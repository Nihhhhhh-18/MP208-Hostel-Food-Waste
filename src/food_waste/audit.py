"""Audit module to detect the 7 data issues in train.csv."""
import hashlib
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np


def detect_leaking_column(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> Dict[str, Any]:
    """Detect columns present in train but missing from holdout (excluding target)."""
    train_cols = set(train_df.columns)
    holdout_cols = set(holdout_df.columns)
    leaks = [c for c in train_cols if c not in holdout_cols and c != "waste_kg"]
    if leaks:
        return {"present": True, "column": leaks[0]}
    return {"present": False, "column": None}


def detect_unit_mixup(train_df: pd.DataFrame) -> Dict[str, Any]:
    """Detect if a group recorded a column in a different unit (e.g. factor of 1000)."""
    for col in ["cooked_kg", "plate_size_cm", "temp_c"]:
        if col in train_df.columns:
            group_means = train_df.groupby("hostel")[col].mean()
            for grp in group_means.index:
                other_mean = group_means.drop(grp).mean()
                if other_mean > 0 and group_means[grp] / other_mean > 500:
                    return {
                        "present": True,
                        "column": col,
                        "group_value": grp,
                        "factor": 1000
                    }
    return {"present": False, "column": None, "group_value": None, "factor": None}


def detect_sentinel_values(train_df: pd.DataFrame) -> Dict[str, Any]:
    """Detect impossible sentinel values (e.g. -999, -1, 9999)."""
    num_cols = ["students_registered", "attendance_pct", "menu_rating", "temp_c", "cooked_kg", "plate_size_cm"]
    for col in num_cols:
        for sentinel in [-999, -99, -1, 999, 9999]:
            cnt = int((train_df[col] == sentinel).sum())
            if cnt > 0:
                return {"present": True, "column": col, "value": sentinel, "count": cnt}
    return {"present": False, "column": None, "value": None, "count": None}


def detect_duplicate_rows(train_df: pd.DataFrame) -> Dict[str, Any]:
    """Detect extra duplicate copies of rows."""
    dup_count = int(train_df.duplicated().sum())
    return {"present": dup_count > 0, "count": dup_count if dup_count > 0 else None}


def detect_stuck_sensor(train_df: pd.DataFrame) -> Dict[str, Any]:
    """Detect sensors sending identical readings across dates."""
    sorted_df = train_df.sort_values("date").reset_index(drop=True)
    num_cols = ["temp_c", "plate_size_cm", "menu_rating"]
    for col in num_cols:
        streaks = (sorted_df[col] != sorted_df[col].shift()).cumsum()
        grouped = sorted_df.groupby(streaks).agg(cnt=(col, "count"), start=("date", "first"), end=("date", "last"))
        long_streaks = grouped[grouped["cnt"] >= 10]
        if not long_streaks.empty:
            row = long_streaks.iloc[0]
            return {
                "present": True,
                "column": col,
                "start_date": str(row["start"]),
                "end_date": str(row["end"])
            }
    return {"present": False, "column": None, "start_date": None, "end_date": None}


def detect_target_typos(train_df: pd.DataFrame) -> Dict[str, Any]:
    """Detect target values typed with an extra zero (10 times too big)."""
    if "waste_kg" not in train_df.columns:
        return {"present": False, "count": None}
    typos = int((train_df["waste_kg"] > 100).sum())
    return {"present": typos > 0, "count": typos if typos > 0 else None}


def detect_spelling_variants(train_df: pd.DataFrame, later_df: pd.DataFrame) -> Dict[str, Any]:
    """Detect text columns with non-standard spelling variants."""
    for col in ["meal", "hostel"]:
        train_unique = train_df[col].nunique()
        later_unique = later_df[col].nunique()
        if train_unique > later_unique:
            return {"present": True, "column": col, "true_levels": int(later_unique)}
    return {"present": False, "column": None, "true_levels": None}


def audit_dataset(train_path: str, later_path: str, holdout_path: str) -> Dict[str, Any]:
    """Run full audit for the 7 data issues."""
    train = pd.read_csv(train_path)
    later = pd.read_csv(later_path)
    holdout = pd.read_csv(holdout_path)

    return {
        "leaking_column": detect_leaking_column(train, holdout),
        "unit_mixup": detect_unit_mixup(train),
        "sentinel_value": detect_sentinel_values(train),
        "duplicate_rows": detect_duplicate_rows(train),
        "stuck_sensor": detect_stuck_sensor(train),
        "target_typos": detect_target_typos(train),
        "spelling_variants": detect_spelling_variants(train, later)
    }


if __name__ == "__main__":
    results = audit_dataset("data/raw/train.csv", "data/raw/later.csv", "data/raw/holdout_features.csv")
    import pprint
    pprint.pprint(results)
