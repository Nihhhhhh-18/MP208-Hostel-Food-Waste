"""Data cleaning and preprocessing module."""
import sys
from pathlib import Path
from typing import Tuple
import pandas as pd

try:
    from src.food_waste.config import load_config
except ModuleNotFoundError:
    # Handle direct script invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.food_waste.config import load_config


def normalize_meal(series: pd.Series) -> pd.Series:
    """Normalize spelling variants in meal column to standard categories: breakfast, lunch, dinner."""
    s = series.astype(str).str.strip().str.lower()
    return s.map(
        lambda x: "lunch" if x.startswith("lu") else (
            "breakfast" if x.startswith("br") else (
                "dinner" if x.startswith("di") else x
            )
        )
    )


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean feature columns for both training and inference time.
    
    - Corrects unit mixup in cooked_kg for Boys Hostel 2 (divides by 1000).
    - Normalizes spelling variants in meal.
    - Drops leaking column waste_weighed_at_bin if present.
    """
    clean_df = df.copy()

    # Drop leaking column if present
    if "waste_weighed_at_bin" in clean_df.columns:
        clean_df = clean_df.drop(columns=["waste_weighed_at_bin"])

    # Fix unit mixup in cooked_kg for Boys Hostel 2
    if "cooked_kg" in clean_df.columns and "hostel" in clean_df.columns:
        bh2_mask = (clean_df["hostel"] == "Boys Hostel 2") & (clean_df["cooked_kg"] > 2000)
        clean_df.loc[bh2_mask, "cooked_kg"] = clean_df.loc[bh2_mask, "cooked_kg"] / 1000.0

    # Fix spelling variants in meal
    if "meal" in clean_df.columns:
        clean_df["meal"] = normalize_meal(clean_df["meal"])

    return clean_df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for training data:
    
    1. Removes duplicate rows.
    2. Fixes target typos in waste_kg (divides >100 by 10).
    3. Cleans features (drops leakage, fixes unit mixup, cleans meal).
    """
    # 1. Remove duplicate rows
    clean_df = df.drop_duplicates().copy()

    # 2. Fix target typos if waste_kg is present
    if "waste_kg" in clean_df.columns:
        typo_mask = clean_df["waste_kg"] > 100.0
        clean_df.loc[typo_mask, "waste_kg"] = clean_df.loc[typo_mask, "waste_kg"] / 10.0

    # 3. Clean features
    clean_df = clean_features(clean_df)

    return clean_df.reset_index(drop=True)


def process_and_save_train_data() -> Tuple[pd.DataFrame, Path]:
    """Load raw train.csv, clean it, and save to processed directory."""
    config = load_config()
    raw_path = Path(config["data"]["raw_train_path"])
    proc_path = Path(config["data"]["processed_train_path"])
    proc_path.parent.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(raw_path)
    cleaned_df = clean_dataset(raw_df)
    cleaned_df.to_csv(proc_path, index=False)
    return cleaned_df, proc_path


if __name__ == "__main__":
    cleaned, path = process_and_save_train_data()
    print(f"Cleaned training data saved to: {path}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(cleaned.describe())
