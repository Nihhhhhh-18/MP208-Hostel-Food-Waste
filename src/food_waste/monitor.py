"""Monitoring and retraining module: PSI drift calculation, group-wise evaluation, and v2 retraining."""
import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import yaml

import sys
try:
    from src.food_waste.config import load_config
    from src.food_waste.data import clean_dataset, clean_features
    from src.food_waste.train import (
        FEATURE_COLS,
        NUMERICAL_FEATURES,
        TARGET_COL,
        build_pipeline,
        get_data_sha256,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.food_waste.config import load_config
    from src.food_waste.data import clean_dataset, clean_features
    from src.food_waste.train import (
        FEATURE_COLS,
        NUMERICAL_FEATURES,
        TARGET_COL,
        build_pipeline,
        get_data_sha256,
    )


def calculate_psi(expected: pd.Series, actual: pd.Series, num_bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) using 10 quantile bins based on expected data."""
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Replace zeros with small epsilon to avoid division by zero
    expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi_val), 4)


def run_monitoring_and_retraining() -> Dict[str, Any]:
    """Execute Stage 8 monitoring and retraining lifecycle."""
    config = load_config()
    raw_train_path = config["data"]["raw_train_path"]
    raw_later_path = config["data"]["raw_later_path"]
    export_path = Path(config["model"]["export_path"])
    data_sha = get_data_sha256(raw_train_path)

    # 1. Load data
    train_clean = clean_dataset(pd.read_csv(raw_train_path))
    later_df = pd.read_csv(raw_later_path)

    # Clean features on later.csv
    later_clean = clean_features(later_df)

    # 2. Calculate PSI for all numeric features
    psi_results = {}
    for col in NUMERICAL_FEATURES:
        psi_results[col] = calculate_psi(train_clean[col], later_clean[col], num_bins=10)

    most_drifted = max(psi_results, key=psi_results.get)

    # 3. Compute v1 MAE on later.csv by hostel group
    v1_model = joblib.load(export_path)
    mae_by_group = {}
    for hostel in ["Boys Hostel 1", "Boys Hostel 2", "Girls Hostel"]:
        sub = later_clean[later_clean["hostel"] == hostel]
        preds = v1_model.predict(sub)
        mae = float(mean_absolute_error(sub[TARGET_COL], preds))
        mae_by_group[hostel] = round(mae, 4)

    worst_served = max(mae_by_group, key=mae_by_group.get)

    # 4. Sort later.csv by date and split 70% / 30%
    later_sorted = later_clean.sort_values("date").reset_index(drop=True)
    split_idx = int(0.7 * len(later_sorted))
    later_70 = later_sorted.iloc[:split_idx].copy()
    later_30 = later_sorted.iloc[split_idx:].copy()

    # 5. Evaluate v1 on last 30% check set
    v1_check_preds = v1_model.predict(later_30[FEATURE_COLS])
    mae_v1_check = round(float(mean_absolute_error(later_30[TARGET_COL], v1_check_preds)), 4)

    # 6. Train v2 on cleaned train + first 70% of later
    combined_train = pd.concat([train_clean, later_70], ignore_index=True)

    mlflow.set_tracking_uri(config["tracking"]["tracking_uri"])
    mlflow.set_experiment(config["tracking"]["experiment_name"])
    client = MlflowClient()

    with mlflow.start_run(run_name="v2_retrained_ridge") as run:
        mlflow.log_param("data_sha256", data_sha)
        mlflow.log_param("model_type", "Ridge_v2")
        mlflow.log_param("alpha", 1.0)
        mlflow.log_param("training_samples", len(combined_train))
        mlflow.log_param("retrain_split_ratio", 0.7)

        v2_pipeline = build_pipeline(Ridge(alpha=1.0, random_state=42))
        v2_pipeline.fit(combined_train[FEATURE_COLS], combined_train[TARGET_COL])

        v2_check_preds = v2_pipeline.predict(later_30[FEATURE_COLS])
        mae_v2_check = round(float(mean_absolute_error(later_30[TARGET_COL], v2_check_preds)), 4)

        improvement = (mae_v1_check - mae_v2_check) / mae_v1_check

        mlflow.log_metric("mae", mae_v2_check)
        mlflow.log_metric("mae_v1_on_check", mae_v1_check)
        mlflow.log_metric("mae_v2_on_check", mae_v2_check)
        mlflow.log_metric("improvement_pct", round(improvement * 100, 2))

        # Register v2 model
        model_info = mlflow.sklearn.log_model(
            sk_model=v2_pipeline,
            artifact_path="model",
            serialization_format="cloudpickle",
        )

        reg_name = config["model"]["registered_model_name"]
        model_v2_version = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=reg_name,
        )

    # 7. Apply Promotion Rule: promote only if MAE on last 30% is >= 5% lower
    threshold = float(config["monitoring"]["promotion_improvement_threshold"])
    if improvement >= threshold:
        decision = "promote"
        # Update champion alias to v2
        client.set_registered_model_alias(
            name=reg_name,
            alias="champion",
            version=model_v2_version.version,
        )
        # Update exported model artifact to v2 champion
        joblib.dump(v2_pipeline, export_path)
    else:
        decision = "keep"

    # 8. Update findings.yaml and reports/findings.yaml
    findings_path = Path("findings.yaml")
    reports_findings_path = Path("reports/findings.yaml")

    with open(findings_path, "r", encoding="utf-8") as f:
        findings_data = yaml.safe_load(f)

    findings_data["monitoring"] = {
        "psi": psi_results,
        "most_drifted_feature": most_drifted,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_served,
        "mae_v1_on_check": mae_v1_check,
        "mae_v2_on_check": mae_v2_check,
        "decision": decision,
    }

    if decision == "promote":
        findings_data["tracking"]["champion_version"] = int(model_v2_version.version)

    for p in [findings_path, reports_findings_path]:
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(findings_data, f, sort_keys=False)

    return {
        "psi": psi_results,
        "most_drifted_feature": most_drifted,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_served,
        "mae_v1_on_check": mae_v1_check,
        "mae_v2_on_check": mae_v2_check,
        "improvement_pct": round(improvement * 100, 2),
        "decision": decision,
        "champion_version": model_v2_version.version if decision == "promote" else 1,
    }


if __name__ == "__main__":
    res = run_monitoring_and_retraining()
    print("\n=== Monitoring & Retraining Summary ===")
    print("PSI (10 bins):", res["psi"])
    print(f"Most Drifted Feature: {res['most_drifted_feature']} (PSI: {res['psi'][res['most_drifted_feature']]})")
    print("MAE by Hostel:", res["mae_by_group"])
    print(f"Worst Served Group: {res['worst_served_group']} (MAE: {res['mae_by_group'][res['worst_served_group']]})")
    print(f"v1 MAE on check (last 30%): {res['mae_v1_on_check']}")
    print(f"v2 MAE on check (last 30%): {res['mae_v2_on_check']}")
    print(f"Improvement: {res['improvement_pct']}% (Threshold: 5.0%)")
    print(f"Decision: {res['decision'].upper()}")
    print(f"Final Champion Version: {res['champion_version']}")
