"""Training module with scikit-learn regression pipelines and MLflow tracking."""
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys
try:
    from src.food_waste.config import load_config
    from src.food_waste.data import clean_dataset
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.food_waste.config import load_config
    from src.food_waste.data import clean_dataset

NUMERICAL_FEATURES = [
    "students_registered",
    "attendance_pct",
    "menu_rating",
    "temp_c",
    "cooked_kg",
    "plate_size_cm",
]
CATEGORICAL_FEATURES = ["meal", "hostel"]
FEATURE_COLS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "waste_kg"


def build_preprocessor() -> ColumnTransformer:
    """Build column transformer for preprocessing features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_pipeline(regressor: Any) -> Pipeline:
    """Assemble scikit-learn pipeline with preprocessing and regressor."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )


def get_data_sha256(raw_train_path: str = "data/raw/train.csv") -> str:
    """Calculate SHA-256 fingerprint of the original raw train.csv."""
    with open(raw_train_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def train_and_track_models() -> Dict[str, Any]:
    """Train at least three model configurations, log to MLflow, and register champion."""
    config = load_config()
    raw_path = config["data"]["raw_train_path"]
    data_sha = get_data_sha256(raw_path)

    # 1. Load and clean training data
    raw_df = pd.read_csv(raw_path)
    clean_df = clean_dataset(raw_df)

    X = clean_df[FEATURE_COLS]
    y = clean_df[TARGET_COL]

    test_size = float(config["model"]["test_size"])
    random_state = int(config["model"]["random_state"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2. Configure MLflow with SQLite tracking
    tracking_uri = config["tracking"]["tracking_uri"]
    experiment_name = config["tracking"]["experiment_name"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    # 3. Define 3 model configurations
    candidates = [
        {
            "name": "Ridge_Baseline",
            "model_type": "Ridge",
            "regressor": Ridge(alpha=1.0, random_state=random_state),
            "params": {"alpha": 1.0, "random_state": random_state},
        },
        {
            "name": "Random_Forest",
            "model_type": "RandomForestRegressor",
            "regressor": RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=random_state
            ),
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": random_state,
            },
        },
        {
            "name": "Gradient_Boosting",
            "model_type": "GradientBoostingRegressor",
            "regressor": GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=4,
                random_state=random_state,
            ),
            "params": {
                "n_estimators": 150,
                "learning_rate": 0.05,
                "max_depth": 4,
                "random_state": random_state,
            },
        },
    ]

    runs_summary = []
    best_run = None
    best_mae = float("inf")
    best_pipeline = None

    for cand in candidates:
        with mlflow.start_run(run_name=cand["name"]) as run:
            run_id = run.info.run_id

            # Log required data_sha256 and hyperparameters
            mlflow.log_param("data_sha256", data_sha)
            mlflow.log_param("model_type", cand["model_type"])
            for p_name, p_val in cand["params"].items():
                mlflow.log_param(p_name, p_val)

            # Build and train pipeline
            pipe = build_pipeline(cand["regressor"])
            pipe.fit(X_train, y_train)

            # Evaluate
            preds = pipe.predict(X_val)
            mae = float(mean_absolute_error(y_val, preds))

            # Log MAE metric (handbook requires metric name containing 'mae')
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("val_mae", mae)

            # Log model artifact with cloudpickle format
            model_info = mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            runs_summary.append(
                {
                    "run_id": run_id,
                    "name": cand["name"],
                    "mae": mae,
                    "model_uri": model_info.model_uri,
                }
            )

            if mae < best_mae:
                best_mae = mae
                best_run = {
                    "run_id": run_id,
                    "name": cand["name"],
                    "mae": mae,
                    "model_uri": model_info.model_uri,
                }
                best_pipeline = pipe

    # 4. Register best model in MLflow Model Registry
    reg_model_name = config["model"]["registered_model_name"]
    model_version = mlflow.register_model(
        model_uri=best_run["model_uri"],
        name=reg_model_name,
    )

    # Set champion alias on the registered version
    champion_alias = config["model"]["champion_alias"]
    client.set_registered_model_alias(
        name=reg_model_name,
        alias=champion_alias,
        version=model_version.version,
    )

    # 5. Export champion model to models/model.joblib for FastAPI service
    export_path = Path(config["model"]["export_path"])
    export_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, export_path)

    return {
        "runs": runs_summary,
        "best_run": best_run,
        "registered_model_name": reg_model_name,
        "champion_version": model_version.version,
        "champion_alias": champion_alias,
        "export_path": str(export_path),
        "data_sha256": data_sha,
    }


if __name__ == "__main__":
    result = train_and_track_models()
    print("\n=== MLflow Experiment Summary ===")
    for r in result["runs"]:
        print(f"Run ID: {r['run_id']} | Model: {r['name']:20s} | MAE: {r['mae']:.4f}")
    print(f"\nBest Model: {result['best_run']['name']} (MAE: {result['best_run']['mae']:.4f})")
    print(f"Registered Name: {result['registered_model_name']}")
    print(f"Champion Version: {result['champion_version']} (Alias: @{result['champion_alias']})")
    print(f"Exported to: {result['export_path']}")
