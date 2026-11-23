"""Prediction module to generate holdout predictions using the champion model."""
from pathlib import Path
from typing import Tuple
import joblib
import numpy as np
import pandas as pd

import sys
try:
    from src.food_waste.config import load_config
    from src.food_waste.data import clean_features
    from src.food_waste.train import FEATURE_COLS
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.food_waste.config import load_config
    from src.food_waste.data import clean_features
    from src.food_waste.train import FEATURE_COLS


def generate_holdout_predictions() -> Tuple[pd.DataFrame, Path]:
    """Load holdout features, run inference using champion model, and save predictions."""
    config = load_config()
    holdout_path = Path(config["data"]["raw_holdout_path"])
    model_path = Path(config["model"]["export_path"])
    output_path = Path("predictions/holdout_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not holdout_path.exists():
        raise FileNotFoundError(f"Holdout features file not found at: {holdout_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Champion model not found at: {model_path}")

    # 1. Load holdout features
    holdout_df = pd.read_csv(holdout_path)
    print(f"Loaded holdout features: {holdout_df.shape[0]} records")

    # 2. Clean features (e.g. meal normalization, unit checking)
    cleaned_features = clean_features(holdout_df)

    # 3. Load champion model and predict
    model = joblib.load(model_path)
    raw_preds = model.predict(cleaned_features[FEATURE_COLS])

    # Round predictions to 2 decimal places and ensure non-negative
    predictions = np.maximum(0.0, np.round(raw_preds, 2))

    # 4. Construct output DataFrame with exactly record_id,prediction
    results_df = pd.DataFrame({
        "record_id": holdout_df["record_id"],
        "prediction": predictions,
    })

    # Assertions for compliance
    assert len(results_df) == 250, f"Expected exactly 250 records, got {len(results_df)}"
    assert list(results_df.columns) == ["record_id", "prediction"], f"Columns mismatch: {results_df.columns}"
    assert results_df["prediction"].isnull().sum() == 0, "Null predictions found"

    # 5. Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(results_df)} predictions to {output_path}")

    return results_df, output_path


if __name__ == "__main__":
    df, out = generate_holdout_predictions()
    print("\n=== Holdout Predictions Head ===")
    print(df.head(10))
    print("\n=== Holdout Predictions Summary ===")
    print(df["prediction"].describe())
