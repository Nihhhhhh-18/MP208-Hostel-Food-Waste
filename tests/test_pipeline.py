"""Unit tests for feature cleaning and preprocessing using synthetic in-memory data.

These tests do not require the raw CSV files to exist on disk.
"""
import pandas as pd
from src.food_waste.data import clean_features, normalize_meal


def test_unit_mixup_correction_for_boys_hostel_2():
    """Verify that cooked_kg in grams for Boys Hostel 2 is correctly divided by 1000."""
    synthetic_df = pd.DataFrame([
        {"hostel": "Boys Hostel 2", "cooked_kg": 350000.0, "meal": "lunch"},
        {"hostel": "Boys Hostel 1", "cooked_kg": 350.0, "meal": "lunch"},
    ])
    cleaned = clean_features(synthetic_df)
    assert cleaned.loc[0, "cooked_kg"] == 350.0
    assert cleaned.loc[1, "cooked_kg"] == 350.0


def test_meal_spelling_normalization():
    """Verify that messy spelling variants of meal are normalized to standard categories."""
    dirty_meals = pd.Series([
        "  Lunch  ",
        "LUNCH",
        "lun.",
        "Breakfast",
        "bre.",
        "dinner",
        "din.",
        "DINNER",
    ])
    cleaned = normalize_meal(dirty_meals)
    expected = [
        "lunch",
        "lunch",
        "lunch",
        "breakfast",
        "breakfast",
        "dinner",
        "dinner",
        "dinner",
    ]
    assert list(cleaned) == expected


def test_leaking_column_stripped():
    """Verify that waste_weighed_at_bin is dropped from incoming features."""
    synthetic_df = pd.DataFrame([
        {
            "students_registered": 600,
            "cooked_kg": 250.0,
            "meal": "lunch",
            "hostel": "Girls Hostel",
            "waste_weighed_at_bin": 42.5,
        }
    ])
    cleaned = clean_features(synthetic_df)
    assert "waste_weighed_at_bin" not in cleaned.columns
