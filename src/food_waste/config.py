"""Configuration loader reading from params.yaml using yaml.safe_load."""
from pathlib import Path
from typing import Any, Dict
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "params.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load and return configuration from params.yaml safely."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
