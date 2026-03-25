import os
import yaml
from pathlib import Path

def load_config(config_path=None, default_config_str=None):
    """
    Load YAML config from file. If not found, load from default_config_str (YAML string).
    Args:
        config_path: Path to config.yaml
        default_config_str: YAML string to use if file not found
    Returns:
        dict: Parsed config
    """
    if config_path is None:
        config_path = os.path.join("configs", "config.yaml")
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    elif default_config_str is not None:
        return yaml.safe_load(default_config_str)
    else:
        raise FileNotFoundError(f"Config file not found at {config_path} and no default_config_str provided.")
