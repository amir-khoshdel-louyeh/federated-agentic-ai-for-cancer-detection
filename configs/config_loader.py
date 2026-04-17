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
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    elif default_config_str is not None:
        config = yaml.safe_load(default_config_str)
    else:
        raise FileNotFoundError(f"Config file not found at {config_path} and no default_config_str provided.")

    if isinstance(config, dict):
        config["_config_path"] = str(config_path)
    return config

def save_config(config, config_path=None):
    """
    Save a configuration dictionary back to YAML.
    Removes internal metadata keys before writing.
    """
    if config_path is None:
        config_path = os.path.join("configs", "config.yaml")
    config_path = Path(config_path)

    if not isinstance(config, dict):
        raise ValueError("save_config requires a dict-like config object.")

    sanitized = dict(config)
    sanitized.pop("_config_path", None)
    if isinstance(sanitized.get("prompt_evolution"), dict):
        sanitized["prompt_evolution"] = dict(sanitized["prompt_evolution"])
        sanitized["prompt_evolution"].pop("_config_path", None)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sanitized, f, sort_keys=False, allow_unicode=True)
