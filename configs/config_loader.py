import os
import yaml
from pathlib import Path
from typing import Any

def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base

def _generated_config_path(config: dict[str, Any] | None, config_path: str | None = None) -> Path:
    if isinstance(config, dict):
        prompt_cfg = config.get("prompt_evolution")
        if isinstance(prompt_cfg, dict):
            custom_path = prompt_cfg.get("generated_config_path")
            if isinstance(custom_path, str) and custom_path.strip():
                return Path(custom_path)

    if isinstance(config, dict):
        out_dir = config.get("out_dir", "outputs")
    elif config_path is not None:
        out_dir = "outputs"
    else:
        out_dir = "outputs"
    return Path(out_dir) / "system" / "ai_generated_config.yaml"

def load_config(config_path=None, default_config_str=None, load_generated_config=True):
    """
    Load YAML config from file. If not found, load from default_config_str (YAML string).
    Args:
        config_path: Path to config.yaml
        default_config_str: YAML string to use if file not found
        load_generated_config: if True, overlay AI-generated config from output directory
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

    if not isinstance(config, dict):
        return config

    config["_config_path"] = str(config_path)

    if load_generated_config:
        generated_path = _generated_config_path(config, str(config_path))
        config["_generated_config_path"] = str(generated_path)
        if generated_path.exists():
            try:
                with open(generated_path, "r", encoding="utf-8") as f:
                    generated = yaml.safe_load(f)
                if isinstance(generated, dict):
                    _deep_update(config, generated)
            except Exception:
                pass

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
