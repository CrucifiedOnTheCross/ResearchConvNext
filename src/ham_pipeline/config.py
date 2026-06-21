from __future__ import annotations

from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_name = cfg.pop("_base_", None)
    if base_name is None:
        return cfg
    base_path = Path(base_name)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    return deep_merge(load_config(base_path), cfg)


def set_config_value(cfg: dict, key: str, raw_value: str) -> None:
    current = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = yaml.safe_load(raw_value)
