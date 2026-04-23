from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


ConfigDict = Dict[str, Any]


def _read_yaml(path: Path) -> ConfigDict:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"配置文件必须是字典结构: {path}")

    return data


def _deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    merged = deepcopy(base)

    for key, value in override.items():
        if key == 'base_config':
            continue

        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged


def _normalize_base_paths(base_config: Any) -> Iterable[str]:
    if not base_config:
        return []
    if isinstance(base_config, str):
        return [base_config]
    if isinstance(base_config, list):
        if not all(isinstance(item, str) for item in base_config):
            raise TypeError('base_config 列表中的所有元素都必须是字符串路径')
        return base_config
    raise TypeError('base_config 必须是字符串或字符串列表')


def _resolve_config_path(config_path: str | Path) -> Path:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _load_config_recursive(config_path: Path, visited: set[Path]) -> ConfigDict:
    config_path = config_path.resolve()

    if config_path in visited:
        cycle = ' -> '.join(str(path) for path in [*visited, config_path])
        raise ValueError(f'检测到循环配置继承: {cycle}')

    visited.add(config_path)
    current_config = _read_yaml(config_path)

    merged_config: ConfigDict = {}
    for base_path_str in _normalize_base_paths(current_config.get('base_config')):
        base_path = Path(base_path_str)
        if not base_path.is_absolute():
            base_path = (config_path.parent / base_path).resolve()
        base_config = _load_config_recursive(base_path, visited)
        merged_config = _deep_merge(merged_config, base_config)

    merged_config = _deep_merge(merged_config, current_config)
    visited.remove(config_path)
    return merged_config


def load_config(config_path: str | Path) -> ConfigDict:
    resolved_path = _resolve_config_path(config_path)
    return _load_config_recursive(resolved_path, visited=set())
