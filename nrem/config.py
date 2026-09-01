from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable
import copy
import yaml

ConfigDict = Dict[str, Any]


def load_yaml(path: str | Path) -> ConfigDict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: str | Path, obj: ConfigDict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if k == '_base_':
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> ConfigDict:
    path = Path(path).expanduser().resolve()
    cfg = load_yaml(path)
    base_ref = cfg.get('_base_')
    if not base_ref:
        return cfg
    base_path = (path.parent / str(base_ref)).resolve()
    base_cfg = load_config(base_path)
    return deep_merge(base_cfg, cfg)


def discover_config_paths(configs: Iterable[str], config_globs: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for c in configs or []:
        p = Path(c).expanduser().resolve()
        if p.is_file():
            out.append(p)
    for g in config_globs or []:
        for p in sorted(Path('.').glob(g)):
            if p.is_file() and p.suffix.lower() in {'.yaml', '.yml'}:
                out.append(p.resolve())
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in out:
        s = str(p)
        if s not in seen:
            uniq.append(p)
            seen.add(s)
    if not uniq:
        raise FileNotFoundError('No config files were found.')
    return uniq
