"""YAML configs with single-parent inheritance (``base:``) and ``key.sub=value`` CLI overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATCHA_DIRNAME = "MATCHA_Multi-Disorder_mental_health_daTa_in_CHinA"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    node = cfg
    *parents, leaf = dotted_key.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("base", None)
    if parent:
        cfg = deep_merge(load_config(path.parent / parent), cfg)
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def resolve_path(path: str | Path | None) -> Path | None:
    """Expand env vars and resolve relative paths against the repository root."""
    if path is None:
        return None
    p = Path(os.path.expandvars(str(path))).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def matcha_root(explicit: str | Path | None = None) -> Path:
    """Location of the raw MATCHA release: argument > $MATCHA_ROOT > repo-local default."""
    if explicit:
        return resolve_path(explicit)
    env = os.environ.get("MATCHA_ROOT")
    if env:
        return resolve_path(env)
    return REPO_ROOT / DEFAULT_MATCHA_DIRNAME
