"""Config loading. Single source of truth for every tunable constant.

Both YAML files are resolved relative to the package root, so the app can be
launched from anywhere. `Config` gives attribute-ish access to nested keys and
raises loudly on a missing key -- a typo in the YAML should fail at startup,
not silently fall back to a hardcoded default hidden in a recognizer.
"""

from __future__ import annotations

import os
from typing import Any

import yaml

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

THRESHOLDS_PATH = os.path.join(PKG_ROOT, "config", "thresholds.yaml")


def resolve(path: str) -> str:
    """Turn a config-relative path into an absolute one."""
    return path if os.path.isabs(path) else os.path.join(PKG_ROOT, path)


class Section:
    """Dict wrapper that fails loudly instead of returning None for typos."""

    def __init__(self, name: str, data: dict):
        self._name = name
        self._data = data or {}

    def __getitem__(self, key: str) -> Any:
        if key not in self._data:
            raise KeyError(
                f"missing key '{key}' in [{self._name}] of thresholds.yaml"
            )
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def as_dict(self) -> dict:
        return dict(self._data)


class Config:
    def __init__(self, thresholds: dict):
        self._raw = thresholds
        for name in (
            "camera",
            "tracker",
            "segmenter",
            "trajectory",
            "pose",
            "builtin",
            "laser",
            "recording",
            "actions",
            "logging",
            "web",
        ):
            if name not in thresholds:
                raise KeyError(f"thresholds.yaml is missing the [{name}] section")
            setattr(self, name, Section(name, thresholds[name]))

    @classmethod
    def load(cls, path: str = THRESHOLDS_PATH) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))
