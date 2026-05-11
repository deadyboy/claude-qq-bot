"""Config loading for QCE style distillation.

Runtime behavior is configured from config/style_distill.json plus optional
config/style_distill.local.json. Environment variables remain available for
machine-specific paths and identifiers.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "style_distill.json"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config" / "style_distill.local.json"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


class StyleDistillSettings:
    def __init__(self, data: Dict[str, Any] | None = None):
        self.data = data or {}

    @classmethod
    def load(cls) -> "StyleDistillSettings":
        config_path = Path(os.getenv("QQBOT_STYLE_DISTILL_CONFIG", "") or DEFAULT_CONFIG_PATH)
        data = _read_json(config_path)
        data = _deep_merge(data, _read_json(LOCAL_CONFIG_PATH))
        return cls(data)

    def get(self, path: str, default: Any = None) -> Any:
        current: Any = self.data
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def str_value(self, path: str, default: str = "") -> str:
        value = self.get(path, default)
        return str(value if value is not None else default)

    def int_value(self, path: str, default: int) -> int:
        try:
            return int(self.get(path, default))
        except (TypeError, ValueError):
            return int(default)

    def float_value(self, path: str, default: float) -> float:
        try:
            return float(self.get(path, default))
        except (TypeError, ValueError):
            return float(default)

    def str_list(self, path: str, default: Sequence[str] = ()) -> tuple[str, ...]:
        value = self.get(path, default)
        if isinstance(value, str):
            items: Iterable[Any] = re.split(r"[,，;\n]+", value)
        elif isinstance(value, Sequence):
            items = value
        else:
            items = default
        return tuple(str(item).strip() for item in items if str(item).strip())

    def dict_list(self, path: str) -> tuple[Dict[str, Any], ...]:
        value = self.get(path, [])
        if not isinstance(value, list):
            return ()
        return tuple(item for item in value if isinstance(item, dict))

    def path_value(self, path: str, default: Path) -> Path:
        text = self.str_value(path, "")
        if not text:
            return default
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        return candidate


DISTILL_SETTINGS = StyleDistillSettings.load()
