"""运行期状态开关。

状态写入 data/runtime_state.json，属于本机运行数据，不进 Git。
"""

import json
from pathlib import Path
from typing import Any, Dict


STATE_FILE = Path("data/runtime_state.json")
DEFAULT_STATE: Dict[str, Any] = {
    "auto_memory_enabled": True,
    "style_raw_fewshot_enabled": False,
    "style_teaching_enabled": False,
}


def load_state() -> Dict[str, Any]:
    """读取运行期状态，缺失或损坏时回退默认值。"""
    if not STATE_FILE.exists():
        return dict(DEFAULT_STATE)

    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_STATE)

    state = dict(DEFAULT_STATE)
    if isinstance(data, dict):
        state.update(data)
    return state


def save_state(state: Dict[str, Any]) -> None:
    """保存运行期状态。"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def is_auto_memory_enabled() -> bool:
    """自动记忆是否启用。"""
    return bool(load_state().get("auto_memory_enabled", True))


def set_auto_memory_enabled(enabled: bool) -> None:
    """设置自动记忆开关。"""
    state = load_state()
    state["auto_memory_enabled"] = enabled
    save_state(state)


def is_style_raw_fewshot_enabled() -> bool:
    """风格草稿是否允许把真实历史原句作为 few-shot 发送给模型。"""
    return bool(load_state().get("style_raw_fewshot_enabled", False))


def set_style_raw_fewshot_enabled(enabled: bool) -> None:
    """设置真实历史原句 few-shot 开关。"""
    state = load_state()
    state["style_raw_fewshot_enabled"] = bool(enabled)
    save_state(state)


def is_style_teaching_enabled() -> bool:
    """影子教学/审核模式是否启用。"""
    return bool(load_state().get("style_teaching_enabled", False))


def set_style_teaching_enabled(enabled: bool) -> None:
    """设置影子教学/审核模式开关。"""
    state = load_state()
    state["style_teaching_enabled"] = bool(enabled)
    save_state(state)
