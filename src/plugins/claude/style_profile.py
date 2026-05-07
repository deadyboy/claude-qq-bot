"""Owner style profile storage and draft generation."""

import json
import re
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .api import llm_client
from .auto_memory import contains_sensitive_content


STYLE_PROFILE_DIR = Path("data/style_profiles")
DEFAULT_PROFILE_NAME = "default"
MAX_STYLE_SAMPLE_LENGTH = 1200
MAX_STYLE_SAMPLES = 12

DEFAULT_STYLE_PROFILE: Dict[str, Any] = {
    "name": DEFAULT_PROFILE_NAME,
    "tone": "自然、直接、不过度热情",
    "length": "中短，优先说重点",
    "emoji": "少量使用；没有把握时不用",
    "habits": [],
    "avoid": ["编造事实", "替主人承诺无法确认的行动", "过度解释自己是机器人"],
    "examples": [],
    "updated_at": None,
}

FIELD_ALIASES = {
    "tone": "tone",
    "语气": "tone",
    "风格": "tone",
    "length": "length",
    "长度": "length",
    "篇幅": "length",
    "emoji": "emoji",
    "表情": "emoji",
    "习惯": "habits",
    "口癖": "habits",
    "常用表达": "habits",
    "avoid": "avoid",
    "避免": "avoid",
    "不要": "avoid",
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _default_profile() -> Dict[str, Any]:
    return deepcopy(DEFAULT_STYLE_PROFILE)


def _normalize_list(value: Any) -> List[str]:
    if isinstance(value, list):
        items = value
    else:
        items = re.split(r"[；;、,，\n]+", str(value))

    normalized = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text[:80])
    return normalized[:20]


def normalize_style_profile(raw: Any) -> Dict[str, Any]:
    """Return a sanitized style profile dict."""
    profile = _default_profile()
    if isinstance(raw, dict):
        profile.update(raw)

    profile["name"] = str(profile.get("name") or DEFAULT_PROFILE_NAME).strip() or DEFAULT_PROFILE_NAME
    for field in ("tone", "length", "emoji"):
        profile[field] = str(profile.get(field) or DEFAULT_STYLE_PROFILE[field]).strip()

    profile["habits"] = _normalize_list(profile.get("habits", []))
    profile["avoid"] = _normalize_list(profile.get("avoid", []))

    examples = []
    for item in profile.get("examples") or []:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            created_at = str(item.get("created_at", "")).strip()
        else:
            text = str(item).strip()
            created_at = ""
        if not text:
            continue
        examples.append({
            "text": text[:MAX_STYLE_SAMPLE_LENGTH],
            "created_at": created_at,
        })
    profile["examples"] = examples[-MAX_STYLE_SAMPLES:]
    return profile


class StyleProfileStore:
    """Small JSON store for owner style profiles."""

    def __init__(self, base_dir: Path | str = STYLE_PROFILE_DIR):
        self.base_dir = Path(base_dir)

    def profile_path(self, name: str = DEFAULT_PROFILE_NAME) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip() or DEFAULT_PROFILE_NAME)
        return self.base_dir / f"{safe_name}.json"

    def load(self, name: str = DEFAULT_PROFILE_NAME) -> Dict[str, Any]:
        path = self.profile_path(name)
        if not path.exists():
            return _default_profile()

        try:
            with path.open("r", encoding="utf-8") as f:
                return normalize_style_profile(json.load(f))
        except (OSError, json.JSONDecodeError):
            return _default_profile()

    def save(self, profile: Dict[str, Any], name: str = DEFAULT_PROFILE_NAME) -> Dict[str, Any]:
        normalized = normalize_style_profile(profile)
        normalized["updated_at"] = _now_iso()
        path = self.profile_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)
        return normalized

    def set_field(self, field: str, value: str, name: str = DEFAULT_PROFILE_NAME) -> tuple[bool, str]:
        canonical = FIELD_ALIASES.get(field.strip().lower()) or FIELD_ALIASES.get(field.strip())
        if canonical not in {"tone", "length", "emoji", "habits", "avoid"}:
            return False, "可设置字段：语气、长度、表情、习惯、避免。"

        text = value.strip()
        if not text:
            return False, "设置内容不能为空。"
        if contains_sensitive_content(text):
            return False, "内容疑似包含敏感信息，已拒绝保存。"

        profile = self.load(name)
        if canonical in {"habits", "avoid"}:
            profile[canonical] = _normalize_list(text)
        else:
            profile[canonical] = text[:120]
        self.save(profile, name)
        return True, f"已更新风格画像：{field.strip()}。"

    def add_example(self, text: str, name: str = DEFAULT_PROFILE_NAME) -> tuple[bool, str]:
        sample = text.strip()
        if not sample:
            return False, "用法：/风格 导入 <一小段你的真实回复样本>"
        if len(sample) > MAX_STYLE_SAMPLE_LENGTH:
            return False, f"样本太长。v1 单条最多 {MAX_STYLE_SAMPLE_LENGTH} 字，批量导入放到下一阶段做。"
        if contains_sensitive_content(sample):
            return False, "样本疑似包含敏感信息，已拒绝保存。"

        profile = self.load(name)
        profile["examples"].append({
            "text": sample,
            "created_at": _now_iso(),
        })
        profile["examples"] = profile["examples"][-MAX_STYLE_SAMPLES:]
        self.save(profile, name)
        return True, f"已保存 1 条风格样本。当前样本数：{len(profile['examples'])}。"

    def clear_examples(self, name: str = DEFAULT_PROFILE_NAME) -> str:
        profile = self.load(name)
        count = len(profile.get("examples") or [])
        profile["examples"] = []
        self.save(profile, name)
        return f"已清空风格样本：{count} 条。"

    def delete_for_tests(self) -> None:
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)


style_store = StyleProfileStore()


def format_style_profile(profile: Dict[str, Any]) -> str:
    """Format a profile without exposing raw bulk data."""
    data = normalize_style_profile(profile)
    lines = [
        "当前风格画像：",
        f"- 语气：{data['tone']}",
        f"- 长度：{data['length']}",
        f"- 表情：{data['emoji']}",
    ]

    habits = data.get("habits") or []
    avoid = data.get("avoid") or []
    examples = data.get("examples") or []
    lines.append(f"- 习惯：{'；'.join(habits) if habits else '未设置'}")
    lines.append(f"- 避免：{'；'.join(avoid) if avoid else '未设置'}")
    lines.append(f"- 样本数：{len(examples)}")

    for index, example in enumerate(examples[-3:], start=max(1, len(examples) - 2)):
        preview = example["text"].replace("\n", " ")[:80]
        lines.append(f"  {index}. {preview}")

    lines.append("用法：/风格 设置 语气=自然简短；/风格 导入 <样本>；/用我的风格回复：<对方消息>")
    return "\n".join(lines)


def parse_style_command(text: str) -> tuple[str, str]:
    """Parse /风格 command into action and payload."""
    stripped = text.strip()
    lowered = stripped.lower()
    for prefix in ("/style", "/风格", "风格"):
        if lowered == prefix.lower():
            return "help", ""
        if lowered.startswith(prefix.lower() + " "):
            stripped = stripped[len(prefix):].strip()
            break
        if stripped.startswith(prefix + "：") or stripped.startswith(prefix + ":"):
            stripped = stripped[len(prefix) + 1:].strip()
            break
    else:
        return "help", ""

    lowered = stripped.lower()
    action_prefixes = {
        "view": ("查看", "show", "view"),
        "set": ("设置", "set"),
        "import": ("导入", "示例", "样本", "import", "add"),
        "clear_examples": ("清空样本", "clear examples"),
        "help": ("帮助", "help"),
    }
    for action, prefixes in action_prefixes.items():
        for prefix in prefixes:
            if lowered == prefix.lower():
                return action, ""
            if lowered.startswith(prefix.lower() + " "):
                return action, stripped[len(prefix):].strip()
            if stripped.startswith(prefix + "：") or stripped.startswith(prefix + ":"):
                return action, stripped[len(prefix) + 1:].strip()

    return "help", stripped


def parse_style_set_payload(payload: str) -> tuple[str, str] | None:
    for separator in ("=", "：", ":"):
        if separator in payload:
            field, value = payload.split(separator, 1)
            field = field.strip()
            value = value.strip()
            if field and value:
                return field, value
    return None


def parse_style_draft_payload(text: str) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    prefixes = (
        "/style draft",
        "/用我的风格回复",
        "用我的风格回复",
        "风格回复",
    )
    for prefix in prefixes:
        if lowered == prefix.lower():
            return ""
        if lowered.startswith(prefix.lower() + " "):
            return stripped[len(prefix):].strip()
        if stripped.startswith(prefix + "：") or stripped.startswith(prefix + ":"):
            return stripped[len(prefix) + 1:].strip()
    return ""


def format_style_help() -> str:
    return "\n".join([
        "风格画像命令：",
        "- /风格 查看：查看当前画像摘要",
        "- /风格 设置 语气=自然、简短、像我本人",
        "- /风格 设置 习惯=短句；少解释；必要时用一点表情",
        "- /风格 导入 <一小段你的真实回复样本>",
        "- /风格 清空样本 确认：删除已导入样本",
        "- /用我的风格回复：<对方消息>：生成一条草稿，不会代替你发送",
    ])


def build_style_system_prompt(profile: Dict[str, Any]) -> str:
    data = normalize_style_profile(profile)
    lines = [
        "你是一个回复草稿生成器，任务是模仿“主人”的日常聊天风格生成一条可复制的中文回复草稿。",
        "只输出草稿正文，不要解释，不要加标题。",
        "不要声称自己是机器人；不要代替主人承诺无法确认的现实行动；不要编造事实。",
        f"语气：{data['tone']}",
        f"长度：{data['length']}",
        f"表情：{data['emoji']}",
    ]
    if data.get("habits"):
        lines.append("表达习惯：")
        lines.extend(f"- {item}" for item in data["habits"][:10])
    if data.get("avoid"):
        lines.append("避免：")
        lines.extend(f"- {item}" for item in data["avoid"][:10])
    if data.get("examples"):
        lines.append("主人真实回复样本，仅学习表达风格，不要照抄隐私内容：")
        for example in data["examples"][-5:]:
            lines.append(f"- {example['text']}")
    return "\n".join(lines)


async def generate_style_draft(message: str, store: StyleProfileStore = style_store) -> str:
    target = message.strip()
    if not target:
        return "用法：/用我的风格回复：<对方消息>"
    if len(target) > 1000:
        return "对方消息太长。v1 先支持 1000 字以内的单条草稿。"

    profile = store.load()
    return await llm_client.chat(
        messages=[{"role": "user", "content": f"对方消息：{target}"}],
        system_prompt=build_style_system_prompt(profile),
        temperature=0.6,
    )
