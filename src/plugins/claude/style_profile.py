"""Owner style profile storage and draft generation."""

import asyncio
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from copy import deepcopy
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

from .api import llm_client
from .auto_memory import contains_sensitive_content


STYLE_PROFILE_DIR = Path("data/style_profiles")
DEFAULT_PROFILE_NAME = "default"
MAX_STYLE_SAMPLE_LENGTH = 1200
MAX_STYLE_SAMPLES = 12
MAX_IMPORT_FILE_BYTES = 2 * 1024 * 1024
MAX_IMPORTED_MESSAGES = 20000
SUPPORTED_IMPORT_EXTENSIONS = {".txt", ".json", ".csv"}

DEFAULT_STYLE_PROFILE: Dict[str, Any] = {
    "name": DEFAULT_PROFILE_NAME,
    "tone": "自然、直接、不过度热情",
    "length": "中短，优先说重点",
    "emoji": "少量使用；没有把握时不用",
    "habits": [],
    "avoid": [
        "编造事实",
        "替主人承诺无法确认的行动",
        "替主人回答当前是否忙、在哪、是否已经完成等未知现实状态",
        "过度解释自己是机器人",
    ],
    "common_phrases": [],
    "punctuation": [],
    "stats": {},
    "source_imports": [],
    "constraints": {
        "draft_only": True,
        "do_not_store_other_party_facts": True,
        "do_not_invent_owner_state": True,
    },
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

CSV_SENDER_COLUMNS = ("sender", "nick", "nickname", "name", "user", "user_id", "qq", "from", "author")
CSV_TEXT_COLUMNS = ("text", "message", "content", "msg", "raw_message")
CSV_ROLE_COLUMNS = ("role", "type")

OWNER_ROLES = {"owner", "me", "self", "mine", "我", "本人", "主人"}
QQ_TEXT_EXPORT_HEADER_RE = re.compile(
    r"^\s*(?P<time>\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}:\d{2})\s+"
    r"(?P<sender>.+?)\s*$"
)
TEXT_HEADER_RE = re.compile(
    r"^\s*(?P<sender>[^:\n：]{1,80})\s*[:：]\s*"
    r"(?P<time>(?:\d{2,4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}|\d{1,2}:\d{2}).*)$"
)
INLINE_MESSAGE_RE = re.compile(r"^\s*(?P<sender>[^:\n：]{1,40})\s*[:：]\s*(?P<text>.+)$")
TXT_METADATA_PREFIXES = ("消息记录", "消息分组:", "消息分组：", "消息对象:", "消息对象：")


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


def _normalize_source_imports(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    imports: List[Dict[str, Any]] = []
    for item in value[-50:]:
        if not isinstance(item, dict):
            continue
        imports.append({
            "import_id": str(item.get("import_id", "")).strip(),
            "file_name": str(item.get("file_name", "")).strip()[:160],
            "imported_at": str(item.get("imported_at", "")).strip(),
            "message_count": int(item.get("message_count") or 0),
            "skipped_sensitive": int(item.get("skipped_sensitive") or 0),
        })
    return [item for item in imports if item["import_id"]]


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
    profile["common_phrases"] = _normalize_list(profile.get("common_phrases", []))
    profile["punctuation"] = _normalize_list(profile.get("punctuation", []))
    profile["source_imports"] = _normalize_source_imports(profile.get("source_imports", []))
    if not isinstance(profile.get("stats"), dict):
        profile["stats"] = {}
    if not isinstance(profile.get("constraints"), dict):
        profile["constraints"] = deepcopy(DEFAULT_STYLE_PROFILE["constraints"])

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


def _normalize_marker(value: str) -> str:
    return str(value).strip().strip("\"'“”").lower()


def split_owner_markers(value: str) -> List[str]:
    """Split owner marker text such as '36,付健'."""
    markers = []
    seen = set()
    for item in re.split(r"[；;、,，|]+", value):
        marker = item.strip().strip("\"'“”")
        normalized = _normalize_marker(marker)
        if marker and normalized not in seen:
            seen.add(normalized)
            markers.append(marker)
    return markers


def parse_style_import_file_payload(payload: str) -> tuple[str, List[str]]:
    """Parse '/风格 导入文件 <path> 我=<marker>' payload."""
    text = payload.strip()
    marker_values = []
    option_re = re.compile(r"(?:^|\s)(?:我|owner|主人|sender|nick|昵称)=([^\s]+)", flags=re.I)
    for match in option_re.finditer(text):
        marker_values.extend(split_owner_markers(match.group(1)))

    path_text = option_re.sub(" ", text).strip().strip("\"'“”")
    return path_text, marker_values


def _clean_import_message(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    return cleaned[:MAX_STYLE_SAMPLE_LENGTH]


def _looks_like_owner(role: str, sender: str, owner_markers: List[str]) -> bool:
    if _normalize_marker(role) in OWNER_ROLES:
        return True

    normalized_sender = _normalize_marker(sender)
    if not normalized_sender:
        return False
    for marker in owner_markers:
        normalized_marker = _normalize_marker(marker)
        if not normalized_marker:
            continue
        if normalized_sender == normalized_marker:
            return True
        if len(normalized_marker) >= 2 and normalized_marker in normalized_sender:
            return True
    return False


def _read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_field(item: Dict[str, Any], candidates: tuple[str, ...]) -> str:
    lower_map = {str(key).lower(): key for key in item.keys()}
    for candidate in candidates:
        key = lower_map.get(candidate.lower())
        if key is not None:
            value = item.get(key)
            if value is not None:
                return str(value)
    return ""


def _iter_json_records(data: Any) -> List[Dict[str, str]]:
    if isinstance(data, dict):
        for key in ("messages", "records", "data", "items", "chat"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]

    if not isinstance(data, list):
        return []

    records: List[Dict[str, str]] = []
    for item in data:
        if isinstance(item, str):
            records.append({"sender": "", "role": "", "text": item})
            continue
        if not isinstance(item, dict):
            continue
        records.append({
            "sender": _extract_field(item, CSV_SENDER_COLUMNS),
            "role": _extract_field(item, CSV_ROLE_COLUMNS),
            "text": _extract_field(item, CSV_TEXT_COLUMNS),
        })
    return records


def _iter_csv_records(text: str) -> List[Dict[str, str]]:
    reader = csv.DictReader(StringIO(text))
    if not reader.fieldnames:
        return []

    records: List[Dict[str, str]] = []
    for row in reader:
        normalized = {str(key): value for key, value in row.items() if key is not None}
        records.append({
            "sender": _extract_field(normalized, CSV_SENDER_COLUMNS),
            "role": _extract_field(normalized, CSV_ROLE_COLUMNS),
            "text": _extract_field(normalized, CSV_TEXT_COLUMNS),
        })
    return records


def _iter_txt_records(text: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    current_sender = ""
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer, current_sender
        if current_sender and buffer:
            records.append({
                "sender": current_sender,
                "role": "",
                "text": "\n".join(buffer).strip(),
            })
        buffer = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(TXT_METADATA_PREFIXES) or set(line) == {"="}:
            continue

        qq_header_match = QQ_TEXT_EXPORT_HEADER_RE.match(line)
        if qq_header_match:
            flush()
            current_sender = qq_header_match.group("sender").strip()
            continue

        header_match = TEXT_HEADER_RE.match(line)
        if header_match:
            flush()
            current_sender = header_match.group("sender").strip()
            rest = line[header_match.end():].strip()
            if rest:
                buffer.append(rest)
            continue

        inline_match = INLINE_MESSAGE_RE.match(line)
        if inline_match and not current_sender:
            records.append({
                "sender": inline_match.group("sender").strip(),
                "role": "",
                "text": inline_match.group("text").strip(),
            })
            continue

        if current_sender:
            buffer.append(line)

    flush()
    return records


def parse_chat_log_text(text: str, suffix: str) -> List[Dict[str, str]]:
    """Parse supported chat-log text into generic sender/text records."""
    normalized_suffix = suffix.lower()
    if normalized_suffix == ".json":
        try:
            return _iter_json_records(json.loads(text))
        except json.JSONDecodeError:
            return []
    if normalized_suffix == ".csv":
        return _iter_csv_records(text)
    return _iter_txt_records(text)


def _message_lengths(samples: List[str]) -> List[int]:
    return [len(sample.strip()) for sample in samples if sample.strip()]


def _emoji_count(text: str) -> int:
    count = 0
    for char in text:
        code = ord(char)
        if (
            0x1F300 <= code <= 0x1FAFF
            or 0x2600 <= code <= 0x27BF
            or char in {"😂", "😭", "🤣", "😊", "😅", "👍", "🙏"}
        ):
            count += 1
    return count


def _extract_common_phrases(samples: List[str]) -> List[str]:
    counter: Counter[str] = Counter()
    for sample in samples:
        parts = re.split(r"[，,。.!！?？、\s~～…]+", sample)
        for part in parts:
            cleaned = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", part)
            if len(cleaned) < 2:
                continue
            if 2 <= len(cleaned) <= 8:
                counter[cleaned] += 1
            for size in (2, 3, 4):
                if len(cleaned) <= size:
                    continue
                for index in range(0, len(cleaned) - size + 1):
                    phrase = cleaned[index:index + size]
                    if not phrase.isdigit():
                        counter[phrase] += 1

    phrases = []
    for phrase, count in counter.most_common(20):
        if count < 2:
            continue
        if contains_sensitive_content(phrase):
            continue
        if any(existing in phrase or phrase in existing for existing in phrases):
            continue
        phrases.append(phrase)
        if len(phrases) >= 8:
            break
    return phrases


def distill_style_from_samples(samples: List[str]) -> Dict[str, Any]:
    """Create a conservative local style summary from owner messages."""
    cleaned_samples = [_clean_import_message(sample) for sample in samples]
    cleaned_samples = [
        sample
        for sample in cleaned_samples
        if sample and not contains_sensitive_content(sample)
    ]
    if not cleaned_samples:
        return {}

    lengths = _message_lengths(cleaned_samples)
    avg_length = round(sum(lengths) / len(lengths), 1)
    sorted_lengths = sorted(lengths)
    median_length = sorted_lengths[len(sorted_lengths) // 2]
    short_ratio = sum(1 for length in lengths if length <= 12) / len(lengths)
    emoji_total = sum(_emoji_count(sample) for sample in cleaned_samples)
    emoji_ratio = emoji_total / len(cleaned_samples)

    if avg_length <= 14:
        length = f"短句为主，平均约 {avg_length} 字"
    elif avg_length <= 35:
        length = f"中短句为主，平均约 {avg_length} 字"
    else:
        length = f"较长回复较多，平均约 {avg_length} 字"

    emoji = "基本不用表情"
    if emoji_ratio >= 0.6:
        emoji = "较常使用表情，但不要过量"
    elif emoji_ratio > 0:
        emoji = "偶尔使用表情，保持克制"

    punctuation = []
    joined = "\n".join(cleaned_samples)
    if joined.count("！") + joined.count("!") >= max(2, len(cleaned_samples) * 0.25):
        punctuation.append("会用感叹号强化语气")
    if "～" in joined or "~" in joined:
        punctuation.append("会用波浪号软化语气")
    if "..." in joined or "…" in joined:
        punctuation.append("会用省略号表达停顿")

    common_phrases = _extract_common_phrases(cleaned_samples)
    habits = [
        f"平均回复约 {avg_length} 字，中位数约 {median_length} 字",
    ]
    if short_ratio >= 0.55:
        habits.append("倾向短句快速回应")
    if common_phrases:
        habits.append("常见表达：" + "、".join(common_phrases[:5]))
    habits.extend(punctuation[:3])

    tone_parts = ["自然口语化", "直接"]
    if short_ratio >= 0.55:
        tone_parts.append("偏轻量闲聊")
    if emoji_ratio == 0:
        tone_parts.append("不依赖表情")

    return {
        "tone": "、".join(tone_parts),
        "length": length,
        "emoji": emoji,
        "habits": habits,
        "common_phrases": common_phrases,
        "punctuation": punctuation,
        "stats": {
            "sample_count": len(cleaned_samples),
            "avg_length": avg_length,
            "median_length": median_length,
            "emoji_total": emoji_total,
        },
    }


class StyleProfileStore:
    """Small JSON store for owner style profiles."""

    def __init__(self, base_dir: Path | str = STYLE_PROFILE_DIR):
        self.base_dir = Path(base_dir)

    @property
    def import_inbox_dir(self) -> Path:
        return self.base_dir / "import_inbox"

    @property
    def pending_dir(self) -> Path:
        return self.base_dir / "pending_imports"

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

    def _pending_path(self, import_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", import_id.strip())
        return self.pending_dir / f"{safe_id}.json"

    def load_style_samples(self, name: str = DEFAULT_PROFILE_NAME) -> List[str]:
        """Load explicit examples only. Bulk imports store summaries, not raw lines."""
        profile = self.load(name)
        return [item["text"] for item in profile.get("examples") or [] if item.get("text")]

    def _resolve_import_path(self, source_path: str | Path) -> Path | None:
        requested = Path(str(source_path).strip().strip("\"'“”"))
        inbox = self.import_inbox_dir.resolve()
        if requested.is_absolute():
            candidate = requested.resolve()
        else:
            candidate = (inbox / requested).resolve()

        try:
            candidate.relative_to(inbox)
        except ValueError:
            return None
        return candidate

    def preview_import_file(
        self,
        source_path: str | Path,
        owner_markers: List[str],
        name: str = DEFAULT_PROFILE_NAME,
    ) -> Dict[str, Any]:
        """Preview owner-side messages from a txt/json/csv log without persisting raw lines."""
        self.import_inbox_dir.mkdir(parents=True, exist_ok=True)
        path = self._resolve_import_path(source_path)
        if path is None:
            return {
                "ok": False,
                "message": (
                    "为避免任意读取本机文件，请先把聊天记录放到 "
                    f"{self.import_inbox_dir}，再用文件名导入。"
                ),
            }
        if not path.exists() or not path.is_file():
            return {
                "ok": False,
                "message": f"没有在导入目录找到这个文件：{path.name}",
            }

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_IMPORT_EXTENSIONS:
            return {"ok": False, "message": "只支持 .txt、.json、.csv 聊天记录。"}

        try:
            size = path.stat().st_size
        except OSError:
            return {"ok": False, "message": "无法读取聊天记录文件。"}
        if size > MAX_IMPORT_FILE_BYTES:
            return {"ok": False, "message": "文件太大。v1 单个文件最多 2MB，先切小文件再导入。"}

        if not owner_markers:
            return {
                "ok": False,
                "message": "请指定哪一方是你：/风格 导入文件 <文件名> 我=<你的昵称或QQ>",
            }

        raw_text = _read_text_with_fallback(path)
        records = parse_chat_log_text(raw_text, suffix)
        if not records:
            return {"ok": False, "message": "没有解析到可用聊天记录。"}

        owner_messages = []
        skipped_sensitive = 0
        for record in records:
            sender = record.get("sender", "")
            role = record.get("role", "")
            if not _looks_like_owner(role, sender, owner_markers):
                continue

            text = _clean_import_message(record.get("text", ""))
            if not text:
                continue
            if contains_sensitive_content(text):
                skipped_sensitive += 1
                continue
            owner_messages.append(text)
            if len(owner_messages) >= MAX_IMPORTED_MESSAGES:
                break

        if not owner_messages:
            return {
                "ok": False,
                "message": "没有找到属于你的可导入消息。请检查“我=...”是否和聊天记录里的昵称/QQ一致。",
            }

        distilled = distill_style_from_samples(owner_messages)
        if not distilled:
            return {"ok": False, "message": "可用样本太少，暂时无法蒸馏风格画像。"}

        digest = hashlib.sha1(
            f"{path.name}:{size}:{len(owner_messages)}:{_now_iso()}".encode("utf-8", errors="ignore")
        ).hexdigest()[:10]
        import_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{digest}"
        pending_doc = {
            "import_id": import_id,
            "file_name": path.name,
            "created_at": _now_iso(),
            "owner_markers": owner_markers,
            "record_count": len(records),
            "message_count": len(owner_messages),
            "skipped_sensitive": skipped_sensitive,
            "distilled": distilled,
        }

        self.pending_dir.mkdir(parents=True, exist_ok=True)
        with self._pending_path(import_id).open("w", encoding="utf-8") as f:
            json.dump(pending_doc, f, ensure_ascii=False, indent=2)

        stats = distilled.get("stats", {})
        return {
            "ok": True,
            "message": (
                "导入预览："
                f"\n- 文件：{path.name}"
                f"\n- 解析记录：{len(records)} 条"
                f"\n- 你的消息：{len(owner_messages)} 条"
                f"\n- 跳过敏感消息：{skipped_sensitive} 条"
                f"\n- 平均长度：{stats.get('avg_length', 0)} 字"
                f"\n- 常见表达：{'、'.join(distilled.get('common_phrases') or []) or '未发现高频短语'}"
                f"\n确认写入画像：/风格 确认导入 {import_id}"
            ),
            "import_id": import_id,
            "message_count": len(owner_messages),
            "skipped_sensitive": skipped_sensitive,
        }

    def confirm_import(self, import_id: str, name: str = DEFAULT_PROFILE_NAME) -> tuple[bool, str]:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", import_id.strip())
        if not safe_id:
            return False, "用法：/风格 确认导入 <import_id>"
        path = self._pending_path(safe_id)
        if not path.exists():
            return False, "没有找到待确认导入。可能已经确认过，或 import_id 不正确。"

        try:
            with path.open("r", encoding="utf-8") as f:
                pending = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False, "待确认导入文件读取失败。"

        distilled = pending.get("distilled") or {}
        if not isinstance(distilled, dict):
            return False, "待确认导入内容无效。"

        profile = self.load(name)
        for field in ("tone", "length", "emoji"):
            if distilled.get(field):
                profile[field] = distilled[field]

        for field in ("habits", "common_phrases", "punctuation"):
            merged = _normalize_list((profile.get(field) or []) + (distilled.get(field) or []))
            profile[field] = merged

        profile["stats"] = distilled.get("stats") or {}
        profile.setdefault("source_imports", []).append({
            "import_id": pending.get("import_id", safe_id),
            "file_name": str(pending.get("file_name", ""))[:160],
            "imported_at": _now_iso(),
            "message_count": int(pending.get("message_count") or 0),
            "skipped_sensitive": int(pending.get("skipped_sensitive") or 0),
        })
        self.save(profile, name)

        try:
            path.unlink()
        except OSError:
            pass

        return True, (
            "已确认导入并更新风格画像。"
            f"\n- 文件：{pending.get('file_name', '')}"
            f"\n- 样本数：{pending.get('message_count', 0)}"
            f"\n- 常见表达：{'、'.join(profile.get('common_phrases') or []) or '未发现高频短语'}"
        )

    def distill(self, name: str = DEFAULT_PROFILE_NAME) -> tuple[bool, str]:
        samples = self.load_style_samples(name)
        distilled = distill_style_from_samples(samples)
        if not distilled:
            return False, "还没有足够的风格样本。先用 /风格 导入 或 /风格 导入文件 添加样本。"

        profile = self.load(name)
        profile.update(distilled)
        self.save(profile, name)

        stats = distilled.get("stats", {})
        return True, (
            "风格画像已蒸馏更新："
            f"\n- 样本数：{stats.get('sample_count', 0)}"
            f"\n- 平均长度：{stats.get('avg_length', 0)} 字"
            f"\n- 常见表达：{'、'.join(distilled.get('common_phrases') or []) or '未发现高频短语'}"
        )

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
    imports = data.get("source_imports") or []
    offline_runs = data.get("offline_distillations") or []
    stats = data.get("stats") or {}
    common_phrases = data.get("common_phrases") or []
    lines.append(f"- 习惯：{'；'.join(habits) if habits else '未设置'}")
    lines.append(f"- 避免：{'；'.join(avoid) if avoid else '未设置'}")
    if common_phrases:
        lines.append(f"- 常见表达：{'、'.join(common_phrases[:8])}")
    if stats:
        lines.append(
            f"- 蒸馏统计：{stats.get('sample_count', 0)} 条样本，"
            f"平均 {stats.get('avg_length', 0)} 字"
        )
        if stats.get("candidate_reply_pairs") is not None:
            lines.append(
                f"- Stage 5B：候选 {stats.get('candidate_reply_pairs', 0)} 条，"
                f"索引 {stats.get('indexed_samples', 0)} 条"
            )
    lines.append(f"- 文件导入：{len(imports)} 个")
    lines.append(f"- 离线蒸馏：{len(offline_runs)} 次")
    lines.append(f"- 样本数：{len(examples)}")

    for index, example in enumerate(examples[-3:], start=max(1, len(examples) - 2)):
        preview = example["text"].replace("\n", " ")[:80]
        lines.append(f"  {index}. {preview}")

    lines.append("用法：/风格 导入文件 <文件名> 我=<昵称或QQ>；/风格 蒸馏；/用我的风格回复：<对方消息>")
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
        "import_file": ("导入文件", "文件导入", "import-file", "importfile", "file"),
        "confirm_import": ("确认导入", "confirm-import", "confirm import"),
        "distill": ("蒸馏", "重建", "distill", "rebuild"),
        "offline_distill": ("离线蒸馏", "qce蒸馏", "qce 蒸馏", "offline-distill", "offline distill"),
        "evaluation": ("评估", "评价", "eval", "evaluate"),
        "relationships": ("关系", "关系画像", "relationship", "relationships"),
        "scenes": ("场景", "场景画像", "scene", "scenes"),
        "retrieve": ("检索", "相似", "相似样本", "retrieve", "search"),
        "raw_fewshot": ("原句", "原文", "真实原句", "fewshot", "few-shot"),
        "auto_reply": ("自动回复", "代聊", "auto-reply", "autoreply", "auto reply"),
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
        "- /风格 导入文件 <文件名> 我=<你的昵称或QQ>：文件需放在 data/style_profiles/import_inbox/",
        "- /风格 确认导入 <import_id>：确认预览并写入画像",
        "- /风格 蒸馏：从已导入样本更新画像摘要",
        "- /风格 离线蒸馏：从 QCE JSON 离线生成 Stage 5B 摘要和样本索引",
        "- /风格 评估：查看 Stage 5B 数据就绪度和评估摘要",
        "- /风格 关系：查看关系/场景来源画像摘要",
        "- /风格 场景：查看场景画像摘要",
        "- /风格 检索 <当前对方消息>：本地检索相似历史样本索引，不返回历史原文",
        "- /风格 原句 开/关：控制真实历史原句 few-shot，开启需要二次确认并写审计",
        "- /风格 自动回复 开/关：控制信任名单内的 owner-style 代聊自动回复",
        "- /风格 清空样本 确认：删除已导入样本",
        "- /用我的风格回复：<对方消息>：生成一条草稿；若有 Stage 5B 结果，会接入检索/关系/场景元数据",
    ])


def format_generation_context_for_prompt(context: Dict[str, Any] | None) -> str:
    """Format Stage 5B metadata for the draft prompt without raw history text."""
    if not context or not context.get("ok"):
        return ""

    lines = [
        "Stage 5B 生成上下文：",
        f"- run_id：{context.get('run_id')}",
        f"- 数据就绪度：{context.get('readiness')}",
    ]
    if context.get("raw_fewshot_included"):
        lines.append("- 历史原文策略：已由主人授权，将少量真实历史原句作为 few-shot 提供给模型；不要照抄隐私事实。")
    else:
        lines.append("- 历史原文策略：不向模型提供历史原文；只使用相似度、长度、场景、关系标签等元数据。")
    target_mapping = context.get("target_mapping") or {}
    if target_mapping.get("matched"):
        lines.append(
            "- 当前对象映射：已匹配到本地关系画像 "
            f"{target_mapping.get('source_file_id')} ({target_mapping.get('chat_type')})；不暴露 QQ/群号。"
        )
    guidance = context.get("guidance") or {}
    if guidance:
        lines.append("生成策略：")
        if guidance.get("length_instruction"):
            lines.append(f"- 长度：{guidance['length_instruction']}，目标约 {guidance.get('target_reply_length')} 字")
        if guidance.get("stance_instruction"):
            lines.append(f"- 应对方式：{guidance['stance_instruction']}")
        if guidance.get("reality_policy"):
            lines.append("- 现实状态：不要编造主人是否忙、在哪、是否完成、是否答应等事实。")

    query_features = context.get("query_features") or {}
    if query_features:
        lines.append(
            "- 当前消息特征："
            f"长度桶={query_features.get('length_bucket')}，"
            f"问句={query_features.get('has_question')}，"
            f"感叹={query_features.get('has_exclamation')}，"
            f"省略={query_features.get('has_ellipsis')}"
        )

    samples = context.get("similar_samples") or []
    if samples:
        lines.append("相似历史样本索引摘要：")
        for index, item in enumerate(samples[:5], start=1):
            lines.append(
                f"- {index}. sim={item.get('similarity')} q={item.get('quality_score')} "
                f"{item.get('chat_type')} reply_len={item.get('reply_char_length')} "
                f"bucket={item.get('reply_length_bucket')} ctx={item.get('context_count')}"
            )

    relationships = context.get("relationship_profiles") or []
    if relationships:
        lines.append("关系/来源画像摘要：")
        for item in relationships[:3]:
            labels = "、".join((item.get("labels") or [])[:5])
            lines.append(
                f"- {item.get('chat_type')} owner_msgs={item.get('owner_text_messages')} "
                f"samples={item.get('candidate_samples')} avg_len={item.get('avg_length')} "
                f"labels={labels}"
            )

    scenes = context.get("scene_profiles") or []
    if scenes:
        lines.append("场景画像摘要：")
        for item in scenes[:3]:
            lines.append(
                f"- {item.get('scene_id')} count={item.get('sample_count')} "
                f"avg_len={item.get('avg_reply_length')} quick={item.get('quick_reply_ratio')} "
                f"style={item.get('recommended_style')}"
            )

    examples = context.get("few_shot_examples") or []
    if examples:
        lines.append("真实历史 few-shot 样本：")
        lines.append("只学习“对方上下文 -> 主人回复”的表达映射，不要照抄其中的具体事实、姓名、时间、地点或承诺。")
        for index, example in enumerate(examples[:3], start=1):
            lines.append(f"样本 {index}：")
            for item in example.get("context") or []:
                role = item.get("role") or "对方"
                lines.append(f"- {role}：{item.get('text')}")
            lines.append(f"- 主人：{example.get('owner_reply')}")

    return "\n".join(lines)


def format_recent_dialogue_for_prompt(messages: List[Dict[str, Any]] | None, limit: int = 8) -> str:
    """Format short recent dialogue context for immediate generation only."""
    if not messages:
        return ""
    lines = ["最近对话（用于理解上下文；不要机械复读上一条回复）："]
    for item in messages[-max(1, int(limit)):]:
        if not isinstance(item, dict):
            continue
        role = "主人" if item.get("role") == "assistant" else "对方"
        text = re.sub(r"\s+", " ", str(item.get("content") or "")).strip()
        if not text:
            continue
        lines.append(f"- {role}：{text[:160]}")
    return "\n".join(lines) if len(lines) > 1 else ""


def build_style_system_prompt(
    profile: Dict[str, Any],
    generation_context: Dict[str, Any] | None = None,
) -> str:
    data = normalize_style_profile(profile)
    lines = [
        "你是一个回复草稿生成器，任务是模仿“主人”的日常聊天风格生成一条可复制的中文回复草稿。",
        "只输出草稿正文，不要解释，不要加标题。",
        "不要声称自己是机器人；不要代替主人承诺无法确认的现实行动；不要编造事实。",
        "必须先理解并回应对方当前消息，不要用万能寒暄敷衍。",
        "如果对方询问主人当前状态、位置、是否有空、是否完成某事等未知现实事实，生成自然但不确认事实的草稿，例如“咋啦”“啥事”“我看下”，不要直接替主人回答“在家/不忙/已经做了”。",
        "不要连续重复同一句或同一个口头禅；如果最近已经说过“刚看到”，下一条不要再用。",
        "遇到[表情]、[动画表情]、[图片]时，可以按语气接话，但不要假装看清图片具体内容。",
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
    if data.get("common_phrases"):
        lines.append("可参考的常见表达；只当风味，不要连续照搬：")
        lines.extend(f"- {item}" for item in data["common_phrases"][:8])
    if data.get("punctuation"):
        lines.append("标点习惯：")
        lines.extend(f"- {item}" for item in data["punctuation"][:5])
    if data.get("examples"):
        lines.append("主人真实回复样本，仅学习表达风格，不要照抄隐私内容：")
        for example in data["examples"][-5:]:
            lines.append(f"- {example['text']}")
    context_prompt = format_generation_context_for_prompt(generation_context)
    if context_prompt:
        lines.append(context_prompt)
    return "\n".join(lines)


def _audit_style_generation(
    *,
    actor_id: str | int | None,
    target_id: str | int | None,
    scope: str,
    context: Dict[str, Any] | None,
    auto_reply: bool,
) -> None:
    if not context or not context.get("ok"):
        return
    raw_examples = context.get("few_shot_examples") or []
    if not raw_examples and not auto_reply:
        return
    try:
        from .confirmation import confirmation_store
        sample_refs = []
        for item in raw_examples[:5]:
            sample_refs.append({
                "sample_id": str(item.get("sample_id") or "")[:24],
                "source_file_id": str(item.get("source_file_id") or "")[:32],
                "chat_type": str(item.get("chat_type") or "")[:16],
                "char_counts": item.get("char_counts") or {},
            })
        action_type = "style_auto_reply" if auto_reply else "style_raw_fewshot_prompt"
        if auto_reply and raw_examples:
            action_type = "style_auto_reply_raw_fewshot"
        result = json.dumps(
            {
                "run_id": context.get("run_id"),
                "scope": scope,
                "target_hash": hashlib.sha1(str(target_id or "").encode("utf-8")).hexdigest()[:12],
                "raw_fewshot_count": len(raw_examples),
                "samples": sample_refs,
                "target_mapping": context.get("target_mapping") or {},
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        confirmation_store.log(
            {
                "id": hashlib.sha1(result.encode("utf-8", errors="ignore")).hexdigest()[:8],
                "type": action_type,
                "summary": (
                    "生成 owner-style 自动回复" if auto_reply
                    else "生成 owner-style 草稿时使用真实历史 few-shot"
                ),
            },
            actor_id=actor_id or "",
            status="executed",
            result=result,
        )
    except Exception:
        return


async def generate_style_draft(
    message: str,
    store: StyleProfileStore = style_store,
    *,
    include_raw_fewshot: bool | None = None,
    chat_type: str | None = None,
    target_id: str | int | None = None,
    actor_id: str | int | None = None,
    scope: str = "private",
    auto_reply: bool = False,
    recent_dialogue: List[Dict[str, Any]] | None = None,
) -> str:
    target = message.strip()
    if not target:
        return "用法：/用我的风格回复：<对方消息>"
    if len(target) > 1000:
        return "对方消息太长。v1 先支持 1000 字以内的单条草稿。"

    profile = store.load()
    generation_context = None
    try:
        from .style_distill import build_style_generation_context
        if include_raw_fewshot is None:
            from .runtime_state import is_style_raw_fewshot_enabled
            include_raw_fewshot = is_style_raw_fewshot_enabled()
        generation_context = await asyncio.to_thread(
            build_style_generation_context,
            target,
            chat_type=chat_type,
            target_id=target_id,
            include_raw_fewshot=bool(include_raw_fewshot),
        )
    except Exception:
        generation_context = None
    _audit_style_generation(
        actor_id=actor_id,
        target_id=target_id,
        scope=scope,
        context=generation_context,
        auto_reply=auto_reply,
    )
    user_prompt_parts = []
    recent_prompt = format_recent_dialogue_for_prompt(recent_dialogue)
    if recent_prompt:
        user_prompt_parts.append(recent_prompt)
    user_prompt_parts.extend([
        f"对方新消息：{target}",
        "生成主人下一条自然回复。需要结合最近对话，避免重复上一条回复。",
    ])
    return await llm_client.chat(
        messages=[{"role": "user", "content": "\n".join(user_prompt_parts)}],
        system_prompt=build_style_system_prompt(profile, generation_context),
        temperature=0.6,
    )
