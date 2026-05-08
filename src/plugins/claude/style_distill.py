"""Offline QCE chat-log distillation for owner style profiles.

This module reads QCE JSON exports and writes only aggregate style summaries
plus message-id based sample indexes. It must not persist raw chat text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Sequence

from .auto_memory import contains_sensitive_content
from .style_profile import DEFAULT_PROFILE_NAME, StyleProfileStore, style_store


DEFAULT_SELF_UIN = "1030400950"
DEFAULT_EXPORT_ROOT = Path(r"F:\ClaudeSpace2\qq-chat-exports")
DEFAULT_QCE_INPUT_DIR = (
    DEFAULT_EXPORT_ROOT
    / "main_1030400950_20260509"
    / "distill-json"
    / "recent_51_text"
)
MAX_INDEX_SAMPLES = 5000
MAX_REPLY_LENGTH = 280
MAX_CONTEXT_MESSAGES = 8
CONTEXT_WINDOW_SECONDS = 30 * 60

TEXT_PLACEHOLDER_RE = re.compile(r"^\s*\[(?:图片|视频|语音|表情|文件|动画表情|转发消息).*\]\s*$")
URL_RE = re.compile(r"https?://|www\.", flags=re.I)
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]",
    flags=re.UNICODE,
)


@dataclass(frozen=True)
class QceDistillOptions:
    input_dir: Path = DEFAULT_QCE_INPUT_DIR
    output_root: Path | None = None
    self_uin: str = DEFAULT_SELF_UIN
    max_index_samples: int = MAX_INDEX_SAMPLES
    context_window_seconds: int = CONTEXT_WINDOW_SECONDS
    max_context_messages: int = MAX_CONTEXT_MESSAGES
    apply_to_profile: bool = True
    profile_name: str = DEFAULT_PROFILE_NAME


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sha1_short(text: str, size: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:size]


def _as_path(value: str | Path | None, default: Path) -> Path:
    if value is None:
        return default
    text = str(value).strip().strip("\"'“”")
    if not text or text in {"默认", "default"}:
        return default
    return Path(text)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_qce_input_dir(input_dir: str | Path | None = None) -> Path:
    """Resolve and validate a QCE JSON input directory.

    Owner-only commands still avoid arbitrary local reads: the default allowed
    locations are the F-drive QCE export root and this project's data folder.
    """
    candidate = _as_path(input_dir, DEFAULT_QCE_INPUT_DIR)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    resolved = candidate.resolve()
    allowed_roots = [
        DEFAULT_EXPORT_ROOT.resolve(),
        (Path.cwd() / "data").resolve(),
    ]
    if not any(_is_relative_to(resolved, root) for root in allowed_roots):
        raise ValueError("离线蒸馏只允许读取 F:\\ClaudeSpace2\\qq-chat-exports 或项目 data 目录下的文件。")
    if not resolved.exists() or not resolved.is_dir():
        raise FileNotFoundError(f"没有找到 QCE JSON 目录：{resolved}")
    if not any(resolved.glob("*.json")):
        raise FileNotFoundError(f"目录中没有 JSON 文件：{resolved}")
    return resolved


def default_output_root_for(input_dir: Path) -> Path:
    if input_dir.parent.name == "distill-json":
        return input_dir.parent.parent / "distill-runs"
    return input_dir.parent / "distill-runs"


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content") or {}
    if not isinstance(content, dict):
        return ""
    return str(content.get("text") or "").strip()


def _message_elements(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = message.get("content") or {}
    if not isinstance(content, dict):
        return []
    elements = content.get("elements") or []
    if not isinstance(elements, list):
        return []
    return [item for item in elements if isinstance(item, dict)]


def _element_types(message: Dict[str, Any]) -> List[str]:
    types = []
    for element in _message_elements(message):
        text = str(element.get("type") or "").strip()
        if text:
            types.append(text[:32])
    return sorted(set(types))


def _sender_uin(message: Dict[str, Any]) -> str:
    sender = message.get("sender") or {}
    if not isinstance(sender, dict):
        return ""
    return str(sender.get("uin") or "").strip()


def _timestamp(message: Dict[str, Any]) -> int:
    try:
        return int(message.get("timestamp") or 0)
    except (TypeError, ValueError):
        return 0


def _is_system_or_recalled(message: Dict[str, Any]) -> bool:
    return bool(message.get("system")) or bool(message.get("recalled"))


def _is_useful_text(text: str) -> bool:
    if not text:
        return False
    if TEXT_PLACEHOLDER_RE.match(text):
        return False
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 2:
        return False
    if not re.search(r"[\w\u4e00-\u9fff]", compact):
        return False
    return True


def _bucket_length(length: int) -> str:
    if length <= 2:
        return "1-2"
    if length <= 6:
        return "3-6"
    if length <= 12:
        return "7-12"
    if length <= 30:
        return "13-30"
    if length <= 80:
        return "31-80"
    return "81+"


def _features_for_text(text: str) -> Dict[str, Any]:
    return {
        "char_length": len(text),
        "length_bucket": _bucket_length(len(text)),
        "emoji_count": len(EMOJI_RE.findall(text)),
        "has_question": "?" in text or "？" in text,
        "has_exclamation": "!" in text or "！" in text,
        "has_ellipsis": "..." in text or "…" in text,
        "has_url": bool(URL_RE.search(text)),
        "line_count": max(1, text.count("\n") + 1),
    }


def _time_bucket(timestamp: int) -> str:
    if timestamp <= 0:
        return ""
    if timestamp > 10_000_000_000:
        timestamp = timestamp // 1000
    try:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:00")
    except (OSError, OverflowError, ValueError):
        return ""


def _message_ref(message: Dict[str, Any], record_index: int) -> Dict[str, Any]:
    timestamp = _timestamp(message)
    return {
        "record_index": record_index,
        "id_hash": _sha1_short(str(message.get("id") or ""), 12),
        "seq_hash": _sha1_short(str(message.get("seq") or ""), 12),
        "time_bucket": _time_bucket(timestamp),
    }


def _context_for(
    messages: Sequence[Dict[str, Any]],
    index: int,
    self_uin: str,
    window_seconds: int,
    max_messages: int,
) -> tuple[List[tuple[int, Dict[str, Any]]], int | None]:
    reply_time = _timestamp(messages[index])
    context: List[tuple[int, Dict[str, Any]]] = []
    first_non_self_delay: int | None = None

    for j in range(index - 1, -1, -1):
        previous = messages[j]
        if _is_system_or_recalled(previous):
            continue
        previous_time = _timestamp(previous)
        if reply_time and previous_time and reply_time - previous_time > window_seconds:
            break
        text = _message_text(previous)
        if not _is_useful_text(text):
            continue
        is_self = _sender_uin(previous) == self_uin
        if not is_self and first_non_self_delay is None and reply_time and previous_time:
            first_non_self_delay = max(0, reply_time - previous_time)
        context.append((j, previous))
        if len(context) >= max_messages:
            break

    context.reverse()
    return context, first_non_self_delay


def _score_candidate(
    *,
    chat_type: str,
    text: str,
    context: Sequence[Dict[str, Any]],
    first_non_self_delay: int | None,
    element_types: Sequence[str],
) -> tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    length = len(text)

    if first_non_self_delay is not None:
        score += 40
        reasons.append("has_recent_other_context")
        if first_non_self_delay <= 120:
            score += 8
            reasons.append("quick_reply")
    else:
        score -= 25
        reasons.append("no_recent_other_context")

    if chat_type == "private":
        score += 18
        reasons.append("private_chat")
    elif chat_type == "group":
        score += 6
        reasons.append("group_chat")

    if "reply" in element_types:
        score += 18
        reasons.append("explicit_reply")
    if "at" in element_types:
        score += 5
        reasons.append("mentions")
    if len(context) >= 2:
        score += min(12, len(context) * 2)
        reasons.append("multi_message_context")

    if 3 <= length <= 80:
        score += 16
        reasons.append("good_length")
    elif 81 <= length <= MAX_REPLY_LENGTH:
        score += 8
        reasons.append("long_but_usable")
    else:
        score -= 15
        reasons.append("weak_length")

    features = _features_for_text(text)
    if features["has_url"]:
        score -= 20
        reasons.append("contains_url")
    if features["line_count"] > 3:
        score -= 8
        reasons.append("multi_line")
    if features["emoji_count"] > 0:
        score += 3
        reasons.append("style_marker_emoji")
    if features["has_question"]:
        score += 2
        reasons.append("question_style")
    if features["has_exclamation"] or features["has_ellipsis"]:
        score += 2
        reasons.append("punctuation_style")

    return score, reasons


def _percentile(values: Sequence[int], ratio: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * ratio) - 1))
    return int(ordered[index])


def _merge_counter(target: Counter, values: Iterable[str]) -> None:
    for value in values:
        if value:
            target[value] += 1


def _style_patch_from_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    owner = stats["owner"]
    lengths = owner["lengths"]
    avg_length = round(sum(lengths) / len(lengths), 1) if lengths else 0
    median_length = int(median(lengths)) if lengths else 0
    short_ratio = owner["length_buckets"].get("1-2", 0) + owner["length_buckets"].get("3-6", 0)
    short_ratio = short_ratio / max(1, owner["text_messages"])
    emoji_ratio = owner["emoji_messages"] / max(1, owner["text_messages"])
    private_count = owner["by_chat_type"].get("private", 0)
    group_count = owner["by_chat_type"].get("group", 0)

    if avg_length <= 10:
        length = f"短句为主，平均约 {avg_length} 字，中位数约 {median_length} 字"
    elif avg_length <= 28:
        length = f"中短句为主，平均约 {avg_length} 字，中位数约 {median_length} 字"
    else:
        length = f"中等长度回复较多，平均约 {avg_length} 字，中位数约 {median_length} 字"

    if emoji_ratio >= 0.35:
        emoji = "会使用表情/贴图语气，但仍以文字为主"
    elif emoji_ratio >= 0.08:
        emoji = "偶尔使用表情或贴图，保持克制"
    else:
        emoji = "很少显式使用 emoji；需要时可用短文本表达语气"

    tone_parts = ["口语化", "直接", "低铺垫"]
    if short_ratio >= 0.45:
        tone_parts.append("偏短句接话")
    if group_count > private_count * 2:
        tone_parts.append("群聊里更像插话和接梗")

    habits = [
        f"从 {owner['text_messages']} 条本人文本消息蒸馏，只保存统计特征，不保存原文",
        f"平均回复约 {avg_length} 字，中位数约 {median_length} 字",
        "优先顺着上下文直接回应，少用正式开场",
    ]
    if short_ratio >= 0.45:
        habits.append("大量回复是短句，适合轻量闲聊和快速确认")
    if group_count > private_count:
        habits.append("群聊发言多，常见形态是短插话、接话、反应和补充")
    if private_count:
        habits.append("私聊里比群聊更适合保留上下文和具体回应")
    if owner["question_messages"] / max(1, owner["text_messages"]) >= 0.08:
        habits.append("会用反问或追问推进对话")
    if owner["ellipsis_messages"] / max(1, owner["text_messages"]) >= 0.03:
        habits.append("会用省略号表现停顿或无语感")

    punctuation = []
    if owner["question_messages"]:
        punctuation.append("会使用问号表达追问或反问")
    if owner["exclamation_messages"]:
        punctuation.append("偶尔用感叹号加强情绪")
    if owner["ellipsis_messages"]:
        punctuation.append("会使用省略号表达停顿")

    return {
        "tone": "、".join(tone_parts),
        "length": length,
        "emoji": emoji,
        "habits": habits,
        "common_phrases": [],
        "punctuation": punctuation,
        "stats": {
            "source": "qce_offline_stage5b",
            "sample_count": owner["text_messages"],
            "candidate_reply_pairs": stats["samples"]["candidate_count"],
            "indexed_samples": stats["samples"]["indexed_count"],
            "avg_length": avg_length,
            "median_length": median_length,
            "p90_length": _percentile(lengths, 0.9),
            "private_self_messages": private_count,
            "group_self_messages": group_count,
        },
    }


def run_qce_style_distillation(
    input_dir: str | Path | None = None,
    *,
    output_root: str | Path | None = None,
    self_uin: str = DEFAULT_SELF_UIN,
    max_index_samples: int = MAX_INDEX_SAMPLES,
    apply_to_profile: bool = True,
    store: StyleProfileStore = style_store,
    profile_name: str = DEFAULT_PROFILE_NAME,
) -> Dict[str, Any]:
    """Run the offline distillation pipeline and optionally update style profile."""
    resolved_input = resolve_qce_input_dir(input_dir)
    resolved_output_root = Path(output_root).resolve() if output_root else default_output_root_for(resolved_input).resolve()
    run_id = f"stage5b_{_now_id()}_{_sha1_short(str(resolved_input), 8)}"
    output_dir = resolved_output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    stats: Dict[str, Any] = {
        "files": {"count": 0, "private": 0, "group": 0},
        "messages": {"total": 0, "private": 0, "group": 0, "system_or_recalled": 0},
        "owner": {
            "total_messages": 0,
            "text_messages": 0,
            "private": 0,
            "group": 0,
            "sensitive_skipped": 0,
            "lengths": [],
            "length_buckets": Counter(),
            "element_types": Counter(),
            "by_chat_type": Counter(),
            "emoji_messages": 0,
            "question_messages": 0,
            "exclamation_messages": 0,
            "ellipsis_messages": 0,
            "url_messages": 0,
        },
        "all_elements": Counter(),
        "samples": {"candidate_count": 0, "indexed_count": 0},
    }
    candidates: List[Dict[str, Any]] = []
    source_catalog: List[Dict[str, Any]] = []
    seen_reply_text = set()
    input_files = sorted(resolved_input.glob("*.json"))

    for source_number, source_path in enumerate(input_files, start=1):
        try:
            data = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        chat_info = data.get("chatInfo") or {}
        chat_type = str(chat_info.get("type") or "unknown")
        messages = data.get("messages") or []
        if not isinstance(messages, list):
            continue

        stats["files"]["count"] += 1
        if chat_type in {"private", "group"}:
            stats["files"][chat_type] += 1

        source_relpath = str(source_path.relative_to(resolved_input.parent))
        source_file_id = f"source_{source_number:04d}_{_sha1_short(source_relpath, 8)}"
        source_catalog.append({
            "source_file_id": source_file_id,
            "relative_path": source_relpath,
            "file_size_bytes": source_path.stat().st_size,
            "chat_type": chat_type,
            "message_count": len(messages),
            "note": "Private local catalog for reproducing sample indexes; no chat text is stored here.",
        })

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            if _is_system_or_recalled(message):
                stats["messages"]["system_or_recalled"] += 1
                continue

            stats["messages"]["total"] += 1
            if chat_type in {"private", "group"}:
                stats["messages"][chat_type] += 1
            _merge_counter(stats["all_elements"], _element_types(message))

            is_owner = _sender_uin(message) == str(self_uin)
            if not is_owner:
                continue

            stats["owner"]["total_messages"] += 1
            stats["owner"]["by_chat_type"][chat_type] += 1
            if chat_type in {"private", "group"}:
                stats["owner"][chat_type] += 1

            text = _message_text(message)
            if not _is_useful_text(text):
                continue
            if contains_sensitive_content(text):
                stats["owner"]["sensitive_skipped"] += 1
                continue

            text_features = _features_for_text(text)
            stats["owner"]["text_messages"] += 1
            stats["owner"]["lengths"].append(text_features["char_length"])
            stats["owner"]["length_buckets"][text_features["length_bucket"]] += 1
            if text_features["emoji_count"]:
                stats["owner"]["emoji_messages"] += 1
            if text_features["has_question"]:
                stats["owner"]["question_messages"] += 1
            if text_features["has_exclamation"]:
                stats["owner"]["exclamation_messages"] += 1
            if text_features["has_ellipsis"]:
                stats["owner"]["ellipsis_messages"] += 1
            if text_features["has_url"]:
                stats["owner"]["url_messages"] += 1
            element_types = _element_types(message)
            _merge_counter(stats["owner"]["element_types"], element_types)

            if len(text) > MAX_REPLY_LENGTH:
                continue
            text_digest = _sha1_short(text, 20)
            if text_digest in seen_reply_text:
                continue
            seen_reply_text.add(text_digest)

            context, first_non_self_delay = _context_for(
                messages,
                index,
                str(self_uin),
                CONTEXT_WINDOW_SECONDS,
                MAX_CONTEXT_MESSAGES,
            )
            if first_non_self_delay is None:
                continue

            score, reasons = _score_candidate(
                chat_type=chat_type,
                text=text,
                context=context,
                first_non_self_delay=first_non_self_delay,
                element_types=element_types,
            )
            if score < 50:
                continue

            stats["samples"]["candidate_count"] += 1
            sample_id = _sha1_short(
                f"{source_file_id}:{index}:{message.get('id')}:{message.get('seq')}:{_timestamp(message)}",
                20,
            )
            context_refs = []
            for context_index, item in context:
                role = "self" if _sender_uin(item) == str(self_uin) else "other"
                context_refs.append({
                    **_message_ref(item, context_index),
                    "role": role,
                    "element_types": _element_types(item),
                })
            candidates.append({
                "sample_id": sample_id,
                "source_file_id": source_file_id,
                "chat_type": chat_type,
                "reply": {
                    **_message_ref(message, index),
                    "char_length": text_features["char_length"],
                    "length_bucket": text_features["length_bucket"],
                    "element_types": element_types,
                    "features": {
                        "emoji_count": text_features["emoji_count"],
                        "has_question": text_features["has_question"],
                        "has_exclamation": text_features["has_exclamation"],
                        "has_ellipsis": text_features["has_ellipsis"],
                        "line_count": text_features["line_count"],
                    },
                },
                "context": {
                    "count": len(context_refs),
                    "first_non_self_delay_seconds": first_non_self_delay,
                    "messages": context_refs,
                },
                "score": score,
                "score_reasons": reasons,
            })

    candidates.sort(
        key=lambda item: (
            -int(item["score"]),
            str(item["source_file_id"]),
            int(item["reply"]["record_index"]),
            item["sample_id"],
        )
    )
    indexed = candidates[: max(0, int(max_index_samples))]
    stats["samples"]["indexed_count"] = len(indexed)

    style_patch = _style_patch_from_stats(stats)
    public_stats = {
        "files": stats["files"],
        "messages": stats["messages"],
        "owner": {
            key: (dict(value) if isinstance(value, Counter) else value)
            for key, value in stats["owner"].items()
            if key != "lengths"
        },
        "owner_length": {
            "avg": style_patch["stats"]["avg_length"],
            "median": style_patch["stats"]["median_length"],
            "p90": style_patch["stats"]["p90_length"],
            "buckets": dict(stats["owner"]["length_buckets"]),
        },
        "all_element_types": dict(stats["all_elements"].most_common(30)),
        "samples": stats["samples"],
    }
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": _now_iso(),
        "input_dir_id": _sha1_short(str(resolved_input), 12),
        "self_uin_hash": _sha1_short(str(self_uin), 12),
        "raw_text_policy": "No raw chat text is stored in this summary or sample index.",
        "sample_index": "sample_index.jsonl",
        "source_catalog": "source_catalog_private.json",
        "stats": public_stats,
        "style_profile_patch": style_patch,
    }

    summary_path = output_dir / "style_profile_summary.json"
    patch_path = output_dir / "style_profile_patch.json"
    index_path = output_dir / "sample_index.jsonl"
    index_summary_path = output_dir / "sample_index_summary.json"
    source_catalog_path = output_dir / "source_catalog_private.json"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    patch_path.write_text(json.dumps(style_patch, ensure_ascii=False, indent=2), encoding="utf-8")
    with index_path.open("w", encoding="utf-8") as f:
        for item in indexed:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    index_summary_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "raw_text_policy": summary["raw_text_policy"],
                "candidate_count": stats["samples"]["candidate_count"],
                "indexed_count": len(indexed),
                "min_score": min((int(item["score"]) for item in indexed), default=0),
                "max_score": max((int(item["score"]) for item in indexed), default=0),
                "score_buckets": dict(Counter(str((int(item["score"]) // 10) * 10) for item in indexed)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    source_catalog_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "raw_text_policy": summary["raw_text_policy"],
                "privacy_note": (
                    "This private catalog maps source_file_id to local QCE export files. "
                    "It may contain QQ/group identifiers in filenames but no chat text."
                ),
                "input_dir": str(resolved_input),
                "self_uin": str(self_uin),
                "sources": source_catalog,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    applied = None
    if apply_to_profile:
        applied = apply_qce_distillation_summary(summary_path, store=store, profile_name=profile_name)

    return {
        "ok": True,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "patch_path": str(patch_path),
        "sample_index_path": str(index_path),
        "indexed_samples": len(indexed),
        "candidate_samples": stats["samples"]["candidate_count"],
        "owner_text_messages": stats["owner"]["text_messages"],
        "total_messages": stats["messages"]["total"],
        "input_files": stats["files"]["count"],
        "applied": applied,
    }


def apply_qce_distillation_summary(
    summary_path: str | Path,
    *,
    store: StyleProfileStore = style_store,
    profile_name: str = DEFAULT_PROFILE_NAME,
) -> Dict[str, Any]:
    path = Path(summary_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    patch = data.get("style_profile_patch") or {}
    if not isinstance(patch, dict):
        raise ValueError("离线蒸馏摘要缺少 style_profile_patch。")

    profile = store.load(profile_name)
    for field in ("tone", "length", "emoji"):
        if patch.get(field):
            profile[field] = patch[field]

    for field in ("habits", "punctuation"):
        merged = []
        seen = set()
        for item in (profile.get(field) or []) + (patch.get(field) or []):
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text[:80])
        profile[field] = merged[:20]

    profile["stats"] = patch.get("stats") or profile.get("stats") or {}

    existing_runs = []
    for item in profile.get("offline_distillations") or []:
        if not isinstance(item, dict):
            continue
        run_id = str(item.get("run_id") or "").strip()
        if not run_id:
            continue
        existing_runs.append({
            "run_id": run_id,
            "applied_at": str(item.get("applied_at") or "").strip(),
            "summary_file": str(item.get("summary_file") or "style_profile_summary.json").strip(),
            "sample_index_file": str(item.get("sample_index_file") or "sample_index.jsonl").strip(),
            "owner_text_messages": int(item.get("owner_text_messages") or 0),
            "indexed_samples": int(item.get("indexed_samples") or 0),
            "raw_text_policy": str(item.get("raw_text_policy") or "").strip(),
        })

    existing_runs.append({
        "run_id": str(data.get("run_id") or ""),
        "applied_at": _now_iso(),
        "summary_file": "style_profile_summary.json",
        "sample_index_file": str(data.get("sample_index") or "sample_index.jsonl"),
        "owner_text_messages": int((data.get("stats") or {}).get("owner", {}).get("text_messages") or 0),
        "indexed_samples": int((data.get("stats") or {}).get("samples", {}).get("indexed_count") or 0),
        "raw_text_policy": str(data.get("raw_text_policy") or ""),
    })
    profile["offline_distillations"] = existing_runs[-20:]
    saved = store.save(profile, profile_name)
    return {
        "profile_path": str(store.profile_path(profile_name)),
        "offline_distillation_count": len(saved.get("offline_distillations") or []),
        "stats": saved.get("stats") or {},
    }


def format_qce_distillation_result(result: Dict[str, Any]) -> str:
    if not result.get("ok"):
        return f"离线蒸馏失败：{result.get('error') or 'unknown'}"
    applied = result.get("applied") or {}
    lines = [
        "Stage 5B 离线蒸馏完成：",
        f"- run_id：{result.get('run_id')}",
        f"- 输入文件：{result.get('input_files')} 个",
        f"- 总消息：{result.get('total_messages')} 条",
        f"- 本人文本：{result.get('owner_text_messages')} 条",
        f"- 候选样本：{result.get('candidate_samples')} 条",
        f"- 索引样本：{result.get('indexed_samples')} 条",
        "- 原文策略：摘要和索引不保存聊天正文",
        f"- 输出目录：{result.get('output_dir')}",
    ]
    if applied:
        lines.append(f"- 已更新画像：{applied.get('profile_path')}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 5B QCE style distillation.")
    parser.add_argument("--input", dest="input_dir", default=None)
    parser.add_argument("--output-root", dest="output_root", default=None)
    parser.add_argument("--self-uin", dest="self_uin", default=DEFAULT_SELF_UIN)
    parser.add_argument("--max-index-samples", type=int, default=MAX_INDEX_SAMPLES)
    parser.add_argument("--no-apply", action="store_true")
    args = parser.parse_args()

    result = run_qce_style_distillation(
        input_dir=args.input_dir,
        output_root=args.output_root,
        self_uin=args.self_uin,
        max_index_samples=args.max_index_samples,
        apply_to_profile=not args.no_apply,
    )
    print(format_qce_distillation_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
