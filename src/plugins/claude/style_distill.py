"""Offline QCE chat-log distillation for owner style profiles.

This module reads QCE JSON exports and writes only aggregate style summaries
plus message-id based sample indexes. It must not persist raw chat text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Sequence

from .auto_memory import contains_sensitive_content
from .style_profile import DEFAULT_PROFILE_NAME, StyleProfileStore, style_store


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SELF_UIN = os.getenv("QQBOT_STYLE_SELF_UIN", "").strip()
DEFAULT_EXPORT_ROOT = Path(os.getenv("QQBOT_STYLE_EXPORT_ROOT") or (PROJECT_ROOT.parent / "qq-chat-exports"))
DEFAULT_QCE_INPUT_DIR_ENV = os.getenv("QQBOT_STYLE_QCE_INPUT_DIR", "").strip()
MAX_INDEX_SAMPLES = 5000
MAX_REPLY_LENGTH = 280
MAX_CONTEXT_MESSAGES = 8
CONTEXT_WINDOW_SECONDS = 30 * 60
MAX_RELATIONSHIP_PROFILES = 500
DEFAULT_RETRIEVAL_LIMIT = 6

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
    input_dir: Path | None = None
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


def default_qce_input_dir() -> Path:
    if DEFAULT_QCE_INPUT_DIR_ENV:
        return Path(DEFAULT_QCE_INPUT_DIR_ENV)
    root = DEFAULT_EXPORT_ROOT
    if root.exists():
        candidates = []
        for path in root.glob("*/distill-json/*"):
            if path.is_dir() and any(path.glob("*.json")):
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    mtime = 0
                candidates.append((mtime, path))
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
    return root


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_qce_input_dir(input_dir: str | Path | None = None) -> Path:
    """Resolve and validate a QCE JSON input directory.

    Owner-only commands still avoid arbitrary local reads: the allowed locations
    are the configured QCE export root and this project's data folder.
    """
    candidate = _as_path(input_dir, default_qce_input_dir())
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    resolved = candidate.resolve()
    allowed_roots = [
        DEFAULT_EXPORT_ROOT.resolve(),
        (Path.cwd() / "data").resolve(),
    ]
    if not any(_is_relative_to(resolved, root) for root in allowed_roots):
        raise ValueError("离线蒸馏只允许读取配置的 QCE 导出目录或项目 data 目录下的文件。")
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


def _text_ngrams(text: str, n: int = 2) -> set[str]:
    compact = re.sub(r"\s+", "", text.lower())
    if not compact:
        return set()
    if len(compact) <= n:
        return {compact}
    return {compact[i:i + n] for i in range(0, len(compact) - n + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


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


def _new_source_stats(source_file_id: str, chat_type: str, message_count: int) -> Dict[str, Any]:
    return {
        "source_file_id": source_file_id,
        "chat_type": chat_type,
        "message_count": message_count,
        "owner_text_messages": 0,
        "owner_lengths": [],
        "candidate_samples": 0,
        "score_total": 0,
        "element_types": Counter(),
        "length_buckets": Counter(),
        "quick_replies": 0,
        "question_messages": 0,
        "exclamation_messages": 0,
        "ellipsis_messages": 0,
    }


def _relationship_labels(item: Dict[str, Any]) -> List[str]:
    labels = []
    chat_type = item["chat_type"]
    message_count = int(item.get("message_count") or 0)
    owner_count = int(item.get("owner_text_messages") or 0)
    candidates = int(item.get("candidate_samples") or 0)
    avg_length = float(item.get("avg_length") or 0)
    owner_ratio = owner_count / max(1, message_count)

    labels.append("private_dialogue" if chat_type == "private" else "group_chat")

    if owner_count < 20:
        labels.append("low_evidence")
    elif owner_count >= 1000:
        labels.append("high_evidence")
    else:
        labels.append("medium_evidence")

    if avg_length <= 8:
        labels.append("terse_replies")
    elif avg_length <= 24:
        labels.append("brief_replies")
    else:
        labels.append("detailed_replies")

    if chat_type == "private":
        if owner_count >= 800 and owner_ratio >= 0.25:
            labels.append("high_familiarity_private")
        elif owner_count >= 100:
            labels.append("active_private")
        else:
            labels.append("light_private")
    else:
        if owner_ratio >= 0.05 and owner_count >= 300:
            labels.append("active_group_participant")
        elif owner_count >= 100:
            labels.append("occasional_group_participant")
        else:
            labels.append("low_frequency_group")

    if candidates >= 50:
        labels.append("strong_context_reply_source")
    elif candidates >= 10:
        labels.append("usable_context_reply_source")
    return labels


def _build_relationship_profiles(source_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    profiles = []
    for item in source_stats.values():
        lengths = item["owner_lengths"]
        avg_length = round(sum(lengths) / len(lengths), 1) if lengths else 0
        median_length = int(median(lengths)) if lengths else 0
        public_item = {
            "source_file_id": item["source_file_id"],
            "chat_type": item["chat_type"],
            "message_count": item["message_count"],
            "owner_text_messages": item["owner_text_messages"],
            "candidate_samples": item["candidate_samples"],
            "avg_length": avg_length,
            "median_length": median_length,
            "length_buckets": dict(item["length_buckets"]),
            "element_types": dict(item["element_types"].most_common(10)),
            "quick_replies": item["quick_replies"],
            "question_messages": item["question_messages"],
            "exclamation_messages": item["exclamation_messages"],
            "ellipsis_messages": item["ellipsis_messages"],
        }
        public_item["labels"] = _relationship_labels(public_item)
        if item["candidate_samples"]:
            public_item["avg_sample_score"] = round(item["score_total"] / item["candidate_samples"], 1)
        else:
            public_item["avg_sample_score"] = 0
        profiles.append(public_item)

    profiles.sort(
        key=lambda p: (
            -int(p["candidate_samples"]),
            -int(p["owner_text_messages"]),
            str(p["source_file_id"]),
        )
    )
    label_counts = Counter(label for profile in profiles for label in profile["labels"])
    return {
        "schema_version": 1,
        "raw_text_policy": "No raw chat text is stored in relationship profiles.",
        "profile_count": len(profiles),
        "label_counts": dict(label_counts.most_common()),
        "profiles": profiles[:MAX_RELATIONSHIP_PROFILES],
    }


def _build_evaluation_report(
    *,
    summary: Dict[str, Any],
    relationship_profiles: Dict[str, Any],
    indexed: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    score_buckets = Counter(str((int(item["score"]) // 10) * 10) for item in indexed)
    relation_profiles = relationship_profiles.get("profiles") or []
    strong_sources = sum(1 for item in relation_profiles if "strong_context_reply_source" in (item.get("labels") or []))
    usable_sources = sum(1 for item in relation_profiles if "usable_context_reply_source" in (item.get("labels") or []))
    low_evidence = sum(1 for item in relation_profiles if "low_evidence" in (item.get("labels") or []))
    samples = (summary.get("stats") or {}).get("samples") or {}
    owner = (summary.get("stats") or {}).get("owner") or {}

    readiness = "weak"
    indexed_count = int(samples.get("indexed_count") or 0)
    owner_text = int(owner.get("text_messages") or 0)
    if indexed_count >= 1000 and strong_sources >= 5:
        readiness = "strong"
    elif indexed_count >= 300 and (strong_sources + usable_sources) >= 3:
        readiness = "usable"

    return {
        "schema_version": 1,
        "run_id": summary.get("run_id"),
        "raw_text_policy": "No raw chat text is stored in this evaluation report.",
        "readiness": readiness,
        "owner_text_messages": owner_text,
        "indexed_samples": indexed_count,
        "candidate_samples": int(samples.get("candidate_count") or 0),
        "score_buckets": dict(score_buckets.most_common()),
        "relationship_sources": {
            "total": len(relation_profiles),
            "strong": strong_sources,
            "usable": usable_sources,
            "low_evidence": low_evidence,
        },
        "recommendations": [
            "先使用草稿模式，不开启自动发送。",
            "下一步优先补关系/场景标签，再做相似样本临时检索。",
            "对高风险现实状态、承诺、金钱和隐私问题继续保持拒绝或模糊草稿。",
        ],
    }


def _scene_key_for_sample(sample: Dict[str, Any]) -> str:
    reply = sample.get("reply") or {}
    features = reply.get("features") or {}
    context = sample.get("context") or {}
    reasons = set(sample.get("score_reasons") or [])
    parts = [str(sample.get("chat_type") or "unknown")]
    if "explicit_reply" in reasons:
        parts.append("explicit_reply")
    elif "mentions" in reasons:
        parts.append("mentioned")
    elif int(context.get("count") or 0) >= 3:
        parts.append("multi_context")
    else:
        parts.append("direct_context")
    if features.get("has_question"):
        parts.append("question_reply")
    elif features.get("has_exclamation"):
        parts.append("emotional_reply")
    elif features.get("has_ellipsis"):
        parts.append("pause_reply")
    else:
        parts.append("plain_reply")
    parts.append(str(reply.get("length_bucket") or "unknown_len"))
    return "::".join(parts)


def _scene_recommendation(scene_id: str, avg_reply_length: float) -> str:
    if "question_reply" in scene_id:
        return "contextual_follow_up"
    if "group" in scene_id and avg_reply_length <= 12:
        return "brief_group_interjection"
    if "private" in scene_id and avg_reply_length >= 20:
        return "contextual_private_reply"
    if "emotional_reply" in scene_id:
        return "short_reaction"
    return "terse_contextual_reply"


def _build_scene_profiles(indexed: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scenes: Dict[str, Dict[str, Any]] = {}
    for sample in indexed:
        scene_id = _scene_key_for_sample(sample)
        reply = sample.get("reply") or {}
        context = sample.get("context") or {}
        features = reply.get("features") or {}
        item = scenes.setdefault(scene_id, {
            "scene_id": scene_id,
            "sample_count": 0,
            "chat_types": Counter(),
            "length_buckets": Counter(),
            "element_types": Counter(),
            "context_count_buckets": Counter(),
            "score_total": 0,
            "reply_length_total": 0,
            "question_replies": 0,
            "exclamation_replies": 0,
            "ellipsis_replies": 0,
            "quick_replies": 0,
            "sample_refs": [],
        })
        item["sample_count"] += 1
        item["chat_types"][str(sample.get("chat_type") or "unknown")] += 1
        item["length_buckets"][str(reply.get("length_bucket") or "unknown")] += 1
        _merge_counter(item["element_types"], reply.get("element_types") or [])
        context_count = int(context.get("count") or 0)
        if context_count <= 1:
            item["context_count_buckets"]["1"] += 1
        elif context_count <= 3:
            item["context_count_buckets"]["2-3"] += 1
        else:
            item["context_count_buckets"]["4+"] += 1
        item["score_total"] += int(sample.get("score") or 0)
        item["reply_length_total"] += int(reply.get("char_length") or 0)
        if features.get("has_question"):
            item["question_replies"] += 1
        if features.get("has_exclamation"):
            item["exclamation_replies"] += 1
        if features.get("has_ellipsis"):
            item["ellipsis_replies"] += 1
        if int(context.get("first_non_self_delay_seconds") or 999999) <= 120:
            item["quick_replies"] += 1
        if len(item["sample_refs"]) < 12:
            item["sample_refs"].append(str(sample.get("sample_id") or ""))

    profiles = []
    for item in scenes.values():
        sample_count = max(1, item["sample_count"])
        avg_reply_length = round(item["reply_length_total"] / sample_count, 1)
        profiles.append({
            "scene_id": item["scene_id"],
            "sample_count": item["sample_count"],
            "chat_types": dict(item["chat_types"]),
            "length_buckets": dict(item["length_buckets"]),
            "element_types": dict(item["element_types"].most_common(10)),
            "context_count_buckets": dict(item["context_count_buckets"]),
            "avg_score": round(item["score_total"] / sample_count, 1),
            "avg_reply_length": avg_reply_length,
            "question_ratio": round(item["question_replies"] / sample_count, 3),
            "exclamation_ratio": round(item["exclamation_replies"] / sample_count, 3),
            "ellipsis_ratio": round(item["ellipsis_replies"] / sample_count, 3),
            "quick_reply_ratio": round(item["quick_replies"] / sample_count, 3),
            "recommended_style": _scene_recommendation(item["scene_id"], avg_reply_length),
            "sample_refs": [ref for ref in item["sample_refs"] if ref],
        })
    profiles.sort(key=lambda item: (-int(item["sample_count"]), -float(item["avg_score"]), item["scene_id"]))
    return {
        "schema_version": 1,
        "raw_text_policy": "No raw chat text is stored in scene profiles.",
        "scene_count": len(profiles),
        "profiles": profiles,
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
    source_stats: Dict[str, Dict[str, Any]] = {}
    seen_reply_text = set()
    input_files = sorted(resolved_input.glob("*.json"))
    configured_self_uin = str(self_uin or "").strip()
    used_self_uins = set()

    for source_number, source_path in enumerate(input_files, start=1):
        try:
            data = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        chat_info = data.get("chatInfo") or {}
        chat_type = str(chat_info.get("type") or "unknown")
        effective_self_uin = configured_self_uin or str(chat_info.get("selfUin") or "").strip()
        messages = data.get("messages") or []
        if not isinstance(messages, list):
            continue

        stats["files"]["count"] += 1
        if chat_type in {"private", "group"}:
            stats["files"][chat_type] += 1

        source_relpath = str(source_path.relative_to(resolved_input.parent))
        source_file_id = f"source_{source_number:04d}_{_sha1_short(source_relpath, 8)}"
        source_stats[source_file_id] = _new_source_stats(source_file_id, chat_type, len(messages))
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

            is_owner = bool(effective_self_uin) and _sender_uin(message) == effective_self_uin
            if not is_owner:
                continue
            used_self_uins.add(effective_self_uin)

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
            source_stats[source_file_id]["owner_text_messages"] += 1
            source_stats[source_file_id]["owner_lengths"].append(text_features["char_length"])
            source_stats[source_file_id]["length_buckets"][text_features["length_bucket"]] += 1
            _merge_counter(source_stats[source_file_id]["element_types"], element_types)
            if text_features["has_question"]:
                source_stats[source_file_id]["question_messages"] += 1
            if text_features["has_exclamation"]:
                source_stats[source_file_id]["exclamation_messages"] += 1
            if text_features["has_ellipsis"]:
                source_stats[source_file_id]["ellipsis_messages"] += 1

            if len(text) > MAX_REPLY_LENGTH:
                continue
            text_digest = _sha1_short(text, 20)
            if text_digest in seen_reply_text:
                continue
            seen_reply_text.add(text_digest)

            context, first_non_self_delay = _context_for(
                messages,
                index,
                effective_self_uin,
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
            source_stats[source_file_id]["candidate_samples"] += 1
            source_stats[source_file_id]["score_total"] += score
            if first_non_self_delay is not None and first_non_self_delay <= 120:
                source_stats[source_file_id]["quick_replies"] += 1
            sample_id = _sha1_short(
                f"{source_file_id}:{index}:{message.get('id')}:{message.get('seq')}:{_timestamp(message)}",
                20,
            )
            context_refs = []
            for context_index, item in context:
                role = "self" if _sender_uin(item) == effective_self_uin else "other"
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
    relationship_profiles = _build_relationship_profiles(source_stats)
    scene_profiles = _build_scene_profiles(indexed)
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
    self_uin_hash_source = ",".join(sorted(used_self_uins)) or configured_self_uin
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": _now_iso(),
        "input_dir_id": _sha1_short(str(resolved_input), 12),
        "self_uin_hash": _sha1_short(self_uin_hash_source, 12) if self_uin_hash_source else "",
        "raw_text_policy": "No raw chat text is stored in this summary or sample index.",
        "sample_index": "sample_index.jsonl",
        "source_catalog": "source_catalog_private.json",
        "relationship_profiles": "relationship_profiles.json",
        "scene_profiles": "scene_profiles.json",
        "evaluation_report": "evaluation_report.json",
        "stats": public_stats,
        "style_profile_patch": style_patch,
    }
    evaluation_report = _build_evaluation_report(
        summary=summary,
        relationship_profiles=relationship_profiles,
        indexed=indexed,
    )

    summary_path = output_dir / "style_profile_summary.json"
    patch_path = output_dir / "style_profile_patch.json"
    index_path = output_dir / "sample_index.jsonl"
    index_summary_path = output_dir / "sample_index_summary.json"
    source_catalog_path = output_dir / "source_catalog_private.json"
    relationship_path = output_dir / "relationship_profiles.json"
    scene_path = output_dir / "scene_profiles.json"
    evaluation_path = output_dir / "evaluation_report.json"

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
    relationship_path.write_text(
        json.dumps(relationship_profiles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    scene_path.write_text(
        json.dumps(scene_profiles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    evaluation_path.write_text(
        json.dumps(evaluation_report, ensure_ascii=False, indent=2),
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
                "self_uin_mode": "explicit" if configured_self_uin else "from_qce_chat_info",
                "self_uin_hash": summary["self_uin_hash"],
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
        "relationship_profiles_path": str(relationship_path),
        "scene_profiles_path": str(scene_path),
        "evaluation_report_path": str(evaluation_path),
        "indexed_samples": len(indexed),
        "candidate_samples": stats["samples"]["candidate_count"],
        "relationship_profiles": relationship_profiles["profile_count"],
        "scene_profiles": scene_profiles["scene_count"],
        "readiness": evaluation_report["readiness"],
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
        f"- 关系/场景画像：{result.get('relationship_profiles', 0)} 个",
        f"- 场景画像：{result.get('scene_profiles', 0)} 个",
        f"- 数据就绪度：{result.get('readiness')}",
        "- 原文策略：摘要和索引不保存聊天正文",
        f"- 输出目录：{result.get('output_dir')}",
    ]
    if applied:
        lines.append(f"- 已更新画像：{applied.get('profile_path')}")
    return "\n".join(lines)


def format_style_relationship_report(run_dir: str | Path | None = None) -> str:
    """Format relationship/scene source profiles without identifiers or text."""
    run_path = find_latest_distill_run(run_dir)
    if run_path is None:
        return "还没有找到 Stage 5B 离线蒸馏结果。先运行 /风格 离线蒸馏。"
    path = run_path / "relationship_profiles.json"
    if not path.exists():
        return f"这个蒸馏结果缺少 relationship_profiles.json：{run_path.name}"
    data = _load_json(path)
    profiles = data.get("profiles") or []
    label_counts = data.get("label_counts") or {}
    strong = sum(1 for item in profiles if "strong_context_reply_source" in (item.get("labels") or []))
    usable = sum(1 for item in profiles if "usable_context_reply_source" in (item.get("labels") or []))
    low = sum(1 for item in profiles if "low_evidence" in (item.get("labels") or []))
    private_count = sum(1 for item in profiles if item.get("chat_type") == "private")
    group_count = sum(1 for item in profiles if item.get("chat_type") == "group")
    lines = [
        "关系/场景画像摘要：",
        f"- run_id：{run_path.name}",
        f"- 来源：{len(profiles)} 个，私聊 {private_count}，群聊 {group_count}",
        f"- 上下文样本源：strong={strong}，usable={usable}，low={low}",
        "- 原文策略：不显示联系人、群名、QQ 或聊天正文",
    ]
    if label_counts:
        labels = "、".join(f"{key}:{value}" for key, value in list(label_counts.items())[:10])
        lines.append(f"- 标签分布：{labels}")
    top = profiles[:5]
    if top:
        lines.append("Top 来源摘要：")
        for index, item in enumerate(top, start=1):
            lines.append(
                f"{index}. {item.get('source_file_id')} {item.get('chat_type')} "
                f"owner={item.get('owner_text_messages')} samples={item.get('candidate_samples')} "
                f"avg={item.get('avg_length')} labels={','.join((item.get('labels') or [])[:3])}"
            )
    return "\n".join(lines)


def format_style_scene_report(run_dir: str | Path | None = None) -> str:
    """Format scene profiles without raw text."""
    run_path = find_latest_distill_run(run_dir)
    if run_path is None:
        return "还没有找到 Stage 5B 离线蒸馏结果。先运行 /风格 离线蒸馏。"
    path = run_path / "scene_profiles.json"
    if not path.exists():
        return f"这个蒸馏结果缺少 scene_profiles.json：{run_path.name}"
    data = _load_json(path)
    profiles = data.get("profiles") or []
    lines = [
        "场景画像摘要：",
        f"- run_id：{run_path.name}",
        f"- 场景数：{len(profiles)}",
        "- 原文策略：不显示历史上下文或真实回复正文",
    ]
    for index, item in enumerate(profiles[:8], start=1):
        lines.append(
            f"{index}. {item.get('scene_id')} count={item.get('sample_count')} "
            f"avg_len={item.get('avg_reply_length')} quick={item.get('quick_reply_ratio')} "
            f"style={item.get('recommended_style')}"
        )
    if not profiles:
        lines.append("尚未生成可用场景画像。")
    return "\n".join(lines)


def find_latest_distill_run(root: str | Path | None = None) -> Path | None:
    """Find the latest local Stage 5B distillation run directory."""
    if root:
        candidates_root = Path(root)
        if candidates_root.is_file():
            candidates_root = candidates_root.parent
        roots = [candidates_root]
    else:
        roots = list(DEFAULT_EXPORT_ROOT.glob("*/distill-runs"))
    runs: List[Path] = []
    for item in roots:
        if not item.exists():
            continue
        if item.name.startswith("stage5b_"):
            runs.append(item)
        else:
            runs.extend(path for path in item.glob("stage5b_*") if path.is_dir())
    if not runs:
        return None
    return max(runs, key=lambda path: path.stat().st_mtime)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_style_evaluation_report(run_dir: str | Path | None = None) -> str:
    """Format the latest Stage 5B readiness report without raw text."""
    run_path = find_latest_distill_run(run_dir)
    if run_path is None:
        return "还没有找到 Stage 5B 离线蒸馏结果。先运行 /风格 离线蒸馏。"
    report_path = run_path / "evaluation_report.json"
    relation_path = run_path / "relationship_profiles.json"
    if not report_path.exists():
        return f"这个蒸馏结果缺少 evaluation_report.json：{run_path.name}"

    report = _load_json(report_path)
    relationships = _load_json(relation_path) if relation_path.exists() else {}
    relation_sources = report.get("relationship_sources") or {}
    label_counts = relationships.get("label_counts") or {}
    lines = [
        "Stage 5B 评估摘要：",
        f"- run_id：{report.get('run_id')}",
        f"- 就绪度：{report.get('readiness')}",
        f"- 本人文本：{report.get('owner_text_messages', 0)} 条",
        f"- 候选样本：{report.get('candidate_samples', 0)} 条",
        f"- 索引样本：{report.get('indexed_samples', 0)} 条",
        (
            "- 关系来源："
            f"{relation_sources.get('total', 0)} 个，"
            f"strong={relation_sources.get('strong', 0)}，"
            f"usable={relation_sources.get('usable', 0)}，"
            f"low={relation_sources.get('low_evidence', 0)}"
        ),
    ]
    if label_counts:
        top_labels = "、".join(f"{key}:{value}" for key, value in list(label_counts.items())[:8])
        lines.append(f"- 场景标签：{top_labels}")
    recommendations = report.get("recommendations") or []
    if recommendations:
        lines.append("- 建议：" + "；".join(str(item) for item in recommendations[:3]))
    lines.append("- 原文策略：评估报告不保存聊天正文")
    return "\n".join(lines)


def _resolve_run_paths(run_dir: str | Path | None = None) -> tuple[Path, Dict[str, Any], List[Dict[str, Any]]]:
    run_path = find_latest_distill_run(run_dir)
    if run_path is None:
        raise FileNotFoundError("还没有找到 Stage 5B 离线蒸馏结果。")
    catalog_path = run_path / "source_catalog_private.json"
    index_path = run_path / "sample_index.jsonl"
    if not catalog_path.exists() or not index_path.exists():
        raise FileNotFoundError("蒸馏结果缺少 source_catalog_private.json 或 sample_index.jsonl。")
    catalog = _load_json(catalog_path)
    samples = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return run_path, catalog, samples


def _load_source_messages(catalog: Dict[str, Any], source_file_id: str, cache: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if source_file_id in cache:
        return cache[source_file_id]
    input_dir = Path(str(catalog.get("input_dir") or ""))
    for source in catalog.get("sources") or []:
        if source.get("source_file_id") != source_file_id:
            continue
        rel = Path(str(source.get("relative_path") or ""))
        path = (input_dir.parent / rel).resolve()
        data = _load_json(path)
        messages = data.get("messages") or []
        cache[source_file_id] = [item for item in messages if isinstance(item, dict)]
        return cache[source_file_id]
    return []


def retrieve_similar_style_samples(
    query: str,
    *,
    run_dir: str | Path | None = None,
    limit: int = DEFAULT_RETRIEVAL_LIMIT,
) -> Dict[str, Any]:
    """Retrieve similar indexed samples using local raw QCE text transiently.

    Return value intentionally contains no raw historical text.
    """
    query_text = query.strip()
    if not query_text:
        return {"ok": False, "message": "用法：/风格 检索 <当前对方消息>"}
    if len(query_text) > 500:
        return {"ok": False, "message": "检索文本太长，先支持 500 字以内。"}

    run_path, catalog, samples = _resolve_run_paths(run_dir)
    query_features = _features_for_text(query_text)
    query_ngrams = _text_ngrams(query_text)
    cache: Dict[str, List[Dict[str, Any]]] = {}
    results = []

    for sample in samples:
        source_file_id = str(sample.get("source_file_id") or "")
        messages = _load_source_messages(catalog, source_file_id, cache)
        if not messages:
            continue
        context_texts = []
        for ref in (sample.get("context") or {}).get("messages") or []:
            if ref.get("role") != "other":
                continue
            try:
                record_index = int(ref.get("record_index"))
            except (TypeError, ValueError):
                continue
            if record_index < 0 or record_index >= len(messages):
                continue
            text = _message_text(messages[record_index])
            if _is_useful_text(text):
                context_texts.append(text)
        if not context_texts:
            continue
        context_text = "\n".join(context_texts[-3:])
        overlap = _jaccard(query_ngrams, _text_ngrams(context_text))
        if overlap <= 0:
            continue
        reply = sample.get("reply") or {}
        feature_bonus = 0.0
        if bool(query_features["has_question"]) == bool(reply.get("features", {}).get("has_question")):
            feature_bonus += 0.04
        if query_features["length_bucket"] == reply.get("length_bucket"):
            feature_bonus += 0.03
        total_score = round(overlap + feature_bonus + (int(sample.get("score") or 0) / 1000), 4)
        results.append({
            "sample_id": sample.get("sample_id"),
            "source_file_id": source_file_id,
            "chat_type": sample.get("chat_type"),
            "similarity": total_score,
            "context_overlap": round(overlap, 4),
            "quality_score": int(sample.get("score") or 0),
            "reply_length_bucket": reply.get("length_bucket"),
            "reply_char_length": reply.get("char_length"),
            "context_count": (sample.get("context") or {}).get("count"),
            "time_bucket": reply.get("time_bucket"),
            "score_reasons": sample.get("score_reasons") or [],
        })

    results.sort(key=lambda item: (-float(item["similarity"]), -int(item["quality_score"]), str(item["sample_id"])))
    return {
        "ok": True,
        "run_id": run_path.name,
        "raw_text_policy": "Historical text was read transiently for similarity only; no raw text is returned or persisted.",
        "query_features": {
            "char_length": query_features["char_length"],
            "length_bucket": query_features["length_bucket"],
            "has_question": query_features["has_question"],
            "has_exclamation": query_features["has_exclamation"],
            "has_ellipsis": query_features["has_ellipsis"],
        },
        "result_count": min(len(results), max(0, int(limit))),
        "results": results[: max(0, int(limit))],
    }


def format_similar_sample_results(result: Dict[str, Any]) -> str:
    if not result.get("ok"):
        return str(result.get("message") or "检索失败。")
    lines = [
        "相似样本检索结果：",
        f"- run_id：{result.get('run_id')}",
        f"- 命中：{result.get('result_count', 0)} 条",
        "- 原文策略：本地临时读取历史文本计算相似度，但不返回、不保存原文",
    ]
    features = result.get("query_features") or {}
    lines.append(
        f"- 查询特征：{features.get('length_bucket')}，"
        f"问句={features.get('has_question')}，"
        f"感叹={features.get('has_exclamation')}"
    )
    for index, item in enumerate(result.get("results") or [], start=1):
        lines.append(
            f"{index}. sim={item.get('similarity')} q={item.get('quality_score')} "
            f"{item.get('chat_type')} {item.get('source_file_id')} "
            f"reply_len={item.get('reply_char_length')} ctx={item.get('context_count')}"
        )
    if not result.get("results"):
        lines.append("没有找到足够相似的历史上下文。")
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
