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

from ...auto_memory import contains_sensitive_content
from ...style_profile import (
    DEFAULT_PROFILE_NAME,
    StyleProfileStore,
    clean_style_common_phrases,
    clean_style_habits,
    style_store,
)
from ...style_skill import (
    candidate_correction_delta,
    format_style_skill_context_for_prompt,
    load_style_skill_context,
)
from .settings import DISTILL_SETTINGS, PROJECT_ROOT


DEFAULT_SELF_UIN = os.getenv("QQBOT_STYLE_SELF_UIN", DISTILL_SETTINGS.str_value("identity.self_uin", "")).strip()
DEFAULT_EXPORT_ROOT = Path(
    os.getenv("QQBOT_STYLE_EXPORT_ROOT")
    or DISTILL_SETTINGS.path_value("paths.export_root", PROJECT_ROOT.parent / "qq-chat-exports")
)
DEFAULT_QCE_INPUT_DIR_ENV = os.getenv("QQBOT_STYLE_QCE_INPUT_DIR", "").strip()
DEFAULT_EXCLUDED_RELATIONSHIP_IDS = set(DISTILL_SETTINGS.str_list("excluded_relationship_ids"))
MAX_INDEX_SAMPLES = DISTILL_SETTINGS.int_value("limits.max_index_samples", 5000)
MAX_REPLY_LENGTH = DISTILL_SETTINGS.int_value("limits.max_reply_length", 280)
MAX_CONTEXT_MESSAGES = DISTILL_SETTINGS.int_value("limits.max_context_messages", 8)
CONTEXT_WINDOW_SECONDS = DISTILL_SETTINGS.int_value("limits.context_window_seconds", 1800)
TURN_MERGE_SECONDS = DISTILL_SETTINGS.int_value("limits.turn_merge_seconds", 60)
PRIVATE_CONTEXT_TURNS = DISTILL_SETTINGS.int_value("limits.private_context_turns", 12)
GROUP_CONTEXT_TURNS = DISTILL_SETTINGS.int_value("limits.group_context_turns", 8)
GROUP_CONTEXT_WINDOW_SECONDS = DISTILL_SETTINGS.int_value("limits.group_context_window_seconds", 600)
MAX_RELATIONSHIP_PROFILES = DISTILL_SETTINGS.int_value("limits.max_relationship_profiles", 500)
DEFAULT_RETRIEVAL_LIMIT = DISTILL_SETTINGS.int_value("limits.default_retrieval_limit", 6)
DEFAULT_GENERATION_CONTEXT_LIMIT = DISTILL_SETTINGS.int_value("limits.default_generation_context_limit", 5)
DEFAULT_RAW_FEWSHOT_LIMIT = DISTILL_SETTINGS.int_value("limits.default_raw_fewshot_limit", 3)
MAX_RAW_FEWSHOT_TEXT_CHARS = DISTILL_SETTINGS.int_value("limits.max_raw_fewshot_text_chars", 180)
MIN_RAW_FEWSHOT_SIMILARITY = DISTILL_SETTINGS.float_value("limits.min_raw_fewshot_similarity", 0.26)
MIN_RAW_FEWSHOT_TEXT_OR_KEYWORD = DISTILL_SETTINGS.float_value("limits.min_raw_fewshot_text_or_keyword", 0.08)

QUESTION_HINTS = DISTILL_SETTINGS.str_list("intent_hints.question")
AVAILABILITY_HINTS = DISTILL_SETTINGS.str_list("intent_hints.availability")
HELP_HINTS = DISTILL_SETTINGS.str_list("intent_hints.help")
INVITATION_HINTS = DISTILL_SETTINGS.str_list("intent_hints.invitation")
GAME_HINTS = DISTILL_SETTINGS.str_list("intent_hints.game")
TASK_HINTS = DISTILL_SETTINGS.str_list("intent_hints.task")
IMAGE_HINTS = DISTILL_SETTINGS.str_list("intent_hints.image")
REALITY_STATE_HINTS = DISTILL_SETTINGS.str_list("intent_hints.reality_state")
HIGH_RISK_COMMITMENT_HINTS = DISTILL_SETTINGS.str_list("intent_hints.high_risk_commitment")
FORMAL_WORK_HINTS = DISTILL_SETTINGS.str_list("intent_hints.formal_work")
EMOTIONAL_HINTS = DISTILL_SETTINGS.str_list("intent_hints.emotional")
EXPLAIN_HINTS = DISTILL_SETTINGS.str_list("intent_hints.explain")
NEGATIVE_PREFIXES = DISTILL_SETTINGS.str_list("reply_prefixes.negative")
CONFIRM_PREFIXES = DISTILL_SETTINGS.str_list("reply_prefixes.confirm")
PLAY_INVITE_RE = re.compile(DISTILL_SETTINGS.str_value("intent_patterns.play_invite", r"(?!)"))
QUESTION_TYPE_RULES = DISTILL_SETTINGS.dict_list("intent_question_type_priority")
COMMITMENT_RISK_RULES = DISTILL_SETTINGS.dict_list("commitment_risk_rules")
PARTICLE_RE = re.compile(DISTILL_SETTINGS.str_value("particle_pattern", r"(?!)"))
TEXT_ONLY_ELEMENTS = {"text", "reply", "at"}
MEDIA_DEPENDENT_ELEMENTS = {
    "image", "pic", "picture", "face", "market_face", "mface", "emoji", "video",
    "record", "voice", "file", "forward", "json", "xml", "ark",
}

TEXT_PLACEHOLDER_RE = re.compile(r"^\s*\[(?:图片|视频|语音|表情|文件|动画表情|转发消息).*\]\s*$")
URL_RE = re.compile(r"https?://|www\.", flags=re.I)
RAW_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", flags=re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
ID_CARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
LONG_ID_RE = re.compile(r"(?<!\d)\d{6,12}(?!\d)")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]",
    flags=re.UNICODE,
)
MARKDOWN_STRUCTURE_RE = re.compile(r"(^|\n)\s{0,3}(?:#{1,4}\s|[-*]\s+|\d+[.)、]\s+|>\s+)", flags=re.M)
MARKDOWN_TABLE_RE = re.compile(r"\|[^|\n]+(?:\|[^|\n]+)+\|")
AI_IDENTITY_RE = re.compile(r"(我是|我叫).{0,20}(?:AI|ai|机器人|助手|科研伙伴|数字伙伴)")
AI_ASSISTANT_MARKERS = DISTILL_SETTINGS.str_list("ai_assistant_markers")


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

def _contains_any(text: str, hints: Sequence[str]) -> bool:
    return any(hint in text for hint in hints)

def _configured_excluded_relationship_ids() -> set[str]:
    raw = os.getenv("QQBOT_STYLE_EXCLUDED_RELATIONSHIP_IDS", "")
    configured = {
        item.strip()
        for item in re.split(r"[,，;\s]+", raw)
        if item.strip()
    }
    return set(DEFAULT_EXCLUDED_RELATIONSHIP_IDS) | configured

def is_ai_assistant_generated_text(text: str) -> bool:
    """Heuristic guard for AI/bot assistant output mixed into QQ exports."""
    raw = str(text or "").strip()
    compact = re.sub(r"\s+", "", raw)
    if len(compact) < 12:
        return False

    lowered = raw.lower()
    score = 0
    if len(compact) >= 120:
        score += 1
    if len(compact) >= 300:
        score += 1
    if AI_IDENTITY_RE.search(raw):
        score += 4
    if MARKDOWN_STRUCTURE_RE.search(raw):
        score += 2
    if MARKDOWN_TABLE_RE.search(raw):
        score += 2
    if "```" in raw:
        score += 3

    marker_hits = 0
    for marker in AI_ASSISTANT_MARKERS:
        marker_text = marker.lower()
        if marker_text in lowered or marker in compact:
            marker_hits += 1
    score += min(5, marker_hits)

    if "让我" in compact[:80] and _contains_any(compact, ("帮你", "整理", "验证", "读取", "查看", "搜索", "创建", "修复")):
        score += 2
    if _contains_any(compact, ("第1步", "第一步", "方案一", "方案1", "##", "---")):
        score += 2
    return score >= 4

def is_style_training_text_allowed(text: str) -> bool:
    """Return whether owner text is suitable as personal style material."""
    if not _is_useful_text(text):
        return False
    if contains_sensitive_content(str(text or "")):
        return False
    if is_ai_assistant_generated_text(str(text or "")):
        return False
    return True

def _style_source_filter(
    messages: Sequence[Dict[str, Any]],
    *,
    chat_type: str,
    self_uin: str,
    relationship_id: str,
) -> Dict[str, Any]:
    """Classify whole sources that should not enter daily owner-style training."""
    result: Dict[str, Any] = {
        "excluded": False,
        "reason": "",
        "ai_like_other_messages": 0,
        "other_text_messages": 0,
        "self_text_messages": 0,
    }
    relationship = str(relationship_id or "").strip()
    if relationship and relationship in _configured_excluded_relationship_ids():
        result["excluded"] = True
        result["reason"] = "configured_ai_or_test_bot_relationship"
        return result
    if relationship and self_uin and relationship == str(self_uin):
        result["excluded"] = True
        result["reason"] = "self_chat_not_daily_style"
        return result

    if chat_type != "private":
        return result

    other_texts = []
    self_texts = []
    for message in messages:
        if not isinstance(message, dict) or _is_system_or_recalled(message):
            continue
        text = _message_text(message)
        if not _is_useful_text(text):
            continue
        if _sender_uin(message) == self_uin:
            self_texts.append(text)
        else:
            other_texts.append(text)

    ai_like_other = sum(1 for item in other_texts if is_ai_assistant_generated_text(item))
    result.update({
        "ai_like_other_messages": ai_like_other,
        "other_text_messages": len(other_texts),
        "self_text_messages": len(self_texts),
    })
    if len(other_texts) >= 3:
        ai_ratio = ai_like_other / max(1, len(other_texts))
        long_other = sum(1 for item in other_texts if len(re.sub(r"\s+", "", item)) >= 180)
        if ai_like_other >= 2 and (ai_ratio >= 0.18 or long_other >= 2):
            result["excluded"] = True
            result["reason"] = "ai_assistant_chat_source"
    return result

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
        "files": {"count": 0, "private": 0, "group": 0, "excluded": 0, "excluded_reasons": Counter()},
        "messages": {"total": 0, "private": 0, "group": 0, "system_or_recalled": 0, "excluded": 0},
        "owner": {
            "total_messages": 0,
            "text_messages": 0,
            "private": 0,
            "group": 0,
            "sensitive_skipped": 0,
            "ai_like_skipped": 0,
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
        "turns": {"count": 0, "private": 0, "group": 0},
        "samples": {
            "candidate_count": 0,
            "indexed_count": 0,
            "dialogue_pair_count": 0,
            "high_quality_count": 0,
            "chat_type_counts": Counter(),
            "scene_counts": Counter(),
        },
    }
    candidates: List[Dict[str, Any]] = []
    raw_turns: List[Dict[str, Any]] = []
    raw_dialogue_pairs: List[Dict[str, Any]] = []
    source_catalog: List[Dict[str, Any]] = []
    source_stats: Dict[str, Dict[str, Any]] = {}
    phrase_stats = {
        "global": _new_phrase_bucket(),
        "private": _new_phrase_bucket(),
        "group": _new_phrase_bucket(),
        "per_relationship": {},
    }
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
        relationship_id = _source_target_id({
            "relative_path": source_relpath,
            "chat_type": chat_type,
        }) or source_file_id
        source_filter = _style_source_filter(
            messages,
            chat_type=chat_type,
            self_uin=effective_self_uin,
            relationship_id=relationship_id,
        )
        catalog_item = {
            "source_file_id": source_file_id,
            "relationship_id": relationship_id,
            "relative_path": source_relpath,
            "file_size_bytes": source_path.stat().st_size,
            "chat_type": chat_type,
            "message_count": len(messages),
            "excluded_from_style_training": bool(source_filter.get("excluded")),
            "excluded_reason": source_filter.get("reason") or "",
            "quality_metrics": {
                key: source_filter.get(key)
                for key in ("ai_like_other_messages", "other_text_messages", "self_text_messages")
            },
            "note": "Private local catalog for reproducing sample indexes; no chat text is stored here.",
        }
        source_catalog.append(catalog_item)
        if source_filter.get("excluded"):
            stats["files"]["excluded"] += 1
            stats["files"]["excluded_reasons"][str(source_filter.get("reason") or "unknown")] += 1
            stats["messages"]["excluded"] += len(messages)
            continue
        source_stats[source_file_id] = _new_source_stats(source_file_id, chat_type, len(messages))

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
            if is_ai_assistant_generated_text(text):
                stats["owner"]["ai_like_skipped"] += 1
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

        turns = _build_turns(
            messages,
            source_file_id=source_file_id,
            relationship_id=relationship_id,
            chat_type=chat_type,
            self_uin=effective_self_uin,
        )
        stats["turns"]["count"] += len(turns)
        if chat_type in {"private", "group"}:
            stats["turns"][chat_type] += len(turns)
        for turn in turns:
            raw_turns.append(_turn_ref(turn, include_text=True))
            if turn.get("role") != "self":
                continue
            text = str(turn.get("text") or "")
            if not _is_useful_text(text):
                continue
            rel_key = _phrase_scope_key(chat_type, relationship_id)
            phrase_stats["per_relationship"].setdefault(rel_key, _new_phrase_bucket())
            for bucket in (
                phrase_stats["global"],
                phrase_stats["private"] if chat_type == "private" else phrase_stats["group"],
                phrase_stats["per_relationship"][rel_key],
            ):
                _add_phrase_observation(bucket, text)

        raw_pairs, index_pairs = _build_dialogue_pairs_for_source(
            turns,
            source_file_id=source_file_id,
            relationship_id=relationship_id,
            chat_type=chat_type,
        )
        raw_dialogue_pairs.extend(raw_pairs)
        candidates.extend(index_pairs)
        for pair in index_pairs:
            score = int(pair.get("score") or 0)
            scene_label = str(pair.get("scene_label") or "unknown")
            stats["samples"]["candidate_count"] += 1
            stats["samples"]["dialogue_pair_count"] += 1
            stats["samples"]["chat_type_counts"][chat_type] += 1
            stats["samples"]["scene_counts"][scene_label] += 1
            if score >= 70:
                stats["samples"]["high_quality_count"] += 1
            source_stats[source_file_id]["candidate_samples"] += 1
            source_stats[source_file_id]["score_total"] += score
            if "private_multi_turn_context" in (pair.get("score_reasons") or []):
                source_stats[source_file_id]["quick_replies"] += 1

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

    phrase_profile = _finalize_phrase_profile(phrase_stats)
    style_patch = _style_patch_from_stats(stats)
    style_patch["common_phrases"] = _common_phrases_from_phrase_profile(phrase_profile)
    relationship_profiles = _build_relationship_profiles(source_stats)
    scene_profiles = _build_scene_profiles(candidates)
    learning_artifacts = _build_learning_artifacts(
        raw_dialogue_pairs,
        phrase_profile=phrase_profile,
        relationship_profiles=relationship_profiles,
    )
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
        "turns": stats["turns"],
        "samples": {
            key: (dict(value) if isinstance(value, Counter) else value)
            for key, value in stats["samples"].items()
        },
    }
    self_uin_hash_source = ",".join(sorted(used_self_uins)) or configured_self_uin
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": _now_iso(),
        "input_dir_id": _sha1_short(str(resolved_input), 12),
        "self_uin_hash": _sha1_short(self_uin_hash_source, 12) if self_uin_hash_source else "",
        "raw_text_policy": (
            "Summary and sample_index store metadata only. turns.jsonl, dialogue_pairs.jsonl, "
            "phrase_profile.json, rag_pool.jsonl, sft_candidates.jsonl, and rerank_style_rules.json "
            "are local raw-text training artifacts and must not be committed or exposed."
        ),
        "sample_index": "sample_index.jsonl",
        "turns": "turns.jsonl",
        "dialogue_pairs": "dialogue_pairs.jsonl",
        "phrase_profile": "phrase_profile.json",
        "rag_pool": "rag_pool.jsonl",
        "sft_candidates": "sft_candidates.jsonl",
        "rerank_style_rules": "rerank_style_rules.json",
        "source_catalog": "source_catalog_private.json",
        "relationship_profiles": "relationship_profiles.json",
        "scene_profiles": "scene_profiles.json",
        "evaluation_report": "evaluation_report.json",
        "stats": public_stats,
        "style_profile_patch": style_patch,
        "learning_taxonomy": learning_artifacts["summary"],
    }
    evaluation_report = _build_evaluation_report(
        summary=summary,
        relationship_profiles=relationship_profiles,
        indexed=indexed,
        phrase_profile=phrase_profile,
        taxonomy_summary=learning_artifacts["summary"],
    )

    summary_path = output_dir / "style_profile_summary.json"
    patch_path = output_dir / "style_profile_patch.json"
    index_path = output_dir / "sample_index.jsonl"
    turns_path = output_dir / "turns.jsonl"
    dialogue_pairs_path = output_dir / "dialogue_pairs.jsonl"
    phrase_profile_path = output_dir / "phrase_profile.json"
    rag_pool_path = output_dir / "rag_pool.jsonl"
    sft_candidates_path = output_dir / "sft_candidates.jsonl"
    rerank_rules_path = output_dir / "rerank_style_rules.json"
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
    with turns_path.open("w", encoding="utf-8") as f:
        for item in raw_turns:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    with dialogue_pairs_path.open("w", encoding="utf-8") as f:
        for item in raw_dialogue_pairs:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    phrase_profile_path.write_text(
        json.dumps(phrase_profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with rag_pool_path.open("w", encoding="utf-8") as f:
        for item in learning_artifacts["rag_pool"]:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    with sft_candidates_path.open("w", encoding="utf-8") as f:
        for item in learning_artifacts["sft_candidates"]:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    rerank_rules_path.write_text(
        json.dumps(learning_artifacts["rerank_style_rules"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    index_summary_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "raw_text_policy": summary["raw_text_policy"],
                "candidate_count": stats["samples"]["candidate_count"],
                "dialogue_pair_count": stats["samples"]["dialogue_pair_count"],
                "high_quality_count": stats["samples"]["high_quality_count"],
                "indexed_count": len(indexed),
                "rag_pool_count": len(learning_artifacts["rag_pool"]),
                "sft_candidate_count": len(learning_artifacts["sft_candidates"]),
                "taxonomy": learning_artifacts["summary"],
                "min_score": min((int(item["score"]) for item in indexed), default=0),
                "max_score": max((int(item["score"]) for item in indexed), default=0),
                "chat_type_counts": dict(stats["samples"]["chat_type_counts"].most_common()),
                "scene_counts": dict(stats["samples"]["scene_counts"].most_common()),
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
        "turns_path": str(turns_path),
        "dialogue_pairs_path": str(dialogue_pairs_path),
        "phrase_profile_path": str(phrase_profile_path),
        "rag_pool_path": str(rag_pool_path),
        "sft_candidates_path": str(sft_candidates_path),
        "rerank_style_rules_path": str(rerank_rules_path),
        "indexed_samples": len(indexed),
        "candidate_samples": stats["samples"]["candidate_count"],
        "dialogue_pair_count": stats["samples"]["dialogue_pair_count"],
        "rag_pool_count": len(learning_artifacts["rag_pool"]),
        "sft_candidate_count": len(learning_artifacts["sft_candidates"]),
        "learning_taxonomy": learning_artifacts["summary"],
        "turn_count": stats["turns"]["count"],
        "high_quality_samples": stats["samples"]["high_quality_count"],
        "common_phrases": style_patch.get("common_phrases") or [],
        "relationship_profiles": relationship_profiles["profile_count"],
        "scene_profiles": scene_profiles["scene_count"],
        "readiness": evaluation_report["readiness"],
        "owner_text_messages": stats["owner"]["text_messages"],
        "total_messages": stats["messages"]["total"],
        "input_files": stats["files"]["count"],
        "excluded_sources": stats["files"]["excluded"],
        "excluded_reasons": dict(stats["files"]["excluded_reasons"].most_common()),
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

    for field in ("habits", "punctuation", "common_phrases"):
        merged = []
        seen = set()
        for item in (profile.get(field) or []) + (patch.get(field) or []):
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text[:80])
        if field == "habits":
            profile[field] = clean_style_habits(merged)
        elif field == "common_phrases":
            profile[field] = clean_style_common_phrases(merged)
        else:
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

# Export local helpers before late imports. Downstream modules import qce_io while
# the late imports below are resolving, so __all__ must already include private helpers.
__all__ = [name for name in globals() if not name.startswith("__")]


# Late imports keep the runner compatible while avoiding import cycles during module load.
def _load_distill_runtime_helpers() -> None:
    global _build_turns, _build_dialogue_pairs_for_source, _new_phrase_bucket, _add_phrase_observation
    global _finalize_phrase_profile, _common_phrases_from_phrase_profile, _style_patch_from_stats
    global _new_source_stats, _relationship_labels, _build_relationship_profiles, _build_evaluation_report
    global _build_scene_profiles, _build_learning_artifacts, _features_for_text, _turn_ref, _phrase_scope_key
    global _merge_counter
    global _source_target_id
    from .phrases import _add_phrase_observation, _common_phrases_from_phrase_profile, _finalize_phrase_profile, _new_phrase_bucket, _phrase_scope_key
    from .retrieval import _source_target_id
    from .turns import _build_dialogue_pairs_for_source, _build_evaluation_report, _build_relationship_profiles, _build_scene_profiles, _build_turns, _features_for_text, _merge_counter, _new_source_stats, _relationship_labels, _style_patch_from_stats, _turn_ref
    from .taxonomy import _build_learning_artifacts

_load_distill_runtime_helpers()
