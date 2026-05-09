"""Runtime owner-style skill layer for 36.

This module keeps the owner-style persona separate from the bot persona.  It
loads local markdown summaries plus a compact correction log and formats a
bounded context for prompt construction.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


STYLE_SKILL_DIR = Path("data/style_profiles/36_skill")
RELATIONSHIP_DIR = STYLE_SKILL_DIR / "relationship_profiles"
EVAL_CASE_DIR = STYLE_SKILL_DIR / "eval_cases"
CORRECTIONS_PATH = STYLE_SKILL_DIR / "corrections.jsonl"

MAX_MD_CHARS = 1800
MAX_RELATIONSHIP_CHARS = 1500
MAX_CORRECTIONS_IN_PROMPT = 5
CORRECTION_TERM_HINTS = (
    "忙", "有空", "在不在", "在吗", "在哪", "做完", "弄完", "今天",
    "帮我", "看下", "看看", "报错", "代码", "截图", "图片", "表情",
    "账号", "登录", "署名", "致谢", "讲下", "解释", "饭", "游戏",
)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _clean_text(value: Any, limit: int = 1200) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _read_md(path: Path, limit: int = MAX_MD_CHARS) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    return text.strip()[: max(0, int(limit))]


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    yield item
    except OSError:
        return


def _ngrams(text: str, n: int = 2) -> set[str]:
    compact = re.sub(r"\s+", "", str(text or "").lower())
    if not compact:
        return set()
    if len(compact) <= n:
        return {compact}
    return {compact[i:i + n] for i in range(len(compact) - n + 1)}


def _text_similarity(left: str, right: str) -> float:
    left_set = _ngrams(left)
    right_set = _ngrams(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def _sha1_short(text: str, size: int = 16) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:size]


def _correction_terms(text: str) -> List[str]:
    compact = re.sub(r"\s+", "", str(text or ""))
    terms: List[str] = []
    for hint in CORRECTION_TERM_HINTS:
        if hint and hint in compact:
            terms.append(hint)
    terms.extend(re.findall(r"[A-Za-z0-9_]{2,24}", str(text or "")))
    seen = set()
    deduped = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            deduped.append(term[:24])
    return deduped[:12]


def _message_hint(text: str) -> str:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return "unknown"
    labels = []
    if any(item in compact for item in ("忙", "有空", "在不在", "在吗", "在哪")):
        labels.append("state_query")
    if any(item in compact for item in ("帮我", "看下", "看看", "报错", "代码", "弄完", "做完")):
        labels.append("task_or_help")
    if any(item in compact for item in ("图片", "截图", "表情", "[图片]", "[表情]")):
        labels.append("media_related")
    if any(item in compact for item in ("账号", "登录", "密码")):
        labels.append("account_risk")
    if not labels and ("?" in compact or "？" in compact or "吗" in compact):
        labels.append("question")
    return ",".join(labels[:4]) or f"len_{min(200, len(compact))}"


def _term_similarity(left_terms: Sequence[str], right_terms: Sequence[str]) -> float:
    left_set = {str(item) for item in left_terms if str(item).strip()}
    right_set = {str(item) for item in right_terms if str(item).strip()}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def _source_scene(metadata: Dict[str, Any]) -> str:
    if metadata.get("scene_label"):
        return str(metadata.get("scene_label") or "")[:40]
    retrieval = metadata.get("retrieval")
    if isinstance(retrieval, dict):
        return str(retrieval.get("scene_label") or "")[:40]
    return ""


def _normalize_correction(raw: Dict[str, Any]) -> Dict[str, Any] | None:
    correction_id = str(raw.get("id") or raw.get("correction_id") or "").strip()
    if not correction_id:
        return None
    status = str(raw.get("status") or "active").strip()[:24] or "active"
    source = raw.get("source") if isinstance(raw.get("source"), dict) else {}
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    corrected_reply = _clean_text(raw.get("corrected_reply"), 500)
    if status == "active" and not corrected_reply:
        return None
    raw_message = _clean_text(raw.get("message"), 500)
    message_hash = str(raw.get("message_hash") or "").strip()[:24] or _sha1_short(raw_message)
    raw_terms = raw.get("message_terms")
    if isinstance(raw_terms, list):
        message_terms = [str(item)[:24] for item in raw_terms if str(item).strip()][:12]
    else:
        message_terms = _correction_terms(raw_message)
    return {
        "schema_version": 1,
        "id": correction_id[:16],
        "time": str(raw.get("time") or "").strip()[:32],
        "actor_id": str(raw.get("actor_id") or "")[:24],
        "review_id": str(raw.get("review_id") or "")[:16],
        "status": status,
        "source": {
            "chat_type": str(source.get("chat_type") or "private")[:16],
            "target_id": str(source.get("target_id") or "")[:24],
            "target_note": str(source.get("target_note") or "")[:80],
            "trigger": str(source.get("trigger") or "")[:24],
        },
        "message_hash": message_hash,
        "message_hint": str(raw.get("message_hint") or _message_hint(raw_message))[:80],
        "message_terms": message_terms,
        "recent_turn_count": int(raw.get("recent_turn_count") or len([
            item for item in (raw.get("recent_dialogue") or []) if isinstance(item, dict)
        ])),
        "bad_candidates": [
            _clean_text(item, 220)
            for item in (raw.get("bad_candidates") or raw.get("candidates") or [])[:8]
            if _clean_text(item, 220)
        ],
        "corrected_reply": corrected_reply,
        "reason": _clean_text(raw.get("reason"), 240),
        "metadata": metadata,
        "scene_label": str(raw.get("scene_label") or _source_scene(metadata))[:40],
        "disabled_at": str(raw.get("disabled_at") or "")[:32],
        "disabled_by": str(raw.get("disabled_by") or "")[:24],
    }


def load_corrections(
    *,
    path: Path | str = CORRECTIONS_PATH,
    include_disabled: bool = False,
) -> List[Dict[str, Any]]:
    """Load compacted correction entries from JSONL."""
    compacted: Dict[str, Dict[str, Any]] = {}
    for raw in _iter_jsonl(Path(path)) or []:
        item = _normalize_correction(raw)
        if not item:
            continue
        correction_id = item["id"]
        if item["status"] in {"disabled", "inactive", "deleted"}:
            previous = compacted.get(correction_id, item)
            previous.update({
                "status": "disabled",
                "disabled_at": item.get("disabled_at") or item.get("time") or _now_iso(),
                "disabled_by": item.get("disabled_by") or item.get("actor_id") or "",
                "reason": item.get("reason") or previous.get("reason") or "",
            })
            compacted[correction_id] = previous
            continue
        compacted[correction_id] = item

    values = [
        item
        for item in compacted.values()
        if include_disabled or item.get("status") == "active"
    ]
    values.sort(key=lambda item: str(item.get("time") or ""), reverse=True)
    return values


def append_correction_from_feedback(
    feedback: Dict[str, Any],
    *,
    path: Path | str = CORRECTIONS_PATH,
) -> Dict[str, Any] | None:
    """Append a generation-facing correction row from /改成 feedback."""
    if str(feedback.get("action") or "") != "correct":
        return None
    corrected = _clean_text(feedback.get("corrected_reply"), 500)
    if not corrected:
        return None
    source = feedback.get("source") if isinstance(feedback.get("source"), dict) else {}
    metadata = feedback.get("metadata") if isinstance(feedback.get("metadata"), dict) else {}
    source_message = _clean_text(feedback.get("message"), 500)
    base = "|".join([
        str(feedback.get("review_id") or ""),
        str(feedback.get("time") or ""),
        corrected,
    ])
    correction = {
        "schema_version": 1,
        "id": "C" + hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()[:9],
        "time": str(feedback.get("time") or _now_iso()),
        "actor_id": str(feedback.get("actor_id") or "")[:24],
        "review_id": str(feedback.get("review_id") or "")[:16],
        "source": source,
        "message_hash": _sha1_short(source_message),
        "message_hint": _message_hint(source_message),
        "message_terms": _correction_terms(source_message),
        "recent_turn_count": len([
            item for item in (feedback.get("recent_dialogue") or []) if isinstance(item, dict)
        ]),
        "bad_candidates": [
            _clean_text(item, 220)
            for item in (feedback.get("candidates") or [])[:8]
            if _clean_text(item, 220)
        ],
        "corrected_reply": corrected,
        "reason": _clean_text(feedback.get("reason"), 240),
        "metadata": metadata,
        "scene_label": _source_scene(metadata),
        "status": "active",
    }
    normalized = _normalize_correction(correction)
    if not normalized:
        return None
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n")
    return normalized


def deactivate_correction(
    correction_id: str,
    *,
    actor_id: str | int = "",
    reason: str = "",
    path: Path | str = CORRECTIONS_PATH,
) -> tuple[bool, str]:
    active = {item["id"]: item for item in load_corrections(path=path)}
    target = correction_id.strip()
    if target not in active:
        return False, "没有找到这个 active 纠正记录。"
    tombstone = {
        "schema_version": 1,
        "id": target,
        "time": _now_iso(),
        "status": "disabled",
        "disabled_at": _now_iso(),
        "disabled_by": str(actor_id)[:24],
        "reason": _clean_text(reason or "手动停用", 240),
    }
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(tombstone, ensure_ascii=False, separators=(",", ":")) + "\n")
    return True, f"已停用纠正记录：{target}"


def correction_stats(*, path: Path | str = CORRECTIONS_PATH) -> Dict[str, Any]:
    all_items = load_corrections(path=path, include_disabled=True)
    active = [item for item in all_items if item.get("status") == "active"]
    return {
        "total_count": len(all_items),
        "active_count": len(active),
        "disabled_count": len(all_items) - len(active),
        "latest_time": active[0].get("time") if active else "",
    }


def select_relevant_corrections(
    latest_message: str,
    *,
    chat_type: str | None = None,
    target_id: str | int | None = None,
    scene_label: str | None = None,
    limit: int = MAX_CORRECTIONS_IN_PROMPT,
    path: Path | str = CORRECTIONS_PATH,
) -> List[Dict[str, Any]]:
    message = str(latest_message or "")
    message_hash = _sha1_short(message)
    message_terms = _correction_terms(message)
    target = str(target_id or "")
    scene = str(scene_label or "")
    ranked = []
    for item in load_corrections(path=path):
        source = item.get("source") or {}
        if chat_type and source.get("chat_type") and source.get("chat_type") != chat_type:
            continue
        same_hash = bool(message_hash and item.get("message_hash") == message_hash)
        similarity = _term_similarity(message_terms, item.get("message_terms") or [])
        same_target = bool(target and source.get("target_id") == target)
        same_scene = bool(scene and item.get("scene_label") == scene)
        score = (
            (0.8 if same_hash else 0.0)
            + similarity
            + (0.35 if same_target else 0.0)
            + (0.15 if same_scene else 0.0)
        )
        if not same_hash and similarity < 0.12 and not (same_target and same_scene):
            continue
        enriched = dict(item)
        enriched["relevance_score"] = round(score, 4)
        ranked.append(enriched)
    ranked.sort(
        key=lambda item: (float(item.get("relevance_score") or 0), str(item.get("time") or "")),
        reverse=True,
    )
    return ranked[: max(0, min(8, int(limit)))]


def load_style_skill_context(
    *,
    chat_type: str | None = None,
    target_id: str | int | None = None,
    scene_label: str | None = None,
    latest_message: str = "",
    root: Path | str = STYLE_SKILL_DIR,
) -> Dict[str, Any]:
    """Load bounded runtime 36.skill context for owner-style generation."""
    root_path = Path(root)
    target = str(target_id or "").strip()
    relationship_text = ""
    relationship_path = None
    if target:
        candidate = root_path / "relationship_profiles" / f"{target}.md"
        if candidate.exists():
            relationship_path = candidate
            relationship_text = _read_md(candidate, MAX_RELATIONSHIP_CHARS)
    corrections = select_relevant_corrections(
        latest_message,
        chat_type=chat_type,
        target_id=target,
        scene_label=scene_label,
        path=root_path / "corrections.jsonl",
    )
    return {
        "ok": True,
        "enabled": root_path.exists(),
        "root": str(root_path),
        "chat_type": str(chat_type or ""),
        "target_id": target,
        "scene_label": str(scene_label or ""),
        "global_persona": _read_md(root_path / "global_persona.md", MAX_MD_CHARS),
        "style_rules": _read_md(root_path / "style_rules.md", MAX_MD_CHARS),
        "memory_patterns": _read_md(root_path / "memory_patterns.md", MAX_MD_CHARS),
        "relationship_profile": relationship_text,
        "relationship_profile_path": str(relationship_path) if relationship_path else "",
        "relationship_profile_found": bool(relationship_text),
        "corrections": corrections,
        "correction_hit_count": len(corrections),
    }


def format_style_skill_context_for_prompt(context: Dict[str, Any] | None) -> str:
    if not context or not context.get("enabled"):
        return ""
    lines = ["36.skill 运行时人格层："]
    if context.get("global_persona"):
        lines.extend(["全局人格：", str(context["global_persona"])])
    if context.get("relationship_profile"):
        lines.extend(["当前关系画像：", str(context["relationship_profile"])])
    if context.get("style_rules"):
        lines.extend(["风格规则：", str(context["style_rules"])])
    if context.get("memory_patterns"):
        lines.extend(["记忆/场景规则：", str(context["memory_patterns"])])
    corrections = context.get("corrections") or []
    if corrections:
        lines.append("主人已纠正过的类似场景：")
        for item in corrections[:MAX_CORRECTIONS_IN_PROMPT]:
            lines.append(
                f"- 触发摘要：{item.get('message_hint') or 'unknown'}; "
                f"匹配词：{'/'.join(item.get('message_terms') or []) or 'none'}; "
                f"应这样回：{item.get('corrected_reply')}; "
                f"不要重复坏候选：{' / '.join((item.get('bad_candidates') or [])[:3])}"
            )
    lines.append("36.skill 优先级：硬安全规则 > 主人纠正 > 当前关系画像 > 相似历史样本 > 全局口癖。")
    return "\n".join(line for line in lines if str(line).strip())


def format_correction_status(*, path: Path | str = CORRECTIONS_PATH) -> str:
    stats = correction_stats(path=path)
    return "\n".join([
        "教学纠正层：",
        f"- active：{stats['active_count']} 条",
        f"- disabled：{stats['disabled_count']} 条",
        f"- 最近更新时间：{stats.get('latest_time') or '无'}",
        "用法：/教学 纠正 最近；/教学 纠正 停用 <id>",
    ])


def format_recent_corrections(limit: int = 8, *, path: Path | str = CORRECTIONS_PATH) -> str:
    items = load_corrections(path=path, include_disabled=True)[: max(1, min(20, int(limit or 8)))]
    if not items:
        return "暂无教学纠正记录。"
    lines = ["最近教学纠正："]
    for item in items:
        source = item.get("source") or {}
        lines.append(
            f"- {item.get('id')} {item.get('status')} "
            f"{source.get('chat_type')}:{source.get('target_id') or 'unknown'} "
            f"触发={str(item.get('message_hint') or '')[:24]} "
            f"改成={str(item.get('corrected_reply') or '')[:24]}"
        )
    lines.append("用法：/教学 纠正 停用 <id>")
    return "\n".join(lines)


def candidate_correction_delta(text: str, corrections: Sequence[Dict[str, Any]]) -> tuple[int, List[str]]:
    """Return rerank delta from selected corrections."""
    delta = 0
    reasons: List[str] = []
    for item in corrections[:MAX_CORRECTIONS_IN_PROMPT]:
        corrected = str(item.get("corrected_reply") or "")
        bad_candidates = item.get("bad_candidates") or []
        good_sim = _text_similarity(text, corrected)
        bad_sim = max((_text_similarity(text, bad) for bad in bad_candidates), default=0.0)
        if good_sim >= 0.45:
            delta += 18
            reasons.append("matches_correction")
        if bad_sim >= 0.45:
            delta -= 30
            reasons.append("matches_rejected_candidate")
    return delta, reasons
