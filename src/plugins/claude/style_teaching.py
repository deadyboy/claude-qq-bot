"""Owner style teaching/review feedback store."""

import json
import re
import time
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


STYLE_TEACHING_DIR = Path("data/style_profiles")
ACTIVE_REVIEWS_PATH = STYLE_TEACHING_DIR / "teaching_reviews.json"
FEEDBACK_LOG_PATH = STYLE_TEACHING_DIR / "teaching_feedback.jsonl"
MAX_ACTIVE_REVIEWS = 80
REVIEW_TTL_SECONDS = 24 * 60 * 60


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _now_ts() -> float:
    return time.time()


def _clean_text(value: Any, limit: int = 1200) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _normalize_candidate(item: Any) -> str:
    if isinstance(item, dict):
        text = item.get("text") or item.get("candidate") or ""
    else:
        text = item
    return _clean_text(text, 300)


def _iter_jsonl(path: Path):
    if not path.exists():
        return
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


def _last_other_text(context: List[Dict[str, Any]]) -> str:
    for item in reversed(context):
        if str(item.get("role") or "") == "other":
            text = _clean_text(item.get("text") or item.get("content"), 700)
            if text:
                return text
    for item in reversed(context):
        text = _clean_text(item.get("text") or item.get("content"), 700)
        if text:
            return text
    return ""


def _normalize_review(raw: Any) -> Dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    review_id = str(raw.get("id") or "").strip()
    if not review_id:
        return None
    try:
        created_ts = float(raw.get("created_ts") or 0)
    except (TypeError, ValueError):
        created_ts = 0
    if created_ts and _now_ts() - created_ts > REVIEW_TTL_SECONDS:
        return None
    candidates = [
        candidate
        for candidate in (_normalize_candidate(item) for item in raw.get("candidates") or [])
        if candidate
    ][:5]
    if not candidates:
        return None
    return {
        "id": review_id[:16],
        "created_at": str(raw.get("created_at") or ""),
        "created_ts": created_ts or _now_ts(),
        "status": str(raw.get("status") or "pending")[:24],
        "source": {
            "chat_type": str((raw.get("source") or {}).get("chat_type") or "private")[:16],
            "target_id": str((raw.get("source") or {}).get("target_id") or "")[:24],
            "target_note": str((raw.get("source") or {}).get("target_note") or "")[:80],
            "trigger": str((raw.get("source") or {}).get("trigger") or "manual")[:24],
        },
        "message": _clean_text(raw.get("message"), 1000),
        "recent_dialogue": [
            {
                "role": str((item or {}).get("role") or "")[:16],
                "content": _clean_text((item or {}).get("content") or (item or {}).get("text"), 500),
            }
            for item in (raw.get("recent_dialogue") or [])[-8:]
            if isinstance(item, dict)
        ],
        "candidates": candidates,
        "metadata": raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
        "reviewer_ids": [str(item)[:24] for item in raw.get("reviewer_ids") or [] if str(item).strip()],
    }


class TeachingReviewStore:
    """JSON active-review store plus JSONL feedback log."""

    def __init__(
        self,
        active_path: Path | str = ACTIVE_REVIEWS_PATH,
        feedback_path: Path | str = FEEDBACK_LOG_PATH,
    ):
        self.active_path = Path(active_path)
        self.feedback_path = Path(feedback_path)

    def _load_active(self) -> Dict[str, Dict[str, Any]]:
        if not self.active_path.exists():
            return {}
        try:
            raw = json.loads(self.active_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        reviews = {}
        for key, value in raw.items():
            review = _normalize_review(value)
            if review:
                reviews[str(key)] = review
        return reviews

    def _save_active(self, reviews: Dict[str, Dict[str, Any]]) -> None:
        items = sorted(
            reviews.values(),
            key=lambda item: float(item.get("created_ts") or 0),
            reverse=True,
        )[:MAX_ACTIVE_REVIEWS]
        self.active_path.parent.mkdir(parents=True, exist_ok=True)
        self.active_path.write_text(
            json.dumps({item["id"]: item for item in items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def create_review(
        self,
        *,
        message: str,
        candidates: List[str],
        chat_type: str = "private",
        target_id: str | int = "",
        target_note: str = "",
        trigger: str = "manual",
        recent_dialogue: List[Dict[str, Any]] | None = None,
        reviewer_ids: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        clean_candidates = [candidate for candidate in (_normalize_candidate(item) for item in candidates) if candidate][:5]
        if not clean_candidates:
            raise ValueError("至少需要 1 条候选回复。")
        review_id = "T" + uuid.uuid4().hex[:7]
        review = _normalize_review({
            "id": review_id,
            "created_at": _now_iso(),
            "created_ts": _now_ts(),
            "status": "pending",
            "source": {
                "chat_type": chat_type,
                "target_id": target_id,
                "target_note": target_note,
                "trigger": trigger,
            },
            "message": message,
            "recent_dialogue": recent_dialogue or [],
            "candidates": clean_candidates,
            "metadata": metadata or {},
            "reviewer_ids": reviewer_ids or [],
        })
        if not review:
            raise ValueError("教学样本创建失败。")
        reviews = self._load_active()
        reviews[review_id] = review
        self._save_active(reviews)
        return deepcopy(review)

    def get(self, review_id: str) -> Dict[str, Any] | None:
        return deepcopy(self._load_active().get(str(review_id).strip()))

    def latest_for_reviewer(self, reviewer_id: str | int) -> Dict[str, Any] | None:
        actor = str(reviewer_id)
        candidates = []
        for review in self._load_active().values():
            reviewers = review.get("reviewer_ids") or []
            if reviewers and actor not in reviewers:
                continue
            if review.get("status") == "pending":
                candidates.append(review)
        if not candidates:
            return None
        return deepcopy(max(candidates, key=lambda item: float(item.get("created_ts") or 0)))

    def list_recent(self, reviewer_id: str | int | None = None, limit: int = 8) -> List[Dict[str, Any]]:
        actor = str(reviewer_id) if reviewer_id is not None else ""
        items = []
        for review in self._load_active().values():
            reviewers = review.get("reviewer_ids") or []
            if actor and reviewers and actor not in reviewers:
                continue
            items.append(review)
        items.sort(key=lambda item: float(item.get("created_ts") or 0), reverse=True)
        return deepcopy(items[: max(0, int(limit))])

    def record_feedback(
        self,
        review_id: str,
        *,
        actor_id: str | int,
        action: str,
        rating: int | None = None,
        selected_index: int | None = None,
        corrected_reply: str = "",
        reason: str = "",
    ) -> tuple[bool, str, Dict[str, Any] | None]:
        reviews = self._load_active()
        review = reviews.get(str(review_id).strip())
        if not review:
            return False, "没有找到这个教学样本，可能已过期。", None
        candidate_text = ""
        if selected_index is not None:
            if selected_index < 1 or selected_index > len(review.get("candidates") or []):
                return False, f"候选编号应为 1-{len(review.get('candidates') or [])}。", review
            candidate_text = review["candidates"][selected_index - 1]
        clean_rating = None
        if rating is not None:
            clean_rating = max(1, min(5, int(rating)))
        entry = {
            "time": _now_iso(),
            "actor_id": str(actor_id)[:24],
            "review_id": review["id"],
            "action": action[:24],
            "rating": clean_rating,
            "selected_index": selected_index,
            "selected_candidate": candidate_text,
            "corrected_reply": _clean_text(corrected_reply, 1000),
            "reason": _clean_text(reason, 500),
            "source": review.get("source") or {},
            "message": review.get("message") or "",
            "candidates": review.get("candidates") or [],
            "metadata": review.get("metadata") or {},
        }
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with self.feedback_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
        review["status"] = "reviewed"
        reviews[review["id"]] = review
        self._save_active(reviews)
        return True, "已记录教学反馈。", deepcopy(entry)

    def feedback_stats(self) -> Dict[str, Any]:
        stats = {
            "feedback_count": 0,
            "action_counts": {},
            "rating_counts": {},
            "latest_time": "",
            "pending_count": len([item for item in self._load_active().values() if item.get("status") == "pending"]),
        }
        if not self.feedback_path.exists():
            return stats
        with self.feedback_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stats["feedback_count"] += 1
                action = str(item.get("action") or "unknown")
                stats["action_counts"][action] = stats["action_counts"].get(action, 0) + 1
                rating = item.get("rating")
                if rating is not None:
                    key = str(rating)
                    stats["rating_counts"][key] = stats["rating_counts"].get(key, 0) + 1
                stats["latest_time"] = str(item.get("time") or stats["latest_time"])
        return stats

    def clear_for_tests(self) -> None:
        for path in (self.active_path, self.feedback_path):
            if path.exists():
                path.unlink()

    def _used_sample_ids(self) -> set[str]:
        used = set()
        for review in self._load_active().values():
            sample_id = str((review.get("metadata") or {}).get("sample_id") or "").strip()
            if sample_id:
                used.add(sample_id)
        for item in _iter_jsonl(self.feedback_path) or []:
            sample_id = str(((item.get("metadata") or {}) if isinstance(item, dict) else {}).get("sample_id") or "").strip()
            if sample_id:
                used.add(sample_id)
        return used

    def create_replay_batch(
        self,
        *,
        count: int = 10,
        reviewer_ids: List[str] | None = None,
        run_dir: str | Path | None = None,
        scene_label: str = "",
    ) -> List[Dict[str, Any]]:
        """Create fast batch review items from text-grounded SFT candidates."""
        from .style_distill import find_latest_distill_run

        run_path = find_latest_distill_run(run_dir)
        if run_path is None:
            raise FileNotFoundError("还没有找到 Stage 5B 离线蒸馏结果。")
        path = run_path / "sft_candidates.jsonl"
        if not path.exists():
            raise FileNotFoundError("当前蒸馏结果缺少 sft_candidates.jsonl。")

        used = self._used_sample_ids()
        created = []
        max_count = max(1, min(30, int(count or 10)))
        for item in _iter_jsonl(path) or []:
            sample_id = str(item.get("sample_id") or "").strip()
            if not sample_id or sample_id in used:
                continue
            if scene_label and item.get("scene_label") != scene_label:
                continue
            message = _last_other_text(item.get("context") or [])
            target = _clean_text(item.get("target"), 300)
            if not message or not target:
                continue
            review = self.create_review(
                message=message,
                candidates=[target],
                chat_type=str(item.get("chat_type") or "private"),
                target_id=str(item.get("relationship_id") or ""),
                trigger="batch_replay",
                recent_dialogue=item.get("context") or [],
                reviewer_ids=reviewer_ids or [],
                metadata={
                    "run_id": run_path.name,
                    "sample_id": sample_id,
                    "source_file_id": item.get("source_file_id"),
                    "scene_label": item.get("scene_label"),
                    "scope": item.get("scope"),
                    "learning_value": item.get("learning_value"),
                    "grounding_type": item.get("grounding_type"),
                },
            )
            created.append(review)
            used.add(sample_id)
            if len(created) >= max_count:
                break
        return created


teaching_store = TeachingReviewStore()


def format_teaching_review_window(review: Dict[str, Any]) -> str:
    source = review.get("source") or {}
    lines = [
        f"教学审核 #{review.get('id')}",
        f"- 来源：{source.get('chat_type')} {source.get('target_id') or 'unknown'}",
        f"- 触发：{source.get('trigger')}",
        "对方：",
        str(review.get("message") or "")[:500],
        "候选：",
    ]
    for index, candidate in enumerate(review.get("candidates") or [], start=1):
        lines.append(f"{index}. {candidate}")
    lines.extend([
        "操作：",
        f"/采纳 {review.get('id')} 1",
        f"/评分 {review.get('id')} 1-5 原因",
        f"/改成 {review.get('id')} 你的正确回复",
        f"/拒绝 {review.get('id')} 原因",
    ])
    return "\n".join(lines)


def format_teaching_status(enabled: bool, stats: Dict[str, Any]) -> str:
    return "\n".join([
        "教学模式：",
        f"- 影子审核：{'开' if enabled else '关'}",
        f"- 待审核：{stats.get('pending_count', 0)} 条",
        f"- 已记录反馈：{stats.get('feedback_count', 0)} 条",
        f"- 操作分布：{stats.get('action_counts') or {}}",
        f"- 评分分布：{stats.get('rating_counts') or {}}",
        "用法：/教学 开；/教学 关；/教学 最近；/采纳 <id> <1-3>；/改成 <id> <正确回复>",
    ])


def format_recent_reviews(reviews: List[Dict[str, Any]]) -> str:
    if not reviews:
        return "暂无待审核教学样本。"
    lines = ["最近教学样本："]
    for review in reviews:
        source = review.get("source") or {}
        lines.append(
            f"- {review.get('id')} {review.get('status')} "
            f"{source.get('chat_type')}:{source.get('target_id') or 'unknown'} "
            f"候选={len(review.get('candidates') or [])}"
        )
    return "\n".join(lines)


def format_teaching_batch(reviews: List[Dict[str, Any]]) -> str:
    if not reviews:
        return "没有创建新的教学题。可能当前筛选条件下的样本已经出完了。"
    lines = [
        f"已创建 {len(reviews)} 条教学题：",
    ]
    for review in reviews[:20]:
        metadata = review.get("metadata") or {}
        lines.append(
            f"- {review.get('id')} {metadata.get('scene_label') or 'unknown'} "
            f"{metadata.get('scope') or ''}：{str(review.get('message') or '')[:36]}"
        )
    lines.append("用 /教学 最近 查看；用 /采纳 <id> 1、/评分 <id> 1-5、/改成 <id> ... 反馈。")
    return "\n".join(lines)
