"""Pending action confirmation and audit logging."""

import json
import os
import threading
import time
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


PENDING_ACTIONS_PATH = Path("data/pending_actions.json")
ACTION_LOG_PATH = Path("data/action_logs.jsonl")
DEFAULT_TTL_SECONDS = 10 * 60
_PATH_LOCKS: dict[Path, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _path_lock(path: Path) -> threading.RLock:
    key = path.resolve()
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _now_ts() -> float:
    return time.time()


def _normalize_actor(value: str | int) -> str:
    return str(value).strip()


def _sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only compact scalar payload values for local pending-action storage."""
    clean: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            text = str(value) if value is not None else ""
            clean[str(key)] = text[:200]
    return clean


def normalize_chat_scope(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    scope = {
        "chat_type": str(raw.get("chat_type") or "").strip()[:16],
        "target_id": str(raw.get("target_id") or "").strip()[:32],
        "session_id": str(raw.get("session_id") or "").strip()[:80],
    }
    return {key: value for key, value in scope.items() if value}


def _scope_matches(action_scope: Dict[str, str], current_scope: Dict[str, str]) -> bool:
    if not action_scope:
        return True
    if not current_scope:
        return False
    for key in ("chat_type", "target_id", "session_id"):
        expected = action_scope.get(key)
        if expected and current_scope.get(key) != expected:
            return False
    return True


def normalize_action(raw: Any) -> Dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    action_id = str(raw.get("id", "")).strip()
    action_type = str(raw.get("type", "")).strip()
    created_by = str(raw.get("created_by", "")).strip()
    summary = str(raw.get("summary", "")).strip()
    if not action_id or not action_type or not created_by or not summary:
        return None

    try:
        expires_at = float(raw.get("expires_at", 0))
    except (TypeError, ValueError):
        expires_at = 0

    payload = raw.get("payload")
    return {
        "id": action_id[:16],
        "type": action_type[:80],
        "created_by": created_by[:24],
        "summary": summary[:400],
        "payload": _sanitize_payload(payload if isinstance(payload, dict) else {}),
        "chat_scope": normalize_chat_scope(raw.get("chat_scope")),
        "created_at": str(raw.get("created_at", "")).strip(),
        "expires_at": expires_at,
    }


class ConfirmationStore:
    """JSON-backed pending confirmation store plus JSONL audit log."""

    def __init__(
        self,
        pending_path: Path | str = PENDING_ACTIONS_PATH,
        log_path: Path | str = ACTION_LOG_PATH,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        self.pending_path = Path(pending_path)
        self.log_path = Path(log_path)
        self.ttl_seconds = ttl_seconds

    def _load_all(self) -> Dict[str, Dict[str, Any]]:
        with _path_lock(self.pending_path):
            if not self.pending_path.exists():
                return {}
            try:
                with self.pending_path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
            except (OSError, json.JSONDecodeError):
                return {}
        if not isinstance(raw, dict):
            return {}

        actions = {}
        now = _now_ts()
        for action_id, action in raw.items():
            normalized = normalize_action(action)
            if not normalized:
                continue
            if normalized["expires_at"] <= now:
                continue
            actions[str(action_id)] = normalized
        return actions

    def _save_all(self, actions: Dict[str, Dict[str, Any]]) -> None:
        with _path_lock(self.pending_path):
            _atomic_write_json(self.pending_path, actions)

    def create(
        self,
        action_type: str,
        created_by: str | int,
        summary: str,
        payload: Dict[str, Any] | None = None,
        chat_scope: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        with _path_lock(self.pending_path):
            actions = self._load_all()
            action_id = uuid.uuid4().hex[:8]
            action = {
                "id": action_id,
                "type": action_type,
                "created_by": _normalize_actor(created_by),
                "summary": summary.strip()[:400],
                "payload": _sanitize_payload(payload or {}),
                "chat_scope": normalize_chat_scope(chat_scope),
                "created_at": _now_iso(),
                "expires_at": _now_ts() + self.ttl_seconds,
            }
            actions[action_id] = action
            self._save_all(actions)
        self.log(action, actor_id=created_by, status="pending")
        return deepcopy(action)

    def list_for_actor(
        self,
        actor_id: str | int,
        chat_scope: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        actor = _normalize_actor(actor_id)
        scope = normalize_chat_scope(chat_scope)
        return [
            deepcopy(action)
            for action in self._load_all().values()
            if action.get("created_by") == actor
            and (not scope or _scope_matches(action.get("chat_scope") or {}, scope))
        ]

    def pop_for_actor(
        self,
        action_id: str,
        actor_id: str | int,
        chat_scope: Dict[str, Any] | None = None,
    ) -> tuple[Dict[str, Any] | None, str]:
        key = action_id.strip()
        with _path_lock(self.pending_path):
            actions = self._load_all()
            action = actions.get(key)
            if not action:
                return None, "没有找到这个待确认操作，可能已过期或已处理。"
            if action.get("created_by") != _normalize_actor(actor_id):
                return None, "这个待确认操作不属于当前用户。"
            if not _scope_matches(action.get("chat_scope") or {}, normalize_chat_scope(chat_scope)):
                return None, "这个待确认操作需要在创建它的私聊或群聊里执行。"
            del actions[key]
            self._save_all(actions)
        return deepcopy(action), ""

    def cancel_for_actor(
        self,
        action_id: str,
        actor_id: str | int,
        chat_scope: Dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        action, error = self.pop_for_actor(action_id, actor_id, chat_scope=chat_scope)
        if not action:
            return False, error
        self.log(action, actor_id=actor_id, status="cancelled")
        return True, f"已取消：{action['summary']}"

    def log(self, action: Dict[str, Any], actor_id: str | int, status: str, result: str = "") -> None:
        entry = {
            "time": _now_iso(),
            "actor_id": _normalize_actor(actor_id),
            "action_id": str(action.get("id", ""))[:16],
            "action_type": str(action.get("type", ""))[:80],
            "status": status[:40],
            "summary": str(action.get("summary", ""))[:400],
            "result": result[:400],
            "chat_scope": normalize_chat_scope(action.get("chat_scope")),
        }
        with _path_lock(self.log_path):
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def clear_for_tests(self) -> None:
        for path in (self.pending_path, self.log_path):
            if path.exists():
                path.unlink()


confirmation_store = ConfirmationStore()


def format_confirmation_request(action: Dict[str, Any]) -> str:
    """Format the prompt shown after creating a pending action."""
    return "\n".join([
        "需要确认：",
        f"- 操作：{action['summary']}",
        f"- 确认ID：{action['id']}",
        f"- 有效期：{DEFAULT_TTL_SECONDS // 60} 分钟",
        f"执行：/确认 {action['id']}",
        f"取消：/取消 {action['id']}",
    ])


def format_pending_actions(actions: List[Dict[str, Any]]) -> str:
    if not actions:
        return "当前没有待确认操作。"
    lines = ["待确认操作："]
    for action in actions[:10]:
        lines.append(f"- {action['id']}：{action['summary']}")
    lines.append("用法：/确认 <id>；/取消 <id>")
    return "\n".join(lines)


def parse_confirmation_payload(text: str, prefixes: tuple[str, ...]) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    for prefix in prefixes:
        if lowered == prefix.lower():
            return ""
        if lowered.startswith(prefix.lower() + " "):
            return stripped[len(prefix):].strip()
        if stripped.startswith(prefix + "：") or stripped.startswith(prefix + ":"):
            return stripped[len(prefix) + 1:].strip()
    return ""
