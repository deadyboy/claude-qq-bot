"""权限配置与 owner 检查。"""

import json
import os
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Set

from dotenv import load_dotenv


OWNER_ENV = "OWNER_QQ_IDS"
ACCESS_POLICY_PATH = Path("data/permissions.json")

DEFAULT_ACCESS_POLICY: Dict[str, Any] = {
    "trusted_private_users": {},
    "trusted_groups": {},
    "updated_at": None,
}

_project_root = Path(__file__).resolve().parents[3]
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=False)


def parse_owner_ids(raw: str | None) -> Set[str]:
    """解析 OWNER_QQ_IDS，支持逗号、分号和空白分隔。"""
    if not raw:
        return set()
    return {
        item.strip()
        for item in re.split(r"[,\s;，；]+", raw)
        if item.strip()
    }


def get_owner_ids() -> Set[str]:
    """读取 owner QQ 号集合。"""
    return parse_owner_ids(os.getenv(OWNER_ENV, ""))


def has_owner_configured(owner_ids: Iterable[str] | None = None) -> bool:
    """是否已配置至少一个 owner。"""
    ids = set(owner_ids) if owner_ids is not None else get_owner_ids()
    return bool(ids)


def is_owner_user_id(user_id: str | int, owner_ids: Iterable[str] | None = None) -> bool:
    """判断用户是否为 owner。"""
    ids = set(owner_ids) if owner_ids is not None else get_owner_ids()
    return str(user_id) in ids


def is_owner_event(event) -> bool:
    """判断事件发送者是否为 owner。"""
    return is_owner_user_id(getattr(event, "user_id", ""))


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _normalize_id(value: str | int) -> str:
    return str(value).strip()


def _looks_like_qq_id(value: str | int) -> bool:
    text = _normalize_id(value)
    return bool(re.fullmatch(r"\d{5,12}", text))


def normalize_access_policy(raw: Any) -> Dict[str, Any]:
    """Return a sanitized access policy dict."""
    policy = deepcopy(DEFAULT_ACCESS_POLICY)
    if isinstance(raw, dict):
        policy.update(raw)

    for key in ("trusted_private_users", "trusted_groups"):
        normalized = {}
        values = policy.get(key, {})
        if isinstance(values, dict):
            items = values.items()
        elif isinstance(values, list):
            items = ((item, {}) for item in values)
        else:
            items = ()

        for raw_id, raw_meta in items:
            target_id = _normalize_id(raw_id)
            if not _looks_like_qq_id(target_id):
                continue
            meta = raw_meta if isinstance(raw_meta, dict) else {}
            normalized[target_id] = {
                "note": str(meta.get("note", "")).strip()[:80],
                "added_at": str(meta.get("added_at", "")).strip(),
                "added_by": _normalize_id(meta.get("added_by", ""))[:24],
            }
        policy[key] = normalized
    return policy


class AccessPolicyStore:
    """Local JSON store for contact/group trust lists."""

    def __init__(self, path: Path | str = ACCESS_POLICY_PATH):
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return deepcopy(DEFAULT_ACCESS_POLICY)
        try:
            with self.path.open("r", encoding="utf-8") as f:
                return normalize_access_policy(json.load(f))
        except (OSError, json.JSONDecodeError):
            return deepcopy(DEFAULT_ACCESS_POLICY)

    def save(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        normalized = normalize_access_policy(policy)
        normalized["updated_at"] = _now_iso()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)
        return normalized

    def is_trusted_user(self, user_id: str | int) -> bool:
        return _normalize_id(user_id) in self.load().get("trusted_private_users", {})

    def is_trusted_group(self, group_id: str | int) -> bool:
        return _normalize_id(group_id) in self.load().get("trusted_groups", {})

    def add_user(self, user_id: str | int, note: str = "", added_by: str | int = "") -> tuple[bool, str]:
        target_id = _normalize_id(user_id)
        if not _looks_like_qq_id(target_id):
            return False, "用户 QQ 号格式不正确。"
        policy = self.load()
        policy.setdefault("trusted_private_users", {})[target_id] = {
            "note": note.strip()[:80],
            "added_at": _now_iso(),
            "added_by": _normalize_id(added_by)[:24],
        }
        self.save(policy)
        return True, f"已加入信任用户：{target_id}"

    def remove_user(self, user_id: str | int) -> tuple[bool, str]:
        target_id = _normalize_id(user_id)
        policy = self.load()
        removed = policy.get("trusted_private_users", {}).pop(target_id, None)
        self.save(policy)
        if removed is None:
            return False, "信任用户名单中没有这个 QQ。"
        return True, f"已移除信任用户：{target_id}"

    def add_group(self, group_id: str | int, note: str = "", added_by: str | int = "") -> tuple[bool, str]:
        target_id = _normalize_id(group_id)
        if not _looks_like_qq_id(target_id):
            return False, "群号格式不正确。"
        policy = self.load()
        policy.setdefault("trusted_groups", {})[target_id] = {
            "note": note.strip()[:80],
            "added_at": _now_iso(),
            "added_by": _normalize_id(added_by)[:24],
        }
        self.save(policy)
        return True, f"已加入信任群：{target_id}"

    def remove_group(self, group_id: str | int) -> tuple[bool, str]:
        target_id = _normalize_id(group_id)
        policy = self.load()
        removed = policy.get("trusted_groups", {}).pop(target_id, None)
        self.save(policy)
        if removed is None:
            return False, "信任群名单中没有这个群。"
        return True, f"已移除信任群：{target_id}"

    def summary(self, include_ids: bool = False) -> str:
        policy = self.load()
        users = policy.get("trusted_private_users", {})
        groups = policy.get("trusted_groups", {})
        lines = [
            "信任名单：",
            f"- 信任用户：{len(users)} 个",
            f"- 信任群：{len(groups)} 个",
            "- 当前作用：仅作为未来自动代聊/高风险工具的权限基座；不会改变普通聊天默认行为。",
        ]
        if include_ids:
            user_line = "、".join(users.keys()) if users else "无"
            group_line = "、".join(groups.keys()) if groups else "无"
            lines.extend([
                f"- 用户列表：{user_line}",
                f"- 群列表：{group_line}",
            ])
        return "\n".join(lines)


access_store = AccessPolicyStore()


def format_permission_status(user_id: str | int) -> str:
    """生成权限状态文本，不暴露完整 owner 列表。"""
    owner_ids = get_owner_ids()
    role = "主人" if is_owner_user_id(user_id, owner_ids) else "普通用户"
    trusted = "是" if access_store.is_trusted_user(user_id) else "否"
    return "\n".join([
        "权限状态：",
        f"- 当前身份：{role}",
        f"- 信任用户：{trusted}",
        "- 管理命令需要主人权限。",
    ])


def owner_required_message(action: str = "这个命令") -> str:
    """权限不足时的提示。"""
    if not has_owner_configured():
        return f"{action}需要主人权限，但当前未配置 OWNER_QQ_IDS。"
    return f"{action}仅主人可用。"
