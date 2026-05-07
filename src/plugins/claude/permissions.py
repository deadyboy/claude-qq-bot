"""权限配置与 owner 检查。"""

import os
import re
from pathlib import Path
from typing import Iterable, Set

from dotenv import load_dotenv


OWNER_ENV = "OWNER_QQ_IDS"

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


def format_permission_status(user_id: str | int) -> str:
    """生成权限状态文本，不暴露完整 owner 列表。"""
    owner_ids = get_owner_ids()
    role = "主人" if is_owner_user_id(user_id, owner_ids) else "普通用户"
    return "\n".join([
        "权限状态：",
        f"- 当前身份：{role}",
        "- 管理命令需要主人权限。",
    ])


def owner_required_message(action: str = "这个命令") -> str:
    """权限不足时的提示。"""
    if not has_owner_configured():
        return f"{action}需要主人权限，但当前未配置 OWNER_QQ_IDS。"
    return f"{action}仅主人可用。"
