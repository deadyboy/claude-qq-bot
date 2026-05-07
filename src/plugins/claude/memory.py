"""会话记忆管理模块"""

import os
import json
import time
import aiofiles
from pathlib import Path
from typing import List, Dict, Any, Optional

SESSION_DIR = Path("data/sessions")

class SessionManager:
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.timeout = int(os.getenv("SESSION_TIMEOUT", "3600"))
        SESSION_DIR.mkdir(parents=True, exist_ok=True)

    def _get_session_file(self, session_id: str) -> Path:
        return SESSION_DIR / f"{session_id}.json"

    async def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话历史消息"""
        session_file = self._get_session_file(session_id)
        if not session_file.exists():
            return []

        try:
            async with aiofiles.open(session_file, "r", encoding="utf-8") as f:
                data = json.loads(await f.read())
            # 检查会话是否超时
            if time.time() - data.get("last_active", 0) > self.timeout:
                return []
            return data.get("messages", [])
        except (json.JSONDecodeError, IOError):
            return []

    async def add_message(self, session_id: str, role: str, content: str):
        """添加消息到会话历史"""
        session_file = self._get_session_file(session_id)
        session_data = {"messages": [], "last_active": time.time()}

        if session_file.exists():
            try:
                async with aiofiles.open(session_file, "r", encoding="utf-8") as f:
                    session_data = json.loads(await f.read())
            except (json.JSONDecodeError, IOError):
                pass

        session_data["messages"].append({"role": role, "content": content})
        # 保留最近的 max_messages 条消息
        if len(session_data["messages"]) > self.max_messages:
            session_data["messages"] = session_data["messages"][-self.max_messages:]

        session_data["last_active"] = time.time()

        async with aiofiles.open(session_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(session_data, ensure_ascii=False, indent=2))

    async def clear_session(self, session_id: str):
        """清空会话历史"""
        session_file = self._get_session_file(session_id)
        if session_file.exists():
            session_file.unlink()

session_manager = SessionManager()
