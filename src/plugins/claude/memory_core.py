"""
混合记忆系统核心模块

三层记忆架构:
1. 短期记忆 - 当前会话的对话历史 (JSON 文件)
2. 长期记忆 - 向量数据库存储，支持语义检索 (ChromaDB)
3. 关键事实 - SQLite 存储任务/用户偏好/承诺
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, asdict
from datetime import datetime
import aiofiles

# 延迟导入，避免 ImportError
try:
    import aiosqlite
except ImportError:
    aiosqlite = None

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

# ==================== 数据结构定义 ====================

@dataclass
class MemoryEntry:
    """记忆条目基类"""
    id: str
    created_at: float
    content: str
    tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ShortTermMemory(MemoryEntry):
    """短期记忆 - 对话历史"""
    session_id: str = ""
    role: str = "user"
    message_index: int = 0

@dataclass
class LongTermMemory(MemoryEntry):
    """长期记忆 - 语义化存储"""
    memory_type: Literal["conversation", "fact", "skill", "experience"] = "conversation"
    importance: float = 0.5  # 重要性评分 0-1
    access_count: int = 0  # 被访问次数
    last_accessed: float = 0
    embedding: Optional[List[float]] = None  # 向量嵌入

@dataclass
class KeyFact(MemoryEntry):
    """关键事实 - 结构化存储"""
    fact_type: Literal["user_profile", "task", "commitment", "preference", "relationship"] = "fact"
    subject: str = ""  # 主体 (如用户 ID)
    predicate: str = ""  # 关系/属性
    object: str = ""  # 值/对象
    confidence: float = 1.0  # 置信度
    verified: bool = False  # 是否已验证

@dataclass
class TaskRecord:
    """任务记录"""
    id: str
    title: str
    description: str
    status: Literal["pending", "in_progress", "blocked", "completed", "cancelled"] = "pending"
    priority: int = 5  # 1-10, 10 最高
    created_at: float = 0
    updated_at: float = 0
    deadline: Optional[float] = None
    parent_task: Optional[str] = None
    subtasks: List[str] = None
    dependencies: List[str] = None
    assigned_to: str = "agent"  # agent 或 user
    result: Optional[str] = None

    def __post_init__(self):
        if self.subtasks is None:
            self.subtasks = []
        if self.dependencies is None:
            self.dependencies = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ==================== 短期记忆管理器 ====================

class ShortTermMemoryManager:
    """
    短期记忆管理器 - 管理当前会话的对话历史
    改进点：支持动态长度、重要性标记、快速访问
    """

    def __init__(self, max_messages: int = 50, timeout: int = 7200):
        self.max_messages = max_messages
        self.timeout = timeout
        self.session_dir = Path("data/sessions")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[Dict]] = {}  # 内存缓存

    def _get_session_file(self, session_id: str) -> Path:
        return self.session_dir / f"{session_id}.json"

    async def get_messages(self, session_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """获取会话历史"""
        # 优先从缓存读取
        if session_id in self._cache:
            messages = self._cache[session_id]
        else:
            session_file = self._get_session_file(session_id)
            if not session_file.exists():
                return []

            async with aiofiles.open(session_file, "r", encoding="utf-8") as f:
                data = json.loads(await f.read())
            messages = data.get("messages", [])
            self._cache[session_id] = messages

        # 检查超时
        session_file = self._get_session_file(session_id)
        if session_file.exists():
            async with aiofiles.open(session_file, "r", encoding="utf-8") as f:
                data = json.loads(await f.read())
            if time.time() - data.get("last_active", 0) > self.timeout:
                return []  # 会话超时

        if limit:
            return messages[-limit:]
        return messages

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ):
        """添加消息到会话"""
        session_file = self._get_session_file(session_id)

        # 加载或创建
        if session_file.exists():
            async with aiofiles.open(session_file, "r", encoding="utf-8") as f:
                session_data = json.loads(await f.read())
        else:
            session_data = {"messages": [], "created_at": time.time()}

        # 添加消息
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            **(metadata or {})
        }
        session_data["messages"].append(message)

        # 裁剪到最大长度
        if len(session_data["messages"]) > self.max_messages:
            # 保留重要的消息
            important = [
                m for m in session_data["messages"][:len(session_data["messages"])//2]
                if m.get("important", False)
            ]
            recent = session_data["messages"][-(self.max_messages - len(important)):]
            session_data["messages"] = important + recent

        session_data["last_active"] = time.time()

        # 保存
        async with aiofiles.open(session_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(session_data, ensure_ascii=False, indent=2))

        # 更新缓存
        self._cache[session_id] = session_data["messages"]

    async def clear(self, session_id: str):
        """清空会话"""
        session_file = self._get_session_file(session_id)
        if session_file.exists():
            session_file.unlink()
        if session_id in self._cache:
            del self._cache[session_id]

    async def mark_important(self, session_id: str, message_index: int):
        """标记重要消息"""
        messages = await self.get_messages(session_id)
        if 0 <= message_index < len(messages):
            messages[message_index]["important"] = True
            session_file = self._get_session_file(session_id)
            async with aiofiles.open(session_file, "w", encoding="utf-8") as f:
                data = {"messages": messages, "last_active": time.time()}
                await f.write(json.dumps(data, ensure_ascii=False, indent=2))

# ==================== 长期记忆管理器 (ChromaDB) ====================

class LongTermMemoryManager:
    """
    长期记忆管理器 - 使用向量数据库支持语义检索

    记忆类型:
    - conversation: 重要对话片段
    - fact: 事实性知识
    - skill: 学到的技能/模式
    - experience: 经验教训
    """

    def __init__(self, persist_dir: str = "data/longterm_memory"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection = None
        self._initialized = False

    async def _ensure_initialized(self):
        """懒加载 ChromaDB"""
        if self._initialized:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            # 初始化持久化客户端
            client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False)
            )

            # 创建或获取集合
            self._collection = client.get_or_create_collection(
                name="longterm_memory",
                metadata={"hnsw:space": "cosine"}
            )
            self._initialized = True
        except ImportError:
            # ChromaDB 未安装，降级到简单存储
            self._collection = None
            self._initialized = True

    async def add(
        self,
        content: str,
        memory_type: str = "conversation",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """添加长期记忆"""
        await self._ensure_initialized()

        memory_id = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()[:16]

        if self._collection is None:
            # 降级模式：简单 JSON 存储
            await self._add_simple(memory_id, content, memory_type, tags, metadata)
            return memory_id

        # 构建元数据
        meta = {
            "memory_id": memory_id,
            "memory_type": memory_type,
            "tags": json.dumps(tags or []),
            "created_at": time.time(),
            "importance": metadata.get("importance", 0.5) if metadata else 0.5,
            **(metadata or {})
        }

        # 添加到向量库
        self._collection.add(
            documents=[content],
            metadatas=[meta],
            ids=[memory_id]
        )

        return memory_id

    async def search(
        self,
        query: str,
        memory_type: Optional[str] = None,
        tags: List[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """语义搜索记忆"""
        await self._ensure_initialized()

        if self._collection is None:
            return await self._search_simple(query, memory_type, tags, limit)

        # 构建过滤条件
        where = {}
        if memory_type:
            where["memory_type"] = memory_type

        results = self._collection.query(
            query_texts=[query],
            n_results=limit,
            where=where if where else None,
            include=["documents", "metadatas", "distances"]
        )

        # 格式化结果
        memories = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                memories.append({
                    "id": meta.get("memory_id", ""),
                    "content": doc,
                    "type": meta.get("memory_type", "conversation"),
                    "tags": json.loads(meta.get("tags", "[]")),
                    "importance": meta.get("importance", 0.5),
                    "created_at": meta.get("created_at", 0),
                    "distance": results["distances"][0][i] if results["distances"] else 0
                })

        return memories

    async def _add_simple(
        self,
        memory_id: str,
        content: str,
        memory_type: str,
        tags: List[str],
        metadata: Dict
    ):
        """降级模式：JSON 文件存储"""
        index_file = self.persist_dir / "memory_index.json"

        if index_file.exists():
            async with aiofiles.open(index_file, "r", encoding="utf-8") as f:
                index = json.loads(await f.read())
        else:
            index = {"memories": {}}

        index["memories"][memory_id] = {
            "content": content,
            "memory_type": memory_type,
            "tags": tags or [],
            "created_at": time.time(),
            **(metadata or {})
        }

        async with aiofiles.open(index_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(index, ensure_ascii=False, indent=2))

    async def _search_simple(
        self,
        query: str,
        memory_type: Optional[str],
        tags: List[str],
        limit: int
    ) -> List[Dict]:
        """降级模式：关键词搜索"""
        index_file = self.persist_dir / "memory_index.json"
        if not index_file.exists():
            return []

        async with aiofiles.open(index_file, "r", encoding="utf-8") as f:
            index = json.loads(await f.read())

        # 简单关键词匹配
        query_words = set(query.lower().split())
        results = []

        for mid, memory in index.get("memories", {}).items():
            if memory_type and memory.get("memory_type") != memory_type:
                continue

            # 关键词匹配分数
            content_words = set(memory["content"].lower().split())
            score = len(query_words & content_words) / max(len(query_words), 1)

            if score > 0.1:  # 阈值
                results.append({
                    "id": mid,
                    "content": memory["content"],
                    "type": memory.get("memory_type", "conversation"),
                    "tags": memory.get("tags", []),
                    "score": score
                })

        return sorted(results, key=lambda x: x["score"], reverse=True)[:limit]

# ==================== 关键事实管理器 (SQLite) ====================

class KeyFactManager:
    """
    关键事实管理器 - SQLite 结构化存储

    存储类型:
    - user_profile: 用户画像信息
    - task: 任务记录
    - commitment: 承诺/待办
    - preference: 用户偏好
    - relationship: 关系图谱
    """

    def __init__(self, db_path: str = "data/key_facts.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None

    async def _ensure_conn(self):
        """确保数据库连接"""
        if self._conn is None:
            import aiosqlite
            self._conn = await aiosqlite.connect(str(self.db_path))
            await self._init_tables()

    async def _init_tables(self):
        """初始化表结构"""
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS key_facts (
                id TEXT PRIMARY KEY,
                fact_type TEXT NOT NULL,
                subject TEXT,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                verified INTEGER DEFAULT 0,
                created_at REAL,
                updated_at REAL,
                metadata TEXT
            )
        """)

        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 5,
                created_at REAL,
                updated_at REAL,
                deadline REAL,
                parent_task TEXT,
                assigned_to TEXT DEFAULT 'agent',
                result TEXT,
                metadata TEXT
            )
        """)

        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fact_type ON key_facts(fact_type)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_subject ON key_facts(subject)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_task_status ON tasks(status)"
        )

        await self._conn.commit()

    # ----- 关键事实操作 -----

    async def add_fact(
        self,
        predicate: str,
        object: str,
        fact_type: str = "fact",
        subject: str = "",
        confidence: float = 1.0,
        metadata: Dict = None
    ) -> str:
        """添加关键事实"""
        await self._ensure_conn()

        fact_id = hashlib.md5(f"{subject}:{predicate}:{object}".encode()).hexdigest()[:16]
        now = time.time()

        await self._conn.execute("""
            INSERT OR REPLACE INTO key_facts
            (id, fact_type, subject, predicate, object, confidence, verified, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fact_id, fact_type, subject, predicate, object,
            confidence, 0, now, now, json.dumps(metadata or {})
        ))
        await self._conn.commit()

        return fact_id

    async def get_facts(
        self,
        fact_type: Optional[str] = None,
        subject: Optional[str] = None,
        verified_only: bool = False
    ) -> List[Dict]:
        """查询事实"""
        await self._ensure_conn()

        query = "SELECT * FROM key_facts WHERE 1=1"
        params = []

        if fact_type:
            query += " AND fact_type = ?"
            params.append(fact_type)
        if subject:
            query += " AND subject = ?"
            params.append(subject)
        if verified_only:
            query += " AND verified = 1"
            params.append(1)

        query += " ORDER BY created_at DESC"

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()

        return [
            {
                "id": row[0],
                "fact_type": row[1],
                "subject": row[2],
                "predicate": row[3],
                "object": row[4],
                "confidence": row[5],
                "verified": bool(row[6]),
                "created_at": row[7],
                "updated_at": row[8],
                "metadata": json.loads(row[9] or "{}")
            }
            for row in rows
        ]

    async def verify_fact(self, fact_id: str):
        """验证事实"""
        await self._ensure_conn()
        await self._conn.execute(
            "UPDATE key_facts SET verified = 1, updated_at = ? WHERE id = ?",
            (time.time(), fact_id)
        )
        await self._conn.commit()

    async def delete_fact(self, fact_id: str) -> bool:
        """删除一条事实。"""
        await self._ensure_conn()
        cursor = await self._conn.execute(
            "DELETE FROM key_facts WHERE id = ?",
            (fact_id,)
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    # ----- 任务操作 -----

    async def add_task(
        self,
        title: str,
        description: str = "",
        priority: int = 5,
        deadline: Optional[float] = None,
        parent_task: Optional[str] = None,
        assigned_to: str = "agent",
        metadata: Dict = None
    ) -> str:
        """添加任务"""
        await self._ensure_conn()

        task_id = hashlib.md5(f"{title}:{time.time()}".encode()).hexdigest()[:16]
        now = time.time()

        await self._conn.execute("""
            INSERT INTO tasks
            (id, title, description, priority, deadline, parent_task, assigned_to, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task_id, title, description, priority, deadline,
            parent_task, assigned_to, now, now, json.dumps(metadata or {})
        ))
        await self._conn.commit()

        return task_id

    async def get_task(self, task_id: str) -> Optional[Dict]:
        """获取任务"""
        await self._ensure_conn()

        cursor = await self._conn.execute(
            "SELECT * FROM tasks WHERE id = ?",
            (task_id,)
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return {
            "id": row[0],
            "title": row[1],
            "description": row[2],
            "status": row[3],
            "priority": row[4],
            "created_at": row[5],
            "updated_at": row[6],
            "deadline": row[7],
            "parent_task": row[8],
            "assigned_to": row[9],
            "result": row[10],
            "metadata": json.loads(row[11] or "{}")
        }

    async def get_tasks(
        self,
        status: Optional[str] = None,
        assigned_to: Optional[str] = None
    ) -> List[Dict]:
        """获取任务列表"""
        await self._ensure_conn()

        query = "SELECT * FROM tasks WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if assigned_to:
            query += " AND assigned_to = ?"
            params.append(assigned_to)

        query += " ORDER BY priority DESC, created_at ASC"

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()

        return [
            {
                "id": row[0],
                "title": row[1],
                "description": row[2],
                "status": row[3],
                "priority": row[4],
                "created_at": row[5],
                "updated_at": row[6],
                "deadline": row[7],
                "parent_task": row[8],
                "assigned_to": row[9],
                "result": row[10],
                "metadata": json.loads(row[11] or "{}")
            }
            for row in rows
        ]

    async def update_task_status(
        self,
        task_id: str,
        status: str,
        result: Optional[str] = None
    ):
        """更新任务状态"""
        await self._ensure_conn()

        now = time.time()
        await self._conn.execute("""
            UPDATE tasks SET status = ?, result = ?, updated_at = ?
            WHERE id = ?
        """, (status, result, now, task_id))
        await self._conn.commit()

    async def close(self):
        """关闭连接"""
        if self._conn:
            await self._conn.close()
            self._conn = None

# ==================== 统一记忆管理器 ====================

class UnifiedMemoryManager:
    """
    统一记忆管理器 - 整合三层记忆系统

    使用示例:
        memory = UnifiedMemoryManager()
        await memory.initialize()

        # 添加对话
        await memory.add_conversation(session_id, "user", "你好")

        # 提取重要事实
        await memory.extract_facts("用户说他住在北京，是一名程序员")

        # 搜索相关记忆
        results = await memory.search("用户的工作相关")

        # 创建任务
        await memory.create_task("完成项目报告", priority=8)
    """

    def __init__(self):
        self.short_term = ShortTermMemoryManager()
        self.long_term = LongTermMemoryManager()
        self.key_facts = KeyFactManager()
        self._initialized = False

    async def initialize(self):
        """初始化所有记忆系统"""
        await self.long_term._ensure_initialized()
        await self.key_facts._ensure_conn()
        self._initialized = True

    # ----- 对话管理 -----

    async def add_conversation(
        self,
        session_id: str,
        role: str,
        content: str,
        auto_extract: bool = True
    ):
        """
        添加对话到短期记忆，可选择自动提取重要信息到长期记忆

        Args:
            session_id: 会话 ID
            role: user 或 assistant
            content: 对话内容
            auto_extract: 是否自动提取事实到长期记忆
        """
        # 添加到短期记忆
        await self.short_term.add_message(session_id, role, content)

        # 可选：自动提取重要信息
        if auto_extract and role == "user":
            await self._auto_extract(session_id, content)

    async def _auto_extract(self, session_id: str, content: str):
        """自动从对话中提取重要信息"""
        # TODO: 调用 LLM 分析内容，提取事实
        # 这里可以集成到 dialogue.py 中
        pass

    # ----- 长期记忆搜索 -----

    async def search(
        self,
        query: str,
        include_types: List[str] = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        搜索长期记忆和关键事实

        Returns:
            按相关性排序的记忆列表
        """
        results = []

        # 搜索长期记忆
        lt_results = await self.long_term.search(
            query,
            limit=limit
        )
        results.extend([{"source": "longterm", **r} for r in lt_results])

        # 搜索关键事实
        facts = await self.key_facts.get_facts()
        for fact in facts:
            if query.lower() in fact["predicate"].lower() or \
               query.lower() in fact["object"].lower():
                results.append({"source": "keyfact", **fact})

        # 按相关性排序（简化版本）
        return results[:limit]

    # ----- 任务管理 -----

    async def create_task(
        self,
        title: str,
        description: str = "",
        priority: int = 5,
        deadline: Optional[float] = None,
        assigned_to: str = "agent"
    ) -> str:
        """创建新任务"""
        return await self.key_facts.add_task(
            title, description, priority, deadline,
            assigned_to=assigned_to
        )

    async def get_pending_tasks(self, assigned_to: str = "agent") -> List[Dict]:
        """获取待处理任务"""
        return await self.key_facts.get_tasks(
            status=None,
            assigned_to=assigned_to
        )

    async def complete_task(self, task_id: str, result: str = ""):
        """完成任务"""
        await self.key_facts.update_task_status(task_id, "completed", result)

    # ----- 用户画像 -----

    async def get_user_profile(self, user_id: str) -> Dict:
        """获取用户画像"""
        facts = await self.key_facts.get_facts(
            fact_type="user_profile",
            subject=user_id
        )

        profile = {
            "user_id": user_id,
            "facts": {},
            "items": facts,
            "preferences": [],
            "commitments": []
        }

        for fact in facts:
            profile["facts"].setdefault(fact["predicate"], fact["object"])

        return profile

    async def remember_about_user(
        self,
        user_id: str,
        predicate: str,
        object: str,
        verified: bool = False
    ) -> str:
        """记住关于用户的事实"""
        fact_id = await self.key_facts.add_fact(
            predicate=predicate,
            object=object,
            fact_type="user_profile",
            subject=user_id
        )

        if verified:
            await self.key_facts.verify_fact(fact_id)

        return fact_id

    async def forget_about_user(self, user_id: str, query: str = "") -> int:
        """忘记当前用户的资料。query 为空或为“全部”时删除全部资料。"""
        normalized = (query or "").strip().lower()
        forget_all = normalized in {"", "全部", "所有", "all", "everything"}

        facts = await self.key_facts.get_facts(
            fact_type="user_profile",
            subject=user_id
        )

        deleted = 0
        for fact in facts:
            haystack = f"{fact['predicate']} {fact['object']}".lower()
            if forget_all or normalized in haystack:
                if await self.key_facts.delete_fact(fact["id"]):
                    deleted += 1

        return deleted

    async def close(self):
        """清理资源"""
        await self.key_facts.close()


# 全局单例
memory_manager = UnifiedMemoryManager()
