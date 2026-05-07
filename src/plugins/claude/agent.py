"""
智能体核心引擎

负责:
- 意图识别
- 任务规划
- 工具调度
- 监督日志
"""

import time
import json
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass, asdict
import aiofiles

from .memory_core import UnifiedMemoryManager
from .api import llm_client

# ==================== 数据结构 ====================

@dataclass
class AgentIntent:
    """识别的意图"""
    intent_type: Literal["chat", "task", "query", "command"]
    confidence: float
    payload: Dict[str, Any] = None

@dataclass
class ToolCall:
    """工具调用请求"""
    tool_name: str
    args: Dict[str, Any]
    call_id: str = ""

@dataclass
class SupervisionLog:
    """监督日志"""
    timestamp: float
    level: Literal["info", "warning", "error", "decision", "action"]
    module: str
    message: str
    details: Dict[str, Any] = None

    def to_dict(self):
        return asdict(self)

# ==================== 意图识别器 ====================

class IntentRecognizer:
    """
    意图识别器 - 分析用户消息，分类到不同处理器

    意图类型:
    - chat: 闲聊/对话
    - task: 任务请求 (需要执行动作)
    - query: 查询请求 (获取信息)
    - command: 系统命令
    """

    def __init__(self):
        # 意图关键词模式
        self.patterns = {
            "task": [
                "帮我", "请帮我", "去做", "执行", "完成", "创建", "写一个",
                "分析", "处理", "运行", "启动", "停止", "删除", "修改",
                "task", "do this", "make a", "create a"
            ],
            "query": [
                "查询", "搜索", "查找", "看看", "检查一下", "获取",
                "是什么", "有没有", "多少", "where", "what", "how", "search"
            ],
            "command": [
                "/clear", "/model", "/help", "/status", "/tasks",
                "清空", "切换", "帮助", "状态", "任务列表"
            ]
        }

    async def recognize(self, message: str, context: Dict = None) -> AgentIntent:
        """
        识别消息意图

        Args:
            message: 用户消息
            context: 上下文信息 (用户 ID, 会话历史等)

        Returns:
            AgentIntent 对象
        """
        message_lower = message.lower().strip()

        # 检查命令
        if message_lower.startswith("/") or message_lower in self.patterns["command"]:
            return AgentIntent(
                intent_type="command",
                confidence=0.95,
                payload={"command": message_lower.split()[0]}
            )

        # 使用 LLM 进行更准确的意图识别
        intent_prompt = f"""分析以下用户消息的意图，返回 JSON 格式：

消息："{message}"

请判断意图类型 (chat/task/query/command) 并提取关键信息。
如果是 task，说明需要什么工具。
如果是 query，说明要查询什么。

返回格式：
{{
    "intent": "chat|task|query|command",
    "confidence": 0.0-1.0,
    "details": {{}}
}}
"""

        try:
            response = await llm_client.chat(
                messages=[{"role": "user", "content": intent_prompt}],
                temperature=0.1
            )
            # 解析 JSON 响应
            result = json.loads(response.strip())
            return AgentIntent(
                intent_type=result.get("intent", "chat"),
                confidence=result.get("confidence", 0.5),
                payload=result.get("details", {})
            )
        except Exception as e:
            # 降级：基于关键词匹配
            return self._keyword_match(message_lower)

    def _keyword_match(self, message: str) -> AgentIntent:
        """基于关键词的简单匹配"""
        scores = {"chat": 0.5, "task": 0, "query": 0, "command": 0}

        for intent_type, patterns in self.patterns.items():
            for pattern in patterns:
                if pattern in message:
                    scores[intent_type] += 0.2

        best_intent = max(scores, key=scores.get)
        return AgentIntent(
            intent_type=best_intent,
            confidence=min(scores[best_intent], 0.9),
            payload={}
        )

# ==================== 工具集 ====================

class ToolRegistry:
    """
    工具注册表 - 管理所有可用工具

    工具类型:
    - file: 文件系统操作
    - web: 网络相关
    - code: 代码执行
    - schedule: 定时任务
    """

    def __init__(self, workspace_root: Optional[str] = None):
        self.tools: Dict[str, Dict] = {}
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self._register_builtins()

    def _register_builtins(self):
        """注册内置工具"""

        # ----- 文件系统工具 -----
        self.register_tool(
            name="read_file",
            description="读取文件内容",
            parameters={
                "path": {"type": "string", "required": True, "description": "文件路径"}
            },
            func=self._read_file
        )

        self.register_tool(
            name="write_file",
            description="写入文件内容",
            parameters={
                "path": {"type": "string", "required": True, "description": "文件路径"},
                "content": {"type": "string", "required": True, "description": "文件内容"}
            },
            func=self._write_file
        )

        self.register_tool(
            name="list_files",
            description="列出目录内容",
            parameters={
                "path": {"type": "string", "required": True, "description": "目录路径"},
                "pattern": {"type": "string", "required": False, "description": "通配符模式"}
            },
            func=self._list_files
        )

        self.register_tool(
            name="search_files",
            description="搜索文件内容",
            parameters={
                "path": {"type": "string", "required": True, "description": "搜索路径"},
                "pattern": {"type": "string", "required": True, "description": "搜索模式"}
            },
            func=self._search_files
        )

        # ----- 网络工具 -----
        self.register_tool(
            name="web_search",
            description="网络搜索",
            parameters={
                "query": {"type": "string", "required": True, "description": "搜索关键词"}
            },
            func=self._web_search
        )

        self.register_tool(
            name="web_fetch",
            description="抓取网页内容",
            parameters={
                "url": {"type": "string", "required": True, "description": "网页 URL"}
            },
            func=self._web_fetch
        )

        # ----- 代码执行工具 -----
        self.register_tool(
            name="run_python",
            description="执行 Python 代码",
            parameters={
                "code": {"type": "string", "required": True, "description": "Python 代码"},
                "timeout": {"type": "integer", "required": False, "description": "超时时间 (秒)"}
            },
            func=self._run_python
        )

        self.register_tool(
            name="run_shell",
            description="执行 Shell 命令",
            parameters={
                "command": {"type": "string", "required": True, "description": "Shell 命令"},
                "timeout": {"type": "integer", "required": False, "description": "超时时间 (秒)"}
            },
            func=self._run_shell
        )

    def register_tool(self, name: str, description: str, parameters: Dict, func: callable):
        """注册一个工具"""
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "func": func
        }

    def get_tool(self, name: str) -> Optional[Dict]:
        """获取工具定义"""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict]:
        """列出所有工具"""
        return [
            {"name": t["name"], "description": t["description"]}
            for t in self.tools.values()
        ]

    async def execute(self, tool_name: str, args: Dict) -> Any:
        """执行工具"""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"未知工具：{tool_name}")

        # 验证必填参数
        for param_name, param_def in tool["parameters"].items():
            if param_def.get("required") and param_name not in args:
                raise ValueError(f"工具 {tool_name} 缺少必填参数：{param_name}")

        # 执行工具
        return await tool["func"](**args)

    # ----- 工具实现 -----

    def _resolve_workspace_path(self, path: str) -> Path:
        """Resolve a tool path and keep it inside the bot workspace."""
        raw_path = Path(path)
        if not raw_path.is_absolute():
            raw_path = self.workspace_root / raw_path

        resolved = raw_path.resolve()
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError:
            raise ValueError(f"路径超出工作目录：{path}")

        return resolved

    async def _read_file(self, path: str) -> str:
        """读取文件"""
        file_path = self._resolve_workspace_path(path)

        if not file_path.exists():
            return f"错误：文件不存在：{path}"

        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            return await f.read()

    async def _write_file(self, path: str, content: str) -> str:
        """写入文件"""
        file_path = self._resolve_workspace_path(path)

        file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(content)

        return f"成功写入文件：{path}"

    async def _list_files(self, path: str, pattern: str = "*") -> List[str]:
        """列出目录"""
        import glob
        dir_path = self._resolve_workspace_path(path)

        if not dir_path.exists():
            return [f"错误：目录不存在：{path}"]

        files = glob.glob(str(dir_path / pattern))
        return [str(f) for f in files]

    async def _search_files(self, path: str, pattern: str) -> List[Dict]:
        """搜索文件内容"""
        import glob
        results = []

        search_path = self._resolve_workspace_path(path)

        for file_path in glob.glob(str(search_path), recursive=True):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if pattern in content:
                    results.append({"file": file_path, "match": True})
            except:
                pass

        return results

    async def _web_search(self, query: str) -> str:
        """网络搜索 - 需要 WebFetch 工具支持"""
        return f"[网络搜索]: {query} - 需要集成 WebSearch 工具"

    async def _web_fetch(self, url: str) -> str:
        """抓取网页"""
        return f"[抓取网页]: {url} - 需要集成 WebFetch 工具"

    async def _run_python(self, code: str, timeout: int = 30) -> str:
        """执行 Python 代码"""
        if os.getenv("AGENT_ALLOW_CODE_EXEC") != "1":
            return "错误：代码执行工具默认禁用。请在本机确认安全后设置 AGENT_ALLOW_CODE_EXEC=1。"

        import subprocess
        import tempfile

        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            output = result.stdout + result.stderr
            return output
        except subprocess.TimeoutExpired:
            return f"错误：代码执行超时 ({timeout}秒)"
        except Exception as e:
            return f"错误：{str(e)}"
        finally:
            os.unlink(temp_file)

    async def _run_shell(self, command: str, timeout: int = 30) -> str:
        """执行 Shell 命令"""
        if os.getenv("AGENT_ALLOW_SHELL") != "1":
            return "错误：Shell 工具默认禁用。请在本机确认安全后设置 AGENT_ALLOW_SHELL=1。"

        import subprocess

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return f"错误：命令执行超时 ({timeout}秒)"
        except Exception as e:
            return f"错误：{str(e)}"

# ==================== 任务规划器 ====================

class TaskPlanner:
    """
    任务规划器 - 分解复杂任务，管理执行顺序

    功能:
    - 任务分解
    - 依赖管理
    - 优先级排序
    - 进度跟踪
    """

    def __init__(self, memory: UnifiedMemoryManager):
        self.memory = memory

    async def plan_task(self, description: str, context: Dict = None) -> List[Dict]:
        """
        将复杂任务分解为子任务

        Returns:
            子任务列表，按执行顺序排列
        """
        prompt = f"""将以下任务分解为可执行的步骤：

任务：{description}

对于每个步骤，说明：
1. 需要做什么
2. 需要什么工具/信息
3. 是否依赖其他步骤

返回 JSON 格式：
[
    {{"step": 1, "action": "...", "tool": "...", "depends_on": []}},
    ...
]
"""
        try:
            response = await llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            steps = json.loads(response.strip())
            return steps
        except:
            # 降级：返回单一步骤
            return [{"step": 1, "action": description, "tool": "auto", "depends_on": []}]

    async def get_next_action(self, task_id: str) -> Optional[Dict]:
        """获取下一个可执行的动作"""
        task = await self.memory.key_facts.get_task(task_id)
        if not task:
            return None

        # 检查依赖
        metadata = task.get("metadata", {})
        subtasks = metadata.get("subtasks", [])

        for subtask in subtasks:
            if subtask.get("status") != "completed":
                # 检查依赖是否满足
                deps = subtask.get("depends_on", [])
                all_deps_done = all(
                    any(st.get("id") == d and st.get("status") == "completed"
                        for st in subtasks)
                    for d in deps
                )
                if all_deps_done:
                    return subtask

        return None

# ==================== 监督日志系统 ====================

class SupervisionLogger:
    """
    监督日志系统 - 记录智能体决策过程

    日志级别:
    - info: 一般信息
    - decision: 重要决策
    - action: 执行动作
    - warning: 警告
    - error: 错误
    """

    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: List[SupervisionLog] = []
        self._flush_interval = 10  # 每 10 条日志刷新一次

    def log(
        self,
        level: str,
        module: str,
        message: str,
        details: Dict = None
    ):
        """记录日志"""
        log_entry = SupervisionLog(
            timestamp=time.time(),
            level=level,
            module=module,
            message=message,
            details=details or {}
        )
        self._buffer.append(log_entry)

        # 控制台输出
        self._console_output(log_entry)

        # 定期刷新到文件
        if len(self._buffer) >= self._flush_interval:
            self._flush()

    def _console_output(self, log: SupervisionLog):
        """控制台输出"""
        from datetime import datetime
        ts = datetime.fromtimestamp(log.timestamp).strftime("%H:%M:%S")
        level_icon = {
            "info": "ℹ️",
            "decision": "🤔",
            "action": "🔧",
            "warning": "⚠️",
            "error": "❌"
        }.get(log.level, "•")

        print(f"[{ts}] {level_icon} [{log.module}] {log.message}")

    def _flush(self):
        """刷新到文件"""
        if not self._buffer:
            return

        today = time.strftime("%Y-%m-%d")
        log_file = self.log_dir / f"supervision_{today}.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            for log in self._buffer:
                f.write(json.dumps(log.to_dict(), ensure_ascii=False) + "\n")

        self._buffer.clear()

    def get_recent_logs(
        self,
        level: Optional[str] = None,
        module: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """获取最近的日志"""
        # 从缓冲区获取
        logs = self._buffer.copy()

        if level:
            logs = [l for l in logs if l.level == level]
        if module:
            logs = [l for l in logs if l.module == module]

        return [l.to_dict() for l in logs[-limit:]]

    def flush(self):
        """强制刷新"""
        self._flush()

# ==================== 智能体引擎 ====================

class AgentEngine:
    """
    智能体引擎 - 整合所有组件

    使用示例:
        engine = AgentEngine()
        await engine.initialize()

        response = await engine.process_message(
            user_id="123456",
            session_id="private_123456",
            message="帮我分析一下这个项目的结构"
        )
    """

    def __init__(self):
        self.memory = UnifiedMemoryManager()
        self.intent_recognizer = IntentRecognizer()
        self.tool_registry = ToolRegistry()
        self.planner = None  # 延迟初始化
        self.logger = SupervisionLogger()
        self._initialized = False

        # 系统提示
        self.system_prompt = """你是一个智能助手，具备以下能力：
- 长期记忆：可以记住用户偏好和历史对话
- 工具使用：可以操作文件、执行代码、网络搜索
- 任务规划：可以分解复杂任务并逐步执行
- 透明决策：所有重要决策都会记录日志供用户审查

当遇到不确定的事情时，先向用户确认再执行。"""

    async def initialize(self):
        """初始化引擎"""
        await self.memory.initialize()
        self.planner = TaskPlanner(self.memory)
        self._initialized = True
        self.logger.log("info", "AgentEngine", "智能体引擎初始化完成")

    async def process_message(
        self,
        user_id: str,
        session_id: str,
        message: str
    ) -> str:
        """
        处理用户消息

        流程:
        1. 意图识别
        2. 检索相关记忆
        3. 根据意图类型处理
        4. 记录决策日志
        5. 返回响应
        """
        if not self._initialized:
            await self.initialize()

        # 1. 意图识别
        intent = await self.intent_recognizer.recognize(message)
        self.logger.log(
            "decision", "IntentRecognizer",
            f"识别意图：{intent.intent_type} (置信度：{intent.confidence:.2f})",
            {"message": message, "payload": intent.payload}
        )

        # 2. 检索相关记忆
        relevant_memories = await self.memory.search(message, limit=3)
        if relevant_memories:
            self.logger.log(
                "info", "Memory",
                f"检索到 {len(relevant_memories)} 条相关记忆",
                {"memories": relevant_memories}
            )

        # 3. 根据意图处理
        if intent.intent_type == "command":
            response = await self._handle_command(
                intent.payload.get("command", ""),
                user_id, session_id, message
            )
        elif intent.intent_type == "task":
            response = await self._handle_task(
                user_id, session_id, message, relevant_memories
            )
        elif intent.intent_type == "query":
            response = await self._handle_query(
                user_id, session_id, message, relevant_memories
            )
        else:  # chat
            response = await self._handle_chat(
                user_id, session_id, message, relevant_memories
            )

        # 4. 添加到对话历史
        await self.memory.add_conversation(session_id, "user", message)
        await self.memory.add_conversation(session_id, "assistant", response)

        return response

    async def _handle_chat(
        self,
        user_id: str,
        session_id: str,
        message: str,
        memories: List[Dict]
    ) -> str:
        """处理闲聊"""
        # 构建上下文
        context = ""
        if memories:
            context = "\n相关记忆:\n" + "\n".join(
                f"- {m.get('content', m.get('object', ''))}"
                for m in memories
            )

        prompt = f"""{self.system_prompt}

当前对话：{message}
{context}

请自然友好地回复。"""

        response = await llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        self.logger.log("info", "ChatHandler", f"闲聊回复：{response[:50]}...")
        return response

    async def _handle_task(
        self,
        user_id: str,
        session_id: str,
        message: str,
        memories: List[Dict]
    ) -> str:
        """处理任务请求"""
        # 创建任务记录
        task_id = await self.memory.create_task(
            title=message[:50],
            description=message,
            priority=5,
            assigned_to="agent"
        )

        self.logger.log(
            "decision", "TaskHandler",
            f"创建任务：{task_id}",
            {"title": message[:50]}
        )

        # 任务分解
        steps = await self.planner.plan_task(message)
        self.logger.log(
            "action", "TaskPlanner",
            f"分解为 {len(steps)} 个步骤",
            {"steps": steps}
        )

        # 执行第一步
        if steps:
            first_step = steps[0]
            tool_name = first_step.get("tool", "auto")

            if tool_name == "auto":
                # 需要进一步分析
                return f"我收到任务：{message}\n\n我计划分 {len(steps)} 步完成：\n" + \
                       "\n".join(f"{i+1}. {s.get('action', s)}" for i, s in enumerate(steps)) + \
                       "\n\n请确认我开始执行。"

            # 执行工具
            try:
                result = await self.tool_registry.execute(
                    tool_name,
                    {"query": message}  # 简化处理
                )
                await self.memory.complete_task(task_id, result)
                return f"任务完成：{result}"
            except Exception as e:
                self.logger.log("error", "TaskHandler", f"执行失败：{e}")
                return f"任务执行失败：{e}"

        return "任务已记录，我会找时间完成。"

    async def _handle_query(
        self,
        user_id: str,
        session_id: str,
        message: str,
        memories: List[Dict]
    ) -> str:
        """处理查询请求"""
        if memories:
            # 从记忆中回答
            best = memories[0]
            return f"根据我的记忆：{best.get('content', best.get('object', '未知'))}"

        # 需要外部查询
        return f"我需要查询一下：{message}"

    async def _handle_command(
        self,
        command: str,
        user_id: str,
        session_id: str,
        message: str
    ) -> str:
        """处理系统命令"""
        if command == "/clear":
            await self.memory.short_term.clear(session_id)
            return "会话历史已清空"

        elif command == "/tasks":
            tasks = await self.memory.get_pending_tasks()
            if not tasks:
                return "没有待处理任务"
            return "\n".join(f"- [{t['priority']}] {t['title']}" for t in tasks)

        elif command == "/status":
            return "智能体运行正常"

        elif command == "/help":
            return """可用命令:
/clear - 清空会话
/tasks - 查看任务列表
/status - 查看状态
/help - 显示帮助"""

        return f"未知命令：{command}"

    async def close(self):
        """清理资源"""
        self.logger.flush()
        await self.memory.close()


# 全局单例
agent_engine = AgentEngine()
