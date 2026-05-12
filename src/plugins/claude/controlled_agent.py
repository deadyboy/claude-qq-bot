"""Controlled agent planning, tool execution, and review drafts.

This is the replacement direction for the archived AGENT_MODE path. It is
command-driven, permission-aware, and confirmation-friendly; it does not run an
autonomous loop or execute hidden tools.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List

from .safe_tools import (
    TodoStore,
    format_current_time,
    format_profile_search_results,
    format_todo_list,
    safe_calculate,
    search_profile,
)


AGENT_DRAFTS_PATH = Path("data/agent_drafts.json")
MAX_DRAFTS = 80
APPROVED_DRAFT_STATUSES = {"accepted", "approved"}
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


@dataclass(frozen=True)
class ControlledToolSpec:
    name: str
    title: str
    description: str
    usage: str
    permission: str = "owner"
    risk: str = "low"
    requires_confirmation: bool = False
    aliases: tuple[str, ...] = ()


@dataclass
class ControlledAgentContext:
    actor_id: str
    session_id: str
    chat_type: str
    is_owner: bool = False


@dataclass
class ToolExecutionResult:
    ok: bool
    tool_name: str
    status: str
    output: str
    requires_confirmation: bool = False


CONTROLLED_TOOLS: dict[str, ControlledToolSpec] = {
    "time": ControlledToolSpec(
        name="time",
        title="时间",
        description="读取本机当前时间。",
        usage="/agent 执行 time",
        permission="owner",
        risk="low",
        aliases=("now", "时间", "当前时间"),
    ),
    "calc": ControlledToolSpec(
        name="calc",
        title="计算",
        description="执行安全数学表达式，只允许数字和四则运算。",
        usage="/agent 执行 calc 1 + 2 * 3",
        permission="owner",
        risk="low",
        aliases=("calculate", "计算"),
    ),
    "todo_list": ControlledToolSpec(
        name="todo_list",
        title="待办列表",
        description="查看当前 QQ 用户自己的待办。",
        usage="/agent 执行 todo_list",
        permission="owner",
        risk="low",
        aliases=("todos", "待办", "待办列表"),
    ),
    "todo_add": ControlledToolSpec(
        name="todo_add",
        title="添加待办",
        description="给当前 QQ 用户添加一条待办。",
        usage="/agent 执行 todo_add 买牛奶",
        permission="owner",
        risk="medium",
        aliases=("add_todo", "添加待办"),
    ),
    "todo_done": ControlledToolSpec(
        name="todo_done",
        title="完成待办",
        description="按编号或 ID 完成当前 QQ 用户的一条待办。",
        usage="/agent 执行 todo_done 1",
        permission="owner",
        risk="medium",
        aliases=("finish_todo", "完成待办"),
    ),
    "memory_query": ControlledToolSpec(
        name="memory_query",
        title="记忆查询",
        description="搜索当前 QQ 用户画像。",
        usage="/agent 执行 memory_query 关键词",
        permission="owner",
        risk="low",
        aliases=("profile_search", "记忆查询", "资料查询"),
    ),
    "status": ControlledToolSpec(
        name="status",
        title="状态摘要",
        description="返回受控 Agent 的本地状态摘要。",
        usage="/agent 执行 status",
        permission="owner",
        risk="low",
        aliases=("状态", "agent_status"),
    ),
    "clear_session": ControlledToolSpec(
        name="clear_session",
        title="清空会话",
        description="清空当前私聊或群聊会话历史。",
        usage="/agent 执行 clear_session",
        permission="owner",
        risk="high",
        requires_confirmation=True,
        aliases=("clear", "清空会话", "清空历史"),
    ),
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _normalize_tool_name(name: str) -> str:
    key = str(name or "").strip()
    lowered = key.lower()
    for tool_name, spec in CONTROLLED_TOOLS.items():
        if lowered == tool_name.lower() or key in spec.aliases or lowered in {a.lower() for a in spec.aliases}:
            return tool_name
    return lowered


def get_controlled_tool(name: str) -> ControlledToolSpec | None:
    return CONTROLLED_TOOLS.get(_normalize_tool_name(name))


def list_controlled_tools() -> list[ControlledToolSpec]:
    return list(CONTROLLED_TOOLS.values())


def format_controlled_agent_help() -> str:
    return "\n".join([
        "受控 Agent：",
        "- /agent 工具：查看受控工具和风险等级",
        "- /agent 计划 <任务>：生成可审核计划，不执行",
        "- /agent 草稿 <任务>：同 /agent 计划，保存审核草稿",
        "- /agent 最近：查看最近草稿",
        "- /agent 执行 <工具> <参数>：执行单个受控工具",
        "- /agent 执行计划 <id>：执行已审核计划；高风险步骤会要求 /确认",
        "- /agent 采纳 <id> / /agent 拒绝 <id>：标记草稿审核结果",
        "约束：只在主人私聊中使用；不恢复旧 AGENT_MODE；高风险动作走确认和审计。",
    ])


def format_tool_catalog() -> str:
    lines = ["受控工具目录："]
    for spec in list_controlled_tools():
        confirm = "需确认" if spec.requires_confirmation else "直接执行"
        lines.append(
            f"- {spec.name}：{spec.title}；权限={spec.permission}；风险={spec.risk}；{confirm}；{spec.usage}"
        )
    return "\n".join(lines)


def parse_agent_command(text: str) -> tuple[str, str]:
    stripped = str(text or "").strip()
    lowered = stripped.lower()
    prefixes = ("/agent", "/智能体", "/受控", "智能体", "受控")
    for prefix in prefixes:
        if lowered == prefix.lower():
            return "help", ""
        if lowered.startswith(prefix.lower() + " "):
            stripped = stripped[len(prefix):].strip()
            break
        if stripped.startswith(prefix + "：") or stripped.startswith(prefix + ":"):
            stripped = stripped[len(prefix) + 1:].strip()
            break
    else:
        return "", ""

    if not stripped:
        return "help", ""

    lowered = stripped.lower()
    command_map = {
        "工具": "tools",
        "tools": "tools",
        "状态": "status",
        "status": "status",
        "最近": "recent",
        "recent": "recent",
    }
    if lowered in command_map:
        return command_map[lowered], ""

    for prefix, action in (
        ("计划", "plan"),
        ("plan", "plan"),
        ("草稿", "draft"),
        ("draft", "draft"),
        ("执行计划", "execute_plan"),
        ("run_plan", "execute_plan"),
        ("执行", "execute_tool"),
        ("run", "execute_tool"),
        ("采纳", "accept"),
        ("accept", "accept"),
        ("拒绝", "reject"),
        ("reject", "reject"),
    ):
        if lowered == prefix.lower():
            return action, ""
        if lowered.startswith(prefix.lower() + " "):
            return action, stripped[len(prefix):].strip()
        if stripped.startswith(prefix + "：") or stripped.startswith(prefix + ":"):
            return action, stripped[len(prefix) + 1:].strip()

    return "plan", stripped


def split_tool_payload(payload: str) -> tuple[str, str]:
    text = str(payload or "").strip()
    if not text:
        return "", ""
    parts = text.split(maxsplit=1)
    tool_name = _normalize_tool_name(parts[0])
    tool_payload = parts[1].strip() if len(parts) > 1 else ""
    return tool_name, tool_payload


def _new_step(tool_name: str, payload: str = "", reason: str = "") -> Dict[str, Any]:
    spec = get_controlled_tool(tool_name)
    if not spec:
        return {
            "tool_name": tool_name,
            "payload": payload[:200],
            "reason": reason[:160],
            "permission": "unknown",
            "risk": "unknown",
            "requires_confirmation": False,
            "status": "unknown_tool",
        }
    return {
        "tool_name": spec.name,
        "payload": payload.strip()[:200],
        "reason": reason[:160],
        "permission": spec.permission,
        "risk": spec.risk,
        "requires_confirmation": spec.requires_confirmation,
        "status": "planned",
    }


def _extract_calc_expression(request: str) -> str:
    text = request.strip()
    for prefix in ("计算", "calc", "calculate"):
        index = text.lower().find(prefix.lower())
        if index >= 0:
            return text[index + len(prefix):].strip(" ：:")
    match = re.search(r"[-+*/().\d\s×÷]{3,}", text)
    return match.group(0).strip() if match else ""


def build_controlled_agent_plan(request: str, context: ControlledAgentContext) -> Dict[str, Any]:
    """Build a deterministic, reviewable plan without executing tools."""
    text = str(request or "").strip()
    lowered = text.lower()
    steps: list[Dict[str, Any]] = []

    if any(token in text for token in ("清空会话", "清空历史", "清除记忆", "/clear")):
        steps.append(_new_step("clear_session", "", "用户要求清空当前会话历史。"))
    elif "待办" in text or "todo" in lowered:
        if any(token in text for token in ("添加", "新增", "加一个")) or "add" in lowered:
            payload = re.sub(r"^(待办|todo)?\s*(添加|新增|add|加一个)?", "", text, flags=re.I).strip(" ：:")
            steps.append(_new_step("todo_add", payload, "用户要求添加待办。"))
        elif any(token in text for token in ("完成", "done", "finish")):
            payload = re.sub(r"^(待办|todo)?\s*(完成|done|finish)?", "", text, flags=re.I).strip(" ：:")
            steps.append(_new_step("todo_done", payload, "用户要求完成待办。"))
        else:
            steps.append(_new_step("todo_list", "", "用户询问待办列表。"))
    elif any(token in text for token in ("记忆查询", "查记忆", "资料查询", "我的资料")) or "memory search" in lowered:
        payload = re.sub(r"^(记忆查询|查记忆|资料查询|我的资料|memory search)", "", text, flags=re.I).strip(" ：:")
        steps.append(_new_step("memory_query", payload, "用户要求查询当前用户资料。"))
    elif any(token in text for token in ("几点", "时间", "日期", "现在")) or lowered in {"time", "now"}:
        steps.append(_new_step("time", "", "用户询问当前时间或日期。"))
    else:
        expression = _extract_calc_expression(text)
        if expression:
            steps.append(_new_step("calc", expression, "用户要求安全计算。"))

    return {
        "id": "",
        "request": text[:500],
        "actor_id": context.actor_id,
        "session_id": context.session_id,
        "chat_type": context.chat_type,
        "created_at": _now_iso(),
        "status": "planned" if steps else "needs_manual_review",
        "steps": steps,
        "review_policy": "draft_first_no_autonomous_execution",
        "notes": (
            "已匹配受控工具；执行前可审核。"
            if steps
            else "没有匹配到受控工具。建议普通聊天处理，或明确指定 /agent 执行 <工具>。"
        ),
    }


def format_agent_plan(plan: Dict[str, Any]) -> str:
    lines = [
        f"受控 Agent 草稿 #{plan.get('id') or '未保存'}",
        f"- 状态：{plan.get('status')}",
        f"- 请求：{plan.get('request') or ''}",
        f"- 策略：{plan.get('review_policy')}",
    ]
    steps = plan.get("steps") or []
    if not steps:
        lines.append(f"- 说明：{plan.get('notes')}")
        return "\n".join(lines)

    lines.append("计划步骤：")
    for index, step in enumerate(steps, start=1):
        confirm = "需确认" if step.get("requires_confirmation") else "可直接执行"
        payload = step.get("payload") or "无"
        lines.append(
            f"{index}. {step.get('tool_name')}({payload})；风险={step.get('risk')}；{confirm}；{step.get('reason')}"
        )
    lines.append(f"标记：/agent 采纳 {plan.get('id')}；/agent 拒绝 {plan.get('id')}")
    lines.append(f"执行：采纳后 /agent 执行计划 {plan.get('id')}")
    return "\n".join(lines)


def format_execution_result(result: ToolExecutionResult) -> str:
    prefix = "执行完成" if result.ok else "执行未完成"
    return "\n".join([
        f"{prefix}：{result.tool_name}",
        f"- 状态：{result.status}",
        f"- 结果：{result.output}",
    ])


def format_plan_execution_results(draft_id: str, results: list[ToolExecutionResult]) -> str:
    lines = [f"计划执行结果 #{draft_id}："]
    for index, result in enumerate(results, start=1):
        marker = "OK" if result.ok else "FAIL"
        lines.append(f"{index}. [{marker}] {result.tool_name}：{result.output}")
    return "\n".join(lines)


class ControlledAgentDraftStore:
    """Small JSON store for owner-reviewed controlled-agent drafts."""

    def __init__(self, path: Path | str = AGENT_DRAFTS_PATH):
        self.path = Path(path)

    def _load(self) -> list[Dict[str, Any]]:
        with _path_lock(self.path):
            if not self.path.exists():
                return []
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                return []
            return data if isinstance(data, list) else []

    def _save(self, drafts: list[Dict[str, Any]]) -> None:
        with _path_lock(self.path):
            _atomic_write_json(self.path, drafts[-MAX_DRAFTS:])

    def create(self, actor_id: str | int, plan: Dict[str, Any]) -> Dict[str, Any]:
        with _path_lock(self.path):
            drafts = self._load()
            draft = deepcopy(plan)
            draft["id"] = uuid.uuid4().hex[:8]
            draft["actor_id"] = str(actor_id)
            draft["created_at_ts"] = time.time()
            draft["updated_at"] = _now_iso()
            drafts.append(draft)
            self._save(drafts)
        return deepcopy(draft)

    def get(self, draft_id: str, actor_id: str | int | None = None) -> Dict[str, Any] | None:
        key = str(draft_id or "").strip()
        actor = str(actor_id) if actor_id is not None else None
        for draft in self._load():
            if str(draft.get("id")) != key:
                continue
            if actor is not None and str(draft.get("actor_id")) != actor:
                return None
            return deepcopy(draft)
        return None

    def list_recent(self, actor_id: str | int, limit: int = 5) -> list[Dict[str, Any]]:
        actor = str(actor_id)
        drafts = [draft for draft in self._load() if str(draft.get("actor_id")) == actor]
        drafts.sort(key=lambda item: float(item.get("created_at_ts") or 0), reverse=True)
        return [deepcopy(draft) for draft in drafts[:limit]]

    def update_status(self, draft_id: str, actor_id: str | int, status: str, result: str = "") -> Dict[str, Any] | None:
        with _path_lock(self.path):
            drafts = self._load()
            actor = str(actor_id)
            updated = None
            for draft in drafts:
                if str(draft.get("id")) == str(draft_id) and str(draft.get("actor_id")) == actor:
                    draft["status"] = status
                    draft["review_result"] = result[:400]
                    draft["updated_at"] = _now_iso()
                    updated = deepcopy(draft)
                    break
            if updated:
                self._save(drafts)
            return updated

    def clear_for_tests(self) -> None:
        if self.path.exists():
            self.path.unlink()


agent_draft_store = ControlledAgentDraftStore()


def format_recent_agent_drafts(drafts: list[Dict[str, Any]]) -> str:
    if not drafts:
        return "最近没有受控 Agent 草稿。"
    lines = ["最近受控 Agent 草稿："]
    for draft in drafts:
        steps = draft.get("steps") or []
        tool_names = "、".join(str(step.get("tool_name")) for step in steps) if steps else "无工具"
        lines.append(f"- {draft.get('id')}：{draft.get('status')}；{tool_names}；{draft.get('request')}")
    lines.append("用法：/agent 执行计划 <id>；/agent 采纳 <id>；/agent 拒绝 <id>")
    return "\n".join(lines)


async def execute_controlled_tool(
    tool_name: str,
    payload: str,
    context: ControlledAgentContext,
    *,
    todo_store: TodoStore | None = None,
    user_profile: Dict[str, Any] | None = None,
    status_lines: list[str] | None = None,
    confirmed: bool = False,
    session_clearer: Callable[[str], Awaitable[None]] | None = None,
) -> ToolExecutionResult:
    spec = get_controlled_tool(tool_name)
    if not spec:
        return ToolExecutionResult(False, tool_name, "unknown_tool", "没有这个受控工具。")
    if spec.permission == "owner" and not context.is_owner:
        return ToolExecutionResult(False, spec.name, "permission_denied", "该工具仅主人可用。")
    if spec.requires_confirmation and not confirmed:
        return ToolExecutionResult(False, spec.name, "needs_confirmation", "该工具需要确认。", True)

    todo = todo_store or TodoStore()
    data = str(payload or "").strip()

    if spec.name == "time":
        return ToolExecutionResult(True, spec.name, "executed", format_current_time())
    if spec.name == "calc":
        return ToolExecutionResult(True, spec.name, "executed", safe_calculate(data))
    if spec.name == "todo_list":
        return ToolExecutionResult(True, spec.name, "executed", format_todo_list(todo.list(context.actor_id)))
    if spec.name == "todo_add":
        try:
            item = todo.add(context.actor_id, data)
        except Exception as e:
            return ToolExecutionResult(False, spec.name, "failed", f"添加待办失败：{e}")
        return ToolExecutionResult(True, spec.name, "executed", f"已添加待办：{item['content']} ({item['id']})")
    if spec.name == "todo_done":
        try:
            item = todo.complete(context.actor_id, data)
        except Exception as e:
            return ToolExecutionResult(False, spec.name, "failed", f"完成待办失败：{e}")
        if not item:
            return ToolExecutionResult(False, spec.name, "not_found", "没有找到对应的待办。")
        return ToolExecutionResult(True, spec.name, "executed", f"已完成：{item['content']}")
    if spec.name == "memory_query":
        profile = user_profile or {"items": []}
        return ToolExecutionResult(
            True,
            spec.name,
            "executed",
            format_profile_search_results(data, search_profile(profile, data)),
        )
    if spec.name == "status":
        lines = status_lines or [
            "受控 Agent 状态：",
            "- 旧 AGENT_MODE：已归档",
            "- 当前模式：命令式计划/审核/确认",
        ]
        return ToolExecutionResult(True, spec.name, "executed", "\n".join(lines))
    if spec.name == "clear_session":
        if not session_clearer:
            return ToolExecutionResult(False, spec.name, "failed", "缺少会话清理器。")
        await session_clearer(context.session_id)
        return ToolExecutionResult(True, spec.name, "executed", f"已清空会话：{context.session_id}")

    return ToolExecutionResult(False, spec.name, "not_implemented", "工具尚未实现。")


async def execute_agent_plan(
    plan: Dict[str, Any],
    context: ControlledAgentContext,
    *,
    todo_store: TodoStore | None = None,
    user_profile: Dict[str, Any] | None = None,
    status_lines: list[str] | None = None,
    confirmed: bool = False,
    session_clearer: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[list[ToolExecutionResult], bool]:
    results: list[ToolExecutionResult] = []
    if plan.get("id") and str(plan.get("status") or "") not in APPROVED_DRAFT_STATUSES:
        return [
            ToolExecutionResult(
                False,
                "controlled_agent_plan",
                "not_approved",
                "受控 Agent 草稿必须先 /agent 采纳 后才能执行；已拒绝或未审核草稿不会执行。",
            )
        ], False

    if not confirmed:
        for step in plan.get("steps") or []:
            spec = get_controlled_tool(str(step.get("tool_name") or ""))
            if spec and spec.requires_confirmation:
                return [
                    ToolExecutionResult(
                        False,
                        spec.name,
                        "needs_confirmation",
                        "计划包含高风险步骤，需要确认后再执行。",
                        True,
                    )
                ], True

    needs_confirmation = False
    for step in plan.get("steps") or []:
        result = await execute_controlled_tool(
            str(step.get("tool_name") or ""),
            str(step.get("payload") or ""),
            context,
            todo_store=todo_store,
            user_profile=user_profile,
            status_lines=status_lines,
            confirmed=confirmed,
            session_clearer=session_clearer,
        )
        results.append(result)
        if result.requires_confirmation:
            needs_confirmation = True
            break
        if not result.ok:
            break
    return results, needs_confirmation


def serialize_execution_results(results: list[ToolExecutionResult]) -> str:
    return "; ".join(f"{item.tool_name}:{item.status}" for item in results)[:400]
