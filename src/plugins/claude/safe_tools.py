"""低风险工具集合。"""

import ast
import json
import math
import operator
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


SAFE_TOOL_DESCRIPTIONS = [
    ("权限", "/权限", "查看当前用户是否为主人。"),
    ("时间", "/time 或 时间", "查看本机当前时间。"),
    ("计算", "计算：1 + 2 * 3", "执行安全四则运算。"),
    ("待办", "待办 添加 买牛奶", "管理当前 QQ 用户自己的待办。"),
    ("记忆查询", "记忆查询 关键词", "搜索当前用户画像里已保存的资料。"),
    ("资料", "我的资料", "查看当前用户画像。"),
]
OWNER_TOOL_DESCRIPTIONS = [
    ("状态", "/status 或 状态", "查看模型、记忆开关、bot 账号和最近错误。"),
    ("记忆开关", "记忆开关 开/关", "启用或关闭自动事实抽取。"),
    ("模型", "/model", "查看或切换模型。"),
    ("群聊清空", "/clear", "清空当前群聊会话历史。"),
    ("风格画像", "/风格 查看/导入/设置", "维护 owner 的说话风格画像。"),
    ("风格草稿", "/用我的风格回复：...", "按 owner 风格生成回复草稿，不自动代发。"),
]

ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
MAX_CALC_EXPR_LENGTH = 120
MAX_ABS_CALC_VALUE = 10 ** 12


def format_tool_list(auto_memory_enabled: bool, include_owner_tools: bool = False) -> str:
    """格式化工具列表。"""
    lines = [
        "当前可用工具：",
        f"- 自动记忆：{'开' if auto_memory_enabled else '关'}",
    ]
    lines.extend(f"- {name}：{usage}；{desc}" for name, usage, desc in SAFE_TOOL_DESCRIPTIONS)
    if include_owner_tools:
        lines.append("主人管理工具：")
        lines.extend(f"- {name}：{usage}；{desc}" for name, usage, desc in OWNER_TOOL_DESCRIPTIONS)
    return "\n".join(lines)


def format_current_time() -> str:
    """返回本机当前时间。"""
    now = datetime.now()
    return now.strftime("当前本机时间：%Y-%m-%d %H:%M:%S")


def _normalize_expression(expression: str) -> str:
    return (
        expression.strip()
        .replace("×", "*")
        .replace("÷", "/")
        .replace("（", "(")
        .replace("）", ")")
    )


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)

    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        if not math.isfinite(node.value) or abs(node.value) > MAX_ABS_CALC_VALUE:
            raise ValueError("数值过大")
        return node.value

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_BIN_OPS:
            raise ValueError("只支持 + - * / // % ** 和括号")
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("指数过大")
        result = ALLOWED_BIN_OPS[op_type](left, right)
        if not isinstance(result, (int, float)) or not math.isfinite(result):
            raise ValueError("结果不是有效数字")
        if abs(result) > MAX_ABS_CALC_VALUE:
            raise ValueError("结果过大")
        return result

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_UNARY_OPS:
            raise ValueError("不支持该一元运算")
        result = ALLOWED_UNARY_OPS[op_type](_eval_ast(node.operand))
        if abs(result) > MAX_ABS_CALC_VALUE:
            raise ValueError("结果过大")
        return result

    raise ValueError("表达式里包含不支持的内容")


def safe_calculate(expression: str) -> str:
    """安全计算纯数学表达式，不允许名称、调用、属性访问。"""
    expr = _normalize_expression(expression)
    if not expr:
        return "用法：计算：1 + 2 * 3"
    if len(expr) > MAX_CALC_EXPR_LENGTH:
        return "表达式太长。"

    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_ast(tree)
    except ZeroDivisionError:
        return "计算失败：除数不能为 0。"
    except Exception as e:
        return f"计算失败：{e}"

    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expr} = {result}"


class TodoStore:
    """按 QQ 用户隔离的本地待办存储。"""

    def __init__(self, path: Path | str = "data/todos.json"):
        self.path = Path(path)

    def _load(self) -> Dict[str, List[Dict[str, Any]]]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add(self, user_id: str, content: str) -> Dict[str, Any]:
        text = content.strip()
        if not text:
            raise ValueError("待办内容不能为空")
        if len(text) > 160:
            raise ValueError("待办内容太长")

        data = self._load()
        item = {
            "id": uuid.uuid4().hex[:8],
            "content": text,
            "done": False,
            "created_at": time.time(),
            "completed_at": None,
        }
        data.setdefault(user_id, []).append(item)
        self._save(data)
        return item

    def list(self, user_id: str, include_done: bool = False) -> List[Dict[str, Any]]:
        items = self._load().get(user_id, [])
        if include_done:
            return items
        return [item for item in items if not item.get("done")]

    def complete(self, user_id: str, selector: str) -> Optional[Dict[str, Any]]:
        key = selector.strip()
        if not key:
            raise ValueError("请提供待办编号或 ID")

        data = self._load()
        items = data.get(user_id, [])
        pending = [item for item in items if not item.get("done")]
        target = None

        if key.isdigit():
            index = int(key) - 1
            if 0 <= index < len(pending):
                target = pending[index]
        else:
            target = next(
                (item for item in pending if item.get("id", "").startswith(key)),
                None,
            )

        if not target:
            return None

        target["done"] = True
        target["completed_at"] = time.time()
        self._save(data)
        return target

    def clear_user(self, user_id: str) -> None:
        data = self._load()
        if user_id in data:
            del data[user_id]
            self._save(data)


def format_todo_list(items: List[Dict[str, Any]]) -> str:
    """格式化待办列表。"""
    if not items:
        return "当前没有待办。"
    lines = ["当前待办："]
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. {item['content']} ({item['id']})")
    return "\n".join(lines)


def parse_todo_command(text: str) -> tuple[str, str]:
    """解析待办命令，返回 action 和 payload。"""
    stripped = text.strip()
    for prefix in ("/todo", "/待办", "待办"):
        if stripped.lower().startswith(prefix.lower()):
            stripped = stripped[len(prefix):].strip(" ：:")
            break

    if not stripped:
        return "list", ""

    for action, prefixes in {
        "add": ("添加", "新增", "add"),
        "done": ("完成", "done", "finish"),
        "list": ("列表", "查看", "list"),
    }.items():
        for prefix in prefixes:
            if stripped.lower() == prefix.lower():
                return action, ""
            if stripped.lower().startswith(prefix.lower() + " "):
                return action, stripped[len(prefix):].strip()
            if stripped.startswith(prefix + "：") or stripped.startswith(prefix + ":"):
                return action, stripped[len(prefix) + 1:].strip()

    return "add", stripped


def search_profile(profile: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    """搜索当前用户画像。"""
    needle = query.strip().lower()
    if not needle:
        return []
    results = []
    for fact in profile.get("items") or []:
        haystack = f"{fact.get('predicate', '')} {fact.get('object', '')}".lower()
        if needle in haystack:
            results.append(fact)
    return results


def format_profile_search_results(query: str, facts: List[Dict[str, Any]]) -> str:
    """格式化用户画像搜索结果。"""
    if not query.strip():
        return "用法：记忆查询 关键词"
    if not facts:
        return f"没有找到和“{query}”匹配的资料。"
    lines = [f"和“{query}”匹配的资料："]
    for fact in facts[:10]:
        verified = "已确认" if fact.get("verified") else "未确认"
        lines.append(f"- {fact['predicate']}：{fact['object']} ({verified})")
    return "\n".join(lines)


def get_latest_error_header(log_path: Path | str = "data/logs/runtime_errors.log") -> str:
    """读取最近一次运行错误标题，不返回完整堆栈。"""
    path = Path(log_path)
    if not path.exists():
        return "无"
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return "读取失败"

    headers = [line.strip("- ") for line in lines if line.startswith("--- ")]
    return headers[-1] if headers else "无"
