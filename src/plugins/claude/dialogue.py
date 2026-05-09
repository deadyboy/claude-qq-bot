"""QQ 机器人对话处理 - 支持智能体模式

模式切换:
- 简单模式：直接调用 LLM API (向后兼容)
- 智能体模式：使用 Agent Engine，支持记忆/工具/任务
"""

import asyncio
import traceback
from pathlib import Path
from typing import Any
import nonebot
from nonebot import on_message
from nonebot.adapters.onebot.v11 import MessageEvent, GroupMessageEvent, PrivateMessageEvent
from nonebot.typing import T_State

from .api import llm_client
from .memory_core import UnifiedMemoryManager, memory_manager as profile_memory
from .formatter import split_qq_msg, format_reply, sanitize_for_qq_text
from .config import model_config
from .persona import render_system_prompt, summarize_persona
from .auto_memory import extract_user_facts, should_attempt_auto_memory
from .confirmation import (
    confirmation_store,
    format_confirmation_request,
    format_pending_actions,
    parse_confirmation_payload,
)
from .permissions import (
    access_store,
    format_permission_status,
    get_owner_ids,
    is_owner_event,
    owner_required_message,
)
from .runtime_state import is_auto_memory_enabled, set_auto_memory_enabled
from .runtime_state import (
    is_style_auto_reply_enabled,
    is_style_raw_fewshot_enabled,
    is_style_teaching_enabled,
    set_style_auto_reply_enabled,
    set_style_raw_fewshot_enabled,
    set_style_teaching_enabled,
)
from .safe_tools import (
    TodoStore,
    format_current_time,
    format_profile_search_results,
    format_todo_list,
    format_tool_list,
    get_latest_error_header,
    parse_todo_command,
    safe_calculate,
    search_profile,
)
from .style_profile import (
    format_style_help,
    format_style_profile,
    generate_style_draft,
    parse_style_command,
    parse_style_draft_payload,
    parse_style_import_file_payload,
    parse_style_set_payload,
    style_store,
)
from .style_distill import (
    format_qce_distillation_result,
    format_similar_sample_results,
    format_style_debug_report,
    format_style_evaluation_report,
    format_style_relationship_report,
    format_style_scene_report,
    find_source_for_target,
    generate_retrieval_first_reply_candidates,
    retrieve_similar_style_samples,
    run_qce_style_distillation,
)
from .style_teaching import (
    format_recent_reviews,
    format_teaching_review_window,
    format_teaching_status,
    teaching_store,
)

# 智能体引擎 (可选启用)
AGENT_MODE = False  # 设置为 True 启用智能体模式
agent_engine = None

# 系统提示词
SYSTEM_PROMPT = "你是一个 QQ 机器人助手，用简洁友好的语气回复用户。"

# 会话管理器 (简单模式)
session_manager = UnifiedMemoryManager() if AGENT_MODE else None
profile_memory_ready = False

REMEMBER_PREFIXES = ("/remember ", "记住：", "记住:", "记住 ")
FORGET_PREFIXES = ("/forget ", "忘记：", "忘记:", "忘记 ")
todo_store = TodoStore()


def get_session_id(event: MessageEvent) -> str:
    """生成会话 ID"""
    if isinstance(event, GroupMessageEvent):
        return f"group_{event.group_id}"
    else:
        return f"private_{event.user_id}"


def _segment_data(segment: Any) -> dict:
    data = getattr(segment, "data", None)
    return data if isinstance(data, dict) else {}


def _segment_to_text(segment: Any) -> str:
    """Convert non-text QQ message segments into compact text hints."""
    segment_type = str(getattr(segment, "type", "") or "")
    data = _segment_data(segment)

    if segment_type == "text":
        return str(data.get("text") or str(segment)).strip()
    if segment_type == "face":
        label = data.get("name") or data.get("text") or data.get("id") or ""
        return f"[表情:{label}]" if label else "[表情]"
    if segment_type in {"mface", "bface", "market_face"}:
        label = (
            data.get("summary")
            or data.get("name")
            or data.get("text")
            or data.get("emoji_id")
            or data.get("id")
            or ""
        )
        return f"[动画表情:{label}]" if label else "[动画表情]"
    if segment_type == "image":
        summary = str(data.get("summary") or "").strip()
        sub_type = str(data.get("sub_type") or "")
        if "动画表情" in summary or sub_type == "1":
            return "[动画表情]"
        if summary:
            return f"[图片:{summary}]"
        return "[图片]"
    if segment_type == "record":
        return "[语音]"
    if segment_type == "video":
        return "[视频]"
    if segment_type == "file":
        name = data.get("name") or data.get("file") or ""
        return f"[文件:{name}]" if name else "[文件]"
    return ""


def extract_text_and_images(event: MessageEvent) -> tuple[str, list[str]]:
    """从消息中提取可供模型理解的文本提示和图片 URL。"""
    text_parts = []
    images = []

    for segment in event.message:
        if getattr(segment, "type", "") == "image":
            url = _segment_data(segment).get("url", "")
            if url:
                images.append(url)
        text = _segment_to_text(segment)
        if text:
            text_parts.append(text)

    return " ".join(text_parts), images


def get_plain_text(event: MessageEvent) -> str:
    """获取去除首尾空白后的纯文本消息。"""
    return event.message.extract_plain_text().strip()


def _is_exact_command(event: MessageEvent, commands: set[str]) -> bool:
    return get_plain_text(event).lower() in commands


def _starts_with_command(text: str, command: str) -> bool:
    text = text.strip()
    return text == command or text.startswith(f"{command} ")


def is_clear_command(event: MessageEvent) -> bool:
    text = get_plain_text(event)
    return text.lower() == "/clear" or text in {"清空历史", "清除记忆"}


def is_model_command(event: MessageEvent) -> bool:
    text = get_plain_text(event)
    return (
        _starts_with_command(text.lower(), "/model")
        or _starts_with_command(text, "/模型")
    )


def is_tasks_command(event: MessageEvent) -> bool:
    return _is_exact_command(event, {"/tasks"})


def is_status_command(event: MessageEvent) -> bool:
    return _is_exact_command(event, {"/status", "状态", "运行状态"})


def is_help_command(event: MessageEvent) -> bool:
    return _is_exact_command(event, {"/help"})


def is_permission_command(event: MessageEvent) -> bool:
    return _is_exact_command(event, {"/owner", "/权限", "权限", "我的权限"})


def is_confirm_command(event: MessageEvent) -> bool:
    text = get_plain_text(event)
    lowered = text.lower()
    return (
        lowered == "/confirm"
        or lowered.startswith("/confirm ")
        or text == "/确认"
        or text.startswith("/确认 ")
        or text.startswith("/确认：")
        or text.startswith("/确认:")
        or text == "确认"
        or text.startswith("确认 ")
        or text.startswith("确认：")
        or text.startswith("确认:")
    )


def is_cancel_command(event: MessageEvent) -> bool:
    text = get_plain_text(event)
    lowered = text.lower()
    return (
        lowered == "/cancel"
        or lowered.startswith("/cancel ")
        or text == "/取消"
        or text.startswith("/取消 ")
        or text.startswith("/取消：")
        or text.startswith("/取消:")
        or text == "取消"
        or text.startswith("取消 ")
        or text.startswith("取消：")
        or text.startswith("取消:")
    )


def is_access_command(event: MessageEvent) -> bool:
    text = get_plain_text(event)
    lowered = text.lower()
    return (
        lowered == "/access"
        or lowered.startswith("/access ")
        or text == "/白名单"
        or text.startswith("/白名单 ")
        or text.startswith("/白名单：")
        or text.startswith("/白名单:")
        or text == "白名单"
        or text.startswith("白名单 ")
        or text.startswith("白名单：")
        or text.startswith("白名单:")
        or text == "信任名单"
        or text.startswith("信任名单 ")
        or text.startswith("信任名单：")
        or text.startswith("信任名单:")
    )


def is_delegate_command(event: MessageEvent) -> bool:
    text = get_plain_text(event)
    lowered = text.lower()
    return (
        lowered == "/auto-reply"
        or lowered.startswith("/auto-reply ")
        or lowered == "/autoreply"
        or lowered.startswith("/autoreply ")
        or text == "/代聊"
        or text.startswith("/代聊 ")
        or text.startswith("/代聊：")
        or text.startswith("/代聊:")
        or text == "代聊"
        or text.startswith("代聊 ")
        or text.startswith("代聊：")
        or text.startswith("代聊:")
    )


def is_tools_command(event: MessageEvent) -> bool:
    return _is_exact_command(event, {"/tools", "/工具", "工具", "工具列表"})


def is_memory_toggle_command(event: MessageEvent) -> bool:
    text = get_plain_text(event)
    lowered = text.lower()
    memory_toggle_words = {
        "开", "开启", "on", "enable", "enabled", "true", "1",
        "关", "关闭", "off", "disable", "disabled", "false", "0",
    }
    return (
        lowered == "/memory"
        or (
            lowered.startswith("/memory ")
            and lowered.split(maxsplit=1)[1] in memory_toggle_words
        )
        or text == "记忆开关"
        or text.startswith("记忆开关 ")
        or text.startswith("记忆开关：")
        or text.startswith("记忆开关:")
    )


def is_time_command(event: MessageEvent) -> bool:
    return _is_exact_command(event, {"/time", "/时间", "时间", "现在几点"})


def is_calc_command(event: MessageEvent) -> bool:
    text = get_plain_text(event)
    lowered = text.lower()
    return (
        lowered.startswith("/calc ")
        or lowered.startswith("/calculate ")
        or text.startswith("计算 ")
        or text.startswith("计算：")
        or text.startswith("计算:")
    )


def is_todo_command(event: MessageEvent) -> bool:
    text = get_plain_text(event)
    lowered = text.lower()
    return (
        lowered == "/todo"
        or lowered.startswith("/todo ")
        or text == "待办"
        or text.startswith("待办 ")
        or text.startswith("待办：")
        or text.startswith("待办:")
        or text.startswith("/待办")
    )


def is_memory_query_command(event: MessageEvent) -> bool:
    text = get_plain_text(event)
    lowered = text.lower()
    return (
        lowered.startswith("/memory search ")
        or lowered.startswith("/memory 查询 ")
        or text.startswith("记忆查询 ")
        or text.startswith("记忆查询：")
        or text.startswith("记忆查询:")
    )


def is_style_draft_command(event: MessageEvent) -> bool:
    text = get_plain_text(event)
    lowered = text.lower()
    return (
        lowered == "/style draft"
        or lowered.startswith("/style draft ")
        or text == "/用我的风格回复"
        or text.startswith("/用我的风格回复 ")
        or text.startswith("/用我的风格回复：")
        or text.startswith("/用我的风格回复:")
        or text == "用我的风格回复"
        or text.startswith("用我的风格回复 ")
        or text.startswith("用我的风格回复：")
        or text.startswith("用我的风格回复:")
        or text == "风格回复"
        or text.startswith("风格回复 ")
        or text.startswith("风格回复：")
        or text.startswith("风格回复:")
    )


def is_teaching_command(event: MessageEvent) -> bool:
    text = get_plain_text(event)
    lowered = text.lower()
    return (
        lowered == "/teach"
        or lowered.startswith("/teach ")
        or text == "/教学"
        or text.startswith("/教学 ")
        or text.startswith("/教学：")
        or text.startswith("/教学:")
        or text == "教学"
        or text.startswith("教学 ")
        or text.startswith("教学：")
        or text.startswith("教学:")
        or text == "/采纳"
        or text.startswith("/采纳 ")
        or text.startswith("/采纳：")
        or text.startswith("/采纳:")
        or text == "/评分"
        or text.startswith("/评分 ")
        or text.startswith("/评分：")
        or text.startswith("/评分:")
        or text == "/改成"
        or text.startswith("/改成 ")
        or text.startswith("/改成：")
        or text.startswith("/改成:")
        or text == "/拒绝"
        or text.startswith("/拒绝 ")
        or text.startswith("/拒绝：")
        or text.startswith("/拒绝:")
    )


def is_style_command(event: MessageEvent) -> bool:
    text = get_plain_text(event)
    lowered = text.lower()
    if lowered == "/style draft" or lowered.startswith("/style draft "):
        return False
    return (
        lowered == "/style"
        or lowered.startswith("/style ")
        or text == "/风格"
        or text.startswith("/风格 ")
        or text.startswith("/风格：")
        or text.startswith("/风格:")
        or text == "风格"
        or text.startswith("风格 ")
        or text.startswith("风格：")
        or text.startswith("风格:")
    )


def is_remember_command(event: MessageEvent) -> bool:
    text = get_plain_text(event)
    lowered = text.lower()
    return (
        lowered.startswith("/remember ")
        or text.startswith("记住：")
        or text.startswith("记住:")
        or text.startswith("记住 ")
        or (text.startswith("记住") and len(text) > 2)
    )


def is_forget_command(event: MessageEvent) -> bool:
    text = get_plain_text(event)
    lowered = text.lower()
    return (
        lowered.startswith("/forget")
        or text.startswith("忘记：")
        or text.startswith("忘记:")
        or text.startswith("忘记 ")
        or text in {"忘记我的资料", "清除我的资料", "删除我的资料"}
    )


def is_profile_command(event: MessageEvent) -> bool:
    return _is_exact_command(
        event,
        {"/profile", "我的资料", "我的记忆", "你记住了什么"}
    )


def is_persona_command(event: MessageEvent) -> bool:
    return _is_exact_command(event, {"/persona", "/identity", "身份", "你是谁"})


def is_to_bot(event: MessageEvent, bot: nonebot.adapters.onebot.v11.Bot) -> bool:
    """判断消息是否明确发给机器人。

    NoneBot 会在预处理阶段移除开头/结尾的 @ 段，并把 event.to_me 设为 True，
    因此这里必须优先使用 is_tome()，不能只遍历 event.message。
    """
    if event.is_tome():
        return True

    self_id = str(bot.self_id)
    return any(
        seg.type == "at" and str(seg.data.get("qq")) == self_id
        for seg in event.message
    )


def should_handle_targeted_event(
    event: MessageEvent,
    bot: nonebot.adapters.onebot.v11.Bot,
) -> bool:
    """私聊总是处理；群聊只处理 @ 或回复机器人的消息。"""
    if not isinstance(event, GroupMessageEvent):
        return True

    is_reply = event.reply and str(event.reply.user_id) == str(bot.self_id)
    return is_to_bot(event, bot) or is_reply


async def require_owner(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    action: str,
) -> bool:
    """要求当前用户是 owner。"""
    if is_owner_event(event):
        return True

    await send_qq_text(bot, event, owner_required_message(action))
    return False


async def ensure_profile_memory_ready():
    """懒加载用户画像存储，避免简单聊天启动时多余初始化。"""
    global profile_memory_ready
    if not profile_memory_ready:
        await profile_memory.initialize()
        profile_memory_ready = True


def _strip_command_prefix(text: str, prefixes: tuple[str, ...]) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix.lower()):
            return stripped[len(prefix):].strip()

    if stripped.startswith("记住"):
        return stripped[2:].lstrip(" ：:，,").strip()
    if stripped.startswith("忘记"):
        return stripped[2:].lstrip(" ：:，,").strip()
    return ""


def parse_remember_payload(text: str) -> str:
    return _strip_command_prefix(text, REMEMBER_PREFIXES)


def parse_forget_payload(text: str) -> str:
    if text in {"忘记我的资料", "清除我的资料", "删除我的资料"}:
        return "全部"
    payload = _strip_command_prefix(text, FORGET_PREFIXES)
    if text.lower().strip() == "/forget":
        return "全部"
    return payload


def parse_memory_toggle_payload(text: str) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    for prefix in ("/memory", "记忆开关"):
        if lowered.startswith(prefix.lower()):
            return stripped[len(prefix):].strip(" ：:").lower()
    return ""


def parse_calc_payload(text: str) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    for prefix in ("/calc", "/calculate", "计算"):
        if lowered.startswith(prefix.lower()):
            return stripped[len(prefix):].strip(" ：:")
    return ""


def parse_memory_query_payload(text: str) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    prefixes = ("/memory search", "/memory 查询", "记忆查询")
    for prefix in prefixes:
        if lowered.startswith(prefix.lower()):
            return stripped[len(prefix):].strip(" ：:")
    return ""


def parse_confirm_payload(text: str) -> str:
    return parse_confirmation_payload(text, ("/confirm", "/确认", "确认"))


def parse_cancel_payload(text: str) -> str:
    return parse_confirmation_payload(text, ("/cancel", "/取消", "取消"))


def parse_access_payload(text: str) -> tuple[str, str, str]:
    """Parse whitelist management command into action, target id, and note."""
    stripped = text.strip()
    lowered = stripped.lower()
    for prefix in ("/access", "/白名单", "白名单", "信任名单"):
        if lowered == prefix.lower():
            return "view", "", ""
        if lowered.startswith(prefix.lower() + " "):
            stripped = stripped[len(prefix):].strip()
            break
        if stripped.startswith(prefix + "：") or stripped.startswith(prefix + ":"):
            stripped = stripped[len(prefix) + 1:].strip()
            break
    else:
        return "view", "", ""

    lowered = stripped.lower()
    actions = {
        "add_user": ("添加用户", "加用户", "add-user", "add user", "user add"),
        "remove_user": ("删除用户", "移除用户", "删用户", "remove-user", "del-user", "delete-user"),
        "add_group": ("添加群", "加群", "add-group", "add group", "group add"),
        "remove_group": ("删除群", "移除群", "删群", "remove-group", "del-group", "delete-group"),
        "view": ("查看", "列表", "list", "view", "show"),
    }
    for action, prefixes in actions.items():
        for prefix in prefixes:
            if lowered == prefix.lower():
                return action, "", ""
            if lowered.startswith(prefix.lower() + " "):
                payload = stripped[len(prefix):].strip()
                parts = payload.split(maxsplit=1)
                target = parts[0] if parts else ""
                note = parts[1] if len(parts) > 1 else ""
                return action, target, note
            if stripped.startswith(prefix + "：") or stripped.startswith(prefix + ":"):
                payload = stripped[len(prefix) + 1:].strip()
                parts = payload.split(maxsplit=1)
                target = parts[0] if parts else ""
                note = parts[1] if len(parts) > 1 else ""
                return action, target, note

    return "view", "", ""


def parse_switch_payload(text: str, prefixes: tuple[str, ...]) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    for prefix in prefixes:
        if lowered == prefix.lower():
            return ""
        if lowered.startswith(prefix.lower() + " "):
            return stripped[len(prefix):].strip(" ：:").lower()
        if stripped.startswith(prefix + "：") or stripped.startswith(prefix + ":"):
            return stripped[len(prefix) + 1:].strip().lower()
    return stripped.strip(" ：:").lower()


def parse_delegate_payload(text: str) -> str:
    return parse_switch_payload(text, ("/auto-reply", "/autoreply", "/代聊", "代聊"))


def parse_style_switch_payload(payload: str) -> str:
    return payload.strip(" ：:").lower()


def parse_teaching_payload(text: str) -> tuple[str, str]:
    stripped = text.strip()
    lowered = stripped.lower()
    prefixes = (
        ("/teach", "control"),
        ("/教学", "control"),
        ("教学", "control"),
        ("/采纳", "accept"),
        ("/评分", "rate"),
        ("/改成", "correct"),
        ("/拒绝", "reject"),
    )
    for prefix, action in prefixes:
        if lowered == prefix.lower():
            return action, ""
        if lowered.startswith(prefix.lower() + " "):
            return action, stripped[len(prefix):].strip(" ：:")
        if stripped.startswith(prefix + "：") or stripped.startswith(prefix + ":"):
            return action, stripped[len(prefix) + 1:].strip()
    return "control", stripped


def _split_review_payload(payload: str) -> tuple[str, str]:
    parts = payload.strip().split(maxsplit=1)
    if not parts:
        return "", ""
    return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""


def _resolve_review_for_feedback(payload: str, actor_id: str | int) -> tuple[str, str]:
    first, rest = _split_review_payload(payload)
    if first.startswith("T") or first.startswith("t"):
        return first, rest
    latest = teaching_store.latest_for_reviewer(actor_id)
    if not latest:
        return "", payload.strip()
    return str(latest.get("id") or ""), payload.strip()


def _is_switch_on(value: str) -> bool:
    return value in {"开", "开启", "on", "enable", "enabled", "true", "1"}


def _is_switch_off(value: str) -> bool:
    return value in {"关", "关闭", "off", "disable", "disabled", "false", "0"}


def infer_user_fact(payload: str) -> tuple[str, str]:
    """把显式记忆文本转成简单的用户画像键值。"""
    text = payload.strip().strip("。.!！")

    for separator in ("=", "：", ":"):
        if separator in text:
            key, value = text.split(separator, 1)
            key = key.strip().removeprefix("我的").strip()
            value = value.strip()
            if key and value:
                return key, value

    patterns = (
        ("以后叫我", "称呼", ""),
        ("以后称呼我", "称呼", ""),
        ("我叫", "称呼", ""),
        ("我是", "身份", ""),
        ("我喜欢", "偏好", "喜欢"),
        ("我不喜欢", "偏好", "不喜欢"),
    )
    for prefix, predicate, value_prefix in patterns:
        if text.startswith(prefix):
            value = text[len(prefix):].strip()
            if value:
                return predicate, f"{value_prefix}{value}" if value_prefix else value

    if text.startswith("我的"):
        body = text[2:]
        for separator in ("是", "为", "叫"):
            if separator in body:
                key, value = body.split(separator, 1)
                key = key.strip()
                value = value.strip()
                if key and value:
                    return key, value

    return "备注", text


def format_user_profile(profile: dict) -> str:
    items = profile.get("items") or []
    if not items:
        return "还没有保存你的资料。"

    lines = []
    for fact in items[:20]:
        verified = "已确认" if fact.get("verified") else "未确认"
        lines.append(f"- {fact['predicate']}：{fact['object']} ({verified})")
    return "\n".join(lines)


def profile_context_for_prompt(profile: dict) -> str:
    items = profile.get("items") or []
    if not items:
        return ""

    lines = []
    for fact in items[:12]:
        lines.append(f"- {fact['predicate']}：{fact['object']}")
    return "\n".join(lines)


async def auto_remember_user_facts(user_id: str, session_id: str, text: str):
    """后台抽取并保存用户画像，不影响当前回复链路。"""
    if not is_auto_memory_enabled():
        return

    if not should_attempt_auto_memory(text):
        return

    try:
        facts = await extract_user_facts(text)
        if not facts:
            return

        await ensure_profile_memory_ready()
        for fact in facts:
            await profile_memory.remember_about_user(
                user_id,
                fact["predicate"],
                fact["object"],
                verified=False,
                confidence=fact.get("confidence", 0.8),
                metadata={
                    "source": "auto_extract",
                    "session_id": session_id,
                },
            )
    except Exception as e:
        write_runtime_error("auto_remember_user_facts", e)


async def send_qq_text(bot: nonebot.adapters.onebot.v11.Bot, event: MessageEvent, text: str):
    """发送 QQ 文本；失败时降级为更保守的纯文本再试一次。"""
    try:
        await bot.send(event, text)
    except Exception:
        fallback = sanitize_for_qq_text(text)
        if fallback and fallback != text:
            await bot.send(event, fallback)
            return
        raise


async def send_owner_private_text(bot: nonebot.adapters.onebot.v11.Bot, owner_id: str, text: str) -> bool:
    """Send a private teaching/review message to one owner."""
    try:
        for part in split_qq_msg(text):
            await bot.call_api("send_private_msg", user_id=int(owner_id), message=part)
        return True
    except Exception as e:
        write_runtime_error("send_owner_private_text", e)
        return False


async def generate_teaching_candidates(
    text: str,
    *,
    chat_type: str,
    target_id: str,
    actor_id: str | int,
    recent_dialogue: list[dict] | None = None,
) -> tuple[list[str], dict]:
    """Generate up to three owner-style candidates for review, without sending them to the contact."""
    metadata: dict[str, Any] = {"generator": "retrieval_first"}
    candidates: list[str] = []
    try:
        retrieval_result = await generate_retrieval_first_reply_candidates(
            text,
            current_context=recent_dialogue,
            chat_type=chat_type,
            target_id=target_id,
        )
        metadata["retrieval"] = {
            "ok": retrieval_result.get("ok"),
            "run_id": retrieval_result.get("run_id"),
            "scene_label": retrieval_result.get("scene_label"),
            "result_count": (retrieval_result.get("retrieval") or {}).get("result_count"),
        }
        for item in retrieval_result.get("candidates") or []:
            if not item.get("accepted"):
                continue
            candidate = str(item.get("text") or "").strip()
            if candidate and candidate not in candidates:
                candidates.append(candidate)
            if len(candidates) >= 3:
                break
    except Exception as e:
        metadata["retrieval_error"] = type(e).__name__
        write_runtime_error("generate_teaching_candidates_retrieval", e)

    if len(candidates) < 3:
        try:
            fallback = await generate_style_draft(
                text,
                include_raw_fewshot=is_style_raw_fewshot_enabled(),
                chat_type=chat_type,
                target_id=target_id,
                actor_id=actor_id,
                scope=chat_type,
                auto_reply=False,
                recent_dialogue=recent_dialogue,
            )
            fallback = format_reply(fallback)
            if fallback and fallback not in candidates:
                candidates.append(fallback)
            metadata["fallback"] = "style_draft"
        except Exception as e:
            metadata["fallback_error"] = type(e).__name__
            write_runtime_error("generate_teaching_candidates_fallback", e)

    return candidates[:3], metadata


def write_runtime_error(scope: str, error: Exception):
    """写入运行期异常，便于独立窗口运行时排查。"""
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "runtime_errors.log").open("a", encoding="utf-8") as f:
        f.write(f"\n--- {scope}: {type(error).__name__}: {error} ---\n")
        f.write(traceback.format_exc())


def create_confirmation(
    event: MessageEvent,
    action_type: str,
    summary: str,
    payload: dict,
) -> str:
    """Create a pending action and return a user-facing confirmation prompt."""
    action = confirmation_store.create(
        action_type=action_type,
        created_by=event.user_id,
        summary=summary,
        payload=payload,
    )
    return format_confirmation_request(action)


async def execute_pending_action(action: dict) -> str:
    """Execute a confirmed pending action."""
    action_type = action.get("type", "")
    payload = action.get("payload") or {}

    if action_type == "access_add_user":
        _, msg = access_store.add_user(
            payload.get("target_id", ""),
            note=payload.get("note", ""),
            added_by=action.get("created_by", ""),
        )
        return msg

    if action_type == "access_remove_user":
        _, msg = access_store.remove_user(payload.get("target_id", ""))
        return msg

    if action_type == "access_add_group":
        _, msg = access_store.add_group(
            payload.get("target_id", ""),
            note=payload.get("note", ""),
            added_by=action.get("created_by", ""),
        )
        return msg

    if action_type == "access_remove_group":
        _, msg = access_store.remove_group(payload.get("target_id", ""))
        return msg

    if action_type == "style_clear_examples":
        return style_store.clear_examples()

    if action_type == "style_raw_fewshot_set":
        enabled = str(payload.get("enabled", "")).lower() in {"true", "1", "yes", "on"}
        set_style_raw_fewshot_enabled(enabled)
        return f"真实历史原句 few-shot 已{'开启' if enabled else '关闭'}。"

    if action_type == "style_auto_reply_set":
        enabled = str(payload.get("enabled", "")).lower() in {"true", "1", "yes", "on"}
        set_style_auto_reply_enabled(enabled)
        return f"受控代聊自动回复已{'开启' if enabled else '关闭'}。"

    if action_type == "clear_session":
        session_id = payload.get("session_id", "")
        if not session_id:
            return "清空失败：缺少会话 ID。"
        if AGENT_MODE and agent_engine:
            await agent_engine.memory.short_term.clear(session_id)
        else:
            from .memory import session_manager as simple_manager
            await simple_manager.clear_session(session_id)
        return "会话历史已清空。"

    return f"未知待确认操作：{action_type}"


def _style_auto_scope(event: MessageEvent) -> tuple[str, str, bool]:
    """Return chat_type, target_id, and whether the target is trusted."""
    if isinstance(event, GroupMessageEvent):
        return "group", str(event.group_id), access_store.is_trusted_group(event.group_id)
    return "private", str(event.user_id), access_store.is_trusted_user(event.user_id)


def _looks_like_command_text(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith("/"):
        return True
    exact_commands = {"时间", "状态", "权限", "工具", "我的资料", "记忆开关", "待办", "风格", "白名单", "代聊", "教学"}
    if stripped in exact_commands:
        return True
    command_prefixes = ("记住", "忘记", "待办", "计算", "记忆查询", "风格", "白名单", "代聊", "教学", "采纳", "评分", "改成", "拒绝")
    return any(
        stripped.startswith(prefix + sep)
        for prefix in command_prefixes
        for sep in (" ", "：", ":")
    )


async def maybe_handle_style_auto_reply(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    text: str,
) -> bool:
    """Return True when style auto-reply mode handled or intentionally suppressed the event."""
    if not is_style_auto_reply_enabled():
        return False
    if not text.strip() or _looks_like_command_text(text):
        return False

    chat_type, target_id, trusted = _style_auto_scope(event)
    if not trusted:
        return True

    mapping = find_source_for_target(target_id, chat_type=chat_type)
    if not mapping.get("matched"):
        confirmation_store.log(
            {
                "id": "style_skip",
                "type": "style_auto_reply_skip",
                "summary": f"受控代聊跳过：未匹配 {chat_type} 本地画像",
            },
            actor_id=getattr(event, "user_id", ""),
            status="skipped",
            result=f"chat_type={chat_type}",
        )
        return True

    from .memory import session_manager as simple_session_manager
    session_id = get_session_id(event)
    history = await simple_session_manager.get_messages(session_id)

    try:
        await simple_session_manager.add_message(session_id, "user", text)
        draft = await generate_style_draft(
            text,
            include_raw_fewshot=is_style_raw_fewshot_enabled(),
            chat_type=chat_type,
            target_id=target_id,
            actor_id=getattr(event, "user_id", ""),
            scope=chat_type,
            auto_reply=True,
            recent_dialogue=history,
        )
        response = format_reply(draft)
        for part in split_qq_msg(response):
            await send_qq_text(bot, event, part)
        await simple_session_manager.add_message(session_id, "assistant", response)
    except Exception as e:
        write_runtime_error("maybe_handle_style_auto_reply", e)
    return True


async def maybe_handle_style_teaching_review(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    text: str,
) -> bool:
    """Shadow mode: generate candidates for owner review instead of replying to trusted contacts."""
    if not is_style_teaching_enabled():
        return False
    if not text.strip() or _looks_like_command_text(text):
        return False
    if isinstance(event, GroupMessageEvent):
        return False
    if is_owner_event(event):
        return False

    chat_type, target_id, trusted = _style_auto_scope(event)
    if not trusted:
        return False
    owner_ids = sorted(get_owner_ids())
    if not owner_ids:
        return False

    from .memory import session_manager as simple_session_manager
    session_id = get_session_id(event)
    history = await simple_session_manager.get_messages(session_id)
    await simple_session_manager.add_message(session_id, "user", text)

    candidates, metadata = await generate_teaching_candidates(
        text,
        chat_type=chat_type,
        target_id=target_id,
        actor_id=getattr(event, "user_id", ""),
        recent_dialogue=history,
    )
    if not candidates:
        confirmation_store.log(
            {
                "id": "teach_fail",
                "type": "style_teaching_review",
                "summary": "教学模式生成候选失败",
            },
            actor_id=getattr(event, "user_id", ""),
            status="failed",
            result=f"target={target_id}",
        )
        return True

    review = teaching_store.create_review(
        message=text,
        candidates=candidates,
        chat_type=chat_type,
        target_id=target_id,
        trigger="shadow",
        recent_dialogue=history,
        reviewer_ids=owner_ids,
        metadata=metadata,
    )
    window = format_teaching_review_window(review)
    sent = False
    for owner_id in owner_ids:
        sent = await send_owner_private_text(bot, owner_id, window) or sent
    confirmation_store.log(
        {
            "id": str(review.get("id") or "")[:16],
            "type": "style_teaching_review",
            "summary": "教学模式生成候选并发送主人审核",
        },
        actor_id=getattr(event, "user_id", ""),
        status="executed" if sent else "notify_failed",
        result=f"target={target_id};candidates={len(candidates)}",
    )
    return True


# ========== 简单模式处理器 (向后兼容) ==========

simple_chat_handler = on_message(priority=5, block=False)


@simple_chat_handler.handle()
async def handle_simple_chat(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """简单模式：直接调用 LLM API"""
    if AGENT_MODE:
        return  # 智能体模式启用时跳过

    import logging
    logger = logging.getLogger('claude_bot')

    # 过滤机器人自己发的消息
    if event.user_id == bot.self_id:
        return

    # 群聊需要 @机器人 或回复
    if not should_handle_targeted_event(event, bot):
        return

    text, images = extract_text_and_images(event)

    # 过滤 @ 和回复标记
    if isinstance(event, GroupMessageEvent):
        text = text.replace(f"@{bot.nickname}", "").strip()
        if not text and not images:
            text = "在不在"

    if not text and not images:
        return

    if await maybe_handle_style_teaching_review(bot, event, text):
        return
    if await maybe_handle_style_auto_reply(bot, event, text):
        return
    if is_style_auto_reply_enabled():
        return

    # 获取会话 ID 和历史
    session_id = get_session_id(event)
    user_id = str(event.user_id)

    # 使用 SimpleSessionManager (向后兼容)
    from .memory import session_manager as simple_session_manager
    history = await simple_session_manager.get_messages(session_id)
    await ensure_profile_memory_ready()
    profile = await profile_memory.get_user_profile(user_id)
    session_kind = "group" if isinstance(event, GroupMessageEvent) else "private"
    system_prompt = render_system_prompt(
        base_prompt=SYSTEM_PROMPT,
        user_profile=profile_context_for_prompt(profile),
        session_kind=session_kind,
    )

    # 添加用户消息
    user_content = text or "[图片]"
    await simple_session_manager.add_message(session_id, "user", user_content)

    # 构建 API 请求
    messages = history + [{"role": "user", "content": text or "请描述这张图片"}]

    try:
        response = await llm_client.chat(messages=messages, system_prompt=system_prompt)
        response = format_reply(response)
        parts = split_qq_msg(response)

        for part in parts:
            await send_qq_text(bot, event, part)

        # 添加 AI 回复
        await simple_session_manager.add_message(session_id, "assistant", response)
        if text and not images and not isinstance(event, GroupMessageEvent):
            asyncio.create_task(auto_remember_user_facts(user_id, session_id, text))

    except Exception as e:
        write_runtime_error("handle_simple_chat", e)
        error_msg = f"API 调用失败：{type(e).__name__}"
        logger.exception(f"错误：{error_msg}: {e}")
        await bot.send(event, error_msg)


# ========== 智能体模式处理器 ==========

agent_chat_handler = on_message(priority=5, block=False)


@agent_chat_handler.handle()
async def handle_agent_chat(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """智能体模式：使用 Agent Engine"""
    if not AGENT_MODE:
        return  # 智能体模式未启用时跳过

    import logging
    logger = logging.getLogger('claude_bot_agent')

    # 过滤机器人自己
    if event.user_id == bot.self_id:
        return

    # 群聊需要 @机器人 或回复
    if isinstance(event, GroupMessageEvent):
        is_mentioned = is_to_bot(event, bot)
        is_reply = event.reply and str(event.reply.user_id) == str(bot.self_id)

        if not is_mentioned and not is_reply:
            return

    text, _ = extract_text_and_images(event)

    # 过滤 @ 标记
    if isinstance(event, GroupMessageEvent):
        text = text.replace(f"@{bot.nickname}", "").strip()
        if not text:
            text = "在不在"

    if not text:
        return

    # 获取会话信息
    session_id = get_session_id(event)
    user_id = str(event.user_id)

    # 初始化智能体引擎
    global agent_engine
    if agent_engine is None:
        from .agent import agent_engine as engine
        agent_engine = engine
        await agent_engine.initialize()

    try:
        # 使用智能体引擎处理
        response = await agent_engine.process_message(
            user_id=user_id,
            session_id=session_id,
            message=text
        )

        # 发送回复
        parts = split_qq_msg(format_reply(response))
        for part in parts:
            await send_qq_text(bot, event, part)

        logger.info(f"智能体回复：{response[:100]}...")

    except Exception as e:
        write_runtime_error("handle_agent_chat", e)
        error_msg = f"智能体处理失败：{type(e).__name__}: {e}"
        logger.exception(f"错误：{error_msg}")
        await bot.send(event, error_msg)


# ========== 命令处理器 ==========

clear_cmd = on_message(rule=is_clear_command, priority=4, block=True)


@clear_cmd.handle()
async def handle_clear(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """处理 /clear 命令"""
    if not should_handle_targeted_event(event, bot):
        return
    if isinstance(event, GroupMessageEvent) and not await require_owner(bot, event, "群聊清空历史"):
        return

    session_id = get_session_id(event)
    if isinstance(event, GroupMessageEvent):
        await send_qq_text(
            bot,
            event,
            create_confirmation(
                event,
                "clear_session",
                f"清空群聊 {event.group_id} 的会话历史",
                {"session_id": session_id, "scope": "group", "group_id": event.group_id},
            ),
        )
        return

    if AGENT_MODE and agent_engine:
        await agent_engine.memory.short_term.clear(session_id)
    else:
        from .memory import session_manager as simple_manager
        await simple_manager.clear_session(session_id)

    await send_qq_text(bot, event, "会话历史已清空")


model_cmd = on_message(rule=is_model_command, priority=4, block=True)


@model_cmd.handle()
async def handle_model_switch(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """处理 /model 命令"""
    if not should_handle_targeted_event(event, bot):
        return
    if not await require_owner(bot, event, "模型管理"):
        return

    text = get_plain_text(event)

    parts = text.split()
    if len(parts) < 2:
        current = model_config.get_current_model()
        api_base = model_config.get_current_api_base()
        available = ", ".join(model_config.list_models())
        msg = f"当前模型：{current}\nAPI Base：{api_base}\n可用模型：{available}\n用法：/model <模型名>"
        await send_qq_text(bot, event, msg)
        return

    model_name = parts[1]
    success, msg = model_config.switch_model(model_name)
    if success:
        llm_client.configure(
            model=model_config.get_current_model(),
            base_url=model_config.get_current_api_base(),
        )
    await send_qq_text(bot, event, msg)


# ========== 状态与低风险工具 ==========

status_cmd = on_message(rule=is_status_command, priority=4, block=True)


@status_cmd.handle()
async def handle_basic_status(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """查看基础运行状态。"""
    if not should_handle_targeted_event(event, bot):
        return
    if not await require_owner(bot, event, "运行状态"):
        return

    await ensure_profile_memory_ready()
    profile = await profile_memory.get_user_profile(str(event.user_id))
    session_kind = "群聊" if isinstance(event, GroupMessageEvent) else "私聊"
    msg = "\n".join([
        "运行状态：",
        f"- Bot QQ：{bot.self_id}",
        f"- 当前场景：{session_kind}",
        f"- 模式：{'Agent Mode' if AGENT_MODE else '简单稳定模式'}",
        f"- 模型：{model_config.get_current_model()}",
        f"- API Base：{model_config.get_current_api_base()}",
        f"- 自动记忆：{'开' if is_auto_memory_enabled() else '关'}",
        f"- 代聊自动回复：{'开' if is_style_auto_reply_enabled() else '关'}",
        f"- 教学影子审核：{'开' if is_style_teaching_enabled() else '关'}",
        f"- 真实原句 few-shot：{'开' if is_style_raw_fewshot_enabled() else '关'}",
        f"- 你的资料：{len(profile.get('items') or [])} 条",
        f"- 最近错误：{get_latest_error_header()}",
    ])
    await send_qq_text(bot, event, msg)


tools_cmd = on_message(rule=is_tools_command, priority=4, block=True)


@tools_cmd.handle()
async def handle_tools(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """列出当前可用工具。"""
    if not should_handle_targeted_event(event, bot):
        return

    await send_qq_text(
        bot,
        event,
        format_tool_list(
            is_auto_memory_enabled(),
            include_owner_tools=is_owner_event(event),
        ),
    )


permission_cmd = on_message(rule=is_permission_command, priority=4, block=True)


@permission_cmd.handle()
async def handle_permission(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """查看当前用户权限。"""
    if not should_handle_targeted_event(event, bot):
        return

    group_id = event.group_id if isinstance(event, GroupMessageEvent) else None
    await send_qq_text(bot, event, format_permission_status(event.user_id, group_id))


confirm_cmd = on_message(rule=is_confirm_command, priority=4, block=True)


@confirm_cmd.handle()
async def handle_confirm_action(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """执行待确认操作。"""
    if not should_handle_targeted_event(event, bot):
        return
    if not await require_owner(bot, event, "确认操作"):
        return

    action_id = parse_confirm_payload(get_plain_text(event))
    if not action_id:
        await send_qq_text(
            bot,
            event,
            format_pending_actions(confirmation_store.list_for_actor(event.user_id)),
        )
        return

    action, error = confirmation_store.pop_for_actor(action_id, event.user_id)
    if not action:
        await send_qq_text(bot, event, error)
        return

    try:
        result = await execute_pending_action(action)
        confirmation_store.log(action, actor_id=event.user_id, status="executed", result=result)
        await send_qq_text(bot, event, "已执行：\n" + result)
    except Exception as e:
        write_runtime_error("handle_confirm_action", e)
        confirmation_store.log(action, actor_id=event.user_id, status="failed", result=f"{type(e).__name__}: {e}")
        await send_qq_text(bot, event, f"执行失败：{type(e).__name__}")


cancel_cmd = on_message(rule=is_cancel_command, priority=4, block=True)


@cancel_cmd.handle()
async def handle_cancel_action(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """取消待确认操作。"""
    if not should_handle_targeted_event(event, bot):
        return
    if not await require_owner(bot, event, "取消确认操作"):
        return

    action_id = parse_cancel_payload(get_plain_text(event))
    if not action_id:
        await send_qq_text(
            bot,
            event,
            format_pending_actions(confirmation_store.list_for_actor(event.user_id)),
        )
        return

    _, msg = confirmation_store.cancel_for_actor(action_id, event.user_id)
    await send_qq_text(bot, event, msg)


access_cmd = on_message(rule=is_access_command, priority=4, block=True)


@access_cmd.handle()
async def handle_access_policy(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """管理未来自动代聊/高风险工具的信任名单。"""
    if not should_handle_targeted_event(event, bot):
        return
    if not await require_owner(bot, event, "信任名单管理"):
        return
    if isinstance(event, GroupMessageEvent):
        await send_qq_text(bot, event, "信任名单管理请在私聊中使用，避免暴露联系人和群号。")
        return

    action, target, note = parse_access_payload(get_plain_text(event))
    if action == "add_user":
        if not target:
            await send_qq_text(bot, event, "用法：/白名单 添加用户 <QQ> [备注]")
            return
        await send_qq_text(
            bot,
            event,
            create_confirmation(
                event,
                "access_add_user",
                f"加入信任用户 {target}",
                {"target_id": target, "note": note},
            ),
        )
        return
    if action == "remove_user":
        if not target:
            await send_qq_text(bot, event, "用法：/白名单 删除用户 <QQ>")
            return
        await send_qq_text(
            bot,
            event,
            create_confirmation(
                event,
                "access_remove_user",
                f"移除信任用户 {target}",
                {"target_id": target},
            ),
        )
        return
    if action == "add_group":
        if not target:
            await send_qq_text(bot, event, "用法：/白名单 添加群 <群号> [备注]")
            return
        await send_qq_text(
            bot,
            event,
            create_confirmation(
                event,
                "access_add_group",
                f"加入信任群 {target}",
                {"target_id": target, "note": note},
            ),
        )
        return
    if action == "remove_group":
        if not target:
            await send_qq_text(bot, event, "用法：/白名单 删除群 <群号>")
            return
        await send_qq_text(
            bot,
            event,
            create_confirmation(
                event,
                "access_remove_group",
                f"移除信任群 {target}",
                {"target_id": target},
            ),
        )
        return

    await send_qq_text(
        bot,
        event,
        access_store.summary(include_ids=True)
        + "\n用法：/白名单 添加用户 <QQ> [备注]；/白名单 添加群 <群号> [备注]；/白名单 删除用户 <QQ>；/白名单 删除群 <群号>",
    )


delegate_cmd = on_message(rule=is_delegate_command, priority=4, block=True)


@delegate_cmd.handle()
async def handle_delegate_mode(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """控制信任名单内的 owner-style 自动回复。"""
    if not should_handle_targeted_event(event, bot):
        return
    if not await require_owner(bot, event, "代聊自动回复"):
        return
    if isinstance(event, GroupMessageEvent):
        await send_qq_text(bot, event, "代聊开关请在私聊中使用，避免公开暴露策略。")
        return

    payload = parse_delegate_payload(get_plain_text(event))
    if not payload or payload in {"状态", "status", "查看"}:
        await send_qq_text(
            bot,
            event,
            "\n".join([
                "代聊自动回复状态：",
                f"- 自动回复：{'开' if is_style_auto_reply_enabled() else '关'}",
                f"- 真实原句 few-shot：{'开' if is_style_raw_fewshot_enabled() else '关'}",
                "- 生效范围：信任用户私聊；信任群中仍只响应 @ 或回复机器人",
                "- 关系映射：必须能在最新 Stage 5B QCE 导出中匹配到该用户/群",
                "用法：/代聊 开；/代聊 关；/白名单 添加用户 <QQ>",
            ]),
        )
        return

    if _is_switch_on(payload):
        await send_qq_text(
            bot,
            event,
            create_confirmation(
                event,
                "style_auto_reply_set",
                "开启信任名单内 owner-style 代聊自动回复",
                {"enabled": "true"},
            ),
        )
        return

    if _is_switch_off(payload):
        set_style_auto_reply_enabled(False)
        confirmation_store.log(
            {
                "id": "style_auto_off",
                "type": "style_auto_reply_set",
                "summary": "关闭 owner-style 代聊自动回复",
            },
            actor_id=event.user_id,
            status="executed",
            result="enabled=false",
        )
        await send_qq_text(bot, event, "受控代聊自动回复已关闭。")
        return

    await send_qq_text(bot, event, "用法：/代聊 状态；/代聊 开；/代聊 关")


memory_toggle_cmd = on_message(rule=is_memory_toggle_command, priority=4, block=True)


@memory_toggle_cmd.handle()
async def handle_memory_toggle(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """启用或关闭自动记忆。"""
    if not should_handle_targeted_event(event, bot):
        return
    if not await require_owner(bot, event, "记忆开关"):
        return

    payload = parse_memory_toggle_payload(get_plain_text(event))
    if not payload:
        msg = f"自动记忆当前为：{'开' if is_auto_memory_enabled() else '关'}\n用法：记忆开关 开 / 记忆开关 关"
        await send_qq_text(bot, event, msg)
        return

    if payload in {"开", "开启", "on", "enable", "enabled", "true", "1"}:
        set_auto_memory_enabled(True)
        await send_qq_text(bot, event, "自动记忆已开启。")
        return

    if payload in {"关", "关闭", "off", "disable", "disabled", "false", "0"}:
        set_auto_memory_enabled(False)
        await send_qq_text(bot, event, "自动记忆已关闭。显式“记住：...”仍然可用。")
        return

    await send_qq_text(bot, event, "用法：记忆开关 开 / 记忆开关 关")


time_cmd = on_message(rule=is_time_command, priority=4, block=True)


@time_cmd.handle()
async def handle_time(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """查看本机当前时间。"""
    if not should_handle_targeted_event(event, bot):
        return

    await send_qq_text(bot, event, format_current_time())


calc_cmd = on_message(rule=is_calc_command, priority=4, block=True)


@calc_cmd.handle()
async def handle_calc(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """执行安全数学计算。"""
    if not should_handle_targeted_event(event, bot):
        return

    await send_qq_text(bot, event, safe_calculate(parse_calc_payload(get_plain_text(event))))


todo_cmd = on_message(rule=is_todo_command, priority=4, block=True)


@todo_cmd.handle()
async def handle_todo(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """管理当前用户待办。"""
    if not should_handle_targeted_event(event, bot):
        return

    user_id = str(event.user_id)
    action, payload = parse_todo_command(get_plain_text(event))

    try:
        if action == "add":
            item = todo_store.add(user_id, payload)
            await send_qq_text(bot, event, f"已添加待办：{item['content']} ({item['id']})")
            return

        if action == "done":
            item = todo_store.complete(user_id, payload)
            if item:
                await send_qq_text(bot, event, f"已完成：{item['content']}")
            else:
                await send_qq_text(bot, event, "没有找到对应的待办。")
            return

        items = todo_store.list(user_id)
        await send_qq_text(bot, event, format_todo_list(items))
    except Exception as e:
        await send_qq_text(bot, event, f"待办操作失败：{e}")


memory_query_cmd = on_message(rule=is_memory_query_command, priority=4, block=True)


@memory_query_cmd.handle()
async def handle_memory_query(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """搜索当前用户画像。"""
    if not should_handle_targeted_event(event, bot):
        return

    query = parse_memory_query_payload(get_plain_text(event))
    await ensure_profile_memory_ready()
    profile = await profile_memory.get_user_profile(str(event.user_id))
    await send_qq_text(
        bot,
        event,
        format_profile_search_results(query, search_profile(profile, query)),
    )


teaching_cmd = on_message(rule=is_teaching_command, priority=4, block=True)


@teaching_cmd.handle()
async def handle_teaching_command(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """Owner teaching/review loop for style drafts."""
    if not should_handle_targeted_event(event, bot):
        return
    if not await require_owner(bot, event, "风格教学"):
        return
    if isinstance(event, GroupMessageEvent):
        await send_qq_text(bot, event, "教学审核请在私聊中使用，避免把草稿和反馈公开到群聊。")
        return

    action, payload = parse_teaching_payload(get_plain_text(event))
    actor_id = event.user_id

    if action == "control":
        switch = payload.strip().lower()
        if not switch or switch in {"状态", "status", "查看"}:
            await send_qq_text(
                bot,
                event,
                format_teaching_status(is_style_teaching_enabled(), teaching_store.feedback_stats()),
            )
            return
        if switch in {"最近", "recent", "列表", "list"}:
            await send_qq_text(bot, event, format_recent_reviews(teaching_store.list_recent(actor_id)))
            return
        if switch in {"复盘", "summary", "统计", "stats"}:
            await send_qq_text(
                bot,
                event,
                format_teaching_status(is_style_teaching_enabled(), teaching_store.feedback_stats()),
            )
            return
        if _is_switch_on(switch):
            set_style_teaching_enabled(True)
            confirmation_store.log(
                {
                    "id": "teach_on",
                    "type": "style_teaching_set",
                    "summary": "开启 owner-style 教学影子审核",
                },
                actor_id=actor_id,
                status="executed",
                result="enabled=true",
            )
            await send_qq_text(bot, event, "教学影子审核已开启。信任用户私聊会生成候选并私发给主人，不自动回复对方。")
            return
        if _is_switch_off(switch):
            set_style_teaching_enabled(False)
            confirmation_store.log(
                {
                    "id": "teach_off",
                    "type": "style_teaching_set",
                    "summary": "关闭 owner-style 教学影子审核",
                },
                actor_id=actor_id,
                status="executed",
                result="enabled=false",
            )
            await send_qq_text(bot, event, "教学影子审核已关闭。")
            return
        await send_qq_text(bot, event, "用法：/教学 状态；/教学 开；/教学 关；/教学 最近；/采纳 <id> <1-3>；/改成 <id> <正确回复>")
        return

    review_id, rest = _resolve_review_for_feedback(payload, actor_id)
    if not review_id:
        await send_qq_text(bot, event, "没有可用的教学样本。请先开启 /教学 开，或使用完整格式：/采纳 <id> <1-3>。")
        return

    if action == "accept":
        first, reason = _split_review_payload(rest)
        try:
            selected = int(first)
        except (TypeError, ValueError):
            await send_qq_text(bot, event, "用法：/采纳 <id> <1-3> [原因]；也可对最近样本用 /采纳 1")
            return
        ok, msg, _ = teaching_store.record_feedback(
            review_id,
            actor_id=actor_id,
            action="accept",
            rating=5,
            selected_index=selected,
            reason=reason,
        )
        await send_qq_text(bot, event, msg if ok else f"记录失败：{msg}")
        return

    if action == "rate":
        first, reason = _split_review_payload(rest)
        try:
            rating = int(first)
        except (TypeError, ValueError):
            await send_qq_text(bot, event, "用法：/评分 <id> <1-5> [原因]；也可对最近样本用 /评分 3 太正式")
            return
        ok, msg, _ = teaching_store.record_feedback(
            review_id,
            actor_id=actor_id,
            action="rate",
            rating=rating,
            reason=reason,
        )
        await send_qq_text(bot, event, msg if ok else f"记录失败：{msg}")
        return

    if action == "correct":
        corrected = rest.strip()
        if not corrected:
            await send_qq_text(bot, event, "用法：/改成 <id> <你会怎么回>；也可对最近样本用 /改成 在，咋了")
            return
        ok, msg, _ = teaching_store.record_feedback(
            review_id,
            actor_id=actor_id,
            action="correct",
            rating=5,
            corrected_reply=corrected,
        )
        await send_qq_text(bot, event, msg if ok else f"记录失败：{msg}")
        return

    if action == "reject":
        reason = rest.strip() or "未说明"
        ok, msg, _ = teaching_store.record_feedback(
            review_id,
            actor_id=actor_id,
            action="reject",
            rating=1,
            reason=reason,
        )
        await send_qq_text(bot, event, msg if ok else f"记录失败：{msg}")
        return

    await send_qq_text(bot, event, "用法：/教学 状态；/采纳 <id> <1-3>；/评分 <id> <1-5>；/改成 <id> <正确回复>；/拒绝 <id> <原因>")


style_draft_cmd = on_message(rule=is_style_draft_command, priority=4, block=True)


@style_draft_cmd.handle()
async def handle_style_draft(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """按 owner 风格生成回复草稿。"""
    if not should_handle_targeted_event(event, bot):
        return
    if not await require_owner(bot, event, "风格草稿"):
        return
    if isinstance(event, GroupMessageEvent):
        await send_qq_text(bot, event, "风格草稿请在私聊中使用，避免把草稿公开到群聊。")
        return

    payload = parse_style_draft_payload(get_plain_text(event))
    if not payload:
        await send_qq_text(bot, event, "用法：/用我的风格回复：<对方消息>")
        return

    try:
        draft = await generate_style_draft(
            payload,
            actor_id=event.user_id,
            scope="private",
        )
        draft_text = format_reply(draft)
        review = teaching_store.create_review(
            message=payload,
            candidates=[draft_text],
            chat_type="private",
            target_id=event.user_id,
            trigger="manual_draft",
            reviewer_ids=[str(event.user_id)],
            metadata={"generator": "style_draft"},
        )
        response = (
            f"草稿 #{review.get('id')}：\n"
            f"{draft_text}\n\n"
            f"可反馈：/评分 {review.get('id')} 1-5 原因；/改成 {review.get('id')} 你的正确回复"
        )
        for part in split_qq_msg(response):
            await send_qq_text(bot, event, part)
    except Exception as e:
        write_runtime_error("handle_style_draft", e)
        await send_qq_text(bot, event, f"风格草稿生成失败：{type(e).__name__}")


style_cmd = on_message(rule=is_style_command, priority=4, block=True)


@style_cmd.handle()
async def handle_style_command(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """管理 owner 风格画像。"""
    if not should_handle_targeted_event(event, bot):
        return
    if not await require_owner(bot, event, "风格画像"):
        return
    if isinstance(event, GroupMessageEvent):
        await send_qq_text(bot, event, "风格画像管理请在私聊中使用，避免暴露样本内容。")
        return

    action, payload = parse_style_command(get_plain_text(event))
    if action == "view":
        await send_qq_text(bot, event, format_style_profile(style_store.load()))
        return

    if action == "set":
        parsed = parse_style_set_payload(payload)
        if not parsed:
            await send_qq_text(bot, event, "用法：/风格 设置 语气=自然、简短、像我本人")
            return
        field, value = parsed
        _, msg = style_store.set_field(field, value)
        await send_qq_text(bot, event, msg)
        return

    if action == "import":
        _, msg = style_store.add_example(payload)
        await send_qq_text(bot, event, msg)
        return

    if action == "import_file":
        path_text, owner_markers = parse_style_import_file_payload(payload)
        if not path_text or not owner_markers:
            await send_qq_text(bot, event, "用法：/风格 导入文件 <文件名> 我=<你的昵称或QQ>；文件需放在 data/style_profiles/import_inbox/")
            return
        result = style_store.preview_import_file(path_text, owner_markers)
        await send_qq_text(bot, event, result["message"])
        return

    if action == "confirm_import":
        ok, msg = style_store.confirm_import(payload)
        confirmation_store.log(
            {
                "id": payload.strip()[:16],
                "type": "style_confirm_import",
                "summary": f"确认导入风格记录 {payload.strip()[:24]}",
            },
            actor_id=event.user_id,
            status="executed" if ok else "failed",
            result=msg,
        )
        await send_qq_text(bot, event, msg)
        return

    if action == "distill":
        _, msg = style_store.distill()
        await send_qq_text(bot, event, msg)
        return

    if action == "offline_distill":
        await send_qq_text(
            bot,
            event,
            "开始 Stage 5B 离线蒸馏。只会写入统计摘要和样本索引，不保存聊天正文；数据量大时可能需要较久。",
        )
        try:
            result = await asyncio.to_thread(run_qce_style_distillation, payload or None)
            await send_qq_text(bot, event, format_qce_distillation_result(result))
        except Exception as e:
            write_runtime_error("handle_style_offline_distill", e)
            await send_qq_text(bot, event, f"离线蒸馏失败：{type(e).__name__}")
        return

    if action == "evaluation":
        await send_qq_text(bot, event, format_style_evaluation_report(payload or None))
        return

    if action == "relationships":
        await send_qq_text(bot, event, format_style_relationship_report(payload or None))
        return

    if action == "scenes":
        await send_qq_text(bot, event, format_style_scene_report(payload or None))
        return

    if action == "retrieve":
        try:
            result = await asyncio.to_thread(retrieve_similar_style_samples, payload)
            await send_qq_text(bot, event, format_similar_sample_results(result))
        except Exception as e:
            write_runtime_error("handle_style_retrieve", e)
            await send_qq_text(bot, event, f"相似样本检索失败：{type(e).__name__}")
        return

    if action == "debug":
        if not payload.strip():
            await send_qq_text(bot, event, "用法：/风格 调试 <当前对方消息>")
            return
        try:
            report = await asyncio.to_thread(
                format_style_debug_report,
                payload,
                chat_type="private",
                target_id=event.user_id,
            )
            for part in split_qq_msg(report):
                await send_qq_text(bot, event, part)
        except Exception as e:
            write_runtime_error("handle_style_debug", e)
            await send_qq_text(bot, event, f"风格调试失败：{type(e).__name__}")
        return

    if action == "raw_fewshot":
        switch = parse_style_switch_payload(payload)
        if not switch or switch in {"状态", "status", "查看"}:
            await send_qq_text(
                bot,
                event,
                "\n".join([
                    "真实历史原句 few-shot：",
                    f"- 当前：{'开' if is_style_raw_fewshot_enabled() else '关'}",
                    "- 作用：开启后，风格草稿/代聊可把少量经过脱敏的真实历史上下文和主人回复发给模型作为 few-shot。",
                    "- 审计：每次使用只记录 sample_id/source_id/字数/目标 hash，不记录原文。",
                    "用法：/风格 原句 开；/风格 原句 关",
                ]),
            )
            return
        if _is_switch_on(switch):
            await send_qq_text(
                bot,
                event,
                create_confirmation(
                    event,
                    "style_raw_fewshot_set",
                    "开启真实历史原句 few-shot 进入模型提示词",
                    {"enabled": "true"},
                ),
            )
            return
        if _is_switch_off(switch):
            set_style_raw_fewshot_enabled(False)
            confirmation_store.log(
                {
                    "id": "style_raw_off",
                    "type": "style_raw_fewshot_set",
                    "summary": "关闭真实历史原句 few-shot",
                },
                actor_id=event.user_id,
                status="executed",
                result="enabled=false",
            )
            await send_qq_text(bot, event, "真实历史原句 few-shot 已关闭。")
            return
        await send_qq_text(bot, event, "用法：/风格 原句 状态；/风格 原句 开；/风格 原句 关")
        return

    if action == "auto_reply":
        switch = parse_style_switch_payload(payload)
        if not switch or switch in {"状态", "status", "查看"}:
            await send_qq_text(
                bot,
                event,
                "\n".join([
                    "代聊自动回复状态：",
                    f"- 自动回复：{'开' if is_style_auto_reply_enabled() else '关'}",
                    f"- 真实原句 few-shot：{'开' if is_style_raw_fewshot_enabled() else '关'}",
                    "- 生效范围：信任用户私聊；信任群中仍只响应 @ 或回复机器人",
                    "- 关系映射：必须能在最新 Stage 5B QCE 导出中匹配到该用户/群",
                    "用法：/风格 自动回复 开；/风格 自动回复 关",
                ]),
            )
            return
        if _is_switch_on(switch):
            await send_qq_text(
                bot,
                event,
                create_confirmation(
                    event,
                    "style_auto_reply_set",
                    "开启信任名单内 owner-style 代聊自动回复",
                    {"enabled": "true"},
                ),
            )
            return
        if _is_switch_off(switch):
            set_style_auto_reply_enabled(False)
            confirmation_store.log(
                {
                    "id": "style_auto_off",
                    "type": "style_auto_reply_set",
                    "summary": "关闭 owner-style 代聊自动回复",
                },
                actor_id=event.user_id,
                status="executed",
                result="enabled=false",
            )
            await send_qq_text(bot, event, "受控代聊自动回复已关闭。")
            return
        await send_qq_text(bot, event, "用法：/风格 自动回复 状态；/风格 自动回复 开；/风格 自动回复 关")
        return

    if action == "clear_examples":
        await send_qq_text(
            bot,
            event,
            create_confirmation(
                event,
                "style_clear_examples",
                "清空手动风格样本",
                {},
            ),
        )
        return

    await send_qq_text(bot, event, format_style_help())


help_cmd = on_message(rule=is_help_command, priority=4, block=True)


@help_cmd.handle()
async def handle_help_basic(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """显示基础帮助。"""
    if not should_handle_targeted_event(event, bot):
        return

    msg = "\n".join([
        "可用命令：",
        "- /权限：查看当前权限",
        "- /status 或 状态：主人查看运行状态",
        "- /tools 或 工具：查看工具列表",
        "- /确认 <id> / /取消 <id>：主人执行或取消待确认操作",
        "- /白名单：主人管理未来自动代聊/高风险工具信任名单",
        "- /model：主人查看或切换模型",
        "- /clear：清空当前会话历史；群聊中仅主人可用",
        "- 记住：...：保存你的资料",
        "- 忘记：...：删除你的资料",
        "- 我的资料：查看你的资料",
        "- 记忆开关 开/关：主人控制自动记忆",
        "- 时间：查看当前时间",
        "- 计算：1 + 2 * 3：安全计算",
        "- 待办 添加/列表/完成：管理待办",
        "- 记忆查询 关键词：搜索你的资料",
        "- /风格 查看/导入/导入文件/设置：主人维护风格画像",
        "- /用我的风格回复：...：主人生成风格草稿",
        "- /教学 开/关：主人开启影子审核并用 /采纳、/评分、/改成 记录反馈",
    ])
    await send_qq_text(bot, event, msg)


# ========== 轻量记忆与身份命令 ==========

remember_cmd = on_message(rule=is_remember_command, priority=4, block=True)


@remember_cmd.handle()
async def handle_remember(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """显式保存当前用户资料。"""
    if not should_handle_targeted_event(event, bot):
        return

    payload = parse_remember_payload(get_plain_text(event))
    if not payload:
        await send_qq_text(event=event, bot=bot, text="用法：记住：我喜欢简洁直接的回答")
        return

    predicate, value = infer_user_fact(payload)
    if not value:
        await send_qq_text(event=event, bot=bot, text="没有识别到要记住的内容。")
        return

    await ensure_profile_memory_ready()
    await profile_memory.remember_about_user(
        str(event.user_id),
        predicate,
        value,
        verified=True,
    )
    await send_qq_text(bot, event, f"记住了：{predicate} = {value}")


forget_cmd = on_message(rule=is_forget_command, priority=4, block=True)


@forget_cmd.handle()
async def handle_forget(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """删除当前用户的资料。"""
    if not should_handle_targeted_event(event, bot):
        return

    payload = parse_forget_payload(get_plain_text(event))
    if not payload:
        await send_qq_text(event=event, bot=bot, text="用法：忘记：偏好；或发送“清除我的资料”。")
        return

    await ensure_profile_memory_ready()
    deleted = await profile_memory.forget_about_user(str(event.user_id), payload)
    if deleted:
        await send_qq_text(bot, event, f"已删除 {deleted} 条资料。")
    else:
        await send_qq_text(bot, event, "没有找到匹配的资料。")


profile_cmd = on_message(rule=is_profile_command, priority=4, block=True)


@profile_cmd.handle()
async def handle_profile(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """查看当前用户资料。"""
    if not should_handle_targeted_event(event, bot):
        return

    await ensure_profile_memory_ready()
    profile = await profile_memory.get_user_profile(str(event.user_id))
    await send_qq_text(bot, event, "我记住的你的资料：\n" + format_user_profile(profile))


persona_cmd = on_message(rule=is_persona_command, priority=4, block=True)


@persona_cmd.handle()
async def handle_persona(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """查看机器人身份设定摘要。"""
    if not should_handle_targeted_event(event, bot):
        return

    await send_qq_text(bot, event, summarize_persona())


# ========== 智能体专属命令 ==========

if AGENT_MODE:
    tasks_cmd = on_message(rule=is_tasks_command, priority=4, block=True)

    @tasks_cmd.handle()
    async def handle_tasks(
        bot: nonebot.adapters.onebot.v11.Bot,
        event: MessageEvent,
        state: T_State,
    ):
        """查看任务列表"""
        if not should_handle_targeted_event(event, bot):
            return
        if not await require_owner(bot, event, "任务列表"):
            return

        if agent_engine:
            tasks = await agent_engine.memory.get_pending_tasks()
            if not tasks:
                await bot.send(event, "没有待处理任务")
                return

            msg = "待处理任务:\n"
            for t in tasks:
                status_icon = {"pending": "⏳", "in_progress": "🔄", "blocked": "🚫"}.get(t.get("status"), "•")
                msg += f"{status_icon} [{t['priority']}] {t['title']}\n"

            await bot.send(event, msg)
