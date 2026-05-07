"""QQ 机器人对话处理 - 支持智能体模式

模式切换:
- 简单模式：直接调用 LLM API (向后兼容)
- 智能体模式：使用 Agent Engine，支持记忆/工具/任务
"""

import asyncio
import traceback
from pathlib import Path
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
from .permissions import (
    format_permission_status,
    is_owner_event,
    owner_required_message,
)
from .runtime_state import is_auto_memory_enabled, set_auto_memory_enabled
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
    parse_style_set_payload,
    style_store,
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


def extract_text_and_images(event: MessageEvent) -> tuple[str, list[str]]:
    """从消息中提取文本和图片 base64"""
    text_parts = []
    images = []

    for segment in event.message:
        if segment.type == "text":
            text_parts.append(str(segment).strip())
        elif segment.type == "image":
            images.append(segment.data.get("url", ""))

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


def write_runtime_error(scope: str, error: Exception):
    """写入运行期异常，便于独立窗口运行时排查。"""
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "runtime_errors.log").open("a", encoding="utf-8") as f:
        f.write(f"\n--- {scope}: {type(error).__name__}: {error} ---\n")
        f.write(traceback.format_exc())


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

    await send_qq_text(bot, event, format_permission_status(event.user_id))


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
        draft = await generate_style_draft(payload)
        response = "草稿：\n" + format_reply(draft)
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

    if action == "clear_examples":
        if payload.strip().lower() not in {"确认", "confirm"}:
            await send_qq_text(bot, event, "清空风格样本需要确认：/风格 清空样本 确认")
            return
        await send_qq_text(bot, event, style_store.clear_examples())
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
        "- /风格 查看/导入/设置：主人维护风格画像",
        "- /用我的风格回复：...：主人生成风格草稿",
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
