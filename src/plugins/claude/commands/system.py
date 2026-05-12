"""System, status, model, help, and persona commands."""

from ..dialogue import *


clear_cmd = on_message(rule=targeted_command_rule(is_clear_command), priority=4, block=True)


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

    await chat_session_manager.clear_session(session_id)

    await send_qq_text(bot, event, "会话历史已清空")


model_cmd = on_message(rule=targeted_command_rule(is_model_command), priority=4, block=True)


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
        vision = model_config.get_current_vision_model()
        api_base = model_config.get_current_api_base()
        vision_api_base = model_config.get_current_vision_api_base()
        available = ", ".join(model_config.list_models())
        notes = model_config.get_compatibility_notes()
        notes_text = "\n".join(notes) + "\n" if notes else ""
        msg = (
            f"当前模型：{current}\n"
            f"图片模型：{vision}\n"
            f"API Base：{api_base}（{model_config.get_api_provider()}）\n"
            f"API Key：{model_config.get_api_key_state()}\n"
            f"图片 API Base：{vision_api_base}（{model_config.get_vision_api_provider()}）\n"
            f"图片 API Key：{model_config.get_vision_api_key_state()}\n"
            f"可用模型：{available}\n"
            f"{notes_text}用法：/model <模型名>"
        )
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

status_cmd = on_message(rule=targeted_command_rule(is_status_command), priority=4, block=True)


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
        "- 模式：普通聊天 + 记忆 + 教学审核",
        f"- 模型：{model_config.get_current_model()}",
        f"- 图片模型：{model_config.get_current_vision_model()}",
        f"- API Base：{model_config.get_current_api_base()}（{model_config.get_api_provider()}）",
        f"- API Key：{model_config.get_api_key_state()}",
        f"- 图片 API Base：{model_config.get_current_vision_api_base()}（{model_config.get_vision_api_provider()}）",
        f"- 图片 API Key：{model_config.get_vision_api_key_state()}",
        f"- 自动记忆：{'开' if is_auto_memory_enabled() else '关'}",
        "- 自动代聊发送：已退役",
        f"- 教学影子审核：{'开' if is_style_teaching_enabled() else '关'}",
        f"- 真实原句 few-shot：{'开' if is_style_raw_fewshot_enabled() else '关'}",
        "- 受控 Agent：已启用（主人私聊 /agent）",
        f"- 你的资料：{len(profile.get('items') or [])} 条",
        f"- 最近错误：{get_latest_error_header()}",
    ])
    await send_qq_text(bot, event, msg)


tools_cmd = on_message(rule=targeted_command_rule(is_tools_command), priority=4, block=True)


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

help_cmd = on_message(rule=targeted_command_rule(is_help_command), priority=4, block=True)


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
        "- /白名单：主人管理教学审核/高风险工具信任名单",
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
        "- /agent：主人私聊使用受控工具、计划草稿和审核执行",
    ])
    await send_qq_text(bot, event, msg)

persona_cmd = on_message(rule=targeted_command_rule(is_persona_command), priority=4, block=True)


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
