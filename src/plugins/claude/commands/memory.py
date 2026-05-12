"""Memory, todo, time, and calculation commands."""

from ..dialogue import *


memory_toggle_cmd = on_message(rule=targeted_command_rule(is_memory_toggle_command), priority=4, block=True)


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


time_cmd = on_message(rule=targeted_command_rule(is_time_command), priority=4, block=True)


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


calc_cmd = on_message(rule=targeted_command_rule(is_calc_command), priority=4, block=True)


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


todo_cmd = on_message(rule=targeted_command_rule(is_todo_command), priority=4, block=True)


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


memory_query_cmd = on_message(rule=targeted_command_rule(is_memory_query_command), priority=4, block=True)


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


# ========== 轻量记忆与身份命令 ==========

remember_cmd = on_message(rule=targeted_command_rule(is_remember_command), priority=4, block=True)


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


forget_cmd = on_message(rule=targeted_command_rule(is_forget_command), priority=4, block=True)


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


profile_cmd = on_message(rule=targeted_command_rule(is_profile_command), priority=4, block=True)


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
