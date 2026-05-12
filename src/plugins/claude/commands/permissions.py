"""Permission, confirmation, and trust-list commands."""

from ..dialogue import *


permission_cmd = on_message(rule=targeted_command_rule(is_permission_command), priority=4, block=True)


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


confirm_cmd = on_message(rule=targeted_command_rule(is_confirm_command), priority=4, block=True)


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
    chat_scope = get_confirmation_scope(event)
    if not action_id:
        await send_qq_text(
            bot,
            event,
            format_pending_actions(confirmation_store.list_for_actor(event.user_id, chat_scope=chat_scope)),
        )
        return

    action, error = confirmation_store.pop_for_actor(action_id, event.user_id, chat_scope=chat_scope)
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


cancel_cmd = on_message(rule=targeted_command_rule(is_cancel_command), priority=4, block=True)


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
    chat_scope = get_confirmation_scope(event)
    if not action_id:
        await send_qq_text(
            bot,
            event,
            format_pending_actions(confirmation_store.list_for_actor(event.user_id, chat_scope=chat_scope)),
        )
        return

    _, msg = confirmation_store.cancel_for_actor(action_id, event.user_id, chat_scope=chat_scope)
    await send_qq_text(bot, event, msg)


access_cmd = on_message(rule=targeted_command_rule(is_access_command), priority=4, block=True)


@access_cmd.handle()
async def handle_access_policy(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """管理教学审核/高风险工具的信任名单。"""
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
