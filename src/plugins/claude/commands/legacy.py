"""Retired and legacy compatibility command handlers."""

from ..dialogue import *


delegate_cmd = on_message(rule=targeted_command_rule(is_delegate_command), priority=4, block=True)


@delegate_cmd.handle()
async def handle_delegate_mode(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """Retired auto-reply command kept as a clear owner-facing notice."""
    if not should_handle_targeted_event(event, bot):
        return
    if not await require_owner(bot, event, "代聊命令"):
        return
    if isinstance(event, GroupMessageEvent):
        await send_qq_text(bot, event, "自动代聊发送已退役；请在私聊使用 /教学 开。")
        return

    await send_qq_text(bot, event, retired_auto_reply_message())
