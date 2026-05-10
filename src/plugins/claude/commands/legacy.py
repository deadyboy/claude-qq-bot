"""Retired and legacy compatibility command handlers."""

from ..dialogue import *


delegate_cmd = on_message(rule=is_delegate_command, priority=4, block=True)


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
