"""Controlled Agent Stage 7/8 commands."""

from ..dialogue import *
from ..controlled_agent import (
    ControlledAgentContext,
    agent_draft_store,
    build_controlled_agent_plan,
    execute_agent_plan,
    execute_controlled_tool,
    format_agent_plan,
    format_controlled_agent_help,
    format_execution_result,
    format_plan_execution_results,
    format_recent_agent_drafts,
    format_tool_catalog,
    get_controlled_tool,
    parse_agent_command,
    serialize_execution_results,
    split_tool_payload,
)


agent_cmd = on_message(rule=is_agent_command, priority=4, block=True)


def _agent_context(event: MessageEvent) -> ControlledAgentContext:
    return ControlledAgentContext(
        actor_id=str(event.user_id),
        session_id=get_session_id(event),
        chat_type="group" if isinstance(event, GroupMessageEvent) else "private",
        is_owner=is_owner_event(event),
    )


def _agent_status_lines() -> list[str]:
    return [
        "受控 Agent 状态：",
        "- Stage 7：受控工具目录、权限、确认、审计已启用",
        "- Stage 8：计划/草稿/审核队列已启用",
        "- 旧 AGENT_MODE：已归档，不参与运行",
        "- 执行策略：主人私聊触发；高风险动作必须 /确认",
    ]


@agent_cmd.handle()
async def handle_controlled_agent(
    bot: nonebot.adapters.onebot.v11.Bot,
    event: MessageEvent,
    state: T_State,
):
    """Owner-only controlled agent entry point."""
    if not should_handle_targeted_event(event, bot):
        return
    if not await require_owner(bot, event, "受控 Agent"):
        return
    if isinstance(event, GroupMessageEvent):
        await send_qq_text(bot, event, "受控 Agent 请在主人私聊中使用，避免把计划、工具结果或审核 ID 暴露到群里。")
        return

    action, payload = parse_agent_command(get_plain_text(event))
    context = _agent_context(event)

    if action in {"", "help"}:
        await send_qq_text(bot, event, format_controlled_agent_help())
        return

    if action == "tools":
        await send_qq_text(bot, event, format_tool_catalog())
        return

    if action == "status":
        await send_qq_text(bot, event, "\n".join(_agent_status_lines()))
        return

    if action == "recent":
        await send_qq_text(
            bot,
            event,
            format_recent_agent_drafts(agent_draft_store.list_recent(event.user_id, limit=8)),
        )
        return

    if action in {"plan", "draft"}:
        if not payload:
            await send_qq_text(bot, event, "用法：/agent 计划 <任务>")
            return
        plan = build_controlled_agent_plan(payload, context)
        draft = agent_draft_store.create(event.user_id, plan)
        confirmation_store.log(
            {"id": draft["id"], "type": "controlled_agent_draft", "summary": f"创建受控 Agent 草稿：{payload[:80]}"},
            actor_id=event.user_id,
            status="draft",
            result=f"steps={len(draft.get('steps') or [])}",
        )
        await send_qq_text(bot, event, format_agent_plan(draft))
        return

    if action == "execute_tool":
        tool_name, tool_payload = split_tool_payload(payload)
        spec = get_controlled_tool(tool_name)
        if not spec:
            await send_qq_text(bot, event, "用法：/agent 执行 <工具> <参数>；发送 /agent 工具 查看工具名。")
            return

        if spec.requires_confirmation:
            await send_qq_text(
                bot,
                event,
                create_confirmation(
                    event,
                    "controlled_agent_tool",
                    f"执行受控工具 {spec.name}",
                    {
                        "tool_name": spec.name,
                        "tool_payload": tool_payload,
                        "session_id": context.session_id,
                        "chat_type": context.chat_type,
                    },
                ),
            )
            return

        await ensure_profile_memory_ready()
        profile = await profile_memory.get_user_profile(str(event.user_id))
        result = await execute_controlled_tool(
            spec.name,
            tool_payload,
            context,
            todo_store=todo_store,
            user_profile=profile,
            status_lines=_agent_status_lines(),
            session_clearer=chat_session_manager.clear_session,
        )
        confirmation_store.log(
            {"id": spec.name, "type": "controlled_agent_tool", "summary": f"执行受控工具 {spec.name}"},
            actor_id=event.user_id,
            status=result.status,
            result=result.output,
        )
        await send_qq_text(bot, event, format_execution_result(result))
        return

    if action == "execute_plan":
        draft_id = payload.strip()
        if not draft_id:
            await send_qq_text(bot, event, "用法：/agent 执行计划 <id>")
            return
        draft = agent_draft_store.get(draft_id, event.user_id)
        if not draft:
            await send_qq_text(bot, event, "没有找到这个受控 Agent 草稿。")
            return

        await ensure_profile_memory_ready()
        profile = await profile_memory.get_user_profile(str(event.user_id))
        results, needs_confirmation = await execute_agent_plan(
            draft,
            context,
            todo_store=todo_store,
            user_profile=profile,
            status_lines=_agent_status_lines(),
            session_clearer=chat_session_manager.clear_session,
        )
        if needs_confirmation:
            await send_qq_text(
                bot,
                event,
                create_confirmation(
                    event,
                    "controlled_agent_plan",
                    f"执行受控 Agent 计划 {draft_id}",
                    {"draft_id": draft_id},
                ),
            )
            return

        agent_draft_store.update_status(
            draft_id,
            event.user_id,
            "executed",
            serialize_execution_results(results),
        )
        confirmation_store.log(
            {"id": draft_id, "type": "controlled_agent_plan", "summary": f"执行受控 Agent 计划 {draft_id}"},
            actor_id=event.user_id,
            status="executed",
            result=serialize_execution_results(results),
        )
        await send_qq_text(bot, event, format_plan_execution_results(draft_id, results))
        return

    if action in {"accept", "reject"}:
        draft_id = payload.strip()
        if not draft_id:
            await send_qq_text(bot, event, f"用法：/agent {'采纳' if action == 'accept' else '拒绝'} <id>")
            return
        status = "accepted" if action == "accept" else "rejected"
        draft = agent_draft_store.update_status(draft_id, event.user_id, status)
        if not draft:
            await send_qq_text(bot, event, "没有找到这个受控 Agent 草稿。")
            return
        confirmation_store.log(
            {"id": draft_id, "type": "controlled_agent_review", "summary": f"标记受控 Agent 草稿 {draft_id} 为 {status}"},
            actor_id=event.user_id,
            status=status,
        )
        await send_qq_text(bot, event, f"已标记草稿 {draft_id}：{status}")
        return

    await send_qq_text(bot, event, format_controlled_agent_help())

