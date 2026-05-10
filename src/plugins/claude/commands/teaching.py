"""Owner-style teaching and review commands."""

from ..dialogue import *
from ..dialogue import _is_switch_off, _is_switch_on, _resolve_review_for_feedback, _split_review_payload


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
        raw_switch = payload.strip()
        switch = raw_switch.lower()
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
        if switch in {"纠正", "correction", "corrections"}:
            await send_qq_text(bot, event, format_correction_status())
            return
        if switch in {"纠正 最近", "纠正 recent", "correction recent", "corrections recent"}:
            await send_qq_text(bot, event, format_recent_corrections())
            return
        if switch.startswith("纠正 停用") or switch.startswith("correction disable") or switch.startswith("corrections disable"):
            parts = raw_switch.split(maxsplit=2)
            correction_id = parts[2].strip() if len(parts) >= 3 else ""
            if not correction_id:
                await send_qq_text(bot, event, "用法：/教学 纠正 停用 <id>")
                return
            ok, msg = deactivate_correction(correction_id, actor_id=actor_id)
            await send_qq_text(bot, event, msg if ok else f"停用失败：{msg}")
            return
        if switch.startswith("出题") or switch.startswith("batch"):
            payload_text = switch.removeprefix("出题").removeprefix("batch").strip()
            parts = payload_text.split()
            count = 10
            scene_label = ""
            if parts:
                try:
                    count = int(parts[0])
                    scene_label = parts[1] if len(parts) > 1 else ""
                except ValueError:
                    scene_label = parts[0]
                    if len(parts) > 1:
                        try:
                            count = int(parts[1])
                        except ValueError:
                            count = 10
            try:
                reviews = await asyncio.to_thread(
                    teaching_store.create_replay_batch,
                    count=count,
                    reviewer_ids=[str(actor_id)],
                    scene_label=scene_label,
                )
                await send_qq_text(bot, event, format_teaching_batch(reviews))
            except Exception as e:
                write_runtime_error("handle_teaching_batch", e)
                await send_qq_text(bot, event, f"创建教学题失败：{type(e).__name__}")
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
        await send_qq_text(bot, event, "用法：/教学 状态；/教学 开；/教学 关；/教学 最近；/教学 纠正 最近；/采纳 <id> <1-8>；/改成 <id> <正确回复>")
        return

    review_id, rest = _resolve_review_for_feedback(payload, actor_id)
    if not review_id:
        await send_qq_text(bot, event, "没有可用的教学样本。请先开启 /教学 开，或使用完整格式：/采纳 <id> <1-8>。")
        return

    if action == "accept":
        first, reason = _split_review_payload(rest)
        try:
            selected = int(first)
        except (TypeError, ValueError):
            await send_qq_text(bot, event, "用法：/采纳 <id> <1-8> [原因]；也可对最近样本用 /采纳 1")
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

    await send_qq_text(bot, event, "用法：/教学 状态；/采纳 <id> <1-8>；/评分 <id> <1-5>；/改成 <id> <正确回复>；/拒绝 <id> <原因>")
