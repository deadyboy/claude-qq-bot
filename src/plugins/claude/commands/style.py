"""Owner style profile and draft commands."""

from ..dialogue import *
from ..dialogue import _is_switch_off, _is_switch_on


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
        result = await generate_owner_private_style_draft(
            event,
            payload,
            scope="private_manual_draft",
            record_dialogue=False,
        )
        draft = str(result.get("draft") or "")
        draft_text = format_reply(draft)
        if not draft_text:
            await send_qq_text(bot, event, "风格草稿生成失败：没有可用候选。")
            return
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
            f"{format_style_draft_debug(result)}\n\n"
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
                    "- 作用：开启后，手动草稿和教学候选可把少量经过脱敏的真实历史上下文和主人回复发给模型作为 few-shot。",
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
        parse_style_switch_payload(payload)
        await send_qq_text(bot, event, retired_auto_reply_message())
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
