#!/usr/bin/env python3
"""快速测试记忆系统核心功能"""

import sys
import asyncio
import json
import os
import uuid
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from src.plugins.claude.memory_core import (
    ShortTermMemoryManager,
    KeyFactManager,
    UnifiedMemoryManager
)
from src.plugins.claude.auto_memory import (
    contains_sensitive_content,
    heuristic_extract_facts,
    normalize_extracted_facts,
    should_attempt_auto_memory,
)
from src.plugins.claude.confirmation import (
    ConfirmationStore,
    format_pending_actions,
)
from src.plugins.claude.controlled_agent import (
    ControlledAgentContext,
    ControlledAgentDraftStore,
    build_controlled_agent_plan,
    execute_agent_plan,
    execute_controlled_tool,
    format_agent_plan,
    format_recent_agent_drafts,
    format_tool_catalog,
    parse_agent_command,
    split_tool_payload,
)
from src.plugins.claude.safe_tools import (
    TodoStore,
    format_tool_list,
    format_todo_list,
    parse_todo_command,
    safe_calculate,
    search_profile,
)
from src.plugins.claude.permissions import (
    AccessPolicyStore,
    format_permission_status,
    get_permission_level,
    is_owner_user_id,
    owner_required_message,
    parse_owner_ids,
)
from src.plugins.claude.style_profile import (
    StyleProfileStore,
    build_style_system_prompt,
    clean_style_common_phrases,
    clean_style_habits,
    format_generation_context_for_prompt,
    format_recent_dialogue_for_prompt,
    format_style_draft_debug,
    format_style_profile,
    parse_chat_log_text,
    parse_style_command,
    parse_style_draft_payload,
    parse_style_import_file_payload,
    parse_style_set_payload,
)
from src.plugins.claude.style.distill.embedding import (
    build_embedding_metadata,
    build_embedding_text,
)
from src.plugins.claude.style_distill import (
    build_retrieval_first_prompt,
    build_style_generation_context,
    classify_reply_behavior,
    detect_message_intent,
    find_source_for_target,
    format_style_debug_report,
    format_style_evaluation_report,
    historical_behavior_distribution,
    infer_scene_label,
    retrieve_dialogue_pair_samples,
    retrieve_similar_style_samples,
    run_qce_style_distillation,
    style_rerank_candidates,
)
from src.plugins.claude.style_teaching import (
    TeachingReviewStore,
    format_teaching_batch,
    format_teaching_review_window,
    format_teaching_status,
)
from src.plugins.claude.style_skill import (
    candidate_correction_delta,
    deactivate_correction,
    format_recent_corrections,
    format_style_skill_context_for_prompt,
    load_corrections,
    load_style_skill_context,
    select_relevant_corrections,
)
from src.plugins.claude import runtime_state

RUN_ID = uuid.uuid4().hex[:8]

async def test_short_term():
    """测试短期记忆"""
    print("[1/10] 测试短期记忆...")
    stm = ShortTermMemoryManager(max_messages=10, timeout=7200)

    session_id = f"test_quick_{RUN_ID}"
    try:
        await stm.clear(session_id)
        await stm.add_message(session_id, "user", "你好，我住在北京")
        await stm.add_message(session_id, "assistant", "你好！北京是个好地方")
        await stm.add_message(session_id, "user", "我是一名程序员")

        messages = await stm.get_messages(session_id)
        assert len(messages) == 3, f"期望 3 条，实际{len(messages)}条"

        print(f"      通过 - 共{len(messages)}条消息")
        for msg in messages:
            print(f"        [{msg['role']}] {msg['content']}")
        return True
    finally:
        await stm.clear(session_id)

async def test_key_facts():
    """测试关键事实"""
    print("[2/10] 测试关键事实...")
    kfm = KeyFactManager()
    subject = f"user_test_{RUN_ID}"

    # 添加事实
    fact_id = await kfm.add_fact(
        predicate="occupation",
        object="程序员",
        fact_type="user_profile",
        subject=subject
    )
    print(f"      添加事实 ID: {fact_id}")

    # 添加任务
    task_id = await kfm.add_task(
        title=f"测试任务_{RUN_ID}",
        description="验证任务功能",
        priority=5,
        assigned_to="agent"
    )
    print(f"      添加任务 ID: {task_id}")

    # 查询事实
    facts = await kfm.get_facts(fact_type="user_profile", subject=subject)
    print(f"      查询到{len(facts)}条事实")

    # 查询任务
    tasks = [
        task
        for task in await kfm.get_tasks(assigned_to="agent")
        if task["id"] == task_id
    ]
    print(f"      查询到{len(tasks)}个任务")
    assert len(tasks) == 1, "未查询到刚创建的任务"

    await kfm.update_task_status(task_id, "completed", "测试完成")
    await kfm.close()
    print("      通过")
    return True

async def test_auto_memory_helpers():
    """测试自动记忆抽取的本地规则与过滤。"""
    print("[3/10] 测试自动记忆规则...")

    assert should_attempt_auto_memory("我叫付健，我喜欢简洁直接的回答")
    assert not should_attempt_auto_memory("今天天气怎么样？")
    assert contains_sensitive_content("我的 API key 是 sk-testsecret123456")
    assert not should_attempt_auto_memory("我的 API key 是 sk-testsecret123456")

    facts = heuristic_extract_facts("我叫付健，我喜欢简洁直接的回答")
    print(f"      规则抽取：{facts}")
    assert {"predicate": "称呼", "object": "付健", "confidence": 0.9} in facts
    assert any(f["predicate"] == "偏好" for f in facts)

    normalized = normalize_extracted_facts([
        {"predicate": "偏好", "object": "喜欢详细步骤", "confidence": 0.9},
        {"predicate": "临时", "object": "因为刚才没看到窗口", "confidence": 0.9},
        {"predicate": "密钥", "object": "sk-testsecret123456", "confidence": 0.99},
        {"predicate": "低置信", "object": "可能喜欢 Java", "confidence": 0.4},
    ])
    print(f"      清洗结果：{normalized}")
    assert len(normalized) == 1
    assert normalized[0]["predicate"] == "偏好"

    print("      通过")
    return True

async def test_safe_tools():
    """测试低风险工具。"""
    print("[4/10] 测试低风险工具...")

    assert safe_calculate("1 + 2 * 3") == "1 + 2 * 3 = 7"
    assert safe_calculate("2 ** 11").startswith("计算失败")
    assert safe_calculate("__import__('os')").startswith("计算失败")

    assert parse_todo_command("待办 添加 买牛奶") == ("add", "买牛奶")
    assert parse_todo_command("待办 完成 1") == ("done", "1")
    assert parse_todo_command("待办") == ("list", "")

    todo_path = Path("data") / f"todos_test_{RUN_ID}.json"
    store = TodoStore(todo_path)
    user_id = f"todo_user_{RUN_ID}"
    try:
        item = store.add(user_id, "买牛奶")
        items = store.list(user_id)
        assert len(items) == 1
        assert item["content"] in format_todo_list(items)
        done = store.complete(user_id, "1")
        assert done and done["id"] == item["id"]
        assert store.list(user_id) == []

        profile = {"items": [{"predicate": "偏好", "object": "喜欢详细步骤"}]}
        assert len(search_profile(profile, "详细")) == 1

        normal_tools = format_tool_list(auto_memory_enabled=True, include_owner_tools=False)
        owner_tools = format_tool_list(auto_memory_enabled=True, include_owner_tools=True)
        assert "/model" not in normal_tools
        assert "/model" in owner_tools
        assert "/风格" not in normal_tools
        assert "/风格" in owner_tools
    finally:
        store.clear_user(user_id)
        if todo_path.exists():
            todo_path.unlink()

    print("      通过")
    return True

async def test_permissions():
    """测试 owner 权限辅助函数。"""
    print("[5/10] 测试权限辅助函数...")

    owner_ids = parse_owner_ids("123, 456；789  100")
    assert owner_ids == {"123", "456", "789", "100"}
    assert is_owner_user_id("123", owner_ids)
    assert is_owner_user_id(456, owner_ids)
    assert not is_owner_user_id("999", owner_ids)

    status = format_permission_status("999")
    assert "权限状态" in status
    assert "当前身份" in status
    assert "权限等级" in status
    assert "Owner 配置" not in status
    assert "主人权限" in owner_required_message("模型管理") or "仅主人可用" in owner_required_message("模型管理")

    policy_path = Path("data") / f"permissions_test_{RUN_ID}.json"
    store = AccessPolicyStore(policy_path)
    try:
        ok, msg = store.add_user("123456", note="测试用户", added_by="999999")
        assert ok, msg
        assert store.is_trusted_user("123456")
        assert not store.is_trusted_user("654321")

        ok, msg = store.add_group("987654321", note="测试群", added_by="999999")
        assert ok, msg
        assert store.is_trusted_group("987654321")
        assert "123456" in store.summary(include_ids=True)

        ok, msg = store.remove_user("123456")
        assert ok, msg
        assert not store.is_trusted_user("123456")
        ok, msg = store.add_user("abc")
        assert not ok
    finally:
        if policy_path.exists():
            policy_path.unlink()

    pending_path = Path("data") / f"pending_actions_test_{RUN_ID}.json"
    log_path = Path("data") / f"action_logs_test_{RUN_ID}.jsonl"
    confirm_store = ConfirmationStore(pending_path, log_path, ttl_seconds=60)
    try:
        action = confirm_store.create(
            "access_add_user",
            created_by="999999",
            summary="加入信任用户 123456",
            payload={"target_id": "123456", "note": "测试"},
        )
        assert action["id"] in format_pending_actions(confirm_store.list_for_actor("999999"))
        popped, error = confirm_store.pop_for_actor(action["id"], "999999")
        assert popped and not error
        confirm_store.log(popped, actor_id="999999", status="executed", result="ok")
        assert log_path.exists()

        action = confirm_store.create("style_clear_examples", "999999", "清空手动风格样本", {})
        ok, msg = confirm_store.cancel_for_actor(action["id"], "999999")
        assert ok and "已取消" in msg
        assert confirm_store.list_for_actor("999999") == []
    finally:
        confirm_store.clear_for_tests()

    assert get_permission_level("nope") == "normal"

    state_path = Path("data") / f"runtime_state_test_{RUN_ID}.json"
    old_state_file = runtime_state.STATE_FILE
    runtime_state.STATE_FILE = state_path
    try:
        assert runtime_state.is_style_raw_fewshot_enabled() is False
        runtime_state.set_style_raw_fewshot_enabled(True)
        assert runtime_state.is_style_raw_fewshot_enabled() is True
        runtime_state.set_style_raw_fewshot_enabled(False)
        assert runtime_state.is_style_raw_fewshot_enabled() is False
    finally:
        runtime_state.STATE_FILE = old_state_file
        if state_path.exists():
            state_path.unlink()

    print("      通过")
    return True

async def test_controlled_agent():
    """测试 Stage 7/8 受控 Agent 计划、工具、草稿和确认门。"""
    print("[6/10] 测试受控 Agent...")

    context = ControlledAgentContext(
        actor_id=f"agent_user_{RUN_ID}",
        session_id=f"private_agent_user_{RUN_ID}",
        chat_type="private",
        is_owner=True,
    )
    todo_path = Path("data") / f"agent_todos_test_{RUN_ID}.json"
    draft_path = Path("data") / f"agent_drafts_test_{RUN_ID}.json"
    todo = TodoStore(todo_path)
    drafts = ControlledAgentDraftStore(draft_path)

    try:
        assert parse_agent_command("/agent 工具") == ("tools", "")
        assert parse_agent_command("/agent 计划 计算 1 + 2") == ("plan", "计算 1 + 2")
        assert split_tool_payload("calc 1 + 2") == ("calc", "1 + 2")
        assert "clear_session" in format_tool_catalog()

        plan = build_controlled_agent_plan("计算 1 + 2 * 3", context)
        assert plan["steps"][0]["tool_name"] == "calc"
        assert plan["steps"][0]["requires_confirmation"] is False

        draft = drafts.create(context.actor_id, plan)
        assert draft["id"] in format_agent_plan(draft)
        assert draft["id"] in format_recent_agent_drafts(drafts.list_recent(context.actor_id))

        result = await execute_controlled_tool(
            "calc",
            "1 + 2 * 3",
            context,
            todo_store=todo,
            user_profile={"items": [{"predicate": "偏好", "object": "喜欢详细步骤"}]},
        )
        assert result.ok and " = 7" in result.output

        todo_result = await execute_controlled_tool("todo_add", "写测试", context, todo_store=todo)
        assert todo_result.ok
        list_result = await execute_controlled_tool("todo_list", "", context, todo_store=todo)
        assert "写测试" in list_result.output

        memory_result = await execute_controlled_tool(
            "memory_query",
            "详细",
            context,
            user_profile={"items": [{"predicate": "偏好", "object": "喜欢详细步骤"}]},
        )
        assert memory_result.ok and "详细" in memory_result.output

        high_plan = build_controlled_agent_plan("清空会话", context)
        results, needs_confirmation = await execute_agent_plan(high_plan, context, todo_store=todo)
        assert needs_confirmation is True
        assert results[0].requires_confirmation is True

        denied = await execute_controlled_tool(
            "time",
            "",
            ControlledAgentContext("normal", "private_normal", "private", is_owner=False),
        )
        assert not denied.ok and denied.status == "permission_denied"

        updated = drafts.update_status(draft["id"], context.actor_id, "accepted")
        assert updated and updated["status"] == "accepted"
    finally:
        todo.clear_user(context.actor_id)
        drafts.clear_for_tests()
        if todo_path.exists():
            todo_path.unlink()

    print("      通过")
    return True

async def test_style_profile():
    """测试风格画像本地存储和解析。"""
    print("[7/10] 测试风格画像...")

    store = StyleProfileStore(Path("data") / f"style_profiles_test_{RUN_ID}")
    try:
        profile = store.load()
        assert profile["name"] == "default"
        assert profile["examples"] == []

        ok, msg = store.set_field("语气", "自然、短句、像我本人")
        assert ok, msg
        ok, msg = store.set_field("习惯", "短句；少解释；必要时用一点表情")
        assert ok, msg
        ok, msg = store.set_field("语气", "sk-testsecret123456")
        assert not ok

        ok, msg = store.add_example("在的在的，刚看到，我来处理。")
        assert ok, msg
        ok, msg = store.add_example("我的 api key 是 sk-testsecret123456")
        assert not ok

        loaded = store.load()
        assert loaded["tone"] == "自然、短句、像我本人"
        assert loaded["habits"] == ["短句", "少解释", "必要时用一点表情"]
        assert len(loaded["examples"]) == 1

        formatted = format_style_profile(loaded)
        assert "样本数：1" in formatted
        assert "在的在的" in formatted

        assert parse_style_command("/风格 设置 语气=自然") == ("set", "语气=自然")
        assert parse_style_set_payload("语气=自然") == ("语气", "自然")
        assert parse_style_command("/风格 导入 在的") == ("import", "在的")
        assert parse_style_command("/风格 导入文件 chat.csv 我=36") == ("import_file", "chat.csv 我=36")
        assert parse_style_command("/风格 确认导入 test123") == ("confirm_import", "test123")
        assert parse_style_command("/风格 评估") == ("evaluation", "")
        assert parse_style_command("/风格 关系") == ("relationships", "")
        assert parse_style_command("/风格 场景") == ("scenes", "")
        assert parse_style_command("/风格 检索 样例问题") == ("retrieve", "样例问题")
        assert parse_style_command("/风格 调试 你现在忙吗") == ("debug", "你现在忙吗")
        assert parse_style_command("/风格 原句 开") == ("raw_fewshot", "开")
        assert parse_style_command("/风格 自动回复 开") == ("auto_reply", "开")
        assert parse_style_command("/风格 清空样本 确认") == ("clear_examples", "确认")
        assert parse_style_draft_payload("用我的风格回复：样例问题") == "样例问题"
        assert parse_style_import_file_payload("chat.csv 我=owner,me") == ("chat.csv", ["owner", "me"])
        assert detect_message_intent("你现在忙吗")["reality_state_query"]
        assert detect_message_intent("在不在")["availability_query"]
        assert detect_message_intent("你在哪")["is_question"]
        assert detect_message_intent("这个怎么弄")["help_request"]
        assert detect_message_intent("能不能帮我看下")["task_request"]
        assert detect_message_intent("有无瓦")["game_invitation"]
        assert detect_message_intent("打不打瓦")["game_invitation"]
        assert detect_message_intent("玩不玩瓦")["game_invitation"]
        assert detect_message_intent("有无ai大手子")["is_question"]
        assert not detect_message_intent("有无ai大手子")["invitation"]

        txt_records = parse_chat_log_text(
            "owner: 05-08 01:00:11\n样例回复A\nfriend: 05-08 01:00:12\n样例问题A",
            ".txt",
        )
        assert len(txt_records) == 2
        assert txt_records[0]["sender"] == "owner"

        qq_export_records = parse_chat_log_text(
            "消息记录（此消息记录为文本格式，不支持重新导入）\n"
            "消息分组:最近联系人\n"
            "消息对象:owner\n"
            "2023-11-29 19:25:07 owner\n"
            "样例回复A\n"
            "2023-11-29 19:26:07 friend\n"
            "样例问题A\n",
            ".txt",
        )
        assert len(qq_export_records) == 2
        assert qq_export_records[0]["sender"] == "owner"
        assert qq_export_records[0]["text"] == "样例回复A"

        json_records = parse_chat_log_text(
            '[{"role":"owner","text":"样例回复B"}, {"role":"other","text":"样例问题B"}]',
            ".json",
        )
        assert json_records[0]["role"] == "owner"

        inbox = store.import_inbox_dir
        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / "chat.csv").write_text(
            "sender,text\n"
            "owner,样例回复A\n"
            "friend,样例问题A\n"
            "owner,样例回复B\n"
            "owner,我的 api key 是 sk-testsecret123456\n",
            encoding="utf-8",
        )
        preview = store.preview_import_file("chat.csv", ["owner"])
        assert preview["ok"], preview["message"]
        assert preview["message_count"] == 2
        assert preview["skipped_sensitive"] == 1

        pending_files = list(store.pending_dir.glob("*.json"))
        assert len(pending_files) == 1
        pending_text = pending_files[0].read_text(encoding="utf-8")
        assert "样例问题A" not in pending_text
        assert "样例回复A" not in pending_text

        ok, msg = store.confirm_import(preview["import_id"])
        assert ok, msg
        imported_profile = store.load()
        assert imported_profile["source_imports"]
        assert imported_profile["stats"]["sample_count"] == 2
        assert "样例问题A" not in format_style_profile(imported_profile)

        prompt = build_style_system_prompt(loaded)
        assert "聊天代写器" in prompt
        assert "自然、短句、像我本人" in prompt
        assert "在的在的，刚看到，我来处理。" in prompt
        assert "不要替主人编造具体状态" in prompt
        assert "语用外壳" in prompt
        assert "相似历史样本是主要风格依据" in prompt

        recent_prompt = format_recent_dialogue_for_prompt([
            {"role": "user", "content": "你现在忙？"},
            {"role": "assistant", "content": "刚看到，咋啦"},
        ])
        assert "最近对话" in recent_prompt
        assert "对方：你现在忙？" in recent_prompt
        assert "主人：刚看到，咋啦" in recent_prompt

        assert clean_style_common_phrases(["图片", "QQ", "@某人", "咋了", "逆天"]) == ["咋了", "逆天"]
        cleaned_habits = clean_style_habits([
            "平均回复约 14 字，中位数约 6 字",
            "倾向短句快速回应",
            "从 28299 条本人文本消息蒸馏，只保存统计特征，不保存原文",
        ])
        assert cleaned_habits == ["倾向短句快速回应"]
    finally:
        store.delete_for_tests()

    print("      通过")
    return True

async def test_style_distill():
    """测试 Stage 5B 离线蒸馏不保存聊天正文。"""
    print("[8/10] 测试 Stage 5B 离线蒸馏...")

    root = Path("data") / f"qce_style_distill_test_{RUN_ID}"
    input_dir = root / "input"
    output_root = root / "runs"
    store = StyleProfileStore(Path("data") / f"style_profiles_distill_test_{RUN_ID}")
    try:
        input_dir.mkdir(parents=True, exist_ok=True)
        sample_export = {
            "chatInfo": {
                "name": "private_test",
                "type": "private",
                "selfUin": "1000000001",
                "selfName": "owner",
            },
            "messages": [
                {
                    "id": "m1",
                    "seq": "1",
                    "timestamp": 1700000000,
                    "sender": {"uin": "2000000002", "name": "friend"},
                    "type": "type_1",
                    "content": {"text": "样例问题A", "elements": [{"type": "text"}]},
                    "recalled": False,
                    "system": False,
                },
                {
                    "id": "m2",
                    "seq": "2",
                    "timestamp": 1700000030,
                    "sender": {"uin": "1000000001", "name": "owner"},
                    "type": "type_1",
                    "content": {"text": "样例回复A", "elements": [{"type": "text"}]},
                    "recalled": False,
                    "system": False,
                },
                {
                    "id": "m3",
                    "seq": "3",
                    "timestamp": 1700000060,
                    "sender": {"uin": "2000000002", "name": "friend"},
                    "type": "type_1",
                    "content": {"text": "样例问题B", "elements": [{"type": "text"}]},
                    "recalled": False,
                    "system": False,
                },
                {
                    "id": "m4",
                    "seq": "4",
                    "timestamp": 1700000120,
                    "sender": {"uin": "1000000001", "name": "owner"},
                    "type": "type_1",
                    "content": {"text": "样例回复B", "elements": [{"type": "text"}]},
                    "recalled": False,
                    "system": False,
                },
                {
                    "id": "m5",
                    "seq": "5",
                    "timestamp": 1700000180,
                    "sender": {"uin": "2000000002", "name": "friend"},
                    "type": "type_1",
                    "content": {"text": "这个怎么弄", "elements": [{"type": "text"}]},
                    "recalled": False,
                    "system": False,
                },
                {
                    "id": "m6",
                    "seq": "6",
                    "timestamp": 1700000210,
                    "sender": {"uin": "1000000001", "name": "owner"},
                    "type": "type_1",
                    "content": {"text": "发我看看", "elements": [{"type": "text"}]},
                    "recalled": False,
                    "system": False,
                },
                {
                    "id": "m7",
                    "seq": "7",
                    "timestamp": 1700000240,
                    "sender": {"uin": "1000000001", "name": "owner"},
                    "type": "type_1",
                    "content": {"text": "我的 api key 是 sk-testsecret123456", "elements": [{"type": "text"}]},
                    "recalled": False,
                    "system": False,
                },
            ],
        }
        (input_dir / "friend_recent_001_private_2000000002_test.json").write_text(
            json.dumps(sample_export, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        ai_export = {
            "chatInfo": {
                "name": "ai_bot_test",
                "type": "private",
                "selfUin": "1000000001",
                "selfName": "owner",
            },
            "messages": [
                {
                    "id": "ai1",
                    "seq": "1",
                    "timestamp": 1700000300,
                    "sender": {"uin": "9000000009", "name": "ai_bot"},
                    "type": "type_1",
                    "content": {
                        "text": "你好！我是 36，你的 AI 科研助手 🤖 有什么我可以帮你的吗？",
                        "elements": [{"type": "text"}],
                    },
                    "recalled": False,
                    "system": False,
                },
                {
                    "id": "ai2",
                    "seq": "2",
                    "timestamp": 1700000330,
                    "sender": {"uin": "1000000001", "name": "owner"},
                    "type": "type_1",
                    "content": {"text": "你叫什么", "elements": [{"type": "text"}]},
                    "recalled": False,
                    "system": False,
                },
                {
                    "id": "ai3",
                    "seq": "3",
                    "timestamp": 1700000360,
                    "sender": {"uin": "9000000009", "name": "ai_bot"},
                    "type": "type_1",
                    "content": {
                        "text": "## 方案 1\n让我帮你整理一下 OpenClaw 的基础命令。\n```text\n/new\n```",
                        "elements": [{"type": "text"}],
                    },
                    "recalled": False,
                    "system": False,
                },
                {
                    "id": "ai4",
                    "seq": "4",
                    "timestamp": 1700000390,
                    "sender": {"uin": "1000000001", "name": "owner"},
                    "type": "type_1",
                    "content": {"text": "创建新的子代理对话是什么意思？", "elements": [{"type": "text"}]},
                    "recalled": False,
                    "system": False,
                },
            ],
        }
        (input_dir / "friend_recent_002_private_9000000009_ai_bot.json").write_text(
            json.dumps(ai_export, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        previous_excluded = os.environ.get("QQBOT_STYLE_EXCLUDED_RELATIONSHIP_IDS")
        os.environ["QQBOT_STYLE_EXCLUDED_RELATIONSHIP_IDS"] = "9000000009"
        try:
            result = run_qce_style_distillation(
                input_dir=input_dir,
                output_root=output_root,
                self_uin="1000000001",
                max_index_samples=10,
                apply_to_profile=True,
                store=store,
            )
        finally:
            if previous_excluded is None:
                os.environ.pop("QQBOT_STYLE_EXCLUDED_RELATIONSHIP_IDS", None)
            else:
                os.environ["QQBOT_STYLE_EXCLUDED_RELATIONSHIP_IDS"] = previous_excluded
        assert result["ok"], result
        assert result["excluded_sources"] >= 1
        assert result["owner_text_messages"] == 3
        assert result["turn_count"] >= 6
        assert result["dialogue_pair_count"] >= 3
        assert result["indexed_samples"] >= 1
        assert result["rag_pool_count"] >= 1
        assert result["sft_candidate_count"] >= 1

        output_dir = Path(result["output_dir"])
        summary_text = (output_dir / "style_profile_summary.json").read_text(encoding="utf-8")
        index_text = (output_dir / "sample_index.jsonl").read_text(encoding="utf-8")
        turns_text = (output_dir / "turns.jsonl").read_text(encoding="utf-8")
        pairs_text = (output_dir / "dialogue_pairs.jsonl").read_text(encoding="utf-8")
        phrase_text = (output_dir / "phrase_profile.json").read_text(encoding="utf-8")
        rag_pool_text = (output_dir / "rag_pool.jsonl").read_text(encoding="utf-8")
        sft_text = (output_dir / "sft_candidates.jsonl").read_text(encoding="utf-8")
        rerank_rules_text = (output_dir / "rerank_style_rules.json").read_text(encoding="utf-8")
        relation_text = (output_dir / "relationship_profiles.json").read_text(encoding="utf-8")
        scene_text = (output_dir / "scene_profiles.json").read_text(encoding="utf-8")
        eval_text = (output_dir / "evaluation_report.json").read_text(encoding="utf-8")
        profile_text = store.profile_path().read_text(encoding="utf-8")
        assert "样例回复A" in turns_text
        assert "样例回复A" in pairs_text
        assert "样例回复A" in rag_pool_text
        assert "样例回复A" in sft_text
        assert "AI 科研助手" not in turns_text
        assert "OpenClaw" not in pairs_text
        assert "创建新的子代理" not in sft_text
        assert "hard_filters" in rerank_rules_text
        assert "high_freq_short_replies" in phrase_text
        assert "dialogue_pair_count" in eval_text
        assert "scene_counts" in eval_text
        assert "taxonomy" in eval_text
        for forbidden in ("样例问题A", "样例问题B", "样例回复A", "样例回复B", "这个怎么弄", "发我看看", "api key", "sk-testsecret"):
            assert forbidden not in summary_text
            assert forbidden not in index_text
            assert forbidden not in relation_text
            assert forbidden not in scene_text
            assert forbidden not in eval_text
            assert forbidden not in profile_text

        profile = store.load()
        assert profile["stats"]["source"] == "qce_offline_stage5b"
        assert profile.get("offline_distillations")
        assert "examples" in profile and profile["examples"] == []

        eval_report = format_style_evaluation_report(output_dir)
        assert "Stage 5B 评估摘要" in eval_report
        assert "样例问题A" not in eval_report

        retrieval = retrieve_similar_style_samples("样例问题", run_dir=output_dir)
        assert retrieval["ok"], retrieval
        assert retrieval["result_count"] >= 1
        assert retrieval["retrieval_strategy"] in {"rules_only", "hybrid_rules_embedding"}
        retrieval_text = json.dumps(retrieval, ensure_ascii=False)
        assert "样例问题A" not in retrieval_text
        assert "样例回复A" not in retrieval_text

        first_pair = json.loads(rag_pool_text.splitlines()[0])
        with patch(
            "src.plugins.claude.style.distill.retrieval._query_embedding_index_for_retrieval",
            return_value={
                "ok": True,
                "model": "test-embedding",
                "result_count": 1,
                "results": [{
                    "pair_id": first_pair["pair_id"],
                    "embedding_similarity": 0.88,
                    "distance": 0.12,
                    "metadata": {
                        "pair_id": first_pair["pair_id"],
                        "source_file_id": first_pair["source_file_id"],
                        "chat_type": first_pair["chat_type"],
                        "scene_label": first_pair["scene_label"],
                        "score": first_pair["score"],
                        "length_bucket": first_pair["length_bucket"],
                        "target_char_length": first_pair["taxonomy"]["target_char_length"],
                        "context_turn_count": first_pair["taxonomy"]["context_turn_count"],
                        "scope": first_pair["taxonomy"]["scope"],
                        "grounding_type": first_pair["taxonomy"]["grounding_type"],
                        "learning_value": first_pair["taxonomy"]["learning_value"],
                    },
                }],
            },
        ):
            hybrid_retrieval = retrieve_similar_style_samples(
                "嵌入检索测试",
                run_dir=output_dir,
                limit=3,
                preferred_chat_type="private",
            )
        assert hybrid_retrieval["ok"], hybrid_retrieval
        assert hybrid_retrieval["retrieval_strategy"] == "hybrid_rules_embedding"
        assert hybrid_retrieval["embedding_status"]["ok"]
        assert hybrid_retrieval["results"][0]["embedding_similarity"] == 0.88
        assert hybrid_retrieval["results"][0]["retrieval_source"] in {"embedding", "hybrid"}
        hybrid_text = json.dumps(hybrid_retrieval, ensure_ascii=False)
        assert "样例问题A" not in hybrid_text
        assert "样例回复A" not in hybrid_text

        mapping = find_source_for_target("2000000002", chat_type="private", run_dir=output_dir)
        assert mapping["matched"], mapping
        assert mapping["chat_type"] == "private"

        generation_context = build_style_generation_context("样例问题", run_dir=output_dir)
        assert generation_context["ok"], generation_context
        assert generation_context["similar_samples"]
        context_text = json.dumps(generation_context, ensure_ascii=False)
        prompt_context = format_generation_context_for_prompt(generation_context)
        assert "Stage 5B 生成上下文" in prompt_context
        assert "样例问题A" not in context_text
        assert "样例回复A" not in context_text
        assert "样例问题A" not in prompt_context
        assert "样例回复A" not in prompt_context

        raw_context = build_style_generation_context(
            "样例问题",
            run_dir=output_dir,
            chat_type="private",
            target_id="2000000002",
            include_raw_fewshot=True,
        )
        assert raw_context["ok"], raw_context
        assert raw_context["raw_fewshot_included"]
        raw_text = json.dumps(raw_context, ensure_ascii=False)
        assert "样例问题A" in raw_text
        assert "样例回复A" in raw_text
        assert "sk-testsecret" not in raw_text
        raw_prompt_context = format_generation_context_for_prompt(raw_context)
        assert "真实历史 few-shot 样本" in raw_prompt_context
        assert "样例回复A" in raw_prompt_context

        help_context = build_style_generation_context(
            "这个怎么弄",
            run_dir=output_dir,
            chat_type="private",
            target_id="2000000002",
            include_raw_fewshot=True,
        )
        assert help_context["query_features"]["intent"]["help_request"]
        assert help_context["guidance"]["intent_summary"]["help_request"]
        help_text = json.dumps(help_context, ensure_ascii=False)
        assert "这个怎么弄" in help_text
        assert "发我看看" in help_text

        debug_report = format_style_debug_report(
            "这个怎么弄",
            run_dir=output_dir,
            chat_type="private",
            target_id="2000000002",
        )
        assert "Stage 5B-RAG 风格调试" in debug_report
        assert "这个怎么弄" in debug_report
        assert "发我看看" in debug_report
        assert "sk-testsecret" not in debug_report

        pair_retrieval = retrieve_dialogue_pair_samples(
            "样例问题",
            run_dir=output_dir,
            chat_type="private",
            target_id="2000000002",
        )
        assert pair_retrieval["ok"], pair_retrieval
        assert pair_retrieval["result_count"] >= 1
        pair_retrieval_text = json.dumps(pair_retrieval, ensure_ascii=False)
        assert "样例回复A" in pair_retrieval_text
        retrieval_prompt = build_retrieval_first_prompt("样例问题", retrieval=pair_retrieval)
        assert "相似样本元数据" in retrieval_prompt
        assert "样例回复A" not in retrieval_prompt
        raw_retrieval_prompt = build_retrieval_first_prompt(
            "样例问题",
            retrieval=pair_retrieval,
            include_raw_samples=True,
        )
        assert "相似真实样本" in raw_retrieval_prompt
        assert "样例回复A" in raw_retrieval_prompt
        assert detect_message_intent("有无瓦")["commitment_risk_level"] == 0
        assert detect_message_intent("你现在忙吗")["commitment_risk_level"] == 2
        assert detect_message_intent("把你账号密码发我")["commitment_risk_level"] == 3
        assert infer_scene_label(
            "在不在",
            chat_type="private",
            current_context=[{"role": "other", "content": "在不在"}],
        ) == "private_short_casual"
        assert infer_scene_label("你给我讲下这个大概是什么逻辑", chat_type="private") == "private_long_explain"
        assert infer_scene_label("你能看下这个代码吗", chat_type="private") == "formal_or_worklike"
        ranked = style_rerank_candidates(
            ["您好，请问有什么可以帮您", "[\"行我看下\"]", "这个问题我无法处理", ","],
            scene_label="private_short_casual",
        )
        assert ranked[0]["text"] == "行我看下"
        assert any(item["text"] == "," and not item["accepted"] for item in ranked)
        copied_ranked = style_rerank_candidates(
            ["我先看下这个事情，今天晚上之前不要等我这边一定确定具体结果", "我看看"],
            scene_label="private_short_casual",
            historical_targets=["我先看下这个事情，今天晚上之前不要等我这边一定确定具体结果"],
        )
        assert copied_ranked[0]["text"] == "我看看"
        assert any(item["text"] == "我先看下这个事情，今天晚上之前不要等我这边一定确定具体结果" and not item["accepted"] for item in copied_ranked)
        short_reuse_ranked = style_rerank_candidates(
            ["行我看下", "我看看"],
            scene_label="private_short_casual",
            historical_targets=["行我看下"],
        )
        assert any(item["text"] == "行我看下" and "reused_history_phrase" in item["reasons"] for item in short_reuse_ranked)
        assert all(item["text"] != "行我看下" or int(item.get("hygiene_penalty") or 0) == 0 for item in short_reuse_ranked)
        state_ranked = style_rerank_candidates(
            ["不忙，来", "咋了", "在的"],
            scene_label="private_short_casual",
            latest_message="你现在忙吗",
        )
        assert all(not item["hard_reject"] for item in state_ranked)
        assert not any("unsafe_owner_state" in item["reasons"] for item in state_ranked)
        game_ranked = style_rerank_candidates(
            ["有的呀！想一起开黑吗？", "有无瓦", "可瓦", "打瓦", "暂无", "可以问问c0", "何意", "咋了"],
            scene_label="private_short_casual",
            target_length=6,
            style_profile={"common_phrases": ["何意", "暂无", "可以问问c0"]},
            latest_message="有无瓦",
        )
        assert game_ranked[0]["text"] in {"暂无", "可以问问c0", "何意"}
        assert "style_score" in game_ranked[0]
        assert "risk_penalty" in game_ranked[0]
        assert game_ranked[0]["commitment_risk_level"] == 0
        assert classify_reply_behavior("在打", latest_message="有无瓦")["label"].startswith("micro_")
        assert classify_reply_behavior("在打", latest_message="有无瓦")["safe_for_context"]
        assert classify_reply_behavior("可以问问c0", latest_message="有无瓦")["label"].startswith("short_")
        assert classify_reply_behavior("谁来", latest_message="有无瓦")["label"].startswith("micro_")
        assert classify_reply_behavior("发你", latest_message="把你账号密码发我")["label"] == "credential_share_risk"
        assert any(item["text"] == "有无瓦" and not item["accepted"] for item in game_ranked)
        assert any(item["text"] == "有的呀！想一起开黑吗？" and not item["hard_reject"] for item in game_ranked)
        draft_debug = format_style_draft_debug({
            "selection_reason": "accepted_candidate",
            "call": {"chat_type": "private", "target_id_present": True, "recent_dialogue_count": 0, "include_raw_fewshot": True},
            "ranked_candidates": game_ranked,
        })
        assert "候选决策矩阵" in draft_debug
        assert "style=" in draft_debug and "scene=" in draft_debug and "risk=-" in draft_debug
        behavior_dist = historical_behavior_distribution(["谁来", "谁来", "等会"], latest_message="有无瓦")
        assert behavior_dist["dominant"].startswith("micro_")
        learned_game_ranked = style_rerank_candidates(
            ["等会", "谁来", "暂无"],
            scene_label="private_short_casual",
            target_length=4,
            historical_targets=["谁来", "谁来", "等会"],
            latest_message="有无瓦",
        )
        assert learned_game_ranked[0]["text"] == "谁来"
        learned_debug = format_style_draft_debug({
            "selection_reason": "accepted_candidate",
            "call": {"chat_type": "private", "target_id_present": True, "recent_dialogue_count": 0, "include_raw_fewshot": True},
            "ranked_candidates": learned_game_ranked,
        })
        assert "历史回复形态分布" in learned_debug
        assert "dominant=micro_" in learned_debug
        task_ranked = style_rerank_candidates(
            ["快了", "难说", "我看看"],
            scene_label="private_short_casual",
            latest_message="这个你今天能弄完吗",
        )
        assert all(not item["hard_reject"] for item in task_ranked)
        credential_ranked = style_rerank_candidates(
            ["行，发你", "别乱搞我号", "不行吧"],
            scene_label="private_short_casual",
            latest_message="你直接把你账号发我我登一下",
        )
        assert credential_ranked[0]["text"] != "行，发你"
        assert any(item["text"] == "行，发你" and not item["accepted"] for item in credential_ranked)
        invalid_ranked = style_rerank_candidates([",", "啥"], latest_message="[图片] 这个咋样")
        assert invalid_ranked[0]["text"] == "啥"
        assert any(item["text"] == "," and not item["accepted"] for item in invalid_ranked)
        corrected_ranked = style_rerank_candidates(
            ["我看下", "我先看下，别等我这边确定"],
            scene_label="formal_or_worklike",
            corrections=[{
                "corrected_reply": "我先看下，别等我这边确定",
                "bad_candidates": ["我看下"],
            }],
        )
        assert corrected_ranked[0]["text"] == "我先看下，别等我这边确定"

        low_context = build_style_generation_context(
            "zzzzzzzzzz",
            run_dir=output_dir,
            chat_type="private",
            include_raw_fewshot=True,
        )
        assert low_context["ok"], low_context
        assert not low_context["raw_fewshot_included"]

        prompt = build_style_system_prompt(store.load(), generation_context)
        assert "Stage 5B 生成上下文" in prompt
        assert "相似历史样本索引摘要" in prompt
        assert "关系/来源画像摘要" in prompt
        assert "场景画像摘要" in prompt

        embedding_pair = {
            "pair_id": "pair_embedding_test",
            "source_file_id": "source_embedding_test",
            "relationship_id": "private_2000000002",
            "chat_type": "private",
            "scene_label": "private_short_casual",
            "score": 88,
            "length_bucket": "short",
            "taxonomy": {
                "scope": "global_style",
                "grounding_type": "text_grounded",
                "learning_value": "high",
                "target_char_length": 4,
                "context_turn_count": 1,
            },
            "context": [{"role": "other", "text": "有无瓦"}],
            "target": {"text": "打啊", "char_length": 2},
        }
        embedding_text = build_embedding_text(embedding_pair)
        embedding_metadata = build_embedding_metadata(embedding_pair)
        metadata_text = json.dumps(embedding_metadata, ensure_ascii=False)
        assert "有无瓦" in embedding_text
        assert "打啊" not in embedding_text
        assert "有无瓦" not in metadata_text
        assert "打啊" not in metadata_text
        assert embedding_metadata["pair_id"] == "pair_embedding_test"
        assert embedding_metadata["embedding_text_chars"] == len(embedding_text)
    finally:
        store.delete_for_tests()
        if root.exists():
            import shutil
            shutil.rmtree(root)

    print("      通过")
    return True


async def test_style_teaching():
    """测试风格教学反馈存储"""
    print("[9/10] 测试风格教学反馈...")
    root = Path("data") / f"test_style_teaching_{RUN_ID}"
    skill_root = root / "36_skill"
    store = TeachingReviewStore(
        active_path=root / "teaching_reviews.json",
        feedback_path=root / "teaching_feedback.jsonl",
        corrections_path=skill_root / "corrections.jsonl",
    )
    try:
        (skill_root / "relationship_profiles").mkdir(parents=True, exist_ok=True)
        (skill_root / "global_persona.md").write_text("短句，像真实熟人聊天。", encoding="utf-8")
        (skill_root / "style_rules.md").write_text("禁止 AI 助手腔；不要承诺未知现实状态。", encoding="utf-8")
        (skill_root / "memory_patterns.md").write_text("熟人私聊先顺着上下文接话。", encoding="utf-8")
        (skill_root / "relationship_profiles" / "2000000002.md").write_text(
            "熟人私聊，轻松直接；承诺类问题先保守接住。",
            encoding="utf-8",
        )
        review = store.create_review(
            message="你现在忙不忙",
            candidates=["咋了", "有事？", "我看下", "啥事", "怎么了", "等下", "发我", "我瞅瞅"],
            chat_type="private",
            target_id="2000000002",
            trigger="shadow",
            reviewer_ids=["1000000001"],
            metadata={
                "scene_label": "private_short_casual",
                "style_skill": {"relationship_profile_found": True, "correction_hit_count": 0},
            },
        )
        assert review["id"].startswith("T")
        assert len(review["candidates"]) == 8
        window = format_teaching_review_window(review)
        assert "教学审核" in window
        assert "候选" in window
        assert "36.skill" in window
        assert "咋了" in window

        latest = store.latest_for_reviewer("1000000001")
        assert latest and latest["id"] == review["id"]

        ok, msg, feedback = store.record_feedback(
            review["id"],
            actor_id="1000000001",
            action="accept",
            rating=5,
            selected_index=1,
            reason="最像",
        )
        assert ok, msg
        assert feedback and feedback["selected_candidate"] == "咋了"
        stats = store.feedback_stats()
        assert stats["feedback_count"] == 1
        assert stats["action_counts"]["accept"] == 1
        assert "教学模式" in format_teaching_status(True, stats)

        review2 = store.create_review(
            message="这个今天能弄完吗",
            candidates=["我看下", "等下看看"],
            reviewer_ids=["1000000001"],
        )
        ok, msg, feedback = store.record_feedback(
            review2["id"],
            actor_id="1000000001",
            action="correct",
            rating=5,
            corrected_reply="我先看下，别等我这边确定",
        )
        assert ok, msg
        assert feedback and "别等我" in feedback["corrected_reply"]
        assert feedback.get("correction_id")
        correction_file_text = (skill_root / "corrections.jsonl").read_text(encoding="utf-8")
        assert "这个今天能弄完吗" not in correction_file_text
        assert "recent_dialogue" not in correction_file_text
        corrections = load_corrections(path=skill_root / "corrections.jsonl")
        assert len(corrections) == 1
        assert not corrections[0].get("message")
        assert corrections[0].get("message_hash")
        assert corrections[0].get("message_terms")
        relevant = select_relevant_corrections(
            "这个今天能弄完吗",
            chat_type="private",
            target_id="2000000002",
            path=skill_root / "corrections.jsonl",
        )
        assert relevant
        context = load_style_skill_context(
            chat_type="private",
            target_id="2000000002",
            scene_label="formal_or_worklike",
            latest_message="这个今天能弄完吗",
            root=skill_root,
        )
        assert context["relationship_profile_found"]
        assert context["correction_hit_count"] == 1
        prompt_context = format_style_skill_context_for_prompt(context)
        assert "36.skill" in prompt_context
        assert "我先看下" in prompt_context
        assert "这个今天能弄完吗" not in prompt_context
        delta_good, _ = candidate_correction_delta("我先看下，别等我这边确定", corrections)
        delta_bad, _ = candidate_correction_delta("我看下", corrections)
        assert delta_good > delta_bad
        assert "最近教学纠正" in format_recent_corrections(path=skill_root / "corrections.jsonl")
        ok, msg = deactivate_correction(feedback["correction_id"], actor_id="1000000001", path=skill_root / "corrections.jsonl")
        assert ok, msg
        assert not load_corrections(path=skill_root / "corrections.jsonl")

        run_dir = root / "stage5b_test"
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "sft_candidates.jsonl").open("w", encoding="utf-8") as f:
            f.write(json.dumps({
                "sample_id": "sample_ai",
                "source_file_id": "source_ai",
                "relationship_id": "9000000009",
                "chat_type": "private",
                "scene_label": "formal_or_worklike",
                "scope": "global_style",
                "learning_value": "high",
                "grounding_type": "text_grounded",
                "context": [{"role": "other", "text": "你好！我是 36，你的 AI 科研助手 🤖 有什么我可以帮你的吗？"}],
                "target": "你叫什么",
            }, ensure_ascii=False) + "\n")
            f.write(json.dumps({
                "sample_id": "sample_a",
                "source_file_id": "source_0001",
                "relationship_id": "2000000002",
                "chat_type": "private",
                "scene_label": "private_short_casual",
                "scope": "familiar_private",
                "learning_value": "high",
                "grounding_type": "text_grounded",
                "context": [{"role": "other", "text": "在不在"}],
                "target": "在的",
            }, ensure_ascii=False) + "\n")
        batch = store.create_replay_batch(count=1, reviewer_ids=["1000000001"], run_dir=run_dir)
        assert len(batch) == 1
        assert batch[0]["message"] == "在不在"
        assert batch[0]["candidates"] == ["在的"]
        assert "已创建 1 条教学题" in format_teaching_batch(batch)
        print("      通过")
        return True
    finally:
        store.clear_for_tests()
        if root.exists():
            import shutil
            shutil.rmtree(root)


async def test_unified():
    """测试统一记忆管理器"""
    print("[10/10] 测试统一记忆管理器...")
    um = UnifiedMemoryManager()
    session_id = f"test_unified_{RUN_ID}"
    user_id = f"user_{RUN_ID}"
    task_id = None

    try:
        await um.initialize()
        print("      初始化完成")

        # 添加对话
        await um.add_conversation(session_id, "user", "我喜欢吃川菜")
        await um.add_conversation(session_id, "assistant", "川菜很有特色")
        print("      添加 2 条对话")

        # 记住用户事实
        await um.remember_about_user(user_id, "food_preference", "川菜", verified=True)
        print("      记住 1 条用户事实")

        # 创建任务
        task_id = await um.create_task(
            title=f"推荐餐厅_{RUN_ID}",
            description="查找川菜馆",
            priority=7,
            assigned_to="agent"
        )
        print(f"      创建任务 ID: {task_id}")

        # 搜索记忆
        results = await um.search("川菜", limit=5)
        print(f"      搜索到{len(results)}条相关记忆")

        # 获取用户画像
        profile = await um.get_user_profile(user_id)
        print(f"      用户画像：{profile.get('facts', {})}")
        assert profile["facts"].get("food_preference") == "川菜", "用户画像未写入"

        # 删除用户事实
        deleted = await um.forget_about_user(user_id, "food_preference")
        print(f"      删除用户事实：{deleted}条")
        assert deleted == 1, "用户画像删除失败"

        # 获取待处理任务
        tasks = [
            task
            for task in await um.get_pending_tasks()
            if task["id"] == task_id
        ]
        print(f"      待处理任务：{len(tasks)}个")
        assert len(tasks) == 1, "未查询到刚创建的待处理任务"

        print("      通过")
        return True
    finally:
        if task_id:
            await um.complete_task(task_id, "测试完成")
        await um.short_term.clear(session_id)
        await um.close()

async def main():
    print("=" * 50)
    print("QQ 机器人 - 记忆系统快速测试")
    print("=" * 50)

    results = []

    try:
        results.append(("短期记忆", await test_short_term()))
    except Exception as e:
        print(f"      失败：{e}")
        results.append(("短期记忆", False))

    try:
        results.append(("关键事实", await test_key_facts()))
    except Exception as e:
        print(f"      失败：{e}")
        results.append(("关键事实", False))

    try:
        results.append(("自动记忆规则", await test_auto_memory_helpers()))
    except Exception as e:
        print(f"      失败：{e}")
        results.append(("自动记忆规则", False))

    try:
        results.append(("低风险工具", await test_safe_tools()))
    except Exception as e:
        print(f"      失败：{e}")
        results.append(("低风险工具", False))

    try:
        results.append(("权限辅助", await test_permissions()))
    except Exception as e:
        print(f"      失败：{e}")
        results.append(("权限辅助", False))

    try:
        results.append(("受控 Agent", await test_controlled_agent()))
    except Exception as e:
        print(f"      失败：{e}")
        results.append(("受控 Agent", False))

    try:
        results.append(("风格画像", await test_style_profile()))
    except Exception as e:
        print(f"      失败：{e}")
        results.append(("风格画像", False))

    try:
        results.append(("Stage 5B 离线蒸馏", await test_style_distill()))
    except Exception as e:
        print(f"      失败：{e}")
        results.append(("Stage 5B 离线蒸馏", False))

    try:
        results.append(("风格教学反馈", await test_style_teaching()))
    except Exception as e:
        print(f"      失败：{e}")
        results.append(("风格教学反馈", False))

    try:
        results.append(("统一记忆", await test_unified()))
    except Exception as e:
        print(f"      失败：{e}")
        results.append(("统一记忆", False))

    print()
    print("=" * 50)
    print("结果汇总:")
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print("=" * 50)
    if all_passed:
        print("所有测试通过！[OK]")
    else:
        print("部分测试失败！[ERROR]")
    print()

    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
