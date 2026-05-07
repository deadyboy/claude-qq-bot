#!/usr/bin/env python3
"""快速测试记忆系统核心功能"""

import sys
import asyncio
import uuid
from pathlib import Path

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
from src.plugins.claude.safe_tools import (
    TodoStore,
    format_tool_list,
    format_todo_list,
    parse_todo_command,
    safe_calculate,
    search_profile,
)
from src.plugins.claude.permissions import (
    format_permission_status,
    is_owner_user_id,
    owner_required_message,
    parse_owner_ids,
)
from src.plugins.claude.style_profile import (
    StyleProfileStore,
    build_style_system_prompt,
    format_style_profile,
    parse_chat_log_text,
    parse_style_command,
    parse_style_draft_payload,
    parse_style_import_file_payload,
    parse_style_set_payload,
)

RUN_ID = uuid.uuid4().hex[:8]

async def test_short_term():
    """测试短期记忆"""
    print("[1/7] 测试短期记忆...")
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
    print("[2/7] 测试关键事实...")
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
    print("[3/7] 测试自动记忆规则...")

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
    print("[4/7] 测试低风险工具...")

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
    print("[5/7] 测试权限辅助函数...")

    owner_ids = parse_owner_ids("123, 456；789  100")
    assert owner_ids == {"123", "456", "789", "100"}
    assert is_owner_user_id("123", owner_ids)
    assert is_owner_user_id(456, owner_ids)
    assert not is_owner_user_id("999", owner_ids)

    status = format_permission_status("999")
    assert "权限状态" in status
    assert "当前身份" in status
    assert "Owner 配置" not in status
    assert "主人权限" in owner_required_message("模型管理") or "仅主人可用" in owner_required_message("模型管理")

    print("      通过")
    return True

async def test_style_profile():
    """测试风格画像本地存储和解析。"""
    print("[6/7] 测试风格画像...")

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
        assert parse_style_command("/风格 清空样本 确认") == ("clear_examples", "确认")
        assert parse_style_draft_payload("用我的风格回复：在不在") == "在不在"
        assert parse_style_import_file_payload("chat.csv 我=36,付健") == ("chat.csv", ["36", "付健"])

        txt_records = parse_chat_log_text(
            "36: 05-08 01:00:11\n在的在的，刚看到\n朋友: 05-08 01:00:12\n你现在忙不忙",
            ".txt",
        )
        assert len(txt_records) == 2
        assert txt_records[0]["sender"] == "36"

        json_records = parse_chat_log_text(
            '[{"role":"owner","text":"我看看"}, {"role":"other","text":"你在哪"}]',
            ".json",
        )
        assert json_records[0]["role"] == "owner"

        inbox = store.import_inbox_dir
        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / "chat.csv").write_text(
            "sender,text\n"
            "36,在的在的，刚看到\n"
            "friend,你现在忙不忙\n"
            "36,我看看，啥事\n"
            "36,我的 api key 是 sk-testsecret123456\n",
            encoding="utf-8",
        )
        preview = store.preview_import_file("chat.csv", ["36"])
        assert preview["ok"], preview["message"]
        assert preview["message_count"] == 2
        assert preview["skipped_sensitive"] == 1

        pending_files = list(store.pending_dir.glob("*.json"))
        assert len(pending_files) == 1
        pending_text = pending_files[0].read_text(encoding="utf-8")
        assert "你现在忙不忙" not in pending_text
        assert "在的在的" not in pending_text

        ok, msg = store.confirm_import(preview["import_id"])
        assert ok, msg
        imported_profile = store.load()
        assert imported_profile["source_imports"]
        assert imported_profile["stats"]["sample_count"] == 2
        assert "你现在忙不忙" not in format_style_profile(imported_profile)

        prompt = build_style_system_prompt(loaded)
        assert "回复草稿生成器" in prompt
        assert "自然、短句、像我本人" in prompt
        assert "在的在的，刚看到，我来处理。" in prompt
        assert "不要直接替主人回答" in prompt
    finally:
        store.delete_for_tests()

    print("      通过")
    return True

async def test_unified():
    """测试统一记忆管理器"""
    print("[7/7] 测试统一记忆管理器...")
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
        results.append(("风格画像", await test_style_profile()))
    except Exception as e:
        print(f"      失败：{e}")
        results.append(("风格画像", False))

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
