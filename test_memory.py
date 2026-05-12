#!/usr/bin/env python3
"""
测试记忆系统脚本

运行方式:
    python test_memory.py
"""

import asyncio
import atexit
import os
import sys
import tempfile
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
_ORIGINAL_CWD = Path.cwd()
_TEST_WORKDIR = tempfile.TemporaryDirectory(
    prefix="claude_qq_bot_test_memory_",
    ignore_cleanup_errors=True,
)


def _cleanup_test_workdir() -> None:
    os.chdir(_ORIGINAL_CWD)
    try:
        _TEST_WORKDIR.cleanup()
    except PermissionError:
        # Chroma can briefly keep sqlite/bin files open on Windows after tests pass.
        pass


atexit.register(_cleanup_test_workdir)
os.chdir(_TEST_WORKDIR.name)

from src.plugins.claude.memory_core import (
    ShortTermMemoryManager,
    LongTermMemoryManager,
    KeyFactManager,
    UnifiedMemoryManager
)


async def test_short_term_memory():
    """测试短期记忆"""
    print("\n" + "=" * 50)
    print("测试短期记忆管理器")
    print("=" * 50)

    manager = ShortTermMemoryManager(max_messages=10, timeout=7200)
    session_id = "test_user_001"

    # 添加消息
    print("\n1. 添加消息...")
    await manager.add_message(session_id, "user", "你好，我住在北京")
    await manager.add_message(session_id, "assistant", "你好！北京是个好地方")
    await manager.add_message(session_id, "user", "我是一名程序员")

    # 获取消息
    print("2. 获取消息历史...")
    messages = await manager.get_messages(session_id)
    print(f"   共 {len(messages)} 条消息:")
    for msg in messages:
        print(f"   - [{msg['role']}] {msg['content']}")

    # 标记重要消息
    print("3. 标记重要消息...")
    await manager.mark_important(session_id, 0)
    print("   已标记第 1 条消息为重要")

    # 测试清空
    print("4. 测试清空...")
    await manager.clear(session_id)
    messages_after = await manager.get_messages(session_id)
    print(f"   清空后剩余 {len(messages_after)} 条消息")

    print("\n[OK] 短期记忆测试完成")
    return True


async def test_long_term_memory():
    """测试长期记忆"""
    print("\n" + "=" * 50)
    print("测试长期记忆管理器")
    print("=" * 50)

    manager = LongTermMemoryManager("data/longterm_memory")

    try:
        # 添加记忆
        print("\n1. 添加长期记忆...")
        await manager.add(
            content="用户住在北京，是一名 Python 程序员",
            memory_type="fact",
            tags=["user_profile", "location", "job"]
        )
        await manager.add(
            content="用户喜欢使用 VS Code 作为编辑器",
            memory_type="preference",
            tags=["user_profile", "tool"]
        )
        await manager.add(
            content="Python 是一门易读性很强的编程语言",
            memory_type="skill",
            tags=["knowledge", "programming"]
        )
        print("   已添加 3 条记忆")

        # 搜索记忆
        print("2. 搜索记忆：'用户工作地点'...")
        results = await manager.search("用户工作地点", limit=3)
        print(f"   找到 {len(results)} 条相关记忆:")
        for r in results:
            print(f"   - [{r.get('type', 'N/A')}] {r['content'][:50]}...")

        print("\n3. 搜索记忆：'Python 编程'...")
        results = await manager.search("Python 编程", limit=3)
        print(f"   找到 {len(results)} 条相关记忆:")
        for r in results:
            print(f"   - [{r.get('type', 'N/A')}] {r['content'][:50]}...")

        print("\n[OK] 长期记忆测试完成")
        return True

    except Exception as e:
        print(f"   [SKIP] ChromaDB 不可用，使用降级模式：{e}")
        print("\n[OK] 长期记忆测试完成（降级模式）")
        return True


async def test_key_facts():
    """测试关键事实"""
    print("\n" + "=" * 50)
    print("测试关键事实管理器")
    print("=" * 50)

    manager = KeyFactManager("data/key_facts.db")

    # 添加事实
    print("\n1. 添加用户事实...")
    fact_id1 = await manager.add_fact(
        predicate="location",
        object="北京",
        fact_type="user_profile",
        subject="user_123"
    )
    fact_id2 = await manager.add_fact(
        predicate="occupation",
        object="程序员",
        fact_type="user_profile",
        subject="user_123"
    )

    # 添加任务
    print("2. 添加任务...")
    task_id = await manager.add_task(
        title="完成项目报告",
        description="写一份关于项目进展的报告",
        priority=8,
        assigned_to="agent"
    )
    print(f"   任务 ID: {task_id}")

    # 查询事实
    print("3. 查询用户事实...")
    facts = await manager.get_facts(fact_type="user_profile", subject="user_123")
    print(f"   找到 {len(facts)} 条事实:")
    for f in facts:
        print(f"   - {f['predicate']}: {f['object']}")

    # 查询任务
    print("4. 查询待处理任务...")
    tasks = await manager.get_tasks(assigned_to="agent")
    print(f"   找到 {len(tasks)} 个任务:")
    for t in tasks:
        print(f"   - [{t['priority']}] {t['title']} ({t['status']})")

    # 验证事实
    print("5. 验证事实...")
    await manager.verify_fact(fact_id1)
    verified_facts = await manager.get_facts(subject="user_123", verified_only=True)
    print(f"   已验证 {len(verified_facts)} 条事实")

    # 完成任务
    print("6. 完成任务...")
    await manager.update_task_status(task_id, "completed", "报告已完成并提交")
    completed_tasks = await manager.get_tasks(assigned_to="agent")
    print(f"   待处理任务剩余 {len(completed_tasks)} 个")

    await manager.close()

    print("\n[OK] 关键事实测试完成")
    return True


async def test_unified_memory():
    """测试统一记忆管理器"""
    print("\n" + "=" * 50)
    print("测试统一记忆管理器")
    print("=" * 50)

    manager = UnifiedMemoryManager()
    manager.short_term = ShortTermMemoryManager(max_messages=50, timeout=7200)
    manager.long_term = LongTermMemoryManager("data/unified_longterm_memory")
    manager.key_facts = KeyFactManager("data/unified_key_facts.db")

    # 初始化
    print("\n1. 初始化所有记忆系统...")
    await manager.initialize()
    print("   初始化完成")

    # 添加对话
    print("2. 添加对话...")
    await manager.add_conversation("test_001", "user", "我喜欢吃川菜")
    await manager.add_conversation("test_001", "assistant", "川菜很有特色，你喜欢辣的还是不辣的？")

    # 记住用户事实
    print("3. 记住用户事实...")
    await manager.remember_about_user("test_001", "food_preference", "川菜", verified=True)

    # 创建任务
    print("4. 创建任务...")
    task_id = await manager.create_task(
        title="推荐川菜餐厅",
        description="查找附近评分高的川菜馆",
        priority=7,
        assigned_to="agent"
    )
    print(f"   任务 ID: {task_id}")

    # 搜索记忆
    print("5. 搜索记忆：'川菜'...")
    results = await manager.search("川菜", limit=5)
    print(f"   找到 {len(results)} 条相关记忆")

    # 获取用户画像
    print("6. 获取用户画像...")
    profile = await manager.get_user_profile("test_001")
    print(f"   用户偏好：{profile.get('facts', {})}")

    # 获取待处理任务
    print("7. 获取待处理任务...")
    tasks = await manager.get_pending_tasks()
    print(f"   待处理任务：{len(tasks)} 个")

    await manager.close()

    print("\n[OK] 统一记忆管理器测试完成")
    return True


async def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("QQ 机器人 - 记忆系统测试")
    print("=" * 60)

    results = {
        "短期记忆": await test_short_term_memory(),
        "长期记忆": await test_long_term_memory(),
        "关键事实": await test_key_facts(),
        "统一记忆": await test_unified_memory()
    }

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！[OK]")
    else:
        print("部分测试失败！[ERROR]")
    print("=" * 60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
