# Legacy Agent Mode

旧 `AGENT_MODE` 已从运行时代码移除。

当前 bot 运行形态是：

- 普通聊天：`dialogue.py` 直接调用 LLM API。
- 会话历史：统一使用 `memory_core.ShortTermMemoryManager`。
- 用户资料、任务和长期记忆：继续由 `memory_core.UnifiedMemoryManager` 管理。
- 安全工具、权限、确认、教学审核、36.skill 风格层：走现有命令模块。

历史 Agent Engine 原型已归档到：

```text
src/plugins/claude/legacy/agent.py
```

归档代码只供参考，不再被 bot 导入，不再注册 `/tasks`，也没有配置开关可以直接启用。

未来如果继续做 Stage 7 受控 Agent，应从现有稳定层重新设计：

- 工具 schema 化。
- 每个工具接入权限等级。
- 高风险动作进入 `/确认`。
- 所有执行写入审计日志。
- 默认只生成草稿或只读结果，不直接执行危险动作。
