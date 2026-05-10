---
name: claude-qq-bot-maintainer
description: Maintain and extend the local F:\ClaudeSpace2\claude-qq-bot NoneBot/NapCat QQ bot. Use when Codex is asked to inspect, fix, optimize, add commands/tools/memory/style-profile features, debug NapCat/OneBot connectivity, manage commits, restart the bot, or continue the staged roadmap for this QQ bot project.
---

# Claude QQ Bot Maintainer

## Core Context

Project path: `F:\ClaudeSpace2\claude-qq-bot`.

NapCat path: `F:\迅雷下载\NapCat.Shell`.

The bot is a NoneBot2 + OneBot v11 QQ bot connected to NapCat by reverse WebSocket on local port `8081`. NapCat WebUI was moved to a high local port; current observed WebUI is `http://127.0.0.1:16099/webui/` and QCE exporter was observed on port `40653`.

Read `references/project-state.md` when a task asks about project history, roadmap, completed stages, memory/tool architecture, or future stage planning.

Full user-facing command documentation is in `docs/command-guide.md`.

## Safety Rules

- Never print or commit `.env`, API keys, NapCat tokens, `data/`, logs, or user memory.
- Keep `.env`, `data/`, runtime logs, cache files, and local DB files ignored by Git.
- Do not revive old `AGENT_MODE`; it has been removed from runtime and archived under `src/plugins/claude/legacy/agent.py`.
- Preserve the user's logged-in NapCat/QQ session. Restart the bot process only when possible; avoid restarting NapCat unless necessary.
- In group chats, keep command handlers gated by `should_handle_targeted_event()` so the bot only responds when @ed or replied to.
- For high-risk tools such as shell/file write/network/bulk memory import, add owner checks and explicit confirmation before execution.
- Owner-only commands currently include `/status`, `/model`, `记忆开关`, group `/clear`, style-profile commands, style-draft commands, and teaching/correction commands.
- Keep user-scoped commands such as `记住：...`, `忘记：...`, `我的资料`, `待办`, `时间`, `计算`, and `记忆查询` available to ordinary users.
- Keep `style_profile` separate from `persona` and `user_profile`; do not write imported chat logs into `key_facts.db` unless a future task explicitly designs that migration.
- Keep style-profile and style-draft commands private-chat only until a later permission/whitelist stage explicitly opens them.
- Bulk style imports must use `data/style_profiles/import_inbox/`, preview-confirm flow, and summary-only persistence; do not persist other-party messages.
- Trust-list commands write only `data/permissions.json`; they scope teaching review and trusted-group generic chat, but do not enable automatic owner-style sending.
- High-risk actions should use `src/plugins/claude/confirmation.py` and log to `data/action_logs.jsonl`.
- Treat deletion as a high-risk action. Before deleting anything, do a read-only review of resolved absolute paths, classify whether the target is protected project/config/data/export/login/credential material or disposable cache/temp output, and avoid deleting protected material unless the user explicitly names the exact path and confirms. Prefer archiving over deletion when the value is uncertain.

## Standard Workflow

1. Check repository state:
   - `git status --short`
   - `git log --oneline -5`
2. Inspect relevant files with `rg` and targeted reads. Prefer:
   - `rg -n "<pattern>" src test_quick.py test_memory.py`
   - `Get-Content -LiteralPath "<file>" -Encoding UTF8`
3. Make focused edits. Prefer existing modules:
   - `src/plugins/claude/dialogue.py` for shared event helpers and command registration.
   - `src/plugins/claude/commands/` for command handlers.
   - `src/plugins/claude/memory_core.py` for session history, SQLite/user profile/task storage.
   - `src/plugins/claude/persona.py` for bot identity prompt.
   - `src/plugins/claude/auto_memory.py` for automatic user fact extraction.
   - `src/plugins/claude/runtime_state.py` for local runtime switches.
   - `src/plugins/claude/safe_tools.py` for low-risk local tools.
   - `src/plugins/claude/style_profile.py` for owner style profiles and draft generation.
   - `src/plugins/claude/permissions.py` for owner checks and trust-list policy.
   - `src/plugins/claude/confirmation.py` for pending confirmations and action logs.
4. Validate before restarting:
   - `python -m compileall -q bot.py src test_quick.py test_memory.py`
   - `python test_quick.py`
   - Run `python test_memory.py` when touching `memory_core.py`, task logic, or SQLite queries.
5. Commit stable changes:
   - `git add <changed files>`
   - `git commit -m "<type>: <concise change>"`
6. Restart only the bot after code changes:
   - Stop the listener on `127.0.0.1:8081`.
   - Start `python -u bot.py` from the project directory in a visible PowerShell window unless the user asks for hidden.
   - Confirm `netstat -ano | Select-String -Pattern ':8081|:16099|:40653'` shows NapCat connected to `8081`.

## Current Commands

Implemented stable commands include:

- `/status` or `状态`: bot QQ, mode, model, API base, auto-memory state, profile count, latest runtime error header.
- `/权限` or `/owner`: current user's owner status.
- `/确认 <id>` and `/取消 <id>`: execute or cancel pending high-risk actions.
- `/白名单`: owner-only trust-list management for teaching review and high-risk tools.
- `/tools` or `工具`: current tool list.
- `记忆开关 开/关`: enable or disable automatic fact extraction.
- `记住：...`, `忘记：...`, `我的资料`, `记忆查询 关键词`.
- `时间`.
- `计算：1 + 2 * 3`.
- `待办 添加 ...`, `待办`, `待办 完成 1`.
- `/风格 查看`, `/风格 设置 ...`, `/风格 导入 ...`, `/风格 导入文件 <文件名> 我=<昵称或QQ>`, `/风格 确认导入 <id>`, `/风格 蒸馏`, `/风格 离线蒸馏`, `/风格 评估`, `/风格 关系`, `/风格 场景`, `/风格 检索 ...`, `/风格 调试 ...`, `/风格 原句 开/关`, `/风格 自动回复`, `/风格 清空样本 确认`.
- `/用我的风格回复：...` generates an owner-style draft only; when Stage 5B outputs exist, it uses no-raw-text retrieval/relationship/scene metadata as generation context.
- `/风格 调试 ...` is owner-only/private-only and may show visible raw historical snippets for debugging. Keep credential-like content skipped and do not persist raw debug snippets to profile/index/log files.
- `/教学 开/关/最近/出题/纠正`: teaching review loop for trusted private users; it sends candidates to the owner only and never auto-sends to contacts.
- `/agent`: owner-only/private-only controlled Agent entry for tool catalog, plans, review drafts, explicit execution, and confirmation-gated high-risk actions.
- `/代聊` and `/风格 自动回复` are retired compatibility commands; they only return a retirement notice and must not create confirmation actions.
- `/model`, `/clear`, `/help`.

## Design Direction

Keep the architecture separated:

- `persona`: global bot identity and speaking style.
- `user_profile`: facts about the QQ user currently talking to the bot.
- `session`: short-term private/group conversation history.
- `style_profile`: owner speaking-style profile and draft generation, stored under `data/style_profiles/`.
- `style_skill`: local 36.skill runtime persona, relationship profiles, memory patterns, and active correction layer.
- `safe_tools`: low-risk commands that do not require agentic autonomy.
- `controlled_agent`: Stage 7/8 command-driven tool plans, review drafts, permission checks, confirmation, and audit logging. Do not revive old `AGENT_MODE` or import `legacy/agent.py` into runtime.

## Subagent Use

Use subagents only when the user explicitly asks for subagents, delegation, or parallel agent work, or when validating a skill with permission.

For larger future stages, split independent read-only investigations:

- Bot startup/config/NapCat connection.
- Memory/session/key_facts schema and storage behavior.
- Command UX and command parsing.
- Permission/privacy risk review.

Keep implementation local to the main agent unless the user explicitly asks for worker subagents to edit code.
