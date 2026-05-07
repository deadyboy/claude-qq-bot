---
name: claude-qq-bot-maintainer
description: Maintain and extend the local F:\ClaudeSpace2\claude-qq-bot NoneBot/NapCat QQ bot. Use when Codex is asked to inspect, fix, optimize, add commands/tools/memory/style-profile features, debug NapCat/OneBot connectivity, manage commits, restart the bot, or continue the staged roadmap for this QQ bot project.
---

# Claude QQ Bot Maintainer

## Core Context

Project path: `F:\ClaudeSpace2\claude-qq-bot`.

NapCat path: `F:\迅雷下载\NapCat.Shell`.

The bot is a NoneBot2 + OneBot v11 QQ bot connected to NapCat by reverse WebSocket on local port `8081`. NapCat WebUI is usually at `http://127.0.0.1:6099/webui/`.

Read `references/project-state.md` when a task asks about project history, roadmap, completed stages, memory/tool architecture, or future stage planning.

## Safety Rules

- Never print or commit `.env`, API keys, NapCat tokens, `data/`, logs, or user memory.
- Keep `.env`, `data/`, runtime logs, cache files, and local DB files ignored by Git.
- Do not enable old `AGENT_MODE` unless explicitly requested; it is unfinished and should be refactored before production use.
- Preserve the user's logged-in NapCat/QQ session. Restart the bot process only when possible; avoid restarting NapCat unless necessary.
- In group chats, keep command handlers gated by `should_handle_targeted_event()` so the bot only responds when @ed or replied to.
- For high-risk tools such as shell/file write/network/bulk memory import, add owner checks and explicit confirmation before execution.

## Standard Workflow

1. Check repository state:
   - `git status --short`
   - `git log --oneline -5`
2. Inspect relevant files with `rg` and targeted reads. Prefer:
   - `rg -n "<pattern>" src test_quick.py test_memory.py`
   - `Get-Content -LiteralPath "<file>" -Encoding UTF8`
3. Make focused edits. Prefer existing modules:
   - `src/plugins/claude/dialogue.py` for command wiring.
   - `src/plugins/claude/memory_core.py` for SQLite/user profile/task storage.
   - `src/plugins/claude/memory.py` for simple session history.
   - `src/plugins/claude/persona.py` for bot identity prompt.
   - `src/plugins/claude/auto_memory.py` for automatic user fact extraction.
   - `src/plugins/claude/runtime_state.py` for local runtime switches.
   - `src/plugins/claude/safe_tools.py` for low-risk local tools.
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
   - Confirm `netstat -ano | Select-String -Pattern ':8081|:6099'` shows NapCat connected to `8081`.

## Current Commands

Implemented stable commands include:

- `/status` or `状态`: bot QQ, mode, model, API base, auto-memory state, profile count, latest runtime error header.
- `/tools` or `工具`: current tool list.
- `记忆开关 开/关`: enable or disable automatic fact extraction.
- `记住：...`, `忘记：...`, `我的资料`, `记忆查询 关键词`.
- `时间`.
- `计算：1 + 2 * 3`.
- `待办 添加 ...`, `待办`, `待办 完成 1`.
- `/model`, `/clear`, `/help`.

## Design Direction

Keep the architecture separated:

- `persona`: global bot identity and speaking style.
- `user_profile`: facts about the QQ user currently talking to the bot.
- `session`: short-term private/group conversation history.
- `style_profile`: future system for distilling the owner's speaking style from imported chat logs.
- `safe_tools`: low-risk commands that do not require agentic autonomy.
- Future `Agent Mode`: refactor later into schema-based tools, permissions, confirmations, and logs.

## Subagent Use

Use subagents only when the user explicitly asks for subagents, delegation, or parallel agent work, or when validating a skill with permission.

For larger future stages, split independent read-only investigations:

- Bot startup/config/NapCat connection.
- Memory/session/key_facts schema and storage behavior.
- Command UX and command parsing.
- Permission/privacy risk review.

Keep implementation local to the main agent unless the user explicitly asks for worker subagents to edit code.
