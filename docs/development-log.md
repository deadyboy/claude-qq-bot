# claude-qq-bot Development Log

## Current Runtime Shape

- Project path: `F:\ClaudeSpace2\claude-qq-bot`
- NapCat path: `F:\迅雷下载\NapCat.Shell`
- NapCat WebUI: `http://127.0.0.1:6099/webui/`
- Bot reverse WebSocket listener: `127.0.0.1:8081`
- Current Git branch: `master`

Do not commit `.env`, `data/`, logs, caches, or user memory.

## Completed Stages

### Connectivity and Stabilization

- Fixed API failures caused by inherited proxy environment via `httpx.AsyncClient(trust_env=False)`.
- Added runtime model/base URL reconfiguration.
- Added Windows UTF-8 stdout/stderr handling in `bot.py`.
- Fixed group @ detection by using `event.is_tome()`.
- Kept emoji sending enabled, with sanitized fallback only on send failure.

### Stage 1: Persona and Explicit Memory

- Added `config/persona.json`.
- Added `src/plugins/claude/persona.py`.
- Added explicit user-profile commands:
  - `记住：...`
  - `忘记：...`
  - `我的资料`
  - `你是谁` / `身份`
- Injects current user profile into the chat system prompt.

### Stage 2: Automatic User Memory

- Added `src/plugins/claude/auto_memory.py`.
- Extracts stable user facts in the background after normal replies.
- Skips sensitive content such as API keys, tokens, passwords, verification codes, ID cards, and bank cards.
- Stores automatic facts as unverified user profile facts.

### Stage 3: Safe Tools and Status Commands

- Added `src/plugins/claude/runtime_state.py`.
- Added `src/plugins/claude/safe_tools.py`.
- Added commands:
  - `/status` / `状态`
  - `/tools` / `工具`
  - `记忆开关 开/关`
  - `时间`
  - `计算：...`
  - `待办 添加/完成/列表`
  - `记忆查询 关键词`
  - `/help`
- Fixed a `verified_only=True` SQL binding bug.
- Changed `get_pending_tasks()` to return only pending tasks.

## Storage Model

- Short-term sessions: `data/sessions/private_<userQQ>.json` and `data/sessions/group_<groupQQ>.json`
- User profiles and tasks: `data/key_facts.db`
- Long-term memory: `data/longterm_memory/`
- Runtime toggles: `data/runtime_state.json`
- Todos: `data/todos.json`
- Bot persona: `config/persona.json`

There is intentionally no `bot_<botQQ>` namespace yet. Current priority is testing on the small QQ account. Add namespaces before running multiple bot QQ accounts against the same data directory.

## Roadmap

### Stage 4: Style Profile v1

Build a separate `style_profile` system for the owner's speaking style. Do not mix imported chat logs into ordinary user profiles.

Initial commands should be draft-first:

- `/风格 导入`
- `/风格 查看`
- `/用我的风格回复：...`

Store outputs under `data/style_profiles/`. The bot should generate reply drafts, not auto-send as the owner.

### Stage 5: Chat Log Import and Distillation

- Import `.txt`, `.json`, or `.csv` chat logs.
- Distinguish owner messages from other-party messages.
- Extract reply length, tone, emoji habits, common phrasing, and topic behavior.
- Generate editable `style_profile.json`.
- Avoid saving other-party private facts into `user_profile`.

### Stage 6: Permission and Contact Whitelist

- Configure owner QQ IDs.
- Restrict admin commands to owner.
- Add confirmation for high-risk actions.
- Add private/group permission tiers.
- Add optional contact/group whitelist for style draft and future automation modes.

### Stage 7: Agent Mode Refactor

- Replace the old `AGENT_MODE` path with a controlled tool loop.
- Tools should have schemas, permission levels, confirmation rules, and execution logs.
- Default high-risk outputs should be drafts or read-only.

## Codex Skill

A local Codex skill was created for future maintenance:

`C:\Users\lenovo\.codex\skills\claude-qq-bot-maintainer`

Use it for future tasks involving this QQbot's memory, tools, status commands, style profile, migrations, permissions, NapCat connectivity, or staged roadmap.
