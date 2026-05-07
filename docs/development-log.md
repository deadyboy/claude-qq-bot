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

### Stage 6A: Minimal Owner Permissions

- Added `OWNER_QQ_IDS` configuration.
- Added `src/plugins/claude/permissions.py`.
- Added `/权限` / `/owner` style permission status command.
- Restricted management commands to owner:
  - `/status`
  - `/model`
  - `记忆开关 开/关`
  - group `/clear`
  - legacy Agent Mode `/tasks`
- Kept per-user commands available to ordinary users:
  - `记住：...`
  - `忘记：...`
  - `我的资料`
  - `待办`
  - `时间`
  - `计算`
  - `记忆查询`
- Tightened group targeting by removing raw QQ-number substring matching.
- Disabled automatic memory extraction in group chats; group users can still use explicit `记住：...`.

### Stage 4: Style Profile v1

- Added `src/plugins/claude/style_profile.py`.
- Added an independent owner style-profile store under `data/style_profiles/`.
- Added owner-only commands:
  - `/风格 查看`
  - `/风格 设置 语气=...`
  - `/风格 设置 习惯=...`
  - `/风格 导入 <一小段主人回复样本>`
  - `/风格 清空样本 确认`
  - `/用我的风格回复：...`
- Style drafts are explicitly draft-only and are not auto-sent as the owner.
- Style commands are private-chat only in v1 to avoid exposing samples or drafts in groups.
- Style drafts now explicitly avoid inventing the owner's current state, location, availability, or completion status.
- Style samples are not written to `user_profile`, `key_facts.db`, or `config/persona.json`.
- Direct `/风格 导入` remains the small text sample path; bulk `.txt/.json/.csv` import is handled by Stage 5 below.

### Stage 5: Chat Log Import and Local Distillation v1

- Added `.txt`, `.json`, and `.csv` chat-log parsing for owner style import.
- Added private owner-only commands:
  - `/风格 导入文件 <文件名> 我=<你的昵称或QQ>`
  - `/风格 确认导入 <import_id>`
  - `/风格 蒸馏`
- Import files must be placed under `data/style_profiles/import_inbox/`; the bot does not read arbitrary filesystem paths from QQ commands.
- Bulk import is preview-first. The first command returns parsed counts, owner-message counts, skipped-sensitive counts, and an import id.
- Confirmation writes only local distilled style summary fields: reply length, emoji habit, punctuation habit, common phrases, stats, and source metadata.
- Other-party messages are not persisted. Bulk-import owner raw lines are not placed into prompt examples by default.
- Existing direct `/风格 导入 <sample>` remains the opt-in path for storing a small raw owner reply example.

## Storage Model

- Short-term sessions: `data/sessions/private_<userQQ>.json` and `data/sessions/group_<groupQQ>.json`
- User profiles and tasks: `data/key_facts.db`
- Long-term memory: `data/longterm_memory/`
- Runtime toggles: `data/runtime_state.json`
- Todos: `data/todos.json`
- Owner style profile, import inbox, and pending summaries: `data/style_profiles/`
- Bot persona: `config/persona.json`

There is intentionally no `bot_<botQQ>` namespace yet. Current priority is testing on the small QQ account. Add namespaces before running multiple bot QQ accounts against the same data directory.

## Roadmap

### Stage 6: Permission and Contact Whitelist

- Expand beyond the 6A owner-only skeleton.
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
