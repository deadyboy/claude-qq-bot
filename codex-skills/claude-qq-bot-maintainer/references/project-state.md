# claude-qq-bot Project State

## Repository

- Project: `F:\ClaudeSpace2\claude-qq-bot`
- Git branch: `master`
- Recent baseline commits:
  - `5adf6e0 feat: initialize qq bot with persona memory`
  - `76bcfa8 feat: add automatic user memory extraction`
  - `58c921e feat: add safe tools and bot status commands`
  - `aaada5d feat: add owner permission checks`

## Runtime

- NapCat WebUI: `http://127.0.0.1:6099/webui/`
- Bot reverse WebSocket listener: `127.0.0.1:8081`
- NapCat config should point reverse WebSocket to `ws://127.0.0.1:8081/onebot/v11/ws`.
- Prefer restarting only the bot listener, not NapCat, to preserve the QQ login session.

## Completed Work

### Connectivity and API

- Fixed API connection failures caused by inherited proxy environment by using `httpx.AsyncClient(trust_env=False)` in `src/plugins/claude/api.py`.
- Added runtime API reconfiguration for model/base URL changes.
- Added UTF-8 stdout/stderr handling in `bot.py` for Windows logging and emoji safety.

### Group Chat

- Fixed group @ detection by using `event.is_tome()` in `is_to_bot()`.
- Pure @ in group defaults to `"在不在"` so it triggers a normal reply.

### Persona and Explicit Memory

- Added `config/persona.json`.
- Added `src/plugins/claude/persona.py`.
- Added explicit commands:
  - `记住：...`
  - `忘记：...`
  - `我的资料`
  - `你是谁` / `身份`
- Replies inject current user profile into the system prompt.

### Automatic User Memory

- Added `src/plugins/claude/auto_memory.py`.
- Automatically extracts stable user facts after a normal reply, without blocking the current reply.
- Sensitive content such as API keys, tokens, passwords, verification codes, ID cards, and bank cards is skipped.
- Automatic memories are stored as unverified user_profile facts; explicit `记住：...` facts are verified.

### Stage 3: Safe Tools and Status

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
- Fixed `verified_only=True` SQL binding bug in `memory_core.py`.
- Changed `get_pending_tasks()` to return only pending tasks.

### Stage 6A: Minimal Owner Permissions

- Added `OWNER_QQ_IDS` configuration.
- Added `src/plugins/claude/permissions.py`.
- Added `/权限` / `/owner` permission status command.
- Restricted owner-only management commands:
  - `/status`
  - `/model`
  - `记忆开关 开/关`
  - group `/clear`
  - legacy Agent Mode `/tasks`
- Kept user-scoped commands available to ordinary users.
- Removed raw QQ-number substring matching from group targeting.
- Disabled automatic memory extraction in group chats; explicit `记住：...` remains available.

### Stage 4: Style Profile v1

- Added `src/plugins/claude/style_profile.py`.
- Stores owner style data separately under `data/style_profiles/`.
- Added owner-only style commands:
  - `/风格 查看`
  - `/风格 设置 语气=...`
  - `/风格 设置 习惯=...`
  - `/风格 导入 <sample>`
  - `/风格 清空样本 确认`
  - `/用我的风格回复：...`
- Style drafts are draft-only and do not auto-send as the owner.
- Style commands are private-chat only in v1 to avoid exposing samples or drafts in groups.
- Style samples do not go into `user_profile`, `key_facts.db`, or `config/persona.json`.
- v1 supports small direct text samples only; bulk file import and distillation are still future work.

## Current Storage Model

- Short-term sessions: `data/sessions/private_<userQQ>.json` or `data/sessions/group_<groupQQ>.json`.
- User profiles and tasks: `data/key_facts.db`.
- Long-term memory: `data/longterm_memory/`.
- Runtime toggles: `data/runtime_state.json`.
- Todos: `data/todos.json`.
- Owner style profile and samples: `data/style_profiles/`.
- Bot persona: `config/persona.json`.

There is currently no `bot_<botQQ>` namespace. This is intentional for the current small-account experiment. Add bot namespaces later before running multiple bot QQ accounts against one data directory.

## Planned Stages

### Stage 5: Chat Log Import and Distillation

- Import `.txt`, `.json`, or `.csv` chat logs.
- Distinguish owner messages from other-party messages.
- Extract reply length, tone, emoji habits, common phrasing, topic behavior.
- Generate editable `style_profile.json`.
- Avoid saving other-party private facts into user_profile unless explicitly requested.

### Stage 6: Permission and Contact Whitelist

- Expand beyond the 6A owner-only skeleton.
- Add confirmation for high-risk actions.
- Add private/group permission tiers.
- Add optional contact/group whitelist for style draft and future automation modes.

### Stage 7: Agent Mode Refactor

- Replace the old `AGENT_MODE` path with a controlled tool loop.
- Tools should have schemas, permission levels, confirmation rules, and execution logs.
- Default high-risk outputs should be drafts or read-only.
