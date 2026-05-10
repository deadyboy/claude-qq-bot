# claude-qq-bot Development Log

## Current Runtime Shape

- Project path: `F:\ClaudeSpace2\claude-qq-bot`
- Stage 6B worktree: `F:\ClaudeSpace2\claude-qq-bot-stage6b` on branch `stage6b`
- NapCat path: `F:\迅雷下载\NapCat.Shell`
- NapCat WebUI: `http://127.0.0.1:6099/webui/`
- Bot reverse WebSocket listener: `127.0.0.1:8081`
- Current Git branch: `master`

Do not commit `.env`, `data/`, logs, caches, or user memory.

## Runtime Repair Notes

### 2026-05-08 NapCat Startup and Git Worktree Repair

- NapCat launcher scripts in `F:\迅雷下载\NapCat.Shell` now change to their own directory before generating `loadNapCat.js`, preventing wrong loader paths when launched from another current directory.
- `loadNapCat.js` was restored to `file:///F:/迅雷下载/NapCat.Shell/napcat.mjs`.
- NapCat global OneBot account was aligned to bot QQ `2920249374`; previous global account value pointed at owner QQ `1030400950`.
- NapCat file logging was enabled for future startup diagnostics.
- `git worktree add` failed under normal Codex permissions because `.git` writes were denied; using elevated git created `F:\ClaudeSpace2\claude-qq-bot-stage6b` on branch `stage6b`.
- Startup order remains: start bot listener on `127.0.0.1:8081`, then start NapCat/QQ.

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

### Stage 6: Permission and Whitelist Base v1

- Expanded `src/plugins/claude/permissions.py` with a local `data/permissions.json` access policy store.
- Added owner-only, private-chat-only trust-list management:
  - `/白名单`
  - `/白名单 添加用户 <QQ> [备注]`
  - `/白名单 删除用户 <QQ>`
  - `/白名单 添加群 <群号> [备注]`
  - `/白名单 删除群 <群号>`
- `/权限` now shows whether the current user is in the trusted-user list.
- The trust list scopes teaching review and high-risk tools. Old owner-style auto-send has been retired; ordinary private chats do not use the trust list to auto-send owner-style replies.

### Stage 6B: Confirmation and Audit Base

- Added `src/plugins/claude/confirmation.py`.
- Added owner-only confirmation commands:
  - `/确认`
  - `/确认 <id>`
  - `/取消`
  - `/取消 <id>`
- Pending actions are stored under `data/pending_actions.json` and expire after 10 minutes.
- Action audit logs are appended to `data/action_logs.jsonl`.
- High-risk actions now use pending confirmation:
  - group `/clear`
  - `/白名单 添加用户/删除用户/添加群/删除群`
  - `/风格 清空样本`
- `/权限` now includes a permission level: `owner`, `trusted_user`, `trusted_group`, or `normal`.

## Storage Model

- Short-term sessions: `data/sessions/private_<userQQ>.json` and `data/sessions/group_<groupQQ>.json`
- User profiles and tasks: `data/key_facts.db`
- Long-term memory: `data/longterm_memory/`
- Runtime toggles: `data/runtime_state.json`
- Todos: `data/todos.json`
- Permission and trust-list policy: `data/permissions.json`
- Pending confirmations and audit logs: `data/pending_actions.json`, `data/action_logs.jsonl`
- Owner style profile, import inbox, and pending summaries: `data/style_profiles/`
- Bot persona: `config/persona.json`

There is intentionally no `bot_<botQQ>` namespace yet. Current priority is testing on the small QQ account. Add namespaces before running multiple bot QQ accounts against the same data directory.

## Roadmap

### Stage 5B: Distillation Quality and Evaluation

- Redesign style-profile fields for stronger imitation beyond draft replies.
- Build evaluation pairs from chat logs: previous message/context plus the owner's real reply.
- Add `/风格 评估` to compare generated replies against real historical replies.
- Improve phrase extraction to avoid generic words such as product names or platform terms.
- Separate global owner style from relationship-specific style.
- Keep this as a focused quality pass after the main framework is in place.

### Stage 6: Permission and Contact Whitelist

- Connect the trust list and permission levels to future style draft, teaching review, and automation modes.
- Add richer confirmation metadata for future file/network/shell tools.

### Stage 7: Agent Mode Refactor

- Replace the old `AGENT_MODE` path with a controlled tool loop.
- Tools should have schemas, permission levels, confirmation rules, and execution logs.
- Default high-risk outputs should be drafts or read-only.

## Codex Skill

A local Codex skill was created for future maintenance:

`C:\Users\lenovo\.codex\skills\claude-qq-bot-maintainer`

Use it for future tasks involving this QQbot's memory, tools, status commands, style profile, migrations, permissions, NapCat connectivity, or staged roadmap.

## Command Guide

Full user-facing command documentation is maintained at `docs/command-guide.md`.
