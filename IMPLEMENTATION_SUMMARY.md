# Implementation Summary

## Runtime Shape

The bot is a NoneBot2 + OneBot v11 QQ bot connected to NapCat by reverse WebSocket on `127.0.0.1:8081`.

Current runtime is no longer dual-mode. The old Agent Engine prototype has been archived and is not imported by the bot.

## Main Modules

| Area | Path | Notes |
|------|------|------|
| Bot entry | `bot.py` | Starts NoneBot and loads the Claude plugin |
| Chat handling | `src/plugins/claude/dialogue.py` | Shared event helpers plus ordinary chat fallback |
| Commands | `src/plugins/claude/commands/` | Split command handlers by domain |
| Memory | `src/plugins/claude/memory_core.py` | Short-term sessions, long-term memory, key facts, tasks |
| Persona | `src/plugins/claude/persona.py` | Bot identity prompt |
| Safe tools | `src/plugins/claude/safe_tools.py` | Time, calculation, todos, profile search, latest errors |
| Permissions | `src/plugins/claude/permissions.py` | Owner and trust-list checks |
| Confirmation | `src/plugins/claude/confirmation.py` | Pending actions and audit log |
| Controlled Agent | `src/plugins/claude/controlled_agent.py` | Stage 7/8 tool plans, review drafts, confirmation-gated execution |
| Style profile | `src/plugins/claude/style_profile.py` | Owner style commands and draft generation |
| 36.skill runtime | `src/plugins/claude/style_skill.py` | Persona, relationship profiles, corrections |
| Stage 5B distill | `src/plugins/claude/style/distill/` | QCE parsing, turns, phrases, taxonomy, retrieval, generation, reports |
| Legacy reference | `src/plugins/claude/legacy/agent.py` | Archived old Agent Engine prototype |

`src/plugins/claude/style_distill.py` remains as a compatibility facade that re-exports the split Stage 5B modules.

## Stable Capabilities

- Private and group targeted replies.
- Explicit user profile memory.
- Automatic low-risk fact extraction.
- `/status`, `/tools`, `/权限`, `/model`, `/clear`, `/help`.
- Time, safe calculation, todos, and profile search.
- Owner trust list and confirmation/audit flow.
- Style profile import, QCE distillation, retrieval-first drafts, teaching review, and correction feedback.
- `/agent` owner-private controlled tools, plans, review drafts, and explicit execution.
- Retired `/代聊` and `/风格 自动回复` compatibility notices.

## Removed Runtime Paths

- `src/plugins/claude/memory.py` was removed. Use `memory_core.session_manager` for session history.
- Old `AGENT_MODE` was removed from `dialogue.py`.
- `/tasks` is no longer registered.
- `agent.py` was moved to `src/plugins/claude/legacy/agent.py` for reference only.

## Validation

Use:

```bash
python -m compileall -q bot.py src test_quick.py test_memory.py
python test_quick.py
python test_memory.py
```

Run `test_memory.py` especially after touching `memory_core.py`, task storage, or SQLite queries.
