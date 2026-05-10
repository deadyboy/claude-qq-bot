# QQ Bot Project Status

**Updated:** 2026-05-10  
**Project:** `F:\ClaudeSpace2\claude-qq-bot`

## Current Architecture

The bot is a NoneBot2 + NapCat/OneBot QQ bot. Runtime code is now centered on ordinary chat, command handlers, memory, permissions, confirmation, and the Stage 5B owner-style teaching loop.

Old Agent Mode has been retired from runtime. The historical implementation is archived at `src/plugins/claude/legacy/agent.py` and should not be re-enabled directly.

## Key Runtime Files

```text
src/plugins/claude/
├── dialogue.py              # shared event helpers and ordinary chat fallback
├── commands/                # system, permissions, memory, style, teaching, legacy notices
├── memory_core.py           # sessions, long-term memory, key facts, tasks
├── permissions.py           # owner/trust-list checks
├── confirmation.py          # pending confirmation and audit logs
├── style_profile.py         # owner style profile and drafts
├── style_skill.py           # 36.skill runtime layer and corrections
├── style/distill/           # Stage 5B QCE distillation/retrieval/generation modules
├── style_distill.py         # compatibility facade
└── legacy/agent.py          # archived old Agent Engine prototype
```

## Completed Work

- NapCat/OneBot connectivity and API proxy fixes.
- Group @/reply targeting.
- Bot persona and explicit user memory.
- Automatic user fact extraction with sensitive-content filtering.
- Safe tools and status commands.
- Owner permissions, trust list, confirmation, and audit logs.
- Style profile v1, chat-log import, QCE distillation, retrieval-first generation, 36.skill runtime layer, teaching review, and correction feedback.
- Command handlers split into `src/plugins/claude/commands/`.
- Stage 5B distillation split into `src/plugins/claude/style/distill/`.
- Session memory consolidated into `memory_core.py`.
- Old `AGENT_MODE`, `/tasks`, and runtime `agent.py` path removed.

## Storage Model

- Sessions: `data/sessions/`
- User profiles/tasks/facts: `data/key_facts.db`
- Long-term memory: `data/longterm_memory/`
- Runtime toggles: `data/runtime_state.json`
- Todos: `data/todos.json`
- Permissions: `data/permissions.json`
- Pending confirmations and audit logs: `data/pending_actions.json`, `data/action_logs.jsonl`
- Owner style data: `data/style_profiles/`
- Bot persona: `config/persona.json`

There is still no `bot_<botQQ>` namespace. Add that before running multiple bot QQ accounts against the same data directory.

## Roadmap

### Stage 5B Quality

- Continue improving retrieval quality, phrase extraction, relationship mapping, and generation/rerank evaluation.
- Add embedding retrieval after the rule-based retrieval baseline is stable.
- Keep real-contact testing in `/教学 开` shadow review unless a new risk-gated auto-send path is designed.

### Stage 7 Controlled Agent

- Do not revive old `AGENT_MODE`.
- Build future agent behavior around schema tools, permissions, confirmations, and audit logs.
- Keep high-risk outputs as drafts or owner-confirmed actions.

## Validation

```bash
python -m compileall -q bot.py src test_quick.py test_memory.py
python test_quick.py
python test_memory.py
```
