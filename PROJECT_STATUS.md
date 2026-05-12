# QQ Bot Project Status

**Updated:** 2026-05-10  
**Project:** `F:\ClaudeSpace2\claude-qq-bot`

## Current Architecture

The bot is a NoneBot2 + NapCat/OneBot QQ bot. It starts with `python -u bot.py` and listens for NapCat/Lagrange reverse WebSocket connections at `ws://127.0.0.1:8081/onebot/v11/ws`. Runtime code is now centered on ordinary chat, command handlers, memory, permissions, confirmation, and the Stage 5B owner-style teaching loop.

Old Agent Mode has been retired from runtime. The historical implementation is archived at `src/plugins/claude/legacy/agent.py` and should not be re-enabled directly.

## Key Runtime Files

```text
src/plugins/claude/
├── dialogue.py              # shared event helpers and ordinary chat fallback
├── commands/                # system, permissions, memory, style, teaching, legacy notices
├── memory_core.py           # sessions, long-term memory, key facts, tasks
├── permissions.py           # owner/trust-list checks
├── confirmation.py          # pending confirmation and audit logs
├── controlled_agent.py      # Stage 7/8 controlled tools, plans, review drafts
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
- Stage 7/8 controlled Agent added: `/agent` owner-private commands, tool schema, draft review queue, confirmation-gated execution, and audit logs.

## Storage Model

- Sessions: `data/sessions/`
- User profiles/tasks/facts: `data/key_facts.db`
- Long-term memory: `data/longterm_memory/`
- Runtime toggles: `data/runtime_state.json`
- Todos: `data/todos.json`
- Permissions: `data/permissions.json`
- Pending confirmations and audit logs: `data/pending_actions.json`, `data/action_logs.jsonl`
- Controlled Agent drafts: `data/agent_drafts.json`
- Owner style data: `data/style_profiles/`
- Bot persona: `config/persona.json`

There is still no `bot_<botQQ>` namespace. Add that before running multiple bot QQ accounts against the same data directory.

## Roadmap

### Stage 5B Quality

- Continue improving retrieval quality, phrase extraction, relationship mapping, and generation/rerank evaluation.
- Add embedding retrieval after the rule-based retrieval baseline is stable.
- Keep real-contact testing in `/教学 开` shadow review unless a new risk-gated auto-send path is designed.

### Stage 7 Controlled Agent

First version complete:

- Do not revive old `AGENT_MODE`.
- `/agent 工具`, `/agent 计划`, `/agent 草稿`, `/agent 执行`, `/agent 执行计划`, `/agent 采纳`, `/agent 拒绝`.
- Tools are schema-like entries with permission, risk, usage, and confirmation flags.
- High-risk execution is confirmation-gated and audited.

### Stage 8 Controlled Review

First version complete:

- Plans/drafts are saved locally for review.
- Drafts can be accepted/rejected without execution.
- Execution is explicit and still respects confirmation rules.
- No automatic owner-style sending was reintroduced.

## Validation

```bash
python -m compileall -q bot.py src test_quick.py test_memory.py
python test_quick.py
python test_memory.py
```

`nb run` is not the default startup path and should not be required in clean installs unless `nb-cli` is installed separately for operator preference.
