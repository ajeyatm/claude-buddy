# Chat Agent Next Steps Plan

Use this checklist to implement features safely in order. We will complete one item at a time.

## Phase 1: Stability First (Do This Before Compaction)

- [x] Add robust tool-call error envelopes
  - Always append a `tool` response even when a tool fails.
  - Include concise error context in tool output.
  - Keep behavior consistent across Read/Write/Bash.

- [x] Add timeout and safe-execution controls for Bash
  - Add subprocess timeout.
  - Return timeout/failure output as tool message.
  - Ensure no hanging command blocks the loop forever.

- [x] Add input guardrails for interactive loop
  - Skip empty inputs.
  - Support exit aliases (`exit`, `quit`, `q`).
  - Keep prompt flow clean.

- [ ] Add message-shape validation helper
  - Validate assistant/tool message shape before append.
  - Prevent invalid history entries (especially `tool_calls`).

## Phase 2: Token Budget Guardrail

- [ ] Add context budget thresholds
  - Configure soft and hard token limits via constants/env.
  - Trigger compaction at soft limit before request.

- [ ] Add message token estimation helper
  - Use a simple heuristic first (char-to-token estimate).
  - Log estimated prompt size before API call.

- [ ] Add preflight budget check in agent loop
  - Check estimated size before each completion call.
  - Compact messages when over threshold.

## Phase 3: Message Compaction (MVP)

- [ ] Implement sliding-window compaction
  - Keep system message + most recent K turns.
  - Drop oldest turns when window exceeds cap.

- [ ] Add compaction metrics
  - Log before/after message count.
  - Log before/after estimated tokens.

- [ ] Verify compaction correctness
  - Ensure conversation still continues coherently.
  - Ensure tools still work after compaction.

## Phase 4: Rolling Summary Memory

- [ ] Add summary generator function
  - Summarize old turns into concise memory bullets.
  - Capture goals, constraints, decisions, pending tasks.

- [ ] Store summary as pinned memory message
  - Keep original system prompt.
  - Insert/update one summary message.
  - Keep recent K turns in full.

- [ ] Add summary refresh policy
  - Refresh only when new old-turn chunk is compacted.
  - Avoid summarizing every single turn.

## Phase 5: Skills Layer (After Compaction + Summary)

- [ ] Define initial skills catalog
  - Example: `explain`, `code-edit`, `debug`, `bash-help`.
  - For each skill, define intent + behavior contract.

- [ ] Add lightweight skill router
  - Route user input to a skill by intent keywords/heuristics.
  - Fall back to default assistant behavior.

- [ ] Add skill-specific system instruction snippets
  - Keep snippets short and composable.
  - Apply only the selected skill context.

## Phase 6: Validation and Hardening

- [ ] Create reproducible manual test set
  - 2-turn, 10-turn, and 20-turn conversations.
  - Tool success/failure scenarios.
  - Long context scenario that forces compaction.

- [ ] Add regression checklist
  - No invalid `tool_calls` history.
  - No crash on subprocess failures.
  - Session token usage still reported.

- [ ] Add optional toggles
  - `SHOW_USAGE=true|false`
  - `CLI_THEME=minimal|high-contrast|default`
  - `COMPACTION_ENABLED=true|false`

---

## Execution Order (Strict)

- [ ] Complete all Phase 1 items.
- [ ] Complete all Phase 2 items.
- [ ] Complete all Phase 3 items.
- [ ] Complete all Phase 4 items.
- [ ] Complete all Phase 5 items.
- [ ] Complete all Phase 6 items.

## Notes

- Keep each change small and test immediately.
- Commit after each completed phase.
- Prefer explicit message schemas over implicit dumps.
