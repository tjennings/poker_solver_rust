---
# poker_solver_rust-u3rf
title: 'Brainstorm Option A: DeepStack gadget as CFR tree node'
status: completed
type: task
priority: high
created_at: 2026-04-24T03:05:56Z
updated_at: 2026-04-24T04:32:49Z
---

Pre-design brainstorm for moving the safe-resolving gadget from post-clamp wrapper to a T/F tree modification at the subgame root. See docs/plans/2026-04-24-option-a-deepstack-gadget-tree.md for substrate. Output: validated design doc + refined akg3 scope.



## Summary of Changes

Brainstorm complete 2026-04-24. All 7 design fork points resolved with user input; full design documented at `docs/plans/2026-04-24-option-a-deepstack-gadget-design.md`.

**Decisions locked:**
1. Acceptance bar: (a) safety only — formal guarantee via Burch §3 sufficiency; exploitability gap may persist.
2. Pattern: (1) structural tree modification (not DeepStack-Leduc's algorithmic Pattern 2).
3. Gadget count: (C) two nested — G_IP → G_OOP → R0 — bounds both players' realized CFVs.
4. Terminal semantics: (ii) neutralized — non-gadget player gets 0 at Terminate.
5. Tree injection: (a) at `PostFlopGame` level (new `with_config_and_gadget` constructor).
6. Evaluator dispatch: (x) `DispatchingBoundaryEvaluator` wrapper in tauri-app.
7. Tests: (1)+(2)+(3)+(4) hard-gate MVP; (5) `JhTh9h` harness informational.

**Research foundation:** ml-researcher pass confirmed DeepStack-Leduc's released code implements the gadget algorithmically (`cfrd_gadget.lua`) rather than structurally; we weighed this against our existing depth-boundary mechanism (which already provides per-hand externally-supplied CFVs and thus makes Pattern 1 more tractable than it was for DeepStack-Leduc) and user chose structural.

**Next steps:**
- `hex:writing-plans` produces implementation plan from the design doc.
- akg3 to be re-scoped (see its bean).
- Implementation routed to rust-developer agents per CLAUDE.md manager-mode.
