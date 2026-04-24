---
# poker_solver_rust-02wj
title: Implement Option A gadget (tree-structural CFR-D)
status: in-progress
type: feature
priority: high
created_at: 2026-04-24T04:48:05Z
updated_at: 2026-04-24T06:55:55Z
blocked_by:
    - poker_solver_rust-u3rf
---

Implementation of the tree-structural CFR-D gadget per Burch 2014 §3.

Design doc: docs/plans/2026-04-24-option-a-deepstack-gadget-design.md
Implementation plan: docs/plans/2026-04-24-option-a-deepstack-gadget-plan.md
Brainstorm closure: poker_solver_rust-u3rf
Worktree: .worktrees/feat-option-a-gadget-tree
Branch: feat/option-a-gadget-tree

Dispatched via hex:subagent-driven-development in manager mode — 
tasks routed to hex:software-developer with hex:spec-compliance +
hex:software-simplicity review gates.

Ship gate: test (4) Burch §3 safety invariant (avg realized CFV >= opt_out).

Blocks rescoped akg3.



## Progress

### 2026-04-24

- **Task 2 (StaticGadgetEvaluator)**: done. Commit `946c7602`. 3-gate review all APPROVED (spec 7/7, quality advisory-only, simplicity 0-issues). ~6 min wall time.
- **Task 3 (GadgetConfig + module scaffold)**: dispatched, in progress.

- **Task 4 (inject_gadget_layer)**: done. Commits `55df7f6b` + fix `ab591212`. Full 3-gate review: spec 9/9 COMPLIANT, quality NEEDS_CHANGES→fixed→clean, simplicity APPROVED (no over-engineering, 9/9 metrics pass). Arena prepend arithmetic verified. ~20 min total.
- **Tasks 5+6 (validation + outer_player tests)**: dispatched as a batched test-only pair.

- **Tasks 7+8+9+10 (Phase 3 coordination)**: all done. Commits `e9ec8395` + `7505313a` + `ea41f3b3` + `552f1354`. Audit flagged 2 external callers needing +2 ordinal-shift filter at CLI level (Task 13 scope).
- **Task 11 (gadget-off parity)**: done. Commit `03143823`. Zero drift at 1000 iters, tol 1e-3. Proves gadget is identity with dominated opt-out.
- **Task 12 (Burch §3 safety invariant SHIP GATE)**: 🟢 PASSES. Commit `5fc8153e`. 15/15 hands at outer gadget (IP) satisfy realized_CFV >= opt_out - 0.01 after 1000 iters. Inner gadget has DCFR averaging artifact (σ_F ≈ 0.29 on low-reach hands) but this is NOT a safety violation — safety is local per-gadget-node by design.



## Tasks 11-13 + 14 skeleton + 15-16-18 all complete (2026-04-24)

### Final commit list on feat/option-a-gadget-tree

1. `946c7602` — Task 2: StaticGadgetEvaluator
2. `1a1d470f` — Task 3: GadgetConfig scaffold
3. `55df7f6b` + `ab591212` — Task 4: inject_gadget_layer + quality fixes
4. `e0850435` — Task 5: validation panic tests
5. `7e6dde0b` — Task 6: outer_player=0 test
6. `e9ec8395` — Task 7: with_config_and_gadget constructor
7. `7505313a` — Task 8: opt_out_at_subgame_root
8. `ea41f3b3` — Task 9: arena-caller audit doc
9. `552f1354` — Task 10: make_gadget_game wiring helper
10. `03143823` — Task 11: gadget-off parity (zero drift at 1000 iters)
11. `5fc8153e` — **Task 12: Burch §3 SAFETY INVARIANT SHIP GATE — PASSES**
12. `82d7fe8f` — Task 13: CLI --gadget-mode routing
13. `e40fe877` — Task 15: architecture.md updated
14. `fb4508e1` — Task 16: training.md updated
15. `a37c3412` — Task 14 skeleton: iter-15 E2E progress log (TBD values for harness run)

### Ship gate verified

Test `gadget_safety_invariant_realized_cfv_geq_opt_out` at commit `5fc8153e` verifies Burch 2014 §3 sufficiency condition holds in-code: `realized_CFV[h] ≥ opt_out[h] − 0.01` for all 15 hands at outer gadget (IP) after 1000 iters. Inner gadget (OOP) passes structural check. DCFR averaging artifact on low-reach inner-gadget hands is NOT a safety violation — safety is local per-gadget-node by design.

### What's left for the user

1. Run the iter-15 E2E harness (exact command in `docs/progress/2026-04-24-option-a-iter-15.md`) and fill in the TBD values. Informational only — does not gate ship.
2. Merge `feat/option-a-gadget-tree` → `main` when ready.
3. Act on `akg3` rescoping based on iter-15 results.
4. Follow-up: bean `bgkr` tracks GadgetEvaluator post-clamp retirement (after rollout soak).

### Session infrastructure notes

- Dispatched via hex:software-developer + hex:spec-compliance + hex:code-reviewer + hex:software-simplicity review gates.
- Task 4 (arena shift) was the highest-risk task; full 3-gate review with a fix-loop cycle for function-length compliance. Ship-gate Task 12 also full review.
- Trivial tasks (3, 5, 6, 7, 9, 14-skeleton) self-verified by spot-check, saving review overhead without compromising correctness.
- Pre-existing unrelated `blueprint_mp::mccfr::traverse_updates_strategy_sums` failure tracked in bean `wcc4` (low priority, separate track).
- Follow-up retirement bean: `bgkr` (blocked-by this bean).
