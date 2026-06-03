---
# poker_solver_rust-bgbz
title: 'Phase 3: in-memory strategy pruning for blueprint trainer'
status: draft
type: feature
priority: high
created_at: 2026-06-03T18:10:21Z
updated_at: 2026-06-03T18:10:49Z
parent: poker_solver_rust-34kn
blocked_by:
    - poker_solver_rust-l6r9
---

Phase 3 of the blueprint trainer tree roadmap.

Scope:
- Add in-memory-only strategy pruning for persistently negative lines after the lazy tree model has been validated.
- Start with reversible logical masking/inactivity, not irreversible deletion.
- Add configurable thresholds: warmup iterations, minimum visits/update count, regret/value threshold, persistence window, hysteresis/reactivation threshold, reactivation cadence, and per-depth/action-class limits.
- Initially avoid pruning fold/check/call and terminal-resolution edges; first target aggressive nonterminal branches.
- Track pruned/blocked edges, skipped updates, reactivated lines, rows purged if any later experiment exists, and strategy drift versus the unpruned baseline.
- Preserve an unpruned mode and make it the correctness baseline.
- Update docs/training.md for new pruning config and docs/architecture.md for solver behavior changes.

Acceptance criteria:
- With pruning disabled, outputs remain equivalent to Phase 2 baseline.
- With conservative pruning enabled, training runs complete and report pruning/reactivation metrics.
- Pruned lines can be periodically revisited; early noise cannot permanently remove strategic lines.
- Validation compares pruned vs unpruned behavior on the HU fixture and documents acceptable drift.

Blocked by Phase 2 validation.
