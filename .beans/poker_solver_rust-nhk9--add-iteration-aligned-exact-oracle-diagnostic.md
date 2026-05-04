---
# poker_solver_rust-nhk9
title: Add iteration-aligned exact_oracle diagnostic
status: in-progress
type: task
priority: high
created_at: 2026-05-04T04:18:13Z
updated_at: 2026-05-04T04:18:13Z
parent: poker_solver_rust-e90m
---

Implement a compare-solve diagnostic that can feed exact_oracle boundary values from an iteration-aligned exact continuation instead of only the finalized average continuation.

Checklist:

[ ] Map where exact per-iteration strategies/CFVs can be captured without perturbing normal solve output.
[ ] Add a hidden compare-solve switch for iteration-aligned oracle behavior.
[ ] Implement the minimal capture/replay path for exact_oracle boundaries.
[ ] Run canonical 200/200 and 1000/1000 comparisons against finalized-average oracle.
[ ] Document results and update training diagnostics docs.
[ ] Run the full warm test suite under 1 minute.
