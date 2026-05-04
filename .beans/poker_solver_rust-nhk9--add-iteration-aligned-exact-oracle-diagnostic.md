---
# poker_solver_rust-nhk9
title: Add iteration-aligned exact_oracle diagnostic
status: completed
type: task
priority: high
created_at: 2026-05-04T04:18:13Z
updated_at: 2026-05-04T04:28:05Z
parent: poker_solver_rust-e90m
---

Implement a compare-solve diagnostic that can feed exact_oracle boundary values from an iteration-aligned exact continuation instead of only the finalized average continuation.

Checklist:

[x] Map where exact per-iteration strategies/CFVs can be captured without perturbing normal solve output.
[x] Add a hidden compare-solve switch for iteration-aligned oracle behavior.
[x] Implement the minimal capture/replay path for exact_oracle boundaries.
[x] Run canonical 200/200 and 1000/1000 comparisons against finalized-average oracle.
[x] Document results and update training diagnostics docs.
[x] Run the full warm test suite under 1 minute.

## Summary of Changes

- Added hidden `--oracle-iteration-aligned` compare-solve diagnostic for exact_oracle runs.
- Runs exact and subgame `solve_step` in lockstep while boundary evaluators read raw CFVs from the shared exact continuation.
- Documented CLI usage and the experiment results.
- Canonical aligned results: 200/200 delta +65.76 mbb/hand; 1000/1000 delta +92.52 mbb/hand.
- Conclusion: iteration alignment does not recover exact. Final-average decoupling contributes at high iterations, but the remaining issue likely lives in multi-boundary root action CFV composition or regret deltas.
- Full warm `cargo test` passed in 53.11s.
