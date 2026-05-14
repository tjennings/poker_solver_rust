---
# poker_solver_rust-y5jo
title: Clarify sampled river datagen solver controls
status: completed
type: bug
priority: high
created_at: 2026-05-14T06:05:54Z
updated_at: 2026-05-14T06:09:10Z
---

Make CFVNet sampled river datagen visibly honor solver_iterations and target_exploitability. Add diagnostics for configured max iterations, target semantics, average iterations, target-hit vs max-iteration completion counts, and avoid misleading progress output.\n\nChecklist:\n- [x] Confirm config values are threaded into SolverConfig\n- [x] Expose per-sample iteration count and completion reason\n- [x] Update progress/final logs with target-hit/max-iter counts\n- [x] Add focused tests for solver completion reason\n- [x] Run targeted verification

## Summary of Changes

Confirmed CPU domain datagen threads `datagen.solver_iterations` and `datagen.target_exploitability` into `SolverConfig`. Added `SolverCompletionReason` and per-sample iteration counts, then surfaced average iterations plus target-stop/max-iteration-stop counts in the progress and final datagen logs. Documented that `target_exploitability` is a pot-fraction chip threshold (`target * pot`), which explains why `0.01` can still display around `5 * pot` mbb/h at 100bb/200-chip stacks. Added focused tests for max-iteration and target-exploitability completion paths.
