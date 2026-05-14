---
# poker_solver_rust-y5jo
title: Clarify sampled river datagen solver controls
status: in-progress
type: bug
priority: high
created_at: 2026-05-14T06:05:54Z
updated_at: 2026-05-14T06:05:54Z
---

Make CFVNet sampled river datagen visibly honor solver_iterations and target_exploitability. Add diagnostics for configured max iterations, target semantics, average iterations, target-hit vs max-iteration completion counts, and avoid misleading progress output.\n\nChecklist:\n- [ ] Confirm config values are threaded into SolverConfig\n- [ ] Expose per-sample iteration count and completion reason\n- [ ] Update progress/final logs with target-hit/max-iter counts\n- [ ] Add focused tests for solver completion reason\n- [ ] Run targeted verification
