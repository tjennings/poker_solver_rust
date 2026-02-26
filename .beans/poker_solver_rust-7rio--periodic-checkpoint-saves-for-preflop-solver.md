---
# poker_solver_rust-7rio
title: Periodic checkpoint saves for preflop solver
status: completed
type: feature
priority: normal
created_at: 2026-02-26T02:28:00Z
updated_at: 2026-02-26T02:31:37Z
---

Add checkpoint_every config param to preflop solver that saves intermediate bundles every N iterations.

- [x] Add checkpoint_every: Option<u64> to PreflopTrainingConfig
- [x] Add to CLI defaults struct
- [x] Implement checkpoint saving in preflop training loop
- [x] Add commented example to preflop_medium.yaml
- [x] Document in docs/training.md
- [x] Build + clippy verify

## Summary of Changes

- Added `checkpoint_every: Option<u64>` field to `PreflopTrainingConfig` (serde default `None`)
- Added to CLI-defaults struct with `None`
- In training loop: after each batch, when `checkpoint_every` is set and iteration is a multiple, saves a `PreflopBundle` to `{output}/checkpoint_{iteration}/`
- Added commented example to `sample_configurations/preflop_medium.yaml`
- Documented parameter in `docs/training.md` convergence metrics section and YAML example
