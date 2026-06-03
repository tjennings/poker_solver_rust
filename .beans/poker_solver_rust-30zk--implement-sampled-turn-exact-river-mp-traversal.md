---
# poker_solver_rust-30zk
title: Implement sampled-turn exact-river MP traversal
status: completed
type: feature
priority: normal
created_at: 2026-05-19T15:57:18Z
updated_at: 2026-05-19T16:11:14Z
---

Add an opt-in lazy MP training mode that samples deals through the turn and exactly averages over legal river cards at river chance boundaries, keeping existing external-sampling action traversal.

## Summary of Changes

- Added training.chance_continuation_mode with sampled_full_deal default and sampled_turn_exact_river opt-in.
- Extended lazy MP traversal with explicit chance nodes and exact river averaging at turn-to-river boundaries and pre-river showdown terminals.
- Precomputed legal river DealWithBuckets variants once per sampled job, preserving averaged regret scale.
- Updated training/architecture docs and added focused config, runout, and lazy-training tests.

## Validation

- cargo fmt --check
- cargo check -p poker-solver-trainer
- cargo test -p poker-solver-core training_chance_continuation_mode_parses -- --nocapture
- cargo test -p poker-solver-core exact_river_runouts_enumerate_legal_rivers_for_turn_prefix -- --nocapture
- cargo test -p poker-solver-core lazy_train_sampled_turn_exact_river_completes -- --nocapture
- cargo test -p poker-solver-trainer run_train_blueprint_mp_lazy_sparse_no_tui_zero_iters_completes -- --nocapture
