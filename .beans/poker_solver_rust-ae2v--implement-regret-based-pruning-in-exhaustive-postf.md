---
# poker_solver_rust-ae2v
title: Implement regret-based pruning in exhaustive postflop CFR
status: completed
type: feature
priority: normal
created_at: 2026-03-01T01:12:16Z
updated_at: 2026-03-01T01:37:34Z
---

Add RBP to postflop_exhaustive.rs: skip subtree traversal for negative-regret actions at hero nodes (95% prune rate after warmup). YAML-configurable: prune_warmup, prune_explore_freq, regret_floor.

## Todo
- [ ] Add config fields to PostflopModelConfig (prune_warmup, prune_explore_freq, regret_floor)
- [ ] Add pruning logic in exhaustive_cfr_traverse hero branch
- [ ] Add regret floor clamping after DCFR discounting
- [ ] Thread config through to traversal calls
- [ ] Add tests for pruning behavior
- [ ] Verify all existing tests pass

## Summary of Changes

Implemented regret-based pruning (Brown & Sandholm, NeurIPS 2015) in the exhaustive postflop CFR backend.

### Files Changed
- `postflop_model.rs` — 3 YAML-configurable fields: `prune_warmup`, `prune_explore_freq`, `regret_floor`
- `postflop_exhaustive.rs` — Single-pass u16 bitmask pruning at hero decision nodes, regret floor clamping, 2 new tests
- `postflop_mccfr.rs` — Config field defaults added to test constructors

### How It Works
After `prune_warmup` iterations, at hero decision nodes where at least one action has positive regret, actions with negative regret are skipped (subtree not traversed). Every `prune_explore_freq` iterations, pruning is disabled to let pruned actions recover. Regret floor prevents unbounded negative accumulation.

### Reviewed
- 2 passes by idiomatic-rust-enforcer (PASS)
- 2 passes by rust-perf-reviewer (PASS)
- All 11 postflop_exhaustive tests pass
