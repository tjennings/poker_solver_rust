---
# poker_solver_rust-ae2v
title: Implement regret-based pruning in exhaustive postflop CFR
status: in-progress
type: feature
created_at: 2026-03-01T01:12:16Z
updated_at: 2026-03-01T01:12:16Z
---

Add RBP to postflop_exhaustive.rs: skip subtree traversal for negative-regret actions at hero nodes (95% prune rate after warmup). YAML-configurable: prune_warmup, prune_explore_freq, regret_floor.

## Todo
- [ ] Add config fields to PostflopModelConfig (prune_warmup, prune_explore_freq, regret_floor)
- [ ] Add pruning logic in exhaustive_cfr_traverse hero branch
- [ ] Add regret floor clamping after DCFR discounting
- [ ] Thread config through to traversal calls
- [ ] Add tests for pruning behavior
- [ ] Verify all existing tests pass
