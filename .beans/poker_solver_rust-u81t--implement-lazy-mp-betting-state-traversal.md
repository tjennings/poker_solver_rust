---
# poker_solver_rust-u81t
title: Implement lazy MP betting-state traversal
status: completed
type: feature
priority: high
created_at: 2026-05-07T13:18:52Z
updated_at: 2026-05-07T14:48:08Z
parent: poker_solver_rust-5kvv
blocked_by:
    - poker_solver_rust-7n72
---

Avoid eager materialization of the full 100bb 6-max MP game tree. Implement traversal over compact public betting state with dynamic legal-action generation, optional memoization for hot states, and stable infoset keys for storage/export. Preserve current eager tree path for small/debug configs during migration.

## Work Started

Starting the first lazy traversal slice: factor compact public-state types and legal-action generation around the existing MP game-tree semantics, then add tests before wiring the full trainer path.

## Summary of Changes

Added core lazy MP public-state traversal over sparse storage. The new LazyMpGame carries compact betting state, generates legal actions on demand from the MP action abstraction, collapses chance/runout nodes against sampled full boards, writes regrets and strategy sums via SparseMpStorage keys, and exposes setup_lazy_training/run_lazy_training/train_blueprint_mp_lazy without materializing the eager tree. Added focused tests for 100bb two-preflop-raise reachability and sparse lazy training updates.
