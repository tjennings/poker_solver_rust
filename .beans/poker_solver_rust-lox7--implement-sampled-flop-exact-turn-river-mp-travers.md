---
# poker_solver_rust-lox7
title: Implement sampled-flop exact turn-river MP traversal
status: completed
type: feature
priority: normal
created_at: 2026-05-19T16:36:26Z
updated_at: 2026-05-19T16:46:21Z
---

Add an opt-in lazy MP training mode that samples private cards plus flop and exactly averages over legal turn/river continuations, keeping existing external-sampling action traversal.

## Summary of Changes

Implemented `sampled_flop_exact_turn_river` for lazy MP traversal. The trainer now precomputes legal turn/river continuations for a sampled flop prefix, traversal averages over exact chance boundaries, config parsing accepts the new mode, and docs/tests cover the behavior.
