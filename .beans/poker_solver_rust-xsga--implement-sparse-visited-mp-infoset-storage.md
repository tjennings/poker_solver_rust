---
# poker_solver_rust-xsga
title: Implement sparse visited MP infoset storage
status: completed
type: feature
priority: high
created_at: 2026-05-07T13:18:47Z
updated_at: 2026-05-07T14:13:55Z
parent: poker_solver_rust-5kvv
blocked_by:
    - poker_solver_rust-7n72
---

Replace or supplement dense MpStorage for 100bb runs with sparse visited-infoset storage keyed by compact betting-state/node identity, street bucket, and action. Reads of unvisited entries must behave as zero/uniform; updates must remain thread-safe under rayon MCCFR batches; snapshot/export must support sparse format.

## Work Started

Starting with the storage layer: sparse infoset keys, thread-safe visited-entry allocation, uniform defaults for missing entries, discounting, snapshots, and focused tests.

## Summary of Changes

Added the sparse MP infoset storage layer with stable lazy keys, sharded visited-entry allocation, atomic regrets and strategy sums, uniform defaults for missing entries, DCFR-style discounting over visited entries only, deterministic sparse snapshots, snapshot restore, focused tests, and architecture documentation.
