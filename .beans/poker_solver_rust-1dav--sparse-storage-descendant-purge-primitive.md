---
# poker_solver_rust-1dav
title: Sparse storage descendant purge primitive
status: completed
type: task
priority: high
created_at: 2026-05-14T14:31:49Z
updated_at: 2026-05-14T15:04:27Z
parent: poker_solver_rust-xl3h
---

Add sparse storage support for removing or invalidating descendants below a pruned action history. Start with the simplest correct implementation: use the exact packed history prefix plus appended action index to identify descendants, scan affected shards, and remove matching rows. If scan cost is too high, document the path to generation-based invalidation. Acceptance: API returns rows purged, slots purged, and duration-friendly counters; it does not remove the parent row; tests cover prefix matching and non-matching sibling preservation.

## Acceptance

- [x] Added sparse purge stats with row/regret/strategy slot counts.
- [x] Added packed-history-prefix descendant purge API on SparseMpStorage.
- [x] Preserves parent prefix rows and matching only uses exact packed history bits, never hash prefix guesses.
- [x] Keeps global and per-shard counters consistent after removal.
- [x] Added focused sparse storage purge tests, including conservative handling beyond packed capacity.

## Summary of Changes

Implemented a conservative sparse storage descendant purge primitive that scans materialized shards, removes rows strictly below an exact packed action-history prefix, and returns purge accounting. Prefixes longer than the 32 packed action slots purge nothing; callers can use generation-based invalidation later if they need support beyond packed history capacity.
