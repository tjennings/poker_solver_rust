---
# poker_solver_rust-1dav
title: Sparse storage descendant purge primitive
status: todo
type: task
priority: high
created_at: 2026-05-14T14:31:49Z
updated_at: 2026-05-14T14:31:49Z
parent: poker_solver_rust-xl3h
blocked_by:
    - poker_solver_rust-6n1e
---

Add sparse storage support for removing or invalidating descendants below a pruned action history. Start with the simplest correct implementation: use the exact packed history prefix plus appended action index to identify descendants, scan affected shards, and remove matching rows. If scan cost is too high, document the path to generation-based invalidation. Acceptance: API returns rows purged, slots purged, and duration-friendly counters; it does not remove the parent row; tests cover prefix matching and non-matching sibling preservation.
