---
# poker_solver_rust-6bs3
title: 'Perf Rank 3: parallel_traverse fold+collect vs fold+reduce'
status: completed
type: task
priority: normal
created_at: 2026-02-28T22:40:17Z
updated_at: 2026-02-28T22:40:17Z
---

Replace fold+reduce with fold+collect in parallel_traverse to reduce allocations from O(log N reduce identities) to O(threads). ~5-20% iteration speedup depending on buffer size. Implemented in commit 4d4fb24.
