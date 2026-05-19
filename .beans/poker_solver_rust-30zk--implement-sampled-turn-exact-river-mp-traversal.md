---
# poker_solver_rust-30zk
title: Implement sampled-turn exact-river MP traversal
status: in-progress
type: feature
priority: normal
created_at: 2026-05-19T15:57:18Z
updated_at: 2026-05-19T15:57:18Z
---

Add an opt-in lazy MP training mode that samples deals through the turn and exactly averages over legal river cards at river chance boundaries, keeping existing external-sampling action traversal.
