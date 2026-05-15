---
# poker_solver_rust-gkja
title: Investigate MP convergence anomalies with subtree purge disabled
status: in-progress
type: bug
priority: high
created_at: 2026-05-15T12:56:04Z
updated_at: 2026-05-15T12:56:04Z
---

User reports unusual converged MP strategies even with negative_action_subtree_purge_enabled=false. Investigate remaining pruning/traversal/averaging paths that can bias lazy sparse MP training independently of persistent negative-action subtree purge, then patch and test any confirmed bug.
