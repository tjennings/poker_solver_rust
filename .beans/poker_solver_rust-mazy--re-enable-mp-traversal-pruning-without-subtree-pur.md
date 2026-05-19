---
# poker_solver_rust-mazy
title: Re-enable MP traversal pruning without subtree purge
status: in-progress
type: task
priority: high
created_at: 2026-05-19T14:44:55Z
updated_at: 2026-05-19T14:44:55Z
---

Add an explicit training config switch for ordinary MP traversal pruning, enable it in the active 250/100/20 config, and keep negative-action subtree purge disabled so pruning does not physically remove stored strategy rows/subtrees.
