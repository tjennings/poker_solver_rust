---
# poker_solver_rust-h2wn
title: Re-enable traversal pruning exploration in 250 config
status: in-progress
type: task
priority: normal
created_at: 2026-05-19T14:55:38Z
updated_at: 2026-05-19T14:55:38Z
---

Set the active 250/100/20 MP config prune_explore_pct back to 0.05 so ordinary traversal pruning periodically explores full branches, while keeping negative-action subtree purge disabled.
