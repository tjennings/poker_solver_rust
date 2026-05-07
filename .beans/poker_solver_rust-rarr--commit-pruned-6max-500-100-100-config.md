---
# poker_solver_rust-rarr
title: Commit pruned 6max 500-100-100 config
status: completed
type: task
priority: normal
created_at: 2026-05-07T03:30:11Z
updated_at: 2026-05-07T03:30:34Z
---

Commit the user's pruned 6max 500/100/100 Blueprint MP training config that removes many postflop betting options for faster training.



## Summary of Changes

Committed the user's pruned 6max 500/100/100 Blueprint MP training config. The config now keeps a narrow postflop action abstraction: flop 75% lead / pot raise, turn pot lead / pot raise, and river pot lead with half-pot/pot raises.

Verification:
- Reviewed git diff before committing.
- No tests were run for this config-only commit because the next step is the MP snapshot bug fix.
