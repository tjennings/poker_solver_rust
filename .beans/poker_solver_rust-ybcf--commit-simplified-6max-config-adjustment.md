---
# poker_solver_rust-ybcf
title: Commit simplified 6max config adjustment
status: completed
type: task
priority: normal
created_at: 2026-05-06T19:28:32Z
updated_at: 2026-05-06T19:28:35Z
---

Commit the user-approved sample configuration change that lowers the simplified 6max bucket path/counts and adjusts preflop raise sizing, keeping it separate from the EMD metric-alignment work.

## Summary of Changes

Committed the user-approved simplified 6max sample config adjustment separately from the clustering metric-alignment work. The config now points at the 200-bucket path/counts and uses the revised preflop raise sizing.
