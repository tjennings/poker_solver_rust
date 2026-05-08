---
# poker_solver_rust-ou7m
title: Point turn-boundary training config at v2 local data
status: completed
type: task
priority: normal
created_at: 2026-05-08T01:06:04Z
updated_at: 2026-05-08T01:07:07Z
---

Update the turn-boundary CFVNet training configuration/example to use local_data/cfvnet/turn_boundary/v2.\n\n- [x] Update sample training config data path\n- [x] Sanity-check the referenced v2 dataset path\n- [x] Commit config and bean update

## Summary of Changes\n\nUpdated the turn-boundary CFVNet sample training command to read local_data/cfvnet/turn_boundary/v2 and write a v2 model output directory. Confirmed the v2 data directory exists locally.
