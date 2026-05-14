---
# poker_solver_rust-9w7c
title: Bump CFVNet GPU buffer refresh to 50 percent
status: in-progress
type: task
priority: normal
created_at: 2026-05-14T00:30:06Z
updated_at: 2026-05-14T00:30:06Z
---

Update boundary training GPU ring buffer refresh from 10% of the active pool per epoch to 50%, so new training runs cycle through the dataset faster.
