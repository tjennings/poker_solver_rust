---
# poker_solver_rust-er8n
title: Fix turn-boundary manifest shard path loading
status: in-progress
type: bug
priority: high
created_at: 2026-05-13T23:45:30Z
updated_at: 2026-05-13T23:45:30Z
---

Training turn-boundary v2 data fails because Python manifest loading resolves a shard as <dataset>/<dataset>/<shard>. Diagnose manifest path normalization and fix loader/writer compatibility so existing datasets train successfully.
