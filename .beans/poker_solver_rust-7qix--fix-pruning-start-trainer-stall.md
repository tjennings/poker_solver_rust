---
# poker_solver_rust-7qix
title: Fix pruning-start trainer stall
status: in-progress
type: bug
priority: high
created_at: 2026-05-14T17:34:59Z
updated_at: 2026-05-14T17:34:59Z
---

Trainer throughput drops to zero with near-zero CPU after pruning/negative-action purge starts. Investigate the lazy sparse pruning and post-DCFR negative-action purge path for deadlock or pathological maintenance work, then fix the stall without touching user config edits.
