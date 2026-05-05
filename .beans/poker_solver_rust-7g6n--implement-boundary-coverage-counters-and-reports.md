---
# poker_solver_rust-7g6n
title: Implement boundary coverage counters and reports
status: completed
type: task
priority: high
created_at: 2026-05-05T02:57:03Z
updated_at: 2026-05-05T03:48:58Z
parent: poker_solver_rust-q93y
---

Track generated examples by pot bucket, effective stack, SPR, raise depth, boundary node type, board texture, range entropy, and source configuration.



Started with manifest-level coverage counters for turn-boundary datagen. First implementation will classify records by pot, stack, SPR, all-in proximity, board texture, range entropy, target source, range source, raise-depth label, and boundary ordinal label.



Completed manifest-backed coverage counters. Turn-boundary runs now report total records plus buckets for pot, stack, SPR, all-in proximity, raise-depth label, boundary ordinal label, board texture, range entropy, range source, and target source.
