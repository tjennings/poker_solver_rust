---
# poker_solver_rust-fga5
title: Replace EHS2 buckets with canonical 169 hands
status: completed
type: epic
priority: normal
created_at: 2026-02-26T05:57:56Z
updated_at: 2026-02-26T07:03:12Z
---

Remove EHS2 bucket-based postflop abstraction. Replace with two backends that index strategies directly by canonical hand (0-168): MCCFR with sampled concrete hands, and Exhaustive vanilla CFR with pre-computed equity tables.

## Summary of Changes\n\nAll 9 tasks completed across 6 commits on branch feature/169-hand-postflop:\n- Created postflop_hands.rs with shared utilities\n- Removed bucket fields from PostflopModelConfig, added Exhaustive variant\n- Removed StreetBuckets from PostflopAbstraction\n- Updated MCCFR backend for 169-hand direct indexing\n- Created Exhaustive backend with pre-computed equity tables\n- Deleted ~4,185 lines of dead code (ehs.rs, hand_buckets.rs, postflop_bucketed.rs)\n- Updated configs, docs, and integration tests
