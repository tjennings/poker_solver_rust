---
# poker_solver_rust-00es
title: Defer UniversalMpLazy probability decoding
status: in-progress
type: bug
priority: high
created_at: 2026-07-28T15:04:50Z
updated_at: 2026-07-28T15:04:50Z
parent: poker_solver_rust-osss
---

The mmap/owned UniversalMpLazy path still calls decode_mp_lazy_probs for every row during load, allocating a full decoded probability vector. Keep the raw validated payload and decode only queried row ranges, while preserving query bit patterns, normalization checks, and public InfosetView/query_mp_lazy compatibility. Add focused equivalence and startup-allocation coverage.
