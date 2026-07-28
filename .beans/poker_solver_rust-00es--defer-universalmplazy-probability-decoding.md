---
# poker_solver_rust-00es
title: Defer UniversalMpLazy probability decoding
status: completed
type: bug
priority: high
created_at: 2026-07-28T15:04:50Z
updated_at: 2026-07-28T15:31:35Z
parent: poker_solver_rust-osss
---

The mmap/owned UniversalMpLazy path still calls decode_mp_lazy_probs for every row during load, allocating a full decoded probability vector. Keep the raw validated payload and decode only queried row ranges, while preserving query bit patterns, normalization checks, and public InfosetView/query_mp_lazy compatibility. Add focused equivalence and startup-allocation coverage.

## Summary of Changes

- Removed eager full-probability decoding from UniversalMpLazy load.
- Added per-row lazy probability decoding/cache with exact probability equivalence.
- Preserved borrowed and owned query APIs and strict normalization/checksum validation.

## Verification

- loader_unified: 30 passed
- universal_explorer_integration: 20 passed
- Targeted rustfmt and git diff --check passed.
- Raw payload copies and per-row cache metadata remain; bounded eviction is a future memory-pressure phase.

## Benchmark\n\nThe representative 1,898,121-row bundle loaded through the release devserver in 3.221 seconds. Integrity scanning was 3.100 seconds; loading was 70 ms, validation 44 ms, and index construction 2 ms. Prior observed load was 33.814 seconds.
