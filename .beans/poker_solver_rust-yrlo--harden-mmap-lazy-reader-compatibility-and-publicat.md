---
# poker_solver_rust-yrlo
title: Harden mmap lazy reader compatibility and publication safety
status: completed
type: bug
priority: high
created_at: 2026-07-28T14:16:36Z
updated_at: 2026-07-28T15:31:44Z
parent: poker_solver_rust-osss
---

Review follow-up for the UniversalMpLazy mmap reader.

- Define and enforce the immutable bundle publication assumption so post-map truncation/modification cannot cause an unsafe SIGBUS path; prefer validation/fallback behavior that preserves a safe error.
- Align mmap header/payload length handling with the existing BundleReader trailing-byte compatibility contract, or document and test an intentional format change.
- Preserve the public query API for MP lazy callers where feasible; avoid exposing an unnecessary incompatible view type or provide a compatible adapter.
- Re-run focused core/Tauri tests and document the final contract.

## Summary of Changes

- Replaced unsafe post-map access risk with owned payload snapshots and metadata stability checks.
- Preserved trailing-byte compatibility with BundleReader.
- Restored query_mp_lazy to Option<InfosetView> and added query_mp_lazy_owned for safe decoded storage.
- Corrected Explorer timing-prefix documentation.
- Deferred MP-lazy probability decoding to per-row OnceLock caches while retaining exact f32 bit patterns and strict validation.

## Verification

- cargo test -p poker-solver-core --test loader_unified: 30 passed
- cargo test -p poker-solver-core --test blueprint_universal_roundtrip: 23 passed
- cargo test -p poker-solver-tauri --test universal_explorer_integration: 20 passed
- Targeted rustfmt and git diff --check passed.
- Residual limitation: raw payload bytes and one cache slot per row remain resident; decoded row probabilities are retained after first visit. This is the follow-up target for bounded memory eviction.

## Benchmark\n\nThe representative 1,898,121-row bundle loaded in 3.221 seconds through the release devserver, down from 33.814 seconds. Reader phases were loading 70 ms, integrity 3,100 ms, validation 44 ms, and index 2 ms. The remaining startup bottleneck is integrity scanning.
