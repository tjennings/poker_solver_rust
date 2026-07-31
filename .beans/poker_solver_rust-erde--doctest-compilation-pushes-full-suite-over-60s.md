---
# poker_solver_rust-erde
title: Doctest compilation pushes full suite over 60s
status: completed
type: bug
priority: high
created_at: 2026-06-10T20:39:58Z
updated_at: 2026-06-10T21:27:21Z
---

Warm full-suite runs measure ~87s wall (gate: 60s). Test execution is only ~36s across 31 binaries; the regression is uncacheable merged-doctest compilation: poker-solver-core has 3 doctests and range-solver has 7, each group forcing a ~10s rustdoc compile+link against the grown core crate on EVERY cargo test run (rustdoc merged-doctest binaries are not cached across runs). Fix: convert the 10 doctests to equivalent #[test] unit tests and mark the doc examples ```ignore so the examples remain in docs without per-run compilation. Discovered 2026-06-10 during Phase 4 integration.

## Summary of Changes

Eliminated uncacheable merged-doctest compilation that was adding ~21s to every warm test run (two ~10.5s doctest groups in poker-solver-core and range-solver, plus one in test-macros).

### Approach
1. Marked all 10 doc code fences as ```ignore so examples stay visible in rustdoc but are never compiled as doctests
2. Added 3 unit tests to preserve assertions from doctests that weren't already covered by existing tests: `doc_example_from_board_len`, `doc_example_bet_size_options`, `doc_example_range`
3. Added `[lib] doctest = false` to poker-solver-core, range-solver, and test-macros Cargo.toml files to prevent the merged-doctest binary compilation phase entirely (the `ignore` attribute alone still triggers a ~10s compile per crate)
4. 7 other doctests had identical assertions already covered by existing unit tests

### Before/After
- Before: ~87s warm, ~80s with ignore-only, **47s with doctest=false**
- After: 47.36s warm (well under 60s gate)
- All tests pass, zero failures, clippy clean
