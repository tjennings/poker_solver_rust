---
# poker_solver_rust-erde
title: Doctest compilation pushes full suite over 60s
status: todo
type: bug
priority: high
created_at: 2026-06-10T20:39:58Z
updated_at: 2026-06-10T20:39:58Z
---

Warm full-suite runs measure ~87s wall (gate: 60s). Test execution is only ~36s across 31 binaries; the regression is uncacheable merged-doctest compilation: poker-solver-core has 3 doctests and range-solver has 7, each group forcing a ~10s rustdoc compile+link against the grown core crate on EVERY cargo test run (rustdoc merged-doctest binaries are not cached across runs). Fix: convert the 10 doctests to equivalent #[test] unit tests and mark the doc examples ```ignore so the examples remain in docs without per-run compilation. Discovered 2026-06-10 during Phase 4 integration.
