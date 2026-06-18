---
# poker_solver_rust-z8jx
title: Full test-suite wall time exceeds 60s gate (giant debug test binaries)
status: todo
type: bug
priority: high
created_at: 2026-06-18T17:13:53Z
updated_at: 2026-06-18T17:13:53Z
---

Warm cargo test --workspace measures ~66-73s wall, over the 60s gate (CLAUDE.md). Diagnosis (2026-06-18): test EXECUTION is healthy and flat at ~29s across 30 binaries; the overage (~35-45s, highly variable with disk/cache contention) is binary load/startup. The deps are enormous debug test binaries: poker-solver-trainer 5x~148MB, cfvnet 5x~125MB, rebel ~116MB, tauri 3x~31MB — several GB loaded sequentially per run. Not caused by any one feature phase; cumulative. Fix directions (cited, with tradeoffs to weigh): (1) workspace [profile.test]/[profile.dev] debug tuning — debug="line-tables-only" or split-debuginfo="unpacked" or strip — to shrink debuginfo-dominated binaries (cargo profile docs; tradeoff: backtrace line info); (2) consolidate the 7 core + 3 tauri tests/*.rs integration files into fewer binaries (each tests/*.rs is a separate binary linking the whole crate — cargo book); (3) gate/feature-flag the heaviest crates' test binaries (cfvnet/trainer) out of the default fast suite. Measure each lever's wall-time delta. Goal: warm cargo test --workspace reliably under 60s.
