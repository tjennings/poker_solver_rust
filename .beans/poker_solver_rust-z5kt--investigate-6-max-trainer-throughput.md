---
# poker_solver_rust-z5kt
title: Investigate 6-max trainer throughput
status: completed
type: task
priority: high
created_at: 2026-05-06T02:38:06Z
updated_at: 2026-05-06T02:41:11Z
---

Understand why N-player blueprint training throughput is around 7k hands/sec versus the heads-up solver's prior million+ hands/sec rate, and identify practical optimization options before implementation.

## Checklist\n\n- [x] Inspect N-player trainer hot path and throughput accounting\n- [x] Compare likely per-hand work versus heads-up trainer\n- [x] Identify optimization options with delivery order\n- [x] Report recommendations before implementation

## Summary of Findings\n\nN-player TUI throughput is meta-iterations/sec: one sampled 6-max deal plus one traversal per seat. Main improvement candidates are benchmarking without TUI telemetry, reducing MP action breadth, partial deal sampling, caching per-deal hand ranks and stack-allocation terminal payoff resolution, sampling/parallelizing traverser seats differently, reducing atomic pressure, and optimizing bucket lookup reuse.
