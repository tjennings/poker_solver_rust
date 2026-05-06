---
# poker_solver_rust-z5kt
title: Investigate 6-max trainer throughput
status: in-progress
type: task
priority: high
created_at: 2026-05-06T02:38:06Z
updated_at: 2026-05-06T02:38:42Z
---

Understand why N-player blueprint training throughput is around 7k hands/sec versus the heads-up solver's prior million+ hands/sec rate, and identify practical optimization options before implementation.

## Checklist\n\n- [ ] Inspect N-player trainer hot path and throughput accounting\n- [ ] Compare likely per-hand work versus heads-up trainer\n- [ ] Identify optimization options with delivery order\n- [ ] Report recommendations before implementation
