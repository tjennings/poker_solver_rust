---
# poker_solver_rust-ev78
title: Strengthen baseline validation sample and TUI coverage
status: todo
type: task
priority: low
created_at: 2026-06-04T04:10:54Z
updated_at: 2026-06-04T04:10:54Z
parent: poker_solver_rust-l6r9
---

Non-blocking follow-up from Phase 2 trainer/TUI integration review.

Scope:
- Add an exact parse/semantic test for `sample_configurations/blueprint_v2_hu_20bb_baseline_validation.yaml`, verifying stack 40, blinds 1/2, limp disabled, 169 preflop buckets, preflop rows `2.5bb` then `5bb`, baseline path, and validation cadence.
- Strengthen TUI rendering tests to assert representative top-5 worst-spot rows/diagnostics, not only aggregate baseline panel presence.

Non-goal: do not change baseline validation semantics or training behavior.
