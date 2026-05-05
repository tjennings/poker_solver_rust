---
# poker_solver_rust-ljo7
title: 'Blueprint action memory: sizing guardrails and budget reports'
status: in-progress
type: task
priority: high
created_at: 2026-05-05T15:01:15Z
updated_at: 2026-05-05T15:06:55Z
parent: poker_solver_rust-ohyt
---

Order 0 / foundation. Add deterministic sizing and memory-budget tooling before changing storage or action abstraction.

Implementation notes:
- Add a dry-run/sizing command or trainer startup mode that builds the tree/layout without entering training.
- Report decision nodes, slots by street, slots by action kind/depth, virtual bytes, touched/resident estimates, and 128GB pass/fail budget.
- Add config examples for current 6-max and at least one richer action candidate.
- Use this to compare every later task.

Acceptance criteria:
- Current sample_configurations/blueprint_mp_6max_200bkt.yaml reports the known baseline around 59.2M nodes, 23.9M decision nodes, 10.5B slots, 63GB virtual storage.
- A richer config can be evaluated without starting a long training run.
- docs/training.md documents the sizing workflow.
