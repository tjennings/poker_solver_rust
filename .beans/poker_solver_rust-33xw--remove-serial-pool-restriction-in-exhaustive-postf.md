---
# poker_solver_rust-33xw
title: Remove serial pool restriction in exhaustive postflop CFR
status: completed
type: task
priority: normal
created_at: 2026-03-01T16:28:22Z
updated_at: 2026-03-01T16:36:14Z
---

Currently exhaustive_solve_one_flop creates a 1-thread rayon pool when num_flops > 1, limiting CPU utilization to num_flops cores. Remove the restriction to let rayon work-stealing handle nested parallelism, enabling full CPU utilization and dynamic reallocation when flops finish.

## Summary of Changes\n\nRemoved the 1-thread serial rayon pool from `exhaustive_solve_one_flop`. Inner `parallel_traverse_into` and `compute_exploitability` now use the global rayon pool, enabling work-stealing across all cores. Also fixed misleading 'MCCFR Solving' progress label.\n\nMerged to main.
