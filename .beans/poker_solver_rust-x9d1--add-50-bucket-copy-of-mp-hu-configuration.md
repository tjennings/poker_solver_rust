---
# poker_solver_rust-x9d1
title: Add 50-bucket copy of MP HU configuration
status: completed
type: task
priority: normal
created_at: 2026-08-04T12:58:19Z
updated_at: 2026-08-04T14:56:58Z
---

Create a copy of mp_hu_500f_100t_100r_nut_high_cap_0p5_v2 configured with 50 buckets on every street.

- [x] Research source configuration and naming conventions
- [x] Brainstorm and approve minimal configuration design
- [x] Implement the copied configuration in an isolated worktree
- [x] Review the implementation
- [x] Run the complete test suite (one-minute ceiling waived by user)
- [x] Update documentation if required
- [x] Integrate and summarize the change

## Summary of Changes

Added a reproducible HU Blueprint-MP preset with the supported 169 canonical preflop buckets and 50 buckets on each postflop street, plus the matching nut-high-cap-0.5 Blueprint-V2 clustering preset. Both use a fresh 50f_50t_50r_v1 bucket/output lineage; no incompatible trained artifacts were copied. Updated training documentation with cluster, inspect, and train commands and reuse warnings. Release config inspection reported 169/50/50/50, HU 100bb, lazy_sparse. The complete workspace suite passed in 288.23 seconds; the user explicitly waived the one-minute runtime requirement.
