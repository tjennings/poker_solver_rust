---
# poker_solver_rust-eldv
title: Plan iterative bucket improvement
status: completed
type: task
priority: normal
created_at: 2026-05-14T01:09:39Z
updated_at: 2026-05-14T01:10:05Z
---

Write a plan for using 500f_100t_100r_v1 audit data to improve potential-aware bucketing and iterate safely.\n\n- [ ] Turn audit findings into metrics\n- [ ] Define bucketing algorithm changes\n- [x] Define iteration/validation loop

## Summary of Changes\n\nPrepared an implementation plan for improving potential-aware bucketing using the 500f_100t_100r_v1 audit data. The plan turns skipped lookups, bucket skew, class mixing, strength inversions, Kxs/Qxs profiles, and nut-distance preservation into acceptance metrics; proposes adding calibrated nut-distance features alongside EMD potential distributions; and defines an iterative generate/audit/train/compare loop with rollback criteria.
