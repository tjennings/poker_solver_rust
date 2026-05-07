---
# poker_solver_rust-hi1g
title: Compare 500-100-100 buckets against 500-50-50
status: completed
type: task
priority: normal
created_at: 2026-05-07T02:43:20Z
updated_at: 2026-05-07T02:47:56Z
---

Analyze the generated 500f_100t_100r bucket model and compare it against the first 500f_50t_50r set using available diagnostics and file-level checks.



## Summary of Changes

Compared local_data/buckets/500f_100t_100r_v1 against local_data/buckets/500f_50t_50r_v1 using diag-clusters size reports and transition audits.

Key results:
- 500/100/100 doubles turn and river bucket counts while assignment file sizes remain unchanged because bucket IDs are stored as u16 per board/combo.
- Turn to river potential consistency improved: separation ratio 9.51 vs 7.24 on a 50-board transition audit, with mean within-bucket EMD falling from 0.0351 to 0.0291.
- Flop to turn potential consistency regressed slightly: separation ratio 16.88 vs 17.67, with mean within-bucket EMD rising from 0.0162 to 0.0169.
- Turn and river bucket occupancy skew increased substantially in the 100/100 model; no empty buckets were reported.

Caveat: diff-clusters cannot directly compare different bucket counts and panics on the river 50 vs 100 mismatch. Raw equity audit remains suspect because the known canonical combo lookup issue still affects audit_bucket_equity.
