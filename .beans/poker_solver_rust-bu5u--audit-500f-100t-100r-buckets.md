---
# poker_solver_rust-bu5u
title: Audit 500f 100t 100r buckets
status: completed
type: task
priority: normal
created_at: 2026-05-14T01:01:40Z
updated_at: 2026-05-14T01:03:01Z
---

Run diagnostics on local_data/buckets/500f_100t_100r_v1 and summarize bucket quality, potential-aware behavior, nut-distance concerns, and hand-class anomalies.\n\n- [ ] Run cluster diagnostics\n- [ ] Review output for failures and suspicious assignments\n- [x] Summarize findings and follow-ups

## Summary of Changes\n\nAudited local_data/buckets/500f_100t_100r_v1 with diag-clusters, including hand-class audit over 200 sample boards and top 15 suspicious rows. No skipped lookups were reported. The audit shows heavy bucket-size skew, especially high-mass river/turn buckets, and repeated Trips/FullHouse strength-order warnings that likely reflect class-strength semantics/nut-distance gaps rather than lookup failures. A focused Kxs/Qxs profile did not reproduce the reported K8s vs Q6s/Q4s inversion: K8s has higher mean bucket labels and comparable or better percentiles on flop, turn, and river.
