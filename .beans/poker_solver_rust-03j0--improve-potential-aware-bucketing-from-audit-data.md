---
# poker_solver_rust-03j0
title: Improve potential-aware bucketing from audit data
status: in-progress
type: epic
priority: high
created_at: 2026-05-14T01:12:59Z
updated_at: 2026-05-14T01:12:59Z
---

Top-level plan to use bucket audit data to improve potential-aware bucketing iteratively. Scope: convert diagnostics into regression metrics, add calibrated nut-distance features, keep potential-aware EMD intact, validate candidate bucket builds against strategy and abstraction quality, and promote only non-regressing models.\n\nAcceptance: each child task is complete or intentionally deferred; generated bucket candidates are compared to 500f_100t_100r_v1 with machine-readable diagnostics and short-run strategy checks.
