---
# poker_solver_rust-93vj
title: Bucket audit regression scorecard
status: todo
type: task
priority: high
created_at: 2026-05-14T01:14:02Z
updated_at: 2026-05-14T01:14:02Z
parent: poker_solver_rust-03j0
---

Implement machine-readable regression metrics for bucket audits.\n\nScope:\n- skipped lookups\n- bucket size skew\n- mixed bucket entropy\n- equity span\n- strength-order inversions\n- nut-distance span\n- Kxs/Qxs sanity profile\n- potential-consistency/distortion\n\nAcceptance: diag-clusters can emit a stable JSON/CSV scorecard for before/after comparison.
