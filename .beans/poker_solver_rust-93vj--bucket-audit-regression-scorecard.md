---
# poker_solver_rust-93vj
title: Bucket audit regression scorecard
status: in-progress
type: task
priority: high
created_at: 2026-05-14T01:14:02Z
updated_at: 2026-05-14T01:56:49Z
parent: poker_solver_rust-03j0
---

Implement machine-readable regression metrics for bucket audits.\n\nScope:\n- skipped lookups\n- bucket size skew\n- mixed bucket entropy\n- equity span\n- strength-order inversions\n- nut-distance span\n- Kxs/Qxs sanity profile\n- potential-consistency/distortion\n\nAcceptance: diag-clusters can emit a stable JSON/CSV scorecard for before/after comparison.

## Implementation Notes

Started implementation with a stable `diag-clusters --scorecard-json <PATH>` output. The scorecard currently includes bucket-size skew metrics for all loaded streets plus hand-class audit metrics when `--hand-class-audit` is enabled: skipped lookups, class/strength spread summaries, mixed-bucket entropy/equity spans, and strength-order inversion summaries.

Verified with `cargo check -p poker-solver-trainer` and a smoke run against `local_data/buckets/500f_100t_100r_v1` using 20 sampled hand-class boards.

Added selected Kxs/Qxs suited-hand profiles and sampled river nut-distance span metrics to the scorecard. Remaining in this task: add potential-consistency/distortion once that signal is exposed.
