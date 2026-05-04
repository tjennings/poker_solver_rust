---
# poker_solver_rust-iueb
title: Audit depth-boundary CFV injection
status: in-progress
type: task
priority: high
created_at: 2026-05-04T01:09:34Z
updated_at: 2026-05-04T04:06:21Z
parent: poker_solver_rust-e90m
---

Trace compute_raw_cfvs_both output into depth-boundary regret updates. Verify injected CFVs affect the correct player/action with the expected sign and magnitude using a forced-choice diagnostic.



## Work Notes

Starting after reach-semantics audit fixed raw per-player boundary caching. The remaining exact_oracle gap is much smaller, so this step focuses on whether depth-boundary values are injected into regret and averaging equivalently to full-depth terminal values.

Audit checklist:

[ ] Map exact full-depth terminal evaluation path into regret deltas and strategy averaging.
[ ] Map depth-boundary raw evaluation path into the same regret deltas and strategy averaging.
[ ] Build a minimal no-choice/one-boundary diagnostic proving raw boundary terminal values equal equivalent terminal values.
[ ] Compare exact_oracle traces before and after a boundary hit for one high-divergence hand/action.
[ ] Document whether the remaining gap is injection semantics, iteration averaging, or expected finite-iteration behavior.
