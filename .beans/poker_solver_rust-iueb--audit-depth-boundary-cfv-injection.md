---
# poker_solver_rust-iueb
title: Audit depth-boundary CFV injection
status: completed
type: task
priority: high
created_at: 2026-05-04T01:09:34Z
updated_at: 2026-05-04T04:16:08Z
parent: poker_solver_rust-e90m
---

Trace compute_raw_cfvs_both output into depth-boundary regret updates. Verify injected CFVs affect the correct player/action with the expected sign and magnitude using a forced-choice diagnostic.



## Work Notes

Starting after reach-semantics audit fixed raw per-player boundary caching. The remaining exact_oracle gap is much smaller, so this step focuses on whether depth-boundary values are injected into regret and averaging equivalently to full-depth terminal values.

Audit checklist:

[x] Map exact full-depth terminal evaluation path into regret deltas and strategy averaging.
[x] Map depth-boundary raw evaluation path into the same regret deltas and strategy averaging.
[x] Build a minimal no-choice/one-boundary diagnostic proving raw boundary terminal values equal equivalent terminal values.
[x] Compare exact_oracle iteration-split diagnostics across high-divergence hands/actions.
[x] Document whether the remaining gap is injection semantics, iteration averaging, or expected finite-iteration behavior.



Injection audit result:

[x] Mapped exact full-depth terminal evaluation path into regret deltas and strategy averaging.
[x] Mapped depth-boundary raw evaluation path into the same regret deltas and strategy averaging.
[x] Verified the existing one-boundary oracle contract still passes.
[x] Added compare-solve hidden iteration overrides to separate exact and subgame iteration counts.
[x] Ran canonical exact_oracle sweep: 200/200 delta +19.79, 1000/200 delta +135.41, 200/1000 delta +122.00, 1000/1000 delta +221.36 mbb/hand.
[x] Documented conclusion in docs/research/oracle_boundary_injection_audit_2026-05-04.md.

Conclusion: local raw boundary injection is unlikely to be the remaining primary bug. The multi-boundary gap behaves like a coupled-policy/final-average decoupling problem. Next diagnostic should feed iteration-aligned exact continuation values into the subgame instead of only the finalized exact average continuation.
