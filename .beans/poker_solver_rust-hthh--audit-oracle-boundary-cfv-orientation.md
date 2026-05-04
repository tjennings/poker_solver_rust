---
# poker_solver_rust-hthh
title: Audit oracle-boundary CFV orientation
status: completed
type: task
priority: high
created_at: 2026-05-04T01:09:23Z
updated_at: 2026-05-04T01:40:05Z
parent: poker_solver_rust-e90m
---

Trace OOP/IP and traverser CFV orientation at depth boundaries. Compare current, swapped, sign-flipped, and swapped+sign-flipped variants to identify whether a player/sign contract mismatch explains the exact_oracle divergence.

## Work Notes

Starting step 3. Goal: audit whether exact_oracle divergence is explained by OOP/IP swap, sign flip, or swapped+sign-flipped CFV orientation at depth boundaries.



## Completion Notes

Completed Step 3 orientation audit. Added hidden compare-solve --oracle-orientation diagnostic for exact_oracle with modes current, swap, sign-flip, and swap-sign-flip; documented results in docs/research/oracle_boundary_orientation_audit_2026-05-04.md.

Canonical JhTh9h7d exact_oracle run at 200 iters did not reveal a valid simple OOP/IP or sign orientation fix: current exp_delta +2837.10, swap +18525.64, sign-flip +611.99 but root strategy mass worsened to 0.677, swap-sign-flip +34607.98.

Conclusion: orientation is unlikely the root cause; proceed to units/normalization, reach semantics, or regret injection audits.

Verification: cargo test -p poker-solver-trainer compare_solve; cargo build --release -p poker-solver-trainer; warm time cargo test in 53.040s.
