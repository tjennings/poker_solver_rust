---
# poker_solver_rust-tzv5
title: 'Trainer unification phase 0: preserve arena tree and establish shared runtime seam'
status: in-progress
type: feature
priority: high
created_at: 2026-06-09T14:55:16Z
updated_at: 2026-06-09T14:59:20Z
parent: poker_solver_rust-osss
---

First implementation slice for the HU/multiplayer trainer unification epic. Scope: research current HU arena/lazy tree and MP lazy traversal boundaries; produce a concrete shared runtime seam that preserves the new arena tree model; add golden/parity tests before behavior migration; identify how the HU TUI becomes the single TUI shell. Non-goal: do not rewrite full traversal or delete either trainer in this slice.

## Research Notes

Required research pass completed. Conclusion: do not merge MCCFR traversal bodies first. The first safe slice is a shared runtime/scheduler seam that calls existing HU and MP traversal engines unchanged.

Key invariants to protect:

- HU average-strategy accounting updates traverser nodes only; MP eager/lazy update sampled-node average strategy for both traverser and opponent decisions. A unified runner must not accidentally apply MP accounting to HU.
- HU chance semantics pre-deal the full board and advance chance nodes by street; MP lazy supports sampled full deals and exact continuation modes. Any shared runtime must make chance policy backend-owned, not global.
- HU sparse identity is arena-node keyed with action-schema fingerprint; MP lazy identity is semantic seat/street/bucket/action-history. Preserve both as opaque backend storage identities.
- HU regret-threshold pruning and MP zero-probability/negative-action pruning are distinct semantics. Do not collapse them behind one boolean.
- HU iteration and MP meta-iteration cadence differ. Shared runtime can report/drive cadence, but traversal backends must own what one unit of work means.

Golden tests recommended before migration:

- HU dense vs HU sparse differential remains mandatory.
- HU average-strategy accounting golden: opponent-sampled pass must not add HU opponent-node strategy sums.
- MP lazy average-strategy accounting golden stays intact.
- MP chance continuation fixed-seed goldens for all continuation modes.
- HU schema-fingerprint storage reuse rejection and MP long-history key behavior.
- HU vs MP pruning semantics remain distinct.
- HU baseline validation semantics remain unchanged.

First slice recommendation: add a shared runtime seam/adapters for scheduling, stop/snapshot/metric reporting, and TUI-facing telemetry; do not move traversal logic yet.
