---
# poker_solver_rust-evag
title: Add comprehensive CFR loop tests for preflop/postflop
status: completed
type: task
priority: normal
created_at: 2026-02-26T15:07:38Z
updated_at: 2026-02-26T15:16:30Z
---

Add tests proving convergence, reach propagation, and regret application for both preflop and postflop CFR solvers. Construct small verifiable scenarios.


## Summary of Changes

Added `crates/core/tests/cfr_mechanics.rs` with 19 comprehensive tests covering:

**Convergence (4 tests):** Vanilla CFR exploitability monotonically decreases, MCCFR converges with full traversal and sampling, strategy stabilizes over training.

**Reach Propagation (4 tests):** Regrets accumulated at all reachable info sets, opponent reach correctly weights regret updates, hero reach correctly weights strategy sums, zero-reach nodes remain untouched.

**Regret Application (6 tests):** Positive regret increases action probability, negative regret decreases it, valid probability distributions maintained throughout training, DCFR discounting accelerates convergence with faster negative regret decay, regret signs match known optimal actions.

**Nash Equilibrium (2 tests):** Kuhn game value matches theoretical -1/18, MCCFR and vanilla CFR converge to the same equilibrium.

**MCCFR-Specific (3 tests):** Alternating traversal covers both players' info sets, average positive regret decreases with training, sample weights preserve unbiasedness.
