---
# poker_solver_rust-zgkr
title: 'Phase 1 prep: differential harness for blueprint trainer tree migration'
status: in-progress
type: task
priority: high
created_at: 2026-06-03T18:10:21Z
updated_at: 2026-06-03T18:33:18Z
parent: poker_solver_rust-34kn
blocking:
    - poker_solver_rust-kqpn
---

Preparatory subtask for Phase 1 of the blueprint trainer tree roadmap.

Scope:
- Build or extend a deterministic differential harness before replacing the map backend.
- Compare current map-backed traversal against the lazy tree path at legal actions, public state/action history, bucket keys, terminal payoffs, sampled opponent action under fixed RNG, regret deltas, strategy-sum deltas, and exported average strategy.
- Keep the fixture small enough for the full suite runtime target.
- Make mismatches actionable with node/history, player, street, board, action coordinates, strategy, regret, and utility deltas.

Acceptance criteria:
- A rust-developer can use this harness while implementing Phase 1.
- The harness can run with the old backend before the migration and with both backends during the migration.
- It is documented in the relevant trainer/testing docs if it adds a command or fixture.

This is a hard correctness guard before the lazy tree rewrite is allowed to replace the map path.


## Phase 0 Scope Correction

Phase 0 found that HU `blueprint_v2` already traverses an eager arena-style `GameTree`; the primary dense pressure point is CFR row allocation plus dense snapshot/export/resume compatibility. For this prep harness, treat the current eager+dense `blueprint_v2` backend as the oracle. The harness should compare current eager/dense behavior against future lazy/sparse behavior at legal actions, terminal values, bucket lookup, regret/strategy-sum deltas, and dense export output. Do not assume there is a hot-path map lookup in HU `blueprint_v2` that must be replaced.
