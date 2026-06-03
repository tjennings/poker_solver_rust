---
# poker_solver_rust-zgkr
title: 'Phase 1 prep: differential harness for blueprint trainer tree migration'
status: todo
type: task
priority: high
created_at: 2026-06-03T18:10:21Z
updated_at: 2026-06-03T18:31:46Z
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
