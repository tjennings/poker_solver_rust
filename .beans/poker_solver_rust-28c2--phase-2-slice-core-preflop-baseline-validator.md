---
# poker_solver_rust-28c2
title: 'Phase 2 slice: core preflop baseline validator'
status: in-progress
type: task
priority: high
created_at: 2026-06-04T03:06:14Z
updated_at: 2026-06-04T03:06:14Z
parent: poker_solver_rust-l6r9
---

Core implementation slice for Phase 2 baseline validation.

Scope:
- Add a `blueprint_v2` baseline validation module for the supplied GTO-style JSON schema.
- Parse preflop baseline spots, action metadata, per-combo action frequencies, aggregate action summaries, and game metadata.
- Resolve baseline preflop path labels to `GameTree` nodes for the 20bb-equivalent HU config.
- Implement context-aware action mapping for `F`, `C`, `R2.5`, `R5`, and `RAI`, including all-in-call mapping to baseline `C`.
- Compute total variation distance metrics over canonical hands/actions using combo weights, skipping zero-mass baseline rows and reporting unsupported/unmapped cases explicitly.
- Produce report structs with aggregate convergence metrics and top-N worst spots plus worst combo rows.
- Add unit tests using inline fixtures and/or cheap tree fixtures: parse baseline shape, six spot path resolution under exact config, root action schema, all-in-call maps to `C`, zero-mass rows are skipped, limp-enabled config reports unsupported root schema.

Non-goals: no TUI rendering, no trainer cadence wiring, no range-solver validation, no EV pass/fail, no pruning or disk eviction.
