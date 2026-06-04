---
# poker_solver_rust-28c2
title: 'Phase 2 slice: core preflop baseline validator'
status: completed
type: task
priority: high
created_at: 2026-06-04T03:06:14Z
updated_at: 2026-06-04T03:16:28Z
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

## Summary of Changes

Implemented the Phase 2 core preflop baseline validator slice in `blueprint_v2`:

- Added baseline JSON parser structs for game/action/spot/action-summary/per-combo strategy data with serde defaults and unknown-field preservation.
- Added preflop spot path resolution for the six target HU baseline spots against the exact 20bb-equivalent tree config.
- Added context-aware action mapping for `F`, `C`, `R2.5`, `R5`, and `RAI`, including all-in-call response mapping to baseline `C`.
- Added storage-compatible validation over `average_strategy(node, bucket)` via a small provider trait.
- Added combo-weighted total variation metrics, zero-mass row skipping, unsupported action/schema reporting, unmapped candidate mass reporting, and top-N report structs.
- Added focused unit tests for parsing, spot resolution, action schemas, all-in-call mapping, zero-mass rows, and limp-enabled unsupported root schema.

Verification:

- `cargo test -p poker-solver-core blueprint_v2::baseline_validation --quiet`
- `cargo test -p poker-solver-core blueprint_v2 --quiet`
- `/usr/bin/time -p cargo test --quiet` (cached steady-state: real 45.27s)
