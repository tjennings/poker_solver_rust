---
# poker_solver_rust-28c2
title: 'Phase 2 slice: core preflop baseline validator'
status: in-progress
type: task
priority: high
created_at: 2026-06-04T03:06:14Z
updated_at: 2026-06-04T03:34:38Z
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

## Review Findings 2026-06-04

Independent review of `d243086c Add blueprint baseline validation core` found the core slice is not ready for trainer/TUI integration.

Blocking fixes required:

- Add exact preflop bucket validation. The validator currently accepts a provider with only `average_strategy`, then uses canonical hand index directly as bucket. It must refuse non-169 preflop storage/config before scoring, or expose the bucket count through the provider boundary.
- Add exact game/tree validation for the supplied baseline. The validator must refuse non-20bb-equivalent trees/configs, especially trees where `RAI` would mean a different all-in stack size. At minimum validate starting stack 40 chips, big blind 2 chips, and expected baseline metadata/action schema before scoring.

Additional fix:

- Malformed/unparsable baseline hand rows must be reported, not silently dropped from aggregate TV/weights.

Reviewer noted that under the exact target tree the six path tests and action mapping are directionally correct, including all-in-call mapping to `C`, but the missing guards can produce plausible-looking validation numbers for the wrong blueprint.

## Corrective Patch Summary

Addressed review blockers for the Phase 2 core baseline validator:

- Added provider preflight validation for exact 169 preflop buckets before scoring.
- Added global precondition failures to the validation report and refuse scoring when preconditions fail.
- Validated the pinned HU 20bb baseline/tree tuple before scoring: tree starting stack 40 chips, validation big blind 2 chips, baseline `stackDepthBb: 20`, baseline `openingSize: 25x`, document action metadata, and the six expected preflop spot schemas/action mappings.
- Reported malformed baseline hand labels through aggregate counters and `invalid_hand_rows` instead of silently dropping them.
- Added regression coverage for non-169 providers, wrong-stack trees that still map root actions, bad baseline metadata, and malformed hand labels.

Verification:

- `cargo test -p poker-solver-core blueprint_v2::baseline_validation --quiet`
- `cargo test -p poker-solver-core blueprint_v2 --quiet`
- `/usr/bin/time -p cargo test --quiet` (first post-change full run passed at real 138.74s after rebuild; warmed run passed at real 47.91s)

## Corrective Review Findings 2026-06-04

Independent review of `b1c98416 Harden baseline validation preconditions` found two remaining issues before trainer/TUI integration can proceed.

Blocking fix required:

- Wrong-tree blind semantics can still pass preflight. `GameTree` does not retain the small/big blind values used at construction, and the validator only checks `BaselineValidationConfig.big_blind`. A tree built with different blinds but adjusted preflop size strings can expose the same chip raise amounts and pass schema checks while representing a different BB stack depth. The validator must receive and verify trusted game config/blind metadata, or otherwise make the trainer integration pass and validate the original `GameConfig` before scoring.

Additional fix:

- The pinned baseline spot check must be exact. The current check verifies the six expected spots exist, but does not reject extra baseline spots; extra resolvable spots could pollute aggregate TV metrics.

The review confirmed that non-169 bucket preflight and malformed-hand reporting now work, and previous action mapping behavior remains intact.
