---
# poker_solver_rust-l6r9
title: 'Phase 2: validate lazy tree trainer against known-good HU game'
status: completed
type: feature
priority: high
created_at: 2026-06-03T18:09:40Z
updated_at: 2026-06-04T04:11:20Z
parent: poker_solver_rust-34kn
---

Phase 2 of the blueprint trainer lazy/sparse storage roadmap.

Scope:
- Use the smallest supplied heads-up cEV baseline: `local_data/baselines/cash_hu_20bb_cev.json` (`game.stackDepthBb = 20`, heads-up cash cEV, 2.5x open size).
- Train the smallest stack-size-equivalent HU blueprint configuration available in this repo, with pruning and disk eviction disabled.
- Extend the blueprint trainer validation path to compare learned preflop strategy against the supplied baseline's preflop spots.
- Extend the blueprint trainer TUI/progress surface to show convergence toward the baseline during training.
- Display the top 5 worst baseline mismatches, with enough data for each spot to diagnose the mismatch: spot key/path/history, position to act, actions, aggregate action deltas, per-combo worst deltas, learned frequencies, baseline frequencies, and a scalar mismatch score.
- Preserve Phase 1 dense export/resume compatibility and support the new sparse/lazy storage backend as an opt-in backend where practical.
- Update `docs/architecture.md` and `docs/training.md` if new commands/config/TUI behavior are added.

Acceptance criteria:
- A reproducible CLI/TUI validation run can train the smallest stack-size-equivalent HU blueprint and report convergence toward `cash_hu_20bb_cev.json`.
- Preflop strategy comparison uses documented action mapping/tolerances between trainer actions and baseline action labels.
- The TUI/progress output includes a baseline convergence metric and top 5 worst spots with diagnostic detail.
- Any mismatch report is actionable: spot/history, player/position, legal actions, baseline strategy, learned strategy, and score are visible.
- Full test suite passes in under 1 minute, or any runtime violation is fixed/beaned before proceeding.
- No Phase 3 pruning or Phase 4 disk eviction logic is enabled during validation.

Implementation must be delegated to rust-developer/worker agents; manager does not write Rust directly.

## Phase 2 Implementation Plan

Research/architecture consensus:

- Validate preflop strategy against `local_data/baselines/cash_hu_20bb_cev.json` as a cheap baseline comparison over `BlueprintCfrStorage`, not via range-solver or postflop EV validation.
- Use a 20bb-equivalent HU config in repo chip units: `stack_depth: 40`, `small_blind: 1`, `big_blind: 2`, `allow_preflop_limp: false`, `preflop.buckets: 169`, and preflop action rows `2.5bb` then `5bb`.
- Map baseline labels contextually: `F` to fold, `R2.5` to raise-to 5 chips, `R5` to raise-to 10 chips, `RAI` to aggressive all-in, and `C` to call or all-in-call depending on tree context.
- Compare only strategy frequencies in the first slice. Do not use baseline EV as a pass/fail signal because trainer EV and postflop abstraction assumptions differ.
- Primary metric is total variation distance over canonical preflop hands and mapped legal actions, weighted by combo count. Report aggregate TV, root TV, first-response TV, worst spot TV, coverage, skipped zero-mass rows, and top 5 worst spots with worst combo rows.
- Do not silently drop unsupported/unmapped actions; report unsupported spots or unmapped candidate mass.
- Keep validation cheap: six preflop spots times 169 canonical hands, no deal sampling, no range solving, no sparse-to-dense projection in the hot validation path.

Implementation split:

- Core validator slice: parser, spot resolver, context-aware action mapping, TV metrics, top-N report structs, unit tests.
- Trainer/TUI slice: config cadence, trainer progress/no-TUI logging, TUI metrics/rendering, sample 20bb validation config, docs, and integration tests.

## Summary of Changes

Phase 2 is complete. The blueprint trainer now supports preflop strategy-frequency validation against the supplied 20bb HU cEV baseline at `local_data/baselines/cash_hu_20bb_cev.json`.

Delivered:

- Added a reviewed `blueprint_v2` core baseline validator for the pinned six-spot HU preflop schema.
- Parses baseline metadata, action metadata, spot metadata, action summaries, and per-combo action frequencies.
- Resolves baseline preflop paths to `GameTree` nodes and maps baseline actions contextually: `F`, `R2.5`, `R5`, `RAI`, and all-in-call-as-`C`.
- Computes combo-weighted total variation distance over strategy frequencies, not EVs, with aggregate/root/first-response/worst-spot metrics, coverage counts, zero-mass row skip reporting, invalid row reporting, unsupported/unmapped action reporting, and top-N worst spot/combo diagnostics.
- Hardened the validator with exact preconditions: 169 preflop buckets, trusted original game config values from integration, stack 40 chips, SB 1, BB 2, limp disabled, pinned baseline metadata, and exact six baseline spots. Wrong-stack/wrong-blind/six-plus-extra baselines are refused before scoring.
- Added disabled-by-default `training.baseline_validation` config with baseline path, cadence, and top-N controls.
- Wired `BlueprintTrainer` to load/score the baseline on cadence against `active_storage()` without dense projection in the validation path.
- Extended no-TUI and TUI surfaces to show baseline convergence and top 5 worst spots with diagnostic data.
- Added `sample_configurations/blueprint_v2_hu_20bb_baseline_validation.yaml` for the smallest stack-equivalent validation run: 20bb in repo chip units, 1/2 blinds, no limp, 169 preflop buckets, `2.5bb` then `5bb`, validation enabled, and pruning/snapshots pushed beyond validation.
- Updated `docs/architecture.md` and `docs/training.md` with config, runbook, metrics, and limitations.

Verification:

- `cargo test -p poker-solver-core blueprint_v2::baseline_validation --quiet` passed.
- `cargo test -p poker-solver-core blueprint_v2::trainer::tests --quiet` passed.
- `cargo test -p poker-solver-trainer blueprint_tui --quiet` passed.
- `cargo test -p poker-solver-core blueprint_v2::config::tests::test_baseline_config --quiet` passed.
- `cargo test -p poker-solver-trainer --quiet` passed.
- Warm full-suite gate: `/usr/bin/time -p cargo test --quiet` passed in `real 49.21`, under one minute. A prior cold-ish run passed but measured `real 66.13`; the warm confirmation is the gate result.

Independent reviews completed:

- Core validator review found missing exact-config guards; fixed in `b1c98416` and `041f8982`.
- Final core validator review found no blockers and cleared trainer/TUI integration.
- Trainer/TUI integration review found no blockers and recommended closing Phase 2.

Known follow-up:

- `poker_solver_rust-ev78` tracks non-blocking test coverage improvements for exact sample YAML parsing and fuller top-5 TUI row assertions.
