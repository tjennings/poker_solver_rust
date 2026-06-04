---
# poker_solver_rust-l6r9
title: 'Phase 2: validate lazy tree trainer against known-good HU game'
status: in-progress
type: feature
priority: high
created_at: 2026-06-03T18:09:40Z
updated_at: 2026-06-04T03:06:14Z
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
