---
# poker_solver_rust-zgkr
title: 'Phase 1 prep: differential harness for blueprint trainer tree migration'
status: in-progress
type: task
priority: high
created_at: 2026-06-03T18:10:21Z
updated_at: 2026-06-03T18:48:19Z
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


## Implementation Plan

Research/brainstorming result for the first harness increment:

- Build around `blueprint_v2::mccfr::traverse_external`, not `BlueprintTrainer::train`, because trainer execution is parallel/atomic and can obscure exact differential comparisons.
- Start in `crates/core/src/blueprint_v2/mccfr.rs` inside the existing test module so the harness can reuse private fixtures and terminal helpers.
- Add a small test-only backend/comparison shape that compares eager+dense against eager+dense today and can accept a lazy/sparse adapter later.
- Use a tiny deterministic HU fixture: toy tree, `[10, 10, 10, 10]` buckets, fixed deals, fixed traversers, fixed RNG seeds, pruning/baselines disabled.
- Seed oracle and candidate storage with identical deterministic non-zero regrets so uniform strategies do not mask row/action-coordinate bugs.
- Compare legal action order before traversal, then traversal EV, regret deltas, strategy-sum deltas, dense average-strategy export, and dense regret save/load round trip.
- Keep this in the default suite and verify it remains fast.

Primary implementation target: `cargo test -p poker-solver-core blueprint_v2::mccfr::tests::differential_harness --quiet` or an equivalent focused filter.

## Summary of Changes

- Added `blueprint_v2::mccfr::tests::differential_harness_eager_dense_self_check`, a deterministic eager+dense self-check around `traverse_external`.
- The harness seeds oracle and candidate dense storage with identical non-zero regrets and strategy sums, compares legal action order before traversal, then checks traversal EV, prune/sample stats, regret deltas, strategy-sum deltas, dense average-strategy export, and dense `regrets.bin` save/load round trip.
- Failure diagnostics decode dense slots back to node/history, player, street, board, bucket, action index, and action label so future lazy/sparse adapter mismatches are actionable.
- Kept the change scoped to `crates/core/src/blueprint_v2/mccfr.rs`; no storage, bundle, docs, or lazy backend changes were needed.

## Verification

- `cargo test -p poker-solver-core blueprint_v2::mccfr::tests::differential_harness_eager_dense_self_check --quiet` passed.
- `cargo test -p poker-solver-core blueprint_v2::mccfr --quiet` passed: 73 passed.
- `/usr/bin/time -p cargo test --quiet` passed warm in `real 40.60`.


## Review Findings 2026-06-03

Independent review found the first harness increment is not sufficient to complete this bean. Required follow-up before Phase 1 can proceed:

- Add an actual backend/adapter trait or equivalent candidate injection point so the harness can accept a future lazy/sparse backend instead of only comparing two concrete `DenseMccfrHarnessBackend` instances.
- Trace and compare sampled opponent actions under fixed RNG, not only final EV/deltas.
- Improve legal-action/public-state comparison so it is not purely dense arena-index based and includes child/terminal semantic context where practical.
- Improve traversal mismatch diagnostics with strategy vectors and utility/action-value context at the point of failure.

The bean was reopened from completed to in-progress until these review items are addressed.
