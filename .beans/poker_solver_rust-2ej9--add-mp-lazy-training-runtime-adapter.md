---
# poker_solver_rust-2ej9
title: Add MP lazy training runtime adapter
status: completed
type: task
priority: high
created_at: 2026-06-09T17:06:45Z
updated_at: 2026-06-09T17:29:15Z
parent: poker_solver_rust-tzv5
---

Add a multiplayer lazy sparse adapter for the shared training runtime. Scope: wrap the existing lazy MP training context with TrainingRuntimeBackend, preserve LazyMpGame traversal, sparse semantic storage identity, sampled/exact chance semantics, snapshot format, and existing training cadence. Seed RuntimeCounters from restored meta-iterations, cap batches with BatchBudget where applicable, bridge RuntimeControls into existing lazy MP quit/pause/snapshot/refresh surfaces or document missing pause support, and add focused adapter/parity tests. Non-goals: do not merge HU/MP traversal code, do not alter lazy_mccfr algorithms, do not change sparse storage keys, do not wire CLI/TUI yet.



## Research Notes

Completed lazy MP research pass. Key invariants:
- Runtime unit must be MetaIteration: one sampled deal plus one traversal per seat.
- Chance continuation modes (sampled full deal, sampled turn/exact river, sampled flop/exact turn+river) remain backend-owned.
- MP average strategy updates at both traverser and opponent sampled nodes; do not collapse into HU accounting.
- Lazy sparse storage identity is semantic key based (seat/street/bucket/history/action shape) with uniform missing-entry semantics; do not alter key or realization behavior.
- Negative-action subtree masking/purge is lazy-MP-specific and must stay inside lazy traversal/storage.
- Lazy MP snapshots currently live in trainer CLI code, not core; sparse_entries.bin format can be restored in memory but resume is not wired.
- Current LazyTrainContext has quit only; no core pause/snapshot/refresh/reload controls.

Research risks:
- Avoid double-counting ctx.iterations vs RuntimeCounters.
- Decide whether this slice moves lazy snapshot helpers into core or defers snapshot hooks until trainer-side integration.
- Config resume exists but current lazy MP path starts fresh; adapter should not pretend resume is complete unless loader is implemented.

## Architecture Notes

Completed architecture/brainstorming pass for the MP lazy adapter.

Decision: add `crates/core/src/blueprint_mp/training_runtime_adapter.rs` with `LazySparseMpTrainingRuntimeAdapter`, export it from `blueprint_mp::mod`, and extract a stateful lazy batch stepper from `blueprint_mp::trainer`.

Key constraints:

• Do not wrap `run_lazy_training` by repeatedly calling it; that would reset local `meta_iter`, pruning RNG, base iteration, and discount cadence.
• Keep `run_lazy_training` as a compatibility wrapper over the same extracted runner.
• Runtime unit is `TrainingUnit::MetaIteration`; do not multiply progress by player count.
• Adapter reports `TrainingBackendKind::MultiplayerLazySparse`.
• `RuntimeLimits.target_units` maps to `config.training.iterations`; `BatchBudget.remaining_target_units()` caps lazy batches.
• Runtime owns `RuntimeCounters`; adapter updates `ctx.iterations` only for existing MP telemetry compatibility.
• Bridge quit by sharing `RuntimeControls::quit_flag()` with `LazyTrainContext::quit`; pause remains runtime-owned between batches.
• Snapshot/reload should be explicit unsupported without injected hooks; do not fake resume support.
• Do not touch lazy traversal, sparse key identity, chance continuation semantics, or negative-action purge internals.

Required focused tests: backend kind/unit, counter seeding from `ctx.iterations`, budget capping without overshoot, `run_until_stopped` target completion without double counting, zero-target no allocation, quit-before-batch, explicit unsupported snapshot without hook, and adapter execution for sampled exact chance modes.

## Implementation Notes

Implemented the MP lazy sparse shared-runtime adapter slice.

Changes:

• Extracted `LazyMpTrainingStepper` from the old lazy training loop so lazy MP can run one capped batch at a time without resetting base iteration, pruning RNG cadence, chance-continuation mode, DCFR discount timing, or negative-action purge timing.
• Kept `run_lazy_training` as the compatibility wrapper over the new stepper for current CLI/TUI paths.
• Added `LazySparseMpTrainingRuntimeAdapter` implementing `TrainingRuntimeBackend` with `TrainingBackendKind::MultiplayerLazySparse` and `TrainingUnit::MetaIteration`.
• Runtime limits map to MP training iterations/time limits; batch budgets cap meta-iterations; runtime counters are seeded from `ctx.iterations` and left under shared-runtime ownership.
• Quit is bridged through the shared runtime control flag; pause remains runtime-owned between batches.
• Snapshot, resume, and config reload remain explicit unsupported core-adapter hooks until trainer-side integration owns real sparse snapshot/resume plumbing.
• Updated `docs/architecture.md` to document the shared runtime and lazy MP adapter boundary.

Review:

• Review agent found no correctness blockers. Residual risk is that the direct-vs-adapter parity test is shallow; code inspection covers the RNG/discount/purge sequencing more directly.

Verification:

• `cargo fmt --all --check`
• `cargo test -p poker-solver-core blueprint_mp::training_runtime_adapter --quiet`
• `cargo test -p poker-solver-core training_runtime --quiet`
• `git diff --check`
• `cargo test --quiet` passed in 43.578s locally, satisfying the under-one-minute project gate on this run.
