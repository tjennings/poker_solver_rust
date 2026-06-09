---
# poker_solver_rust-tzv5
title: 'Trainer unification phase 0: preserve arena tree and establish shared runtime seam'
status: in-progress
type: feature
priority: high
created_at: 2026-06-09T14:55:16Z
updated_at: 2026-06-09T15:02:54Z
parent: poker_solver_rust-osss
---

First implementation slice for the HU/multiplayer trainer unification epic. Scope: research current HU arena/lazy tree and MP lazy traversal boundaries; produce a concrete shared runtime seam that preserves the new arena tree model; add golden/parity tests before behavior migration; identify how the HU TUI becomes the single TUI shell. Non-goal: do not rewrite full traversal or delete either trainer in this slice.

## Research Notes

Required research pass completed. Conclusion: do not merge MCCFR traversal bodies first. The first safe slice is a shared runtime/scheduler seam that calls existing HU and MP traversal engines unchanged.

Key invariants to protect:

- HU average-strategy accounting updates traverser nodes only; MP eager/lazy update sampled-node average strategy for both traverser and opponent decisions. A unified runner must not accidentally apply MP accounting to HU.
- HU chance semantics pre-deal the full board and advance chance nodes by street; MP lazy supports sampled full deals and exact continuation modes. Any shared runtime must make chance policy backend-owned, not global.
- HU sparse identity is arena-node keyed with action-schema fingerprint; MP lazy identity is semantic seat/street/bucket/action-history. Preserve both as opaque backend storage identities.
- HU regret-threshold pruning and MP zero-probability/negative-action pruning are distinct semantics. Do not collapse them behind one boolean.
- HU iteration and MP meta-iteration cadence differ. Shared runtime can report/drive cadence, but traversal backends must own what one unit of work means.

Golden tests recommended before migration:

- HU dense vs HU sparse differential remains mandatory.
- HU average-strategy accounting golden: opponent-sampled pass must not add HU opponent-node strategy sums.
- MP lazy average-strategy accounting golden stays intact.
- MP chance continuation fixed-seed goldens for all continuation modes.
- HU schema-fingerprint storage reuse rejection and MP long-history key behavior.
- HU vs MP pruning semantics remain distinct.
- HU baseline validation semantics remain unchanged.

First slice recommendation: add a shared runtime seam/adapters for scheduling, stop/snapshot/metric reporting, and TUI-facing telemetry; do not move traversal logic yet.

## Architecture / Brainstorming Notes

Required architecture pass completed. It agrees with research: Phase 0 should add a shared runtime/scheduler layer that talks in generic training units while backend adapters continue to own poker semantics and call existing traversal engines unchanged.

Proposed seam:

- Add `crates/core/src/training_runtime.rs` for neutral runtime primitives: backend kind, training unit label, limits, controls, counters, batch outcome, telemetry sink, and runtime-backend trait.
- Later add `blueprint_v2::runtime` and `blueprint_mp::runtime` adapters; those adapters call current HU and MP traversal code rather than merging it.
- Later add trainer-side `unified_train.rs` and eventually a HU-TUI-based unified shell; initially keep `BlueprintTuiMetrics` as the shared bridge because HU and MP TUIs already use it.

Compatibility requirements:

- Preserve both CLI command names and config schemas in Phase 0.
- Preserve HU dense/sparse bundle and snapshot formats.
- Preserve MP eager and lazy snapshot formats.
- Preserve HU arena-node/action-schema identity and MP semantic seat/bucket/history identity as opaque backend keys.
- Preserve unit labels: HU iterations vs MP meta-iterations.

Files to avoid rewriting in Phase 0:

- `crates/core/src/blueprint_v2/mccfr.rs`
- `crates/core/src/blueprint_mp/mccfr.rs`
- `crates/core/src/blueprint_mp/lazy_mccfr.rs`
- `crates/core/src/blueprint_v2/game_tree.rs`
- `crates/core/src/blueprint_mp/game_tree.rs`
- HU/MP storage key semantics in sparse storage modules.

Child work slices:

1. Runtime primitives and fake-backend tests.
2. HU adapter around `BlueprintTrainer`, preserving arena/lazy tree and current traversal.
3. MP lazy adapter around `LazyTrainContext`, preserving chance mode, sparse identity, and negative-action telemetry.
4. MP eager adapter for compatibility only.
5. Unified CLI entry while preserving command names/configs.
6. HU-TUI-based unified shell with backend-specific scenario providers.
7. Shared snapshot/telemetry trigger/status flow with backend-owned serialization.

Decision for this turn: implement slice 1 only. Do not touch traversal logic or TUI shells yet.
