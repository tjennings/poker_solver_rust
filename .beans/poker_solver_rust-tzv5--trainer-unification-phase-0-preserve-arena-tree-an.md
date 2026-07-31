---
# poker_solver_rust-tzv5
title: 'Trainer unification phase 0: preserve arena tree and establish shared runtime seam'
status: completed
type: feature
priority: high
created_at: 2026-06-09T14:55:16Z
updated_at: 2026-06-23T21:20:13Z
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

## 2026-06-23 Slice 1 Activation

Branch: `codex/training-runtime-seam`.

Preflight:

- Working tree clean before branch creation.
- Cold/noisy redirected full suite passed but missed the gate: `/usr/bin/time -p sh -c 'cargo test --workspace --quiet >/tmp/poker_solver_tzv5_preflight.log 2>&1'` -> `real 147.99`, `user 97.68`, `sys 18.41`.
- Hot redirected rerun passed under the gate: `/usr/bin/time -p sh -c 'cargo test --workspace --quiet >/tmp/poker_solver_tzv5_preflight_hot.log 2>&1'` -> `real 44.34`, `user 99.47`, `sys 14.83`.

Slice 1 checklist:

- [x] Dispatch fresh research review for the neutral runtime seam and invariants.
- [x] Dispatch fresh architecture/brainstorming review for the minimal API/test shape.
- [x] Implement runtime primitives and fake-backend tests only; do not touch traversal/TUI/snapshot formats.
- [x] Update docs only if the public architecture/training docs need to mention the new seam.
- [x] Run focused tests, diff hygiene, and hot full workspace suite under one minute.

## 2026-06-23 Runtime Seam Reconciliation Notes

Fresh research and architecture passes completed for this turn. Both confirmed the seam must be scheduler/runtime only: target/time limits, pause/quit/snapshot/telemetry/config-reload requests, unit counters, batch budget capping, stop reasons, event ordering, and backend trait are runtime-owned; traversal, chance policy, bucket lookup, regret/average-strategy accounting, pruning, storage identity, snapshots, TUI extraction, and validation remain backend-owned.

Important finding: the requested runtime seam work is already present in this branch ancestry rather than absent. Relevant commits found in history include:

- `3b34ee76 Add shared training runtime primitives`
- `9e4a2596 Add HU blueprint runtime adapter`
- `2bb84dda Add MP lazy shared runtime adapter`
- `89004a91 Wire MP lazy trainer through shared runtime`
- `9d9a2068 Add snapshot.format config flag (legacy | universal | both)`
- `d294c838 Fix review findings: error propagation, MP snapshot gating, TUI stub, dedup`

Current files include `crates/core/src/training_runtime.rs`, `crates/core/src/blueprint_v2/training_runtime_adapter.rs`, `crates/core/src/blueprint_mp/training_runtime_adapter.rs`, exported modules, and `docs/architecture.md` runtime documentation. The remaining work for this turn is validation/review and tracker closeout, not duplicating the implementation.

## Summary of Changes

- Reconciled Phase 0 Slice 1 against the current branch and confirmed the shared training runtime seam is already implemented in ancestry: `training_runtime.rs`, HU runtime adapter, MP lazy runtime adapter, MP lazy runtime wiring, snapshot format support, and architecture documentation are present.
- Re-ran the required research and architecture/brainstorming passes. Both confirmed the seam preserves the intended boundary: runtime owns scheduling/control/counters/telemetry/batch budgets, while traversal, chance policy, pruning, storage identity, snapshots, TUI extraction, and validation stay backend-owned.
- Verified that HU arena/sparse identities and MP lazy sparse semantic identities remain opaque to the runtime and are not interpreted by the shared seam.
- Closure review found no blockers. One deferred follow-up candidate remains: consolidate shared snapshot trigger/status flow later; current direct MP lazy TUI snapshot handling is consistent with the documented trainer-side boundary.

Verification:

- `cargo test -p poker-solver-core --lib training_runtime -- --nocapture` passed: 42 relevant runtime/adapter tests.
- `cargo test -p poker-solver-core --lib blueprint_v2::training_runtime_adapter -- --nocapture` passed: 4 HU adapter tests.
- `cargo test -p poker-solver-core --lib blueprint_mp::training_runtime_adapter -- --nocapture` passed: 16 MP lazy adapter tests.
- `git diff --check` passed.
- Full redirected workspace suite passed once with `real 283.79`, then immediate hot retry passed under the gate: `/usr/bin/time -p sh -c 'cargo test --workspace --quiet >/tmp/poker_solver_tzv5_final_hot2.log 2>&1'` -> `real 45.37`, `user 106.42`, `sys 15.21`.
