---
# poker_solver_rust-osss
title: Unify HU and multiplayer blueprint trainers
status: in-progress
type: epic
priority: high
created_at: 2026-06-09T13:03:45Z
updated_at: 2026-06-09T14:55:16Z
---

There are currently separate HU blueprint_v2 and multiplayer/6-max trainer paths. Plan and eventually migrate toward one trainer architecture across all player counts. This is intentionally large and requires a detailed architecture plan before implementation. The plan must cover shared game/tree abstractions, storage/key identity, traversal/sampling, action abstraction, snapshot/bundle format, TUI/metrics, validation baselines, migration compatibility, and incremental rollout slices.

## Detailed Plan

Problem statement: HU training currently lives in `blueprint_v2` (`crates/core/src/blueprint_v2/*`, `train-blueprint`) while 3-8 player training lives in `blueprint_mp` (`crates/core/src/blueprint_mp/*`, `train-blueprint-mp`). The two paths duplicate solver concerns but diverge in config shape, game-state representation, storage identity, traversal loop, snapshot format, TUI metrics, and CLI behavior. The desired end state is one blueprint trainer architecture that supports all player counts, with HU as the 2-player specialization rather than a separate trainer family.

Current split:

- CLI: `crates/trainer/src/main.rs` has separate `TrainBlueprint` and `TrainBlueprintMp` command paths.
- Config: `blueprint_v2::config::BlueprintV2Config` uses HU fields (`players`, `small_blind`, `big_blind`, HU action rows); `blueprint_mp::config::BlueprintMpConfig` uses N-player seats, forced bets, lead/raise split, lazy backend settings, and MP snapshot config.
- Tree/state: `blueprint_v2::game_tree::GameTree` eagerly materializes HU public tree; `blueprint_mp::game_tree::MpGameTree` eagerly materializes N-player tree; `blueprint_mp::lazy_mccfr::LazyMpGame` traverses compact public state on demand.
- Storage: HU dense/sparse uses `(decision node, bucket)` identity through `BlueprintCfrStorage`; MP dense uses `MpStorage`; lazy MP uses `SparseMpStorage` keyed by `InfoKey` / action history.
- Traversal: HU `blueprint_v2::mccfr` and `BlueprintTrainer` implement external sampling, baseline validation, BR/SAPCFR variants, dense-compatible snapshots, and HU TUI callbacks. MP has separate eager/lazy traversal in `blueprint_mp::mccfr`, `blueprint_mp::lazy_mccfr`, and `blueprint_mp::trainer`.
- Snapshots/bundles: HU writes dense-compatible `strategy.bin`, `regrets.bin`, CBVs, hand EVs, buckets, and metadata. Lazy MP writes sparse entries and metadata via trainer CLI helper code. Explorer/Tauri currently consume HU `blueprint_v2` bundle format.
- UI/metrics: HU TUI lives in `blueprint_tui*`; MP TUI lives in `mp_tui*`, with overlapping controls but different scenario resolution, telemetry, and snapshot status wiring.

Target architecture:

- Introduce a unified blueprint training kernel with an N-player public-state interface. Heads-up is represented as a 2-seat game config using the same seat/blind/action abstractions as multiplayer.
- Prefer the lazy public-state model as the canonical traversal substrate. Eager dense tree/storage can remain as a compatibility/testing backend, but should not be the conceptual owner of trainer behavior.
- Define shared traits or structs for:
  - `BlueprintGameSpec`: seats, blinds/antes, stack units, limp policy, action abstraction, street bucket counts.
  - `PublicState`: legal actions, advance, terminal evaluation boundary, current player/seat, street, pot/stacks, all-in/fold state.
  - `BucketProvider`: street-aware bucket lookup over shared `blueprint_v2` bucket files/per-flop files.
  - `CfrStorage`: current strategy, average strategy, regret update, strategy-sum update, discounting, snapshot/export hooks.
  - `TrainerRuntime`: batch loop, pause/quit/snapshot triggers, metric callbacks, validation hooks, and cadence scheduling.
- Preserve existing HU bundle compatibility during migration. Existing Explorer/Tauri consumers should continue to load old `blueprint_v2` bundles until a new versioned universal bundle format is deliberately introduced.
- Make baseline validation a plugin-style trainer diagnostic over a storage/provider interface, not HU trainer-specific logic. The current 20bb HU preflop baseline remains a 2-player validation module.

Migration phases:

1. Inventory and invariants
   - Document exact behavioral contracts of HU `BlueprintTrainer` and MP lazy trainer: config defaults, action labels, snapshot metadata, iteration accounting, storage scaling, pruning, validation, and TUI metrics.
   - Add golden smoke tests for current HU 20bb baseline validation and MP lazy sparse 2-player/6-player one-iteration runs before migration begins.

2. Shared config model without behavior changes
   - Add conversion/adaptation layer between `BlueprintV2Config` and a unified `BlueprintGameSpec`.
   - Add conversion/adaptation layer between `BlueprintMpConfig` and the same `BlueprintGameSpec`.
   - Keep both CLI commands operational, but make both print the normalized game spec for auditability.

3. Shared bucket/action/runtime utilities
   - Move shared bucket loading, bucket validation, cadence scheduling, snapshot candidate selection, and metric trigger code into reusable modules.
   - Replace duplicated HU/MP snapshot scheduling semantics with one tested scheduler.
   - Keep traversal algorithms unchanged in this phase.

4. Unified storage provider boundary
   - Define a storage trait that can represent HU dense, HU sparse, MP dense, and MP sparse/lazy rows.
   - Add dense export adapters for old HU bundles and sparse export adapters for MP lazy checkpoints.
   - Prove HU dense and HU sparse still produce equivalent strategy reads under existing tests.

5. Make lazy public-state traversal support HU parity
   - Extend or adapt `LazyMpGame` so a 2-seat game with HU action config can reproduce HU legal actions, betting sequence semantics, and terminal payoffs.
   - Build a 2-player lazy traversal smoke that validates root and key preflop action labels against the current HU tree.
   - Compare aggregate baseline TV progression between old HU trainer and unified lazy 2-player trainer on the 20bb baseline sample.

6. Move HU trainer onto unified runtime
   - Keep `train-blueprint` as a compatibility command, but route it through unified runtime for 2-player configs once parity tests pass.
   - Preserve old bundle output or emit a versioned compatibility export.
   - Keep `blueprint_v2` code available behind tests until Explorer and validation consumers are migrated.

7. Move MP eager/lazy trainer onto unified runtime
   - Route `train-blueprint-mp` through the same runtime loop, cadence scheduler, snapshot service, metric bus, and storage abstraction.
   - Retain lazy sparse as the default scalable backend for 6-max.
   - Validate 6-max smoke configs and negative-action prune telemetry against current behavior.

8. CLI and TUI consolidation
   - Decide whether `train-blueprint` should accept all player counts and deprecate `train-blueprint-mp`, or whether both commands remain aliases over one runtime for a deprecation window.
   - Merge HU/MP TUI shared widgets and metrics while preserving domain-specific scenario resolvers.
   - Align manual snapshot status, pause behavior, scenario refresh, and baseline/telemetry panels.

9. Bundle/explorer migration
   - Design a versioned universal blueprint bundle with game spec, seat map, action schema, storage backend metadata, strategy export, and optional sparse checkpoint payloads.
   - Update Explorer/Tauri/devserver loaders to read both legacy HU bundles and universal bundles.
   - Add migration docs and compatibility tests.

10. Retirement
   - Once parity and compatibility are proven, retire duplicate trainer loops and stale CLI branches.
   - Keep narrow compatibility loaders for legacy bundles, but remove duplicate traversal/runtime code where possible.

Non-negotiable tests:

- HU 20bb baseline validation still runs against `local_data/baselines/cash_hu_20bb_cev.json` and reports the same schema/coverage semantics.
- Existing HU dense and sparse snapshot/resume compatibility remains intact.
- 2-player unified lazy game can represent fold/call/raise/all-in semantics matching HU preflop baseline spots.
- 6-max lazy sparse smoke remains under the one-minute suite gate.
- Snapshot scheduling/resume behavior is shared and tested once, not separately reimplemented.
- Bundle loaders accept old HU bundles and any new universal bundle format during migration.

Risks:

- HU and MP action abstractions are not merely syntactic variants; HU uses compact per-depth rows while MP distinguishes lead/raise and multi-seat closing action constraints.
- Storage identity migration is the hardest correctness problem. Node-indexed HU dense storage is not naturally compatible with MP action-history keys.
- Explorer compatibility can become a hidden blocker if universal bundles are introduced before readers are ready.
- Performance regressions are likely if the unified layer erases hot-path details. Traits should be compile-time/generic or localized outside inner traversal loops where possible.
- Do not migrate pruning/eviction experiments at the same time as trainer unification unless the slice explicitly owns that risk.

Recommended first child beans:

- Architecture spike: write a concrete `BlueprintGameSpec` / `PublicState` / `CfrStorage` proposal with compatibility matrix.
- Golden tests: lock current HU baseline validation and MP lazy sparse smoke behavior.
- Shared snapshot scheduler: extract and test common snapshot/resume semantics before touching traversal.
- Config adapter: normalize HU and MP config into a shared audited game spec without changing training behavior.

## Current Direction

User requested starting the trainer merge now, with two explicit constraints:

- Retain the new arena/lazy tree structure built in the recent blueprint trainer phase; do not collapse back to map-based/eager-only traversal as part of unification.
- End with one TUI. The HU TUI can be kept as the base and iterated into the unified TUI rather than introducing a third dashboard.

Workflow note: the project instructions call for `hex:brainstorming`, but no callable `hex`/brainstorming tool is available in this environment. Use delegated research/architecture agents plus bean planning as the fallback brainstorming record.
