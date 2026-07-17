---
# poker_solver_rust-ql5z
title: Enable Tauri GameExplorer for universal MP storage
status: in-progress
type: bug
priority: high
created_at: 2026-07-17T16:31:40Z
updated_at: 2026-07-17T16:31:40Z
parent: poker_solver_rust-osss
---

The Tauri Game view recognizes universal MP lazy bundles and displays their manifest metadata, but startup still calls the HU-only GameSession path. That path rejects StrategySource::UniversalMp with: game_new requires a BlueprintV2 source (MP browsing not yet supported). Implement a lazy MP exploration/session path for the current N-player storage model, preserving the arena/universal row identity and avoiding full-tree materialization. The first user-visible gate is the 2-player universal_mp_lazy bundle shown in the current Tauri screenshots; keep the design extensible to N players.

## Checklist

- [ ] Research the existing LazyMpGame/LazyResolvedSpot and universal row query APIs.
- [ ] Add an MP session/state representation that advances the lazy public cursor and derives sparse lookup keys.
- [ ] Expose MP strategy/action rows through the Tauri game commands without routing them through BlueprintV2.
- [ ] Preserve existing HU GameSession behavior.
- [ ] Add focused Tauri regression coverage for loading, game_new, root state, and one action advance on a universal MP lazy bundle.
- [ ] Update explorer/training documentation for supported universal MP browsing scope.
- [ ] Run focused tests and the repository-approved full-suite verification.
