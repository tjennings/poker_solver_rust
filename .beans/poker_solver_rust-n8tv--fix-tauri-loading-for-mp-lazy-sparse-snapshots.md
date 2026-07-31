---
# poker_solver_rust-n8tv
title: Fix Tauri loading for MP lazy sparse snapshots
status: in-progress
type: bug
priority: high
created_at: 2026-06-24T18:25:48Z
updated_at: 2026-06-24T20:35:52Z
parent: poker_solver_rust-osss
blocked_by:
    - poker_solver_rust-8xsd
---

cargo tauri dev cannot load snapshots produced by the new Blueprint MP lazy_sparse backend. Two observed failures: (1) the trainer/snapshot writer does not copy the MP config.yaml to the blueprint root in the shape the Explorer/Tauri loader expects; (2) after manually copying the config, Tauri fails with game missing field 'players', indicating the loader is trying to deserialize an MP config or universal/lazy snapshot as a legacy HU BlueprintV2Config instead of accepting MP's game.num_players / universal metadata. Scope: make lazy_sparse snapshot/export output include the required config/metadata at the blueprint root; update Tauri/devserver bundle detection and config loading so MP lazy sparse bundles do not require HU-only game.players; preserve legacy HU loading; add regression coverage for loading an MP lazy_sparse snapshot/bundle through the Tauri/devserver loader path; update docs/explorer.md or docs/training.md if the expected bundle layout changes. Acceptance: a snapshot produced from sample_configurations/blueprint_mp_hu_500f_100t_100r.yaml loads in cargo tauri dev without manual config copying; no game.players error; existing HU bundles still load.

## Work Log

- 2026-06-24: Started investigation. Preflight will verify clean tree and full-suite timing before dispatching implementation.

## Checklist

- [x] Preflight full suite passes within the required time budget.
- [x] Identify the snapshot/export layout mismatch for MP lazy_sparse bundles.
- [x] Identify the Tauri/devserver config-loading path that assumes legacy HU game.players.
- [x] Implement bundle export/config metadata fix.
- [x] Implement Explorer loader compatibility for MP lazy_sparse bundles while preserving legacy HU bundles.
- [x] Add regression coverage.
- [x] Run focused verification.
- [ ] Run full-suite verification after resolving poker_solver_rust-8xsd.
- [x] Update docs if the bundle layout or user workflow changes.


## Diagnosis

- MP snapshot writers assumed root `config.yaml` already existed; tests manually created it, but training output may not.
- The HU MP lazy sample omitted `snapshots.format`, so it defaulted to legacy sparse output only.
- Universal MP lazy snapshots are written under `snapshot_NNNN/universal/`, but core/Tauri detection only checked root, `final/`, or direct `snapshot_NNNN/blueprint.json`.
- The Explorer snapshot path calls `load_blueprint_v2`, which parses `config.yaml` as HU `BlueprintV2Config` and fails on MP `game.num_players` configs with missing `game.players`.

## Implementation Notes

- MP eager and lazy snapshot saves now persist root `config.yaml` from the effective snapshot config before writing snapshot payloads.
- The HU MP lazy sample now uses `snapshots.format: both` so lazy sparse checkpoints keep `sparse_entries.bin` and also write Explorer-loadable nested universal bundles.
- Core and Tauri universal detection now recognize `snapshot_NNNN/universal/blueprint.json`; mixed direct/nested universal snapshots choose the newest snapshot overall while preserving final/root precedence and legacy HU behavior.
- `load_blueprint_v2_core` now delegates to universal loading for snapshot-specific nested universal bundles before attempting HU-only `BlueprintV2Config` parsing.
- Snapshot listing marks nested universal snapshots loadable and reads both `iterations` and `iteration`.
- Added regression coverage for root MP config plus nested lazy universal loading, snapshot-specific Explorer loading, mixed direct/nested newest selection, and legacy strategy fallback skipping snapshots without `strategy.bin`.

## Verification

- PASS: `git diff --check`.
- PASS: `cargo test -p poker-solver-core --test loader_unified` (25 passed; test runner 0.03s; wall 30.80s with rebuild).
- PASS: `cargo test -p poker-solver-tauri --test universal_explorer_integration` (8 passed; test runner 0.03s; wall 50.47s with relink).
- PASS: `cargo test -p poker-solver-trainer --bin poker-solver-trainer` (341 passed, 1 ignored; test runner 1.25s; wall 55.90s with rebuild).
- PASS: `cargo run -p poker-solver-trainer -- inspect-mp-config --config sample_configurations/blueprint_mp_hu_500f_100t_100r.yaml` parsed the sample and reported `Selected backend: lazy_sparse`.
- BLOCKED: broad `cargo test --workspace --quiet` and `cargo test --workspace --quiet --lib --tests` repeatedly stalled after many passing binaries with only Cargo visible; tracked as poker_solver_rust-8xsd.
