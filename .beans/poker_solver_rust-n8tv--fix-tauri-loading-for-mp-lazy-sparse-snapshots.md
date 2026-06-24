---
# poker_solver_rust-n8tv
title: Fix Tauri loading for MP lazy sparse snapshots
status: in-progress
type: bug
priority: high
created_at: 2026-06-24T18:25:48Z
updated_at: 2026-06-24T19:35:06Z
parent: poker_solver_rust-osss
---

cargo tauri dev cannot load snapshots produced by the new Blueprint MP lazy_sparse backend. Two observed failures: (1) the trainer/snapshot writer does not copy the MP config.yaml to the blueprint root in the shape the Explorer/Tauri loader expects; (2) after manually copying the config, Tauri fails with game missing field 'players', indicating the loader is trying to deserialize an MP config or universal/lazy snapshot as a legacy HU BlueprintV2Config instead of accepting MP's game.num_players / universal metadata. Scope: make lazy_sparse snapshot/export output include the required config/metadata at the blueprint root; update Tauri/devserver bundle detection and config loading so MP lazy sparse bundles do not require HU-only game.players; preserve legacy HU loading; add regression coverage for loading an MP lazy_sparse snapshot/bundle through the Tauri/devserver loader path; update docs/explorer.md or docs/training.md if the expected bundle layout changes. Acceptance: a snapshot produced from sample_configurations/blueprint_mp_hu_500f_100t_100r.yaml loads in cargo tauri dev without manual config copying; no game.players error; existing HU bundles still load.

## Work Log

- 2026-06-24: Started investigation. Preflight will verify clean tree and full-suite timing before dispatching implementation.

## Checklist

- [x] Preflight full suite passes within the required time budget.
- [x] Identify the snapshot/export layout mismatch for MP lazy_sparse bundles.
- [x] Identify the Tauri/devserver config-loading path that assumes legacy HU game.players.
- [ ] Implement bundle export/config metadata fix.
- [ ] Implement Explorer loader compatibility for MP lazy_sparse bundles while preserving legacy HU bundles.
- [ ] Add regression coverage.
- [ ] Run focused and full-suite verification.
- [ ] Update docs if the bundle layout or user workflow changes.


## Diagnosis

- MP snapshot writers assumed root `config.yaml` already existed; tests manually created it, but training output may not.
- The HU MP lazy sample omitted `snapshots.format`, so it defaulted to legacy sparse output only.
- Universal MP lazy snapshots are written under `snapshot_NNNN/universal/`, but core/Tauri detection only checked root, `final/`, or direct `snapshot_NNNN/blueprint.json`.
- The Explorer snapshot path calls `load_blueprint_v2`, which parses `config.yaml` as HU `BlueprintV2Config` and fails on MP `game.num_players` configs with missing `game.players`.
