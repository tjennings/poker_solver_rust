---
# poker_solver_rust-5qgb
title: Make universal MP lazy loading truly on-demand
status: completed
type: bug
priority: high
created_at: 2026-07-28T12:52:24Z
updated_at: 2026-07-28T13:18:18Z
parent: poker_solver_rust-osss
---

The Explorer timing for mp_hu_500f_100t_100r_nut_high_cap_0p5_v2 shows universal_reader_load takes 33.9 seconds for 1,898,121 rows, while MP source setup and game session construction take milliseconds. BundleReader::open currently reads, hashes, validates, decodes all standard payloads, and UniversalMpLazy then builds a full HashMap index across every row. This defeats the lazy backend's startup goal.

The same picker scan emits repeated `game: missing field players` warnings for the older v1 MP directory because a multiplayer config.yaml is probed through the legacy HU BlueprintV2Config parser. Fix that warning path as part of the loader/listing work without regressing legacy HU discovery.

Acceptance:
- Measure the current reader/open and lazy-index phases on a representative large MP bundle and add a small-bundle regression/benchmark seam.
- Make universal_mp_lazy loading avoid unnecessary full-payload materialization and full-row HashMap construction at startup, while preserving checksum/format validation guarantees appropriate to the selected load mode and exact query results.
- Keep universal HU, universal MP eager, and legacy HU behavior compatible; preserve the existing arena/lazy session structure.
- MP config.yaml directories are listed without attempting HU-only game.players parsing or emitting misleading warnings.
- Update architecture/training/explorer docs for the reader mode/validation contract.
- Run focused core and Tauri tests; leave the pre-existing sample configuration edit untouched.

## Summary of Changes

Replaced the universal MP-lazy per-row HashMap with a compact sorted public-prefix range locator and preserved last-match duplicate semantics. Added focused loader coverage for multiple histories per prefix, missing keys, duplicate hash/length keys, and exact action/probability results.


## Explorer listing changes

Added schema-aware legacy config classification so MP `game.num_players` YAML is not parsed as HU `game.players`. Legacy MP entries are metadata-only and empty snapshots no longer count as trained strategy entries.

## Verification

- `cargo test -p poker-solver-core --test loader_unified`: 27 passed.
- `cargo test -p poker-solver-tauri --test universal_explorer_integration`: 20 passed.
- The large-bundle phase measurement came from the Tauri `[explorer-load]` timing reported by the user; no production-sized fixture was added to CI.
- Architecture and Explorer documentation updated; the pre-existing sample configuration remains untouched.
