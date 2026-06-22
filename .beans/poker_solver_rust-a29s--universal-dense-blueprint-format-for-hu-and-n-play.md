---
# poker_solver_rust-a29s
title: Universal dense blueprint format for HU and N-player
status: completed
type: feature
priority: high
created_at: 2026-06-09T18:40:56Z
updated_at: 2026-06-22T18:13:40Z
parent: poker_solver_rust-osss
---

Design and implement a versioned dense blueprint bundle format that can represent HU blueprint_v2 and N-player blueprint_mp strategies under one loader/exporter surface. The format must carry explicit game/player metadata, action schemas, bucket metadata, row identity, strategy payloads, checksums/fingerprints, compatibility flags, and read-only MP lazy sparse exports. Old HU bundles must remain readable. MP lazy resume is out of scope until snapshot schema includes blocked-edge purge/runtime cadence state.

## Scope Note (2026-06-10)

User guidance: no need to convert/migrate existing legacy-format data. Legacy bundles stay readable via existing untouched code paths, but conversion tooling is a convenience, not a contract. Later phases should target universal-only support for newly written artifacts; prefer trainers writing universal natively over polishing legacy converters.

## Scope Update (2026-06-10)

Supersedes 'Old HU bundles must remain readable': per the user's goal recap on the parent epic, the only compatibility obligation is the Tauri Explorer's ability to load exported strategies. Legacy readability is transitional convenience only. num_players target widened to 2-10 (spec updated).

## Summary of Changes

All 8 phases of the universal dense blueprint format are complete (merged to codex/blueprint-lazy-tree-roadmap):
- Phase 0 (6op8): spec (docs/blueprint_format.md)
- Phase 1 (l75m): core format module (manifest + binary payloads + CRC-64/SHA-256 + validating reader)
- Phase 2 (le5g): HU legacy->universal exporter (export-universal)
- Phase 3 (v1it): MP eager exporter (export-universal-mp)
- Phase 4 (klpj): MP lazy sparse exporter (semantic side table, Opaque actions)
- Phase 5 (g54j): unified bundle detection + loader (LoadedBundle enum)
- Phase 6 (9p59): Explorer + devserver load universal bundles (config.yaml retained; universal HU renders via reconstruction; MP read-only)
- Phase 7 (0iye): trainers write universal natively behind snapshot.format flag (all 3 backends, byte-identical to post-hoc) + train dispatcher + docs

The broader trainer consolidation (one TUI, 2-10 players, retire HU/eager training) continues under the parent epic osss via beans 8jan/nbzo/wa53/rp9r/hoq8/lu4f/tzv5/mt3l. The only hard external compatibility surface is the Tauri Explorer loading exported strategies. Suite wall-time fragility tracked in z8jx.
