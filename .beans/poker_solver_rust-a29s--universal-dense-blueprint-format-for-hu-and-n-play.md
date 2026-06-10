---
# poker_solver_rust-a29s
title: Universal dense blueprint format for HU and N-player
status: in-progress
type: feature
priority: high
created_at: 2026-06-09T18:40:56Z
updated_at: 2026-06-10T16:59:37Z
parent: poker_solver_rust-osss
---

Design and implement a versioned dense blueprint bundle format that can represent HU blueprint_v2 and N-player blueprint_mp strategies under one loader/exporter surface. The format must carry explicit game/player metadata, action schemas, bucket metadata, row identity, strategy payloads, checksums/fingerprints, compatibility flags, and read-only MP lazy sparse exports. Old HU bundles must remain readable. MP lazy resume is out of scope until snapshot schema includes blocked-edge purge/runtime cadence state.

## Scope Note (2026-06-10)

User guidance: no need to convert/migrate existing legacy-format data. Legacy bundles stay readable via existing untouched code paths, but conversion tooling is a convenience, not a contract. Later phases should target universal-only support for newly written artifacts; prefer trainers writing universal natively over polishing legacy converters.
