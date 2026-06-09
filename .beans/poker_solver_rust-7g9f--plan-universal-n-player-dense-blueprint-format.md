---
# poker_solver_rust-7g9f
title: Plan universal N-player dense blueprint format
status: completed
type: task
priority: high
created_at: 2026-06-09T18:35:50Z
updated_at: 2026-06-09T18:41:22Z
parent: poker_solver_rust-tzv5
---

Design a new dense blueprint strategy/snapshot format that can represent both heads-up blueprint_v2 and N-player blueprint_mp strategies. The plan must cover schema/versioning, player/seat metadata, game/action abstraction metadata, bucket metadata, row/action layout, dense eager and lazy sparse export, loader compatibility, Explorer/TUI/API integration, migration/compatibility, validation tests, and phased implementation beans. Non-goals for this planning bean: do not implement the format yet and do not change traversal/storage semantics.

## Planning Summary

Completed research and architecture planning for a universal dense blueprint format that supports both HU `blueprint_v2` and N-player `blueprint_mp` strategies.

Decision: use one universal directory-based dense blueprint envelope, not separate HU/MP formats. The old HU `BlueprintV2Strategy` remains a compatibility artifact, but the new primary format should be a versioned manifest plus explicit row/action/probability binary payloads.

Recommended artifact layout:

```text
bundle_or_snapshot/
  blueprint.json
  strategy.rows.bin
  strategy.actions.bin
  strategy.probs.f32.bin
  cfr.snapshot.bin       # optional, only when resumable/diagnostic state is complete
  checksums.json
  config.yaml            # retained for human/debug/back-compat context
  buckets/ or bucket_refs.json
```

Core schema requirements:

• Versioned manifest with format name/version, producer, compatibility features, checksums, and semantic fingerprints.
• Explicit game metadata: num players, seats, blinds/antes/straddles, stacks, button/dealer, rake, units.
• Explicit action abstraction and bucket metadata/fingerprints.
• Row descriptors with namespace (`hu_arena`, `mp_arena`, `mp_semantic`), acting seat, street, bucket, source node when applicable, semantic key when applicable, action/probability offsets, and action schema fingerprint.
• Action descriptors with stable machine-readable kind/amount/order, not display labels alone.
• Strategy payload is probabilities-first (`f32`, finite, normalized per row). Regrets/strategy sums are optional typed payloads with scaling metadata and only imply resumability when complete.
• Lazy MP semantic keys must be preserved verbatim; do not synthesize arena node IDs for lazy rows.
• Lazy MP universal export is analysis/read-only first. Resume remains out of scope until sparse snapshots persist blocked-edge purge state and runtime/cadence metadata.

Created child roadmap under feature `poker_solver_rust-a29s`:

• `poker_solver_rust-6op8` Phase 0: universal dense format spec.
• `poker_solver_rust-l75m` Phase 1: core universal format module.
• `poker_solver_rust-le5g` Phase 2: HU legacy-to-universal exporter.
• `poker_solver_rust-v1it` Phase 3: MP eager universal exporter.
• `poker_solver_rust-klpj` Phase 4: MP lazy sparse universal exporter.
• `poker_solver_rust-g54j` Phase 5: unified bundle detection and loader.
• `poker_solver_rust-9p59` Phase 6: Explorer and devserver universal format integration.
• `poker_solver_rust-0iye` Phase 7: trainer/cloud/docs migration.
