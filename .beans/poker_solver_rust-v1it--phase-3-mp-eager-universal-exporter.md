---
# poker_solver_rust-v1it
title: 'Phase 3: MP eager universal exporter'
status: completed
type: task
priority: high
created_at: 2026-06-09T18:40:56Z
updated_at: 2026-06-10T18:48:07Z
parent: poker_solver_rust-a29s
---

Export MP eager dense snapshots into the universal format with explicit seat/action metadata. Acceptance: row descriptors include acting seat, street, bucket, arena node, ordered actions, and fingerprints; probabilities match MpStorage::average_strategy for known nodes; existing MP snapshot artifacts are preserved during migration.

## Note from Phase 2 (2026-06-10)

When implementing the MP eager exporter, consider introducing the spec's StrategyRowSource trait (now that there are 2+ export sources), harden action_abstraction_fingerprint away from Debug-format hashing, and add config-vs-storage bucket count cross-checks. See Phase 2 bean (le5g) deferred list.

## Summary of Changes

Implemented as `crates/core/src/blueprint_universal/mp_eager_export.rs` + shared `export_common.rs` + the `export-universal-mp` trainer subcommand (merged in the Phase 3 merge commit):
- In-memory export from live MpStorage::average_strategy (f64 -> f32 via `as f32`, matching the trainer projection) and disk export from snapshot dirs (bitwise f32 pass-through from projected strategy.bin), both through shared row machinery factored out of hu_export.rs (~150 lines deduplicated; chose shared functions over the spec StrategyRowSource trait for two call sites — revisit in Phase 4).
- Rows under mp_arena namespace with acting seat, street, bucket, arena node, ordered actions, row/action fingerprints; spec sort order exercised with 3-player tests.
- MP action amounts verified CHIP-denominated on all streets (game_tree.rs try_add_sized_action resolves preflop bb sizes to chips at build time) — no BB conversion, size_key "{v}chips".
- Phase 2 deferrals addressed: MP action-abstraction fingerprint hashes typed values (mp_action_abstraction_fingerprint_v1, no Debug-format hashing); config-vs-storage and config-vs-strategy bucket-count mismatches are hard errors.
- 38 tests added: average_strategy bitwise identity for every (node,bucket), disk-vs-in-memory byte identity, action mapping across streets, seat sort order, zero-mass uniform, precise rejections (incl. metadata kind \!= blueprint_mp).

Acceptance verified: row descriptors complete; probabilities match MpStorage::average_strategy; no changes to blueprint_mp/ or snapshot writing; HU exporter tests unchanged and passing. Full suite green in 45s warm.

## Deferred
- HU action_abstraction_fingerprint still Debug-based (informational; changing it would alter existing manifests).
- Minor nits from round 2: test-file compiler warnings, dead sb_bb_blinds helper, unused params, MpTrainingConfig test literal duplication.
