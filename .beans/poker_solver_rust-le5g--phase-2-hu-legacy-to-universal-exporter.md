---
# poker_solver_rust-le5g
title: 'Phase 2: HU legacy-to-universal exporter'
status: completed
type: task
priority: high
created_at: 2026-06-09T18:40:56Z
updated_at: 2026-06-10T17:49:11Z
parent: poker_solver_rust-a29s
---

Export existing HU blueprint_v2 snapshots into the universal dense format. Acceptance: old strategy.bin remains readable; universal export probabilities match BlueprintV2Strategy row-for-row; HU sparse projected snapshots export identically to dense; Explorer behavior remains unchanged.

## Summary of Changes

Implemented as `crates/core/src/blueprint_universal/hu_export.rs` plus the `export-universal` trainer subcommand (merged in d1d2d9ea):
- In-memory exporter (config + GameTree + BlueprintV2Strategy -> ExportOutput) and disk exporter (legacy bundle dir + snapshot name -> universal bundle dir), built on the Phase 1 write_bundle/BundleReader API.
- Probabilities pass through bitwise from strategy.bin (no recompute); rows emitted in spec sort order under the hu_arena namespace; per-row action_schema_fingerprint reuses the blueprint_v2 function via a new shared Fnv1aHasher (hash values regression-pinned).
- Manifest populated from config + snapshot metadata.json (source_backend hu_dense / hu_sparse_projected, missing_row_policy reject).
- 37 tests added incl. bitwise row-for-row identity vs BlueprintV2Strategy, dense-vs-sparse payload identity, action mapping, sort order, zero-mass uniform preservation, and precise rejection tests.

Acceptance verified: legacy strategy.bin remains readable (no legacy behavior changes; storage.rs only refactored onto the shared hasher with pinned values); export matches BlueprintV2Strategy row-for-row bitwise; sparse-projected exports byte-identical to dense except training.source_backend; Explorer untouched. Full suite green in 44s warm.

## Deferred (noted for later phases)
- Bucket file references in the manifest are empty (exporter reads projected strategy.bin, not cluster files); Phase 5 loader should decide validation policy.
- Call/AllIn amount_chips = 0 (GameTree nodes carry no pot state); spec "when applicable" escape hatch used, size_key carries sizing.
- action_abstraction_fingerprint hashes Debug formatting (fragile across versions); harden when Phase 3 touches fingerprints.
- Disk export path does not cross-check config bucket counts vs strategy.bucket_counts.
