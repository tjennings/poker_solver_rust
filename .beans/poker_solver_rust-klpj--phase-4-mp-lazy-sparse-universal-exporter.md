---
# poker_solver_rust-klpj
title: 'Phase 4: MP lazy sparse universal exporter'
status: completed
type: task
priority: high
created_at: 2026-06-09T18:40:56Z
updated_at: 2026-06-10T21:28:59Z
parent: poker_solver_rust-a29s
---

Export realized MP lazy sparse rows into the universal format for analysis/read-only loading. Acceptance: SparseSnapshotEntry rows are sorted by semantic key and exported with verbatim semantic identity; strategy sums normalize correctly; zero-sum rows use documented uniform fallback; artifact is marked non-resumable.

## Summary of Changes

Implemented as `crates/core/src/blueprint_universal/mp_lazy_export.rs` + reader/format extensions (merged in 1557a66b):
- Semantic-key side table: strategy.semantic.bin (magic BPSEM001, 32-byte LE records: history_hi/lo u64, history_hash u64, history_len u16, reserved[6]); rows reference records by index via semantic_key_kind=1; record type lives in descriptors.rs with the other wire formats.
- Rows under mp_semantic namespace: seat/street/local bucket unpacked from MpInfosetKey (street = bucket >> 14, local = bucket & 0x3FFF), global_bucket = packed value widened, source_node_idx = u32::MAX, fingerprint over the full semantic identity (mp_semantic_row_key_fingerprint_v1).
- Opaque action descriptors (kind gated by mp_semantic_rows_v1 required feature): action kinds/amounts unrecoverable from sparse snapshots (>32-action histories are hash-only); real actions arrive with bean mt3l (store action identity at realization).
- Normalization matches SparseMpStorage::average_strategy incl. zero-mass -> uniform; missing_row_policy uniform_legal; bundles marked non-resumable via new CompatibilityMetadata.resumable flag.
- Reader extensions: side-table header/CRC/SHA validation, record-index range checks, Opaque-without-feature rejection (tested), feature gating via SUPPORTED_FEATURES; HU/MP eager bundles take the unchanged path.
- export-universal-mp dispatches on the recorded metadata.json kind field (blueprint_mp vs blueprint_mp_lazy_sparse; unknown kinds are hard errors).
- 26+ tests added incl. verbatim identity round-trip (with >32-length hash-only history), normalization equivalence vs rebuilt storage, gating, and precise rejections. Review fix rounds caught two real disk-path bugs (config.yaml read from wrong dir; LazyExportConfig fields not matching real YAML) and a silent kind-detection fallback.

Follow-up during integration: doctest compilation regression fixed (bean erde) — warm suite 47.4s, under the 60s gate.
