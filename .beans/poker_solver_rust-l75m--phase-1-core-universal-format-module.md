---
# poker_solver_rust-l75m
title: 'Phase 1: core universal format module'
status: completed
type: task
priority: high
created_at: 2026-06-09T18:40:56Z
updated_at: 2026-06-10T15:28:08Z
parent: poker_solver_rust-a29s
---

Add the core format module and read/write primitives for synthetic universal bundles. Acceptance: empty/synthetic bundles round-trip; bad magic/version/checksum/truncated payload/unknown-required-feature cases reject precisely; old format code remains untouched.

## Summary of Changes

Implemented as `crates/core/src/blueprint_universal/` (merged in e63772cd):
- `manifest.rs`: serde types for `blueprint.json` covering all spec-required fields (game/seats/rake, training provenance, layout incl. row_namespace, buckets with per-street counts and file refs, manifest-level fingerprints, files). Unknown JSON fields ignored on read.
- `header.rs`: fixed 48-byte little-endian binary header (BPROW001/BPACT001/BPPRO001 magics), CRC-64/XZ payload checksums.
- `descriptors.rs`: fixed-width row descriptors (documented byte offsets) and action descriptors.
- `bundle.rs`: `write_bundle`/`BundleData` writer and validating `BundleReader` (manifest, format_name, required features, file lengths, SHA-256, CRC-64, sorted-unique row identities, offsets, probability normalization) exposing read-only lookup.
- `error.rs`: `FormatError` with 16 precise variants; every rejection test asserts the specific variant.
- Tests: 40 total (unit + proptest + integration round-trip/rejection) in-module and in `crates/core/tests/blueprint_universal_roundtrip.rs`.

Acceptance verified: empty/synthetic bundles round-trip; bad magic/version/checksum/truncated/unknown-required-feature reject precisely; no legacy format code modified. Full workspace suite green in 46s. Built via TDD with three-lens review (spec compliance, correctness, simplicity) plus a fix round addressing all 18 findings (overflow-checked header arithmetic, hard error on missing manifest file entries, spec-complete manifest types, API simplification).
