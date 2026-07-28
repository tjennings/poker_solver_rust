---
# poker_solver_rust-xtvm
title: Implement mmap-backed universal MP lazy payload loading
status: completed
type: bug
priority: critical
created_at: 2026-07-28T13:46:24Z
updated_at: 2026-07-28T15:31:27Z
parent: poker_solver_rust-osss
---

The compact MP-lazy index did not change real startup time: the Explorer still reports universal_reader_load around 33.8 seconds for 1,898,121 rows. Therefore BundleReader::open itself is dominant: it std::fs::read-copies all payloads, computes full SHA-256/CRC checks, decodes every row/action/probability/semantic record, and validates the entire bundle before the first preflop view.

Implement a private mmap-backed or offset-backed reader path for universal_mp_lazy bundles that preserves strict integrity validation and exact query results but avoids duplicate heap materialization and decoding action descriptors that are not queried. Keep existing BundleReader::open behavior for HU, MP eager, legacy, and compatibility tests unless a shared validated abstraction is proven safe. Do not weaken corruption detection silently; if validation moves behind a mode, make the mode explicit and document it.

Acceptance:
- Separate timing for payload mapping/read, checksum/CRC validation, decoding/validation, and MP-lazy load completion.
- Representative large MP lazy load materially improves over the observed 33.8s, or the timing proves the remaining phase and records the next concrete bottleneck.
- Query action probabilities and descriptors are byte-for-byte/semantically equivalent for existing small HU, eager MP, and lazy MP fixtures.
- Strict checksum, header, offset, semantic-key, normalization, and corruption tests remain green.
- No on-disk format change unless required and documented; preserve arena/lazy session structure.
- Update docs/architecture.md and docs/explorer.md for the reader mode and validation contract.
- Run focused core/Tauri tests; leave sample_configurations/blueprint_mp_hu_500f_100t_100r.yaml untouched.

\n\n## Summary of Changes\n\n- Added a private memmap2-backed UniversalMpLazy reader for fixed-width rows, actions, probabilities, and semantic payloads.\n- Preserved BundleReader for HU, eager MP, and legacy paths.\n- Kept strict header, manifest, SHA-256, CRC-64, record-length, ordering, duplicate, offset, semantic, action-kind, and normalization validation at load time.\n- Added phase timing logs for mapping, integrity, validation, reader-ready, and MP-lazy range-index construction.\n- Added an owned MP-lazy query view so little-endian probabilities are decoded safely without changing borrowed HU/eager InfosetView callers.\n- Added full action-descriptor and probability-bit-pattern equivalence assertions plus mapped checksum corruption coverage.\n- Representative large-bundle benchmark was not run in this turn; strict checksum and full structural scans remain the likely residual startup cost.\n\n## Verification\n\n- cargo check -p poker-solver-core\n- cargo check -p poker-solver-tauri\n- cargo test -p poker-solver-core --test loader_unified (28 passed)\n- cargo test -p poker-solver-core --test blueprint_universal_roundtrip (23 passed)\n- cargo test -p poker-solver-core --test mp_lazy_universal_export (12 passed)\n- targeted rustfmt --check and git diff --check passed.\n

## Benchmark\n\nRepresentative bundle:\n- path: local_data/blueprints/mp_hu_500f_100t_100r_nut_high_cap_0p5_v2\n- rows: 1,898,121\n- previous Explorer load: 33,814 ms\n- current Explorer load: 3,221 ms\n- phases: loading 70 ms, integrity 3,100 ms, validation 44 ms, index 2 ms\n- end-to-end curl request: 3.23 s\n- observed improvement: approximately 10.5x\n\nThe remaining startup cost is integrity checksum/CRC scanning, not MP index construction or descriptor/probability decoding.
