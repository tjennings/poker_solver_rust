---
# poker_solver_rust-xtvm
title: Implement mmap-backed universal MP lazy payload loading
status: in-progress
type: bug
priority: critical
created_at: 2026-07-28T13:46:24Z
updated_at: 2026-07-28T13:46:24Z
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
