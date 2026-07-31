---
# poker_solver_rust-g54j
title: 'Phase 5: unified bundle detection and loader'
status: completed
type: task
priority: high
created_at: 2026-06-09T18:40:56Z
updated_at: 2026-06-18T14:45:06Z
parent: poker_solver_rust-a29s
---

Add one loader detection path for legacy HU bundles and universal HU/MP bundles. Acceptance: loader distinguishes old config.yaml+strategy.bin, universal HU, universal MP eager, universal MP lazy; errors explain missing/incompatible files; old HU discovery remains compatible.

## Scope Note (2026-06-10 goal recap)

Only the Tauri Explorer matters as a consumer. The unified loader needs universal-bundle loading plus whatever the Explorer currently uses during transition; legacy read paths are transitional, not contractual.

## Summary of Changes

Implemented as `crates/core/src/blueprint_universal/loader.rs` (merged in the Phase 5 merge commit):
- `detect_bundle_kind(dir)`: cheap manifest-only detection of the four kinds (LegacyHu, UniversalHu, UniversalMpEager, UniversalMpLazy). blueprint.json takes precedence over a retained config.yaml; source_backend is cross-checked against row_namespace and (for lazy) the mp_semantic_rows_v1 required feature. Resolves root / final/ / latest snapshot_NNNN/ nesting uniformly for both formats.
- `load_bundle(dir)` -> `LoadedBundle` enum (chosen over a trait: four closed variants, single consumer). Legacy HU reuses blueprint_v2 primitives (load_config, BlueprintV2Strategy::load, GameTree::build_with_options); universal kinds wrap BundleReader with HashMap lookup indices built at load.
- Unified infoset query API (query_hu / query_mp_eager via MpLazyKey for lazy) returning borrowed prob slices + action descriptors. Precise LoaderError variants (NotABundle, ManifestParse, UnknownSourceBackend, MissingPayloadFile, MissingRequiredFeature, NamespaceMismatch) — no silent fallbacks.
- 25 integration tests: detection for all four kinds + nesting + per-variant error cases; load+query each kind; zero-mass lazy uniform; blueprint.json-over-config.yaml precedence; and the acceptance equivalence test asserting legacy vs universal-HU return bitwise-identical probabilities for every infoset.

Acceptance verified: four-way detection with precise errors; legacy HU discovery unchanged and proven equivalent to universal HU. Core only — Explorer/devserver wiring deferred to Phase 6 (g54j is the loader; 9p59 is the integration). Full suite green in ~48s warm. NOTE: the Bash command sandbox now blocks the two blueprint_v2 per-flop trainer tests (create_dir_all on buckets/); they pass unsandboxed — environmental, not a code issue.
