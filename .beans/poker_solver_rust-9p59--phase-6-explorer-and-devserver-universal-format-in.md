---
# poker_solver_rust-9p59
title: 'Phase 6: Explorer and devserver universal format integration'
status: completed
type: task
priority: normal
created_at: 2026-06-09T18:40:56Z
updated_at: 2026-06-18T17:13:37Z
parent: poker_solver_rust-a29s
---

Teach Explorer/devserver bundle APIs about universal format metadata. Acceptance: snapshot listing reports format kind/player count; HU universal bundles load through existing views; MP bundles expose read-only bundle info and row lookup APIs before full MP browsing UI.

## Scope Note (2026-06-10 goal recap)

Explorer compatibility is THE hard requirement of the whole effort: the end state is the Explorer browsing universal bundles (including lazy sparse exports, which need real action descriptors — see the blocking bean on action identity in sparse rows). Legacy bundle browsing can be dropped once universal loading works.

## Scope (Phase 6, 2026-06-18)

Explore finding: universal bundles do NOT retain config.yaml and the manifest lacks the action abstraction, so the Explorer cannot rebuild the V2GameTree it navigates by. Resolution (spec-sanctioned — directory layout lists config.yaml as optional retained config): exporters retain config.yaml in the universal bundle so it is self-contained.

This phase (unblocked; mt3l block removed — mt3l gates only future MP-lazy RICH rendering, not Phase 6's read-only MP support):
- Exporters write config.yaml into universal bundle output (HU required for tree rebuild; MP for consistency/future).
- Explorer + devserver load via the unified loader (detect blueprint.json -> load_bundle); legacy HU path preserved.
- Universal HU renders through EXISTING views via a tree rebuilt from retained config; must be identical to legacy (proven by a test comparing a legacy bundle and its universal export through the actual exploration commands).
- Snapshot/bundle listing + bundle info report format kind + player count from the manifest.
- MP bundles: load + read-only bundle info + row-lookup exposure; NO MP browsing UI (deferred; MP-lazy action labels remain Opaque pending mt3l).

## Summary of Changes

Merged in the Phase 6 merge commit. Universal dense bundles now load in the Tauri Explorer and devserver:
- Disk-wrapper exporters retain config.yaml in the universal bundle (shared export_common::retain_config_yaml), making bundles self-contained; manifest config_path references it.
- Explorer load_bundle_core detects blueprint.json BEFORE config.yaml and routes through blueprint_universal::loader::load_bundle.
- Universal HU renders through the EXISTING HU views: a BlueprintV2Strategy is reconstructed from the universal rows (new BlueprintV2Strategy::from_raw_with_tree, additive) with the tree rebuilt from the retained config.yaml — bitwise-identical to legacy, proven by universal_hu_renders_identically_to_legacy through the actual view commands.
- MP bundles load read-only (StrategySource::UniversalMp): get_bundle_info reports kind/num_players/seats/stacks/buckets from the manifest; HU-only views return a clean "MP browsing not yet supported" error (no panic). Listings report format kind + player count.
- Devserver auto-mirrors via the shared _core functions. No frontend changes needed (universal HU reuses the BlueprintV2 UI state).

Acceptance verified: listing reports kind+players; universal HU renders identically to legacy through existing views; MP bundles load with read-only info and do not error. blueprint_v2 change is additive only (new constructor). All tests pass.

## Caveat (tracked separately)
Full-suite WALL time now measures ~66-73s, over the 60s gate. Cause is NOT Phase 6 (test execution is flat at ~29s; the new tauri test runs in 0.05s) but pre-existing binary-load overhead — ~30 test binaries totaling several GB of debug binaries (trainer 5x148MB, cfvnet 5x125MB). Tracked in a dedicated high-priority bean.
