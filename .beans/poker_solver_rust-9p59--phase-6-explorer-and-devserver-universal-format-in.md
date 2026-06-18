---
# poker_solver_rust-9p59
title: 'Phase 6: Explorer and devserver universal format integration'
status: in-progress
type: task
priority: normal
created_at: 2026-06-09T18:40:56Z
updated_at: 2026-06-18T15:05:52Z
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
