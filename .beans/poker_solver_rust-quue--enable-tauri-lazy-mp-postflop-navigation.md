---
# poker_solver_rust-quue
title: Enable Tauri lazy MP postflop navigation
status: completed
type: feature
priority: high
created_at: 2026-07-17T17:37:03Z
updated_at: 2026-07-17T18:44:58Z
parent: poker_solver_rust-osss
---

The Tauri GameExplorer currently starts two-player universal_mp_lazy sessions and supports preflop navigation, but errors when the flop is dealt because it cannot map public board cards and the current player state to the semantic lazy MP bucket key. Extend the separate lazy MP session without reconstructing a dense tree.

The first delivery target is a 2-player universal_mp_lazy bundle with retained full config: deal a legal flop from the preflop session, derive the trainer-compatible postflop bucket, query the matching sparse row, render legal actions/strategy, and preserve back navigation. Keep the adapter extensible to turn/river and N-player support, but do not silently fabricate postflop strategy when the bucket model or row is unavailable.

## Checklist

- [x] Research the existing postflop bucket APIs and lazy MP key contract.
- [x] Define the supported 2-player flop/turn/river navigation boundary and failure behavior.
- [x] Implement card dealing, board state, bucket derivation, and postflop lazy row lookup.
- [x] Preserve preflop/HU behavior and use the existing arena semantic key.
- [x] Add regression tests for flop transition, postflop action navigation, and back navigation.
- [x] Update explorer/architecture documentation.
- [x] Run focused tests and repository-approved verification.

Parent: unified HU/MP trainer runtime epic.

## Summary of Changes

Added a separate two-player universal_mp_lazy flop-navigation path to the Tauri GameExplorer. The session now retains typed board state, exposes transactional chance states for partial flops, resolves file-backed buckets through the trainer-compatible AllBuckets canonicalization, aggregates blocked-combo postflop matrix probabilities from semantic sparse rows, validates row action descriptors, and preserves back replay. Missing bucket sources, mappings, and sparse rows are explicit errors; turn/river and N-player browsing remain deferred. Fixed preflop bucket clamping for sub-169 configurations. Focused Tauri tests pass: 376 library tests and 15 integration tests; the workspace suite remains over the repository one-minute gate.
