---
# poker_solver_rust-quue
title: Enable Tauri lazy MP postflop navigation
status: in-progress
type: feature
priority: high
created_at: 2026-07-17T17:37:03Z
updated_at: 2026-07-17T17:37:03Z
parent: poker_solver_rust-osss
---

The Tauri GameExplorer currently starts two-player universal_mp_lazy sessions and supports preflop navigation, but errors when the flop is dealt because it cannot map public board cards and the current player state to the semantic lazy MP bucket key. Extend the separate lazy MP session without reconstructing a dense tree.

The first delivery target is a 2-player universal_mp_lazy bundle with retained full config: deal a legal flop from the preflop session, derive the trainer-compatible postflop bucket, query the matching sparse row, render legal actions/strategy, and preserve back navigation. Keep the adapter extensible to turn/river and N-player support, but do not silently fabricate postflop strategy when the bucket model or row is unavailable.

## Checklist

- [ ] Research the existing postflop bucket APIs and lazy MP key contract.
- [ ] Define the supported 2-player flop/turn/river navigation boundary and failure behavior.
- [ ] Implement card dealing, board state, bucket derivation, and postflop lazy row lookup.
- [ ] Preserve preflop/HU behavior and use the existing arena semantic key.
- [ ] Add regression tests for flop transition, postflop action navigation, and back navigation.
- [ ] Update explorer/architecture documentation.
- [ ] Run focused tests and repository-approved verification.

Parent: unified HU/MP trainer runtime epic.
