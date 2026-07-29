---
# poker_solver_rust-g7yj
title: UniversalMpLazy turn and river exact solving
status: in-progress
type: feature
priority: high
created_at: 2026-07-29T12:59:55Z
updated_at: 2026-07-29T12:59:55Z
parent: poker_solver_rust-mk2k
---

Extend the UniversalMpLazy Exact solve path beyond flop decisions without weakening the capability contract.

- [ ] Define a street-aware exact root contract carrying board, street, action vectors, betting snapshot, raw reaches, and lossless supported chip units.
- [ ] Adapt UniversalMpLazy turn and river states into the existing range-solver/full-depth backend.
- [ ] Preserve action semantics, depth-specific raises, root actor/facing-bet state, and exact cache/generation invalidation.
- [ ] Hide or disable Exact solve controls only for states that remain genuinely unsupported.
- [ ] Add turn/river parity, fractional-unit, cache, and UI regression coverage.
- [ ] Update docs and retain the two-player limitation unless N-player support is explicitly added.
