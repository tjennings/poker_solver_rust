---
# poker_solver_rust-lrsj
title: Design street-aware MP exact root contract
status: in-progress
type: task
priority: high
created_at: 2026-07-29T13:00:06Z
updated_at: 2026-07-29T13:08:26Z
parent: poker_solver_rust-g7yj
---

Define and test the shared contract between UniversalMpLazy and the exact range solver for flop, turn, and river roots. Include board/street, current actor, street bets, facing bet, prior aggression, raise depth vectors, raw reaches, and lossless chip-unit policy. Reconcile existing p989 and iu44 requirements without implementing the full adapter yet.



## Checklist

- [ ] Research existing exact-root/session contracts and p989 fractional-chip boundary
- [ ] Brainstorm and decide the street-aware contract shape
- [ ] Implement contract and focused construction/validation tests via rust-developer
- [ ] Review implementation and preserve turn/river navigation and exact-cache behavior
- [ ] Run formatting, focused tests, full test suite timing, and diff checks
- [ ] Document temporary UniversalMpLazy turn/river guard boundary
- [ ] Commit code, tests, and bean atomically while excluding unrelated YAML
