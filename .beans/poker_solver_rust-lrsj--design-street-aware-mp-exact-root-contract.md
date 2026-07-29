---
# poker_solver_rust-lrsj
title: Design street-aware MP exact root contract
status: in-progress
type: task
priority: high
created_at: 2026-07-29T13:00:06Z
updated_at: 2026-07-29T13:16:22Z
parent: poker_solver_rust-g7yj
---

Define and test the shared contract between UniversalMpLazy and the exact range solver for flop, turn, and river roots. Include board/street, current actor, street bets, facing bet, prior aggression, raise depth vectors, raw reaches, and lossless chip-unit policy. Reconcile existing p989 and iu44 requirements without implementing the full adapter yet.



## Checklist

- [x] Research existing exact-root/session contracts and p989 fractional-chip boundary
- [x] Brainstorm and decide the street-aware contract shape
- [ ] Implement contract and focused construction/validation tests via rust-developer
- [ ] Review implementation and preserve turn/river navigation and exact-cache behavior
- [ ] Run formatting, focused tests, full test suite timing, and diff checks
- [ ] Document temporary UniversalMpLazy turn/river guard boundary
- [ ] Commit code, tests, and bean atomically while excluding unrelated YAML



## Design Decision

Use an explicit postflop street-aware snapshot with `MpStreet`, current board, actual-seat raw reaches, root actor, raw `f64` chip amounts, betting metadata, and full Flop/Turn/River lead plus raise-depth vectors. Keep the existing integer `SolveGameRoot` projection isolated behind a named legacy adapter conversion so the lossless snapshot does not imply that p989 is solved. Snapshot construction supports non-terminal Flop/Turn/River roots; `game_solve_core` retains an explicit UniversalMpLazy Turn/River rejection until the adapter phase.



## Status

Implementation was intentionally stopped at the user's request before Rust edits. Research and architecture brainstorming identified the contract shape, but no code, tests, or adapter changes landed. The implementation worktree remains clean; the existing main-worktree YAML edit is preserved.
