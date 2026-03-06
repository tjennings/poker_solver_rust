---
# poker_solver_rust-y0o3
title: 'Range Solver: Output-identical postflop DCFR reimplementation'
status: completed
type: epic
priority: normal
created_at: 2026-03-06T18:10:44Z
updated_at: 2026-03-06T20:01:09Z
---

Reimplement b-inary/postflop-solver as crates/range-solver. Output-identical, comparable performance. Includes range-solve CLI and 1000-game comparison harness.

## Tasks

- [x] Task 1: Scaffold crate + workspace integration
- [x] Task 2: Card types + encoding
- [x] Task 3: MutexLike + AtomicFloat
- [x] Task 4: Range parsing
- [x] Task 5: Bet size parsing
- [x] Task 6: Action enum + TreeConfig
- [x] Task 7: Action tree building
- [x] Task 8: Game/GameNode traits (interface)
- [x] Task 9: PostFlopNode + PostFlopGame structs
- [x] Task 10: Hand evaluator
- [x] Task 11: Game tree building (interpreter)
- [x] Task 12: Terminal evaluation
- [x] Task 13: Isomorphism
- [x] Task 14: DCFR solver
- [x] Task 15: Public query API
- [x] Task 16: End-to-end validation
- [x] Task 17: CLI subcommand
- [x] Task 18: Comparison crate + identity test
- [x] Task 19: Performance benchmark


## Summary of Changes

Complete reimplementation of b-inary/postflop-solver as `crates/range-solver` (~9,500 lines Rust).

**Output identity**: 1000/1000 random configs produce exact f32 equality on exploitability, strategy, EV, and equity.
**Performance**: 1.00x — identical speed to the original (416ms vs 415ms on 10-config benchmark).

### What was built:
- `crates/range-solver/` — Self-contained DCFR postflop solver (no dependency on core crate)
  - Card encoding, PioSOLVER range parsing, bet size parsing
  - Action tree construction with threshold-based sizing
  - Game tree building with arena-allocated nodes
  - 7-card hand evaluator with sorted strength arrays
  - Suit isomorphism detection and swap tables
  - Terminal evaluation (fold + showdown + rake)
  - DCFR solver with exact discount formulas (α=1.5, β=0.5, γ=3.0)
  - Public query API (strategy, EV, equity, navigation)
- `crates/range-solver-compare/` — Comparison harness against original
  - 1000-config identity test (21s in release mode)
  - Performance parity benchmark
- `range-solve` CLI subcommand in trainer

### Commits: 20 atomic commits on feat/range-solver branch
### Tests: 188 unit tests + 7 doc tests + 5 comparison tests
