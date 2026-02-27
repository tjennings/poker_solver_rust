---
# poker_solver_rust-i9gy
title: Replace raise size multipliers with BB amounts and pot fractions
status: completed
type: feature
priority: normal
created_at: 2026-02-27T13:52:30Z
updated_at: 2026-02-27T14:11:18Z
---

Replace the preflop raise_sizes multiplier system with tagged RaiseSize enum (Bb/PotFraction).

## Tasks
- [ ] Define RaiseSize enum with serde in config.rs
- [ ] Update PreflopConfig fields and constructors
- [ ] Update tree.rs (PreflopAction::Raise, compute_raise_amount, tree walk)
- [ ] Update solver.rs type changes
- [ ] Update preflop_convergence.rs tests
- [ ] Update sample YAML configs
- [ ] Update docs/training.md

## Summary of Changes

Replaced the preflop raise size multiplier system (`current_bet * multiplier`) with a tagged `RaiseSize` enum supporting two intuitive formats:
- `Bb(x)` / `"2.5bb"` — raise TO x big blinds
- `PotFraction(f)` / `"0.75p"` — raise BY f × pot after call

20 files modified across config, tree builder, solver, exploration, visualization, tests, sample configs, and docs.
