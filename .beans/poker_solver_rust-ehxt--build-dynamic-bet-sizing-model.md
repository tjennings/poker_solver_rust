---
# poker_solver_rust-ehxt
title: Build dynamic bet sizing model
status: todo
type: feature
priority: normal
created_at: 2026-03-23T13:14:11Z
updated_at: 2026-03-23T13:14:11Z
---

Build an automated bet sizing selection model inspired by GTO Wizard's Dynamic Sizing approach. The model should select optimal bet/raise sizes per node based on game context (SPR, board texture, position, street).

## Context
- GTO Wizard's Dynamic Sizing 2.0 solves depth-limited trees for candidate size subsets and picks the lowest-EV-loss set
- Their benchmarks show 1 well-chosen size captures 99.95% of single-size EV; simplified strategies can outperform complex ones in practice
- Current solver uses static bet sizes configured in YAML per street/depth

## Research Findings
- SRP flop: ~33%, ~67%, ~127% pot (small/medium/large pattern)
- Sizes should scale with SPR (smaller sizes in 3bet/4bet pots)
- Overbets (122-150% flop, 140-170% turn) matter on specific textures
- River benefits from small bet option (25-33%)
- Geometric sizing is suboptimal on early streets; solvers prefer smaller-than-geometric on flop, larger-than-geometric on turn/river

## Possible Approaches
1. **Iterative pruning**: Start with many candidate sizes, remove least valuable via regret analysis
2. **Build-up selection**: Solve depth-limited trees per candidate subset, pick lowest EV-loss (GTO Wizard v2)
3. **Heuristic rules**: SPR-based size selection without solving (simpler, less optimal)

## Todo
- [ ] Research: formalize EV-loss metric for size comparison
- [ ] Design: define candidate size generation strategy
- [ ] Design: depth-limited tree solving for size evaluation
- [ ] Implement: size selection pipeline
- [ ] Evaluate: benchmark against static sizing configs
