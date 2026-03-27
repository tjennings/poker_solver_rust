---
# poker_solver_rust-eqt3
title: 'Implement preflop handling for ReBeL: use blueprint strategy for preflop'
status: in-progress
type: feature
created_at: 2026-03-27T11:13:57Z
updated_at: 2026-03-27T11:13:57Z
---

Use the blueprint's preflop strategy in ReBeL self-play instead of subgame solving. The range-solver is postflop only, so preflop needs special handling: sample preflop actions from the blueprint, update reach probabilities, determine pot/stack sizes entering the flop, skip preflop training examples, and start subgame solving from flop onward.
