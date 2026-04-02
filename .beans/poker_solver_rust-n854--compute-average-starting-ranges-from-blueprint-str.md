---
# poker_solver_rust-n854
title: Compute average starting ranges from blueprint strategy
status: todo
type: task
priority: high
created_at: 2026-04-01T22:25:55Z
updated_at: 2026-04-01T22:25:55Z
---

Post-processing step after blueprint training. Walk the blueprint tree and compute average reach-weighted ranges at each street boundary for all action sequences.

For each canonical hand (169), compute the probability of reaching the flop/turn/river under the blueprint's average strategy. Save as a lookup table that cfvnet datagen can load to generate realistic ranges instead of random RSP ranges.

Approach: standalone CLI command or snapshot post-processing step.
- Load saved strategy.bin + game tree + bucket files
- For each action sequence through preflop → flop → turn:
  - Compute reach per canonical hand (product of strategy probabilities)
  - At turn entry: save the ranges (169 hands × 2 players × weight)
- Save to a file (e.g. turn_ranges.bin) alongside the snapshot

This replaces random RSP ranges in cfvnet datagen, reducing game tree size from ~6GB to ~200MB per subgame (realistic ranges have ~200-400 combos vs ~1000).
