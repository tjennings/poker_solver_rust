---
# poker_solver_rust-bxip
title: Refactor build_subgame_solver 14-param signature into options struct
status: todo
type: task
priority: normal
created_at: 2026-04-17T16:52:22Z
updated_at: 2026-04-17T16:52:22Z
---

Pre-existing tech debt flagged by simplicity review during perf/subgame-rollout work.

**Current state:** crates/tauri-app/src/postflop.rs:573-588 has 14 positional parameters — CLAUDE.md's simplicity limit is ~4 params. The commit for rollout hands/sec telemetry added the 14th (hand_counter), making the problem visible.

**Smell:** Every test and call site has a long trailing list of `None` arguments. Parameter names at call sites are unclear. Adding new optional features (bias_factor, num_samples, opponent_samples, neural_boundary_evaluator, hand_counter, ...) requires touching every caller.

**Proposed fix:** Group the optional solver configuration into a `SubgameSolverOptions` struct with defaults. Keep the small set of required args (board_cards, bet_sizes_per_depth, pot, stacks, weights, player, abstract_node_idx) as positional; move everything else to the options struct with `..Default::default()` ergonomics.

**Not urgent:** The function works. Just painful to extend.
