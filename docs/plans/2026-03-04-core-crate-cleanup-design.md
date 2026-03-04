# Core Crate Function Cleanup Design

**Date**: 2026-03-04
**Scope**: 16 production functions > 60 LOC in `crates/core/src/`
**Approach**: Independent refactors per module, phase-helper extraction, TraverseCtx pattern

## Principles

- Each refactored function targets **< 60 LOC**
- Helpers are `#[inline]` on hot paths, private
- Context structs for functions with > 7 args
- No behavioral changes — pure structural refactoring
- `cargo test` + `cargo clippy` verification after each function
- Independent refactors per module (no cross-module unification)

## Group 1: `postflop_exhaustive.rs` (5 functions)

| Function | LOC | Strategy |
|-|-|-|
| `exhaustive_solve_one_flop` | 148 | Extract 5 phase helpers: traverse, dcfr_discount_and_merge, clamp_regrets, check_exploitability, report_progress. Loop body becomes ~15 lines of phase calls. |
| `compute_equity_table` | 128 | Extract 4 sequential phases: build_combo_index, enumerate_boards, compute_ranks_per_board, aggregate_equity. |
| `build_exhaustive` | 93 | Extract per-flop closure body into `solve_and_extract_flop` helper. |
| `best_response_ev` | 86 | Apply TraverseCtx pattern — context struct for immutable args, extract hero/opponent decision helpers. |
| `eval_with_avg_strategy` | 66 | Same TraverseCtx pattern. Could share context struct with `best_response_ev` using a mode flag. |

## Group 2: `postflop_mccfr.rs` (5 functions)

| Function | LOC | Strategy |
|-|-|-|
| `mccfr_traverse` | 143 | Create `MccfrTraverseCtx` (13 args -> ctx + varying args). Extract hero/opponent decision helpers. Mirror the exhaustive refactor pattern. |
| `mccfr_solve_one_flop` | 90 | Phase helpers: sample_deals, traverse_iteration, merge_deltas, report_progress. |
| `mccfr_eval_with_avg_strategy` | 90 | TraverseCtx pattern with shared ctx struct. |
| `mccfr_extract_values` | 84 | Extract deal-enumeration loop body into helper. |
| `build_mccfr` | 63 | Extract per-flop closure body (mirrors `build_exhaustive` pattern). |

## Group 3: Other Modules (6 functions)

| Function | LOC | Strategy |
|-|-|-|
| `tree.rs:build_recursive` | 99 | Extract terminal_check, generate_actions, build_children into helpers. |
| `rank_array_cache.rs:derive_equity_table` | 91 | Extract combo_index_building, per_board_accumulation phases. |
| `rank_array_cache.rs:compute_rank_arrays` | 62 | Extract combo_list_building, board_enumeration. |
| `simulation.rs:run_simulation` | 74 | Extract agent_setup, game_loop, aggregate_results. |
| `solver.rs:cfr_traverse` | 73 | TraverseCtx pattern (lighter — already fairly tight). |
| `postflop_tree.rs` (if applicable) | ~68 | Extract if identified during implementation. |

## Excluded

- **Test functions** (6): `exhaustive_cfr_fold_terminal_payoff`, `exhaustive_cfr_showdown_terminal_payoff`, `mccfr_traverse_fold_terminal`, `mccfr_traverse_showdown_terminal`, `assert_allin_recursive`, `assert_allin_at_every_node`
- **Test-only code**: `compute_equity_table_reference` (`#[cfg(test)]`, intentionally verbose)
- **Test utilities**: `info_key.rs:all_169_canonical_hands_unique`

## Execution Order

Three independent workstreams that can run in parallel:

1. **Exhaustive module** (Group 1) — highest value, 5 functions
2. **MCCFR module** (Group 2) — mirrors Group 1 patterns, 5 functions
3. **Other modules** (Group 3) — smaller independent changes, 6 functions

Each workstream gets its own `rust-developer` agent in a worktree. Post-refactor audit with `idiomatic-rust-enforcer` + `rust-perf-reviewer` per group.
