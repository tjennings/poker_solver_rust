---
# poker_solver_rust-b4s8
title: Always include all-in in postflop tree actions
status: completed
type: feature
priority: normal
created_at: 2026-02-26T14:30:03Z
updated_at: 2026-02-26T14:49:51Z
---

Ensure all-in is always offered at every postflop decision point, matching PioSolver behavior. Changes to constrained_bet_actions, constrained_raise_actions, build_after_facing_bet_spr, and tests in postflop_tree.rs.

## Summary of Changes

Modified `crates/core/src/preflop/postflop_tree.rs`:

1. **constrained_bet_actions**: Always adds all-in if not already present and >5% pot gap from largest bet
2. **constrained_raise_actions**: All-in shove always available even at raise cap (raises_remaining=0); deduplicates against sized raises within 5% of all-in
3. **build_after_facing_bet_spr**: Uses `saturating_sub(1)` to prevent u8 underflow
4. **Tests**: Added `spr_tree_deep_allin_always_present_in_bets`, `spr_tree_allin_available_at_raise_limit`; updated `spr_tree_deep_matches_unconstrained` → `spr_tree_deep_has_more_nodes_than_unconstrained`

All 26 postflop_tree tests pass. Clippy clean.

### Preflop Tree Fix (follow-up)

Same issue in `crates/core/src/preflop/tree.rs`:
- All-in moved outside the `raise_count < raise_cap` guard
- Condition changed from `player_stack > to_call` to `player_stack > 0` so all-in is always an option
- `count_max_raises` renamed to `count_max_sized_raises` (only counts `Raise(_)`, not `AllIn`)
- Added 2 new tests: deep stacks at raise cap, stack == to_call
