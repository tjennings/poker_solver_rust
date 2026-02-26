---
# poker_solver_rust-62hb
title: Fix inverted postflop position mapping and update preflop viz labels
status: completed
type: bug
priority: normal
created_at: 2026-02-26T15:38:55Z
updated_at: 2026-02-26T15:50:40Z
---

## Problem

1. **postflop_showdown_value position swap bug**: In solver.rs:761, `hero_tree_pos = u8::from(hero_pos == ip_player)` inverts the position mapping. SB (pos 0, IP) gets OOP values and BB (pos 1, OOP) gets IP values. Both preflop and postflop trees use the same convention (0=SB, 1=BB), so this should be `hero_tree_pos = hero_pos`. This explains the "raising/shoving almost all hands" behavior.

2. **Preflop training viz generic labels**: The matrix legend shows "small raise / mid raise / all-in" instead of the actual configured raise sizes from PreflopConfig.

## Tasks

- [x] Fix position mapping in postflop_showdown_value (solver.rs)
- [x] Fix position mapping in exploitability (inherits fix via same function)
- [x] Add test verifying correct position mapping
- [x] Update preflop viz legend to show actual raise sizes
- [x] Run tests to verify fix

## Summary of Changes

1. **Fixed inverted position mapping** in `postflop_showdown_value` (`solver.rs:755-766`): Changed `hero_tree_pos = u8::from(hero_pos == ip_player)` to `hero_tree_pos = hero_pos`. Both preflop and postflop trees use identical position numbering (0=SB/IP, 1=BB/OOP), so no remapping is needed. The old code inverted the lookup — SB read OOP values and BB read IP values — biasing both players toward maximum aggression.

2. **Exploitability fix**: `exploitability.rs` calls the same `postflop_showdown_value` function, so it inherits the fix automatically.

3. **Added position mapping test**: `postflop_showdown_value_position_mapping` constructs a `PostflopState` with known asymmetric IP/OOP EV values and verifies SB reads pos-0 (IP) values and BB reads pos-1 (OOP) values.

4. **Updated viz legend**: `print_hand_matrix` now accepts optional `&[PreflopAction]` and displays actual raise sizes (e.g. "2.5x", "3.0x", "all-in") in the legend instead of generic labels. Callers in `print_preflop_matrices` pass the tree node actions; other callers pass `None` for fallback generic labels.
