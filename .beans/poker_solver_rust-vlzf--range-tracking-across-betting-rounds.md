---
# poker_solver_rust-vlzf
title: Range tracking across betting rounds
status: completed
type: feature
priority: normal
created_at: 2026-03-06T02:16:51Z
updated_at: 2026-03-06T03:16:44Z
---

Add reaching-probability range tracking to the strategy explorer. PlayerRange data structure shared between UI and solver. Bar height scaling, manual editing, node-level overrides, combo expansion at postflop.

## Tasks

- [x] Task 1: PlayerRange data structure
- [x] Task 2: compute_reaching_range function
- [x] Task 3: Wire reaching weights into StrategyMatrix
- [x] Task 4: Update dev server
- [x] Task 5: Frontend bar height scaling
- [x] Task 6: Range editing UI
- [x] Task 7: Cleanup old threshold/filter code

## Summary of Changes

Added range tracking across betting rounds to the strategy explorer. As users navigate the game tree, hands that would have folded at earlier decisions are visually narrowed via bar height scaling.

### Backend
- `crates/core/src/range.rs` (new): PlayerRange struct with [f64; 169] reaching probabilities, RangeSource enum, multiply_action/set_hand methods, custom serde, 6 unit tests
- `crates/tauri-app/src/exploration.rs`: compute_reaching_range replays action history through blueprint strategy. StrategyMatrix response includes reaching_p1/reaching_p2. Removed threshold, is_hand_in_range, action_meets_threshold
- `crates/devserver/src/main.rs`: Removed threshold from params

### Frontend
- Bar height scaling proportional to reaching probability (below 1% hidden)
- Range editing: click cells to cycle weight (1.0/0.5/0.0), yellow dot indicator, range toolbar
- Node-level snapshots: ranges snapshot on navigation, restore on rewind, reset on new hand
- Removed all threshold/filter references

### Commits (6)
034e0da PlayerRange | 418c768 compute_reaching_range | 1bf376e Wire weights | abfe6bc Bar scaling | 8f42fa2 Editing UI | 892fd03 Fix rewind bug
