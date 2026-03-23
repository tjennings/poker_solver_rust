# HU Position Model & Value Model Refactor

**Date:** 2026-03-23

## Problem

1. **Position is broken**: Player 0 = SB acts first on all streets. In HU poker, SB = button = IP postflop (acts last). The entire codebase assumes player 0 = OOP postflop, which changes the game fundamentally.

2. **Value model is fragile**: Separate `blinds` and `invested` arrays on terminal nodes cause double-counting bugs and confusion about what "invested" means (voluntary only? total? including blinds?).

## Design

### Position: Dealer-based acting order

Inspired by robopoker's `(dealer + ticker) % N` model:

- Game tree gets a `dealer: u8` field (which seat is the button)
- Acting order is **derived**, not hardcoded:
  - Preflop first actor: `dealer` (SB/button acts first)
  - Postflop first actor: `1 - dealer` (BB/OOP acts first)
- `to_act` on Decision nodes is set by the tree builder based on `dealer` and street
- Strategy is keyed by position (the tree node encodes which position acts), not player identity
- For standard HU: `dealer = 0` means seat 0 is SB/button/IP

### Value model: pot + invested (per-street) + stack

**State tracked:**
- `pot: f64` — total chips in the middle, incremented by every bet/call/blind posting
- `invested: [f64; 2]` — per-player chips committed **this street** only (resets each street). Used solely for computing to-call amounts during tree construction.
- `stack` — chips behind (implicit: `starting_stack - total_spent`)

**Terminal nodes store only `pot`:**
```
Terminal {
    kind: Fold { winner } | Showdown,
    pot: f64,
}
```

No `invested` array on terminals. No `blinds` array on the tree.

**Terminal payoffs:**
- Fold winner: `pot` (gets everything)
- Fold loser: `0` (walks away)
- Showdown winner: `pot`
- Showdown loser: `0`
- Showdown tie: `pot / 2`

**Depth boundaries:** store `pot`, pass it forward to the rollout.

### Blind posting

At tree construction time:
- `pot` starts at 0
- SB (seat `dealer`) posts small blind: `pot += SB`, `invested[dealer] += SB`
- BB (seat `1-dealer`) posts big blind: `pot += BB`, `invested[1-dealer] += BB`
- Preflop `to_act = dealer` (SB acts first, facing the BB)
- `to_call = invested[opponent] - invested[actor]` = BB - SB

### Street transitions

At chance nodes (new street):
- `invested` resets to `[0, 0]`
- `pot` carries forward unchanged
- `to_act = 1 - dealer` (OOP acts first postflop)

### Bet/raise handling

When a player bets/raises:
- `invested[actor] = new_amount` (this street's commitment)
- `pot += (new_amount - old_invested[actor])` (increment by the additional chips)
- `stack` decreases accordingly

### What changes

| Component | Current | New |
|-----------|---------|-----|
| `GameTree` | `blinds: [f64; 2]` | `dealer: u8` |
| `BuildState` | `blinds`, `invested`, `to_act` hardcoded | `dealer`, `pot`, `invested` (per-street), `to_act` derived |
| `Terminal` | `pot`, `invested: [f64; 2]` | `pot` only |
| `terminal_value` (MCCFR) | `blinds[o] + vol_o` / `-(blinds[t] + vol_t)` | `pot` / `0` |
| Subgame solver fold | `pot - invested[winner]` / `0` | `pot` / `0` |
| Subgame solver showdown | `pot - invested[t]` / `-invested[t]` | `pot` / `0` |
| Rollout terminals | Uses abstract tree's `pot`/`invested` | Uses carried `pot` |
| Depth boundaries | `pot`, `invested` | `pot` only |
| `LeafEvaluator` requests | `(pot, eff_stack, traverser, invested)` | `(pot, eff_stack, traverser)` |
| TUI labels | Hardcoded `player 0 = SB` | Derived from `dealer` |
| Explorer labels | Hardcoded `player 0 = SB` | Derived from `dealer` |
| Subgame solver OOP/IP | Hardcoded player 0 = OOP | Derived from `dealer` |
| EV display offset | `fold_value_at_node` subtracts blind | Not needed — fold is 0 |

### What doesn't change

- Strategy storage: `regrets[node_idx][bucket]` — node_idx already encodes position through the tree structure
- Bucket computation: `buckets[player][street]` — still indexed by seat
- MCCFR traversal structure: `traverse_traverser` / `traverse_opponent` — still uses `player == traverser` check
- Hand dealing: `deal.hole_cards[seat]` — unchanged

### Validation

After refactoring:
- SB open strategy should show ~67% raise, ~24% call, ~9% fold
- AA EV should be ~7BB
- Q2s EV should be ~0.43BB
- Fold EV displays as 0
- All existing tests updated to new model
