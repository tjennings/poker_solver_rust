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

### Position resolution: system-by-system changes

#### 1. Game tree constructor (`game_tree.rs`)

**Current:** `to_act: 0` hardcoded at root and at postflop street transitions. `blinds: [small_blind, big_blind]` stored on `GameTree`.

**Change:**
- `GameTree` gets `dealer: u8` instead of `blinds: [f64; 2]`
- `BuildState` gets `dealer: u8`, drops `blinds`
- Root: `to_act = dealer` (SB/button acts first preflop)
- Street transitions (Chance nodes): `to_act = 1 - dealer` (BB/OOP acts first postflop)
- Blind posting: computed from `dealer` during tree construction — `invested[dealer] += small_blind`, `invested[1-dealer] += big_blind`, `pot += small_blind + big_blind`
- `build_subgame` also takes `dealer: u8` and sets `to_act = 1 - dealer` (postflop subgames always start with OOP)
- `build_child` for `Fold` / `Call` / `Bet` / `Raise` / `AllIn`: uses `pot` (carried) instead of recomputing from blinds+invested

**Test:** Build a tree with `dealer=0`, verify:
- Root node: `player = 0` (SB acts first preflop)
- First flop decision: `player = 1` (BB acts first postflop)
- Fold terminal at root: `pot = small_blind + big_blind` (1.5BB)
- After SB raise to 2BB + BB call: `pot = 4.0BB` (0.5 + 1.0 + 1.5 + 1.0)

#### 2. MCCFR traversal (`mccfr.rs`)

**Current:** `terminal_value` takes `blinds`, `invested`, computes `-(blinds[t] + vol_t)` for folder. `traverse_external` passes `deal`, `traverser`, reads `player` from tree node.

**Change:**
- `terminal_value` simplifies: takes `pot`, `traverser`, `winner`
  - Folder: `0`
  - Winner: `pot`
  - Showdown: compare hands, winner gets `pot`, loser gets `0`, tie gets `pot/2`
- Remove `blinds` parameter from `terminal_value`
- Remove `invested` from `GameNode::Terminal` (only `pot` remains)
- `GameTree.blinds` removed — `dealer` stored instead
- `ScenarioEvTracker` EV offset (`fold_value_at_node`): no longer needed since fold = 0

**Test:**
- Unit test `terminal_value`: fold winner gets `pot`, loser gets `0`
- Unit test showdown: winner gets `pot`, tie splits
- Integration: traverse a small tree, verify EV of pure-fold hand = 0

#### 3. Subgame solver (`cfv_subgame_solver.rs`)

**Current:** Comments say "player 0 = OOP, player 1 = IP". `oop_reach_init` / `ip_reach_init` fields assume player 0 = OOP. Fold payoff uses `pot - invested[winner]` / `0`.

**Change:**
- Rename `oop_reach_init` → `reach_init: [Vec<f64>; 2]` indexed by seat, or keep names but document they mean "first-to-act" / "second-to-act" not hardcoded OOP
- Better: accept `dealer: u8` in constructor, derive which seat is OOP/IP
- `showdown_value_avg` / `showdown_value_single`: uses `pot` only — winner gets `pot`, loser gets `0`
- `compute_conditional_showdowns`: returns `avg_eq * pot` for winner (not `avg_eq * pot - invested`)
- Fold terminals in `cfr_traverse_vectorized`: folder gets `0`, winner gets `pot`
- Depth boundary: stores `pot` only (no `invested`)
- `warm_start_from_blueprint`: unchanged (uses `blueprint_decision_idx`, doesn't depend on position)

**Test:**
- Build a subgame with `dealer=0`, verify BB (seat 1) acts first at the root
- Fold payoff: folder gets 0, winner gets pot
- Showdown: verify correct hand ranking determines winner, payoff = pot

#### 4. Rollout evaluator (`continuation.rs`, `postflop.rs`)

**Current:** `rollout_inner` carries `pot` and `invested` through the abstract tree. Terminal payoffs use `pot - invested[player]` / `-invested[player]`.

**Change:**
- `rollout_inner` carries only `pot` (drop `invested` parameter)
- `apply_action` updates `pot` only (increment by bet amount). Track `invested` locally per-street for to-call logic within `apply_action`, reset at chance nodes.
- Terminal payoffs: fold winner = `pot`, loser = `0`. Showdown winner = `pot`, loser = `0`.
- `rollout_from_boundary` signature: drop `invested` parameter
- `RolloutLeafEvaluator`: drop `invested` from `evaluate_boundaries` requests
- `LeafEvaluator` trait: requests become `(pot, eff_stack, traverser)` — drop `invested`

**Test:**
- Unit test: rollout through fold terminal → winner gets pot, loser gets 0
- Unit test: rollout through showdown → winner gets pot
- Unit test: rollout through chance node → pot carries forward, invested resets

#### 5. Explorer (`exploration.rs`)

**Current:** `get_strategy_matrix_v2` reaching probability loop hardcodes preflop bucket indices for player 0. `walk_v2_tree` auto-skips chance nodes. TUI-facing code assumes `player 0 = SB`.

**Change:**
- `get_strategy_matrix_v2`: use `tree.dealer` to determine which seat is SB/BB
- `StrategyMatrix.to_act`: already set from `walk.to_act` (correct after tree fix)
- `blueprint_propagate_ranges`: uses tree walking, inherits correct `to_act` from tree
- Position labels in returned data: derive from `dealer`, not hardcoded

**Test:**
- Load a blueprint, navigate to a flop node, verify the acting player label is "BB" (OOP postflop)
- Verify preflop root shows "SB" as the acting player

#### 6. TUI scenarios (`blueprint_tui_scenarios.rs`, `main.rs`)

**Current:** `if *player == 0 { "SB" } else { "BB" }` hardcoded.

**Change:**
- Use `tree.dealer` to map seat index → position label:
  ```
  fn position_label(seat: u8, dealer: u8) -> &str {
      if seat == dealer { "SB" } else { "BB" }
  }
  ```
- Scenario config `player: SB` resolves to the seat that matches `dealer` (not hardcoded to 0)
- EV display: fold offset no longer needed (fold = 0 in the new model)

**Test:**
- With `dealer=0`: player 0 displays as "SB", player 1 as "BB"
- With `dealer=1`: player 1 displays as "SB", player 0 as "BB"
- "SB Open" scenario resolves to the correct tree root node

#### 7. Postflop explorer frontend (`PostflopExplorer.tsx`, `Explorer.tsx`)

**Current:** `matrix.player === 0 ? 'SB' : 'BB'` hardcoded. `position.to_act === 0 ? 'SB' : 'BB'` hardcoded.

**Change:**
- Backend returns `dealer` (or position label) alongside `to_act`
- Frontend derives label: `to_act === dealer ? 'SB' : 'BB'`
- Or backend resolves the label and returns `position_name: "SB" | "BB"` directly

**Test:**
- Postflop UI shows "BB" as the first actor on the flop
- Preflop UI shows "SB" as the first actor

#### 8. Range solver / full-depth solver (`full_depth_solver.rs`, range-solver crate)

**Current:** Comments say "OOP = player 0". The range-solver builds its own tree internally.

**Change:**
- Verify the range-solver's internal tree uses correct position
- If the range-solver has its own tree builder, it needs the same `dealer`-based fix
- `build_game` in `postflop.rs` passes position info to the range-solver

**Test:**
- Range-solve a flop spot, verify BB acts first
- Compare range-solver output against known commercial solver results

### What doesn't change

- Strategy storage: `regrets[node_idx][bucket]` — node_idx already encodes position through the tree structure
- Bucket computation: `buckets[player][street]` — still indexed by seat
- MCCFR traversal structure: `traverse_traverser` / `traverse_opponent` — still uses `player == traverser` check from tree node
- Hand dealing: `deal.hole_cards[seat]` — unchanged

### Validation plan

#### Unit tests (per component)
Each system above has specific tests listed. All existing tests must be updated to the new model.

#### Integration test: known equilibrium
- Train a small blueprint (169 preflop buckets, 10 postflop buckets, simple bet sizes) for 1B+ iterations
- Compare SB open frequencies against known HU equilibrium:
  - SB raise: ~67%, call: ~24%, fold: ~9%
- Compare AA EV: should be ~7BB (not 4.3)
- Compare Q2s EV: should be ~0.43BB (not 0.06)

#### Regression test: symmetry
- Build tree with `dealer=0` and `dealer=1`
- Verify the trees are structurally identical (mirror image)
- Verify strategies trained with either dealer assignment converge to the same equilibrium

#### Smoke test: explorer
- Load a trained blueprint in the explorer
- Navigate preflop: SB label on first actor
- Navigate to flop: BB label on first actor
- Verify strategy matrices look reasonable (not all-fold, not all-shove)

#### Comparison test: before/after
- Save the current (broken) blueprint's SB open strategy as a reference
- After refactor, train a new blueprint and compare
- The new blueprint should be more aggressive (more raises, fewer calls) at the SB open
