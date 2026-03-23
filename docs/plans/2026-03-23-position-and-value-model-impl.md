# HU Position Model & Value Model Refactor — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Fix the HU position model (dealer-based acting order) and simplify the value model (pot-only terminals, fold=0) across all 8 systems.

**Architecture:** Three phases — (1) domain types & game tree, (2) solvers (MCCFR + subgame), (3) UI/explorer. Each phase is independently testable. The game tree change is the foundation that all other systems depend on.

**Tech Stack:** Rust (poker-solver-core, poker-solver-tauri), TypeScript/React (frontend).

---

## Phase 1: Domain Types & Game Tree

### Task 1: Simplify GameNode::Terminal — remove invested, keep pot only

**Files:**
- Modify: `crates/core/src/blueprint_v2/game_tree.rs:35-40`

**Step 1: Write failing test**

```rust
#[test]
fn terminal_fold_stores_pot_only() {
    let tree = GameTree::build(
        50.0, 0.5, 1.0,
        &[vec!["2bb".into()]],
        &[vec![0.5]], &[vec![0.5]], &[vec![0.5]],
    );
    // Find a fold terminal at the root (SB folds)
    if let GameNode::Decision { children, actions, .. } = &tree.nodes[tree.root as usize] {
        let fold_idx = actions.iter().position(|a| matches!(a, TreeAction::Fold)).unwrap();
        if let GameNode::Terminal { kind, pot, .. } = &tree.nodes[children[fold_idx] as usize] {
            assert!(matches!(kind, TerminalKind::Fold { .. }));
            // Pot should be SB + BB = 1.5
            assert!((pot - 1.5).abs() < 0.01, "Fold pot should be 1.5, got {pot}");
        } else {
            panic!("Expected terminal");
        }
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core --lib blueprint_v2::game_tree::tests::terminal_fold_stores_pot_only`
Expected: FAIL (test doesn't exist yet, or terminal still has invested)

**Step 3: Remove `invested` from `GameNode::Terminal`**

Change `GameNode::Terminal` to:
```rust
Terminal {
    kind: TerminalKind,
    pot: f64,
}
```

Update ALL construction sites in `game_tree.rs` (`build_child`, `make_showdown_or_chance`, etc.) to stop passing `invested`. The `pot` field must be computed correctly as a running total:

In `BuildState`, add a `pot: f64` field. Initialize it to `small_blind + big_blind` at the root. Increment it in `build_child` when processing Call/Bet/Raise/AllIn actions. Pass it through to terminals.

The `invested` field on `BuildState` stays — it's still needed for computing to-call amounts. But it doesn't propagate to terminal nodes.

**Step 4: Fix all compilation errors**

Every file that pattern-matches on `GameNode::Terminal { pot, invested, .. }` needs to drop `invested`. This includes:
- `mccfr.rs` — `traverse_external`, `terminal_value`
- `cfv_subgame_solver.rs` — `cfr_traverse_vectorized`, `eval_combo_value`, `NodeInfo`
- `continuation.rs` — `rollout_inner`
- `cbv_compute.rs` — test helpers
- `postflop.rs` — various

For now, just fix compilation — the payoff logic changes come in Phase 2.

**Step 5: Run all tests, fix failures**

Run: `cargo test -p poker-solver-core --lib`
Many tests will break. Fix each one to work with the new terminal structure.

**Step 6: Commit**

```bash
git commit -m "refactor: remove invested from GameNode::Terminal, pot-only terminals"
```

### Task 2: Add `dealer` to GameTree, derive `to_act` from dealer + street

**Files:**
- Modify: `crates/core/src/blueprint_v2/game_tree.rs:62-69` (GameTree struct)
- Modify: `crates/core/src/blueprint_v2/game_tree.rs:66-68` (BuildState)
- Modify: `crates/core/src/blueprint_v2/game_tree.rs:140-189` (GameTree::build)
- Modify: `crates/core/src/blueprint_v2/game_tree.rs:510-530` (Chance node / street transition)
- Modify: `crates/core/src/blueprint_v2/game_tree.rs:755-798` (build_subgame)

**Step 1: Write failing test**

```rust
#[test]
fn postflop_bb_acts_first_with_dealer_0() {
    let tree = GameTree::build(
        50.0, 0.5, 1.0,
        &[vec!["2bb".into()]],
        &[vec![0.5]], &[vec![0.5]], &[vec![0.5]],
    );
    assert_eq!(tree.dealer, 0); // SB = seat 0
    // Root: seat 0 (SB/dealer) acts first preflop
    if let GameNode::Decision { player, .. } = &tree.nodes[tree.root as usize] {
        assert_eq!(*player, 0, "SB acts first preflop");
    }
    // Find flop node: seat 1 (BB/non-dealer) should act first
    for node in &tree.nodes {
        if let GameNode::Chance { next_street: Street::Flop, child } = node {
            if let GameNode::Decision { player, street, .. } = &tree.nodes[*child as usize] {
                assert_eq!(*player, 1, "BB (seat 1) acts first on flop when dealer=0");
                assert_eq!(*street, Street::Flop);
            }
            return;
        }
    }
    panic!("No flop chance node found");
}
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `player` will be 0, not 1.

**Step 3: Implement dealer-based acting order**

Replace `GameTree.blinds: [f64; 2]` with `GameTree.dealer: u8`.

In `GameTree::build`:
```rust
let initial_state = BuildState {
    starting_stack: stack_depth,
    invested: [0.0, 0.0],
    pot: small_blind + big_blind, // pot starts with both blinds
    dealer,
    street: Street::Preflop,
    num_raises: 0,
    to_act: dealer as u8, // SB (dealer) acts first preflop
    facing_bet: true,
    last_raise_to: big_blind,
};
```

Add `dealer: u8` and `pot: f64` to `BuildState`, remove `blinds`.

In `Chance` node street transition (line ~526):
```rust
to_act: 1 - state.dealer, // BB (non-dealer) acts first postflop
```

In `build_subgame`:
```rust
pub fn build_subgame(
    street: Street,
    pot: f64,
    invested: [f64; 2],
    starting_stack: f64,
    bet_sizes: &[Vec<f64>],
    depth_limit: Option<u8>,
    dealer: u8, // NEW parameter
) -> Self {
    ...
    to_act: 1 - dealer, // OOP acts first in postflop subgames
    ...
}
```

Update `blind` handling: in `BuildState`, compute invested from dealer:
```rust
invested: {
    let mut inv = [0.0; 2];
    inv[dealer as usize] = small_blind; // SB posts
    inv[1 - dealer as usize] = big_blind; // BB posts
    inv
},
```

**Step 4: Fix all references to `tree.blinds`**

Every reference to `tree.blinds` changes to use `tree.dealer`. Key files:
- `mccfr.rs`: `terminal_value` currently takes `&tree.blinds` — will change in Phase 2
- `trainer.rs`: references `tree.blinds` for EV display offset
- `exploration.rs`: `invested_at_v2_node` uses `tree.blinds`
- `postflop.rs`: `build_subgame_solver` passes blinds

**Step 5: Fix compilation and tests**

Run: `cargo test -p poker-solver-core --lib`

**Step 6: Commit**

```bash
git commit -m "feat: dealer-based acting order — BB acts first postflop"
```

### Task 3: Add position helper function

**Files:**
- Modify: `crates/core/src/blueprint_v2/game_tree.rs`

**Step 1: Add a helper to resolve position labels**

```rust
impl GameTree {
    /// Map a seat index to a position label based on dealer.
    pub fn position_label(&self, seat: u8) -> &'static str {
        if seat == self.dealer { "SB" } else { "BB" }
    }

    /// The seat index of the small blind (dealer/button).
    pub fn sb_seat(&self) -> u8 { self.dealer }

    /// The seat index of the big blind.
    pub fn bb_seat(&self) -> u8 { 1 - self.dealer }
}
```

**Step 2: Write test**

```rust
#[test]
fn position_labels() {
    let tree = /* build with dealer=0 */;
    assert_eq!(tree.position_label(0), "SB");
    assert_eq!(tree.position_label(1), "BB");
    assert_eq!(tree.sb_seat(), 0);
    assert_eq!(tree.bb_seat(), 1);
}
```

**Step 3: Commit**

```bash
git commit -m "feat: add position_label/sb_seat/bb_seat helpers to GameTree"
```

---

## Phase 2: Solver Value Models

### Task 4: Simplify MCCFR terminal_value — pot-only payoffs

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs:557-606` (terminal_value)
- Modify: `crates/core/src/blueprint_v2/mccfr.rs:482-483` (call site in traverse_external)

**Step 1: Write failing test**

```rust
#[test]
fn terminal_fold_winner_gets_pot() {
    let deal = make_deal();
    let pot = 4.0;
    // Winner (traverser) gets pot
    let v = terminal_value(TerminalKind::Fold { winner: 0 }, pot, 0, &deal, 0.0, 0.0);
    assert!((v - 4.0).abs() < 1e-10, "Fold winner gets pot, got {v}");
}

#[test]
fn terminal_fold_loser_gets_zero() {
    let deal = make_deal();
    let pot = 4.0;
    // Loser (traverser) gets 0
    let v = terminal_value(TerminalKind::Fold { winner: 1 }, pot, 0, &deal, 0.0, 0.0);
    assert!(v.abs() < 1e-10, "Fold loser gets 0, got {v}");
}
```

**Step 2: Simplify terminal_value signature and implementation**

```rust
fn terminal_value(
    kind: TerminalKind,
    pot: f64,
    traverser: u8,
    deal: &Deal,
    rake_rate: f64,
    rake_cap: f64,
) -> f64 {
    let rake = if rake_rate > 0.0 {
        let uncapped = pot * rake_rate;
        if rake_cap > 0.0 { uncapped.min(rake_cap) } else { uncapped }
    } else {
        0.0
    };
    match kind {
        TerminalKind::Fold { winner } => {
            if winner == traverser { pot - rake } else { 0.0 }
        }
        TerminalKind::Showdown => {
            let t = traverser as usize;
            let o = 1 - t;
            let rank_t = rank_hand(deal.hole_cards[t], &deal.board);
            let rank_o = rank_hand(deal.hole_cards[o], &deal.board);
            match rank_t.cmp(&rank_o) {
                Ordering::Greater => pot - rake,
                Ordering::Less => 0.0,
                Ordering::Equal => pot / 2.0 - rake / 2.0,
            }
        }
        TerminalKind::DepthBoundary => unreachable!(),
    }
}
```

**Step 3: Update call site in traverse_external**

The call in `traverse_external` (line ~482) currently passes `invested` and `blinds`:
```rust
// Old:
(terminal_value(*kind, invested, &tree.blinds, traverser, &deal.deal, rake_rate, rake_cap), ...)
// New:
(terminal_value(*kind, *pot, traverser, &deal.deal, rake_rate, rake_cap), ...)
```

**Step 4: Update all terminal_value tests**

Update existing tests (`terminal_fold_payoff`, `terminal_showdown_payoff`, `terminal_fold_with_rake`, `terminal_dead_money_model`, etc.) to match the new signature and expected values.

**Step 5: Remove fold_value_at_node EV offset**

In `trainer.rs`, the `fold_value_at_node` function and the EV offset logic (lines ~654-663) are no longer needed since fold = 0. Remove them — the displayed EVs will naturally show fold as 0.

**Step 6: Run tests**

Run: `cargo test -p poker-solver-core --lib blueprint_v2::mccfr`

**Step 7: Commit**

```bash
git commit -m "feat: pot-only terminal payoffs in MCCFR — fold=0, winner=pot"
```

### Task 5: Simplify subgame solver terminal values

**Files:**
- Modify: `crates/core/src/blueprint_v2/cfv_subgame_solver.rs:562-610` (cfr_traverse_vectorized terminals)
- Modify: `crates/core/src/blueprint_v2/cfv_subgame_solver.rs:1245-1275` (compute_conditional_showdowns)
- Modify: `crates/core/src/blueprint_v2/cfv_subgame_solver.rs:1055-1065` (eval_combo_value / showdown_value_avg)
- Modify: `crates/core/src/blueprint_v2/cfv_subgame_solver.rs:1296-1330` (showdown_value_single)

**Step 1: Update fold payoff in cfr_traverse_vectorized**

```rust
TerminalKind::Fold { winner } => {
    let payoff = if winner == traverser { pot } else { 0.0 };
    for i in 0..n {
        cfv_buf[out_start + i] = payoff;
    }
}
```

**Step 2: Update compute_conditional_showdowns**

Change from `avg_eq * pot - invested_traverser` to `avg_eq * pot`:

```rust
fn compute_conditional_showdowns(
    combos: &[[Card; 2]],
    equity_matrix: &[Vec<f64>],
    reach_opponent: &[f64],
    pot: f64,
    out: &mut [f64],   // Drop invested_traverser parameter
) {
    out.par_iter_mut().enumerate().for_each(|(i, val)| {
        // ... same equity computation ...
        *val = if reach_sum > 0.0 {
            let avg_eq = eq_sum / reach_sum;
            avg_eq * pot   // Winner gets eq * pot, loser gets (1-eq) * pot → net = eq*pot
        } else {
            0.0
        };
    });
}
```

**Step 3: Update showdown_value_single and showdown_value_avg**

Drop `invested_traverser` parameter:
```rust
fn showdown_value_single(
    hero_combo: usize,
    hands: &SubgameHands,
    equity_matrix: &[Vec<f64>],
    opp_reach: &[f64],
    pot: f64,
) -> f64 {
    // ... same equity computation ...
    avg_equity * pot  // Drop "- invested_traverser"
}
```

**Step 4: Update NodeInfo to drop invested**

```rust
enum NodeInfo {
    Terminal { kind: TerminalKind, pot: f64 },  // Drop invested
    Chance { child: u32 },
    Decision { player: u8, num_actions: usize, children_buf: [u32; 16] },
}
```

**Step 5: Update depth boundary — pot only**

```rust
TerminalKind::DepthBoundary => {
    let ordinal = self.node_to_boundary[node_idx];
    if ordinal != usize::MAX {
        let half_pot = pot / 2.0;
        for i in 0..n {
            cfv_buf[out_start + i] = self.leaf_cfvs[ordinal]
                .get(i).copied().unwrap_or(0.0) * half_pot;
        }
    }
}
```

Note: `half_pot` is still correct here — leaf_cfvs are in pot-fraction units.

**Step 6: Update BoundaryInfo to drop invested**

```rust
pub boundaries: Vec<(usize, f64)>,  // (node_index, pot) — drop invested
```

Update all code that constructs/reads boundary info.

**Step 7: Update LeafEvaluator trait — drop invested from requests**

```rust
fn evaluate_boundaries(
    &self,
    combos: &[[Card; 2]],
    board: &[Card],
    oop_range: &[f64],
    ip_range: &[f64],
    requests: &[(f64, f64, u8)],  // (pot, effective_stack, traverser) — no invested
) -> Vec<Vec<f64>>;
```

**Step 8: Run tests and fix**

Run: `cargo test -p poker-solver-core --lib blueprint_v2::cfv_subgame`

**Step 9: Commit**

```bash
git commit -m "feat: pot-only terminal payoffs in subgame solver"
```

### Task 6: Simplify rollout terminal values

**Files:**
- Modify: `crates/core/src/blueprint_v2/continuation.rs:162-245` (rollout_inner)
- Modify: `crates/tauri-app/src/postflop.rs` (RolloutLeafEvaluator)

**Step 1: Update rollout_inner terminals**

```rust
GameNode::Terminal { kind, .. } => {
    match kind {
        TerminalKind::Fold { winner } => {
            if ctx.player == *winner { pot } else { 0.0 }
        }
        TerminalKind::Showdown => {
            let player_rank = rank_hand(hero_hand, board);
            let opponent_rank = rank_hand(opponent_hand, board);
            match player_rank.cmp(&opponent_rank) {
                Ordering::Greater => pot,
                Ordering::Less => 0.0,
                Ordering::Equal => pot / 2.0,
            }
        }
        TerminalKind::DepthBoundary => panic!("..."),
    }
}
```

**Step 2: Drop `invested` parameter from rollout_from_boundary**

The function currently carries `pot` and `invested`. Drop `invested`:
```rust
pub fn rollout_from_boundary(
    hero_hand: [Card; 2],
    opponent_hand: [Card; 2],
    board: &[Card],
    ctx: &RolloutContext<'_>,
    abstract_node: u32,
    rng: &mut impl Rng,
    pot: f64,
) -> f64 {
```

**Step 3: Update apply_action to track pot only**

`apply_action` currently returns `(pot, invested)`. Change to return just `pot`:
```rust
fn apply_action(action: &TreeAction, pot: f64, actor_invested: f64, opponent_invested: f64, starting_stack: f64) -> (f64, f64, f64) {
    // Returns (new_pot, new_actor_invested, new_opponent_invested)
    // actor_invested/opponent_invested are per-street for to-call only
    // pot is the running total
}
```

Or simpler: `apply_action` just returns the new pot. Track per-street invested locally in `rollout_inner` for the Call to-call computation.

**Step 4: Update RolloutLeafEvaluator in postflop.rs**

Drop `invested` from the evaluate flow. The rollout now returns chip values based on pot only.

**Step 5: Run tests**

Run: `cargo test -p poker-solver-core --lib blueprint_v2::continuation`

**Step 6: Commit**

```bash
git commit -m "feat: pot-only rollout terminals, drop invested from rollout"
```

---

## Phase 3: UI & Explorer

### Task 7: Update TUI position labels

**Files:**
- Modify: `crates/trainer/src/blueprint_tui_scenarios.rs:165-167`
- Modify: `crates/trainer/src/main.rs` (scenario resolution)

**Step 1: Use `tree.position_label(player)` instead of hardcoded mapping**

```rust
// Old:
if *player == 0 { "SB" } else { "BB" }
// New:
tree.position_label(*player)
```

**Step 2: Resolve scenario `player: SB` config using dealer**

```rust
let seat = match scenario.player {
    PlayerLabel::Sb => tree.sb_seat(),
    PlayerLabel::Bb => tree.bb_seat(),
};
```

**Step 3: Remove fold_value_at_node EV offset** (if not already done in Task 4)

**Step 4: Commit**

```bash
git commit -m "feat: TUI labels derived from dealer position"
```

### Task 8: Update Explorer position labels

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs` (StrategyMatrix, labels)
- Modify: `crates/tauri-app/src/postflop.rs` (subgame solver setup, player labels)
- Modify: `crates/devserver/src/main.rs` (if position info needed)

**Step 1: Add `dealer` to StrategyMatrix response**

```rust
pub struct StrategyMatrix {
    // ... existing fields ...
    pub dealer: u8,  // So frontend can derive labels
}
```

**Step 2: Update postflop solver to pass dealer to subgame**

`build_subgame_solver` needs to pass `dealer` to `GameTree::build_subgame`.

**Step 3: Commit**

```bash
git commit -m "feat: pass dealer through explorer and subgame solver"
```

### Task 9: Update Frontend position labels

**Files:**
- Modify: `frontend/src/PostflopExplorer.tsx`
- Modify: `frontend/src/Explorer.tsx`
- Modify: `frontend/src/types.ts`

**Step 1: Add `dealer` to StrategyMatrix TypeScript type**

```typescript
export interface StrategyMatrix {
    // ... existing fields ...
    dealer: number;
}
```

**Step 2: Replace hardcoded labels**

```typescript
// Old:
position: matrix.player === 0 ? 'SB' : 'BB'
// New:
position: matrix.player === matrix.dealer ? 'SB' : 'BB'
```

Apply to all label sites in `PostflopExplorer.tsx` and `Explorer.tsx`.

**Step 3: Commit**

```bash
git commit -m "feat: frontend derives position labels from dealer"
```

---

## Phase 4: Integration Validation

### Task 10: Full integration test

**Step 1: Build and run all tests**

```bash
cargo test
cd frontend && npx tsc --noEmit && npx vitest run
```

**Step 2: Train a small blueprint and validate**

Train with the 1kbkt config for ~1B iterations. Check:
- SB open: ~67% raise, ~24% call, ~9% fold
- AA EV: ~7BB
- Q2s EV: ~0.43BB
- Fold EV displays as 0
- Flop: BB acts first (correct position)
- TUI labels: correct SB/BB assignment

**Step 3: Test explorer**

- Load blueprint, navigate preflop → verify SB label
- Navigate to flop → verify BB acts first
- Click solve → verify subgame runs correctly with new model

**Step 4: Commit**

```bash
git commit -m "test: validate position and value model refactor"
```

---

## Summary of file changes

| File | Phase | Changes |
|------|-------|---------|
| `game_tree.rs` | 1 | Remove invested from Terminal, add dealer, derive to_act |
| `mccfr.rs` | 2 | Simplify terminal_value (pot only, fold=0) |
| `trainer.rs` | 2 | Remove fold_value_at_node EV offset |
| `cfv_subgame_solver.rs` | 2 | Pot-only fold/showdown, drop invested from boundaries |
| `continuation.rs` | 2 | Pot-only rollout terminals, drop invested param |
| `postflop.rs` | 2+3 | RolloutLeafEvaluator, subgame setup, dealer passthrough |
| `blueprint_tui_scenarios.rs` | 3 | Dealer-based position labels |
| `main.rs` (trainer) | 3 | Scenario resolution via dealer |
| `exploration.rs` | 3 | StrategyMatrix gets dealer field |
| `devserver/main.rs` | 3 | Pass-through if needed |
| `PostflopExplorer.tsx` | 3 | Derive labels from dealer |
| `Explorer.tsx` | 3 | Derive labels from dealer |
| `types.ts` | 3 | Add dealer to StrategyMatrix |
