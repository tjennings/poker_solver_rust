# Postflop Solve Matrix Bugs — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Fix three bugs in postflop explorer: position label flip, EV data loss, stale matrix race condition.

**Architecture:** All changes are in `crates/tauri-app/src/postflop.rs`. Add `dealer` field to `MatrixSnapshot`, plumb it through all snapshot sources, remove hardcoded `dealer: 1`. Compute EVs for depth-limited solver. Clear stale snapshot at solve start.

**Tech Stack:** Rust, Tauri (backend only — no frontend changes needed)

---

### Task 1: Add `dealer` field to `MatrixSnapshot` and plumb through range-solver path

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs:842-853` (struct), `:899-910` (capture), `:985-993` (build)

**Step 1: Add `dealer: u8` field to `MatrixSnapshot` struct**

In the struct at line 842, add `dealer` after `board`:

```rust
struct MatrixSnapshot {
    player: usize,
    strategy: Vec<f32>,
    private_cards: Vec<(u8, u8)>,
    initial_weights: Vec<f32>,
    num_hands: usize,
    actions: Vec<ActionInfo>,
    pot: i32,
    stacks: [i32; 2],
    hand_evs: Option<Vec<f32>>,
    board: Vec<String>,
    dealer: u8,
}
```

**Step 2: Set `dealer: 1` in `capture_matrix_snapshot` (range-solver path)**

In the `MatrixSnapshot` construction at line 899, add `dealer: 1`:

```rust
    MatrixSnapshot {
        player,
        strategy,
        private_cards,
        initial_weights,
        num_hands,
        actions,
        pot,
        stacks,
        hand_evs,
        board,
        dealer: 1, // range-solver: player 1 = IP = SB = dealer
    }
```

**Step 3: Use `snap.dealer` in `build_matrix_from_snapshot`**

Replace the hardcoded `dealer: 1` at line 992 with `snap.dealer`:

```rust
    PostflopStrategyMatrix {
        cells,
        actions: snap.actions,
        player: snap.player,
        pot: snap.pot,
        stacks: snap.stacks,
        board: snap.board,
        dealer: snap.dealer,
    }
```

**Step 4: Build and verify no compile errors**

Run: `cargo build -p poker-solver-tauri 2>&1 | head -30`

Expected: Compile errors in `snapshot_from_subgame` about missing `dealer` field. This is expected — we fix it in Task 2.

**Step 5: Commit**

Do NOT commit yet — the code won't compile until Task 2 is done.

---

### Task 2: Plumb `dealer` through the subgame/depth-limited path

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs:1007-1062` (snapshot_from_subgame), `:1580-1618` (solve_depth_limited make_matrix), `:2019-2029` (subgame_node_to_result)

**Step 1: Add `dealer: u8` parameter to `snapshot_from_subgame`**

At line 1007, add `dealer: u8` parameter after `node_idx`:

```rust
fn snapshot_from_subgame(
    hands: &SubgameHands,
    strategy: &SubgameStrategy,
    action_infos: Vec<ActionInfo>,
    weights: &[f32],
    board_strings: &[String],
    pot: i32,
    stacks: [i32; 2],
    player: usize,
    node_idx: u32,
    dealer: u8,
) -> MatrixSnapshot {
```

And add `dealer` to the `MatrixSnapshot` construction at line 1051:

```rust
    MatrixSnapshot {
        player,
        strategy: flat_strategy,
        private_cards,
        initial_weights,
        num_hands,
        actions: action_infos,
        pot,
        stacks,
        hand_evs: None,
        board: board_strings.to_vec(),
        dealer,
    }
```

**Step 2: Fix `solve_depth_limited` make_matrix closure**

At line 1581-1618, replace the hardcoded `player = 0` with the tree root's actual player, and pass `tree.dealer`:

```rust
        // Get root player from V2 tree (OOP = 1 - dealer in V2 convention).
        let root_player = match &tree.nodes[tree.root as usize] {
            GameNode::Decision { player, .. } => *player as usize,
            _ => 0,
        };
        let tree_dealer = tree.dealer;

        match build_subgame_solver(
            &board_cards,
            &bet_sizes_per_depth,
            pot as u32,
            [eff_stack as u32, eff_stack as u32],
            &oop_w,
            &ip_w,
            root_player,
            cbv_context.as_deref(),
            abstract_node_idx,
            rollout_bias_factor,
            rollout_num_samples,
            rollout_opponent_samples,
        ) {
            Ok((mut solver, hands, action_infos, tree, initial_pot, starting_stack)) => {
```

Wait — `build_subgame_solver` returns the tree. We can't read from it before calling. The tree is only available after the call. Let me re-check.

The `tree` variable is returned by `build_subgame_solver`. So we read root player AFTER the call:

Replace lines 1582 and 1610-1618:

```rust
        // Remove: let player = 0; // OOP acts first at root
```

After `build_subgame_solver` returns (inside the `Ok` arm, after line 1597):

```rust
            Ok((mut solver, hands, action_infos, tree, initial_pot, starting_stack)) => {
                // Get root player and dealer from V2 tree convention.
                let root_player = match &tree.nodes[tree.root as usize] {
                    GameNode::Decision { player, .. } => *player as usize,
                    _ => 0, // shouldn't happen — root is always a decision node
                };
                let tree_dealer = tree.dealer;
```

Then update the `make_matrix` closure at line 1611:

```rust
                let make_matrix = |strat: &SubgameStrategy| {
                    let snap = snapshot_from_subgame(
                        &hands, strat, action_infos.clone(),
                        &oop_w, &board_strings, pot,
                        [eff_stack, eff_stack], root_player, 0, tree_dealer,
                    );
                    build_matrix_from_snapshot(snap)
                };
```

Also update the diagnostic section around line 1662 that references `player`:

```rust
                        let reach = if root_player == 0 { &oop_w } else { &ip_w };
```

Wait — the `player` variable is also used in the `build_subgame_solver` call at line 1590. Let me re-read that:

Looking at line 1583-1596: `build_subgame_solver` takes `_player: usize` (unused param at line 644). So we can pass anything. But it's cleaner to pass `0` (OOP) since the param is ignored. Actually, keep it simple — just remove the `let player = 0` line and replace all uses.

Actually, the `player` variable at line 1590 is passed to `build_subgame_solver` as `_player` (unused). And at line 1662 it's used to select weights for diagnostics. Since `root_player` won't be available until after `build_subgame_solver`, let's just pass `0` to the builder directly.

Revised plan for this step:

1. Remove `let player = 0;` at line 1582
2. Pass `0` directly at line 1590 (unused param)
3. After `build_subgame_solver` returns, extract `root_player` and `tree_dealer` from the tree
4. Update `make_matrix` closure to use `root_player` and `tree_dealer`
5. Update diagnostic `reach` selection at line 1662 — use `root_player` (note: in V2 convention root_player=1 for OOP, but `root_cfvs(0)` at line 1696 takes positional 0=OOP — these are different conventions within the solver itself). Actually, the `reach` at 1662 just needs OOP weights since we're looking at the acting player's combos at root. Since OOP acts first and we always pass `&oop_w` to `snapshot_from_subgame`, we can just use `&oop_w` directly.

**Step 3: Fix `subgame_node_to_result` at line 2019**

Pass `result.tree.dealer` to `snapshot_from_subgame`:

```rust
            let snap = snapshot_from_subgame(
                &result.hands,
                &result.strategy,
                action_infos,
                &weights,
                &board_card_strings,
                pot_i32,
                stacks_i32,
                player_usize,
                node_idx,
                result.tree.dealer,
            );
```

**Step 4: Build and verify**

Run: `cargo build -p poker-solver-tauri 2>&1 | head -30`

Expected: Successful compilation.

**Step 5: Run tests**

Run: `cargo test -p poker-solver-tauri`

Expected: All tests pass.

**Step 6: Commit**

```
git add crates/tauri-app/src/postflop.rs
git commit -m "fix: plumb dealer through MatrixSnapshot, fix position label flip"
```

---

### Task 3: Clear stale `matrix_snapshot` at solve start

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs:1386-1390`

**Step 1: Add snapshot clear**

After line 1390 (`*state.solve_start.write() = ...`), add:

```rust
    *state.matrix_snapshot.write() = None;
```

**Step 2: Build and test**

Run: `cargo build -p poker-solver-tauri && cargo test -p poker-solver-tauri`

Expected: Pass.

**Step 3: Commit**

```
git add crates/tauri-app/src/postflop.rs
git commit -m "fix: clear stale matrix_snapshot at solve start"
```

---

### Task 4: Compute EVs after depth-limited solve completes

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs:1007-1062` (snapshot_from_subgame hand_evs), `:1643-1645` (final snapshot)

**Step 1: Add `hand_evs` parameter to `snapshot_from_subgame`**

Change the function signature at line 1007 to accept optional EVs:

```rust
fn snapshot_from_subgame(
    hands: &SubgameHands,
    strategy: &SubgameStrategy,
    action_infos: Vec<ActionInfo>,
    weights: &[f32],
    board_strings: &[String],
    pot: i32,
    stacks: [i32; 2],
    player: usize,
    node_idx: u32,
    dealer: u8,
    hand_evs: Option<Vec<f32>>,
) -> MatrixSnapshot {
```

And use it in the `MatrixSnapshot` construction (replacing the hardcoded `None`):

```rust
    MatrixSnapshot {
        player,
        strategy: flat_strategy,
        private_cards,
        initial_weights,
        num_hands,
        actions: action_infos,
        pot,
        stacks,
        hand_evs,
        board: board_strings.to_vec(),
        dealer,
    }
```

**Step 2: Update all callers to pass `None` for non-final snapshots**

In `solve_depth_limited` `make_matrix` closure (updated in Task 2):

```rust
                let make_matrix = |strat: &SubgameStrategy, evs: Option<Vec<f32>>| {
                    let snap = snapshot_from_subgame(
                        &hands, strat, action_infos.clone(),
                        &oop_w, &board_strings, pot,
                        [eff_stack, eff_stack], root_player, 0, tree_dealer, evs,
                    );
                    build_matrix_from_snapshot(snap)
                };
```

Update callers of `make_matrix`:
- Line 1622 (initial): `make_matrix(&initial_strategy, None)`
- Line 1640 (iteration): `make_matrix(&strategy, None)`
- Line 1645 (final): see Step 3

In `subgame_node_to_result` (line 2019):

```rust
            let snap = snapshot_from_subgame(
                &result.hands,
                &result.strategy,
                action_infos,
                &weights,
                &board_card_strings,
                pot_i32,
                stacks_i32,
                player_usize,
                node_idx,
                result.tree.dealer,
                None, // EVs not available during navigation
            );
```

**Step 3: Compute and pass EVs for the final snapshot**

At line 1643-1645 (final strategy snapshot), compute root CFVs and build per-combo EV vector. The `root_cfvs(0)` returns per-combo CFVs for OOP (the root player in the subgame). We need to map these from SubgameHands combo ordering to the same ordering used by `snapshot_from_subgame` (which filters zero-weight combos).

Actually, `snapshot_from_subgame` already filters combos by weight. The EVs need to be in the same order as the filtered combos. Since `root_cfvs` returns one value per combo in `hands.combos` order, and `snapshot_from_subgame` iterates `hands.combos` skipping zero-weight entries, we need to pre-filter the EVs the same way.

The simplest approach: build the EV vector inside `snapshot_from_subgame` by passing the full CFV vector and letting it filter alongside the combos. But that complicates the signature. Instead, pre-filter:

```rust
                // Final strategy snapshot with EVs.
                let strategy = solver.strategy();
                let cfvs = solver.root_cfvs(0); // 0 = OOP (positional convention)
                // Filter to match snapshot_from_subgame's combo filtering (skip zero-weight).
                let hand_evs: Vec<f32> = hands.combos.iter().enumerate()
                    .filter_map(|(combo_idx, combo)| {
                        let id0 = rs_poker_card_to_id(combo[0]);
                        let id1 = rs_poker_card_to_id(combo[1]);
                        let ci = card_pair_to_index(id0, id1);
                        if oop_w[ci] > 0.0 {
                            Some(cfvs[combo_idx] as f32)
                        } else {
                            None
                        }
                    })
                    .collect();
                *shared.matrix_snapshot.write() = Some(make_matrix(&strategy, Some(hand_evs)));
```

**Step 4: Build and test**

Run: `cargo build -p poker-solver-tauri && cargo test -p poker-solver-tauri`

Expected: Pass.

**Step 5: Commit**

```
git add crates/tauri-app/src/postflop.rs
git commit -m "feat: compute EVs after depth-limited solve for matrix display"
```

---

### Task 5: Write regression test for dealer convention

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs` (test module at line 2488)

**Step 1: Write test for `build_matrix_from_snapshot` dealer field**

```rust
    #[test]
    fn test_matrix_snapshot_dealer_passthrough() {
        // Minimal snapshot with no combos — just verify dealer is passed through.
        let snap = MatrixSnapshot {
            player: 0,
            strategy: vec![],
            private_cards: vec![],
            initial_weights: vec![],
            num_hands: 0,
            actions: vec![],
            pot: 100,
            stacks: [900, 900],
            hand_evs: None,
            board: vec!["Td".into(), "9d".into(), "6h".into()],
            dealer: 0,
        };
        let matrix = build_matrix_from_snapshot(snap);
        assert_eq!(matrix.dealer, 0, "dealer should pass through from snapshot");
        assert_eq!(matrix.player, 0);

        // Range-solver convention: dealer = 1
        let snap2 = MatrixSnapshot {
            player: 0,
            strategy: vec![],
            private_cards: vec![],
            initial_weights: vec![],
            num_hands: 0,
            actions: vec![],
            pot: 100,
            stacks: [900, 900],
            hand_evs: None,
            board: vec!["Td".into(), "9d".into(), "6h".into()],
            dealer: 1,
        };
        let matrix2 = build_matrix_from_snapshot(snap2);
        assert_eq!(matrix2.dealer, 1, "range-solver dealer convention");
    }
```

**Step 2: Run the test**

Run: `cargo test -p poker-solver-tauri test_matrix_snapshot_dealer_passthrough`

Expected: PASS.

**Step 3: Commit**

```
git add crates/tauri-app/src/postflop.rs
git commit -m "test: regression test for matrix snapshot dealer passthrough"
```

---

### Task 6: Run full test suite and verify

**Step 1: Run all tests**

Run: `cargo test`

Expected: All tests pass in under 1 minute.

**Step 2: Run clippy**

Run: `cargo clippy`

Expected: No new warnings.
