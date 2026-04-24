# Option A: DeepStack-style gadget as CFR tree modification — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `hex:executing-plans` to implement this plan task-by-task.

**Goal:** Replace the post-clamp gadget (bean `lay5`, obsoleted by iter-14 failure mode) with a Burch 2014 §3 structural CFR-D construction — two nested `Decision(player, [Terminate, Follow])` nodes prepended at the `PostFlopGame::node_arena` root, with `Terminate` children as per-hand opt-out boundary terminals. Safety (avg realized CFV ≥ opt-out, per-hand, per-gadget-player) is the ship-gating criterion.

**Architecture:** Gadget injection happens inside `PostFlopGame` at construction time via a new `with_config_and_gadget` constructor — the abstract `ActionTree` is untouched. The two gadget terminals use the existing depth-boundary mechanism with reserved ordinals 0 and 1. A simple `StaticGadgetEvaluator` returns pre-computed per-hand opt-out CFVs for those ordinals; existing cfvnet/exact evaluators handle ordinals 2+ via the already-present `per_boundary_evaluators` dispatch slot. Opt-out values come from `BlueprintCbvOptOut` extended with a new `opt_out_at_subgame_root` method that looks up the parent turn-chance CBV as an MVP approximation (safety holds under approximation; tighter bounds deferred to re-scoped bean `akg3`).

**Tech Stack:** Rust (stable), `cargo`, existing crates `range-solver` (DCFR solver, `PostFlopGame`, depth-boundary machinery), `poker-solver-core` (`CbvTable`, blueprint strategy), `tauri-app` (existing `OptOutProvider` / `BlueprintCbvOptOut`). No new dependencies.

**Design source:** `docs/plans/2026-04-24-option-a-deepstack-gadget-design.md`.
**Parent bean (implementation):** to be created as the first task below.

---

## Plan-stage decisions (resolved — no further user input needed)

Recorded here so implementers don't re-litigate:

1. **Arena layout:** prepend (gadget at arena indices 0..=3, all existing nodes shift by +4). Preserves the "children come after parent in arena order" invariant that `build_tree_recursive` depends on. External callers holding arena indices must walk from `game.root()`; this plan audits callsites in Task 9.
2. **Nesting order:** IP outer (G_IP at arena 0, G_OOP at arena 2, R0 at arena 4). `GadgetConfig.outer_player` field exposes this for A/B testing but MVP ships with IP outer per iter-14 data.
3. **Evaluator dispatch:** use the existing `per_boundary_evaluators: Vec<Arc<dyn BoundaryEvaluator>>` field on `PostFlopGame` (see `crates/range-solver/src/game/mod.rs:245` and dispatch at `game/evaluation.rs:111-123`). Populate `per_boundary_evaluators[0]` and `per_boundary_evaluators[1]` with `StaticGadgetEvaluator` instances. Ordinals 2+ fall through to the global `boundary_evaluator` (unchanged). No `DispatchingBoundaryEvaluator` wrapper — the existing dispatch is sufficient.
4. **CBV lookup at subgame root:** MVP uses the parent turn-chance CBV (stored in `CbvTable` and accessible via `CbvTable::build_node_to_ordinal_map`) as the opt-out for the decision-node subgame root. This is an approximation (averages over turn cards rather than the specific turn). Safety holds under approximation per Burch 2014 §3 — regret matching ensures realized ≥ opt-out regardless of opt-out tightness. Tighter bounds via backward-induction deferred to bean `akg3`.
5. **CLI flags:** keep `--gadget` (Option A, the new default when gadget is enabled); add `--gadget-clamp` as an opt-in flag routing to the old post-clamp `GadgetEvaluator` for diagnostic A/B comparison.

---

## Overall approach

- **Branching:** one feature branch `feat/option-a-gadget-tree` with a worktree per `hex:using-git-worktrees` when dispatched to agents.
- **TDD:** every task writes the failing test first, runs to confirm failure, implements minimally, runs to confirm pass, commits. Per CLAUDE.md manager-mode, tasks are dispatched to `rust-developer` agents (not written by the coordinator).
- **Commit discipline:** one commit per task (small, atomic). No `--no-verify`. Include bean updates in the task's commit.
- **Verification cadence:** after every task, run `cargo test -p <affected-crate>` with a <60s target per CLAUDE.md. After every phase, run the full workspace: `cargo test && cargo clippy`.

---

# Phase 1 — Domain: `StaticGadgetEvaluator` (simplest, no deps)

## Task 1: Create implementation bean

**Files:**
- None (bean only).

**Step 1:** Create the parent implementation bean.

```bash
beans create --json \
  "Implement Option A gadget (tree-structural CFR-D)" \
  -t feature -p high \
  -d "Implementation plan at docs/plans/2026-04-24-option-a-deepstack-gadget-plan.md.
Design at docs/plans/2026-04-24-option-a-deepstack-gadget-design.md.
Brainstorm closure in bean u3rf.
Blocks re-scoped akg3."
```

**Step 2:** Link dependencies.

```bash
# Link u3rf (brainstorm) as parent
beans update <new-id> --parent poker_solver_rust-u3rf
# Mark akg3 as blocked by this
beans update poker_solver_rust-akg3 --blocked-by <new-id>
```

**Step 3:** Commit the bean state.

```bash
git add .beans/ && git commit -m "chore(beans): open Option A implementation bean"
```

## Task 2: `StaticGadgetEvaluator` trait impl — unit tests first

**Files:**
- Create: `crates/tauri-app/src/gadget.rs` (add struct to existing file, after `GadgetEvaluator`)
- Test: same file, `#[cfg(test)] mod tests`

**Step 1:** Write the failing test in `crates/tauri-app/src/gadget.rs` `#[cfg(test)] mod tests`:

```rust
#[test]
fn static_gadget_evaluator_returns_stored_cfvs_regardless_of_reach() {
    let oop = vec![0.1, 0.2, 0.3];
    let ip  = vec![-0.5, -0.6];
    let eval = StaticGadgetEvaluator::new(oop.clone(), ip.clone());

    // compute_cfvs_both returns stored CFVs independent of reach/pot/etc.
    let (o, i) = eval.compute_cfvs_both(
        /* pot */      100,
        /* stack */    50.0,
        /* oop_reach*/ &[1.0; 3],
        /* ip_reach */ &[1.0; 2],
        /* num_oop */  3,
        /* num_ip  */  2,
        /* cont_idx */ 0,
    );
    assert_eq!(o, oop);
    assert_eq!(i, ip);

    // Reach values are ignored.
    let (o2, i2) = eval.compute_cfvs_both(
        50, 10.0, &[0.01; 3], &[0.01; 2], 3, 2, 0,
    );
    assert_eq!(o2, oop);
    assert_eq!(i2, ip);
}

#[test]
fn static_gadget_evaluator_compute_cfvs_single_side() {
    let oop = vec![0.1, 0.2];
    let ip  = vec![0.3, 0.4];
    let eval = StaticGadgetEvaluator::new(oop.clone(), ip.clone());

    assert_eq!(
        eval.compute_cfvs(0, 100, 50.0, &[1.0; 2], 2, 0),
        oop,
    );
    assert_eq!(
        eval.compute_cfvs(1, 100, 50.0, &[1.0; 2], 2, 0),
        ip,
    );
}
```

**Step 2:** Run the test, expect compile failure:

```bash
cargo test -p tauri-app static_gadget_evaluator 2>&1 | tail -20
```
Expected: `error[E0422]: cannot find struct StaticGadgetEvaluator in this scope`

**Step 3:** Implement minimally in the same file, below existing `GadgetEvaluator`:

```rust
/// BoundaryEvaluator that returns pre-stored per-hand CFVs, ignoring reach.
/// Used for the gadget Terminate terminals (boundary ordinals 0 and 1).
pub struct StaticGadgetEvaluator {
    oop_cfvs: Vec<f32>,
    ip_cfvs: Vec<f32>,
}

impl StaticGadgetEvaluator {
    pub fn new(oop_cfvs: Vec<f32>, ip_cfvs: Vec<f32>) -> Self {
        Self { oop_cfvs, ip_cfvs }
    }
}

impl range_solver::game::BoundaryEvaluator for StaticGadgetEvaluator {
    fn compute_cfvs(
        &self,
        player: usize,
        _pot: i32,
        _remaining_stack: f64,
        _opponent_reach: &[f32],
        _num_hands: usize,
        _continuation_index: usize,
    ) -> Vec<f32> {
        if player == 0 { self.oop_cfvs.clone() } else { self.ip_cfvs.clone() }
    }

    fn compute_cfvs_both(
        &self,
        _pot: i32,
        _remaining_stack: f64,
        _oop_reach: &[f32],
        _ip_reach: &[f32],
        _num_oop: usize,
        _num_ip: usize,
        _continuation_index: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        (self.oop_cfvs.clone(), self.ip_cfvs.clone())
    }
}
```

**Step 4:** Run tests, expect pass:

```bash
cargo test -p tauri-app static_gadget_evaluator -- --nocapture
```
Expected: `2 passed`

**Step 5:** Commit.

```bash
git add crates/tauri-app/src/gadget.rs
git commit -m "feat(gadget): add StaticGadgetEvaluator for gadget terminal CFVs

Returns pre-stored per-hand CFVs independent of reach — used at the
two gadget Terminate terminals (boundary ordinals 0/1). Plugs into
PostFlopGame::per_boundary_evaluators for ordinal-based dispatch."
```

---

# Phase 2 — Domain: `GadgetConfig` + `inject_gadget_layer` (arena-level tree modification)

## Task 3: `GadgetConfig` struct + stub module

**Files:**
- Create: `crates/range-solver/src/game/gadget.rs`
- Modify: `crates/range-solver/src/game/mod.rs` (add `mod gadget;`)

**Step 1:** Create `crates/range-solver/src/game/gadget.rs`:

```rust
//! Structural CFR-D gadget: two nested Decision(player, [Terminate, Follow])
//! nodes prepended at the PostFlopGame::node_arena root, using existing
//! depth-boundary terminals for per-hand opt-out CFVs.
//!
//! See docs/plans/2026-04-24-option-a-deepstack-gadget-design.md and
//! docs/plans/2026-04-24-option-a-deepstack-gadget-plan.md.

use crate::action_tree::*;

/// Gadget configuration passed to PostFlopGame::with_config_and_gadget.
#[derive(Debug, Clone)]
pub struct GadgetConfig {
    /// Per-hand OOP opt-out in bcfv units; len = num_private_hands(OOP).
    pub opt_out_oop: Vec<f32>,
    /// Per-hand IP opt-out in bcfv units; len = num_private_hands(IP).
    pub opt_out_ip: Vec<f32>,
    /// Which player's Decision is the outer gadget layer (0 = OOP, 1 = IP).
    /// Default: 1 (IP outer). Affects convergence dynamics but not safety
    /// guarantee — Burch §3 sufficiency holds at each gadget decision node
    /// independently of nesting order.
    pub outer_player: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gadget_config_constructs() {
        let cfg = GadgetConfig {
            opt_out_oop: vec![0.1; 3],
            opt_out_ip:  vec![0.2; 2],
            outer_player: 1,
        };
        assert_eq!(cfg.opt_out_oop.len(), 3);
        assert_eq!(cfg.opt_out_ip.len(), 2);
        assert_eq!(cfg.outer_player, 1);
    }
}
```

**Step 2:** Register the module in `crates/range-solver/src/game/mod.rs:1-5`:

```rust
mod evaluation;
mod interpreter;
mod node;
mod query;
pub mod gadget;  // NEW
```

**Step 3:** Verify compilation and test pass:

```bash
cargo test -p range-solver gadget_config_constructs
```
Expected: `1 passed`

**Step 4:** Commit.

```bash
git add crates/range-solver/src/game/{mod.rs,gadget.rs}
git commit -m "feat(gadget): scaffold range-solver gadget module with GadgetConfig"
```

## Task 4: `inject_gadget_layer` — tree-construction test (happy path)

**Files:**
- Modify: `crates/range-solver/src/game/gadget.rs`

**Step 1:** Add failing test:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::action_tree::ActionTree;
    use crate::card::flop_from_str;
    use crate::interface::PLAYER_DEPTH_BOUNDARY_FLAG;
    use crate::range::Range;
    use crate::bet_size::BetSizeOptions;
    use crate::game::{CardConfig, PostFlopGame};
    use crate::BoardState;

    fn minimal_game() -> PostFlopGame {
        let oop: Range = "AA,KK".parse().unwrap();
        let ip:  Range = "QQ,JJ".parse().unwrap();
        let cc = CardConfig {
            range: [oop, ip],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: crate::card::card_from_str("8d").unwrap(),
            river: crate::card::card_from_str("3s").unwrap(),
        };
        let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
        let tc = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 200,
            river_bet_sizes: [sizes.clone(), sizes],
            ..Default::default()
        };
        let at = ActionTree::new(tc).unwrap();
        PostFlopGame::with_config(cc, at).unwrap()
    }

    #[test]
    fn inject_gadget_layer_prepends_four_nodes() {
        let mut game = minimal_game();
        let arena_before = game.node_arena_len_for_test();

        let cfg = GadgetConfig {
            opt_out_oop: vec![0.0; game.num_private_hands(0)],
            opt_out_ip:  vec![0.0; game.num_private_hands(1)],
            outer_player: 1,
        };
        inject_gadget_layer(&mut game, &cfg);

        assert_eq!(
            game.node_arena_len_for_test(),
            arena_before + 4,
            "gadget injection should add exactly 4 nodes"
        );
    }

    #[test]
    fn inject_gadget_layer_root_is_gadget_decision() {
        let mut game = minimal_game();
        let cfg = GadgetConfig {
            opt_out_oop: vec![0.0; game.num_private_hands(0)],
            opt_out_ip:  vec![0.0; game.num_private_hands(1)],
            outer_player: 1,
        };
        inject_gadget_layer(&mut game, &cfg);

        let root = game.root();
        assert!(!root.is_terminal(), "root should be a decision node");
        assert!(!root.is_chance(),   "root should not be a chance node");
        assert_eq!(root.num_actions(), 2, "root should have [Terminate, Follow]");
        assert_eq!(root.acting_player(), 1, "IP is outer per config");
    }

    #[test]
    fn inject_gadget_layer_terminates_are_depth_boundaries() {
        let mut game = minimal_game();
        let cfg = GadgetConfig {
            opt_out_oop: vec![0.0; game.num_private_hands(0)],
            opt_out_ip:  vec![0.0; game.num_private_hands(1)],
            outer_player: 1,
        };
        inject_gadget_layer(&mut game, &cfg);

        // Terminate at arena 1 (after G_IP at arena 0)
        let t_ip = game.node_arena_get_for_test(1);
        assert!(t_ip.is_depth_boundary(), "G_IP.Terminate should be depth boundary");

        // Terminate at arena 3 (after G_OOP at arena 2)
        let t_oop = game.node_arena_get_for_test(3);
        assert!(t_oop.is_depth_boundary(), "G_OOP.Terminate should be depth boundary");
    }

    #[test]
    fn inject_gadget_layer_assigns_reserved_boundary_ordinals() {
        let mut game = minimal_game();
        let cfg = GadgetConfig {
            opt_out_oop: vec![0.0; game.num_private_hands(0)],
            opt_out_ip:  vec![0.0; game.num_private_hands(1)],
            outer_player: 1,
        };
        inject_gadget_layer(&mut game, &cfg);

        // G_IP.Terminate at arena 1 -> ordinal 0
        assert_eq!(game.boundary_ordinal(1), Some(0));
        // G_OOP.Terminate at arena 3 -> ordinal 1
        assert_eq!(game.boundary_ordinal(3), Some(1));
    }
}
```

**Step 2:** Add the test-only accessors to `PostFlopGame` in `crates/range-solver/src/game/mod.rs` (or `interpreter.rs`):

```rust
#[cfg(test)]
impl PostFlopGame {
    pub fn node_arena_len_for_test(&self) -> usize { self.node_arena.len() }
    pub fn node_arena_get_for_test(&self, idx: usize) -> std::sync::MutexGuard<'_, PostFlopNode> {
        // NOTE: adapt to MutexLike API — may need raw access
        unimplemented!("see Task 4 implementer note")
    }
}
```

Implementer note: `MutexLike` has custom lock semantics; use `node_arena[idx].lock()` or expose the existing `node_index(&node)` pattern. Confirm correct API at implementation time.

**Step 3:** Run tests, expect failure:

```bash
cargo test -p range-solver inject_gadget_layer -- --nocapture
```
Expected: all 4 tests fail with `inject_gadget_layer not defined` or similar.

**Step 4:** Implement `inject_gadget_layer` in `gadget.rs`. Pseudocode (implementer to fill in concrete Rust with careful attention to arena layout):

```rust
pub fn inject_gadget_layer(game: &mut PostFlopGame, config: &GadgetConfig) {
    assert_eq!(
        config.opt_out_oop.len(),
        game.num_private_hands(0),
        "opt_out_oop length must match OOP hand count"
    );
    assert_eq!(
        config.opt_out_ip.len(),
        game.num_private_hands(1),
        "opt_out_ip length must match IP hand count"
    );
    assert!(
        config.outer_player == 0 || config.outer_player == 1,
        "outer_player must be 0 or 1"
    );

    // 1. Shift all existing nodes by +4 positions in node_arena.
    //    Prepend 4 new PostFlopNode::default() entries at the front.
    //    Update each existing node's children_offset (it's relative, not
    //    absolute — verify: if relative, NO update needed; if absolute, +4).
    //    Inspect crates/range-solver/src/action_tree.rs line ~877 to confirm:
    //    `node.children_offset = (info.turn_index - node_index) as u32;`
    //    This is a RELATIVE offset from self_index. Since both self and
    //    children shift by the same amount, relative offsets are unchanged.

    // 2. Configure the 4 new nodes at arena indices 0..=3:
    //    [0] G_IP        (Decision, player = 1, num_children = 2, children_offset = 1)
    //        -> children at arena 1 (Terminate) and arena 2 (G_OOP)
    //    [1] G_IP.Terminate (depth-boundary terminal, prev_action = OptOut)
    //    [2] G_OOP       (Decision, player = 0, num_children = 2, children_offset = 1)
    //        -> children at arena 3 (Terminate) and arena 4 (former root, now shifted)
    //    [3] G_OOP.Terminate (depth-boundary terminal, prev_action = OptOut)

    //    For Decision nodes, set node.num_elements = 2 * num_private_hands(player).
    //    For depth-boundary terminals, set node.player = PLAYER_TERMINAL_FLAG
    //                                               | PLAYER_DEPTH_BOUNDARY_FLAG.

    // 3. Rebuild node_to_boundary mapping:
    //    New ordinal 0 = arena 1 (G_IP.Terminate)
    //    New ordinal 1 = arena 3 (G_OOP.Terminate)
    //    Each previous ordinal_k shifts: arena index_new = arena_index_old + 4,
    //    ordinal_new = ordinal_old + 2.

    // 4. Update boundary storage size:
    //    boundary_cfvs.len() should now be (old_count + 2) * 2 slots
    //    (2 players per boundary). Existing slots slide to indices 4..(old+2)*2;
    //    new slots 0..4 are for the two gadget boundaries.

    // 5. Pre-populate boundary_cfvs for the gadget terminals:
    //    boundary_cfvs[0 * 2 + 0] = vec![0.0; num_hands_oop]      (OOP neutralized at G_IP.Terminate)
    //    boundary_cfvs[0 * 2 + 1] = config.opt_out_ip.clone()     (IP takes opt_out at G_IP.Terminate)
    //    boundary_cfvs[1 * 2 + 0] = config.opt_out_oop.clone()    (OOP takes opt_out at G_OOP.Terminate)
    //    boundary_cfvs[1 * 2 + 1] = vec![0.0; num_hands_ip]       (IP neutralized at G_OOP.Terminate)

    // 6. If config.outer_player == 0 (OOP outer), swap the Decision
    //    acting_player values and the opt-out assignments accordingly.

    // 7. Reinvoke game.allocate_memory_nodes() so the new Decision nodes
    //    get correct storage pointers (storage1/storage2 for strategy/regrets).
    //    Verify: num_storage accounting was recomputed BEFORE allocation.

    // 8. Mark the PostFlopGame state appropriately — tree changes should
    //    not reset `State::MemoryAllocated` if we're re-wiring pointers.
}
```

Implementer: implement step-by-step, run tests after each step. This is the highest-risk task — careful arena arithmetic required. If in doubt, raise blockers back to the coordinator.

**Step 5:** Run tests, expect pass:

```bash
cargo test -p range-solver inject_gadget_layer
```
Expected: `4 passed`

**Step 6:** Commit.

```bash
git add crates/range-solver/src/game/gadget.rs crates/range-solver/src/game/mod.rs
git commit -m "feat(gadget): implement inject_gadget_layer with arena prepend

Prepends 4 nodes (G_IP decision, G_IP.Terminate, G_OOP decision,
G_OOP.Terminate) at arena indices 0..=3. Shifts existing tree by +4
(children_offset is relative so offsets themselves stay unchanged).
Sets reserved boundary ordinals 0 and 1, populates boundary_cfvs
for both gadget terminals per the neutralized-opponent convention."
```

## Task 5: Edge cases — `inject_gadget_layer` panics on invalid config

**Files:**
- Modify: `crates/range-solver/src/game/gadget.rs`

**Step 1:** Add failing tests:

```rust
#[test]
#[should_panic(expected = "opt_out_oop length must match")]
fn inject_panics_on_oop_length_mismatch() {
    let mut game = minimal_game();
    let cfg = GadgetConfig {
        opt_out_oop: vec![0.0; 99],  // wrong length
        opt_out_ip:  vec![0.0; game.num_private_hands(1)],
        outer_player: 1,
    };
    inject_gadget_layer(&mut game, &cfg);
}

#[test]
#[should_panic(expected = "outer_player must be 0 or 1")]
fn inject_panics_on_invalid_outer_player() {
    let mut game = minimal_game();
    let cfg = GadgetConfig {
        opt_out_oop: vec![0.0; game.num_private_hands(0)],
        opt_out_ip:  vec![0.0; game.num_private_hands(1)],
        outer_player: 42,
    };
    inject_gadget_layer(&mut game, &cfg);
}
```

**Step 2:** Run, expect pass (Task 4's impl already contains these asserts).

```bash
cargo test -p range-solver inject_panics -- --nocapture
```
Expected: `2 passed`

**Step 3:** Commit.

```bash
git add crates/range-solver/src/game/gadget.rs
git commit -m "test(gadget): assert length and player-index validation"
```

## Task 6: `inject_gadget_layer` with `outer_player=0` (OOP outer)

**Files:**
- Modify: `crates/range-solver/src/game/gadget.rs`

**Step 1:** Write failing test:

```rust
#[test]
fn inject_with_oop_outer_swaps_nesting() {
    let mut game = minimal_game();
    let cfg = GadgetConfig {
        opt_out_oop: vec![0.5; game.num_private_hands(0)],
        opt_out_ip:  vec![-0.5; game.num_private_hands(1)],
        outer_player: 0,  // OOP outer
    };
    inject_gadget_layer(&mut game, &cfg);

    let root = game.root();
    assert_eq!(root.acting_player(), 0, "OOP is now outer");

    // Boundary ordinal 0 = G_OOP.Terminate
    // Boundary ordinal 1 = G_IP.Terminate
    // CFVs at ordinal 0: OOP gets opt_out_oop, IP gets zeros.
    // CFVs at ordinal 1: IP gets opt_out_ip, OOP gets zeros.
    // (Verify via get_boundary_cfvs accessor or pull raw values.)
}
```

**Step 2:** Run, expect pass if Task 4 handles `outer_player` correctly; else diagnose and fix.

```bash
cargo test -p range-solver inject_with_oop_outer
```

**Step 3:** Commit.

```bash
git add crates/range-solver/src/game/gadget.rs
git commit -m "test(gadget): cover outer_player=0 (OOP outer) nesting path"
```

---

# Phase 3 — Coordination: `PostFlopGame::with_config_and_gadget`

## Task 7: New constructor wiring

**Files:**
- Modify: `crates/range-solver/src/game/interpreter.rs` (or `mod.rs` — wherever `with_config` lives)

**Step 1:** Failing test in `crates/range-solver/src/game/gadget.rs`:

```rust
#[test]
fn with_config_and_gadget_returns_ready_game() {
    // Same card/tree config as minimal_game(), but go through the new constructor.
    let oop: Range = "AA,KK".parse().unwrap();
    let ip:  Range = "QQ,JJ".parse().unwrap();
    let cc = CardConfig { /* ...as minimal_game()... */ };
    let tc = TreeConfig { /* ... */ };
    let at = ActionTree::new(tc).unwrap();

    let n_oop = cc.range[0].num_hands_approx(); // or use actual count
    let n_ip  = cc.range[1].num_hands_approx();

    let cfg = GadgetConfig {
        opt_out_oop: vec![0.1; n_oop],
        opt_out_ip:  vec![0.1; n_ip],
        outer_player: 1,
    };

    let game = PostFlopGame::with_config_and_gadget(cc, at, cfg).unwrap();
    // Root is the gadget; it should be a decision node.
    assert!(!game.root().is_terminal());
    // Two boundary terminals added.
    assert!(game.num_boundary_nodes() >= 2);
}
```

**Step 2:** Run, expect failure:

```bash
cargo test -p range-solver with_config_and_gadget
```

**Step 3:** Implement in `crates/range-solver/src/game/interpreter.rs` (same `impl PostFlopGame` block that has `with_config`):

```rust
pub fn with_config_and_gadget(
    card_config: CardConfig,
    action_tree: ActionTree,
    gadget_config: crate::game::gadget::GadgetConfig,
) -> Result<Self, String> {
    let mut game = Self::with_config(card_config, action_tree)?;
    // Allocate memory first so inject can read num_private_hands etc.
    // NOTE: allocate_memory returns after setting storage — inject will
    // reallocate nodes as needed. Implementer: verify order is correct
    // (allocate before or after inject?) — likely AFTER inject so the
    // 4 new decision nodes get storage too.
    crate::game::gadget::inject_gadget_layer(&mut game, &gadget_config);
    Ok(game)
}
```

**Step 4:** Run tests, expect pass:

```bash
cargo test -p range-solver with_config_and_gadget
```
Expected: `1 passed`

**Step 5:** Commit.

```bash
git add crates/range-solver/src/game/interpreter.rs crates/range-solver/src/game/gadget.rs
git commit -m "feat(gadget): add PostFlopGame::with_config_and_gadget constructor"
```

---

# Phase 4 — Domain: `BlueprintCbvOptOut::opt_out_at_subgame_root`

## Task 8: Implement root-lookup method

**Files:**
- Modify: `crates/tauri-app/src/gadget.rs`

**Step 1:** Write failing test in existing `#[cfg(test)] mod tests`:

```rust
#[test]
fn opt_out_at_subgame_root_returns_two_hand_vectors() {
    let (ctx, abstract_root, board, private_cards) = make_cbv_test_context();
    let subgame_pot_chips = 100.0f32;

    let [opt_oop, opt_ip] = BlueprintCbvOptOut::opt_out_at_subgame_root(
        &ctx,
        abstract_root,
        subgame_pot_chips,
        &board,
        &private_cards,
    );

    assert_eq!(opt_oop.len(), private_cards[0].len());
    assert_eq!(opt_ip.len(),  private_cards[1].len());
    // Every value should be finite and in a reasonable bcfv range.
    for v in opt_oop.iter().chain(opt_ip.iter()) {
        assert!(v.is_finite(), "opt-out bcfv must be finite");
        assert!(*v >= -10.0 && *v <= 10.0, "bcfv unreasonable: {v}");
    }
}

#[test]
fn opt_out_at_subgame_root_uses_parent_chance_cbv() {
    // Hand-built fixture where we know the expected value:
    // chance-parent CBV bucket 0 = 100 chips, half_pot = 50, so bcfv = +1.0.
    // Subgame pot = 100 chips -> half_pot = 50.
    // Expected: per-hand bcfv = (100 - 50) / 50 = +1.0 for every hand in bucket 0.
    let (ctx, abstract_root, board, private_cards) = make_cbv_test_context();
    let [opt_oop, opt_ip] = BlueprintCbvOptOut::opt_out_at_subgame_root(
        &ctx, abstract_root, 100.0, &board, &private_cards,
    );
    // Verify the values match what chip_cfv_to_bcfv produces.
    // (Actual expected value depends on the fixture's CbvTable; use existing
    //  make_cbv_test_context helper values — update this assertion once
    //  the implementer sees the fixture's CbvTable values.)
    assert!(opt_oop.iter().all(|&v| v.is_finite()));
    assert!(opt_ip.iter().all(|&v| v.is_finite()));
}
```

**Step 2:** Run, expect failure:

```bash
cargo test -p tauri-app opt_out_at_subgame_root
```

**Step 3:** Implement. Add as an associated function (not method) on `BlueprintCbvOptOut` in `crates/tauri-app/src/gadget.rs`:

```rust
impl BlueprintCbvOptOut {
    /// Compute per-hand opt-out bcfv values at the subgame ROOT (not at
    /// chance descendants). MVP approximation: uses the parent turn-chance
    /// CBV as the root's opt-out, normalised by the subgame's starting
    /// half-pot.
    ///
    /// Returns `[opt_out_oop, opt_out_ip]`.
    ///
    /// # Panics
    ///
    /// Panics if `subgame_pot_chips <= 0`, if the CbvTable is empty, or if
    /// no chance parent is found for `abstract_root`.
    pub fn opt_out_at_subgame_root(
        cbv_context: &crate::postflop::CbvContext,
        abstract_root: u32,
        subgame_pot_chips: f32,
        board: &[u8],
        private_cards: &[Vec<(u8, u8)>; 2],
    ) -> [Vec<f32>; 2] {
        use poker_solver_core::blueprint_v2::cbv::CbvTable;
        use poker_solver_core::blueprint_v2::Street;

        assert!(subgame_pot_chips > 0.0, "subgame_pot_chips must be positive");
        assert!(
            !cbv_context.cbv_table.values.is_empty(),
            "CbvTable has no values; cannot construct opt-out"
        );

        // Find the parent chance node of abstract_root.
        // GameTree stores chance_descendants(); we need the NEAREST-CHANCE-
        // ANCESTOR. Implementer: add GameTree::nearest_chance_ancestor(node)
        // in poker-solver-core if it doesn't exist, or traverse parents
        // manually.
        let ordinal_map = CbvTable::build_node_to_ordinal_map(&cbv_context.abstract_tree);
        let chance_ancestor = cbv_context.abstract_tree
            .nearest_chance_ancestor(abstract_root)
            .expect("subgame root must have a chance ancestor in the abstract tree");
        let cbv_ordinal = CbvTable::require_ordinal(&ordinal_map, chance_ancestor);

        let half_pot = subgame_pot_chips / 2.0;
        assert!(half_pot > 0.0);

        let street = match board.len() {
            3 => Street::Flop,
            4 => Street::Turn,
            5 => Street::River,
            n => panic!("unexpected board length {n}"),
        };

        let rs_board: Vec<poker_solver_core::poker::Card> = board
            .iter()
            .map(|&id| crate::exploration::range_solver_to_rs_card(id))
            .collect();

        let mut per_hand: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
        for player in 0..2 {
            let hands = &private_cards[player];
            per_hand[player].reserve(hands.len());
            for &(c1, c2) in hands {
                let rs_c1 = crate::exploration::range_solver_to_rs_card(c1);
                let rs_c2 = crate::exploration::range_solver_to_rs_card(c2);
                let bucket = cbv_context.all_buckets.get_bucket(
                    street, [rs_c1, rs_c2], &rs_board,
                );
                let chip_cbv = cbv_context.cbv_table.lookup(cbv_ordinal, bucket as usize);
                per_hand[player].push(chip_cfv_to_bcfv(chip_cbv, half_pot));
            }
        }
        per_hand
    }
}
```

**Step 4:** Implement `GameTree::nearest_chance_ancestor` in `crates/core/src/blueprint_v2/game_tree.rs` (if not already present):

```rust
impl GameTree {
    /// Walk up from `node_idx` following parent pointers until a Chance
    /// node is found. Returns its arena index, or `None` if none exists.
    pub fn nearest_chance_ancestor(&self, node_idx: u32) -> Option<u32> {
        // Implementer note: GameTree may not store parent pointers directly.
        // If not, either add parent pointers during tree construction OR
        // do a BFS from root matching child references. For MVP, either
        // works — BFS is ~O(N) per call and opt_out_at_subgame_root is
        // called once per subgame setup, so acceptable.
        let mut parent = vec![None; self.nodes.len()];
        for (idx, node) in self.nodes.iter().enumerate() {
            match node {
                GameNode::Decision { children, .. } => {
                    for &c in children {
                        parent[c as usize] = Some(idx as u32);
                    }
                }
                GameNode::Chance { child, .. } => {
                    parent[*child as usize] = Some(idx as u32);
                }
                GameNode::Terminal { .. } => {}
            }
        }
        let mut cur = node_idx;
        while let Some(p) = parent[cur as usize] {
            if matches!(self.nodes[p as usize], GameNode::Chance { .. }) {
                return Some(p);
            }
            cur = p;
        }
        None
    }
}
```

Add a unit test for `nearest_chance_ancestor` in the same file.

**Step 5:** Run tests, expect pass:

```bash
cargo test -p tauri-app opt_out_at_subgame_root
cargo test -p poker-solver-core nearest_chance_ancestor
```

**Step 6:** Commit.

```bash
git add crates/tauri-app/src/gadget.rs crates/core/src/blueprint_v2/game_tree.rs
git commit -m "feat(gadget): add BlueprintCbvOptOut::opt_out_at_subgame_root

MVP approximation: uses the parent chance-ancestor CBV (already stored
in CbvTable) as the subgame-root opt-out. Safety holds under approximation
per Burch 2014 §3 — regret matching ensures realized ≥ opt_out regardless
of tightness. Tighter backward-induction bound deferred to re-scoped akg3."
```

---

# Phase 5 — Coordination: `per_boundary_evaluators` wiring

## Task 9: Audit external arena-index callers

**Files:**
- None (investigation only).

**Step 1:** Grep for arena-index usage that might break after the +4 shift:

```bash
grep -rn "node_index\|node_arena\[\|boundary_node_indices\|root().*lock" \
  crates/ --include="*.rs" | grep -v test
```

**Step 2:** For each hit, determine:
- Does the caller derive the index from `game.root()` or `game.boundary_node_indices()`? (Safe — indices shift consistently.)
- Does the caller hardcode an index like `node_arena[5].lock()`? (Breaks.)

**Step 3:** Document findings in a comment at the top of `crates/range-solver/src/game/gadget.rs`:

```rust
//! Known callers holding arena indices (audited 2026-04-24, Task 9):
//! - <list findings here>
//! All confirmed safe under the +4 arena shift.
```

If any breakage is found, file a follow-up bean and address in the same commit.

**Step 4:** Commit (documentation-only unless fixes needed).

```bash
git add crates/range-solver/src/game/gadget.rs
git commit -m "docs(gadget): audit external arena-index callers for +4 shift safety"
```

## Task 10: Wire `StaticGadgetEvaluator` into `per_boundary_evaluators`

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs` or wherever the hybrid-mode wiring lives today (find via grep: `per_boundary_evaluators`).

**Step 1:** Failing integration test in a new test file `crates/tauri-app/tests/gadget_integration.rs`:

```rust
#[test]
fn gadget_solve_reads_static_cfvs_at_ordinals_0_and_1() {
    // Build a minimal PostFlopGame via with_config_and_gadget with known
    // opt-outs; confirm solve_step() reads them without calling any
    // dynamic evaluator.

    // Outline:
    // 1. Create GadgetConfig with opt_out_oop = vec![0.7; n_oop],
    //    opt_out_ip = vec![-0.3; n_ip].
    // 2. Build game via with_config_and_gadget.
    // 3. Allocate memory.
    // 4. Set game.per_boundary_evaluators = vec![
    //        Arc::new(StaticGadgetEvaluator::new(vec![0.0; n_oop], vec![-0.3; n_ip])),
    //        Arc::new(StaticGadgetEvaluator::new(vec![0.7; n_oop], vec![0.0; n_ip])),
    //    ];
    //    game.boundary_evaluator = Some(Arc::new(ExactSubtreeEvaluator)); // or cfvnet mock
    // 5. Run solve_step(game, 0).
    // 6. Verify game.boundary_cfvs[0*2+1] (IP at G_IP.T) equals vec![-0.3; n_ip].
    //    Verify game.boundary_cfvs[1*2+0] (OOP at G_OOP.T) equals vec![0.7; n_oop].
}
```

**Step 2:** Run, expect failure until `with_config_and_gadget` auto-populates `per_boundary_evaluators` OR the caller does so manually:

```bash
cargo test -p tauri-app --test gadget_integration gadget_solve_reads_static
```

**Step 3:** Two implementation options; implementer picks based on what's simpler:

**Option A:** `inject_gadget_layer` auto-populates `per_boundary_evaluators[0]` and `[1]` with `StaticGadgetEvaluator` instances. Caller doesn't need to know. Lives in `range-solver` but requires `StaticGadgetEvaluator` in `tauri-app` — circular dep unless we move the evaluator to `range-solver`.

**Option B:** `inject_gadget_layer` ONLY does the tree work; caller wires `per_boundary_evaluators` manually (in `tauri-app::exploration`). Recommended — keeps `range-solver` evaluator-agnostic.

Implement Option B: add a helper in `tauri-app` that composes `with_config_and_gadget` + sets up evaluators:

```rust
// in crates/tauri-app/src/gadget.rs or postflop.rs
pub fn make_gadget_game(
    card_config: CardConfig,
    action_tree: ActionTree,
    gadget_config: GadgetConfig,
    inner_evaluator: Arc<dyn BoundaryEvaluator>,
) -> Result<PostFlopGame, String> {
    let mut game = PostFlopGame::with_config_and_gadget(
        card_config, action_tree, gadget_config.clone(),
    )?;
    game.allocate_memory(false);

    let n_oop = game.num_private_hands(0);
    let n_ip  = game.num_private_hands(1);

    // Ordinal 0 = G_IP.Terminate: IP takes opt_out_ip, OOP neutralized.
    let eval_ord_0 = Arc::new(StaticGadgetEvaluator::new(
        vec![0.0; n_oop], gadget_config.opt_out_ip,
    ));
    // Ordinal 1 = G_OOP.Terminate: OOP takes opt_out_oop, IP neutralized.
    let eval_ord_1 = Arc::new(StaticGadgetEvaluator::new(
        gadget_config.opt_out_oop, vec![0.0; n_ip],
    ));
    game.per_boundary_evaluators = vec![eval_ord_0, eval_ord_1];
    game.boundary_evaluator = Some(inner_evaluator);

    Ok(game)
}
```

(Remember to invert these if `outer_player == 0`.)

**Step 4:** Run test, expect pass:

```bash
cargo test -p tauri-app --test gadget_integration gadget_solve_reads_static
```

**Step 5:** Commit.

```bash
git add crates/tauri-app/src/gadget.rs crates/tauri-app/tests/gadget_integration.rs
git commit -m "feat(gadget): wire StaticGadgetEvaluator via make_gadget_game helper

Populates per_boundary_evaluators[0/1] with static opt-out returns;
existing cfvnet/exact evaluator fills boundary_evaluator for ordinals 2+.
Uses the existing hybrid-mode dispatch (per_boundary_evaluators first,
boundary_evaluator fallback) — no wrapper needed."
```

---

# Phase 6 — Integration tests: parity & safety (ship gates)

## Task 11: Test (3) — gadget-off parity with `ConstantOptOut(-inf)`

**Files:**
- Modify: `crates/tauri-app/tests/gadget_integration.rs`

**Step 1:** Failing test:

```rust
#[test]
fn gadget_with_neg_inf_opt_out_matches_no_gadget() {
    // Build two games: one with gadget using ConstantOptOut(f32::NEG_INFINITY),
    // one without gadget. Solve both with identical seed/iters. Compare
    // final average strategies at all shared decision nodes (arena index
    // 4+ in the gadget game, arena index 0+ in the non-gadget game, i.e.
    // mapping gadget_arena_idx - 4 -> non_gadget_arena_idx).

    let oop: Range = "AA,KK".parse().unwrap();
    let ip:  Range = "QQ,JJ".parse().unwrap();
    // ... build common card_config, tree_config, action_tree ...

    // Non-gadget game
    let mut g_noop = PostFlopGame::with_config(cc.clone(), at.clone()).unwrap();
    g_noop.allocate_memory(false);
    g_noop.boundary_evaluator = Some(Arc::new(SubtreeExactEvaluator::default()));
    solver::solve(&mut g_noop, 50, 1e-4, false);

    // Gadget game with opt_out = -infinity (T-branch dominated everywhere)
    let cfg = GadgetConfig {
        opt_out_oop: vec![f32::NEG_INFINITY; n_oop],
        opt_out_ip:  vec![f32::NEG_INFINITY; n_ip],
        outer_player: 1,
    };
    let mut g_gad = make_gadget_game(
        cc, at, cfg, Arc::new(SubtreeExactEvaluator::default()),
    ).unwrap();
    solver::solve(&mut g_gad, 50, 1e-4, false);

    // Compare strategies at shared decision nodes.
    // For each node idx in g_noop (0..N), corresponding idx in g_gad is (idx + 4).
    // Strategy arrays should match within 1e-6 after the same number of iters.
    for noop_idx in 0..g_noop.node_arena_len_for_test() {
        let noop_node = g_noop.node_arena_get_for_test(noop_idx);
        if noop_node.is_terminal() || noop_node.is_chance() { continue; }
        let gad_node = g_gad.node_arena_get_for_test(noop_idx + 4);
        let noop_strat = noop_node.strategy();
        let gad_strat  = gad_node.strategy();
        assert_eq!(
            noop_strat.len(), gad_strat.len(),
            "strategy length mismatch at node {noop_idx}",
        );
        for (a, b) in noop_strat.iter().zip(gad_strat.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "strategy drift at node {noop_idx}: noop={a} vs gad={b}",
            );
        }
    }
}
```

**Step 2:** Run, expect pass (assuming Tasks 1-10 are correct):

```bash
cargo test -p tauri-app --test gadget_integration gadget_with_neg_inf -- --nocapture
```

**Step 3:** If the test fails, diagnose the divergence point:
- Regret at the gadget decision node under `opt_out = -infinity` should dominate F branch, so T never taken. Verify via a single-iter inspection (add a debug println).
- If strategy drift is >1e-5, the gadget is leaking something — likely a reach scaling issue or boundary cache inconsistency.

**Step 4:** Commit once green:

```bash
git add crates/tauri-app/tests/gadget_integration.rs
git commit -m "test(gadget): test (3) — gadget-off parity with ConstantOptOut(-inf)

Validates that the gadget is a no-op when T-branch is dominated, ensuring
we haven't broken the underlying solver path."
```

## Task 12: Test (4) — safety invariant

**Files:**
- Modify: `crates/tauri-app/tests/gadget_integration.rs`

**Step 1:** Failing test:

```rust
#[test]
fn gadget_safety_invariant_realized_cfv_geq_opt_out() {
    // Build a game with non-trivial opt-out values; solve for N iters;
    // assert realized CFV at each gadget decision ≥ opt_out per hand.

    let oop: Range = "AA,KK,QQ".parse().unwrap();
    let ip:  Range = "TT,99,88".parse().unwrap();
    // ...card/tree config; pick spot where we know opt-outs differ per-hand...

    let cfg = GadgetConfig {
        opt_out_oop: vec![0.3; n_oop],   // non-trivial, non-uniform OK
        opt_out_ip:  vec![-0.1; n_ip],
        outer_player: 1,
    };
    let mut game = make_gadget_game(
        cc, at, cfg.clone(), Arc::new(SubtreeExactEvaluator::default()),
    ).unwrap();

    let iters = 200;
    let warmup = iters / 2;
    solver::solve(&mut game, iters, 1e-4, false);
    finalize(&mut game);  // convert regrets to average strategies

    // Extract the gadget decision nodes:
    //   outer (G_IP) at arena 0, with player=1
    //   inner (G_OOP) at arena 2, with player=0

    // For each hand of the gadget player, compute realized CFV at that node:
    //   realized_CFV[h] = σ(T|h) * V_T[h] + σ(F|h) * V_F[h]
    // where V_T = pre-set opt_out, V_F comes from the subtree CFV.

    // Use game.cfvalues_at_node(node_idx) + game.strategy_at_node(node_idx)
    // to pull the needed data.
    // (Implementer: verify these getter names against the PostFlopGame API;
    //  add them if missing.)

    let tolerance = 0.01_f32;

    // G_IP at arena 0 — gadget player is IP
    let outer = game.node_arena_get_for_test(0);
    assert_eq!(outer.acting_player(), 1);  // IP
    let strat_outer = outer.strategy();    // [num_hands * 2] (T, F actions)
    let cfvs_outer  = outer.cfvalues();    // if separate storage; else reconstruct

    // Safety check: For each IP hand, realized CFV ≥ opt_out_ip[hand] - tolerance.
    for (h, &opt) in cfg.opt_out_ip.iter().enumerate() {
        // Extract σ(T|h), σ(F|h); V_T[h] = opt; V_F[h] from subtree.
        // Compute realized = σ(T|h) * opt + σ(F|h) * V_F[h].
        // Assert: realized ≥ opt - tolerance.
        // Implementer: complete the arithmetic based on the storage layout.
    }

    // G_OOP at arena 2 — gadget player is OOP
    let inner = game.node_arena_get_for_test(2);
    assert_eq!(inner.acting_player(), 0);  // OOP
    for (h, &opt) in cfg.opt_out_oop.iter().enumerate() {
        // Analogous safety check for OOP.
    }
}
```

**Step 2:** This test is the FORMAL SAFETY CLAIM. If it fails, the gadget isn't safe — do not ship. Expected behavior: after ~200 iters with tolerance 0.01, it passes.

**Step 3:** If the test doesn't compile (getter APIs missing), add helper methods to `PostFlopGame`:

```rust
#[cfg(test)]
impl PostFlopGame {
    /// Returns (σ(T|h), σ(F|h), V_T[h], V_F[h]) per gadget player hand.
    /// Used only in gadget safety tests.
    pub fn gadget_decision_diagnostic(
        &self,
        gadget_arena_idx: usize,
        hand: usize,
    ) -> (f32, f32, f32, f32) {
        // Implementer: pull from storage1 (strategy), storage2 (cfvalues).
        // Numeric layout: [num_hands] per action.
        unimplemented!()
    }
}
```

**Step 4:** Run, expect pass:

```bash
cargo test -p tauri-app --test gadget_integration gadget_safety_invariant
```

**Step 5:** Commit.

```bash
git add crates/tauri-app/tests/gadget_integration.rs crates/range-solver/src/game/mod.rs
git commit -m "test(gadget): test (4) — Burch §3 safety invariant

For every gadget player hand, avg_realized_CFV ≥ opt_out[hand] - 0.01
after 200 iters with warmup=100. This is the formal safety claim we
ship. If this test fails, the gadget is not safe."
```

---

# Phase 7 — Adapter: CLI / compare-solve wiring

## Task 13: `--gadget-mode` flag routing

**Files:**
- Modify: `crates/poker-solver-trainer/src/bin/<compare-solve>.rs` (find exact filename via `find crates/poker-solver-trainer -name "*.rs" | xargs grep -l 'compare-solve'`)
- Modify: `crates/tauri-app/src/` (if compare-solve lives there)

**Step 1:** Write failing CLI-level test (may be an integration test in the trainer crate):

```rust
// tests/cli_gadget_mode.rs
#[test]
fn compare_solve_with_gadget_flag_uses_option_a() {
    // Run `compare-solve --gadget --iters 10 ...`. Assert:
    // (a) exit code 0
    // (b) stdout contains "gadget mode: tree" (or similar indicator)
    //     to distinguish from old post-clamp path.
    // (c) no panic.
}

#[test]
fn compare_solve_with_gadget_clamp_flag_uses_legacy_postclamp() {
    // Run `compare-solve --gadget-clamp --iters 10 ...`. Assert:
    // (a) exit code 0
    // (b) stdout contains "gadget mode: clamp"
}
```

**Step 2:** Run, expect failure.

**Step 3:** Implement the flag routing:

```rust
// At the argparse level:
//   --gadget          enables Option A (tree) — DEFAULT when gadget is on
//   --gadget-clamp    enables legacy post-clamp (diagnostic-only)
//   (mutual exclusion: --gadget-clamp implies --gadget; cannot both be "on")
//
// In the solve setup:
//   if args.gadget_clamp {
//       // use old GadgetEvaluator path (unchanged)
//   } else if args.gadget {
//       // build game via make_gadget_game (new path)
//   } else {
//       // no gadget
//   }
//
// Print the mode indicator at startup:
//   eprintln!("gadget mode: {}", if gadget_clamp { "clamp" } else if gadget { "tree" } else { "off" });
```

**Step 4:** Run tests:

```bash
cargo test -p poker-solver-trainer cli_gadget_mode
```

**Step 5:** Commit.

```bash
git add crates/poker-solver-trainer/
git commit -m "feat(cli): route --gadget to Option A tree path; add --gadget-clamp diagnostic"
```

## Task 14: End-to-end harness run on `JhTh9h|…|7d` (test 5 — informational)

**Files:**
- Create: `docs/progress/2026-04-24-option-a-iter-15.md` (progress log)

**Step 1:** Run the harness:

```bash
cargo build --release -p poker-solver-trainer
./target/release/poker-solver-trainer compare-solve \
    --bundle ./local_data/blueprints/1k_100bb_brdcfr_v2 \
    --snapshot snapshot_0013 \
    --spot 'sb:2bb,bb:10bb,sb:22bb,bb:call|JhTh9h|bb:15bb,sb:call|7d' \
    --river-boundary cfvnet \
    --river-model ./local_data/models/cfvnet_river_py_v2/checkpoint_epoch675.onnx \
    --gadget --iters 40 --tolerance 0.001 \
    2>&1 | tee /tmp/option_a_iter15.log
```

**Step 2:** Analyze output: pull `exact_exp`, `subgame_exp`, `worst_delta`. Record in `docs/progress/2026-04-24-option-a-iter-15.md`:

```markdown
# Iteration 15 — Option A gadget tree E2E

**Spot:** `JhTh9h|…|7d` (same as iter 14)
**Compare against:** iter 10 baseline (no gadget) = 21k mbb; iter 14 (post-clamp) = 40k mbb.

**Result:**
- `exact_exp`:    ???
- `subgame_exp`:  ???
- `worst_delta`:  ???

**Verdict (soft, informational):** [one sentence: gadget beats / matches / trails iter 10 baseline]

**Safety invariant check:** test (4) passed / failed [link to CI run]

**Commentary:**
- [sentences on what the result means for akg3's rescoping]
```

**Step 3:** Commit the progress log:

```bash
git add docs/progress/2026-04-24-option-a-iter-15.md
git commit -m "docs(progress): iteration 15 — Option A gadget tree E2E result"
```

---

# Phase 8 — Docs + cleanup

## Task 15: Update `docs/architecture.md`

**Files:**
- Modify: `docs/architecture.md`

**Step 1:** Add a section describing the gadget-at-root construction:

```markdown
## Safe subgame solving — CFR-D gadget (Option A)

The subgame solver (range-solver::PostFlopGame) supports a safe
re-solving gadget per the Burch 2014 §3 CFR-D construction. When
enabled (via `--gadget` CLI or the `enable_gadget` Tauri command flag),
`PostFlopGame::with_config_and_gadget` prepends two nested
`Decision(player, [Terminate, Follow])` nodes at arena indices 0–3,
with Terminate children as depth-boundary terminals at reserved
boundary ordinals 0 and 1. Per-hand opt-out CFVs come from the
blueprint's CbvTable via `BlueprintCbvOptOut::opt_out_at_subgame_root`
(MVP: parent-chance approximation). Regret matching at each gadget
Decision node ensures the gadget player's realized CFV ≥ opt_out,
bounding subgame exploitability per Brown & Sandholm 2017 Theorem 2.

See `docs/plans/2026-04-24-option-a-deepstack-gadget-design.md`.
```

**Step 2:** Commit.

```bash
git add docs/architecture.md
git commit -m "docs(architecture): document Option A gadget-at-root construction"
```

## Task 16: Update `docs/training.md` if CLI changed

**Files:**
- Modify: `docs/training.md`

**Step 1:** Document `--gadget` (now Option A, default) and `--gadget-clamp` (diagnostic-only, legacy).

**Step 2:** Commit.

```bash
git add docs/training.md
git commit -m "docs(training): document --gadget (tree) and --gadget-clamp (legacy) flags"
```

## Task 17: Close implementation bean

**Files:**
- Bean update.

**Step 1:** Update the implementation bean to completed with a summary pointing at the design doc, progress log, and tests.

```bash
beans update <impl-bean-id> -s completed --body-append "

## Summary of Changes

[sentence on what was built]

**Tests:** (1) tree-construction, (2) opt-out placement, (3) gadget-off parity,
(4) safety invariant — all passing. (5) end-to-end harness: see
docs/progress/2026-04-24-option-a-iter-15.md.

**Files:** <list new + modified>"
```

**Step 2:** Decide akg3: completed / still-relevant / scrap.

Based on the iter-15 result:
- If `subgame_exp` ≈ iter-10 baseline: akg3 is LOW-priority; mark deferred.
- If `subgame_exp` ≫ iter-10 baseline: akg3 is HIGH-priority (opt-out tightness matters); mark todo.
- If project direction has shifted away from exploitability tightness entirely: scrap akg3.

**Step 3:** Commit.

```bash
git add .beans/
git commit -m "chore(beans): close Option A implementation bean; triage akg3"
```

## Task 18: Open follow-up bean — retire `GadgetEvaluator` (post-clamp)

**Files:**
- Bean.

**Step 1:** Open a follow-up:

```bash
beans create --json \
  "Retire GadgetEvaluator (post-clamp) now that Option A ships" \
  -t task -p low \
  -d "Bean lay5's GadgetEvaluator is retained in crates/tauri-app/src/gadget.rs
for A/B diagnostic comparison during Option A rollout. Once Option A ships
with test (4) safety invariant passing (see u3rf / this-impl-bean), delete
GadgetEvaluator along with --gadget-clamp CLI flag. Update docs/training.md
to remove the --gadget-clamp mention."
```

---

# Verification checklist (end of plan)

Before declaring the plan complete:

- [ ] All 18 tasks committed, each with a passing test.
- [ ] `cargo test` full workspace passes, under 60 seconds (per CLAUDE.md).
- [ ] `cargo clippy` clean.
- [ ] Test (4) safety invariant passes — formal Burch §3 check.
- [ ] `docs/architecture.md` documents the gadget.
- [ ] `docs/training.md` documents the CLI flag changes.
- [ ] `docs/progress/2026-04-24-option-a-iter-15.md` records the end-to-end result.
- [ ] Implementation bean closed with summary.
- [ ] `akg3` re-scoped based on iter-15 data.
- [ ] Follow-up bean opened for GadgetEvaluator retirement.

---

# References

- **Design doc:** `docs/plans/2026-04-24-option-a-deepstack-gadget-design.md`
- **Brainstorm bean:** `poker_solver_rust-u3rf`
- **Iteration history:** `docs/progress/2026-04-22-subgame-exact-parity.md`
- **Canonical CFR-D citation:** [Burch, Johanson, Bowling 2014. Solving Imperfect-Information Games Using Decomposition. AAAI-14.](https://webdocs.cs.ualberta.ca/~bowling/papers/14aaai-cfrd.pdf)
- **DeepStack-Leduc reference:** [lifrordi/DeepStack-Leduc](https://github.com/lifrordi/DeepStack-Leduc) (used Pattern 2 algorithmic; we chose Pattern 1 structural)
- **Safety theorem 2Δ bound:** [Brown & Sandholm 2017. Safe and Nested Subgame Solving.](https://arxiv.org/abs/1705.02955)

---

# Sub-skills referenced

- `hex:using-git-worktrees` — create feature branch worktree before dispatching
- `hex:executing-plans` — agent uses this to walk the plan task-by-task
- `hex:subagent-driven-development` — parallel implementer-reviewer streams if chosen
- `hex:test-driven-development` — TDD discipline per task
- `hex:verification-before-completion` — run tests before claiming done
- `hex:requesting-code-review` — review cycle for each task's PR
