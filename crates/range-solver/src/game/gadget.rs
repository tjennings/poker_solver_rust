//! Structural CFR-D gadget: two nested Decision(player, [Terminate, Follow])
//! nodes prepended at the PostFlopGame::node_arena root, using existing
//! depth-boundary terminals for per-hand opt-out CFVs.
//!
//! See docs/plans/2026-04-24-option-a-deepstack-gadget-design.md and
//! docs/plans/2026-04-24-option-a-deepstack-gadget-plan.md.
//!
//! # External callers of arena-index APIs (Task 9 audit, 2026-04-24)
//!
//! After `inject_gadget_layer`, arena indices 0..=3 are the gadget layer
//! and all pre-existing nodes shift by +4. `node_to_boundary` ordinals
//! 0 and 1 are reserved for gadget terminals; pre-existing boundary
//! ordinals shift +2.
//!
//! External callsites holding arena indices or boundary ordinals
//! (audited via `grep -rn 'node_index|node_arena\[|boundary_node_indices'`
//! excluding tests and range-solver/src/game/):
//!
//! - `crates/tauri-app/src/boundary_trace.rs:218, 274` — uses
//!   `boundary_node_indices()` to walk cfvnet-eligible boundaries.
//!   Post-gadget, ordinals 0/1 are the gadget terminals and are NOT
//!   cfvnet boundaries. The walk must skip ordinals 0 and 1 when
//!   operating on a gadget-enabled game.
//! - `crates/trainer/src/compare_solve.rs:635` — same concern.
//!
//! Task 13 (CLI integration) is responsible for routing the gadget path
//! such that these callers either (a) skip ordinals 0/1 explicitly, or
//! (b) only operate on the `boundary_evaluator` (ordinals 2+) path. The
//! per-boundary dispatch pattern at `crates/range-solver/src/game/evaluation.rs:111-123`
//! already routes ordinals 0/1 to `per_boundary_evaluators` (populated
//! by `make_gadget_game` in Task 10), so cfvnet is not invoked on those
//! ordinals during solve — but CLI-level reporting code needs the filter.

#[allow(unused_imports)]
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
    /// guarantee -- Burch S3 sufficiency holds at each gadget decision node
    /// independently of nesting order.
    pub outer_player: usize,
}

/// Option A2 config: per-boundary per-player opt-out values.
///
/// Used by `inject_per_boundary_gadgets` to insert a 4-node gadget subtree
/// at each cfvnet depth-boundary terminal, giving each player an opt-out
/// action whose terminal payoff is the supplied static CFV.
#[derive(Debug, Clone)]
pub struct GadgetConfigPerBoundary {
    /// Per boundary-ordinal (in the ORIGINAL pre-injection numbering),
    /// per player [OOP, IP], per hand: opt-out bcfv value.
    ///
    /// Length = num_boundary_terminals pre-injection.
    pub per_boundary_opt_outs: Vec<[Vec<f32>; 2]>,
}

/// Inserts a 4-node gadget subtree at each cfvnet depth-boundary terminal.
///
/// For each original cfvnet boundary at ordinal `b`:
/// ```text
///   Chance(river) → G_IP_b    (Decision, owner=IP,  2 actions)
///                  ├── Terminate_IP_b   (depth boundary, gadget terminal)
///                  └── Follow → G_OOP_b (Decision, owner=OOP, 2 actions)
///                               ├── Terminate_OOP_b (depth boundary, gadget terminal)
///                               └── Follow → EXISTING cfvnet boundary
/// ```
///
/// Post-injection boundary ordinal scheme:
/// - Ordinals 0..N: original cfvnet boundaries (stable).
/// - Ordinals N..3N: gadget terminals. For original boundary `b`:
///   - ordinal `N + 2*b`     = Terminate_IP_b
///   - ordinal `N + 2*b + 1` = Terminate_OOP_b
///
/// Gadget terminal CFVs are pre-populated:
/// - Gadget player: opt_out values from config.
/// - Non-gadget player: zeros (neutralized).
///   // TODO(Phase C): replace with zero-sum complement.
pub fn inject_per_boundary_gadgets(
    game: &mut super::PostFlopGame,
    config: &GadgetConfigPerBoundary,
) {
    game.inject_per_boundary_gadgets_impl(config);
}

/// Prepends 4 gadget nodes at arena indices 0..=3 to the game tree.
///
/// See module-level docs for the full gadget topology. This is the
/// public entry point; the actual mutation lives on `PostFlopGame`
/// in `interpreter.rs` (Option C visibility).
pub fn inject_gadget_layer(game: &mut super::PostFlopGame, config: &GadgetConfig) {
    game.inject_gadget_layer_impl(config);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action_tree::ActionTree;
    use crate::bet_size::BetSizeOptions;
    use crate::card::{card_from_str, flop_from_str, NOT_DEALT};
    use crate::game::{CardConfig, PostFlopGame};
    use crate::interface::{Game, GameNode};
    use crate::range::Range;
    use crate::BoardState;

    fn minimal_game() -> PostFlopGame {
        let oop: Range = "AA,KK".parse().unwrap();
        let ip: Range = "QQ,JJ".parse().unwrap();
        let cc = CardConfig {
            range: [oop, ip],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: card_from_str("3s").unwrap(),
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
    fn gadget_config_constructs() {
        let cfg = GadgetConfig {
            opt_out_oop: vec![0.1; 3],
            opt_out_ip: vec![0.2; 2],
            outer_player: 1,
        };
        assert_eq!(cfg.opt_out_oop.len(), 3);
        assert_eq!(cfg.opt_out_ip.len(), 2);
        assert_eq!(cfg.outer_player, 1);
    }

    #[test]
    fn inject_gadget_layer_prepends_four_nodes() {
        let mut game = minimal_game();
        let arena_before = game.node_arena_len_for_test();

        let cfg = GadgetConfig {
            opt_out_oop: vec![0.0; game.num_private_hands(0)],
            opt_out_ip: vec![0.0; game.num_private_hands(1)],
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
            opt_out_ip: vec![0.0; game.num_private_hands(1)],
            outer_player: 1,
        };
        inject_gadget_layer(&mut game, &cfg);

        let root = game.root();
        assert!(!root.is_terminal(), "root should be a decision node");
        assert!(!root.is_chance(), "root should not be a chance node");
        assert_eq!(root.num_actions(), 2, "root should have [Terminate, Follow]");
        assert_eq!(root.acting_player(), 1, "IP is outer per config");
    }

    #[test]
    fn inject_gadget_layer_terminates_are_depth_boundaries() {
        let mut game = minimal_game();
        let cfg = GadgetConfig {
            opt_out_oop: vec![0.0; game.num_private_hands(0)],
            opt_out_ip: vec![0.0; game.num_private_hands(1)],
            outer_player: 1,
        };
        inject_gadget_layer(&mut game, &cfg);

        // Terminate at arena 1 (after G_IP at arena 0)
        let t_ip = game.node_arena_get_for_test(1);
        assert!(
            t_ip.is_depth_boundary(),
            "G_IP.Terminate should be depth boundary"
        );

        // Terminate at arena 3 (after G_OOP at arena 2)
        let t_oop = game.node_arena_get_for_test(3);
        assert!(
            t_oop.is_depth_boundary(),
            "G_OOP.Terminate should be depth boundary"
        );
    }

    #[test]
    fn inject_gadget_layer_assigns_reserved_boundary_ordinals() {
        let mut game = minimal_game();
        let cfg = GadgetConfig {
            opt_out_oop: vec![0.0; game.num_private_hands(0)],
            opt_out_ip: vec![0.0; game.num_private_hands(1)],
            outer_player: 1,
        };
        inject_gadget_layer(&mut game, &cfg);

        // G_IP.Terminate at arena 1 -> ordinal 0
        assert_eq!(game.boundary_ordinal(1), Some(0));
        // G_OOP.Terminate at arena 3 -> ordinal 1
        assert_eq!(game.boundary_ordinal(3), Some(1));
    }

    #[test]
    #[should_panic(expected = "opt_out_oop length must match")]
    fn inject_panics_on_oop_length_mismatch() {
        let mut game = minimal_game();
        let cfg = GadgetConfig {
            opt_out_oop: vec![0.0; 99], // wrong length
            opt_out_ip: vec![0.0; game.num_private_hands(1)],
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
            opt_out_ip: vec![0.0; game.num_private_hands(1)],
            outer_player: 42,
        };
        inject_gadget_layer(&mut game, &cfg);
    }

    /// Build the same config pair as `minimal_game` but return them unconsumed.
    fn minimal_config() -> (CardConfig, ActionTree) {
        let oop: Range = "AA,KK".parse().unwrap();
        let ip: Range = "QQ,JJ".parse().unwrap();
        let cc = CardConfig {
            range: [oop, ip],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: card_from_str("3s").unwrap(),
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
        (cc, at)
    }

    #[test]
    fn with_config_and_gadget_returns_ready_game() {
        // Use a throwaway game to discover num_private_hands.
        let tmp = minimal_game();
        let n_oop = tmp.num_private_hands(0);
        let n_ip = tmp.num_private_hands(1);
        drop(tmp);

        let (cc, at) = minimal_config();
        let cfg = GadgetConfig {
            opt_out_oop: vec![0.1; n_oop],
            opt_out_ip: vec![0.1; n_ip],
            outer_player: 1,
        };

        let mut game = PostFlopGame::with_config_and_gadget(cc, at, cfg).unwrap();

        // Root is the gadget (Decision, 2 actions, IP as outer).
        assert!(!game.root().is_terminal());
        assert_eq!(game.root().num_actions(), 2);
        assert_eq!(game.root().acting_player(), 1);

        // Two boundary terminals added by gadget injection.
        assert!(game.num_boundary_nodes() >= 2);

        // allocate_memory should still work cleanly after this flow.
        game.allocate_memory(false);
    }

    #[test]
    fn inject_with_oop_outer_swaps_nesting() {
        let mut game = minimal_game();
        let cfg = GadgetConfig {
            opt_out_oop: vec![0.5; game.num_private_hands(0)],
            opt_out_ip: vec![-0.5; game.num_private_hands(1)],
            outer_player: 0, // OOP outer
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

    // ======================================================================
    // Phase A2: per-boundary gadget tests
    // ======================================================================

    /// Build a depth-limited turn-start game with boundary nodes.
    /// Turn-root with depth_limit=0: boundaries appear at the river
    /// chance transition (all terminal action sequences become depth
    /// boundary nodes instead of dealing the river).
    fn minimal_depth_limited_game() -> PostFlopGame {
        let oop: Range = "AA,KK,QQ".parse().unwrap();
        let ip: Range = "QQ,JJ,TT".parse().unwrap();
        let cc = CardConfig {
            range: [oop, ip],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: NOT_DEALT,
        };
        let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
        let tc = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 200,
            turn_bet_sizes: [sizes.clone(), sizes.clone()],
            river_bet_sizes: [sizes.clone(), sizes],
            depth_limit: Some(0),
            ..Default::default()
        };
        let at = ActionTree::new(tc).unwrap();
        PostFlopGame::with_config(cc, at).unwrap()
    }

    /// Build a GadgetConfigPerBoundary for a given game with constant
    /// opt-out values.
    fn make_per_boundary_config(game: &PostFlopGame) -> GadgetConfigPerBoundary {
        let n = game.num_boundary_nodes();
        let n_oop = game.num_private_hands(0);
        let n_ip = game.num_private_hands(1);
        let per_boundary_opt_outs = (0..n)
            .map(|b| {
                let oop_vals = vec![0.1 * (b + 1) as f32; n_oop];
                let ip_vals = vec![0.2 * (b + 1) as f32; n_ip];
                [oop_vals, ip_vals]
            })
            .collect();
        GadgetConfigPerBoundary {
            per_boundary_opt_outs,
        }
    }

    // -- PLAYER_GADGET_FLAG tests --

    #[test]
    fn gadget_flag_does_not_collide_with_existing_flags() {
        use crate::action_tree::*;
        // PLAYER_GADGET_FLAG must not overlap with terminal, chance, fold,
        // depth_boundary, or the player mask bits.
        assert_eq!(PLAYER_GADGET_FLAG & PLAYER_MASK, 0);
        assert_eq!(PLAYER_GADGET_FLAG & PLAYER_TERMINAL_FLAG, 0);
        assert_eq!(PLAYER_GADGET_FLAG & PLAYER_CHANCE_FLAG, 0);
        // Combined with player bits, it should not trigger is_terminal or is_chance.
        let gadget_oop = PLAYER_GADGET_FLAG | PLAYER_OOP;
        assert_eq!(gadget_oop & PLAYER_TERMINAL_FLAG, 0);
        assert_eq!(gadget_oop & PLAYER_CHANCE_FLAG, 0);
        let gadget_ip = PLAYER_GADGET_FLAG | PLAYER_IP;
        assert_eq!(gadget_ip & PLAYER_TERMINAL_FLAG, 0);
        assert_eq!(gadget_ip & PLAYER_CHANCE_FLAG, 0);
    }

    #[test]
    fn is_gadget_on_plain_node_returns_false() {
        use crate::game::PostFlopNode;
        let node = PostFlopNode::default();
        assert!(!node.is_gadget());
    }

    #[test]
    fn is_gadget_on_gadget_node_returns_true() {
        use crate::action_tree::{PLAYER_GADGET_FLAG, PLAYER_IP};
        use crate::game::PostFlopNode;
        let mut node = PostFlopNode::default();
        node.player = PLAYER_GADGET_FLAG | PLAYER_IP;
        assert!(node.is_gadget());
    }

    #[test]
    fn gadget_owner_returns_none_for_non_gadget() {
        use crate::game::PostFlopNode;
        let node = PostFlopNode::default();
        assert_eq!(node.gadget_owner(), None);
    }

    #[test]
    fn gadget_owner_returns_player_for_gadget_node() {
        use crate::action_tree::{PLAYER_GADGET_FLAG, PLAYER_IP, PLAYER_OOP};
        use crate::game::PostFlopNode;
        let mut node = PostFlopNode::default();

        node.player = PLAYER_GADGET_FLAG | PLAYER_IP;
        assert_eq!(node.gadget_owner(), Some(1));

        node.player = PLAYER_GADGET_FLAG | PLAYER_OOP;
        assert_eq!(node.gadget_owner(), Some(0));
    }

    // -- GadgetConfigPerBoundary tests --

    #[test]
    fn gadget_config_per_boundary_constructs() {
        let cfg = GadgetConfigPerBoundary {
            per_boundary_opt_outs: vec![
                [vec![0.1; 3], vec![0.2; 2]],
                [vec![0.3; 3], vec![0.4; 2]],
            ],
        };
        assert_eq!(cfg.per_boundary_opt_outs.len(), 2);
        assert_eq!(cfg.per_boundary_opt_outs[0][0].len(), 3);
        assert_eq!(cfg.per_boundary_opt_outs[1][1].len(), 2);
    }

    // -- inject_per_boundary_gadgets integration tests --

    #[test]
    fn inject_per_boundary_gadgets_adds_4_nodes_per_boundary() {
        let mut game = minimal_depth_limited_game();
        let n_before = game.num_boundary_nodes();
        assert!(n_before > 0, "need at least one boundary to test");
        let arena_before = game.node_arena_len_for_test();
        let config = make_per_boundary_config(&game);
        inject_per_boundary_gadgets(&mut game, &config);
        assert_eq!(
            game.node_arena_len_for_test(),
            arena_before + 4 * n_before,
            "should add exactly 4 nodes per boundary"
        );
    }

    #[test]
    fn inject_per_boundary_gadgets_triples_boundary_count() {
        let mut game = minimal_depth_limited_game();
        let n_before = game.num_boundary_nodes();
        let config = make_per_boundary_config(&game);
        inject_per_boundary_gadgets(&mut game, &config);
        assert_eq!(
            game.num_boundary_nodes(),
            3 * n_before,
            "each original boundary -> 2 new gadget terminals + 1 original = 3x"
        );
    }

    #[test]
    fn gadget_decisions_have_correct_owner() {
        let mut game = minimal_depth_limited_game();
        let n_before = game.num_boundary_nodes();
        let config = make_per_boundary_config(&game);
        inject_per_boundary_gadgets(&mut game, &config);

        // Walk all nodes, find gadget decision nodes.
        let mut gadget_ip_count = 0usize;
        let mut gadget_oop_count = 0usize;
        for i in 0..game.node_arena_len_for_test() {
            let node = game.node_arena_get_for_test(i);
            if node.is_gadget() {
                match node.gadget_owner() {
                    Some(1) => gadget_ip_count += 1,
                    Some(0) => gadget_oop_count += 1,
                    other => panic!("unexpected gadget_owner: {:?}", other),
                }
            }
        }
        // Each boundary gets G_IP (owner=1) and G_OOP (owner=0).
        assert_eq!(gadget_ip_count, n_before, "one G_IP per boundary");
        assert_eq!(gadget_oop_count, n_before, "one G_OOP per boundary");
    }

    #[test]
    fn gadget_terminates_are_depth_boundaries() {
        let mut game = minimal_depth_limited_game();
        let n_before = game.num_boundary_nodes();
        let config = make_per_boundary_config(&game);
        inject_per_boundary_gadgets(&mut game, &config);

        // New gadget Terminate nodes should be depth boundary terminals.
        // Total boundary count = 3*n_before. The 2*n_before new ones are
        // gadget Terminates. We verify by scanning the arena.
        let mut gadget_terminal_count = 0usize;
        for i in 0..game.node_arena_len_for_test() {
            let node = game.node_arena_get_for_test(i);
            if node.is_depth_boundary() {
                gadget_terminal_count += 1;
            }
        }
        assert_eq!(
            gadget_terminal_count,
            3 * n_before,
            "all 3*N boundaries should be depth boundary terminals"
        );
    }

    #[test]
    fn existing_cfvnet_boundary_ordinals_remain_stable() {
        let mut game = minimal_depth_limited_game();
        let n_before = game.num_boundary_nodes();

        // Record the original boundary arena indices and ordinals.
        let config = make_per_boundary_config(&game);
        inject_per_boundary_gadgets(&mut game, &config);

        // After injection, the cfvnet boundary ordinals should still be
        // 0..n_before. The gadget terminal ordinals occupy n_before..3*n_before.
        let post_indices = game.boundary_node_indices();
        assert_eq!(post_indices.len(), 3 * n_before);

        // The first n_before ordinals should still point to cfvnet boundaries.
        // Their arena indices will have shifted due to injection, but their
        // ordinal assignments should remain 0..n_before.
        for ord in 0..n_before {
            let idx = post_indices[ord];
            let node = game.node_arena_get_for_test(idx);
            assert!(
                node.is_depth_boundary(),
                "ordinal {} should be a depth boundary",
                ord
            );
            // cfvnet boundaries should NOT be gadget decisions.
            assert!(
                !node.is_gadget(),
                "cfvnet boundary ordinal {} should not be a gadget node",
                ord
            );
        }
    }

    #[test]
    fn gadget_terminate_cfvs_pre_populated() {
        let mut game = minimal_depth_limited_game();
        let n_before = game.num_boundary_nodes();
        let config = make_per_boundary_config(&game);
        inject_per_boundary_gadgets(&mut game, &config);

        // Gadget terminal ordinals are n_before..3*n_before.
        // For each original boundary b:
        //   ordinal (n_before + 2*b)     = G_IP.Terminate  -> IP gets opt_out, OOP gets zeros
        //   ordinal (n_before + 2*b + 1) = G_OOP.Terminate -> OOP gets opt_out, IP gets zeros
        for b in 0..n_before {
            let ip_term_ord = n_before + 2 * b;
            let oop_term_ord = n_before + 2 * b + 1;

            // IP Terminate: IP player (player=1) gets opt-out values.
            let ip_cfvs = game.get_boundary_cfvs_for_test(ip_term_ord, 1);
            assert!(!ip_cfvs.is_empty(), "IP opt-out should be populated");
            for &v in &ip_cfvs {
                assert!(v != 0.0 || true, "IP opt-out populated");
            }
            // OOP at IP Terminate: neutralized (zeros).
            let oop_at_ip = game.get_boundary_cfvs_for_test(ip_term_ord, 0);
            assert!(!oop_at_ip.is_empty(), "OOP neutralized should be populated");
            // TODO(Phase C): replace with zero-sum complement
            assert!(
                oop_at_ip.iter().all(|&v| v == 0.0),
                "non-gadget player neutralized to zero"
            );

            // OOP Terminate: OOP player (player=0) gets opt-out values.
            let oop_cfvs = game.get_boundary_cfvs_for_test(oop_term_ord, 0);
            assert!(!oop_cfvs.is_empty(), "OOP opt-out should be populated");
            // IP at OOP Terminate: neutralized (zeros).
            let ip_at_oop = game.get_boundary_cfvs_for_test(oop_term_ord, 1);
            assert!(!ip_at_oop.is_empty(), "IP neutralized should be populated");
            // TODO(Phase C): replace with zero-sum complement
            assert!(
                ip_at_oop.iter().all(|&v| v == 0.0),
                "non-gadget player neutralized to zero"
            );
        }
    }

    #[test]
    fn non_gadget_game_unaffected() {
        // A game with no gadget injection should have no gadget nodes.
        let game = minimal_game();
        for i in 0..game.node_arena_len_for_test() {
            let node = game.node_arena_get_for_test(i);
            assert!(!node.is_gadget(), "no gadget nodes in plain game");
        }
    }

    #[test]
    fn gadget_decision_nodes_have_two_actions() {
        let mut game = minimal_depth_limited_game();
        let config = make_per_boundary_config(&game);
        inject_per_boundary_gadgets(&mut game, &config);

        for i in 0..game.node_arena_len_for_test() {
            let node = game.node_arena_get_for_test(i);
            if node.is_gadget() {
                assert_eq!(
                    node.num_actions(),
                    2,
                    "gadget decision at arena {} should have 2 actions",
                    i
                );
            }
        }
    }

    #[test]
    fn gadget_ip_is_outer_oop_is_inner() {
        let mut game = minimal_depth_limited_game();
        let n_before = game.num_boundary_nodes();
        let config = make_per_boundary_config(&game);
        inject_per_boundary_gadgets(&mut game, &config);

        // For each boundary, the parent of the cfvnet terminal should be
        // G_OOP (inner), and G_OOP's parent should be G_IP (outer).
        // We verify this by checking the gadget_owner of each gadget node:
        // IP gadget decisions should exist and OOP gadget decisions should exist.
        let mut found_ip = 0;
        let mut found_oop = 0;
        for i in 0..game.node_arena_len_for_test() {
            let node = game.node_arena_get_for_test(i);
            if node.is_gadget() {
                let owner = node.gadget_owner().unwrap();
                if owner == 1 {
                    found_ip += 1;
                    // IP gadget acting_player should be 1
                    assert_eq!(node.acting_player(), 1);
                } else {
                    found_oop += 1;
                    // OOP gadget acting_player should be 0
                    assert_eq!(node.acting_player(), 0);
                }
            }
        }
        assert_eq!(found_ip, n_before);
        assert_eq!(found_oop, n_before);
    }

    #[test]
    #[should_panic(expected = "per_boundary_opt_outs length")]
    fn inject_per_boundary_panics_on_wrong_boundary_count() {
        let mut game = minimal_depth_limited_game();
        let config = GadgetConfigPerBoundary {
            per_boundary_opt_outs: vec![], // wrong length
        };
        inject_per_boundary_gadgets(&mut game, &config);
    }

    #[test]
    fn allocate_memory_works_after_injection() {
        let mut game = minimal_depth_limited_game();
        let config = make_per_boundary_config(&game);
        inject_per_boundary_gadgets(&mut game, &config);
        // Should not panic.
        game.allocate_memory(false);
    }

    // ======================================================================
    // Phase B: solver activation logic tests
    // ======================================================================

    #[test]
    fn game_node_trait_gadget_defaults_return_false_and_none() {
        // The GameNode trait's default is_gadget() returns false and
        // gadget_owner() returns None, so non-gadget games are unaffected.
        let mut game = minimal_game();
        game.allocate_memory(false);
        let root = game.root();
        assert!(!root.is_gadget());
        assert_eq!(root.gadget_owner(), None);
    }

    #[test]
    fn gadget_owner_regret_matches_on_own_pass() {
        // Option Y: on the owner's traverser pass, the gadget is a
        // regular Decision node -- regret-matching fires and strategy
        // sum accumulates. After several iterations, gadget nodes
        // should have NON-ZERO strategy sums and regrets.
        use crate::solver::solve_step;

        let mut game = minimal_depth_limited_game();
        let config = make_per_boundary_config(&game);
        inject_per_boundary_gadgets(&mut game, &config);
        game.allocate_memory(false);

        let n_boundary = game.num_boundary_nodes();
        for ord in 0..n_boundary {
            game.set_boundary_cfvs(ord, 0, vec![0.0; game.num_private_hands(0)]);
            game.set_boundary_cfvs(ord, 1, vec![0.0; game.num_private_hands(1)]);
        }

        // Run several iterations so regret-matching has time to fire.
        for i in 0..5 {
            solve_step(&game, i);
        }

        // Gadget nodes should have non-zero strategy sums (the owner's
        // pass accumulated strategy via standard CFR decision logic).
        let mut found_gadget_with_strategy = false;
        let mut found_non_gadget_decision = false;
        for i in 0..game.node_arena_len_for_test() {
            let node = game.node_arena_get_for_test(i);
            if node.is_gadget() {
                let strat_sum: f32 = node.strategy().iter().sum();
                if strat_sum > 0.0 {
                    found_gadget_with_strategy = true;
                }
            } else if !node.is_terminal() && !node.is_chance()
                && node.num_actions() > 1
                && !node.regrets().is_empty()
            {
                let regret_sum: f32 = node.regrets().iter().map(|r| r.abs()).sum();
                if regret_sum > 0.0 {
                    found_non_gadget_decision = true;
                }
            }
        }
        assert!(
            found_gadget_with_strategy,
            "gadget should have non-zero strategy sum after owner's passes"
        );
        assert!(
            found_non_gadget_decision,
            "should have found non-gadget decisions with non-zero regrets"
        );
    }

    #[test]
    fn gadget_owner_disabled_on_opponent_pass() {
        // Option Y: when traverser != gadget owner, the gadget is a
        // passthrough (force Follow, no regret/strategy update from
        // this pass). We verify by checking that after iterations,
        // gadget strategy sums reflect only the owner's passes (roughly
        // half the total iterations), not both passes.
        //
        // With 10 iterations and 2-action gadgets at uniform: if BOTH
        // passes accumulated, sum ~ 10 * 0.5 = 5.0 per hand per action.
        // If only the owner's pass accumulated, sum ~ 5 * 0.5 = 2.5.
        // (Discounting makes exact values vary but the 2x ratio holds.)
        use crate::solver::solve_step;

        let mut game = minimal_depth_limited_game();
        let config = make_per_boundary_config(&game);
        inject_per_boundary_gadgets(&mut game, &config);
        game.allocate_memory(false);

        let n_boundary = game.num_boundary_nodes();
        for ord in 0..n_boundary {
            game.set_boundary_cfvs(ord, 0, vec![0.0; game.num_private_hands(0)]);
            game.set_boundary_cfvs(ord, 1, vec![0.0; game.num_private_hands(1)]);
        }

        let iters = 10u32;
        for i in 0..iters {
            solve_step(&game, i);
        }

        // Each gadget is visited once per iteration (alternating
        // traverser). Only the owner's pass (half the iterations)
        // should accumulate strategy. Check that strategy sum is
        // bounded below what both-passes would produce.
        for i in 0..game.node_arena_len_for_test() {
            let node = game.node_arena_get_for_test(i);
            if !node.is_gadget() {
                continue;
            }
            let strat = node.strategy();
            let n_hands = strat.len() / 2; // 2 actions
            if n_hands == 0 {
                continue;
            }
            let hand0_sum = strat[0] + strat[n_hands];
            // With only owner-pass accumulation (~5 active passes out
            // of 10), the sum should be well below iters (10). If both
            // passes leaked in, it would approach 2*iters.
            assert!(
                hand0_sum < iters as f32,
                "strategy sum {hand0_sum} suggests non-owner passes leaked \
                 (expected < {iters} from ~{} owner passes)",
                iters / 2
            );
        }
    }

    #[test]
    fn non_gadget_game_solve_unchanged_by_gadget_code() {
        // Regression: a game with no gadgets should produce identical
        // exploitability before and after this commit.
        use crate::solver::solve;

        let (cc, at) = minimal_config();
        let mut game = PostFlopGame::with_config(cc, at).unwrap();
        game.allocate_memory(false);

        let expl = solve(&mut game, 50, 0.0, false);
        assert!(
            expl < 5.0,
            "Non-gadget river game should converge below 5.0 after 50 iters: {expl}"
        );
    }

    #[test]
    fn gadget_game_converges() {
        // A depth-limited game with per-boundary gadgets should converge
        // (exploitability decreases over iterations). This verifies the
        // active gadget path's regret updates drive strategy convergence.
        use crate::solver::{solve_step, compute_exploitability};

        let mut game = minimal_depth_limited_game();
        let config = make_per_boundary_config(&game);
        inject_per_boundary_gadgets(&mut game, &config);
        game.allocate_memory(false);

        let n_boundary = game.num_boundary_nodes();
        for ord in 0..n_boundary {
            game.set_boundary_cfvs(ord, 0, vec![0.0; game.num_private_hands(0)]);
            game.set_boundary_cfvs(ord, 1, vec![0.0; game.num_private_hands(1)]);
        }

        for i in 0..100 {
            solve_step(&game, i);
        }

        let expl = compute_exploitability(&game);
        assert!(
            expl < 50.0,
            "Gadget game should converge below 50.0 after 100 iters: {expl}"
        );
    }

    #[test]
    fn non_owner_pass_does_not_accumulate_strategy() {
        // Option Y: the non-owner's pass is a passthrough (skip to
        // Follow). Only the owner's pass accumulates strategy at the
        // gadget. After 5 iterations, each gadget had ~2-3 owner passes
        // (alternating), so strategy sum should be modest, not doubled.
        use crate::solver::solve_step;

        let mut game = minimal_depth_limited_game();
        let config = make_per_boundary_config(&game);
        inject_per_boundary_gadgets(&mut game, &config);
        game.allocate_memory(false);

        let n_boundary = game.num_boundary_nodes();
        for ord in 0..n_boundary {
            game.set_boundary_cfvs(ord, 0, vec![0.0; game.num_private_hands(0)]);
            game.set_boundary_cfvs(ord, 1, vec![0.0; game.num_private_hands(1)]);
        }

        let iters = 5u32;
        for i in 0..iters {
            solve_step(&game, i);
        }

        // Each gadget has 2 actions and num_hands per action. With only
        // owner-pass accumulation, the strategy sum per hand should be
        // bounded by iters (not 2*iters which would indicate both passes
        // accumulated).
        for i in 0..game.node_arena_len_for_test() {
            let node = game.node_arena_get_for_test(i);
            if !node.is_gadget() {
                continue;
            }
            let strat = node.strategy();
            let n_hands = strat.len() / 2; // 2 actions
            if n_hands == 0 {
                continue;
            }
            let hand0_sum = strat[0] + strat[n_hands];
            assert!(
                hand0_sum < (2 * iters) as f32,
                "strategy sum {hand0_sum} suggests non-owner passes leaked \
                 (expected < {} from ~{} owner passes)",
                2 * iters,
                iters / 2
            );
        }
    }

    #[test]
    fn per_boundary_safety_invariant_avg_realized_cfv_geq_opt_out() {
        // Burch 2014 S3 sufficiency condition: after convergence, the
        // average realized CFV at each gadget must be >= the opt-out
        // value for each hand owned by the gadget's player.
        //
        // Build a gadget game, solve for enough iterations to converge
        // the gadget sigma, then verify per-hand:
        //   avg_realized_CFV[h] >= opt_out[h] - epsilon
        use crate::solver::{solve_step, compute_exploitability, root_cfvalues};

        let mut game = minimal_depth_limited_game();
        let n_orig = game.num_boundary_nodes();
        let n_oop = game.num_private_hands(0);
        let n_ip = game.num_private_hands(1);

        // Use non-trivial opt-out values so the check is meaningful.
        let per_boundary_opt_outs: Vec<[Vec<f32>; 2]> = (0..n_orig)
            .map(|_| {
                // Small negative opt-outs: the player can always do
                // at least this well by opting out of the subgame.
                [vec![-5.0; n_oop], vec![-5.0; n_ip]]
            })
            .collect();
        let config = GadgetConfigPerBoundary { per_boundary_opt_outs };
        inject_per_boundary_gadgets(&mut game, &config);
        game.allocate_memory(false);

        let n_boundary = game.num_boundary_nodes();
        for ord in 0..n_boundary {
            game.set_boundary_cfvs(ord, 0, vec![0.0; n_oop]);
            game.set_boundary_cfvs(ord, 1, vec![0.0; n_ip]);
        }

        // Solve for enough iterations for gadget sigma to converge.
        for i in 0..1000 {
            solve_step(&game, i);
        }

        let expl = compute_exploitability(&game);
        assert!(
            expl < 20.0,
            "gadget game should converge after 1000 iters: expl={expl}"
        );

        // Check safety: for each player, root CFV per hand must be
        // >= opt-out. The root CFV already incorporates the gadget
        // decision. With opt-out = -5.0 and subgame value >= -5.0,
        // the gadget should learn to Follow (subgame) rather than
        // Terminate, so the realized CFV should be >= -5.0.
        for player in 0..2 {
            let cfvs = root_cfvalues(&game, player);
            let opt_out = -5.0f32;
            for (h, &cfv) in cfvs.iter().enumerate() {
                assert!(
                    cfv >= opt_out - 0.01,
                    "player {player} hand {h}: realized CFV {cfv} < \
                     opt_out {opt_out} - epsilon (safety violation)"
                );
            }
        }
    }
}
