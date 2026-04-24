//! Structural CFR-D gadget: two nested Decision(player, [Terminate, Follow])
//! nodes prepended at the PostFlopGame::node_arena root, using existing
//! depth-boundary terminals for per-hand opt-out CFVs.
//!
//! See docs/plans/2026-04-24-option-a-deepstack-gadget-design.md and
//! docs/plans/2026-04-24-option-a-deepstack-gadget-plan.md.

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
    use crate::card::{card_from_str, flop_from_str};
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
}
