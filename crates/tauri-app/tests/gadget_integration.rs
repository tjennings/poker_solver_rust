//! Integration tests for per-boundary gadget (Option A2) game construction and solve.

use range_solver::action_tree::{ActionTree, TreeConfig};
use range_solver::bet_size::BetSizeOptions;
use range_solver::card::{card_from_str, flop_from_str, CardConfig, NOT_DEALT};
use range_solver::interface::{Game, GameNode};
use range_solver::range::Range;
use range_solver::BoardState;

/// Build a depth-limited turn-start game with boundary nodes for A2 tests.
fn depth_limited_turn_configs() -> (CardConfig, TreeConfig) {
    let oop: Range = "AA,KK,QQ".parse().unwrap();
    let ip: Range = "TT,99,88".parse().unwrap();
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
    (cc, tc)
}

#[test]
fn per_boundary_gadget_game_root_is_real_subgame() {
    let (cc, tc) = depth_limited_turn_configs();
    let at = ActionTree::new(tc).unwrap();
    let mut game = range_solver::PostFlopGame::with_config(cc, at).unwrap();

    let n_boundaries = game.num_boundary_nodes();
    assert!(n_boundaries > 0, "need boundaries to test A2");
    let n_oop = game.num_private_hands(0);
    let n_ip = game.num_private_hands(1);

    let config = range_solver::game::gadget::GadgetConfigPerBoundary {
        per_boundary_opt_outs: (0..n_boundaries)
            .map(|_| [vec![0.0; n_oop], vec![0.0; n_ip]])
            .collect(),
    };
    range_solver::game::gadget::inject_per_boundary_gadgets(&mut game, &config);
    game.allocate_memory(false);

    // Root is the REAL subgame root (OOP acts first in turn), not a gadget.
    let root = game.root();
    assert!(!root.is_gadget(), "root should NOT be a gadget node under A2");
    assert_eq!(root.acting_player(), 0, "OOP acts first at turn root");
    assert!(
        root.num_actions() > 2,
        "real subgame root has >2 actions (bet sizes), got {}",
        root.num_actions()
    );
}

#[test]
fn per_boundary_gadget_triples_boundary_count() {
    let (cc, tc) = depth_limited_turn_configs();
    let at = ActionTree::new(tc).unwrap();
    let mut game = range_solver::PostFlopGame::with_config(cc, at).unwrap();

    let n_before = game.num_boundary_nodes();
    let n_oop = game.num_private_hands(0);
    let n_ip = game.num_private_hands(1);

    let config = range_solver::game::gadget::GadgetConfigPerBoundary {
        per_boundary_opt_outs: (0..n_before)
            .map(|_| [vec![0.0; n_oop], vec![0.0; n_ip]])
            .collect(),
    };
    range_solver::game::gadget::inject_per_boundary_gadgets(&mut game, &config);

    assert_eq!(
        game.num_boundary_nodes(),
        3 * n_before,
        "A2: each original boundary -> original + 2 gadget terminals = 3x"
    );
}

#[test]
fn per_boundary_gadget_solve_step_does_not_panic() {
    let (cc, tc) = depth_limited_turn_configs();
    let at = ActionTree::new(tc).unwrap();
    let mut game = range_solver::PostFlopGame::with_config(cc, at).unwrap();

    let n_boundaries = game.num_boundary_nodes();
    let n_oop = game.num_private_hands(0);
    let n_ip = game.num_private_hands(1);

    let config = range_solver::game::gadget::GadgetConfigPerBoundary {
        per_boundary_opt_outs: (0..n_boundaries)
            .map(|_| [vec![0.0; n_oop], vec![0.0; n_ip]])
            .collect(),
    };
    range_solver::game::gadget::inject_per_boundary_gadgets(&mut game, &config);
    game.allocate_memory(false);

    // Set boundary CFVs to zero for all boundaries.
    let n_total = game.num_boundary_nodes();
    for ord in 0..n_total {
        game.set_boundary_cfvs(ord, 0, vec![0.0; n_oop]);
        game.set_boundary_cfvs(ord, 1, vec![0.0; n_ip]);
    }

    // Should not panic.
    range_solver::solve_step(&game, 0);
    range_solver::solve_step(&game, 1);
}

#[test]
fn per_boundary_gadget_converges() {
    let (cc, tc) = depth_limited_turn_configs();
    let at = ActionTree::new(tc).unwrap();
    let mut game = range_solver::PostFlopGame::with_config(cc, at).unwrap();

    let n_boundaries = game.num_boundary_nodes();
    let n_oop = game.num_private_hands(0);
    let n_ip = game.num_private_hands(1);

    let config = range_solver::game::gadget::GadgetConfigPerBoundary {
        per_boundary_opt_outs: (0..n_boundaries)
            .map(|_| [vec![-5.0; n_oop], vec![-5.0; n_ip]])
            .collect(),
    };
    range_solver::game::gadget::inject_per_boundary_gadgets(&mut game, &config);
    game.allocate_memory(false);

    let n_total = game.num_boundary_nodes();
    for ord in 0..n_total {
        game.set_boundary_cfvs(ord, 0, vec![0.0; n_oop]);
        game.set_boundary_cfvs(ord, 1, vec![0.0; n_ip]);
    }

    for i in 0..200 {
        range_solver::solve_step(&game, i);
    }

    let expl = range_solver::compute_exploitability(&game);
    assert!(
        expl < 50.0,
        "A2 gadget game should converge after 200 iters: expl={expl}"
    );
}
