//! Integration tests for gadget-enabled PostFlopGame construction and solve.

use std::sync::Arc;

use poker_solver_tauri::gadget::make_gadget_game;
use range_solver::action_tree::{ActionTree, TreeConfig};
use range_solver::bet_size::BetSizeOptions;
use range_solver::card::{card_from_str, flop_from_str, CardConfig};
use range_solver::game::BoundaryEvaluator;
use range_solver::interface::Game;
use range_solver::range::Range;
use range_solver::BoardState;

/// Build a river-only card config + tree config pair for gadget tests.
/// Returns `(CardConfig, TreeConfig)` -- callers build ActionTree from TreeConfig.
fn river_test_configs() -> (CardConfig, TreeConfig) {
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
    (cc, tc)
}

/// Inner evaluator that panics if called -- proves gadget ordinals are
/// served by the StaticGadgetEvaluator, not this fallback.
struct PanicEvaluator;
impl BoundaryEvaluator for PanicEvaluator {
    fn compute_cfvs(
        &self,
        _player: usize,
        _pot: i32,
        _remaining_stack: f64,
        _opponent_reach: &[f32],
        _num_hands: usize,
        _continuation_index: usize,
    ) -> Vec<f32> {
        panic!("inner evaluator should not be called for gadget ordinals")
    }
}

#[test]
fn make_gadget_game_returns_ok() {
    let (cc, tc) = river_test_configs();
    let at = ActionTree::new(tc.clone()).unwrap();

    // Discover hand counts from a throwaway game.
    let tmp = range_solver::PostFlopGame::with_config(cc.clone(), ActionTree::new(tc).unwrap()).unwrap();
    let n_oop = tmp.num_private_hands(0);
    let n_ip = tmp.num_private_hands(1);
    drop(tmp);

    let cfg = range_solver::game::gadget::GadgetConfig {
        opt_out_oop: vec![0.5; n_oop],
        opt_out_ip: vec![-0.3; n_ip],
        outer_player: 1,
    };
    let inner: Arc<dyn BoundaryEvaluator> = Arc::new(PanicEvaluator);

    let result = make_gadget_game(cc, at, cfg, inner);
    assert!(result.is_ok(), "make_gadget_game should return Ok");
}

#[test]
fn make_gadget_game_has_exactly_two_gadget_boundaries() {
    let (cc, tc) = river_test_configs();
    let at = ActionTree::new(tc.clone()).unwrap();

    let tmp = range_solver::PostFlopGame::with_config(cc.clone(), ActionTree::new(tc).unwrap()).unwrap();
    let n_oop = tmp.num_private_hands(0);
    let n_ip = tmp.num_private_hands(1);
    drop(tmp);

    let cfg = range_solver::game::gadget::GadgetConfig {
        opt_out_oop: vec![0.0; n_oop],
        opt_out_ip: vec![0.0; n_ip],
        outer_player: 1,
    };
    let inner: Arc<dyn BoundaryEvaluator> = Arc::new(PanicEvaluator);

    let game = make_gadget_game(cc, at, cfg, inner).unwrap();

    // River game with no depth_limit: only the 2 gadget terminals are boundaries.
    assert_eq!(
        game.num_boundary_nodes(),
        2,
        "river game should have exactly 2 gadget boundary terminals"
    );
}

#[test]
fn make_gadget_game_populates_per_boundary_evaluators() {
    let (cc, tc) = river_test_configs();
    let at = ActionTree::new(tc.clone()).unwrap();

    let tmp = range_solver::PostFlopGame::with_config(cc.clone(), ActionTree::new(tc).unwrap()).unwrap();
    let n_oop = tmp.num_private_hands(0);
    let n_ip = tmp.num_private_hands(1);
    drop(tmp);

    let cfg = range_solver::game::gadget::GadgetConfig {
        opt_out_oop: vec![0.7; n_oop],
        opt_out_ip: vec![-0.3; n_ip],
        outer_player: 1,
    };
    let inner: Arc<dyn BoundaryEvaluator> = Arc::new(PanicEvaluator);

    let game = make_gadget_game(cc, at, cfg, inner).unwrap();

    assert_eq!(
        game.per_boundary_evaluators.len(),
        2,
        "per_boundary_evaluators should have exactly 2 entries"
    );
    assert!(
        game.boundary_evaluator.is_some(),
        "boundary_evaluator (inner) should be set"
    );
}

#[test]
fn gadget_solve_step_does_not_panic() {
    let (cc, tc) = river_test_configs();
    let at = ActionTree::new(tc.clone()).unwrap();

    let tmp = range_solver::PostFlopGame::with_config(cc.clone(), ActionTree::new(tc).unwrap()).unwrap();
    let n_oop = tmp.num_private_hands(0);
    let n_ip = tmp.num_private_hands(1);
    drop(tmp);

    let opt_out_oop = vec![0.7f32; n_oop];
    let opt_out_ip = vec![-0.3f32; n_ip];

    let cfg = range_solver::game::gadget::GadgetConfig {
        opt_out_oop: opt_out_oop.clone(),
        opt_out_ip: opt_out_ip.clone(),
        outer_player: 1,
    };
    let inner: Arc<dyn BoundaryEvaluator> = Arc::new(PanicEvaluator);

    let game = make_gadget_game(cc, at, cfg, inner).unwrap();

    // solve_step should not panic: gadget ordinals 0/1 are served by
    // StaticGadgetEvaluator, and there are no real boundaries (ordinals 2+)
    // in a river-only game.
    range_solver::solve_step(&game, 0);
}

#[test]
fn gadget_static_cfvs_correct_ip_outer() {
    // With outer_player=1 (IP outer):
    //   ordinal 0 = G_IP.Terminate  -> IP gets opt_out_ip, OOP gets zero
    //   ordinal 1 = G_OOP.Terminate -> OOP gets opt_out_oop, IP gets zero
    let (cc, tc) = river_test_configs();
    let at = ActionTree::new(tc.clone()).unwrap();

    let tmp = range_solver::PostFlopGame::with_config(cc.clone(), ActionTree::new(tc).unwrap()).unwrap();
    let n_oop = tmp.num_private_hands(0);
    let n_ip = tmp.num_private_hands(1);
    drop(tmp);

    let opt_out_oop = vec![0.7f32; n_oop];
    let opt_out_ip = vec![-0.3f32; n_ip];

    let cfg = range_solver::game::gadget::GadgetConfig {
        opt_out_oop: opt_out_oop.clone(),
        opt_out_ip: opt_out_ip.clone(),
        outer_player: 1,
    };
    let inner: Arc<dyn BoundaryEvaluator> = Arc::new(PanicEvaluator);

    let game = make_gadget_game(cc, at, cfg, inner).unwrap();

    // Verify per_boundary_evaluators[0] = G_IP.Terminate
    let eval_0 = &game.per_boundary_evaluators[0];
    let (oop_at_0, ip_at_0) = eval_0.compute_cfvs_both(100, 50.0, &[], &[], n_oop, n_ip, 0);
    assert_eq!(oop_at_0, vec![0.0f32; n_oop], "OOP at G_IP.Terminate should be zero");
    assert_eq!(ip_at_0, opt_out_ip, "IP at G_IP.Terminate should be opt_out_ip");

    // Verify per_boundary_evaluators[1] = G_OOP.Terminate
    let eval_1 = &game.per_boundary_evaluators[1];
    let (oop_at_1, ip_at_1) = eval_1.compute_cfvs_both(100, 50.0, &[], &[], n_oop, n_ip, 0);
    assert_eq!(oop_at_1, opt_out_oop, "OOP at G_OOP.Terminate should be opt_out_oop");
    assert_eq!(ip_at_1, vec![0.0f32; n_ip], "IP at G_OOP.Terminate should be zero");
}

#[test]
fn gadget_static_cfvs_correct_oop_outer() {
    // With outer_player=0 (OOP outer), ordinal mapping inverts:
    //   ordinal 0 = G_OOP.Terminate -> OOP gets opt_out_oop, IP gets zero
    //   ordinal 1 = G_IP.Terminate  -> IP gets opt_out_ip, OOP gets zero
    let (cc, tc) = river_test_configs();
    let at = ActionTree::new(tc.clone()).unwrap();

    let tmp = range_solver::PostFlopGame::with_config(cc.clone(), ActionTree::new(tc).unwrap()).unwrap();
    let n_oop = tmp.num_private_hands(0);
    let n_ip = tmp.num_private_hands(1);
    drop(tmp);

    let opt_out_oop = vec![0.7f32; n_oop];
    let opt_out_ip = vec![-0.3f32; n_ip];

    let cfg = range_solver::game::gadget::GadgetConfig {
        opt_out_oop: opt_out_oop.clone(),
        opt_out_ip: opt_out_ip.clone(),
        outer_player: 0,
    };
    let inner: Arc<dyn BoundaryEvaluator> = Arc::new(PanicEvaluator);

    let game = make_gadget_game(cc, at, cfg, inner).unwrap();

    // Verify per_boundary_evaluators[0] = G_OOP.Terminate (inverted)
    let eval_0 = &game.per_boundary_evaluators[0];
    let (oop_at_0, ip_at_0) = eval_0.compute_cfvs_both(100, 50.0, &[], &[], n_oop, n_ip, 0);
    assert_eq!(oop_at_0, opt_out_oop, "OOP at G_OOP.Terminate should be opt_out_oop");
    assert_eq!(ip_at_0, vec![0.0f32; n_ip], "IP at G_OOP.Terminate should be zero");

    // Verify per_boundary_evaluators[1] = G_IP.Terminate (inverted)
    let eval_1 = &game.per_boundary_evaluators[1];
    let (oop_at_1, ip_at_1) = eval_1.compute_cfvs_both(100, 50.0, &[], &[], n_oop, n_ip, 0);
    assert_eq!(oop_at_1, vec![0.0f32; n_oop], "OOP at G_IP.Terminate should be zero");
    assert_eq!(ip_at_1, opt_out_ip, "IP at G_IP.Terminate should be opt_out_ip");
}
