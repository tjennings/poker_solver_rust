//! Integration tests for gadget-enabled PostFlopGame construction and solve.

use std::sync::Arc;

use poker_solver_tauri::gadget::make_gadget_game;
use range_solver::action_tree::{ActionTree, TreeConfig};
use range_solver::bet_size::BetSizeOptions;
use range_solver::card::{card_from_str, flop_from_str, CardConfig};
use range_solver::game::BoundaryEvaluator;
use range_solver::interface::{Game, GameNode};
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

/// With opt-out values so negative they are always dominated, the T-branch
/// (Terminate) never accumulates positive regret, so regret matching always
/// picks F (Follow). The gadget reduces to the identity: strategies at all
/// non-gadget decision nodes must match a no-gadget solve on the same spot.
///
/// We use `-1e9` rather than `f32::NEG_INFINITY` because the solver's
/// reach-scaling arithmetic produces NaN when infinity propagates through
/// subtraction/multiplication. A finite-but-dominated value achieves the
/// same behavioral result without corrupting float arithmetic.
#[test]
fn gadget_with_neg_inf_opt_out_matches_no_gadget() {
    // A value so negative that T-branch regret can never go positive,
    // making Follow the only rational choice at every gadget decision.
    const DOMINATED_OPT_OUT: f32 = -1e9;

    // -- Shared parameters --
    let oop: Range = "AA,KK,QQ".parse().unwrap();
    let ip: Range = "TT,99,88".parse().unwrap();
    let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
    let mk_config = |oop_r: Range, ip_r: Range| {
        let cc = CardConfig {
            range: [oop_r, ip_r],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: card_from_str("3s").unwrap(),
        };
        let tc = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 200,
            river_bet_sizes: [sizes.clone(), sizes.clone()],
            ..Default::default()
        };
        (cc, tc)
    };

    // Enough iterations for DCFR discounting to wash out the gadget's
    // warmup period (first few iterations where T-branch has ~50% weight
    // before regret drives it to zero).
    let iters: u32 = 1000;
    let target_expl: f32 = 1e-4;

    // -- No-gadget game --
    let (cc_nog, tc_nog) = mk_config(oop.clone(), ip.clone());
    let at_nog = ActionTree::new(tc_nog).unwrap();
    let mut g_nog = range_solver::PostFlopGame::with_config(cc_nog, at_nog).unwrap();
    g_nog.allocate_memory(false);
    range_solver::solve(&mut g_nog, iters, target_expl, false);

    // -- Gadget game with dominated opt-out --
    let (cc_gad, tc_gad) = mk_config(oop, ip);
    let at_gad = ActionTree::new(tc_gad).unwrap();

    let n_oop = g_nog.num_private_hands(0);
    let n_ip = g_nog.num_private_hands(1);

    let cfg = range_solver::game::gadget::GadgetConfig {
        opt_out_oop: vec![DOMINATED_OPT_OUT; n_oop],
        opt_out_ip: vec![DOMINATED_OPT_OUT; n_ip],
        outer_player: 1,
    };
    let inner: Arc<dyn BoundaryEvaluator> = Arc::new(PanicEvaluator);
    let mut g_gad = make_gadget_game(cc_gad, at_gad, cfg, inner).unwrap();
    range_solver::solve(&mut g_gad, iters, target_expl, false);

    // -- Compare normalized strategies at all shared decision nodes --
    // The gadget prepends 4 arena nodes (2 gadget-decision + 2 gadget-terminal),
    // so no-gadget arena index N maps to gadget arena index N + 4.
    // Use strategy_at_index() which returns the normalized average strategy,
    // not the raw cumulative sums from GameNode::strategy().
    let n_nog = g_nog.num_nodes();
    let tol: f32 = 1e-3;
    let mut compared = 0usize;
    let mut drift_count = 0usize;

    for nog_idx in 0..n_nog {
        let nog_node = g_nog.node_at(nog_idx);
        if nog_node.is_terminal() || nog_node.is_chance() {
            continue;
        }
        drop(nog_node);

        let nog_strat = g_nog.strategy_at_index(nog_idx);
        let gad_idx = nog_idx + 4;
        let gad_strat = g_gad.strategy_at_index(gad_idx);

        assert_eq!(
            nog_strat.len(),
            gad_strat.len(),
            "strategy length mismatch at nog arena idx {nog_idx}",
        );

        for (i, (a, b)) in nog_strat.iter().zip(gad_strat.iter()).enumerate() {
            let d = (a - b).abs();
            if d > tol {
                drift_count += 1;
                if drift_count <= 10 {
                    eprintln!(
                        "[parity drift] nog_idx={nog_idx} slot={i}: nog={a:.6} gad={b:.6} d={d:.6}"
                    );
                }
            }
        }
        compared += 1;
    }

    assert!(compared > 0, "should have compared at least one decision node");
    assert_eq!(
        drift_count, 0,
        "strategies drifted at {drift_count} slots (tolerance {tol}); see stderr for details"
    );
}
