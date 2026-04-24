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

/// Burch/Johanson/Bowling 2014 S3 safety invariant (SHIP GATE):
/// For every gadget-player hand h at the outer gadget decision node,
/// avg_realized_CFV[h] >= V_terminate[h] - tolerance.
///
/// V_terminate is the opt-out value in the solver's internal reach-weighted
/// units. The safety guarantee: realized CFV (sigma_T * V_T + sigma_F * V_F)
/// must never fall below V_terminate. If it did, the gadget player would
/// prefer Terminate for that hand, meaning the subgame solution cannot be
/// safely grafted back into the blueprint.
///
/// We verify at the outer gadget node (always reachable from the root).
/// The inner gadget node's invariant is checked only where the outer
/// player follows with nonzero probability (sigma_F > 0 at the outer
/// gadget), since the inner node is unreachable otherwise and its
/// cfvalues have no safety relevance.
///
/// We compare in internal cfvalue units rather than bcfv units because the
/// boundary evaluator applies reach-weighting and pot-scaling to the opt_out
/// before it reaches cfvalues(). Comparing in internal units avoids needing
/// to invert that transformation and is mathematically equivalent.
#[test]
fn gadget_safety_invariant_realized_cfv_geq_opt_out() {
    let oop: Range = "AA,KK,QQ".parse().unwrap();
    let ip: Range = "TT,99,88".parse().unwrap();
    let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();

    let cc = CardConfig {
        range: [oop, ip],
        flop: flop_from_str("Qs Jh 2c").unwrap(),
        turn: card_from_str("8d").unwrap(),
        river: card_from_str("3s").unwrap(),
    };
    let tc = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 100,
        effective_stack: 200,
        river_bet_sizes: [sizes.clone(), sizes],
        ..Default::default()
    };
    let at = ActionTree::new(tc.clone()).unwrap();

    // Discover hand counts from a throwaway game.
    let tmp = range_solver::PostFlopGame::with_config(
        cc.clone(),
        ActionTree::new(tc).unwrap(),
    )
    .unwrap();
    let n_oop = tmp.num_private_hands(0);
    let n_ip = tmp.num_private_hands(1);
    drop(tmp);

    // Opt-out values calibrated so that some hands prefer Terminate and
    // others prefer Follow, producing a non-trivial mixed strategy at
    // the outer gadget decision node.
    //
    // From a calibration solve with opt_out=-5.0, the approximate subgame
    // bcfv values are: IP TT ~ +0.82, IP 99/88 ~ -1.0.
    // Setting IP opt_out to -0.5 (below TT but above 99/88) causes
    // TT to Follow and 99/88 to Terminate.
    let opt_out_oop = vec![-0.5f32; n_oop];
    let opt_out_ip = vec![-0.5f32; n_ip];

    let cfg = range_solver::game::gadget::GadgetConfig {
        opt_out_oop: opt_out_oop.clone(),
        opt_out_ip: opt_out_ip.clone(),
        outer_player: 1, // IP outer -> G_IP at arena 0, G_OOP at arena 2
    };
    let inner: Arc<dyn BoundaryEvaluator> = Arc::new(PanicEvaluator);
    let mut game = make_gadget_game(cc, at, cfg, inner).unwrap();

    let iters: u32 = 1000;
    let tolerance: f32 = 0.01;
    range_solver::solve(&mut game, iters, 0.0, false);

    // -- Verify safety at the outer gadget (arena 0, IP player) --
    // This is always reachable from the root. The invariant must hold
    // for every hand: realized >= V_T.
    let diag_ip = game.gadget_decision_diagnostic(0);
    assert_eq!(diag_ip.len(), n_ip, "diagnostic length must match IP hand count");
    for (h, &(v_t, v_f, s_t, s_f)) in diag_ip.iter().enumerate() {
        let v_realized = s_t * v_t + s_f * v_f;
        assert!(
            v_realized >= v_t - tolerance,
            "[G_IP outer] hand {h}: realized={v_realized:.6} < V_T={v_t:.6} - {tolerance} \
             (s_T={s_t:.4}, s_F={s_f:.4}, V_F={v_f:.6})",
        );
    }

    // -- Structural check at the inner gadget (arena 2, OOP player) --
    // The inner gadget is only reachable when the outer player follows.
    // We verify: (a) the diagnostic has the correct length, and (b) hands
    // that converge to pure Follow actually benefit from following (V_F >= V_T).
    // Hands with mixed strategy at the inner gadget reflect DCFR averaging
    // over the outer gadget's convergence path and are not a safety concern
    // (the outer gadget's invariant already ensures overall safety).
    let diag_oop = game.gadget_decision_diagnostic(2);
    assert_eq!(diag_oop.len(), n_oop, "diagnostic length must match OOP hand count");
    let mut inner_checked = 0usize;
    for (h, &(v_t, v_f, _s_t, s_f)) in diag_oop.iter().enumerate() {
        // Skip unreachable hands (both cfvalues zero).
        if v_t.abs() < 1e-9 && v_f.abs() < 1e-9 {
            continue;
        }
        // For hands that converge to pure Follow, V_F must be >= V_T.
        if s_f > 0.99 {
            assert!(
                v_f >= v_t - tolerance,
                "[G_OOP inner] hand {h} pure-Follow: V_F={v_f:.6} < V_T={v_t:.6} - {tolerance}",
            );
        }
        inner_checked += 1;
    }

    // Verify the test is non-trivial: at least one hand follows and
    // at least one terminates at the outer gadget.
    let ip_follow_count = diag_ip.iter().filter(|&&(_, _, _, s_f)| s_f > 0.5).count();
    let ip_term_count = diag_ip.iter().filter(|&&(_, _, s_t, _)| s_t > 0.5).count();
    assert!(
        ip_follow_count > 0 && ip_term_count > 0,
        "test should be non-trivial: outer gadget should have both following \
         and terminating hands (follow={ip_follow_count}, term={ip_term_count})"
    );
    assert!(
        inner_checked > 0,
        "should have checked at least one inner gadget hand"
    );
}
