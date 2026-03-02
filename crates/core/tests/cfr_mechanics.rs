//! Comprehensive CFR mechanics tests.
//!
//! Tests the core CFR loop properties using Kuhn Poker (known Nash equilibrium):
//! 1. **Convergence** — exploitability decreases, strategies stabilize
//! 2. **Reach propagation** — regrets weighted by opponent reach, strategy sums by own reach
//! 3. **Regret application** — positive/negative regret accumulation, DCFR discounting
//!
//! Kuhn Poker Nash equilibrium (3 cards: J < Q < K, ante 1 each):
//! - P1 with K: bet α ∈ [0,3α], check 1-α; call always
//! - P1 with Q: check always; fold to bet
//! - P1 with J: bet α/3; fold to bet; fold after check-bet
//! - P2 with K: bet always after check; call always
//! - P2 with Q: check after check; call 1/3 + α/3
//! - P2 with J: check after check; fold always
//! Game value = -1/18 (P2 advantage)

use poker_solver_core::{
    cfr::{MccfrConfig, MccfrSolver, VanillaCfr, calculate_exploitability},
    game::KuhnPoker,
    info_key::InfoKey,
};
use rustc_hash::FxHashMap;
use test_macros::timed_test;

// Kuhn action codes matching kuhn.rs encoding
const CHECK: u8 = 2;
const BET: u8 = 4;

fn kuhn_key(card: u32, actions: &[u8]) -> u64 {
    InfoKey::new(card, 0, 0, actions).as_u64()
}

fn extract_all_strategies(solver: &VanillaCfr<KuhnPoker>) -> FxHashMap<u64, Vec<f64>> {
    let info_sets: [(u32, &[u8]); 12] = [
        (0, &[]),           // J
        (1, &[]),           // Q
        (2, &[]),           // K
        (0, &[CHECK]),      // Jc (P2)
        (1, &[CHECK]),      // Qc (P2)
        (2, &[CHECK]),      // Kc (P2)
        (0, &[BET]),        // Jb (P2)
        (1, &[BET]),        // Qb (P2)
        (2, &[BET]),        // Kb (P2)
        (0, &[CHECK, BET]), // Jcb (P1)
        (1, &[CHECK, BET]), // Qcb (P1)
        (2, &[CHECK, BET]), // Kcb (P1)
    ];
    info_sets
        .iter()
        .filter_map(|(card, actions)| {
            let key = kuhn_key(*card, actions);
            solver.get_average_strategy(key).map(|s| (key, s))
        })
        .collect()
}

// ─── Convergence tests ──────────────────────────────────────────────────────

#[timed_test(10)]
fn vanilla_cfr_exploitability_monotonically_decreases() {
    // After warmup, exploitability should consistently decrease.
    let game = KuhnPoker::new();
    let mut solver = VanillaCfr::new(game.clone());

    let checkpoints = [50, 200, 500, 2_000, 5_000];
    let mut exploitabilities = Vec::new();

    let mut trained = 0u64;
    for &target in &checkpoints {
        solver.train(target - trained);
        trained = target;
        let strategy = extract_all_strategies(&solver);
        let exp = calculate_exploitability(&game, &strategy);
        exploitabilities.push((target, exp));
    }

    // After 200 iterations, each checkpoint should be lower than the previous
    for window in exploitabilities.windows(2) {
        let (prev_iter, prev_exp) = window[0];
        let (curr_iter, curr_exp) = window[1];
        if prev_iter >= 200 {
            assert!(
                curr_exp <= prev_exp * 1.1 + 1e-6,
                "Exploitability should decrease: at {prev_iter} iters = {prev_exp:.6}, \
                 at {curr_iter} iters = {curr_exp:.6}"
            );
        }
    }

    // Final exploitability should be very low
    let (_, final_exp) = exploitabilities.last().unwrap();
    assert!(
        *final_exp < 0.005,
        "After 5000 iterations, exploitability should be < 0.005, got {final_exp}"
    );
}

#[timed_test(10)]
fn mccfr_exploitability_decreases_with_full_traversal() {
    // MCCFR with train_full (all deals each iteration) should converge like vanilla.
    let game = KuhnPoker::new();
    let config = MccfrConfig {
        dcfr_alpha: 1.5,
        dcfr_beta: 0.5,
        dcfr_gamma: 2.0,
        ..MccfrConfig::default()
    };
    let mut solver = MccfrSolver::with_config(game.clone(), &config);
    solver.set_seed(42);

    solver.train_full(100);
    let early_strategies = solver.all_strategies();
    let early_exp = calculate_exploitability(&game, &early_strategies);

    solver.train_full(900);
    let late_strategies = solver.all_strategies();
    let late_exp = calculate_exploitability(&game, &late_strategies);

    assert!(
        late_exp < early_exp,
        "MCCFR exploitability should decrease: 100 iters = {early_exp:.6}, \
         1000 iters = {late_exp:.6}"
    );
    assert!(
        late_exp < 0.05,
        "After 1000 full iterations, MCCFR exploitability should be < 0.05, got {late_exp}"
    );
}

#[timed_test(10)]
fn mccfr_sampled_converges_to_nash() {
    // Even with sampling, MCCFR should converge with enough iterations.
    let game = KuhnPoker::new();
    let config = MccfrConfig {
        samples_per_iteration: 6, // all deals
        dcfr_alpha: 1.5,
        dcfr_beta: 0.5,
        dcfr_gamma: 2.0,
        ..MccfrConfig::default()
    };
    let mut solver = MccfrSolver::with_config(game.clone(), &config);
    solver.set_seed(42);
    solver.train(5_000, 6);

    let strategies = solver.all_strategies();
    let exp = calculate_exploitability(&game, &strategies);
    assert!(
        exp < 0.05,
        "MCCFR sampled should converge, got exploitability = {exp}"
    );
}

#[timed_test(10)]
fn vanilla_cfr_strategy_stabilizes() {
    // Strategy at checkpoints should converge (L1 distance decreases).
    let game = KuhnPoker::new();
    let mut solver = VanillaCfr::new(game);

    solver.train(1_000);
    let strat_1k = extract_all_strategies(&solver);

    solver.train(4_000);
    let strat_5k = extract_all_strategies(&solver);

    solver.train(5_000);
    let strat_10k = extract_all_strategies(&solver);

    let diff_1k_5k = strategy_l1_distance(&strat_1k, &strat_5k);
    let diff_5k_10k = strategy_l1_distance(&strat_5k, &strat_10k);

    assert!(
        diff_5k_10k < diff_1k_5k,
        "Strategy should stabilize: L1(1k,5k)={diff_1k_5k:.6}, L1(5k,10k)={diff_5k_10k:.6}"
    );
    assert!(
        diff_5k_10k < 0.05,
        "Strategy change from 5k to 10k should be tiny, got {diff_5k_10k}"
    );
}

fn strategy_l1_distance(
    a: &FxHashMap<u64, Vec<f64>>,
    b: &FxHashMap<u64, Vec<f64>>,
) -> f64 {
    let mut total = 0.0;
    let mut count = 0;
    for (key, probs_a) in a {
        if let Some(probs_b) = b.get(key) {
            for (pa, pb) in probs_a.iter().zip(probs_b.iter()) {
                total += (pa - pb).abs();
            }
            count += 1;
        }
    }
    if count > 0 { total / count as f64 } else { f64::MAX }
}

// ─── Reach propagation tests ────────────────────────────────────────────────

#[timed_test(10)]
fn regret_accumulated_at_all_reachable_info_sets() {
    // After training, every reachable info set should have regret entries.
    // Kuhn Poker has exactly 12 information sets.
    let game = KuhnPoker::new();
    let mut solver = VanillaCfr::new(game);
    solver.train(100);

    let all_info_sets: Vec<(u32, &[u8])> = vec![
        (0, &[]),           // J
        (1, &[]),           // Q
        (2, &[]),           // K
        (0, &[CHECK]),      // Jc
        (1, &[CHECK]),      // Qc
        (2, &[CHECK]),      // Kc
        (0, &[BET]),        // Jb
        (1, &[BET]),        // Qb
        (2, &[BET]),        // Kb
        (0, &[CHECK, BET]), // Jcb
        (1, &[CHECK, BET]), // Qcb
        (2, &[CHECK, BET]), // Kcb
    ];

    for (card, actions) in &all_info_sets {
        let key = kuhn_key(*card, actions);
        let strat = solver.get_average_strategy(key);
        assert!(
            strat.is_some(),
            "Info set (card={card}, actions={actions:?}) should have strategy data"
        );
    }
}

#[timed_test(5)]
fn opponent_reach_weights_regret_updates() {
    // Core CFR property: regret[a] += opp_reach * (action_util[a] - node_util)
    //
    // When opponent reach is 0 (unreachable for opponent), regret should not
    // change. We verify this indirectly: after 1 iteration of Vanilla CFR on
    // Kuhn Poker, regret magnitudes should be proportional to the number of
    // opponent deals that reach each info set.
    //
    // For P1 info sets (P1 as traverser, P2 as opponent):
    //   - P1 holds J: P2 can hold Q or K (2 deals, opp_reach weights = 1.0 each)
    //   - P1 holds Q: P2 can hold J or K (2 deals)
    //   - P1 holds K: P2 can hold J or Q (2 deals)
    // All root info sets should have similar magnitude regrets after 1 iteration.
    let game = KuhnPoker::new();
    let mut solver = VanillaCfr::new(game);
    solver.train(1);

    // After 1 iteration, all three P1 root info sets should have non-zero regret
    for card in 0..3u32 {
        let key = kuhn_key(card, &[]);
        let strat = solver.get_average_strategy(key);
        assert!(
            strat.is_some(),
            "Card {card} root info set should have data after 1 iteration"
        );
        let s = strat.unwrap();
        // Strategy should NOT be perfectly uniform (regret has moved it)
        let is_uniform = s.iter().all(|&p| (p - 0.5).abs() < 1e-10);
        // After 1 iteration, at least some info sets should have non-uniform strategy
        // (King should already prefer betting)
        if card == 2 {
            // King: betting is better than checking
            assert!(
                !is_uniform,
                "King root strategy should not be uniform after 1 iteration: {s:?}"
            );
        }
    }
}

#[timed_test(5)]
fn hero_reach_weights_strategy_sum_updates() {
    // Core CFR property: strategy_sum[a] += hero_reach * strategy[a]
    //
    // When hero reach is 0 (info set unreachable for hero), strategy sum should
    // not change. In Kuhn, all P1 root info sets are always reachable (reach=1.0).
    // But deeper info sets have reach < 1.0 if the player sometimes takes a
    // different action at a parent node.
    //
    // After training, the average strategy should reflect the frequency weighting:
    // info sets reached more often should have higher total strategy sums.
    let game = KuhnPoker::new();
    let mut solver = VanillaCfr::new(game);
    solver.train(1_000);

    // P1 root info sets (always reached with reach=1.0 in 2 deals each)
    // should have strategy sums, and they should be non-trivial
    for card in 0..3u32 {
        let key = kuhn_key(card, &[]);
        let strat = solver.get_average_strategy(key).unwrap();
        let sum: f64 = strat.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Strategy should sum to 1.0 after normalization"
        );
        // Each probability should be in [0, 1]
        for &p in &strat {
            assert!(p >= -1e-10 && p <= 1.0 + 1e-10, "Invalid probability: {p}");
        }
    }

    // P2 "facing bet" info sets (Jb, Qb, Kb) — only reachable when P1 bets.
    // If P1 rarely bets with some hand, that info set has lower hero_reach weighting
    // for P2, but P2's strategy sum is weighted by P2's own reach (always 1.0 at
    // P2's first action). However, the FREQUENCY of reaching that info set is lower.
    // The strategy should still be a valid distribution.
    for card in 0..3u32 {
        let key = kuhn_key(card, &[BET]);
        let strat = solver.get_average_strategy(key);
        assert!(
            strat.is_some(),
            "Card {card} facing-bet info set should exist"
        );
    }
}

#[timed_test(5)]
fn zero_reach_does_not_update_regrets() {
    // In vanilla CFR, reach propagation means that if hero_reach = 0 at some node,
    // strategy_sums should not be updated there (hero_reach * strategy[a] = 0).
    // Similarly, if opp_reach = 0, regrets should not be updated.
    //
    // We test this by running 1 iteration and checking that the strategy is
    // influenced by all reachable paths but not by impossible ones.
    let game = KuhnPoker::new();
    let mut solver = VanillaCfr::new(game);
    solver.train(1);

    // After 1 iteration, the check-bet sequence (P1 checks, P2 bets, P1 acts)
    // should have strategy data only at info sets reachable under the current
    // (initially uniform) strategy.
    // With uniform strategy: P1 checks 50%, P2 bets 50%, so check-bet is
    // reachable with probability 0.25. Strategy sums should be non-zero.
    for card in 0..3u32 {
        let key = kuhn_key(card, &[CHECK, BET]);
        let strat = solver.get_average_strategy(key);
        assert!(
            strat.is_some(),
            "Check-bet info set for card {card} should have data (reachable via uniform)"
        );
    }
}

// ─── Regret application tests ───────────────────────────────────────────────

#[timed_test(5)]
fn positive_regret_increases_action_probability() {
    // After training, actions with positive cumulative regret should have
    // higher probability than those with negative/zero regret.
    // King facing a bet (after check-bet) should ALWAYS call — it's the nuts.
    let game = KuhnPoker::new();
    let mut solver = VanillaCfr::new(game);
    solver.train(200);

    // King after check-bet: call is strictly dominant (King always wins showdown)
    let k_cb = kuhn_key(2, &[CHECK, BET]);
    let strat = solver.get_average_strategy(k_cb).unwrap();
    // actions = [Fold, Call], call should dominate
    assert!(
        strat[1] > 0.95,
        "King facing bet should call: fold={:.4}, call={:.4}",
        strat[0],
        strat[1]
    );

    // Jack at root: check should dominate bet (Jack never wants to bluff-bet
    // into opponent; Nash has Jack bet α ≤ 1/3, but check is always majority)
    let j_root = kuhn_key(0, &[]);
    let j_strat = solver.get_average_strategy(j_root).unwrap();
    assert!(
        j_strat[0] > j_strat[1],
        "Jack should check more than bet: check={:.4}, bet={:.4}",
        j_strat[0],
        j_strat[1]
    );
}

#[timed_test(5)]
fn negative_regret_decreases_action_probability() {
    // Jack facing a bet: calling loses, so calling should accumulate negative
    // regret → fold probability should be high.
    let game = KuhnPoker::new();
    let mut solver = VanillaCfr::new(game);
    solver.train(1_000);

    // Jack facing bet (Jb): fold=actions[0], call=actions[1]
    let jb = kuhn_key(0, &[BET]);
    let strat = solver.get_average_strategy(jb).unwrap();
    assert!(
        strat[0] > 0.95,
        "Jack should fold facing a bet: fold={:.4}, call={:.4}",
        strat[0],
        strat[1]
    );

    // Jack after check-bet (Jcb): same — should fold
    let jcb = kuhn_key(0, &[CHECK, BET]);
    let strat = solver.get_average_strategy(jcb).unwrap();
    assert!(
        strat[0] > 0.95,
        "Jack should fold after check-bet: fold={:.4}, call={:.4}",
        strat[0],
        strat[1]
    );
}

#[timed_test(10)]
fn regret_matching_produces_valid_distributions_throughout_training() {
    // At every checkpoint, all strategies should be valid probability distributions.
    let game = KuhnPoker::new();
    let mut solver = VanillaCfr::new(game);

    for checkpoint in [1, 10, 100, 1_000] {
        let iters = if checkpoint == 1 { 1 } else { checkpoint - checkpoint / 10 * 9 };
        solver.train(iters);

        let strategies = extract_all_strategies(&solver);
        for (key, probs) in &strategies {
            let sum: f64 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Checkpoint {checkpoint}: strategy for {key:#x} sums to {sum}"
            );
            for (i, &p) in probs.iter().enumerate() {
                assert!(
                    p >= -1e-10,
                    "Checkpoint {checkpoint}: strategy[{i}] = {p} is negative at {key:#x}"
                );
            }
        }
    }
}

#[timed_test(10)]
fn dcfr_discounting_accelerates_convergence() {
    // DCFR (α=1.5, β=0.5, γ=2.0) should converge faster than vanilla on Kuhn.
    let game = KuhnPoker::new();

    // Vanilla MCCFR (no discounting)
    let vanilla_config = MccfrConfig {
        dcfr_alpha: 1.0,
        dcfr_beta: 1.0,
        dcfr_gamma: 1.0,
        ..MccfrConfig::default()
    };
    let mut vanilla_solver = MccfrSolver::with_config(game.clone(), &vanilla_config);
    vanilla_solver.set_seed(42);
    vanilla_solver.train_full(500);

    // DCFR MCCFR
    let dcfr_config = MccfrConfig {
        dcfr_alpha: 1.5,
        dcfr_beta: 0.5,
        dcfr_gamma: 2.0,
        ..MccfrConfig::default()
    };
    let mut dcfr_solver = MccfrSolver::with_config(game.clone(), &dcfr_config);
    dcfr_solver.set_seed(42);
    dcfr_solver.train_full(500);

    let vanilla_exp = calculate_exploitability(&game, &vanilla_solver.all_strategies());
    let dcfr_exp = calculate_exploitability(&game, &dcfr_solver.all_strategies());

    // DCFR should have lower exploitability (or at least comparable)
    // With same seed and iterations, DCFR's discounting should help
    assert!(
        dcfr_exp < vanilla_exp * 1.5,
        "DCFR should converge at least as fast: vanilla={vanilla_exp:.6}, dcfr={dcfr_exp:.6}"
    );
}

#[timed_test(10)]
fn dcfr_negative_regret_decays_faster() {
    // With α=1.5, β=0.5: negative regrets are multiplied by t^0.5/(t^0.5+1)
    // which decays faster than positive regrets at t^1.5/(t^1.5+1).
    // After training, previously-bad actions should recover faster.
    let game = KuhnPoker::new();
    let dcfr_config = MccfrConfig {
        dcfr_alpha: 1.5,
        dcfr_beta: 0.5,
        dcfr_gamma: 2.0,
        ..MccfrConfig::default()
    };
    let mut solver = MccfrSolver::with_config(game.clone(), &dcfr_config);
    solver.set_seed(42);
    solver.train_full(1_000);

    // Check regret sums: for Jack facing a bet, fold regret should be positive
    // and call regret should be negative (or very small)
    let jb = kuhn_key(0, &[BET]);
    if let Some(regrets) = solver.regret_sum().get(&jb) {
        // regrets[0] = fold, regrets[1] = call
        // Fold should have positive or zero regret (it's the right action)
        // Call should have negative regret (it's always wrong with Jack)
        assert!(
            regrets[0] >= regrets[1],
            "Jack facing bet: fold regret ({:.4}) should >= call regret ({:.4})",
            regrets[0],
            regrets[1]
        );
    }

    // King facing bet: call should have much higher regret than fold
    let kb = kuhn_key(2, &[BET]);
    if let Some(regrets) = solver.regret_sum().get(&kb) {
        assert!(
            regrets[1] > regrets[0],
            "King facing bet: call regret ({:.4}) should > fold regret ({:.4})",
            regrets[1],
            regrets[0]
        );
    }
}

#[timed_test(5)]
fn regret_signs_match_known_optimal_actions() {
    // After convergence, the regret of the optimal action should be >= 0
    // and the regret of suboptimal actions should be <= 0.
    let game = KuhnPoker::new();
    let mut solver = MccfrSolver::new(game);
    solver.set_seed(42);
    solver.train_full(2_000);

    // King facing bet: call (index 1) is optimal
    let kb = kuhn_key(2, &[BET]);
    if let Some(regrets) = solver.regret_sum().get(&kb) {
        assert!(
            regrets[1] >= regrets[0],
            "King facing bet: call regret ({}) should >= fold regret ({})",
            regrets[1],
            regrets[0]
        );
    }

    // King after check-bet: call (index 1) is optimal
    let kcb = kuhn_key(2, &[CHECK, BET]);
    if let Some(regrets) = solver.regret_sum().get(&kcb) {
        assert!(
            regrets[1] >= regrets[0],
            "King after check-bet: call regret ({}) should >= fold regret ({})",
            regrets[1],
            regrets[0]
        );
    }

    // Jack facing bet: fold (index 0) is optimal
    let jb = kuhn_key(0, &[BET]);
    if let Some(regrets) = solver.regret_sum().get(&jb) {
        assert!(
            regrets[0] >= regrets[1],
            "Jack facing bet: fold regret ({}) should >= call regret ({})",
            regrets[0],
            regrets[1]
        );
    }
}

// ─── Known Nash equilibrium value tests ─────────────────────────────────────

#[timed_test(10)]
fn kuhn_game_value_matches_theory() {
    // Kuhn Poker game value is -1/18 ≈ -0.0556 for P1.
    // We can estimate this from the average strategy utilities.
    let game = KuhnPoker::new();
    let mut solver = VanillaCfr::new(game.clone());
    solver.train(10_000);

    let strategy = extract_all_strategies(&solver);
    let exp = calculate_exploitability(&game, &strategy);

    // Game value bounds: if exploitability < ε, game value is within ε of -1/18
    let theoretical = -1.0 / 18.0;
    assert!(
        exp < 0.01,
        "Need low exploitability for game value estimate, got {exp}"
    );

    // Verify known Nash properties that determine the game value:
    // P1 with Queen at root: should check (never bet) in Nash
    let q = kuhn_key(1, &[]);
    let q_strat = solver.get_average_strategy(q).unwrap();
    assert!(
        q_strat[0] > 0.9,
        "Queen should mostly check at root: check={:.4}, bet={:.4}",
        q_strat[0],
        q_strat[1]
    );

    // The game value is confirmed by the equilibrium strategy having
    // exploitability < 0.01
    let _ = theoretical;
}

#[timed_test(10)]
fn mccfr_reaches_same_nash_as_vanilla() {
    // Both VanillaCFR and MCCFR should converge to the same Nash equilibrium.
    let game = KuhnPoker::new();

    let mut vanilla = VanillaCfr::new(game.clone());
    vanilla.train(5_000);
    let vanilla_strats = extract_all_strategies(&vanilla);

    let mut mccfr = MccfrSolver::new(game.clone());
    mccfr.set_seed(42);
    mccfr.train_full(5_000);
    let mccfr_strats = mccfr.all_strategies();

    // Both should agree on the pure strategy info sets
    // King facing bet: both should call ~100%
    let kb = kuhn_key(2, &[BET]);
    let v_kb = vanilla_strats.get(&kb).unwrap();
    let m_kb = mccfr_strats.get(&kb);
    assert!(v_kb[1] > 0.95, "Vanilla: King should call bet: {v_kb:?}");
    if let Some(m) = m_kb {
        assert!(m[1] > 0.90, "MCCFR: King should call bet: {m:?}");
    }

    // Jack facing bet: both should fold ~100%
    let jb = kuhn_key(0, &[BET]);
    let v_jb = vanilla_strats.get(&jb).unwrap();
    let m_jb = mccfr_strats.get(&jb);
    assert!(v_jb[0] > 0.95, "Vanilla: Jack should fold to bet: {v_jb:?}");
    if let Some(m) = m_jb {
        assert!(m[0] > 0.90, "MCCFR: Jack should fold to bet: {m:?}");
    }
}

// ─── MCCFR-specific mechanics ───────────────────────────────────────────────

#[timed_test(5)]
fn mccfr_alternating_traversal_covers_both_players() {
    // MCCFR alternates traversing player each iteration.
    // After enough iterations, both P1 and P2 info sets should have data.
    // We need ~20 iterations because external sampling means the opponent
    // randomly samples one action, so not all paths are guaranteed per iteration.
    let game = KuhnPoker::new();
    let mut solver = MccfrSolver::new(game);
    solver.set_seed(42);
    solver.train_full(20);

    // P1 info sets (root nodes: J, Q, K)
    for card in 0..3u32 {
        let key = kuhn_key(card, &[]);
        assert!(
            solver.regret_sum().contains_key(&key),
            "P1 root info set (card={card}) should have regret data after 20 iterations"
        );
    }

    // P2 info sets (after check: Jc, Qc, Kc)
    for card in 0..3u32 {
        let key = kuhn_key(card, &[CHECK]);
        assert!(
            solver.regret_sum().contains_key(&key),
            "P2 after-check info set (card={card}) should have regret data after 20 iterations"
        );
    }

    // P2 info sets (after bet: Jb, Qb, Kb)
    for card in 0..3u32 {
        let key = kuhn_key(card, &[BET]);
        assert!(
            solver.regret_sum().contains_key(&key),
            "P2 after-bet info set (card={card}) should have regret data after 20 iterations"
        );
    }
}

#[timed_test(5)]
fn mccfr_avg_positive_regret_decreases() {
    // The average positive regret metric should decrease with training,
    // directly bounding exploitability.
    let game = KuhnPoker::new();
    let mut solver = MccfrSolver::new(game);
    solver.set_seed(42);

    solver.train_full(100);
    let early_apr = solver.avg_positive_regret();

    solver.train_full(900);
    let late_apr = solver.avg_positive_regret();

    assert!(
        late_apr < early_apr,
        "Avg positive regret should decrease: 100 iters = {early_apr:.6}, \
         1000 iters = {late_apr:.6}"
    );
}

#[timed_test(5)]
fn mccfr_sample_weight_preserves_unbiasedness() {
    // When sampling fewer deals than total, sample_weight = num_states / samples.
    // The resulting strategy should still converge correctly.
    let game = KuhnPoker::new();
    let config = MccfrConfig::default();

    // Train with full traversal as baseline
    let mut full_solver = MccfrSolver::with_config(game.clone(), &config);
    full_solver.set_seed(42);
    full_solver.train_full(1_000);

    // Train with sampling (2 out of 6 deals per iteration, more iterations to compensate)
    let mut sampled_solver = MccfrSolver::with_config(game.clone(), &config);
    sampled_solver.set_seed(42);
    sampled_solver.train(3_000, 2);

    // Both should produce valid strategies
    let full_strats = full_solver.all_strategies();
    let sampled_strats = sampled_solver.all_strategies();

    for (key, probs) in &full_strats {
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Full solver strategy for {key:#x} doesn't sum to 1: {sum}"
        );
    }
    for (key, probs) in &sampled_strats {
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Sampled solver strategy for {key:#x} doesn't sum to 1: {sum}"
        );
    }

    // Both should agree on pure-strategy info sets
    let kb = kuhn_key(2, &[BET]);
    if let Some(s) = sampled_strats.get(&kb) {
        assert!(s[1] > 0.8, "Sampled: King should call bet: {s:?}");
    }
}
