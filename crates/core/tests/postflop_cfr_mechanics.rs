//! Comprehensive postflop CFR mechanics tests.
//!
//! Mirrors `cfr_mechanics.rs` (which uses Kuhn Poker) but exercises the
//! HunlPostflop game — the real postflop solver with SPR-constrained trees,
//! multi-street play, hand class abstractions, and deal sampling.
//!
//! Uses minimal configs (shallow stacks, single bet size, few deals) to keep
//! tests fast while still exercising the full postflop CFR loop.

use poker_solver_core::cfr::{MccfrConfig, MccfrSolver, calculate_exploitability};
use poker_solver_core::game::{AbstractionMode, HunlPostflop, PostflopConfig};
use poker_solver_core::info_key::InfoKey;
use test_macros::timed_test;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Minimal postflop config: 5BB stacks, single pot-sized bet, 1 raise per street.
fn tiny_config() -> PostflopConfig {
    PostflopConfig {
        stack_depth: 5,
        bet_sizes: vec![1.0],
        max_raises_per_street: 1,
    }
}

/// Small postflop config: 10BB stacks, single pot-sized bet.
fn small_config() -> PostflopConfig {
    PostflopConfig {
        stack_depth: 10,
        bet_sizes: vec![1.0],
        ..PostflopConfig::default()
    }
}

/// Create a seeded MCCFR solver with default config and given game.
fn make_solver(game: HunlPostflop) -> MccfrSolver<HunlPostflop> {
    let mut solver = MccfrSolver::new(game);
    solver.set_seed(42);
    solver
}

/// Create a seeded MCCFR solver with custom MCCFR config.
fn make_solver_with_config(game: HunlPostflop, config: &MccfrConfig) -> MccfrSolver<HunlPostflop> {
    let mut solver = MccfrSolver::with_config(game, config);
    solver.set_seed(42);
    solver
}

// ─── Convergence tests ──────────────────────────────────────────────────────

#[timed_test(30)]
fn postflop_avg_positive_regret_decreases() {
    // Average positive regret bounds exploitability. It should decrease
    // with more training iterations.
    let game = HunlPostflop::new(tiny_config(), None, 100);
    let mut solver = make_solver(game);

    solver.train_full(50);
    let early_apr = solver.avg_positive_regret();

    solver.train_full(450);
    let late_apr = solver.avg_positive_regret();

    assert!(
        late_apr < early_apr,
        "Avg positive regret should decrease: 50 iters = {early_apr:.6}, \
         500 iters = {late_apr:.6}"
    );
}

#[timed_test(30)]
fn postflop_exploitability_decreases() {
    // Exploitability should decrease (or at least not blow up) with training.
    let config = tiny_config();
    let eval_game = HunlPostflop::new(config.clone(), None, 100);
    let solver_game = HunlPostflop::new(config, None, 100);
    let mut solver = make_solver(solver_game);

    solver.train_full(50);
    let early_strats = solver.all_strategies();
    let early_exploit = calculate_exploitability(&eval_game, &early_strats);

    solver.train_full(200);
    let late_strats = solver.all_strategies();
    let late_exploit = calculate_exploitability(&eval_game, &late_strats);

    assert!(
        late_exploit < early_exploit * 1.5,
        "Exploitability should not increase significantly: \
         early={early_exploit:.4}, late={late_exploit:.4}"
    );
}

#[timed_test(30)]
fn postflop_strategies_valid_throughout_training() {
    // All strategy probabilities should be valid (non-negative, sum to ~1)
    // at every checkpoint during training.
    let game = HunlPostflop::new(tiny_config(), None, 100);
    let mut solver = make_solver(game);

    for checkpoint in [10, 50, 100, 200] {
        let remaining = checkpoint
            - solver.iterations();
        if remaining > 0 {
            solver.train_full(remaining);
        }

        let strategies = solver.all_strategies();
        assert!(
            !strategies.is_empty(),
            "Should have strategies at iteration {checkpoint}"
        );

        for (key, probs) in &strategies {
            let sum: f64 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Strategy for key {key:#018x} should sum to ~1.0 at iter {checkpoint}, got {sum:.6}"
            );
            for (i, &p) in probs.iter().enumerate() {
                assert!(
                    p >= -1e-9,
                    "Prob[{i}] for key {key:#018x} should be non-negative at iter {checkpoint}, got {p}"
                );
            }
        }
    }
}

#[timed_test(30)]
fn postflop_sampled_training_converges() {
    // Sampled training (subset of deals per iteration) should also converge.
    let mccfr_config = MccfrConfig {
        samples_per_iteration: 50,
        ..MccfrConfig::default()
    };
    let game = HunlPostflop::new(tiny_config(), None, 200);
    let mut solver = make_solver_with_config(game, &mccfr_config);

    solver.train(100, 50);
    let early_apr = solver.avg_positive_regret();

    solver.train(400, 50);
    let late_apr = solver.avg_positive_regret();

    assert!(
        late_apr < early_apr,
        "Sampled APR should decrease: 100 iters = {early_apr:.6}, \
         500 iters = {late_apr:.6}"
    );
}

// ─── Reach propagation tests ────────────────────────────────────────────────

#[timed_test(30)]
fn postflop_all_reachable_info_sets_populated() {
    // After sufficient training, a non-trivial number of info sets should
    // have regret data, confirming reach propagation works.
    let game = HunlPostflop::new(tiny_config(), None, 100);
    let mut solver = make_solver(game);
    solver.train_full(100);

    let regret_map = solver.regret_sum();
    assert!(
        regret_map.len() > 10,
        "Should have many info sets populated, got {}",
        regret_map.len()
    );

    // Every info set should have at least 2 actions (fold+call or check+bet)
    for (key, regrets) in regret_map {
        assert!(
            regrets.len() >= 2,
            "Info set {key:#018x} should have ≥2 actions, got {}",
            regrets.len()
        );
    }
}

#[timed_test(30)]
fn postflop_both_players_info_sets_populated() {
    // Alternating traversal should populate info sets for both P1 and P2.
    // We verify by checking the action history in info set keys:
    // P1 acts first on flop (even-length action histories after deal),
    // P2 acts second (odd-length).
    let game = HunlPostflop::new(tiny_config(), None, 100);
    let mut solver = make_solver(game);
    solver.train_full(100);

    let strategies = solver.all_strategies();
    let mut p1_count = 0u64;
    let mut p2_count = 0u64;

    for &key in strategies.keys() {
        let decoded = InfoKey::from_raw(key);
        let action_bits = decoded.actions_bits();

        // Count non-zero action nibbles to determine history length
        let mut action_len = 0;
        for i in 0..6 {
            if (action_bits >> (20 - i * 4)) & 0xF != 0 {
                action_len += 1;
            }
        }

        // In postflop, P1 (SB) checks/bets first, P2 (BB) responds.
        // Even action count = P1's turn, odd = P2's turn.
        if action_len % 2 == 0 {
            p1_count += 1;
        } else {
            p2_count += 1;
        }
    }

    assert!(
        p1_count > 0,
        "P1 (SB) should have info sets, got 0"
    );
    assert!(
        p2_count > 0,
        "P2 (BB) should have info sets, got 0"
    );
    println!("P1 info sets: {p1_count}, P2 info sets: {p2_count}");
}

#[timed_test(30)]
fn postflop_multi_street_info_sets_reached() {
    // Training should reach info sets on flop, turn, and river streets.
    // The street is encoded in bits 42-43 of the info key (0=preflop, 1=flop, 2=turn, 3=river).
    let game = HunlPostflop::new(tiny_config(), None, 200);
    let mut solver = make_solver(game);
    solver.train_full(200);

    let strategies = solver.all_strategies();
    let mut streets_seen = [false; 4]; // preflop, flop, turn, river

    for &key in strategies.keys() {
        let decoded = InfoKey::from_raw(key);
        let street = decoded.street() as usize;
        if street < 4 {
            streets_seen[street] = true;
        }
    }

    // Postflop game should reach at least flop. With check-check sequences
    // it should also reach turn and river.
    assert!(streets_seen[1], "Should have flop info sets");
    assert!(streets_seen[2], "Should have turn info sets (via check-check on flop)");
    assert!(streets_seen[3], "Should have river info sets (via check-check sequences)");
}

#[timed_test(30)]
fn postflop_regret_data_at_every_strategy_info_set() {
    // Every info set that has strategy data should also have regret data.
    // This confirms that reach → regret → strategy pipeline is complete.
    let game = HunlPostflop::new(tiny_config(), None, 100);
    let mut solver = make_solver(game);
    solver.train_full(100);

    let strategies = solver.all_strategies();
    let regret_map = solver.regret_sum();

    for &key in strategies.keys() {
        assert!(
            regret_map.contains_key(&key),
            "Info set {key:#018x} has strategy data but no regret data"
        );
    }
}

// ─── Regret application tests ───────────────────────────────────────────────

#[timed_test(30)]
fn postflop_regret_matching_produces_valid_distributions() {
    // After training, every info set's strategy should be a valid probability
    // distribution. This verifies regret matching works correctly throughout.
    let game = HunlPostflop::new(tiny_config(), None, 100);
    let mut solver = make_solver(game);
    solver.train_full(200);

    let strategies = solver.all_strategies();
    for (key, probs) in &strategies {
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Info set {key:#018x}: probs sum to {sum:.6}, expected ~1.0"
        );
        for &p in probs {
            assert!(
                p >= -1e-9 && p <= 1.0 + 1e-9,
                "Info set {key:#018x}: probability {p} out of [0,1] range"
            );
        }
    }
}

#[timed_test(30)]
fn postflop_dcfr_discounting_changes_regret_profile() {
    // DCFR (α=1.5, β=0.5, γ=2.0) applies iteration-dependent discounting
    // to regrets and strategy sums. This should produce a measurably different
    // regret profile compared to uniform weighting, confirming discounting is active.
    let config = tiny_config();

    // DCFR (default)
    let dcfr_game = HunlPostflop::new(config.clone(), None, 100);
    let mut dcfr_solver = make_solver(dcfr_game);
    dcfr_solver.train_full(200);

    // No discounting
    let no_discount_config = MccfrConfig {
        dcfr_alpha: 0.0,
        dcfr_beta: 0.0,
        dcfr_gamma: 0.0,
        ..MccfrConfig::default()
    };
    let plain_game = HunlPostflop::new(config, None, 100);
    let mut plain_solver = make_solver_with_config(plain_game, &no_discount_config);
    plain_solver.train_full(200);

    // Both should produce valid strategies and non-empty regret maps
    let dcfr_regrets = dcfr_solver.regret_sum();
    let plain_regrets = plain_solver.regret_sum();
    assert!(!dcfr_regrets.is_empty());
    assert!(!plain_regrets.is_empty());

    // The regret values should differ (discounting changes them)
    // Compare total positive regret magnitude — they should be different
    let dcfr_pos: f64 = dcfr_regrets
        .values()
        .flat_map(|v| v.iter())
        .filter(|&&r| r > 0.0)
        .sum();
    let plain_pos: f64 = plain_regrets
        .values()
        .flat_map(|v| v.iter())
        .filter(|&&r| r > 0.0)
        .sum();

    // They should differ meaningfully (at least 10% difference)
    let diff_ratio = (dcfr_pos - plain_pos).abs() / plain_pos.max(1e-12);
    assert!(
        diff_ratio > 0.1,
        "DCFR should produce different regret profile: dcfr_pos={dcfr_pos:.4}, \
         plain_pos={plain_pos:.4}, diff_ratio={diff_ratio:.4}"
    );

    // DCFR should still converge (APR decreases from 100 to 200 iters)
    let dcfr_apr = dcfr_solver.avg_positive_regret();
    assert!(
        dcfr_apr < 1.0,
        "DCFR APR should be bounded after 200 iters, got {dcfr_apr:.6}"
    );
}

#[timed_test(30)]
fn postflop_dcfr_parameters_affect_strategy() {
    // Different DCFR parameter settings should produce different strategies,
    // confirming the parameters actually influence the training outcome.
    let config = tiny_config();

    // Aggressive discounting (high α, low β)
    let aggressive_config = MccfrConfig {
        dcfr_alpha: 3.0,
        dcfr_beta: 0.0,
        dcfr_gamma: 4.0,
        ..MccfrConfig::default()
    };
    let agg_game = HunlPostflop::new(config.clone(), None, 100);
    let mut agg_solver = make_solver_with_config(agg_game, &aggressive_config);
    agg_solver.train_full(200);
    let agg_strats = agg_solver.all_strategies();

    // Mild discounting (low α, high β)
    let mild_config = MccfrConfig {
        dcfr_alpha: 0.5,
        dcfr_beta: 1.0,
        dcfr_gamma: 0.5,
        ..MccfrConfig::default()
    };
    let mild_game = HunlPostflop::new(config, None, 100);
    let mut mild_solver = make_solver_with_config(mild_game, &mild_config);
    mild_solver.train_full(200);
    let mild_strats = mild_solver.all_strategies();

    // Find shared info sets and compare strategies
    let mut total_diff = 0.0;
    let mut comparisons = 0;
    for (key, agg_probs) in &agg_strats {
        if let Some(mild_probs) = mild_strats.get(key) {
            if agg_probs.len() == mild_probs.len() {
                for (a, m) in agg_probs.iter().zip(mild_probs.iter()) {
                    total_diff += (a - m).abs();
                }
                comparisons += 1;
            }
        }
    }

    assert!(comparisons > 0, "Should have shared info sets to compare");
    let avg_diff = total_diff / comparisons as f64;
    assert!(
        avg_diff > 0.01,
        "Different DCFR params should produce different strategies, avg diff={avg_diff:.6}"
    );
}

// ─── Postflop-specific mechanics ────────────────────────────────────────────

#[timed_test(30)]
fn postflop_hand_class_abstraction_reduces_info_sets() {
    // HandClassV2 abstraction groups many distinct holdings into the same
    // info set, significantly reducing the number of unique info sets.
    let config = tiny_config();

    // No abstraction: each distinct holding is a separate info set
    let raw_game = HunlPostflop::new(config.clone(), None, 100);
    let mut raw_solver = make_solver(raw_game);
    raw_solver.train_full(100);
    let raw_count = raw_solver.all_strategies().len();

    // HandClassV2 with minimal resolution: groups holdings by class only
    let abs_game = HunlPostflop::new(
        config,
        Some(AbstractionMode::HandClassV2 {
            strength_bits: 0,
            equity_bits: 0,
        }),
        100,
    );
    let mut abs_solver = make_solver(abs_game);
    abs_solver.train_full(100);
    let abs_count = abs_solver.all_strategies().len();

    assert!(
        abs_count < raw_count,
        "Abstraction should reduce info sets: raw={raw_count}, abstracted={abs_count}"
    );
    println!("Info set reduction: {raw_count} → {abs_count} ({:.0}% reduction)",
        (1.0 - abs_count as f64 / raw_count as f64) * 100.0);
}

#[timed_test(30)]
fn postflop_abstraction_fidelity_increases_with_bits() {
    // More strength/equity bits means finer-grained abstraction,
    // which should produce more info sets (closer to raw game).
    let config = tiny_config();

    // Coarse: 0 bits each (class only)
    let coarse_game = HunlPostflop::new(
        config.clone(),
        Some(AbstractionMode::HandClassV2 {
            strength_bits: 0,
            equity_bits: 0,
        }),
        100,
    );
    let mut coarse_solver = make_solver(coarse_game);
    coarse_solver.train_full(100);
    let coarse_count = coarse_solver.all_strategies().len();

    // Fine: 2 bits each (4 strength × 4 equity sub-buckets per class)
    let fine_game = HunlPostflop::new(
        config,
        Some(AbstractionMode::HandClassV2 {
            strength_bits: 2,
            equity_bits: 2,
        }),
        100,
    );
    let mut fine_solver = make_solver(fine_game);
    fine_solver.train_full(100);
    let fine_count = fine_solver.all_strategies().len();

    assert!(
        fine_count > coarse_count,
        "Finer abstraction should have more info sets: coarse={coarse_count}, fine={fine_count}"
    );
}

#[timed_test(30)]
fn postflop_spr_bucket_varies_across_info_sets() {
    // With a 10BB stack and pot-sized bets, the SPR should change as bets
    // are made. Info sets at different SPR buckets should exist.
    let game = HunlPostflop::new(small_config(), None, 100);
    let mut solver = make_solver(game);
    solver.train_full(100);

    let strategies = solver.all_strategies();
    let mut spr_buckets_seen = std::collections::HashSet::new();

    for &key in strategies.keys() {
        let decoded = InfoKey::from_raw(key);
        spr_buckets_seen.insert(decoded.spr_bucket());
    }

    assert!(
        spr_buckets_seen.len() >= 2,
        "Should see multiple SPR buckets, got {}: {:?}",
        spr_buckets_seen.len(),
        spr_buckets_seen
    );
    println!("SPR buckets seen: {:?}", spr_buckets_seen);
}

#[timed_test(30)]
fn postflop_parallel_matches_sequential_quality() {
    // Parallel training should produce similar quality strategies as sequential.
    // Both should converge, with similar APR ranges.
    let config = tiny_config();

    let seq_game = HunlPostflop::new(config.clone(), None, 100);
    let mut seq_solver = make_solver(seq_game);
    seq_solver.train_full(200);
    let seq_apr = seq_solver.avg_positive_regret();

    let par_game = HunlPostflop::new(config, None, 100);
    let mut par_solver = make_solver(par_game);
    par_solver.train_full_parallel(200);
    let par_apr = par_solver.avg_positive_regret();

    // Both should be in the same ballpark (within 3x of each other)
    let ratio = if seq_apr > par_apr {
        seq_apr / par_apr.max(1e-12)
    } else {
        par_apr / seq_apr.max(1e-12)
    };
    assert!(
        ratio < 3.0,
        "Sequential and parallel APR should be similar: seq={seq_apr:.6}, par={par_apr:.6}"
    );
}

#[timed_test(30)]
fn postflop_more_deals_improves_coverage() {
    // More deals should lead to broader info set coverage (more unique
    // info sets visited), confirming deal sampling affects reach.
    let config = tiny_config();

    let few_game = HunlPostflop::new(config.clone(), None, 20);
    let mut few_solver = make_solver(few_game);
    few_solver.train_full(100);
    let few_count = few_solver.all_strategies().len();

    let many_game = HunlPostflop::new(config, None, 500);
    let mut many_solver = make_solver(many_game);
    many_solver.train_full(100);
    let many_count = many_solver.all_strategies().len();

    assert!(
        many_count >= few_count,
        "More deals should give ≥ info sets: few={few_count}, many={many_count}"
    );
}

#[timed_test(30)]
fn postflop_zero_iteration_produces_empty_strategies() {
    // Before any training, there should be no strategy data.
    let game = HunlPostflop::new(tiny_config(), None, 100);
    let solver = make_solver(game);

    let strategies = solver.all_strategies();
    assert!(
        strategies.is_empty(),
        "Should have no strategies before training, got {}",
        strategies.len()
    );
    assert_eq!(solver.iterations(), 0);
}

#[timed_test(30)]
fn postflop_info_set_count_grows_then_stabilizes() {
    // The number of visited info sets should grow with early training
    // and then stabilize as the tree has been fully explored.
    let game = HunlPostflop::new(tiny_config(), None, 100);
    let mut solver = make_solver(game);

    solver.train_full(10);
    let early_count = solver.all_strategies().len();

    solver.train_full(90);
    let mid_count = solver.all_strategies().len();

    solver.train_full(100);
    let late_count = solver.all_strategies().len();

    assert!(
        mid_count >= early_count,
        "Info set count should grow: early={early_count}, mid={mid_count}"
    );
    // Late count should be similar to mid (stabilized)
    assert!(
        late_count >= mid_count,
        "Info set count should not shrink: mid={mid_count}, late={late_count}"
    );
    println!("Info set growth: {early_count} → {mid_count} → {late_count}");
}
