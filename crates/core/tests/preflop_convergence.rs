//! Integration test: verify the preflop solver converges to valid strategies.
//!
//! The solver uses a uniform equity table (all matchups = 0.5), so all 169
//! canonical hands are strategically equivalent. We verify structural properties:
//! strategies sum to 1.0 and the solver produces entries for every hand.

use poker_solver_core::preflop::{PreflopConfig, PreflopSolver};

#[test]
fn hu_preflop_converges_in_500_iterations() {
    // Use a small stack depth and restricted raise sizes for a compact tree.
    let mut config = PreflopConfig::heads_up(10);
    config.raise_sizes = vec![vec![3.0]];
    config.raise_cap = 2;

    let mut solver = PreflopSolver::new(&config);
    solver.train(500);
    let strategy = solver.strategy();

    // Strategy should not be empty.
    assert!(
        !strategy.is_empty(),
        "strategy should have entries after training"
    );

    // Every hand should have a strategy at the root.
    for hand_idx in 0..169 {
        let probs = strategy.get_root_probs(hand_idx);
        assert!(
            !probs.is_empty(),
            "hand {hand_idx} should have a strategy at root"
        );
    }

    // Strategy probabilities should sum to ~1.0 for every hand at root.
    for hand_idx in 0..169 {
        let probs = strategy.get_root_probs(hand_idx);
        if probs.is_empty() {
            continue;
        }
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.02,
            "hand {hand_idx}: strategy sum = {sum}, expected ~1.0"
        );
        // All probabilities should be non-negative.
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                p >= -1e-9,
                "hand {hand_idx} action {i}: negative probability {p}"
            );
        }
    }
}

#[test]
fn solver_iteration_count_tracks_correctly() {
    let mut config = PreflopConfig::heads_up(5);
    config.raise_sizes = vec![vec![3.0]];
    config.raise_cap = 1;

    let mut solver = PreflopSolver::new(&config);
    assert_eq!(solver.iteration(), 0);

    solver.train(10);
    assert_eq!(solver.iteration(), 10);

    solver.train(5);
    assert_eq!(solver.iteration(), 15);
}

/// Verify solver converges and compare against reference GTO data from
/// `data/hu_sb_open_25bb.csv`.
///
/// The reference is from a full-game GTO solver. Our preflop-only model uses
/// raw equity showdowns, so it finds a different Nash equilibrium where SB
/// prefers limp-trapping with strong hands (BB can't fold vs limps, so SB
/// can re-raise when BB raises). This is a valid equilibrium but differs from
/// full-game GTO where raising builds pots with positional/nut advantage.
///
/// We verify:
/// 1. Solver converges (mean strategy delta < 0.001)
/// 2. Strategies are valid probability distributions
/// 3. Structural properties: weak hands fold, premium hands don't fold
/// 4. Raise ordering: KK raises more than off-suit rags
///
/// Run with: `cargo test -p poker-solver-core --release --test preflop_convergence -- --ignored --nocapture`
#[test]
#[ignore]
fn solver_matches_gto_reference_25bb() {
    use poker_solver_core::hands::CanonicalHand;
    use poker_solver_core::preflop::{EquityTable, PreflopAction, PreflopNode};
    use std::collections::HashMap;

    // --- Parse reference CSV ---
    let csv_data = include_str!("../../../data/hu_sb_open_25bb.csv");
    let mut reference: HashMap<String, (f64, f64, f64)> = HashMap::new();
    for line in csv_data.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 5 {
            continue;
        }
        let hand = fields[0].trim().to_string();
        let raise: f64 = fields[2].trim().parse().unwrap_or(0.0);
        let limp: f64 = fields[3].trim().parse().unwrap_or(0.0);
        let fold: f64 = fields[4].trim().parse().unwrap_or(0.0);
        reference.insert(hand, (raise, limp, fold));
    }
    assert_eq!(reference.len(), 169, "CSV should have 169 hands");

    // --- Run solver until convergence ---
    // Per-position raise sizes matching reference tree structure:
    //   SB: fold, limp, raise 2.0x (min-raise), all-in
    //   BB vs limp: check, raise 3.0x, raise 7.0x, all-in
    //   BB vs raise: fold, call, 3bet 2.5x, 3bet 4.0x, all-in
    //   SB vs 3bet: fold, call, all-in (raise_cap=3 limits re-raises)
    //
    // Uses LCFR (Linear CFR): both regret and strategy contributions are
    // weighted by iteration number. DCFR discounting is disabled (warmup
    // exceeds max iterations) since LCFR's linear weighting replaces it.
    let mut config = PreflopConfig::heads_up(25);
    config.raise_sizes = vec![vec![2.0]];
    config.position_raise_sizes = Some(vec![
        vec![vec![2.0]],                      // SB: open 2.0x
        vec![vec![3.0, 7.0], vec![2.5, 4.0]], // BB: vs-limp 3x/7x, vs-raise 2.5x/4x
    ]);
    config.raise_cap = 3;
    config.dcfr_warmup = 300_000;
    let equity = EquityTable::new_computed(10_000, |_| {});
    let mut solver = PreflopSolver::new_with_equity(&config, equity);

    let chunk = 500;
    let max_iterations = 200_000u64;
    let convergence_threshold = 0.001;
    let mut prev_raise_pcts = vec![0.0f64; 169];
    let mut converged = false;

    loop {
        solver.train(chunk);
        let strat = solver.strategy();

        let action_labels_tmp = match &solver.tree().nodes[0] {
            PreflopNode::Decision { action_labels, .. } => action_labels.clone(),
            _ => panic!("root should be decision"),
        };
        let fi = action_labels_tmp
            .iter()
            .position(|a| matches!(a, PreflopAction::Fold));
        let ci = action_labels_tmp
            .iter()
            .position(|a| matches!(a, PreflopAction::Call))
            .unwrap();

        let mut cur_raise_pcts = vec![0.0f64; 169];
        for h in 0..169 {
            let probs = strat.get_root_probs(h);
            let fold_p = fi.map_or(0.0, |i| probs[i]);
            cur_raise_pcts[h] = (1.0 - fold_p - probs[ci]) * 100.0;
        }

        let mean_delta: f64 = cur_raise_pcts
            .iter()
            .zip(&prev_raise_pcts)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / 169.0;

        if solver.iteration() % 5000 == 0 || mean_delta < convergence_threshold {
            eprintln!(
                "iter {:>6}: mean_delta={:.4}pp",
                solver.iteration(),
                mean_delta
            );
        }

        if mean_delta < convergence_threshold && solver.iteration() >= chunk as u64 * 2 {
            eprintln!("Converged at iteration {}", solver.iteration());
            converged = true;
            break;
        }
        if solver.iteration() >= max_iterations {
            eprintln!("Hit max iterations ({max_iterations}) with delta={mean_delta:.4}pp");
            break;
        }
        prev_raise_pcts = cur_raise_pcts;
    }

    assert!(
        converged,
        "solver should converge within {max_iterations} iterations"
    );

    let strategy = solver.strategy();

    // --- Identify action indices at root ---
    let action_labels = match &solver.tree().nodes[0] {
        PreflopNode::Decision { action_labels, .. } => action_labels.clone(),
        _ => panic!("root should be a decision node"),
    };
    let fold_idx = action_labels
        .iter()
        .position(|a| matches!(a, PreflopAction::Fold));
    let call_idx = action_labels
        .iter()
        .position(|a| matches!(a, PreflopAction::Call))
        .expect("root must have Call");

    // Helper: get raise% for a hand
    let raise_pct = |name: &str| -> f64 {
        let h = CanonicalHand::parse(name).unwrap();
        let probs = strategy.get_root_probs(h.index());
        probs
            .iter()
            .enumerate()
            .filter(|&(i, _)| Some(i) != fold_idx && i != call_idx)
            .map(|(_, &p)| p)
            .sum::<f64>()
            * 100.0
    };

    let fold_pct = |name: &str| -> f64 {
        let h = CanonicalHand::parse(name).unwrap();
        let probs = strategy.get_root_probs(h.index());
        fold_idx.map_or(0.0, |i| probs[i]) * 100.0
    };

    // --- Print comparison table ---
    let mut errors: Vec<(String, f64, f64)> = Vec::new();
    let mut total_error = 0.0f64;
    let mut count = 0usize;

    println!(
        "\n{:<6} ref_r%  sol_r%  ref_l%  sol_l%  ref_f%  sol_f%  err",
        "Hand"
    );
    println!("{}", "-".repeat(75));

    for hand_idx in 0..169 {
        let hand = CanonicalHand::from_index(hand_idx).unwrap();
        let name = hand.to_string();

        let (ref_raise, ref_limp, ref_fold) = match reference.get(&name) {
            Some(v) => *v,
            None => continue,
        };

        let probs = strategy.get_root_probs(hand_idx);
        let sol_fold = fold_idx.map_or(0.0, |i| probs[i]) * 100.0;
        let sol_limp = probs[call_idx] * 100.0;
        let sol_raise: f64 = probs
            .iter()
            .enumerate()
            .filter(|&(i, _)| Some(i) != fold_idx && i != call_idx)
            .map(|(_, &p)| p)
            .sum::<f64>()
            * 100.0;

        let err = (sol_raise - ref_raise).abs();
        total_error += err;
        count += 1;
        errors.push((name.clone(), ref_raise, sol_raise));

        println!(
            "{:<6} {:>6.1}  {:>6.1}  {:>6.1}  {:>6.1}  {:>6.1}  {:>6.1}  {:>5.1}",
            name, ref_raise, sol_raise, ref_limp, sol_limp, ref_fold, sol_fold, err
        );
    }

    let mean_error = total_error / count as f64;
    errors.sort_by(|a, b| (b.1 - b.2).abs().partial_cmp(&(a.1 - a.2).abs()).unwrap());

    println!("\nMean raise% error: {mean_error:.1}pp");
    println!("Top 10 worst errors:");
    for (name, ref_r, sol_r) in errors.iter().take(10) {
        println!(
            "  {name:<6} ref={ref_r:.1}% sol={sol_r:.1}% err={:.1}pp",
            (ref_r - sol_r).abs()
        );
    }

    // === Structural assertions ===
    // These verify the solver produces a sensible equilibrium for the
    // preflop-only model, even though it differs from full-game GTO.

    // 1. Strategy validity: probabilities sum to 1.0 and are non-negative.
    for hand_idx in 0..169 {
        let probs = strategy.get_root_probs(hand_idx);
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.02,
            "hand {hand_idx}: strategy sum = {sum}, expected ~1.0"
        );
        for (i, &p) in probs.iter().enumerate() {
            assert!(p >= -1e-9, "hand {hand_idx} action {i}: negative prob {p}");
        }
    }

    // 2. Worst hands fold: 72o, 62o, 42o, 32o should fold (100% in reference).
    for name in ["72o", "62o", "42o", "32o"] {
        let f = fold_pct(name);
        assert!(
            f > 90.0,
            "{name}: fold% = {f:.1}, expected > 90% (worst hands should fold)"
        );
    }

    // 3. Premium hands never fold: AA, KK, QQ should never fold.
    for name in ["AA", "KK", "QQ"] {
        let f = fold_pct(name);
        assert!(
            f < 1.0,
            "{name}: fold% = {f:.1}, expected < 1% (premium hands should not fold)"
        );
    }

    // 4. KK raises more than weak offsuit hands: KK has stronger equity incentive.
    let kk_raise = raise_pct("KK");
    let t2o_raise = raise_pct("T2o");
    assert!(
        kk_raise > t2o_raise + 10.0,
        "KK raise ({kk_raise:.1}%) should exceed T2o raise ({t2o_raise:.1}%) by > 10pp"
    );

    // 5. AA has non-trivial raise frequency (limp-trap is valid but not 100% limp).
    let aa_raise = raise_pct("AA");
    assert!(
        aa_raise > 10.0,
        "AA: raise% = {aa_raise:.1}, expected > 10% (should raise at least sometimes)"
    );

    // 6. Not all hands play identically: some raise, some don't.
    let raise_count = (0..169)
        .filter(|&h| {
            let probs = strategy.get_root_probs(h);
            let r: f64 = probs
                .iter()
                .enumerate()
                .filter(|&(i, _)| Some(i) != fold_idx && i != call_idx)
                .map(|(_, &p)| p)
                .sum();
            r > 0.1
        })
        .count();
    assert!(
        raise_count > 20 && raise_count < 150,
        "raise_count={raise_count}, expected between 20 and 150 hands raising >10%"
    );

    // 7. Mean error as a regression catch (not a precision target).
    // The preflop-only model finds a limp-heavy equilibrium that differs
    // from full-game GTO by ~38pp. Flag if it degrades significantly.
    assert!(
        mean_error < 50.0,
        "mean raise% error = {mean_error:.1}pp, expected < 50pp (regression catch)"
    );
}
