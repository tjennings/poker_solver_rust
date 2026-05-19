use std::time::Instant;

use range_solver_compare::*;

const RIVER_SMOKE_ITERATIONS: u32 = 200;
// Exploitability is a scalar f32 returned after a fixed deterministic solve.
// 1e-4 chips is tight enough to catch solver drift while allowing harmless
// floating-point accumulation differences across the two implementations.
const EXPLOITABILITY_TOLERANCE: f32 = 1.0e-4;
const ROOT_STRATEGY_TOLERANCE: f32 = 1.0e-4;
const EV_TOLERANCE: f32 = 1.0e-3;
const EQUITY_TOLERANCE: f32 = 1.0e-6;

/// First-slice audit: deterministic hand-authored river spots only.
#[test]
fn river_smoke_structural_and_numeric_parity() {
    let spots = river_smoke_spots();

    for spot in &spots {
        let ours = run_ours(spot, RIVER_SMOKE_ITERATIONS);
        let original = run_original(spot, RIVER_SMOKE_ITERATIONS);
        assert_river_smoke_parity(spot, &ours, &original);
    }
}

/// Legacy random smoke. Kept available, but not part of the deterministic first slice.
#[test]
#[ignore = "random mixed coverage is outside the deterministic river smoke slice"]
fn test_identity_5_mixed() {
    run_identity_test(&generate_configs(5, 42), 200);
}

/// Legacy random river coverage. Kept available, but not part of the deterministic first slice.
#[test]
fn test_identity_50_river() {
    run_identity_test(&generate_river_configs(50, 99), 200);
}

/// Legacy random river soak.
#[test]
#[ignore = "random soak coverage is outside the deterministic river smoke slice"]
fn test_identity_1000_river() {
    run_identity_test(&generate_river_configs(1000, 42), 200);
}

/// Soak test: 1000 configs across all street depths.
/// Takes hours to run -- use `cargo test --release -- --ignored` explicitly.
#[test]
#[ignore]
fn test_identity_1000_mixed() {
    run_identity_test(&generate_configs(1000, 42), 200);
}

fn run_identity_test(configs: &[TestConfig], iterations: u32) {
    let num_configs = configs.len();
    let mut failures = Vec::new();

    for (i, config) in configs.iter().enumerate() {
        let ours = run_ours(config, iterations);
        let original = run_original(config, iterations);

        let mut mismatches = Vec::new();

        if ours.exploitability != original.exploitability {
            mismatches.push(format!(
                "exploitability: ours={} orig={}",
                ours.exploitability, original.exploitability
            ));
        }

        if ours.root_strategy != original.root_strategy {
            let diff_count = ours
                .root_strategy
                .iter()
                .zip(&original.root_strategy)
                .filter(|(a, b)| a != b)
                .count();
            mismatches.push(format!(
                "strategy: {diff_count}/{} diffs",
                ours.root_strategy.len()
            ));
        }

        if ours.ev_oop != original.ev_oop {
            let diff_count = ours
                .ev_oop
                .iter()
                .zip(&original.ev_oop)
                .filter(|(a, b)| a != b)
                .count();
            mismatches.push(format!("ev_oop: {diff_count}/{} diffs", ours.ev_oop.len()));
        }

        if ours.ev_ip != original.ev_ip {
            let diff_count = ours
                .ev_ip
                .iter()
                .zip(&original.ev_ip)
                .filter(|(a, b)| a != b)
                .count();
            mismatches.push(format!("ev_ip: {diff_count}/{} diffs", ours.ev_ip.len()));
        }

        if ours.equity_oop != original.equity_oop {
            let diff_count = ours
                .equity_oop
                .iter()
                .zip(&original.equity_oop)
                .filter(|(a, b)| a != b)
                .count();
            mismatches.push(format!(
                "equity_oop: {diff_count}/{} diffs",
                ours.equity_oop.len()
            ));
        }

        if ours.equity_ip != original.equity_ip {
            let diff_count = ours
                .equity_ip
                .iter()
                .zip(&original.equity_ip)
                .filter(|(a, b)| a != b)
                .count();
            mismatches.push(format!(
                "equity_ip: {diff_count}/{} diffs",
                ours.equity_ip.len()
            ));
        }

        if !mismatches.is_empty() {
            failures.push((i, config_summary(config), mismatches));
        }

        if (i + 1) % 10 == 0 || i + 1 == num_configs {
            eprintln!(
                "Progress: {}/{} ({} failures so far)",
                i + 1,
                num_configs,
                failures.len()
            );
        }
    }

    if !failures.is_empty() {
        eprintln!("\n=== MISMATCHES ===");
        for (i, summary, mismatches) in &failures[..failures.len().min(10)] {
            eprintln!("Config #{i} ({summary}): {}", mismatches.join(", "));
        }
        if failures.len() > 10 {
            eprintln!("... and {} more", failures.len() - 10);
        }
        panic!("{} / {} configs mismatched", failures.len(), num_configs);
    }
}

/// Performance parity benchmark: 10 river configs x 1000 iterations.
/// Run with:
/// cargo test --manifest-path crates/range-solver-compare/Cargo.toml --release test_performance_parity -- --nocapture --test-threads=1
#[test]
#[ignore = "performance coverage is outside the deterministic river smoke slice"]
fn test_performance_parity() {
    let configs = generate_river_configs(10, 99);
    let iterations = 1000;

    let mut total_ours_ms = 0u128;
    let mut total_orig_ms = 0u128;

    for (i, config) in configs.iter().enumerate() {
        let t1 = Instant::now();
        let _ours = run_ours(config, iterations);
        let ours_ms = t1.elapsed().as_millis();

        let t2 = Instant::now();
        let _orig = run_original(config, iterations);
        let orig_ms = t2.elapsed().as_millis();

        total_ours_ms += ours_ms;
        total_orig_ms += orig_ms;

        let ratio = ours_ms as f64 / orig_ms.max(1) as f64;
        eprintln!("Config {i}: ours={ours_ms}ms orig={orig_ms}ms ratio={ratio:.2}x");
    }

    let overall_ratio = total_ours_ms as f64 / total_orig_ms.max(1) as f64;
    eprintln!("\nOverall: ours={total_ours_ms}ms orig={total_orig_ms}ms ratio={overall_ratio:.2}x");
    assert!(
        overall_ratio < 1.5,
        "Performance regression: {overall_ratio:.2}x slower overall"
    );
}

fn config_summary(config: &TestConfig) -> String {
    let street = match (config.turn, config.river) {
        (None, _) => "flop",
        (Some(_), None) => "turn",
        (Some(_), Some(_)) => "river",
    };
    format!(
        "{} {} pot={} stack={} bets={:?}",
        config.name, street, config.pot, config.stack, config.bet_pcts
    )
}

fn river_smoke_spots() -> Vec<TestConfig> {
    vec![
        river_spot(
            "ak_high_single_raised",
            "AA-QQ,AKs,AQs,KQs,AKo",
            "JJ-88,AQs-AJs,KQs,QJs,JTs,AQo,KQo",
            "As7d2c",
            "Kh",
            "3s",
            120,
            900,
            vec![0.50],
            vec![2.5],
        ),
        river_spot(
            "paired_double_broadway",
            "AA-99,AQs,KQs,QJs,AQo,KQo",
            "JJ-66,AQs-ATs,KQs-KTs,QJs,JTs,AQo,KQo,QJo",
            "QsQd7h",
            "2c",
            "2s",
            180,
            720,
            vec![0.33, 0.75],
            vec![2.5],
        ),
        river_spot(
            "four_flush_river",
            "AA-TT,AKs-ATs,KQs-KTs,QJs,JTs,AKo",
            "QQ-77,AQs-A2s,KQs-K9s,QJs-QTs,JTs,T9s,AQo",
            "Td8d4d",
            "2s",
            "Ad",
            240,
            600,
            vec![0.50, 1.00],
            vec![2.5],
        ),
        river_spot(
            "one_liner_straight",
            "TT-77,A9s-A7s,K9s,Q9s,JTs,T9s,98s,87s,A9o,K9o",
            "JJ-66,ATs-A5s,KTs-K8s,QTs-Q8s,JTs,T9s,98s,87s,76s",
            "9c8d6s",
            "5h",
            "7c",
            96,
            420,
            vec![0.33, 0.67],
            vec![2.5],
        ),
    ]
}

fn river_spot(
    name: &str,
    oop_range: &str,
    ip_range: &str,
    flop: &str,
    turn: &str,
    river: &str,
    pot: i32,
    stack: i32,
    bet_pcts: Vec<f64>,
    raise_pcts: Vec<f64>,
) -> TestConfig {
    TestConfig {
        name: name.to_string(),
        oop_range: oop_range.to_string(),
        ip_range: ip_range.to_string(),
        flop: range_solver::card::flop_from_str(flop).unwrap(),
        turn: Some(range_solver::card::card_from_str(turn).unwrap()),
        river: Some(range_solver::card::card_from_str(river).unwrap()),
        pot,
        stack,
        bet_pcts,
        raise_pcts,
    }
}

fn assert_river_smoke_parity(spot: &TestConfig, ours: &SolveResult, original: &SolveResult) {
    let mut failures = Vec::new();

    compare_private_hand_counts(spot, ours, original, &mut failures);
    compare_root_actions(spot, ours, original, &mut failures);
    compare_lengths(spot, ours, original, &mut failures);
    compare_numeric_scalar(
        spot,
        "exploitability",
        ours.exploitability,
        original.exploitability,
        EXPLOITABILITY_TOLERANCE,
        &mut failures,
    );

    compare_numeric_vector(
        spot,
        "root_strategy",
        &ours.root_strategy,
        &original.root_strategy,
        ROOT_STRATEGY_TOLERANCE,
        &mut failures,
    );
    compare_numeric_vector(
        spot,
        "ev_oop",
        &ours.ev_oop,
        &original.ev_oop,
        EV_TOLERANCE,
        &mut failures,
    );
    compare_numeric_vector(
        spot,
        "ev_ip",
        &ours.ev_ip,
        &original.ev_ip,
        EV_TOLERANCE,
        &mut failures,
    );
    compare_numeric_vector(
        spot,
        "equity_oop",
        &ours.equity_oop,
        &original.equity_oop,
        EQUITY_TOLERANCE,
        &mut failures,
    );
    compare_numeric_vector(
        spot,
        "equity_ip",
        &ours.equity_ip,
        &original.equity_ip,
        EQUITY_TOLERANCE,
        &mut failures,
    );

    assert!(
        failures.is_empty(),
        "river smoke parity failed:\n{}",
        failures.join("\n")
    );
}

fn compare_private_hand_counts(
    spot: &TestConfig,
    ours: &SolveResult,
    original: &SolveResult,
    failures: &mut Vec<String>,
) {
    for player in 0..2 {
        if ours.private_hand_counts[player] != original.private_hand_counts[player] {
            failures.push(format_mismatch(
                spot,
                "private_hand_count",
                &format!("player={player}"),
                format!(
                    "ours={} original={}",
                    ours.private_hand_counts[player], original.private_hand_counts[player]
                ),
            ));
        }
    }
}

fn compare_root_actions(
    spot: &TestConfig,
    ours: &SolveResult,
    original: &SolveResult,
    failures: &mut Vec<String>,
) {
    if ours.root_actions.len() != original.root_actions.len() {
        failures.push(format_mismatch(
            spot,
            "root_actions",
            "len",
            format!(
                "ours_len={} original_len={} ours={:?} original={:?}",
                ours.root_actions.len(),
                original.root_actions.len(),
                ours.root_actions,
                original.root_actions
            ),
        ));
        return;
    }

    for (index, (ours_action, original_action)) in ours
        .root_actions
        .iter()
        .zip(original.root_actions.iter())
        .enumerate()
    {
        if ours_action != original_action {
            failures.push(format_mismatch(
                spot,
                "root_actions",
                &index.to_string(),
                format!("ours={ours_action} original={original_action}"),
            ));
        }
    }
}

fn compare_lengths(
    spot: &TestConfig,
    ours: &SolveResult,
    original: &SolveResult,
    failures: &mut Vec<String>,
) {
    compare_len(
        spot,
        "root_strategy",
        ours.root_strategy.len(),
        original.root_strategy.len(),
        failures,
    );
    compare_len(
        spot,
        "ev_oop",
        ours.ev_oop.len(),
        original.ev_oop.len(),
        failures,
    );
    compare_len(
        spot,
        "ev_ip",
        ours.ev_ip.len(),
        original.ev_ip.len(),
        failures,
    );
    compare_len(
        spot,
        "equity_oop",
        ours.equity_oop.len(),
        original.equity_oop.len(),
        failures,
    );
    compare_len(
        spot,
        "equity_ip",
        ours.equity_ip.len(),
        original.equity_ip.len(),
        failures,
    );
}

fn compare_len(
    spot: &TestConfig,
    metric: &str,
    ours_len: usize,
    original_len: usize,
    failures: &mut Vec<String>,
) {
    if ours_len != original_len {
        failures.push(format_mismatch(
            spot,
            metric,
            "len",
            format!("ours_len={ours_len} original_len={original_len}"),
        ));
    }
}

fn compare_numeric_scalar(
    spot: &TestConfig,
    metric: &str,
    ours: f32,
    original: f32,
    tolerance: f32,
    failures: &mut Vec<String>,
) {
    let diff = (ours - original).abs();
    if diff > tolerance {
        failures.push(format_mismatch(
            spot,
            metric,
            "scalar",
            format!(
                "ours={ours:.8} original={original:.8} diff={diff:.8} tolerance={tolerance:.8}"
            ),
        ));
    }
}

fn compare_numeric_vector(
    spot: &TestConfig,
    metric: &str,
    ours: &[f32],
    original: &[f32],
    tolerance: f32,
    failures: &mut Vec<String>,
) {
    if ours.len() != original.len() {
        return;
    }

    let mut mismatch_count = 0usize;
    let mut max_diff = 0.0f32;
    let mut max_index = 0usize;
    let mut examples = Vec::new();

    for (index, (&ours_value, &original_value)) in ours.iter().zip(original.iter()).enumerate() {
        let diff = (ours_value - original_value).abs();
        if diff > max_diff {
            max_diff = diff;
            max_index = index;
        }
        if diff > tolerance {
            mismatch_count += 1;
            if examples.len() < 5 {
                examples.push(format!(
                    "index={index} ours={ours_value:.8} original={original_value:.8} diff={diff:.8}"
                ));
            }
        }
    }

    if mismatch_count > 0 {
        failures.push(format_mismatch(
            spot,
            metric,
            &max_index.to_string(),
            format!(
                "mismatches={mismatch_count}/{} tolerance={tolerance:.8} max_diff={max_diff:.8}; {}",
                ours.len(),
                examples.join("; ")
            ),
        ));
    }
}

fn format_mismatch(spot: &TestConfig, metric: &str, index: &str, detail: String) -> String {
    format!(
        "spot={} board={} oop_range={} ip_range={} pot={} stack={} metric={} index={} {}",
        spot.name,
        board_string(spot),
        spot.oop_range,
        spot.ip_range,
        spot.pot,
        spot.stack,
        metric,
        index,
        detail
    )
}

fn board_string(spot: &TestConfig) -> String {
    let mut cards = spot.flop.to_vec();
    if let Some(turn) = spot.turn {
        cards.push(turn);
    }
    if let Some(river) = spot.river {
        cards.push(river);
    }
    cards
        .into_iter()
        .map(|card| range_solver::card::card_to_string(card).unwrap())
        .collect::<Vec<_>>()
        .join("")
}
