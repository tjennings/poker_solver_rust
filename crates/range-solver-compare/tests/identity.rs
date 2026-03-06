use std::time::Instant;

use range_solver_compare::*;

/// Fast smoke test: 5 random configs across all street depths.
#[test]
fn test_identity_5_mixed() {
    run_identity_test(&generate_configs(5, 42), 200);
}

/// Medium test: 50 river-only configs (fast to solve).
#[test]
fn test_identity_50_river() {
    run_identity_test(&generate_river_configs(50, 99), 200);
}

/// 1000 river-only configs — practical identity test (~2 min release).
#[test]
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
            mismatches.push(format!(
                "ev_oop: {diff_count}/{} diffs",
                ours.ev_oop.len()
            ));
        }

        if ours.ev_ip != original.ev_ip {
            let diff_count = ours
                .ev_ip
                .iter()
                .zip(&original.ev_ip)
                .filter(|(a, b)| a != b)
                .count();
            mismatches.push(format!(
                "ev_ip: {diff_count}/{} diffs",
                ours.ev_ip.len()
            ));
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
/// Run with: cargo test -p range-solver-compare --release test_performance_parity -- --nocapture --test-threads=1
#[test]
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
        eprintln!(
            "Config {i}: ours={ours_ms}ms orig={orig_ms}ms ratio={ratio:.2}x"
        );
    }

    let overall_ratio = total_ours_ms as f64 / total_orig_ms.max(1) as f64;
    eprintln!(
        "\nOverall: ours={total_ours_ms}ms orig={total_orig_ms}ms ratio={overall_ratio:.2}x"
    );
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
        "{} pot={} stack={} bets={:?}",
        street, config.pot, config.stack, config.bet_pcts
    )
}
