//! `validate-rollout` subcommand: compare exhaustive vs sampled rollout CFVs.
//!
//! Runs the rollout evaluator twice per traverser -- once with exhaustive
//! enumeration (no sampling at decision nodes) and once with the default
//! depth-gated sampling. Reports per-combo diff statistics in pot-fraction
//! and mbb/hand units. Multiple sampled runs are aggregated to separate
//! stochastic noise from systematic bias.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use poker_solver_tauri::postflop::build_rollout_evaluator;
use range_solver::range::Range;

use crate::bench_rollout::{find_first_flop_node, load_bundle, parse_board_cards};

/// Per-traverser comparison metrics between exhaustive and sampled CFVs.
#[derive(Debug, Clone)]
pub struct DiffMetrics {
    /// Maximum absolute difference across combos (pot-fraction).
    pub max_abs_diff: f64,
    /// Mean absolute difference across combos (pot-fraction).
    pub mean_abs_diff: f64,
    /// L2 norm of the difference vector: sqrt(sum(diff^2)).
    pub l2_diff: f64,
    /// Number of combos with nonzero weight.
    pub nonzero_combos: usize,
}

/// Aggregated metrics over multiple sampled runs.
#[derive(Debug, Clone)]
pub struct AggregatedMetrics {
    /// Mean of max_abs_diff across runs.
    pub mean_max_abs_diff: f64,
    /// Stddev of max_abs_diff across runs.
    pub stddev_max_abs_diff: f64,
    /// Mean of mean_abs_diff across runs.
    pub mean_mean_abs_diff: f64,
    /// Stddev of mean_abs_diff across runs.
    pub stddev_mean_abs_diff: f64,
    /// Mean of l2_diff across runs.
    pub mean_l2_diff: f64,
    /// Stddev of l2_diff across runs.
    pub stddev_l2_diff: f64,
    /// Nonzero combos (same across runs).
    pub nonzero_combos: usize,
}

/// Compute per-combo diff metrics between two CFV vectors.
///
/// Only considers combos where the weight (from the traverser's range) is
/// nonzero. Returns `None` if there are no nonzero-weight combos.
pub fn compute_diff_metrics(
    exhaustive: &[f64],
    sampled: &[f64],
    weights: &[f64],
) -> Option<DiffMetrics> {
    debug_assert_eq!(exhaustive.len(), sampled.len());
    debug_assert_eq!(exhaustive.len(), weights.len());

    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut count = 0usize;

    for i in 0..exhaustive.len() {
        if weights[i] <= 0.0 {
            continue;
        }
        let diff = (sampled[i] - exhaustive[i]).abs();
        max_abs = max_abs.max(diff);
        sum_abs += diff;
        sum_sq += diff * diff;
        count += 1;
    }

    if count == 0 {
        return None;
    }

    Some(DiffMetrics {
        max_abs_diff: max_abs,
        mean_abs_diff: sum_abs / count as f64,
        l2_diff: sum_sq.sqrt(),
        nonzero_combos: count,
    })
}

/// Aggregate `DiffMetrics` from multiple runs into mean and stddev.
///
/// Returns `None` if `runs` is empty.
pub fn aggregate_metrics(runs: &[DiffMetrics]) -> Option<AggregatedMetrics> {
    if runs.is_empty() {
        return None;
    }
    let n = runs.len() as f64;

    let mean_max = runs.iter().map(|r| r.max_abs_diff).sum::<f64>() / n;
    let mean_mean = runs.iter().map(|r| r.mean_abs_diff).sum::<f64>() / n;
    let mean_l2 = runs.iter().map(|r| r.l2_diff).sum::<f64>() / n;

    let stddev = |values: &[f64], mean: f64| -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let variance = values
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / (values.len() - 1) as f64;
        variance.sqrt()
    };

    let max_vals: Vec<f64> = runs.iter().map(|r| r.max_abs_diff).collect();
    let mean_vals: Vec<f64> = runs.iter().map(|r| r.mean_abs_diff).collect();
    let l2_vals: Vec<f64> = runs.iter().map(|r| r.l2_diff).collect();

    Some(AggregatedMetrics {
        mean_max_abs_diff: mean_max,
        stddev_max_abs_diff: stddev(&max_vals, mean_max),
        mean_mean_abs_diff: mean_mean,
        stddev_mean_abs_diff: stddev(&mean_vals, mean_mean),
        mean_l2_diff: mean_l2,
        stddev_l2_diff: stddev(&l2_vals, mean_l2),
        nonzero_combos: runs[0].nonzero_combos,
    })
}

/// Convert a pot-fraction value to approximate mbb/hand.
///
/// Uses the task's convention: `mbb = pot_fraction * 100`. This maps
/// 0.02 pot-fraction to 2 mbb/hand, matching the pass criterion.
pub fn pot_fraction_to_mbb(pot_fraction: f64) -> f64 {
    pot_fraction * 100.0
}

/// Format and print the validation results for one traverser.
pub fn format_traverser_report(
    name: &str,
    agg: &AggregatedMetrics,
    num_runs: usize,
    pass_threshold: f64,
) -> (String, bool) {
    let max_mbb = pot_fraction_to_mbb(agg.mean_max_abs_diff);
    let mean_mbb = pot_fraction_to_mbb(agg.mean_mean_abs_diff);
    let threshold_mbb = pot_fraction_to_mbb(pass_threshold);
    let passed = agg.mean_max_abs_diff < pass_threshold;
    let result_str = if passed { "PASS" } else { "FAIL" };

    let report = format!(
        "{name} traverser (sampled vs exhaustive over {num_runs} runs):\n  \
         nonzero combos:  {}\n  \
         max_abs_diff:    {:.6} pot-fraction  ({:.2} mbb/hand)\n  \
         mean_abs_diff:   {:.6} pot-fraction  ({:.2} mbb/hand)\n  \
         L2 diff:         {:.6}  (stddev over {num_runs} runs: {:.6})\n  \
         pass criterion:  max_abs_diff < {pass_threshold:.4} ({threshold_mbb:.1} mbb/hand)\n  \
         result: {result_str}",
        agg.nonzero_combos,
        agg.mean_max_abs_diff,
        max_mbb,
        agg.mean_mean_abs_diff,
        mean_mbb,
        agg.mean_l2_diff,
        agg.stddev_l2_diff,
    );

    (report, passed)
}

/// Run one rollout evaluation pass for both traversers, returning the
/// per-combo chip values for OOP (traverser 0) and IP (traverser 1).
fn run_rollout_pass(
    bench_ctx: &poker_solver_tauri::postflop::RolloutBenchContext,
    boundary_pot: f64,
    boundary_invested: [f64; 2],
) -> (Vec<f64>, Vec<f64>) {
    let oop_values = bench_ctx.evaluator.rollout_chip_values_with_state(
        &bench_ctx.combos,
        &bench_ctx.board,
        &bench_ctx.oop_range,
        &bench_ctx.ip_range,
        0,
        boundary_pot,
        boundary_invested,
    );
    let ip_values = bench_ctx.evaluator.rollout_chip_values_with_state(
        &bench_ctx.combos,
        &bench_ctx.board,
        &bench_ctx.oop_range,
        &bench_ctx.ip_range,
        1,
        boundary_pot,
        boundary_invested,
    );
    (oop_values, ip_values)
}

/// Run the validate-rollout command.
///
/// Loads the bundle, builds evaluators for exhaustive and sampled modes,
/// runs multiple sampled passes, and reports diff metrics per traverser.
pub fn run(
    bundle_dir: &Path,
    board_str: &str,
    pot: u32,
    stacks: u32,
    num_runs: usize,
    pass_threshold: f64,
) -> Result<(), String> {
    let board_cards = parse_board_cards(board_str)?;
    if board_cards.len() != 3 {
        return Err(format!(
            "Validate requires a flop (3 cards), got {} cards",
            board_cards.len()
        ));
    }

    let (ctx, tree, _decision_map) = load_bundle(bundle_dir)?;

    let flop_node = find_first_flop_node(&tree)
        .ok_or("No flop decision node found in abstract tree")?;
    eprintln!("[validate] abstract flop node index: {flop_node}");

    let pot_f = f64::from(pot);
    let starting_stack = f64::from(stacks) + pot_f / 2.0;

    let oop_range: Range = "TT+,AQs+,AKo"
        .parse()
        .map_err(|e| format!("Bad OOP range: {e}"))?;
    let ip_range: Range = "JJ+,AKs,AQs"
        .parse()
        .map_err(|e| format!("Bad IP range: {e}"))?;

    eprintln!(
        "[validate] board={board_str} pot={pot} stacks={stacks} runs={num_runs}"
    );

    let counter = Arc::new(AtomicU64::new(0));

    // Build exhaustive evaluator
    let mut exhaustive_ctx = build_rollout_evaluator(
        &board_cards,
        &ctx,
        flop_node,
        pot_f,
        starting_stack,
        Some(Arc::clone(&counter)),
        Some(&oop_range),
        Some(&ip_range),
    );
    exhaustive_ctx.evaluator.exhaustive = true;

    // Build sampled evaluator (default: exhaustive=false)
    let sampled_ctx = build_rollout_evaluator(
        &board_cards,
        &ctx,
        flop_node,
        pot_f,
        starting_stack,
        Some(Arc::clone(&counter)),
        Some(&oop_range),
        Some(&ip_range),
    );

    let boundary_pot = pot_f;
    let boundary_invested = [pot_f / 2.0, pot_f / 2.0];

    eprintln!(
        "[validate] {} combos, running exhaustive baseline...",
        exhaustive_ctx.combos.len()
    );

    // Run exhaustive once (deterministic at decision nodes)
    let (ex_oop, ex_ip) =
        run_rollout_pass(&exhaustive_ctx, boundary_pot, boundary_invested);

    eprintln!(
        "[validate] exhaustive done, running {num_runs} sampled passes..."
    );

    // Run sampled N times
    let mut oop_runs: Vec<DiffMetrics> = Vec::with_capacity(num_runs);
    let mut ip_runs: Vec<DiffMetrics> = Vec::with_capacity(num_runs);

    for run_idx in 0..num_runs {
        let (s_oop, s_ip) =
            run_rollout_pass(&sampled_ctx, boundary_pot, boundary_invested);

        // Convert chip values to pot-fraction before computing metrics
        let ex_oop_pf: Vec<f64> = ex_oop.iter().map(|&v| v / pot_f).collect();
        let s_oop_pf: Vec<f64> = s_oop.iter().map(|&v| v / pot_f).collect();
        let ex_ip_pf: Vec<f64> = ex_ip.iter().map(|&v| v / pot_f).collect();
        let s_ip_pf: Vec<f64> = s_ip.iter().map(|&v| v / pot_f).collect();

        if let Some(m) =
            compute_diff_metrics(&ex_oop_pf, &s_oop_pf, &exhaustive_ctx.oop_range)
        {
            oop_runs.push(m);
        }
        if let Some(m) =
            compute_diff_metrics(&ex_ip_pf, &s_ip_pf, &exhaustive_ctx.ip_range)
        {
            ip_runs.push(m);
        }
        eprintln!("  run {}/{num_runs} complete", run_idx + 1);
    }

    // Aggregate and report
    let oop_agg = aggregate_metrics(&oop_runs)
        .ok_or("No OOP metrics computed (all combos zero weight?)")?;
    let ip_agg = aggregate_metrics(&ip_runs)
        .ok_or("No IP metrics computed (all combos zero weight?)")?;

    let (oop_report, oop_pass) =
        format_traverser_report("OOP", &oop_agg, num_runs, pass_threshold);
    let (ip_report, ip_pass) =
        format_traverser_report("IP", &ip_agg, num_runs, pass_threshold);

    println!("{oop_report}");
    println!();
    println!("{ip_report}");
    println!();

    let overall = oop_pass && ip_pass;
    let overall_str = if overall { "PASS" } else { "FAIL" };
    println!("Overall: {overall_str}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- compute_diff_metrics tests ----

    #[test]
    fn diff_metrics_identical_vectors_all_zero() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let w = vec![1.0, 1.0, 1.0];
        let m = compute_diff_metrics(&a, &b, &w).unwrap();
        assert_eq!(m.max_abs_diff, 0.0);
        assert_eq!(m.mean_abs_diff, 0.0);
        assert_eq!(m.l2_diff, 0.0);
        assert_eq!(m.nonzero_combos, 3);
    }

    #[test]
    fn diff_metrics_known_diffs() {
        let exhaustive = vec![1.0, 2.0, 3.0, 4.0];
        let sampled = vec![1.1, 2.0, 2.8, 4.5];
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let m = compute_diff_metrics(&exhaustive, &sampled, &weights).unwrap();
        // diffs: 0.1, 0.0, 0.2, 0.5
        assert!((m.max_abs_diff - 0.5).abs() < 1e-10);
        assert!((m.mean_abs_diff - 0.2).abs() < 1e-10);
        // L2 = sqrt(0.01 + 0 + 0.04 + 0.25) = sqrt(0.30)
        assert!((m.l2_diff - 0.30_f64.sqrt()).abs() < 1e-10);
        assert_eq!(m.nonzero_combos, 4);
    }

    #[test]
    fn diff_metrics_skips_zero_weight_combos() {
        let exhaustive = vec![1.0, 2.0, 3.0];
        let sampled = vec![1.1, 999.0, 2.8];
        let weights = vec![1.0, 0.0, 1.0];
        let m = compute_diff_metrics(&exhaustive, &sampled, &weights).unwrap();
        // Only combos 0 and 2: diffs 0.1, 0.2
        assert!((m.max_abs_diff - 0.2).abs() < 1e-10);
        assert!((m.mean_abs_diff - 0.15).abs() < 1e-10);
        assert_eq!(m.nonzero_combos, 2);
    }

    #[test]
    fn diff_metrics_all_zero_weight_returns_none() {
        let a = vec![1.0, 2.0];
        let b = vec![1.1, 2.2];
        let w = vec![0.0, 0.0];
        assert!(compute_diff_metrics(&a, &b, &w).is_none());
    }

    #[test]
    fn diff_metrics_single_combo() {
        let a = vec![5.0];
        let b = vec![5.3];
        let w = vec![1.0];
        let m = compute_diff_metrics(&a, &b, &w).unwrap();
        assert!((m.max_abs_diff - 0.3).abs() < 1e-10);
        assert!((m.mean_abs_diff - 0.3).abs() < 1e-10);
        assert!((m.l2_diff - 0.3).abs() < 1e-10);
        assert_eq!(m.nonzero_combos, 1);
    }

    #[test]
    fn diff_metrics_negative_diffs_use_abs() {
        let a = vec![2.0, 3.0];
        let b = vec![1.5, 3.5];
        let w = vec![1.0, 1.0];
        let m = compute_diff_metrics(&a, &b, &w).unwrap();
        assert!((m.max_abs_diff - 0.5).abs() < 1e-10);
        assert!((m.mean_abs_diff - 0.5).abs() < 1e-10);
    }

    // ---- aggregate_metrics tests ----

    #[test]
    fn aggregate_empty_returns_none() {
        assert!(aggregate_metrics(&[]).is_none());
    }

    #[test]
    fn aggregate_single_run_zero_stddev() {
        let run = DiffMetrics {
            max_abs_diff: 0.1,
            mean_abs_diff: 0.05,
            l2_diff: 0.2,
            nonzero_combos: 10,
        };
        let agg = aggregate_metrics(&[run]).unwrap();
        assert!((agg.mean_max_abs_diff - 0.1).abs() < 1e-10);
        assert_eq!(agg.stddev_max_abs_diff, 0.0);
        assert!((agg.mean_mean_abs_diff - 0.05).abs() < 1e-10);
        assert_eq!(agg.stddev_mean_abs_diff, 0.0);
        assert!((agg.mean_l2_diff - 0.2).abs() < 1e-10);
        assert_eq!(agg.stddev_l2_diff, 0.0);
        assert_eq!(agg.nonzero_combos, 10);
    }

    #[test]
    fn aggregate_multiple_runs_computes_mean_and_stddev() {
        let runs = vec![
            DiffMetrics {
                max_abs_diff: 0.1,
                mean_abs_diff: 0.01,
                l2_diff: 0.05,
                nonzero_combos: 5,
            },
            DiffMetrics {
                max_abs_diff: 0.2,
                mean_abs_diff: 0.02,
                l2_diff: 0.10,
                nonzero_combos: 5,
            },
            DiffMetrics {
                max_abs_diff: 0.3,
                mean_abs_diff: 0.03,
                l2_diff: 0.15,
                nonzero_combos: 5,
            },
        ];
        let agg = aggregate_metrics(&runs).unwrap();
        assert!((agg.mean_max_abs_diff - 0.2).abs() < 1e-10);
        // stddev = sqrt(((0.1-0.2)^2+(0.2-0.2)^2+(0.3-0.2)^2)/2) = 0.1
        assert!((agg.stddev_max_abs_diff - 0.1).abs() < 1e-10);
        assert!((agg.mean_mean_abs_diff - 0.02).abs() < 1e-10);
        assert_eq!(agg.nonzero_combos, 5);
    }

    // ---- pot_fraction_to_mbb tests ----

    #[test]
    fn pot_fraction_to_mbb_at_standard_pot() {
        let mbb = pot_fraction_to_mbb(0.02);
        assert!((mbb - 2.0).abs() < 1e-10);
    }

    #[test]
    fn pot_fraction_to_mbb_zero() {
        assert_eq!(pot_fraction_to_mbb(0.0), 0.0);
    }

    #[test]
    fn pot_fraction_to_mbb_one() {
        let mbb = pot_fraction_to_mbb(1.0);
        assert!((mbb - 100.0).abs() < 1e-10);
    }

    // ---- format_traverser_report tests ----

    #[test]
    fn format_report_pass() {
        let agg = AggregatedMetrics {
            mean_max_abs_diff: 0.01,
            stddev_max_abs_diff: 0.002,
            mean_mean_abs_diff: 0.005,
            stddev_mean_abs_diff: 0.001,
            mean_l2_diff: 0.03,
            stddev_l2_diff: 0.005,
            nonzero_combos: 100,
        };
        let (report, passed) =
            format_traverser_report("OOP", &agg, 5, 0.02);
        assert!(passed);
        assert!(
            report.contains("PASS"),
            "report should say PASS: {report}"
        );
        assert!(report.contains("OOP traverser"));
    }

    #[test]
    fn format_report_fail() {
        let agg = AggregatedMetrics {
            mean_max_abs_diff: 0.05,
            stddev_max_abs_diff: 0.01,
            mean_mean_abs_diff: 0.02,
            stddev_mean_abs_diff: 0.005,
            mean_l2_diff: 0.1,
            stddev_l2_diff: 0.01,
            nonzero_combos: 100,
        };
        let (report, passed) =
            format_traverser_report("IP", &agg, 5, 0.02);
        assert!(!passed);
        assert!(
            report.contains("FAIL"),
            "report should say FAIL: {report}"
        );
        assert!(report.contains("IP traverser"));
    }

    #[test]
    fn format_report_contains_all_metrics() {
        let agg = AggregatedMetrics {
            mean_max_abs_diff: 0.015,
            stddev_max_abs_diff: 0.003,
            mean_mean_abs_diff: 0.008,
            stddev_mean_abs_diff: 0.002,
            mean_l2_diff: 0.04,
            stddev_l2_diff: 0.006,
            nonzero_combos: 50,
        };
        let (report, _) =
            format_traverser_report("OOP", &agg, 3, 0.02);
        assert!(
            report.contains("max_abs_diff"),
            "missing max_abs_diff: {report}"
        );
        assert!(
            report.contains("mean_abs_diff"),
            "missing mean_abs_diff: {report}"
        );
        assert!(
            report.contains("L2 diff"),
            "missing L2 diff: {report}"
        );
        assert!(
            report.contains("pass criterion"),
            "missing criterion: {report}"
        );
        assert!(
            report.contains("mbb/hand"),
            "missing mbb/hand: {report}"
        );
        assert!(
            report.contains("50"),
            "missing nonzero combos: {report}"
        );
    }
}
