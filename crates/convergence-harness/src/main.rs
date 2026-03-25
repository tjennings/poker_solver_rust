use clap::{Parser, Subcommand};
use convergence_harness::{baseline, comparison, config::ConvergenceConfig, harness, reporter};

#[derive(Parser)]
#[command(name = "convergence-harness")]
#[command(about = "Convergence validation harness for CFR algorithms")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Solve Flop Poker exactly and persist the golden baseline
    GenerateBaseline {
        /// Path to YAML config file
        #[arg(long)]
        config: String,

        /// Output directory for baseline artifacts
        #[arg(long, default_value = "baselines/convergence")]
        output_dir: String,
    },
    /// Compare a saved solver result against the baseline
    Compare {
        /// Path to baseline directory
        #[arg(long, default_value = "baselines/flop_poker_v1")]
        baseline_dir: String,

        /// Path to solver result directory
        #[arg(long)]
        result_dir: String,
    },
    /// Run a solver against the Flop Poker game and compare to baseline
    RunSolver {
        /// Path to YAML config file
        #[arg(long)]
        config: String,

        /// Path to baseline directory (auto-generated if missing)
        #[arg(long, default_value = "baselines/convergence")]
        baseline_dir: String,

        /// Output directory for results
        #[arg(long, default_value = "results/mccfr_run")]
        output_dir: String,
    },
}

/// Load a baseline and a solver result, compute comparison metrics.
///
/// Both `baseline_dir` and `result_dir` use the same `Baseline` file format.
fn run_compare(
    baseline_dir: &std::path::Path,
    result_dir: &std::path::Path,
) -> Result<reporter::ComparisonResult, Box<dyn std::error::Error>> {
    let baseline = baseline::Baseline::load(baseline_dir)?;
    let solver_result = baseline::Baseline::load(result_dir)?;

    let (per_node_l1, overall_l1) = comparison::l1_strategy_distance(
        &baseline.strategy,
        &solver_result.strategy,
        baseline.summary.num_combos_per_player,
    );

    let (per_node_ev_diff, overall_ev_diff) =
        comparison::combo_ev_diff(&baseline.combo_evs, &solver_result.combo_evs);

    Ok(reporter::ComparisonResult {
        solver_name: solver_result.summary.solver_name,
        total_iterations: solver_result.summary.total_iterations,
        total_time_ms: solver_result.summary.total_time_ms,
        final_exploitability: solver_result.summary.final_exploitability,
        baseline_exploitability: baseline.summary.final_exploitability,
        overall_l1_distance: overall_l1,
        overall_max_ev_diff: overall_ev_diff,
        convergence_curve: solver_result.convergence_curve,
        per_node_l1,
        per_node_ev_diff,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::GenerateBaseline { config, output_dir } => {
            let cfg = ConvergenceConfig::load(std::path::Path::new(&config))?;

            println!("Generating baseline for {} flop(s):", cfg.game.flops.len());
            for flop in &cfg.game.flops {
                println!("  - {}", flop);
            }

            let baseline = harness::generate_baseline_from_config(&cfg)?;

            let dir = std::path::Path::new(&output_dir);
            baseline.save(dir)?;

            println!("\n=== Baseline Summary ===");
            println!("Solver: {}", baseline.summary.solver_name);
            println!("Iterations: {}", baseline.summary.total_iterations);
            println!(
                "Final exploitability: {:.4e}",
                baseline.summary.final_exploitability
            );
            println!(
                "Time: {:.1}s",
                baseline.summary.total_time_ms as f64 / 1000.0
            );
            println!("Info sets captured: {}", baseline.summary.num_info_sets);
            println!(
                "Combos per player: {}",
                baseline.summary.num_combos_per_player
            );
            println!("Saved to: {}", output_dir);

            Ok(())
        }
        Commands::Compare {
            baseline_dir,
            result_dir,
        } => {
            let baseline_path = std::path::Path::new(&baseline_dir);
            let result_path = std::path::Path::new(&result_dir);

            println!("Loading baseline from: {}", baseline_dir);
            println!("Loading solver result from: {}", result_dir);

            let comparison = run_compare(baseline_path, result_path)?;

            println!("{}", comparison.human_summary());

            let output_dir = result_path.join("comparison");
            comparison.save(&output_dir)?;
            println!("Comparison artifacts saved to: {}", output_dir.display());

            Ok(())
        }
        Commands::RunSolver {
            config,
            baseline_dir,
            output_dir,
        } => {
            let cfg = ConvergenceConfig::load(std::path::Path::new(&config))?;

            println!("Starting MCCFR solver");
            println!("Flops: {:?}", cfg.game.flops);
            println!("Max iterations: {}", cfg.mccfr.iterations);
            println!("Checkpoints: {:?}", cfg.mccfr.checkpoints);

            // Ensure a baseline exists so h2h comparison is fair
            let baseline_path = std::path::Path::new(&baseline_dir);
            if !baseline_path.join("summary.json").exists() {
                println!(
                    "\nNo baseline at {}. Generating matching baseline...",
                    baseline_dir
                );
                let matching_baseline = harness::generate_baseline_from_config(&cfg)?;
                matching_baseline.save(baseline_path)?;
                println!("Baseline saved to: {}\n", baseline_dir);
            }

            // Load baseline for head-to-head EV computation during run
            let loaded_baseline = baseline::Baseline::load(baseline_path).ok();
            let result =
                harness::run_mccfr_solver_from_config(&cfg, loaded_baseline.as_ref())?;

            // Save results
            let result_dir = std::path::Path::new(&output_dir);
            result.save(result_dir)?;

            // Clean summary block
            println!("\n=== Result ===");
            println!("solver:     {}", result.summary.solver_name);
            println!("iterations: {}", result.summary.total_iterations);
            println!(
                "time:       {:.1}s",
                result.summary.total_time_ms as f64 / 1000.0
            );
            println!("mbb/hand:   {:.2}", result.summary.final_exploitability);
            println!("output:     {}", output_dir);

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use baseline::{Baseline, BaselineSummary, ConvergenceSample};
    use clap::Parser;
    use std::collections::BTreeMap;

    #[test]
    fn parse_generate_baseline_requires_config() {
        let result = Cli::try_parse_from(["convergence-harness", "generate-baseline"]);
        assert!(
            result.is_err(),
            "generate-baseline should require --config"
        );
    }

    #[test]
    fn parse_generate_baseline_with_config() {
        let cli = Cli::parse_from([
            "convergence-harness",
            "generate-baseline",
            "--config",
            "my_config.yaml",
        ]);
        match cli.command {
            Commands::GenerateBaseline { config, output_dir } => {
                assert_eq!(config, "my_config.yaml");
                assert_eq!(output_dir, "baselines/convergence");
            }
            _ => panic!("expected GenerateBaseline"),
        }
    }

    #[test]
    fn parse_generate_baseline_custom_output() {
        let cli = Cli::parse_from([
            "convergence-harness",
            "generate-baseline",
            "--config",
            "my_config.yaml",
            "--output-dir",
            "/tmp/my_baseline",
        ]);
        match cli.command {
            Commands::GenerateBaseline { config, output_dir } => {
                assert_eq!(config, "my_config.yaml");
                assert_eq!(output_dir, "/tmp/my_baseline");
            }
            _ => panic!("expected GenerateBaseline"),
        }
    }

    #[test]
    fn parse_compare_with_required_result_dir() {
        let cli = Cli::parse_from([
            "convergence-harness",
            "compare",
            "--result-dir",
            "/tmp/results",
        ]);
        match cli.command {
            Commands::Compare {
                baseline_dir,
                result_dir,
            } => {
                assert_eq!(baseline_dir, "baselines/flop_poker_v1");
                assert_eq!(result_dir, "/tmp/results");
            }
            _ => panic!("expected Compare"),
        }
    }

    #[test]
    fn parse_compare_custom_baseline() {
        let cli = Cli::parse_from([
            "convergence-harness",
            "compare",
            "--baseline-dir",
            "/tmp/custom_baseline",
            "--result-dir",
            "/tmp/results",
        ]);
        match cli.command {
            Commands::Compare {
                baseline_dir,
                result_dir,
            } => {
                assert_eq!(baseline_dir, "/tmp/custom_baseline");
                assert_eq!(result_dir, "/tmp/results");
            }
            _ => panic!("expected Compare"),
        }
    }

    #[test]
    fn compare_requires_result_dir() {
        let result = Cli::try_parse_from(["convergence-harness", "compare"]);
        assert!(result.is_err(), "compare should require --result-dir");
    }

    /// Helper to build a sample baseline for testing compare logic.
    fn sample_baseline(solver_name: &str, exploitability: f64) -> Baseline {
        Baseline {
            summary: BaselineSummary {
                solver_name: solver_name.into(),
                total_iterations: 100,
                final_exploitability: exploitability,
                total_time_ms: 5000,
                num_info_sets: 1,
                num_combos_per_player: 3,
                game_description: "Test game".into(),
            },
            convergence_curve: vec![
                ConvergenceSample {
                    iteration: 1,
                    exploitability: 1.0,
                    elapsed_ms: 0,
                },
                ConvergenceSample {
                    iteration: 100,
                    exploitability,
                    elapsed_ms: 5000,
                },
            ],
            strategy: {
                let mut m = BTreeMap::new();
                m.insert(0, vec![0.5, 0.3, 0.2, 0.5, 0.7, 0.8]);
                m
            },
            combo_evs: {
                let mut m = BTreeMap::new();
                m.insert(0, [vec![1.5, -0.5, 0.0], vec![-1.5, 0.5, 0.0]]);
                m
            },
        }
    }

    #[test]
    fn run_compare_loads_baselines_and_produces_report() {
        let dir = tempfile::TempDir::new().unwrap();
        let baseline_dir = dir.path().join("baseline");
        let result_dir = dir.path().join("result");

        let baseline = sample_baseline("Exhaustive DCFR", 0.001);
        baseline.save(&baseline_dir).unwrap();

        // Solver result is slightly different
        let mut solver_result = sample_baseline("Test Solver", 0.01);
        solver_result
            .strategy
            .insert(0, vec![0.6, 0.2, 0.2, 0.4, 0.8, 0.8]);
        solver_result
            .combo_evs
            .insert(0, [vec![1.0, -1.0, 0.5], vec![-1.0, 1.0, -0.5]]);
        solver_result.save(&result_dir).unwrap();

        let comparison = run_compare(&baseline_dir, &result_dir).unwrap();

        assert_eq!(comparison.solver_name, "Test Solver");
        assert!((comparison.baseline_exploitability - 0.001).abs() < 1e-9);
        assert!((comparison.final_exploitability - 0.01).abs() < 1e-9);
        assert!(
            comparison.overall_l1_distance > 0.0,
            "L1 distance should be positive"
        );
        assert!(
            comparison.overall_max_ev_diff > 0.0,
            "EV diff should be positive"
        );
    }

    #[test]
    fn run_compare_saves_artifacts() {
        let dir = tempfile::TempDir::new().unwrap();
        let baseline_dir = dir.path().join("baseline");
        let result_dir = dir.path().join("result");

        let baseline = sample_baseline("Exhaustive DCFR", 0.001);
        baseline.save(&baseline_dir).unwrap();

        let solver_result = sample_baseline("Test Solver", 0.01);
        solver_result.save(&result_dir).unwrap();

        let comparison = run_compare(&baseline_dir, &result_dir).unwrap();

        let output_dir = result_dir.join("comparison");
        comparison.save(&output_dir).unwrap();

        assert!(output_dir.join("summary.json").exists());
        assert!(output_dir.join("report.txt").exists());
        assert!(output_dir.join("strategy_distance.csv").exists());
        assert!(output_dir.join("combo_ev_diff.csv").exists());
    }

    #[test]
    fn run_compare_identical_baselines_have_zero_distance() {
        let dir = tempfile::TempDir::new().unwrap();
        let baseline_dir = dir.path().join("baseline");
        let result_dir = dir.path().join("result");

        let baseline = sample_baseline("Exhaustive DCFR", 0.001);
        baseline.save(&baseline_dir).unwrap();
        baseline.save(&result_dir).unwrap();

        let comparison = run_compare(&baseline_dir, &result_dir).unwrap();

        assert!(
            comparison.overall_l1_distance.abs() < 1e-9,
            "Identical strategies should have 0 L1 distance"
        );
        assert!(
            comparison.overall_max_ev_diff.abs() < 1e-9,
            "Identical EVs should have 0 diff"
        );
    }

    #[test]
    fn run_compare_missing_baseline_returns_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let result = run_compare(
            &dir.path().join("nonexistent"),
            &dir.path().join("also_nonexistent"),
        );
        assert!(result.is_err());
    }

    #[test]
    fn parse_run_solver_requires_config() {
        let result = Cli::try_parse_from(["convergence-harness", "run-solver"]);
        assert!(result.is_err(), "run-solver should require --config");
    }

    #[test]
    fn parse_run_solver_defaults() {
        let cli = Cli::parse_from([
            "convergence-harness",
            "run-solver",
            "--config",
            "test.yaml",
        ]);
        match cli.command {
            Commands::RunSolver {
                config,
                baseline_dir,
                output_dir,
            } => {
                assert_eq!(config, "test.yaml");
                assert_eq!(baseline_dir, "baselines/convergence");
                assert_eq!(output_dir, "results/mccfr_run");
            }
            _ => panic!("expected RunSolver"),
        }
    }

    #[test]
    fn parse_run_solver_custom() {
        let cli = Cli::parse_from([
            "convergence-harness",
            "run-solver",
            "--config",
            "custom.yaml",
            "--baseline-dir",
            "/tmp/baseline",
            "--output-dir",
            "/tmp/output",
        ]);
        match cli.command {
            Commands::RunSolver {
                config,
                baseline_dir,
                output_dir,
            } => {
                assert_eq!(config, "custom.yaml");
                assert_eq!(baseline_dir, "/tmp/baseline");
                assert_eq!(output_dir, "/tmp/output");
            }
            _ => panic!("expected RunSolver"),
        }
    }
}
