use clap::{Parser, Subcommand};

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
        /// Output directory for baseline artifacts
        #[arg(long, default_value = "baselines/flop_poker_v1")]
        output_dir: String,

        /// Maximum iterations for the solver
        #[arg(long, default_value_t = 1000)]
        iterations: u32,

        /// Target exploitability (fraction of pot)
        #[arg(long, default_value_t = 0.001)]
        target_exploitability: f32,
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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::GenerateBaseline {
            output_dir: _,
            iterations: _,
            target_exploitability: _,
        } => {
            println!("generate-baseline: not yet implemented");
            Ok(())
        }
        Commands::Compare {
            baseline_dir: _,
            result_dir: _,
        } => {
            println!("compare: not yet implemented");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parse_generate_baseline_defaults() {
        let cli = Cli::parse_from(["convergence-harness", "generate-baseline"]);
        match cli.command {
            Commands::GenerateBaseline {
                output_dir,
                iterations,
                target_exploitability,
            } => {
                assert_eq!(output_dir, "baselines/flop_poker_v1");
                assert_eq!(iterations, 1000);
                assert!((target_exploitability - 0.001).abs() < 1e-6);
            }
            _ => panic!("expected GenerateBaseline"),
        }
    }

    #[test]
    fn parse_generate_baseline_custom_args() {
        let cli = Cli::parse_from([
            "convergence-harness",
            "generate-baseline",
            "--output-dir",
            "/tmp/my_baseline",
            "--iterations",
            "5000",
            "--target-exploitability",
            "0.01",
        ]);
        match cli.command {
            Commands::GenerateBaseline {
                output_dir,
                iterations,
                target_exploitability,
            } => {
                assert_eq!(output_dir, "/tmp/my_baseline");
                assert_eq!(iterations, 5000);
                assert!((target_exploitability - 0.01).abs() < 1e-6);
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
        // --result-dir has no default, so omitting it should fail parsing
        let result = Cli::try_parse_from(["convergence-harness", "compare"]);
        assert!(result.is_err(), "compare should require --result-dir");
    }
}
