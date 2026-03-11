use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "cfvnet", about = "Deep Counterfactual Value Network toolkit")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate training data by solving random river subgames
    Generate {
        /// Path to YAML config file
        #[arg(short, long)]
        config: PathBuf,
        /// Output path for binary training data
        #[arg(short, long)]
        output: PathBuf,
        /// Override num_samples from config
        #[arg(long)]
        num_samples: Option<u64>,
        /// Override thread count from config
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Train the CFVnet model
    Train {
        /// Path to YAML config file
        #[arg(short, long)]
        config: PathBuf,
        /// Path to binary training data
        #[arg(short, long)]
        data: PathBuf,
        /// Output directory for checkpoints
        #[arg(short, long)]
        output: PathBuf,
        /// Backend: ndarray (default) or wgpu
        #[arg(long, default_value = "ndarray")]
        backend: String,
    },
    /// Evaluate model on held-out validation data
    Evaluate {
        /// Path to saved model directory
        #[arg(short, long)]
        model: PathBuf,
        /// Path to binary evaluation data
        #[arg(short, long)]
        data: PathBuf,
    },
    /// Compare model predictions against exact solves
    Compare {
        /// Path to saved model directory
        #[arg(short, long)]
        model: PathBuf,
        /// Number of random spots to compare
        #[arg(long, default_value = "100")]
        num_spots: usize,
        /// Thread count for parallel solves
        #[arg(long)]
        threads: Option<usize>,
        /// Optional YAML config for game parameters
        #[arg(short, long)]
        config: Option<PathBuf>,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            config,
            output,
            num_samples,
            threads,
        } => cmd_generate(config, output, num_samples, threads),
        Commands::Train {
            config: _,
            data: _,
            output: _,
            backend: _,
        } => {
            eprintln!("Train not yet implemented (pending model/training.rs)");
            std::process::exit(1);
        }
        Commands::Evaluate {
            model: _,
            data: _,
        } => {
            eprintln!("Evaluate not yet implemented (needs model loading)");
            std::process::exit(1);
        }
        Commands::Compare {
            model: _,
            num_spots: _,
            threads: _,
            config: _,
        } => {
            eprintln!("Compare not yet implemented (needs model loading)");
            std::process::exit(1);
        }
    }
}

fn cmd_generate(
    config_path: PathBuf,
    output: PathBuf,
    num_samples: Option<u64>,
    threads: Option<usize>,
) {
    let yaml = std::fs::read_to_string(&config_path).unwrap_or_else(|e| {
        eprintln!("failed to read config {}: {e}", config_path.display());
        std::process::exit(1);
    });

    let mut cfg: cfvnet::config::CfvnetConfig = serde_yaml::from_str(&yaml).unwrap_or_else(|e| {
        eprintln!("failed to parse config: {e}");
        std::process::exit(1);
    });

    if let Err(e) = cfg.game.validate() {
        eprintln!("invalid game config: {e}");
        std::process::exit(1);
    }

    if let Some(n) = num_samples {
        cfg.datagen.num_samples = n;
    }
    if let Some(t) = threads {
        cfg.datagen.threads = t;
    }

    println!(
        "Generating {} training samples to {}...",
        cfg.datagen.num_samples,
        output.display()
    );

    if let Err(e) = cfvnet::datagen::generate::generate_training_data(&cfg, &output) {
        eprintln!("data generation failed: {e}");
        std::process::exit(1);
    }

    println!("Done.");
}
