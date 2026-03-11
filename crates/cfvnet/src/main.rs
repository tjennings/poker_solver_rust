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

fn ensure_parent_dir(path: &std::path::Path) {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).unwrap_or_else(|e| {
                eprintln!("failed to create directory {}: {e}", parent.display());
                std::process::exit(1);
            });
        }
    }
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            config,
            output,
            num_samples,
            threads,
        } => {
            ensure_parent_dir(&output);
            cmd_generate(config, output, num_samples, threads);
        }
        Commands::Train {
            config,
            data,
            output,
            backend: _,
        } => {
            ensure_parent_dir(&output);
            cmd_train(config, data, output);
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

fn cmd_train(config_path: PathBuf, data: PathBuf, output: PathBuf) {
    let yaml = std::fs::read_to_string(&config_path).unwrap_or_else(|e| {
        eprintln!("failed to read config {}: {e}", config_path.display());
        std::process::exit(1);
    });

    let cfg: cfvnet::config::CfvnetConfig = serde_yaml::from_str(&yaml).unwrap_or_else(|e| {
        eprintln!("failed to parse config: {e}");
        std::process::exit(1);
    });

    let dataset = cfvnet::model::dataset::CfvDataset::from_file(&data).unwrap_or_else(|e| {
        eprintln!("failed to load dataset: {e}");
        std::process::exit(1);
    });
    println!("Loaded {} training records", dataset.len());

    let train_config = cfvnet::model::training::TrainConfig {
        hidden_layers: cfg.training.hidden_layers,
        hidden_size: cfg.training.hidden_size,
        batch_size: cfg.training.batch_size,
        epochs: cfg.training.epochs,
        learning_rate: cfg.training.learning_rate,
        lr_min: cfg.training.lr_min,
        huber_delta: cfg.training.huber_delta,
        aux_loss_weight: cfg.training.aux_loss_weight,
        validation_split: cfg.training.validation_split,
        checkpoint_every_n_batches: cfg.training.checkpoint_every_n_batches,
    };

    use burn::backend::{Autodiff, NdArray};
    type B = Autodiff<NdArray>;
    let device = Default::default();

    let result = cfvnet::model::training::train::<B>(&device, &dataset, &train_config, Some(&output));
    println!("Training complete. Final loss: {}", result.final_train_loss);
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
