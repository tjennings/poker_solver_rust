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
        Commands::Evaluate { model, data } => cmd_evaluate(model, data),
        Commands::Compare {
            model,
            num_spots,
            threads,
            config,
        } => cmd_compare(model, num_spots, threads, config),
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

    let board_cards = cfvnet::config::board_cards_for_street(&cfg.datagen.street);
    let dataset = cfvnet::model::dataset::CfvDataset::from_file(&data, board_cards).unwrap_or_else(|e| {
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
        checkpoint_every_n_epochs: cfg.training.checkpoint_every_n_epochs,
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

fn cmd_evaluate(model_dir: PathBuf, data_path: PathBuf) {
    use burn::backend::NdArray;
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
    use burn::tensor::{Tensor, TensorData};
    use cfvnet::eval::metrics::compute_prediction_metrics;
    use cfvnet::model::dataset::CfvDataset;
    use cfvnet::model::network::{CfvNet, input_size};

    // TODO: support turn/flop models via --street flag
    let board_cards = 5;
    let in_size = input_size(board_cards);

    type B = NdArray;
    let device = Default::default();
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
    let model_path = model_dir.join("model");

    let model = CfvNet::<B>::new(&device, 7, 500, in_size)
        .load_file(&model_path, &recorder, &device)
        .unwrap_or_else(|e| {
            eprintln!("failed to load model from {}: {e}", model_path.display());
            std::process::exit(1);
        });

    let dataset = CfvDataset::from_file(&data_path, board_cards).unwrap_or_else(|e| {
        eprintln!("failed to load dataset: {e}");
        std::process::exit(1);
    });

    println!("Evaluating {} records...", dataset.len());

    let mut total_mae = 0.0_f64;
    let mut total_max_error = 0.0_f64;
    let mut total_mbb = 0.0_f64;
    let mut count = 0_u64;

    for i in 0..dataset.len() {
        // INVARIANT: index is in bounds because we iterate up to dataset.len().
        let item = dataset.get(i).unwrap();

        let input = Tensor::<B, 2>::from_data(
            TensorData::new(item.input.clone(), [1, in_size]),
            &device,
        );
        let pred = model.forward(input);
        let pred_vec: Vec<f32> = pred.into_data().to_vec::<f32>().unwrap();

        let mask: Vec<bool> = item.mask.iter().map(|&v| v > 0.5).collect();
        let pot = item.input[2657] * 400.0;

        let metrics = compute_prediction_metrics(&pred_vec, &item.target, &mask, pot);
        total_mae += metrics.mae;
        total_max_error += metrics.max_error;
        total_mbb += metrics.mbb_error;
        count += 1;
    }

    if count == 0 {
        println!("No records to evaluate.");
        return;
    }

    let n = count as f64;
    println!("Results ({count} records):");
    println!("  MAE:       {:.6}", total_mae / n);
    println!("  Max Error: {:.6}", total_max_error / n);
    println!("  mBB:       {:.2}", total_mbb / n);
}

fn cmd_compare(
    model_dir: PathBuf,
    num_spots: usize,
    threads: Option<usize>,
    config_path: Option<PathBuf>,
) {
    use burn::backend::NdArray;
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
    use burn::tensor::{Tensor, TensorData};
    use cfvnet::config::GameConfig;
    use cfvnet::eval::compare::run_comparison;
    use cfvnet::model::dataset::encode_situation_for_inference;
    use cfvnet::model::network::{CfvNet, input_size};

    // TODO: support turn/flop models via --street flag
    let board_cards = 5;
    let in_size = input_size(board_cards);

    type B = NdArray;
    let device = Default::default();
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
    let model_path = model_dir.join("model");

    let model = CfvNet::<B>::new(&device, 7, 500, in_size)
        .load_file(&model_path, &recorder, &device)
        .unwrap_or_else(|e| {
            eprintln!("failed to load model from {}: {e}", model_path.display());
            std::process::exit(1);
        });

    let game_config = match config_path {
        Some(path) => {
            let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                eprintln!("failed to read config {}: {e}", path.display());
                std::process::exit(1);
            });
            let cfg: cfvnet::config::CfvnetConfig =
                serde_yaml::from_str(&yaml).unwrap_or_else(|e| {
                    eprintln!("failed to parse config: {e}");
                    std::process::exit(1);
                });
            cfg.game
        }
        None => GameConfig::default(),
    };

    if let Some(t) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .ok(); // Ignore error if pool already initialized.
    }

    println!("Comparing {num_spots} spots against exact solver...");

    let summary = run_comparison(&game_config, num_spots, 42, |sit, _solve_result| {
        let input_data = encode_situation_for_inference(sit, 0);
        let input = Tensor::<B, 2>::from_data(
            TensorData::new(input_data, [1, in_size]),
            &device,
        );
        let pred = model.forward(input);
        pred.into_data().to_vec::<f32>().unwrap()
    })
    .unwrap_or_else(|e| {
        eprintln!("comparison failed: {e}");
        std::process::exit(1);
    });

    println!("Results ({} spots):", summary.num_spots);
    println!("  Mean MAE:       {:.6}", summary.mean_mae);
    println!("  Mean Max Error: {:.6}", summary.mean_max_error);
    println!("  Mean mBB:       {:.2}", summary.mean_mbb);
    println!("  Worst MAE:      {:.6}", summary.worst_mae);
    println!("  Worst mBB:      {:.2}", summary.worst_mbb);
}

