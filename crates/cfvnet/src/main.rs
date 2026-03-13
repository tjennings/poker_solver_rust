use clap::{Parser, Subcommand};
use std::path::PathBuf;

fn default_backend() -> String {
    if cfg!(feature = "cuda") {
        "cuda".into()
    } else {
        "wgpu".into()
    }
}

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
        /// Max samples per output file; splits into multiple files if total exceeds this
        #[arg(long)]
        per_file: Option<u64>,
    },
    /// Train the CFVnet model
    Train {
        /// Path to YAML config file
        #[arg(short, long)]
        config: PathBuf,
        /// Path to training data (file or directory of files)
        #[arg(short, long)]
        data: PathBuf,
        /// Output directory for checkpoints
        #[arg(short, long)]
        output: PathBuf,
        /// Backend: wgpu (Metal/Vulkan), ndarray (CPU), or cuda (NVIDIA, requires --features cuda).
        /// Defaults to cuda when built with --features cuda, otherwise wgpu.
        #[arg(long, default_value_t = default_backend())]
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
    },
    /// Compare turn model against CfvSubgameSolver + RiverNetEvaluator
    CompareNet {
        /// Path to turn model directory
        #[arg(short, long)]
        model: PathBuf,
        /// Path to river model directory
        #[arg(long)]
        river_model: PathBuf,
        /// Number of spots to compare
        #[arg(long, default_value = "100")]
        num_spots: usize,
    },
    /// Compare turn model against CfvSubgameSolver + ExactRiverEvaluator
    CompareExact {
        /// Path to turn model directory
        #[arg(short, long)]
        model: PathBuf,
        /// Number of spots to compare
        #[arg(long, default_value = "20")]
        num_spots: usize,
    },
}

fn append_random_suffix(path: &std::path::Path) -> PathBuf {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let suffix: String = (0..5)
        .map(|_| {
            let idx = rng.gen_range(0..52);
            if idx < 26 {
                (b'a' + idx) as char
            } else {
                (b'A' + idx - 26) as char
            }
        })
        .collect();
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let ext = path.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
    let new_name = format!("{stem}_{suffix}{ext}");
    path.with_file_name(new_name)
}

/// Resolve a model path from user input.
///
/// Accepts either:
/// - A directory: appends `/model` (burn adds `.mpk.gz`)
/// - A `.mpk.gz` file: strips `.mpk.gz` (burn re-adds it)
fn resolve_model_path(path: &std::path::Path) -> PathBuf {
    if path.is_dir() {
        path.join("model")
    } else {
        // Strip .mpk.gz extension(s) since burn's load_file adds them back.
        let s = path.to_string_lossy();
        let stripped = s.strip_suffix(".mpk.gz").unwrap_or(&s);
        PathBuf::from(stripped.to_string())
    }
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
            per_file,
        } => {
            ensure_parent_dir(&output);
            cmd_generate(config, output, num_samples, threads, per_file);
        }
        Commands::Train {
            config,
            data,
            output,
            backend,
        } => {
            ensure_parent_dir(&output);
            cmd_train(config, data, output, &backend);
        }
        Commands::Evaluate { model, data } => cmd_evaluate(model, data),
        Commands::Compare {
            model,
            num_spots,
            threads,
        } => cmd_compare(model, num_spots, threads),
        Commands::CompareNet {
            model,
            river_model,
            num_spots,
        } => cmd_compare_net(model, river_model, num_spots),
        Commands::CompareExact {
            model,
            num_spots,
        } => cmd_compare_exact(model, num_spots),
    }
}

fn cmd_train(config_path: PathBuf, data: PathBuf, output: PathBuf, backend: &str) {
    let yaml = std::fs::read_to_string(&config_path).unwrap_or_else(|e| {
        eprintln!("failed to read config {}: {e}", config_path.display());
        std::process::exit(1);
    });

    let cfg: cfvnet::config::CfvnetConfig = serde_yaml::from_str(&yaml).unwrap_or_else(|e| {
        eprintln!("failed to parse config: {e}");
        std::process::exit(1);
    });

    let board_cards = cfvnet::config::board_cards_for_street(&cfg.datagen.street);

    let train_config = cfvnet::model::training::TrainConfig {
        hidden_layers: cfg.training.hidden_layers,
        hidden_size: cfg.training.hidden_size,
        batch_size: cfg.training.batch_size,
        epochs: cfg.training.epochs,
        learning_rate: cfg.training.learning_rate,
        lr_min: cfg.training.lr_min,
        huber_delta: cfg.training.huber_delta,

        validation_split: cfg.training.validation_split,
        checkpoint_every_n_epochs: cfg.training.checkpoint_every_n_epochs,
        gpu_chunk_size: cfg.training.gpu_chunk_size,
        epochs_per_chunk: cfg.training.epochs_per_chunk,
        prefetch_chunks: cfg.training.prefetch_chunks,
    };

    // Ensure output directory exists before writing config.
    std::fs::create_dir_all(&output).unwrap_or_else(|e| {
        eprintln!("failed to create output directory {}: {e}", output.display());
        std::process::exit(1);
    });

    // Write config early so it's available alongside checkpoints.
    let config_out = output.join("config.yaml");
    let config_yaml = serde_yaml::to_string(&cfg).unwrap_or_else(|e| {
        eprintln!("failed to serialize config: {e}");
        std::process::exit(1);
    });
    std::fs::write(&config_out, &config_yaml).unwrap_or_else(|e| {
        eprintln!("failed to write config to {}: {e}", config_out.display());
        std::process::exit(1);
    });
    println!("Config saved to {}", config_out.display());

    match backend {
        "wgpu" => {
            use burn::backend::{Autodiff, wgpu::{Wgpu, WgpuDevice}};
            type B = Autodiff<Wgpu>;
            let device = WgpuDevice::DefaultDevice;
            println!("Using wgpu backend (Metal GPU on macOS)");
            let result = cfvnet::model::training::train::<B>(&device, &data, board_cards, &train_config, Some(&output));
            println!("Training complete. Final loss: {}", result.final_train_loss);
        }
        "ndarray" => {
            use burn::backend::{Autodiff, NdArray};
            type B = Autodiff<NdArray>;
            let device = Default::default();
            println!("Using ndarray backend (CPU)");
            let result = cfvnet::model::training::train::<B>(&device, &data, board_cards, &train_config, Some(&output));
            println!("Training complete. Final loss: {}", result.final_train_loss);
        }
        #[cfg(feature = "cuda")]
        "cuda" => {
            use burn::backend::{Autodiff, CudaJit, cuda_jit::CudaDevice};
            type B = Autodiff<CudaJit<f32>>;
            let device = CudaDevice::default();
            println!("Using CUDA backend (NVIDIA GPU)");
            let result = cfvnet::model::training::train::<B>(&device, &data, board_cards, &train_config, Some(&output));
            println!("Training complete. Final loss: {}", result.final_train_loss);
        }
        #[cfg(not(feature = "cuda"))]
        "cuda" => {
            eprintln!("CUDA backend not enabled. Rebuild with: cargo build -p cfvnet --features cuda --release");
            std::process::exit(1);
        }
        other => {
            eprintln!("unknown backend '{other}', expected 'ndarray', 'wgpu', or 'cuda'");
            std::process::exit(1);
        }
    }

}

fn cmd_generate(
    config_path: PathBuf,
    output: PathBuf,
    num_samples: Option<u64>,
    threads: Option<usize>,
    per_file: Option<u64>,
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

    let total = cfg.datagen.num_samples;
    let chunk_size = per_file.unwrap_or(total);
    let num_files = (total + chunk_size - 1) / chunk_size;

    let street = cfg.datagen.street.as_str();
    if num_files > 1 {
        println!(
            "Generating {total} {street} training samples across {num_files} files ({chunk_size} per file)..."
        );
    } else {
        println!(
            "Generating {total} {street} training samples...",
        );
    }

    let base_seed = cfg.datagen.seed;
    let mut remaining = total;
    let mut file_idx = 0u64;
    while remaining > 0 {
        let this_chunk = remaining.min(chunk_size);
        remaining -= this_chunk;

        cfg.datagen.num_samples = this_chunk;
        // Advance seed per file so each file gets different situations.
        cfg.datagen.seed = base_seed.wrapping_add(file_idx * 1_000_000);

        let file_output = append_random_suffix(&output);

        if num_files > 1 {
            println!(
                "  File {}/{num_files}: {this_chunk} samples → {}",
                file_idx + 1,
                file_output.display()
            );
        } else {
            println!("  → {}", file_output.display());
        }

        let result = match street {
            "turn" => cfvnet::datagen::turn_generate::generate_turn_training_data(&cfg, &file_output),
            _ => cfvnet::datagen::generate::generate_training_data(&cfg, &file_output),
        };

        if let Err(e) = result {
            eprintln!("data generation failed: {e}");
            std::process::exit(1);
        }

        file_idx += 1;
    }

    println!("Done.");
}

fn cmd_evaluate(model_dir: PathBuf, data_path: PathBuf) {
    use burn::backend::NdArray;
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
    use burn::tensor::{Tensor, TensorData};
    use cfvnet::config::board_cards_for_street;
    use cfvnet::eval::metrics::compute_prediction_metrics;
    use cfvnet::model::dataset::CfvDataset;
    use cfvnet::model::network::{CfvNet, input_size};

    let cfg = load_model_config(&model_dir);
    let board_cards = board_cards_for_street(&cfg.datagen.street);
    let in_size = input_size(board_cards);

    type B = NdArray;
    let device = Default::default();
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
    let model_path = resolve_model_path(&model_dir);

    let model = CfvNet::<B>::new(&device, cfg.training.hidden_layers, cfg.training.hidden_size, in_size)
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
        let item = dataset.get(i).unwrap();

        let input = Tensor::<B, 2>::from_data(
            TensorData::new(item.input.clone(), [1, in_size]),
            &device,
        );
        let range_oop = Tensor::<B, 2>::from_data(
            TensorData::new(item.oop_range.clone(), [1, 1326]),
            &device,
        );
        let range_ip = Tensor::<B, 2>::from_data(
            TensorData::new(item.ip_range.clone(), [1, 1326]),
            &device,
        );
        let pred = model.forward(input, range_oop, range_ip);
        let pred_vec: Vec<f32> = pred.into_data().to_vec::<f32>().unwrap();

        let mask: Vec<bool> = item.mask.iter().map(|&v| v > 0.5).collect();
        let pot = item.pot;

        // Split dual output (2652) into OOP (first 1326) and IP (last 1326)
        // to match the 1326-element mask.
        let pred_oop = &pred_vec[..1326];
        let pred_ip = &pred_vec[1326..];

        let metrics_oop = compute_prediction_metrics(pred_oop, &item.oop_target, &mask, pot);
        let metrics_ip = compute_prediction_metrics(pred_ip, &item.ip_target, &mask, pot);

        // Average OOP and IP metrics for overall evaluation.
        total_mae += (metrics_oop.mae + metrics_ip.mae) / 2.0;
        total_max_error += f64::max(metrics_oop.max_error, metrics_ip.max_error);
        total_mbb += (metrics_oop.mbb_error + metrics_ip.mbb_error) / 2.0;
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
) {
    use burn::backend::NdArray;
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
    use burn::tensor::{Tensor, TensorData};
    use cfvnet::config::board_cards_for_street;
    use cfvnet::eval::compare::run_comparison;
    use cfvnet::model::dataset::encode_situation_for_inference;
    use cfvnet::model::network::{CfvNet, NUM_COMBOS, input_size};

    let cfg = load_model_config(&model_dir);
    let board_cards = board_cards_for_street(&cfg.datagen.street);
    let in_size = input_size(board_cards);

    type B = NdArray;
    let device = Default::default();
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
    let model_path = resolve_model_path(&model_dir);

    let model = CfvNet::<B>::new(&device, cfg.training.hidden_layers, cfg.training.hidden_size, in_size)
        .load_file(&model_path, &recorder, &device)
        .unwrap_or_else(|e| {
            eprintln!("failed to load model from {}: {e}", model_path.display());
            std::process::exit(1);
        });

    if let Some(t) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .ok();
    }

    println!("Comparing {num_spots} spots against exact solver...");

    let summary = run_comparison(&cfg.game, &cfg.datagen, num_spots, cfg.datagen.seed, |sit, _solve_result| {
        let input_data = encode_situation_for_inference(sit);
        let input = Tensor::<B, 2>::from_data(
            TensorData::new(input_data, [1, in_size]),
            &device,
        );
        let range_oop = Tensor::<B, 2>::from_data(
            TensorData::new(sit.ranges[0].to_vec(), [1, NUM_COMBOS]),
            &device,
        );
        let range_ip = Tensor::<B, 2>::from_data(
            TensorData::new(sit.ranges[1].to_vec(), [1, NUM_COMBOS]),
            &device,
        );
        let pred = model.forward(input, range_oop, range_ip);
        let all_cfvs = pred.into_data().to_vec::<f32>().unwrap();
        // Return only OOP CFVs (first 1326) for comparison against solver OOP EVs.
        all_cfvs[..1326].to_vec()
    })
    .unwrap_or_else(|e| {
        eprintln!("comparison failed: {e}");
        std::process::exit(1);
    });

    print_summary(&summary);
}

fn cmd_compare_net(
    model_dir: PathBuf,
    river_model_dir: PathBuf,
    num_spots: usize,
) {
    use cfvnet::eval::compare_turn::run_turn_comparison_net;

    let cfg = load_model_config(&model_dir);
    let model_path = resolve_model_path(&model_dir);
    let river_path = resolve_model_path(&river_model_dir);

    println!("Comparing {num_spots} turn spots against CfvSubgameSolver + RiverNetEvaluator...");

    let summary = run_turn_comparison_net(&cfg, &model_path, &river_path, num_spots, cfg.datagen.seed)
        .unwrap_or_else(|e| {
            eprintln!("comparison failed: {e}");
            std::process::exit(1);
        });

    print_summary(&summary);
}

fn cmd_compare_exact(model_dir: PathBuf, num_spots: usize) {
    use cfvnet::eval::compare_turn::run_turn_comparison_exact;

    let cfg = load_model_config(&model_dir);
    let model_path = resolve_model_path(&model_dir);

    println!("Comparing {num_spots} turn spots against CfvSubgameSolver + ExactRiverEvaluator...");

    let summary = run_turn_comparison_exact(&cfg, &model_path, num_spots, cfg.datagen.seed).unwrap_or_else(|e| {
        eprintln!("comparison failed: {e}");
        std::process::exit(1);
    });

    print_summary(&summary);
}

fn load_model_config(model_path: &std::path::Path) -> cfvnet::config::CfvnetConfig {
    let dir = if model_path.is_dir() {
        model_path.to_path_buf()
    } else {
        model_path.parent().unwrap_or(model_path).to_path_buf()
    };
    let config_path = dir.join("config.yaml");
    let yaml = std::fs::read_to_string(&config_path).unwrap_or_else(|e| {
        eprintln!(
            "failed to read model config {}: {e}\n\
             hint: this model was saved before config embedding was added — \
             re-train or manually place a config.yaml in the model directory",
            config_path.display()
        );
        std::process::exit(1);
    });
    serde_yaml::from_str(&yaml).unwrap_or_else(|e| {
        eprintln!("failed to parse model config {}: {e}", config_path.display());
        std::process::exit(1);
    })
}

fn print_summary(summary: &cfvnet::eval::compare::ComparisonSummary) {
    println!("Results ({} spots):", summary.num_spots);
    println!("  Mean MAE:       {:.6}", summary.mean_mae);
    println!("  Mean Max Error: {:.6}", summary.mean_max_error);
    println!("  Mean mBB:       {:.2}", summary.mean_mbb);
    println!("  Worst MAE:      {:.6}", summary.worst_mae);
    println!("  Worst mBB:      {:.2}", summary.worst_mbb);

    if summary.spots.is_empty() {
        return;
    }

    let mut sorted: Vec<_> = summary.spots.iter().collect();
    sorted.sort_by(|a, b| a.mbb.total_cmp(&b.mbb));

    let top_n = sorted.len().min(3);

    println!("\nBest {} spots (by mBB):", top_n);
    for (i, spot) in sorted.iter().take(top_n).enumerate() {
        println!("  {}. {}  Pot: {:<5} Stack: {:<5} MAE: {:.6}  mBB: {:.2}",
            i + 1, format_board(&spot.board, spot.board_size),
            spot.pot, spot.effective_stack, spot.mae, spot.mbb);
    }

    println!("\nWorst {} spots (by mBB):", top_n);
    for (i, spot) in sorted.iter().rev().take(top_n).enumerate() {
        println!("  {}. {}  Pot: {:<5} Stack: {:<5} MAE: {:.6}  mBB: {:.2}",
            i + 1, format_board(&spot.board, spot.board_size),
            spot.pot, spot.effective_stack, spot.mae, spot.mbb);
    }
}

fn format_board(board: &[u8; 5], board_size: usize) -> String {
    use range_solver::card::card_to_string;
    let cards: Vec<String> = board[..board_size]
        .iter()
        .map(|&c| card_to_string(c).unwrap_or_else(|_| "??".into()))
        .collect();
    format!("Board: {}", cards.join(" "))
}

