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
        /// Output path for binary training data. Overrides config datagen.turn_output.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Override num_samples from config
        #[arg(long)]
        num_samples: Option<u64>,
        /// Override thread count from config
        #[arg(long)]
        threads: Option<usize>,
        /// Max samples per output file; splits into multiple files if total exceeds this
        #[arg(long)]
        per_file: Option<u64>,
        /// Backend for inference: ndarray (CPU, default) or cuda (GPU, requires --features cuda).
        #[arg(long, default_value = "ndarray")]
        backend: String,
        /// Output path for river training data (exact mode only)
        #[arg(long)]
        river_output: Option<PathBuf>,
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
    /// Compare turn model against PostFlopGame + RiverNetEvaluator
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
    /// Compare turn model against PostFlopGame + exact river evaluation
    CompareExact {
        /// Path to turn model directory
        #[arg(short, long)]
        model: PathBuf,
        /// Number of spots to compare
        #[arg(long, default_value = "20")]
        num_spots: usize,
    },
    /// Print distribution histograms for generated training data
    DatagenEval {
        /// Path to binary training data (file or directory)
        #[arg(short, long)]
        data: PathBuf,
    },
    /// Benchmark sample + solve throughput (no output).
    BenchSolve {
        /// Path to the YAML configuration file.
        #[arg(long)]
        config: PathBuf,
        /// Number of situations to solve (default: 1000).
        #[arg(long, default_value = "1000")]
        num_samples: u64,
        /// Number of threads (default: 1 for profiling).
        #[arg(long, default_value = "1")]
        threads: usize,
    },
    /// Train the BoundaryNet model (normalized EV output for range-solver integration)
    TrainBoundary {
        #[arg(short, long)]
        config: PathBuf,
        #[arg(short, long)]
        data: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = default_backend())]
        backend: String,
    },
    /// Evaluate BoundaryNet on held-out data
    EvalBoundary {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(short, long)]
        data: PathBuf,
    },
    /// Compare BoundaryNet boundary CFVs against ground truth on reference positions
    CompareBoundary {
        /// Path to trained boundary model directory
        #[arg(short, long)]
        model: PathBuf,
        /// Path to binary test data (same format as training data)
        #[arg(short, long)]
        data: PathBuf,
        /// Max positions to compare
        #[arg(short, long, default_value = "100")]
        num_positions: usize,
    },
    /// Inspect a preflop_ranges.bin file: prints per-path label, frequency,
    /// and range-shape stats (density, entropy, top-10 mass).
    #[command(name = "inspect-preflop-ranges")]
    InspectPreflopRanges {
        /// Path to the preflop_ranges.bin file
        #[arg(short, long)]
        path: PathBuf,
    },
    /// Filter a preflop_ranges.bin to a single path by label.
    #[command(name = "filter-preflop-ranges")]
    FilterPreflopRanges {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        /// Exact path label to keep (e.g. "2bb/10bb/22bb/call")
        #[arg(short, long)]
        label: String,
    },
    /// Diagnose per-combo CFV deltas between ONNX model and ground truth
    #[command(name = "diagnose-boundary")]
    DiagnoseBoundary {
        /// Path to ONNX model file
        #[arg(short, long)]
        model: PathBuf,
        /// Path to binary training/validation data file
        #[arg(short, long)]
        data: PathBuf,
        /// Max records to process (default: all)
        #[arg(long)]
        num_records: Option<usize>,
        /// Optional CSV dump path (record_idx, combo_idx, category, net_cfv, truth_cfv, delta)
        #[arg(long)]
        csv_out: Option<PathBuf>,
    },
    /// Precompute average turn entry ranges from a blueprint strategy, bucketed by SPR.
    #[command(name = "precompute-ranges")]
    PrecomputeRanges {
        /// Path to blueprint bundle directory
        #[arg(short, long)]
        blueprint: PathBuf,
        /// Output file for cached ranges
        #[arg(short, long)]
        output: PathBuf,
        /// Number of deals to sample per SPR bucket (default 10000)
        #[arg(long, default_value = "10000")]
        samples_per_bucket: usize,
        /// SPR bucket boundaries (comma-separated, e.g. "0,1,3,8,50")
        #[arg(long, default_value = "0,0.5,1.5,4,8,50")]
        spr_boundaries: String,
        /// Initial stack size in chips (default 200)
        #[arg(long, default_value = "200")]
        initial_stack: i32,
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
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).unwrap_or_else(|e| {
            eprintln!("failed to create directory {}: {e}", parent.display());
            std::process::exit(1);
        });
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
            backend,
            river_output,
        } => {
            if let Some(ref o) = output {
                ensure_parent_dir(o);
            }
            cmd_generate(config, output, num_samples, threads, per_file, &backend, river_output);
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
        Commands::TrainBoundary {
            config,
            data,
            output,
            backend,
        } => {
            ensure_parent_dir(&output);
            cmd_train_boundary(config, data, output, &backend);
        }
        Commands::EvalBoundary { model, data } => cmd_eval_boundary(model, data),
        Commands::CompareBoundary { model, data, num_positions } => {
            cmd_compare_boundary(&model, &data, num_positions);
        }
        Commands::DatagenEval { data } => cmd_datagen_eval(data),
        Commands::BenchSolve { config, num_samples, threads } => {
            cmd_bench_solve(config, num_samples, threads);
        }
        Commands::DiagnoseBoundary { model, data, num_records, csv_out } => {
            cmd_diagnose_boundary(model, data, num_records, csv_out);
        }
        Commands::InspectPreflopRanges { path } => cmd_inspect_preflop_ranges(path),
        Commands::FilterPreflopRanges { input, output, label } => {
            cmd_filter_preflop_ranges(input, output, label);
        }
        Commands::PrecomputeRanges {
            blueprint,
            output,
            samples_per_bucket,
            spr_boundaries,
            initial_stack,
        } => {
            ensure_parent_dir(&output);
            cmd_precompute_ranges(blueprint, output, samples_per_bucket, &spr_boundaries, initial_stack);
        }
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

    // Create output directory and write config before training starts,
    // so the config is preserved even if training is interrupted.
    std::fs::create_dir_all(&output).unwrap_or_else(|e| {
        eprintln!("failed to create output directory {}: {e}", output.display());
        std::process::exit(1);
    });
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
        shuffle_buffer_size: cfg.training.shuffle_buffer_size,
        prefetch_depth: cfg.training.prefetch_depth,
        encoder_threads: cfg.training.encoder_threads,
        gpu_prefetch: cfg.training.gpu_prefetch,
        grad_clip_norm: cfg.training.grad_clip_norm,
    };

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

fn cmd_train_boundary(config_path: PathBuf, data: PathBuf, output: PathBuf, backend: &str) {
    let yaml = std::fs::read_to_string(&config_path).unwrap_or_else(|e| {
        eprintln!("failed to read config {}: {e}", config_path.display());
        std::process::exit(1);
    });

    let cfg: cfvnet::config::CfvnetConfig = serde_yaml::from_str(&yaml).unwrap_or_else(|e| {
        eprintln!("failed to parse config: {e}");
        std::process::exit(1);
    });

    let board_cards = cfvnet::config::board_cards_for_street(&cfg.datagen.street);

    // Create output directory and write config before training starts.
    std::fs::create_dir_all(&output).unwrap_or_else(|e| {
        eprintln!("failed to create output directory {}: {e}", output.display());
        std::process::exit(1);
    });
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

    let train_config = cfvnet::model::boundary_training::BoundaryTrainConfig {
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
        shuffle_buffer_size: cfg.training.shuffle_buffer_size,
        prefetch_depth: cfg.training.prefetch_depth,
        encoder_threads: cfg.training.encoder_threads,
        gpu_prefetch: cfg.training.gpu_prefetch,
        grad_clip_norm: cfg.training.grad_clip_norm,
    };

    match backend {
        "wgpu" => {
            use burn::backend::{Autodiff, wgpu::{Wgpu, WgpuDevice}};
            type B = Autodiff<Wgpu>;
            let device = WgpuDevice::DefaultDevice;
            println!("Using wgpu backend (Metal GPU on macOS)");
            let result = cfvnet::model::boundary_training::train_boundary::<B>(&device, &data, board_cards, &train_config, Some(&output));
            println!("Training complete. Final loss: {}", result.final_train_loss);
        }
        "ndarray" => {
            use burn::backend::{Autodiff, NdArray};
            type B = Autodiff<NdArray>;
            let device = Default::default();
            println!("Using ndarray backend (CPU)");
            let result = cfvnet::model::boundary_training::train_boundary::<B>(&device, &data, board_cards, &train_config, Some(&output));
            println!("Training complete. Final loss: {}", result.final_train_loss);
        }
        #[cfg(feature = "cuda")]
        "cuda" => {
            use burn::backend::{Autodiff, CudaJit, cuda_jit::CudaDevice};
            type B = Autodiff<CudaJit<f32>>;
            let device = CudaDevice::default();
            println!("Using CUDA backend (NVIDIA GPU)");
            let result = cfvnet::model::boundary_training::train_boundary::<B>(&device, &data, board_cards, &train_config, Some(&output));
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
    output: Option<PathBuf>,
    num_samples: Option<u64>,
    threads: Option<usize>,
    per_file: Option<u64>,
    _backend: &str,
    river_output: Option<PathBuf>,
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
    if let Some(rp) = river_output {
        cfg.datagen.river_output = Some(rp.to_string_lossy().into_owned());
    }

    // Resolve output path: CLI -o > config turn_output/river_output (by street).
    let output = output
        .or_else(|| {
            let path = match cfg.datagen.street.as_str() {
                "river" => cfg.datagen.river_output.as_ref(),
                _ => cfg.datagen.turn_output.as_ref(),
            };
            path.map(PathBuf::from)
        })
        .unwrap_or_else(|| {
            let field = match cfg.datagen.street.as_str() {
                "river" => "datagen.river_output",
                _ => "datagen.turn_output",
            };
            eprintln!("No output path specified. Use -o or set {field} in config.");
            std::process::exit(1);
        });
    ensure_parent_dir(&output);

    let total = cfg.datagen.num_samples;
    let chunk_size = per_file.or(cfg.datagen.per_file).unwrap_or(total);
    let num_files = total.div_ceil(chunk_size);

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

    let base_seed = cfvnet::config::resolve_seed(cfg.datagen.seed);
    println!("Using seed: {base_seed}");
    let mut remaining = total;
    let mut file_idx = 0u64;
    while remaining > 0 {
        let this_chunk = remaining.min(chunk_size);
        remaining -= this_chunk;

        cfg.datagen.num_samples = this_chunk;
        // Advance seed per file so each file gets different situations.
        cfg.datagen.seed = Some(base_seed.wrapping_add(file_idx * 1_000_000));

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
            "turn" | "river" => cfvnet::datagen::domain::pipeline::DomainPipeline::run(&cfg, &file_output),
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

fn cmd_bench_solve(config_path: PathBuf, num_samples: u64, threads: usize) {
    let yaml = std::fs::read_to_string(&config_path).unwrap_or_else(|e| {
        eprintln!("failed to read config {}: {e}", config_path.display());
        std::process::exit(1);
    });
    let cfg: cfvnet::config::CfvnetConfig = serde_yaml::from_str(&yaml).unwrap_or_else(|e| {
        eprintln!("failed to parse config: {e}");
        std::process::exit(1);
    });
    if let Err(e) = cfg.game.validate() {
        eprintln!("invalid game config: {e}");
        std::process::exit(1);
    }

    let bet_str = cfg.game.bet_sizes.join_flat(",");
    let bet_sizes = range_solver::bet_size::BetSizeOptions::try_from((bet_str.as_str(), ""))
        .unwrap_or_else(|e| {
            eprintln!("invalid bet sizes: {e}");
            std::process::exit(1);
        });
    let solve_config = cfvnet::datagen::solver::SolveConfig {
        bet_sizes,
        solver_iterations: cfg.datagen.solver_iterations,
        target_exploitability: cfg.datagen.target_exploitability,
        add_allin_threshold: cfg.game.add_allin_threshold,
        force_allin_threshold: cfg.game.force_allin_threshold,
    };

    let board_size = cfg.game.board_size;
    let seed = cfvnet::config::resolve_seed(cfg.datagen.seed);

    eprintln!("Benchmarking sample+solve: {num_samples} situations, {threads} thread(s)");

    let start = std::time::Instant::now();

    if threads > 1 {
        use rayon::prelude::*;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            (0..num_samples).into_par_iter().for_each(|i| {
                use rand::SeedableRng;
                let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed.wrapping_add(i));
                let situation = cfvnet::datagen::sampler::sample_situation(
                    &cfg.datagen,
                    cfg.game.initial_stack,
                    board_size,
                    &mut rng,
                );
                if situation.effective_stack > 0 {
                    let _ = cfvnet::datagen::solver::solve_situation(&situation, &solve_config);
                }
            });
        });
    } else {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        for _ in 0..num_samples {
            let situation = cfvnet::datagen::sampler::sample_situation(
                &cfg.datagen,
                cfg.game.initial_stack,
                board_size,
                &mut rng,
            );
            if situation.effective_stack > 0 {
                let _ = cfvnet::datagen::solver::solve_situation(&situation, &solve_config);
            }
        }
    }

    let elapsed = start.elapsed();
    let per_solve = elapsed / num_samples as u32;
    let solves_per_sec = num_samples as f64 / elapsed.as_secs_f64();
    eprintln!(
        "Solved {} situations in {:.2?} ({:.1} solves/sec, {:.2?}/solve)",
        num_samples, elapsed, solves_per_sec, per_solve,
    );
}

fn cmd_inspect_preflop_ranges(path: PathBuf) {
    use cfvnet::datagen::precompute_ranges::PrecomputedRanges;

    let ranges = PrecomputedRanges::load(&path).unwrap_or_else(|e| {
        eprintln!("failed to load {}: {e}", path.display());
        std::process::exit(1);
    });

    let total_freq: f64 = ranges.paths.iter().map(|p| p.frequency).sum();
    println!("Loaded {} preflop paths from {}", ranges.paths.len(), path.display());
    println!("Total frequency (sum): {total_freq:.6}\n");

    // Column headers
    println!(
        "{:<28} {:>10} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "label", "freq", "oop_nz", "ip_nz",
        "oop_ent", "ip_ent", "oop_top10", "ip_top10",
    );
    println!("{}", "-".repeat(100));

    for p in &ranges.paths {
        let (oop_ent, oop_top10) = range_shape(&p.oop_range);
        let (ip_ent, ip_top10) = range_shape(&p.ip_range);
        let pct = if total_freq > 0.0 { p.frequency / total_freq * 100.0 } else { 0.0 };
        println!(
            "{:<28} {:>10.4} {:>8} {:>8} {:>10.3} {:>10.3} {:>10.4} {:>10.4}",
            p.label,
            pct,
            p.oop_nonzero,
            p.ip_nonzero,
            oop_ent,
            ip_ent,
            oop_top10,
            ip_top10,
        );
    }
    println!("\nfreq column is % of total. oop_nz / ip_nz = non-zero combos (>0.01).");
    println!("Entropy in bits over L1-normalized range. top-10 = sum of top 10 weights / total.");
}

fn cmd_filter_preflop_ranges(input: PathBuf, output: PathBuf, label: String) {
    use cfvnet::datagen::precompute_ranges::PrecomputedRanges;
    let ranges = PrecomputedRanges::load(&input).unwrap_or_else(|e| {
        eprintln!("failed to load {}: {e}", input.display());
        std::process::exit(1);
    });
    let kept: Vec<_> = ranges.paths.into_iter().filter(|p| p.label == label).collect();
    if kept.is_empty() {
        eprintln!("no paths match label '{label}' in {}", input.display());
        std::process::exit(1);
    }
    let out = PrecomputedRanges { paths: kept };
    out.save(&output).unwrap_or_else(|e| {
        eprintln!("failed to save {}: {e}", output.display());
        std::process::exit(1);
    });
    println!("Wrote {} path(s) matching '{label}' to {}", out.paths.len(), output.display());
}

/// Returns (entropy_bits, top10_mass_frac) for a range vector.
fn range_shape(range: &[f32]) -> (f64, f64) {
    let total: f64 = range.iter().map(|&v| v as f64).sum();
    if total <= 0.0 { return (0.0, 0.0); }
    let entropy = range.iter().map(|&v| v as f64).filter(|&v| v > 0.0)
        .map(|v| { let p = v / total; -p * p.log2() })
        .sum::<f64>();
    let mut sorted: Vec<f64> = range.iter().map(|&v| v as f64).collect();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let top10: f64 = sorted.iter().take(10).sum();
    (entropy, top10 / total)
}

fn cmd_precompute_ranges(
    blueprint: PathBuf,
    output: PathBuf,
    _samples_per_bucket: usize,
    _spr_boundaries_str: &str,
    _initial_stack: i32,
) {
    use cfvnet::datagen::blueprint_ranges::BlueprintRangeGenerator;
    use cfvnet::datagen::precompute_ranges::compute_preflop_paths;

    let generator = BlueprintRangeGenerator::load(&blueprint).unwrap_or_else(|e| {
        eprintln!("Failed to load blueprint: {e}");
        std::process::exit(1);
    });

    println!("Computing preflop paths to flop...");

    let ranges = compute_preflop_paths(
        generator.strategy(),
        generator.tree(),
        generator.decision_map(),
    );

    ranges.save(&output).unwrap_or_else(|e| {
        eprintln!("Failed to save: {e}");
        std::process::exit(1);
    });

    println!("Saved {} preflop paths to {}", ranges.paths.len(), output.display());
    for path in &ranges.paths {
        println!(
            "  {:<40} freq={:.4}  OOP ~{} combos, IP ~{} combos",
            path.label, path.frequency, path.oop_nonzero, path.ip_nonzero
        );
    }
}

fn cmd_evaluate(model_dir: PathBuf, data_path: PathBuf) {
    use burn::backend::NdArray;
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
    use burn::tensor::{Tensor, TensorData};
    use cfvnet::config::board_cards_for_street;
    use cfvnet::eval::metrics::compute_prediction_metrics;
    use cfvnet::model::dataset::CfvDataset;
    use cfvnet::model::network::{CfvNet, INPUT_SIZE, POT_INDEX};

    let cfg = load_model_config(&model_dir);
    let board_cards = board_cards_for_street(&cfg.datagen.street);

    type B = NdArray;
    let device = Default::default();
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
    let model_path = resolve_model_path(&model_dir);

    let model = CfvNet::<B>::new(&device, cfg.training.hidden_layers, cfg.training.hidden_size, INPUT_SIZE)
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
            TensorData::new(item.input.clone(), [1, INPUT_SIZE]),
            &device,
        );
        let pred = model.forward(input);
        let pred_vec: Vec<f32> = pred.into_data().to_vec::<f32>().unwrap();

        let mask: Vec<bool> = item.mask.iter().map(|&v| v > 0.5).collect();
        // Pot is at POT_INDEX, normalized by 400.0 during encoding
        let pot = item.input[POT_INDEX] * 400.0;

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

/// Infer the number of board cards from a binary data file by reading
/// the first byte, which stores `board_size` (3=flop, 4=turn, 5=river).
/// This avoids requiring a `--street` CLI flag or a `config.yaml` for ONNX models.
#[cfg(any(feature = "onnx", test))]
fn infer_board_cards_from_data(data_path: &std::path::Path) -> usize {
    let mut f = std::fs::File::open(data_path).unwrap_or_else(|e| {
        eprintln!("failed to open data file {}: {e}", data_path.display());
        std::process::exit(1);
    });
    let mut buf = [0u8; 1];
    std::io::Read::read_exact(&mut f, &mut buf).unwrap_or_else(|e| {
        eprintln!("failed to read board_size from {}: {e}", data_path.display());
        std::process::exit(1);
    });
    let board_cards = buf[0] as usize;
    assert!(
        (3..=5).contains(&board_cards),
        "unexpected board_size {board_cards} in data file (expected 3-5)"
    );
    board_cards
}

/// Evaluate a `.onnx` model against binary boundary data using ONNX Runtime.
///
/// Board card count is inferred from the data file's first record (option b),
/// avoiding the need for a `config.yaml` or extra CLI flags.
#[cfg(feature = "onnx")]
fn cmd_eval_boundary_onnx(model_path: PathBuf, data_path: PathBuf) {
    use cfvnet::datagen::storage::{read_record, record_size};
    use cfvnet::eval::metrics::compute_normalized_mae;
    use cfvnet::model::boundary_dataset::encode_boundary_record;
    use cfvnet::model::network::INPUT_SIZE;
    use ort::session::{Session, builder::GraphOptimizationLevel};

    let session = Session::builder()
        .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
        .and_then(|b| b.commit_from_file(&model_path))
        .unwrap_or_else(|e| {
            eprintln!("failed to load ONNX model from {}: {e}", model_path.display());
            std::process::exit(1);
        });

    let board_cards = infer_board_cards_from_data(&data_path);
    let rec_size = record_size(board_cards) as u64;
    let file_size = std::fs::metadata(&data_path)
        .unwrap_or_else(|e| {
            eprintln!("failed to read data file {}: {e}", data_path.display());
            std::process::exit(1);
        })
        .len();
    let num_records = file_size / rec_size;
    let street_name = match board_cards {
        3 => "flop",
        4 => "turn",
        _ => "river",
    };
    println!("Evaluating {num_records} records (ONNX, {street_name}) ...");

    let file = std::fs::File::open(&data_path).unwrap_or_else(|e| {
        eprintln!("failed to open data file {}: {e}", data_path.display());
        std::process::exit(1);
    });
    let mut reader = std::io::BufReader::new(file);

    let mut all_maes: Vec<f64> = Vec::with_capacity(num_records as usize);
    let mut spr_maes: [Vec<f64>; 4] = [vec![], vec![], vec![], vec![]];

    while let Ok(rec) = read_record(&mut reader) {
        let item = encode_boundary_record(&rec);
        let mask: Vec<bool> = item.mask.iter().map(|&v| v > 0.5).collect();

        let input_tensor = ort::value::Tensor::from_array(
            ([1_i64, INPUT_SIZE as i64], item.input.clone()),
        )
        .expect("ort tensor creation");
        let outputs = session
            .run(ort::inputs![input_tensor].expect("ort inputs"))
            .expect("ort session run");
        let output_view = outputs[0]
            .try_extract_tensor::<f32>()
            .expect("ort output extract f32");
        let pred_vec: Vec<f32> = output_view.iter().copied().collect();

        let mae = compute_normalized_mae(&pred_vec, &item.target, &mask);
        all_maes.push(mae);

        let spr = if rec.pot > 0.0 {
            rec.effective_stack as f64 / rec.pot as f64
        } else {
            0.0
        };
        let bucket = if spr < 1.0 {
            0
        } else if spr < 3.0 {
            1
        } else if spr < 10.0 {
            2
        } else {
            3
        };
        spr_maes[bucket].push(mae);
    }

    if all_maes.is_empty() {
        println!("No records to evaluate.");
        return;
    }

    println!("Results ({} records):", all_maes.len());
    print_error_stats("  Overall", &mut all_maes);

    let bucket_labels = ["<1", "1-3", "3-10", "10+"];
    println!("\nMAE by SPR bucket:");
    for (i, label) in bucket_labels.iter().enumerate() {
        if spr_maes[i].is_empty() {
            println!("  SPR {:<5}: N/A     (0 records)", label);
        } else {
            print_error_stats(&format!("  SPR {:<5}", label), &mut spr_maes[i]);
        }
    }
}

fn cmd_diagnose_boundary(
    model_path: PathBuf,
    data_path: PathBuf,
    num_records: Option<usize>,
    csv_out: Option<PathBuf>,
) {
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (&model_path, &data_path, &num_records, &csv_out);
        eprintln!("diagnose-boundary requires building with --features onnx");
        std::process::exit(1);
    }
    #[cfg(feature = "onnx")]
    {
        cmd_diagnose_boundary_onnx(model_path, data_path, num_records, csv_out);
    }
}

#[cfg(feature = "onnx")]
fn cmd_diagnose_boundary_onnx(
    model_path: PathBuf,
    data_path: PathBuf,
    num_records: Option<usize>,
    csv_out: Option<PathBuf>,
) {
    use cfvnet::datagen::storage::{read_record, record_size};
    use cfvnet::model::boundary_dataset::encode_boundary_record;
    use cfvnet::model::network::INPUT_SIZE;
    use ort::session::{Session, builder::GraphOptimizationLevel};
    use range_solver::card::index_to_card_pair;
    use std::collections::HashMap;

    let session = Session::builder()
        .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
        .and_then(|b| b.commit_from_file(&model_path))
        .unwrap_or_else(|e| {
            eprintln!("failed to load ONNX model from {}: {e}", model_path.display());
            std::process::exit(1);
        });

    let board_cards = infer_board_cards_from_data(&data_path);
    if board_cards != 5 {
        eprintln!("diagnose-boundary requires river data (5-card board), got {board_cards}-card");
        std::process::exit(1);
    }

    let rec_size = record_size(board_cards) as u64;
    let file_size = std::fs::metadata(&data_path)
        .unwrap_or_else(|e| {
            eprintln!("failed to read data file {}: {e}", data_path.display());
            std::process::exit(1);
        })
        .len();
    let total_records = file_size / rec_size;
    let limit = num_records.map_or(total_records as usize, |n| n.min(total_records as usize));

    println!(
        "=== CFV Delta Analysis: {} vs {} ===",
        model_path.display(),
        data_path.display()
    );
    println!("Records: {limit}");

    let file = std::fs::File::open(&data_path).unwrap_or_else(|e| {
        eprintln!("failed to open data file {}: {e}", data_path.display());
        std::process::exit(1);
    });
    let mut reader = std::io::BufReader::new(file);

    // Collect deltas into bins
    const NUM_BINS: usize = 13;
    const COMBOS: usize = 1326;
    let bin_size = COMBOS / NUM_BINS; // 102
    let mut bin_deltas: Vec<Vec<f64>> = (0..NUM_BINS).map(|_| Vec::new()).collect();
    let mut cat_deltas: HashMap<HandCategory, Vec<f64>> = HashMap::new();

    let mut csv_writer = csv_out.map(|path| {
        let f = std::fs::File::create(&path).unwrap_or_else(|e| {
            eprintln!("failed to create CSV file {}: {e}", path.display());
            std::process::exit(1);
        });
        let mut w = std::io::BufWriter::new(f);
        use std::io::Write;
        writeln!(w, "record_idx,combo_idx,category,net_cfv,truth_cfv,delta").unwrap();
        w
    });

    for rec_idx in 0..limit {
        let rec = match read_record(&mut reader) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("error reading record {rec_idx}: {e}");
                break;
            }
        };
        let item = encode_boundary_record(&rec);
        let mask: Vec<bool> = item.mask.iter().map(|&v| v > 0.5).collect();

        let input_tensor = ort::value::Tensor::from_array(
            ([1_i64, INPUT_SIZE as i64], item.input.clone()),
        )
        .expect("ort tensor creation");
        let outputs = session
            .run(ort::inputs![input_tensor].expect("ort inputs"))
            .expect("ort session run");
        let pred_vec: Vec<f32> = outputs[0]
            .try_extract_tensor::<f32>()
            .expect("ort output extract f32")
            .iter()
            .copied()
            .collect();

        let board = &rec.board;
        for combo_idx in 0..COMBOS {
            if !mask[combo_idx] {
                continue;
            }
            let delta = (pred_vec[combo_idx] - item.target[combo_idx]) as f64;

            // Index bin
            let bin = (combo_idx / bin_size).min(NUM_BINS - 1);
            bin_deltas[bin].push(delta);

            // Hand category
            let (c1, c2) = index_to_card_pair(combo_idx);
            let category = classify_made_hand(c1, c2, board);
            cat_deltas.entry(category).or_default().push(delta);

            // CSV output
            if let Some(ref mut w) = csv_writer {
                use std::io::Write;
                writeln!(
                    w,
                    "{rec_idx},{combo_idx},{},{:.6},{:.6},{:.6}",
                    category.label(),
                    pred_vec[combo_idx],
                    item.target[combo_idx],
                    delta as f32
                )
                .unwrap();
            }
        }
    }

    // Print per-index-bin stats
    println!("\n--- Per combo-index bin ({NUM_BINS} x ~{bin_size} combos) ---");
    for (i, deltas) in bin_deltas.iter_mut().enumerate() {
        let lo = i * bin_size;
        let hi = if i == NUM_BINS - 1 { COMBOS } else { lo + bin_size };
        if deltas.is_empty() {
            println!("bin [{lo:04}..{hi:04}]: n=0");
            continue;
        }
        print_delta_stats(&format!("bin [{lo:04}..{hi:04}]"), deltas);
    }

    // Print per-category stats sorted by count descending
    println!("\n--- Per made-hand category ---");
    let mut cat_order: Vec<HandCategory> = HandCategory::ALL
        .iter()
        .copied()
        .filter(|cat| cat_deltas.contains_key(cat))
        .collect();
    cat_order.sort_by(|a, b| {
        cat_deltas[b].len().cmp(&cat_deltas[a].len())
    });
    for cat in &cat_order {
        let deltas = cat_deltas.get_mut(cat).unwrap();
        print_delta_stats(&format!("{:<14}", cat), deltas);
    }

    if let Some(ref mut w) = csv_writer {
        use std::io::Write;
        w.flush().unwrap();
    }
}

/// Print delta statistics: count, mean, std, |mean|, p5, p95, min, max.
#[allow(dead_code)]
fn print_delta_stats(label: &str, values: &mut [f64]) {
    let n = values.len();
    if n == 0 {
        println!("{label}: n=0");
        return;
    }
    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();
    let abs_mean: f64 = values.iter().map(|v| v.abs()).sum::<f64>() / n as f64;

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p = |frac: f64| values[((frac * (n - 1) as f64) as usize).min(n - 1)];

    println!(
        "{label}: n={n} mean={mean:+.4} std={std_dev:.4} |mean|={abs_mean:.4} p5={:.4} p95={:.4} min={:.4} max={:.4}",
        p(0.05),
        p(0.95),
        values[0],
        values[n - 1]
    );
}

fn cmd_eval_boundary(model_dir: PathBuf, data_path: PathBuf) {
    // Dispatch: .onnx files use ONNX Runtime; everything else uses the Burn path.
    if is_onnx_model(&model_dir) {
        #[cfg(feature = "onnx")]
        {
            return cmd_eval_boundary_onnx(model_dir, data_path);
        }
        #[cfg(not(feature = "onnx"))]
        {
            eprintln!("ONNX models require building with --features onnx");
            std::process::exit(1);
        }
    }

    use burn::backend::NdArray;
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
    use burn::tensor::{Tensor, TensorData};
    use cfvnet::datagen::storage::{read_record, record_size};
    use cfvnet::eval::metrics::compute_normalized_mae;
    use cfvnet::model::boundary_dataset::encode_boundary_record;
    use cfvnet::model::boundary_net::BoundaryNet;
    use cfvnet::model::network::INPUT_SIZE;

    let cfg = load_model_config(&model_dir);
    let board_cards = cfvnet::config::board_cards_for_street(&cfg.datagen.street);

    type B = NdArray;
    let device = Default::default();
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
    let model_path = resolve_model_path(&model_dir);

    let model = BoundaryNet::<B>::new(&device, cfg.training.hidden_layers, cfg.training.hidden_size)
        .load_file(&model_path, &recorder, &device)
        .unwrap_or_else(|e| {
            eprintln!("failed to load model from {}: {e}", model_path.display());
            std::process::exit(1);
        });

    // Count records
    let rec_size = record_size(board_cards) as u64;
    let file_size = std::fs::metadata(&data_path)
        .unwrap_or_else(|e| {
            eprintln!("failed to read data file {}: {e}", data_path.display());
            std::process::exit(1);
        })
        .len();
    let num_records = file_size / rec_size;
    println!("Evaluating {num_records} records...");

    let file = std::fs::File::open(&data_path).unwrap_or_else(|e| {
        eprintln!("failed to open data file {}: {e}", data_path.display());
        std::process::exit(1);
    });
    let mut reader = std::io::BufReader::new(file);

    // Collect per-record MAE and per-SPR-bucket MAE values.
    let mut all_maes = Vec::new();
    let mut spr_maes: [Vec<f64>; 4] = [vec![], vec![], vec![], vec![]];

    while let Ok(rec) = read_record(&mut reader) {
        let item = encode_boundary_record(&rec);
        let mask: Vec<bool> = item.mask.iter().map(|&v| v > 0.5).collect();

        let input = Tensor::<B, 2>::from_data(
            TensorData::new(item.input.clone(), [1, INPUT_SIZE]),
            &device,
        );
        let pred = model.forward(input);
        let pred_vec: Vec<f32> = pred.into_data().to_vec::<f32>().unwrap();

        let mae = compute_normalized_mae(&pred_vec, &item.target, &mask);
        all_maes.push(mae);

        let spr = if rec.pot > 0.0 {
            rec.effective_stack as f64 / rec.pot as f64
        } else {
            0.0
        };
        let bucket = if spr < 1.0 {
            0
        } else if spr < 3.0 {
            1
        } else if spr < 10.0 {
            2
        } else {
            3
        };
        spr_maes[bucket].push(mae);
    }

    if all_maes.is_empty() {
        println!("No records to evaluate.");
        return;
    }

    println!("Results ({} records):", all_maes.len());
    print_error_stats("  Overall", &mut all_maes);

    let bucket_labels = ["<1", "1-3", "3-10", "10+"];
    println!("\nMAE by SPR bucket:");
    for (i, label) in bucket_labels.iter().enumerate() {
        if spr_maes[i].is_empty() {
            println!("  SPR {:<5}: N/A     (0 records)", label);
        } else {
            print_error_stats(&format!("  SPR {:<5}", label), &mut spr_maes[i]);
        }
    }
}

fn print_error_stats(label: &str, values: &mut [f64]) {
    let n = values.len();
    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p = |frac: f64| values[(frac * (n - 1) as f64) as usize];

    println!(
        "{}: mean={:.6} std={:.4} p50={:.4} p90={:.4} p95={:.4} p99={:.4} max={:.4}  ({n} records)",
        label, mean, std_dev, p(0.5), p(0.9), p(0.95), p(0.99),
        values.last().unwrap_or(&0.0)
    );
}

fn cmd_compare_boundary(model_dir: &std::path::Path, data_path: &std::path::Path, num_positions: usize) {
    use burn::backend::NdArray;
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
    use burn::tensor::{Tensor, TensorData};
    use cfvnet::datagen::storage::read_record;
    use cfvnet::eval::metrics::compute_normalized_mae;
    use cfvnet::model::boundary_dataset::encode_boundary_record;
    use cfvnet::model::boundary_net::BoundaryNet;
    use cfvnet::model::network::INPUT_SIZE;

    let cfg = load_model_config(model_dir);

    type B = NdArray;
    let device = Default::default();
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
    let model_path = resolve_model_path(model_dir);

    let model = BoundaryNet::<B>::new(&device, cfg.training.hidden_layers, cfg.training.hidden_size)
        .load_file(&model_path, &recorder, &device)
        .unwrap_or_else(|e| {
            eprintln!("failed to load model from {}: {e}", model_path.display());
            std::process::exit(1);
        });

    let file = std::fs::File::open(data_path).unwrap_or_else(|e| {
        eprintln!("failed to open data file {}: {e}", data_path.display());
        std::process::exit(1);
    });
    let mut reader = std::io::BufReader::new(file);

    let mut total_mae = 0.0_f64;
    let mut worst_mae = 0.0_f64;
    let mut count = 0_u64;

    // SPR buckets: <1, 1-3, 3-10, 10+
    let mut spr_mae = [0.0_f64; 4];
    let mut spr_count = [0_u64; 4];
    let mut spr_worst = [0.0_f64; 4];

    while count < num_positions as u64 {
        let rec = match read_record(&mut reader) {
            Ok(r) => r,
            Err(_) => break,
        };

        let item = encode_boundary_record(&rec);
        let mask: Vec<bool> = item.mask.iter().map(|&v| v > 0.5).collect();

        let input = Tensor::<B, 2>::from_data(
            TensorData::new(item.input.clone(), [1, INPUT_SIZE]),
            &device,
        );
        let pred = model.forward(input);
        let pred_vec: Vec<f32> = pred.into_data().to_vec::<f32>().unwrap();

        let mae = compute_normalized_mae(&pred_vec, &item.target, &mask);
        total_mae += mae;
        if mae > worst_mae {
            worst_mae = mae;
        }
        count += 1;

        // SPR bucket
        let spr = if rec.pot > 0.0 {
            rec.effective_stack as f64 / rec.pot as f64
        } else {
            0.0
        };
        let bucket = if spr < 1.0 {
            0
        } else if spr < 3.0 {
            1
        } else if spr < 10.0 {
            2
        } else {
            3
        };
        spr_mae[bucket] += mae;
        spr_count[bucket] += 1;
        if mae > spr_worst[bucket] {
            spr_worst[bucket] = mae;
        }
    }

    if count == 0 {
        println!("No records to compare.");
        return;
    }

    let n = count as f64;
    println!("Comparison results ({count} positions):");
    println!("  Normalized MAE: {:.6}", total_mae / n);
    println!("  Worst-case MAE: {:.6}", worst_mae);

    let bucket_labels = ["<1", "1-3", "3-10", "10+"];
    println!("\nMAE by SPR bucket:");
    for (i, label) in bucket_labels.iter().enumerate() {
        if spr_count[i] > 0 {
            println!(
                "  SPR {:<5}: MAE={:.6}  worst={:.6}  ({} positions)",
                label,
                spr_mae[i] / spr_count[i] as f64,
                spr_worst[i],
                spr_count[i]
            );
        } else {
            println!("  SPR {:<5}: N/A  (0 positions)", label);
        }
    }
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
    use cfvnet::eval::compare::run_comparison;
    use cfvnet::model::dataset::encode_situation_for_inference;
    use cfvnet::model::network::{CfvNet, INPUT_SIZE};

    let cfg = load_model_config(&model_dir);

    type B = NdArray;
    let device = Default::default();
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
    let model_path = resolve_model_path(&model_dir);

    let model = CfvNet::<B>::new(&device, cfg.training.hidden_layers, cfg.training.hidden_size, INPUT_SIZE)
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

    let summary = run_comparison(&cfg.game, &cfg.datagen, num_spots, cfvnet::config::resolve_seed(cfg.datagen.seed), |sit, _solve_result| {
        let input_data = encode_situation_for_inference(sit, 0);
        let input = Tensor::<B, 2>::from_data(
            TensorData::new(input_data, [1, INPUT_SIZE]),
            &device,
        );
        let pred = model.forward(input);
        pred.into_data().to_vec::<f32>().unwrap()
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

    println!("Comparing {num_spots} turn spots against PostFlopGame + RiverNetEvaluator...");

    let summary = run_turn_comparison_net(&cfg, &model_path, &river_path, num_spots, cfvnet::config::resolve_seed(cfg.datagen.seed))
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

    println!("Comparing {num_spots} turn spots against PostFlopGame + exact river evaluation...");

    let summary = run_turn_comparison_exact(&cfg, &model_path, num_spots, cfvnet::config::resolve_seed(cfg.datagen.seed)).unwrap_or_else(|e| {
        eprintln!("comparison failed: {e}");
        std::process::exit(1);
    });

    print_summary(&summary);
}

fn cmd_datagen_eval(data: PathBuf) {
    use cfvnet::datagen::storage::read_record;
    use std::io::BufReader;

    let paths: Vec<PathBuf> = if data.is_dir() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&data)
            .unwrap_or_else(|e| {
                eprintln!("failed to read directory {}: {e}", data.display());
                std::process::exit(1);
            })
            .filter_map(|entry| {
                let entry = entry.ok()?;
                if entry.file_type().ok()?.is_file() { Some(entry.path()) } else { None }
            })
            .collect();
        entries.sort();
        entries
    } else {
        vec![data.clone()]
    };

    if paths.is_empty() {
        eprintln!("no files found in {}", data.display());
        std::process::exit(1);
    }

    let mut pots = Vec::new();
    let mut stacks = Vec::new();
    let mut game_values = Vec::new();
    let mut cfv_mins = Vec::new();
    let mut cfv_maxs = Vec::new();
    let mut cfv_abs_maxs = Vec::new();
    let mut total_stakes = Vec::new();
    let mut norm_target_abs_maxs = Vec::new();
    let mut card_max: u8 = 0;
    let mut num_extreme_records = 0u64;
    let mut extreme_examples: Vec<String> = Vec::new();

    // Range shape statistics
    let mut oop_densities = Vec::new();
    let mut oop_entropies = Vec::new();
    let mut oop_top10_concs = Vec::new();
    let mut oop_max_mean_ratios = Vec::new();
    let mut oop_totals = Vec::new();
    let mut ip_densities = Vec::new();
    let mut ip_entropies = Vec::new();
    let mut ip_top10_concs = Vec::new();
    let mut ip_max_mean_ratios = Vec::new();
    let mut ip_totals = Vec::new();

    for path in &paths {
        let file = std::fs::File::open(path).unwrap_or_else(|e| {
            eprintln!("failed to open {}: {e}", path.display());
            std::process::exit(1);
        });
        let mut reader = BufReader::new(file);
        let mut rec_idx = 0u64;
        loop {
            match read_record(&mut reader) {
                Ok(rec) => {
                    pots.push(rec.pot as f64);
                    stacks.push(rec.effective_stack as f64);
                    game_values.push(rec.game_value as f64);

                    // Track board card IDs
                    for &c in &rec.board {
                        if c > card_max { card_max = c; }
                    }

                    // CFV statistics (over valid entries only)
                    let mut cfv_min = f64::INFINITY;
                    let mut cfv_max = f64::NEG_INFINITY;
                    for (i, &cfv) in rec.cfvs.iter().enumerate() {
                        if rec.valid_mask[i] != 0 {
                            let v = cfv as f64;
                            if v < cfv_min { cfv_min = v; }
                            if v > cfv_max { cfv_max = v; }
                        }
                    }
                    if cfv_min.is_finite() {
                        cfv_mins.push(cfv_min);
                        cfv_maxs.push(cfv_max);
                        cfv_abs_maxs.push(cfv_min.abs().max(cfv_max.abs()));
                    }

                    // Normalized target statistics (BoundaryNet encoding)
                    let total_stake = rec.pot as f64 + rec.effective_stack as f64;
                    total_stakes.push(total_stake);
                    if total_stake > 0.0 {
                        let pot_over_norm = rec.pot as f64 / total_stake;
                        let mut norm_abs_max = 0.0_f64;
                        for (i, &cfv) in rec.cfvs.iter().enumerate() {
                            if rec.valid_mask[i] != 0 {
                                let norm = (cfv as f64) * pot_over_norm;
                                if norm.abs() > norm_abs_max { norm_abs_max = norm.abs(); }
                            }
                        }
                        norm_target_abs_maxs.push(norm_abs_max);

                        // Flag extreme records
                        let norm_gv = (rec.game_value as f64 * pot_over_norm).abs();
                        if norm_abs_max > 5.0 || norm_gv > 5.0 || total_stake < 1.0 {
                            num_extreme_records += 1;
                            if extreme_examples.len() < 10 {
                                extreme_examples.push(format!(
                                    "  file={} rec={}: pot={:.1} stack={:.1} total={:.1} gv={:.4} norm_gv={:.4} max_norm_cfv={:.4} board={:?}",
                                    path.file_name().unwrap_or_default().to_string_lossy(),
                                    rec_idx, rec.pot, rec.effective_stack, total_stake,
                                    rec.game_value, norm_gv, norm_abs_max, rec.board
                                ));
                            }
                        }
                    }
                    // Range shape statistics
                    let (d, e, t10, mm, tot) = range_stats(&rec.oop_range);
                    oop_densities.push(d);
                    oop_entropies.push(e);
                    oop_top10_concs.push(t10);
                    oop_max_mean_ratios.push(mm);
                    oop_totals.push(tot);

                    let (d, e, t10, mm, tot) = range_stats(&rec.ip_range);
                    ip_densities.push(d);
                    ip_entropies.push(e);
                    ip_top10_concs.push(t10);
                    ip_max_mean_ratios.push(mm);
                    ip_totals.push(tot);

                    rec_idx += 1;
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => {
                    eprintln!("read error in {}: {e}", path.display());
                    std::process::exit(1);
                }
            }
        }
    }

    let num_records = pots.len();
    println!("Loaded {num_records} records from {} file(s).", paths.len());

    if num_records < 2 {
        println!("Not enough records for histograms.");
        return;
    }

    // Summary statistics
    println!("\nValue Statistics:");
    print_stats("  pot", &pots);
    print_stats("  effective_stack", &stacks);
    print_stats("  total_stake (pot+stack)", &total_stakes);
    print_stats("  game_value (pot-relative)", &game_values);
    print_stats("  cfv_min (pot-relative)", &cfv_mins);
    print_stats("  cfv_max (pot-relative)", &cfv_maxs);
    print_stats("  |cfv|_max (pot-relative)", &cfv_abs_maxs);
    print_stats("  |norm_target|_max (boundary)", &norm_target_abs_maxs);
    println!("  max board card ID: {card_max} (expected < 52)");

    // Extreme records
    if num_extreme_records > 0 {
        println!("\nExtreme Records ({num_extreme_records} total, |norm_cfv|>5 or |norm_gv|>5 or total_stake<1):");
        for ex in &extreme_examples {
            println!("{ex}");
        }
        if num_extreme_records > 10 {
            println!("  ... and {} more", num_extreme_records - 10);
        }
    } else {
        println!("\nNo extreme records found.");
    }

    print_raw_frequency_histogram("Frequency by Stack Size", &stacks);
    print_raw_frequency_histogram("Frequency by Pot Size", &pots);
    print_raw_frequency_histogram("Frequency by Game Value", &game_values);
    print_raw_frequency_histogram("Frequency by |CFV|_max", &cfv_abs_maxs);
    print_raw_frequency_histogram("Frequency by |Normalized Target|_max", &norm_target_abs_maxs);

    let sprs: Vec<f64> = pots.iter().zip(&stacks)
        .filter(|(p, _)| **p > 0.0)
        .map(|(p, s)| *s / *p)
        .collect();
    print_spr_frequency_histogram("Frequency by SPR", &sprs);

    // Range Statistics
    let oop_nonzero_mass = oop_totals.iter().filter(|&&t| t > 0.0).count();
    let ip_nonzero_mass = ip_totals.iter().filter(|&&t| t > 0.0).count();
    println!("\n=== Range Statistics ===\n");
    println!("Records with nonzero OOP mass: {}/{}", oop_nonzero_mass, num_records);
    println!("Records with nonzero IP mass:  {}/{}", ip_nonzero_mass, num_records);

    println!("\nOOP ranges (across {} records):", num_records);
    print_stats("  Density       ", &oop_densities);
    print_stats("  Entropy (bits)", &oop_entropies);
    print_stats("  Top-10 mass   ", &oop_top10_concs);
    print_stats("  Max/mean      ", &oop_max_mean_ratios);
    print_stats("  Total mass    ", &oop_totals);

    println!("\nIP ranges (across {} records):", num_records);
    print_stats("  Density       ", &ip_densities);
    print_stats("  Entropy (bits)", &ip_entropies);
    print_stats("  Top-10 mass   ", &ip_top10_concs);
    print_stats("  Max/mean      ", &ip_max_mean_ratios);
    print_stats("  Total mass    ", &ip_totals);
}

/// Compute range-shape statistics for a single range vector.
/// Returns (density, entropy_bits, top10_concentration, max_mean_ratio, total_mass).
fn range_stats(range: &[f32]) -> (f64, f64, f64, f64, f64) {
    let total: f64 = range.iter().map(|&v| v as f64).sum();
    let nonzero_count = range.iter().filter(|&&v| v > 0.0).count();
    let density = nonzero_count as f64 / range.len() as f64;

    let entropy = if total > 0.0 {
        range.iter()
            .filter(|&&v| v > 0.0)
            .map(|&v| {
                let p = v as f64 / total;
                -p * p.log2()
            })
            .sum::<f64>()
    } else {
        0.0
    };

    let top10_mass = if total > 0.0 {
        let mut sorted: Vec<f64> = range.iter().map(|&v| v as f64).collect();
        sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let top: f64 = sorted.iter().take(10).sum();
        top / total
    } else {
        0.0
    };

    let max_mean_ratio = if nonzero_count > 0 && total > 0.0 {
        let mean = total / nonzero_count as f64;
        let max = range.iter().map(|&v| v as f64).fold(0.0_f64, f64::max);
        max / mean
    } else {
        0.0
    };

    (density, entropy, top10_mass, max_mean_ratio, total)
}

fn print_stats(label: &str, values: &[f64]) {
    if values.is_empty() {
        println!("{label}: no data");
        return;
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = values.iter().sum();
    let mean = sum / values.len() as f64;

    // Percentiles via sorted copy (only if reasonable size, else sample)
    let (p1, p50, p99) = if values.len() <= 20_000_000 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p = |frac: f64| sorted[(frac * (sorted.len() - 1) as f64) as usize];
        (p(0.01), p(0.5), p(0.99))
    } else {
        (f64::NAN, f64::NAN, f64::NAN)
    };
    println!("{label}: min={min:.4} p1={p1:.4} median={p50:.4} mean={mean:.4} p99={p99:.4} max={max:.4} (n={})", values.len());
}

fn print_raw_frequency_histogram(title: &str, values: &[f64]) {
    const NUM_BUCKETS: usize = 20;
    const BAR_WIDTH: usize = 40;

    let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if (max_val - min_val).abs() < 1e-9 {
        println!("\n{title}: all values = {:.0}", min_val);
        return;
    }

    let bucket_width = (max_val - min_val) / NUM_BUCKETS as f64;
    let mut bucket_counts = [0_u32; NUM_BUCKETS];

    for &v in values {
        let idx = ((v - min_val) / bucket_width) as usize;
        let idx = idx.min(NUM_BUCKETS - 1);
        bucket_counts[idx] += 1;
    }

    let max_count = *bucket_counts.iter().max().unwrap_or(&1);

    println!("\n{title}:");
    for (i, &count) in bucket_counts.iter().enumerate() {
        let lo = min_val + i as f64 * bucket_width;
        let hi = lo + bucket_width;
        let bar_len = ((count as f64 / max_count as f64) * BAR_WIDTH as f64).round() as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  {:>6.0}-{:<6.0} |{:<width$}| {}", lo, hi, bar, count, width = BAR_WIDTH);
    }
}

const SPR_BOUNDARIES: &[f64] = &[0.0, 0.5, 1.5, 4.0, 8.0, 50.0];
const SPR_LABELS: &[&str] = &[
    "0.0-0.5  jam/fold   ",
    "0.5-1.5  ~1 PSB     ",
    "1.5-4.0  bread&butter",
    "4.0-8.0  single st  ",
    "8.0-50.0 deep/limped",
];

fn spr_bucket(spr: f64) -> Option<usize> {
    for i in 0..SPR_LABELS.len() {
        if spr >= SPR_BOUNDARIES[i] && spr < SPR_BOUNDARIES[i + 1] {
            return Some(i);
        }
    }
    // Include the upper boundary in the last bucket
    if (spr - SPR_BOUNDARIES[SPR_LABELS.len()]).abs() < 1e-9 {
        return Some(SPR_LABELS.len() - 1);
    }
    None
}

fn print_spr_frequency_histogram(title: &str, sprs: &[f64]) {
    const BAR_WIDTH: usize = 40;

    let mut counts = vec![0_u32; SPR_LABELS.len()];
    for &spr in sprs {
        if let Some(idx) = spr_bucket(spr) {
            counts[idx] += 1;
        }
    }

    let max_count = *counts.iter().max().unwrap_or(&1);
    if max_count == 0 {
        return;
    }

    println!("\n{title}:");
    for (i, label) in SPR_LABELS.iter().enumerate() {
        let count = counts[i];
        let bar_len = ((count as f64 / max_count as f64) * BAR_WIDTH as f64).round() as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  {label} |{:<width$}| {}", bar, count, width = BAR_WIDTH);
    }
}

fn print_spr_mbb_histogram(title: &str, spr_mbb: &[(f64, f64)]) {
    const BAR_WIDTH: usize = 40;

    let mut sums = vec![0.0_f64; SPR_LABELS.len()];
    let mut counts = vec![0_u32; SPR_LABELS.len()];

    for &(spr, mbb) in spr_mbb {
        if let Some(idx) = spr_bucket(spr) {
            sums[idx] += mbb;
            counts[idx] += 1;
        }
    }

    let means: Vec<f64> = sums.iter().zip(&counts)
        .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
        .collect();

    let max_val = means.iter().copied().fold(0.0_f64, f64::max);
    if max_val <= 0.0 {
        return;
    }

    println!("\n{title}:");
    for (i, label) in SPR_LABELS.iter().enumerate() {
        let mean = means[i];
        let count = counts[i];
        let bar_len = ((mean / max_val) * BAR_WIDTH as f64).round() as usize;
        let bar: String = "█".repeat(bar_len);
        if count > 0 {
            println!("  {label} |{:<width$}| {:.2} mBB  (n={})", bar, mean, count, width = BAR_WIDTH);
        } else {
            println!("  {label} |{:<width$}|              (n=0)", "", width = BAR_WIDTH);
        }
    }
}

/// Board texture labels (non-exclusive — a board can match multiple).
const TEXTURE_LABELS: &[&str] = &[
    "Monotone     ",
    "4-flush      ",
    "Three-flush  ",
    "Two-tone     ",
    "Rainbow      ",
    "4-straight   ",
    "3-straight   ",
    "Paired       ",
    "Dry          ",
];

/// Return all texture tags that apply to this board.
fn board_texture_tags(board: &[u8; 5], board_size: usize) -> Vec<&'static str> {
    let cards = &board[..board_size];

    // Suit counts.
    let mut suit_counts = [0u8; 4];
    for &card in cards {
        suit_counts[(card % 4) as usize] += 1;
    }
    let max_suit = suit_counts.iter().copied().max().unwrap_or(0);

    // Rank presence + pair detection.
    let mut rank_counts = [0u8; 13];
    for &card in cards {
        rank_counts[(card / 4) as usize] += 1;
    }
    let paired = rank_counts.iter().any(|&c| c >= 2);

    // Max consecutive ranks (including ace-low wrap for wheels).
    let max_consec = {
        let mut best = 0u8;
        let mut run = 0u8;
        for &c in &rank_counts {
            if c > 0 { run += 1; best = best.max(run); } else { run = 0; }
        }
        // Check ace-low: if ace present, count consecutive from rank 0.
        if rank_counts[12] > 0 {
            let mut wheel_run = 1u8; // ace counts as low
            for &c in &rank_counts[..12] {
                if c > 0 { wheel_run += 1; } else { break; }
            }
            best = best.max(wheel_run);
        }
        best
    };

    let mut tags = Vec::new();

    // Suit textures (mutually exclusive).
    match max_suit {
        5.. => tags.push("Monotone     "),
        4 => tags.push("4-flush      "),
        3 => tags.push("Three-flush  "),
        2 => tags.push("Two-tone     "),
        _ => tags.push("Rainbow      "),
    }

    // Straight textures (non-exclusive with suit textures).
    if max_consec >= 4 { tags.push("4-straight   "); }
    else if max_consec >= 3 { tags.push("3-straight   "); }

    if paired { tags.push("Paired       "); }

    // Dry: no flush draw (max suit < 3), no straight draw (max consec < 3), not paired.
    if max_suit < 3 && max_consec < 3 && !paired {
        tags.push("Dry          ");
    }

    tags
}

fn print_texture_histogram(spots: &[cfvnet::eval::compare::SpotResult]) {
    const BAR_WIDTH: usize = 40;

    let mut mae_sums = [0.0_f64; TEXTURE_LABELS.len()];
    let mut mbb_sums = [0.0_f64; TEXTURE_LABELS.len()];
    let mut counts = [0_u32; TEXTURE_LABELS.len()];

    for spot in spots {
        let tags = board_texture_tags(&spot.board, spot.board_size);
        for tag in tags {
            if let Some(idx) = TEXTURE_LABELS.iter().position(|&l| l == tag) {
                mae_sums[idx] += spot.mae;
                mbb_sums[idx] += spot.mbb;
                counts[idx] += 1;
            }
        }
    }

    let mae_means: Vec<f64> = mae_sums.iter().zip(&counts)
        .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
        .collect();
    let mbb_means: Vec<f64> = mbb_sums.iter().zip(&counts)
        .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
        .collect();

    let max_mae = mae_means.iter().copied().fold(0.0_f64, f64::max);
    if max_mae <= 0.0 {
        return;
    }

    println!("\nMAE / mBB Error by Board Texture:");
    for (i, label) in TEXTURE_LABELS.iter().enumerate() {
        let mae = mae_means[i];
        let mbb = mbb_means[i];
        let count = counts[i];
        let bar_len = ((mae / max_mae) * BAR_WIDTH as f64).round() as usize;
        let bar: String = "█".repeat(bar_len);
        if count > 0 {
            println!("  {label} |{:<width$}| MAE {:.4}  {:.0} mBB  (n={})", bar, mae, mbb, count, width = BAR_WIDTH);
        } else {
            println!("  {label} |{:<width$}|                          (n=0)", "", width = BAR_WIDTH);
        }
    }
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
    sorted.sort_by(|a, b| a.mae.total_cmp(&b.mae));

    let top_n = sorted.len().min(10);

    println!("\nBest {} spots (by MAE):", top_n);
    for (i, spot) in sorted.iter().take(top_n).enumerate() {
        println!("  {}. {}  Pot: {:<5} Stack: {:<5} MAE: {:.6}  mBB: {:.2}",
            i + 1, format_board(&spot.board, spot.board_size),
            spot.pot, spot.effective_stack, spot.mae, spot.mbb);
    }

    println!("\nWorst {} spots (by MAE):", top_n);
    for (i, spot) in sorted.iter().rev().take(top_n).enumerate() {
        println!("  {}. {}  Pot: {:<5} Stack: {:<5} MAE: {:.6}  mBB: {:.2}",
            i + 1, format_board(&spot.board, spot.board_size),
            spot.pot, spot.effective_stack, spot.mae, spot.mbb);
    }

    print_histogram("mBB Error by Stack Size", &summary.spots, |s| s.effective_stack as f64, |s| s.mbb);
    print_histogram("mBB Error by Pot Size", &summary.spots, |s| s.pot as f64, |s| s.mbb);
    print_frequency_histogram("Frequency by Stack Size", &summary.spots, |s| s.effective_stack as f64);
    print_frequency_histogram("Frequency by Pot Size", &summary.spots, |s| s.pot as f64);

    let sprs: Vec<f64> = summary.spots.iter()
        .filter(|s| s.pot > 0)
        .map(|s| s.effective_stack as f64 / s.pot as f64)
        .collect();
    print_spr_frequency_histogram("Frequency by SPR", &sprs);

    let spr_mbb: Vec<(f64, f64)> = summary.spots.iter()
        .filter(|s| s.pot > 0)
        .map(|s| (s.effective_stack as f64 / s.pot as f64, s.mbb))
        .collect();
    print_spr_mbb_histogram("mBB Error by SPR", &spr_mbb);

    print_texture_histogram(&summary.spots);
}

fn print_histogram<F, G>(title: &str, spots: &[cfvnet::eval::compare::SpotResult], key_fn: F, val_fn: G)
where
    F: Fn(&cfvnet::eval::compare::SpotResult) -> f64,
    G: Fn(&cfvnet::eval::compare::SpotResult) -> f64,
{
    const NUM_BUCKETS: usize = 20;
    const BAR_WIDTH: usize = 40;

    if spots.len() < 2 {
        return;
    }

    let min_key = spots.iter().map(&key_fn).fold(f64::INFINITY, f64::min);
    let max_key = spots.iter().map(&key_fn).fold(f64::NEG_INFINITY, f64::max);

    if (max_key - min_key).abs() < 1e-9 {
        return;
    }

    let bucket_width = (max_key - min_key) / NUM_BUCKETS as f64;
    let mut bucket_sums = [0.0_f64; NUM_BUCKETS];
    let mut bucket_counts = [0_u32; NUM_BUCKETS];

    for spot in spots {
        let k = key_fn(spot);
        let idx = ((k - min_key) / bucket_width) as usize;
        let idx = idx.min(NUM_BUCKETS - 1);
        bucket_sums[idx] += val_fn(spot);
        bucket_counts[idx] += 1;
    }

    let bucket_means: Vec<f64> = bucket_sums
        .iter()
        .zip(&bucket_counts)
        .map(|(&sum, &count)| if count > 0 { sum / count as f64 } else { 0.0 })
        .collect();

    let max_val = bucket_means.iter().copied().fold(0.0_f64, f64::max);
    if max_val <= 0.0 {
        return;
    }

    println!("\n{title}:");
    for i in 0..NUM_BUCKETS {
        let lo = min_key + i as f64 * bucket_width;
        let hi = lo + bucket_width;
        let mean = bucket_means[i];
        let bar_len = ((mean / max_val) * BAR_WIDTH as f64).round() as usize;
        let bar: String = "█".repeat(bar_len);
        let count = bucket_counts[i];
        if count > 0 {
            println!("  {:>6.0}-{:<6.0} |{:<width$}| {:.2} mBB  (n={})", lo, hi, bar, mean, count, width = BAR_WIDTH);
        } else {
            println!("  {:>6.0}-{:<6.0} |{:<width$}|              (n=0)", lo, hi, "", width = BAR_WIDTH);
        }
    }
}

fn print_frequency_histogram<F>(title: &str, spots: &[cfvnet::eval::compare::SpotResult], key_fn: F)
where
    F: Fn(&cfvnet::eval::compare::SpotResult) -> f64,
{
    const NUM_BUCKETS: usize = 20;
    const BAR_WIDTH: usize = 40;

    if spots.len() < 2 {
        return;
    }

    let min_key = spots.iter().map(&key_fn).fold(f64::INFINITY, f64::min);
    let max_key = spots.iter().map(&key_fn).fold(f64::NEG_INFINITY, f64::max);

    if (max_key - min_key).abs() < 1e-9 {
        return;
    }

    let bucket_width = (max_key - min_key) / NUM_BUCKETS as f64;
    let mut bucket_counts = [0_u32; NUM_BUCKETS];

    for spot in spots {
        let k = key_fn(spot);
        let idx = ((k - min_key) / bucket_width) as usize;
        let idx = idx.min(NUM_BUCKETS - 1);
        bucket_counts[idx] += 1;
    }

    let max_count = *bucket_counts.iter().max().unwrap_or(&1);
    if max_count == 0 {
        return;
    }

    println!("\n{title}:");
    for (i, &count) in bucket_counts.iter().enumerate() {
        let lo = min_key + i as f64 * bucket_width;
        let hi = lo + bucket_width;
        let bar_len = ((count as f64 / max_count as f64) * BAR_WIDTH as f64).round() as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  {:>6.0}-{:<6.0} |{:<width$}| {}", lo, hi, bar, count, width = BAR_WIDTH);
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

// ---------------------------------------------------------------------------
// Hand category classification for diagnose-boundary
// ---------------------------------------------------------------------------

/// Poker made-hand categories for diagnostic grouping.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
enum HandCategory {
    HighCard,
    Pair,
    TwoPair,
    Trips,
    Straight,
    Flush,
    FullHouse,
    Quads,
    StraightFlush,
}

impl HandCategory {
    #[allow(dead_code)]
    const ALL: [HandCategory; 9] = [
        Self::HighCard,
        Self::Pair,
        Self::TwoPair,
        Self::Trips,
        Self::Straight,
        Self::Flush,
        Self::FullHouse,
        Self::Quads,
        Self::StraightFlush,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::HighCard => "HighCard",
            Self::Pair => "Pair",
            Self::TwoPair => "TwoPair",
            Self::Trips => "Trips",
            Self::Straight => "Straight",
            Self::Flush => "Flush",
            Self::FullHouse => "FullHouse",
            Self::Quads => "Quads",
            Self::StraightFlush => "StraightFlush",
        }
    }
}

impl std::fmt::Display for HandCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(self.label())
    }
}

/// Classify a 7-card hand (2 hole cards + 5 board cards) into a `HandCategory`.
///
/// Reuses the inline evaluator from `datagen::range_gen` which encodes the
/// category in bits 26..28 of the returned score.
fn classify_made_hand(c1: u8, c2: u8, board: &[u8]) -> HandCategory {
    let score = cfvnet::datagen::range_gen::evaluate_7_slice(c1, c2, board);
    match score >> 26 {
        0 => HandCategory::HighCard,
        1 => HandCategory::Pair,
        2 => HandCategory::TwoPair,
        3 => HandCategory::Trips,
        4 => HandCategory::Straight,
        5 => HandCategory::Flush,
        6 => HandCategory::FullHouse,
        7 => HandCategory::Quads,
        8 => HandCategory::StraightFlush,
        other => panic!("unexpected hand category {other} from evaluator"),
    }
}

/// Returns `true` if the model path has an `.onnx` extension, indicating
/// that the ONNX inference path should be used instead of the Burn path.
fn is_onnx_model(path: &std::path::Path) -> bool {
    path.extension().map_or(false, |ext| ext == "onnx")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn is_onnx_model_detects_onnx_extension() {
        assert!(is_onnx_model(Path::new("model.onnx")));
        assert!(is_onnx_model(Path::new("/some/path/checkpoint_epoch675.onnx")));
    }

    #[test]
    fn is_onnx_model_rejects_non_onnx() {
        assert!(!is_onnx_model(Path::new("model.mpk.gz")));
        assert!(!is_onnx_model(Path::new("/some/dir/model")));
        assert!(!is_onnx_model(Path::new("model_dir/")));
        assert!(!is_onnx_model(Path::new("model.onnx.bak")));
    }

    #[test]
    fn is_onnx_model_handles_edge_cases() {
        assert!(!is_onnx_model(Path::new("")));
        assert!(!is_onnx_model(Path::new(".")));
        // ".onnx" is a hidden file with no extension on Unix, not an ONNX model
        assert!(!is_onnx_model(Path::new(".onnx")));
    }

    #[test]
    fn infer_board_cards_river() {
        let dir = std::env::temp_dir().join("cfvnet_test_infer_river");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_river.bin");
        // First byte = 5 (river)
        std::fs::write(&path, &[5u8]).unwrap();
        assert_eq!(infer_board_cards_from_data(&path), 5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn infer_board_cards_flop() {
        let dir = std::env::temp_dir().join("cfvnet_test_infer_flop");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_flop.bin");
        // First byte = 3 (flop)
        std::fs::write(&path, &[3u8]).unwrap();
        assert_eq!(infer_board_cards_from_data(&path), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn infer_board_cards_turn() {
        let dir = std::env::temp_dir().join("cfvnet_test_infer_turn");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_turn.bin");
        // First byte = 4 (turn)
        std::fs::write(&path, &[4u8]).unwrap();
        assert_eq!(infer_board_cards_from_data(&path), 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[should_panic(expected = "unexpected board_size")]
    fn infer_board_cards_rejects_invalid() {
        let dir = std::env::temp_dir().join("cfvnet_test_infer_invalid");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_bad.bin");
        // First byte = 0 (invalid)
        std::fs::write(&path, &[0u8]).unwrap();
        infer_board_cards_from_data(&path);
        // cleanup won't run due to panic, but temp dir is fine
    }

    #[test]
    fn range_stats_uniform_nonzero() {
        let range = [1.0_f32; 1326];
        let (density, entropy, top10, max_mean, total) = range_stats(&range);
        assert!((density - 1.0).abs() < 1e-9, "density={density}");
        let expected_entropy = (1326.0_f64).log2();
        assert!((entropy - expected_entropy).abs() < 0.01, "entropy={entropy}");
        let expected_top10 = 10.0 / 1326.0;
        assert!((top10 - expected_top10).abs() < 1e-6, "top10={top10}");
        assert!((max_mean - 1.0).abs() < 1e-9, "max_mean={max_mean}");
        assert!((total - 1326.0).abs() < 1e-6, "total={total}");
    }

    #[test]
    fn range_stats_all_zeros() {
        let range = [0.0_f32; 1326];
        let (density, entropy, top10, max_mean, total) = range_stats(&range);
        assert!((density).abs() < 1e-9);
        assert!((entropy).abs() < 1e-9);
        assert!((top10).abs() < 1e-9);
        assert!((max_mean).abs() < 1e-9);
        assert!((total).abs() < 1e-9);
    }

    #[test]
    fn range_stats_single_spike() {
        let mut range = [0.0_f32; 1326];
        range[500] = 100.0;
        let (density, entropy, top10, max_mean, total) = range_stats(&range);
        let expected_density = 1.0 / 1326.0;
        assert!((density - expected_density).abs() < 1e-9, "density={density}");
        assert!((entropy).abs() < 1e-9, "entropy={entropy}");
        assert!((top10 - 1.0).abs() < 1e-9, "top10={top10}");
        assert!((max_mean - 1.0).abs() < 1e-9, "max_mean={max_mean}");
        assert!((total - 100.0).abs() < 1e-6, "total={total}");
    }

    #[test]
    fn range_stats_partial_range() {
        // 100 entries at 2.0, rest zero
        let mut range = [0.0_f32; 1326];
        for i in 0..100 {
            range[i] = 2.0;
        }
        let (density, entropy, top10, max_mean, total) = range_stats(&range);
        let expected_density = 100.0 / 1326.0;
        assert!((density - expected_density).abs() < 1e-6, "density={density}");
        // Uniform over 100 entries: entropy = log2(100)
        let expected_entropy = (100.0_f64).log2();
        assert!((entropy - expected_entropy).abs() < 0.01, "entropy={entropy}");
        // Top 10 of 100 uniform entries = 10/100 = 0.1
        let expected_top10 = 10.0 / 100.0;
        assert!((top10 - expected_top10).abs() < 1e-6, "top10={top10}");
        assert!((max_mean - 1.0).abs() < 1e-9, "max_mean={max_mean}");
        assert!((total - 200.0).abs() < 1e-6, "total={total}");
    }

    #[test]
    fn classify_made_hand_known_hands() {
        // Card encoding: card_id = 4 * rank + suit
        // rank: 2→0, 3→1, ..., A→12; suit: c→0, d→1, h→2, s→3

        // TwoPair: As Ks on board [Ah Kh 7c 2d 3s] → AA KK
        assert_eq!(classify_made_hand(51, 47, &[50, 46, 20, 1, 7]), HandCategory::TwoPair);

        // HighCard: Ah Ks on board [9c 7d 5h 3c 2d]
        assert_eq!(classify_made_hand(50, 47, &[28, 21, 14, 4, 1]), HandCategory::HighCard);

        // Pair: As Kh on board [Ac 7h 5d 3c 2d] → pair of aces
        assert_eq!(classify_made_hand(51, 46, &[48, 22, 13, 4, 1]), HandCategory::Pair);

        // Trips: Ah Ad on board [As 7h 5c 2d 3s] → AAA
        assert_eq!(classify_made_hand(50, 49, &[51, 22, 12, 1, 7]), HandCategory::Trips);

        // Straight: 9h 8h on board [7c 6d 5s 2c 3d] → 9-high straight
        assert_eq!(classify_made_hand(30, 26, &[20, 17, 15, 0, 5]), HandCategory::Straight);

        // Flush: Ah Kh on board [Qh 9h 2h 3c 4d] → heart flush
        assert_eq!(classify_made_hand(50, 46, &[42, 30, 2, 4, 9]), HandCategory::Flush);

        // FullHouse: Ah Ad on board [As 7h 7c 2d 3s] → AAA 77
        assert_eq!(classify_made_hand(50, 49, &[51, 22, 20, 1, 7]), HandCategory::FullHouse);

        // Quads: Ah Ad on board [As Ac 7h 2d 3s] → AAAA
        assert_eq!(classify_made_hand(50, 49, &[51, 48, 22, 1, 7]), HandCategory::Quads);

        // StraightFlush: 9h 8h on board [7h 6h 5h 2c 3d] → 9-high straight flush
        assert_eq!(classify_made_hand(30, 26, &[22, 18, 14, 0, 5]), HandCategory::StraightFlush);
    }

    #[test]
    fn range_stats_spiky_distribution() {
        // One entry at 90, nine entries at 1 => top10 mass = 99/99 = 1.0
        // max/mean: nonzero count=10, total=99, mean=9.9, max=90 => ratio=90/9.9
        let mut range = [0.0_f32; 1326];
        range[0] = 90.0;
        for i in 1..10 {
            range[i] = 1.0;
        }
        let (density, _entropy, top10, max_mean, total) = range_stats(&range);
        let expected_density = 10.0 / 1326.0;
        assert!((density - expected_density).abs() < 1e-6, "density={density}");
        // All 10 entries are in top 10, so top10 concentration = 1.0
        assert!((top10 - 1.0).abs() < 1e-6, "top10={top10}");
        // max/mean = 90 / (99/10) = 90/9.9 ~= 9.0909
        let expected_ratio = 90.0 / 9.9;
        assert!((max_mean - expected_ratio).abs() < 0.001, "max_mean={max_mean}");
        assert!((total - 99.0).abs() < 1e-6, "total={total}");
    }
}
