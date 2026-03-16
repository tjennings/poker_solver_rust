mod blueprint_tui;
mod blueprint_tui_config;
mod blueprint_tui_metrics;
mod blueprint_tui_scenarios;
mod blueprint_tui_widgets;
mod log_file;

use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use poker_solver_core::blueprint_v2::config::BlueprintV2Config;
use poker_solver_core::blueprint_v2::trainer::BlueprintTrainer;

#[derive(Parser)]
#[command(name = "poker-solver-trainer")]
#[command(about = "Poker solver training tools: blueprint training, clustering, range solving")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Parser)]
enum Commands {
    /// Train a full-game blueprint strategy using MCCFR (Blueprint V2)
    TrainBlueprint {
        /// YAML config file (BlueprintV2Config)
        #[arg(short, long)]
        config: PathBuf,
        /// Disable the TUI dashboard even when tui.enabled is true in config
        #[arg(long)]
        no_tui: bool,
    },
    /// Run the clustering pipeline to build bucket assignments (Blueprint V2)
    Cluster {
        /// YAML config file (BlueprintV2Config — uses clustering section)
        #[arg(short, long)]
        config: PathBuf,
        /// Output directory for bucket files
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Show diagnostics for pre-computed cluster bucket files (Blueprint V2)
    DiagClusters {
        /// Directory containing .buckets files
        #[arg(short = 'd', long)]
        cluster_dir: PathBuf,
        /// Audit intra-bucket equity quality by sampling boards
        #[arg(long)]
        audit: bool,
        /// Number of boards to sample for audit (default 50)
        #[arg(long, default_value = "50")]
        audit_boards: usize,
        /// Print cross-street transition matrices for adjacent street pairs
        #[arg(long)]
        transitions: bool,
        /// Reconstruct centroids and print pairwise EMD report for a street
        #[arg(long)]
        centroid_emd: Option<String>,
        /// Show sample hands from a specific bucket (STREET BUCKET_ID)
        #[arg(long, num_args = 2, value_names = ["STREET", "BUCKET"])]
        sample_bucket: Option<Vec<String>>,
        /// Audit river bucket equity using cfvnet training data (path to cfvnet dir)
        #[arg(long)]
        cfvnet_audit: Option<PathBuf>,
    },
    /// Pre-compute the equity+delta lookup cache for fast expected-delta bucketing.
    /// Generates turn table first (averaging over river cards), then flop table
    /// (using turn table for two-street lookahead). Saves to a binary file.
    PrecomputeEquityDelta {
        /// Output file path for the cache
        #[arg(short, long, default_value = "cache/equity_delta.bin")]
        output: PathBuf,
    },
    /// Solve a postflop spot on GPU using DCFR+
    #[cfg(feature = "cuda")]
    GpuSolve {
        /// OOP player's range (PioSOLVER format, e.g. "QQ+,AKs,AKo")
        #[arg(long)]
        oop_range: String,
        /// IP player's range
        #[arg(long)]
        ip_range: String,
        /// Flop cards (e.g. "Qs Jh 2c")
        #[arg(long)]
        flop: String,
        /// Turn card (optional, e.g. "8d")
        #[arg(long)]
        turn: Option<String>,
        /// River card (optional, e.g. "3s")
        #[arg(long)]
        river: Option<String>,
        /// Starting pot size
        #[arg(long, default_value = "100")]
        pot: i32,
        /// Effective stack size
        #[arg(long, default_value = "100")]
        effective_stack: i32,
        /// Maximum iterations
        #[arg(long, default_value = "1000")]
        iterations: u32,
        /// Target exploitability (stops early if reached)
        #[arg(long, default_value = "0.5")]
        target_exploitability: f32,
        /// OOP bet sizes (comma-separated, e.g. "50%,100%,a")
        #[arg(long, default_value = "50%,100%")]
        oop_bet_sizes: String,
        /// OOP raise sizes
        #[arg(long, default_value = "60%,100%")]
        oop_raise_sizes: String,
        /// IP bet sizes
        #[arg(long, default_value = "50%,100%")]
        ip_bet_sizes: String,
        /// IP raise sizes
        #[arg(long, default_value = "60%,100%")]
        ip_raise_sizes: String,
    },
    /// Train a river CFVNet on GPU using Supremus-style pipeline
    #[cfg(feature = "gpu-training")]
    GpuTrainRiver {
        /// Total training samples to generate
        #[arg(long, default_value_t = 50_000_000)]
        num_samples: u64,
        /// DCFR+ iterations per solve batch
        #[arg(long, default_value_t = 4000)]
        solve_iterations: u32,
        /// River spots per solve batch
        #[arg(long, default_value_t = 1000)]
        batch_size: usize,
        /// GPU reservoir capacity (max training records)
        #[arg(long, default_value_t = 100_000)]
        reservoir_capacity: usize,
        /// Number of hidden layers
        #[arg(long, default_value_t = 7)]
        hidden_layers: usize,
        /// Hidden layer width
        #[arg(long, default_value_t = 500)]
        hidden_size: usize,
        /// Mini-batch size for training
        #[arg(long, default_value_t = 1024)]
        train_batch_size: usize,
        /// Training steps per solve batch
        #[arg(long, default_value_t = 10)]
        train_steps_per_batch: usize,
        /// Learning rate
        #[arg(long, default_value_t = 0.001)]
        learning_rate: f64,
        /// Huber loss delta
        #[arg(long, default_value_t = 1.0)]
        huber_delta: f64,
        /// Auxiliary game-value loss weight
        #[arg(long, default_value_t = 1.0)]
        aux_loss_weight: f64,
        /// Validation reporting interval (samples)
        #[arg(long, default_value_t = 10_000)]
        validation_interval: u64,
        /// Checkpoint save interval (samples)
        #[arg(long, default_value_t = 1_000_000)]
        checkpoint_interval: u64,
        /// Number of ground-truth validation positions
        #[arg(long, default_value_t = 100)]
        gt_positions: usize,
        /// Iterations for ground-truth solves
        #[arg(long, default_value_t = 10_000)]
        gt_iterations: u32,
        /// Output directory for model and checkpoints
        #[arg(long)]
        output: PathBuf,
        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Reference pot size
        #[arg(long, default_value_t = 100)]
        ref_pot: i32,
        /// Reference effective stack
        #[arg(long, default_value_t = 100)]
        ref_stack: i32,
        /// Use persistent mega-kernel (single launch, eliminates kernel launch overhead)
        #[arg(long, default_value_t = false)]
        persistent_kernel: bool,
    },
    /// Train a turn CFVNet on GPU using trained river model as leaf evaluator
    #[cfg(feature = "gpu-training")]
    GpuTrainTurn {
        /// Path to trained river model (without .mpk.gz extension)
        #[arg(long)]
        river_model: PathBuf,
        /// Number of hidden layers in the river model (must match trained model)
        #[arg(long, default_value_t = 7)]
        river_hidden_layers: usize,
        /// Hidden size of the river model (must match trained model)
        #[arg(long, default_value_t = 500)]
        river_hidden_size: usize,
        /// Total training samples to generate
        #[arg(long, default_value_t = 20_000_000)]
        num_samples: u64,
        /// DCFR+ iterations per solve batch
        #[arg(long, default_value_t = 4000)]
        solve_iterations: u32,
        /// Turn spots per solve batch
        #[arg(long, default_value_t = 1000)]
        batch_size: usize,
        /// GPU reservoir capacity (max training records)
        #[arg(long, default_value_t = 100_000)]
        reservoir_capacity: usize,
        /// Number of hidden layers in the turn model
        #[arg(long, default_value_t = 7)]
        hidden_layers: usize,
        /// Hidden layer width for the turn model
        #[arg(long, default_value_t = 500)]
        hidden_size: usize,
        /// Mini-batch size for training
        #[arg(long, default_value_t = 1024)]
        train_batch_size: usize,
        /// Training steps per solve batch
        #[arg(long, default_value_t = 10)]
        train_steps_per_batch: usize,
        /// Learning rate
        #[arg(long, default_value_t = 0.001)]
        learning_rate: f64,
        /// Huber loss delta
        #[arg(long, default_value_t = 1.0)]
        huber_delta: f64,
        /// Auxiliary game-value loss weight
        #[arg(long, default_value_t = 1.0)]
        aux_loss_weight: f64,
        /// Validation reporting interval (samples)
        #[arg(long, default_value_t = 10_000)]
        validation_interval: u64,
        /// Checkpoint save interval (samples)
        #[arg(long, default_value_t = 1_000_000)]
        checkpoint_interval: u64,
        /// Output directory for model and checkpoints
        #[arg(long)]
        output: PathBuf,
        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Reference pot size
        #[arg(long, default_value_t = 100)]
        ref_pot: i32,
        /// Reference effective stack
        #[arg(long, default_value_t = 100)]
        ref_stack: i32,
    },
    /// Train a flop CFVNet on GPU using trained turn model as leaf evaluator
    #[cfg(feature = "gpu-training")]
    GpuTrainFlop {
        /// Path to trained turn model (without .mpk.gz extension)
        #[arg(long)]
        turn_model: PathBuf,
        /// Number of hidden layers in the turn model (must match trained model)
        #[arg(long, default_value_t = 7)]
        turn_hidden_layers: usize,
        /// Hidden size of the turn model (must match trained model)
        #[arg(long, default_value_t = 500)]
        turn_hidden_size: usize,
        /// Total training samples to generate
        #[arg(long, default_value_t = 20_000_000)]
        num_samples: u64,
        /// DCFR+ iterations per solve batch
        #[arg(long, default_value_t = 4000)]
        solve_iterations: u32,
        /// Flop spots per solve batch
        #[arg(long, default_value_t = 1000)]
        batch_size: usize,
        /// GPU reservoir capacity (max training records)
        #[arg(long, default_value_t = 100_000)]
        reservoir_capacity: usize,
        /// Number of hidden layers in the flop model
        #[arg(long, default_value_t = 7)]
        hidden_layers: usize,
        /// Hidden layer width for the flop model
        #[arg(long, default_value_t = 500)]
        hidden_size: usize,
        /// Mini-batch size for training
        #[arg(long, default_value_t = 1024)]
        train_batch_size: usize,
        /// Training steps per solve batch
        #[arg(long, default_value_t = 10)]
        train_steps_per_batch: usize,
        /// Learning rate
        #[arg(long, default_value_t = 0.001)]
        learning_rate: f64,
        /// Huber loss delta
        #[arg(long, default_value_t = 1.0)]
        huber_delta: f64,
        /// Auxiliary game-value loss weight
        #[arg(long, default_value_t = 1.0)]
        aux_loss_weight: f64,
        /// Validation reporting interval (samples)
        #[arg(long, default_value_t = 10_000)]
        validation_interval: u64,
        /// Checkpoint save interval (samples)
        #[arg(long, default_value_t = 1_000_000)]
        checkpoint_interval: u64,
        /// Output directory for model and checkpoints
        #[arg(long)]
        output: PathBuf,
        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Reference pot size
        #[arg(long, default_value_t = 100)]
        ref_pot: i32,
        /// Reference effective stack
        #[arg(long, default_value_t = 100)]
        ref_stack: i32,
    },
    /// Train a preflop auxiliary CFVNet on GPU using trained flop model (inference-only, no CFR)
    #[cfg(feature = "gpu-training")]
    GpuTrainPreflop {
        /// Path to trained flop model (without .mpk.gz extension)
        #[arg(long)]
        flop_model: PathBuf,
        /// Number of hidden layers in the flop model (must match trained model)
        #[arg(long, default_value_t = 7)]
        flop_hidden_layers: usize,
        /// Hidden size of the flop model (must match trained model)
        #[arg(long, default_value_t = 500)]
        flop_hidden_size: usize,
        /// Total training samples to generate
        #[arg(long, default_value_t = 10_000_000)]
        num_samples: u64,
        /// Spots per inference batch (limited by VRAM: 22100 flops per spot)
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        /// GPU reservoir capacity (max training records)
        #[arg(long, default_value_t = 100_000)]
        reservoir_capacity: usize,
        /// Number of hidden layers in the preflop model
        #[arg(long, default_value_t = 7)]
        hidden_layers: usize,
        /// Hidden layer width for the preflop model
        #[arg(long, default_value_t = 500)]
        hidden_size: usize,
        /// Mini-batch size for training
        #[arg(long, default_value_t = 1024)]
        train_batch_size: usize,
        /// Training steps per sample batch
        #[arg(long, default_value_t = 10)]
        train_steps_per_batch: usize,
        /// Learning rate
        #[arg(long, default_value_t = 0.001)]
        learning_rate: f64,
        /// Huber loss delta
        #[arg(long, default_value_t = 1.0)]
        huber_delta: f64,
        /// Auxiliary game-value loss weight
        #[arg(long, default_value_t = 1.0)]
        aux_loss_weight: f64,
        /// Validation reporting interval (samples)
        #[arg(long, default_value_t = 10_000)]
        validation_interval: u64,
        /// Checkpoint save interval (samples)
        #[arg(long, default_value_t = 1_000_000)]
        checkpoint_interval: u64,
        /// Output directory for model and checkpoints
        #[arg(long)]
        output: PathBuf,
        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Reference pot size
        #[arg(long, default_value_t = 100)]
        ref_pot: i32,
        /// Reference effective stack
        #[arg(long, default_value_t = 100)]
        ref_stack: i32,
    },
    /// Train the full CFVNet model stack (river → turn → flop → preflop) from one YAML config
    #[cfg(feature = "gpu-training")]
    GpuTrainStack {
        /// YAML config file (GpuTrainingStackConfig)
        #[arg(short, long)]
        config: PathBuf,
        /// Output directory (creates river/, turn/, flop/, preflop/ subdirs)
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Solve a postflop spot with exact (no abstraction) DCFR
    RangeSolve {
        /// OOP player's range (PioSOLVER format, e.g. "QQ+,AKs,AKo")
        #[arg(long)]
        oop_range: String,
        /// IP player's range
        #[arg(long)]
        ip_range: String,
        /// Flop cards (e.g. "Qs Jh 2c")
        #[arg(long)]
        flop: String,
        /// Turn card (optional, e.g. "8d")
        #[arg(long)]
        turn: Option<String>,
        /// River card (optional, e.g. "3s")
        #[arg(long)]
        river: Option<String>,
        /// Starting pot size
        #[arg(long, default_value = "100")]
        pot: i32,
        /// Effective stack size
        #[arg(long, default_value = "100")]
        effective_stack: i32,
        /// Maximum iterations
        #[arg(long, default_value = "1000")]
        iterations: u32,
        /// Target exploitability (stops early if reached)
        #[arg(long, default_value = "0.5")]
        target_exploitability: f32,
        /// OOP bet sizes (comma-separated, e.g. "50%,100%,a")
        #[arg(long, default_value = "50%,100%")]
        oop_bet_sizes: String,
        /// OOP raise sizes
        #[arg(long, default_value = "60%,100%")]
        oop_raise_sizes: String,
        /// IP bet sizes
        #[arg(long, default_value = "50%,100%")]
        ip_bet_sizes: String,
        /// IP raise sizes
        #[arg(long, default_value = "60%,100%")]
        ip_raise_sizes: String,
        /// Use 16-bit compressed storage
        #[arg(long)]
        compressed: bool,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::TrainBlueprint { config, no_tui } => {
            let yaml = std::fs::read_to_string(&config)?;
            let bp_config: BlueprintV2Config = serde_yaml::from_str(&yaml)?;
            let tui_config = blueprint_tui_config::parse_tui_config(&yaml);

            eprintln!("Blueprint V2 Training");
            eprintln!("  Stack: {}BB", bp_config.game.stack_depth);
            eprintln!(
                "  Buckets: preflop={}, flop={}, turn={}, river={}",
                bp_config.clustering.preflop.buckets,
                bp_config.clustering.flop.buckets,
                bp_config.clustering.turn.buckets,
                bp_config.clustering.river.buckets,
            );
            eprintln!("  Actions: preflop_depths={} flop_depths={} turn_depths={} river_depths={}",
                bp_config.action_abstraction.preflop.len(),
                bp_config.action_abstraction.flop.len(),
                bp_config.action_abstraction.turn.len(),
                bp_config.action_abstraction.river.len(),
            );
            if let Some(iters) = bp_config.training.iterations {
                eprintln!("  Iterations: {iters}");
            }
            if let Some(mins) = bp_config.training.time_limit_minutes {
                eprintln!("  Time limit: {mins} min");
            }
            eprintln!();

            let mut trainer = BlueprintTrainer::new(bp_config);
            trainer.try_resume()?;
            let use_tui = tui_config.enabled && !no_tui;

            if use_tui {
                let metrics = Arc::new(blueprint_tui_metrics::BlueprintTuiMetrics::new(
                    trainer.config.training.iterations,
                    trainer.config.training.time_limit_minutes,
                ));

                // Share atomics between trainer and TUI.
                trainer.paused = Arc::clone(&metrics.paused);
                trainer.quit_requested = Arc::clone(&metrics.quit_requested);
                trainer.shared_iterations = Arc::clone(&metrics.iterations);
                trainer.snapshot_trigger = Arc::clone(&metrics.snapshot_trigger);

                // Pre-compute board cards for each scenario.
                let scenario_boards: Vec<Vec<poker_solver_core::poker::Card>> = tui_config
                    .scenarios
                    .iter()
                    .map(|sc| {
                        sc.board
                            .as_ref()
                            .map(|strings| {
                                strings
                                    .iter()
                                    .filter_map(|s| {
                                        poker_solver_core::poker::parse_card(s)
                                    })
                                    .collect()
                            })
                            .unwrap_or_default()
                    })
                    .collect();

                // Resolve scenarios to game-tree nodes.
                let scenarios: Vec<blueprint_tui::ResolvedScenario> = tui_config
                    .scenarios
                    .iter()
                    .enumerate()
                    .map(|(i, sc)| {
                        let node_idx = blueprint_tui_scenarios::resolve_action_path(
                            &trainer.tree,
                            &sc.actions,
                        )
                        .unwrap_or(trainer.tree.root);
                        let grid = blueprint_tui_scenarios::extract_strategy_grid(
                            &trainer.tree,
                            &trainer.storage,
                            node_idx,
                            &scenario_boards[i],
                        );
                        blueprint_tui::ResolvedScenario {
                            name: sc.name.clone(),
                            node_idx,
                            grid: blueprint_tui_widgets::HandGridState {
                                cells: grid,
                                prev_cells: None,
                                scenario_name: sc.name.clone(),
                                action_path: sc.actions.clone(),
                                board_display: sc
                                    .board
                                    .as_ref()
                                    .map(|b| b.join(" ")),
                                cluster_id: None,
                                street_label: sc
                                    .street
                                    .map_or("Preflop".into(), |s| format!("{s:?}")),
                                iteration_at_snapshot: 0,
                            },
                        }
                    })
                    .collect();

                // Wire strategy refresh callback from trainer to TUI metrics.
                let scenarios_node_indices: Vec<u32> =
                    scenarios.iter().map(|s| s.node_idx).collect();
                trainer.scenario_node_indices = scenarios_node_indices;
                trainer.strategy_refresh_interval_secs =
                    tui_config.telemetry.strategy_delta_interval_seconds;

                let metrics_for_refresh = Arc::clone(&metrics);
                let boards_for_refresh = scenario_boards;
                trainer.on_strategy_refresh =
                    Some(Box::new(move |scenario_idx, node_idx, storage, tree| {
                        let board = &boards_for_refresh[scenario_idx];
                        let grid = blueprint_tui_scenarios::extract_strategy_grid(
                            tree, storage, node_idx, board,
                        );
                        metrics_for_refresh.update_scenario_grid(scenario_idx, grid);
                    }));

                let metrics_for_delta = Arc::clone(&metrics);
                trainer.on_strategy_delta = Some(Box::new(move |delta| {
                    metrics_for_delta.push_strategy_delta(delta);
                }));

                let metrics_for_leaf = Arc::clone(&metrics);
                trainer.on_leaf_movement = Some(Box::new(move |pct| {
                    metrics_for_leaf.push_leaf_movement(pct);
                }));

                let metrics_for_regret = Arc::clone(&metrics);
                trainer.on_min_regret = Some(Box::new(move |val| {
                    metrics_for_regret.push_min_regret(val);
                }));

                let metrics_for_max_regret = Arc::clone(&metrics);
                trainer.on_max_regret = Some(Box::new(move |val| {
                    metrics_for_max_regret.push_max_regret(val);
                }));

                let metrics_for_avg_regret = Arc::clone(&metrics);
                trainer.on_avg_pos_regret = Some(Box::new(move |val| {
                    metrics_for_avg_regret.push_avg_pos_regret(val);
                }));

                let metrics_for_prune = Arc::clone(&metrics);
                trainer.on_prune_fraction = Some(Box::new(move |frac| {
                    metrics_for_prune.push_prune_fraction(frac);
                }));

                // Random scenario carousel.
                if tui_config.random_scenario.enabled {
                    trainer.random_scenario_hold_minutes =
                        tui_config.random_scenario.hold_minutes;
                    let metrics_for_random = Arc::clone(&metrics);
                    let pool = tui_config.random_scenario.pool.clone();
                    trainer.on_random_scenario =
                        Some(Box::new(move |storage, tree| {
                            use rand::seq::IndexedRandom;
                            let mut rng = rand::rng();

                            let Some(street_label) = pool.choose(&mut rng) else {
                                return;
                            };
                            let street = match street_label {
                                blueprint_tui_config::StreetLabel::Preflop => {
                                    poker_solver_core::blueprint_v2::Street::Preflop
                                }
                                blueprint_tui_config::StreetLabel::Flop => {
                                    poker_solver_core::blueprint_v2::Street::Flop
                                }
                                blueprint_tui_config::StreetLabel::Turn => {
                                    poker_solver_core::blueprint_v2::Street::Turn
                                }
                                blueprint_tui_config::StreetLabel::River => {
                                    poker_solver_core::blueprint_v2::Street::River
                                }
                            };

                            let candidates =
                                blueprint_tui_scenarios::decision_nodes_at_street(tree, street);
                            let Some(&node_idx) = candidates.choose(&mut rng) else {
                                return;
                            };

                            let board =
                                blueprint_tui_scenarios::random_board(street, &mut rng);
                            let board_display = if board.is_empty() {
                                String::new()
                            } else {
                                board
                                    .iter()
                                    .map(|c| format!("{c}"))
                                    .collect::<Vec<_>>()
                                    .join(" ")
                            };

                            let grid = blueprint_tui_scenarios::extract_strategy_grid(
                                tree, storage, node_idx, &board,
                            );

                            let name = blueprint_tui_scenarios::random_scenario_name(
                                tree, node_idx, &board_display,
                            );

                            let street_label_str = format!("{street:?}");

                            metrics_for_random.update_random_scenario(
                                name, node_idx, grid, board_display, street_label_str,
                            );
                        }));
                }

                trainer.tui_active = true;

                let refresh = Duration::from_millis(tui_config.refresh_rate_ms);
                let tui_handle = blueprint_tui::run_blueprint_tui(
                    Arc::clone(&metrics),
                    scenarios,
                    tui_config.telemetry.clone(),
                    refresh,
                );

                trainer.train()?;
                metrics
                    .quit_requested
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                let _ = tui_handle.join();
            } else {
                trainer.train()?;
            }

            eprintln!("\nTraining complete: {} iterations", trainer.iterations);
        }
        Commands::Cluster { config, output } => {
            let yaml = std::fs::read_to_string(&config)?;
            let bp_config: BlueprintV2Config = serde_yaml::from_str(&yaml)?;

            std::fs::create_dir_all(&output)?;

            eprintln!("Blueprint V2 Clustering Pipeline");
            eprintln!("  Output: {}", output.display());
            eprintln!(
                "  Buckets: preflop={}, flop={}, turn={}, river={}",
                bp_config.clustering.preflop.buckets,
                bp_config.clustering.flop.buckets,
                bp_config.clustering.turn.buckets,
                bp_config.clustering.river.buckets,
            );
            eprintln!();

            if poker_solver_core::blueprint_v2::trainer::bucket_files_exist(&output) {
                eprintln!("Bucket files already exist in {}, skipping clustering", output.display());
            } else {
                let pb = ProgressBar::new(1000);
                pb.set_style(
                    ProgressStyle::with_template("  [{msg}] {bar:40.cyan/blue} {pos}/{len}")
                        .unwrap()
                        .progress_chars("##-"),
                );
                let current_street = std::sync::Mutex::new(String::new());

                poker_solver_core::blueprint_v2::cluster_pipeline::run_clustering_pipeline(
                    &bp_config.clustering,
                    &output,
                    |street, p| {
                        let mut cur = current_street.lock().unwrap();
                        if *cur != street {
                            if !cur.is_empty() {
                                pb.finish_with_message(format!("{cur} done"));
                                eprintln!();
                            }
                            *cur = street.to_string();
                            pb.reset();
                        }
                        drop(cur);

                        let phase = if p < 0.8 { "features" } else { "k-means" };
                        pb.set_message(format!("{street} {phase}"));
                        pb.set_position((p * 1000.0) as u64);
                    },
                )?;

                pb.finish_with_message("done");
                eprintln!();
                eprintln!("Clustering complete. Files saved to {}", output.display());
            }
        }
        Commands::DiagClusters {
            cluster_dir,
            audit,
            audit_boards,
            transitions,
            centroid_emd,
            sample_bucket,
            cfvnet_audit,
        } => {
            use poker_solver_core::blueprint_v2::bucket_file::BucketFile;
            use poker_solver_core::blueprint_v2::cluster_diagnostics::{
                cross_street_transition_matrix, sample_hands_for_bucket,
            };

            let reports = poker_solver_core::blueprint_v2::cluster_diagnostics::diagnose_cluster_dir(&cluster_dir)?;
            if reports.is_empty() {
                eprintln!("No .buckets files found in {}", cluster_dir.display());
            } else {
                for report in &reports {
                    eprintln!("{}", report.summary());
                }
            }
            if audit {
                eprintln!("\nEquity audit ({audit_boards} sample boards per street)...");
                let audit_reports = poker_solver_core::blueprint_v2::cluster_diagnostics::audit_cluster_dir(
                    &cluster_dir,
                    audit_boards,
                    42,
                )?;
                for report in &audit_reports {
                    eprintln!("\n{}", report.summary());
                }
            }
            if let Some(ref cfvnet_dir) = cfvnet_audit {
                eprintln!("\nCFVnet river equity audit (sampling every 10th record)...");
                let river_path = cluster_dir.join("river.buckets");
                if !river_path.exists() {
                    eprintln!("No river.buckets found in {}", cluster_dir.display());
                } else {
                    let bf = BucketFile::load(&river_path)?;
                    let report = poker_solver_core::blueprint_v2::cluster_diagnostics::audit_cfvnet_buckets(
                        cfvnet_dir,
                        &bf,
                        10,
                        |p| eprint!("\r  progress: {:.0}%", p * 100.0),
                    )?;
                    eprintln!("\r                    ");
                    eprintln!("{}", report.summary());
                }
            }
            if transitions {
                let pairs = [("preflop", "flop"), ("flop", "turn"), ("turn", "river")];
                for (from_name, to_name) in &pairs {
                    let from_path = cluster_dir.join(format!("{from_name}.buckets"));
                    let to_path = cluster_dir.join(format!("{to_name}.buckets"));
                    if from_path.exists() && to_path.exists() {
                        let from_bf = BucketFile::load(&from_path)?;
                        let to_bf = BucketFile::load(&to_path)?;
                        let matrix = cross_street_transition_matrix(&from_bf, &to_bf);
                        eprintln!("\n{}", matrix.summary());
                    }
                }
            }
            if let Some(_street) = centroid_emd {
                eprintln!(
                    "\nCentroid EMD requires feature vectors which are not stored \
                     in bucket files — run during clustering to capture this diagnostic."
                );
            }
            if let Some(ref args) = sample_bucket {
                let street_str = &args[0];
                let bucket_id: u16 = args[1].parse().map_err(|e| {
                    format!("invalid bucket id '{}': {e}", args[1])
                })?;
                let path = cluster_dir.join(format!("{street_str}.buckets"));
                if !path.exists() {
                    eprintln!("Bucket file not found: {}", path.display());
                } else {
                    let bf = BucketFile::load(&path)?;
                    let samples = sample_hands_for_bucket(&bf, bucket_id, 10, 42);
                    if samples.is_empty() {
                        eprintln!("No entries found for bucket {bucket_id} in {street_str}");
                    } else {
                        eprintln!(
                            "{} sample(s) from {street_str} bucket {bucket_id}:",
                            samples.len()
                        );
                        for sample in &samples {
                            eprintln!("  {}", sample.display());
                        }
                    }
                }
            }
        }
        Commands::PrecomputeEquityDelta { output } => {
            use poker_solver_core::blueprint_v2::equity_cache::EquityDeltaCache;

            if output.exists() {
                eprintln!("Cache already exists at {}, nothing to do", output.display());
                return Ok(());
            }

            eprintln!("Pre-computing equity+delta cache...");
            eprintln!("  Output: {}", output.display());
            eprintln!();

            let pb = ProgressBar::new(10000);
            pb.set_style(
                ProgressStyle::with_template("  [{msg}] {bar:40.cyan/blue} {pos}/10000 ({eta})")
                    .unwrap()
                    .progress_chars("##-"),
            );
            pb.enable_steady_tick(Duration::from_millis(200));
            let current_street = std::sync::Mutex::new(String::new());

            let start = Instant::now();
            let cache = EquityDeltaCache::generate(|street, frac| {
                let mut cur = current_street.lock().unwrap();
                if *cur != street {
                    if !cur.is_empty() {
                        pb.finish_with_message(format!("{cur} done"));
                        eprintln!();
                    }
                    *cur = street.to_string();
                    pb.reset();
                }
                drop(cur);
                pb.set_message(street.to_string());
                pb.set_position((frac * 10000.0) as u64);
            });

            pb.finish_with_message("done");
            eprintln!();

            let elapsed = start.elapsed();
            eprintln!(
                "Generated in {:.1}s: turn={} entries, flop={} entries",
                elapsed.as_secs_f64(),
                cache.turn_entries(),
                cache.flop_entries(),
            );

            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            cache.save(&output)?;
            eprintln!("Saved to {}", output.display());
        }
        #[cfg(feature = "gpu-training")]
        Commands::GpuTrainRiver {
            num_samples,
            solve_iterations,
            batch_size,
            reservoir_capacity,
            hidden_layers,
            hidden_size,
            train_batch_size,
            train_steps_per_batch,
            learning_rate,
            huber_delta,
            aux_loss_weight,
            validation_interval,
            checkpoint_interval,
            gt_positions,
            gt_iterations,
            output,
            seed,
            ref_pot,
            ref_stack,
            persistent_kernel,
        } => {
            run_gpu_train_river(
                num_samples,
                solve_iterations,
                batch_size,
                reservoir_capacity,
                hidden_layers,
                hidden_size,
                train_batch_size,
                train_steps_per_batch,
                learning_rate,
                huber_delta,
                aux_loss_weight,
                validation_interval,
                checkpoint_interval,
                gt_positions,
                gt_iterations,
                output,
                seed,
                ref_pot,
                ref_stack,
                persistent_kernel,
            )?;
        }
        #[cfg(feature = "gpu-training")]
        Commands::GpuTrainTurn {
            river_model,
            river_hidden_layers,
            river_hidden_size,
            num_samples,
            solve_iterations,
            batch_size,
            reservoir_capacity,
            hidden_layers,
            hidden_size,
            train_batch_size,
            train_steps_per_batch,
            learning_rate,
            huber_delta,
            aux_loss_weight,
            validation_interval,
            checkpoint_interval,
            output,
            seed,
            ref_pot,
            ref_stack,
        } => {
            run_gpu_train_turn(
                river_model,
                river_hidden_layers,
                river_hidden_size,
                num_samples,
                solve_iterations,
                batch_size,
                reservoir_capacity,
                hidden_layers,
                hidden_size,
                train_batch_size,
                train_steps_per_batch,
                learning_rate,
                huber_delta,
                aux_loss_weight,
                validation_interval,
                checkpoint_interval,
                output,
                seed,
                ref_pot,
                ref_stack,
            )?;
        }
        #[cfg(feature = "gpu-training")]
        Commands::GpuTrainFlop {
            turn_model,
            turn_hidden_layers,
            turn_hidden_size,
            num_samples,
            solve_iterations,
            batch_size,
            reservoir_capacity,
            hidden_layers,
            hidden_size,
            train_batch_size,
            train_steps_per_batch,
            learning_rate,
            huber_delta,
            aux_loss_weight,
            validation_interval,
            checkpoint_interval,
            output,
            seed,
            ref_pot,
            ref_stack,
        } => {
            run_gpu_train_flop(
                turn_model,
                turn_hidden_layers,
                turn_hidden_size,
                num_samples,
                solve_iterations,
                batch_size,
                reservoir_capacity,
                hidden_layers,
                hidden_size,
                train_batch_size,
                train_steps_per_batch,
                learning_rate,
                huber_delta,
                aux_loss_weight,
                validation_interval,
                checkpoint_interval,
                output,
                seed,
                ref_pot,
                ref_stack,
            )?;
        }
        #[cfg(feature = "gpu-training")]
        Commands::GpuTrainStack { config, output } => {
            run_gpu_train_stack(config, output)?;
        }
        #[cfg(feature = "gpu-training")]
        Commands::GpuTrainPreflop {
            flop_model,
            flop_hidden_layers,
            flop_hidden_size,
            num_samples,
            batch_size,
            reservoir_capacity,
            hidden_layers,
            hidden_size,
            train_batch_size,
            train_steps_per_batch,
            learning_rate,
            huber_delta,
            aux_loss_weight,
            validation_interval,
            checkpoint_interval,
            output,
            seed,
            ref_pot,
            ref_stack,
        } => {
            run_gpu_train_preflop(
                flop_model,
                flop_hidden_layers,
                flop_hidden_size,
                num_samples,
                batch_size,
                reservoir_capacity,
                hidden_layers,
                hidden_size,
                train_batch_size,
                train_steps_per_batch,
                learning_rate,
                huber_delta,
                aux_loss_weight,
                validation_interval,
                checkpoint_interval,
                output,
                seed,
                ref_pot,
                ref_stack,
            )?;
        }
        #[cfg(feature = "cuda")]
        Commands::GpuSolve {
            oop_range,
            ip_range,
            flop,
            turn,
            river,
            pot,
            effective_stack,
            iterations,
            target_exploitability,
            oop_bet_sizes,
            oop_raise_sizes,
            ip_bet_sizes,
            ip_raise_sizes,
        } => {
            run_gpu_solve(
                &oop_range,
                &ip_range,
                &flop,
                turn.as_deref(),
                river.as_deref(),
                pot,
                effective_stack,
                iterations,
                target_exploitability,
                &oop_bet_sizes,
                &oop_raise_sizes,
                &ip_bet_sizes,
                &ip_raise_sizes,
            )?;
        }
        Commands::RangeSolve {
            oop_range,
            ip_range,
            flop,
            turn,
            river,
            pot,
            effective_stack,
            iterations,
            target_exploitability,
            oop_bet_sizes,
            oop_raise_sizes,
            ip_bet_sizes,
            ip_raise_sizes,
            compressed,
        } => {
            run_range_solve(
                &oop_range,
                &ip_range,
                &flop,
                turn.as_deref(),
                river.as_deref(),
                pot,
                effective_stack,
                iterations,
                target_exploitability,
                &oop_bet_sizes,
                &oop_raise_sizes,
                &ip_bet_sizes,
                &ip_raise_sizes,
                compressed,
            )?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// GPU river CFVNet training
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu-training")]
#[allow(clippy::too_many_arguments)]
fn run_gpu_train_river(
    num_samples: u64,
    solve_iterations: u32,
    batch_size: usize,
    reservoir_capacity: usize,
    hidden_layers: usize,
    hidden_size: usize,
    train_batch_size: usize,
    train_steps_per_batch: usize,
    learning_rate: f64,
    huber_delta: f64,
    aux_loss_weight: f64,
    validation_interval: u64,
    checkpoint_interval: u64,
    gt_positions: usize,
    gt_iterations: u32,
    output: PathBuf,
    seed: u64,
    ref_pot: i32,
    ref_stack: i32,
    persistent_kernel: bool,
) -> Result<(), Box<dyn Error>> {
    use burn::backend::{Autodiff, NdArray};
    use poker_solver_gpu::training::pipeline::{train_river_cfvnet, RiverTrainingConfig};

    // Use NdArray backend by default. When burn-cuda is available,
    // this can be switched to Autodiff<CudaJit<f32>>.
    type B = Autodiff<NdArray>;

    let config = RiverTrainingConfig {
        num_samples,
        solve_iterations,
        batch_size,
        reservoir_capacity,
        hidden_layers,
        hidden_size,
        train_batch_size,
        train_steps_per_batch,
        learning_rate,
        huber_delta,
        aux_loss_weight,
        validation_interval,
        checkpoint_interval,
        gt_validation_positions: gt_positions,
        gt_solve_iterations: gt_iterations,
        output_dir: output,
        seed,
        ref_pot,
        ref_stack,
        use_persistent_kernel: persistent_kernel,
    };

    let device = Default::default();
    train_river_cfvnet::<B>(&config, &device)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// GPU turn CFVNet training
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu-training")]
#[allow(clippy::too_many_arguments)]
fn run_gpu_train_turn(
    river_model: PathBuf,
    river_hidden_layers: usize,
    river_hidden_size: usize,
    num_samples: u64,
    solve_iterations: u32,
    batch_size: usize,
    reservoir_capacity: usize,
    hidden_layers: usize,
    hidden_size: usize,
    train_batch_size: usize,
    train_steps_per_batch: usize,
    learning_rate: f64,
    huber_delta: f64,
    aux_loss_weight: f64,
    validation_interval: u64,
    checkpoint_interval: u64,
    output: PathBuf,
    seed: u64,
    ref_pot: i32,
    ref_stack: i32,
) -> Result<(), Box<dyn Error>> {
    use burn::backend::{Autodiff, NdArray};
    use poker_solver_gpu::training::turn_pipeline::{train_turn_cfvnet_cuda, TurnTrainingConfig};

    type B = Autodiff<NdArray>;

    let config = TurnTrainingConfig {
        river_model_path: river_model,
        river_hidden_layers,
        river_hidden_size,
        num_samples,
        solve_iterations,
        batch_size,
        reservoir_capacity,
        hidden_layers,
        hidden_size,
        train_batch_size,
        train_steps_per_batch,
        learning_rate,
        huber_delta,
        aux_loss_weight,
        validation_interval,
        checkpoint_interval,
        gt_validation_positions: 0, // not implemented for turn yet
        gt_solve_iterations: 0,
        output_dir: output,
        seed,
        ref_pot,
        ref_stack,
    };

    let device = Default::default();
    train_turn_cfvnet_cuda::<B>(&config, &device)?;
    Ok(())
}

#[cfg(feature = "gpu-training")]
#[allow(clippy::too_many_arguments)]
fn run_gpu_train_flop(
    turn_model: PathBuf,
    turn_hidden_layers: usize,
    turn_hidden_size: usize,
    num_samples: u64,
    solve_iterations: u32,
    batch_size: usize,
    reservoir_capacity: usize,
    hidden_layers: usize,
    hidden_size: usize,
    train_batch_size: usize,
    train_steps_per_batch: usize,
    learning_rate: f64,
    huber_delta: f64,
    aux_loss_weight: f64,
    validation_interval: u64,
    checkpoint_interval: u64,
    output: PathBuf,
    seed: u64,
    ref_pot: i32,
    ref_stack: i32,
) -> Result<(), Box<dyn Error>> {
    use burn::backend::{Autodiff, NdArray};
    use poker_solver_gpu::training::flop_pipeline::{train_flop_cfvnet_cuda, FlopTrainingConfig};

    type B = Autodiff<NdArray>;

    let config = FlopTrainingConfig {
        turn_model_path: turn_model,
        turn_hidden_layers,
        turn_hidden_size,
        num_samples,
        solve_iterations,
        batch_size,
        reservoir_capacity,
        hidden_layers,
        hidden_size,
        train_batch_size,
        train_steps_per_batch,
        learning_rate,
        huber_delta,
        aux_loss_weight,
        validation_interval,
        checkpoint_interval,
        output_dir: output,
        seed,
        ref_pot,
        ref_stack,
    };

    let device = Default::default();
    train_flop_cfvnet_cuda::<B>(&config, &device)?;
    Ok(())
}

#[cfg(feature = "gpu-training")]
#[allow(clippy::too_many_arguments)]
fn run_gpu_train_preflop(
    flop_model: PathBuf,
    flop_hidden_layers: usize,
    flop_hidden_size: usize,
    num_samples: u64,
    batch_size: usize,
    reservoir_capacity: usize,
    hidden_layers: usize,
    hidden_size: usize,
    train_batch_size: usize,
    train_steps_per_batch: usize,
    learning_rate: f64,
    huber_delta: f64,
    aux_loss_weight: f64,
    validation_interval: u64,
    checkpoint_interval: u64,
    output: PathBuf,
    seed: u64,
    ref_pot: i32,
    ref_stack: i32,
) -> Result<(), Box<dyn Error>> {
    use burn::backend::{Autodiff, NdArray};
    use poker_solver_gpu::training::preflop_pipeline::{train_preflop_cfvnet_cuda, PreflopTrainingConfig};

    type B = Autodiff<NdArray>;

    let config = PreflopTrainingConfig {
        flop_model_path: flop_model,
        flop_hidden_layers,
        flop_hidden_size,
        num_samples,
        reservoir_capacity,
        hidden_layers,
        hidden_size,
        train_batch_size,
        train_steps_per_batch,
        batch_size,
        learning_rate,
        huber_delta,
        aux_loss_weight,
        validation_interval,
        checkpoint_interval,
        output_dir: output,
        seed,
        ref_pot,
        ref_stack,
    };

    let device = Default::default();
    train_preflop_cfvnet_cuda::<B>(&config, &device)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// GPU stack training (all 4 streets from YAML config)
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu-training")]
fn run_gpu_train_stack(
    config_path: PathBuf,
    output: PathBuf,
) -> Result<(), Box<dyn Error>> {
    use burn::backend::{Autodiff, NdArray};
    use poker_solver_gpu::training::stack_config::{train_full_stack, GpuTrainingStackConfig};

    type B = Autodiff<NdArray>;

    let yaml = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("Failed to read config {}: {e}", config_path.display()))?;
    let stack_config: GpuTrainingStackConfig = serde_yaml::from_str(&yaml)
        .map_err(|e| format!("Failed to parse config: {e}"))?;

    eprintln!("GPU Train Stack");
    eprintln!("  Config: {}", config_path.display());
    eprintln!("  Output: {}", output.display());
    eprintln!("  Model:  {} layers x {} hidden", stack_config.model.hidden_layers, stack_config.model.hidden_size);
    eprintln!("  River model: {} ({}x{})", stack_config.river_model.path, stack_config.river_model.hidden_layers, stack_config.river_model.hidden_size);
    eprintln!("  Turn:   {} samples", stack_config.turn.num_samples);
    eprintln!("  Flop:   {} samples", stack_config.flop.num_samples);
    eprintln!("  Preflop:{} samples", stack_config.preflop.num_samples);
    eprintln!();

    std::fs::create_dir_all(&output)
        .map_err(|e| format!("Failed to create output dir: {e}"))?;

    let device = Default::default();
    train_full_stack::<B>(&stack_config, &output, &device)
        .map_err(|e| format!("Stack training failed: {e}"))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Range solver
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_range_solve(
    oop_range_str: &str,
    ip_range_str: &str,
    flop_str: &str,
    turn_str: Option<&str>,
    river_str: Option<&str>,
    pot: i32,
    effective_stack: i32,
    iterations: u32,
    target_exploitability: f32,
    oop_bet_str: &str,
    oop_raise_str: &str,
    ip_bet_str: &str,
    ip_raise_str: &str,
    compressed: bool,
) -> Result<(), Box<dyn Error>> {
    use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::{card_from_str, flop_from_str, hole_to_string, CardConfig, NOT_DEALT};
    use range_solver::range::Range;
    use range_solver::{solve, PostFlopGame};

    // --- Parse inputs ---
    let oop_range: Range = oop_range_str
        .parse()
        .map_err(|e: String| format!("Invalid OOP range: {e}"))?;
    let ip_range: Range = ip_range_str
        .parse()
        .map_err(|e: String| format!("Invalid IP range: {e}"))?;

    let flop = flop_from_str(flop_str).map_err(|e| format!("Invalid flop: {e}"))?;

    let turn = match turn_str {
        Some(s) => card_from_str(s).map_err(|e| format!("Invalid turn card: {e}"))?,
        None => NOT_DEALT,
    };

    let river = match river_str {
        Some(s) => card_from_str(s).map_err(|e| format!("Invalid river card: {e}"))?,
        None => NOT_DEALT,
    };

    // Determine initial board state
    let initial_state = if river != NOT_DEALT {
        BoardState::River
    } else if turn != NOT_DEALT {
        BoardState::Turn
    } else {
        BoardState::Flop
    };

    // Parse bet sizes
    let oop_sizes = BetSizeOptions::try_from((oop_bet_str, oop_raise_str))
        .map_err(|e| format!("Invalid OOP bet sizes: {e}"))?;
    let ip_sizes = BetSizeOptions::try_from((ip_bet_str, ip_raise_str))
        .map_err(|e| format!("Invalid IP bet sizes: {e}"))?;

    // --- Build game ---
    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop,
        turn,
        river,
    };

    let tree_config = TreeConfig {
        initial_state,
        starting_pot: pot,
        effective_stack,
        flop_bet_sizes: [oop_sizes.clone(), ip_sizes.clone()],
        turn_bet_sizes: [oop_sizes.clone(), ip_sizes.clone()],
        river_bet_sizes: [oop_sizes, ip_sizes],
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
        ..Default::default()
    };

    let action_tree =
        ActionTree::new(tree_config).map_err(|e| format!("Failed to build action tree: {e}"))?;

    let mut game = PostFlopGame::with_config(card_config, action_tree)
        .map_err(|e| format!("Failed to build game: {e}"))?;

    // --- Print game info ---
    let (mem_uncompressed, mem_compressed) = game.memory_usage();
    let mem = if compressed {
        mem_compressed
    } else {
        mem_uncompressed
    };
    eprintln!("Range Solver (Discounted CFR)");
    eprintln!("  Board: {flop_str}{}", format_board_suffix(turn_str, river_str));
    eprintln!("  Initial state: {initial_state}");
    eprintln!("  Pot: {pot}, Effective stack: {effective_stack}");
    eprintln!(
        "  OOP hands: {}, IP hands: {}",
        game.private_cards(0).len(),
        game.private_cards(1).len(),
    );
    eprintln!("  Memory: {:.1} MB", mem as f64 / (1024.0 * 1024.0));
    eprintln!(
        "  Compression: {}",
        if compressed { "enabled" } else { "disabled" }
    );
    eprintln!();

    // --- Allocate and solve ---
    game.allocate_memory(compressed);

    let start = Instant::now();
    let exploitability = solve(&mut game, iterations, target_exploitability, true);
    let elapsed = start.elapsed();

    eprintln!();
    eprintln!(
        "Solved in {:.2}s ({} iterations)",
        elapsed.as_secs_f64(),
        iterations,
    );
    eprintln!("Final exploitability: {exploitability:.4}");
    eprintln!();

    // --- Print root actions and strategy summary ---
    game.back_to_root();
    let actions = game.available_actions();
    let player = game.current_player();
    let hands = game.private_cards(player);
    let strategy = game.strategy();
    let num_hands = hands.len();
    let num_actions = actions.len();

    println!("Root node: {} to act ({num_actions} actions, {num_hands} hands)",
        if player == 0 { "OOP" } else { "IP" });
    println!();

    // Print header
    print!("{:<10}", "Hand");
    for action in &actions {
        print!("  {:>10}", action.to_string());
    }
    println!();

    // Print per-hand strategy (limit to first 30 hands for readability)
    let display_count = num_hands.min(30);
    for h in 0..display_count {
        let hand_str = hole_to_string(hands[h]).unwrap_or_else(|_| "??".to_string());
        print!("{:<10}", hand_str);
        for a in 0..num_actions {
            let prob = strategy[a * num_hands + h];
            print!("  {:>10.1}%", prob * 100.0);
        }
        println!();
    }

    if num_hands > display_count {
        println!("... and {} more hands", num_hands - display_count);
    }

    Ok(())
}

fn format_board_suffix(turn: Option<&str>, river: Option<&str>) -> String {
    let mut s = String::new();
    if let Some(t) = turn {
        s.push(' ');
        s.push_str(t);
    }
    if let Some(r) = river {
        s.push(' ');
        s.push_str(r);
    }
    s
}

// ---------------------------------------------------------------------------
// GPU solver
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_gpu_solve(
    oop_range_str: &str,
    ip_range_str: &str,
    flop_str: &str,
    turn_str: Option<&str>,
    river_str: Option<&str>,
    pot: i32,
    effective_stack: i32,
    iterations: u32,
    target_exploitability: f32,
    oop_bet_str: &str,
    oop_raise_str: &str,
    ip_bet_str: &str,
    ip_raise_str: &str,
) -> Result<(), Box<dyn Error>> {
    use poker_solver_gpu::gpu::GpuContext;
    use poker_solver_gpu::solver::GpuSolver;
    use poker_solver_gpu::tree::FlatTree;
    use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::{card_from_str, flop_from_str, hole_to_string, CardConfig, NOT_DEALT};
    use range_solver::interface::Game;
    use range_solver::range::Range;
    use range_solver::PostFlopGame;

    // --- Parse inputs ---
    let oop_range: Range = oop_range_str
        .parse()
        .map_err(|e: String| format!("Invalid OOP range: {e}"))?;
    let ip_range: Range = ip_range_str
        .parse()
        .map_err(|e: String| format!("Invalid IP range: {e}"))?;

    let flop = flop_from_str(flop_str).map_err(|e| format!("Invalid flop: {e}"))?;

    let turn = match turn_str {
        Some(s) => card_from_str(s).map_err(|e| format!("Invalid turn card: {e}"))?,
        None => NOT_DEALT,
    };

    let river = match river_str {
        Some(s) => card_from_str(s).map_err(|e| format!("Invalid river card: {e}"))?,
        None => NOT_DEALT,
    };

    // Determine initial board state
    let initial_state = if river != NOT_DEALT {
        BoardState::River
    } else if turn != NOT_DEALT {
        BoardState::Turn
    } else {
        BoardState::Flop
    };

    // Parse bet sizes
    let oop_sizes = BetSizeOptions::try_from((oop_bet_str, oop_raise_str))
        .map_err(|e| format!("Invalid OOP bet sizes: {e}"))?;
    let ip_sizes = BetSizeOptions::try_from((ip_bet_str, ip_raise_str))
        .map_err(|e| format!("Invalid IP bet sizes: {e}"))?;

    // --- Build game ---
    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop,
        turn,
        river,
    };

    let tree_config = TreeConfig {
        initial_state,
        starting_pot: pot,
        effective_stack,
        flop_bet_sizes: [oop_sizes.clone(), ip_sizes.clone()],
        turn_bet_sizes: [oop_sizes.clone(), ip_sizes.clone()],
        river_bet_sizes: [oop_sizes, ip_sizes],
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
        ..Default::default()
    };

    let action_tree =
        ActionTree::new(tree_config).map_err(|e| format!("Failed to build action tree: {e}"))?;

    let mut game = PostFlopGame::with_config(card_config, action_tree)
        .map_err(|e| format!("Failed to build game: {e}"))?;

    game.allocate_memory(false);

    // --- Print game info ---
    eprintln!("GPU DCFR+ Solver");
    eprintln!("  Board: {flop_str}{}", format_board_suffix(turn_str, river_str));
    eprintln!("  Initial state: {initial_state}");
    eprintln!("  Pot: {pot}, Effective stack: {effective_stack}");
    eprintln!(
        "  OOP hands: {}, IP hands: {}",
        game.num_private_hands(0),
        game.num_private_hands(1),
    );
    eprintln!("  Iterations: {iterations}");
    eprintln!("  Target exploitability: {target_exploitability}");
    eprintln!();

    // --- Build flat tree ---
    let flat_tree = FlatTree::from_postflop_game(&mut game);
    eprintln!(
        "  Tree: {} nodes, {} infosets, {} levels",
        flat_tree.num_nodes(),
        flat_tree.num_infosets,
        flat_tree.num_levels(),
    );
    eprintln!();

    // --- Initialize GPU and solve ---
    let gpu = GpuContext::new(0).map_err(|e| format!("Failed to initialize GPU: {e}"))?;
    let mut solver =
        GpuSolver::new(&gpu, &flat_tree).map_err(|e| format!("Failed to create solver: {e}"))?;

    let start = Instant::now();
    let result = solver
        .solve(iterations, Some(target_exploitability))
        .map_err(|e| format!("Solver failed: {e}"))?;
    let elapsed = start.elapsed();

    eprintln!(
        "Solved in {:.2}s ({} iterations)",
        elapsed.as_secs_f64(),
        result.iterations,
    );
    eprintln!();

    // --- Print root strategy ---
    game.back_to_root();
    let actions = game.available_actions();
    let player = game.current_player();
    let hands = game.private_cards(player);
    let num_hands = flat_tree.num_hands;
    let max_actions = flat_tree.max_actions();
    let root_infoset = flat_tree.infoset_ids[0] as usize;
    let num_actions = flat_tree.infoset_num_actions[root_infoset] as usize;

    println!(
        "Root node: {} to act ({num_actions} actions, {} hands)",
        if player == 0 { "OOP" } else { "IP" },
        hands.len(),
    );
    println!();

    // Print header
    print!("{:<10}", "Hand");
    for action in actions.iter().take(num_actions) {
        print!("  {:>10}", action.to_string());
    }
    println!();

    // Print per-hand strategy (limit to first 30 hands)
    let display_count = hands.len().min(30);
    for h in 0..display_count {
        let hand_str = hole_to_string(hands[h]).unwrap_or_else(|_| "??".to_string());
        print!("{:<10}", hand_str);
        for a in 0..num_actions {
            let idx = (root_infoset * max_actions + a) * num_hands + h;
            let prob = result.strategy[idx];
            print!("  {:>10.1}%", prob * 100.0);
        }
        println!();
    }

    if hands.len() > display_count {
        println!("... and {} more hands", hands.len() - display_count);
    }

    Ok(())
}
