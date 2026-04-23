mod blueprint_tui;
mod blueprint_tui_audit;
mod blueprint_tui_audit_widget;
mod blueprint_tui_config;
mod blueprint_tui_metrics;
mod blueprint_tui_resolve;
mod blueprint_tui_scenarios;
mod blueprint_tui_widgets;
mod bench_rollout;
mod boundary_trace;
mod compare_solve;
mod inspect_spot;
mod validate_rollout;
mod mp_tui;
mod mp_tui_scenarios;
mod mp_tui_widgets;
#[allow(dead_code)]
mod log_file;
#[allow(dead_code)]
mod validate_blueprint;
mod validation_spots;

use std::error::Error;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use poker_solver_core::blueprint_mp::config::BlueprintMpConfig;
use poker_solver_core::blueprint_mp::trainer::train_blueprint_mp;
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
    /// Train an N-player blueprint strategy using MCCFR (multiplayer)
    TrainBlueprintMp {
        /// YAML config file (BlueprintMpConfig)
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
        /// Audit transition consistency (do combos in same bucket go to similar next-street buckets?)
        #[arg(long)]
        transition_audit: bool,
        /// Number of boards to sample for transition audit (default 20)
        #[arg(long, default_value = "20")]
        transition_audit_boards: usize,
    },
    /// Pre-compute the equity+delta lookup cache for fast expected-delta bucketing.
    /// Generates turn table first (averaging over river cards), then flop table
    /// (using turn table for two-street lookahead). Saves to a binary file.
    PrecomputeEquityDelta {
        /// Output file path for the cache
        #[arg(short, long, default_value = "cache/equity_delta.bin")]
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
    /// Solve a postflop spot using GPU-accelerated DCFR
    GpuRangeSolve {
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
    /// Validate a blueprint strategy against exact range-solver solutions
    ValidateBlueprint {
        /// Path to blueprint bundle directory
        #[arg(short, long)]
        blueprint: PathBuf,
        /// Path to validation spots YAML file
        #[arg(short, long)]
        spots: PathBuf,
        /// Optional cluster directory (for per-flop bucket lookup)
        #[arg(long)]
        cluster_dir: Option<PathBuf>,
    },
    /// Compare two sets of cluster bucket files
    DiffClusters {
        /// Directory A containing .buckets files
        #[arg(long)]
        dir_a: PathBuf,
        /// Directory B containing .buckets files
        #[arg(long)]
        dir_b: PathBuf,
        /// Number of boards to sample for equity audit (0 = skip equity)
        #[arg(long, default_value = "200")]
        sample_boards: usize,
        /// Show per-bucket equity histogram breakdown
        #[arg(long)]
        verbose: bool,
    },
    /// Generate PBS training data from blueprint play for ReBeL offline seeding
    #[command(name = "rebel-seed")]
    RebelSeed {
        /// Path to ReBeL YAML configuration file
        #[arg(short, long)]
        config: PathBuf,
    },
    /// Run ReBeL training: offline seeding then live self-play
    #[command(name = "rebel-train")]
    RebelTrain {
        /// Path to ReBeL YAML configuration file
        #[arg(short, long)]
        config: PathBuf,

        /// Skip offline seeding and start from an existing model
        #[arg(long)]
        model: Option<String>,

        /// Run offline seeding only (no self-play)
        #[arg(long)]
        offline_only: bool,
    },
    /// Evaluate a trained ReBeL model
    #[command(name = "rebel-eval")]
    RebelEval {
        /// Path to ReBeL YAML configuration file
        #[arg(short, long)]
        config: PathBuf,

        /// Path to trained model checkpoint
        #[arg(long)]
        model: String,

        /// Evaluation mode: mse or h2h
        #[arg(long, default_value = "mse")]
        mode: String,

        /// Number of hands for head-to-head evaluation
        #[arg(long, default_value_t = 100000)]
        num_hands: usize,
    },
    /// Inspect a blueprint strategy at a specific spot encoding
    InspectSpot {
        /// Path to blueprint config YAML
        #[arg(short, long)]
        config: PathBuf,

        /// Spot encoding string (e.g. "sb:2bb,bb:call|Td9d6h|bb:check,sb:4bb")
        #[arg(long)]
        spot: String,
    },
    /// Benchmark rollout throughput: drives the rollout evaluator directly
    /// (does not run DCFR). Reports hands/sec.
    /// Requires a bundle with buckets/ and strategy.bin.
    BenchRollout {
        /// Path to blueprint bundle directory (must contain config.yaml,
        /// strategy.bin or snapshot_*/strategy.bin, and buckets/)
        #[arg(short, long)]
        bundle: PathBuf,

        /// Wall-time duration in seconds
        #[arg(long, default_value = "10")]
        duration_secs: u64,

        /// Flop board cards (e.g. "Ks7h2c"). Default: Ks7h2c.
        #[arg(long, default_value = "Ks7h2c")]
        board: String,

        /// Starting pot size (chips)
        #[arg(long, default_value = "100")]
        pot: u32,

        /// Starting stack per player (chips)
        #[arg(long, default_value = "200")]
        stacks: u32,

        /// Decision levels to enumerate before sampling (higher = more accurate, slower)
        #[arg(long)]
        enumerate_depth: Option<u8>,

        /// Opponent hands sampled per hero combo (higher = less variance, slower)
        #[arg(long)]
        opponent_samples: Option<u32>,
    },
    /// Compare exhaustive vs sampled rollout CFVs per combo.
    /// Reports max/mean/L2 diff in pot-fraction and mbb/hand units.
    /// Runs the sampled side multiple times and aggregates.
    ValidateRollout {
        /// Path to blueprint bundle directory
        #[arg(short, long)]
        bundle: PathBuf,

        /// Flop board cards (e.g. "Ks7h2c"). Default: Ks7h2c.
        #[arg(long, default_value = "Ks7h2c")]
        board: String,

        /// Starting pot size (chips)
        #[arg(long, default_value = "100")]
        pot: u32,

        /// Starting stack per player (chips)
        #[arg(long, default_value = "200")]
        stacks: u32,

        /// Number of sampled runs to aggregate
        #[arg(long, default_value = "5")]
        num_runs: usize,

        /// Pass threshold for max_abs_diff in pot-fraction units
        #[arg(long, default_value = "0.02")]
        pass_threshold: f64,

        /// Decision levels to enumerate before sampling (higher = more accurate, slower)
        #[arg(long)]
        enumerate_depth: Option<u8>,

        /// Opponent hands sampled per hero combo (higher = less variance, slower)
        #[arg(long)]
        opponent_samples: Option<u32>,
    },
    /// Compare subgame vs exact solve on a given spot.
    /// Reports per-hand mass moved, per-action-class bias, and exploitability delta.
    #[command(name = "compare-solve")]
    CompareSolve {
        /// Path to blueprint bundle directory
        #[arg(short, long)]
        bundle: PathBuf,

        /// Snapshot name (e.g. snapshot_0013); defaults to latest
        #[arg(long)]
        snapshot: Option<String>,

        /// Spot encoding (e.g. "sb:2bb,bb:10bb,sb:22bb,bb:call|Jd9d7d")
        #[arg(long)]
        spot: String,

        /// DCFR iterations for each solve
        #[arg(long, default_value_t = 200)]
        iters: u32,

        /// Print per-iteration progress
        #[arg(long)]
        verbose: bool,

        /// Dump per-boundary CFV statistics after precompute (before DCFR)
        #[arg(long)]
        dump_boundary_cfvs: bool,

        /// Flop boundary mode: "exact" or "cfvnet"
        #[arg(long, default_value = "exact")]
        flop_boundary: String,

        /// ONNX model path for flop boundary (required when --flop-boundary=cfvnet)
        #[arg(long)]
        flop_model: Option<String>,

        /// Turn boundary mode: "exact" or "cfvnet"
        #[arg(long, default_value = "exact")]
        turn_boundary: String,

        /// ONNX model path for turn boundary (required when --turn-boundary=cfvnet)
        #[arg(long)]
        turn_model: Option<String>,

        /// River boundary mode: "exact" or "cfvnet"
        #[arg(long, default_value = "exact")]
        river_boundary: String,

        /// ONNX model path for river boundary (required when --river-boundary=cfvnet)
        #[arg(long)]
        river_model: Option<String>,

        /// Boundary ordinals to trace (comma-separated, or "all").
        /// Produces one JSONL file per ordinal in --trace-dir.
        #[arg(long)]
        trace_boundaries: Option<String>,

        /// Iterations to trace (comma-separated, "all", or "last"; default "last").
        #[arg(long, default_value = "last")]
        trace_iters: String,

        /// Directory to write trace files (default: ./traces/).
        #[arg(long, default_value = "./traces")]
        trace_dir: PathBuf,

        /// Maximum acceptable per-cell strategy delta (fraction, e.g. 0.001
        /// = 0.1%). If any aggregated 13x13 cell probability differs by more
        /// than this between exact and subgame, the harness exits with a
        /// non-zero status. `0.0` disables the check (default).
        #[arg(long, default_value_t = 0.0)]
        tolerance: f32,

        /// Enable DeepStack-style range gadget at subgame boundaries.
        /// Constrains opponent CFVs to be at least as good as the blueprint
        /// CBV opt-out, producing a safe re-solve.
        #[arg(long, default_value_t = false)]
        gadget: bool,
    },
    /// Generate a held-out validation set for ReBeL
    #[command(name = "rebel-validate")]
    RebelValidate {
        /// Path to ReBeL YAML configuration file
        #[arg(short, long)]
        config: PathBuf,

        /// Number of validation examples to generate
        #[arg(long, default_value_t = 100)]
        num_examples: usize,

        /// Output path for validation set binary file
        #[arg(short, long)]
        output: String,
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
                trainer.strategy_refresh_trigger = Arc::clone(&metrics.strategy_refresh_trigger);

                // Resolve scenarios using spot notation.
                let resolved = blueprint_tui_resolve::resolve_scenarios(
                    &trainer.tree,
                    &trainer.storage,
                    &tui_config.scenarios,
                );
                let scenarios = resolved.scenarios;
                let shared_boards: Arc<RwLock<Vec<Vec<poker_solver_core::poker::Card>>>> =
                    Arc::new(RwLock::new(resolved.boards));

                // Resolve regret audits.
                let resolved_audit = blueprint_tui_resolve::resolve_audits(
                    &trainer.tree,
                    &trainer.storage,
                    &tui_config.regret_audits,
                    tui_config.telemetry.sparkline_window,
                );
                let shared_audits: Arc<RwLock<Vec<blueprint_tui_audit::ResolvedRegretAudit>>> =
                    Arc::new(RwLock::new(resolved_audit.audits));
                let audit_panel = resolved_audit.panel;

                // Wire config reload trigger.
                trainer.config_reload_trigger =
                    Arc::clone(&metrics.config_reload_trigger);

                // Wire strategy refresh callback from trainer to TUI metrics.
                let scenarios_node_indices: Vec<u32> =
                    scenarios.iter().map(|s| s.node_idx).collect();
                trainer.scenario_ev_tracker.set_nodes(scenarios_node_indices.clone());
                trainer.scenario_node_indices = scenarios_node_indices;
                trainer.strategy_refresh_interval_secs =
                    tui_config.telemetry.strategy_delta_interval_seconds;

                let boards_for_refresh = Arc::clone(&shared_boards);
                let metrics_for_refresh = Arc::clone(&metrics);
                trainer.on_strategy_refresh =
                    Some(Box::new(move |scenario_idx, node_idx, storage, tree, hand_evs| {
                        let boards = boards_for_refresh.read().unwrap();
                        if scenario_idx < boards.len() {
                            let grid = blueprint_tui_scenarios::extract_strategy_grid(
                                tree, storage, node_idx, &boards[scenario_idx], Some(hand_evs),
                            );
                            metrics_for_refresh.update_scenario_grid(scenario_idx, grid);
                        }
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

                let metrics_for_exploit = Arc::clone(&metrics);
                trainer.on_exploitability = Some(Box::new(move |val| {
                    metrics_for_exploit.push_exploitability(val);
                }));

                let metrics_for_exploit_start = Arc::clone(&metrics);
                trainer.on_exploitability_start = Some(Box::new(move |total| {
                    metrics_for_exploit_start.start_exploitability_pass(total);
                }));

                let metrics_for_exploit_tick = Arc::clone(&metrics);
                trainer.on_exploitability_tick = Some(Arc::new(move || {
                    metrics_for_exploit_tick.tick_exploitability_progress();
                }));

                let metrics_for_exploit_finish = Arc::clone(&metrics);
                trainer.on_exploitability_finish = Some(Box::new(move || {
                    metrics_for_exploit_finish.finish_exploitability_pass();
                }));

                let metrics_for_spots = Arc::clone(&metrics);
                trainer.on_exploitable_spots = Some(Box::new(move |spots| {
                    metrics_for_spots.set_exploitable_spots(spots);
                }));

                // Wire audit refresh callback.
                {
                    let audits_for_refresh = Arc::clone(&shared_audits);
                    let metrics_for_audit = Arc::clone(&metrics);
                    trainer.on_audit_refresh = Some(Box::new(move |storage| {
                        let mut audits = audits_for_refresh.write().unwrap();
                        if audits.is_empty() {
                            return;
                        }
                        for audit in audits.iter_mut() {
                            audit.tick(storage);
                        }
                        let snapshots: Vec<_> =
                            audits.iter().map(|a| a.snapshot()).collect();
                        metrics_for_audit.update_regret_audits(snapshots);
                    }));
                }

                // Wire config reload callback.
                {
                    let config_path_for_reload = config.clone();
                    let shared_boards_for_reload = Arc::clone(&shared_boards);
                    let shared_audits_for_reload = Arc::clone(&shared_audits);
                    let metrics_for_reload = Arc::clone(&metrics);
                    let reloaded_indices = Arc::clone(&trainer.reloaded_node_indices);
                    let sparkline_window = tui_config.telemetry.sparkline_window;
                    trainer.on_config_reload = Some(Box::new(move |tree, storage| {
                        let Ok(yaml) = std::fs::read_to_string(&config_path_for_reload) else {
                            return;
                        };
                        let new_tui_config = blueprint_tui_config::parse_tui_config(&yaml);

                        // Re-resolve scenarios.
                        let resolved = blueprint_tui_resolve::resolve_scenarios(
                            tree, storage, &new_tui_config.scenarios,
                        );

                        // Re-resolve audits.
                        let audits = blueprint_tui_resolve::resolve_audits(
                            tree, storage, &new_tui_config.regret_audits, sparkline_window,
                        );

                        // Swap shared data for callbacks.
                        *shared_boards_for_reload.write().unwrap() = resolved.boards;
                        *shared_audits_for_reload.write().unwrap() = audits.audits;

                        // Provide new node indices so the trainer updates tracking.
                        let new_indices: Vec<u32> =
                            resolved.scenarios.iter().map(|s| s.node_idx).collect();
                        *reloaded_indices.lock().unwrap() = Some(new_indices);

                        // Push new UI state to TUI.
                        let state = blueprint_tui_metrics::ReloadedTuiState {
                            scenarios: resolved.scenarios,
                            audit_panel: audits.panel,
                        };
                        *metrics_for_reload.reloaded_tui_state.lock().unwrap() = Some(state);
                    }));
                }

                // Random scenario carousel.
                if tui_config.random_scenario.enabled {
                    trainer.random_scenario_hold_minutes =
                        tui_config.random_scenario.hold_minutes;
                    let metrics_for_random = Arc::clone(&metrics);
                    let pool = tui_config.random_scenario.pool.clone();
                    trainer.on_random_scenario =
                        Some(Box::new(move |storage, tree, hand_evs| {
                            use rand::seq::IndexedRandom;
                            use poker_solver_core::blueprint_v2::game_tree::GameNode;
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

                            // Select the correct position's EVs based on the node's player.
                            let player = match &tree.nodes[node_idx as usize] {
                                GameNode::Decision { player, .. } => *player as usize,
                                _ => 0,
                            };
                            let node_hand_evs = &hand_evs[player];

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
                                tree, storage, node_idx, &board, Some(node_hand_evs),
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
                    audit_panel,
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
        Commands::TrainBlueprintMp { config, no_tui } => {
            run_train_blueprint_mp(config.to_str().expect("invalid config path"), no_tui)?;
        }
        Commands::Cluster { config, output } => {
            let yaml = std::fs::read_to_string(&config)?;
            let bp_config: BlueprintV2Config = serde_yaml::from_str(&yaml)?;

            std::fs::create_dir_all(&output)?;

            if bp_config.clustering.per_flop.is_some() {
                let pf_cfg = bp_config.clustering.per_flop.as_ref().unwrap();
                eprintln!("Per-Flop Clustering Pipeline");
                eprintln!("  Output: {}", output.display());
                eprintln!(
                    "  Buckets: flop={}, turn={}/flop, river={}/flop",
                    bp_config.clustering.flop.buckets,
                    pf_cfg.turn_buckets,
                    pf_cfg.river_buckets,
                );
                eprintln!();

                let per_flop_config = poker_solver_core::blueprint_v2::cluster_pipeline::PerFlopClusteringConfig {
                    flop_buckets: bp_config.clustering.flop.buckets,
                    turn_buckets: pf_cfg.turn_buckets,
                    river_buckets: pf_cfg.river_buckets,
                    kmeans_iterations: bp_config.clustering.kmeans_iterations,
                    seed: bp_config.clustering.seed,
                };

                let num_flops = 1755_u64; // canonical flop count

                let mp = MultiProgress::new();

                // Bar 1: Overall flop completion (0 / 1755)
                let flop_bar = mp.add(ProgressBar::new(num_flops));
                flop_bar.set_style(
                    ProgressStyle::with_template("  {msg:>20} {bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}] ETA {eta_precise}")
                        .unwrap()
                        .progress_chars("##-"),
                );
                flop_bar.set_message("Flops completed");

                // Bars 2..N: one per active thread, showing current flop + phase
                let thread_count = rayon::current_num_threads();
                let bar_style = ProgressStyle::with_template("  {msg:>30} {bar:30.white/black} {pos}/{len} ETA {eta}")
                    .unwrap()
                    .progress_chars("##-");
                let spinner_style = ProgressStyle::with_template("  {msg:>30} {spinner:.cyan} {elapsed_precise}")
                    .unwrap();
                let thread_bars: Vec<ProgressBar> = (0..thread_count)
                    .map(|_| {
                        let bar = mp.add(ProgressBar::new(100));
                        bar.set_style(bar_style.clone());
                        bar.set_message("");
                        bar
                    })
                    .collect();
                let bar_style_clone = bar_style.clone();
                let spinner_style_clone = spinner_style.clone();

                poker_solver_core::blueprint_v2::cluster_pipeline::run_per_flop_pipeline(
                    &per_flop_config,
                    &output,
                    None,
                    |stage, msg, p| {
                        match stage {
                            "done" => {
                                let done = (p * num_flops as f64).round() as u64;
                                flop_bar.set_position(done);
                                // Clear the thread bar that finished
                                let tid = rayon::current_thread_index().unwrap_or(0);
                                if let Some(bar) = thread_bars.get(tid) {
                                    bar.set_message("");
                                }
                            }
                            "resume" => {
                                let done = (p * num_flops as f64).round() as u64;
                                flop_bar.set_position(done);
                                flop_bar.set_message(format!("Resumed ({done} cached)"));
                            }
                            "flop-clustering" | "preflop" => {
                                flop_bar.set_position(num_flops);
                                flop_bar.set_message(format!("{stage}: {msg}"));
                            }
                            _ => {
                                // stage = "NNNN [cards]", msg = "river 5/48" or "turn-kmeans 0/1" etc
                                let tid = rayon::current_thread_index().unwrap_or(0);
                                if let Some(bar) = thread_bars.get(tid) {
                                    // Parse "phase pos/total" from msg
                                    if let Some((phase, counts)) = msg.rsplit_once(' ') {
                                        if let Some((pos_s, total_s)) = counts.split_once('/') {
                                            if let (Ok(pos), Ok(total)) = (pos_s.parse::<u64>(), total_s.parse::<u64>()) {
                                                let new_msg = format!("{stage} {phase}");
                                                if total <= 1 {
                                                    // No meaningful progress — show spinner
                                                    bar.set_style(spinner_style_clone.clone());
                                                    bar.set_message(new_msg);
                                                    bar.enable_steady_tick(std::time::Duration::from_millis(100));
                                                } else {
                                                    // Real progress bar
                                                    if bar.message() != new_msg {
                                                        bar.disable_steady_tick();
                                                        bar.set_style(bar_style_clone.clone());
                                                        bar.reset_eta();
                                                        bar.set_message(new_msg);
                                                        bar.set_length(total);
                                                    }
                                                    bar.set_position(pos);
                                                }
                                                return;
                                            }
                                        }
                                    }
                                    bar.set_message(format!("{stage} {msg}"));
                                }
                            }
                        }
                    },
                )?;

                flop_bar.finish_with_message("done");
                for bar in &thread_bars {
                    bar.finish_and_clear();
                }
                eprintln!("Per-flop clustering complete. Files saved to {}", output.display());
            } else {
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
                    let mp = MultiProgress::new();
                    let street_bar = mp.add(ProgressBar::new(4));
                    street_bar.set_style(
                        ProgressStyle::with_template("  {msg:>12} {bar:40.cyan/blue} {pos}/{len}")
                            .unwrap()
                            .progress_chars("##-"),
                    );
                    street_bar.set_message("clustering");

                    let phase_bar = mp.add(ProgressBar::new(1000));
                    phase_bar.set_style(
                        ProgressStyle::with_template("  {msg:>12} {bar:40.white/black} {pos}/{len}")
                            .unwrap()
                            .progress_chars("##-"),
                    );

                    let current_street = std::sync::Mutex::new(String::new());
                    let street_count = std::sync::atomic::AtomicU32::new(0);

                    poker_solver_core::blueprint_v2::cluster_pipeline::run_clustering_pipeline(
                        &bp_config.clustering,
                        &output,
                        |street, phase, p| {
                            let mut cur = current_street.lock().unwrap();
                            if *cur != street {
                                if !cur.is_empty() {
                                    phase_bar.finish_and_clear();
                                }
                                *cur = street.to_string();
                                street_bar.set_message(street.to_string());
                                street_bar.set_position(
                                    street_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                                        as u64,
                                );
                                phase_bar.reset();
                            }
                            drop(cur);

                            phase_bar.set_message(phase.to_string());
                            phase_bar.set_position((p * 1000.0) as u64);
                        },
                    )?;

                    street_bar.finish_with_message("done");
                    phase_bar.finish_and_clear();
                    eprintln!("Clustering complete. Files saved to {}", output.display());
                }
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
            transition_audit,
            transition_audit_boards,
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
            if transition_audit {
                use poker_solver_core::blueprint_v2::cluster_diagnostics::audit_transition_consistency;
                use poker_solver_core::blueprint_v2::centroid_file::CentroidFile;
                let pairs = [("flop", "turn"), ("turn", "river")];
                for (from_name, to_name) in &pairs {
                    let from_path = cluster_dir.join(format!("{from_name}.buckets"));
                    let to_path = cluster_dir.join(format!("{to_name}.buckets"));
                    if from_path.exists() && to_path.exists() {
                        eprintln!("\nTransition consistency audit: {from_name} → {to_name} ({transition_audit_boards} sample boards)...");
                        let from_bf = BucketFile::load(&from_path)?;
                        let to_bf = BucketFile::load(&to_path)?;
                        // Load centroid file for the current (from) street if available.
                        let centroid_path = cluster_dir.join(format!("{from_name}.centroids"));
                        let centroids = if centroid_path.exists() {
                            CentroidFile::load(&centroid_path).ok()
                        } else {
                            None
                        };
                        let report = audit_transition_consistency(
                            &from_bf, &to_bf, transition_audit_boards, 42,
                            centroids.as_ref(),
                        );
                        eprintln!("{}", report.summary());
                    }
                }
            }
            // Per-flop bucket file diagnostics
            let per_flop_marker = cluster_dir.join("flop_0000.buckets");
            if per_flop_marker.exists() {
                eprintln!("\nPer-flop bucket files detected.");
                let pf_report = poker_solver_core::blueprint_v2::cluster_diagnostics::diagnose_per_flop_dir(&cluster_dir, 10)?;
                eprintln!("  Files: {}, sampled: {}", pf_report.total_flop_files, pf_report.sampled_files);
                eprintln!("  Turn buckets: {}, River buckets: {}", pf_report.turn_bucket_count, pf_report.river_bucket_count);
                eprintln!("  Avg turn cards: {:.1}, Avg rivers/turn: {:.1}", pf_report.avg_turn_cards, pf_report.avg_river_cards_per_turn);

                if audit {
                    let pf_flop_samples = audit_boards.min(20);
                    eprintln!("\nPer-flop equity audit ({} flop samples, 5 rivers/turn)...", pf_flop_samples);
                    let pf_audit = poker_solver_core::blueprint_v2::cluster_diagnostics::audit_per_flop_equity(
                        &cluster_dir, pf_flop_samples, 5, 42,
                    )?;
                    eprintln!("{}", pf_audit.summary());
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
        Commands::GpuRangeSolve {
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
            run_gpu_range_solve(
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
        Commands::ValidateBlueprint {
            blueprint,
            spots,
            cluster_dir,
        } => {
            let spots_file = validation_spots::ValidationSpotsFile::load(&spots)?;

            eprintln!("Blueprint Validation");
            eprintln!("  Blueprint: {}", blueprint.display());
            eprintln!("  Spots file: {} ({} spots)", spots.display(), spots_file.spots.len());
            if let Some(ref dir) = cluster_dir {
                eprintln!("  Cluster dir: {}", dir.display());
            }
            eprintln!();

            let mut results = Vec::new();

            for spot in &spots_file.spots {
                eprintln!("  Solving [{}]...", spot.name);
                let vspot = validate_blueprint::ValidationSpot {
                    name: spot.name.clone(),
                    board: spot.board.clone(),
                    oop_range: spot.oop_range.clone(),
                    ip_range: spot.ip_range.clone(),
                    pot: spot.pot,
                    effective_stack: spot.effective_stack,
                };

                match validate_blueprint::solve_spot(&vspot) {
                    Ok(result) => {
                        eprintln!(
                            "    {} hands, {} actions, exploitability={:.4}",
                            result.num_hands, result.num_actions, result.exploitability,
                        );
                        // Print average action frequencies
                        for (ai, action_name) in result.actions_display.iter().enumerate() {
                            let avg_freq: f32 = (0..result.num_hands)
                                .map(|h| result.strategy[ai * result.num_hands + h])
                                .sum::<f32>() / result.num_hands as f32;
                            eprintln!("      {action_name}: {:.1}%", avg_freq * 100.0);
                        }
                        results.push((spot.name.clone(), result));
                    }
                    Err(e) => {
                        eprintln!("    ERROR: {e}");
                    }
                }
                eprintln!();
            }

            // Summary table
            if !results.is_empty() {
                println!();
                println!("{:<40} {:>6} {:>6} {:>12}", "Spot", "Hands", "Acts", "Exploit");
                println!("{}", "-".repeat(66));
                for (name, r) in &results {
                    println!(
                        "{:<40} {:>6} {:>6} {:>12.4}",
                        name, r.num_hands, r.num_actions, r.exploitability,
                    );
                }
            }
        }
        Commands::DiffClusters {
            dir_a,
            dir_b,
            sample_boards,
            verbose,
        } => {
            use poker_solver_core::blueprint_v2::bucket_file::BucketFile;
            use poker_solver_core::blueprint_v2::cluster_diagnostics::diff_bucket_files;

            let streets = ["river", "turn", "flop", "preflop"];
            let mut any_found = false;

            for street_name in &streets {
                let path_a = dir_a.join(format!("{street_name}.buckets"));
                let path_b = dir_b.join(format!("{street_name}.buckets"));

                if !path_a.exists() && !path_b.exists() {
                    continue;
                }
                if !path_a.exists() {
                    eprintln!("warning: {street_name}.buckets missing from dir-a, skipping");
                    continue;
                }
                if !path_b.exists() {
                    eprintln!("warning: {street_name}.buckets missing from dir-b, skipping");
                    continue;
                }

                let bf_a = BucketFile::load(&path_a)?;
                let bf_b = BucketFile::load(&path_b)?;

                eprintln!("diffing {street_name}...");
                let report = diff_bucket_files(&bf_a, &bf_b, sample_boards, 42);
                println!("{}", report.summary(verbose));

                any_found = true;
            }

            if !any_found {
                eprintln!("no matching .buckets files found in both directories");
            }
        }
        Commands::InspectSpot { config, spot } => {
            inspect_spot::run(&config, &spot)
                .map_err(|e| -> Box<dyn Error> { e.into() })?;
        }
        Commands::BenchRollout {
            bundle,
            duration_secs,
            board,
            pot,
            stacks,
            enumerate_depth,
            opponent_samples,
        } => {
            bench_rollout::run(
                &bundle, duration_secs, &board, pot, stacks,
                enumerate_depth, opponent_samples,
            )
            .map_err(|e| -> Box<dyn Error> { e.into() })?;
        }
        Commands::ValidateRollout {
            bundle,
            board,
            pot,
            stacks,
            num_runs,
            pass_threshold,
            enumerate_depth,
            opponent_samples,
        } => {
            validate_rollout::run(
                &bundle, &board, pot, stacks, num_runs, pass_threshold,
                enumerate_depth, opponent_samples,
            )
            .map_err(|e| -> Box<dyn Error> { e.into() })?;
        }
        Commands::CompareSolve {
            bundle,
            snapshot,
            spot,
            iters,
            verbose,
            dump_boundary_cfvs,
            flop_boundary,
            flop_model,
            turn_boundary,
            turn_model,
            river_boundary,
            river_model,
            trace_boundaries,
            trace_iters,
            trace_dir,
            tolerance,
            gadget,
        } => {
            let parse_mode = |mode: &str, model: Option<String>, street: &str|
                -> Result<poker_solver_tauri::StreetBoundaryMode, Box<dyn Error>> {
                match mode {
                    "exact" => Ok(poker_solver_tauri::StreetBoundaryMode::Exact),
                    "cfvnet" => {
                        let path = model.ok_or_else(|| {
                            format!("--{street}-model is required when --{street}-boundary=cfvnet")
                        })?;
                        Ok(poker_solver_tauri::StreetBoundaryMode::Cfvnet { model_path: path })
                    }
                    "exact_subtree" => Ok(poker_solver_tauri::StreetBoundaryMode::ExactSubtree),
                    other => Err(format!(
                        "invalid --{street}-boundary value '{other}': \
                         expected 'exact', 'cfvnet', or 'exact_subtree'"
                    ).into()),
                }
            };
            let sbc = poker_solver_tauri::StreetBoundaryConfig {
                flop: parse_mode(&flop_boundary, flop_model, "flop")?,
                turn: parse_mode(&turn_boundary, turn_model, "turn")?,
                river: parse_mode(&river_boundary, river_model, "river")?,
            };
            let trace_config = boundary_trace::TraceConfig {
                boundaries: trace_boundaries,
                iters_str: trace_iters,
                dir: trace_dir,
            };
            compare_solve::run(
                &bundle,
                snapshot.as_deref(),
                &spot,
                iters,
                verbose,
                dump_boundary_cfvs,
                sbc,
                trace_config,
                tolerance,
                gadget,
            )
            .map_err(|e| -> Box<dyn Error> { e.into() })?;
        }
        Commands::RebelSeed { config } => {
            run_rebel_seed(&config)?;
        }
        Commands::RebelTrain {
            config,
            model,
            offline_only,
        } => {
            run_rebel_train(&config, model.as_deref(), offline_only)?;
        }
        Commands::RebelEval {
            config,
            model,
            mode,
            num_hands,
        } => {
            run_rebel_eval(&config, &model, &mode, num_hands)?;
        }
        Commands::RebelValidate {
            config,
            num_examples,
            output,
        } => {
            run_rebel_validate(&config, num_examples, &output)?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// ReBeL seed data generation
// ---------------------------------------------------------------------------

fn run_rebel_seed(config_path: &std::path::Path) -> Result<(), Box<dyn Error>> {
    use poker_solver_core::blueprint_v2::bucket_file::BucketFile;
    use poker_solver_core::blueprint_v2::bundle::{load_config, BlueprintV2Strategy};
    use poker_solver_core::blueprint_v2::game_tree::GameTree;
    use poker_solver_core::blueprint_v2::mccfr::AllBuckets;

    let yaml = std::fs::read_to_string(config_path)
        .map_err(|e| format!("Failed to read config: {e}"))?;
    let rebel_config: rebel::config::RebelConfig = serde_yaml::from_str(&yaml)
        .map_err(|e| format!("Failed to parse config: {e}"))?;

    // Create output directory if needed
    std::fs::create_dir_all(&rebel_config.output_dir)
        .map_err(|e| format!("Failed to create output dir: {e}"))?;

    // Load blueprint strategy
    let strategy_path = std::path::Path::new(&rebel_config.blueprint_path).join("strategy.bin");
    eprintln!("Loading blueprint from {}...", strategy_path.display());
    let strategy = BlueprintV2Strategy::load(&strategy_path)
        .map_err(|e| format!("Failed to load blueprint: {e}"))?;

    // Load bucket files from cluster directory
    eprintln!("Loading bucket files from {}...", rebel_config.cluster_dir);
    let cluster_dir = std::path::Path::new(&rebel_config.cluster_dir);
    let bucket_names = ["preflop.buckets", "flop.buckets", "turn.buckets", "river.buckets"];
    let mut bucket_files: [Option<BucketFile>; 4] = [None, None, None, None];
    for (i, name) in bucket_names.iter().enumerate() {
        let path = cluster_dir.join(name);
        if path.exists() {
            match BucketFile::load(&path) {
                Ok(bf) => {
                    eprintln!(
                        "  Loaded {}: {} boards, {} combos/board, {} buckets",
                        name, bf.header.board_count, bf.header.combos_per_board, bf.header.bucket_count,
                    );
                    bucket_files[i] = Some(bf);
                }
                Err(e) => eprintln!("  Warning: failed to load {}: {e}", path.display()),
            }
        }
    }

    let bucket_counts = [
        strategy.bucket_counts[0],
        strategy.bucket_counts[1],
        strategy.bucket_counts[2],
        strategy.bucket_counts[3],
    ];
    let buckets = AllBuckets::new(bucket_counts, bucket_files);

    // Auto-detect per-flop bucket files
    let buckets = {
        let per_flop_marker = cluster_dir.join("flop_0000.buckets");
        if per_flop_marker.exists() {
            eprintln!("  Detected per-flop bucket files in {}", cluster_dir.display());
            buckets.with_per_flop_dir(cluster_dir.to_path_buf())
        } else {
            buckets
        }
    };

    // Build game tree from blueprint config
    let bp_config_path = std::path::Path::new(&rebel_config.blueprint_path).join("config.yaml");
    eprintln!("Loading blueprint config from {}...", bp_config_path.display());
    let bp_config = load_config(std::path::Path::new(&rebel_config.blueprint_path))
        .map_err(|e| format!("Failed to load blueprint config: {e}"))?;
    let tree = GameTree::build(
        bp_config.game.stack_depth,
        bp_config.game.small_blind,
        bp_config.game.big_blind,
        &bp_config.action_abstraction.preflop,
        &bp_config.action_abstraction.flop,
        &bp_config.action_abstraction.turn,
        &bp_config.action_abstraction.river,
    );

    // Open or create buffer
    let buffer_path = std::path::Path::new(&rebel_config.output_dir)
        .join(&rebel_config.buffer.path);
    let buffer = if buffer_path.exists() {
        let buf = rebel::data_buffer::DiskBuffer::open(&buffer_path, rebel_config.buffer.max_records)
            .map_err(|e| format!("Failed to open buffer: {e}"))?;
        eprintln!("Resuming from existing buffer: {} records at {}", buf.len(), buffer_path.display());
        std::sync::Mutex::new(buf)
    } else {
        eprintln!("Creating buffer at {}...", buffer_path.display());
        std::sync::Mutex::new(
            rebel::data_buffer::DiskBuffer::create(&buffer_path, rebel_config.buffer.max_records)
                .map_err(|e| format!("Failed to create buffer: {e}"))?
        )
    };

    // Skip PBS generation if buffer already has records
    let existing_count = buffer.lock().unwrap().len();
    if existing_count > 0 {
        eprintln!("Buffer has {} existing records, skipping PBS generation", existing_count);
    } else {
        eprintln!(
            "Generating {} hands with {} threads...",
            rebel_config.seed.num_hands, rebel_config.seed.threads
        );

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(rebel_config.seed.threads)
            .build()
            .map_err(|e| format!("Failed to create thread pool: {e}"))?;

        let pbs_count = pool.install(|| {
            rebel::generate::generate_pbs(&strategy, &tree, &buckets, &rebel_config, &buffer)
        });
        eprintln!("Generated {} PBS snapshots", pbs_count);
    }

    let mut buf = buffer.into_inner().unwrap();
    let record_count = buf.len();

    // --- Step 3: Solve PBSs in buffer → fill CFVs ---
    eprintln!("Solving {} records...", record_count);
    let solve_config = rebel::generate::build_solve_config(&rebel_config.seed);
    let solved = rebel::generate::solve_buffer_records(
        &mut buf,
        &solve_config,
        None,
        rebel_config.seed.threads,
    );
    eprintln!(
        "Solved {}/{} records successfully",
        solved, record_count
    );

    // --- Step 4: Export buffer → cfvnet training files ---
    let export_path = std::path::Path::new(&rebel_config.output_dir)
        .join("training_data.bin");
    eprintln!("Exporting training data to {}...", export_path.display());
    let exported = rebel::training::export_training_data(&buf, &export_path)
        .map_err(|e| format!("Failed to export training data: {e}"))?;
    eprintln!(
        "Done! Exported {} training records to {}",
        exported,
        export_path.display()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// ReBeL training (offline seeding + optional self-play)
// ---------------------------------------------------------------------------

fn run_rebel_train(
    config_path: &std::path::Path,
    existing_model: Option<&str>,
    offline_only: bool,
) -> Result<(), Box<dyn Error>> {
    use poker_solver_core::blueprint_v2::bucket_file::BucketFile;
    use poker_solver_core::blueprint_v2::bundle::{load_config, BlueprintV2Strategy};
    use poker_solver_core::blueprint_v2::game_tree::GameTree;
    use poker_solver_core::blueprint_v2::mccfr::AllBuckets;

    let yaml = std::fs::read_to_string(config_path)
        .map_err(|e| format!("Failed to read config: {e}"))?;
    let rebel_config: rebel::config::RebelConfig = serde_yaml::from_str(&yaml)
        .map_err(|e| format!("Failed to parse config: {e}"))?;

    eprintln!("ReBeL Training Pipeline");
    eprintln!("  Blueprint: {}", rebel_config.blueprint_path);
    eprintln!("  Cluster dir: {}", rebel_config.cluster_dir);
    eprintln!("  Output dir: {}", rebel_config.output_dir);
    if let Some(model_path) = existing_model {
        eprintln!("  Existing model: {model_path} (skipping offline seeding)");
    }
    if offline_only {
        eprintln!("  Mode: offline seeding only (no self-play)");
    }
    eprintln!();

    // Create output directory
    std::fs::create_dir_all(&rebel_config.output_dir)
        .map_err(|e| format!("Failed to create output dir: {e}"))?;

    // Load blueprint strategy
    let strategy_path = std::path::Path::new(&rebel_config.blueprint_path).join("strategy.bin");
    eprintln!("Loading blueprint from {}...", strategy_path.display());
    let strategy = BlueprintV2Strategy::load(&strategy_path)
        .map_err(|e| format!("Failed to load blueprint: {e}"))?;

    // Load bucket files
    eprintln!("Loading bucket files from {}...", rebel_config.cluster_dir);
    let cluster_dir = std::path::Path::new(&rebel_config.cluster_dir);
    let bucket_names = ["preflop.buckets", "flop.buckets", "turn.buckets", "river.buckets"];
    let mut bucket_files: [Option<BucketFile>; 4] = [None, None, None, None];
    for (i, name) in bucket_names.iter().enumerate() {
        let path = cluster_dir.join(name);
        if path.exists() {
            match BucketFile::load(&path) {
                Ok(bf) => {
                    eprintln!(
                        "  Loaded {}: {} boards, {} combos/board, {} buckets",
                        name, bf.header.board_count, bf.header.combos_per_board, bf.header.bucket_count,
                    );
                    bucket_files[i] = Some(bf);
                }
                Err(e) => eprintln!("  Warning: failed to load {}: {e}", path.display()),
            }
        }
    }

    let bucket_counts = [
        strategy.bucket_counts[0],
        strategy.bucket_counts[1],
        strategy.bucket_counts[2],
        strategy.bucket_counts[3],
    ];
    let buckets = AllBuckets::new(bucket_counts, bucket_files);

    // Auto-detect per-flop bucket files
    let buckets = {
        let per_flop_marker = cluster_dir.join("flop_0000.buckets");
        if per_flop_marker.exists() {
            eprintln!("  Detected per-flop bucket files in {}", cluster_dir.display());
            buckets.with_per_flop_dir(cluster_dir.to_path_buf())
        } else {
            buckets
        }
    };

    // Build game tree from blueprint config
    let bp_config_path = std::path::Path::new(&rebel_config.blueprint_path).join("config.yaml");
    eprintln!("Loading blueprint config from {}...", bp_config_path.display());
    let bp_config = load_config(std::path::Path::new(&rebel_config.blueprint_path))
        .map_err(|e| format!("Failed to load blueprint config: {e}"))?;
    let tree = GameTree::build(
        bp_config.game.stack_depth,
        bp_config.game.small_blind,
        bp_config.game.big_blind,
        &bp_config.action_abstraction.preflop,
        &bp_config.action_abstraction.flop,
        &bp_config.action_abstraction.turn,
        &bp_config.action_abstraction.river,
    );

    // Step 1: Offline seeding (unless an existing model was provided)
    let model_path = if let Some(model_path) = existing_model {
        eprintln!("Skipping offline seeding — using existing model: {model_path}");
        std::path::PathBuf::from(model_path)
    } else {
        eprintln!("\n--- Phase 1: Offline Seeding ---");
        let result = rebel::orchestration::run_offline_seeding(
            &rebel_config, &strategy, &tree, &buckets,
        )?;
        eprintln!(
            "\nOffline seeding complete: {} total records, model at {}",
            result.total_records,
            result.model_path.display()
        );
        for sr in &result.per_street {
            eprintln!(
                "  {:?}: {} PBSs, {} solved, loss={:.6}",
                sr.street,
                sr.pbs_generated,
                sr.records_solved,
                sr.training_loss.unwrap_or(0.0)
            );
        }
        result.model_path
    };

    // Step 2: Self-play (unless --offline-only)
    if offline_only {
        eprintln!("\nOffline-only mode — skipping self-play.");
        eprintln!("Model saved at: {}", model_path.display());
    } else {
        use std::sync::atomic::{AtomicBool, Ordering};

        use burn::backend::{Autodiff, wgpu::Wgpu};
        use burn::module::Module;
        use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
        use cfvnet::model::network::CfvNet;
        use rebel::inference_server::{spawn_inference_server, InferenceServerConfig};
        use rebel::replay_buffer::ReplayBuffer;
        use rebel::self_play::{self_play_training_loop, SelfPlayConfig};

        type TrainBackend = Autodiff<Wgpu>;

        eprintln!("\n--- Phase 2: Self-Play Training ---");
        eprintln!("Model at: {}", model_path.display());

        let device = burn::backend::wgpu::WgpuDevice::default();
        let mut model = CfvNet::<TrainBackend>::new(
            &device,
            rebel_config.training.hidden_layers,
            rebel_config.training.hidden_size,
            cfvnet::model::network::INPUT_SIZE,
        );

        // Load weights from checkpoint if available
        let recorder = burn::record::NamedMpkGzFileRecorder::<
            burn::record::FullPrecisionSettings,
        >::new();
        let model_file = model_path.join("model");
        if model_file.with_extension("mpk.gz").exists() {
            match model.clone().load_file(&model_file, &recorder, &device) {
                Ok(loaded) => {
                    model = loaded;
                    eprintln!("  Loaded model from {}", model_file.display());
                }
                Err(e) => {
                    eprintln!("  Warning: failed to load model, using random init: {e}");
                }
            }
        } else {
            eprintln!("  No checkpoint found at {}, using random init", model_file.display());
        }
        eprintln!(
            "  CfvNet: {} layers x {} hidden",
            rebel_config.training.hidden_layers, rebel_config.training.hidden_size,
        );

        // Create replay buffer.
        let replay_buffer = Arc::new(ReplayBuffer::new(
            rebel_config.inference.replay_capacity,
        ));

        // Spawn inference server.
        let shutdown = Arc::new(AtomicBool::new(false));
        let inf_config = InferenceServerConfig {
            batch_size: rebel_config.inference.batch_size,
            batch_timeout_us: rebel_config.inference.batch_timeout_us,
            train_every_n_solves: rebel_config.inference.train_every_n_solves,
            train_batch_size: rebel_config.inference.train_batch_size,
            learning_rate: rebel_config.inference.learning_rate,
            checkpoint_dir: Some(model_path.clone()),
            checkpoint_every_n_steps: 100,
        };
        let (handle, server_thread) = spawn_inference_server(
            model,
            device,
            inf_config,
            Arc::clone(&replay_buffer),
            Arc::clone(&shutdown),
        );
        eprintln!("  Inference server spawned (batch_size={}, timeout={}us)",
            rebel_config.inference.batch_size, rebel_config.inference.batch_timeout_us);

        // Build SolveConfig and SelfPlayConfig.
        let solve_config = rebel::generate::build_solve_config(&rebel_config.seed);
        let sp_config = SelfPlayConfig {
            num_hands: rebel_config.seed.num_hands,
            cfr_iterations: rebel_config.seed.solver_iterations,
            exploration_epsilon: 0.25,
            initial_stack: rebel_config.game.initial_stack,
            small_blind: rebel_config.game.small_blind,
            big_blind: rebel_config.game.big_blind,
            hands_per_training_batch: 100,
            seed: rebel_config.seed.seed,
        };

        // Run self-play training loop.
        let total = self_play_training_loop(
            &handle,
            &solve_config,
            &sp_config,
            &strategy,
            &tree,
            &buckets,
            &replay_buffer,
        );

        // Shutdown inference server.
        shutdown.store(true, Ordering::Relaxed);
        server_thread.join().expect("inference server thread panicked");
        eprintln!("Self-play complete: {total} examples generated");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// ReBeL evaluation
// ---------------------------------------------------------------------------

fn run_rebel_eval(
    config_path: &std::path::Path,
    model_path: &str,
    mode: &str,
    num_hands: usize,
) -> Result<(), Box<dyn Error>> {
    let yaml = std::fs::read_to_string(config_path)
        .map_err(|e| format!("Failed to read config: {e}"))?;
    let rebel_config: rebel::config::RebelConfig = serde_yaml::from_str(&yaml)
        .map_err(|e| format!("Failed to parse config: {e}"))?;

    eprintln!("ReBeL Evaluation");
    eprintln!("  Config: {}", config_path.display());
    eprintln!("  Model: {model_path}");
    eprintln!("  Mode: {mode}");
    if mode == "h2h" {
        eprintln!("  Num hands: {num_hands}");
    }
    eprintln!();

    // Verify model exists
    let model_file = std::path::Path::new(model_path);
    if !model_file.exists() {
        return Err(format!("Model not found at: {model_path}").into());
    }

    match mode {
        "mse" => {
            // Generate a small held-out validation set (river only for now)
            eprintln!("Generating held-out validation set...");
            let solve_config = rebel::generate::build_solve_config(&rebel_config.seed);
            let val_records = rebel::validation::generate_validation_set(
                100,  // 100 validation examples
                &solve_config,
                rebel_config.seed.seed + 999999,  // different seed from training
            );
            eprintln!("Generated {} validation records", val_records.len());

            eprintln!("  River records: {}", val_records.iter().filter(|r| r.board_card_count == 5).count());

            // Model MSE evaluation requires loading CfvNet — not yet wired.
            eprintln!();
            eprintln!(
                "Model MSE evaluation requires loading CfvNet ({} layers x {} units) — not yet wired.",
                rebel_config.training.hidden_layers, rebel_config.training.hidden_size
            );
            eprintln!("Validation set generated with {} records.", val_records.len());
        }
        "h2h" => {
            eprintln!("Head-to-head evaluation: {} hands", num_hands);
            eprintln!("This requires both a ReBeL agent (subgame solving at each decision)");
            eprintln!("and a blueprint agent (table lookup). Not yet implemented.");
        }
        other => {
            return Err(format!("Unknown evaluation mode: '{other}'. Use 'mse' or 'h2h'.").into());
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// ReBeL validation set generation
// ---------------------------------------------------------------------------

fn run_rebel_validate(
    config_path: &std::path::Path,
    num_examples: usize,
    output: &str,
) -> Result<(), Box<dyn Error>> {
    let yaml = std::fs::read_to_string(config_path)
        .map_err(|e| format!("Failed to read config: {e}"))?;
    let rebel_config: rebel::config::RebelConfig = serde_yaml::from_str(&yaml)
        .map_err(|e| format!("Failed to parse config: {e}"))?;

    eprintln!("ReBeL Validation Set Generator");
    eprintln!("  Config: {}", config_path.display());
    eprintln!("  Examples: {num_examples}");
    eprintln!("  Output: {output}");
    eprintln!();

    // Build solve config from rebel seed settings
    let solve_config = rebel::generate::build_solve_config(&rebel_config.seed);

    // Use a different seed from training to ensure held-out data
    let val_seed = rebel_config.seed.seed + 999999;

    eprintln!("Generating validation set (seed={val_seed})...");
    let start = std::time::Instant::now();
    let val_records = rebel::validation::generate_validation_set(
        num_examples,
        &solve_config,
        val_seed,
    );
    let elapsed = start.elapsed();

    let river_count = val_records.iter().filter(|r| r.board_card_count == 5).count();
    eprintln!("Generated {} records ({} river) in {:.1}s", val_records.len(), river_count, elapsed.as_secs_f64());

    // Save to output file using DiskBuffer
    let output_path = std::path::Path::new(output);
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create output directory: {e}"))?;
        }
    }

    let mut buf = rebel::data_buffer::DiskBuffer::create(output_path, val_records.len())
        .map_err(|e| format!("Failed to create output file: {e}"))?;
    for rec in &val_records {
        buf.append(rec).map_err(|e| format!("Failed to write record: {e}"))?;
    }
    eprintln!("Saved {} validation records to {output}", buf.len());

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

fn run_gpu_range_solve(
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
    use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::{card_from_str, flop_from_str, hole_to_string, CardConfig, NOT_DEALT};
    use range_solver::range::Range;

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

    let mut game = range_solver::PostFlopGame::with_config(card_config, action_tree)
        .map_err(|e| format!("Failed to build game: {e}"))?;

    // --- Print game info ---
    eprintln!("GPU Range Solver (Discounted CFR)");
    eprintln!("  Board: {flop_str}{}", format_board_suffix(turn_str, river_str));
    eprintln!("  Initial state: {initial_state}");
    eprintln!("  Pot: {pot}, Effective stack: {effective_stack}");
    eprintln!(
        "  OOP hands: {}, IP hands: {}",
        game.private_cards(0).len(),
        game.private_cards(1).len(),
    );
    eprintln!();

    // --- Allocate memory and solve on GPU ---
    game.allocate_memory(false);

    let config = gpu_range_solver::GpuSolverConfig {
        max_iterations: iterations,
        target_exploitability,
        print_progress: true,
    };

    let start = Instant::now();
    let result = gpu_range_solver::gpu_solve_hand_parallel(&game, &config);
    let elapsed = start.elapsed();

    eprintln!();
    eprintln!(
        "Solved in {:.2}s ({} iterations)",
        elapsed.as_secs_f64(),
        result.iterations_run,
    );
    eprintln!("Final exploitability: {:.4}", result.exploitability);
    eprintln!();

    // --- Print root actions and strategy summary ---
    game.back_to_root();
    let actions = game.available_actions();
    let player = game.current_player();
    let hands = game.private_cards(player);
    let num_hands = hands.len();
    let num_actions = actions.len();

    println!(
        "Root node: {} to act ({num_actions} actions, {num_hands} hands)",
        if player == 0 { "OOP" } else { "IP" }
    );
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
            let prob = result.root_strategy[a * num_hands + h];
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
// N-player blueprint training
// ---------------------------------------------------------------------------

fn run_train_blueprint_mp(path: &str, no_tui: bool) -> Result<(), Box<dyn Error>> {
    let yaml = std::fs::read_to_string(path)?;
    let config: BlueprintMpConfig = serde_yaml::from_str(&yaml)?;
    config
        .game
        .validate()
        .map_err(|e| format!("invalid config: {e}"))?;
    let tui_config = blueprint_tui_config::parse_tui_config(&yaml);
    eprintln!(
        "Starting N-player blueprint training: {} ({} players, {}bb deep)",
        config.game.name,
        config.game.num_players,
        config.game.stack_depth / 2.0
    );
    if tui_config.enabled && !no_tui {
        run_mp_with_tui(&config, &tui_config)?;
    } else {
        let result = train_blueprint_mp(&config);
        eprintln!("Training complete: {} meta-iterations", result.meta_iterations);
    }
    Ok(())
}

fn run_mp_with_tui(
    config: &BlueprintMpConfig,
    tui_config: &blueprint_tui_config::BlueprintTuiConfig,
) -> Result<(), Box<dyn Error>> {
    use poker_solver_core::blueprint_mp::trainer::{run_training, setup_training};

    let ctx = setup_training(config);
    let scenarios = resolve_tui_scenarios(
        &ctx.tree, &tui_config.scenarios, config.game.num_players,
    );
    let metrics = Arc::new(blueprint_tui_metrics::BlueprintTuiMetrics::new(
        config.training.iterations,
        config.training.time_limit_minutes,
    ));
    let shared_iters = Arc::clone(&ctx.iterations);
    let quit_flag = Arc::clone(&ctx.quit);
    let storage = Arc::clone(&ctx.storage);
    let tree = Arc::clone(&ctx.tree);
    let scenario_node_ids: Vec<u32> = scenarios.iter().map(|s| s.node_idx).collect();
    let tui_handle = spawn_mp_tui(
        &metrics, scenarios, tui_config, config.game.num_players,
    );
    let train_config = config.clone();
    let train_handle = std::thread::spawn(move || {
        run_training(&ctx, &train_config.training, &train_config.game)
    });
    bridge_mp_iterations(
        &shared_iters, &storage, &tree, &scenario_node_ids,
        &metrics, &quit_flag, &train_handle,
    );
    // Signal both the TUI and training thread to stop
    metrics.quit_requested.store(true, Ordering::Relaxed);
    quit_flag.store(true, Ordering::Relaxed);
    let result = train_handle.join().expect("training thread panicked");
    let _ = tui_handle.join();
    eprintln!("Training complete: {} meta-iterations", result.meta_iterations);
    Ok(())
}

fn spawn_mp_tui(
    metrics: &Arc<blueprint_tui_metrics::BlueprintTuiMetrics>,
    scenarios: Vec<mp_tui::ResolvedMpScenario>,
    tui_config: &blueprint_tui_config::BlueprintTuiConfig,
    num_players: u8,
) -> std::thread::JoinHandle<()> {
    let refresh = Duration::from_millis(tui_config.refresh_rate_ms);
    mp_tui::run_mp_tui(
        Arc::clone(metrics),
        scenarios,
        tui_config.telemetry.clone(),
        refresh,
        num_players,
    )
}

fn bridge_mp_iterations<T>(
    source: &Arc<AtomicU64>,
    storage: &Arc<poker_solver_core::blueprint_mp::storage::MpStorage>,
    tree: &Arc<poker_solver_core::blueprint_mp::game_tree::MpGameTree>,
    scenario_node_ids: &[u32],
    metrics: &Arc<blueprint_tui_metrics::BlueprintTuiMetrics>,
    quit_flag: &Arc<std::sync::atomic::AtomicBool>,
    handle: &std::thread::JoinHandle<T>,
) {
    let mut last_telemetry = Instant::now();
    let telemetry_interval = Duration::from_secs(10);
    loop {
        std::thread::sleep(Duration::from_millis(50));
        let iters = source.load(Ordering::Relaxed);
        metrics.iterations.store(iters, Ordering::Relaxed);
        if last_telemetry.elapsed() >= telemetry_interval {
            // Spawn telemetry scan on a background thread to avoid blocking
            // the iteration counter bridge (the scan touches billions of entries).
            let s = Arc::clone(storage);
            let t = Arc::clone(tree);
            let m = Arc::clone(metrics);
            let nodes = scenario_node_ids.to_vec();
            std::thread::spawn(move || push_mp_telemetry(&s, &t, &nodes, &m, iters));
            last_telemetry = Instant::now();
        }
        if handle.is_finished() {
            metrics.iterations.store(source.load(Ordering::Relaxed), Ordering::Relaxed);
            break;
        }
        if metrics.quit_requested.load(Ordering::Relaxed) {
            quit_flag.store(true, Ordering::Relaxed);
            break;
        }
    }
}

fn push_mp_telemetry(
    storage: &poker_solver_core::blueprint_mp::storage::MpStorage,
    tree: &poker_solver_core::blueprint_mp::game_tree::MpGameTree,
    scenario_node_ids: &[u32],
    metrics: &blueprint_tui_metrics::BlueprintTuiMetrics,
    iters: u64,
) {
    // Push regret telemetry
    mp_tui::push_regret_telemetry(
        &storage.regrets,
        poker_solver_core::blueprint_mp::storage::REGRET_SCALE,
        metrics,
    );
    // Push prune stats
    use poker_solver_core::blueprint_mp::trainer::{PRUNE_HITS, PRUNE_TOTAL};
    let hits = PRUNE_HITS.swap(0, Ordering::Relaxed);
    let total = PRUNE_TOTAL.swap(0, Ordering::Relaxed);
    let prune_pct = if total > 0 { hits as f64 / total as f64 * 100.0 } else { 0.0 };
    metrics.push_prune_fraction(prune_pct);
    // Push strategy grids for each scenario
    for (idx, &node_idx) in scenario_node_ids.iter().enumerate() {
        let grid_state = mp_tui_scenarios::extract_mp_grid(tree, storage, node_idx, iters, "");
        if let Ok(mut grids) = metrics.strategy_grids.lock() {
            if idx < grids.len() {
                grids[idx] = Some(grid_state.cells);
            }
        }
    }
}

fn resolve_tui_scenarios(
    tree: &poker_solver_core::blueprint_mp::game_tree::MpGameTree,
    configs: &[blueprint_tui_config::ScenarioConfig],
    num_players: u8,
) -> Vec<mp_tui::ResolvedMpScenario> {
    configs
        .iter()
        .filter_map(|sc| {
            let (node_idx, _board) = mp_tui_scenarios::resolve_mp_spot(tree, &sc.spot, num_players)?;
            Some(mp_tui::ResolvedMpScenario {
                name: sc.name.clone(),
                node_idx,
                grid: default_hand_grid_state(&sc.name),
            })
        })
        .collect()
}

fn default_hand_grid_state(name: &str) -> blueprint_tui_widgets::HandGridState {
    blueprint_tui_widgets::HandGridState {
        cells: std::array::from_fn(|_| std::array::from_fn(|_| Default::default())),
        prev_cells: None,
        scenario_name: name.to_string(),
        action_path: vec![],
        board_display: None,
        cluster_id: None,
        street_label: "Preflop".to_string(),
        iteration_at_snapshot: 0,
        error_message: None,
    }
}

#[cfg(test)]
mod tests {
    use poker_solver_core::blueprint_mp::config::BlueprintMpConfig;
    use poker_solver_core::blueprint_v2::config::BlueprintV2Config;
    use poker_solver_core::blueprint_v2::cluster_pipeline::PerFlopClusteringConfig;
    use test_macros::timed_test;

    /// The sample per_flop_200bkt.yaml must parse and have per_flop set.
    #[test]
    fn per_flop_sample_yaml_parses() {
        let yaml = std::fs::read_to_string(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../sample_configurations/per_flop_200bkt.yaml"),
        )
        .expect("sample per_flop_200bkt.yaml must exist");
        let cfg: BlueprintV2Config =
            serde_yaml::from_str(&yaml).expect("YAML must parse as BlueprintV2Config");
        assert!(
            cfg.clustering.per_flop.is_some(),
            "per_flop section must be present"
        );
        let pf = cfg.clustering.per_flop.as_ref().unwrap();
        assert_eq!(pf.turn_buckets, 200);
        assert_eq!(pf.river_buckets, 200);
    }

    /// PerFlopClusteringConfig can be constructed from a parsed config with per_flop.
    #[test]
    fn per_flop_config_construction_from_yaml() {
        let yaml = std::fs::read_to_string(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../sample_configurations/per_flop_200bkt.yaml"),
        )
        .expect("sample per_flop_200bkt.yaml must exist");
        let cfg: BlueprintV2Config =
            serde_yaml::from_str(&yaml).expect("YAML must parse");

        let pf = cfg.clustering.per_flop.as_ref().unwrap();
        let per_flop_config = PerFlopClusteringConfig {
            flop_buckets: cfg.clustering.flop.buckets,
            turn_buckets: pf.turn_buckets,
            river_buckets: pf.river_buckets,
            kmeans_iterations: cfg.clustering.kmeans_iterations,
            seed: cfg.clustering.seed,
        };

        assert_eq!(per_flop_config.flop_buckets, 200);
        assert_eq!(per_flop_config.turn_buckets, 200);
        assert_eq!(per_flop_config.river_buckets, 200);
        assert_eq!(per_flop_config.seed, 42);
    }

    /// The gpu-range-solve subcommand should be parseable by clap.
    #[test]
    fn gpu_range_solve_cli_parses() {
        use clap::Parser;
        let cli = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "gpu-range-solve",
            "--oop-range", "AA",
            "--ip-range", "KK",
            "--flop", "Qs Jh 2c",
            "--turn", "8d",
            "--river", "3s",
            "--pot", "100",
            "--effective-stack", "100",
            "--iterations", "100",
        ]);
        assert!(cli.is_ok(), "gpu-range-solve CLI must parse: {:?}", cli.err());
    }

    /// run_gpu_range_solve constructs a game and solves it without error.
    #[test]
    fn run_gpu_range_solve_returns_ok() {
        let result = super::run_gpu_range_solve(
            "AA",
            "KK",
            "Qs Jh 2c",
            Some("8d"),
            Some("3s"),
            100,
            100,
            10,  // low iterations for speed
            10.0, // high target so it stops fast
            "100%",
            "",
            "100%",
            "",
        );
        assert!(result.is_ok(), "run_gpu_range_solve must succeed: {:?}", result.err());
    }

    /// A config without per_flop should have per_flop as None.
    #[test]
    fn standard_config_has_no_per_flop() {
        let yaml = std::fs::read_to_string(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../sample_configurations/blueprint_v2_500bkt.yaml"),
        )
        .expect("blueprint_v2_500bkt.yaml must exist");
        let cfg: BlueprintV2Config =
            serde_yaml::from_str(&yaml).expect("YAML must parse");
        assert!(
            cfg.clustering.per_flop.is_none(),
            "standard config should not have per_flop"
        );
    }

    /// The sample TUI config with regret_audits must parse and contain two audit entries.
    #[test]
    fn tui_sample_yaml_parses_regret_audits() {
        let yaml = std::fs::read_to_string(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../sample_configurations/blueprint_v2_with_tui.yaml"),
        )
        .expect("blueprint_v2_with_tui.yaml must exist");
        let tui_cfg = crate::blueprint_tui_config::parse_tui_config(&yaml);
        assert_eq!(
            tui_cfg.regret_audits.len(),
            2,
            "expected 2 regret_audits entries in sample config"
        );
        assert_eq!(tui_cfg.regret_audits[0].name, "AKo SB open");
        assert_eq!(tui_cfg.regret_audits[0].hand, "AKo");
        assert_eq!(
            tui_cfg.regret_audits[0].player,
            crate::blueprint_tui_config::PlayerLabel::Sb,
        );
        assert_eq!(tui_cfg.regret_audits[1].name, "72o SB open");
        assert_eq!(tui_cfg.regret_audits[1].hand, "72o");
    }

    // -- train-blueprint-mp CLI tests --

    /// The train-blueprint-mp subcommand should be parseable by clap.
    #[timed_test]
    fn train_blueprint_mp_cli_parses() {
        use clap::Parser;
        let cli = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "train-blueprint-mp",
            "--config",
            "/tmp/test.yaml",
        ]);
        assert!(
            cli.is_ok(),
            "train-blueprint-mp CLI must parse: {:?}",
            cli.err()
        );
    }

    /// The 3-player sample config must parse as BlueprintMpConfig.
    #[timed_test]
    fn mp_3player_sample_yaml_parses() {
        let yaml = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sample_configurations/blueprint_mp_3player.yaml"
        ))
        .expect("blueprint_mp_3player.yaml must exist");
        let cfg: BlueprintMpConfig =
            serde_yaml::from_str(&yaml).expect("YAML must parse as BlueprintMpConfig");
        assert_eq!(cfg.game.num_players, 3);
        assert_eq!(cfg.game.blinds.len(), 2);
        assert!(cfg.game.validate().is_ok());
    }

    /// The 6-player ante sample config must parse as BlueprintMpConfig.
    #[timed_test]
    fn mp_6player_ante_sample_yaml_parses() {
        let yaml = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sample_configurations/blueprint_mp_6player_ante.yaml"
        ))
        .expect("blueprint_mp_6player_ante.yaml must exist");
        let cfg: BlueprintMpConfig =
            serde_yaml::from_str(&yaml).expect("YAML must parse as BlueprintMpConfig");
        assert_eq!(cfg.game.num_players, 6);
        assert_eq!(cfg.game.blinds.len(), 3);
        assert!(cfg.game.validate().is_ok());
    }

    /// run_train_blueprint_mp returns an error for a non-existent file.
    #[timed_test]
    fn run_train_blueprint_mp_missing_file_errors() {
        let result = super::run_train_blueprint_mp("/tmp/nonexistent_mp_config.yaml", false);
        assert!(result.is_err());
    }

    /// run_train_blueprint_mp returns an error for invalid YAML content.
    #[timed_test]
    fn run_train_blueprint_mp_invalid_yaml_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.yaml");
        std::fs::write(&path, "not: valid: blueprint: mp: config: [").unwrap();
        let result = super::run_train_blueprint_mp(path.to_str().unwrap(), false);
        assert!(result.is_err());
    }

    /// run_train_blueprint_mp returns an error when game config is invalid.
    #[timed_test]
    fn run_train_blueprint_mp_invalid_game_config_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("invalid_game.yaml");
        let yaml = r#"
game:
  name: "bad"
  num_players: 99
  stack_depth: 100
  blinds:
    - seat: 0
      type: small_blind
      amount: 1

action_abstraction:
  preflop:
    lead: [1.0]
    raise:
      - [1.0]
  flop:
    lead: [1.0]
    raise:
      - [1.0]
  turn:
    lead: [1.0]
    raise:
      - [1.0]
  river:
    lead: [1.0]
    raise:
      - [1.0]

clustering:
  preflop:
    buckets: 10
  flop:
    buckets: 10
  turn:
    buckets: 10
  river:
    buckets: 10

training:
  iterations: 1

snapshots:
  warmup_minutes: 1
  snapshot_every_minutes: 1
  output_dir: "/tmp/bad"
"#;
        std::fs::write(&path, yaml).unwrap();
        let result = super::run_train_blueprint_mp(path.to_str().unwrap(), false);
        assert!(result.is_err(), "should reject num_players=99");
    }

    /// The 6-player ante sample config should have a tui section after update.
    #[timed_test]
    fn mp_6player_tui_section_parses() {
        let yaml = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sample_configurations/blueprint_mp_6player_ante.yaml"
        ))
        .expect("blueprint_mp_6player_ante.yaml must exist");
        let tui_cfg = crate::blueprint_tui_config::parse_tui_config(&yaml);
        // TUI should be disabled by default but scenarios should be present
        assert!(!tui_cfg.enabled);
        assert!(!tui_cfg.scenarios.is_empty(), "should have TUI scenarios");
    }

    /// The train-blueprint-mp subcommand should accept --no-tui.
    #[timed_test]
    fn train_blueprint_mp_no_tui_cli_parses() {
        use clap::Parser;
        let cli = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "train-blueprint-mp",
            "--config",
            "/tmp/test.yaml",
            "--no-tui",
        ]);
        assert!(
            cli.is_ok(),
            "train-blueprint-mp --no-tui must parse: {:?}",
            cli.err()
        );
    }

    /// resolve_tui_scenarios should produce ResolvedMpScenario from configs.
    #[timed_test(10)]
    fn resolve_tui_scenarios_from_tree() {
        use poker_solver_core::blueprint_mp::config::*;
        use poker_solver_core::blueprint_mp::game_tree::MpGameTree;

        fn yaml_f64(v: f64) -> serde_yaml::Value {
            serde_yaml::Value::Number(serde_yaml::Number::from(v))
        }

        let game = MpGameConfig {
            name: "test".into(),
            num_players: 6,
            stack_depth: 200.0,
            blinds: vec![
                ForcedBet { seat: 4, kind: ForcedBetKind::SmallBlind, amount: 1.0 },
                ForcedBet { seat: 5, kind: ForcedBetKind::BigBlind, amount: 2.0 },
            ],
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        let preflop = MpStreetSizes {
            lead: vec![serde_yaml::Value::String("5bb".into())],
            raise: vec![vec![serde_yaml::Value::String("5bb".into())]],
        };
        let postflop = MpStreetSizes {
            lead: vec![yaml_f64(0.67)],
            raise: vec![vec![yaml_f64(1.0)]],
        };
        let action = MpActionAbstractionConfig {
            preflop,
            flop: postflop.clone(),
            turn: postflop.clone(),
            river: postflop,
        };
        let tree = MpGameTree::build(&game, &action);
        let scenarios = vec![
            crate::blueprint_tui_config::ScenarioConfig {
                name: "UTG open".into(),
                spot: String::new(),
            },
            crate::blueprint_tui_config::ScenarioConfig {
                name: "HJ vs UTG".into(),
                spot: "utg:5bb".into(),
            },
            crate::blueprint_tui_config::ScenarioConfig {
                name: "Invalid spot".into(),
                spot: "xyz:999bb".into(),
            },
        ];
        let resolved = super::resolve_tui_scenarios(&tree, &scenarios, 6);
        // First two should resolve; third should be filtered out
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].name, "UTG open");
        assert_eq!(resolved[1].name, "HJ vs UTG");
    }

    /// compare-solve should accept per-street boundary flags.
    #[test]
    fn compare_solve_street_boundary_cli_flags_parse() {
        use clap::Parser;

        // All-exact defaults
        let cli = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "compare-solve",
            "--bundle", "/tmp/test",
            "--spot", "sb:2bb,bb:call|Jd9d7d",
        ]);
        assert!(cli.is_ok(), "default flags must parse: {:?}", cli.err());

        if let super::Commands::CompareSolve {
            flop_boundary,
            turn_boundary,
            river_boundary,
            river_model,
            ..
        } = cli.unwrap().command {
            assert_eq!(flop_boundary, "exact");
            assert_eq!(turn_boundary, "exact");
            assert_eq!(river_boundary, "exact");
            assert!(river_model.is_none());
        } else {
            panic!("expected CompareSolve variant");
        }

        // River cfvnet with model path
        let cli2 = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "compare-solve",
            "--bundle", "/tmp/test",
            "--spot", "sb:2bb,bb:call|Jd9d7d",
            "--river-boundary", "cfvnet",
            "--river-model", "/path/to/model.onnx",
        ]);
        assert!(cli2.is_ok(), "river-cfvnet flags must parse: {:?}", cli2.err());

        if let super::Commands::CompareSolve {
            river_boundary,
            river_model,
            ..
        } = cli2.unwrap().command {
            assert_eq!(river_boundary, "cfvnet");
            assert_eq!(river_model.as_deref(), Some("/path/to/model.onnx"));
        } else {
            panic!("expected CompareSolve variant");
        }
    }

    /// compare-solve should accept --trace-boundaries, --trace-iters, --trace-dir flags.
    #[test]
    fn compare_solve_trace_flags_parse() {
        use clap::Parser;

        // With trace flags
        let cli = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "compare-solve",
            "--bundle", "/tmp/test",
            "--spot", "sb:2bb,bb:call|Jd9d7d",
            "--trace-boundaries", "0,42,100",
            "--trace-iters", "0,9",
            "--trace-dir", "/tmp/traces",
        ]);
        assert!(cli.is_ok(), "trace flags must parse: {:?}", cli.err());

        if let super::Commands::CompareSolve {
            trace_boundaries,
            trace_iters,
            trace_dir,
            ..
        } = cli.unwrap().command {
            assert_eq!(trace_boundaries.as_deref(), Some("0,42,100"));
            assert_eq!(trace_iters, "0,9");
            assert_eq!(trace_dir, std::path::PathBuf::from("/tmp/traces"));
        } else {
            panic!("expected CompareSolve variant");
        }

        // Default trace values (no --trace-boundaries means None)
        let cli2 = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "compare-solve",
            "--bundle", "/tmp/test",
            "--spot", "sb:2bb,bb:call|Jd9d7d",
        ]);
        assert!(cli2.is_ok());

        if let super::Commands::CompareSolve {
            trace_boundaries,
            trace_iters,
            trace_dir,
            ..
        } = cli2.unwrap().command {
            assert!(trace_boundaries.is_none());
            assert_eq!(trace_iters, "last");
            assert_eq!(trace_dir, std::path::PathBuf::from("./traces"));
        } else {
            panic!("expected CompareSolve variant");
        }
    }

    /// compare-solve should accept --gadget flag.
    #[test]
    fn compare_solve_gadget_flag_parse() {
        use clap::Parser;

        // Default: gadget off
        let cli = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "compare-solve",
            "--bundle", "/tmp/test",
            "--spot", "sb:2bb,bb:call|Jd9d7d",
        ]);
        assert!(cli.is_ok());
        if let super::Commands::CompareSolve { gadget, .. } = cli.unwrap().command {
            assert!(!gadget, "gadget should default to false");
        } else {
            panic!("expected CompareSolve variant");
        }

        // Explicit --gadget
        let cli2 = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "compare-solve",
            "--bundle", "/tmp/test",
            "--spot", "sb:2bb,bb:call|Jd9d7d",
            "--gadget",
        ]);
        assert!(cli2.is_ok());
        if let super::Commands::CompareSolve { gadget, .. } = cli2.unwrap().command {
            assert!(gadget, "gadget should be true when --gadget is passed");
        } else {
            panic!("expected CompareSolve variant");
        }
    }
}
