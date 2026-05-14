mod bench_rollout;
mod blueprint_tui;
mod blueprint_tui_audit;
mod blueprint_tui_audit_widget;
mod blueprint_tui_config;
mod blueprint_tui_metrics;
mod blueprint_tui_resolve;
mod blueprint_tui_scenarios;
mod blueprint_tui_widgets;
mod boundary_trace;
mod compare_solve;
mod inspect_spot;
#[allow(dead_code)]
mod log_file;
mod mp_tui;
mod mp_tui_scenarios;
mod mp_tui_widgets;
#[allow(dead_code)]
mod validate_blueprint;
mod validate_rollout;
mod validation_spots;

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use poker_solver_core::blueprint_mp::config::{BlueprintMpConfig, MpTrainingBackend};
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
    /// Inspect a multiplayer blueprint config before training
    InspectMpConfig {
        /// YAML config file (BlueprintMpConfig)
        #[arg(short, long)]
        config: PathBuf,
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
        /// Audit bucket assignments by made hand class, intra-class strength, and equity decile
        #[arg(long)]
        hand_class_audit: bool,
        /// Number of boards to sample for hand-class audit (default 10)
        #[arg(long, default_value = "10")]
        hand_class_audit_boards: usize,
        /// Number of rows to show per hand-class audit section (default 10)
        #[arg(long, default_value = "10")]
        hand_class_audit_top: usize,
        /// Write a machine-readable bucket audit scorecard to this JSON file
        #[arg(long)]
        scorecard_json: Option<PathBuf>,
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

        /// Diagnostic override for exact solve iterations.
        #[arg(long, hide = true)]
        exact_iters: Option<u32>,

        /// Diagnostic override for subgame solve iterations.
        #[arg(long, hide = true)]
        subgame_iters: Option<u32>,

        /// Print per-iteration progress
        #[arg(long)]
        verbose: bool,

        /// Dump per-boundary CFV statistics after precompute (before DCFR)
        #[arg(long)]
        dump_boundary_cfvs: bool,

        /// Flop boundary mode: "exact", "cfvnet", "exact_subtree", or "exact_oracle"
        #[arg(long, default_value = "exact")]
        flop_boundary: String,

        /// ONNX model path for flop boundary (required when --flop-boundary=cfvnet)
        #[arg(long)]
        flop_model: Option<String>,

        /// Flop cfvnet inference mode: "direct", "river_enumerated_turn", or "direct_normalized_legacy"
        #[arg(long, default_value = "direct")]
        flop_model_kind: String,

        /// Turn boundary mode: "exact", "cfvnet", "exact_subtree", or "exact_oracle"
        #[arg(long, default_value = "exact")]
        turn_boundary: String,

        /// ONNX model path for turn boundary (required when --turn-boundary=cfvnet)
        #[arg(long)]
        turn_model: Option<String>,

        /// Turn cfvnet inference mode: "direct", "river_enumerated_turn", or "direct_normalized_legacy"
        #[arg(long, default_value = "direct")]
        turn_model_kind: String,

        /// River boundary mode: "exact", "cfvnet", "exact_subtree", or "exact_oracle"
        #[arg(long, default_value = "exact")]
        river_boundary: String,

        /// ONNX model path for river boundary (required when --river-boundary=cfvnet)
        #[arg(long)]
        river_model: Option<String>,

        /// River cfvnet inference mode: "direct", "river_enumerated_turn", or "direct_normalized_legacy"
        #[arg(long, default_value = "direct")]
        river_model_kind: String,

        /// Diagnostic exact_oracle CFV orientation transform.
        #[arg(long, default_value = "current", hide = true)]
        oracle_orientation: String,

        /// Diagnostic exact_oracle raw CFV multiplier.
        #[arg(long, default_value_t = 1.0, hide = true)]
        oracle_scale: f32,

        /// Diagnostic exact_oracle mode: solve exact and subgame in lockstep.
        #[arg(long, default_value_t = false, hide = true)]
        oracle_iteration_aligned: bool,

        /// Diagnostic: emit root action CFV/regret update traces for iterations.
        #[arg(long, hide = true)]
        root_update_trace_iters: Option<String>,

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

        /// Enable per-boundary CFR-D gadget (Option A2).
        /// Injects gadget subtrees at each depth-boundary terminal for safe re-solve.
        /// Mutually exclusive with --gadget-clamp.
        #[arg(long, default_value_t = false, conflicts_with = "gadget_clamp")]
        gadget: bool,

        /// Enable legacy post-clamp GadgetEvaluator at subgame boundaries.
        /// Clamps opponent CFVs upward to blueprint opt-out values.
        /// Mutually exclusive with --gadget.
        #[arg(long, default_value_t = false, conflicts_with = "gadget")]
        gadget_clamp: bool,

        /// Opt-out provider when --gadget or --gadget-clamp is set.
        /// "blueprint-cbv" reads from the bundle's CbvTable (production).
        /// "constant" uses a fixed value from --gadget-constant (diagnostic).
        #[arg(long, default_value = "blueprint-cbv")]
        gadget_provider: String,

        /// Constant opt-out value (pot-normalised bcfv) when
        /// --gadget-provider=constant. Ignored otherwise.
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        gadget_constant: f32,
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
            eprintln!(
                "  Actions: preflop_depths={} flop_depths={} turn_depths={} river_depths={}",
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
                trainer.config_reload_trigger = Arc::clone(&metrics.config_reload_trigger);

                // Wire strategy refresh callback from trainer to TUI metrics.
                let scenarios_node_indices: Vec<u32> =
                    scenarios.iter().map(|s| s.node_idx).collect();
                trainer
                    .scenario_ev_tracker
                    .set_nodes(scenarios_node_indices.clone());
                trainer.scenario_node_indices = scenarios_node_indices;
                trainer.strategy_refresh_interval_secs =
                    tui_config.telemetry.strategy_delta_interval_seconds;

                let boards_for_refresh = Arc::clone(&shared_boards);
                let metrics_for_refresh = Arc::clone(&metrics);
                trainer.on_strategy_refresh = Some(Box::new(
                    move |scenario_idx, node_idx, storage, tree, hand_evs| {
                        let boards = boards_for_refresh.read().unwrap();
                        if scenario_idx < boards.len() {
                            let grid = blueprint_tui_scenarios::extract_strategy_grid(
                                tree,
                                storage,
                                node_idx,
                                &boards[scenario_idx],
                                Some(hand_evs),
                            );
                            metrics_for_refresh.update_scenario_grid(scenario_idx, grid);
                        }
                    },
                ));

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
                        let snapshots: Vec<_> = audits.iter().map(|a| a.snapshot()).collect();
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
                            tree,
                            storage,
                            &new_tui_config.scenarios,
                        );

                        // Re-resolve audits.
                        let audits = blueprint_tui_resolve::resolve_audits(
                            tree,
                            storage,
                            &new_tui_config.regret_audits,
                            sparkline_window,
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
                    trainer.random_scenario_hold_minutes = tui_config.random_scenario.hold_minutes;
                    let metrics_for_random = Arc::clone(&metrics);
                    let pool = tui_config.random_scenario.pool.clone();
                    trainer.on_random_scenario = Some(Box::new(move |storage, tree, hand_evs| {
                        use poker_solver_core::blueprint_v2::game_tree::GameNode;
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

                        // Select the correct position's EVs based on the node's player.
                        let player = match &tree.nodes[node_idx as usize] {
                            GameNode::Decision { player, .. } => *player as usize,
                            _ => 0,
                        };
                        let node_hand_evs = &hand_evs[player];

                        let board = blueprint_tui_scenarios::random_board(street, &mut rng);
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
                            tree,
                            storage,
                            node_idx,
                            &board,
                            Some(node_hand_evs),
                        );

                        let name = blueprint_tui_scenarios::random_scenario_name(
                            tree,
                            node_idx,
                            &board_display,
                        );

                        let street_label_str = format!("{street:?}");

                        metrics_for_random.update_random_scenario(
                            name,
                            node_idx,
                            grid,
                            board_display,
                            street_label_str,
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
        Commands::InspectMpConfig { config } => {
            run_inspect_mp_config(config.to_str().expect("invalid config path"))?;
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
                    bp_config.clustering.flop.buckets, pf_cfg.turn_buckets, pf_cfg.river_buckets,
                );
                eprintln!();

                let per_flop_config =
                    poker_solver_core::blueprint_v2::cluster_pipeline::PerFlopClusteringConfig {
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
                let bar_style = ProgressStyle::with_template(
                    "  {msg:>30} {bar:30.white/black} {pos}/{len} ETA {eta}",
                )
                .unwrap()
                .progress_chars("##-");
                let spinner_style =
                    ProgressStyle::with_template("  {msg:>30} {spinner:.cyan} {elapsed_precise}")
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
                                            if let (Ok(pos), Ok(total)) =
                                                (pos_s.parse::<u64>(), total_s.parse::<u64>())
                                            {
                                                let new_msg = format!("{stage} {phase}");
                                                if total <= 1 {
                                                    // No meaningful progress — show spinner
                                                    bar.set_style(spinner_style_clone.clone());
                                                    bar.set_message(new_msg);
                                                    bar.enable_steady_tick(
                                                        std::time::Duration::from_millis(100),
                                                    );
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
                eprintln!(
                    "Per-flop clustering complete. Files saved to {}",
                    output.display()
                );
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
                    eprintln!(
                        "Bucket files already exist in {}, skipping clustering",
                        output.display()
                    );
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
                        ProgressStyle::with_template(
                            "  {msg:>12} {bar:40.white/black} {pos}/{len}",
                        )
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
            hand_class_audit,
            hand_class_audit_boards,
            hand_class_audit_top,
            scorecard_json,
        } => {
            use poker_solver_core::blueprint_v2::bucket_file::BucketFile;
            use poker_solver_core::blueprint_v2::cluster_diagnostics::{
                audit_hand_class_bucket_assignments, cross_street_transition_matrix,
                sample_hands_for_bucket,
            };

            let reports =
                poker_solver_core::blueprint_v2::cluster_diagnostics::diagnose_cluster_dir(
                    &cluster_dir,
                )?;
            let mut hand_class_reports = Vec::new();
            if reports.is_empty() {
                eprintln!("No .buckets files found in {}", cluster_dir.display());
            } else {
                for report in &reports {
                    eprintln!("{}", report.summary());
                }
            }
            if audit {
                eprintln!("\nEquity audit ({audit_boards} sample boards per street)...");
                let audit_reports =
                    poker_solver_core::blueprint_v2::cluster_diagnostics::audit_cluster_dir(
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
                    let report =
                        poker_solver_core::blueprint_v2::cluster_diagnostics::audit_cfvnet_buckets(
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
                use poker_solver_core::blueprint_v2::centroid_file::CentroidFile;
                use poker_solver_core::blueprint_v2::cluster_diagnostics::audit_transition_consistency;
                let pairs = [("flop", "turn"), ("turn", "river")];
                for (from_name, to_name) in &pairs {
                    let from_path = cluster_dir.join(format!("{from_name}.buckets"));
                    let to_path = cluster_dir.join(format!("{to_name}.buckets"));
                    if from_path.exists() && to_path.exists() {
                        eprintln!(
                            "\nTransition consistency audit: {from_name} → {to_name} ({transition_audit_boards} sample boards)..."
                        );
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
                            &from_bf,
                            &to_bf,
                            transition_audit_boards,
                            42,
                            centroids.as_ref(),
                        );
                        eprintln!("{}", report.summary());
                    }
                }
            }
            if hand_class_audit {
                for street_name in &["flop", "turn", "river"] {
                    let path = cluster_dir.join(format!("{street_name}.buckets"));
                    if path.exists() {
                        eprintln!(
                            "\nHand-class bucket audit: {street_name} ({hand_class_audit_boards} sample boards)..."
                        );
                        let bf = BucketFile::load(&path)?;
                        let report = audit_hand_class_bucket_assignments(
                            &bf,
                            hand_class_audit_boards,
                            42,
                            hand_class_audit_top,
                        );
                        eprintln!("{}", report.summary());
                        hand_class_reports.push(report);
                    }
                }
            }
            if let Some(path) = scorecard_json {
                let scorecard =
                    build_cluster_scorecard_json(&cluster_dir, &reports, &hand_class_reports)?;
                let json = serde_json::to_string_pretty(&scorecard)?;
                std::fs::write(&path, json)?;
                eprintln!("\nWrote bucket audit scorecard to {}", path.display());
            }
            // Per-flop bucket file diagnostics
            let per_flop_marker = cluster_dir.join("flop_0000.buckets");
            if per_flop_marker.exists() {
                eprintln!("\nPer-flop bucket files detected.");
                let pf_report =
                    poker_solver_core::blueprint_v2::cluster_diagnostics::diagnose_per_flop_dir(
                        &cluster_dir,
                        10,
                    )?;
                eprintln!(
                    "  Files: {}, sampled: {}",
                    pf_report.total_flop_files, pf_report.sampled_files
                );
                eprintln!(
                    "  Turn buckets: {}, River buckets: {}",
                    pf_report.turn_bucket_count, pf_report.river_bucket_count
                );
                eprintln!(
                    "  Avg turn cards: {:.1}, Avg rivers/turn: {:.1}",
                    pf_report.avg_turn_cards, pf_report.avg_river_cards_per_turn
                );

                if audit {
                    let pf_flop_samples = audit_boards.min(20);
                    eprintln!(
                        "\nPer-flop equity audit ({} flop samples, 5 rivers/turn)...",
                        pf_flop_samples
                    );
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
                let bucket_id: u16 = args[1]
                    .parse()
                    .map_err(|e| format!("invalid bucket id '{}': {e}", args[1]))?;
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
                eprintln!(
                    "Cache already exists at {}, nothing to do",
                    output.display()
                );
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
            eprintln!(
                "  Spots file: {} ({} spots)",
                spots.display(),
                spots_file.spots.len()
            );
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
                                .sum::<f32>()
                                / result.num_hands as f32;
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
                println!(
                    "{:<40} {:>6} {:>6} {:>12}",
                    "Spot", "Hands", "Acts", "Exploit"
                );
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
            inspect_spot::run(&config, &spot).map_err(|e| -> Box<dyn Error> { e.into() })?;
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
                &bundle,
                duration_secs,
                &board,
                pot,
                stacks,
                enumerate_depth,
                opponent_samples,
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
                &bundle,
                &board,
                pot,
                stacks,
                num_runs,
                pass_threshold,
                enumerate_depth,
                opponent_samples,
            )
            .map_err(|e| -> Box<dyn Error> { e.into() })?;
        }
        Commands::CompareSolve {
            bundle,
            snapshot,
            spot,
            iters,
            exact_iters,
            subgame_iters,
            verbose,
            dump_boundary_cfvs,
            flop_boundary,
            flop_model,
            flop_model_kind,
            turn_boundary,
            turn_model,
            turn_model_kind,
            river_boundary,
            river_model,
            river_model_kind,
            oracle_orientation,
            oracle_scale,
            oracle_iteration_aligned,
            root_update_trace_iters,
            trace_boundaries,
            trace_iters,
            trace_dir,
            tolerance,
            gadget,
            gadget_clamp,
            gadget_provider,
            gadget_constant,
        } => {
            let parse_mode =
                |mode: &str,
                 model: Option<String>,
                 model_kind: &str,
                 street: &str|
                 -> Result<poker_solver_tauri::StreetBoundaryMode, Box<dyn Error>> {
                    match mode {
                        "exact" => Ok(poker_solver_tauri::StreetBoundaryMode::Exact),
                        "cfvnet" => {
                            let path = model.ok_or_else(|| {
                                format!(
                                    "--{street}-model is required when --{street}-boundary=cfvnet"
                                )
                            })?;
                            let inference_mode = match model_kind {
                                "river_enumerated_turn" | "river-enumerated-turn" => {
                                    cfvnet::eval::boundary_evaluator::BoundaryInferenceMode::RiverEnumeratedTurn
                                }
                                "direct" => {
                                    cfvnet::eval::boundary_evaluator::BoundaryInferenceMode::Direct
                                }
                                "direct_normalized_legacy" | "direct-normalized-legacy" => {
                                    cfvnet::eval::boundary_evaluator::BoundaryInferenceMode::DirectNormalizedLegacy
                                }
                                other => {
                                    return Err(format!(
                                        "invalid --{street}-model-kind value '{other}': expected \
                                         'river_enumerated_turn', 'direct', or \
                                         'direct_normalized_legacy'"
                                    )
                                    .into());
                                }
                            };
                            Ok(poker_solver_tauri::StreetBoundaryMode::Cfvnet {
                                model_path: path,
                                inference_mode,
                            })
                        }
                        "exact_subtree" => Ok(poker_solver_tauri::StreetBoundaryMode::ExactSubtree),
                        "exact_oracle" => Ok(poker_solver_tauri::StreetBoundaryMode::ExactSubtree),
                        other => Err(format!(
                            "invalid --{street}-boundary value '{other}': \
                         expected 'exact', 'cfvnet', 'exact_subtree', or 'exact_oracle'"
                        )
                        .into()),
                    }
                };
            let oracle_boundary_flags = [
                flop_boundary == "exact_oracle",
                turn_boundary == "exact_oracle",
                river_boundary == "exact_oracle",
            ];
            let oracle_orientation =
                compare_solve::OracleCfvOrientation::parse(&oracle_orientation)
                    .map_err(|e| -> Box<dyn Error> { e.into() })?;
            let sbc = poker_solver_tauri::StreetBoundaryConfig {
                flop: parse_mode(&flop_boundary, flop_model, &flop_model_kind, "flop")?,
                turn: parse_mode(&turn_boundary, turn_model, &turn_model_kind, "turn")?,
                river: parse_mode(&river_boundary, river_model, &river_model_kind, "river")?,
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
                exact_iters,
                subgame_iters,
                verbose,
                dump_boundary_cfvs,
                sbc,
                oracle_boundary_flags,
                oracle_orientation,
                oracle_scale,
                oracle_iteration_aligned,
                root_update_trace_iters.as_deref(),
                trace_config,
                tolerance,
                gadget,
                gadget_clamp,
                &gadget_provider,
                gadget_constant,
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
    use poker_solver_core::blueprint_v2::bundle::{BlueprintV2Strategy, load_config};
    use poker_solver_core::blueprint_v2::game_tree::GameTree;
    use poker_solver_core::blueprint_v2::mccfr::AllBuckets;

    let yaml =
        std::fs::read_to_string(config_path).map_err(|e| format!("Failed to read config: {e}"))?;
    let rebel_config: rebel::config::RebelConfig =
        serde_yaml::from_str(&yaml).map_err(|e| format!("Failed to parse config: {e}"))?;

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
    let bucket_names = [
        "preflop.buckets",
        "flop.buckets",
        "turn.buckets",
        "river.buckets",
    ];
    let mut bucket_files: [Option<BucketFile>; 4] = [None, None, None, None];
    for (i, name) in bucket_names.iter().enumerate() {
        let path = cluster_dir.join(name);
        if path.exists() {
            match BucketFile::load(&path) {
                Ok(bf) => {
                    eprintln!(
                        "  Loaded {}: {} boards, {} combos/board, {} buckets",
                        name,
                        bf.header.board_count,
                        bf.header.combos_per_board,
                        bf.header.bucket_count,
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
            eprintln!(
                "  Detected per-flop bucket files in {}",
                cluster_dir.display()
            );
            buckets.with_per_flop_dir(cluster_dir.to_path_buf())
        } else {
            buckets
        }
    };

    // Build game tree from blueprint config
    let bp_config_path = std::path::Path::new(&rebel_config.blueprint_path).join("config.yaml");
    eprintln!(
        "Loading blueprint config from {}...",
        bp_config_path.display()
    );
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
    let buffer_path =
        std::path::Path::new(&rebel_config.output_dir).join(&rebel_config.buffer.path);
    let buffer = if buffer_path.exists() {
        let buf =
            rebel::data_buffer::DiskBuffer::open(&buffer_path, rebel_config.buffer.max_records)
                .map_err(|e| format!("Failed to open buffer: {e}"))?;
        eprintln!(
            "Resuming from existing buffer: {} records at {}",
            buf.len(),
            buffer_path.display()
        );
        std::sync::Mutex::new(buf)
    } else {
        eprintln!("Creating buffer at {}...", buffer_path.display());
        std::sync::Mutex::new(
            rebel::data_buffer::DiskBuffer::create(&buffer_path, rebel_config.buffer.max_records)
                .map_err(|e| format!("Failed to create buffer: {e}"))?,
        )
    };

    // Skip PBS generation if buffer already has records
    let existing_count = buffer.lock().unwrap().len();
    if existing_count > 0 {
        eprintln!(
            "Buffer has {} existing records, skipping PBS generation",
            existing_count
        );
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
    eprintln!("Solved {}/{} records successfully", solved, record_count);

    // --- Step 4: Export buffer → cfvnet training files ---
    let export_path = std::path::Path::new(&rebel_config.output_dir).join("training_data.bin");
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
    use poker_solver_core::blueprint_v2::bundle::{BlueprintV2Strategy, load_config};
    use poker_solver_core::blueprint_v2::game_tree::GameTree;
    use poker_solver_core::blueprint_v2::mccfr::AllBuckets;

    let yaml =
        std::fs::read_to_string(config_path).map_err(|e| format!("Failed to read config: {e}"))?;
    let rebel_config: rebel::config::RebelConfig =
        serde_yaml::from_str(&yaml).map_err(|e| format!("Failed to parse config: {e}"))?;

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
    let bucket_names = [
        "preflop.buckets",
        "flop.buckets",
        "turn.buckets",
        "river.buckets",
    ];
    let mut bucket_files: [Option<BucketFile>; 4] = [None, None, None, None];
    for (i, name) in bucket_names.iter().enumerate() {
        let path = cluster_dir.join(name);
        if path.exists() {
            match BucketFile::load(&path) {
                Ok(bf) => {
                    eprintln!(
                        "  Loaded {}: {} boards, {} combos/board, {} buckets",
                        name,
                        bf.header.board_count,
                        bf.header.combos_per_board,
                        bf.header.bucket_count,
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
            eprintln!(
                "  Detected per-flop bucket files in {}",
                cluster_dir.display()
            );
            buckets.with_per_flop_dir(cluster_dir.to_path_buf())
        } else {
            buckets
        }
    };

    // Build game tree from blueprint config
    let bp_config_path = std::path::Path::new(&rebel_config.blueprint_path).join("config.yaml");
    eprintln!(
        "Loading blueprint config from {}...",
        bp_config_path.display()
    );
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
        let result =
            rebel::orchestration::run_offline_seeding(&rebel_config, &strategy, &tree, &buckets)?;
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
        use rebel::inference_server::{InferenceServerConfig, spawn_inference_server};
        use rebel::replay_buffer::ReplayBuffer;
        use rebel::self_play::{SelfPlayConfig, self_play_training_loop};

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
        let recorder =
            burn::record::NamedMpkGzFileRecorder::<burn::record::FullPrecisionSettings>::new();
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
            eprintln!(
                "  No checkpoint found at {}, using random init",
                model_file.display()
            );
        }
        eprintln!(
            "  CfvNet: {} layers x {} hidden",
            rebel_config.training.hidden_layers, rebel_config.training.hidden_size,
        );

        // Create replay buffer.
        let replay_buffer = Arc::new(ReplayBuffer::new(rebel_config.inference.replay_capacity));

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
        eprintln!(
            "  Inference server spawned (batch_size={}, timeout={}us)",
            rebel_config.inference.batch_size, rebel_config.inference.batch_timeout_us
        );

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
        server_thread
            .join()
            .expect("inference server thread panicked");
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
    let yaml =
        std::fs::read_to_string(config_path).map_err(|e| format!("Failed to read config: {e}"))?;
    let rebel_config: rebel::config::RebelConfig =
        serde_yaml::from_str(&yaml).map_err(|e| format!("Failed to parse config: {e}"))?;

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
                100, // 100 validation examples
                &solve_config,
                rebel_config.seed.seed + 999999, // different seed from training
            );
            eprintln!("Generated {} validation records", val_records.len());

            eprintln!(
                "  River records: {}",
                val_records
                    .iter()
                    .filter(|r| r.board_card_count == 5)
                    .count()
            );

            // Model MSE evaluation requires loading CfvNet — not yet wired.
            eprintln!();
            eprintln!(
                "Model MSE evaluation requires loading CfvNet ({} layers x {} units) — not yet wired.",
                rebel_config.training.hidden_layers, rebel_config.training.hidden_size
            );
            eprintln!(
                "Validation set generated with {} records.",
                val_records.len()
            );
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
    let yaml =
        std::fs::read_to_string(config_path).map_err(|e| format!("Failed to read config: {e}"))?;
    let rebel_config: rebel::config::RebelConfig =
        serde_yaml::from_str(&yaml).map_err(|e| format!("Failed to parse config: {e}"))?;

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
    let val_records =
        rebel::validation::generate_validation_set(num_examples, &solve_config, val_seed);
    let elapsed = start.elapsed();

    let river_count = val_records
        .iter()
        .filter(|r| r.board_card_count == 5)
        .count();
    eprintln!(
        "Generated {} records ({} river) in {:.1}s",
        val_records.len(),
        river_count,
        elapsed.as_secs_f64()
    );

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
        buf.append(rec)
            .map_err(|e| format!("Failed to write record: {e}"))?;
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
    use range_solver::card::{CardConfig, NOT_DEALT, card_from_str, flop_from_str, hole_to_string};
    use range_solver::range::Range;
    use range_solver::{PostFlopGame, solve};

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
    eprintln!(
        "  Board: {flop_str}{}",
        format_board_suffix(turn_str, river_str)
    );
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
    use range_solver::card::{CardConfig, NOT_DEALT, card_from_str, flop_from_str, hole_to_string};
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
    eprintln!(
        "  Board: {flop_str}{}",
        format_board_suffix(turn_str, river_str)
    );
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

fn run_inspect_mp_config(path: &str) -> Result<(), Box<dyn Error>> {
    let yaml = std::fs::read_to_string(path)?;
    let config: BlueprintMpConfig = serde_yaml::from_str(&yaml)?;
    config
        .game
        .validate()
        .map_err(|e| format!("invalid config: {e}"))?;
    let report = inspect_mp_config(&config)?;
    print_mp_config_report(&report);
    Ok(())
}

fn run_train_blueprint_mp(path: &str, no_tui: bool) -> Result<(), Box<dyn Error>> {
    let yaml = std::fs::read_to_string(path)?;
    let config: BlueprintMpConfig = serde_yaml::from_str(&yaml)?;
    config
        .game
        .validate()
        .map_err(|e| format!("invalid config: {e}"))?;
    let report = inspect_mp_config(&config)?;
    if report.backend == MpTrainingBackend::Eager {
        if let Some(risk) = &report.eager_risk {
            return Err(format!(
                "MP config is too large for the current eager backend: {risk}. Run inspect-mp-config for details."
            )
            .into());
        }
    }
    let tui_config = blueprint_tui_config::parse_tui_config(&yaml);
    eprintln!(
        "Starting N-player blueprint training: {} ({} players, {}bb deep)",
        config.game.name,
        config.game.num_players,
        config.game.stack_depth / 2.0
    );
    if config.training.backend == MpTrainingBackend::LazySparse {
        if tui_config.enabled && !no_tui {
            run_mp_with_tui_lazy(&config, &tui_config)?;
        } else {
            run_mp_without_tui_lazy(&config)?;
        }
    } else if tui_config.enabled && !no_tui {
        run_mp_with_tui(&config, &tui_config)?;
    } else {
        run_mp_without_tui(&config)?;
    }
    Ok(())
}

#[derive(Debug)]
struct MpConfigReport {
    name: String,
    num_players: u8,
    stack_chips: f64,
    big_blind: f64,
    stack_bb: f64,
    bucket_counts: [u16; 4],
    preflop_lead_sizes: usize,
    preflop_raise_rows: usize,
    postflop_raise_rows: [usize; 3],
    backend: MpTrainingBackend,
    eager_risk: Option<String>,
}

fn inspect_mp_config(config: &BlueprintMpConfig) -> Result<MpConfigReport, Box<dyn Error>> {
    let big_blind = mp_big_blind_amount(config)
        .ok_or_else(|| "config must include a positive big_blind forced bet".to_string())?;
    let stack_bb = config.game.stack_depth / big_blind;
    let preflop_raise_rows = config.action_abstraction.preflop.raise.len();
    let postflop_raise_rows = [
        config.action_abstraction.flop.raise.len(),
        config.action_abstraction.turn.raise.len(),
        config.action_abstraction.river.raise.len(),
    ];
    let eager_risk = mp_eager_risk(
        config.game.num_players,
        stack_bb,
        preflop_raise_rows,
        postflop_raise_rows,
    );
    Ok(MpConfigReport {
        name: config.game.name.clone(),
        num_players: config.game.num_players,
        stack_chips: config.game.stack_depth,
        big_blind,
        stack_bb,
        bucket_counts: config.clustering.bucket_counts(),
        preflop_lead_sizes: config.action_abstraction.preflop.lead.len(),
        preflop_raise_rows,
        postflop_raise_rows,
        backend: config.training.backend,
        eager_risk,
    })
}

fn mp_big_blind_amount(config: &BlueprintMpConfig) -> Option<f64> {
    use poker_solver_core::blueprint_mp::config::ForcedBetKind;

    config
        .game
        .blinds
        .iter()
        .find(|blind| blind.kind == ForcedBetKind::BigBlind && blind.amount > 0.0)
        .map(|blind| blind.amount)
}

fn mp_eager_risk(
    num_players: u8,
    stack_bb: f64,
    preflop_raise_rows: usize,
    postflop_raise_rows: [usize; 3],
) -> Option<String> {
    let total_postflop_rows: usize = postflop_raise_rows.iter().sum();
    if num_players >= 6 && stack_bb >= 80.0 && preflop_raise_rows >= 2 {
        return Some(
            "unsafe for current eager backend: 100bb-scale 6-max with multiple preflop raise rows can exceed hundreds of millions of nodes and hundreds of GB of dense storage; use the planned lazy_sparse backend once implemented".to_string(),
        );
    }
    if num_players >= 6 && stack_bb >= 80.0 && total_postflop_rows >= 4 {
        return Some(
            "high risk for current eager backend: 100bb-scale 6-max with broad postflop raise rows may materialize an impractically large dense tree".to_string(),
        );
    }
    None
}

fn print_mp_config_report(report: &MpConfigReport) {
    eprintln!("Blueprint MP config preflight");
    eprintln!("  Name: {}", report.name);
    eprintln!("  Players: {}", report.num_players);
    eprintln!(
        "  Stack: {:.1} chips ({:.1}bb, BB={:.1} chips)",
        report.stack_chips, report.stack_bb, report.big_blind
    );
    eprintln!(
        "  Buckets: preflop={}, flop={}, turn={}, river={}",
        report.bucket_counts[0],
        report.bucket_counts[1],
        report.bucket_counts[2],
        report.bucket_counts[3]
    );
    eprintln!(
        "  Actions: preflop_leads={}, preflop_raise_rows={}, flop_raise_rows={}, turn_raise_rows={}, river_raise_rows={}",
        report.preflop_lead_sizes,
        report.preflop_raise_rows,
        report.postflop_raise_rows[0],
        report.postflop_raise_rows[1],
        report.postflop_raise_rows[2]
    );
    eprintln!("  Selected backend: {}", mp_backend_label(report.backend));
    if let Some(risk) = &report.eager_risk {
        eprintln!("  Eager backend: {risk}");
        if report.backend == MpTrainingBackend::LazySparse {
            eprintln!("  Lazy sparse backend: selected; eager dense risk will not block training");
        }
    } else {
        eprintln!("  Eager backend: no known 100bb-scale risk pattern detected");
    }
}

fn mp_backend_label(backend: MpTrainingBackend) -> &'static str {
    match backend {
        MpTrainingBackend::Eager => "eager",
        MpTrainingBackend::LazySparse => "lazy_sparse",
    }
}

fn run_mp_without_tui(config: &BlueprintMpConfig) -> Result<(), Box<dyn Error>> {
    use poker_solver_core::blueprint_mp::trainer::{run_training, setup_training};

    let ctx = setup_training(config);
    let shared_iters = Arc::clone(&ctx.iterations);
    let storage = Arc::clone(&ctx.storage);
    let train_config = config.clone();
    let train_handle =
        std::thread::spawn(move || run_training(&ctx, &train_config.training, &train_config.game));

    let mut heartbeat = MpNoTuiHeartbeat::new();
    eprintln!("  no-TUI progress: heartbeat every 60s");
    while !train_handle.is_finished() {
        std::thread::sleep(Duration::from_secs(1));
        if heartbeat.should_print() {
            heartbeat.print(&shared_iters, &storage);
        }
    }
    heartbeat.print(&shared_iters, &storage);
    let result = train_handle.join().expect("training thread panicked");
    eprintln!(
        "Training complete: {} meta-iterations",
        result.meta_iterations
    );
    Ok(())
}

fn run_mp_without_tui_lazy(config: &BlueprintMpConfig) -> Result<(), Box<dyn Error>> {
    use poker_solver_core::blueprint_mp::trainer::{run_lazy_training, setup_lazy_training};

    let ctx = setup_lazy_training(config);
    let shared_iters = Arc::clone(&ctx.iterations);
    let storage = Arc::clone(&ctx.storage);
    let train_config = config.clone();
    let train_handle = std::thread::spawn(move || {
        run_lazy_training(&ctx, &train_config.training, &train_config.game)
    });

    let mut heartbeat = MpNoTuiHeartbeat::new();
    eprintln!("  no-TUI lazy_sparse progress: heartbeat every 60s");
    while !train_handle.is_finished() {
        std::thread::sleep(Duration::from_secs(1));
        if heartbeat.should_print() {
            heartbeat.print_sparse(&shared_iters, &storage);
        }
    }
    heartbeat.print_sparse(&shared_iters, &storage);
    let result = train_handle.join().expect("lazy training thread panicked");
    eprintln!(
        "Lazy sparse training complete: {} meta-iterations",
        result.meta_iterations
    );
    Ok(())
}

struct MpNoTuiHeartbeat {
    started: Instant,
    last_print: Instant,
    last_iters: u64,
    last_sparse_entries: usize,
    last_sparse_bytes: usize,
    last_sparse_activity: SparseStorageActivity,
    last_sparse_insert_attribution: SparseInsertAttribution,
}

type SparseStorageActivity = poker_solver_core::blueprint_mp::sparse_storage::SparseStorageActivity;
type SparseInsertAttribution =
    poker_solver_core::blueprint_mp::sparse_storage::SparseInsertAttribution;
type SparseTelemetrySample = poker_solver_core::blueprint_mp::sparse_storage::SparseTelemetrySample;
type LazyActionLimitSnapshot = poker_solver_core::blueprint_mp::lazy_mccfr::LazyActionLimitSnapshot;
const STREET_LABELS: [&str; 4] = ["pf", "f", "t", "r"];
const LAZY_TUI_TELEMETRY_SAMPLE_ENTRIES: usize = 8192;
const HISTORY_LEN_BIN_LABELS: [&str; 8] = ["0", "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64+"];
const ACTION_COUNT_BIN_LABELS: [&str; 8] =
    ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65+"];

impl MpNoTuiHeartbeat {
    const INTERVAL: Duration = Duration::from_secs(60);

    fn new() -> Self {
        let now = Instant::now();
        Self {
            started: now,
            last_print: now,
            last_iters: 0,
            last_sparse_entries: 0,
            last_sparse_bytes: 0,
            last_sparse_activity: Default::default(),
            last_sparse_insert_attribution: Default::default(),
        }
    }

    fn should_print(&self) -> bool {
        self.last_print.elapsed() >= Self::INTERVAL
    }

    fn print(
        &mut self,
        iterations: &AtomicU64,
        storage: &poker_solver_core::blueprint_mp::storage::MpStorage,
    ) {
        let now = Instant::now();
        let iters = iterations.load(Ordering::Relaxed);
        let since_last = now.duration_since(self.last_print).as_secs_f64();
        let elapsed = now.duration_since(self.started);
        let interval_iters = iters.saturating_sub(self.last_iters);
        let interval_rate = if since_last > 0.0 {
            interval_iters as f64 / since_last
        } else {
            0.0
        };
        let elapsed_secs = elapsed.as_secs_f64();
        let avg_rate = if elapsed_secs > 0.0 {
            iters as f64 / elapsed_secs
        } else {
            0.0
        };
        let regret = sample_mp_regret_summary(
            &storage.regrets,
            poker_solver_core::blueprint_mp::storage::REGRET_SCALE,
            1_000_000,
        );
        let prune_pct = take_mp_prune_pct();
        let regret_text = regret.map_or_else(
            || "regret[n/a]".to_string(),
            |r| {
                format!(
                    "regret[max+={:.1}, max-={:.1}, avg+={:.3e}, pos={}/{} sampled]",
                    r.max_positive, r.max_negative, r.avg_positive, r.positive_count, r.samples
                )
            },
        );
        eprintln!(
            "  iter={} ips={:.0} avg_ips={:.0} elapsed={} {} prune={:.1}%",
            iters,
            interval_rate,
            avg_rate,
            format_duration_compact(elapsed),
            regret_text,
            prune_pct,
        );
        self.last_print = now;
        self.last_iters = iters;
    }

    fn print_sparse(
        &mut self,
        iterations: &AtomicU64,
        storage: &poker_solver_core::blueprint_mp::sparse_storage::SparseMpStorage,
    ) {
        use poker_solver_core::blueprint_mp::lazy_mccfr::take_lazy_action_limit_snapshot;
        use poker_solver_core::blueprint_mp::trainer::take_lazy_mp_timing_snapshot;

        let now = Instant::now();
        let iters = iterations.load(Ordering::Relaxed);
        let since_last = now.duration_since(self.last_print).as_secs_f64();
        let elapsed = now.duration_since(self.started);
        let interval_iters = iters.saturating_sub(self.last_iters);
        let interval_rate = if since_last > 0.0 {
            interval_iters as f64 / since_last
        } else {
            0.0
        };
        let elapsed_secs = elapsed.as_secs_f64();
        let avg_rate = if elapsed_secs > 0.0 {
            iters as f64 / elapsed_secs
        } else {
            0.0
        };
        let loop_timing = take_lazy_mp_timing_snapshot();
        let action_limit = take_lazy_action_limit_snapshot();
        let stats_started = Instant::now();
        let stats = storage.stats();
        let activity = storage.activity();
        let activity_delta = sparse_activity_delta(activity, self.last_sparse_activity);
        let insert_attribution = storage.insert_attribution();
        let insert_attribution_delta = sparse_insert_attribution_delta(
            insert_attribution,
            self.last_sparse_insert_attribution,
        );
        let stats_nanos = u64::try_from(stats_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let entries_since_last = stats.entries.saturating_sub(self.last_sparse_entries);
        let bytes_since_last = stats.approx_bytes.saturating_sub(self.last_sparse_bytes);
        let entries_per_sec = if since_last > 0.0 {
            entries_since_last as f64 / since_last
        } else {
            0.0
        };
        let bytes_per_sec = if since_last > 0.0 {
            bytes_since_last as f64 / since_last
        } else {
            0.0
        };
        let avg_entries_per_shard = if stats.shard_count > 0 {
            stats.entries as f64 / stats.shard_count as f64
        } else {
            0.0
        };
        let read_probe_rate = per_second(activity_delta.read_probes, since_last);
        let write_probe_rate = per_second(activity_delta.write_probes, since_last);
        let insert_rate = per_second(activity_delta.inserts, since_last);
        let read_hit_pct = percent(activity_delta.read_hits, activity_delta.read_probes);
        let write_hit_pct = percent(activity_delta.write_hits, activity_delta.write_probes);
        let insert_by_text = format_sparse_insert_attribution(insert_attribution_delta, since_last);
        let action_limit_text = format_lazy_action_limit(action_limit, since_last);
        let prune_pct = take_mp_prune_pct();
        eprintln!(
            "  iter={} ips={:.0} avg_ips={:.0} elapsed={} sparse[entries={} (+{:.0}/s), regret_slots={}, strategy_slots={}, approx={} (+{}/s), shards={}/{}, avg_shard={:.0}, max_shard={}] activity[read={:.0}/s hit={:.1}%, write={:.0}/s hit={:.1}%, inserts={:.0}/s] {} {} timing[batch_wall={}, deal={}, buckets={}, traverse={}, discount={}, stats={}] tail[traversals={}, max_job={}@iter{}, slow_jobs={}, max_trav={}@iter{}/p{}, slow_trav={}] prune={:.1}%",
            iters,
            interval_rate,
            avg_rate,
            format_duration_compact(elapsed),
            stats.entries,
            entries_per_sec,
            stats.regret_slots,
            stats.strategy_slots,
            format_bytes_decimal(stats.approx_bytes),
            format_bytes_decimal(bytes_per_sec as usize),
            stats.nonempty_shards,
            stats.shard_count,
            avg_entries_per_shard,
            stats.max_entries_per_shard,
            read_probe_rate,
            read_hit_pct,
            write_probe_rate,
            write_hit_pct,
            insert_rate,
            insert_by_text,
            action_limit_text,
            format_nanos_millis(loop_timing.batch_wall_nanos),
            format_nanos_millis(loop_timing.deal_nanos),
            format_nanos_millis(loop_timing.bucket_nanos),
            format_nanos_millis(loop_timing.traverse_nanos),
            format_nanos_millis(loop_timing.discount_nanos),
            format_nanos_millis(stats_nanos),
            loop_timing.traversal_count,
            format_nanos_millis(loop_timing.max_job_nanos),
            loop_timing.max_job_iter,
            loop_timing.slow_jobs,
            format_nanos_millis(loop_timing.max_traverser_nanos),
            loop_timing.max_traverser_iter,
            loop_timing.max_traverser_seat,
            loop_timing.slow_traversers,
            prune_pct,
        );
        self.last_print = now;
        self.last_iters = iters;
        self.last_sparse_entries = stats.entries;
        self.last_sparse_bytes = stats.approx_bytes;
        self.last_sparse_activity = activity;
        self.last_sparse_insert_attribution = insert_attribution;
    }
}

fn sparse_activity_delta(
    current: SparseStorageActivity,
    previous: SparseStorageActivity,
) -> SparseStorageActivity {
    SparseStorageActivity {
        read_probes: current.read_probes.saturating_sub(previous.read_probes),
        read_hits: current.read_hits.saturating_sub(previous.read_hits),
        write_probes: current.write_probes.saturating_sub(previous.write_probes),
        write_hits: current.write_hits.saturating_sub(previous.write_hits),
        inserts: current.inserts.saturating_sub(previous.inserts),
    }
}

fn sparse_insert_attribution_delta(
    current: SparseInsertAttribution,
    previous: SparseInsertAttribution,
) -> SparseInsertAttribution {
    SparseInsertAttribution {
        by_street: array_saturating_sub(current.by_street, previous.by_street),
        by_seat: array_saturating_sub(current.by_seat, previous.by_seat),
        history_len_bins: array_saturating_sub(current.history_len_bins, previous.history_len_bins),
        action_count_bins: array_saturating_sub(
            current.action_count_bins,
            previous.action_count_bins,
        ),
        history_len_max: current.history_len_max,
        action_count_sum: current
            .action_count_sum
            .saturating_sub(previous.action_count_sum),
        action_count_max: current.action_count_max,
    }
}

fn array_saturating_sub<const N: usize>(current: [u64; N], previous: [u64; N]) -> [u64; N] {
    std::array::from_fn(|idx| current[idx].saturating_sub(previous[idx]))
}

fn format_sparse_insert_attribution(
    attribution: SparseInsertAttribution,
    elapsed_secs: f64,
) -> String {
    let street_text = format!(
        "pf:{:.0}/f:{:.0}/t:{:.0}/r:{:.0}",
        per_second(attribution.by_street[0], elapsed_secs),
        per_second(attribution.by_street[1], elapsed_secs),
        per_second(attribution.by_street[2], elapsed_secs),
        per_second(attribution.by_street[3], elapsed_secs),
    );
    let (seat_idx, seat_count) = top_index(&attribution.by_seat);
    let (history_idx, history_count) = top_index(&attribution.history_len_bins);
    let (action_idx, action_count) = top_index(&attribution.action_count_bins);
    let insert_count: u64 = attribution.by_street.iter().sum();
    let action_avg = if insert_count > 0 {
        attribution.action_count_sum as f64 / insert_count as f64
    } else {
        0.0
    };
    format!(
        "insert_by[st={}, seat_top=p{}:{:.0}/s, hist_top={}:{:.0}/s max_seen={}, act_avg={:.1} max_seen={} top={}:{:.0}/s]",
        street_text,
        seat_idx,
        per_second(seat_count, elapsed_secs),
        HISTORY_LEN_BIN_LABELS[history_idx],
        per_second(history_count, elapsed_secs),
        attribution.history_len_max,
        action_avg,
        attribution.action_count_max,
        ACTION_COUNT_BIN_LABELS[action_idx],
        per_second(action_count, elapsed_secs),
    )
}

fn format_lazy_action_limit(snapshot: LazyActionLimitSnapshot, elapsed_secs: f64) -> String {
    format!(
        "action_limit[max={}, over_dec={}, over_aggr={}, allin_aggr={}]",
        format_street_counts_raw(&snapshot.max_raise_count),
        format_street_counts_rate(&snapshot.over_config_decisions, elapsed_secs),
        format_street_counts_rate(&snapshot.over_config_aggressions, elapsed_secs),
        format_street_counts_rate(&snapshot.all_in_aggressions, elapsed_secs),
    )
}

fn format_street_counts_raw(values: &[u64; 4]) -> String {
    STREET_LABELS
        .iter()
        .zip(values)
        .map(|(label, value)| format!("{label}:{value}"))
        .collect::<Vec<_>>()
        .join("/")
}

fn format_street_counts_rate(values: &[u64; 4], elapsed_secs: f64) -> String {
    STREET_LABELS
        .iter()
        .zip(values)
        .map(|(label, value)| format!("{label}:{:.0}/s", per_second(*value, elapsed_secs)))
        .collect::<Vec<_>>()
        .join("/")
}

fn top_index(values: &[u64]) -> (usize, u64) {
    values
        .iter()
        .copied()
        .enumerate()
        .max_by_key(|(_, count)| *count)
        .unwrap_or((0, 0))
}

fn per_second(count: u64, elapsed_secs: f64) -> f64 {
    if elapsed_secs > 0.0 {
        count as f64 / elapsed_secs
    } else {
        0.0
    }
}

fn percent(part: u64, total: u64) -> f64 {
    if total > 0 {
        part as f64 * 100.0 / total as f64
    } else {
        0.0
    }
}

#[derive(Debug, Clone, Copy)]
struct MpRegretSummary {
    max_positive: f64,
    max_negative: f64,
    avg_positive: f64,
    positive_count: u64,
    samples: u64,
}

fn sample_mp_regret_summary(
    regrets: &[AtomicI32],
    regret_scale: f64,
    max_samples: usize,
) -> Option<MpRegretSummary> {
    if regrets.is_empty() || max_samples == 0 {
        return None;
    }
    let step = (regrets.len() / max_samples).max(1);
    let mut max_positive = 0_i32;
    let mut min_negative = 0_i32;
    let mut positive_sum = 0_i64;
    let mut positive_count = 0_u64;
    let mut samples = 0_u64;

    for atom in regrets.iter().step_by(step).take(max_samples) {
        let v = atom.load(Ordering::Relaxed);
        max_positive = max_positive.max(v);
        min_negative = min_negative.min(v);
        if v > 0 {
            positive_sum += i64::from(v);
            positive_count += 1;
        }
        samples += 1;
    }

    let avg_positive = if positive_count > 0 {
        (positive_sum as f64 / positive_count as f64) / regret_scale
    } else {
        0.0
    };
    Some(MpRegretSummary {
        max_positive: f64::from(max_positive) / regret_scale,
        max_negative: f64::from(min_negative) / regret_scale,
        avg_positive,
        positive_count,
        samples,
    })
}

fn take_mp_prune_pct() -> f64 {
    use poker_solver_core::blueprint_mp::trainer::{PRUNE_HITS, PRUNE_TOTAL};

    let hits = PRUNE_HITS.swap(0, Ordering::Relaxed);
    let total = PRUNE_TOTAL.swap(0, Ordering::Relaxed);
    if total > 0 {
        hits as f64 / total as f64 * 100.0
    } else {
        0.0
    }
}

fn format_duration_compact(duration: Duration) -> String {
    let secs = duration.as_secs();
    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    if hours > 0 {
        format!("{hours}h{minutes:02}m{seconds:02}s")
    } else if minutes > 0 {
        format!("{minutes}m{seconds:02}s")
    } else {
        format!("{seconds}s")
    }
}

fn format_nanos_millis(nanos: u64) -> String {
    format!("{:.1}ms", nanos as f64 / 1_000_000.0)
}

fn format_bytes_decimal(bytes: usize) -> String {
    const KB: f64 = 1_000.0;
    const MB: f64 = KB * 1_000.0;
    const GB: f64 = MB * 1_000.0;
    let bytes_f = bytes as f64;
    if bytes_f >= GB {
        format!("{:.2}GB", bytes_f / GB)
    } else if bytes_f >= MB {
        format!("{:.1}MB", bytes_f / MB)
    } else if bytes_f >= KB {
        format!("{:.1}KB", bytes_f / KB)
    } else {
        format!("{bytes}B")
    }
}

fn run_mp_with_tui(
    config: &BlueprintMpConfig,
    tui_config: &blueprint_tui_config::BlueprintTuiConfig,
) -> Result<(), Box<dyn Error>> {
    use poker_solver_core::blueprint_mp::trainer::{run_training, setup_training};

    let ctx = setup_training(config);
    let scenarios =
        resolve_tui_scenarios(&ctx.tree, &tui_config.scenarios, config.game.num_players);
    let metrics = Arc::new(blueprint_tui_metrics::BlueprintTuiMetrics::new(
        config.training.iterations,
        config.training.time_limit_minutes,
    ));
    let shared_iters = Arc::clone(&ctx.iterations);
    let quit_flag = Arc::clone(&ctx.quit);
    let storage = Arc::clone(&ctx.storage);
    let tree = Arc::clone(&ctx.tree);
    let scenario_node_ids: Vec<u32> = scenarios.iter().map(|s| s.node_idx).collect();
    let tui_handle = spawn_mp_tui(&metrics, scenarios, tui_config, config.game.num_players);
    let train_config = config.clone();
    let train_handle =
        std::thread::spawn(move || run_training(&ctx, &train_config.training, &train_config.game));
    let telemetry_in_flight = Arc::new(AtomicBool::new(false));
    bridge_mp_iterations(
        &shared_iters,
        &storage,
        &tree,
        &scenario_node_ids,
        &metrics,
        &quit_flag,
        &train_handle,
        &config.snapshots,
        &telemetry_in_flight,
    );
    // Signal both the TUI and training thread to stop
    metrics.quit_requested.store(true, Ordering::Relaxed);
    quit_flag.store(true, Ordering::Relaxed);
    let result = train_handle.join().expect("training thread panicked");
    let _ = tui_handle.join();
    eprintln!(
        "Training complete: {} meta-iterations",
        result.meta_iterations
    );
    Ok(())
}

fn run_mp_with_tui_lazy(
    config: &BlueprintMpConfig,
    tui_config: &blueprint_tui_config::BlueprintTuiConfig,
) -> Result<(), Box<dyn Error>> {
    use poker_solver_core::blueprint_mp::trainer::{run_lazy_training, setup_lazy_training};

    let ctx = setup_lazy_training(config);
    let (scenarios, lazy_scenario_spots) = resolve_lazy_tui_scenarios(
        &ctx.game,
        &ctx.storage,
        ctx.bucket_counts,
        &tui_config.scenarios,
        config.game.num_players,
        0,
    );
    let metrics = Arc::new(blueprint_tui_metrics::BlueprintTuiMetrics::new(
        config.training.iterations,
        config.training.time_limit_minutes,
    ));
    let shared_iters = Arc::clone(&ctx.iterations);
    let quit_flag = Arc::clone(&ctx.quit);
    let storage = Arc::clone(&ctx.storage);
    let game = Arc::clone(&ctx.game);
    let bucket_counts = ctx.bucket_counts;
    let tui_handle = spawn_mp_tui(&metrics, scenarios, tui_config, config.game.num_players);
    let train_config = config.clone();
    let train_handle = std::thread::spawn(move || {
        run_lazy_training(&ctx, &train_config.training, &train_config.game)
    });

    bridge_mp_lazy_iterations(
        &shared_iters,
        &storage,
        &game,
        &lazy_scenario_spots,
        bucket_counts,
        &metrics,
        &quit_flag,
        &train_handle,
        &config.snapshots,
    );

    metrics.quit_requested.store(true, Ordering::Relaxed);
    quit_flag.store(true, Ordering::Relaxed);
    let result = train_handle.join().expect("lazy training thread panicked");
    let _ = tui_handle.join();
    eprintln!(
        "Lazy sparse training complete: {} meta-iterations",
        result.meta_iterations
    );
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
    snapshot_config: &poker_solver_core::blueprint_mp::config::MpSnapshotConfig,
    telemetry_in_flight: &Arc<AtomicBool>,
) {
    let mut last_telemetry = Instant::now();
    let telemetry_interval = Duration::from_secs(10);
    let started = Instant::now();
    loop {
        std::thread::sleep(Duration::from_millis(50));
        let iters = source.load(Ordering::Relaxed);
        metrics.iterations.store(iters, Ordering::Relaxed);
        if metrics.take_snapshot_trigger() {
            match save_mp_snapshot(snapshot_config, storage, tree, iters, started.elapsed()) {
                Ok(path) => eprintln!("  MP snapshot saved to {}", path.display()),
                Err(e) => eprintln!("  Warning: failed to save MP snapshot: {e}"),
            }
        }
        if last_telemetry.elapsed() >= telemetry_interval {
            if !telemetry_in_flight.swap(true, Ordering::Relaxed) {
                // Spawn telemetry scan on a background thread to avoid blocking
                // the iteration counter bridge. Keep only one full-storage scan
                // active so large abstractions cannot stack memory-bandwidth work.
                let s = Arc::clone(storage);
                let t = Arc::clone(tree);
                let m = Arc::clone(metrics);
                let nodes = scenario_node_ids.to_vec();
                let gate = Arc::clone(telemetry_in_flight);
                std::thread::spawn(move || {
                    push_mp_telemetry(&s, &t, &nodes, &m, iters);
                    gate.store(false, Ordering::Relaxed);
                });
            }
            last_telemetry = Instant::now();
        }
        if handle.is_finished() {
            metrics
                .iterations
                .store(source.load(Ordering::Relaxed), Ordering::Relaxed);
            break;
        }
        if metrics.quit_requested.load(Ordering::Relaxed) {
            quit_flag.store(true, Ordering::Relaxed);
            break;
        }
    }
}

fn bridge_mp_lazy_iterations<T>(
    source: &Arc<AtomicU64>,
    storage: &Arc<poker_solver_core::blueprint_mp::sparse_storage::SparseMpStorage>,
    game: &Arc<poker_solver_core::blueprint_mp::lazy_mccfr::LazyMpGame>,
    scenario_spots: &[poker_solver_core::blueprint_mp::lazy_mccfr::LazyResolvedSpot],
    bucket_counts: [u16; 4],
    metrics: &Arc<blueprint_tui_metrics::BlueprintTuiMetrics>,
    quit_flag: &Arc<std::sync::atomic::AtomicBool>,
    handle: &std::thread::JoinHandle<T>,
    snapshot_config: &poker_solver_core::blueprint_mp::config::MpSnapshotConfig,
) {
    let mut last_telemetry = Instant::now();
    let telemetry_interval = Duration::from_secs(10);
    let started = Instant::now();
    let mut previous_strategy_fingerprint = None;
    loop {
        std::thread::sleep(Duration::from_millis(50));
        let iters = source.load(Ordering::Relaxed);
        metrics.iterations.store(iters, Ordering::Relaxed);
        if metrics.take_snapshot_trigger() {
            match save_lazy_mp_snapshot(snapshot_config, storage, iters, started.elapsed()) {
                Ok(path) => eprintln!("  Lazy MP snapshot saved to {}", path.display()),
                Err(e) => eprintln!("  Warning: failed to save lazy MP snapshot: {e}"),
            }
        }
        if last_telemetry.elapsed() >= telemetry_interval {
            let sample = storage.telemetry_sample(LAZY_TUI_TELEMETRY_SAMPLE_ENTRIES);
            push_sparse_mp_telemetry(sample, &mut previous_strategy_fingerprint, metrics);
            push_lazy_mp_strategy_grids(
                storage,
                game,
                scenario_spots,
                bucket_counts,
                metrics,
                iters,
            );
            metrics.push_prune_fraction(take_mp_prune_pct());
            last_telemetry = Instant::now();
        }
        if handle.is_finished() {
            metrics
                .iterations
                .store(source.load(Ordering::Relaxed), Ordering::Relaxed);
            break;
        }
        if metrics.quit_requested.load(Ordering::Relaxed) {
            quit_flag.store(true, Ordering::Relaxed);
            break;
        }
    }
}

fn save_mp_snapshot(
    snapshot_config: &poker_solver_core::blueprint_mp::config::MpSnapshotConfig,
    storage: &poker_solver_core::blueprint_mp::storage::MpStorage,
    tree: &poker_solver_core::blueprint_mp::game_tree::MpGameTree,
    iterations: u64,
    elapsed: Duration,
) -> std::io::Result<PathBuf> {
    let output_dir = PathBuf::from(&snapshot_config.output_dir);
    std::fs::create_dir_all(&output_dir)?;
    let snapshot_idx = next_snapshot_index(&output_dir)?;
    let snapshot_dir = output_dir.join(format!("snapshot_{snapshot_idx:04}"));
    std::fs::create_dir_all(&snapshot_dir)?;

    let strategy = mp_strategy_from_storage(storage, tree, iterations, elapsed.as_secs() / 60);
    strategy.save(&snapshot_dir.join("strategy.bin"))?;
    save_mp_storage(storage, &snapshot_dir.join("regrets.bin"))?;

    let metadata = serde_json::json!({
        "kind": "blueprint_mp",
        "snapshot_index": snapshot_idx,
        "iterations": iterations,
        "elapsed_seconds": elapsed.as_secs(),
        "elapsed_minutes": elapsed.as_secs() / 60,
        "bucket_counts": storage.bucket_counts,
    });
    let metadata_json = serde_json::to_string_pretty(&metadata)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    std::fs::write(snapshot_dir.join("metadata.json"), metadata_json)?;
    Ok(snapshot_dir)
}

fn save_lazy_mp_snapshot(
    snapshot_config: &poker_solver_core::blueprint_mp::config::MpSnapshotConfig,
    storage: &poker_solver_core::blueprint_mp::sparse_storage::SparseMpStorage,
    iterations: u64,
    elapsed: Duration,
) -> std::io::Result<PathBuf> {
    let output_dir = PathBuf::from(&snapshot_config.output_dir);
    std::fs::create_dir_all(&output_dir)?;
    let snapshot_idx = next_snapshot_index(&output_dir)?;
    let snapshot_dir = output_dir.join(format!("snapshot_{snapshot_idx:04}"));
    std::fs::create_dir_all(&snapshot_dir)?;

    let entries = storage.snapshot_entries();
    let file = std::fs::File::create(snapshot_dir.join("sparse_entries.bin"))?;
    let writer = std::io::BufWriter::new(file);
    bincode::serialize_into(writer, &entries).map_err(|e| std::io::Error::other(e.to_string()))?;

    let stats = storage.stats();
    let metadata = serde_json::json!({
        "kind": "blueprint_mp_lazy_sparse",
        "snapshot_index": snapshot_idx,
        "iterations": iterations,
        "elapsed_seconds": elapsed.as_secs(),
        "elapsed_minutes": elapsed.as_secs() / 60,
        "entries": stats.entries,
        "regret_slots": stats.regret_slots,
        "strategy_slots": stats.strategy_slots,
        "approx_bytes": stats.approx_bytes,
    });
    let metadata_json = serde_json::to_string_pretty(&metadata)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    std::fs::write(snapshot_dir.join("metadata.json"), metadata_json)?;
    Ok(snapshot_dir)
}

fn next_snapshot_index(output_dir: &Path) -> std::io::Result<u32> {
    let mut next = 0_u32;
    for entry in std::fs::read_dir(output_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(num) = name.strip_prefix("snapshot_") else {
            continue;
        };
        if let Ok(idx) = num.parse::<u32>() {
            next = next.max(idx.saturating_add(1));
        }
    }
    Ok(next)
}

#[allow(clippy::cast_precision_loss)]
fn build_cluster_scorecard_json(
    cluster_dir: &Path,
    reports: &[poker_solver_core::blueprint_v2::cluster_diagnostics::ClusterReport],
    hand_class_reports: &[poker_solver_core::blueprint_v2::cluster_diagnostics::HandClassBucketAuditReport],
) -> Result<serde_json::Value, Box<dyn Error>> {
    Ok(serde_json::json!({
        "schema_version": 1,
        "cluster_dir": cluster_dir.display().to_string(),
        "cluster_reports": reports.iter().map(cluster_report_scorecard_json).collect::<Vec<_>>(),
        "hand_class_audits": hand_class_reports
            .iter()
            .map(hand_class_audit_scorecard_json)
            .collect::<Vec<_>>(),
        "selected_suited_hand_profiles": selected_suited_hand_profiles(cluster_dir)?,
    }))
}

#[allow(clippy::cast_precision_loss)]
fn cluster_report_scorecard_json(
    report: &poker_solver_core::blueprint_v2::cluster_diagnostics::ClusterReport,
) -> serde_json::Value {
    let mean = report.size_stats.mean;
    let max_to_mean = if mean > 0.0 {
        report.size_stats.max as f64 / mean
    } else {
        0.0
    };
    let std_to_mean = if mean > 0.0 {
        report.size_stats.std_dev / mean
    } else {
        0.0
    };
    let empty_buckets = report
        .bucket_sizes
        .iter()
        .filter(|&&count| count == 0)
        .count();

    serde_json::json!({
        "street": &report.street,
        "bucket_count": report.bucket_count,
        "board_count": report.board_count,
        "combos_per_board": report.combos_per_board,
        "total_entries": report.total_entries,
        "bucket_size": {
            "min": report.size_stats.min,
            "p50": percentile_usize(&report.bucket_sizes, 0.50),
            "p90": percentile_usize(&report.bucket_sizes, 0.90),
            "p99": percentile_usize(&report.bucket_sizes, 0.99),
            "max": report.size_stats.max,
            "mean": report.size_stats.mean,
            "std_dev": report.size_stats.std_dev,
            "max_to_mean": max_to_mean,
            "std_to_mean": std_to_mean,
            "empty_buckets": empty_buckets,
        },
    })
}

fn hand_class_audit_scorecard_json(
    report: &poker_solver_core::blueprint_v2::cluster_diagnostics::HandClassBucketAuditReport,
) -> serde_json::Value {
    let max_distinct_buckets = report
        .class_strength_groups
        .iter()
        .map(|group| group.distinct_buckets)
        .max()
        .unwrap_or(0);
    let max_bucket_entropy = report
        .class_strength_groups
        .iter()
        .map(|group| group.bucket_entropy)
        .fold(0.0_f64, f64::max);
    let max_class_entropy = report
        .bucket_mixes
        .iter()
        .map(|mix| mix.class_entropy)
        .fold(0.0_f64, f64::max);
    let max_equity_span = report
        .bucket_mixes
        .iter()
        .map(|mix| mix.max_equity - mix.min_equity)
        .fold(0.0_f64, f64::max);
    let max_inversion_delta = report
        .strength_inversions
        .iter()
        .map(|inversion| inversion.delta_buckets)
        .fold(0.0_f64, f64::max);

    serde_json::json!({
        "street": &report.street,
        "bucket_count": report.bucket_count,
        "sample_boards": report.sample_boards,
        "assignments": report.assignments,
        "skipped_lookups": report.skipped_lookups,
        "summary": {
            "class_strength_group_count": report.class_strength_groups.len(),
            "bucket_mix_count": report.bucket_mixes.len(),
            "strength_inversion_count": report.strength_inversions.len(),
            "max_distinct_buckets": max_distinct_buckets,
            "max_bucket_entropy": max_bucket_entropy,
            "max_class_entropy": max_class_entropy,
            "max_equity_span": max_equity_span,
            "max_inversion_delta": max_inversion_delta,
        },
        "worst_class_strength_spreads": report
            .class_strength_groups
            .iter()
            .take(report.top_n)
            .map(class_strength_group_scorecard_json)
            .collect::<Vec<_>>(),
        "worst_mixed_buckets": report
            .bucket_mixes
            .iter()
            .take(report.top_n)
            .map(bucket_mix_scorecard_json)
            .collect::<Vec<_>>(),
        "strength_inversions": report
            .strength_inversions
            .iter()
            .take(report.top_n)
            .map(strength_inversion_scorecard_json)
            .collect::<Vec<_>>(),
    })
}

fn class_strength_group_scorecard_json(
    group: &poker_solver_core::blueprint_v2::cluster_diagnostics::HandClassStrengthBucketStats,
) -> serde_json::Value {
    serde_json::json!({
        "contribution": &group.contribution,
        "hand_class": &group.hand_class,
        "strength": group.strength,
        "equity_decile": group.equity_decile,
        "count": group.count,
        "distinct_buckets": group.distinct_buckets,
        "bucket_entropy": group.bucket_entropy,
        "mean_bucket": group.mean_bucket,
        "top_buckets": group.top_buckets.iter().map(|bucket| {
            serde_json::json!({
                "bucket_id": bucket.bucket_id,
                "count": bucket.count,
                "share": bucket.share,
            })
        }).collect::<Vec<_>>(),
    })
}

fn bucket_mix_scorecard_json(
    mix: &poker_solver_core::blueprint_v2::cluster_diagnostics::BucketHandClassMixStats,
) -> serde_json::Value {
    serde_json::json!({
        "bucket_id": mix.bucket_id,
        "count": mix.count,
        "dominant_class": &mix.dominant_class,
        "dominant_share": mix.dominant_share,
        "class_entropy": mix.class_entropy,
        "mean_equity": mix.mean_equity,
        "min_equity": mix.min_equity,
        "max_equity": mix.max_equity,
        "equity_span": mix.max_equity - mix.min_equity,
        "top_classes": mix.top_classes.iter().map(|class| {
            serde_json::json!({
                "class_label": &class.class_label,
                "count": class.count,
                "share": class.share,
            })
        }).collect::<Vec<_>>(),
    })
}

fn strength_inversion_scorecard_json(
    inversion: &poker_solver_core::blueprint_v2::cluster_diagnostics::HandClassStrengthInversion,
) -> serde_json::Value {
    serde_json::json!({
        "contribution": &inversion.contribution,
        "hand_class": &inversion.hand_class,
        "weaker_strength": inversion.weaker_strength,
        "weaker_mean_bucket": inversion.weaker_mean_bucket,
        "stronger_strength": inversion.stronger_strength,
        "stronger_mean_bucket": inversion.stronger_mean_bucket,
        "delta_buckets": inversion.delta_buckets,
    })
}

fn percentile_usize(values: &[usize], percentile: f64) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    let idx = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn selected_suited_hand_profiles(
    cluster_dir: &Path,
) -> Result<Vec<serde_json::Value>, Box<dyn Error>> {
    let mut profiles = Vec::new();
    for (street_name, board_cards) in [("flop", 3_usize), ("turn", 4), ("river", 5)] {
        let path = cluster_dir.join(format!("{street_name}.buckets"));
        if !path.exists() {
            continue;
        }
        let bucket_file = poker_solver_core::blueprint_v2::bucket_file::BucketFile::load(&path)?;
        for (label, high, low) in selected_suited_hands() {
            let buckets = suited_hand_bucket_values(&bucket_file, board_cards, high, low);
            profiles.push(serde_json::json!({
                "street": street_name,
                "hand": label,
                "count": buckets.len(),
                "bucket_count": bucket_file.header.bucket_count,
                "mean_bucket": mean_u16(&buckets),
                "normalized_mean_bucket": normalized_mean_u16(&buckets, bucket_file.header.bucket_count),
                "min_bucket": buckets.iter().copied().min().unwrap_or(0),
                "p10_bucket": percentile_u16(&buckets, 0.10),
                "p50_bucket": percentile_u16(&buckets, 0.50),
                "p90_bucket": percentile_u16(&buckets, 0.90),
                "max_bucket": buckets.iter().copied().max().unwrap_or(0),
            }));
        }
    }
    Ok(profiles)
}

fn selected_suited_hands() -> Vec<(
    &'static str,
    poker_solver_core::poker::Value,
    poker_solver_core::poker::Value,
)> {
    use poker_solver_core::poker::Value;
    vec![
        ("KTs", Value::King, Value::Ten),
        ("K9s", Value::King, Value::Nine),
        ("K8s", Value::King, Value::Eight),
        ("K7s", Value::King, Value::Seven),
        ("K6s", Value::King, Value::Six),
        ("K5s", Value::King, Value::Five),
        ("K4s", Value::King, Value::Four),
        ("K3s", Value::King, Value::Three),
        ("K2s", Value::King, Value::Two),
        ("Q6s", Value::Queen, Value::Six),
        ("Q5s", Value::Queen, Value::Five),
        ("Q4s", Value::Queen, Value::Four),
    ]
}

fn suited_hand_bucket_values(
    bucket_file: &poker_solver_core::blueprint_v2::bucket_file::BucketFile,
    board_cards: usize,
    high: poker_solver_core::poker::Value,
    low: poker_solver_core::poker::Value,
) -> Vec<u16> {
    use poker_solver_core::blueprint_v2::cluster_pipeline::combo_index;
    use poker_solver_core::poker::{Card, Suit};

    let suits = [Suit::Spade, Suit::Heart, Suit::Diamond, Suit::Club];
    let holes = suits.map(|suit| [Card::new(high, suit), Card::new(low, suit)]);
    let combo_indices = holes.map(|hole| combo_index(hole[0], hole[1]));
    let mut buckets = Vec::new();

    for (board_idx, packed_board) in bucket_file.boards.iter().enumerate() {
        let board = packed_board.to_cards(board_cards);
        for (hole, &combo_idx) in holes.iter().zip(combo_indices.iter()) {
            if board.contains(&hole[0]) || board.contains(&hole[1]) {
                continue;
            }
            #[allow(clippy::cast_possible_truncation)]
            buckets.push(bucket_file.get_bucket(board_idx as u32, combo_idx));
        }
    }

    buckets
}

#[allow(clippy::cast_precision_loss)]
fn mean_u16(values: &[u16]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|&value| f64::from(value)).sum::<f64>() / values.len() as f64
}

#[allow(clippy::cast_precision_loss)]
fn normalized_mean_u16(values: &[u16], bucket_count: u16) -> f64 {
    if bucket_count <= 1 {
        return 0.0;
    }
    mean_u16(values) / f64::from(bucket_count - 1)
}

fn percentile_u16(values: &[u16], percentile: f64) -> u16 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    let idx = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn mp_strategy_from_storage(
    storage: &poker_solver_core::blueprint_mp::storage::MpStorage,
    tree: &poker_solver_core::blueprint_mp::game_tree::MpGameTree,
    iterations: u64,
    elapsed_minutes: u64,
) -> poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy {
    use poker_solver_core::blueprint_mp::game_tree::MpGameNode;

    let mut action_probs = Vec::new();
    let mut node_action_counts = Vec::new();
    let mut node_street_indices = Vec::new();

    for (node_idx, node) in tree.nodes.iter().enumerate() {
        if let MpGameNode::Decision {
            street, actions, ..
        } = node
        {
            let num_actions = actions.len();
            let street_idx = *street as u8;
            let buckets = storage.bucket_counts[street_idx as usize];
            let mut avg = vec![0.0_f64; num_actions];

            node_action_counts.push(num_actions as u16);
            node_street_indices.push(street_idx);
            for bucket in 0..buckets {
                storage.average_strategy(node_idx as u32, bucket, num_actions, &mut avg);
                action_probs.extend(avg.iter().map(|&p| p as f32));
            }
        }
    }

    let mut strategy = poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy {
        action_probs,
        node_action_counts,
        node_street_indices,
        bucket_counts: storage.bucket_counts,
        iterations,
        elapsed_minutes,
        node_offsets: Vec::new(),
    };
    strategy.post_deserialize();
    strategy
}

fn save_mp_storage(
    storage: &poker_solver_core::blueprint_mp::storage::MpStorage,
    path: &Path,
) -> std::io::Result<()> {
    let regrets: Vec<i32> = storage
        .regrets
        .iter()
        .map(|atom| atom.load(Ordering::Relaxed))
        .collect();
    let strategy_sums: Vec<u64> = storage
        .strategy_sums
        .iter()
        .map(|atom| atom.load(Ordering::Relaxed))
        .collect();
    let payload = (&storage.bucket_counts, &regrets, &strategy_sums);
    let file = std::fs::File::create(path)?;
    let writer = std::io::BufWriter::new(file);
    bincode::serialize_into(writer, &payload).map_err(|e| std::io::Error::other(e.to_string()))
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
    metrics.push_prune_fraction(take_mp_prune_pct());
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

fn push_sparse_mp_telemetry(
    sample: SparseTelemetrySample,
    previous_strategy_fingerprint: &mut Option<f64>,
    metrics: &blueprint_tui_metrics::BlueprintTuiMetrics,
) {
    if sample.entries_sampled == 0 {
        return;
    }
    let scale = poker_solver_core::blueprint_mp::storage::REGRET_SCALE;
    metrics.push_max_regret(f64::from(sample.max_positive_regret) / scale);
    metrics.push_min_regret(f64::from(sample.max_negative_regret) / scale);
    metrics.push_avg_pos_regret(sample.avg_positive_regret / scale);
    let delta = previous_strategy_fingerprint
        .map(|previous| (sample.strategy_fingerprint - previous).abs())
        .unwrap_or(0.0);
    *previous_strategy_fingerprint = Some(sample.strategy_fingerprint);
    metrics.push_strategy_delta(delta);
}

fn push_lazy_mp_strategy_grids(
    storage: &poker_solver_core::blueprint_mp::sparse_storage::SparseMpStorage,
    game: &poker_solver_core::blueprint_mp::lazy_mccfr::LazyMpGame,
    scenario_spots: &[poker_solver_core::blueprint_mp::lazy_mccfr::LazyResolvedSpot],
    bucket_counts: [u16; 4],
    metrics: &blueprint_tui_metrics::BlueprintTuiMetrics,
    iters: u64,
) {
    for (idx, &spot) in scenario_spots.iter().enumerate() {
        let grid_state = mp_tui_scenarios::extract_lazy_mp_grid(
            game,
            storage,
            spot,
            bucket_counts,
            iters,
            "",
            "",
            &[],
        );
        metrics.update_scenario_grid(idx, grid_state.cells);
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
            let (node_idx, _board) =
                mp_tui_scenarios::resolve_mp_spot(tree, &sc.spot, num_players)?;
            Some(mp_tui::ResolvedMpScenario {
                name: sc.name.clone(),
                node_idx,
                grid: default_hand_grid_state(&sc.name),
            })
        })
        .collect()
}

fn resolve_lazy_tui_scenarios(
    game: &poker_solver_core::blueprint_mp::lazy_mccfr::LazyMpGame,
    storage: &poker_solver_core::blueprint_mp::sparse_storage::SparseMpStorage,
    bucket_counts: [u16; 4],
    configs: &[blueprint_tui_config::ScenarioConfig],
    num_players: u8,
    iteration: u64,
) -> (
    Vec<mp_tui::ResolvedMpScenario>,
    Vec<poker_solver_core::blueprint_mp::lazy_mccfr::LazyResolvedSpot>,
) {
    let mut scenarios = Vec::new();
    let mut spots = Vec::new();
    for sc in configs {
        let Some((spot, board)) =
            mp_tui_scenarios::resolve_lazy_mp_spot(game, &sc.spot, num_players)
        else {
            continue;
        };
        let grid = mp_tui_scenarios::extract_lazy_mp_grid(
            game,
            storage,
            spot,
            bucket_counts,
            iteration,
            &sc.name,
            &sc.spot,
            &board,
        );
        let node_idx = u32::try_from(scenarios.len()).unwrap_or(u32::MAX);
        scenarios.push(mp_tui::ResolvedMpScenario {
            name: sc.name.clone(),
            node_idx,
            grid,
        });
        spots.push(spot);
    }
    (scenarios, spots)
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
    use poker_solver_core::blueprint_mp::config::{BlueprintMpConfig, MpTrainingBackend};
    use poker_solver_core::blueprint_v2::cluster_pipeline::PerFlopClusteringConfig;
    use poker_solver_core::blueprint_v2::config::BlueprintV2Config;
    use test_macros::timed_test;

    /// The sample per_flop_200bkt.yaml must parse and have per_flop set.
    #[test]
    fn per_flop_sample_yaml_parses() {
        let yaml = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sample_configurations/per_flop_200bkt.yaml"
        ))
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
        let yaml = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sample_configurations/per_flop_200bkt.yaml"
        ))
        .expect("sample per_flop_200bkt.yaml must exist");
        let cfg: BlueprintV2Config = serde_yaml::from_str(&yaml).expect("YAML must parse");

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
            "--oop-range",
            "AA",
            "--ip-range",
            "KK",
            "--flop",
            "Qs Jh 2c",
            "--turn",
            "8d",
            "--river",
            "3s",
            "--pot",
            "100",
            "--effective-stack",
            "100",
            "--iterations",
            "100",
        ]);
        assert!(
            cli.is_ok(),
            "gpu-range-solve CLI must parse: {:?}",
            cli.err()
        );
    }

    /// run_gpu_range_solve constructs a game and solves it without error.
    #[test]
    #[ignore = "requires CUDA/NVRTC runtime libraries"]
    fn run_gpu_range_solve_returns_ok() {
        let result = super::run_gpu_range_solve(
            "AA",
            "KK",
            "Qs Jh 2c",
            Some("8d"),
            Some("3s"),
            100,
            100,
            10,   // low iterations for speed
            10.0, // high target so it stops fast
            "100%",
            "",
            "100%",
            "",
        );
        assert!(
            result.is_ok(),
            "run_gpu_range_solve must succeed: {:?}",
            result.err()
        );
    }

    /// A config without per_flop should have per_flop as None.
    #[test]
    fn standard_config_has_no_per_flop() {
        let yaml = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sample_configurations/blueprint_v2_500bkt.yaml"
        ))
        .expect("blueprint_v2_500bkt.yaml must exist");
        let cfg: BlueprintV2Config = serde_yaml::from_str(&yaml).expect("YAML must parse");
        assert!(
            cfg.clustering.per_flop.is_none(),
            "standard config should not have per_flop"
        );
    }

    /// The sample TUI config with regret_audits must parse and contain two audit entries.
    #[test]
    fn tui_sample_yaml_parses_regret_audits() {
        let yaml = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sample_configurations/blueprint_v2_with_tui.yaml"
        ))
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

    #[test]
    fn inspect_mp_config_cli_parses() {
        use clap::Parser;
        let cli = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "inspect-mp-config",
            "--config",
            "/tmp/test.yaml",
        ]);
        assert!(
            cli.is_ok(),
            "inspect-mp-config CLI must parse: {:?}",
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

    #[timed_test(2)]
    fn run_train_blueprint_mp_lazy_sparse_no_tui_zero_iters_completes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lazy.yaml");
        let yaml = r#"
game:
  name: "lazy tiny"
  num_players: 2
  stack_depth: 20
  blinds:
    - seat: 0
      type: small_blind
      amount: 1
    - seat: 1
      type: big_blind
      amount: 2

action_abstraction:
  preflop:
    lead: []
    raise: []
  flop:
    lead: []
    raise: []
  turn:
    lead: []
    raise: []
  river:
    lead: []
    raise: []

clustering:
  preflop: { buckets: 10 }
  flop: { buckets: 10 }
  turn: { buckets: 10 }
  river: { buckets: 10 }

training:
  backend: lazy_sparse
  iterations: 0

snapshots:
  warmup_minutes: 1
  snapshot_every_minutes: 1
  output_dir: "/tmp/lazy_tiny"

tui:
  enabled: false
"#;
        std::fs::write(&path, yaml).unwrap();

        let result = super::run_train_blueprint_mp(path.to_str().unwrap(), true);

        assert!(result.is_ok());
    }

    #[timed_test(5)]
    fn mp_100bb_lazy_sparse_smoke_config_advances_without_dense_setup() {
        use poker_solver_core::blueprint_mp::trainer::{run_lazy_training, setup_lazy_training};

        let yaml = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sample_configurations/blueprint_mp_6max_100bb_lazy_sparse_smoke.yaml"
        ))
        .expect("blueprint_mp_6max_100bb_lazy_sparse_smoke.yaml must exist");
        let config: BlueprintMpConfig =
            serde_yaml::from_str(&yaml).expect("YAML must parse as BlueprintMpConfig");

        let report = super::inspect_mp_config(&config).unwrap();
        assert_eq!(report.num_players, 6);
        assert_eq!(report.stack_bb, 100.0);
        assert_eq!(report.preflop_raise_rows, 2);
        assert_eq!(report.backend, MpTrainingBackend::LazySparse);
        assert!(
            report.eager_risk.is_some(),
            "the sample must remain a known dense eager risk"
        );

        let ctx = setup_lazy_training(&config);
        assert_eq!(
            ctx.storage.stats().entries,
            0,
            "lazy setup should not allocate visited infosets"
        );

        let result = run_lazy_training(&ctx, &config.training, &config.game);
        let stats = ctx.storage.stats();

        assert_eq!(result.meta_iterations, 1);
        assert_eq!(ctx.iterations.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert!(
            stats.entries > 0,
            "one lazy meta-iteration should visit and allocate sparse infosets"
        );
        assert!(
            stats.approx_bytes < 10 * 1024 * 1024,
            "one lazy smoke iteration should stay bounded, got {} bytes",
            stats.approx_bytes
        );
    }

    #[test]
    fn inspect_mp_config_flags_100bb_multi_preflop_raise_rows() {
        let yaml = r#"
game:
  name: "100bb risk"
  num_players: 6
  stack_depth: 200
  blinds:
    - seat: 4
      type: small_blind
      amount: 1
    - seat: 5
      type: big_blind
      amount: 2

action_abstraction:
  preflop:
    lead: ["2bb"]
    raise:
      - ["1.0x"]
      - ["1.0x"]
  flop:
    lead: [0.75]
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
  preflop: { buckets: 169 }
  flop: { buckets: 500 }
  turn: { buckets: 100 }
  river: { buckets: 100 }

training:
  iterations: 1

snapshots:
  warmup_minutes: 1
  snapshot_every_minutes: 1
  output_dir: "/tmp/risk"
"#;
        let config: BlueprintMpConfig = serde_yaml::from_str(yaml).unwrap();

        let report = super::inspect_mp_config(&config).unwrap();

        assert_eq!(report.stack_bb, 100.0);
        assert_eq!(report.preflop_raise_rows, 2);
        assert!(
            report.eager_risk.is_some(),
            "100bb 6-max with two preflop raise rows should be flagged"
        );
    }

    #[test]
    fn inspect_mp_config_reports_lazy_sparse_backend_without_clearing_eager_risk() {
        let yaml = r#"
game:
  name: "100bb sparse"
  num_players: 6
  stack_depth: 200
  blinds:
    - seat: 4
      type: small_blind
      amount: 1
    - seat: 5
      type: big_blind
      amount: 2

action_abstraction:
  preflop:
    lead: ["2bb"]
    raise:
      - ["1.0x"]
      - ["1.0x"]
  flop:
    lead: [0.75]
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
  preflop: { buckets: 169 }
  flop: { buckets: 500 }
  turn: { buckets: 100 }
  river: { buckets: 100 }

training:
  backend: lazy_sparse
  iterations: 1

snapshots:
  warmup_minutes: 1
  snapshot_every_minutes: 1
  output_dir: "/tmp/sparse"
"#;
        let config: BlueprintMpConfig = serde_yaml::from_str(yaml).unwrap();

        let report = super::inspect_mp_config(&config).unwrap();

        assert_eq!(report.backend, MpTrainingBackend::LazySparse);
        assert!(
            report.eager_risk.is_some(),
            "preflight should still show the dense backend risk for context"
        );
    }

    #[test]
    fn mp_no_tui_heartbeat_interval_is_one_minute() {
        assert_eq!(
            super::MpNoTuiHeartbeat::INTERVAL,
            std::time::Duration::from_secs(60)
        );
    }

    #[test]
    fn mp_snapshot_save_creates_strategy_and_metadata() {
        use poker_solver_core::blueprint_mp::config::{
            ForcedBet, ForcedBetKind, MpActionAbstractionConfig, MpClusteringConfig, MpGameConfig,
            MpSnapshotConfig, MpStreetCluster, MpStreetSizes, MpTrainingConfig,
        };
        use poker_solver_core::blueprint_mp::trainer::setup_training;
        use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;

        let tiny_preflop_size = MpStreetSizes {
            lead: vec![serde_yaml::Value::String("1bb".into())],
            raise: vec![],
        };
        let tiny_postflop_size = MpStreetSizes {
            lead: vec![serde_yaml::Value::Number(serde_yaml::Number::from(1))],
            raise: vec![],
        };
        let config = BlueprintMpConfig {
            game: MpGameConfig {
                name: "snapshot test".into(),
                num_players: 2,
                stack_depth: 6.0,
                blinds: vec![
                    ForcedBet {
                        seat: 0,
                        kind: ForcedBetKind::SmallBlind,
                        amount: 1.0,
                    },
                    ForcedBet {
                        seat: 1,
                        kind: ForcedBetKind::BigBlind,
                        amount: 2.0,
                    },
                ],
                rake_rate: 0.0,
                rake_cap: 0.0,
            },
            action_abstraction: MpActionAbstractionConfig {
                max_flop_players: None,
                preflop: tiny_preflop_size,
                flop: tiny_postflop_size.clone(),
                turn: tiny_postflop_size.clone(),
                river: tiny_postflop_size,
            },
            clustering: MpClusteringConfig {
                preflop: MpStreetCluster { buckets: 2 },
                flop: MpStreetCluster { buckets: 2 },
                turn: MpStreetCluster { buckets: 2 },
                river: MpStreetCluster { buckets: 2 },
            },
            training: MpTrainingConfig {
                backend: MpTrainingBackend::Eager,
                cluster_path: None,
                iterations: Some(1),
                time_limit_minutes: None,
                lcfr_warmup_iterations: 0,
                lcfr_discount_interval: 50,
                prune_after_iterations: 1_000_000,
                prune_threshold: -250,
                prune_explore_pct: 0.05,
                batch_size: 1,
                dcfr_alpha: 1.5,
                dcfr_beta: 0.0,
                dcfr_gamma: 2.0,
                print_every_minutes: 999,
                purify_threshold: 0.0,
                exploitability_interval_minutes: 0,
                exploitability_samples: 0,
            },
            snapshots: MpSnapshotConfig {
                warmup_minutes: 0,
                snapshot_every_minutes: 1,
                output_dir: String::new(),
                resume: false,
                max_snapshots: None,
            },
        };
        let ctx = setup_training(&config);
        let dir = tempfile::tempdir().unwrap();
        let snapshot_config = MpSnapshotConfig {
            output_dir: dir.path().to_string_lossy().into_owned(),
            ..config.snapshots
        };

        let snapshot_dir = super::save_mp_snapshot(
            &snapshot_config,
            &ctx.storage,
            &ctx.tree,
            123,
            std::time::Duration::from_secs(7),
        )
        .expect("snapshot save should succeed");

        assert!(snapshot_dir.join("strategy.bin").exists());
        assert!(snapshot_dir.join("regrets.bin").exists());
        assert!(snapshot_dir.join("metadata.json").exists());

        let strategy = BlueprintV2Strategy::load(&snapshot_dir.join("strategy.bin")).unwrap();
        assert_eq!(strategy.iterations, 123);
        assert_eq!(strategy.bucket_counts, [2, 2, 2, 2]);
        assert!(
            !strategy.node_action_counts.is_empty(),
            "snapshot should include decision-node strategy data"
        );

        let metadata: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(snapshot_dir.join("metadata.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(metadata["kind"], "blueprint_mp");
        assert_eq!(metadata["iterations"], 123);
    }

    #[test]
    fn lazy_mp_snapshot_save_creates_sparse_entries_and_metadata() {
        use poker_solver_core::blueprint_mp::config::MpSnapshotConfig;
        use poker_solver_core::blueprint_mp::sparse_storage::MpInfosetKey;
        use poker_solver_core::blueprint_mp::types::{Seat, Street};

        let storage =
            poker_solver_core::blueprint_mp::sparse_storage::SparseMpStorage::with_shards(4);
        let key =
            MpInfosetKey::from_street_bucket(Seat::from_raw(0), Street::Preflop, 1, 0, 0, 0, 0);
        storage.add_regret(key, 2, 0, 25);
        storage.add_strategy_sum(key, 2, 1, 50);
        let dir = tempfile::tempdir().unwrap();
        let snapshot_config = MpSnapshotConfig {
            warmup_minutes: 0,
            snapshot_every_minutes: 1,
            output_dir: dir.path().to_string_lossy().into_owned(),
            resume: false,
            max_snapshots: None,
        };

        let snapshot_dir = super::save_lazy_mp_snapshot(
            &snapshot_config,
            &storage,
            456,
            std::time::Duration::from_secs(11),
        )
        .expect("lazy sparse snapshot save should succeed");

        assert!(snapshot_dir.join("sparse_entries.bin").exists());
        assert!(snapshot_dir.join("metadata.json").exists());
        let metadata: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(snapshot_dir.join("metadata.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(metadata["kind"], "blueprint_mp_lazy_sparse");
        assert_eq!(metadata["iterations"], 456);
        assert_eq!(metadata["entries"], 1);
    }

    #[test]
    fn sparse_mp_telemetry_pushes_regret_and_strategy_delta() {
        let metrics = crate::blueprint_tui_metrics::BlueprintTuiMetrics::new(None, None);
        let mut previous = None;
        let first = super::SparseTelemetrySample {
            entries_sampled: 2,
            regret_slots_sampled: 5,
            max_positive_regret: 2000,
            max_negative_regret: -1000,
            avg_positive_regret: 1500.0,
            strategy_fingerprint: 2.0,
        };
        let second = super::SparseTelemetrySample {
            strategy_fingerprint: 2.25,
            ..first.clone()
        };

        super::push_sparse_mp_telemetry(first, &mut previous, &metrics);
        super::push_sparse_mp_telemetry(second, &mut previous, &metrics);

        assert_eq!(
            *metrics.max_regret_history.lock().unwrap(),
            vec![2000.0, 2000.0]
        );
        assert_eq!(
            *metrics.min_regret_history.lock().unwrap(),
            vec![-1000.0, -1000.0]
        );
        assert_eq!(
            *metrics.avg_pos_regret_history.lock().unwrap(),
            vec![1500.0, 1500.0]
        );
        assert_eq!(
            *metrics.strategy_delta_history.lock().unwrap(),
            vec![0.0, 0.25]
        );
    }

    #[test]
    fn sample_mp_regret_summary_reports_scaled_stats() {
        let regrets = [
            std::sync::atomic::AtomicI32::new(100),
            std::sync::atomic::AtomicI32::new(-60),
            std::sync::atomic::AtomicI32::new(40),
            std::sync::atomic::AtomicI32::new(0),
        ];

        let summary = super::sample_mp_regret_summary(&regrets, 20.0, 10).unwrap();

        assert_eq!(summary.max_positive, 5.0);
        assert_eq!(summary.max_negative, -3.0);
        assert!((summary.avg_positive - 3.5).abs() < 1e-9);
        assert_eq!(summary.positive_count, 2);
        assert_eq!(summary.samples, 4);
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
        // The sample enables TUI scenarios for interactive monitoring.
        assert!(tui_cfg.enabled);
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
    #[timed_test(30)]
    fn resolve_tui_scenarios_from_tree() {
        use poker_solver_core::blueprint_mp::config::*;
        use poker_solver_core::blueprint_mp::game_tree::MpGameTree;

        fn yaml_f64(v: f64) -> serde_yaml::Value {
            serde_yaml::Value::Number(serde_yaml::Number::from(v))
        }

        let game = MpGameConfig {
            name: "test".into(),
            num_players: 6,
            stack_depth: 40.0,
            blinds: vec![
                ForcedBet {
                    seat: 4,
                    kind: ForcedBetKind::SmallBlind,
                    amount: 1.0,
                },
                ForcedBet {
                    seat: 5,
                    kind: ForcedBetKind::BigBlind,
                    amount: 2.0,
                },
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
            max_flop_players: None,
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

    #[timed_test(20)]
    fn resolve_lazy_tui_scenarios_from_lazy_game() {
        use poker_solver_core::blueprint_mp::config::*;
        use poker_solver_core::blueprint_mp::lazy_mccfr::LazyMpGame;
        use poker_solver_core::blueprint_mp::sparse_storage::SparseMpStorage;

        let game = MpGameConfig {
            name: "test".into(),
            num_players: 6,
            stack_depth: 40.0,
            blinds: vec![
                ForcedBet {
                    seat: 4,
                    kind: ForcedBetKind::SmallBlind,
                    amount: 1.0,
                },
                ForcedBet {
                    seat: 5,
                    kind: ForcedBetKind::BigBlind,
                    amount: 2.0,
                },
            ],
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        let preflop = MpStreetSizes {
            lead: vec![serde_yaml::Value::String("5bb".into())],
            raise: vec![vec![serde_yaml::Value::String("5bb".into())]],
        };
        let empty = MpStreetSizes {
            lead: vec![],
            raise: vec![],
        };
        let action = MpActionAbstractionConfig {
            max_flop_players: None,
            preflop,
            flop: empty.clone(),
            turn: empty.clone(),
            river: empty,
        };
        let lazy_game = LazyMpGame::new(&game, &action);
        let storage = SparseMpStorage::with_shards(4);
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

        let (resolved, spots) = super::resolve_lazy_tui_scenarios(
            &lazy_game,
            &storage,
            [169, 50, 50, 50],
            &scenarios,
            6,
            0,
        );

        assert_eq!(resolved.len(), 2);
        assert_eq!(spots.len(), 2);
        assert_eq!(resolved[0].name, "UTG open");
        assert_eq!(resolved[1].name, "HJ vs UTG");
        assert!(!resolved[0].grid.cells[0][0].actions.is_empty());
        assert!(!resolved[1].grid.cells[0][0].actions.is_empty());
    }

    /// compare-solve should accept per-street boundary flags.
    #[test]
    fn compare_solve_street_boundary_cli_flags_parse() {
        use clap::Parser;

        // All-exact defaults
        let cli = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "compare-solve",
            "--bundle",
            "/tmp/test",
            "--spot",
            "sb:2bb,bb:call|Jd9d7d",
        ]);
        assert!(cli.is_ok(), "default flags must parse: {:?}", cli.err());

        if let super::Commands::CompareSolve {
            flop_boundary,
            flop_model_kind,
            turn_boundary,
            turn_model_kind,
            river_boundary,
            river_model,
            river_model_kind,
            ..
        } = cli.unwrap().command
        {
            assert_eq!(flop_boundary, "exact");
            assert_eq!(flop_model_kind, "direct");
            assert_eq!(turn_boundary, "exact");
            assert_eq!(turn_model_kind, "direct");
            assert_eq!(river_boundary, "exact");
            assert_eq!(river_model_kind, "direct");
            assert!(river_model.is_none());
        } else {
            panic!("expected CompareSolve variant");
        }

        // River cfvnet defaults to direct model inference unless the caller
        // explicitly requests the legacy river-enumerated turn adapter.
        let cli2 = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "compare-solve",
            "--bundle",
            "/tmp/test",
            "--spot",
            "sb:2bb,bb:call|Jd9d7d",
            "--river-boundary",
            "cfvnet",
            "--river-model",
            "/path/to/model.onnx",
        ]);
        assert!(
            cli2.is_ok(),
            "river-cfvnet flags must parse: {:?}",
            cli2.err()
        );

        if let super::Commands::CompareSolve {
            river_boundary,
            river_model,
            river_model_kind,
            ..
        } = cli2.unwrap().command
        {
            assert_eq!(river_boundary, "cfvnet");
            assert_eq!(river_model.as_deref(), Some("/path/to/model.onnx"));
            assert_eq!(river_model_kind, "direct");
        } else {
            panic!("expected CompareSolve variant");
        }

        let cli2_legacy = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "compare-solve",
            "--bundle",
            "/tmp/test",
            "--spot",
            "sb:2bb,bb:call|Jd9d7d",
            "--turn-boundary",
            "cfvnet",
            "--turn-model",
            "/path/to/river-model.onnx",
            "--turn-model-kind",
            "river_enumerated_turn",
        ]);
        assert!(
            cli2_legacy.is_ok(),
            "legacy river-enumerated adapter must still parse: {:?}",
            cli2_legacy.err()
        );

        if let super::Commands::CompareSolve {
            turn_boundary,
            turn_model,
            turn_model_kind,
            ..
        } = cli2_legacy.unwrap().command
        {
            assert_eq!(turn_boundary, "cfvnet");
            assert_eq!(turn_model.as_deref(), Some("/path/to/river-model.onnx"));
            assert_eq!(turn_model_kind, "river_enumerated_turn");
        } else {
            panic!("expected CompareSolve variant");
        }

        // River exact_oracle does not require a model path.
        let cli3 = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "compare-solve",
            "--bundle",
            "/tmp/test",
            "--spot",
            "sb:2bb,bb:call|Jd9d7d",
            "--river-boundary",
            "exact_oracle",
        ]);
        assert!(
            cli3.is_ok(),
            "river-exact-oracle flags must parse: {:?}",
            cli3.err()
        );

        if let super::Commands::CompareSolve {
            river_boundary,
            river_model,
            ..
        } = cli3.unwrap().command
        {
            assert_eq!(river_boundary, "exact_oracle");
            assert!(river_model.is_none());
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
            "--bundle",
            "/tmp/test",
            "--spot",
            "sb:2bb,bb:call|Jd9d7d",
            "--trace-boundaries",
            "0,42,100",
            "--trace-iters",
            "0,9",
            "--trace-dir",
            "/tmp/traces",
        ]);
        assert!(cli.is_ok(), "trace flags must parse: {:?}", cli.err());

        if let super::Commands::CompareSolve {
            trace_boundaries,
            trace_iters,
            trace_dir,
            ..
        } = cli.unwrap().command
        {
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
            "--bundle",
            "/tmp/test",
            "--spot",
            "sb:2bb,bb:call|Jd9d7d",
        ]);
        assert!(cli2.is_ok());

        if let super::Commands::CompareSolve {
            trace_boundaries,
            trace_iters,
            trace_dir,
            ..
        } = cli2.unwrap().command
        {
            assert!(trace_boundaries.is_none());
            assert_eq!(trace_iters, "last");
            assert_eq!(trace_dir, std::path::PathBuf::from("./traces"));
        } else {
            panic!("expected CompareSolve variant");
        }
    }

    /// compare-solve hidden iteration overrides should parse independently.
    #[test]
    fn compare_solve_iteration_override_flags_parse() {
        use clap::Parser;

        let cli = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "compare-solve",
            "--bundle",
            "/tmp/test",
            "--spot",
            "sb:2bb,bb:call|Jd9d7d",
            "--iters",
            "200",
            "--exact-iters",
            "1000",
            "--subgame-iters",
            "400",
            "--oracle-iteration-aligned",
            "--root-update-trace-iters",
            "0,999",
        ]);
        assert!(
            cli.is_ok(),
            "iteration override flags must parse: {:?}",
            cli.err()
        );

        if let super::Commands::CompareSolve {
            iters,
            exact_iters,
            subgame_iters,
            oracle_iteration_aligned,
            root_update_trace_iters,
            ..
        } = cli.unwrap().command
        {
            assert_eq!(iters, 200);
            assert_eq!(exact_iters, Some(1000));
            assert_eq!(subgame_iters, Some(400));
            assert!(oracle_iteration_aligned);
            assert_eq!(root_update_trace_iters.as_deref(), Some("0,999"));
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
            "--bundle",
            "/tmp/test",
            "--spot",
            "sb:2bb,bb:call|Jd9d7d",
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
            "--bundle",
            "/tmp/test",
            "--spot",
            "sb:2bb,bb:call|Jd9d7d",
            "--gadget",
        ]);
        assert!(cli2.is_ok());
        if let super::Commands::CompareSolve { gadget, .. } = cli2.unwrap().command {
            assert!(gadget, "gadget should be true when --gadget is passed");
        } else {
            panic!("expected CompareSolve variant");
        }
    }

    /// --gadget-provider and --gadget-constant flags parse correctly.
    #[test]
    fn compare_solve_accepts_gadget_provider_flag() {
        use clap::Parser;

        let cli = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "compare-solve",
            "--bundle",
            "/tmp/dummy",
            "--spot",
            "any",
            "--river-boundary",
            "cfvnet",
            "--river-model",
            "/tmp/m",
            "--gadget",
            "--gadget-provider",
            "constant",
            "--gadget-constant",
            "-0.5",
        ])
        .expect("should parse");
        if let super::Commands::CompareSolve {
            gadget,
            gadget_provider,
            gadget_constant,
            ..
        } = cli.command
        {
            assert!(gadget);
            assert_eq!(gadget_provider, "constant");
            assert_eq!(gadget_constant, -0.5);
        } else {
            panic!("expected CompareSolve");
        }
    }

    /// --gadget-provider defaults to "blueprint-cbv" and --gadget-constant defaults to 0.0.
    #[test]
    fn compare_solve_gadget_provider_defaults() {
        use clap::Parser;

        let cli = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "compare-solve",
            "--bundle",
            "/tmp/dummy",
            "--spot",
            "any",
            "--gadget",
        ])
        .expect("should parse");
        if let super::Commands::CompareSolve {
            gadget,
            gadget_provider,
            gadget_constant,
            ..
        } = cli.command
        {
            assert!(gadget);
            assert_eq!(gadget_provider, "blueprint-cbv");
            assert_eq!(gadget_constant, 0.0);
        } else {
            panic!("expected CompareSolve");
        }
    }

    /// --gadget-clamp flag defaults to false.
    #[test]
    fn compare_solve_gadget_clamp_defaults_false() {
        use clap::Parser;

        let cli = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "compare-solve",
            "--bundle",
            "/tmp/dummy",
            "--spot",
            "any",
        ])
        .expect("should parse");
        if let super::Commands::CompareSolve { gadget_clamp, .. } = cli.command {
            assert!(!gadget_clamp, "gadget_clamp should default to false");
        } else {
            panic!("expected CompareSolve");
        }
    }

    /// --gadget-clamp can be set explicitly.
    #[test]
    fn compare_solve_gadget_clamp_can_be_set() {
        use clap::Parser;

        let cli = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "compare-solve",
            "--bundle",
            "/tmp/dummy",
            "--spot",
            "any",
            "--gadget-clamp",
        ])
        .expect("should parse");
        if let super::Commands::CompareSolve { gadget_clamp, .. } = cli.command {
            assert!(
                gadget_clamp,
                "gadget_clamp should be true when --gadget-clamp is passed"
            );
        } else {
            panic!("expected CompareSolve");
        }
    }

    /// --gadget and --gadget-clamp are mutually exclusive: passing both is an error.
    #[test]
    fn compare_solve_gadget_and_gadget_clamp_mutually_exclusive() {
        use clap::Parser;

        let result = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "compare-solve",
            "--bundle",
            "/tmp/dummy",
            "--spot",
            "any",
            "--gadget",
            "--gadget-clamp",
        ]);
        assert!(
            result.is_err(),
            "--gadget and --gadget-clamp should be mutually exclusive"
        );
    }

    /// --gadget-clamp accepts --gadget-provider and --gadget-constant flags.
    #[test]
    fn compare_solve_gadget_clamp_with_provider_and_constant() {
        use clap::Parser;

        let cli = super::Cli::try_parse_from([
            "poker-solver-trainer",
            "compare-solve",
            "--bundle",
            "/tmp/dummy",
            "--spot",
            "any",
            "--gadget-clamp",
            "--gadget-provider",
            "constant",
            "--gadget-constant",
            "-0.5",
        ])
        .expect("should parse");
        if let super::Commands::CompareSolve {
            gadget,
            gadget_clamp,
            gadget_provider,
            gadget_constant,
            ..
        } = cli.command
        {
            assert!(!gadget);
            assert!(gadget_clamp);
            assert_eq!(gadget_provider, "constant");
            assert_eq!(gadget_constant, -0.5);
        } else {
            panic!("expected CompareSolve");
        }
    }

    /// gadget_mode_label returns correct mode strings.
    #[test]
    fn gadget_mode_label_returns_correct_modes() {
        assert_eq!(super::compare_solve::gadget_mode_label(true, false), "tree");
        assert_eq!(
            super::compare_solve::gadget_mode_label(false, true),
            "clamp"
        );
        assert_eq!(super::compare_solve::gadget_mode_label(false, false), "off");
    }
}
