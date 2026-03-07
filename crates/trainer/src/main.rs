mod blueprint_tui;
mod blueprint_tui_config;
mod blueprint_tui_metrics;
mod blueprint_tui_scenarios;
mod blueprint_tui_widgets;
mod bucket_diagnostics;
mod hand_trace;
mod lhe_viz;
mod log_file;
mod tree_dump;
mod tui;
mod tui_metrics;

use std::error::Error;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use poker_solver_core::flops::{self, CanonicalFlop, RankTexture, SuitTexture};
use poker_solver_core::preflop::{
    EquityTable, PostflopBundle, PostflopModelConfig, PostflopSolveType, PreflopAction,
    PreflopBundle, PreflopConfig, PreflopNode, PreflopSolver, PreflopTree, SolverCounters,
};
use poker_solver_core::preflop::postflop_abstraction::{BuildPhase, FlopStage, PostflopAbstraction};
use poker_solver_core::preflop::postflop_hands::{build_combo_map, parse_flops, sample_canonical_flops};
use poker_solver_core::preflop::postflop_exhaustive::compute_equity_table;
use poker_solver_core::preflop::equity_table_cache::EquityTableCache;
use poker_solver_core::preflop::rank_array_cache::{
    compute_rank_arrays, derive_equity_table, RankArrayCache,
};
use poker_solver_core::blueprint_v2::config::BlueprintV2Config;
use poker_solver_core::blueprint_v2::trainer::BlueprintTrainer;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[command(name = "poker-solver-trainer")]
#[command(about = "Poker solver training tools: preflop/postflop solving, diagnostics")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Parser)]
enum Commands {
    /// Solve preflop strategy using Linear CFR.
    /// Requires a config file with `postflop_model_path` pointing to a pre-built bundle.
    SolvePreflop {
        /// YAML config file (contains PreflopConfig + training params + postflop_model_path).
        #[arg(short, long)]
        config: PathBuf,
        /// Number of LCFR iterations (overrides config)
        #[arg(long)]
        iterations: Option<u64>,
        /// Output directory for the preflop bundle
        #[arg(short, long)]
        output: PathBuf,
        /// Print strategy matrices every N iterations (0 = only at end; overrides config)
        #[arg(long)]
        print_every: Option<u64>,
        /// Monte Carlo samples per hand matchup for equity table (0 = uniform; overrides config)
        #[arg(long)]
        equity_samples: Option<u32>,
        /// Print strategy matrices in plain text (no ANSI colors) for machine consumption
        #[arg(long)]
        claude_debug: bool,
    },
    /// Build a postflop abstraction and save it as a reusable bundle
    SolvePostflop {
        /// YAML config file (same format as solve-preflop, reads postflop_model section)
        #[arg(short, long)]
        config: PathBuf,
        /// Output directory for the postflop bundle
        #[arg(short, long)]
        output: PathBuf,
        /// TUI dashboard refresh interval in seconds (default: 1.0)
        #[arg(long, default_value = "1.0")]
        tui_refresh: f64,
    },
    /// List all 1,755 canonical (suit-isomorphic) flops
    Flops {
        /// Output format
        #[arg(short, long, default_value = "json")]
        format: OutputFormat,
        /// Output file (defaults to stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Run EHS bucket diagnostics on a postflop abstraction config
    DiagBuckets {
        /// YAML config file (same format as solve-preflop)
        #[arg(short, long)]
        config: PathBuf,
        /// Directory for abstraction cache
        #[arg(long, default_value = "cache/postflop")]
        cache_dir: PathBuf,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Trace all 169 hands through the full postflop pipeline (EHS → buckets → EV)
    TraceHand {
        /// YAML config file (same format as solve-preflop)
        #[arg(short, long)]
        config: PathBuf,
    },
    /// Precompute equity tables for all 1,755 canonical flops and save to disk.
    /// These tables are auto-loaded by solve-postflop to skip the expensive
    /// per-flop equity computation at startup.
    PrecomputeEquity {
        /// Output file path
        #[arg(short, long, default_value = "cache/equity_tables.bin")]
        output: PathBuf,
    },
    /// Dump the postflop betting tree as indented text or Graphviz DOT.
    DumpTree {
        /// YAML config file (same format as solve-postflop)
        #[arg(short, long)]
        config: PathBuf,
        /// Emit Graphviz DOT instead of indented text
        #[arg(long)]
        dot: bool,
        /// Output file (defaults to stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Stack-to-pot ratio
        #[arg(long, default_value = "3.0")]
        spr: f64,
        /// Pot type label (limped, raised, 3bet, 4bet)
        #[arg(long, default_value = "raised")]
        pot_type: String,
    },
    /// Inspect a saved preflop bundle: dump strategy & action labels at key nodes.
    InspectPreflop {
        /// Path to preflop bundle directory (or checkpoint directory)
        #[arg(short, long)]
        path: PathBuf,
        /// Comma-separated hand names (e.g. "AA,Q8s,K5s")
        #[arg(long, default_value = "AA,KK,AKs,K9s,K7s,K5s,Q9s,Q8s,Q7s,JTs,J9s,T9s,T8s,98s,87s,76s,65s,A5s,72o")]
        hands: String,
        /// Walk to a specific node by action history (e.g. "r:3" for BB facing SB 2bb raise)
        #[arg(long)]
        history: Option<String>,
    },
    /// Trace raw regret/strategy-sum evolution for specific hands over first N iterations.
    TraceRegrets {
        /// YAML config file (same as solve-preflop)
        #[arg(short, long)]
        config: PathBuf,
        /// Number of iterations to trace
        #[arg(long, default_value = "10")]
        iterations: u64,
        /// Comma-separated hand names
        #[arg(long, default_value = "AA,Q8s,K5s,T9s,87s,72o")]
        hands: String,
        /// Action history to reach the target node (e.g. "r:3")
        #[arg(long, default_value = "r:3")]
        history: String,
    },
    /// Decompose per-opponent regret contributions at a target node after N iterations.
    DecomposeRegrets {
        /// YAML config file (same as solve-preflop)
        #[arg(short, long)]
        config: PathBuf,
        /// Warm-up iterations before decomposition
        #[arg(long, default_value = "5")]
        iterations: u64,
        /// Hero hand to decompose (e.g. "Q8s")
        #[arg(long)]
        hand: String,
        /// Action history to reach the target node (e.g. "r:3")
        #[arg(long, default_value = "r:3")]
        history: String,
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
    },
    /// Dump terminal node values from a decision node for a specific matchup.
    TraceTerminals {
        /// YAML config file (same as solve-preflop)
        #[arg(short, long)]
        config: PathBuf,
        /// Hero hand (e.g. "AA")
        #[arg(long)]
        hero: String,
        /// Opponent hand (e.g. "72o")
        #[arg(long)]
        opp: String,
        /// Action history to reach the target node (e.g. "call,r:3")
        #[arg(long, default_value = "call,r:3")]
        history: String,
    },
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

/// Output format for the flops command.
#[derive(Debug, Clone, ValueEnum)]
enum OutputFormat {
    Json,
    Csv,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::SolvePreflop {
            config,
            iterations,
            output,
            print_every,
            equity_samples,
            claude_debug,
        } => {
            run_solve_preflop(
                &config,
                iterations,
                &output,
                print_every,
                equity_samples,
                claude_debug,
            )?;
        }
        Commands::SolvePostflop { config, output, tui_refresh } => {
            run_solve_postflop(&config, &output, tui_refresh)?;
        }
        Commands::Flops { format, output } => {
            run_flops(format, output)?;
        }
        Commands::DiagBuckets { config, cache_dir, json } => {
            let yaml = std::fs::read_to_string(&config)?;
            let pf_solve: PostflopSolveConfig = serde_yaml::from_str(&yaml)?;
            let all_passed = bucket_diagnostics::run(&pf_solve.postflop_model, &cache_dir, json);
            if !all_passed {
                std::process::exit(1);
            }
        }
        Commands::TraceHand { config } => {
            let yaml = std::fs::read_to_string(&config)?;
            let pf_solve: PostflopSolveConfig = serde_yaml::from_str(&yaml)?;
            let pf_config = pf_solve.postflop_model;

            eprintln!("Building postflop abstraction from scratch...");
            let abstraction = PostflopAbstraction::build(
                &pf_config, None, |phase| {
                    eprintln!("  {phase:?}");
                },
            )?;

            hand_trace::run_with_abstraction(&pf_config, &abstraction)?;
        }
        Commands::PrecomputeEquity { output } => {
            use poker_solver_core::preflop::postflop_hands::canonical_flops;

            let rank_cache_path = output.with_file_name("rank_arrays.bin");

            let total = 1755_u64;

            // Step 1: Build or load rank arrays
            let rank_cache = if let Some(cache) = RankArrayCache::load(&rank_cache_path) {
                // Early exit if equity tables also exist already
                if output.exists() {
                    eprintln!("Both caches already exist, nothing to do");
                    eprintln!("  rank arrays:   {}", rank_cache_path.display());
                    eprintln!("  equity tables: {}", output.display());
                    return Ok(());
                }
                eprintln!(
                    "Loaded rank arrays for {} flops from cache",
                    cache.num_flops()
                );
                cache
            } else {
                let pb = ProgressBar::new(total);
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} flops ({eta})")
                        .unwrap()
                        .progress_chars("=>-"),
                );
                pb.enable_steady_tick(Duration::from_millis(200));
                eprintln!("Computing rank arrays for {total} canonical flops...");

                let flops = canonical_flops();
                let start = Instant::now();
                let completed = AtomicU32::new(0);

                let entries: Vec<_> = flops
                    .par_iter()
                    .map(|flop| {
                        let combo_map = build_combo_map(flop);
                        let data = compute_rank_arrays(&combo_map, *flop);
                        let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                        pb.set_position(u64::from(done));
                        data
                    })
                    .collect();

                pb.finish_with_message("done");
                let elapsed = start.elapsed();
                eprintln!(
                    "Computed rank arrays in {:.1}s",
                    elapsed.as_secs_f64()
                );

                let cache = RankArrayCache {
                    flops: flops.clone(),
                    entries,
                };
                if let Err(e) = cache.save(&rank_cache_path) {
                    eprintln!("Warning: failed to save rank cache: {e}");
                } else {
                    eprintln!("Rank cache saved to {}", rank_cache_path.display());
                }
                cache
            };

            // Step 2: Derive equity tables (skip if already cached)
            // Destructure so we can borrow entries for par_iter and move flops into from_parts.
            let RankArrayCache { flops: rc_flops, entries: rc_entries } = rank_cache;
            if output.exists() {
                eprintln!("Equity tables already exist at {}, skipping derivation", output.display());
            } else {
                eprintln!("Deriving equity tables from rank arrays...");
                let derive_start = Instant::now();
                let tables: Vec<Vec<f64>> = rc_flops
                    .par_iter()
                    .zip(rc_entries.par_iter())
                    .map(|(flop, entry)| {
                        let combo_map = build_combo_map(flop);
                        derive_equity_table(entry, &combo_map)
                    })
                    .collect();
                eprintln!(
                    "Derived {} equity tables in {:.1}s",
                    tables.len(),
                    derive_start.elapsed().as_secs_f64()
                );

                let eq_cache = EquityTableCache::from_parts(rc_flops, tables);
                eq_cache.save(&output)?;
                eprintln!("Saved equity tables to {}", output.display());
            }
        }
        Commands::DumpTree { config, dot, output, spr, pot_type } => {
            tree_dump::run(dot, output, config, spr, pot_type)?;
        }
        Commands::InspectPreflop { path, hands, history } => {
            run_inspect_preflop(&path, &hands, history.as_deref())?;
        }
        Commands::TraceRegrets { config, iterations, hands, history } => {
            run_trace_regrets(&config, iterations, &hands, &history)?;
        }
        Commands::DecomposeRegrets { config, iterations, hand, history } => {
            run_decompose_regrets(&config, iterations, &hand, &history)?;
        }
        Commands::DiagClusters { cluster_dir, audit, audit_boards } => {
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
        }
        Commands::TraceTerminals { config, hero, opp, history } => {
            run_trace_terminals(&config, &hero, &opp, &history)?;
        }
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
            eprintln!("  Actions: max_raises={}", bp_config.action_abstraction.max_raises);
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
                                        poker_solver_core::preflop::postflop_hands::parse_card(s)
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

                trainer.tui_active = true;

                let refresh_ms = tui_config.refresh_rate_ms;
                let refresh = Duration::from_millis(refresh_ms);
                let refresh_hz = 1000.0 / refresh_ms as f64;
                let tui_handle = blueprint_tui::run_blueprint_tui(
                    Arc::clone(&metrics),
                    scenarios,
                    tui_config.telemetry.clone(),
                    refresh,
                    refresh_hz,
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

            poker_solver_core::blueprint_v2::cluster_pipeline::run_clustering_pipeline(
                &bp_config.clustering,
                &output,
                |street, p| {
                    let pct = (p * 100.0) as u32;
                    if p < 0.8 {
                        eprint!("\r  [{street}] features {pct}%");
                    } else {
                        eprint!("\r  [{street}] k-means {pct}% ");
                    }
                    if p >= 1.0 {
                        eprintln!(" done");
                    }
                },
            )?;

            eprintln!("\nClustering complete. Files saved to {}", output.display());
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
// Preflop solver
// ---------------------------------------------------------------------------

/// YAML config file for preflop training. Contains both game config and training params.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PreflopTrainingConfig {
    #[serde(flatten)]
    pub game: PreflopConfig,
    #[serde(default = "default_iterations")]
    pub iterations: u64,
    #[serde(default = "default_equity_samples")]
    pub equity_samples: u32,
    #[serde(default = "default_print_every")]
    pub print_every: u64,
    /// Preflop exploitability stopping threshold in mBB/hand.
    /// Training stops when exploitability drops below this value.
    #[serde(default = "default_preflop_exploitability_threshold_mbb", alias = "convergence_threshold_mbb")]
    pub preflop_exploitability_threshold_mbb: f64,
    /// Save a checkpoint bundle every N iterations during training.
    /// When `None`, no intermediate checkpoints are saved.
    #[serde(default)]
    pub checkpoint_every: Option<u64>,
    /// Path to a pre-built postflop bundle directory.
    /// Build one with `solve-postflop` first, then reference it here.
    pub postflop_model_path: PathBuf,
    /// Comma-separated canonical hands for postflop EV diagnostics.
    /// When set, prints avg EV per hand, pairwise matchups, and per-flop
    /// breakdown after building the postflop abstraction.
    /// Example: `"AA,KK,AKs,72o,65s"`
    #[serde(default)]
    pub ev_diagnostic_hands: Option<String>,
}

/// Lightweight config wrapper for `solve-postflop` — avoids requiring
/// the full `PreflopTrainingConfig` (positions, blinds, stacks, etc.).
#[derive(Debug, Deserialize)]
pub(crate) struct PostflopSolveConfig {
    postflop_model: PostflopModelConfig,
}

fn default_iterations() -> u64 { 5000 }
fn default_equity_samples() -> u32 { 20000 }
fn default_print_every() -> u64 { 1000 }
fn default_preflop_exploitability_threshold_mbb() -> f64 { 25.0 }

/// Parse a comma-separated list of canonical hands (e.g. "AA,KK,AKs,72o")
/// into a deduplicated, ordered list of `(label, canonical_index)` pairs.
fn parse_ev_diagnostic_hands(s: &str) -> Vec<(String, usize)> {
    use poker_solver_core::hands::CanonicalHand;
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();
    for token in s.split(',') {
        let token = token.trim();
        if token.is_empty() { continue; }
        if let Ok(ch) = CanonicalHand::parse(token) {
            if seen.insert(ch.index()) {
                result.push((ch.to_string(), ch.index()));
            }
        } else {
            eprintln!("warning: ignoring unrecognised hand '{token}'");
        }
    }
    result
}

/// Print postflop EV diagnostics for the given hands.
fn print_postflop_ev_diagnostics(
    abstraction: &PostflopAbstraction,
    hands: &[(String, usize)],
) {
    solver_print!("\n=== Postflop EV Table (pot-fraction units) ===");
    solver_print!("{:>5} {:>10} {:>10}", "Hand", "IP(SB)EV", "OOP(BB)EV");
    for (name, hi) in hands {
        let mut ip_sum = 0.0f64;
        let mut oop_sum = 0.0f64;
        let mut count = 0;
        for opp in 0..169 {
            let ev_ip = abstraction.avg_ev(0, *hi, opp);
            let ev_oop = abstraction.avg_ev(1, *hi, opp);
            if ev_ip != 0.0 || ev_oop != 0.0 {
                ip_sum += ev_ip;
                oop_sum += ev_oop;
                count += 1;
            }
        }
        if count > 0 {
            solver_print!(
                "{:>5} {:>10.4} {:>10.4}  (avg over {count} opps)",
                name,
                ip_sum / count as f64,
                oop_sum / count as f64,
            );
        } else {
            solver_print!("{:>5}  no data", name);
        }
    }

    // Coverage
    let mut nonzero_ip = 0usize;
    let mut nonzero_oop = 0usize;
    for h in 0..169 {
        for o in 0..169 {
            if abstraction.avg_ev(0, h, o) != 0.0 { nonzero_ip += 1; }
            if abstraction.avg_ev(1, h, o) != 0.0 { nonzero_oop += 1; }
        }
    }
    let total = 169 * 169;
    solver_print!("  Coverage (avg): IP {nonzero_ip}/{total} ({:.1}%)  OOP {nonzero_oop}/{total} ({:.1}%)",
        100.0 * nonzero_ip as f64 / total as f64,
        100.0 * nonzero_oop as f64 / total as f64,
    );

    // Per-flop coverage
    for fi in 0..abstraction.values.num_flops() {
        let mut nz = 0usize;
        for h in 0..169u16 {
            for o in 0..169u16 {
                if abstraction.values.get_by_flop(fi, 0, h, o) != 0.0 { nz += 1; }
            }
        }
        let flop_name = if fi < abstraction.flops.len() {
            format!("{:?}", abstraction.flops[fi])
        } else {
            format!("flop {fi}")
        };
        solver_print!("  Flop {fi} ({flop_name}): {nz}/{total} ({:.1}%) non-zero",
            100.0 * nz as f64 / total as f64);
    }

    // Pairwise matchups
    if hands.len() >= 2 {
        solver_print!("\n=== Pairwise matchups (avg_ev) ===");
        for i in 0..hands.len() {
            for j in (i+1)..hands.len() {
                let (h_name, hi) = &hands[i];
                let (o_name, oi) = &hands[j];
                let ip = abstraction.avg_ev(0, *hi, *oi);
                let oop = abstraction.avg_ev(1, *hi, *oi);
                let mut line = format!("  {h_name:>4} vs {o_name:>4}: IP(SB)={ip:+.4}  OOP(BB)={oop:+.4}");
                for fi in 0..abstraction.values.num_flops() {
                    let fip = abstraction.values.get_by_flop(fi, 0, *hi as u16, *oi as u16);
                    let foop = abstraction.values.get_by_flop(fi, 1, *hi as u16, *oi as u16);
                    use std::fmt::Write as _;
                    let _ = write!(line, "  |f{fi}: IP={fip:+.3} OOP={foop:+.3}");
                }
                solver_print!("{line}");
            }
        }
    }
    solver_print!("");
}

// ---------------------------------------------------------------------------
// Shared postflop progress infrastructure
// ---------------------------------------------------------------------------

/// Build one or more `PostflopAbstraction`s (one per configured SPR) with
/// a full-screen TUI dashboard showing per-flop convergence, pruning stats,
/// and traversal throughput.
///
/// When stderr is not a TTY (piped output, CI, etc.), falls back to simple
/// line-based progress logging to avoid corrupting output with ANSI escape
/// sequences.
fn build_postflop_with_progress(
    pf_config: &PostflopModelConfig,
    equity: Option<&EquityTable>,
    tui_refresh: f64,
) -> Result<Vec<PostflopAbstraction>, Box<dyn Error>> {
    log_file::init_log_file();

    let sprs = &pf_config.postflop_sprs;
    let total_sprs = sprs.len() as u32;
    let use_tui = std::io::stderr().is_terminal();

    // Resolve the actual flop list so the TUI shows the correct count.
    let flops = if let Some(ref names) = pf_config.fixed_flops {
        parse_flops(names).map_err(|e| format!("invalid flops: {e}"))?
    } else {
        sample_canonical_flops(pf_config.max_flop_boards)
    };
    let estimated_flops = flops.len() as u32;
    let counters = Arc::new(SolverCounters::default());

    // TUI-mode resources: only allocated when stderr is a TTY.
    let metrics = if use_tui {
        Some(Arc::new(tui_metrics::TuiMetrics::new(total_sprs, estimated_flops)))
    } else {
        None
    };
    let done = if use_tui {
        Some(Arc::new(AtomicBool::new(false)))
    } else {
        None
    };
    let tui_handle = if let (Some(m), Some(d)) = (&metrics, &done) {
        log_file::TUI_ACTIVE.store(true, Ordering::Relaxed);
        Some(tui::run_tui(
            Arc::clone(m),
            Arc::clone(&counters),
            Duration::from_secs_f64(tui_refresh),
            Arc::clone(d),
        ))
    } else {
        None
    };

    let pf_start = Instant::now();
    let mut abstractions = Vec::with_capacity(sprs.len());

    // Try loading pre-computed equity tables from disk cache.
    // Falls back to inline computation if cache is missing/invalid.
    let cache_path = Path::new("cache/equity_tables.bin");
    let equity_tables: Vec<Vec<f64>> =
        if pf_config.solve_type == PostflopSolveType::Exhaustive {
            if let Some(cache) = EquityTableCache::load(cache_path) {
                match cache.extract_tables(&flops) {
                    Some(tables) => {
                        solver_log!(
                            "Loaded equity tables for {}/{} flops from cache",
                            tables.len(),
                            cache.num_flops(),
                        );
                        tables
                    }
                    None => {
                        solver_log!(
                            "Warning: cache missing some requested flops, falling back to inline computation"
                        );
                        vec![]
                    }
                }
            } else {
                // No equity table cache — try rank array cache
                let rank_cache_path = cache_path.with_file_name("rank_arrays.bin");
                if let Some(rank_cache) = RankArrayCache::load(&rank_cache_path) {
                    solver_log!(
                        "Deriving equity tables from rank cache ({} flops)...",
                        rank_cache.num_flops()
                    );
                    let tables: Vec<Vec<f64>> = flops
                        .par_iter()
                        .map(|flop| {
                            let combo_map = build_combo_map(flop);
                            if let Some(data) = rank_cache.get_flop_data(flop) {
                                derive_equity_table(data, &combo_map)
                            } else {
                                // Flop not in rank cache, compute directly
                                compute_equity_table(&combo_map, *flop)
                            }
                        })
                        .collect();
                    tables
                } else if sprs.len() > 1 {
                    // No cache at all — fall back to inline pre-computation for multi-SPR runs.
                    if let Some(m) = &metrics {
                        m.equity_tables_total.store(flops.len() as u32, Ordering::Relaxed);
                        m.set_phase(4); // equity table pre-computation phase
                    }
                    solver_log!(
                        "Pre-computing equity tables for {} flops...",
                        flops.len()
                    );
                    let eq_completed = AtomicU32::new(0);
                    let tables: Vec<Vec<f64>> = flops
                        .par_iter()
                        .map(|flop| {
                            let combo_map = build_combo_map(flop);
                            let table = compute_equity_table(&combo_map, *flop);
                            let done = eq_completed.fetch_add(1, Ordering::Relaxed) + 1;
                            if let Some(m) = &metrics {
                                m.equity_tables_completed.store(done, Ordering::Relaxed);
                            }
                            table
                        })
                        .collect();
                    if let Some(m) = &metrics {
                        m.set_phase(0); // back to idle before SPR loop
                    }
                    tables
                } else {
                    vec![]
                }
            }
        } else {
            vec![]
        };
    let pre_tables = if equity_tables.is_empty() {
        None
    } else {
        Some(equity_tables.as_slice())
    };

    for (i, &spr) in sprs.iter().enumerate() {
        if let Some(m) = &metrics {
            m.start_spr(i as u32, estimated_flops);
        }
        // Reset solver counters for each SPR so TUI shows per-SPR rates.
        counters.traversal_count.store(0, Ordering::Relaxed);
        counters.pruned_traversal_count.store(0, Ordering::Relaxed);
        counters.total_action_slots.store(0, Ordering::Relaxed);
        counters.pruned_action_slots.store(0, Ordering::Relaxed);
        counters.total_expected_traversals.store(0, Ordering::Relaxed);
        solver_log!("SPR={spr} ({}/{total_sprs})", i + 1);

        let metrics_ref = &metrics;

        let abstraction = PostflopAbstraction::build_for_spr(
            pf_config,
            spr,
            equity,
            pre_tables,
            Some(&*counters),
            |phase| {
                if let Some(m) = metrics_ref {
                    match &phase {
                        BuildPhase::BuildingTree => {
                            m.set_phase(1);
                        }
                        BuildPhase::ComputingEquityTables => {}
                        BuildPhase::FlopProgress { flop_name, stage } => {
                            // Transition to solving phase on first flop progress.
                            if m.spr_phase.load(Ordering::Relaxed) < 2 {
                                m.set_phase(2);
                            }
                            match stage {
                                FlopStage::Solving { iteration, max_iterations, delta, total_action_slots, pruned_action_slots, max_positive_regret, min_negative_regret, .. } => {
                                    let pct_act = if *total_action_slots > 0 {
                                        *pruned_action_slots as f64 * 100.0 / *total_action_slots as f64
                                    } else {
                                        0.0
                                    };
                                    m.update_flop(flop_name, *iteration, *max_iterations, *delta, pct_act, *max_positive_regret, *min_negative_regret);
                                }
                                FlopStage::EstimatingEv { .. } => {}
                                FlopStage::Done => {
                                    m.remove_flop(flop_name);
                                }
                            }
                        }
                        BuildPhase::MccfrFlopsCompleted { completed, total } => {
                            m.flops_completed.store(*completed as u32, Ordering::Relaxed);
                            m.total_flops.store(*total as u32, Ordering::Relaxed);
                        }
                        BuildPhase::ComputingValues => {
                            m.set_phase(3);
                        }
                    }
                } else {
                    // Non-TTY fallback: line-based progress to log + stderr.
                    match &phase {
                        BuildPhase::BuildingTree => {
                            solver_log!("  Building game tree...");
                        }
                        BuildPhase::ComputingEquityTables => {
                            solver_log!("  Computing equity tables...");
                        }
                        BuildPhase::FlopProgress { flop_name, stage } => {
                            if let FlopStage::Solving { iteration, max_iterations, delta, metric_label, .. } = stage {
                                // Log every ~10 iterations to avoid flooding.
                                if *iteration % 10 == 0 || *iteration == *max_iterations {
                                    solver_log!(
                                        "    {flop_name}: iter {iteration}/{max_iterations} {metric_label}={delta:.2}",
                                    );
                                }
                            }
                        }
                        BuildPhase::MccfrFlopsCompleted { completed, total } => {
                            solver_log!("  SPR={spr}: {completed}/{total} flops completed");
                        }
                        BuildPhase::ComputingValues => {
                            solver_log!("  Computing hand average values...");
                        }
                    }
                }
            },
        )
        .map_err(|e| format!("postflop SPR={spr}: {e}"))?;

        abstractions.push(abstraction);
    }

    // Signal the TUI to exit and wait for it to restore the terminal.
    if let Some(d) = &done {
        d.store(true, Ordering::Relaxed);
    }
    if let Some(handle) = tui_handle {
        let _ = handle.join();
    }
    log_file::TUI_ACTIVE.store(false, Ordering::Relaxed);

    solver_log!(
        "Postflop solve complete in {:.1?} ({total_sprs} SPR model{})",
        pf_start.elapsed(),
        if total_sprs == 1 { "" } else { "s" },
    );

    Ok(abstractions)
}

// ---------------------------------------------------------------------------
// Postflop bundle builder
// ---------------------------------------------------------------------------

fn run_solve_postflop(config_path: &Path, output: &Path, tui_refresh: f64) -> Result<(), Box<dyn Error>> {
    log_file::init_log_file();
    log_file::install_panic_hook();

    let yaml = std::fs::read_to_string(config_path)?;
    let config: PostflopSolveConfig = serde_yaml::from_str(&yaml)?;
    let pf_config = config.postflop_model;

    solver_log!("Building postflop abstraction ({} SPR models)...", pf_config.postflop_sprs.len());

    let abstractions = build_postflop_with_progress(&pf_config, None, tui_refresh)?;

    let refs: Vec<&PostflopAbstraction> = abstractions.iter().collect();
    PostflopBundle::save_multi(&pf_config, &refs, output)?;
    solver_log!(
        "Postflop bundle saved to {} ({} SPR models)",
        output.display(),
        refs.len(),
    );

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_solve_preflop(
    config_path: &Path,
    cli_iterations: Option<u64>,
    output: &Path,
    cli_print_every: Option<u64>,
    cli_equity_samples: Option<u32>,
    claude_debug: bool,
) -> Result<(), Box<dyn Error>> {
    let yaml = std::fs::read_to_string(config_path)?;
    let mut training: PreflopTrainingConfig = serde_yaml::from_str(&yaml)?;

    // CLI overrides
    if let Some(v) = cli_iterations { training.iterations = v; }
    if let Some(v) = cli_equity_samples { training.equity_samples = v; }
    if let Some(v) = cli_print_every { training.print_every = v; }

    let postflop_model_path = training.postflop_model_path;
    let config = training.game;
    let iterations = training.iterations;
    let equity_samples = training.equity_samples;
    let print_every = training.print_every;
    let exploitability_threshold_mbb = training.preflop_exploitability_threshold_mbb;
    let checkpoint_every = training.checkpoint_every;
    let players = config.positions.len();

    let cache_base = std::path::Path::new("cache/postflop");

    let equity = if equity_samples > 0 {
        use poker_solver_core::preflop::equity_cache;

        if let Some(cached) = equity_cache::load(cache_base, equity_samples) {
            eprintln!(
                "Equity cache hit: {}",
                equity_cache::cache_dir(cache_base, equity_samples).display()
            );
            cached
        } else {
            let total_pairs = 169 * 168 / 2;
            let eq_pb = ProgressBar::new(total_pairs as u64);
            eq_pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} Computing equities [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                    .expect("valid template")
                    .progress_chars("#>-"),
            );
            let eq_start = Instant::now();
            let table = EquityTable::new_computed(equity_samples, |done| {
                eq_pb.set_position(done as u64);
            });
            eq_pb.finish_with_message("done");
            println!("Equity table built in {:.1?}", eq_start.elapsed());

            if let Err(e) = equity_cache::save(cache_base, equity_samples, &table) {
                eprintln!("Warning: failed to save equity cache: {e}");
            } else {
                eprintln!(
                    "Equity cache saved: {}",
                    equity_cache::cache_dir(cache_base, equity_samples).display()
                );
            }

            lhe_viz::print_equity_matrix(&table);
            table
        }
    } else {
        println!("Using uniform equities (--equity-samples 0)");
        EquityTable::new_uniform()
    };

    println!("Solving preflop: {players}p, {iterations} iterations");
    let start = Instant::now();

    let tree = PreflopTree::build(&config);
    let bb_node = lhe_viz::find_raise_child(&tree, 0);
    let bb_call_node = lhe_viz::find_call_child(&tree, 0);

    // Load multi-SPR postflop abstraction(s) from pre-built bundle.
    eprintln!("Loading postflop bundle from {}", postflop_model_path.display());
    let pf_start = Instant::now();
    let pf_config_yaml = std::fs::read_to_string(postflop_model_path.join("config.yaml"))
        .map_err(|e| format!("failed to read postflop config: {e}"))?;
    let pf_config: PostflopModelConfig = serde_yaml::from_str(&pf_config_yaml)
        .map_err(|e| format!("failed to parse postflop config: {e}"))?;
    let abstractions = PostflopBundle::load_multi(&pf_config, &postflop_model_path)
        .map_err(|e| format!("failed to load postflop bundle: {e}"))?;
    eprintln!(
        "Postflop bundle loaded in {:.1?} ({} SPR models)",
        pf_start.elapsed(),
        abstractions.len(),
    );
    let postflop = Some(abstractions);

    let mut solver = PreflopSolver::new_with_equity(&config, equity);
    // Parse ev_diagnostic_hands once for use in diagnostics and per-iteration output.
    let ev_diagnostic_hands: Vec<(String, usize)> = training.ev_diagnostic_hands.as_deref()
        .map(parse_ev_diagnostic_hands)
        .unwrap_or_default();

    // Clone hand_avg_values from first model before abstractions are consumed.
    let hand_avg_values = postflop.as_ref()
        .and_then(|abs| abs.first())
        .map(|a| a.hand_avg_values.clone());

    if let Some(ref abstractions) = postflop
        && !ev_diagnostic_hands.is_empty()
        && let Some(first) = abstractions.first()
    {
        print_postflop_ev_diagnostics(first, &ev_diagnostic_hands);
    }
    if let Some(abstractions) = postflop {
        solver.attach_postflop(abstractions, &config);
    }

    print_preflop_matrices(&solver.strategy(), &tree, bb_node, bb_call_node, 0, claude_debug);

    let pb = ProgressBar::new(iterations);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .expect("valid template")
            .progress_chars("#>-"),
    );

    let chunk = if print_every > 0 { print_every } else { std::cmp::max(iterations / 100, 1) };
    let mut done = 0u64;
    let mut converged_early = false;
    let mut expl_history: Vec<f64> = Vec::new();
    let mut prev_strategy: Option<FxHashMap<u64, Vec<f64>>> = None;
    while done < iterations {
        let batch = std::cmp::min(chunk, iterations - done);
        solver.train(batch);
        done += batch;
        pb.set_position(done);

        if print_every > 0 && done < iterations && done.is_multiple_of(print_every) {
            let mut early_stop = false;
            pb.suspend(|| {
                let strat = solver.strategy();
                print_preflop_matrices(&strat, &tree, bb_node, bb_call_node, done, claude_debug);
                let strat_map = strat.into_inner();
                let delta = prev_strategy.as_ref()
                    .map_or(0.0, |prev| strategy_delta(prev, &strat_map));
                let expl = solver.exploitability();
                let mbb = expl * 500.0;
                expl_history.push(mbb);
                println!("  Strategy \u{03b4}: {delta:.6}  Exploitability: {expl:.6} (mBB/hand: {mbb:.2})");
                prev_strategy = Some(strat_map);
                print_regret_sparkline(&expl_history, 10);
                if mbb < exploitability_threshold_mbb {
                    println!("Exploitability {mbb:.2} mBB/hand < {exploitability_threshold_mbb} — stopping early at iteration {done}");
                    early_stop = true;
                }
            });
            if early_stop {
                converged_early = true;
                break;
            }
        }

        // Periodic checkpoint save
        if let Some(interval) = checkpoint_every
            && interval > 0 && done.is_multiple_of(interval)
        {
            pb.suspend(|| {
                let cp_dir = output.join(format!("checkpoint_{done}"));
                let bundle = PreflopBundle::new(config.clone(), solver.strategy());
                match bundle.save(&cp_dir) {
                    Ok(()) => println!("  Saved checkpoint to {}/", cp_dir.display()),
                    Err(e) => eprintln!("  Warning: failed to save checkpoint: {e}"),
                }
            });
        }
    }
    pb.finish_with_message("done");

    let elapsed = start.elapsed();
    let strategy = solver.strategy();
    let label = if converged_early { "Converged" } else { "Finished" };
    println!("{label} in {elapsed:.1?} — {done} iterations, {} info sets", strategy.len());

    print_preflop_matrices(&strategy, &tree, bb_node, bb_call_node, done, claude_debug);

    let expl = solver.exploitability();
    let mbb = expl * 500.0;
    println!("  Final exploitability: {expl:.6} (mBB/hand: {mbb:.2})");

    let bundle = PreflopBundle::with_ev_table(config, strategy, hand_avg_values);
    bundle.save(output)?;
    println!("Saved to {}", output.display());

    Ok(())
}

// ---------------------------------------------------------------------------
// Strategy convergence utility
// ---------------------------------------------------------------------------

/// Mean L1 distance between two strategy maps.
///
/// For each info set present in both maps, computes the sum of absolute
/// differences between corresponding action probabilities, then averages
/// across all shared info sets. Returns 0.0 if no info sets are shared.
#[allow(clippy::implicit_hasher)]
fn strategy_delta(prev: &FxHashMap<u64, Vec<f64>>, curr: &FxHashMap<u64, Vec<f64>>) -> f64 {
    let mut total_delta = 0.0;
    let mut count = 0u64;

    for (key, prev_probs) in prev {
        if let Some(curr_probs) = curr.get(key) {
            let l1: f64 = prev_probs
                .iter()
                .zip(curr_probs.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            total_delta += l1;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        {
            total_delta / count as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Preflop display helpers
// ---------------------------------------------------------------------------

/// Print a small terminal chart of exploitability history (y=mBB/hand, x=iteration).
///
/// Uses Unicode braille characters for sub-cell resolution (2×4 dots per cell),
/// giving effective resolution of `width*2 × height*4` pixels.
fn print_regret_sparkline(history: &[f64], height: usize) {
    if history.len() < 2 {
        return;
    }
    let window = if history.len() > 100 { &history[history.len() - 100..] } else { history };
    let width = 60usize.min(window.len());
    // Resample window to `width * 2` points (braille has 2 columns per cell).
    let cols = width * 2;
    let rows = height * 4; // braille has 4 rows per cell
    let n = window.len();
    let mut samples = Vec::with_capacity(cols);
    for i in 0..cols {
        let idx_f = i as f64 * (n - 1) as f64 / (cols - 1) as f64;
        let lo = idx_f as usize;
        let hi = (lo + 1).min(n - 1);
        let frac = idx_f - lo as f64;
        samples.push(window[lo] * (1.0 - frac) + window[hi] * frac);
    }

    let y_max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_range = if (y_max - y_min).abs() < 1e-12 { 1.0 } else { y_max - y_min };

    // Map each sample to a row (0 = bottom, rows-1 = top).
    let mapped: Vec<usize> = samples
        .iter()
        .map(|&v| {
            let norm = (v - y_min) / y_range;
            ((norm * (rows - 1) as f64).round() as usize).min(rows - 1)
        })
        .collect();

    // Braille base: U+2800. Dot positions in a 2×4 cell:
    //   col0: rows 0-3 → bits 0,1,2,6
    //   col1: rows 0-3 → bits 3,4,5,7
    const DOT_MAP: [[u32; 4]; 2] = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]];

    // Build grid of braille cells.
    let mut grid = vec![vec![0u32; width]; height];
    for (col_idx, &row_val) in mapped.iter().enumerate() {
        let cell_x = col_idx / 2;
        let dot_x = col_idx % 2;
        // row_val is from bottom; convert to cell coords.
        let cell_y = height - 1 - row_val / 4;
        let dot_y = 3 - (row_val % 4);
        grid[cell_y][cell_x] |= DOT_MAP[dot_x][dot_y];
    }

    // Print with y-axis labels.
    for (row, cells) in grid.iter().enumerate() {
        let label = if row == 0 {
            format!("{y_max:.1}")
        } else if row == height - 1 {
            format!("{y_min:.1}")
        } else {
            String::new()
        };
        let braille: String = cells
            .iter()
            .map(|&bits| char::from_u32(0x2800 + bits).unwrap_or(' '))
            .collect();
        println!("  {label:>10} │{braille}");
    }
    // x-axis line
    println!("  {:>10} └{}", "", "─".repeat(width));
}

fn preflop_node_actions(tree: &PreflopTree, idx: u32) -> Option<&[PreflopAction]> {
    match &tree.nodes[idx as usize] {
        PreflopNode::Decision { action_labels, .. } => Some(action_labels),
        PreflopNode::Terminal { .. } => None,
    }
}

fn print_preflop_matrices(
    strategy: &poker_solver_core::preflop::PreflopStrategy,
    tree: &PreflopTree,
    bb_raise_node: Option<u32>,
    bb_call_node: Option<u32>,
    iteration: u64,
    plain: bool,
) {
    let print = if plain { lhe_viz::print_hand_matrix_plain } else { lhe_viz::print_hand_matrix };

    let sb_matrix = lhe_viz::preflop_strategy_matrix(strategy, tree, 0);
    let sb_actions = preflop_node_actions(tree, 0);
    print(&sb_matrix, &format!("SB RFI — iteration {iteration}"), sb_actions);

    if let Some(bb_idx) = bb_call_node {
        let bb_matrix = lhe_viz::preflop_strategy_matrix(strategy, tree, bb_idx);
        let bb_actions = preflop_node_actions(tree, bb_idx);
        print(&bb_matrix, &format!("BB vs SB Call — iteration {iteration}"), bb_actions);
    }

    if let Some(bb_idx) = bb_raise_node {
        let bb_matrix = lhe_viz::preflop_strategy_matrix(strategy, tree, bb_idx);
        let bb_actions = preflop_node_actions(tree, bb_idx);
        print(&bb_matrix, &format!("BB vs Raise — iteration {iteration}"), bb_actions);
    }
}

// ---------------------------------------------------------------------------
// Flops command
// ---------------------------------------------------------------------------

fn run_flops(format: OutputFormat, output: Option<PathBuf>) -> Result<(), Box<dyn Error>> {
    let flops = flops::all_flops();

    let mut writer: Box<dyn Write> = match &output {
        Some(path) => Box::new(std::fs::File::create(path)?),
        None => Box::new(std::io::stdout().lock()),
    };

    match format {
        OutputFormat::Json => write_flops_json(&flops, &mut writer)?,
        OutputFormat::Csv => write_flops_csv(&flops, &mut writer)?,
    }

    if let Some(path) = &output {
        eprintln!("Wrote {} flops to {}", flops.len(), path.display());
    }

    Ok(())
}

fn flop_to_card_strings(flop: &CanonicalFlop) -> [String; 3] {
    let cards = flop.cards();
    [
        cards[0].to_string(),
        cards[1].to_string(),
        cards[2].to_string(),
    ]
}

fn suit_texture_str(flop: &CanonicalFlop) -> &'static str {
    match flop.suit_texture() {
        SuitTexture::Rainbow => "rainbow",
        SuitTexture::TwoTone => "two_tone",
        SuitTexture::Monotone => "monotone",
    }
}

fn rank_texture_str(flop: &CanonicalFlop) -> &'static str {
    match flop.rank_texture() {
        RankTexture::Unpaired => "unpaired",
        RankTexture::Paired => "paired",
        RankTexture::Trips => "trips",
    }
}

fn high_card_class_str(flop: &CanonicalFlop) -> &'static str {
    match flop.high_card_class() {
        flops::HighCardClass::Broadway => "broadway",
        flops::HighCardClass::Middle => "middle",
        flops::HighCardClass::Low => "low",
    }
}

fn write_flops_json(flops: &[CanonicalFlop], writer: &mut dyn Write) -> Result<(), Box<dyn Error>> {
    let entries: Vec<serde_json::Value> = flops
        .iter()
        .map(|f| {
            let cards = flop_to_card_strings(f);
            let conn = f.connectedness();
            serde_json::json!({
                "cards": cards,
                "suit_texture": suit_texture_str(f),
                "rank_texture": rank_texture_str(f),
                "high_card_class": high_card_class_str(f),
                "gap_high_mid": conn.gap_high_mid,
                "gap_mid_low": conn.gap_mid_low,
                "has_straight_potential": conn.has_straight_potential,
                "weight": f.weight(),
            })
        })
        .collect();

    serde_json::to_writer_pretty(writer, &entries)?;
    Ok(())
}

fn write_flops_csv(flops: &[CanonicalFlop], writer: &mut dyn Write) -> Result<(), Box<dyn Error>> {
    writeln!(
        writer,
        "card1,card2,card3,suit_texture,rank_texture,high_card_class,gap_high_mid,gap_mid_low,has_straight_potential,weight"
    )?;
    for f in flops {
        let cards = flop_to_card_strings(f);
        let conn = f.connectedness();
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{}",
            cards[0],
            cards[1],
            cards[2],
            suit_texture_str(f),
            rank_texture_str(f),
            high_card_class_str(f),
            conn.gap_high_mid,
            conn.gap_mid_low,
            conn.has_straight_potential,
            f.weight(),
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Inspect preflop bundle
// ---------------------------------------------------------------------------

fn run_inspect_preflop(
    path: &Path,
    hands_str: &str,
    history: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    use poker_solver_core::info_key::canonical_hand_index_from_str;
    use poker_solver_core::preflop::{PreflopNode, PreflopTree};

    let bundle = PreflopBundle::load(path)?;
    let tree = PreflopTree::build(&bundle.config);

    // Resolve hand names to indices
    let hand_names: Vec<&str> = hands_str.split(',').map(str::trim).collect();
    let hand_indices: Vec<(usize, &str)> = hand_names
        .iter()
        .filter_map(|&name| {
            canonical_hand_index_from_str(name).map(|idx| (idx as usize, name))
        })
        .collect();

    // Determine which nodes to inspect
    let nodes_to_inspect = if let Some(hist) = history {
        // Walk tree following action history like "r:3" or "r:3,call"
        let actions: Vec<&str> = hist.split(',').map(str::trim).collect();
        let mut node_idx = 0u32;
        for action_str in &actions {
            node_idx = find_child_by_label(&tree, node_idx, action_str)
                .ok_or_else(|| format!("Action '{}' not found at node {}", action_str, node_idx))?;
        }
        vec![(node_idx, format!("after {}", hist))]
    } else {
        // Default: root (SB decision) + BB-vs-raise nodes
        let mut nodes = vec![(0u32, "SB root".to_string())];
        // Find all raise children from root (BB facing different raise sizes)
        if let PreflopNode::Decision { action_labels, children, .. } = &tree.nodes[0] {
            for (i, action) in action_labels.iter().enumerate() {
                if matches!(action, PreflopAction::Raise(_) | PreflopAction::AllIn) {
                    nodes.push((children[i], format!("BB vs {action:?}")));
                }
            }
            // Also add BB-vs-limp (call child)
            for (i, action) in action_labels.iter().enumerate() {
                if matches!(action, PreflopAction::Call) {
                    nodes.push((children[i], "BB vs limp".to_string()));
                }
            }
        }
        nodes
    };

    // Print strategy at each node
    for (node_idx, label) in &nodes_to_inspect {
        let (action_labels, _) = match &tree.nodes[*node_idx as usize] {
            PreflopNode::Decision { action_labels, children, position, .. } => {
                println!("\n=== Node {} ({}) — position {} ===", node_idx, label, position);
                (action_labels.clone(), children.clone())
            }
            PreflopNode::Terminal { .. } => {
                println!("\n=== Node {} ({}) — TERMINAL ===", node_idx, label);
                continue;
            }
        };

        // Header
        let action_strs: Vec<String> = action_labels.iter().map(|a| format!("{a:?}")).collect();
        print!("{:>5}", "Hand");
        for a in &action_strs {
            print!("  {:>8}", a);
        }
        println!();

        // Per-hand strategy
        for &(hand_idx, name) in &hand_indices {
            let probs = bundle.strategy.get_probs(*node_idx, hand_idx);
            print!("{:>5}", name);
            if probs.is_empty() {
                println!("  (no data)");
            } else {
                for &p in probs {
                    print!("  {:>7.1}%", p * 100.0);
                }
                println!();
            }
        }
    }

    Ok(())
}

/// Find a child node by action label string (e.g. "r:3", "call", "fold", "allin").
fn find_child_by_label(
    tree: &PreflopTree,
    node_idx: u32,
    action_str: &str,
) -> Option<u32> {
    use poker_solver_core::preflop::PreflopNode;
    match &tree.nodes[node_idx as usize] {
        PreflopNode::Decision { action_labels, children, .. } => {
            for (i, action) in action_labels.iter().enumerate() {
                let matches = match action_str.to_lowercase().as_str() {
                    "fold" => matches!(action, PreflopAction::Fold),
                    "call" => matches!(action, PreflopAction::Call),
                    "allin" | "all-in" => matches!(action, PreflopAction::AllIn),
                    s if s.starts_with("r:") => {
                        // Match by action index: "r:3" means 3rd action (0-indexed)
                        if let Ok(idx) = s[2..].parse::<usize>() {
                            i == idx
                        } else {
                            false
                        }
                    }
                    _ => false,
                };
                if matches {
                    return Some(children[i]);
                }
            }
            None
        }
        PreflopNode::Terminal { .. } => None,
    }
}

// ---------------------------------------------------------------------------
// Trace regret / strategy-sum evolution
// ---------------------------------------------------------------------------

fn run_trace_regrets(
    config_path: &Path,
    iterations: u64,
    hands_str: &str,
    history: &str,
) -> Result<(), Box<dyn Error>> {
    use poker_solver_core::info_key::canonical_hand_index_from_str;
    use poker_solver_core::preflop::{PreflopNode, PreflopTree};

    let yaml = std::fs::read_to_string(config_path)?;
    let training: PreflopTrainingConfig = serde_yaml::from_str(&yaml)?;

    let postflop_model_path = training.postflop_model_path;
    let config = training.game;

    // Load equity
    let cache_base = Path::new("cache/postflop");
    let equity = if training.equity_samples > 0 {
        use poker_solver_core::preflop::equity_cache;
        equity_cache::load(cache_base, training.equity_samples)
            .ok_or("Equity cache not found — run solve-preflop first to build it")?
    } else {
        EquityTable::new_uniform()
    };

    // Build tree + solver
    let tree = PreflopTree::build(&config);
    let mut solver = PreflopSolver::new_with_equity(&config, equity);

    // Load postflop model
    let pf_config_yaml = std::fs::read_to_string(postflop_model_path.join("config.yaml"))?;
    let pf_config: PostflopModelConfig = serde_yaml::from_str(&pf_config_yaml)?;
    let abstractions = PostflopBundle::load_multi(&pf_config, &postflop_model_path)?;
    solver.attach_postflop(abstractions, &config);

    // Resolve target node
    let actions: Vec<&str> = history.split(',').map(str::trim).collect();
    let mut node_idx = 0u32;
    for action_str in &actions {
        node_idx = find_child_by_label(&tree, node_idx, action_str)
            .ok_or_else(|| format!("Action '{}' not found at node {}", action_str, node_idx))?;
    }

    // Get action labels at target node
    let action_labels = match &tree.nodes[node_idx as usize] {
        PreflopNode::Decision { action_labels, .. } => action_labels.clone(),
        PreflopNode::Terminal { .. } => return Err("Target node is terminal".into()),
    };
    let action_strs: Vec<String> = action_labels.iter().map(|a| format!("{a:?}")).collect();

    // Resolve hand names
    let hand_entries: Vec<(usize, &str)> = hands_str
        .split(',')
        .filter_map(|name| {
            let name = name.trim();
            canonical_hand_index_from_str(name).map(|idx| (idx as usize, name))
        })
        .collect();

    println!("Tracing node {} (after {}) — {} actions: {:?}",
        node_idx, history, action_strs.len(), action_strs);
    println!();

    // Run iteration by iteration
    for iter in 0..iterations {
        solver.train(1);

        println!("=== After iteration {} ===", iter + 1);
        for &(hand_idx, name) in &hand_entries {
            let regrets = solver.regret_at(node_idx, hand_idx);
            let strat_sums = solver.strategy_sum_at(node_idx, hand_idx);

            // Compute current strategy from regret matching
            let total_pos: f64 = regrets.iter().filter(|&&r| r > 0.0).sum();
            let current_strat: Vec<f64> = if total_pos > 0.0 {
                regrets.iter().map(|&r| if r > 0.0 { r / total_pos } else { 0.0 }).collect()
            } else {
                vec![1.0 / regrets.len() as f64; regrets.len()]
            };

            // Compute average strategy from strategy sums
            let total_ss: f64 = strat_sums.iter().sum();
            let avg_strat: Vec<f64> = if total_ss > 0.0 {
                strat_sums.iter().map(|&s| s / total_ss).collect()
            } else {
                vec![1.0 / strat_sums.len() as f64; strat_sums.len()]
            };

            print!("  {:<5} regret_sum: [", name);
            for (i, r) in regrets.iter().enumerate() {
                if i > 0 { print!(", "); }
                print!("{:>10.2}", r);
            }
            println!("]");

            print!("  {:<5} strat_sum:  [", "");
            for (i, s) in strat_sums.iter().enumerate() {
                if i > 0 { print!(", "); }
                print!("{:>10.2}", s);
            }
            println!("]");

            print!("  {:<5} curr_strat: [", "");
            for (i, s) in current_strat.iter().enumerate() {
                if i > 0 { print!(", "); }
                print!("{:>10.1}%", s * 100.0);
            }
            println!("]");

            print!("  {:<5} avg_strat:  [", "");
            for (i, s) in avg_strat.iter().enumerate() {
                if i > 0 { print!(", "); }
                print!("{:>10.1}%", s * 100.0);
            }
            println!("]");
            println!();
        }
    }

    println!("Actions: {:?}", action_strs);

    Ok(())
}

fn run_decompose_regrets(
    config_path: &Path,
    iterations: u64,
    hand_str: &str,
    history: &str,
) -> Result<(), Box<dyn Error>> {
    use poker_solver_core::hands::CanonicalHand;
    use poker_solver_core::info_key::canonical_hand_index_from_str;
    use poker_solver_core::preflop::{PreflopNode, PreflopTree};

    let yaml = std::fs::read_to_string(config_path)?;
    let training: PreflopTrainingConfig = serde_yaml::from_str(&yaml)?;

    let postflop_model_path = training.postflop_model_path;
    let config = training.game;

    // Load equity
    let cache_base = Path::new("cache/postflop");
    let equity = if training.equity_samples > 0 {
        use poker_solver_core::preflop::equity_cache;
        equity_cache::load(cache_base, training.equity_samples)
            .ok_or("Equity cache not found — run solve-preflop first to build it")?
    } else {
        EquityTable::new_uniform()
    };

    // Build tree + solver
    let tree = PreflopTree::build(&config);
    let mut solver = PreflopSolver::new_with_equity(&config, equity);

    // Load postflop model
    let pf_config_yaml = std::fs::read_to_string(postflop_model_path.join("config.yaml"))?;
    let pf_config: PostflopModelConfig = serde_yaml::from_str(&pf_config_yaml)?;
    let abstractions = PostflopBundle::load_multi(&pf_config, &postflop_model_path)?;
    solver.attach_postflop(abstractions, &config);

    // Resolve hero hand
    let hero_hand = canonical_hand_index_from_str(hand_str)
        .ok_or_else(|| format!("Invalid hand: {hand_str}"))?;

    // Resolve target node
    let actions: Vec<&str> = history.split(',').map(str::trim).collect();
    let mut node_idx = 0u32;
    for action_str in &actions {
        node_idx = find_child_by_label(&tree, node_idx, action_str)
            .ok_or_else(|| format!("Action '{}' not found at node {}", action_str, node_idx))?;
    }

    // Get action labels and hero position at target node
    let (hero_pos, action_labels) = match &tree.nodes[node_idx as usize] {
        PreflopNode::Decision { position, action_labels, .. } => (*position, action_labels.clone()),
        PreflopNode::Terminal { .. } => return Err("Target node is terminal".into()),
    };
    let action_strs: Vec<String> = action_labels.iter().map(|a| format!("{a:?}")).collect();

    println!("Decomposing regrets for {} (index {}) at node {} (hero_pos={})",
        hand_str, hero_hand, node_idx, hero_pos);
    println!("Actions: {:?}", action_strs);
    println!("Running {} warm-up iterations...", iterations);

    // Run warm-up iterations
    solver.train(iterations);

    // Show aggregate regrets first
    let regrets = solver.regret_at(node_idx, hero_hand as usize);
    println!("\nAggregate regret_sum after {} iterations:", iterations);
    for (i, label) in action_strs.iter().enumerate() {
        println!("  {:>12}: {:>12.2}", label, regrets[i]);
    }

    // Decompose
    println!("\nPer-opponent breakdown (sorted by fold-call delta, descending):");
    println!("{:>6} {:>8} {:>12} {:>12} {:>12}   (remaining actions...)",
        "Opp", "Weight", "Fold Δ", "Call Δ", "F-C diff");
    println!("{}", "-".repeat(80));

    let mut results = solver.decompose_regrets_at(node_idx, hero_hand, hero_pos);

    // Sort by (fold_delta - call_delta) descending — shows which opponents make fold look better
    results.sort_by(|a, b| {
        let diff_a = if a.2.len() >= 2 { a.2[0] - a.2[1] } else { 0.0 };
        let diff_b = if b.2.len() >= 2 { b.2[0] - b.2[1] } else { 0.0 };
        diff_b.partial_cmp(&diff_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut total_fold = 0.0;
    let mut total_call = 0.0;
    let mut count = 0;

    for (opp_hand, weight, deltas, _) in &results {
        let opp_name = CanonicalHand::from_index(*opp_hand as usize)
            .map(|h| h.to_string())
            .unwrap_or_else(|| format!("#{opp_hand}"));
        let fold_d = deltas.first().copied().unwrap_or(0.0);
        let call_d = deltas.get(1).copied().unwrap_or(0.0);
        let diff = fold_d - call_d;

        total_fold += fold_d;
        total_call += call_d;
        count += 1;

        // Print top 30 and bottom 30
        if count <= 30 || results.len() - count < 30 {
            print!("{:>6} {:>8.4} {:>12.2} {:>12.2} {:>12.2}",
                opp_name, weight, fold_d, call_d, diff);
            // Print remaining action deltas
            for d in deltas.iter().skip(2) {
                print!("  {:>10.2}", d);
            }
            println!();
        } else if count == 31 {
            println!("  ... ({} more opponents) ...", results.len() - 60);
        }
    }

    println!("{}", "-".repeat(80));
    println!("{:>6} {:>8} {:>12.2} {:>12.2} {:>12.2}",
        "TOTAL", "", total_fold, total_call, total_fold - total_call);

    println!("\nIf fold-call diff > 0, those opponents make fold look better than call.");
    println!("If fold-call diff < 0, those opponents make call look better than fold.");

    Ok(())
}

fn run_trace_terminals(
    config_path: &Path,
    hero_str: &str,
    opp_str: &str,
    history: &str,
) -> Result<(), Box<dyn Error>> {
    use poker_solver_core::info_key::canonical_hand_index_from_str;

    let yaml = std::fs::read_to_string(config_path)?;
    let training: PreflopTrainingConfig = serde_yaml::from_str(&yaml)?;

    let postflop_model_path = training.postflop_model_path;
    let config = training.game;

    // Load equity
    let cache_base = Path::new("cache/postflop");
    let equity = if training.equity_samples > 0 {
        use poker_solver_core::preflop::equity_cache;
        equity_cache::load(cache_base, training.equity_samples)
            .ok_or("Equity cache not found — run solve-preflop first to build it")?
    } else {
        EquityTable::new_uniform()
    };

    // Build solver
    let tree = PreflopTree::build(&config);
    let mut solver = PreflopSolver::new_with_equity(&config, equity);

    // Load postflop model
    let pf_config_yaml = std::fs::read_to_string(postflop_model_path.join("config.yaml"))?;
    let pf_config: PostflopModelConfig = serde_yaml::from_str(&pf_config_yaml)?;
    let abstractions = PostflopBundle::load_multi(&pf_config, &postflop_model_path)?;
    solver.attach_postflop(abstractions, &config);

    // Resolve hands
    let hero_hand = canonical_hand_index_from_str(hero_str)
        .ok_or_else(|| format!("Invalid hand: {hero_str}"))?;
    let opp_hand = canonical_hand_index_from_str(opp_str)
        .ok_or_else(|| format!("Invalid hand: {opp_str}"))?;

    // Resolve target node
    let actions: Vec<&str> = history.split(',').map(str::trim).collect();
    let mut node_idx = 0u32;
    for action_str in &actions {
        node_idx = find_child_by_label(&tree, node_idx, action_str)
            .ok_or_else(|| format!("Action '{}' not found at node {}", action_str, node_idx))?;
    }

    // Get hero position
    let hero_pos = match &tree.nodes[node_idx as usize] {
        PreflopNode::Decision { position, .. } => *position,
        PreflopNode::Terminal { .. } => return Err("Target node is terminal".into()),
    };

    println!("Hero: {} (idx {}), Opp: {} (idx {})", hero_str, hero_hand, opp_str, opp_hand);
    solver.dump_terminal_values(node_idx, hero_hand as u16, opp_hand as u16, hero_pos);

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
