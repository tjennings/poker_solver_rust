mod lhe_diagnose;
mod lhe_viz;
mod tree;

use std::error::Error;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use poker_solver_core::Game;
use poker_solver_core::HandClass;
use poker_solver_core::abstract_game::{self, AbstractDealConfig};
use poker_solver_core::abstraction::{AbstractionConfig, BoundaryGenerator};
use poker_solver_core::blueprint::{
    AbstractionModeConfig, BlueprintStrategy, BundleConfig, StrategyBundle,
};
use poker_solver_core::cfr::convergence;
use poker_solver_core::cfr::{
    DealInfo, MccfrConfig, MccfrSolver, SequenceCfrConfig, SequenceCfrSolver, materialize_postflop,
};
use poker_solver_core::flops::{self, CanonicalFlop, RankTexture, SuitTexture};
use poker_solver_core::game::{
    AbstractionMode, Action, HunlPostflop, LimitHoldem, Player, PostflopConfig, PostflopState,
};
use poker_solver_core::hand_class::HandClassification;
use poker_solver_core::info_key::{
    InfoKey, canonical_hand_index_from_str, hand_label_from_bits, reverse_canonical_index,
    spr_bucket,
};
use poker_solver_core::preflop::{
    EquityTable, PostflopModelConfig, PreflopBundle, PreflopConfig, PreflopSolver, PreflopTree,
};
use poker_solver_core::preflop::postflop_abstraction::{BuildPhase, PostflopAbstraction};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[command(name = "poker-solver-trainer")]
#[command(about = "Train poker blueprint strategies using MCCFR")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Parser)]
enum Commands {
    /// Train a blueprint strategy
    Train {
        /// Path to YAML config file
        #[arg(short, long)]
        config: PathBuf,
        /// Number of threads for parallel training (default: all cores)
        #[arg(short, long)]
        threads: Option<usize>,
        /// Solver backend: mccfr (default) or sequence (full-traversal)
        #[arg(long, default_value = "mccfr")]
        solver: SolverMode,
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
    /// Print materialized game tree statistics for a training config
    TreeStats {
        /// Path to YAML config file
        #[arg(short, long)]
        config: PathBuf,
        /// Number of deals to sample for info set count estimation
        #[arg(long, default_value = "100")]
        sample_deals: usize,
    },
    /// Generate exhaustive abstract deals for tabular CFR
    GenerateDeals {
        /// Path to YAML config file
        #[arg(short, long)]
        config: PathBuf,
        /// Output directory for abstract deals
        #[arg(short, long)]
        output: PathBuf,
        /// Dry run: estimate deal count without generating
        #[arg(long)]
        dry_run: bool,
        /// Number of threads for parallel generation (default: all cores)
        #[arg(short, long)]
        threads: Option<usize>,
        /// Number of canonical flops per batch (0 = in-memory, no batching)
        #[arg(long, default_value = "20")]
        batch_size: usize,
    },
    /// Inspect pre-generated abstract deal files
    InspectDeals {
        /// Directory containing abstract_deals.bin and manifest.yaml
        #[arg(short, long)]
        input: PathBuf,
        /// Number of sample deals to display (0 = summary only)
        #[arg(short, long, default_value = "10")]
        limit: usize,
        /// Sort deals by: weight, equity, or none
        #[arg(long, default_value = "weight")]
        sort: DealSortOrder,
        /// Export deals to CSV file
        #[arg(long)]
        csv: Option<PathBuf>,
    },
    /// Merge batch files from a batched generate-deals run
    MergeDeals {
        /// Directory containing batch_*.bin files
        #[arg(short, long)]
        input: PathBuf,
        /// Output directory for abstract_deals.bin + manifest.yaml
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Analyze game tree or translate info set keys
    Tree {
        /// Path to trained strategy bundle
        #[arg(short, long)]
        bundle: PathBuf,
        /// Max tree depth to traverse
        #[arg(short, long, default_value = "4")]
        depth: usize,
        /// Prune branches below this probability
        #[arg(short, long, default_value = "0.01")]
        min_prob: f32,
        /// RNG seed for deal selection
        #[arg(short, long, default_value = "42")]
        seed: u64,
        /// Info set key (hex or compose format) for key describe mode
        #[arg(short, long)]
        key: Option<String>,
        /// Filter deals to this canonical hand (e.g. "AKs", "QQ")
        #[arg(long)]
        hand: Option<String>,
    },
    /// Evaluate a trained LHE SD-CFR model (exploitability + strategy visualization)
    EvalLhe {
        /// Training output directory (reads training_config.yaml for defaults)
        #[arg(short, long)]
        dir: PathBuf,
        /// Specific checkpoint subdir (default: latest_checkpoint)
        #[arg(long)]
        checkpoint: Option<String>,
        /// Number of deals for exploitability sampling
        #[arg(long, default_value = "10000")]
        eval_deals: usize,
        /// Override: number of actions in the trained network
        #[arg(long)]
        num_actions: Option<usize>,
        /// Override: hidden dimension of the trained network
        #[arg(long)]
        hidden_dim: Option<usize>,
        /// Override: random seed
        #[arg(long)]
        seed: Option<u64>,
        /// Override: LHE stack depth in BB
        #[arg(long)]
        stack_depth: Option<u32>,
        /// Override: number of streets (2=flop HE, 4=full)
        #[arg(long)]
        num_streets: Option<u8>,
        /// Board samples for strategy visualization
        #[arg(long, default_value = "100")]
        board_samples: usize,
    },
    /// Trace LHE strategy evolution across SD-CFR checkpoints
    TraceLhe {
        /// Training output directory containing lhe_checkpoint_N subdirs
        #[arg(short, long)]
        dir: PathBuf,
        /// Spot to trace (repeatable, e.g. "SB AA", "BB.R AKs")
        #[arg(long)]
        spot: Vec<String>,
        /// YAML file with spot definitions
        #[arg(long)]
        spots_file: Option<PathBuf>,
        /// Sample every Nth checkpoint
        #[arg(long, default_value = "1")]
        every: u32,
        /// Board samples per hand per checkpoint
        #[arg(long, default_value = "50")]
        board_samples: usize,
        /// Override: number of actions in the trained network
        #[arg(long)]
        num_actions: Option<usize>,
        /// Override: hidden dimension of the trained network
        #[arg(long)]
        hidden_dim: Option<usize>,
        /// Override: random seed
        #[arg(long)]
        seed: Option<u64>,
        /// Override: LHE stack depth in BB
        #[arg(long)]
        stack_depth: Option<u32>,
        /// Override: number of streets (2=flop HE, 4=full)
        #[arg(long)]
        num_streets: Option<u8>,
    },
    /// Solve preflop strategy using Linear CFR
    SolvePreflop {
        /// YAML config file (contains PreflopConfig + training params).
        /// CLI flags override values from the config file.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Stack depth in big blinds
        #[arg(long)]
        stack_depth: Option<u32>,
        /// Number of players (2 = HU, 6 = six-max)
        #[arg(long)]
        players: Option<u8>,
        /// Number of LCFR iterations
        #[arg(long)]
        iterations: Option<u64>,
        /// Output directory for the preflop bundle
        #[arg(short, long)]
        output: PathBuf,
        /// Print strategy matrices every N iterations (0 = only at end)
        #[arg(long)]
        print_every: Option<u64>,
        /// Monte Carlo samples per hand matchup for equity table (0 = uniform)
        #[arg(long)]
        equity_samples: Option<u32>,
        /// Postflop model preset: fast, medium, standard, or accurate.
        /// Overrides any postflop_model in the config file.
        #[arg(long)]
        postflop_model: Option<String>,
    },
}

/// Solver backend selection.
#[derive(Debug, Clone, ValueEnum)]
enum SolverMode {
    /// External sampling MCCFR with parallel Rayon training (default).
    Mccfr,
    /// Full-traversal sequence-form CFR on materialized game tree.
    Sequence,
    /// GPU-accelerated sequence-form CFR via wgpu compute shaders.
    /// Requires `--features gpu` at build time.
    Gpu,
    /// Single Deep CFR with neural network advantage estimation.
    SdCfr,
}

/// Output format for the flops command.
#[derive(Debug, Clone, ValueEnum)]
enum OutputFormat {
    Json,
    Csv,
}

/// Sort order for inspect-deals sample output.
#[derive(Debug, Clone, ValueEnum)]
enum DealSortOrder {
    Weight,
    Equity,
    None,
}

/// CFR variant: `dcfr` (default) or `linear` (alpha=beta=gamma=1).
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
enum CfrVariant {
    #[default]
    Dcfr,
    Linear,
}

fn default_cfr_variant() -> CfrVariant {
    CfrVariant::Dcfr
}

#[derive(Debug, Deserialize)]
struct TrainingConfig {
    game: PostflopConfig,
    /// EHS2 abstraction config. Omit entirely for hand_class mode.
    abstraction: Option<AbstractionConfig>,
    training: TrainingParams,
}

#[derive(Debug, Deserialize)]
struct TrainingParams {
    #[serde(default)]
    iterations: u64,
    seed: u64,
    output_dir: String,
    #[serde(default = "default_mccfr_samples")]
    mccfr_samples: usize,
    #[serde(default = "default_deal_count")]
    deal_count: usize,
    /// Which abstraction mode to use: `ehs2`, `hand_class`, or `hand_class_v2`.
    #[serde(default)]
    abstraction_mode: AbstractionModeConfig,
    /// Minimum deals per hand class for stratified generation (0 = disabled).
    #[serde(default)]
    min_deals_per_class: usize,
    /// Maximum rejection-sample attempts per deficit class.
    #[serde(default = "default_max_rejections")]
    max_rejections_per_class: usize,
    /// Number of bits for intra-class strength (0-4). Only used with `hand_class_v2`.
    #[serde(default = "default_strength_bits")]
    strength_bits: u8,
    /// Number of bits for equity bin (0-4). Only used with `hand_class_v2`.
    #[serde(default = "default_equity_bits")]
    equity_bits: u8,
    /// Strategy delta threshold for convergence-based stopping.
    /// When set, training continues until the mean L1 strategy delta between
    /// consecutive checkpoints drops below this value, overriding `iterations`.
    #[serde(default)]
    convergence_threshold: Option<f64>,
    /// How often (in iterations) to check convergence (default: 100).
    #[serde(default = "default_convergence_check_interval")]
    convergence_check_interval: u64,
    /// Enable regret-based pruning during training.
    #[serde(default)]
    pruning: bool,
    /// Fraction of total iterations to complete before enabling pruning.
    #[serde(default = "default_pruning_warmup_fraction")]
    pruning_warmup_fraction: f64,
    /// Run a full un-pruned probe iteration every N iterations.
    #[serde(default = "default_pruning_probe_interval")]
    pruning_probe_interval: u64,
    /// Regret threshold below which actions are pruned (default 0.0).
    /// With DCFR, a negative value (e.g. -5.0) lets DCFR decay bring
    /// regrets back above the line between probes.
    #[serde(default)]
    pruning_threshold: f64,
    /// CFR variant: `dcfr` (default) or `linear` (alpha=beta=gamma=1).
    #[serde(default = "default_cfr_variant")]
    cfr_variant: CfrVariant,
    /// Use exhaustive abstract deal enumeration instead of random sampling.
    /// Only valid with `hand_class_v2` abstraction mode.
    #[serde(default)]
    exhaustive: bool,
    /// Pre-generated abstract deals directory. If set, loads deals from disk.
    /// If absent with `exhaustive: true`, generates in-memory.
    #[serde(default)]
    abstract_deals_dir: Option<String>,
    /// GPU tile size for tiled solver. `None` or absent = auto-detect,
    /// `Some(0)` = force untiled solver.
    #[serde(default)]
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    tile_size: Option<u32>,
}

fn default_convergence_check_interval() -> u64 {
    100
}

fn default_mccfr_samples() -> usize {
    500
}

fn default_deal_count() -> usize {
    50_000
}

fn default_max_rejections() -> usize {
    500_000
}

fn default_strength_bits() -> u8 {
    4
}

fn default_equity_bits() -> u8 {
    4
}

fn default_pruning_warmup_fraction() -> f64 {
    0.2
}

fn default_pruning_probe_interval() -> u64 {
    20
}

// ---------------------------------------------------------------------------
// SD-CFR YAML config structs
// ---------------------------------------------------------------------------

/// Game type selection for SD-CFR training.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SdCfrGameType {
    HunlPostflop,
    LimitHoldem,
}

fn default_sdcfr_game_type() -> SdCfrGameType {
    SdCfrGameType::HunlPostflop
}

#[derive(Debug, Deserialize)]
pub(crate) struct SdCfrTrainingConfig {
    #[serde(default = "default_sdcfr_game_type")]
    pub(crate) game_type: SdCfrGameType,
    pub(crate) game: PostflopConfig,
    #[serde(default)]
    pub(crate) lhe_game: Option<poker_solver_core::game::LimitHoldemConfig>,
    pub(crate) deals: SdCfrDealConfig,
    pub(crate) training: SdCfrTrainingParams,
    pub(crate) network: SdCfrNetworkConfig,
    pub(crate) sgd: SdCfrSgdConfig,
    pub(crate) memory: SdCfrMemoryConfig,
    pub(crate) checkpoint: SdCfrCheckpointConfig,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)] // Fields parsed from YAML; some reserved for future checkpoint/stratification tasks
pub(crate) struct SdCfrDealConfig {
    pub(crate) count: usize,
    #[serde(default)]
    pub(crate) min_per_class: usize,
    #[serde(default = "default_max_rejections")]
    pub(crate) max_rejections: usize,
    pub(crate) seed: u64,
}

fn default_parallel_traversals() -> usize {
    256
}

#[derive(Debug, Deserialize)]
pub(crate) struct SdCfrTrainingParams {
    pub(crate) iterations: u32,
    pub(crate) traversals_per_iter: u32,
    pub(crate) seed: u64,
    pub(crate) output_dir: String,
    #[serde(default = "default_parallel_traversals")]
    pub(crate) parallel_traversals: usize,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SdCfrNetworkConfig {
    pub(crate) hidden_dim: usize,
    pub(crate) num_actions: usize,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SdCfrSgdConfig {
    pub(crate) steps: usize,
    pub(crate) batch_size: usize,
    pub(crate) learning_rate: f64,
    #[serde(default = "default_grad_clip")]
    pub(crate) grad_clip_norm: f64,
}

fn default_grad_clip() -> f64 {
    1.0
}

#[derive(Debug, Deserialize)]
pub(crate) struct SdCfrMemoryConfig {
    pub(crate) advantage_cap: usize,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SdCfrCheckpointConfig {
    pub(crate) interval: u32,
}

/// Hands to display in the SB preflop strategy table.
const DISPLAY_HANDS: &[&str] = &["AA", "KK", "QQ", "AKs", "AKo", "JTs", "76s", "72o"];

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            config,
            threads,
            solver,
        } => {
            if let Some(n) = threads {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build_global()
                    .expect("failed to configure rayon thread pool");
            }
            match solver {
                SolverMode::SdCfr => {
                    run_sdcfr_training(&config)?;
                }
                SolverMode::Mccfr | SolverMode::Sequence | SolverMode::Gpu => {
                    let yaml = std::fs::read_to_string(&config)?;
                    let training_config: TrainingConfig = serde_yaml::from_str(&yaml)?;
                    match solver {
                        SolverMode::Mccfr => run_mccfr_training(training_config)?,
                        SolverMode::Sequence => run_sequence_training(training_config)?,
                        SolverMode::Gpu => {
                            #[cfg(feature = "gpu")]
                            {
                                run_gpu_training(training_config)?;
                            }
                            #[cfg(not(feature = "gpu"))]
                            {
                                eprintln!(
                                    "Error: GPU solver requires `--features gpu` at build time."
                                );
                                eprintln!(
                                    "Build with: cargo run -p poker-solver-trainer --features gpu --release -- train ..."
                                );
                                std::process::exit(1);
                            }
                        }
                        SolverMode::SdCfr => unreachable!("SdCfr handled in outer match"),
                    }
                }
            }
        }
        Commands::GenerateDeals {
            config,
            output,
            dry_run,
            threads,
            batch_size,
        } => {
            if let Some(n) = threads {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build_global()
                    .expect("failed to configure rayon thread pool");
            }
            let yaml = std::fs::read_to_string(&config)?;
            let training_config: TrainingConfig = serde_yaml::from_str(&yaml)?;
            run_generate_deals(training_config, &output, dry_run, batch_size)?;
        }
        Commands::MergeDeals { input, output } => {
            run_merge_deals(&input, &output)?;
        }
        Commands::InspectDeals {
            input,
            limit,
            sort,
            csv,
        } => {
            run_inspect_deals(&input, limit, sort, csv.as_deref())?;
        }
        Commands::TreeStats {
            config,
            sample_deals,
        } => {
            let yaml = std::fs::read_to_string(&config)?;
            let training_config: TrainingConfig = serde_yaml::from_str(&yaml)?;
            run_tree_stats(training_config, sample_deals)?;
        }
        Commands::Flops { format, output } => {
            run_flops(format, output)?;
        }
        Commands::Tree {
            bundle,
            depth,
            min_prob,
            seed,
            key,
            hand,
        } => {
            tree::run_tree(
                &bundle,
                depth,
                min_prob,
                seed,
                key.as_deref(),
                hand.as_deref(),
            )?;
        }
        Commands::EvalLhe {
            dir,
            checkpoint,
            eval_deals,
            num_actions,
            hidden_dim,
            seed,
            stack_depth,
            num_streets,
            board_samples,
        } => {
            run_eval_lhe(
                &dir,
                checkpoint.as_deref(),
                eval_deals,
                num_actions,
                hidden_dim,
                seed,
                stack_depth,
                num_streets,
                board_samples,
            )?;
        }
        Commands::TraceLhe {
            dir,
            spot,
            spots_file,
            every,
            board_samples,
            num_actions,
            hidden_dim,
            seed,
            stack_depth,
            num_streets,
        } => {
            lhe_diagnose::run_trace_lhe(
                &dir,
                &spot,
                spots_file.as_deref(),
                every,
                board_samples,
                num_actions,
                hidden_dim,
                seed,
                stack_depth,
                num_streets,
            )?;
        }
        Commands::SolvePreflop {
            config,
            stack_depth,
            players,
            iterations,
            output,
            print_every,
            equity_samples,
            postflop_model,
        } => {
            run_solve_preflop(
                config.as_deref(),
                stack_depth,
                players,
                iterations,
                &output,
                print_every,
                equity_samples,
                postflop_model.as_deref(),
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
}

fn default_iterations() -> u64 { 5000 }
fn default_equity_samples() -> u32 { 20000 }
fn default_print_every() -> u64 { 1000 }

#[allow(clippy::too_many_arguments)]
fn run_solve_preflop(
    config_path: Option<&Path>,
    cli_stack_depth: Option<u32>,
    cli_players: Option<u8>,
    cli_iterations: Option<u64>,
    output: &Path,
    cli_print_every: Option<u64>,
    cli_equity_samples: Option<u32>,
    cli_postflop_model: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    // Load config: from YAML file or defaults based on CLI args.
    let mut training = if let Some(path) = config_path {
        let yaml = std::fs::read_to_string(path)?;
        serde_yaml::from_str::<PreflopTrainingConfig>(&yaml)?
    } else {
        let stack_depth = cli_stack_depth.unwrap_or(100);
        let players = cli_players.unwrap_or(2);
        let game = match players {
            2 => PreflopConfig::heads_up(stack_depth),
            6 => PreflopConfig::six_max(stack_depth),
            n => return Err(format!("unsupported player count: {n} (use 2 or 6)").into()),
        };
        PreflopTrainingConfig {
            game,
            iterations: 5000,
            equity_samples: 20000,
            print_every: 1000,
        }
    };

    // CLI overrides
    if let Some(v) = cli_iterations { training.iterations = v; }
    if let Some(v) = cli_equity_samples { training.equity_samples = v; }
    if let Some(v) = cli_print_every { training.print_every = v; }
    if let Some(preset) = cli_postflop_model {
        training.game.postflop_model = Some(PostflopModelConfig::from_preset(preset)
            .ok_or_else(|| format!("unknown postflop preset: {preset} (use fast/medium/standard/accurate)"))?);
    }

    let config = training.game;
    let iterations = training.iterations;
    let equity_samples = training.equity_samples;
    let print_every = training.print_every;
    let players = config.positions.len();

    let equity = if equity_samples > 0 {
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
        lhe_viz::print_equity_matrix(&table);
        table
    } else {
        println!("Using uniform equities (--equity-samples 0)");
        EquityTable::new_uniform()
    };

    println!("Solving preflop: {players}p, {iterations} iterations");
    let start = Instant::now();

    let tree = PreflopTree::build(&config);
    let bb_node = lhe_viz::find_raise_child(&tree, 0);

    // Build postflop abstraction before the solver takes equity ownership.
    let postflop = if let Some(pf_config) = &config.postflop_model {
        let pf_pb = ProgressBar::new(0);
        pf_pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} {msg} [{bar:40.cyan/blue}] {pos}/{len}")
                .expect("valid template")
                .progress_chars("#>-"),
        );
        pf_pb.set_message("Building postflop abstraction");
        let pf_start = Instant::now();
        let cache_base = std::path::Path::new("cache/postflop");
        let abstraction = PostflopAbstraction::build(pf_config, Some(&equity), Some(cache_base), |phase| {
            match &phase {
                BuildPhase::HandBuckets(done, total) => {
                    pf_pb.set_length(*total as u64);
                    pf_pb.set_position(*done as u64);
                    pf_pb.set_message("Hand buckets");
                }
                BuildPhase::SolvingPostflop(iter, total) => {
                    pf_pb.set_length(*total as u64);
                    pf_pb.set_position(*iter as u64);
                    pf_pb.set_message("Solving postflop");
                }
                _ => {
                    pf_pb.set_message(format!("{phase}..."));
                }
            }
        })
        .map_err(|e| format!("postflop abstraction: {e}"))?;
        pf_pb.finish_with_message(format!(
            "done in {:.1?} (values: {} entries)",
            pf_start.elapsed(),
            abstraction.values.len(),
        ));
        Some(abstraction)
    } else {
        None
    };

    let mut solver = PreflopSolver::new_with_equity(&config, equity);
    if let Some(abstraction) = postflop {
        solver.attach_postflop(abstraction);
    }

    let mut prev_matrices = print_preflop_matrices(&solver.strategy(), &tree, bb_node, 0, None);

    let pb = ProgressBar::new(iterations);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .expect("valid template")
            .progress_chars("#>-"),
    );

    let chunk = std::cmp::max(iterations / 100, 1);
    let mut done = 0u64;
    while done < iterations {
        let batch = std::cmp::min(chunk, iterations - done);
        solver.train(batch);
        done += batch;
        pb.set_position(done);

        if print_every > 0 && done < iterations && done.is_multiple_of(print_every) {
            pb.suspend(|| {
                prev_matrices = print_preflop_matrices(
                    &solver.strategy(), &tree, bb_node, done, Some(&prev_matrices),
                );
            });
        }
    }
    pb.finish_with_message("done");

    let elapsed = start.elapsed();
    let strategy = solver.strategy();
    println!("Converged in {elapsed:.1?} — {} info sets", strategy.len());

    print_preflop_matrices(&strategy, &tree, bb_node, iterations, Some(&prev_matrices));

    let bundle = PreflopBundle::new(config, strategy);
    bundle.save(output)?;
    println!("Saved to {}", output.display());

    Ok(())
}

/// Previous SB and BB matrices for delta computation.
type PrevMatrices = (lhe_viz::HandMatrix, Option<lhe_viz::HandMatrix>);

fn print_preflop_matrices(
    strategy: &poker_solver_core::preflop::PreflopStrategy,
    tree: &PreflopTree,
    bb_node: Option<u32>,
    iteration: u64,
    prev: Option<&PrevMatrices>,
) -> PrevMatrices {
    let sb_matrix = lhe_viz::preflop_strategy_matrix(strategy, tree, 0);
    let sb_delta = prev.and_then(|(p, _)| lhe_viz::matrix_delta(p, &sb_matrix));
    let sb_title = match sb_delta {
        Some(d) => format!("SB RFI — iteration {iteration}  (Δ={d:.4})"),
        None => format!("SB RFI — iteration {iteration}"),
    };
    lhe_viz::print_hand_matrix(&sb_matrix, &sb_title);

    let bb_matrix = bb_node.map(|bb_idx| {
        let m = lhe_viz::preflop_strategy_matrix(strategy, tree, bb_idx);
        let bb_delta = prev
            .and_then(|(_, prev_bb)| prev_bb.as_ref())
            .and_then(|p| lhe_viz::matrix_delta(p, &m));
        let bb_title = match bb_delta {
            Some(d) => format!("BB vs Raise — iteration {iteration}  (Δ={d:.4})"),
            None => format!("BB vs Raise — iteration {iteration}"),
        };
        lhe_viz::print_hand_matrix(&m, &bb_title);
        m
    });

    (sb_matrix, bb_matrix)
}

// ---------------------------------------------------------------------------
// Shared training loop infrastructure
// ---------------------------------------------------------------------------

/// Trait abstracting over solver backends for the generic training loop.
trait TrainingSolver {
    /// Train for `iterations` iterations, calling `callback` after each.
    fn train_batch(&mut self, iterations: u64, callback: &dyn Fn(u64));

    /// Extract the full average strategy map.
    fn all_strategies(&self) -> FxHashMap<u64, Vec<f64>>;

    /// Return the total number of iterations completed.
    fn iterations(&self) -> u64;

    /// Naming prefix for checkpoint directories (e.g. "checkpoint_", "seq_checkpoint_").
    fn checkpoint_prefix(&self) -> &str;

    /// Called at each checkpoint. Should print progress/metrics and return the
    /// strategy delta (if computable from a previous checkpoint).
    fn checkpoint_report(
        &mut self,
        checkpoint_num: u64,
        is_convergence: bool,
        total_checkpoints: u64,
        total_iterations: u64,
    ) -> Option<f64>;
}

/// Configuration for [`run_training_loop`].
struct TrainingLoopConfig<'a> {
    convergence_threshold: Option<f64>,
    check_interval: u64,
    iterations: u64,
    output_dir: &'a str,
    bundle_config: &'a BundleConfig,
    boundaries: &'a Option<poker_solver_core::abstraction::BucketBoundaries>,
}

/// Run the convergence or fixed-iteration training loop, then save the final bundle.
fn run_training_loop<S: TrainingSolver>(
    solver: &mut S,
    loop_config: &TrainingLoopConfig<'_>,
) -> Result<(), Box<dyn Error>> {
    let training_start = Instant::now();

    if let Some(threshold) = loop_config.convergence_threshold {
        run_convergence_loop(solver, threshold, loop_config);
    } else {
        run_fixed_iteration_loop(solver, loop_config);
    }

    save_final_bundle(solver, loop_config)?;

    println!("\n=== Training Complete ===");
    println!("Total time: {:?}", training_start.elapsed());

    Ok(())
}

fn run_convergence_loop<S: TrainingSolver>(
    solver: &mut S,
    threshold: f64,
    config: &TrainingLoopConfig<'_>,
) {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] {pos} iters ({per_sec})",
        )
        .expect("valid template"),
    );

    let mut checkpoint_num = 0u64;

    loop {
        solver.train_batch(config.check_interval, &|_| pb.inc(1));
        checkpoint_num += 1;

        let converged = pb.suspend(|| {
            let delta = solver.checkpoint_report(checkpoint_num, true, 0, 0);

            save_checkpoint(
                solver.all_strategies(),
                solver.iterations(),
                &format!("{}{checkpoint_num}", solver.checkpoint_prefix()),
                config.output_dir,
                config.bundle_config,
                config.boundaries,
            );

            delta.is_some_and(|d| d < threshold)
        });

        if converged {
            pb.finish_with_message(format!(
                "Converged after {} iterations",
                solver.iterations()
            ));
            break;
        }
    }
}

fn run_fixed_iteration_loop<S: TrainingSolver>(solver: &mut S, config: &TrainingLoopConfig<'_>) {
    let total = config.iterations;
    let checkpoint_interval = (total / 10).max(1);

    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} iters ({per_sec}, ETA: {eta})",
        )
        .expect("valid template")
        .progress_chars("=>-"),
    );

    for checkpoint in 1..=10 {
        solver.train_batch(checkpoint_interval, &|_| pb.inc(1));

        pb.suspend(|| {
            solver.checkpoint_report(checkpoint, false, 10, total);

            save_checkpoint(
                solver.all_strategies(),
                solver.iterations(),
                &format!("{}{checkpoint}_of_10", solver.checkpoint_prefix()),
                config.output_dir,
                config.bundle_config,
                config.boundaries,
            );
        });
    }

    let trained_so_far = checkpoint_interval * 10;
    if trained_so_far < total {
        solver.train_batch(total - trained_so_far, &|_| pb.inc(1));
    }

    pb.finish_with_message("Training complete");
}

fn save_final_bundle<S: TrainingSolver>(
    solver: &S,
    config: &TrainingLoopConfig<'_>,
) -> Result<(), Box<dyn Error>> {
    println!("\nSaving strategy bundle...");
    let strategies = solver.all_strategies();
    let blueprint = BlueprintStrategy::from_strategies(strategies, solver.iterations());
    println!(
        "  {} info sets, {} iterations",
        blueprint.len(),
        blueprint.iterations_trained()
    );

    let bundle = StrategyBundle::new(
        config.bundle_config.clone(),
        blueprint,
        config.boundaries.clone(),
    );
    let output_path = PathBuf::from(config.output_dir);
    bundle.save(&output_path)?;
    println!("  Saved to {}/", config.output_dir);

    Ok(())
}

// ---------------------------------------------------------------------------
// MCCFR solver wrapper
// ---------------------------------------------------------------------------

struct MccfrTrainingSolver<'a> {
    solver: MccfrSolver<HunlPostflop>,
    mccfr_samples: usize,
    previous_strategies: Option<FxHashMap<u64, Vec<f64>>>,
    training_start: Instant,
    header: String,
    action_labels: Vec<String>,
    stack_depth: u32,
    output_dir: &'a str,
    abs_mode: AbstractionModeConfig,
    convergence_threshold: Option<f64>,
}

impl TrainingSolver for MccfrTrainingSolver<'_> {
    fn train_batch(&mut self, iterations: u64, callback: &dyn Fn(u64)) {
        self.solver
            .train_parallel_with_callback(iterations, self.mccfr_samples, callback);
    }

    fn all_strategies(&self) -> FxHashMap<u64, Vec<f64>> {
        self.solver.all_strategies()
    }

    fn iterations(&self) -> u64 {
        self.solver.iterations()
    }

    fn checkpoint_prefix(&self) -> &str {
        "checkpoint_"
    }

    fn checkpoint_report(
        &mut self,
        checkpoint_num: u64,
        is_convergence: bool,
        total_checkpoints: u64,
        total_iterations: u64,
    ) -> Option<f64> {
        let strategies = self.solver.all_strategies_best_effort();

        let report_ctx = StrategyReportCtx {
            strategies: &strategies,
            previous: &self.previous_strategies,
            checkpoint_num,
            is_convergence,
            total_checkpoints,
            total_iterations,
            current_iterations: self.solver.iterations(),
            training_start: &self.training_start,
            header: &self.header,
            action_labels: &self.action_labels,
            stack_depth: self.stack_depth,
            abs_mode: self.abs_mode,
            convergence_threshold: if is_convergence {
                self.convergence_threshold
            } else {
                None
            },
            max_regret: None,
        };
        let delta = print_strategy_report(&report_ctx);

        // MCCFR-specific extras: pruning stats, regret metrics, extreme regret keys
        let mccfr_ctx = MccfrCheckpointCtx {
            output_dir: self.output_dir,
        };
        print_mccfr_extras(&self.solver, checkpoint_num, &mccfr_ctx);

        self.previous_strategies = Some(strategies);
        delta
    }
}

// ---------------------------------------------------------------------------
// Sequence/GPU solver wrappers
// ---------------------------------------------------------------------------

struct SimpleTrainingSolver<'a, S> {
    solver: S,
    prefix: &'a str,
    previous: Option<FxHashMap<u64, Vec<f64>>>,
    training_start: Instant,
    header: String,
    action_labels: Vec<String>,
    stack_depth: u32,
    abs_mode: AbstractionModeConfig,
    convergence_threshold: Option<f64>,
}

impl<S> TrainingSolver for SimpleTrainingSolver<'_, S>
where
    S: SimpleTrainingSolverBackend,
{
    fn train_batch(&mut self, iterations: u64, callback: &dyn Fn(u64)) {
        self.solver.train_with_cb(iterations, callback);
    }

    fn all_strategies(&self) -> FxHashMap<u64, Vec<f64>> {
        self.solver.strategies()
    }

    fn iterations(&self) -> u64 {
        self.solver.iters()
    }

    fn checkpoint_prefix(&self) -> &str {
        self.prefix
    }

    fn checkpoint_report(
        &mut self,
        checkpoint_num: u64,
        is_convergence: bool,
        total_checkpoints: u64,
        total_iterations: u64,
    ) -> Option<f64> {
        let strategies = self.solver.strategies_best_effort();

        let report_ctx = StrategyReportCtx {
            strategies: &strategies,
            previous: &self.previous,
            checkpoint_num,
            is_convergence,
            total_checkpoints,
            total_iterations,
            current_iterations: self.solver.iters(),
            training_start: &self.training_start,
            header: &self.header,
            action_labels: &self.action_labels,
            stack_depth: self.stack_depth,
            abs_mode: self.abs_mode,
            convergence_threshold: if is_convergence {
                self.convergence_threshold
            } else {
                None
            },
            max_regret: self.solver.max_regret(),
        };
        let delta = print_strategy_report(&report_ctx);

        self.previous = Some(strategies);
        delta
    }
}

/// Minimal interface that both `SequenceCfrSolver` and `TabularGpuCfrSolver` satisfy.
trait SimpleTrainingSolverBackend {
    fn train_with_cb(&mut self, iterations: u64, callback: &dyn Fn(u64));
    fn strategies(&self) -> FxHashMap<u64, Vec<f64>>;
    fn strategies_best_effort(&self) -> FxHashMap<u64, Vec<f64>>;
    fn iters(&self) -> u64;
    /// GPU max regret (upper bound on exploitability). `None` for CPU solvers.
    fn max_regret(&self) -> Option<f32> {
        None
    }
}

impl SimpleTrainingSolverBackend for SequenceCfrSolver {
    fn train_with_cb(&mut self, iterations: u64, callback: &dyn Fn(u64)) {
        self.train_with_callback(iterations, callback);
    }
    fn strategies(&self) -> FxHashMap<u64, Vec<f64>> {
        self.all_strategies()
    }
    fn strategies_best_effort(&self) -> FxHashMap<u64, Vec<f64>> {
        self.all_strategies_best_effort()
    }
    fn iters(&self) -> u64 {
        self.iterations()
    }
}

#[cfg(feature = "gpu")]
impl SimpleTrainingSolverBackend for poker_solver_gpu_cfr::tabular::TabularGpuCfrSolver {
    fn train_with_cb(&mut self, iterations: u64, callback: &dyn Fn(u64)) {
        self.train_with_callback(iterations, callback);
    }
    fn strategies(&self) -> FxHashMap<u64, Vec<f64>> {
        self.all_strategies()
    }
    fn strategies_best_effort(&self) -> FxHashMap<u64, Vec<f64>> {
        self.all_strategies()
    }
    fn iters(&self) -> u64 {
        self.iterations()
    }
    fn max_regret(&self) -> Option<f32> {
        Some(self.max_regret())
    }
}

#[cfg(feature = "gpu")]
impl SimpleTrainingSolverBackend for poker_solver_gpu_cfr::tiled::TiledTabularGpuCfrSolver {
    fn train_with_cb(&mut self, iterations: u64, callback: &dyn Fn(u64)) {
        self.train_with_callback(iterations, callback);
    }
    fn strategies(&self) -> FxHashMap<u64, Vec<f64>> {
        self.all_strategies()
    }
    fn strategies_best_effort(&self) -> FxHashMap<u64, Vec<f64>> {
        self.all_strategies()
    }
    fn iters(&self) -> u64 {
        self.iterations()
    }
    fn max_regret(&self) -> Option<f32> {
        Some(self.max_regret())
    }
}

fn run_mccfr_training(config: TrainingConfig) -> Result<(), Box<dyn Error>> {
    let abs_mode = config.training.abstraction_mode;
    let use_hand_class_v2 = abs_mode == AbstractionModeConfig::HandClassV2;

    println!("=== Poker Blueprint Trainer (MCCFR) ===\n");
    println!("Game config:");
    println!("  Stack depth: {} BB", config.game.stack_depth);
    println!("  Bet sizes: {:?} pot", config.game.bet_sizes);
    println!("  Deal pool size: {}", config.training.deal_count);
    println!();

    if use_hand_class_v2 {
        println!(
            "Abstraction: hand_class_v2 (strength_bits={}, equity_bits={})",
            config.training.strength_bits, config.training.equity_bits
        );
    } else if let Some(ref abs) = config.abstraction {
        println!("Abstraction config:");
        println!("  Flop buckets: {}", abs.flop_buckets);
        println!("  Turn buckets: {}", abs.turn_buckets);
        println!("  River buckets: {}", abs.river_buckets);
        println!("  Samples/street: {}", abs.samples_per_street);
    }
    println!();

    // Validate stopping criteria
    let convergence_mode = config.training.convergence_threshold.is_some();
    if convergence_mode && config.training.iterations > 0 {
        eprintln!(
            "WARNING: both convergence_threshold ({:.6}) and iterations ({}) are set; \
             convergence_threshold will be used for stopping (iterations value ignored)",
            config.training.convergence_threshold.unwrap(),
            config.training.iterations,
        );
    }
    if !convergence_mode && config.training.iterations == 0 {
        return Err("Either iterations or convergence_threshold must be set".into());
    }

    if let Some(threshold) = config.training.convergence_threshold {
        println!(
            "Training: converge to delta < {:.6}, check every {} iters, {} samples/iter, seed {}",
            threshold,
            config.training.convergence_check_interval,
            config.training.mccfr_samples,
            config.training.seed,
        );
    } else {
        println!(
            "Training: {} iterations, {} samples/iter, seed {}",
            config.training.iterations, config.training.mccfr_samples, config.training.seed
        );
    }
    println!("Output: {}", config.training.output_dir);
    println!();

    // Generate bucket boundaries (only for EHS2 mode)
    let boundaries = if abs_mode.is_hand_class() {
        println!("Skipping boundary generation ({abs_mode:?} mode)\n");
        None
    } else if let Some(ref abs_config) = config.abstraction {
        println!("Generating bucket boundaries...");
        let start = Instant::now();
        let generator = BoundaryGenerator::new(abs_config.clone());
        let b = generator.generate(config.training.seed);
        println!("  Done in {:?}\n", start.elapsed());
        Some(b)
    } else {
        None
    };

    // Select abstraction mode for the game
    let abstraction_mode = if use_hand_class_v2 {
        Some(AbstractionMode::HandClassV2 {
            strength_bits: config.training.strength_bits,
            equity_bits: config.training.equity_bits,
        })
    } else {
        None
    };

    // Create game with deal pool
    let mut game = HunlPostflop::new(
        config.game.clone(),
        abstraction_mode,
        config.training.deal_count,
    );
    if config.training.min_deals_per_class > 0 {
        game = game.with_stratification(
            config.training.min_deals_per_class,
            config.training.max_rejections_per_class,
        );
    }

    // Create MCCFR solver
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let pruning_warmup =
        (config.training.pruning_warmup_fraction * config.training.iterations as f64) as u64;

    let mccfr_config = MccfrConfig {
        samples_per_iteration: config.training.mccfr_samples,
        pruning: config.training.pruning,
        pruning_warmup,
        pruning_probe_interval: config.training.pruning_probe_interval,
        pruning_threshold: config.training.pruning_threshold,
        ..MccfrConfig::default()
    };

    println!("Creating MCCFR solver...");
    println!(
        "  DCFR: α={}, β={}, γ={}",
        mccfr_config.dcfr_alpha, mccfr_config.dcfr_beta, mccfr_config.dcfr_gamma
    );
    if config.training.pruning {
        println!(
            "  Pruning: enabled (warmup {:.0}%, probe every {} iters, threshold {:.1})",
            config.training.pruning_warmup_fraction * 100.0,
            config.training.pruning_probe_interval,
            config.training.pruning_threshold,
        );
    }
    let start = Instant::now();
    let mut solver = MccfrSolver::with_config(game, &mccfr_config);
    solver.set_seed(config.training.seed);
    println!("  Created in {:?}\n", start.elapsed());

    // Derive action labels from a single deal (all deals share the same preflop actions)
    let label_game = HunlPostflop::new(config.game.clone(), None, 1);
    let initial_states = label_game.initial_states();
    let actions = label_game.actions(&initial_states[0]);
    let action_labels = format_action_labels(&actions);
    let header = format_table_header(&action_labels);

    // Build bundle config for intermediate saves
    let bundle_config = BundleConfig {
        game: config.game.clone(),
        abstraction: config.abstraction.clone(),
        abstraction_mode: abs_mode,
        strength_bits: if use_hand_class_v2 {
            config.training.strength_bits
        } else {
            0
        },
        equity_bits: if use_hand_class_v2 {
            config.training.equity_bits
        } else {
            0
        },
        ..BundleConfig::default()
    };

    let num_threads = rayon::current_num_threads();
    if let Some(threshold) = config.training.convergence_threshold {
        println!(
            "  Parallel training with {} threads (target delta < {:.6})\n",
            num_threads, threshold
        );
    } else {
        println!("  Parallel training with {} threads\n", num_threads);
    }

    let mut wrapper = MccfrTrainingSolver {
        solver,
        mccfr_samples: config.training.mccfr_samples,
        previous_strategies: None,
        training_start: Instant::now(),
        header,
        action_labels,
        stack_depth: config.game.stack_depth,
        output_dir: &config.training.output_dir,
        abs_mode,
        convergence_threshold: config.training.convergence_threshold,
    };

    // Baseline checkpoint (iteration 0)
    wrapper.checkpoint_report(0, false, 10, config.training.iterations);

    let loop_config = TrainingLoopConfig {
        convergence_threshold: config.training.convergence_threshold,
        check_interval: config.training.convergence_check_interval,
        iterations: config.training.iterations,
        output_dir: &config.training.output_dir,
        bundle_config: &bundle_config,
        boundaries: &boundaries,
    };

    run_training_loop(&mut wrapper, &loop_config)
}

fn run_sdcfr_training(config_path: &Path) -> Result<(), Box<dyn Error>> {
    let yaml = std::fs::read_to_string(config_path)?;
    let config: SdCfrTrainingConfig = serde_yaml::from_str(&yaml)?;

    match config.game_type {
        SdCfrGameType::HunlPostflop => run_sdcfr_hunl(config),
        SdCfrGameType::LimitHoldem => run_sdcfr_lhe(config, config_path),
    }
}

fn run_sdcfr_hunl(config: SdCfrTrainingConfig) -> Result<(), Box<dyn Error>> {
    print_sdcfr_config(&config);

    let device = poker_solver_deep_cfr::best_available_device();
    println!("  Compute device: {:?}", device);

    let game = HunlPostflop::new(config.game.clone(), None, config.deals.count);
    let deal_pool = game.initial_states();
    println!("  Generated {} deals", deal_pool.len());

    let encoder =
        poker_solver_deep_cfr::hunl_encoder::HunlStateEncoder::new(config.game.bet_sizes.clone());
    let sdcfr_config = build_sdcfr_config(&config);
    let mut solver = poker_solver_deep_cfr::solver::SdCfrSolver::new(game, encoder, sdcfr_config)?;

    let output_dir = config.training.output_dir.clone();
    let game_config = config.game.clone();
    let num_actions = config.network.num_actions;
    let hidden_dim = config.network.hidden_dim;

    // Derive action labels from a single deal (same pattern as MCCFR setup)
    let label_game = HunlPostflop::new(config.game.clone(), None, 1);
    let initial_states = label_game.initial_states();
    let actions = label_game.actions(&initial_states[0]);
    let action_labels = format_action_labels(&actions);
    let header = format_table_header(&action_labels);

    let total_iterations = config.training.iterations;
    let checkpoint_interval = config.checkpoint.interval;
    let total_checkpoints = total_iterations
        .checked_div(checkpoint_interval)
        .unwrap_or(0);
    let training_start = Instant::now();
    let mut previous_strategies: Option<FxHashMap<u64, Vec<f64>>> = None;
    let mut checkpoint_count = 0u64;

    let mut report_state = SdCfrReportState {
        training_start: &training_start,
        header: &header,
        action_labels: &action_labels,
        previous_strategies: &mut previous_strategies,
        checkpoint_count: 0,
        total_checkpoints: u64::from(total_checkpoints),
        total_iterations: u64::from(total_iterations),
        stack_depth: config.game.stack_depth,
    };

    println!("\nStarting SD-CFR training ({total_iterations} iterations)...");
    let start = Instant::now();

    for i in 1..=total_iterations {
        let iter_start = Instant::now();
        eprintln!(
            "  [iter {i}/{total_iterations}] starting (elapsed: {:.1}s)...",
            start.elapsed().as_secs_f64(),
        );

        solver.step_with_deals(Some(&deal_pool))?;

        let iter_elapsed = iter_start.elapsed();
        let total_elapsed = start.elapsed().as_secs_f64();
        let rate = f64::from(i) / total_elapsed;
        let remaining = f64::from(total_iterations - i) / rate;
        let stats = solver.buffer_stats();

        eprintln!(
            "  [iter {i}/{total_iterations}] done in {:.1}s | rate: {rate:.2} iter/s | \
             buf P1: {}/{} P2: {}/{} | ETA: {:.0}s",
            iter_elapsed.as_secs_f64(),
            stats[0].0,
            stats[0].1,
            stats[1].0,
            stats[1].1,
            remaining,
        );

        // Run checkpoint callback at the specified interval
        if checkpoint_interval > 0 && i % checkpoint_interval == 0 {
            checkpoint_count += 1;
            report_state.checkpoint_count = checkpoint_count;
            let snap = solver.snapshot();
            save_sdcfr_checkpoint(
                &snap,
                i,
                &output_dir,
                &game_config,
                num_actions,
                hidden_dim,
                &mut report_state,
            )?;
        }
    }

    let elapsed = start.elapsed();
    let trained = solver.take_result();
    print_sdcfr_summary(&trained, elapsed);
    Ok(())
}

fn run_sdcfr_lhe(config: SdCfrTrainingConfig, config_path: &Path) -> Result<(), Box<dyn Error>> {
    print_sdcfr_lhe_config(&config);

    let device = poker_solver_deep_cfr::best_available_device();
    println!("  Compute device: {:?}", device);

    // Persist training config in output directory
    let output_dir = PathBuf::from(&config.training.output_dir);
    std::fs::create_dir_all(&output_dir)?;
    std::fs::copy(config_path, output_dir.join("training_config.yaml"))?;

    let lhe_config = config.lhe_game.clone().unwrap_or_default();
    let game = LimitHoldem::new(lhe_config.clone(), config.deals.count, config.deals.seed);
    let deal_pool = game.initial_states();
    println!("  Generated {} deals", deal_pool.len());

    let encoder = poker_solver_deep_cfr::lhe_encoder::LheEncoder::new();
    let sdcfr_config = build_sdcfr_config(&config);
    let mut solver = poker_solver_deep_cfr::solver::SdCfrSolver::new(game, encoder, sdcfr_config)?;

    let total_iterations = config.training.iterations;
    println!("\nStarting SD-CFR LHE training ({total_iterations} iterations)...");
    let start = Instant::now();

    for i in 1..=total_iterations {
        let iter_start = Instant::now();
        eprintln!(
            "  [iter {i}/{total_iterations}] starting (elapsed: {:.1}s)...",
            start.elapsed().as_secs_f64(),
        );

        solver.step_with_deals(Some(&deal_pool))?;

        let iter_elapsed = iter_start.elapsed();
        let total_elapsed = start.elapsed().as_secs_f64();
        let rate = f64::from(i) / total_elapsed;
        let remaining = f64::from(total_iterations - i) / rate;
        let stats = solver.buffer_stats();

        eprintln!(
            "  [iter {i}/{total_iterations}] done in {:.1}s | rate: {rate:.2} iter/s | \
             buf P1: {}/{} P2: {}/{} | ETA: {:.0}s",
            iter_elapsed.as_secs_f64(),
            stats[0].0,
            stats[0].1,
            stats[1].0,
            stats[1].1,
            remaining,
        );

        // Save LHE checkpoint at the configured interval
        let checkpoint_interval = config.checkpoint.interval;
        if checkpoint_interval > 0 && i % checkpoint_interval == 0 {
            let snap = solver.snapshot();
            save_lhe_checkpoint(
                &snap,
                i,
                &config.training.output_dir,
                config.network.num_actions,
                config.network.hidden_dim,
                &lhe_config,
                config.deals.seed,
                50,
            )?;
        }
    }

    // Final checkpoint
    let elapsed = start.elapsed();
    let trained = solver.take_result();
    save_lhe_checkpoint(
        &trained,
        total_iterations,
        &config.training.output_dir,
        config.network.num_actions,
        config.network.hidden_dim,
        &lhe_config,
        config.deals.seed,
        50,
    )?;
    print_sdcfr_summary(&trained, elapsed);
    Ok(())
}

/// Save LHE model buffers (p1.bin, p2.bin) to a checkpoint directory,
/// update the `latest_checkpoint` symlink, and print preflop strategy matrices.
#[allow(clippy::too_many_arguments)]
fn save_lhe_checkpoint(
    trained: &poker_solver_deep_cfr::solver::TrainedSdCfr,
    iteration: u32,
    output_dir: &str,
    num_actions: usize,
    hidden_dim: usize,
    lhe_config: &poker_solver_core::game::LimitHoldemConfig,
    seed: u64,
    board_samples: usize,
) -> Result<(), Box<dyn Error>> {
    let base = PathBuf::from(output_dir);
    let checkpoint_name = format!("lhe_checkpoint_{iteration}");
    let dir = base.join(&checkpoint_name);
    std::fs::create_dir_all(&dir)?;
    trained.model_buffers[0].save_to_file(&dir.join("p1.bin"))?;
    trained.model_buffers[1].save_to_file(&dir.join("p2.bin"))?;

    // Update latest_checkpoint symlink (relative target for portability)
    let link = base.join("latest_checkpoint");
    let _ = std::fs::remove_file(&link);
    #[cfg(unix)]
    std::os::unix::fs::symlink(&checkpoint_name, &link)?;

    println!("  Saved LHE checkpoint to {}", dir.display());

    // Print preflop strategy matrices
    let device = candle_core::Device::Cpu;
    let policies = [
        poker_solver_deep_cfr::eval::ExplicitPolicy::from_buffer(
            &trained.model_buffers[0],
            num_actions,
            hidden_dim,
            &device,
        )?,
        poker_solver_deep_cfr::eval::ExplicitPolicy::from_buffer(
            &trained.model_buffers[1],
            num_actions,
            hidden_dim,
            &device,
        )?,
    ];

    let rfi_matrix =
        lhe_viz::preflop_rfi_matrix(&policies, lhe_config, num_actions, board_samples, seed);
    lhe_viz::print_hand_matrix(&rfi_matrix, "SB Preflop RFI (Fold/Call/Raise)");

    let response_matrix =
        lhe_viz::preflop_response_matrix(&policies, lhe_config, num_actions, board_samples, seed);
    lhe_viz::print_hand_matrix(&response_matrix, "BB vs SB Raise (Fold/Call/3-bet)");

    Ok(())
}

// ---------------------------------------------------------------------------
// eval-lhe command
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_eval_lhe(
    dir: &Path,
    checkpoint: Option<&str>,
    eval_deals: usize,
    num_actions_override: Option<usize>,
    hidden_dim_override: Option<usize>,
    seed_override: Option<u64>,
    stack_depth_override: Option<u32>,
    num_streets_override: Option<u8>,
    board_samples: usize,
) -> Result<(), Box<dyn Error>> {
    // Try to load training config from output directory
    let config_path = dir.join("training_config.yaml");
    let saved_config = if config_path.exists() {
        let yaml = std::fs::read_to_string(&config_path)?;
        Some(serde_yaml::from_str::<SdCfrTrainingConfig>(&yaml)?)
    } else {
        None
    };

    // Resolve parameters: CLI override > saved config > hardcoded default
    let num_actions = num_actions_override
        .or(saved_config.as_ref().map(|c| c.network.num_actions))
        .unwrap_or(3);
    let hidden_dim = hidden_dim_override
        .or(saved_config.as_ref().map(|c| c.network.hidden_dim))
        .unwrap_or(64);
    let seed = seed_override
        .or(saved_config.as_ref().map(|c| c.deals.seed))
        .unwrap_or(42);
    let stack_depth = stack_depth_override
        .or(saved_config.as_ref().and_then(|c| c.lhe_game.as_ref().map(|g| g.stack_depth)))
        .unwrap_or(20);
    let num_streets = num_streets_override
        .or(saved_config.as_ref().and_then(|c| c.lhe_game.as_ref().map(|g| g.num_streets)))
        .unwrap_or(4);

    // Resolve checkpoint directory
    let checkpoint_dir = match checkpoint {
        Some(name) => dir.join(name),
        None => dir.join("latest_checkpoint"),
    };

    println!("=== LHE Model Evaluation ===");
    println!("  Directory: {}", dir.display());
    println!("  Checkpoint: {}", checkpoint_dir.display());
    if saved_config.is_some() {
        println!("  Config: loaded from training_config.yaml");
    }
    println!("  Eval deals: {eval_deals}");
    println!("  Network: num_actions={num_actions}, hidden_dim={hidden_dim}");
    println!("  LHE: stack_depth={stack_depth} BB, streets={num_streets}");
    println!("  Seed: {seed}");
    println!();

    // 1. Load model checkpoint
    let p1_path = checkpoint_dir.join("p1.bin");
    let p2_path = checkpoint_dir.join("p2.bin");
    let p1_buf = poker_solver_deep_cfr::model_buffer::ModelBuffer::load_from_file(&p1_path)?;
    let p2_buf = poker_solver_deep_cfr::model_buffer::ModelBuffer::load_from_file(&p2_path)?;
    println!(
        "  Loaded model buffers: P1={} nets, P2={} nets",
        p1_buf.len(),
        p2_buf.len(),
    );

    // 2. Build explicit policies
    let device = candle_core::Device::Cpu;
    let policies = [
        poker_solver_deep_cfr::eval::ExplicitPolicy::from_buffer(
            &p1_buf,
            num_actions,
            hidden_dim,
            &device,
        )?,
        poker_solver_deep_cfr::eval::ExplicitPolicy::from_buffer(
            &p2_buf,
            num_actions,
            hidden_dim,
            &device,
        )?,
    ];

    // 3. Create LHE game + encoder
    let lhe_config = poker_solver_core::game::LimitHoldemConfig {
        stack_depth,
        num_streets,
        ..Default::default()
    };
    let game = LimitHoldem::new(lhe_config.clone(), eval_deals, seed);
    let encoder = poker_solver_deep_cfr::lhe_encoder::LheEncoder::new();

    // 4. Generate eval deals and compute exploitability with progress bar
    let deals = game.initial_states();
    println!("\nComputing sampled exploitability ({} deals)...", deals.len());
    let start = Instant::now();

    let pb = ProgressBar::new((2 * deals.len()) as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner} [{elapsed_precise}] [{bar:40}] {pos}/{len} ({per_sec}, ETA: {eta})")
            .expect("valid progress bar template")
            .progress_chars("=> "),
    );

    use rayon::prelude::*;

    let progress = AtomicU64::new(0);
    let br_p1_sum: f64 = deals
        .par_iter()
        .map(|deal| {
            let ev = poker_solver_deep_cfr::lhe_exploitability::br_ev_for_deal(
                &game, &encoder, &policies, deal, Player::Player1, num_actions,
            );
            let prev = progress.fetch_add(1, Ordering::Relaxed);
            if prev.is_multiple_of(64) {
                pb.set_position(prev + 1);
            }
            ev
        })
        .collect::<Result<Vec<f64>, _>>()?
        .iter()
        .sum();

    let br_p2_sum: f64 = deals
        .par_iter()
        .map(|deal| {
            let ev = poker_solver_deep_cfr::lhe_exploitability::br_ev_for_deal(
                &game, &encoder, &policies, deal, Player::Player2, num_actions,
            );
            let prev = progress.fetch_add(1, Ordering::Relaxed);
            if prev.is_multiple_of(64) {
                pb.set_position(prev + 1);
            }
            ev
        })
        .collect::<Result<Vec<f64>, _>>()?
        .iter()
        .sum();

    pb.finish_and_clear();

    let n = deals.len() as f64;
    let mbb = ((br_p1_sum / n) + (br_p2_sum / n)) / 2.0 * 1000.0;
    let elapsed = start.elapsed();

    println!("  Exploitability: {mbb:.1} mbb/h");
    println!("  Computed in {:.1}s", elapsed.as_secs_f64());

    // 5. Preflop strategy visualization
    println!("\nComputing preflop strategies ({board_samples} board samples per hand)...");
    let viz_start = Instant::now();

    let rfi_matrix =
        lhe_viz::preflop_rfi_matrix(&policies, &lhe_config, num_actions, board_samples, seed);
    lhe_viz::print_hand_matrix(&rfi_matrix, "SB Preflop RFI (Fold/Call/Raise)");

    let response_matrix =
        lhe_viz::preflop_response_matrix(&policies, &lhe_config, num_actions, board_samples, seed);
    lhe_viz::print_hand_matrix(&response_matrix, "BB vs SB Raise (Fold/Call/3-bet)");

    println!(
        "\nVisualization computed in {:.1}s",
        viz_start.elapsed().as_secs_f64()
    );

    Ok(())
}

fn print_sdcfr_config(config: &SdCfrTrainingConfig) {
    println!("=== Poker Blueprint Trainer (Single Deep CFR) ===");
    println!("  Game type: {:?}", config.game_type);
    println!("  Stack depth: {} BB", config.game.stack_depth);
    println!("  Bet sizes: {:?}", config.game.bet_sizes);
    println!("  Iterations: {}", config.training.iterations);
    println!("  Traversals/iter: {}", config.training.traversals_per_iter);
    println!("  Deal pool: {} deals", config.deals.count);
    println!(
        "  Network: hidden_dim={}, num_actions={}",
        config.network.hidden_dim, config.network.num_actions
    );
    println!(
        "  SGD: steps={}, batch={}, lr={}",
        config.sgd.steps, config.sgd.batch_size, config.sgd.learning_rate
    );
    println!("  Memory cap: {}", config.memory.advantage_cap);
    println!(
        "  Parallel traversals: {}",
        config.training.parallel_traversals
    );
    println!("  Checkpoint every {} iters", config.checkpoint.interval);
}

fn print_sdcfr_lhe_config(config: &SdCfrTrainingConfig) {
    let lhe = config.lhe_game.as_ref().cloned().unwrap_or_default();
    println!("=== Poker Blueprint Trainer (SD-CFR Limit Hold'em) ===");
    println!("  Stack depth: {} BB", lhe.stack_depth);
    println!("  Streets: {}", lhe.num_streets);
    println!("  Max raises (early): {}", lhe.max_raises_early);
    println!("  Max raises (late): {}", lhe.max_raises_late);
    println!("  Small bet: {} SB units", lhe.small_bet);
    println!("  Big bet: {} SB units", lhe.big_bet);
    println!("  Iterations: {}", config.training.iterations);
    println!("  Traversals/iter: {}", config.training.traversals_per_iter);
    println!(
        "  Deal pool: {} deals (seed: {})",
        config.deals.count, config.deals.seed
    );
    println!(
        "  Network: hidden_dim={}, num_actions={}",
        config.network.hidden_dim, config.network.num_actions
    );
    println!(
        "  SGD: steps={}, batch={}, lr={}",
        config.sgd.steps, config.sgd.batch_size, config.sgd.learning_rate
    );
    println!("  Memory cap: {}", config.memory.advantage_cap);
    println!(
        "  Parallel traversals: {}",
        config.training.parallel_traversals
    );
}

fn build_sdcfr_config(config: &SdCfrTrainingConfig) -> poker_solver_deep_cfr::SdCfrConfig {
    poker_solver_deep_cfr::SdCfrConfig {
        cfr_iterations: config.training.iterations,
        traversals_per_iter: config.training.traversals_per_iter,
        advantage_memory_cap: config.memory.advantage_cap,
        hidden_dim: config.network.hidden_dim,
        num_actions: config.network.num_actions,
        sgd_steps: config.sgd.steps,
        batch_size: config.sgd.batch_size,
        learning_rate: config.sgd.learning_rate,
        grad_clip_norm: config.sgd.grad_clip_norm,
        seed: config.training.seed,
        checkpoint_interval: config.checkpoint.interval,
        parallel_traversals: config.training.parallel_traversals,
    }
}

fn print_sdcfr_summary(
    trained: &poker_solver_deep_cfr::solver::TrainedSdCfr,
    elapsed: std::time::Duration,
) {
    println!("\n=== SD-CFR Training Complete ===");
    println!("  Total time: {:.1}s", elapsed.as_secs_f64());
    println!("  P1 model snapshots: {}", trained.model_buffers[0].len(),);
    println!("  P2 model snapshots: {}", trained.model_buffers[1].len(),);
    println!(
        "  Time/iteration: {:.2}s",
        elapsed.as_secs_f64() / f64::from(trained.config.cfr_iterations),
    );
}

/// Mutable state carried across SD-CFR checkpoints for reporting.
struct SdCfrReportState<'a> {
    training_start: &'a Instant,
    header: &'a str,
    action_labels: &'a [String],
    previous_strategies: &'a mut Option<FxHashMap<u64, Vec<f64>>>,
    checkpoint_count: u64,
    total_checkpoints: u64,
    total_iterations: u64,
    stack_depth: u32,
}

fn save_sdcfr_checkpoint(
    trained: &poker_solver_deep_cfr::solver::TrainedSdCfr,
    iteration: u32,
    output_dir: &str,
    game_config: &PostflopConfig,
    num_actions: usize,
    hidden_dim: usize,
    report: &mut SdCfrReportState<'_>,
) -> Result<(), poker_solver_deep_cfr::SdCfrError> {
    let device = candle_core::Device::Cpu;

    let policies = build_explicit_policies(trained, num_actions, hidden_dim, &device)?;
    let strategy_map = walk_deals_for_strategies(game_config, &policies)?;

    let report_ctx = StrategyReportCtx {
        strategies: &strategy_map,
        previous: report.previous_strategies,
        checkpoint_num: report.checkpoint_count,
        is_convergence: false,
        total_checkpoints: report.total_checkpoints,
        total_iterations: report.total_iterations,
        current_iterations: u64::from(iteration),
        training_start: report.training_start,
        header: report.header,
        action_labels: report.action_labels,
        stack_depth: report.stack_depth,
        abs_mode: AbstractionModeConfig::default(),
        convergence_threshold: None,
        max_regret: None,
    };
    print_strategy_report(&report_ctx);

    *report.previous_strategies = Some(strategy_map.clone());

    save_strategy_bundle(game_config, strategy_map, iteration, output_dir)
}

/// Build `ExplicitPolicy` for both players from the trained model buffers.
fn build_explicit_policies(
    trained: &poker_solver_deep_cfr::solver::TrainedSdCfr,
    num_actions: usize,
    hidden_dim: usize,
    device: &candle_core::Device,
) -> Result<[poker_solver_deep_cfr::eval::ExplicitPolicy; 2], poker_solver_deep_cfr::SdCfrError> {
    let p1 = poker_solver_deep_cfr::eval::ExplicitPolicy::from_buffer(
        &trained.model_buffers[0],
        num_actions,
        hidden_dim,
        device,
    )?;
    let p2 = poker_solver_deep_cfr::eval::ExplicitPolicy::from_buffer(
        &trained.model_buffers[1],
        num_actions,
        hidden_dim,
        device,
    )?;
    Ok([p1, p2])
}

/// Sample deals and walk the game tree to extract strategies at every info set.
fn walk_deals_for_strategies(
    game_config: &PostflopConfig,
    policies: &[poker_solver_deep_cfr::eval::ExplicitPolicy; 2],
) -> Result<FxHashMap<u64, Vec<f64>>, poker_solver_deep_cfr::SdCfrError> {
    let encoder =
        poker_solver_deep_cfr::hunl_encoder::HunlStateEncoder::new(game_config.bet_sizes.clone());
    let tree_game = HunlPostflop::new(game_config.clone(), None, 200);
    let sample_deals = tree_game.initial_states();

    let mut strategy_map: FxHashMap<u64, Vec<f64>> = FxHashMap::default();
    for deal in &sample_deals {
        extract_strategies_dfs(&tree_game, deal, policies, &encoder, &mut strategy_map)?;
    }
    Ok(strategy_map)
}

/// Save the extracted strategy map as a `StrategyBundle` to disk.
fn save_strategy_bundle(
    game_config: &PostflopConfig,
    strategy_map: FxHashMap<u64, Vec<f64>>,
    iteration: u32,
    output_dir: &str,
) -> Result<(), poker_solver_deep_cfr::SdCfrError> {
    let bundle_config = BundleConfig {
        game: game_config.clone(),
        abstraction: None,
        abstraction_mode: AbstractionModeConfig::default(),
        ..BundleConfig::default()
    };
    let blueprint = BlueprintStrategy::from_strategies(strategy_map, u64::from(iteration));
    let bundle = StrategyBundle::new(bundle_config, blueprint, None);

    let dir = PathBuf::from(output_dir).join(format!("checkpoint_{iteration}"));
    bundle
        .save(&dir)
        .map_err(|e| poker_solver_deep_cfr::SdCfrError::Io(std::io::Error::other(e.to_string())))?;
    println!("    Saved to {}", dir.display());
    Ok(())
}

/// Recursive DFS through the game tree, collecting info_set_key -> action
/// probabilities from the neural network policies.
fn extract_strategies_dfs(
    game: &HunlPostflop,
    state: &PostflopState,
    policies: &[poker_solver_deep_cfr::eval::ExplicitPolicy; 2],
    encoder: &poker_solver_deep_cfr::hunl_encoder::HunlStateEncoder,
    strategy_map: &mut FxHashMap<u64, Vec<f64>>,
) -> Result<(), poker_solver_deep_cfr::SdCfrError> {
    if game.is_terminal(state) {
        return Ok(());
    }

    let key = game.info_set_key(state);
    if let std::collections::hash_map::Entry::Vacant(entry) = strategy_map.entry(key) {
        let strat = query_policy(game, state, policies, encoder)?;
        entry.insert(strat);
    }

    let actions = game.actions(state);
    for action in actions {
        let next = game.next_state(state, action);
        extract_strategies_dfs(game, &next, policies, encoder, strategy_map)?;
    }
    Ok(())
}

/// Query the neural network policy for the current player at the given state.
fn query_policy(
    game: &HunlPostflop,
    state: &PostflopState,
    policies: &[poker_solver_deep_cfr::eval::ExplicitPolicy; 2],
    encoder: &poker_solver_deep_cfr::hunl_encoder::HunlStateEncoder,
) -> Result<Vec<f64>, poker_solver_deep_cfr::SdCfrError> {
    use poker_solver_deep_cfr::traverse::StateEncoder;

    let player = game.player(state);
    let pi = match player {
        Player::Player1 => 0,
        Player::Player2 => 1,
    };
    let features = encoder.encode(state, player);
    let probs = policies[pi].strategy(&features)?;

    let actions = game.actions(state);
    let n = actions.len().min(probs.len());
    Ok(probs[..n].iter().map(|&p| f64::from(p)).collect())
}

fn run_generate_deals(
    config: TrainingConfig,
    output: &Path,
    dry_run: bool,
    batch_size: usize,
) -> Result<(), Box<dyn Error>> {
    let abs_mode = config.training.abstraction_mode;
    if abs_mode != AbstractionModeConfig::HandClassV2 {
        return Err("generate-deals requires hand_class_v2 abstraction mode".into());
    }

    let deal_config = AbstractDealConfig {
        stack_depth: config.game.stack_depth,
        strength_bits: config.training.strength_bits,
        equity_bits: config.training.equity_bits,
        max_hole_pairs: 0, // all pairs
    };

    println!("=== Abstract Deal Generator ===\n");
    println!("Config:");
    println!("  Stack depth: {} BB", deal_config.stack_depth);
    println!("  Strength bits: {}", deal_config.strength_bits);
    println!("  Equity bits: {}", deal_config.equity_bits);
    if batch_size > 0 {
        println!("  Batch size: {} flops", batch_size);
    }
    println!();

    if dry_run {
        let (est_abstract, concrete) = abstract_game::estimate_deal_count(&deal_config);
        println!("Estimated deal counts:");
        println!("  Concrete: {concrete}");
        println!("  Abstract: ~{est_abstract}");
        println!(
            "  Estimated memory: ~{:.0} MB (at 48 bytes/deal)",
            est_abstract as f64 * 48.0 / 1_048_576.0
        );
        return Ok(());
    }

    let start = Instant::now();

    if batch_size > 0 {
        run_generate_deals_batched(&config, &deal_config, output, batch_size)?;
    } else {
        run_generate_deals_in_memory(&config, &deal_config, output)?;
    }

    println!("\nTotal time: {:?}", start.elapsed());
    Ok(())
}

fn run_generate_deals_in_memory(
    config: &TrainingConfig,
    deal_config: &AbstractDealConfig,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let (deals, stats) = abstract_game::generate_abstract_deals(deal_config);

    println!("\nGeneration complete");
    println!("  Concrete deals: {}", stats.concrete_deals);
    println!("  Abstract deals: {}", stats.abstract_deals);
    println!("  Compression: {:.1}x", stats.compression_ratio);

    save_deals_and_manifest(config, &deals, &stats, output)
}

fn run_generate_deals_batched(
    config: &TrainingConfig,
    deal_config: &AbstractDealConfig,
    output: &Path,
    batch_size: usize,
) -> Result<(), Box<dyn Error>> {
    use abstract_game::DealGenContext;

    let canonical_flops = flops::all_flops();
    let total_flops = canonical_flops.len();
    let num_batches = total_flops.div_ceil(batch_size);

    println!("Building deal generation context...");
    let ctx = DealGenContext::new(deal_config);

    let batches_dir = output.join("batches");
    std::fs::create_dir_all(&batches_dir)?;

    println!(
        "Processing {} flops in {} batches of up to {}...\n",
        total_flops, num_batches, batch_size
    );

    let mut batch_paths = Vec::with_capacity(num_batches);
    let mut total_entries = 0usize;
    let mut total_concrete = 0u64;

    for (i, chunk) in canonical_flops.chunks(batch_size).enumerate() {
        let batch_path = batches_dir.join(format!("batch_{i:04}.bin"));
        let batch_start = Instant::now();

        let (entries, concrete) =
            abstract_game::generate_deals_batch(&ctx, deal_config, chunk, i as u32, &batch_path)?;

        total_entries += entries;
        total_concrete += concrete;
        batch_paths.push(batch_path);

        println!(
            "  batch {}/{}: {} flops, {} entries, {} concrete deals ({:?})",
            i + 1,
            num_batches,
            chunk.len(),
            entries,
            concrete,
            batch_start.elapsed(),
        );
    }

    println!(
        "\nAll batches written ({} total entries, {} concrete deals)",
        total_entries, total_concrete
    );
    println!("Merging batches...");

    let deals_path = output.join("abstract_deals.bin");
    let stats = abstract_game::merge_deal_batches(&batch_paths, &deals_path)?;

    println!(
        "  {} concrete -> {} abstract deals ({:.1}x compression)",
        stats.concrete_deals, stats.abstract_deals, stats.compression_ratio
    );

    save_manifest(config, &stats, output)?;
    let file_size = std::fs::metadata(&deals_path)?.len();
    println!(
        "  Saved to {}/  ({:.2} MB)",
        output.display(),
        file_size as f64 / 1_048_576.0
    );

    Ok(())
}

fn save_deals_and_manifest(
    config: &TrainingConfig,
    deals: &[abstract_game::AbstractDeal],
    stats: &abstract_game::GenerationStats,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(output)?;
    let deals_path = output.join("abstract_deals.bin");
    let data = bincode::serialize(
        &deals
            .iter()
            .map(|d| (d.hand_bits_p1, d.hand_bits_p2, d.p1_equity, d.weight))
            .collect::<Vec<_>>(),
    )?;
    std::fs::write(&deals_path, &data)?;

    save_manifest(config, stats, output)?;

    println!("  Saved to {}/", output.display());
    println!("  File size: {:.2} MB", data.len() as f64 / 1_048_576.0);
    Ok(())
}

fn save_manifest(
    config: &TrainingConfig,
    stats: &abstract_game::GenerationStats,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let manifest = serde_yaml::to_string(&serde_yaml::Value::Mapping({
        let mut m = serde_yaml::Mapping::new();
        m.insert("stack_depth".into(), config.game.stack_depth.into());
        m.insert("strength_bits".into(), config.training.strength_bits.into());
        m.insert("equity_bits".into(), config.training.equity_bits.into());
        m.insert(
            "concrete_deals".into(),
            (stats.concrete_deals as i64).into(),
        );
        m.insert(
            "abstract_deals".into(),
            (stats.abstract_deals as i64).into(),
        );
        m.insert(
            "compression_ratio".into(),
            serde_yaml::Value::Number(serde_yaml::Number::from(stats.compression_ratio)),
        );
        m
    }))?;
    std::fs::write(output.join("manifest.yaml"), manifest)?;
    Ok(())
}

fn run_merge_deals(input: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    println!("=== Merge Deal Batches ===\n");
    println!("Input:  {}/", input.display());
    println!("Output: {}/\n", output.display());

    let mut batch_paths: Vec<PathBuf> = std::fs::read_dir(input)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|ext| ext == "bin")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("batch_"))
        })
        .collect();
    batch_paths.sort();

    if batch_paths.is_empty() {
        return Err(format!("no batch_*.bin files found in {}", input.display()).into());
    }

    println!("Found {} batch files", batch_paths.len());

    std::fs::create_dir_all(output)?;
    let deals_path = output.join("abstract_deals.bin");

    let start = Instant::now();
    let stats = abstract_game::merge_deal_batches(&batch_paths, &deals_path)?;
    let elapsed = start.elapsed();

    println!("\nMerge complete in {elapsed:?}");
    println!("  Concrete deals: {}", stats.concrete_deals);
    println!("  Abstract deals: {}", stats.abstract_deals);
    println!("  Compression: {:.1}x", stats.compression_ratio);

    let file_size = std::fs::metadata(&deals_path)?.len();
    println!(
        "  Saved to {}/abstract_deals.bin ({:.2} MB)",
        output.display(),
        file_size as f64 / 1_048_576.0
    );

    Ok(())
}

fn run_tree_stats(config: TrainingConfig, sample_deals: usize) -> Result<(), Box<dyn Error>> {
    println!("=== Game Tree Statistics ===\n");
    println!("Game config:");
    println!("  Stack depth: {} BB", config.game.stack_depth);
    println!("  Bet sizes: {:?} pot", config.game.bet_sizes);
    println!();

    // Build game with a small deal pool (just need the tree structure)
    let abs_mode = config.training.abstraction_mode;
    let abstraction_mode = build_abstraction_mode(&config);

    let game = HunlPostflop::new(config.game.clone(), abstraction_mode, sample_deals);
    let states = game.initial_states();

    println!("Materializing tree from first deal...");
    let start = Instant::now();
    let tree = materialize_postflop(&game, &states[0]);
    let elapsed = start.elapsed();
    println!("  Done in {elapsed:?}\n");

    println!("{}", tree.stats);

    let position_keys = tree.unique_position_keys();
    println!("Unique position keys: {position_keys}");
    println!("  (action-history positions ignoring hand bits)\n");

    // Estimate info sets for different abstraction sizes
    let hand_class_count = 28u32;
    println!("Estimated info sets (upper bound):");
    println!(
        "  hand_class ({hand_class_count} classes): {}",
        tree.estimated_info_sets(hand_class_count)
    );

    // For hand_class_v2, estimate based on configured bits
    if abs_mode == AbstractionModeConfig::HandClassV2 {
        let class_bits = 5u32;
        let strength_bins = 1u32 << config.training.strength_bits;
        let equity_bins = 1u32 << config.training.equity_bits;
        let draw_combos = 128u32; // 7 draw flag bits
        let v2_combos = (1 << class_bits) * strength_bins * equity_bins * draw_combos;
        println!(
            "  hand_class_v2 (s={}, e={}, 7 draw bits): {} (theoretical max)",
            config.training.strength_bits,
            config.training.equity_bits,
            tree.estimated_info_sets(v2_combos)
        );
    }
    println!();

    // Count actual unique info set keys across sampled deals
    println!("Sampling {sample_deals} deals to count actual info sets...");
    let start = Instant::now();
    let actual_info_sets = count_actual_info_sets(&game, &tree, &states, sample_deals);
    let elapsed = start.elapsed();
    println!("  Unique info set keys found: {actual_info_sets}");
    println!("  Done in {elapsed:?}\n");

    // Memory estimates
    let node_bytes =
        tree.nodes.len() * std::mem::size_of::<poker_solver_core::cfr::game_tree::TreeNode>();
    let strategy_bytes_est = actual_info_sets as usize * tree.nodes[0].children.len() * 8;
    println!("Memory estimates:");
    println!("  Tree nodes: {:.2} MB", node_bytes as f64 / 1_048_576.0);
    println!(
        "  Strategy arrays (f64): {:.2} MB",
        strategy_bytes_est as f64 / 1_048_576.0
    );

    Ok(())
}

fn run_sequence_training(config: TrainingConfig) -> Result<(), Box<dyn Error>> {
    let abs_mode = config.training.abstraction_mode;
    let abstraction_mode = build_abstraction_mode(&config);
    let abstraction_for_deals = build_abstraction_mode(&config);

    println!("=== Poker Blueprint Trainer (Sequence-Form CFR) ===\n");
    println!("Game config:");
    println!("  Stack depth: {} BB", config.game.stack_depth);
    println!("  Bet sizes: {:?} pot", config.game.bet_sizes);
    if config.training.exhaustive {
        println!("  Mode: exhaustive abstract deals");
    } else {
        println!("  Deal pool size: {}", config.training.deal_count);
    }
    println!();

    let iterations = if let Some(threshold) = config.training.convergence_threshold {
        println!("Convergence mode: delta < {threshold:.6}");
        u64::MAX // will stop when converged
    } else if config.training.iterations > 0 {
        config.training.iterations
    } else {
        return Err("Either iterations or convergence_threshold must be set".into());
    };

    // Build deals: either exhaustive abstract or random concrete
    let deals = if config.training.exhaustive {
        build_exhaustive_deals(&config)?
    } else if let Some(ref dir) = config.training.abstract_deals_dir {
        load_abstract_deals(dir)?
    } else {
        let mut game = HunlPostflop::new(
            config.game.clone(),
            abstraction_mode.clone(),
            config.training.deal_count,
        );
        if config.training.min_deals_per_class > 0 {
            game = game.with_stratification(
                config.training.min_deals_per_class,
                config.training.max_rejections_per_class,
            );
        }
        let states = game.initial_states();
        println!("Building deal info for {} deals...", states.len());
        let start = Instant::now();
        let deals = build_deal_infos(&game, &states, &abstraction_for_deals);
        println!("  Done in {:?}", start.elapsed());
        deals
    };

    println!("  {} deals loaded\n", deals.len());

    // Materialize tree (abstraction mode not needed — tree shape depends only on bet sizes/stacks)
    let tree_game = HunlPostflop::new(config.game.clone(), None, 1);
    let tree_states = tree_game.initial_states();

    println!("Materializing game tree...");
    let start = Instant::now();
    let tree = materialize_postflop(&tree_game, &tree_states[0]);
    println!(
        "  {} nodes, done in {:?}",
        tree.stats.total_nodes,
        start.elapsed()
    );

    // Build solver
    let seq_config = match config.training.cfr_variant {
        CfrVariant::Linear => SequenceCfrConfig::linear_cfr(),
        CfrVariant::Dcfr => SequenceCfrConfig::default(),
    };
    let solver = SequenceCfrSolver::new(tree, deals, seq_config);

    // Build bundle config for saves
    let bundle_config = BundleConfig {
        game: config.game.clone(),
        abstraction: config.abstraction.clone(),
        abstraction_mode: abs_mode,
        strength_bits: if abs_mode == AbstractionModeConfig::HandClassV2 {
            config.training.strength_bits
        } else {
            0
        },
        equity_bits: if abs_mode == AbstractionModeConfig::HandClassV2 {
            config.training.equity_bits
        } else {
            0
        },
        ..BundleConfig::default()
    };

    let boundaries = None; // sequence solver doesn't use EHS2

    // Derive action labels from a single deal (all deals share the same preflop actions)
    let label_game = HunlPostflop::new(config.game.clone(), None, 1);
    let label_states = label_game.initial_states();
    let actions = label_game.actions(&label_states[0]);
    let action_labels = format_action_labels(&actions);
    let header = format_table_header(&action_labels);

    let mut wrapper = SimpleTrainingSolver {
        solver,
        prefix: "seq_checkpoint_",
        previous: None,
        training_start: Instant::now(),
        header,
        action_labels,
        stack_depth: config.game.stack_depth,
        abs_mode,
        convergence_threshold: config.training.convergence_threshold,
    };

    let loop_config = TrainingLoopConfig {
        convergence_threshold: config.training.convergence_threshold,
        check_interval: config.training.convergence_check_interval,
        iterations,
        output_dir: &config.training.output_dir,
        bundle_config: &bundle_config,
        boundaries: &boundaries,
    };

    run_training_loop(&mut wrapper, &loop_config)
}

#[cfg(feature = "gpu")]
fn run_gpu_training(config: TrainingConfig) -> Result<(), Box<dyn Error>> {
    use poker_solver_gpu_cfr::GpuCfrConfig;
    use poker_solver_gpu_cfr::tabular::TabularGpuCfrSolver;
    use poker_solver_gpu_cfr::tiled::TiledTabularGpuCfrSolver;

    let abs_mode = config.training.abstraction_mode;
    let abstraction_mode = build_abstraction_mode(&config);
    let abstraction_for_deals = build_abstraction_mode(&config);

    println!("=== Poker Blueprint Trainer (Tabular GPU CFR) ===\n");
    println!("Game config:");
    println!("  Stack depth: {} BB", config.game.stack_depth);
    println!("  Bet sizes: {:?} pot", config.game.bet_sizes);
    if config.training.exhaustive {
        println!("  Mode: exhaustive abstract deals");
    } else {
        println!("  Deal pool size: {}", config.training.deal_count);
    }
    println!();

    let iterations = if let Some(threshold) = config.training.convergence_threshold {
        println!("Convergence mode: delta < {threshold:.6}");
        u64::MAX
    } else if config.training.iterations > 0 {
        config.training.iterations
    } else {
        return Err("Either iterations or convergence_threshold must be set".into());
    };

    // Build deals: either exhaustive abstract or random concrete
    println!("Step 1/4: Loading deals...");
    let step_start = Instant::now();
    let deals = if config.training.exhaustive {
        build_exhaustive_deals(&config)?
    } else if let Some(ref dir) = config.training.abstract_deals_dir {
        load_abstract_deals(dir)?
    } else {
        let mut game = HunlPostflop::new(
            config.game.clone(),
            abstraction_mode.clone(),
            config.training.deal_count,
        );
        if config.training.min_deals_per_class > 0 {
            game = game.with_stratification(
                config.training.min_deals_per_class,
                config.training.max_rejections_per_class,
            );
        }
        let states = game.initial_states();
        println!("  Building deal info for {} deals...", states.len());
        let start = Instant::now();
        let deals = build_deal_infos(&game, &states, &abstraction_for_deals);
        println!("  Deal info built in {:?}", start.elapsed());
        deals
    };
    println!(
        "  {} deals loaded in {:?}\n",
        deals.len(),
        step_start.elapsed()
    );

    println!("Step 2/4: Materializing game tree...");
    let start = Instant::now();
    let tree_game = HunlPostflop::new(config.game.clone(), None, 1);
    let tree_states = tree_game.initial_states();
    let tree = materialize_postflop(&tree_game, &tree_states[0]);
    println!(
        "  {} nodes, done in {:?}\n",
        tree.stats.total_nodes,
        start.elapsed()
    );

    println!("Step 3/4: Initializing tabular GPU solver...");
    let start = Instant::now();
    let gpu_config = GpuCfrConfig {
        dcfr_alpha: 1.5,
        dcfr_beta: 0.5,
        dcfr_gamma: 2.0,
        tile_size: config.training.tile_size,
        ..Default::default()
    };

    // Choose tiled vs untiled solver based on tile_size config.
    // tile_size: None => auto-detect (use tiled if coupling matrices too large)
    // tile_size: Some(0) => force untiled
    // tile_size: Some(n) => force tiled with given tile size
    let use_tiled = match config.training.tile_size {
        Some(0) => false,
        Some(_) => true,
        None => {
            // Auto-detect: use tiled if coupling matrices would exceed ~8 GB
            let n1 = deals
                .iter()
                .map(|d| d.hand_bits_p1)
                .collect::<std::collections::HashSet<_>>()
                .len() as u64;
            let n2 = deals
                .iter()
                .map(|d| d.hand_bits_p2)
                .collect::<std::collections::HashSet<_>>()
                .len() as u64;
            let coupling_bytes = 6 * n1 * n2 * 4; // 6 matrices (3 + 3 transposed) x f32
            coupling_bytes > 8 * 1024 * 1024 * 1024
        }
    };

    // Build bundle config for saves (shared by both paths)
    let bundle_config = BundleConfig {
        game: config.game.clone(),
        abstraction: config.abstraction.clone(),
        abstraction_mode: abs_mode,
        strength_bits: if abs_mode == AbstractionModeConfig::HandClassV2 {
            config.training.strength_bits
        } else {
            0
        },
        equity_bits: if abs_mode == AbstractionModeConfig::HandClassV2 {
            config.training.equity_bits
        } else {
            0
        },
        ..BundleConfig::default()
    };
    let boundaries = None;
    let actions = tree_game.actions(&tree_states[0]);
    let action_labels = format_action_labels(&actions);
    let header = format_table_header(&action_labels);

    if use_tiled {
        let tile_sz = config.training.tile_size.unwrap_or(32_768);
        println!("  Using tiled solver (tile_size={tile_sz})");
        let solver = TiledTabularGpuCfrSolver::new(&tree, deals, gpu_config)?;
        println!(
            "  {} info sets, initialized in {:?}\n",
            solver.num_info_sets(),
            start.elapsed()
        );
        let mut wrapper = SimpleTrainingSolver {
            solver,
            prefix: "gpu_checkpoint_",
            previous: None,
            training_start: Instant::now(),
            header,
            action_labels,
            stack_depth: config.game.stack_depth,
            abs_mode,
            convergence_threshold: config.training.convergence_threshold,
        };
        let loop_config = TrainingLoopConfig {
            convergence_threshold: config.training.convergence_threshold,
            check_interval: config.training.convergence_check_interval,
            iterations,
            output_dir: &config.training.output_dir,
            bundle_config: &bundle_config,
            boundaries: &boundaries,
        };
        println!("Step 4/4: Training ({iterations} iterations)...\n");
        run_training_loop(&mut wrapper, &loop_config)
    } else {
        println!("  Using untiled solver");
        let solver = TabularGpuCfrSolver::new(&tree, deals, gpu_config)?;
        println!(
            "  {} info sets, initialized in {:?}\n",
            solver.num_info_sets(),
            start.elapsed()
        );
        let mut wrapper = SimpleTrainingSolver {
            solver,
            prefix: "gpu_checkpoint_",
            previous: None,
            training_start: Instant::now(),
            header,
            action_labels,
            stack_depth: config.game.stack_depth,
            abs_mode,
            convergence_threshold: config.training.convergence_threshold,
        };
        let loop_config = TrainingLoopConfig {
            convergence_threshold: config.training.convergence_threshold,
            check_interval: config.training.convergence_check_interval,
            iterations,
            output_dir: &config.training.output_dir,
            bundle_config: &bundle_config,
            boundaries: &boundaries,
        };
        println!("Step 4/4: Training ({iterations} iterations)...\n");
        run_training_loop(&mut wrapper, &loop_config)
    }
}

fn save_checkpoint(
    strategies: FxHashMap<u64, Vec<f64>>,
    iterations: u64,
    dir_name: &str,
    output_dir: &str,
    bundle_config: &BundleConfig,
    boundaries: &Option<poker_solver_core::abstraction::BucketBoundaries>,
) {
    let dir = PathBuf::from(output_dir).join(dir_name);
    let blueprint = BlueprintStrategy::from_strategies(strategies, iterations);
    let bundle = StrategyBundle::new(bundle_config.clone(), blueprint, boundaries.clone());

    match bundle.save(&dir) {
        Ok(()) => println!("  Saved checkpoint to {}/", dir.display()),
        Err(e) => eprintln!("  Warning: failed to save checkpoint: {e}"),
    }
}

fn build_deal_infos(
    _game: &HunlPostflop,
    states: &[PostflopState],
    abstraction: &Option<AbstractionMode>,
) -> Vec<DealInfo> {
    states
        .iter()
        .map(|state| {
            let hand_bits_p1 = compute_per_street_hand_bits(state, true, abstraction);
            let hand_bits_p2 = compute_per_street_hand_bits(state, false, abstraction);

            let p1_equity = match (&state.p1_cache.rank, &state.p2_cache.rank) {
                (Some(r1), Some(r2)) => {
                    use std::cmp::Ordering;
                    match r1.cmp(r2) {
                        Ordering::Greater => 1.0,
                        Ordering::Less => 0.0,
                        Ordering::Equal => 0.5,
                    }
                }
                _ => 0.5,
            };

            DealInfo {
                hand_bits_p1,
                hand_bits_p2,
                p1_equity,
                weight: 1.0,
            }
        })
        .collect()
}

/// Compute hand bits for all 4 streets (preflop, flop, turn, river).
///
/// Preflop always uses canonical_hand_index. Postflop streets use the
/// abstraction mode to encode the hand classification.
fn compute_per_street_hand_bits(
    state: &PostflopState,
    is_p1: bool,
    abstraction: &Option<AbstractionMode>,
) -> [u32; 4] {
    use poker_solver_core::info_key::{canonical_hand_index, compute_hand_bits_v2};

    let holding = if is_p1 {
        state.p1_holding
    } else {
        state.p2_holding
    };
    let preflop_bits = u32::from(canonical_hand_index(holding));

    let Some(full_board) = state.full_board else {
        return [preflop_bits; 4];
    };

    let postflop_bits = |board: &[poker_solver_core::poker::Card]| -> u32 {
        match abstraction {
            Some(AbstractionMode::HandClassV2 {
                strength_bits,
                equity_bits,
            }) => compute_hand_bits_v2(holding, board, *strength_bits, *equity_bits),
            _ => preflop_bits,
        }
    };

    [
        preflop_bits,
        postflop_bits(&full_board[..3]),
        postflop_bits(&full_board[..4]),
        postflop_bits(&full_board[..5]),
    ]
}

/// Build abstract deals via exhaustive enumeration.
fn build_exhaustive_deals(config: &TrainingConfig) -> Result<Vec<DealInfo>, Box<dyn Error>> {
    if config.training.abstraction_mode != AbstractionModeConfig::HandClassV2 {
        return Err("Exhaustive deals require hand_class_v2 abstraction mode".into());
    }

    let deal_config = AbstractDealConfig {
        stack_depth: config.game.stack_depth,
        strength_bits: config.training.strength_bits,
        equity_bits: config.training.equity_bits,
        max_hole_pairs: 0,
    };

    println!("Generating exhaustive abstract deals...");
    let start = Instant::now();
    let (deals, stats) = abstract_game::generate_abstract_deals(&deal_config);
    println!(
        "  {} concrete → {} abstract ({:.1}x compression) in {:?}",
        stats.concrete_deals,
        stats.abstract_deals,
        stats.compression_ratio,
        start.elapsed()
    );

    Ok(abstract_game::to_deal_infos(&deals))
}

/// Load pre-generated abstract deals from a directory.
///
/// Auto-detects format by reading the first 4 bytes:
/// - `b"DEAL"` → new raw binary format from streaming merge
/// - Otherwise → legacy bincode `Vec<(p1, p2, eq, w)>` format
fn load_abstract_deals(dir: &str) -> Result<Vec<DealInfo>, Box<dyn Error>> {
    let deals_path = PathBuf::from(dir).join("abstract_deals.bin");
    println!("Loading abstract deals from {}...", deals_path.display());
    let start = Instant::now();

    // Peek at first 4 bytes to detect format
    let mut file = std::fs::File::open(&deals_path)?;
    let mut magic = [0u8; 4];
    use std::io::Read;
    file.read_exact(&mut magic)?;
    drop(file);

    let deals = if &magic == b"DEAL" {
        println!("  detected raw binary (DEAL) format");
        abstract_game::load_raw_deal_file(&deals_path)?
    } else {
        println!("  detected legacy bincode format");
        let data = std::fs::read(&deals_path)?;
        let tuples: Vec<([u32; 4], [u32; 4], f64, f64)> = bincode::deserialize(&data)?;
        tuples
            .into_iter()
            .map(|(p1, p2, eq, w)| DealInfo {
                hand_bits_p1: p1,
                hand_bits_p2: p2,
                p1_equity: eq,
                weight: w,
            })
            .collect()
    };

    println!("  {} deals loaded in {:?}", deals.len(), start.elapsed());
    Ok(deals)
}

fn run_inspect_deals(
    input: &std::path::Path,
    limit: usize,
    sort: DealSortOrder,
    csv: Option<&std::path::Path>,
) -> Result<(), Box<dyn Error>> {
    // Load manifest
    let manifest_path = input.join("manifest.yaml");
    let manifest_str = std::fs::read_to_string(&manifest_path)?;
    let manifest: serde_yaml::Value = serde_yaml::from_str(&manifest_str)?;

    println!("=== Abstract Deal Inspector ===\n");
    println!("Source: {}/", input.display());
    print_manifest(&manifest);

    // Load deals
    let deals = load_abstract_deals(&input.to_string_lossy())?;

    // Compute and print statistics
    let equities: Vec<f64> = deals.iter().map(|d| d.p1_equity).collect();
    let weights: Vec<f64> = deals.iter().map(|d| d.weight).collect();

    println!("\n--- Summary Statistics ---");
    println!("Total deals: {}", deals.len());
    println!("Total weight: {:.1}", weights.iter().sum::<f64>());
    println!();

    print_equity_stats(&equities);
    print_weight_stats(&weights);
    print_class_histogram(&deals);

    // Print sample deals
    if limit > 0 {
        print_sample_deals(&deals, limit, &sort);
    }

    // Optional CSV export
    if let Some(csv_path) = csv {
        export_csv(&deals, csv_path)?;
    }

    Ok(())
}

fn print_manifest(manifest: &serde_yaml::Value) {
    println!("Manifest:");
    if let Some(map) = manifest.as_mapping() {
        for (k, v) in map {
            if let Some(key) = k.as_str() {
                match v {
                    serde_yaml::Value::Number(n) => println!("  {key}: {n}"),
                    serde_yaml::Value::String(s) => println!("  {key}: {s}"),
                    serde_yaml::Value::Bool(b) => println!("  {key}: {b}"),
                    other => println!("  {key}: {other:?}"),
                }
            }
        }
    }
}

fn print_equity_stats(equities: &[f64]) {
    if equities.is_empty() {
        return;
    }
    let min = equities.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = equities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = equities.iter().sum::<f64>() / equities.len() as f64;
    let median = {
        let mut sorted = equities.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    };
    let variance = equities.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / equities.len() as f64;
    let stddev = variance.sqrt();

    println!("Equity distribution:");
    println!(
        "  min={min:.4}  max={max:.4}  mean={mean:.4}  median={median:.4}  stddev={stddev:.4}"
    );
}

fn print_weight_stats(weights: &[f64]) {
    if weights.is_empty() {
        return;
    }
    let min = weights.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = weights.iter().sum::<f64>() / weights.len() as f64;

    println!("Weight distribution:");
    println!("  min={min:.4}  max={max:.4}  mean={mean:.4}");
}

fn print_class_histogram(deals: &[DealInfo]) {
    let mut counts: FxHashMap<u8, usize> = FxHashMap::default();
    for deal in deals {
        let river_bits = deal.hand_bits_p1[3];
        let class_id = ((river_bits >> 23) & 0x1F) as u8;
        *counts.entry(class_id).or_default() += 1;
    }

    let mut entries: Vec<(u8, usize)> = counts.into_iter().collect();
    entries.sort_by_key(|e| std::cmp::Reverse(e.1));

    println!("\nP1 river hand class frequency:");
    println!("  {:<20} {:>8} {:>7}", "Class", "Count", "%");
    println!("  {}", "-".repeat(37));
    let total = deals.len() as f64;
    for (class_id, count) in &entries {
        let name = HandClass::ALL
            .get(*class_id as usize)
            .map_or_else(|| format!("class{class_id}"), |c| c.to_string());
        let pct = *count as f64 / total * 100.0;
        println!("  {name:<20} {count:>8} {pct:>6.1}%");
    }
}

fn decode_trajectory(bits: [u32; 4]) -> [String; 4] {
    let preflop = reverse_canonical_index(bits[0] as u16).to_string();
    let flop = hand_label_from_bits(bits[1], 1, AbstractionModeConfig::HandClassV2);
    let turn = hand_label_from_bits(bits[2], 2, AbstractionModeConfig::HandClassV2);
    let river = hand_label_from_bits(bits[3], 3, AbstractionModeConfig::HandClassV2);
    [preflop, flop, turn, river]
}

fn print_sample_deals(deals: &[DealInfo], limit: usize, sort: &DealSortOrder) {
    let mut indices: Vec<usize> = (0..deals.len()).collect();
    match sort {
        DealSortOrder::Weight => {
            indices.sort_by(|&a, &b| {
                deals[b]
                    .weight
                    .partial_cmp(&deals[a].weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        DealSortOrder::Equity => {
            indices.sort_by(|&a, &b| {
                deals[b]
                    .p1_equity
                    .partial_cmp(&deals[a].p1_equity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        DealSortOrder::None => {}
    }

    let n = limit.min(indices.len());
    println!("\n--- Sample Deals (top {n} by {sort:?}) ---");
    for (rank, &idx) in indices.iter().take(n).enumerate() {
        let deal = &deals[idx];
        let p1 = decode_trajectory(deal.hand_bits_p1);
        let p2 = decode_trajectory(deal.hand_bits_p2);
        println!(
            "\nDeal #{} (weight={:.1}, equity={:.4}):",
            rank + 1,
            deal.weight,
            deal.p1_equity,
        );
        println!("  P1: {} -> {} -> {} -> {}", p1[0], p1[1], p1[2], p1[3]);
        println!("  P2: {} -> {} -> {} -> {}", p2[0], p2[1], p2[2], p2[3]);
    }
    println!();
}

fn export_csv(deals: &[DealInfo], path: &std::path::Path) -> Result<(), Box<dyn Error>> {
    let mut file = std::fs::File::create(path)?;
    writeln!(
        file,
        "p1_preflop,p1_flop,p1_turn,p1_river,p2_preflop,p2_flop,p2_turn,p2_river,equity,weight"
    )?;
    for deal in deals {
        let p1 = decode_trajectory(deal.hand_bits_p1);
        let p2 = decode_trajectory(deal.hand_bits_p2);
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{:.6},{:.6}",
            p1[0], p1[1], p1[2], p1[3], p2[0], p2[1], p2[2], p2[3], deal.p1_equity, deal.weight,
        )?;
    }
    println!("Exported {} deals to {}", deals.len(), path.display());
    Ok(())
}

fn build_abstraction_mode(config: &TrainingConfig) -> Option<AbstractionMode> {
    let abs_mode = config.training.abstraction_mode;
    if abs_mode == AbstractionModeConfig::HandClassV2 {
        Some(AbstractionMode::HandClassV2 {
            strength_bits: config.training.strength_bits,
            equity_bits: config.training.equity_bits,
        })
    } else {
        None
    }
}

fn count_actual_info_sets(
    game: &HunlPostflop,
    _tree: &poker_solver_core::cfr::GameTree,
    states: &[PostflopState],
    sample_count: usize,
) -> u64 {
    use std::collections::HashSet;

    let mut seen = HashSet::new();

    for state in states.iter().take(sample_count) {
        collect_info_keys_dfs(game, state, &mut seen);
    }

    seen.len() as u64
}

fn collect_info_keys_dfs(
    game: &HunlPostflop,
    state: &PostflopState,
    seen: &mut std::collections::HashSet<u64>,
) {
    if game.is_terminal(state) {
        return;
    }

    let key = game.info_set_key(state);
    seen.insert(key);

    for &action in &game.actions(state) {
        let next = game.next_state(state, action);
        collect_info_keys_dfs(game, &next, seen);
    }
}

fn format_action_labels(actions: &[Action]) -> Vec<String> {
    use poker_solver_core::game::ALL_IN;
    actions
        .iter()
        .map(|a| match a {
            Action::Fold => "Fold".to_string(),
            Action::Check => "Check".to_string(),
            Action::Call => "Call".to_string(),
            Action::Bet(idx) if *idx == ALL_IN => "B-AI".to_string(),
            Action::Bet(idx) => format!("B{idx}"),
            Action::Raise(idx) if *idx == ALL_IN => "R-AI".to_string(),
            Action::Raise(idx) => format!("R{idx}"),
        })
        .collect()
}

fn format_table_header(labels: &[String]) -> String {
    let action_cols: String = labels.iter().map(|l| format!("{:>6}", l)).collect();
    format!("{:<6}|{action_cols}", "Hand")
}

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

/// Context for MCCFR-specific checkpoint extras (regret data, pruning stats).
struct MccfrCheckpointCtx<'a> {
    output_dir: &'a str,
}

/// Context for strategy-based checkpoint output shared by all solvers.
struct StrategyReportCtx<'a> {
    strategies: &'a FxHashMap<u64, Vec<f64>>,
    previous: &'a Option<FxHashMap<u64, Vec<f64>>>,
    checkpoint_num: u64,
    is_convergence: bool,
    total_checkpoints: u64,
    total_iterations: u64,
    current_iterations: u64,
    training_start: &'a Instant,
    header: &'a str,
    action_labels: &'a [String],
    stack_depth: u32,
    abs_mode: AbstractionModeConfig,
    convergence_threshold: Option<f64>,
    /// GPU-computed max regret (upper bound on exploitability). `None` for CPU solvers.
    max_regret: Option<f32>,
}

/// Print strategy-based checkpoint output shared by all solvers.
///
/// Prints: header, info set count, time/ETA, strategy delta, entropy,
/// preflop strategy table, and river strategy tables.
/// Returns the strategy delta (if a previous checkpoint exists).
fn print_strategy_report(ctx: &StrategyReportCtx) -> Option<f64> {
    let elapsed = ctx.training_start.elapsed().as_secs_f64();

    print_checkpoint_header(ctx);
    println!("Info sets: {}", ctx.strategies.len());
    print_time_estimate(ctx.checkpoint_num, elapsed, ctx);

    let delta = if ctx.checkpoint_num > 0 {
        compute_and_print_convergence(ctx)
    } else {
        None
    };

    print_preflop_table(ctx);
    print_flop_strategies(ctx.strategies, ctx.abs_mode);
    print_river_strategies(ctx.strategies, ctx.abs_mode);

    delta
}

fn print_checkpoint_header(ctx: &StrategyReportCtx) {
    if ctx.is_convergence {
        println!(
            "\n=== Convergence Check {} ({} iterations) ===",
            ctx.checkpoint_num, ctx.current_iterations
        );
    } else {
        let current_iter = if ctx.checkpoint_num == 0 {
            0
        } else {
            (ctx.total_iterations / ctx.total_checkpoints) * ctx.checkpoint_num
        };
        println!(
            "\n=== Checkpoint {}/{} ({}/{} iterations) ===",
            ctx.checkpoint_num, ctx.total_checkpoints, current_iter, ctx.total_iterations
        );
    }
}

fn print_time_estimate(checkpoint: u64, elapsed: f64, ctx: &StrategyReportCtx) {
    if checkpoint == 0 {
        return;
    }
    if !ctx.is_convergence && ctx.total_checkpoints > 0 {
        let rate = elapsed / checkpoint as f64;
        let remaining = rate * (ctx.total_checkpoints - checkpoint) as f64;
        println!("Time: {elapsed:.1}s elapsed, ~{remaining:.1}s remaining");
    } else {
        println!("Time: {elapsed:.1}s elapsed");
    }
}

fn compute_and_print_convergence(ctx: &StrategyReportCtx) -> Option<f64> {
    let entropy = convergence::strategy_entropy(ctx.strategies);
    let delta = ctx
        .previous
        .as_ref()
        .map(|prev| convergence::strategy_delta(prev, ctx.strategies));
    let delta_str = match delta {
        Some(d) => format!("{d:.6}"),
        None => "(first checkpoint)".to_string(),
    };

    println!("\nConvergence Metrics:");
    if let Some(mr) = ctx.max_regret {
        println!("  Max regret:       {mr:.6}");
    }
    println!("  Strategy delta:   {delta_str}");
    if let Some(threshold) = ctx.convergence_threshold {
        let status = match delta {
            Some(d) if d < threshold => "CONVERGED",
            _ => "not converged",
        };
        println!("  Target delta:     {threshold:.6} ({status})");
    }
    println!("  Strategy entropy: {entropy:.4}");

    delta
}

fn print_preflop_table(ctx: &StrategyReportCtx) {
    println!("\nSB Opening Strategy (preflop, facing BB):");
    println!("{}", ctx.header);
    println!("{}", "-".repeat(ctx.header.len()));

    let preflop_eff_stack = ctx.stack_depth * 2 - 2;
    let spr_b = spr_bucket(3, preflop_eff_stack);

    for &hand in DISPLAY_HANDS {
        if let Some(hand_idx) = canonical_hand_index_from_str(hand) {
            let info_key = InfoKey::new(u32::from(hand_idx), 0, spr_b, &[]).as_u64();
            if let Some(probs) = ctx.strategies.get(&info_key) {
                let prob_cols: String = probs
                    .iter()
                    .take(ctx.action_labels.len())
                    .map(|p| format!("{:>6.2}", p))
                    .collect();
                println!("{:<6}|{prob_cols}", hand);
            } else {
                println!("{:<6}|  (no data)", hand);
            }
        } else {
            println!("{:<6}|  (invalid hand)", hand);
        }
    }
    println!();
}

/// Print MCCFR-specific checkpoint extras: pruning stats, regret metrics, extreme regret keys.
fn print_mccfr_extras(
    solver: &MccfrSolver<HunlPostflop>,
    checkpoint_num: u64,
    mccfr_ctx: &MccfrCheckpointCtx,
) {
    let (pruned, total) = solver.pruning_stats();
    if total > 0 {
        let skip_pct = 100.0 * pruned as f64 / total as f64;
        println!("Pruned: {pruned}/{total} traversals ({skip_pct:.1}% skip rate)");
    }

    if checkpoint_num > 0 {
        let regrets = solver.regret_sum();
        let iters = solver.iterations();
        let max_r = convergence::max_regret(regrets, iters);
        let avg_r = convergence::avg_regret(regrets, iters);
        println!("  Max regret:       {max_r:.6}");
        println!("  Avg regret:       {avg_r:.6}");
        print_extreme_regret_keys(regrets, iters, mccfr_ctx.output_dir);
    }
}

/// Print the 2 highest and 2 lowest total-regret info set keys.
///
/// Useful for investigating convergence outliers with the `tree --key` command.
fn print_extreme_regret_keys(
    regret_sum: &FxHashMap<u64, Vec<f64>>,
    iterations: u64,
    output_dir: &str,
) {
    if regret_sum.is_empty() || iterations == 0 {
        return;
    }

    let iter_f = iterations as f64;

    // Collect (key, total_positive_regret_per_iter)
    let mut keyed: Vec<(u64, f64)> = regret_sum
        .iter()
        .map(|(&k, regrets)| {
            let total: f64 = regrets.iter().filter(|&&r| r > 0.0).sum();
            (k, total / iter_f)
        })
        .collect();

    keyed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let lowest: Vec<_> = keyed.iter().take(2).collect();
    let highest: Vec<_> = keyed.iter().rev().take(2).collect();

    println!("\nExtreme Regret Keys (for `tree --key`):");
    for &(key, regret) in &highest {
        let info = InfoKey::from_raw(*key);
        println!(
            "  highest: {key:#018x}  regret={regret:.6}  street={} spr={}",
            info.street(),
            info.spr_bucket(),
        );
    }
    for &(key, regret) in &lowest {
        let info = InfoKey::from_raw(*key);
        println!(
            "  lowest:  {key:#018x}  regret={regret:.6}  street={} spr={}",
            info.street(),
            info.spr_bucket(),
        );
    }

    // Negative regret diagnostic (DCFR health check)
    let (neg_count, most_neg_regret, most_neg_key) = negative_regret_stats(regret_sum);
    if neg_count > 0 {
        let info = InfoKey::from_raw(most_neg_key);
        println!(
            "  negative regrets: {} actions, min={:.6} at {:#018x} street={} spr={}",
            neg_count,
            most_neg_regret,
            most_neg_key,
            info.street(),
            info.spr_bucket()
        );
    } else {
        println!("  negative regrets: NONE (possible regret flooring issue)");
    }

    // Print a ready-to-copy command for the highest regret key
    if let Some(&(top_key, _)) = keyed.last() {
        println!(
            "  run: cargo run -p poker-solver-trainer -- tree -b {output_dir} --key {top_key:#018x}"
        );
    }
}

/// Find the most negative action regret and count of negative-regret actions.
///
/// Returns `(count, min_regret, key_of_min)`.
fn negative_regret_stats(regret_sum: &FxHashMap<u64, Vec<f64>>) -> (u64, f64, u64) {
    let mut most_neg = 0.0_f64;
    let mut most_neg_key = 0u64;
    let mut count = 0u64;

    for (&k, regrets) in regret_sum {
        for &r in regrets {
            if r < 0.0 {
                count += 1;
                if r < most_neg {
                    most_neg = r;
                    most_neg_key = k;
                }
            }
        }
    }

    (count, most_neg, most_neg_key)
}

/// Hand classes to display in the river strategy table (strongest → weakest).
const RIVER_DISPLAY_CLASSES: &[HandClass] = &[
    HandClass::Flush,
    HandClass::Straight,
    HandClass::Set,
    HandClass::TwoPair,
    HandClass::Overpair,
    HandClass::Pair,
    HandClass::HighCard,
];

/// Hand classes to display in the flop strategy table (made hands + draws).
const FLOP_DISPLAY_CLASSES: &[HandClass] = &[
    HandClass::Flush,
    HandClass::Straight,
    HandClass::Set,
    HandClass::TwoPair,
    HandClass::Overpair,
    HandClass::Pair,
    HandClass::HighCard,
    HandClass::ComboDraw,
    HandClass::FlushDraw,
    HandClass::Oesd,
    HandClass::Gutshot,
];

/// Return the strongest (lowest-discriminant) made-hand class for a hand_bits value,
/// or `None` if no recognized class is set.
///
/// For `HandClass` mode, `hand_bits` is a classification bitset.
/// For `HandClassV2` mode, `hand_bits` encodes class_id in bits 27-23.
fn strongest_class(hand_bits: u32, abs_mode: AbstractionModeConfig) -> Option<HandClass> {
    if abs_mode == AbstractionModeConfig::HandClassV2 {
        let class_id = ((hand_bits >> 23) & 0x1F) as u8;
        HandClass::from_discriminant(class_id)
    } else {
        HandClassification::from_bits(hand_bits).iter().next()
    }
}

/// Group key for postflop scenarios: (spr_bucket, actions_bits).
type PostflopScenario = (u32, u32);

/// Print postflop strategy tables for a given street.
fn print_postflop_strategies(
    strategies: &FxHashMap<u64, Vec<f64>>,
    abs_mode: AbstractionModeConfig,
    street_code: u8,
    street_name: &str,
    display_classes: &[HandClass],
) {
    if !abs_mode.is_hand_class() {
        return;
    }

    // Collect keys for the target street, grouped by scenario
    let mut first_to_act: FxHashMap<PostflopScenario, Vec<(u32, Vec<f64>)>> = FxHashMap::default();
    let mut facing_action: FxHashMap<PostflopScenario, Vec<(u32, Vec<f64>)>> = FxHashMap::default();

    for (&raw_key, probs) in strategies {
        let key = InfoKey::from_raw(raw_key);
        if key.street() != street_code {
            continue;
        }
        let hand_bits = key.hand_bits();
        let spr = key.spr_bucket();
        let actions = key.actions_bits();
        let scenario = (spr, actions);

        if actions == 0 {
            first_to_act
                .entry(scenario)
                .or_default()
                .push((hand_bits, probs.clone()));
        } else {
            facing_action
                .entry(scenario)
                .or_default()
                .push((hand_bits, probs.clone()));
        }
    }

    if let Some(scenario) = most_populated_scenario(&first_to_act) {
        let entries = &first_to_act[&scenario];
        let num_actions = entries.first().map_or(0, |(_, p)| p.len());
        let labels = first_to_act_labels(num_actions);
        print_postflop_table(
            street_name,
            "first to act",
            scenario,
            entries,
            &labels,
            abs_mode,
            display_classes,
        );
    }

    if let Some(scenario) = most_populated_scenario(&facing_action) {
        let entries = &facing_action[&scenario];
        let num_actions = entries.first().map_or(0, |(_, p)| p.len());
        let labels = facing_bet_labels(num_actions);
        print_postflop_table(
            street_name,
            "facing bet",
            scenario,
            entries,
            &labels,
            abs_mode,
            display_classes,
        );
    }
}

/// Print flop strategy tables for the most populated scenarios.
fn print_flop_strategies(strategies: &FxHashMap<u64, Vec<f64>>, abs_mode: AbstractionModeConfig) {
    print_postflop_strategies(strategies, abs_mode, 1, "Flop", FLOP_DISPLAY_CLASSES);
}

/// Print river strategy tables for the most populated first-to-act and facing-bet scenarios.
fn print_river_strategies(strategies: &FxHashMap<u64, Vec<f64>>, abs_mode: AbstractionModeConfig) {
    print_postflop_strategies(strategies, abs_mode, 3, "River", RIVER_DISPLAY_CLASSES);
}

/// Find the scenario key with the most entries.
fn most_populated_scenario(
    groups: &FxHashMap<PostflopScenario, Vec<(u32, Vec<f64>)>>,
) -> Option<PostflopScenario> {
    groups
        .iter()
        .max_by_key(|(_, entries)| entries.len())
        .map(|(&k, _)| k)
}

/// Build action labels for first-to-act river spots.
fn first_to_act_labels(num_actions: usize) -> Vec<String> {
    let mut labels = vec!["Check".to_string()];
    for i in 0..num_actions.saturating_sub(2) {
        labels.push(format!("B{i}"));
    }
    if num_actions > 1 {
        labels.push("B-AI".to_string());
    }
    labels
}

/// Build action labels for facing-bet river spots.
fn facing_bet_labels(num_actions: usize) -> Vec<String> {
    let mut labels = vec!["Fold".to_string(), "Call".to_string()];
    for i in 0..num_actions.saturating_sub(3) {
        labels.push(format!("R{i}"));
    }
    if num_actions > 2 {
        labels.push("R-AI".to_string());
    }
    labels
}

/// Print a single postflop strategy table (used for both flop and river).
fn print_postflop_table(
    street_name: &str,
    context: &str,
    scenario: PostflopScenario,
    entries: &[(u32, Vec<f64>)],
    labels: &[String],
    abs_mode: AbstractionModeConfig,
    display_classes: &[HandClass],
) {
    // Deduplicate by strongest hand class — keep entry with most data
    let mut by_class: FxHashMap<u8, &Vec<f64>> = FxHashMap::default();
    for (hand_bits, probs) in entries {
        if let Some(class) = strongest_class(*hand_bits, abs_mode) {
            let disc = class as u8;
            by_class.entry(disc).or_insert(probs);
        }
    }

    let (spr_b, _) = scenario;
    // Convert bucket back to approximate value
    let approx_spr = spr_b as f64 / 2.0;

    let action_cols: String = labels.iter().map(|l| format!("{:>6}", l)).collect();
    let header = format!("{:<12}|{action_cols}", "Class");
    let separator = "-".repeat(header.len());

    println!("{street_name} Strategy ({context}, SPR ~{approx_spr:.1}):");
    println!("{header}");
    println!("{separator}");

    for &class in display_classes {
        let disc = class as u8;
        if let Some(probs) = by_class.get(&disc) {
            let prob_cols: String = probs
                .iter()
                .take(labels.len())
                .map(|p| format!("{:>6.2}", p))
                .collect();
            println!("{:<12}|{prob_cols}", class);
        }
    }
    println!();
}
