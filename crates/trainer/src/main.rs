mod tree;

use std::error::Error;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use poker_solver_core::Game;
use poker_solver_core::HandClass;
use poker_solver_core::hand_class::HandClassification;
use poker_solver_core::abstraction::{AbstractionConfig, BoundaryGenerator};
use poker_solver_core::blueprint::{AbstractionModeConfig, BlueprintStrategy, BundleConfig, StrategyBundle};
use poker_solver_core::cfr::convergence;
use poker_solver_core::cfr::{MccfrConfig, MccfrSolver};
use poker_solver_core::flops::{self, CanonicalFlop, RankTexture, SuitTexture};
use poker_solver_core::game::{AbstractionMode, Action, HunlPostflop, PostflopConfig};
use poker_solver_core::info_key::{canonical_hand_index_from_str, depth_bucket, spr_bucket, InfoKey};
use rustc_hash::FxHashMap;
use serde::Deserialize;

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
}

/// Output format for the flops command.
#[derive(Debug, Clone, ValueEnum)]
enum OutputFormat {
    Json,
    Csv,
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
    /// Enable zero-regret pruning during training.
    #[serde(default)]
    pruning: bool,
    /// Fraction of total iterations to complete before enabling pruning.
    #[serde(default = "default_pruning_warmup_fraction")]
    pruning_warmup_fraction: f64,
    /// Run a full un-pruned probe iteration every N iterations.
    #[serde(default = "default_pruning_probe_interval")]
    pruning_probe_interval: u64,
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

/// Hands to display in the SB preflop strategy table.
const DISPLAY_HANDS: &[&str] = &["AA", "KK", "QQ", "AKs", "AKo", "JTs", "76s", "72o"];

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train { config, threads } => {
            if let Some(n) = threads {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build_global()
                    .expect("failed to configure rayon thread pool");
            }
            let yaml = std::fs::read_to_string(&config)?;
            let training_config: TrainingConfig = serde_yaml::from_str(&yaml)?;
            run_mccfr_training(training_config)?;
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
    }

    Ok(())
}

fn run_mccfr_training(config: TrainingConfig) -> Result<(), Box<dyn Error>> {
    let abs_mode = config.training.abstraction_mode;
    let use_hand_class = abs_mode == AbstractionModeConfig::HandClass;
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
    } else if use_hand_class {
        println!("Abstraction: hand_class (classify()-based, no EHS2)");
    } else if let Some(ref abs) = config.abstraction {
        println!("Abstraction config:");
        println!("  Flop buckets: {}", abs.flop_buckets);
        println!("  Turn buckets: {}", abs.turn_buckets);
        println!("  River buckets: {}", abs.river_buckets);
        println!("  Samples/street: {}", abs.samples_per_street);
    }
    println!();
    println!(
        "Training: {} iterations, {} samples/iter, seed {}",
        config.training.iterations, config.training.mccfr_samples, config.training.seed
    );
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
    } else if use_hand_class {
        Some(AbstractionMode::HandClass)
    } else {
        None
    };

    // Create game with deal pool
    let mut game = HunlPostflop::new(config.game.clone(), abstraction_mode, config.training.deal_count);
    if config.training.min_deals_per_class > 0 {
        game = game.with_stratification(
            config.training.min_deals_per_class,
            config.training.max_rejections_per_class,
        );
    }

    // Create MCCFR solver
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let pruning_warmup = (config.training.pruning_warmup_fraction
        * config.training.iterations as f64) as u64;

    let mccfr_config = MccfrConfig {
        samples_per_iteration: config.training.mccfr_samples,
        pruning: config.training.pruning,
        pruning_warmup,
        pruning_probe_interval: config.training.pruning_probe_interval,
        ..MccfrConfig::default()
    };

    println!("Creating MCCFR solver...");
    println!("  DCFR: α={}, β={}, γ={}", mccfr_config.dcfr_alpha, mccfr_config.dcfr_beta, mccfr_config.dcfr_gamma);
    if config.training.pruning {
        println!(
            "  Pruning: enabled (warmup {:.0}%, probe every {} iters)",
            config.training.pruning_warmup_fraction * 100.0,
            config.training.pruning_probe_interval
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
        strength_bits: if use_hand_class_v2 { config.training.strength_bits } else { 0 },
        equity_bits: if use_hand_class_v2 { config.training.equity_bits } else { 0 },
    };

    // Training loop with 10 checkpoints
    let total = config.training.iterations;
    let checkpoint_interval = (total / 10).max(1);
    let training_start = Instant::now();
    let mut previous_strategies: Option<FxHashMap<u64, Vec<f64>>> = None;

    // Baseline checkpoint (iteration 0)
    let ckpt_ctx = CheckpointCtx {
        total_checkpoints: 10,
        total_iterations: total,
        training_start: &training_start,
        header: &header,
        action_labels: &action_labels,
        stack_depth: config.game.stack_depth,
        output_dir: &config.training.output_dir,
        abs_mode,
        previous_strategies: &previous_strategies,
    };
    print_checkpoint(&solver, 0, &ckpt_ctx);

    // Progress bar for training iterations
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} iters ({per_sec}, ETA: {eta})",
        )
        .expect("valid template")
        .progress_chars("=>-"),
    );

    let num_threads = rayon::current_num_threads();
    println!("  Parallel training with {} threads\n", num_threads);

    for checkpoint in 1..=10 {
        solver.train_parallel_with_callback(
            checkpoint_interval,
            config.training.mccfr_samples,
            |_| {
                pb.inc(1);
            },
        );

        pb.suspend(|| {
            let ckpt_ctx = CheckpointCtx {
                total_checkpoints: 10,
                total_iterations: total,
                training_start: &training_start,
                header: &header,
                action_labels: &action_labels,
                stack_depth: config.game.stack_depth,
                output_dir: &config.training.output_dir,
                abs_mode,
                previous_strategies: &previous_strategies,
            };
            print_checkpoint(&solver, checkpoint, &ckpt_ctx);

            // Snapshot strategies for next delta computation
            previous_strategies = Some(solver.all_strategies_best_effort());

            // Save intermediate checkpoint bundle
            save_checkpoint_bundle(
                &solver,
                checkpoint,
                &config.training.output_dir,
                &bundle_config,
                &boundaries,
            );
        });
    }

    // Handle remainder iterations
    let trained_so_far = checkpoint_interval * 10;
    if trained_so_far < total {
        solver.train_parallel_with_callback(
            total - trained_so_far,
            config.training.mccfr_samples,
            |_| {
                pb.inc(1);
            },
        );
    }

    pb.finish_with_message("Training complete");

    // Save final bundle
    println!("\nSaving strategy bundle...");
    let strategies = solver.all_strategies();
    let blueprint = BlueprintStrategy::from_strategies(strategies, solver.iterations());
    println!(
        "  {} info sets, {} iterations",
        blueprint.len(),
        blueprint.iterations_trained()
    );

    let bundle = StrategyBundle::new(bundle_config, blueprint, boundaries);
    let output_path = PathBuf::from(&config.training.output_dir);
    bundle.save(&output_path)?;
    println!("  Saved to {}/", config.training.output_dir);

    // Verify loads
    let loaded = StrategyBundle::load(&output_path)?;
    println!("  Verified: {} info sets loaded\n", loaded.blueprint.len());

    println!("=== Training Complete ===");
    println!("Total time: {:?}", training_start.elapsed());

    Ok(())
}

fn save_checkpoint_bundle(
    solver: &MccfrSolver<HunlPostflop>,
    checkpoint: u64,
    output_dir: &str,
    bundle_config: &BundleConfig,
    boundaries: &Option<poker_solver_core::abstraction::BucketBoundaries>,
) {
    let dir = PathBuf::from(output_dir).join(format!("checkpoint_{checkpoint}_of_10"));
    let strategies = solver.all_strategies();
    let blueprint = BlueprintStrategy::from_strategies(strategies, solver.iterations());
    let bundle = StrategyBundle::new(bundle_config.clone(), blueprint, boundaries.clone());

    match bundle.save(&dir) {
        Ok(()) => println!("  Saved checkpoint to {}/", dir.display()),
        Err(e) => eprintln!("  Warning: failed to save checkpoint: {e}"),
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

struct CheckpointCtx<'a> {
    total_checkpoints: u64,
    total_iterations: u64,
    training_start: &'a Instant,
    header: &'a str,
    action_labels: &'a [String],
    stack_depth: u32,
    output_dir: &'a str,
    abs_mode: AbstractionModeConfig,
    previous_strategies: &'a Option<FxHashMap<u64, Vec<f64>>>,
}

fn print_checkpoint(
    solver: &MccfrSolver<HunlPostflop>,
    checkpoint: u64,
    ctx: &CheckpointCtx,
) {
    let current_iter = if checkpoint == 0 {
        0
    } else {
        (ctx.total_iterations / ctx.total_checkpoints) * checkpoint
    };

    let strategies = solver.all_strategies_best_effort();

    let elapsed = ctx.training_start.elapsed().as_secs_f64();
    let remaining = if checkpoint > 0 {
        let rate = elapsed / checkpoint as f64;
        rate * (ctx.total_checkpoints - checkpoint) as f64
    } else {
        0.0
    };

    println!(
        "\n=== Checkpoint {}/{} ({}/{} iterations) ===",
        checkpoint, ctx.total_checkpoints, current_iter, ctx.total_iterations
    );
    println!("Info sets: {}", strategies.len());

    if checkpoint > 0 {
        println!(
            "Time: {:.1}s elapsed, ~{:.1}s remaining",
            elapsed, remaining
        );
    }

    let (pruned, total) = solver.pruning_stats();
    if total > 0 {
        let skip_pct = 100.0 * pruned as f64 / total as f64;
        println!("Pruned: {pruned}/{total} traversals ({skip_pct:.1}% skip rate)");
    }

    // Convergence metrics
    if checkpoint > 0 {
        let regrets = solver.regret_sum();
        let iters = solver.iterations();
        let max_r = convergence::max_regret(regrets, iters);
        let avg_r = convergence::avg_regret(regrets, iters);
        let entropy = convergence::strategy_entropy(&strategies);

        let delta_str = match ctx.previous_strategies {
            Some(prev) => format!("{:.6}", convergence::strategy_delta(prev, &strategies)),
            None => "(first checkpoint)".to_string(),
        };

        println!("\nConvergence Metrics:");
        println!("  Strategy delta:   {delta_str}");
        println!("  Max regret:       {max_r:.6}");
        println!("  Avg regret:       {avg_r:.6}");
        println!("  Strategy entropy: {entropy:.4}");

        print_extreme_regret_keys(regrets, iters, ctx.output_dir);
    }

    // SB preflop strategy table
    println!("\nSB Opening Strategy (preflop, facing BB):");
    println!("{}", ctx.header);
    println!("{}", "-".repeat(ctx.header.len()));

    // Preflop initial state: pot=3 (SB+BB), stacks=stack_depth*2-1, stack_depth*2-2
    let preflop_eff_stack = ctx.stack_depth * 2 - 2;
    let spr_b = spr_bucket(3, preflop_eff_stack);
    let depth_b = depth_bucket(preflop_eff_stack);

    for &hand in DISPLAY_HANDS {
        if let Some(hand_idx) = canonical_hand_index_from_str(hand) {
            let info_key =
                InfoKey::new(u32::from(hand_idx), 0, spr_b, depth_b, &[]).as_u64();
            if let Some(probs) = strategies.get(&info_key) {
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

    print_river_strategies(&strategies, ctx.abs_mode);
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
            "  highest: {key:#018x}  regret={regret:.6}  street={} spr={} depth={}",
            info.street(),
            info.spr_bucket(),
            info.depth_bucket(),
        );
    }
    for &(key, regret) in &lowest {
        let info = InfoKey::from_raw(*key);
        println!(
            "  lowest:  {key:#018x}  regret={regret:.6}  street={} spr={} depth={}",
            info.street(),
            info.spr_bucket(),
            info.depth_bucket(),
        );
    }

    // Negative regret diagnostic (DCFR health check)
    let (neg_count, most_neg_regret, most_neg_key) = negative_regret_stats(regret_sum);
    if neg_count > 0 {
        let info = InfoKey::from_raw(most_neg_key);
        println!(
            "  negative regrets: {} actions, min={:.6} at {:#018x} street={} spr={} depth={}",
            neg_count, most_neg_regret, most_neg_key,
            info.street(), info.spr_bucket(), info.depth_bucket()
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
    HandClass::NutFlush,
    HandClass::Straight,
    HandClass::TopSet,
    HandClass::TwoPair,
    HandClass::Overpair,
    HandClass::TopPair,
    HandClass::SecondPair,
    HandClass::AceHigh,
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

/// Return a short display name for a `HandClass`.
fn class_label(class: HandClass) -> &'static str {
    match class {
        HandClass::NutFlush => "NutFlush",
        HandClass::Straight => "Straight",
        HandClass::TopSet => "TopSet",
        HandClass::TwoPair => "TwoPair",
        HandClass::Overpair => "Overpair",
        HandClass::TopPair => "TopPair",
        HandClass::SecondPair => "SecondPair",
        HandClass::AceHigh => "AceHigh",
        other => {
            // Fallback — use the Display impl (leaks a String, acceptable for rare cases)
            Box::leak(other.to_string().into_boxed_str())
        }
    }
}

/// Group key for river scenarios: (spr_bucket, depth_bucket, actions_bits).
type RiverScenario = (u32, u32, u32);

/// Print river strategy tables for the most populated first-to-act and facing-bet scenarios.
fn print_river_strategies(strategies: &FxHashMap<u64, Vec<f64>>, abs_mode: AbstractionModeConfig) {
    // Collect river keys grouped by scenario
    let mut first_to_act: FxHashMap<RiverScenario, Vec<(u32, Vec<f64>)>> = FxHashMap::default();
    let mut facing_action: FxHashMap<RiverScenario, Vec<(u32, Vec<f64>)>> = FxHashMap::default();

    for (&raw_key, probs) in strategies {
        let key = InfoKey::from_raw(raw_key);
        if key.street() != 3 {
            continue;
        }
        let hand_bits = key.hand_bits();
        let spr = key.spr_bucket();
        let depth = key.depth_bucket();
        let actions = key.actions_bits();
        let scenario = (spr, depth, actions);

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
        print_river_table("first to act", scenario, entries, &labels, abs_mode);
    }

    if let Some(scenario) = most_populated_scenario(&facing_action) {
        let entries = &facing_action[&scenario];
        let num_actions = entries.first().map_or(0, |(_, p)| p.len());
        let labels = facing_bet_labels(num_actions);
        print_river_table("facing bet", scenario, entries, &labels, abs_mode);
    }
}

/// Find the scenario key with the most entries.
fn most_populated_scenario(
    groups: &FxHashMap<RiverScenario, Vec<(u32, Vec<f64>)>>,
) -> Option<RiverScenario> {
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

/// Print a single river strategy table.
fn print_river_table(
    context: &str,
    scenario: RiverScenario,
    entries: &[(u32, Vec<f64>)],
    labels: &[String],
    abs_mode: AbstractionModeConfig,
) {
    // Deduplicate by strongest hand class — keep entry with most data
    let mut by_class: FxHashMap<u8, &Vec<f64>> = FxHashMap::default();
    for (hand_bits, probs) in entries {
        if let Some(class) = strongest_class(*hand_bits, abs_mode) {
            let disc = class as u8;
            by_class.entry(disc).or_insert(probs);
        }
    }

    let (spr_b, depth_b, _) = scenario;
    // Convert buckets back to approximate values
    let approx_spr = spr_b as f64 / 2.0;
    let approx_stack_bb = depth_b * 13 / 2;

    let action_cols: String = labels.iter().map(|l| format!("{:>6}", l)).collect();
    let header = format!("{:<12}|{action_cols}", "Class");
    let separator = "-".repeat(header.len());

    println!(
        "River Strategy ({context}, SPR ~{approx_spr:.1}, stack ~{approx_stack_bb}BB):"
    );
    println!("{header}");
    println!("{separator}");

    for &class in RIVER_DISPLAY_CLASSES {
        let disc = class as u8;
        if let Some(probs) = by_class.get(&disc) {
            let prob_cols: String = probs
                .iter()
                .take(labels.len())
                .map(|p| format!("{:>6.2}", p))
                .collect();
            println!("{:<12}|{prob_cols}", class_label(class));
        }
    }
    println!();
}
