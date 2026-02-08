use std::error::Error;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use poker_solver_core::Game;
use poker_solver_core::abstraction::{AbstractionConfig, BoundaryGenerator};
use poker_solver_core::blueprint::{BlueprintStrategy, BundleConfig, StrategyBundle};
use poker_solver_core::cfr::{MccfrConfig, MccfrSolver};
use poker_solver_core::flops::{self, CanonicalFlop, RankTexture, SuitTexture};
use poker_solver_core::game::{Action, HunlPostflop, PostflopConfig};
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
    abstraction: AbstractionConfig,
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
}

fn default_mccfr_samples() -> usize {
    500
}

fn default_deal_count() -> usize {
    50_000
}

/// Hands to display in the SB preflop strategy table.
const DISPLAY_HANDS: &[&str] = &["AA", "KK", "QQ", "AKs", "AKo", "JTs", "76s", "72o"];

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train { config } => {
            let yaml = std::fs::read_to_string(&config)?;
            let training_config: TrainingConfig = serde_yaml::from_str(&yaml)?;
            run_mccfr_training(training_config)?;
        }
        Commands::Flops { format, output } => {
            run_flops(format, output)?;
        }
    }

    Ok(())
}

fn run_mccfr_training(config: TrainingConfig) -> Result<(), Box<dyn Error>> {
    println!("=== Poker Blueprint Trainer (MCCFR) ===\n");
    println!("Game config:");
    println!("  Stack depth: {} BB", config.game.stack_depth);
    println!("  Bet sizes: {:?} pot", config.game.bet_sizes);
    println!("  Deal pool size: {}", config.training.deal_count);
    println!();
    println!("Abstraction config:");
    println!("  Flop buckets: {}", config.abstraction.flop_buckets);
    println!("  Turn buckets: {}", config.abstraction.turn_buckets);
    println!("  River buckets: {}", config.abstraction.river_buckets);
    println!(
        "  Samples/street: {}",
        config.abstraction.samples_per_street
    );
    println!();
    println!(
        "Training: {} iterations, {} samples/iter, seed {}",
        config.training.iterations, config.training.mccfr_samples, config.training.seed
    );
    println!("Output: {}", config.training.output_dir);
    println!();

    // Generate bucket boundaries
    println!("Generating bucket boundaries...");
    let start = Instant::now();
    let generator = BoundaryGenerator::new(config.abstraction.clone());
    let boundaries = generator.generate(config.training.seed);
    println!("  Done in {:?}\n", start.elapsed());

    // Create game with deal pool
    let game_config = PostflopConfig {
        stack_depth: config.game.stack_depth,
        bet_sizes: config.game.bet_sizes.clone(),
        samples_per_iteration: config.training.deal_count,
        ..PostflopConfig::default()
    };
    let game = HunlPostflop::new(game_config, None);

    // Create MCCFR solver
    let skip_first = config.training.iterations / 2;
    let mccfr_config = MccfrConfig {
        samples_per_iteration: config.training.mccfr_samples,
        use_cfr_plus: true,
        discount_iterations: Some(30),
        skip_first_iterations: Some(skip_first),
    };

    println!("Creating MCCFR solver...");
    println!(
        "  Skip first {} iterations for average strategy (50%)",
        skip_first
    );
    let start = Instant::now();
    let mut solver = MccfrSolver::with_config(game, &mccfr_config);
    solver.set_seed(config.training.seed);
    println!("  Created in {:?}\n", start.elapsed());

    // Derive action labels from a single deal (all deals share the same preflop actions)
    let label_game = HunlPostflop::new(
        PostflopConfig {
            stack_depth: config.game.stack_depth,
            bet_sizes: config.game.bet_sizes.clone(),
            samples_per_iteration: 1,
            ..PostflopConfig::default()
        },
        None,
    );
    let initial_states = label_game.initial_states();
    let actions = label_game.actions(&initial_states[0]);
    let action_labels = format_action_labels(&actions);
    let header = format_table_header(&action_labels);

    // Training loop with 10 checkpoints
    let total = config.training.iterations;
    let checkpoint_interval = total / 10;
    let training_start = Instant::now();
    // Baseline checkpoint (iteration 0)
    print_checkpoint(
        &solver,
        0,
        10,
        total,
        &training_start,
        &header,
        &action_labels,
    );

    // Progress bar for training iterations
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} iters ({per_sec}, ETA: {eta})",
        )
        .expect("valid template")
        .progress_chars("=>-"),
    );

    for checkpoint in 1..=10 {
        solver.train_with_callback(checkpoint_interval, config.training.mccfr_samples, |_| {
            pb.inc(1);
        });

        pb.suspend(|| {
            print_checkpoint(
                &solver,
                checkpoint,
                10,
                total,
                &training_start,
                &header,
                &action_labels,
            );
        });
    }

    // Handle remainder iterations
    let trained_so_far = checkpoint_interval * 10;
    if trained_so_far < total {
        solver.train_with_callback(
            total - trained_so_far,
            config.training.mccfr_samples,
            |_| {
                pb.inc(1);
            },
        );
    }

    pb.finish_with_message("Training complete");

    // Save bundle
    println!("\nSaving strategy bundle...");
    let strategies = solver.all_strategies();
    let blueprint = BlueprintStrategy::from_strategies(strategies, solver.iterations());
    println!(
        "  {} info sets, {} iterations",
        blueprint.len(),
        blueprint.iterations_trained()
    );

    let bundle_config = BundleConfig {
        game: config.game,
        abstraction: config.abstraction,
    };
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

fn format_action_labels(actions: &[Action]) -> Vec<String> {
    actions
        .iter()
        .map(|a| match a {
            Action::Fold => "Fold".to_string(),
            Action::Check => "Check".to_string(),
            Action::Call => "Call".to_string(),
            Action::Bet(n) => format!("B{n}"),
            Action::Raise(n) => format!("R{n}"),
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

fn print_checkpoint(
    solver: &MccfrSolver<HunlPostflop>,
    checkpoint: u64,
    total_checkpoints: u64,
    total_iterations: u64,
    training_start: &Instant,
    header: &str,
    action_labels: &[String],
) {
    let current_iter = if checkpoint == 0 {
        0
    } else {
        (total_iterations / total_checkpoints) * checkpoint
    };

    let strategies = solver.all_strategies();

    let elapsed = training_start.elapsed().as_secs_f64();
    let remaining = if checkpoint > 0 {
        let rate = elapsed / checkpoint as f64;
        rate * (total_checkpoints - checkpoint) as f64
    } else {
        0.0
    };

    // Header
    println!(
        "\n=== Checkpoint {}/{} ({}/{} iterations) ===",
        checkpoint, total_checkpoints, current_iter, total_iterations
    );
    println!("Info sets: {}", strategies.len());

    // Timing
    if checkpoint > 0 {
        println!(
            "Time: {:.1}s elapsed, ~{:.1}s remaining",
            elapsed, remaining
        );
    }

    // SB preflop strategy table
    println!("\nSB Opening Strategy (preflop, facing BB):");
    println!("{header}");
    println!("{}", "-".repeat(header.len()));

    for &hand in DISPLAY_HANDS {
        let info_key = format!("{hand}|P|");
        if let Some(probs) = strategies.get(&info_key) {
            let prob_cols: String = probs
                .iter()
                .take(action_labels.len())
                .map(|p| format!("{:>6.2}", p))
                .collect();
            println!("{:<6}|{prob_cols}", hand);
        } else {
            println!("{:<6}|  (no data)", hand);
        }
    }
    println!();
}
