mod bucket_diagnostics;
mod hand_trace;
mod lhe_viz;
mod tui;
mod tui_metrics;

use std::error::Error;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use clap::{Parser, ValueEnum};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use poker_solver_core::flops::{self, CanonicalFlop, RankTexture, SuitTexture};
use poker_solver_core::preflop::{
    EquityTable, PostflopBundle, PostflopModelConfig, PreflopAction, PreflopBundle, PreflopConfig,
    PreflopNode, PreflopSolver, PreflopTree,
};
use poker_solver_core::preflop::postflop_abstraction::{BuildPhase, FlopStage, PostflopAbstraction};
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
        Commands::SolvePostflop { config, output } => {
            run_solve_postflop(&config, &output)?;
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
struct PostflopSolveConfig {
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
    println!("\n=== Postflop EV Table (pot-fraction units) ===");
    println!("{:>5} {:>10} {:>10}", "Hand", "IP(SB)EV", "OOP(BB)EV");
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
            println!(
                "{:>5} {:>10.4} {:>10.4}  (avg over {count} opps)",
                name,
                ip_sum / count as f64,
                oop_sum / count as f64,
            );
        } else {
            println!("{:>5}  no data", name);
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
    println!("  Coverage (avg): IP {nonzero_ip}/{total} ({:.1}%)  OOP {nonzero_oop}/{total} ({:.1}%)",
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
        println!("  Flop {fi} ({flop_name}): {nz}/{total} ({:.1}%) non-zero",
            100.0 * nz as f64 / total as f64);
    }

    // Pairwise matchups
    if hands.len() >= 2 {
        println!("\n=== Pairwise matchups (avg_ev) ===");
        for i in 0..hands.len() {
            for j in (i+1)..hands.len() {
                let (h_name, hi) = &hands[i];
                let (o_name, oi) = &hands[j];
                let ip = abstraction.avg_ev(0, *hi, *oi);
                let oop = abstraction.avg_ev(1, *hi, *oi);
                print!("  {h_name:>4} vs {o_name:>4}: IP(SB)={ip:+.4}  OOP(BB)={oop:+.4}");
                for fi in 0..abstraction.values.num_flops() {
                    let fip = abstraction.values.get_by_flop(fi, 0, *hi as u16, *oi as u16);
                    let foop = abstraction.values.get_by_flop(fi, 1, *hi as u16, *oi as u16);
                    print!("  |f{fi}: IP={fip:+.3} OOP={foop:+.3}");
                }
                println!();
            }
        }
    }
    println!();
}

// ---------------------------------------------------------------------------
// Shared postflop progress infrastructure
// ---------------------------------------------------------------------------

/// Build one or more `PostflopAbstraction`s (one per configured SPR) with
/// multi-bar progress display showing per-flop CFR delta / EV extraction
/// progress alongside a main phase bar.
fn build_postflop_with_progress(
    pf_config: &PostflopModelConfig,
    equity: Option<&EquityTable>,
) -> Result<Vec<PostflopAbstraction>, Box<dyn Error>> {
    let bar_style = ProgressStyle::default_bar()
        .template("{spinner:.green} {msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .expect("valid template")
        .progress_chars("#>-");
    let spinner_style = ProgressStyle::default_spinner()
        .template("{spinner:.green} {msg}")
        .expect("valid template");

    let multi = MultiProgress::new();
    let phase_bar = multi.add(ProgressBar::new_spinner());
    phase_bar.set_style(spinner_style.clone());
    phase_bar.set_message("Postflop abstraction");
    phase_bar.enable_steady_tick(std::time::Duration::from_millis(500));

    const MAX_FLOP_BARS: usize = 10;

    struct FlopSlotData {
        sort_key: f64,
        position: u64,
        length: u64,
        message: String,
    }

    struct FlopBarState {
        states: HashMap<String, FlopSlotData>,
        slots: Vec<ProgressBar>,
        last_refresh: Instant,
    }

    #[allow(clippy::cast_precision_loss)]
    fn refresh_flop_slots(
        states: &HashMap<String, FlopSlotData>,
        slots: &mut Vec<ProgressBar>,
        multi: &MultiProgress,
        style: &ProgressStyle,
    ) {
        let mut sorted: Vec<_> = states.iter().collect();
        sorted.sort_by(|a, b| {
            b.1.sort_key
                .partial_cmp(&a.1.sort_key)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let visible = sorted.len().min(MAX_FLOP_BARS);
        while slots.len() < visible {
            let b = multi.add(ProgressBar::new(0));
            b.set_style(style.clone());
            b.enable_steady_tick(std::time::Duration::from_millis(500));
            slots.push(b);
        }
        while slots.len() > visible {
            if let Some(bar) = slots.pop() {
                bar.finish_and_clear();
                multi.remove(&bar);
            }
        }
        for (i, (_, data)) in sorted.iter().take(visible).enumerate() {
            slots[i].set_length(data.length);
            slots[i].set_position(data.position);
            slots[i].set_message(data.message.clone());
        }
    }

    let flop_state: Arc<Mutex<FlopBarState>> =
        Arc::new(Mutex::new(FlopBarState {
            states: HashMap::new(),
            slots: Vec::new(),
            last_refresh: Instant::now(),
        }));

    let pf_start = Instant::now();
    let sprs = &pf_config.postflop_sprs;
    let total_sprs = sprs.len();
    let mut abstractions = Vec::with_capacity(total_sprs);

    for (i, &spr) in sprs.iter().enumerate() {
        phase_bar.set_style(spinner_style.clone());
        phase_bar.set_length(0);
        phase_bar.set_position(0);
        phase_bar.set_message(format!(
            "Postflop SPR={spr} ({}/{})",
            i + 1,
            total_sprs,
        ));

        let abstraction = PostflopAbstraction::build_for_spr(
            pf_config,
            spr,
            equity,
            None,
            |phase| {
                match &phase {
                    BuildPhase::FlopProgress { flop_name, stage } => {
                        let mut guard = flop_state.lock().unwrap();
                        let fbs = &mut *guard;
                        match stage {
                            FlopStage::Solving { iteration, max_iterations, delta, metric_label } => {
                                #[allow(clippy::cast_precision_loss)]
                                let key = 1.0 + *iteration as f64 / (*max_iterations).max(1) as f64;
                                fbs.states.insert(flop_name.clone(), FlopSlotData {
                                    sort_key: key,
                                    position: *iteration as u64,
                                    length: *max_iterations as u64,
                                    message: format!("Flop '{flop_name}' CFR {metric_label}={delta:.4}"),
                                });
                            }
                            FlopStage::EstimatingEv { sample, total_samples } => {
                                #[allow(clippy::cast_precision_loss)]
                                let key = 2.0 + *sample as f64 / (*total_samples).max(1) as f64;
                                fbs.states.insert(flop_name.clone(), FlopSlotData {
                                    sort_key: key,
                                    position: *sample as u64,
                                    length: *total_samples as u64,
                                    message: format!("Flop '{flop_name}' EV Extraction"),
                                });
                            }
                            FlopStage::Done => {
                                fbs.states.remove(flop_name);
                                refresh_flop_slots(&fbs.states, &mut fbs.slots, &multi, &bar_style);
                                fbs.last_refresh = Instant::now();
                            }
                        }
                        if fbs.last_refresh.elapsed() >= std::time::Duration::from_secs(10) {
                            fbs.last_refresh = Instant::now();
                            refresh_flop_slots(&fbs.states, &mut fbs.slots, &multi, &bar_style);
                        }
                    }
                    BuildPhase::MccfrFlopsCompleted { completed, total } => {
                        phase_bar.set_style(bar_style.clone());
                        phase_bar.set_length(*total as u64);
                        phase_bar.set_position(*completed as u64);
                        phase_bar.set_message(format!("SPR={spr} MCCFR Solving"));
                    }
                    _ => {
                        phase_bar.set_style(spinner_style.clone());
                        phase_bar.set_message(format!("SPR={spr} {phase}..."));
                    }
                }
            },
        )
        .map_err(|e| format!("postflop SPR={spr}: {e}"))?;

        // Clean up flop bars between SPR solves.
        {
            let mut guard = flop_state.lock().unwrap();
            let fbs = &mut *guard;
            fbs.states.clear();
            for bar in fbs.slots.drain(..) {
                bar.finish_and_clear();
                multi.remove(&bar);
            }
            fbs.last_refresh = Instant::now();
        }

        eprintln!(
            "  SPR={spr} done ({}/{total_sprs}, values: {} entries)",
            i + 1,
            abstraction.values.len(),
        );
        abstractions.push(abstraction);
    }

    phase_bar.set_style(bar_style);
    phase_bar.finish_with_message(format!(
        "done in {:.1?} ({total_sprs} SPR model{})",
        pf_start.elapsed(),
        if total_sprs == 1 { "" } else { "s" },
    ));

    Ok(abstractions)
}

// ---------------------------------------------------------------------------
// Postflop bundle builder
// ---------------------------------------------------------------------------

fn run_solve_postflop(config_path: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let yaml = std::fs::read_to_string(config_path)?;
    let config: PostflopSolveConfig = serde_yaml::from_str(&yaml)?;
    let pf_config = config.postflop_model;

    eprintln!("Building postflop abstraction ({} SPR models)...", pf_config.postflop_sprs.len());

    let abstractions = build_postflop_with_progress(&pf_config, None)?;

    let refs: Vec<&PostflopAbstraction> = abstractions.iter().collect();
    PostflopBundle::save_multi(&pf_config, &refs, output)?;
    eprintln!(
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

    if let Some(ref abstractions) = postflop {
        if !ev_diagnostic_hands.is_empty() {
            if let Some(first) = abstractions.first() {
                print_postflop_ev_diagnostics(first, &ev_diagnostic_hands);
            }
        }
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
        if let Some(interval) = checkpoint_every {
            if interval > 0 && done.is_multiple_of(interval) {
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
    for row in 0..height {
        let label = if row == 0 {
            format!("{y_max:.1}")
        } else if row == height - 1 {
            format!("{y_min:.1}")
        } else {
            String::new()
        };
        let braille: String = grid[row]
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
