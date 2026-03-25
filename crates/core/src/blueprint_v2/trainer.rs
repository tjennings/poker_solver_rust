//! Training loop for the Blueprint V2 MCCFR solver.
//!
//! Drives external-sampling MCCFR iterations with LCFR weighting,
//! negative-regret pruning, periodic progress logging, and time-based
//! snapshot checkpoints.

// Arena indices are u32, bucket indices u16. Truncation and precision
// loss on small counts cast to f64 are safe.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::error::Error;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::Instant;

use rand::prelude::*;
use rand::rngs::{SmallRng, StdRng};
use rayon::prelude::*;

use super::bucket_file::BucketFile;
use super::bundle::{self, BlueprintV2Strategy};
use super::config::BlueprintV2Config;
use super::game_tree::GameTree;
use super::mccfr::{traverse_external, AllBuckets, Deal, DealWithBuckets, FullTreeEvTracker, PruneStats, ScenarioEvTracker, PRUNE_HITS, PRUNE_TOTAL};
use super::storage::BlueprintStorage;
use crate::hands::CanonicalHand;
use crate::poker::{Card, ALL_SUITS, ALL_VALUES};

/// Pre-initialized canonical deck — copied into the trainer's deck buffer
/// via memcpy instead of rebuilding from VALUE×SUIT loops each deal.
static CANONICAL_DECK: LazyLock<[Card; 52]> = LazyLock::new(|| {
    let mut deck = [Card::new(ALL_VALUES[0], ALL_SUITS[0]); 52];
    let mut idx = 0;
    for &v in &ALL_VALUES {
        for &s in &ALL_SUITS {
            deck[idx] = Card::new(v, s);
            idx += 1;
        }
    }
    deck
});

/// Sample a random deal using the provided RNG (thread-safe, no &mut self).
///
/// Copies the canonical deck onto the stack and applies a partial
/// Fisher-Yates shuffle of the first 9 positions.
fn sample_deal_with_rng(rng: &mut impl Rng) -> Deal {
    let mut deck = *CANONICAL_DECK;
    for i in 0..9 {
        let j = rng.random_range(i..52);
        deck.swap(i, j);
    }
    Deal {
        hole_cards: [
            [deck[0], deck[1]],
            [deck[2], deck[3]],
        ],
        board: [deck[4], deck[5], deck[6], deck[7], deck[8]],
    }
}

/// Check whether all 4 bucket files already exist in the given directory.
#[must_use]
pub fn bucket_files_exist(dir: &Path) -> bool {
    ["river.buckets", "turn.buckets", "flop.buckets", "preflop.buckets"]
        .iter()
        .all(|name| dir.join(name).exists())
}

/// Attempt to load `.buckets` files from the given directory.
///
/// Looks for `preflop.buckets`, `flop.buckets`, `turn.buckets`, and
/// `river.buckets`. Missing files are silently skipped (returning `None`
/// for that street). Load errors are logged to stderr but do not cause
/// a hard failure.
fn load_bucket_files(dir: &Path) -> [Option<BucketFile>; 4] {
    const NAMES: [&str; 4] = [
        "preflop.buckets",
        "flop.buckets",
        "turn.buckets",
        "river.buckets",
    ];
    let mut files: [Option<BucketFile>; 4] = [None, None, None, None];
    for (i, name) in NAMES.iter().enumerate() {
        let path = dir.join(name);
        if path.exists() {
            match BucketFile::load(&path) {
                Ok(bf) => {
                    eprintln!("  Loaded bucket file: {} ({} boards, {} combos/board, {} buckets)",
                        path.display(),
                        bf.header.board_count,
                        bf.header.combos_per_board,
                        bf.header.bucket_count,
                    );
                    files[i] = Some(bf);
                }
                Err(e) => eprintln!("Warning: failed to load {}: {e}", path.display()),
            }
        }
    }
    files
}

/// Callback type for pushing per-scenario strategy data to the TUI.
type StrategyRefreshCallback =
    Box<dyn Fn(usize, u32, &BlueprintStorage, &GameTree, &[f64; 169]) + Send>;

/// Callback type for pushing a random scenario to the TUI.
/// Receives per-position EV arrays `[position][hand_index]` so the callback
/// can select the correct position based on the chosen node's player.
type RandomScenarioCallback =
    Box<dyn Fn(&BlueprintStorage, &GameTree, &[[f64; 169]; 2]) + Send>;

/// Outer training driver for Blueprint V2.
///
/// Holds the game tree, regret/strategy storage, bucket lookup, and all
/// timing state needed to orchestrate LCFR-weighted MCCFR training with
/// periodic snapshots.
pub struct BlueprintTrainer {
    pub tree: GameTree,
    pub storage: BlueprintStorage,
    pub buckets: AllBuckets,
    pub config: BlueprintV2Config,
    pub rng: StdRng,
    pub start_time: Instant,
    pub iterations: u64,
    last_discount_time: u64,
    last_print_time: u64,
    last_snapshot_time: u64,
    snapshot_count: u32,
    /// Pre-allocated deck for [`sample_deal`](Self::sample_deal), avoiding
    /// a 52-element `Vec` allocation on every call.
    deck: [Card; 52],

    // --- Per-scenario-node EV tracking ---
    /// Per-scenario-node EV accumulator for TUI display.
    pub scenario_ev_tracker: ScenarioEvTracker,
    /// Full-tree EV tracker for all decision nodes (persisted in snapshots).
    pub full_ev_tracker: FullTreeEvTracker,

    // --- TUI shared state ---
    /// Iteration counter visible to the TUI thread.
    pub shared_iterations: Arc<AtomicU64>,
    /// Skip the bucket-file validation check in `train()`. Only for tests.
    pub skip_bucket_validation: bool,
    /// Per-street bucket visit counts. `bucket_visits[street][bucket]` = number
    /// of deals that mapped to this bucket. Updated atomically during training.
    pub bucket_visits: [Vec<AtomicU64>; 4],
    /// When `true`, the training loop sleeps until unpaused.
    pub paused: Arc<AtomicBool>,
    /// When `true`, the training loop exits at the next iteration boundary.
    pub quit_requested: Arc<AtomicBool>,
    /// One-shot trigger: the TUI sets this to request an immediate snapshot.
    pub snapshot_trigger: Arc<AtomicBool>,

    // --- TUI integration ---
    /// When `true`, suppress `eprintln!()` output that would corrupt the TUI.
    pub tui_active: bool,

    // --- TUI strategy refresh ---
    /// Seconds between strategy refresh pushes to TUI.
    pub strategy_refresh_interval_secs: u64,
    /// Node indices for scenarios to refresh.
    pub scenario_node_indices: Vec<u32>,
    /// Callback to push strategy data to TUI metrics.
    /// Args: (`scenario_index`, `node_idx`, `&BlueprintStorage`, `&GameTree`)
    pub on_strategy_refresh: Option<StrategyRefreshCallback>,
    /// Callback to push strategy delta values to TUI metrics.
    pub on_strategy_delta: Option<Box<dyn Fn(f64) + Send>>,
    /// Callback to push leaf movement fraction to TUI metrics.
    pub on_leaf_movement: Option<Box<dyn Fn(f64) + Send>>,
    /// Callback to push the minimum (most-negative) regret value to TUI metrics.
    pub on_min_regret: Option<Box<dyn Fn(f64) + Send>>,
    /// Callback to push the maximum (most-positive) regret value to TUI metrics.
    pub on_max_regret: Option<Box<dyn Fn(f64) + Send>>,
    /// Callback to push the average positive regret value to TUI metrics.
    pub on_avg_pos_regret: Option<Box<dyn Fn(f64) + Send>>,
    /// Callback to push fraction of actions below prune threshold to TUI.
    pub on_prune_fraction: Option<Box<dyn Fn(f64) + Send>>,
    /// Last time (in seconds) a strategy refresh was performed.
    last_strategy_refresh_secs: u64,

    // --- Strategy delta stopping ---
    /// Previous strategy-sum snapshot for delta computation.
    prev_strategy_sums: Option<Vec<i64>>,
    /// Most recent strategy delta value.
    pub last_strategy_delta: f64,
    /// Most recent fraction of info sets with max action delta > 0.20.
    pub last_pct_moving: f64,

    // --- Random scenario carousel ---
    /// Callback to push a random scenario to TUI.
    /// Receives a reference to storage and the game tree.
    pub on_random_scenario: Option<RandomScenarioCallback>,
    /// Minutes between random scenario rotations.
    pub random_scenario_hold_minutes: u64,
    /// Last time (in minutes) a random scenario was pushed.
    last_random_scenario_min: u64,
}

impl BlueprintTrainer {
    /// Build a trainer from a config: constructs the game tree, allocates
    /// storage, and initialises timing state.
    #[must_use]
    pub fn new(config: BlueprintV2Config) -> Self {
        let tree = GameTree::build(
            config.game.stack_depth,
            config.game.small_blind,
            config.game.big_blind,
            &config.action_abstraction.preflop,
            &config.action_abstraction.flop,
            &config.action_abstraction.turn,
            &config.action_abstraction.river,
        );

        let bucket_counts = [
            config.clustering.preflop.buckets,
            config.clustering.flop.buckets,
            config.clustering.turn.buckets,
            config.clustering.river.buckets,
        ];

        let storage = BlueprintStorage::new_with_baselines(
            &tree,
            bucket_counts,
            config.training.use_baselines,
        );
        let bucket_files = match &config.training.cluster_path {
            Some(path) => load_bucket_files(Path::new(path)),
            None => [None, None, None, None],
        };
        let buckets = AllBuckets::new(bucket_counts, bucket_files);

        // Auto-detect per-flop bucket files
        let buckets = if let Some(ref cluster_path) = config.training.cluster_path {
            let per_flop_marker = Path::new(cluster_path).join("flop_0000.buckets");
            if per_flop_marker.exists() {
                eprintln!("  Detected per-flop bucket files in {cluster_path}");
                buckets.with_per_flop_dir(Path::new(cluster_path).to_path_buf())
            } else {
                buckets
            }
        } else {
            buckets
        };

        // Copy bucket files into the snapshot output directory so the bundle
        // is self-contained and the explorer can find them.
        if let Some(ref cluster_path) = config.training.cluster_path {
            let dest = Path::new(&config.snapshots.output_dir).join("buckets");
            if !dest.exists() {
                std::fs::create_dir_all(&dest).expect("failed to create buckets/ in output dir");
                let src = Path::new(cluster_path);
                let mut copied = 0u32;
                for entry in std::fs::read_dir(src).expect("failed to read cluster_path") {
                    let entry = entry.expect("failed to read dir entry");
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.ends_with(".buckets") {
                        let dest_file = dest.join(&name);
                        std::fs::copy(entry.path(), &dest_file).unwrap_or_else(|e| {
                            panic!("failed to copy {} to {}: {e}", entry.path().display(), dest_file.display())
                        });
                        copied += 1;
                    }
                }
                eprintln!("  Copied {copied} bucket files from {cluster_path} to {}", dest.display());
            } else {
                eprintln!("  Bucket files already present at {}", dest.display());
            }
        }

        let rng = StdRng::seed_from_u64(config.clustering.seed);

        let deck = *CANONICAL_DECK;

        let full_ev_tracker = FullTreeEvTracker::new(&tree);

        Self {
            tree,
            storage,
            buckets,
            config,
            rng,
            start_time: Instant::now(),
            iterations: 0,
            last_discount_time: 0,
            last_print_time: 0,
            last_snapshot_time: 0,
            snapshot_count: 0,
            deck,
            scenario_ev_tracker: ScenarioEvTracker::new(vec![]),
            full_ev_tracker,
            skip_bucket_validation: false,
            bucket_visits: [
                (0..bucket_counts[0]).map(|_| AtomicU64::new(0)).collect(),
                (0..bucket_counts[1]).map(|_| AtomicU64::new(0)).collect(),
                (0..bucket_counts[2]).map(|_| AtomicU64::new(0)).collect(),
                (0..bucket_counts[3]).map(|_| AtomicU64::new(0)).collect(),
            ],
            shared_iterations: Arc::new(AtomicU64::new(0)),
            paused: Arc::new(AtomicBool::new(false)),
            quit_requested: Arc::new(AtomicBool::new(false)),
            snapshot_trigger: Arc::new(AtomicBool::new(false)),
            tui_active: false,
            strategy_refresh_interval_secs: 30,
            scenario_node_indices: Vec::new(),
            on_strategy_refresh: None,
            on_strategy_delta: None,
            on_leaf_movement: None,
            on_min_regret: None,
            on_max_regret: None,
            on_avg_pos_regret: None,
            on_prune_fraction: None,
            last_strategy_refresh_secs: 0,
            prev_strategy_sums: None,
            last_strategy_delta: f64::INFINITY,
            last_pct_moving: 1.0,
            on_random_scenario: None,
            random_scenario_hold_minutes: 3,
            last_random_scenario_min: 0,
        }
    }

    /// Attempt to resume from the latest snapshot in `output_dir`.
    ///
    /// When `config.snapshots.resume` is `true`, scans for `snapshot_NNNN`
    /// directories (and a `final/` directory), picks the one with the
    /// highest number, and loads its `regrets.bin` and `metadata.json`.
    ///
    /// Does nothing if `resume` is `false` or `output_dir` does not exist.
    ///
    /// # Errors
    ///
    /// Returns an error if a snapshot directory is found but its files
    /// cannot be read or parsed.
    pub fn try_resume(&mut self) -> Result<(), Box<dyn Error>> {
        if !self.config.snapshots.resume {
            return Ok(());
        }

        let output_dir = Path::new(&self.config.snapshots.output_dir);
        if !output_dir.exists() {
            return Ok(());
        }

        // Find the latest snapshot directory.
        let mut best: Option<(u32, std::path::PathBuf)> = None;

        if let Ok(entries) = std::fs::read_dir(output_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if let Some(num_str) = name_str.strip_prefix("snapshot_")
                    && let Ok(num) = num_str.parse::<u32>()
                    && entry.path().join("regrets.bin").exists()
                    && best.as_ref().is_none_or(|(n, _)| num > *n)
                {
                    best = Some((num, entry.path()));
                }
            }
        }

        // A `final/` directory always wins over numbered snapshots.
        let final_dir = output_dir.join("final");
        if final_dir.join("regrets.bin").exists() {
            let num = best.as_ref().map_or(0, |(n, _)| n + 1);
            best = Some((num, final_dir));
        }

        let Some((snapshot_num, snapshot_dir)) = best else {
            eprintln!("Resume: no snapshots found in {}", output_dir.display());
            return Ok(());
        };

        // Load regrets.
        let regrets_path = snapshot_dir.join("regrets.bin");
        let bucket_counts = [
            self.config.clustering.preflop.buckets,
            self.config.clustering.flop.buckets,
            self.config.clustering.turn.buckets,
            self.config.clustering.river.buckets,
        ];
        self.storage = BlueprintStorage::load_regrets(&regrets_path, &self.tree, bucket_counts)?;

        // Load metadata for iteration count and elapsed time.
        let meta_path = snapshot_dir.join("metadata.json");
        if meta_path.exists() {
            let meta_str = std::fs::read_to_string(&meta_path)?;
            if let Some(iter_val) = extract_json_u64(&meta_str, "iteration") {
                self.iterations = iter_val;
                self.shared_iterations.store(iter_val, Ordering::Relaxed);
            }
            // Backdate start_time so elapsed_minutes() reflects total
            // training time, not just this process's wall time. This
            // ensures pruning warmup and other time-gated actions
            // activate correctly after resume.
            if let Some(prev_min) = extract_json_u64(&meta_str, "elapsed_minutes") {
                let backdate = std::time::Duration::from_secs(prev_min * 60);
                self.start_time = Instant::now().checked_sub(backdate).unwrap_or(self.start_time);
            }
        }

        self.snapshot_count = snapshot_num + 1;

        // Load full-tree EV tracker if available.
        let hand_ev_path = snapshot_dir.join("hand_ev.bin");
        if hand_ev_path.exists() {
            self.full_ev_tracker.load(&hand_ev_path)?;
        }

        // Seed the strategy-delta baseline so the first check after resume
        // compares against the loaded state instead of producing zero.
        self.prev_strategy_sums = Some(self.storage.snapshot_strategy_sums());

        eprintln!(
            "Resumed from {}: {} iterations, {:.0}min elapsed, mean_pos_regret={:.2}",
            snapshot_dir.display(),
            self.iterations,
            self.elapsed_minutes(),
            self.mean_positive_regret(),
        );

        Ok(())
    }

    /// Run the training loop until a stopping criterion is met.
    ///
    /// # Errors
    ///
    /// Returns an error if no postflop bucket files are found (equity
    /// fallback produces meaningless abstractions) or if a snapshot
    /// write fails.
    pub fn train(&mut self) -> Result<(), Box<dyn Error>> {
        // Validate bucket files unless explicitly skipped or no cluster_path configured
        // (no cluster_path means intentional equity-only mode).
        if !self.skip_bucket_validation && self.config.training.cluster_path.is_some() {
        const STREET_NAMES: [&str; 3] = ["flop", "turn", "river"];
        let mut missing = Vec::new();
        for (i, name) in STREET_NAMES.iter().enumerate() {
            if self.buckets.bucket_files[i + 1].is_none() {
                missing.push(*name);
            }
        }
        if !missing.is_empty() {
            return Err(format!(
                "No bucket files found for: {}. \
                 Run the clustering pipeline first (cluster_path: {:?}). \
                 Training without proper buckets produces meaningless strategies.",
                missing.join(", "),
                self.config.training.cluster_path,
            )
            .into());
        }
        }

        let batch_size = self.config.training.batch_size;

        while !self.should_stop() {
            // Honour pause requests from the TUI.
            while self.paused.load(Ordering::Relaxed) {
                if self.quit_requested.load(Ordering::Relaxed) {
                    return Ok(());
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }

            // Calculate how many iterations remain (respect iteration limit).
            let remaining = self
                .config
                .training
                .iterations
                .map_or(batch_size, |max| max.saturating_sub(self.iterations));
            let this_batch = batch_size.min(remaining);
            if this_batch == 0 {
                break;
            }

            // Pre-seed per-deal RNGs from the main RNG (only sequential part).
            let thread_seeds: Vec<u64> = (0..this_batch)
                .map(|_| self.rng.random())
                .collect();

            let prune = self.should_prune();
            // Config is in BB units; stored regrets are ×1000.
            let threshold = self.config.training.prune_threshold.saturating_mul(1000);

            let tree = &self.tree;
            let storage = &self.storage;
            let buckets_ref = &self.buckets;
            let ev_tracker = &self.scenario_ev_tracker;
            let full_ev_tracker = &self.full_ev_tracker;
            let visit_counters = &self.bucket_visits;

            let rake_rate = self.config.game.rake_rate;
            let rake_cap = self.config.game.rake_cap;
            let baseline_alpha = if self.storage.baselines.is_some() {
                self.config.training.baseline_alpha
            } else {
                0.0
            };

            // Fully parallel: each thread samples its own deal, precomputes
            // buckets, and traverses — no sequential deal generation.
            let batch_prune_stats: PruneStats = thread_seeds.into_par_iter().map(|seed| {
                let mut rng = SmallRng::seed_from_u64(seed);
                let deal = sample_deal_with_rng(&mut rng);
                let buckets = buckets_ref.precompute_buckets(&deal);

                // Count bucket visits (both players, all streets).
                for player_buckets in &buckets {
                    for (street, &bucket) in player_buckets.iter().enumerate() {
                        if (bucket as usize) < visit_counters[street].len() {
                            visit_counters[street][bucket as usize]
                                .fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }

                let deal = DealWithBuckets { deal, buckets };
                let mut stats = PruneStats::default();

                let (_, s0) = traverse_external(
                    tree, storage, &deal, 0, tree.root, prune, threshold, &mut rng,
                    rake_rate, rake_cap, Some(ev_tracker), Some(full_ev_tracker),
                    baseline_alpha,
                );
                stats.merge(s0);
                let (_, s1) = traverse_external(
                    tree, storage, &deal, 1, tree.root, prune, threshold, &mut rng,
                    rake_rate, rake_cap, Some(ev_tracker), Some(full_ev_tracker),
                    baseline_alpha,
                );
                stats.merge(s1);

                stats
            }).reduce(PruneStats::default, |mut a, b| { a.merge(b); a });

            // Single atomic update for the whole batch.
            PRUNE_HITS.fetch_add(batch_prune_stats.hits, Ordering::Relaxed);
            PRUNE_TOTAL.fetch_add(batch_prune_stats.total, Ordering::Relaxed);

            // 3. Sequential: update counters and check timed actions.
            self.iterations += this_batch;
            self.shared_iterations
                .store(self.iterations, Ordering::Relaxed);
            self.check_timed_actions()?;
        }
        Ok(())
    }

    /// Sample a random deal by partial Fisher-Yates on a 52-card deck.
    ///
    /// Re-initialises the deck from canonical order each call, then
    /// shuffles only the first 9 positions (2×2 hole + 5 board).
    pub fn sample_deal(&mut self) -> Deal {
        // Reset deck to canonical order (avoids tracking swap state).
        self.deck = *CANONICAL_DECK;

        // Partial Fisher-Yates: shuffle only the first 9 positions.
        for i in 0..9 {
            let j = self.rng.random_range(i..52);
            self.deck.swap(i, j);
        }

        Deal {
            hole_cards: [
                [self.deck[0], self.deck[1]],
                [self.deck[2], self.deck[3]],
            ],
            board: [
                self.deck[4],
                self.deck[5],
                self.deck[6],
                self.deck[7],
                self.deck[8],
            ],
        }
    }

    /// True when either the iteration limit or time limit has been reached.
    fn should_stop(&self) -> bool {
        if self.quit_requested.load(Ordering::Relaxed) {
            return true;
        }
        if let Some(max_iter) = self.config.training.iterations
            && self.iterations >= max_iter
        {
            return true;
        }
        if let Some(max_min) = self.config.training.time_limit_minutes
            && self.elapsed_minutes() >= max_min
        {
            return true;
        }
        if let Some(target) = self.config.training.target_strategy_delta
            && self.last_strategy_delta <= target
        {
            return true;
        }
        false
    }

    /// Minutes elapsed since training started.
    fn elapsed_minutes(&self) -> u64 {
        self.start_time.elapsed().as_secs() / 60
    }

    /// Determine whether the current iteration should use pruning.
    ///
    /// Pruning activates after `prune_after_iterations` have elapsed and
    /// applies to `1 - prune_explore_pct` of iterations (the rest
    /// explore all actions to avoid permanently losing information).
    fn should_prune(&mut self) -> bool {
        if self.iterations < self.config.training.prune_after_iterations {
            return false;
        }
        let explore: f64 = self.rng.random();
        explore >= self.config.training.prune_explore_pct
    }

    /// Check and execute time-gated actions: LCFR discount, progress
    /// logging, and snapshot saving.
    fn check_timed_actions(&mut self) -> Result<(), Box<dyn Error>> {
        let elapsed_min = self.elapsed_minutes();

        // LCFR discount.
        let interval = self.config.training.lcfr_discount_interval.max(1);
        if self.iterations >= self.config.training.lcfr_warmup_iterations
            && self.iterations >= self.last_discount_time + interval
        {
            self.apply_lcfr_discount();
        }

        // Strategy delta: compute on whichever cadence fires first (print
        // interval or TUI refresh interval). Only compute once per check to
        // avoid overwriting prev_strategy_sums and getting a near-zero second
        // reading.
        let elapsed_secs = self.start_time.elapsed().as_secs();
        let print_due =
            elapsed_min >= self.last_print_time + self.config.training.print_every_minutes;
        let refresh_due =
            elapsed_secs >= self.last_strategy_refresh_secs + self.strategy_refresh_interval_secs;

        if print_due || refresh_due {
            self.update_strategy_delta();
        }

        if print_due {
            self.print_metrics();
        }

        // Snapshot: either timed or TUI-triggered.
        let tui_triggered = self.snapshot_trigger.swap(false, Ordering::Relaxed);
        if tui_triggered
            || (elapsed_min >= self.config.snapshots.warmup_minutes
                && elapsed_min
                    >= self.last_snapshot_time + self.config.snapshots.snapshot_every_minutes)
        {
            self.save_snapshot()?;
        }

        // Strategy refresh for TUI.
        if refresh_due {
            let decision_map = self.tree.decision_index_map();
            if let Some(ref callback) = self.on_strategy_refresh {
                for (i, &node_idx) in self.scenario_node_indices.iter().enumerate() {
                    let player = match &self.tree.nodes[node_idx as usize] {
                        super::game_tree::GameNode::Decision { player, .. } => *player as usize,
                        _ => 0,
                    };
                    // Read EVs from the full-tree tracker.
                    let dec_idx = decision_map[node_idx as usize];
                    let mut hand_evs = if dec_idx != u32::MAX {
                        self.full_ev_tracker.hand_ev_array(dec_idx as usize, player)
                    } else {
                        [0.0; 169]
                    };
                    // Normalize to decision-point frame: fold = 0.
                    // The traversal measures from hand start, so node_value
                    // includes the sunk cost of reaching this node. Subtract
                    // the fold terminal value (= -vol_self) so fold shows 0.
                    let fold_offset = Self::fold_value_at_node(&self.tree, node_idx, player);
                    for ev in &mut hand_evs {
                        if *ev != 0.0 {
                            *ev -= fold_offset;
                        }
                    }
                    callback(i, node_idx, &self.storage, &self.tree, &hand_evs);
                }
            }
            // Reset EV accumulators after reading so displayed values
            // reflect only the most recent window, not cumulative history.
            self.scenario_ev_tracker.reset();
            self.full_ev_tracker.reset();
            self.last_strategy_refresh_secs = elapsed_secs;
        }

        // Random scenario carousel rotation.
        if let Some(ref callback) = self.on_random_scenario
            && elapsed_min >= self.last_random_scenario_min + self.random_scenario_hold_minutes
        {
            // Use scenario 0 (root) EVs for random scenario display.
            let hand_evs = if self.scenario_ev_tracker.node_indices.is_empty() {
                [[0.0; 169], [0.0; 169]]
            } else {
                [self.scenario_ev_tracker.hand_ev_array(0, 0), self.scenario_ev_tracker.hand_ev_array(0, 1)]
            };
            callback(&self.storage, &self.tree, &hand_evs);
            self.last_random_scenario_min = elapsed_min;
        }

        Ok(())
    }

    /// Apply DCFR (Discounted CFR) discounting with separate exponents
    /// for positive regrets (α), negative regrets (β), and strategy sums (γ).
    ///
    /// - `d_pos = t^α / (t^α + 1)` — higher α retains positive regrets longer
    /// - `d_neg = t^β / (t^β + 1)` — lower β decays negative regrets faster
    /// - `d_strat = (t / (t + 1))^γ` — higher γ weights recent strategies more
    ///
    /// Setting α = β = γ = 1.0 recovers standard LCFR.
    fn apply_lcfr_discount(&mut self) {
        let interval = self.config.training.lcfr_discount_interval.max(1);
        let t = self.iterations / interval;
        let t = self.config.training.dcfr_epoch_cap.map_or(t, |cap| t.min(cap));
        let tf = t as f64;

        let alpha = self.config.training.dcfr_alpha;
        let beta = self.config.training.dcfr_beta;
        let gamma = self.config.training.dcfr_gamma;

        let t_alpha = tf.powf(alpha);
        let t_beta = tf.powf(beta);
        let d_pos = t_alpha / (t_alpha + 1.0);
        let d_neg = t_beta / (t_beta + 1.0);
        let d_strat = (tf / (tf + 1.0)).powf(gamma);

        self.storage.regrets.par_iter().for_each(|atom| {
            let v = atom.load(Ordering::Relaxed);
            let d = if v >= 0 { d_pos } else { d_neg };
            atom.store((f64::from(v) * d) as i32, Ordering::Relaxed);
        });
        self.storage.strategy_sums.par_iter().for_each(|atom| {
            let v = atom.load(Ordering::Relaxed);
            atom.store((v as f64 * d_strat) as i64, Ordering::Relaxed);
        });

        self.last_discount_time = self.iterations;
    }

    /// Compute and store the strategy delta vs the previous snapshot.
    fn update_strategy_delta(&mut self) {
        if let Some(ref prev) = self.prev_strategy_sums {
            let (delta, pct_moving) = self.storage.strategy_delta(prev);
            self.last_strategy_delta = delta;
            self.last_pct_moving = pct_moving;
            if let Some(ref cb) = self.on_strategy_delta {
                cb(delta);
            }
            if let Some(ref cb) = self.on_leaf_movement {
                cb(pct_moving);
            }
            if let Some(ref cb) = self.on_min_regret {
                cb(self.min_regret());
            }
            if let Some(ref cb) = self.on_max_regret {
                cb(self.max_regret());
            }
            if let Some(ref cb) = self.on_avg_pos_regret {
                cb(self.avg_pos_regret());
            }
            if let Some(ref cb) = self.on_prune_fraction {
                cb(self.traversal_prune_rate());
            }
        }
        self.prev_strategy_sums = Some(self.storage.snapshot_strategy_sums());
    }

    /// Log a one-line progress summary to stderr.
    fn print_metrics(&mut self) {
        self.last_print_time = self.elapsed_minutes();
        if self.tui_active {
            return;
        }

        let elapsed = self.start_time.elapsed();
        let secs = elapsed.as_secs_f64();
        let its_per_sec = if secs > 0.0 {
            self.iterations as f64 / secs
        } else {
            0.0
        };

        eprintln!(
            "[{:>6.1}m] iter={:<10} {:.0} it/s  mean_pos_regret={:.2}  δ={:.6}  moving={:.1}%",
            secs / 60.0,
            self.iterations,
            its_per_sec,
            self.mean_positive_regret(),
            self.last_strategy_delta,
            self.last_pct_moving * 100.0,
        );
    }

    /// The most-negative regret value across all info-set entries,
    /// divided by the ×1000 scaling factor used in storage.
    #[must_use]
    pub fn min_regret(&self) -> f64 {
        let min_raw = self
            .storage
            .regrets
            .iter()
            .map(|atom| atom.load(Ordering::Relaxed))
            .min()
            .unwrap_or(0);
        f64::from(min_raw) / 1000.0
    }

    /// The most-positive regret value across all info-set entries,
    /// divided by the ×1000 scaling factor used in storage.
    #[must_use]
    pub fn max_regret(&self) -> f64 {
        let max_raw = self
            .storage
            .regrets
            .iter()
            .map(|atom| atom.load(Ordering::Relaxed))
            .max()
            .unwrap_or(0);
        f64::from(max_raw) / 1000.0
    }

    /// Average positive regret per iteration: mean of positive regret
    /// entries divided by iteration count. This is the actual convergence
    /// signal — should decrease as O(1/√T).
    #[must_use]
    pub fn avg_pos_regret(&self) -> f64 {
        if self.iterations == 0 {
            return 0.0;
        }
        let (sum, count) = self
            .storage
            .regrets
            .iter()
            .fold((0.0_f64, 0_u64), |(s, c), atom| {
                let r = atom.load(Ordering::Relaxed);
                if r > 0 {
                    (s + f64::from(r), c + 1)
                } else {
                    (s, c)
                }
            });
        if count > 0 {
            sum / count as f64 / 1000.0 / self.iterations as f64
        } else {
            0.0
        }
    }

    /// Fraction of regret entries below the prune threshold (0.0–1.0).
    #[must_use]
    pub fn prune_fraction(&self) -> f64 {
        // Config is in BB units; stored regrets are ×1000.
        let threshold = self.config.training.prune_threshold.saturating_mul(1000);
        let total = self.storage.regrets.len() as f64;
        if total == 0.0 {
            return 0.0;
        }
        let below = self
            .storage
            .regrets
            .iter()
            .filter(|atom| atom.load(Ordering::Relaxed) < threshold)
            .count() as f64;
        below / total
    }

    /// Actual traversal prune rate: fraction of traverser-node actions
    /// that were skipped due to pruning since the last call (0.0–1.0).
    /// Resets the counters on each read.
    #[must_use]
    pub fn traversal_prune_rate(&self) -> f64 {
        let hits = PRUNE_HITS.swap(0, Ordering::Relaxed);
        let total = PRUNE_TOTAL.swap(0, Ordering::Relaxed);
        if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Mean of all strictly-positive regret entries.
    #[must_use]
    pub fn mean_positive_regret(&self) -> f64 {
        let (sum, count) = self
            .storage
            .regrets
            .iter()
            .fold((0.0_f64, 0_u64), |(s, c), atom| {
                let r = atom.load(Ordering::Relaxed);
                if r > 0 {
                    (s + f64::from(r), c + 1)
                } else {
                    (s, c)
                }
            });
        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }

    /// Compute the fold terminal value at a decision node for a given player.
    ///
    /// Returns the dead-money fold payoff (e.g. -2.0 if the player has
    /// voluntarily invested 2.0). At the root preflop node this is 0.0.
    /// Returns 0.0 if the node has no Fold action.
    fn fold_value_at_node(tree: &GameTree, node_idx: u32, player: usize) -> f64 {
        use super::game_tree::{GameNode, TreeAction, TerminalKind};
        if let GameNode::Decision { actions, children, .. } = &tree.nodes[node_idx as usize] {
            for (a, &child_idx) in children.iter().enumerate() {
                if actions[a] == TreeAction::Fold {
                    if let GameNode::Terminal { kind: TerminalKind::Fold { .. }, stacks, .. } =
                        &tree.nodes[child_idx as usize]
                    {
                        // Fold payoff = stacks[folder] - starting_stack (negative, loses blind)
                        return stacks[player] - tree.starting_stack;
                    }
                }
            }
        }
        0.0
    }

    /// Mean chip EV per canonical hand for a given position (0 or 1),
    /// as a fixed-size array (indexed by hand index 0..169).
    ///
    /// Uses scenario 0 (root) if the tracker has scenario nodes,
    /// otherwise returns all zeros.
    #[must_use]
    pub fn hand_ev_array(&self, player: usize) -> [f64; 169] {
        if self.scenario_ev_tracker.node_indices.is_empty() {
            [0.0; 169]
        } else {
            self.scenario_ev_tracker.hand_ev_array(0, player)
        }
    }

    /// Per-hand EV averages with names and sample counts for JSON export.
    /// Uses scenario 0 (root) for backward compatibility.
    #[must_use]
    pub fn hand_ev_averages(&self, player: usize) -> Vec<(String, f64, u64)> {
        use std::sync::atomic::Ordering as AO;
        (0..169)
            .map(|i| {
                let hand = CanonicalHand::from_index(i).expect("valid index 0..169");
                if self.scenario_ev_tracker.node_indices.is_empty() {
                    return (hand.to_string(), 0.0, 0);
                }
                let sum = self.scenario_ev_tracker.ev_sum[0][player][i].load(AO::Relaxed) as f64 / 1000.0;
                let count = self.scenario_ev_tracker.ev_count[0][player][i].load(AO::Relaxed);
                let avg = if count > 0 { sum / count as f64 } else { 0.0 };
                (hand.to_string(), avg, count)
            })
            .collect()
    }

    /// Combined (cross-position average) EV per canonical hand for JSON export.
    /// Uses scenario 0 (root) for backward compatibility.
    #[must_use]
    fn hand_ev_averages_combined(&self) -> Vec<(String, f64, u64)> {
        use std::sync::atomic::Ordering as AO;
        (0..169)
            .map(|i| {
                let hand = CanonicalHand::from_index(i).expect("valid index 0..169");
                if self.scenario_ev_tracker.node_indices.is_empty() {
                    return (hand.to_string(), 0.0, 0);
                }
                let sum0 = self.scenario_ev_tracker.ev_sum[0][0][i].load(AO::Relaxed) as f64 / 1000.0;
                let count0 = self.scenario_ev_tracker.ev_count[0][0][i].load(AO::Relaxed);
                let sum1 = self.scenario_ev_tracker.ev_sum[0][1][i].load(AO::Relaxed) as f64 / 1000.0;
                let count1 = self.scenario_ev_tracker.ev_count[0][1][i].load(AO::Relaxed);
                let total_count = count0 + count1;
                let avg = if total_count > 0 {
                    (sum0 + sum1) / total_count as f64
                } else {
                    0.0
                };
                (hand.to_string(), avg, total_count)
            })
            .collect()
    }

    /// Write a snapshot (regrets + metadata) to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the output directory cannot be created or
    /// the files cannot be written.
    pub fn save_snapshot(&mut self) -> Result<(), Box<dyn Error>> {
        use std::fmt::Write;
        let output_dir = Path::new(&self.config.snapshots.output_dir);
        std::fs::create_dir_all(output_dir)?;

        // Write config.yaml on the first snapshot so the explorer can discover this blueprint.
        if !output_dir.join("config.yaml").exists() {
            bundle::save_config(output_dir, &self.config)?;
        }

        let snapshot_dir = output_dir.join(format!("snapshot_{:04}", self.snapshot_count));

        let mut strategy = BlueprintV2Strategy::from_storage_with_threshold(
            &self.storage,
            &self.tree,
            self.config.training.purify_threshold,
        );
        strategy.iterations = self.iterations;
        strategy.elapsed_minutes = self.elapsed_minutes();

        let metadata = format!(
            "{{\"iteration\": {}, \"elapsed_minutes\": {}, \"mean_positive_regret\": {:.2}}}",
            self.iterations,
            self.elapsed_minutes(),
            self.mean_positive_regret(),
        );

        bundle::save_snapshot(&snapshot_dir, &strategy, &self.storage, &metadata)?;

        // Save full-tree EV tracker.
        self.full_ev_tracker.save(&snapshot_dir.join("hand_ev.bin"))?;

        // Compute and save counterfactual boundary values (CBVs) for
        // real-time subgame solving. One table per player, indexed by
        // (chance_node, bucket).
        let bucket_counts = self.storage.bucket_counts;
        let transitions = crate::blueprint_v2::cbv_compute::build_transitions_from_buckets(
            &self.buckets.bucket_files,
        );
        let [p0_cbvs, p1_cbvs] =
            crate::blueprint_v2::cbv_compute::compute_cbvs_with_transitions(
                &strategy, &self.tree, bucket_counts, &transitions,
            );
        p0_cbvs.save(&snapshot_dir.join("cbv_p0.bin"))?;
        p1_cbvs.save(&snapshot_dir.join("cbv_p1.bin"))?;

        // Write bucket visit counts.
        {
            let street_names = ["preflop", "flop", "turn", "river"];
            let mut visit_json = String::from("{\n");
            for (s, name) in street_names.iter().enumerate() {
                let counts: Vec<u64> = self.bucket_visits[s]
                    .iter()
                    .map(|a| a.load(Ordering::Relaxed))
                    .collect();
                let total: u64 = counts.iter().sum();
                let nonzero = counts.iter().filter(|&&c| c > 0).count();
                let max = counts.iter().max().copied().unwrap_or(0);
                let min_nonzero = counts.iter().filter(|&&c| c > 0).min().copied().unwrap_or(0);
                eprintln!(
                    "[bucket visits] {name}: {nonzero}/{} buckets visited, total={total}, min={min_nonzero}, max={max}",
                    counts.len()
                );
                let _ = write!(visit_json, "  \"{name}\": {:?}", counts);
                if s < 3 { visit_json.push(','); }
                visit_json.push('\n');
            }
            visit_json.push('}');
            std::fs::write(snapshot_dir.join("bucket_visits.json"), visit_json)?;
        }

        // Write per-hand chip EV averages with sample counts (averaged across both positions
        // for backward compatibility with existing JSON consumers).
        let hand_evs = self.hand_ev_averages_combined();
        let mut ev_json = String::from("{\n");
        for (i, (name, ev, count)) in hand_evs.iter().enumerate() {
            let _ = write!(
                ev_json,
                "  \"{name}\": {{\"ev\": {ev:.4}, \"samples\": {count}}}"
            );
            if i < hand_evs.len() - 1 {
                ev_json.push(',');
            }
            ev_json.push('\n');
        }
        ev_json.push('}');
        std::fs::write(snapshot_dir.join("hand_ev.json"), ev_json)?;

        self.snapshot_count += 1;
        self.last_snapshot_time = self.elapsed_minutes();

        if !self.tui_active {
            eprintln!("  Snapshot saved to {}", snapshot_dir.display());
        }

        // Prune old snapshots if retention limit is set.
        if let Some(max) = self.config.snapshots.max_snapshots {
            self.prune_old_snapshots(output_dir, max)?;
        }

        Ok(())
    }

    /// Delete the oldest `snapshot_NNNN` directories until at most `max`
    /// remain. Directories are sorted by their numeric suffix; lower
    /// numbers are deleted first. The `final/` directory is never pruned.
    fn prune_old_snapshots(&self, output_dir: &Path, max: u32) -> Result<(), Box<dyn Error>> {
        let mut numbered: Vec<(u32, std::path::PathBuf)> = Vec::new();
        for entry in std::fs::read_dir(output_dir)?.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if let Some(num_str) = name_str.strip_prefix("snapshot_")
                && let Ok(num) = num_str.parse::<u32>()
            {
                numbered.push((num, entry.path()));
            }
        }

        if numbered.len() as u32 <= max {
            return Ok(());
        }

        numbered.sort_by_key(|(num, _)| *num);
        let to_remove = numbered.len() - max as usize;
        for (_, path) in &numbered[..to_remove] {
            std::fs::remove_dir_all(path)?;
            if !self.tui_active {
                eprintln!("  Pruned old snapshot: {}", path.display());
            }
        }

        Ok(())
    }
}

/// Extract a `u64` value for the given `key` from a simple JSON string.
///
/// Avoids pulling in `serde_json` for a single metadata read.
fn extract_json_u64(json: &str, key: &str) -> Option<u64> {
    let pattern = format!("\"{key}\"");
    let idx = json.find(&pattern)?;
    let after_key = &json[idx + pattern.len()..];
    let colon_idx = after_key.find(':')?;
    let after_colon = after_key[colon_idx + 1..].trim_start();
    let num_str: String = after_colon
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    num_str.parse().ok()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::blueprint_v2::config::*;

    fn toy_config() -> BlueprintV2Config {
        BlueprintV2Config {
            game: GameConfig {
                name: "Test".to_string(),
                players: 2,
                stack_depth: 10.0,
                small_blind: 0.5,
                big_blind: 1.0,
                rake_rate: 0.0,
                rake_cap: 0.0,
            },
            clustering: ClusteringConfig {
                algorithm: ClusteringAlgorithm::PotentialAwareEmd,
                preflop: StreetClusterConfig { buckets: 10, delta_bins: None, expected_delta: false, sample_boards: None },
                flop: StreetClusterConfig { buckets: 10, delta_bins: None, expected_delta: false, sample_boards: None },
                turn: StreetClusterConfig { buckets: 10, delta_bins: None, expected_delta: false, sample_boards: None },
                river: StreetClusterConfig { buckets: 10, delta_bins: None, expected_delta: false, sample_boards: None },
                seed: 42,
                kmeans_iterations: 50,
                cfvnet_river_data: None,
                per_flop: None,
            },
            action_abstraction: ActionAbstractionConfig {
                preflop: vec![vec!["2.5bb".into()]],
                flop: vec![vec![1.0]],
                turn: vec![vec![1.0]],
                river: vec![vec![1.0]],
            },
            training: TrainingConfig {
                cluster_path: None,
                iterations: Some(100),
                time_limit_minutes: None,
                lcfr_warmup_iterations: 0,
                lcfr_discount_interval: 1,
                prune_after_iterations: 9999,
                prune_threshold: 0,
                prune_explore_pct: 0.05,
                print_every_minutes: 9999,
                batch_size: 200,
                target_strategy_delta: None,
                purify_threshold: 0.0,
                equity_cache_path: None,
                dcfr_alpha: 1.0,
                dcfr_beta: 1.0,
                dcfr_gamma: 1.0,
                dcfr_epoch_cap: None,
                optimizer: "dcfr".to_string(),
                sapcfr_eta: 0.5,
                use_baselines: false,
                baseline_alpha: 0.01,
            },
            snapshots: SnapshotConfig {
                warmup_minutes: 9999,
                snapshot_every_minutes: 9999,
                output_dir: "/tmp/test_blueprint_v2_snapshots".into(),
                resume: false,
                max_snapshots: None,
            },
        }
    }

    fn toy_trainer(config: BlueprintV2Config) -> BlueprintTrainer {
        let mut t = BlueprintTrainer::new(config);
        t.skip_bucket_validation = true;
        t
    }

    #[test]
    fn trainer_creation() {
        let config = toy_config();
        let trainer = BlueprintTrainer::new(config);
        assert_eq!(trainer.iterations, 0);
        assert!(!trainer.storage.regrets.is_empty());
    }

    #[test]
    fn sample_deal_no_duplicates() {
        let config = toy_config();
        let mut trainer = BlueprintTrainer::new(config);
        let deal = trainer.sample_deal();

        let mut all_cards: Vec<Card> = Vec::new();
        all_cards.extend_from_slice(&deal.hole_cards[0]);
        all_cards.extend_from_slice(&deal.hole_cards[1]);
        all_cards.extend_from_slice(&deal.board);

        for i in 0..all_cards.len() {
            for j in (i + 1)..all_cards.len() {
                assert_ne!(all_cards[i], all_cards[j], "duplicate cards in deal");
            }
        }
    }

    #[test]
    fn train_runs_iterations() {
        let mut config = toy_config();
        config.training.iterations = Some(50);
        let mut trainer = toy_trainer(config);
        trainer.train().expect("training should complete");
        assert_eq!(trainer.iterations, 50);
    }

    #[test]
    fn train_updates_storage() {
        let mut config = toy_config();
        config.training.iterations = Some(20);
        let mut trainer = toy_trainer(config);

        assert!(trainer
            .storage
            .regrets
            .iter()
            .all(|r| r.load(Ordering::Relaxed) == 0));

        trainer.train().expect("training should complete");

        assert!(
            trainer
                .storage
                .regrets
                .iter()
                .any(|r| r.load(Ordering::Relaxed) != 0),
            "regrets should be updated after training"
        );
    }

    #[test]
    fn mean_positive_regret_initially_zero() {
        let config = toy_config();
        let trainer = BlueprintTrainer::new(config);

        assert!(
            (trainer.mean_positive_regret() - 0.0).abs() < 1e-10,
            "initially zero"
        );

        if trainer.storage.regrets.len() >= 3 {
            trainer.storage.regrets[0].store(100, Ordering::Relaxed);
            trainer.storage.regrets[1].store(-50, Ordering::Relaxed);
            trainer.storage.regrets[2].store(200, Ordering::Relaxed);
        }

        let mean = trainer.mean_positive_regret();
        assert!(mean > 0.0, "mean positive regret should be > 0");
    }

    #[test]
    fn lcfr_discount_at_t_zero() {
        let config = toy_config();
        let mut trainer = BlueprintTrainer::new(config);

        trainer.storage.regrets[0].store(1000, Ordering::Relaxed);
        trainer.storage.strategy_sums[0].store(2000, Ordering::Relaxed);

        // t = elapsed_min / interval = 0 / 1 = 0, so d = 0/(0+1) = 0.
        trainer.apply_lcfr_discount();

        assert_eq!(trainer.storage.regrets[0].load(Ordering::Relaxed), 0);
        assert_eq!(
            trainer.storage.strategy_sums[0].load(Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn dcfr_epoch_cap_limits_discount() {
        let mut config = toy_config();
        config.training.dcfr_alpha = 1.0;
        config.training.dcfr_beta = 1.0;
        config.training.dcfr_gamma = 1.0;
        config.training.lcfr_discount_interval = 1;
        config.training.dcfr_epoch_cap = Some(5);
        let mut trainer = BlueprintTrainer::new(config);

        // Seed a known regret and strategy sum value.
        trainer.storage.regrets[0].store(1000, Ordering::Relaxed);
        trainer.storage.strategy_sums[0].store(2000, Ordering::Relaxed);

        // Set iterations to 10, which without the cap would give t=10.
        // With cap=5, t should be clamped to 5.
        trainer.iterations = 10;
        trainer.apply_lcfr_discount();

        // d_pos = 5^1 / (5^1 + 1) = 5/6 ≈ 0.8333
        // regret: (1000 * 5/6) as i32 = 833
        let expected_regret = (1000.0 * 5.0 / 6.0) as i32;
        assert_eq!(
            trainer.storage.regrets[0].load(Ordering::Relaxed),
            expected_regret,
        );

        // d_strat = (5/6)^1 = 5/6 ≈ 0.8333
        // strategy_sum: (2000 * 5/6) as i64 = 1666
        let expected_strat = (2000.0 * 5.0 / 6.0) as i64;
        assert_eq!(
            trainer.storage.strategy_sums[0].load(Ordering::Relaxed),
            expected_strat,
        );
    }

    #[test]
    fn dcfr_epoch_cap_none_preserves_behavior() {
        let mut config = toy_config();
        config.training.dcfr_alpha = 1.0;
        config.training.dcfr_beta = 1.0;
        config.training.dcfr_gamma = 1.0;
        config.training.lcfr_discount_interval = 1;
        config.training.dcfr_epoch_cap = None;
        let mut trainer = BlueprintTrainer::new(config);

        trainer.storage.regrets[0].store(1000, Ordering::Relaxed);
        trainer.storage.strategy_sums[0].store(2000, Ordering::Relaxed);

        // With no cap and iterations=10, t=10 is used directly.
        trainer.iterations = 10;
        trainer.apply_lcfr_discount();

        // d_pos = 10^1 / (10^1 + 1) = 10/11 ≈ 0.9091
        let expected_regret = (1000.0 * 10.0 / 11.0) as i32;
        assert_eq!(
            trainer.storage.regrets[0].load(Ordering::Relaxed),
            expected_regret,
        );

        // d_strat = (10/11)^1 ≈ 0.9091
        let expected_strat = (2000.0 * 10.0 / 11.0) as i64;
        assert_eq!(
            trainer.storage.strategy_sums[0].load(Ordering::Relaxed),
            expected_strat,
        );
    }

    #[test]
    fn snapshot_save() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let mut config = toy_config();
        config.snapshots.output_dir = dir.path().to_string_lossy().to_string();
        config.training.iterations = Some(10);
        let mut trainer = toy_trainer(config);

        trainer.train().expect("training should complete");
        trainer.save_snapshot().expect("snapshot should save");

        let snapshot_dir = dir.path().join("snapshot_0000");
        assert!(snapshot_dir.join("strategy.bin").exists());
        assert!(snapshot_dir.join("regrets.bin").exists());
        assert!(snapshot_dir.join("metadata.json").exists());
        assert!(snapshot_dir.join("hand_ev.json").exists());
        assert!(snapshot_dir.join("hand_ev.bin").exists(), "full-tree EV tracker should be saved");

        // Verify hand_ev.json is valid JSON with 169 entries.
        let ev_json = std::fs::read_to_string(snapshot_dir.join("hand_ev.json"))
            .expect("read hand_ev.json");
        let ev_map: std::collections::BTreeMap<String, serde_json::Value> =
            serde_json::from_str(&ev_json).expect("parse hand_ev.json");
        assert_eq!(ev_map.len(), 169, "should have 169 hand entries");
        // Verify each entry has ev and samples fields.
        for val in ev_map.values() {
            assert!(val.get("ev").and_then(|v| v.as_f64()).is_some());
            assert!(val.get("samples").and_then(|v| v.as_u64()).is_some());
        }
    }

    #[test]
    fn snapshot_retention_prunes_oldest() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let mut config = toy_config();
        config.snapshots.output_dir = dir.path().to_string_lossy().to_string();
        config.snapshots.max_snapshots = Some(2);
        config.training.iterations = Some(10);
        let mut trainer = toy_trainer(config);
        trainer.train().expect("training should complete");

        // Save 4 snapshots; retention limit is 2.
        for _ in 0..4 {
            trainer.save_snapshot().expect("snapshot should save");
        }

        // Only the 2 newest should remain.
        let remaining: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("snapshot_")
            })
            .collect();
        assert_eq!(remaining.len(), 2, "should keep exactly max_snapshots");

        // The kept ones should be snapshot_0002 and snapshot_0003.
        let mut names: Vec<String> = remaining
            .iter()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        names.sort();
        assert_eq!(names, vec!["snapshot_0002", "snapshot_0003"]);
    }

    #[test]
    fn train_batch_iterations() {
        let mut config = toy_config();
        config.training.iterations = Some(50);
        config.training.batch_size = 10;
        let mut trainer = toy_trainer(config);
        trainer.train().expect("training should complete");
        assert_eq!(trainer.iterations, 50);
    }

    #[test]
    fn parallel_batch_produces_regret_updates() {
        let mut config = toy_config();
        config.training.iterations = Some(200);
        config.training.batch_size = 50;
        let mut trainer = toy_trainer(config);
        trainer.train().expect("training should complete");
        assert_eq!(trainer.iterations, 200);
        assert!(trainer
            .storage
            .regrets
            .iter()
            .any(|r| r.load(Ordering::Relaxed) != 0));
        assert!(trainer
            .storage
            .strategy_sums
            .iter()
            .any(|s| s.load(Ordering::Relaxed) != 0));
    }

    #[test]
    fn batch_size_larger_than_iterations() {
        let mut config = toy_config();
        config.training.iterations = Some(10);
        config.training.batch_size = 200;
        let mut trainer = toy_trainer(config);
        trainer.train().expect("training should complete");
        assert_eq!(trainer.iterations, 10);
    }

    #[test]
    fn strategy_delta_stops_training() {
        let mut config = toy_config();
        // No iteration limit — only delta-based stopping.
        config.training.iterations = None;
        config.training.time_limit_minutes = Some(1); // safety timeout
        config.training.target_strategy_delta = Some(0.5);
        config.training.print_every_minutes = 0; // check every batch
        config.training.batch_size = 50;
        let mut trainer = toy_trainer(config);
        trainer.train().expect("training should complete");
        // Should have stopped due to delta, not the 1-minute limit.
        assert!(trainer.iterations > 0, "should have run some iterations");
        assert!(
            trainer.last_strategy_delta <= 0.5,
            "should have stopped when delta <= 0.5, got {}",
            trainer.last_strategy_delta,
        );
    }

    #[test]
    fn extract_json_u64_works() {
        let json = r#"{"iteration": 12345, "elapsed_minutes": 5}"#;
        assert_eq!(extract_json_u64(json, "iteration"), Some(12345));
        assert_eq!(extract_json_u64(json, "elapsed_minutes"), Some(5));
        assert_eq!(extract_json_u64(json, "missing"), None);
    }

    #[test]
    fn resume_from_snapshot() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let snapshot_dir = dir.path().join("snapshot_0000");

        // Train 20 iterations and save a snapshot.
        let mut config = toy_config();
        config.training.iterations = Some(20);
        config.snapshots.output_dir = dir.path().to_string_lossy().to_string();
        let mut trainer = toy_trainer(config.clone());
        trainer.train().expect("initial training");
        trainer.save_snapshot().expect("save snapshot");
        assert!(snapshot_dir.join("regrets.bin").exists());

        // Create a new trainer with resume=true and train 20 more.
        config.training.iterations = Some(40); // total target
        config.snapshots.resume = true;
        let mut trainer2 = toy_trainer(config);
        trainer2.try_resume().expect("resume should succeed");
        assert_eq!(trainer2.iterations, 20, "should resume at iteration 20");
        assert_eq!(trainer2.snapshot_count, 1, "should start at snapshot 1");

        // Regrets should be non-zero (loaded from snapshot).
        assert!(
            trainer2
                .storage
                .regrets
                .iter()
                .any(|r| r.load(Ordering::Relaxed) != 0),
            "regrets should be loaded from snapshot"
        );

        // Full EV tracker should have been loaded from hand_ev.bin.
        assert!(snapshot_dir.join("hand_ev.bin").exists(), "hand_ev.bin should exist in snapshot");

        trainer2.train().expect("resumed training");
        assert_eq!(trainer2.iterations, 40, "should reach 40 total");
    }

    #[test]
    fn hand_ev_array_per_position_initially_zero() {
        let config = toy_config();
        let trainer = BlueprintTrainer::new(config);
        let evs_p0 = trainer.hand_ev_array(0);
        let evs_p1 = trainer.hand_ev_array(1);
        assert!(evs_p0.iter().all(|&v| v == 0.0), "position 0 EVs should start at zero");
        assert!(evs_p1.iter().all(|&v| v == 0.0), "position 1 EVs should start at zero");
    }

    #[test]
    fn hand_ev_array_per_position_independent() {
        let config = toy_config();
        let mut trainer = BlueprintTrainer::new(config);

        // Set up tracker with a single node (root).
        trainer.scenario_ev_tracker.set_nodes(vec![trainer.tree.root]);

        // Manually set EV data for position 0, hand index 5.
        trainer.scenario_ev_tracker.accumulate(0, 0, 5, 3.0);
        trainer.scenario_ev_tracker.accumulate(0, 0, 5, 0.0); // avg = 1.5

        // Manually set EV data for position 1, hand index 5.
        trainer.scenario_ev_tracker.accumulate(0, 1, 5, -1.0);

        let evs_p0 = trainer.hand_ev_array(0);
        let evs_p1 = trainer.hand_ev_array(1);

        assert!((evs_p0[5] - 1.5).abs() < 1e-2, "position 0 hand 5 EV should be 1.5, got {}", evs_p0[5]);
        assert!((evs_p1[5] - (-1.0)).abs() < 1e-2, "position 1 hand 5 EV should be -1.0, got {}", evs_p1[5]);
        // Other hands should still be zero.
        assert!(evs_p0[0] == 0.0);
        assert!(evs_p1[0] == 0.0);
    }

    #[test]
    fn hand_ev_averages_per_position() {
        let config = toy_config();
        let mut trainer = BlueprintTrainer::new(config);

        // Set up tracker with a single node (root).
        trainer.scenario_ev_tracker.set_nodes(vec![trainer.tree.root]);

        trainer.scenario_ev_tracker.accumulate(0, 0, 0, 5.0);
        trainer.scenario_ev_tracker.accumulate(0, 1, 0, 5.0);
        trainer.scenario_ev_tracker.accumulate(0, 1, 0, 5.0);

        let avg_p0 = trainer.hand_ev_averages(0);
        let avg_p1 = trainer.hand_ev_averages(1);

        assert!((avg_p0[0].1 - 5.0).abs() < 1e-2, "p0 avg should be 5.0");
        assert_eq!(avg_p0[0].2, 1);
        assert!((avg_p1[0].1 - 5.0).abs() < 1e-2, "p1 avg should be 5.0");
        assert_eq!(avg_p1[0].2, 2);
    }

    #[test]
    fn train_accumulates_ev_per_position() {
        use super::super::game_tree::GameNode;

        let mut config = toy_config();
        config.training.iterations = Some(200);
        config.training.batch_size = 50;
        let mut trainer = toy_trainer(config);

        // Collect decision nodes for both players.
        let mut tracked_nodes: Vec<u32> = Vec::new();
        for (i, node) in trainer.tree.nodes.iter().enumerate() {
            if let GameNode::Decision { .. } = node {
                tracked_nodes.push(i as u32);
            }
        }
        assert!(!tracked_nodes.is_empty(), "should have decision nodes");

        // Set up tracker with all decision nodes.
        trainer.scenario_ev_tracker.set_nodes(tracked_nodes.clone());
        trainer.scenario_node_indices = tracked_nodes;

        trainer.train().expect("training should complete");

        // Root (scenario 0) is player 0's decision -- should have EVs for p0.
        let evs_p0 = trainer.scenario_ev_tracker.hand_ev_array(0, 0);
        assert!(evs_p0.iter().any(|&v| v != 0.0), "position 0 should have some non-zero EVs at root");

        // Find a player 1 scenario node and check it has EVs.
        let mut found_p1 = false;
        for (si, &ni) in trainer.scenario_ev_tracker.node_indices.iter().enumerate() {
            if let GameNode::Decision { player: 1, .. } = &trainer.tree.nodes[ni as usize] {
                let evs = trainer.scenario_ev_tracker.hand_ev_array(si, 1);
                if evs.iter().any(|&v| v != 0.0) {
                    found_p1 = true;
                    break;
                }
            }
        }
        assert!(found_p1, "position 1 should have some non-zero EVs at its decision nodes");
    }

    #[test]
    fn trainer_auto_detects_per_flop_buckets() {
        use crate::blueprint_v2::per_flop_bucket_file::PerFlopBucketFile;
        use crate::poker::{Card, Suit, Value};

        let dir = tempfile::tempdir().expect("tempdir");

        // Create a minimal per-flop bucket file (flop_0000.buckets)
        let flop = [
            Card::new(Value::Queen, Suit::Spade),
            Card::new(Value::Jack, Suit::Heart),
            Card::new(Value::Two, Suit::Diamond),
        ];
        let turn_card = Card::new(Value::Ace, Suit::Club);
        let river_card = Card::new(Value::Ten, Suit::Club);

        // 1 turn card x 1326 combos
        let turn_buckets = vec![0u16; 1326];
        // 1 turn x 1 river x 1326 combos
        let river_buckets_per_turn = vec![vec![0u16; 1326]];

        let pf = PerFlopBucketFile {
            flop_cards: flop,
            turn_bucket_count: 10,
            river_bucket_count: 10,
            turn_cards: vec![turn_card],
            turn_buckets,
            river_cards_per_turn: vec![vec![river_card]],
            river_buckets_per_turn,
        };
        pf.save(&dir.path().join("flop_0000.buckets")).expect("save per-flop file");

        let mut config = toy_config();
        config.training.cluster_path = Some(dir.path().to_string_lossy().into_owned());

        let trainer = toy_trainer(config);
        assert!(trainer.buckets.has_per_flop_dir(), "trainer should auto-detect per-flop bucket files");
    }

    #[test]
    fn trainer_no_per_flop_without_marker_file() {
        let dir = tempfile::tempdir().expect("tempdir");

        // Empty directory — no flop_0000.buckets
        let mut config = toy_config();
        config.training.cluster_path = Some(dir.path().to_string_lossy().into_owned());

        let trainer = toy_trainer(config);
        assert!(!trainer.buckets.has_per_flop_dir(), "trainer should not enable per-flop without marker file");
    }

    #[test]
    fn train_with_baselines_runs_iterations() {
        let mut config = toy_config();
        config.training.iterations = Some(50);
        config.training.use_baselines = true;
        config.training.baseline_alpha = 0.05;
        let mut trainer = toy_trainer(config);
        trainer.train().expect("training with baselines should complete");
        assert_eq!(trainer.iterations, 50);
    }

    #[test]
    fn train_with_baselines_updates_storage() {
        let mut config = toy_config();
        config.training.iterations = Some(20);
        config.training.use_baselines = true;
        config.training.baseline_alpha = 0.1;
        let mut trainer = toy_trainer(config);
        trainer.train().expect("training should complete");

        // Verify that some regrets have been updated (training happened).
        let any_nonzero = trainer
            .storage
            .regrets
            .iter()
            .any(|r| r.load(std::sync::atomic::Ordering::Relaxed) != 0);
        assert!(any_nonzero, "training with baselines should update regrets");
    }
}
