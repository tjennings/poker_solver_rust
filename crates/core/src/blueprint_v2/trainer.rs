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
use std::sync::Arc;
use std::time::Instant;

use rand::prelude::*;
use rand::rngs::{SmallRng, StdRng};
use rayon::prelude::*;

use super::bucket_file::BucketFile;
use super::bundle::{self, BlueprintV2Strategy};
use super::config::BlueprintV2Config;
use super::game_tree::GameTree;
use super::mccfr::{traverse_external, AllBuckets, Deal};
use super::storage::BlueprintStorage;
use crate::poker::{Card, ALL_SUITS, ALL_VALUES};

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

    // --- TUI shared state ---
    /// Iteration counter visible to the TUI thread.
    pub shared_iterations: Arc<AtomicU64>,
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
    /// Args: (scenario_index, node_idx, &BlueprintStorage, &GameTree)
    pub on_strategy_refresh: Option<Box<dyn Fn(usize, u32, &BlueprintStorage, &GameTree) + Send>>,
    /// Callback to push strategy delta values to TUI metrics.
    pub on_strategy_delta: Option<Box<dyn Fn(f64) + Send>>,
    /// Callback to push leaf movement fraction to TUI metrics.
    pub on_leaf_movement: Option<Box<dyn Fn(f64) + Send>>,
    /// Last time (in seconds) a strategy refresh was performed.
    last_strategy_refresh_secs: u64,

    // --- Strategy delta stopping ---
    /// Previous strategy-sum snapshot for delta computation.
    prev_strategy_sums: Option<Vec<i64>>,
    /// Most recent strategy delta value.
    pub last_strategy_delta: f64,
    /// Most recent fraction of info sets with max action delta > 0.20.
    pub last_pct_moving: f64,
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
            config.action_abstraction.max_raises,
        );

        let bucket_counts = [
            config.clustering.preflop.buckets,
            config.clustering.flop.buckets,
            config.clustering.turn.buckets,
            config.clustering.river.buckets,
        ];

        let storage = BlueprintStorage::new(&tree, bucket_counts);
        let bucket_files = load_bucket_files(Path::new(&config.training.cluster_path));
        let buckets = AllBuckets {
            bucket_counts,
            bucket_files,
        };
        let rng = StdRng::seed_from_u64(config.clustering.seed);

        let mut deck = [Card::new(ALL_VALUES[0], ALL_SUITS[0]); 52];
        let mut idx = 0;
        for &v in &ALL_VALUES {
            for &s in &ALL_SUITS {
                deck[idx] = Card::new(v, s);
                idx += 1;
            }
        }

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
            last_strategy_refresh_secs: 0,
            prev_strategy_sums: None,
            last_strategy_delta: f64::INFINITY,
            last_pct_moving: 1.0,
        }
    }

    /// Run the training loop until a stopping criterion is met.
    ///
    /// # Errors
    ///
    /// Returns an error if a snapshot write fails.
    pub fn train(&mut self) -> Result<(), Box<dyn Error>> {
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
                .map(|max| max.saturating_sub(self.iterations))
                .unwrap_or(batch_size);
            let this_batch = batch_size.min(remaining);
            if this_batch == 0 {
                break;
            }

            // 1. Generate batch of deals (sequential, fast).
            let deals: Vec<Deal> = (0..this_batch).map(|_| self.sample_deal()).collect();

            let prune = self.should_prune();
            let threshold = self.config.training.prune_threshold;

            // 2. Parallel traversal.
            let tree = &self.tree;
            let storage = &self.storage;
            let buckets = &self.buckets;

            deals.par_iter().for_each(|deal| {
                let mut rng = SmallRng::from_os_rng();

                traverse_external(
                    tree, storage, buckets, deal, 0, tree.root, prune, threshold, &mut rng,
                );
                traverse_external(
                    tree, storage, buckets, deal, 1, tree.root, prune, threshold, &mut rng,
                );
            });

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
        let mut idx = 0;
        for &v in &ALL_VALUES {
            for &s in &ALL_SUITS {
                self.deck[idx] = Card::new(v, s);
                idx += 1;
            }
        }

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
    /// Pruning activates after `prune_after_minutes` have elapsed and
    /// applies to `1 - prune_explore_pct` of iterations (the rest
    /// explore all actions to avoid permanently losing information).
    fn should_prune(&mut self) -> bool {
        let elapsed_min = self.elapsed_minutes();
        if elapsed_min < self.config.training.prune_after_minutes {
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
        if elapsed_min >= self.config.training.lcfr_warmup_minutes
            && elapsed_min >= self.last_discount_time + interval
        {
            self.apply_lcfr_discount();
        }

        // Strategy delta computation (same cadence as progress logging).
        if elapsed_min >= self.last_print_time + self.config.training.print_every_minutes {
            self.update_strategy_delta();
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
        // An interval of 0 means refresh on every check.
        let elapsed_secs = self.start_time.elapsed().as_secs();
        if elapsed_secs
            >= self.last_strategy_refresh_secs + self.strategy_refresh_interval_secs
        {
            if let Some(ref callback) = self.on_strategy_refresh {
                for (i, &node_idx) in self.scenario_node_indices.iter().enumerate() {
                    callback(i, node_idx, &self.storage, &self.tree);
                }
            }
            self.last_strategy_refresh_secs = elapsed_secs;
        }

        Ok(())
    }

    /// Apply LCFR (Linear CFR) discounting: multiply all regrets and
    /// strategy sums by `t / (t + 1)` where `t` is the number of
    /// discount intervals elapsed.
    fn apply_lcfr_discount(&mut self) {
        let elapsed_min = self.elapsed_minutes();
        let interval = self.config.training.lcfr_discount_interval.max(1);
        let t = elapsed_min / interval;
        let d = t as f64 / (t as f64 + 1.0);

        for atom in &self.storage.regrets {
            let v = atom.load(Ordering::Relaxed);
            atom.store((f64::from(v) * d) as i32, Ordering::Relaxed);
        }
        for atom in &self.storage.strategy_sums {
            let v = atom.load(Ordering::Relaxed);
            atom.store((v as f64 * d) as i64, Ordering::Relaxed);
        }

        self.last_discount_time = elapsed_min;
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

    /// Write a snapshot (regrets + metadata) to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the output directory cannot be created or
    /// the files cannot be written.
    pub fn save_snapshot(&mut self) -> Result<(), Box<dyn Error>> {
        let output_dir = Path::new(&self.config.snapshots.output_dir);
        std::fs::create_dir_all(output_dir)?;

        let snapshot_dir = output_dir.join(format!("snapshot_{:04}", self.snapshot_count));

        let mut strategy = BlueprintV2Strategy::from_storage(&self.storage, &self.tree);
        strategy.iterations = self.iterations;
        strategy.elapsed_minutes = self.elapsed_minutes();

        let metadata = format!(
            "{{\"iteration\": {}, \"elapsed_minutes\": {}, \"mean_positive_regret\": {:.2}}}",
            self.iterations,
            self.elapsed_minutes(),
            self.mean_positive_regret(),
        );

        bundle::save_snapshot(&snapshot_dir, &strategy, &self.storage, &metadata)?;

        self.snapshot_count += 1;
        self.last_snapshot_time = self.elapsed_minutes();

        if !self.tui_active {
            eprintln!("  Snapshot saved to {}", snapshot_dir.display());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::blueprint_v2::config::*;

    fn toy_config() -> BlueprintV2Config {
        BlueprintV2Config {
            game: GameConfig {
                players: 2,
                stack_depth: 10.0,
                small_blind: 0.5,
                big_blind: 1.0,
            },
            clustering: ClusteringConfig {
                algorithm: ClusteringAlgorithm::PotentialAwareEmd,
                preflop: StreetClusterConfig { buckets: 10 },
                flop: StreetClusterConfig { buckets: 10 },
                turn: StreetClusterConfig { buckets: 10 },
                river: StreetClusterConfig { buckets: 10 },
                seed: 42,
                kmeans_iterations: 50,
            },
            action_abstraction: ActionAbstractionConfig {
                preflop: vec![vec!["2.5bb".into()]],
                flop: vec![vec![1.0]],
                turn: vec![vec![1.0]],
                river: vec![vec![1.0]],
                max_raises: 1,
            },
            training: TrainingConfig {
                cluster_path: "clusters/".into(),
                iterations: Some(100),
                time_limit_minutes: None,
                lcfr_warmup_minutes: 0,
                lcfr_discount_interval: 1,
                prune_after_minutes: 9999,
                prune_threshold: -310_000_000,
                prune_explore_pct: 0.05,
                print_every_minutes: 9999,
                batch_size: 200,
                target_strategy_delta: None,
            },
            snapshots: SnapshotConfig {
                warmup_minutes: 9999,
                snapshot_every_minutes: 9999,
                output_dir: "/tmp/test_blueprint_v2_snapshots".into(),
            },
        }
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
        let mut trainer = BlueprintTrainer::new(config);
        trainer.train().expect("training should complete");
        assert_eq!(trainer.iterations, 50);
    }

    #[test]
    fn train_updates_storage() {
        let mut config = toy_config();
        config.training.iterations = Some(20);
        let mut trainer = BlueprintTrainer::new(config);

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
    fn snapshot_save() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let mut config = toy_config();
        config.snapshots.output_dir = dir.path().to_string_lossy().to_string();
        config.training.iterations = Some(10);
        let mut trainer = BlueprintTrainer::new(config);

        trainer.train().expect("training should complete");
        trainer.save_snapshot().expect("snapshot should save");

        let snapshot_dir = dir.path().join("snapshot_0000");
        assert!(snapshot_dir.join("strategy.bin").exists());
        assert!(snapshot_dir.join("regrets.bin").exists());
        assert!(snapshot_dir.join("metadata.json").exists());
    }

    #[test]
    fn train_batch_iterations() {
        let mut config = toy_config();
        config.training.iterations = Some(50);
        config.training.batch_size = 10;
        let mut trainer = BlueprintTrainer::new(config);
        trainer.train().expect("training should complete");
        assert_eq!(trainer.iterations, 50);
    }

    #[test]
    fn parallel_batch_produces_regret_updates() {
        let mut config = toy_config();
        config.training.iterations = Some(200);
        config.training.batch_size = 50;
        let mut trainer = BlueprintTrainer::new(config);
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
        let mut trainer = BlueprintTrainer::new(config);
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
        let mut trainer = BlueprintTrainer::new(config);
        trainer.train().expect("training should complete");
        // Should have stopped due to delta, not the 1-minute limit.
        assert!(trainer.iterations > 0, "should have run some iterations");
        assert!(
            trainer.last_strategy_delta <= 0.5,
            "should have stopped when delta <= 0.5, got {}",
            trainer.last_strategy_delta,
        );
    }
}
