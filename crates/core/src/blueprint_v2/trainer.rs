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
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::Instant;

use rand::prelude::*;
use rand::rngs::{SmallRng, StdRng};
use rayon::prelude::*;

use super::baseline_validation::{
    BaselineDocument, BaselineGamePreconditions, BaselineValidationConfig,
    BaselineValidationReport, format_baseline_validation_lines, parse_baseline_json,
    validate_baseline,
};
use super::bucket_file::BucketFile;
use super::bundle::{self, BlueprintV2Strategy};
use super::config::{BlueprintV2Config, SnapshotFormat};
use super::game_tree::GameTree;
use super::mccfr::{
    AllBuckets, Deal, DealWithBuckets, FullTreeEvTracker, PRUNE_HITS, PRUNE_TOTAL, PruneStats,
    ScenarioEvTracker, traverse_best_response, traverse_external,
};
use super::sparse_storage::SparseBlueprintStorage;
use super::storage::{BlueprintCfrStorage, BlueprintStorage, CfrStorageStats};
use crate::cfr::optimizer::{BrcfrPlusOptimizer, CfrOptimizer, DcfrOptimizer, SapcfrPlusOptimizer};
use crate::hands::CanonicalHand;
use crate::poker::{ALL_SUITS, ALL_VALUES, Card};

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
        hole_cards: [[deck[0], deck[1]], [deck[2], deck[3]]],
        board: [deck[4], deck[5], deck[6], deck[7], deck[8]],
    }
}

/// Check whether all 4 bucket files already exist in the given directory.
#[must_use]
pub fn bucket_files_exist(dir: &Path) -> bool {
    [
        "river.buckets",
        "turn.buckets",
        "flop.buckets",
        "preflop.buckets",
    ]
    .iter()
    .all(|name| dir.join(name).exists())
}

/// Attempt to load `.buckets` files from the given directory.
///
/// Looks for `preflop.buckets`, `flop.buckets`, `turn.buckets`, and
/// `river.buckets`. Missing files are silently skipped (returning `None`
/// for that street). Load errors are logged to stderr but do not cause
/// a hard failure.
pub fn load_bucket_files(dir: &Path) -> [Option<BucketFile>; 4] {
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
                    eprintln!(
                        "  Loaded bucket file: {} ({} boards, {} combos/board, {} buckets)",
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
type RandomScenarioCallback = Box<dyn Fn(&BlueprintStorage, &GameTree, &[[f64; 169]; 2]) + Send>;

/// Callback type for pushing baseline convergence reports to the TUI.
type BaselineValidationCallback = Box<dyn Fn(BaselineValidationReport) + Send>;

/// Outer training driver for Blueprint V2.
///
/// Holds the game tree, regret/strategy storage, bucket lookup, and all
/// timing state needed to orchestrate LCFR-weighted MCCFR training with
/// periodic snapshots.
pub struct BlueprintTrainer {
    pub tree: GameTree,
    /// Dense storage when `training.storage_backend = "dense"`.
    ///
    /// When the active backend is sparse, this is a zero-slot compatibility
    /// placeholder; use [`dense_storage_projection`](Self::dense_storage_projection)
    /// to obtain a dense export/resume view.
    pub storage: BlueprintStorage,
    sparse_storage: Option<SparseBlueprintStorage>,
    pub buckets: AllBuckets,
    pub config: BlueprintV2Config,
    pub rng: StdRng,
    pub start_time: Instant,
    pub iterations: u64,
    last_discount_time: u64,
    last_print_time: u64,
    /// Iteration of the most recent BR prediction pass (BRCFR+).
    pub last_br_iteration: u64,
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
    /// One-shot trigger: the TUI sets this to request an immediate strategy refresh.
    pub strategy_refresh_trigger: Arc<AtomicBool>,

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

    // --- Regret audit ---
    /// Called at strategy-refresh interval to update regret audit state.
    pub on_audit_refresh: Option<Box<dyn FnMut(&BlueprintStorage) + Send>>,

    // --- Exploitability measurement ---
    /// Callback to push top exploitable spots to TUI after a BR pass.
    pub on_exploitable_spots:
        Option<Box<dyn Fn(Vec<super::exploitable_spots::ExploitableSpot>) + Send>>,
    /// Callback to push exploitability values to TUI metrics.
    pub on_exploitability: Option<Box<dyn Fn(f64) + Send>>,
    /// Called at the start of an exploitability/BR pass with the total sample count.
    pub on_exploitability_start: Option<Box<dyn Fn(u64) + Send>>,
    /// Called once per completed deal during an exploitability/BR pass.
    /// Must be `Arc` (not `Box`) because it is shared across rayon threads.
    pub on_exploitability_tick: Option<Arc<dyn Fn() + Send + Sync>>,
    /// Called when an exploitability/BR pass finishes.
    pub on_exploitability_finish: Option<Box<dyn Fn() + Send>>,
    /// Last time (in minutes) an exploitability measurement was performed.
    last_exploitability_time: u64,

    // --- External baseline validation ---
    baseline_validation_document: Option<BaselineDocument>,
    last_baseline_validation_iteration: u64,
    last_baseline_validation_time: u64,
    /// Callback to push baseline strategy-frequency validation reports to TUI.
    pub on_baseline_validation: Option<BaselineValidationCallback>,

    // --- Config reload ---
    /// One-shot trigger: the TUI sets this to request a config reload.
    pub config_reload_trigger: Arc<AtomicBool>,
    /// Callback invoked when config reload is triggered.
    pub on_config_reload: Option<Box<dyn FnMut(&GameTree, &BlueprintStorage) + Send>>,
    /// After a config reload, the callback stores new node indices here
    /// so the trainer can update `scenario_node_indices` and `scenario_ev_tracker`.
    pub reloaded_node_indices: Arc<std::sync::Mutex<Option<Vec<u32>>>>,
}

impl BlueprintTrainer {
    fn configured_storage_backend(config: &BlueprintV2Config) -> String {
        config.training.storage_backend.trim().to_ascii_lowercase()
    }

    fn reject_unsupported_storage_backend(config: &BlueprintV2Config) {
        let backend = Self::configured_storage_backend(config);
        match backend.as_str() {
            "dense" | "sparse" | "lazy" => {}
            _ => panic!(
                "unsupported blueprint_v2 storage_backend={}; expected \"dense\" or \"sparse\"",
                config.training.storage_backend
            ),
        }

        if matches!(backend.as_str(), "sparse" | "lazy")
            && config
                .training
                .optimizer
                .trim()
                .eq_ignore_ascii_case("brcfr+")
        {
            panic!(
                "blueprint_v2 sparse storage does not support brcfr+ in this slice; use storage_backend=\"dense\" or optimizer=\"dcfr\"/\"sapcfr+\""
            );
        }
    }

    fn new_optimizer(config: &BlueprintV2Config) -> Arc<dyn CfrOptimizer> {
        if config.training.optimizer == "brcfr+" {
            Arc::new(BrcfrPlusOptimizer::new(
                config.training.dcfr_alpha,
                config.training.dcfr_gamma,
                config.training.brcfr_eta,
            ))
        } else if config.training.optimizer == "sapcfr+" {
            Arc::new(SapcfrPlusOptimizer {
                alpha: config.training.dcfr_alpha,
                gamma: config.training.dcfr_gamma,
                eta: config.training.sapcfr_eta,
            })
        } else {
            Arc::new(DcfrOptimizer {
                alpha: config.training.dcfr_alpha,
                beta: config.training.dcfr_beta,
                gamma: config.training.dcfr_gamma,
            })
        }
    }

    fn configure_optimizer(config: &BlueprintV2Config, optimizer: &Arc<dyn CfrOptimizer>) {
        let samples = config.training.exploitability_samples.max(1) as f64;
        let br_interval = config.training.brcfr_interval.max(1) as f64;
        optimizer.set_br_scale(br_interval / samples);
    }

    fn scaled_regret_floor(config: &BlueprintV2Config) -> Option<i32> {
        config
            .training
            .regret_floor
            .map(|floor| (floor as f64 * super::storage::REGRET_SCALE) as i32)
    }

    fn configure_dense_storage(config: &BlueprintV2Config, storage: &mut BlueprintStorage) {
        if let Some(scaled_floor) = Self::scaled_regret_floor(config) {
            storage.set_regret_floor(scaled_floor);
        }
        let optimizer = Self::new_optimizer(config);
        Self::configure_optimizer(config, &optimizer);
        if optimizer.needs_predictions() {
            storage.enable_predictions();
        }
        storage.set_optimizer(optimizer);
    }

    fn configure_sparse_storage(config: &BlueprintV2Config, storage: &mut SparseBlueprintStorage) {
        if let Some(scaled_floor) = Self::scaled_regret_floor(config) {
            storage.set_regret_floor(scaled_floor);
        }
        let optimizer = Self::new_optimizer(config);
        Self::configure_optimizer(config, &optimizer);
        if optimizer.needs_predictions() {
            storage.enable_predictions();
        }
        storage.set_optimizer(optimizer);
    }

    fn build_sparse_storage(
        tree: &GameTree,
        bucket_counts: [u16; 4],
        config: &BlueprintV2Config,
    ) -> SparseBlueprintStorage {
        let mut storage = SparseBlueprintStorage::new_with_baselines(
            tree,
            bucket_counts,
            config.training.use_baselines,
        );
        Self::configure_sparse_storage(config, &mut storage);
        storage
    }

    fn clamp_loaded_regrets(config: &BlueprintV2Config, storage: &BlueprintStorage) {
        if let Some(floor) = config.training.regret_floor {
            let scaled_floor = (floor as f64 * super::storage::REGRET_SCALE) as i32;
            let mut clamped = 0u64;
            for atom in &storage.regrets {
                let v = atom.load(Ordering::Relaxed);
                if v < scaled_floor {
                    atom.store(scaled_floor, Ordering::Relaxed);
                    clamped += 1;
                }
            }
            if clamped > 0 {
                eprintln!("  Clamped {clamped} regret values to floor ({floor} chips)");
            }
        }
    }

    fn active_storage(&self) -> &(dyn BlueprintCfrStorage + Sync) {
        self.sparse_storage
            .as_ref()
            .map_or(&self.storage as &(dyn BlueprintCfrStorage + Sync), |s| {
                s as &(dyn BlueprintCfrStorage + Sync)
            })
    }

    #[must_use]
    pub fn is_sparse_storage(&self) -> bool {
        self.sparse_storage.is_some()
    }

    #[must_use]
    pub fn storage_stats(&self) -> CfrStorageStats {
        self.active_storage().storage_stats()
    }

    #[must_use]
    pub fn dense_storage_projection(&self) -> BlueprintStorage {
        if let Some(ref sparse) = self.sparse_storage {
            return sparse.to_dense_storage(&self.tree);
        }

        let dense = BlueprintStorage::new_unlogged(&self.tree, self.storage.bucket_counts);
        let (regrets, sums) = self.storage.project_dense_regrets_and_sums(&self.tree);
        debug_assert_eq!(regrets.len(), dense.regrets.len());
        debug_assert_eq!(sums.len(), dense.strategy_sums.len());
        for (atom, value) in dense.regrets.iter().zip(regrets) {
            atom.store(value, Ordering::Relaxed);
        }
        for (atom, value) in dense.strategy_sums.iter().zip(sums) {
            atom.store(value, Ordering::Relaxed);
        }
        dense
    }

    fn with_dense_storage<R>(&self, f: impl FnOnce(&BlueprintStorage) -> R) -> R {
        if self.sparse_storage.is_some() {
            let dense = self.dense_storage_projection();
            f(&dense)
        } else {
            f(&self.storage)
        }
    }

    fn active_strategy_sums_snapshot(&self) -> Vec<i64> {
        if self.sparse_storage.is_some() {
            let (_, sums) = self
                .active_storage()
                .project_dense_regrets_and_sums(&self.tree);
            sums
        } else {
            self.storage.snapshot_strategy_sums()
        }
    }

    /// Build a trainer from a config: constructs the game tree, allocates
    /// storage, and initialises timing state.
    #[must_use]
    pub fn new(config: BlueprintV2Config) -> Self {
        Self::reject_unsupported_storage_backend(&config);

        let tree = GameTree::build_with_options(
            config.game.stack_depth,
            config.game.small_blind,
            config.game.big_blind,
            &config.action_abstraction.preflop,
            &config.action_abstraction.flop,
            &config.action_abstraction.turn,
            &config.action_abstraction.river,
            config.game.allow_preflop_limp,
        );

        let bucket_counts = [
            config.clustering.preflop.buckets,
            config.clustering.flop.buckets,
            config.clustering.turn.buckets,
            config.clustering.river.buckets,
        ];

        let backend = Self::configured_storage_backend(&config);
        let (storage, sparse_storage) = if matches!(backend.as_str(), "sparse" | "lazy") {
            let sparse_storage = Self::build_sparse_storage(&tree, bucket_counts, &config);
            (
                BlueprintStorage::new_projection_stub(&tree, bucket_counts),
                Some(sparse_storage),
            )
        } else {
            let mut storage = BlueprintStorage::new_with_baselines(
                &tree,
                bucket_counts,
                config.training.use_baselines,
            );
            Self::configure_dense_storage(&config, &mut storage);
            (storage, None)
        };

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
                            panic!(
                                "failed to copy {} to {}: {e}",
                                entry.path().display(),
                                dest_file.display()
                            )
                        });
                        copied += 1;
                    }
                }
                eprintln!(
                    "  Copied {copied} bucket files from {cluster_path} to {}",
                    dest.display()
                );
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
            sparse_storage,
            buckets,
            config,
            rng,
            start_time: Instant::now(),
            iterations: 0,
            last_discount_time: 0,
            last_print_time: 0,
            last_br_iteration: 0,
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
            strategy_refresh_trigger: Arc::new(AtomicBool::new(false)),
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
            on_audit_refresh: None,
            on_exploitable_spots: None,
            on_exploitability: None,
            on_exploitability_start: None,
            on_exploitability_tick: None,
            on_exploitability_finish: None,
            last_exploitability_time: 0,
            baseline_validation_document: None,
            last_baseline_validation_iteration: 0,
            last_baseline_validation_time: 0,
            on_baseline_validation: None,
            config_reload_trigger: Arc::new(AtomicBool::new(false)),
            on_config_reload: None,
            reloaded_node_indices: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Attempt to resume from the latest snapshot in `output_dir`.
    ///
    /// When `config.snapshots.resume` is `true`, scans for valid
    /// `snapshot_NNNN` directories and `final/`, picks the newest candidate
    /// by metadata iteration, metadata elapsed minutes, final checkpoint
    /// status, then snapshot number, and loads its `regrets.bin` and
    /// `metadata.json`.
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

        // Find the latest valid snapshot directory. Metadata wins over
        // directory naming so a stale `final/` cannot mask a newer numbered
        // checkpoint, while an equal-or-newer `final/` is preferred.
        let mut candidates = Vec::new();
        let mut highest_numbered_snapshot = None;

        if let Ok(entries) = std::fs::read_dir(output_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if let Some(num_str) = name_str.strip_prefix("snapshot_")
                    && let Ok(num) = num_str.parse::<u32>()
                    && let Some(candidate) =
                        ResumeCandidate::from_dir(entry.path(), Some(num), false)
                {
                    highest_numbered_snapshot = Some(
                        highest_numbered_snapshot.map_or(num, |highest: u32| highest.max(num)),
                    );
                    candidates.push(candidate);
                }
            }
        }

        let final_dir = output_dir.join("final");
        if let Some(candidate) = ResumeCandidate::from_dir(final_dir, None, true) {
            candidates.push(candidate);
        }

        let Some(candidate) = candidates.into_iter().max() else {
            eprintln!("Resume: no snapshots found in {}", output_dir.display());
            return Ok(());
        };
        let snapshot_dir = candidate.path;

        // Load regrets.
        let regrets_path = snapshot_dir.join("regrets.bin");
        let bucket_counts = [
            self.config.clustering.preflop.buckets,
            self.config.clustering.flop.buckets,
            self.config.clustering.turn.buckets,
            self.config.clustering.river.buckets,
        ];
        let mut loaded = BlueprintStorage::load_regrets(&regrets_path, &self.tree, bucket_counts)?;
        Self::clamp_loaded_regrets(&self.config, &loaded);

        let backend = Self::configured_storage_backend(&self.config);
        if matches!(backend.as_str(), "sparse" | "lazy") {
            let sparse_storage =
                Self::build_sparse_storage(&self.tree, bucket_counts, &self.config);
            sparse_storage.load_dense_projection(&self.tree, &loaded);
            self.storage = BlueprintStorage::new_projection_stub(&self.tree, bucket_counts);
            self.sparse_storage = Some(sparse_storage);
        } else {
            Self::configure_dense_storage(&self.config, &mut loaded);
            if self.config.training.use_baselines && loaded.baselines.is_none() {
                let total = loaded.regrets.len();
                loaded.baselines = Some(
                    (0..total)
                        .map(|_| std::sync::atomic::AtomicI32::new(0))
                        .collect(),
                );
            }
            self.storage = loaded;
            self.sparse_storage = None;
        }

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
                self.start_time = Instant::now()
                    .checked_sub(backdate)
                    .unwrap_or(self.start_time);
                self.last_snapshot_time = prev_min;
            }
        }

        self.snapshot_count = candidate.snapshot_num.map_or_else(
            || highest_numbered_snapshot.map_or(1, |num| num + 1),
            |num| num + 1,
        );

        // Prevent spurious immediate discount: align last_discount_time
        // with the restored iteration count so the next discount waits
        // a full interval.
        self.last_discount_time = self.iterations;

        // Load full-tree EV tracker if available.
        let hand_ev_path = snapshot_dir.join("hand_ev.bin");
        if hand_ev_path.exists() {
            self.full_ev_tracker.load(&hand_ev_path)?;
        }

        // Seed the strategy-delta baseline so the first check after resume
        // compares against the loaded state instead of producing zero.
        self.prev_strategy_sums = Some(self.active_strategy_sums_snapshot());

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
        self.validate_training_startup()?;

        while !self.should_stop() {
            // Honour pause requests from the TUI.
            while self.paused.load(Ordering::Relaxed) {
                if self.quit_requested.load(Ordering::Relaxed) {
                    return Ok(());
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }

            if self.run_batch_with_budget(None)? == 0 {
                break;
            }
        }
        Ok(())
    }

    /// Run startup validations that must pass before any training batch executes.
    pub(crate) fn validate_training_startup(&mut self) -> Result<(), Box<dyn Error>> {
        self.initialize_baseline_validation()?;

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

        Ok(())
    }

    /// Run one parallel MCCFR batch, capped by config and an optional runtime budget.
    ///
    /// Returns the number of completed iterations.
    pub fn run_batch_with_budget(&mut self, max_units: Option<u64>) -> Result<u64, Box<dyn Error>> {
        let batch_size = self.config.training.batch_size;

        // Calculate how many iterations remain (respect iteration limit).
        let remaining = self
            .config
            .training
            .iterations
            .map_or(batch_size, |max| max.saturating_sub(self.iterations));
        let this_batch = max_units
            .map_or(batch_size, |max| batch_size.min(max))
            .min(remaining);
        if this_batch == 0 {
            return Ok(0);
        }

        // Pre-seed per-deal RNGs from the main RNG (only sequential part).
        let thread_seeds: Vec<u64> = (0..this_batch).map(|_| self.rng.random()).collect();

        let prune = self.should_prune();
        // Config prune_threshold is in chip units; scale to match stored regrets.
        let threshold =
            (f64::from(self.config.training.prune_threshold) * super::storage::REGRET_SCALE) as i32;
        let prune_streets = self.config.training.prune_street_mask();

        let tree = &self.tree;
        let storage = self.active_storage();
        let buckets_ref = &self.buckets;
        let ev_tracker = &self.scenario_ev_tracker;
        let full_ev_tracker = &self.full_ev_tracker;
        let visit_counters = &self.bucket_visits;

        let rake_rate = self.config.game.rake_rate;
        let rake_cap = self.config.game.rake_cap;
        let baseline_alpha = if self.config.training.use_baselines {
            self.config.training.baseline_alpha
        } else {
            0.0
        };

        // Fully parallel: each thread samples its own deal, precomputes
        // buckets, and traverses — no sequential deal generation.
        let batch_prune_stats: PruneStats = thread_seeds
            .into_par_iter()
            .map(|seed| {
                let mut rng = SmallRng::seed_from_u64(seed);
                let deal = sample_deal_with_rng(&mut rng);
                let buckets = buckets_ref.precompute_buckets(&deal);

                // Count bucket visits (both players, all streets).
                for player_buckets in &buckets {
                    for (street, &bucket) in player_buckets.iter().enumerate() {
                        if (bucket as usize) < visit_counters[street].len() {
                            visit_counters[street][bucket as usize].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }

                let deal = DealWithBuckets { deal, buckets };
                let mut stats = PruneStats::default();

                let (_, s0) = traverse_external(
                    tree,
                    storage,
                    &deal,
                    0,
                    tree.root,
                    prune,
                    threshold,
                    prune_streets,
                    &mut rng,
                    rake_rate,
                    rake_cap,
                    Some(ev_tracker),
                    Some(full_ev_tracker),
                    baseline_alpha,
                );
                stats.merge(s0);
                let (_, s1) = traverse_external(
                    tree,
                    storage,
                    &deal,
                    1,
                    tree.root,
                    prune,
                    threshold,
                    prune_streets,
                    &mut rng,
                    rake_rate,
                    rake_cap,
                    Some(ev_tracker),
                    Some(full_ev_tracker),
                    baseline_alpha,
                );
                stats.merge(s1);

                stats
            })
            .reduce(PruneStats::default, |mut a, b| {
                a.merge(b);
                a
            });

        // Single atomic update for the whole batch.
        PRUNE_HITS.fetch_add(batch_prune_stats.hits, Ordering::Relaxed);
        PRUNE_TOTAL.fetch_add(batch_prune_stats.total, Ordering::Relaxed);

        // 3. Sequential: update counters and check timed actions.
        self.iterations += this_batch;
        self.shared_iterations
            .store(self.iterations, Ordering::Relaxed);
        self.check_timed_actions()?;

        Ok(this_batch)
    }

    fn initialize_baseline_validation(&mut self) -> Result<(), Box<dyn Error>> {
        if !self.config.training.baseline_validation.enabled {
            return Ok(());
        }
        if self.baseline_validation_document.is_none() {
            let path = self
                .config
                .training
                .baseline_validation
                .baseline_path
                .as_ref()
                .ok_or("training.baseline_validation.enabled requires baseline_path")?;
            let raw = std::fs::read_to_string(path).map_err(|err| {
                format!(
                    "failed to read baseline validation file {}: {err}",
                    path.display()
                )
            })?;
            let document = parse_baseline_json(&raw)?;
            self.baseline_validation_document = Some(document);
            if !self.tui_active {
                eprintln!("  Loaded baseline validation: {}", path.display());
            }
        }

        self.run_baseline_validation("initial")?;
        self.last_baseline_validation_iteration = self.iterations;
        self.last_baseline_validation_time = self.elapsed_minutes();
        Ok(())
    }

    fn should_run_baseline_validation(&self, elapsed_min: u64) -> bool {
        if !self.config.training.baseline_validation.enabled
            || self.baseline_validation_document.is_none()
        {
            return false;
        }

        let settings = &self.config.training.baseline_validation;
        let iteration_due = settings.interval_iterations > 0
            && self.iterations
                >= self.last_baseline_validation_iteration + settings.interval_iterations;
        let time_due = settings.interval_minutes > 0
            && elapsed_min >= self.last_baseline_validation_time + settings.interval_minutes;
        iteration_due || time_due
    }

    fn baseline_validation_preconditions(&self) -> BaselineGamePreconditions {
        BaselineGamePreconditions {
            starting_stack: self.config.game.stack_depth,
            small_blind: self.config.game.small_blind,
            big_blind: self.config.game.big_blind,
            allow_preflop_limp: self.config.game.allow_preflop_limp,
        }
    }

    /// Compute a validation report against the active storage backend.
    ///
    /// This intentionally reads through `BlueprintCfrStorage::average_strategy`
    /// via the validator provider boundary and does not project sparse storage
    /// to dense storage.
    #[must_use]
    pub fn baseline_validation_report(&self) -> Option<BaselineValidationReport> {
        let baseline = self.baseline_validation_document.as_ref()?;
        let settings = &self.config.training.baseline_validation;
        let top_n = settings
            .top_n_spots
            .max(settings.top_n_combos_per_spot)
            .max(1);
        let mut report = validate_baseline(
            baseline,
            &self.tree,
            self.active_storage(),
            BaselineValidationConfig {
                game: Some(self.baseline_validation_preconditions()),
                top_n,
            },
        );
        report.worst_spots.truncate(settings.top_n_spots);
        report
            .worst_combo_rows
            .truncate(settings.top_n_combos_per_spot);
        Some(report)
    }

    fn run_baseline_validation(&mut self, reason: &str) -> Result<(), Box<dyn Error>> {
        let Some(report) = self.baseline_validation_report() else {
            return Ok(());
        };
        let settings = &self.config.training.baseline_validation;

        if !self.tui_active {
            eprintln!(
                "  Baseline validation ({reason}) at iter={}",
                self.iterations
            );
            for line in format_baseline_validation_lines(
                &report,
                settings.top_n_spots,
                settings.top_n_combos_per_spot,
            ) {
                eprintln!("{line}");
            }
        }

        if let Some(ref cb) = self.on_baseline_validation {
            cb(report.clone());
        }

        if !report.precondition_failures.is_empty() {
            let first = &report.precondition_failures[0];
            return Err(format!(
                "baseline validation preconditions failed: {} expected={} actual={} ({})",
                first.field, first.expected, first.actual, first.reason
            )
            .into());
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
            hole_cards: [[self.deck[0], self.deck[1]], [self.deck[2], self.deck[3]]],
            board: [
                self.deck[4],
                self.deck[5],
                self.deck[6],
                self.deck[7],
                self.deck[8],
            ],
        }
    }

    /// Compute sample-based best-response exploitability in mbb/hand.
    ///
    /// Samples N random deals, computes BR value for each player,
    /// and returns the average exploitability.
    pub fn compute_exploitability(&self) -> f64 {
        let n = self.config.training.exploitability_samples.max(1);
        let tree = &self.tree;
        let projected_storage;
        let storage = if self.sparse_storage.is_some() {
            projected_storage = Some(self.dense_storage_projection());
            projected_storage.as_ref().expect("projection exists")
        } else {
            &self.storage
        };
        let buckets_ref = &self.buckets;
        let rake_rate = self.config.game.rake_rate;
        let rake_cap = self.config.game.rake_cap;
        let big_blind = self.config.game.big_blind;

        if let Some(ref cb) = self.on_exploitability_start {
            cb(n);
        }
        let tick = self.on_exploitability_tick.clone();

        let (sum_p0, sum_p1): (f64, f64) = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut rng = SmallRng::seed_from_u64(i.wrapping_mul(0x9E3779B97F4A7C15));
                let deal = sample_deal_with_rng(&mut rng);
                let buckets = buckets_ref.precompute_buckets(&deal);
                let deal = DealWithBuckets { deal, buckets };

                let br0 = traverse_best_response(
                    tree, storage, &deal, 0, tree.root, rake_rate, rake_cap, None, None,
                );
                let br1 = traverse_best_response(
                    tree, storage, &deal, 1, tree.root, rake_rate, rake_cap, None, None,
                );
                if let Some(ref cb) = tick {
                    cb();
                }
                (br0, br1)
            })
            .reduce(|| (0.0, 0.0), |(a0, a1), (b0, b1)| (a0 + b0, a1 + b1));

        if let Some(ref cb) = self.on_exploitability_finish {
            cb();
        }

        let n_f = n as f64;
        // Exploitability in mbb/hand: avg BR value in chips, convert to mBB.
        (sum_p0 / n_f + sum_p1 / n_f) / 2.0 / big_blind * 1000.0
    }

    fn run_br_prediction_pass(&mut self) -> f64 {
        assert!(
            self.sparse_storage.is_none(),
            "brcfr+ prediction pass requires dense blueprint_v2 storage"
        );
        let n = self.config.training.exploitability_samples.max(1);
        let tree = &self.tree;
        let storage = &self.storage;
        let buckets_ref = &self.buckets;
        let rake_rate = self.config.game.rake_rate;
        let rake_cap = self.config.game.rake_cap;
        let big_blind = self.config.game.big_blind;

        if let Some(ref cb) = self.on_exploitability_start {
            cb(n);
        }
        let tick = self.on_exploitability_tick.clone();

        // Unlock and zero predictions for fresh accumulation.
        self.storage.unlock_predictions();
        self.storage.zero_predictions();

        // Allocate temporary visit counts (same layout as predictions).
        let pred_len = self.storage.predictions.as_ref().map_or(0, |p| p.len());
        let visit_counts: Vec<AtomicU32> = (0..pred_len).map(|_| AtomicU32::new(0)).collect();
        let vc_ref = &visit_counts;

        let (sum_p0, sum_p1): (f64, f64) = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut rng = SmallRng::seed_from_u64(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let deal = sample_deal_with_rng(&mut rng);
                let buckets = buckets_ref.precompute_buckets(&deal);
                let deal = DealWithBuckets { deal, buckets };

                let br0 = traverse_best_response(
                    tree,
                    storage,
                    &deal,
                    0,
                    tree.root,
                    rake_rate,
                    rake_cap,
                    Some(storage),
                    Some(vc_ref),
                );
                let br1 = traverse_best_response(
                    tree,
                    storage,
                    &deal,
                    1,
                    tree.root,
                    rake_rate,
                    rake_cap,
                    Some(storage),
                    Some(vc_ref),
                );
                if let Some(ref cb) = tick {
                    cb();
                }
                (br0, br1)
            })
            .reduce(|| (0.0, 0.0), |(a0, a1), (b0, b1)| (a0 + b0, a1 + b1));

        if let Some(ref cb) = self.on_exploitability_finish {
            cb();
        }

        // Normalize predictions by visit count.
        if let Some(ref preds) = self.storage.predictions {
            for (i, pred) in preds.iter().enumerate() {
                let count = visit_counts[i].load(Ordering::Relaxed);
                if count > 0 {
                    let val = pred.load(Ordering::Relaxed);
                    pred.store(val / count as i32, Ordering::Relaxed);
                }
            }
        }

        // Lock predictions to prevent MCCFR from overwriting.
        self.storage.lock_predictions();
        self.last_br_iteration = self.iterations;

        // Find and push top exploitable spots to TUI.
        if let Some(ref cb) = self.on_exploitable_spots {
            let spots =
                super::exploitable_spots::find_top_exploitable_spots(&self.tree, &self.storage, 10);
            cb(spots);
        }

        let n_f = n as f64;
        (sum_p0 / n_f + sum_p1 / n_f) / 2.0 / big_blind * 1000.0
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
        if self.trainer_specific_stop_reached() {
            return true;
        }
        false
    }

    /// True when a trainer-owned stopping condition outside shared runtime limits is met.
    pub(crate) fn trainer_specific_stop_reached(&self) -> bool {
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
        let tui_refresh_triggered = self.strategy_refresh_trigger.swap(false, Ordering::Relaxed);
        let refresh_due = tui_refresh_triggered
            || elapsed_secs
                >= self.last_strategy_refresh_secs + self.strategy_refresh_interval_secs;

        if print_due || refresh_due {
            self.update_strategy_delta();
        }

        if print_due {
            self.print_metrics();
        }

        if self.should_run_baseline_validation(elapsed_min) {
            self.run_baseline_validation("cadence")?;
            self.last_baseline_validation_iteration = self.iterations;
            self.last_baseline_validation_time = elapsed_min;
        }

        // Config reload: TUI-triggered.
        if self.config_reload_trigger.swap(false, Ordering::Relaxed) {
            self.reload_config_now();
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
            self.push_strategy_refresh(elapsed_secs);
        }

        // Exploitability measurement.
        if self.config.training.exploitability_interval_minutes > 0 {
            let interval = self.config.training.exploitability_interval_minutes;
            if elapsed_min >= self.last_exploitability_time + interval {
                let exploit = self.compute_exploitability();
                if !self.tui_active {
                    eprintln!("  Exploitability: {exploit:.2} mbb/hand");
                }
                if let Some(ref cb) = self.on_exploitability {
                    cb(exploit);
                }
                self.last_exploitability_time = elapsed_min;
            }
        }

        // Random scenario carousel rotation.
        if let Some(ref callback) = self.on_random_scenario
            && elapsed_min >= self.last_random_scenario_min + self.random_scenario_hold_minutes
        {
            // Use scenario 0 (root) EVs for random scenario display.
            let hand_evs = if self.scenario_ev_tracker.node_indices.is_empty() {
                [[0.0; 169], [0.0; 169]]
            } else {
                [
                    self.scenario_ev_tracker.hand_ev_array(0, 0),
                    self.scenario_ev_tracker.hand_ev_array(0, 1),
                ]
            };
            let projected_storage = self
                .sparse_storage
                .is_some()
                .then(|| self.dense_storage_projection());
            let callback_storage = projected_storage.as_ref().unwrap_or(&self.storage);
            callback(callback_storage, &self.tree, &hand_evs);
            self.last_random_scenario_min = elapsed_min;
        }

        // BRCFR+ BR prediction pass (iteration-based).
        if self.config.training.optimizer == "brcfr+" {
            let warmup = self.config.training.brcfr_warmup_iterations;
            let interval = self.config.training.brcfr_interval.max(1);

            if self.iterations >= warmup {
                let should_br = if self.last_br_iteration < warmup {
                    true // First pass right at warmup boundary
                } else {
                    self.iterations >= self.last_br_iteration + interval
                };

                if should_br {
                    let exploit = self.run_br_prediction_pass();
                    if !self.tui_active {
                        eprintln!("  BRCFR+ BR pass: exploitability = {exploit:.2} mbb/hand");
                    }
                    if let Some(ref cb) = self.on_exploitability {
                        cb(exploit);
                    }
                }
            }

            // Update decay on the optimizer.
            let decay = if self.iterations < warmup || self.last_br_iteration == 0 {
                0.0
            } else {
                let elapsed_iters = self.iterations.saturating_sub(self.last_br_iteration) as f64;
                (1.0 - elapsed_iters / interval as f64).max(0.0)
            };
            if let Some(ref sparse) = self.sparse_storage {
                sparse.set_optimizer_decay(decay);
            } else if let Some(ref opt) = self.storage.optimizer {
                opt.set_decay(decay);
            }
        }

        Ok(())
    }

    /// Run the config-reload callback path without evaluating timed snapshot cadence.
    pub fn reload_config_now(&mut self) {
        let projected_storage = self
            .sparse_storage
            .is_some()
            .then(|| self.dense_storage_projection());
        let callback_storage = projected_storage.as_ref().unwrap_or(&self.storage);
        if let Some(ref mut reload_fn) = self.on_config_reload {
            reload_fn(&self.tree, callback_storage);
        }
        // Update scenario tracking if the callback provided new indices.
        if let Some(new_indices) = self.reloaded_node_indices.lock().unwrap().take() {
            self.scenario_node_indices = new_indices.clone();
            self.scenario_ev_tracker.set_nodes(new_indices);
        }
    }

    /// Run the strategy refresh callback path without evaluating timed snapshot cadence.
    pub fn refresh_strategy_telemetry_now(&mut self) {
        self.update_strategy_delta();
        self.push_strategy_refresh(self.start_time.elapsed().as_secs());
    }

    fn push_strategy_refresh(&mut self, elapsed_secs: u64) {
        let decision_map = self.tree.decision_index_map();
        let projected_storage = self
            .sparse_storage
            .is_some()
            .then(|| self.dense_storage_projection());
        let callback_storage = projected_storage.as_ref().unwrap_or(&self.storage);
        if let Some(ref callback) = self.on_strategy_refresh {
            for (i, &node_idx) in self.scenario_node_indices.iter().enumerate() {
                let player = match &self.tree.nodes[node_idx as usize] {
                    super::game_tree::GameNode::Decision { player, .. } => *player as usize,
                    _ => 0,
                };
                // Read EVs from the full-tree tracker.
                // EVs are raw counterfactual values measured from hand start.
                // Blinds are included: fold at root = -0.5 BB for SB.
                let dec_idx = decision_map[node_idx as usize];
                let hand_evs = if dec_idx != u32::MAX {
                    self.full_ev_tracker.hand_ev_array(dec_idx as usize, player)
                } else {
                    [0.0; 169]
                };
                callback(i, node_idx, callback_storage, &self.tree, &hand_evs);
            }
        }
        if let Some(ref mut cb) = self.on_audit_refresh {
            cb(callback_storage);
        }
        // Reset the scenario (windowed) EV tracker so TUI displays
        // only the most recent window. The full_ev_tracker is cumulative
        // and must NOT be reset — it persists to hand_ev.bin at snapshot time.
        self.scenario_ev_tracker.reset();
        self.last_strategy_refresh_secs = elapsed_secs;
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
        let t = self
            .config
            .training
            .dcfr_epoch_cap
            .map_or(t, |cap| t.min(cap));

        if let Some(ref sparse) = self.sparse_storage {
            sparse.apply_optimizer_discount(t);
        } else if let Some(ref opt) = self.storage.optimizer {
            opt.apply_discount(
                &self.storage.regrets,
                &self.storage.strategy_sums,
                self.storage.predictions.as_deref(),
                t,
            );
        } else {
            // Fallback: inline DCFR (should not happen when optimizer is wired).
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
                atom.store((v as f64 * d) as i32, Ordering::Relaxed);
            });
            self.storage.strategy_sums.par_iter().for_each(|atom| {
                let v = atom.load(Ordering::Relaxed);
                atom.store((v as f64 * d_strat) as i64, Ordering::Relaxed);
            });
        }

        self.last_discount_time = self.iterations;
    }

    /// Compute and store the strategy delta vs the previous snapshot.
    fn update_strategy_delta(&mut self) {
        if let Some(ref prev) = self.prev_strategy_sums {
            let (delta, pct_moving) =
                self.with_dense_storage(|storage| storage.strategy_delta(prev));
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
        self.prev_strategy_sums = Some(self.active_strategy_sums_snapshot());
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

        if self.sparse_storage.is_some() {
            let stats = self.storage_stats();
            eprintln!(
                "  storage=sparse rows={} slots={} inserts={} reads={}/{} writes={}/{} dense_slots={} dense_bytes={} sparse_bytes={}",
                stats.realized_rows,
                stats.realized_slots,
                stats.inserts,
                stats.read_hits,
                stats.read_probes,
                stats.write_hits,
                stats.write_probes,
                stats.dense_equivalent_slots,
                stats.dense_equivalent_bytes,
                stats.sparse_resident_bytes,
            );
        }
    }

    /// The most-negative regret value across all info-set entries
    /// (in chip units, after dividing by `REGRET_SCALE`).
    #[must_use]
    pub fn min_regret(&self) -> f64 {
        let min_raw = self.with_dense_storage(|storage| {
            storage
                .regrets
                .iter()
                .map(|atom| atom.load(Ordering::Relaxed))
                .min()
                .unwrap_or(0)
        });
        min_raw as f64 / super::storage::REGRET_SCALE
    }

    /// The most-positive regret value across all info-set entries
    /// (in chip units, after dividing by `REGRET_SCALE`).
    #[must_use]
    pub fn max_regret(&self) -> f64 {
        let max_raw = self.with_dense_storage(|storage| {
            storage
                .regrets
                .iter()
                .map(|atom| atom.load(Ordering::Relaxed))
                .max()
                .unwrap_or(0)
        });
        max_raw as f64 / super::storage::REGRET_SCALE
    }

    /// Average positive regret per iteration: mean of positive regret
    /// entries divided by iteration count. This is the actual convergence
    /// signal — should decrease as O(1/√T).
    #[must_use]
    pub fn avg_pos_regret(&self) -> f64 {
        if self.iterations == 0 {
            return 0.0;
        }
        let (sum, count) = self.with_dense_storage(|storage| {
            storage
                .regrets
                .iter()
                .fold((0.0_f64, 0_u64), |(s, c), atom| {
                    let r = atom.load(Ordering::Relaxed);
                    if r > 0 { (s + r as f64, c + 1) } else { (s, c) }
                })
        });
        if count > 0 {
            sum / count as f64 / self.iterations as f64 / super::storage::REGRET_SCALE
        } else {
            0.0
        }
    }

    /// Fraction of regret entries below the prune threshold (0.0–1.0).
    #[must_use]
    pub fn prune_fraction(&self) -> f64 {
        // Config prune_threshold is in chip units; scale to match stored regrets.
        let threshold =
            (f64::from(self.config.training.prune_threshold) * super::storage::REGRET_SCALE) as i32;
        let (below, total) = self.with_dense_storage(|storage| {
            let total = storage.regrets.len();
            let below = storage
                .regrets
                .iter()
                .filter(|atom| atom.load(Ordering::Relaxed) < threshold)
                .count();
            (below, total)
        });
        let total = total as f64;
        if total == 0.0 {
            return 0.0;
        }
        below as f64 / total
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

    /// Mean of all strictly-positive regret entries (in chip units).
    #[must_use]
    pub fn mean_positive_regret(&self) -> f64 {
        let (sum, count) = self.with_dense_storage(|storage| {
            storage
                .regrets
                .iter()
                .fold((0.0_f64, 0_u64), |(s, c), atom| {
                    let r = atom.load(Ordering::Relaxed);
                    if r > 0 { (s + r as f64, c + 1) } else { (s, c) }
                })
        });
        if count > 0 {
            sum / count as f64 / super::storage::REGRET_SCALE
        } else {
            0.0
        }
    }

    /// Compute the fold terminal value at a decision node for a given player.
    ///
    /// Returns the dead-money fold payoff (e.g. -2.0 if the player has
    /// voluntarily invested 2.0). At the root preflop node this is 0.0.
    /// Returns 0.0 if the node has no Fold action.
    #[allow(dead_code)]
    fn fold_value_at_node(tree: &GameTree, node_idx: u32, player: usize) -> f64 {
        use super::game_tree::{GameNode, TerminalKind, TreeAction};
        if let GameNode::Decision {
            actions, children, ..
        } = &tree.nodes[node_idx as usize]
        {
            for (a, &child_idx) in children.iter().enumerate() {
                if actions[a] == TreeAction::Fold {
                    if let GameNode::Terminal {
                        kind: TerminalKind::Fold { .. },
                        stacks,
                        ..
                    } = &tree.nodes[child_idx as usize]
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
                let sum =
                    self.scenario_ev_tracker.ev_sum[0][player][i].load(AO::Relaxed) as f64 / 1000.0;
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
                let sum0 =
                    self.scenario_ev_tracker.ev_sum[0][0][i].load(AO::Relaxed) as f64 / 1000.0;
                let count0 = self.scenario_ev_tracker.ev_count[0][0][i].load(AO::Relaxed);
                let sum1 =
                    self.scenario_ev_tracker.ev_sum[0][1][i].load(AO::Relaxed) as f64 / 1000.0;
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
        let format = self.config.snapshots.format;

        let projected_storage;
        let storage = if self.sparse_storage.is_some() {
            projected_storage = Some(self.dense_storage_projection());
            projected_storage.as_ref().expect("projection exists")
        } else {
            &self.storage
        };

        let mut strategy = BlueprintV2Strategy::from_storage_with_threshold(
            storage,
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

        let write_legacy = matches!(format, SnapshotFormat::Legacy | SnapshotFormat::Both);
        let write_universal = matches!(format, SnapshotFormat::Universal | SnapshotFormat::Both);

        if write_legacy {
            bundle::save_snapshot(&snapshot_dir, &strategy, storage, &metadata)?;
        } else {
            // Universal-only still needs snapshot_dir and metadata
            std::fs::create_dir_all(&snapshot_dir)?;
            std::fs::write(snapshot_dir.join("metadata.json"), &metadata)?;
        }

        if write_universal {
            self.write_universal_bundle(&snapshot_dir, &strategy)?;
        }

        // Save full-tree EV tracker.
        self.full_ev_tracker
            .save(&snapshot_dir.join("hand_ev.bin"))?;

        // Compute and save counterfactual boundary values (CBVs) for
        // real-time subgame solving. One table per player, indexed by
        // (chance_node, bucket).
        let bucket_counts = storage.bucket_counts;
        let transitions = crate::blueprint_v2::cbv_compute::build_transitions_from_buckets(
            &self.buckets.bucket_files,
        );
        let [p0_cbvs, p1_cbvs] = crate::blueprint_v2::cbv_compute::compute_cbvs_with_transitions(
            &strategy,
            &self.tree,
            bucket_counts,
            &transitions,
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
                let min_nonzero = counts
                    .iter()
                    .filter(|&&c| c > 0)
                    .min()
                    .copied()
                    .unwrap_or(0);
                if !self.tui_active {
                    eprintln!(
                        "[bucket visits] {name}: {nonzero}/{} buckets visited, total={total}, min={min_nonzero}, max={max}",
                        counts.len()
                    );
                }
                let _ = write!(visit_json, "  \"{name}\": {:?}", counts);
                if s < 3 {
                    visit_json.push(',');
                }
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

    /// Write a universal dense bundle into `snapshot_dir/universal/`.
    fn write_universal_bundle(
        &self,
        snapshot_dir: &Path,
        strategy: &BlueprintV2Strategy,
    ) -> Result<(), Box<dyn Error>> {
        let output_dir = Path::new(&self.config.snapshots.output_dir);
        let config_path = output_dir.join("config.yaml");
        let universal_dir = snapshot_dir.join("universal");

        crate::blueprint_universal::hu_export::write_hu_universal_snapshot(
            &self.config,
            &self.tree,
            strategy,
            self.iterations,
            self.elapsed_minutes() as f64,
            &config_path,
            &universal_dir,
        )?;

        if !self.tui_active {
            eprintln!(
                "  Universal bundle written to {}",
                universal_dir.display(),
            );
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

#[derive(Debug, Eq, PartialEq)]
struct ResumeCandidate {
    path: PathBuf,
    snapshot_num: Option<u32>,
    iteration: u64,
    elapsed_minutes: u64,
    is_final: bool,
}

impl ResumeCandidate {
    fn from_dir(path: PathBuf, snapshot_num: Option<u32>, is_final: bool) -> Option<Self> {
        if !path.join("regrets.bin").exists() {
            return None;
        }

        let metadata = std::fs::read_to_string(path.join("metadata.json")).ok()?;
        let iteration = extract_json_u64(&metadata, "iteration")?;
        let elapsed_minutes = extract_json_u64(&metadata, "elapsed_minutes")?;

        Some(Self {
            path,
            snapshot_num,
            iteration,
            elapsed_minutes,
            is_final,
        })
    }
}

impl Ord for ResumeCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.iteration
            .cmp(&other.iteration)
            .then_with(|| self.elapsed_minutes.cmp(&other.elapsed_minutes))
            .then_with(|| self.is_final.cmp(&other.is_final))
            .then_with(|| self.snapshot_num.cmp(&other.snapshot_num))
            .then_with(|| self.path.cmp(&other.path))
    }
}

impl PartialOrd for ResumeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    use test_macros::timed_test;

    use super::*;
    use crate::blueprint_v2::config::*;

    fn toy_config() -> BlueprintV2Config {
        BlueprintV2Config {
            game: GameConfig {
                name: "Test".to_string(),
                players: 2,
                stack_depth: 20.0,
                small_blind: 1.0,
                big_blind: 2.0,
                rake_rate: 0.0,
                rake_cap: 0.0,
                allow_preflop_limp: true,
            },
            clustering: ClusteringConfig {
                algorithm: ClusteringAlgorithm::PotentialAwareEmd,
                preflop: StreetClusterConfig {
                    buckets: 10,
                    delta_bins: None,
                    expected_delta: false,
                    sample_boards: None,
                    metric: Default::default(),
                },
                flop: StreetClusterConfig {
                    buckets: 10,
                    delta_bins: None,
                    expected_delta: false,
                    sample_boards: None,
                    metric: Default::default(),
                },
                turn: StreetClusterConfig {
                    buckets: 10,
                    delta_bins: None,
                    expected_delta: false,
                    sample_boards: None,
                    metric: Default::default(),
                },
                river: StreetClusterConfig {
                    buckets: 10,
                    delta_bins: None,
                    expected_delta: false,
                    sample_boards: None,
                    metric: Default::default(),
                },
                seed: 42,
                kmeans_iterations: 50,
                cfvnet_river_data: None,
                per_flop: None,
            },
            action_abstraction: ActionAbstractionConfig {
                preflop: vec![vec!["5bb".into()]],
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
                storage_backend: "dense".to_string(),
                sapcfr_eta: 0.5,
                brcfr_eta: 0.6,
                brcfr_warmup_iterations: 0,
                brcfr_interval: 100_000_000,
                use_baselines: false,
                baseline_alpha: 0.01,
                baseline_validation: BaselineValidationTrainingConfig::default(),
                prune_streets: None,
                regret_floor: None,
                exploitability_interval_minutes: 0,
                exploitability_samples: 100_000,
            },
            snapshots: SnapshotConfig {
                warmup_minutes: 9999,
                snapshot_every_minutes: 9999,
                output_dir: "/tmp/test_blueprint_v2_snapshots".into(),
                resume: false,
                max_snapshots: None,
                format: SnapshotFormat::Legacy,
            },
        }
    }

    fn toy_trainer(config: BlueprintV2Config) -> BlueprintTrainer {
        let mut t = BlueprintTrainer::new(config);
        t.skip_bucket_validation = true;
        t
    }

    fn baseline_validation_config() -> BlueprintV2Config {
        let mut config = toy_config();
        config.game.stack_depth = 40.0;
        config.game.allow_preflop_limp = false;
        config.clustering.preflop.buckets = 169;
        config.action_abstraction.preflop =
            vec![vec!["2.5bb".to_string()], vec!["5bb".to_string()]];
        config.training.iterations = Some(0);
        config.training.batch_size = 1;
        config.training.baseline_validation = BaselineValidationTrainingConfig {
            enabled: true,
            baseline_path: Some(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../local_data/baselines/cash_hu_20bb_cev.json"),
            ),
            interval_iterations: 1,
            interval_minutes: 0,
            top_n_spots: 5,
            top_n_combos_per_spot: 5,
        };
        config
    }

    fn tiny_training_config(storage_backend: &str) -> BlueprintV2Config {
        let mut config = toy_config();
        config.training.storage_backend = storage_backend.to_string();
        config.training.iterations = Some(24);
        config.training.batch_size = 1;
        config.training.lcfr_warmup_iterations = 999_999;
        config.training.prune_after_iterations = 999_999;
        config.training.print_every_minutes = 999_999;
        config.training.exploitability_interval_minutes = 0;
        config.snapshots.warmup_minutes = 999_999;
        config
    }

    fn dense_values(storage: &BlueprintStorage) -> (Vec<i32>, Vec<i64>) {
        (
            storage
                .regrets
                .iter()
                .map(|slot| slot.load(Ordering::Relaxed))
                .collect(),
            storage
                .strategy_sums
                .iter()
                .map(|slot| slot.load(Ordering::Relaxed))
                .collect(),
        )
    }

    fn clone_snapshot_regrets(source_snapshot: &Path, dest_snapshot: &Path) {
        std::fs::create_dir_all(dest_snapshot).expect("create snapshot dir");
        std::fs::copy(
            source_snapshot.join("regrets.bin"),
            dest_snapshot.join("regrets.bin"),
        )
        .expect("copy regrets fixture");
    }

    fn write_snapshot_metadata(snapshot_dir: &Path, iteration: u64, elapsed_minutes: u64) {
        std::fs::write(
            snapshot_dir.join("metadata.json"),
            format!(
                "{{\"iteration\": {iteration}, \"elapsed_minutes\": {elapsed_minutes}, \"mean_positive_regret\": 0.00}}"
            ),
        )
        .expect("write metadata fixture");
    }

    #[test]
    fn resume_candidate_final_ordering_requires_equal_or_newer_metadata() {
        let numbered = ResumeCandidate {
            path: PathBuf::from("snapshot_0011"),
            snapshot_num: Some(11),
            iteration: 200,
            elapsed_minutes: 20,
            is_final: false,
        };
        let stale_final = ResumeCandidate {
            path: PathBuf::from("final"),
            snapshot_num: None,
            iteration: 199,
            elapsed_minutes: 999,
            is_final: true,
        };
        let tied_final = ResumeCandidate {
            path: PathBuf::from("final"),
            snapshot_num: None,
            iteration: 200,
            elapsed_minutes: 20,
            is_final: true,
        };
        let newer_final = ResumeCandidate {
            path: PathBuf::from("final"),
            snapshot_num: None,
            iteration: 200,
            elapsed_minutes: 21,
            is_final: true,
        };

        assert!(numbered > stale_final);
        assert!(tied_final > numbered);
        assert!(newer_final > numbered);
    }

    #[test]
    fn trainer_creation() {
        let config = toy_config();
        let trainer = BlueprintTrainer::new(config);
        assert_eq!(trainer.iterations, 0);
        assert!(!trainer.storage.regrets.is_empty());
        assert!(!trainer.is_sparse_storage());
    }

    #[test]
    fn trainer_creates_sparse_backend_from_config() {
        let config = tiny_training_config("sparse");
        let trainer = BlueprintTrainer::new(config);
        assert!(trainer.is_sparse_storage());
        assert!(
            trainer.storage.regrets.is_empty(),
            "public storage should be a dense compatibility placeholder"
        );
        let stats = trainer.storage_stats();
        assert_eq!(stats.realized_rows, 0);
        assert_eq!(stats.realized_slots, 0);
        assert!(stats.dense_equivalent_slots > 0);
        assert!(stats.dense_equivalent_bytes > 0);
    }

    #[test]
    #[should_panic(expected = "sparse storage does not support brcfr+")]
    fn sparse_storage_rejects_brcfr_plus() {
        let mut config = tiny_training_config("sparse");
        config.training.optimizer = "brcfr+".to_string();
        let _ = BlueprintTrainer::new(config);
    }

    #[test]
    fn baseline_validation_initial_report_uses_actual_config_preconditions() {
        let config = baseline_validation_config();
        let mut trainer = toy_trainer(config);

        trainer.train().expect("baseline validation should pass");
        let report = trainer
            .baseline_validation_report()
            .expect("baseline document should remain loaded");

        assert_eq!(report.aggregate.precondition_failures, 0);
        assert_eq!(report.aggregate.spots_total, 6);
        assert_eq!(report.aggregate.spots_scored, 6);
        assert!(report.aggregate.combo_rows_scored > 0);
    }

    #[test]
    fn baseline_validation_rejects_actual_wrong_game_config() {
        let mut config = baseline_validation_config();
        config.game.small_blind = 3.0;
        let mut trainer = toy_trainer(config);

        let err = trainer
            .train()
            .expect_err("wrong trusted game config should be rejected");
        let text = err.to_string();
        assert!(text.contains("trusted_game.small_blind"), "{text}");
        assert!(text.contains("actual=3.0"), "{text}");
    }

    #[test]
    fn baseline_validation_cadence_is_batch_bound() {
        let mut config = baseline_validation_config();
        config.training.iterations = Some(3);
        config.training.batch_size = 1;
        config.training.baseline_validation.interval_iterations = 2;

        let reports = Arc::new(AtomicU64::new(0));
        let reports_for_callback = Arc::clone(&reports);
        let mut trainer = toy_trainer(config);
        trainer.on_baseline_validation = Some(Box::new(move |_report| {
            reports_for_callback.fetch_add(1, Ordering::Relaxed);
        }));

        trainer.train().expect("short training should complete");

        assert_eq!(
            reports.load(Ordering::Relaxed),
            2,
            "expected initial report plus one cadence report at iteration 2"
        );
    }

    #[test]
    fn baseline_validation_log_format_includes_required_diagnostics() {
        let config = baseline_validation_config();
        let mut trainer = toy_trainer(config);
        trainer.train().expect("baseline validation should pass");
        let report = trainer
            .baseline_validation_report()
            .expect("baseline document should remain loaded");
        let text = format_baseline_validation_lines(&report, 5, 5).join("\n");

        for needle in [
            "aggregate_tv=",
            "root_tv=",
            "first_response_tv=",
            "worst_spot_tv=",
            "coverage=",
            "skipped_zero_mass=",
            "invalid_rows=",
            "unsupported_spots=",
            "unsupported_actions=",
            "worst spots:",
            "worst combo rows:",
        ] {
            assert!(text.contains(needle), "missing {needle} in {text}");
        }
    }

    #[test]
    fn config_reload_trigger_default_false() {
        let config = toy_config();
        let trainer = BlueprintTrainer::new(config);
        assert!(!trainer.config_reload_trigger.load(Ordering::Relaxed));
    }

    #[test]
    fn on_config_reload_default_none() {
        let config = toy_config();
        let trainer = BlueprintTrainer::new(config);
        assert!(trainer.on_config_reload.is_none());
    }

    #[test]
    fn config_reload_trigger_fires_callback() {
        use std::sync::atomic::AtomicBool;
        let config = toy_config();
        let mut trainer = toy_trainer(config);
        let called = Arc::new(AtomicBool::new(false));
        let called_clone = Arc::clone(&called);
        trainer.on_config_reload = Some(Box::new(move |_tree, _storage| {
            called_clone.store(true, Ordering::Relaxed);
        }));
        // Set trigger
        trainer.config_reload_trigger.store(true, Ordering::Relaxed);
        // check_timed_actions should fire the callback and clear the trigger
        let _ = trainer.check_timed_actions();
        assert!(called.load(Ordering::Relaxed));
        assert!(!trainer.config_reload_trigger.load(Ordering::Relaxed));
    }

    #[test]
    fn config_reload_updates_scenario_node_indices() {
        let config = toy_config();
        let mut trainer = toy_trainer(config);
        let new_indices = vec![42u32, 7u32];
        let shared = Arc::clone(&trainer.reloaded_node_indices);
        trainer.on_config_reload = Some(Box::new(move |_tree, _storage| {
            *shared.lock().unwrap() = Some(vec![42, 7]);
        }));
        trainer.config_reload_trigger.store(true, Ordering::Relaxed);
        let _ = trainer.check_timed_actions();
        assert_eq!(trainer.scenario_node_indices, new_indices);
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

        assert!(
            trainer
                .storage
                .regrets
                .iter()
                .all(|r| r.load(Ordering::Relaxed) == 0)
        );

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
    fn train_with_sparse_storage_reports_stats() {
        let config = tiny_training_config("sparse");
        let mut trainer = toy_trainer(config);
        trainer.train().expect("sparse training should complete");

        let stats = trainer.storage_stats();
        assert!(stats.realized_rows > 0, "training should realize rows");
        assert!(stats.realized_slots > 0, "training should realize slots");
        assert!(stats.inserts > 0, "training should insert sparse rows");
        assert!(stats.read_probes > 0, "training should probe sparse reads");
        assert!(
            stats.write_probes > 0,
            "training should probe sparse writes"
        );
        assert!(stats.dense_equivalent_slots >= stats.realized_slots);
        assert!(stats.sparse_resident_bytes > 0);
    }

    #[test]
    fn dense_and_sparse_training_project_same_dense_values() {
        let dense_config = tiny_training_config("dense");
        let sparse_config = tiny_training_config("sparse");

        let mut dense_trainer = toy_trainer(dense_config);
        let mut sparse_trainer = toy_trainer(sparse_config);

        dense_trainer
            .train()
            .expect("dense training should complete");
        sparse_trainer
            .train()
            .expect("sparse training should complete");

        let dense_projection = dense_trainer.dense_storage_projection();
        let sparse_projection = sparse_trainer.dense_storage_projection();
        assert_eq!(
            dense_values(&dense_projection),
            dense_values(&sparse_projection)
        );

        let dense_strategy =
            BlueprintV2Strategy::from_storage(&dense_projection, &dense_trainer.tree);
        let sparse_strategy =
            BlueprintV2Strategy::from_storage(&sparse_projection, &sparse_trainer.tree);
        assert_eq!(dense_strategy.action_probs, sparse_strategy.action_probs);
        assert_eq!(
            dense_strategy.node_action_counts,
            sparse_strategy.node_action_counts
        );
    }

    #[test]
    fn sparse_snapshot_writes_dense_resume_compatible_files() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let mut config = tiny_training_config("sparse");
        config.snapshots.output_dir = dir.path().to_string_lossy().to_string();
        let mut trainer = toy_trainer(config);

        trainer.train().expect("sparse training should complete");
        trainer.save_snapshot().expect("snapshot should save");

        let snapshot_dir = dir.path().join("snapshot_0000");
        assert!(snapshot_dir.join("strategy.bin").exists());
        assert!(snapshot_dir.join("regrets.bin").exists());

        let loaded = BlueprintStorage::load_regrets(
            &snapshot_dir.join("regrets.bin"),
            &trainer.tree,
            trainer.storage.bucket_counts,
        )
        .expect("dense regrets should load");
        let projected = trainer.dense_storage_projection();
        assert_eq!(dense_values(&loaded), dense_values(&projected));

        let strategy = BlueprintV2Strategy::load(&snapshot_dir.join("strategy.bin"))
            .expect("strategy should load");
        assert_eq!(strategy.bucket_counts, trainer.storage.bucket_counts);
        assert!(!strategy.action_probs.is_empty());
    }

    #[test]
    fn sparse_training_honors_prediction_baseline_and_regret_floor_config() {
        let mut config = tiny_training_config("sparse");
        config.training.optimizer = "sapcfr+".to_string();
        config.training.use_baselines = true;
        config.training.regret_floor = Some(0);
        let mut trainer = toy_trainer(config);

        trainer
            .train()
            .expect("sparse sapcfr+ baseline training should complete");

        let projected = trainer.dense_storage_projection();
        assert!(
            projected
                .regrets
                .iter()
                .all(|slot| slot.load(Ordering::Relaxed) >= 0),
            "sparse regret floor should clamp projected regrets"
        );
        let stats = trainer.storage_stats();
        assert!(stats.realized_rows > 0);
        assert!(stats.sparse_resident_bytes > stats.realized_slots * 12);
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
        assert_eq!(trainer.storage.strategy_sums[0].load(Ordering::Relaxed), 0);
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
        assert!(
            snapshot_dir.join("hand_ev.bin").exists(),
            "full-tree EV tracker should be saved"
        );

        // Verify hand_ev.json is valid JSON with 169 entries.
        let ev_json =
            std::fs::read_to_string(snapshot_dir.join("hand_ev.json")).expect("read hand_ev.json");
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
            .filter(|e| e.file_name().to_string_lossy().starts_with("snapshot_"))
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
        assert!(
            trainer
                .storage
                .regrets
                .iter()
                .any(|r| r.load(Ordering::Relaxed) != 0)
        );
        assert!(
            trainer
                .storage
                .strategy_sums
                .iter()
                .any(|s| s.load(Ordering::Relaxed) != 0)
        );
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
        assert!(
            snapshot_dir.join("hand_ev.bin").exists(),
            "hand_ev.bin should exist in snapshot"
        );

        trainer2.train().expect("resumed training");
        assert_eq!(trainer2.iterations, 40, "should reach 40 total");
    }

    #[test]
    fn resume_prefers_newer_numbered_snapshot_over_stale_final() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let source_snapshot = dir.path().join("snapshot_0000");

        let mut config = toy_config();
        config.snapshots.output_dir = dir.path().to_string_lossy().to_string();
        let mut trainer = toy_trainer(config.clone());
        trainer.save_snapshot().expect("save fixture snapshot");

        let newer_numbered = dir.path().join("snapshot_0011");
        clone_snapshot_regrets(&source_snapshot, &newer_numbered);
        write_snapshot_metadata(&newer_numbered, 2_961_501_400, 140);

        let stale_final = dir.path().join("final");
        clone_snapshot_regrets(&source_snapshot, &stale_final);
        write_snapshot_metadata(&stale_final, 2_000_000_000, 999);

        config.snapshots.resume = true;
        let mut resumed = toy_trainer(config);
        resumed.try_resume().expect("resume should succeed");

        assert_eq!(resumed.iterations, 2_961_501_400);
        assert_eq!(
            resumed.snapshot_count, 12,
            "numbered snapshot_0011 should drive the next snapshot index"
        );
    }

    #[test]
    fn resume_prefers_final_on_equal_metadata_and_keeps_next_numbered_index() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let source_snapshot = dir.path().join("snapshot_0000");

        let mut config = toy_config();
        config.snapshots.output_dir = dir.path().to_string_lossy().to_string();
        let mut trainer = toy_trainer(config.clone());
        trainer.save_snapshot().expect("save fixture snapshot");

        let tied_numbered = dir.path().join("snapshot_0001");
        clone_snapshot_regrets(&source_snapshot, &tied_numbered);
        write_snapshot_metadata(&tied_numbered, 700, 70);

        let lower_metadata_high_index = dir.path().join("snapshot_0006");
        clone_snapshot_regrets(&source_snapshot, &lower_metadata_high_index);
        write_snapshot_metadata(&lower_metadata_high_index, 100, 10);

        let tied_final = dir.path().join("final");
        clone_snapshot_regrets(&source_snapshot, &tied_final);
        write_snapshot_metadata(&tied_final, 700, 70);

        config.snapshots.resume = true;
        let mut resumed = toy_trainer(config);
        resumed.try_resume().expect("resume should succeed");

        assert_eq!(resumed.iterations, 700);
        assert_eq!(resumed.last_snapshot_time, 70);
        assert_eq!(
            resumed.snapshot_count, 7,
            "resuming from final should continue after the highest valid numbered snapshot"
        );
    }

    #[test]
    fn resume_sets_snapshot_count_after_selected_numbered_snapshot() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let source_snapshot = dir.path().join("snapshot_0000");

        let mut config = toy_config();
        config.snapshots.output_dir = dir.path().to_string_lossy().to_string();
        let mut trainer = toy_trainer(config.clone());
        trainer.save_snapshot().expect("save fixture snapshot");

        let selected = dir.path().join("snapshot_0006");
        clone_snapshot_regrets(&source_snapshot, &selected);
        write_snapshot_metadata(&selected, 600, 60);

        config.snapshots.resume = true;
        let mut resumed = toy_trainer(config);
        resumed.try_resume().expect("resume should succeed");

        assert_eq!(resumed.iterations, 600);
        assert_eq!(resumed.snapshot_count, 7);
    }

    #[test]
    fn resume_restores_last_snapshot_time_from_metadata() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let source_snapshot = dir.path().join("snapshot_0000");

        let mut config = toy_config();
        config.snapshots.output_dir = dir.path().to_string_lossy().to_string();
        let mut trainer = toy_trainer(config.clone());
        trainer.save_snapshot().expect("save fixture snapshot");
        write_snapshot_metadata(&source_snapshot, 321, 77);

        config.snapshots.resume = true;
        let mut resumed = toy_trainer(config);
        resumed.try_resume().expect("resume should succeed");

        assert_eq!(resumed.iterations, 321);
        assert_eq!(resumed.last_snapshot_time, 77);
    }

    #[test]
    fn resume_skips_snapshots_missing_metadata() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let source_snapshot = dir.path().join("snapshot_0000");

        let mut config = toy_config();
        config.training.iterations = Some(20);
        config.snapshots.output_dir = dir.path().to_string_lossy().to_string();
        let mut trainer = toy_trainer(config.clone());
        trainer.train().expect("initial training");
        trainer.save_snapshot().expect("save fixture snapshot");

        let metadata_missing = dir.path().join("snapshot_0009");
        clone_snapshot_regrets(&source_snapshot, &metadata_missing);
        std::fs::remove_dir_all(&source_snapshot).expect("remove valid source snapshot");

        config.snapshots.resume = true;
        let mut resumed = toy_trainer(config);
        let before_regrets = resumed
            .storage
            .regrets
            .iter()
            .map(|slot| slot.load(Ordering::Relaxed))
            .collect::<Vec<_>>();

        resumed
            .try_resume()
            .expect("metadata-missing resume is skipped");

        let after_regrets = resumed
            .storage
            .regrets
            .iter()
            .map(|slot| slot.load(Ordering::Relaxed))
            .collect::<Vec<_>>();
        assert_eq!(resumed.iterations, 0);
        assert_eq!(resumed.snapshot_count, 0);
        assert_eq!(after_regrets, before_regrets);
    }

    #[test]
    fn hand_ev_array_per_position_initially_zero() {
        let config = toy_config();
        let trainer = BlueprintTrainer::new(config);
        let evs_p0 = trainer.hand_ev_array(0);
        let evs_p1 = trainer.hand_ev_array(1);
        assert!(
            evs_p0.iter().all(|&v| v == 0.0),
            "position 0 EVs should start at zero"
        );
        assert!(
            evs_p1.iter().all(|&v| v == 0.0),
            "position 1 EVs should start at zero"
        );
    }

    #[test]
    fn hand_ev_array_per_position_independent() {
        let config = toy_config();
        let mut trainer = BlueprintTrainer::new(config);

        // Set up tracker with a single node (root).
        trainer
            .scenario_ev_tracker
            .set_nodes(vec![trainer.tree.root]);

        // Manually set EV data for position 0, hand index 5.
        trainer.scenario_ev_tracker.accumulate(0, 0, 5, 3.0);
        trainer.scenario_ev_tracker.accumulate(0, 0, 5, 0.0); // avg = 1.5

        // Manually set EV data for position 1, hand index 5.
        trainer.scenario_ev_tracker.accumulate(0, 1, 5, -1.0);

        let evs_p0 = trainer.hand_ev_array(0);
        let evs_p1 = trainer.hand_ev_array(1);

        assert!(
            (evs_p0[5] - 1.5).abs() < 1e-2,
            "position 0 hand 5 EV should be 1.5, got {}",
            evs_p0[5]
        );
        assert!(
            (evs_p1[5] - (-1.0)).abs() < 1e-2,
            "position 1 hand 5 EV should be -1.0, got {}",
            evs_p1[5]
        );
        // Other hands should still be zero.
        assert!(evs_p0[0] == 0.0);
        assert!(evs_p1[0] == 0.0);
    }

    #[test]
    fn hand_ev_averages_per_position() {
        let config = toy_config();
        let mut trainer = BlueprintTrainer::new(config);

        // Set up tracker with a single node (root).
        trainer
            .scenario_ev_tracker
            .set_nodes(vec![trainer.tree.root]);

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
        assert!(
            evs_p0.iter().any(|&v| v != 0.0),
            "position 0 should have some non-zero EVs at root"
        );

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
        assert!(
            found_p1,
            "position 1 should have some non-zero EVs at its decision nodes"
        );
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
        pf.save(&dir.path().join("flop_0000.buckets"))
            .expect("save per-flop file");

        let mut config = toy_config();
        config.training.cluster_path = Some(dir.path().to_string_lossy().into_owned());

        let trainer = toy_trainer(config);
        assert!(
            trainer.buckets.has_per_flop_dir(),
            "trainer should auto-detect per-flop bucket files"
        );
    }

    #[test]
    fn trainer_no_per_flop_without_marker_file() {
        let dir = tempfile::tempdir().expect("tempdir");

        // Empty directory — no flop_0000.buckets
        let mut config = toy_config();
        config.training.cluster_path = Some(dir.path().to_string_lossy().into_owned());

        let trainer = toy_trainer(config);
        assert!(
            !trainer.buckets.has_per_flop_dir(),
            "trainer should not enable per-flop without marker file"
        );
    }

    #[test]
    fn train_with_baselines_runs_iterations() {
        let mut config = toy_config();
        config.training.iterations = Some(50);
        config.training.use_baselines = true;
        config.training.baseline_alpha = 0.05;
        let mut trainer = toy_trainer(config);
        trainer
            .train()
            .expect("training with baselines should complete");
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

    #[test]
    fn trainer_creates_dcfr_optimizer_by_default() {
        let config = toy_config();
        let trainer = BlueprintTrainer::new(config);
        let opt = trainer
            .storage
            .optimizer
            .as_ref()
            .expect("optimizer should be set");
        assert_eq!(opt.name(), "dcfr");
        assert!(!opt.needs_predictions());
        assert!(trainer.storage.predictions.is_none());
    }

    #[test]
    fn trainer_creates_sapcfr_optimizer() {
        let mut config = toy_config();
        config.training.optimizer = "sapcfr+".to_string();
        config.training.sapcfr_eta = 0.75;
        let trainer = BlueprintTrainer::new(config);
        let opt = trainer
            .storage
            .optimizer
            .as_ref()
            .expect("optimizer should be set");
        assert_eq!(opt.name(), "sapcfr+");
        assert!(opt.needs_predictions());
        assert!(trainer.storage.predictions.is_some());
    }

    #[test]
    fn train_with_sapcfr_optimizer() {
        let mut config = toy_config();
        config.training.optimizer = "sapcfr+".to_string();
        config.training.sapcfr_eta = 0.5;
        config.training.iterations = Some(50);
        let mut trainer = toy_trainer(config);
        trainer
            .train()
            .expect("training with sapcfr+ should complete");
        assert_eq!(trainer.iterations, 50);
        assert!(
            trainer
                .storage
                .regrets
                .iter()
                .any(|r| r.load(Ordering::Relaxed) != 0),
            "regrets should be updated after sapcfr+ training"
        );
    }

    #[test]
    fn sapcfr_training_populates_predictions() {
        let mut config = toy_config();
        config.training.optimizer = "sapcfr+".to_string();
        config.training.sapcfr_eta = 0.5;
        config.training.iterations = Some(50);
        let mut trainer = toy_trainer(config);
        trainer.train().expect("training should complete");
        let preds = trainer
            .storage
            .predictions
            .as_ref()
            .expect("predictions should be Some");
        assert!(
            preds.iter().any(|p| p.load(Ordering::Relaxed) != 0),
            "predictions should be populated after training"
        );
    }

    #[timed_test]
    fn prune_threshold_scaling() {
        // prune_threshold is in chip units; prune_fraction() should scale it
        // by REGRET_SCALE (1,000) before comparing against stored regrets.
        let mut config = toy_config();
        config.training.prune_threshold = -5; // -5 chips
        let trainer = toy_trainer(config);

        let scale = crate::blueprint_v2::storage::REGRET_SCALE as i32; // 1_000
        let n = trainer.storage.regrets.len();
        assert!(n >= 6, "need at least 6 regret slots for this test");

        // Scaled threshold = -5 * 1_000 = -5_000.
        // Values *below* -5_000 are pruned; values >= -5_000 are not.
        //
        // Set 6 regrets at known scaled values:
        //   [0] = -6_000  (below threshold, pruned)
        //   [1] = -5_001  (below threshold, pruned)
        //   [2] = -5_000  (at threshold, NOT pruned -- filter uses strict <)
        //   [3] = -4_999  (above threshold, NOT pruned)
        //   [4] =  0      (above threshold, NOT pruned)
        //   [5] =  300 * scale = 300_000  (above threshold, NOT pruned)
        trainer.storage.regrets[0].store(-6_000, Ordering::Relaxed);
        trainer.storage.regrets[1].store(-5_001, Ordering::Relaxed);
        trainer.storage.regrets[2].store(-5_000, Ordering::Relaxed);
        trainer.storage.regrets[3].store(-4_999, Ordering::Relaxed);
        trainer.storage.regrets[4].store(0, Ordering::Relaxed);
        trainer.storage.regrets[5].store(300 * scale, Ordering::Relaxed);

        // Remaining regret slots are 0 (default), which is above threshold.
        // Total slots = n, below count = 2 (indices 0 and 1).
        let fraction = trainer.prune_fraction();
        let expected = 2.0 / n as f64;
        assert!(
            (fraction - expected).abs() < 1e-10,
            "expected prune fraction {expected}, got {fraction}"
        );
    }

    #[test]
    fn full_ev_tracker_not_reset_on_strategy_refresh() {
        let config = toy_config();
        let mut trainer = toy_trainer(config);

        // Accumulate EV data into the full_ev_tracker at root.
        let root = trainer.tree.root;
        trainer.full_ev_tracker.accumulate(root, 0, 42, 10.0);

        // Verify it was recorded.
        let dec_map = trainer.tree.decision_index_map();
        let dec_idx = dec_map[root as usize] as usize;
        let evs_before = trainer.full_ev_tracker.hand_ev_array(dec_idx, 0);
        assert!(
            (evs_before[42] - 10.0).abs() < 0.01,
            "precondition: full_ev_tracker should have data"
        );

        // Also accumulate into the scenario_ev_tracker.
        trainer.scenario_ev_tracker.set_nodes(vec![root]);
        trainer.scenario_ev_tracker.accumulate(0, 0, 42, 5.0);

        // Force strategy refresh: set interval to 0 so refresh fires immediately.
        trainer.strategy_refresh_interval_secs = 0;
        trainer.last_strategy_refresh_secs = 0;
        trainer.iterations = 1; // need at least 1 iteration to pass

        // Trigger check_timed_actions which includes strategy refresh.
        trainer
            .check_timed_actions()
            .expect("check_timed_actions should not fail");

        // scenario_ev_tracker SHOULD be reset (windowed for TUI display).
        let scenario_evs = trainer.scenario_ev_tracker.hand_ev_array(0, 0);
        assert!(
            (scenario_evs[42] - 0.0).abs() < 1e-10,
            "scenario_ev_tracker should be reset, got {}",
            scenario_evs[42],
        );

        // full_ev_tracker should NOT be reset (cumulative for hand_ev.bin persistence).
        let evs_after = trainer.full_ev_tracker.hand_ev_array(dec_idx, 0);
        assert!(
            (evs_after[42] - 10.0).abs() < 0.01,
            "full_ev_tracker should NOT be reset after strategy refresh, got {}",
            evs_after[42],
        );
    }

    #[test]
    fn compute_exploitability_returns_nonneg() {
        let mut config = toy_config();
        config.training.exploitability_samples = 100;
        let trainer = toy_trainer(config);
        let exploit = trainer.compute_exploitability();
        assert!(
            exploit >= -0.01,
            "Exploitability should be non-negative, got {exploit}",
        );
        assert!(exploit.is_finite(), "Exploitability should be finite");
    }

    #[test]
    fn compute_exploitability_units_are_mbb() {
        let mut config = toy_config();
        config.training.exploitability_samples = 100;
        let trainer = toy_trainer(config);
        let exploit = trainer.compute_exploitability();
        // With a uniform (untrained) strategy, exploitability should be
        // reasonably bounded. Stack depth is 20 chips = 10 BB = 10,000 mbb.
        // Exploitability must be within [0, 10,000] mbb/hand.
        assert!(
            exploit <= 10_000.0,
            "Exploitability should be <= stack depth in mbb, got {exploit}",
        );
    }

    #[test]
    fn trainer_creates_brcfr_optimizer() {
        let mut config = toy_config();
        config.training.optimizer = "brcfr+".to_string();
        config.training.brcfr_eta = 0.7;
        let trainer = BlueprintTrainer::new(config);
        let opt = trainer
            .storage
            .optimizer
            .as_ref()
            .expect("optimizer should be set");
        assert_eq!(opt.name(), "brcfr+");
        assert!(
            trainer.storage.predictions.is_some(),
            "predictions should be enabled"
        );
    }

    #[test]
    fn brcfr_runs_br_pass_after_warmup() {
        let mut config = toy_config();
        config.training.optimizer = "brcfr+".to_string();
        config.training.brcfr_eta = 0.6;
        config.training.brcfr_warmup_iterations = 200;
        config.training.brcfr_interval = 200;
        config.training.iterations = Some(400);
        config.training.exploitability_samples = 100;
        let mut trainer = toy_trainer(config);
        trainer.train().expect("training should complete");
        // After training 400 iters with warmup=200, interval=200, we should have
        // had at least 1 BR pass (at iter 200).
        // Predictions should be populated.
        let preds = trainer
            .storage
            .predictions
            .as_ref()
            .expect("predictions enabled");
        let any_nonzero = preds.iter().any(|a| a.load(Ordering::Relaxed) != 0);
        assert!(any_nonzero, "predictions should be populated after BR pass");
        // last_br_iteration should have been advanced past 0.
        assert!(
            trainer.last_br_iteration > 0,
            "last_br_iteration should be > 0 after BR pass, got {}",
            trainer.last_br_iteration,
        );
        // predictions should be locked after the BR pass
        assert!(
            trainer.storage.predictions_locked.load(Ordering::Relaxed),
            "predictions should be locked after BR pass",
        );
    }

    #[test]
    fn on_exploitable_spots_callback_fires_after_br_pass() {
        use std::sync::Mutex;
        let mut config = toy_config();
        config.training.optimizer = "brcfr+".to_string();
        config.training.brcfr_eta = 0.6;
        config.training.brcfr_warmup_iterations = 200;
        config.training.brcfr_interval = 200;
        config.training.iterations = Some(400);
        config.training.exploitability_samples = 100;
        let mut trainer = toy_trainer(config);

        let received = Arc::new(Mutex::new(Vec::new()));
        let received_clone = Arc::clone(&received);
        trainer.on_exploitable_spots = Some(Box::new(move |spots| {
            received_clone.lock().unwrap().push(spots);
        }));

        trainer.train().expect("training should complete");

        let calls = received.lock().unwrap();
        assert!(
            !calls.is_empty(),
            "on_exploitable_spots should have been called at least once",
        );
        // Each call should have received a non-empty vec (there are predictions)
        for (i, spots) in calls.iter().enumerate() {
            assert!(!spots.is_empty(), "call {i} should have non-empty spots",);
        }
    }

    #[test]
    fn on_exploitable_spots_default_is_none() {
        let config = toy_config();
        let trainer = BlueprintTrainer::new(config);
        assert!(trainer.on_exploitable_spots.is_none());
    }
}
