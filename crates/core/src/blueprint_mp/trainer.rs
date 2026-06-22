//! Training loop for the N-player multiplayer MCCFR blueprint solver.
//!
//! Drives external-sampling MCCFR iterations with DCFR discounting
//! and parallel batches via rayon.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_arguments
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// Global prune counters — accumulated per batch, read+reset by TUI bridge.
pub static PRUNE_HITS: AtomicU64 = AtomicU64::new(0);
pub static PRUNE_TOTAL: AtomicU64 = AtomicU64::new(0);
static LAZY_BATCH_WALL_NANOS: AtomicU64 = AtomicU64::new(0);
static LAZY_DEAL_NANOS: AtomicU64 = AtomicU64::new(0);
static LAZY_BUCKET_NANOS: AtomicU64 = AtomicU64::new(0);
static LAZY_TRAVERSE_NANOS: AtomicU64 = AtomicU64::new(0);
static LAZY_DISCOUNT_NANOS: AtomicU64 = AtomicU64::new(0);
static LAZY_TRAVERSAL_COUNT: AtomicU64 = AtomicU64::new(0);
static LAZY_MAX_JOB_NANOS: AtomicU64 = AtomicU64::new(0);
static LAZY_MAX_JOB_ITER: AtomicU64 = AtomicU64::new(0);
static LAZY_SLOW_JOBS: AtomicU64 = AtomicU64::new(0);
static LAZY_MAX_TRAVERSER_NANOS: AtomicU64 = AtomicU64::new(0);
static LAZY_MAX_TRAVERSER_CONTEXT: AtomicU64 = AtomicU64::new(0);
static LAZY_SLOW_TRAVERSERS: AtomicU64 = AtomicU64::new(0);
const SLOW_LAZY_JOB_NANOS: u64 = 1_000_000_000;
const SLOW_LAZY_TRAVERSER_NANOS: u64 = 1_000_000_000;

use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;

use super::MAX_PLAYERS;
use super::config::{
    BlueprintMpConfig, MpChanceContinuationMode, MpGameConfig, MpNegativeActionPurgeMode,
    MpTrainingConfig,
};
use super::game_tree::MpGameTree;
use super::lazy_mccfr::{
    ExactChanceRunouts, ExactTurnRunout, LazyMpGame, NegativeActionTraversalConfig,
    traverse_external_lazy,
};
use super::mccfr::{sample_deal, traverse_external};
use super::sparse_storage::SparseMpStorage;
use super::storage::{MpStorage, REGRET_SCALE};
use super::types::{Bucket, Chips, Deal, DealWithBuckets, Seat};
use crate::blueprint_v2::mccfr::AllBuckets;
use crate::blueprint_v2::trainer::load_bucket_files;
use crate::poker::{Card, full_deck};

/// Result of a training run.
pub struct TrainResult {
    pub meta_iterations: u64,
    pub final_strategy_delta: f64,
}

/// Timing counters accumulated by the lazy sparse MP training loop.
///
/// Worker component timings are summed across Rayon workers, so they can exceed
/// wall-clock time. `batch_wall_nanos` and `discount_nanos` are wall-clock
/// measurements from the coordinator thread.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LazyMpTimingSnapshot {
    pub batch_wall_nanos: u64,
    pub deal_nanos: u64,
    pub bucket_nanos: u64,
    pub traverse_nanos: u64,
    pub discount_nanos: u64,
    pub traversal_count: u64,
    pub max_job_nanos: u64,
    pub max_job_iter: u64,
    pub slow_jobs: u64,
    pub max_traverser_nanos: u64,
    pub max_traverser_iter: u64,
    pub max_traverser_seat: u8,
    pub slow_traversers: u64,
}

/// Read and reset lazy sparse MP training timing counters.
#[must_use]
pub fn take_lazy_mp_timing_snapshot() -> LazyMpTimingSnapshot {
    let max_traverser_context = LAZY_MAX_TRAVERSER_CONTEXT.swap(0, Ordering::Relaxed);
    LazyMpTimingSnapshot {
        batch_wall_nanos: LAZY_BATCH_WALL_NANOS.swap(0, Ordering::Relaxed),
        deal_nanos: LAZY_DEAL_NANOS.swap(0, Ordering::Relaxed),
        bucket_nanos: LAZY_BUCKET_NANOS.swap(0, Ordering::Relaxed),
        traverse_nanos: LAZY_TRAVERSE_NANOS.swap(0, Ordering::Relaxed),
        discount_nanos: LAZY_DISCOUNT_NANOS.swap(0, Ordering::Relaxed),
        traversal_count: LAZY_TRAVERSAL_COUNT.swap(0, Ordering::Relaxed),
        max_job_nanos: LAZY_MAX_JOB_NANOS.swap(0, Ordering::Relaxed),
        max_job_iter: LAZY_MAX_JOB_ITER.swap(0, Ordering::Relaxed),
        slow_jobs: LAZY_SLOW_JOBS.swap(0, Ordering::Relaxed),
        max_traverser_nanos: LAZY_MAX_TRAVERSER_NANOS.swap(0, Ordering::Relaxed),
        max_traverser_iter: max_traverser_context >> 8,
        max_traverser_seat: (max_traverser_context & 0xFF) as u8,
        slow_traversers: LAZY_SLOW_TRAVERSERS.swap(0, Ordering::Relaxed),
    }
}

/// Shared training state accessible from outside the training loop.
pub struct TrainContext {
    pub tree: Arc<MpGameTree>,
    pub storage: Arc<MpStorage>,
    pub buckets: Arc<AllBuckets>,
    pub iterations: Arc<AtomicU64>,
    /// Set to `true` to signal the training loop to stop after the current batch.
    pub quit: Arc<AtomicBool>,
    pub num_players: u8,
    pub bucket_counts: [u16; 4],
}

/// Shared lazy/sparse training state accessible from outside the training loop.
pub struct LazyTrainContext {
    pub game: Arc<LazyMpGame>,
    pub storage: Arc<SparseMpStorage>,
    pub buckets: Arc<AllBuckets>,
    pub iterations: Arc<AtomicU64>,
    /// Set to `true` to signal the training loop to stop after the current batch.
    pub quit: Arc<AtomicBool>,
    pub num_players: u8,
    pub bucket_counts: [u16; 4],
}

/// Build tree and storage without starting training.
#[must_use]
pub fn setup_training(config: &BlueprintMpConfig) -> TrainContext {
    let tree = MpGameTree::build(&config.game, &config.action_abstraction);
    let bucket_counts = config.clustering.bucket_counts();
    let storage = MpStorage::new(&tree, bucket_counts);
    let bucket_files = config.training.cluster_path.as_ref().map_or_else(
        || [None, None, None, None],
        |path| load_bucket_files(std::path::Path::new(path)),
    );
    let mut all_buckets = AllBuckets::new(bucket_counts, bucket_files);
    if config.training.cluster_path.is_none() {
        all_buckets.equity_fallback = true;
    }
    TrainContext {
        tree: Arc::new(tree),
        storage: Arc::new(storage),
        buckets: Arc::new(all_buckets),
        iterations: Arc::new(AtomicU64::new(0)),
        quit: Arc::new(AtomicBool::new(false)),
        num_players: config.game.num_players,
        bucket_counts,
    }
}

/// Build lazy public game and sparse storage without materializing the eager tree.
#[must_use]
pub fn setup_lazy_training(config: &BlueprintMpConfig) -> LazyTrainContext {
    let game = LazyMpGame::new(&config.game, &config.action_abstraction);
    let bucket_counts = config.clustering.bucket_counts();
    let bucket_files = config.training.cluster_path.as_ref().map_or_else(
        || [None, None, None, None],
        |path| load_bucket_files(std::path::Path::new(path)),
    );
    let mut all_buckets = AllBuckets::new(bucket_counts, bucket_files);
    if config.training.cluster_path.is_none() {
        all_buckets.equity_fallback = true;
    }
    LazyTrainContext {
        game: Arc::new(game),
        storage: Arc::new(SparseMpStorage::new()),
        buckets: Arc::new(all_buckets),
        iterations: Arc::new(AtomicU64::new(0)),
        quit: Arc::new(AtomicBool::new(false)),
        num_players: config.game.num_players,
        bucket_counts,
    }
}

/// Run training on an existing context. Updates `ctx.iterations` atomically.
#[must_use]
pub fn run_training(
    ctx: &TrainContext,
    training: &MpTrainingConfig,
    game: &MpGameConfig,
) -> TrainResult {
    training_loop(
        &ctx.tree,
        &ctx.storage,
        &ctx.buckets,
        training,
        ctx.num_players,
        ctx.bucket_counts,
        game.rake_rate,
        Chips(game.rake_cap),
        &ctx.iterations,
        &ctx.quit,
    )
}

/// Run lazy/sparse training on an existing context. Updates `ctx.iterations` atomically.
#[must_use]
pub fn run_lazy_training(
    ctx: &LazyTrainContext,
    training: &MpTrainingConfig,
    game: &MpGameConfig,
) -> TrainResult {
    LazyMpTrainingStepper::from_context(ctx, training, game, 0).run_until_stopped()
}

/// Train an N-player blueprint strategy (convenience wrapper).
///
/// One meta-iteration = N traversals (one per seat as traverser).
#[must_use]
pub fn train_blueprint_mp(config: &BlueprintMpConfig) -> TrainResult {
    let ctx = setup_training(config);
    run_training(&ctx, &config.training, &config.game)
}

/// Train an N-player blueprint strategy with lazy traversal and sparse storage.
///
/// One meta-iteration = N traversals (one per seat as traverser).
#[must_use]
pub fn train_blueprint_mp_lazy(config: &BlueprintMpConfig) -> TrainResult {
    let ctx = setup_lazy_training(config);
    run_lazy_training(&ctx, &config.training, &config.game)
}

fn training_loop(
    tree: &MpGameTree,
    storage: &MpStorage,
    all_buckets: &AllBuckets,
    config: &MpTrainingConfig,
    num_players: u8,
    bucket_counts: [u16; 4],
    rake_rate: f64,
    rake_cap: Chips,
    iterations: &AtomicU64,
    quit: &AtomicBool,
) -> TrainResult {
    let max_iters = config.iterations.unwrap_or(u64::MAX);
    let scaled_threshold = (f64::from(config.prune_threshold) * REGRET_SCALE)
        .clamp(f64::from(i32::MIN), f64::from(i32::MAX))
        .round() as i32;
    let mut meta_iter: u64 = 0;
    let mut rng = SmallRng::seed_from_u64(0xDEAD_BEEF_CAFE_1234);

    loop {
        if meta_iter >= max_iters || quit.load(Ordering::Relaxed) {
            break;
        }
        let remaining = max_iters.saturating_sub(meta_iter);
        let batch = config.batch_size.min(remaining);
        if batch == 0 {
            break;
        }

        let prune = should_prune(meta_iter, config, &mut rng);
        run_batch(
            tree,
            storage,
            all_buckets,
            num_players,
            bucket_counts,
            rake_rate,
            rake_cap,
            batch,
            meta_iter,
            prune,
            scaled_threshold,
        );
        meta_iter += batch;
        iterations.store(meta_iter, Ordering::Relaxed);

        if should_discount(meta_iter, config) {
            apply_dcfr_discount(storage, meta_iter, config);
        }
    }

    TrainResult {
        meta_iterations: meta_iter,
        final_strategy_delta: 0.0,
    }
}

/// Stateful batch runner for lazy sparse multiplayer blueprint training.
///
/// One completed unit is one meta-iteration: one sampled deal followed by one
/// traversal for each seat. The runner owns the batch-local state that used to
/// live inside `training_loop_lazy`, including the current base meta-iteration
/// and pruning RNG cadence.
pub struct LazyMpTrainingStepper {
    game: Arc<LazyMpGame>,
    storage: Arc<SparseMpStorage>,
    buckets: Arc<AllBuckets>,
    training: MpTrainingConfig,
    num_players: u8,
    bucket_counts: [u16; 4],
    rake_rate: f64,
    rake_cap: Chips,
    iterations: Arc<AtomicU64>,
    quit: Arc<AtomicBool>,
    meta_iter: u64,
    rng: SmallRng,
    scaled_threshold: i32,
}

impl LazyMpTrainingStepper {
    /// Build a stepper from an existing lazy training context.
    #[must_use]
    pub fn from_context(
        ctx: &LazyTrainContext,
        training: &MpTrainingConfig,
        game: &MpGameConfig,
        start_meta_iter: u64,
    ) -> Self {
        Self::new(
            Arc::clone(&ctx.game),
            Arc::clone(&ctx.storage),
            Arc::clone(&ctx.buckets),
            training.clone(),
            ctx.num_players,
            ctx.bucket_counts,
            game.rake_rate,
            Chips(game.rake_cap),
            Arc::clone(&ctx.iterations),
            Arc::clone(&ctx.quit),
            start_meta_iter,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        game: Arc<LazyMpGame>,
        storage: Arc<SparseMpStorage>,
        buckets: Arc<AllBuckets>,
        training: MpTrainingConfig,
        num_players: u8,
        bucket_counts: [u16; 4],
        rake_rate: f64,
        rake_cap: Chips,
        iterations: Arc<AtomicU64>,
        quit: Arc<AtomicBool>,
        start_meta_iter: u64,
    ) -> Self {
        let scaled_threshold = (f64::from(training.prune_threshold) * REGRET_SCALE)
            .clamp(f64::from(i32::MIN), f64::from(i32::MAX))
            .round() as i32;
        let rng = pruning_rng_at_meta_iter(start_meta_iter, &training);
        iterations.store(start_meta_iter, Ordering::Relaxed);
        Self {
            game,
            storage,
            buckets,
            training,
            num_players,
            bucket_counts,
            rake_rate,
            rake_cap,
            iterations,
            quit,
            meta_iter: start_meta_iter,
            rng,
            scaled_threshold,
        }
    }

    /// Current completed lazy sparse meta-iterations.
    #[must_use]
    pub const fn meta_iterations(&self) -> u64 {
        self.meta_iter
    }

    /// Run at most one configured batch, optionally capped by a runtime budget.
    ///
    /// Returns the number of completed meta-iterations. Runtime adapters should
    /// report this value to the shared runtime instead of mutating runtime
    /// counters directly.
    pub fn run_next_batch(&mut self, max_meta_iterations: Option<u64>) -> u64 {
        let max_iters = self.training.iterations.unwrap_or(u64::MAX);
        if self.meta_iter >= max_iters || self.quit.load(Ordering::Relaxed) {
            return 0;
        }

        let remaining = max_iters.saturating_sub(self.meta_iter);
        let mut batch = self.training.batch_size.min(remaining);
        if let Some(max_meta_iterations) = max_meta_iterations {
            batch = batch.min(max_meta_iterations);
        }
        if batch == 0 {
            return 0;
        }

        let prune = should_prune(self.meta_iter, &self.training, &mut self.rng);
        let negative_action = negative_action_traversal_config(&self.training, self.meta_iter);
        run_lazy_batch(
            &self.game,
            &self.storage,
            &self.buckets,
            self.num_players,
            self.bucket_counts,
            self.rake_rate,
            self.rake_cap,
            batch,
            self.meta_iter,
            prune,
            self.scaled_threshold,
            negative_action,
            self.training.chance_continuation_mode,
        );
        self.meta_iter += batch;
        self.iterations.store(self.meta_iter, Ordering::Relaxed);

        if should_discount(self.meta_iter, &self.training) {
            let discount_started = Instant::now();
            apply_dcfr_discount_lazy(&self.storage, self.meta_iter, &self.training);
            purge_negative_action_subtrees_after_discount(&self.storage, &self.training);
            LAZY_DISCOUNT_NANOS.fetch_add(nanos_since(discount_started), Ordering::Relaxed);
        }

        batch
    }

    /// Run until the configured iteration cap or quit flag stops training.
    #[must_use]
    pub fn run_until_stopped(&mut self) -> TrainResult {
        while self.run_next_batch(None) > 0 {}
        TrainResult {
            meta_iterations: self.meta_iter,
            final_strategy_delta: 0.0,
        }
    }
}

fn pruning_rng_at_meta_iter(start_meta_iter: u64, config: &MpTrainingConfig) -> SmallRng {
    let mut rng = SmallRng::seed_from_u64(0xC0DE_5EED_1234_5678);
    let mut meta_iter = 0;
    while meta_iter < start_meta_iter {
        let remaining = start_meta_iter.saturating_sub(meta_iter);
        let batch = config.batch_size.min(remaining);
        if batch == 0 {
            break;
        }
        let _ = should_prune(meta_iter, config, &mut rng);
        meta_iter += batch;
    }
    rng
}

/// Determine whether the current batch should use ordinary traversal pruning.
///
/// This only skips eligible traverser-side action branches for the current
/// batch; it does not physically remove sparse rows or strategy sums. Keep it
/// opt-in because the previous always-on MP traversal-pruning path could starve
/// branches when configured too aggressively.
fn should_prune(meta_iter: u64, config: &MpTrainingConfig, rng: &mut impl Rng) -> bool {
    if !config.traversal_pruning_enabled || meta_iter < config.prune_after_iterations {
        return false;
    }
    let explore: f64 = rng.random();
    explore >= config.prune_explore_pct
}

fn run_batch(
    tree: &MpGameTree,
    storage: &MpStorage,
    all_buckets: &AllBuckets,
    num_players: u8,
    bucket_counts: [u16; 4],
    rake_rate: f64,
    rake_cap: Chips,
    batch_size: u64,
    base_iter: u64,
    prune: bool,
    prune_threshold: i32,
) {
    use super::mccfr::PruneStats;
    let batch_stats: PruneStats = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let seed = base_iter
                .wrapping_add(i)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut rng = SmallRng::seed_from_u64(seed);
            let deal = sample_deal(num_players, &mut rng);
            let buckets = compute_deal_buckets(&deal, all_buckets, bucket_counts);
            let mut local = PruneStats::default();
            for traverser in 0..num_players {
                let (_, stats) = traverse_external(
                    tree,
                    storage,
                    &buckets,
                    Seat::from_raw(traverser),
                    tree.root,
                    &mut rng,
                    rake_rate,
                    rake_cap,
                    prune,
                    prune_threshold,
                );
                local.merge(stats);
            }
            local
        })
        .reduce(PruneStats::default, |mut a, b| {
            a.merge(b);
            a
        });
    PRUNE_HITS.fetch_add(batch_stats.hits, Ordering::Relaxed);
    PRUNE_TOTAL.fetch_add(batch_stats.total, Ordering::Relaxed);
}

fn run_lazy_batch(
    game: &LazyMpGame,
    storage: &SparseMpStorage,
    all_buckets: &AllBuckets,
    num_players: u8,
    bucket_counts: [u16; 4],
    rake_rate: f64,
    rake_cap: Chips,
    batch_size: u64,
    base_iter: u64,
    prune: bool,
    prune_threshold: i32,
    negative_action: NegativeActionTraversalConfig,
    chance_mode: MpChanceContinuationMode,
) {
    use super::mccfr::PruneStats;
    let batch_started = Instant::now();
    let (batch_stats, timing): (PruneStats, LazyWorkerTiming) = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let job_started = Instant::now();
            let job_iter = base_iter.wrapping_add(i);
            let seed = base_iter
                .wrapping_add(i)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut rng = SmallRng::seed_from_u64(seed);

            let deal_started = Instant::now();
            let deal = sample_deal(num_players, &mut rng);
            let deal_nanos = nanos_since(deal_started);

            let bucket_started = Instant::now();
            let buckets = compute_deal_buckets(&deal, all_buckets, bucket_counts);
            let exact_runouts =
                exact_chance_runouts_for_deal(&deal, all_buckets, bucket_counts, chance_mode);
            let bucket_nanos = nanos_since(bucket_started);

            let mut local = PruneStats::default();
            let traversers_started = Instant::now();
            let mut max_traverser_nanos = 0;
            let mut max_traverser_seat = 0;
            let mut slow_traversers = 0;
            for traverser in 0..num_players {
                let traverser_started = Instant::now();
                let (_, stats) = traverse_external_lazy(
                    game,
                    storage,
                    &buckets,
                    &exact_runouts,
                    chance_mode,
                    Seat::from_raw(traverser),
                    &mut rng,
                    rake_rate,
                    rake_cap,
                    prune,
                    prune_threshold,
                    negative_action,
                );
                let traverser_nanos = nanos_since(traverser_started);
                if traverser_nanos > max_traverser_nanos {
                    max_traverser_nanos = traverser_nanos;
                    max_traverser_seat = traverser;
                }
                if traverser_nanos >= SLOW_LAZY_TRAVERSER_NANOS {
                    slow_traversers += 1;
                }
                local.merge(stats);
            }
            let traverse_nanos = nanos_since(traversers_started);
            let job_nanos = nanos_since(job_started);
            (
                local,
                LazyWorkerTiming {
                    deal_nanos,
                    bucket_nanos,
                    traverse_nanos,
                    traversal_count: u64::from(num_players),
                    max_job_nanos: job_nanos,
                    max_job_iter: job_iter,
                    slow_jobs: if job_nanos >= SLOW_LAZY_JOB_NANOS {
                        1
                    } else {
                        0
                    },
                    max_traverser_nanos,
                    max_traverser_iter: job_iter,
                    max_traverser_seat,
                    slow_traversers,
                },
            )
        })
        .reduce(
            || (PruneStats::default(), LazyWorkerTiming::default()),
            |mut a, b| {
                a.0.merge(b.0);
                a.1.merge(b.1);
                a
            },
        );
    LAZY_BATCH_WALL_NANOS.fetch_add(nanos_since(batch_started), Ordering::Relaxed);
    LAZY_DEAL_NANOS.fetch_add(timing.deal_nanos, Ordering::Relaxed);
    LAZY_BUCKET_NANOS.fetch_add(timing.bucket_nanos, Ordering::Relaxed);
    LAZY_TRAVERSE_NANOS.fetch_add(timing.traverse_nanos, Ordering::Relaxed);
    LAZY_TRAVERSAL_COUNT.fetch_add(timing.traversal_count, Ordering::Relaxed);
    record_lazy_max(
        &LAZY_MAX_JOB_NANOS,
        &LAZY_MAX_JOB_ITER,
        timing.max_job_nanos,
        timing.max_job_iter,
    );
    LAZY_SLOW_JOBS.fetch_add(timing.slow_jobs, Ordering::Relaxed);
    record_lazy_max(
        &LAZY_MAX_TRAVERSER_NANOS,
        &LAZY_MAX_TRAVERSER_CONTEXT,
        timing.max_traverser_nanos,
        pack_traverser_context(timing.max_traverser_iter, timing.max_traverser_seat),
    );
    LAZY_SLOW_TRAVERSERS.fetch_add(timing.slow_traversers, Ordering::Relaxed);
    PRUNE_HITS.fetch_add(batch_stats.hits, Ordering::Relaxed);
    PRUNE_TOTAL.fetch_add(batch_stats.total, Ordering::Relaxed);
}

fn negative_action_traversal_config(
    config: &MpTrainingConfig,
    meta_iter: u64,
) -> NegativeActionTraversalConfig {
    NegativeActionTraversalConfig {
        enabled: config.negative_action_subtree_purge_enabled
            && meta_iter >= config.prune_after_iterations
            && matches!(
                config.negative_action_purge_mode,
                MpNegativeActionPurgeMode::ScanHistoryPrefix
            ),
        prune_below: config.negative_action_prune_below,
        reactivate_at: config.negative_action_reactivate_at,
    }
}

#[derive(Clone, Copy, Default)]
struct LazyWorkerTiming {
    deal_nanos: u64,
    bucket_nanos: u64,
    traverse_nanos: u64,
    traversal_count: u64,
    max_job_nanos: u64,
    max_job_iter: u64,
    slow_jobs: u64,
    max_traverser_nanos: u64,
    max_traverser_iter: u64,
    max_traverser_seat: u8,
    slow_traversers: u64,
}

impl LazyWorkerTiming {
    fn merge(&mut self, other: Self) {
        self.deal_nanos = self.deal_nanos.saturating_add(other.deal_nanos);
        self.bucket_nanos = self.bucket_nanos.saturating_add(other.bucket_nanos);
        self.traverse_nanos = self.traverse_nanos.saturating_add(other.traverse_nanos);
        self.traversal_count = self.traversal_count.saturating_add(other.traversal_count);
        self.slow_jobs = self.slow_jobs.saturating_add(other.slow_jobs);
        self.slow_traversers = self.slow_traversers.saturating_add(other.slow_traversers);
        if other.max_job_nanos > self.max_job_nanos {
            self.max_job_nanos = other.max_job_nanos;
            self.max_job_iter = other.max_job_iter;
        }
        if other.max_traverser_nanos > self.max_traverser_nanos {
            self.max_traverser_nanos = other.max_traverser_nanos;
            self.max_traverser_iter = other.max_traverser_iter;
            self.max_traverser_seat = other.max_traverser_seat;
        }
    }
}

fn record_lazy_max(max_nanos: &AtomicU64, max_context: &AtomicU64, nanos: u64, context: u64) {
    let mut current = max_nanos.load(Ordering::Relaxed);
    while nanos > current {
        match max_nanos.compare_exchange_weak(current, nanos, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => {
                max_context.store(context, Ordering::Relaxed);
                break;
            }
            Err(next) => current = next,
        }
    }
}

fn pack_traverser_context(iter: u64, seat: u8) -> u64 {
    (iter << 8) | u64::from(seat)
}

fn nanos_since(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn should_discount(meta_iter: u64, config: &MpTrainingConfig) -> bool {
    if meta_iter < config.lcfr_warmup_iterations {
        return false;
    }
    let interval = config.lcfr_discount_interval.max(1);
    meta_iter.is_multiple_of(interval)
}

fn apply_dcfr_discount(storage: &MpStorage, meta_iter: u64, config: &MpTrainingConfig) {
    let interval = config.lcfr_discount_interval.max(1);
    let epoch = meta_iter / interval;
    let (d_pos, d_neg) = regret_discount_factors(epoch, config.dcfr_alpha, config.dcfr_beta);
    let d_strat = strategy_discount_factor(epoch, config.dcfr_gamma);

    storage.regrets.par_iter().for_each(|atom| {
        let v = atom.load(Ordering::Relaxed);
        let d = if v >= 0 { d_pos } else { d_neg };
        let discounted = (f64::from(v) * d)
            .round()
            .clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32;
        atom.store(discounted, Ordering::Relaxed);
    });
    storage.strategy_sums.par_iter().for_each(|atom| {
        let v = atom.load(Ordering::Relaxed);
        let discounted = ((v as f64) * d_strat).clamp(0.0, u64::MAX as f64) as u64;
        atom.store(discounted, Ordering::Relaxed);
    });
}

fn apply_dcfr_discount_lazy(storage: &SparseMpStorage, meta_iter: u64, config: &MpTrainingConfig) {
    let interval = config.lcfr_discount_interval.max(1);
    let epoch = meta_iter / interval;
    let (d_pos, d_neg) = regret_discount_factors(epoch, config.dcfr_alpha, config.dcfr_beta);
    let d_strat = strategy_discount_factor(epoch, config.dcfr_gamma);

    storage.discount(d_pos, d_neg, d_strat);
}

fn purge_negative_action_subtrees_after_discount(
    storage: &SparseMpStorage,
    config: &MpTrainingConfig,
) {
    if !config.negative_action_subtree_purge_enabled {
        return;
    }
    if config.negative_action_purge_mode != MpNegativeActionPurgeMode::ScanHistoryPrefix {
        return;
    }
    storage.purge_blocked_negative_action_subtrees_after_discount(
        config.negative_action_reactivate_at,
    );
}

fn regret_discount_factors(epoch: u64, alpha: f64, beta: f64) -> (f64, f64) {
    let t = epoch as f64;
    let ta = t.powf(alpha);
    let d_pos = ta / (ta + 1.0);
    let tb = t.powf(beta);
    let d_neg = tb / (tb + 1.0);
    (d_pos, d_neg)
}

fn strategy_discount_factor(epoch: u64, gamma: f64) -> f64 {
    let t = epoch as f64;
    (t / (t + 1.0)).powf(gamma)
}

/// Compute bucket assignments for a deal using real clustering when available.
fn compute_deal_buckets(
    deal: &Deal,
    all_buckets: &AllBuckets,
    _bucket_counts: [u16; 4],
) -> DealWithBuckets {
    use crate::blueprint_v2::Street as V2Street;
    let streets = [
        V2Street::Preflop,
        V2Street::Flop,
        V2Street::Turn,
        V2Street::River,
    ];
    let board_slices: [&[crate::poker::Card]; 4] =
        [&[], &deal.board[..3], &deal.board[..4], &deal.board[..5]];
    let mut buckets = [[Bucket(0); 4]; MAX_PLAYERS];
    for (p, seat_buckets) in buckets
        .iter_mut()
        .enumerate()
        .take(deal.num_players as usize)
    {
        for (s, (&street, board)) in streets.iter().zip(board_slices.iter()).enumerate() {
            seat_buckets[s] = Bucket(all_buckets.get_bucket(street, deal.hole_cards[p], board));
        }
    }
    DealWithBuckets {
        deal: deal.clone(),
        buckets,
    }
}

fn exact_chance_runouts_for_deal(
    deal: &Deal,
    all_buckets: &AllBuckets,
    bucket_counts: [u16; 4],
    chance_mode: MpChanceContinuationMode,
) -> ExactChanceRunouts {
    match chance_mode {
        MpChanceContinuationMode::SampledFullDeal => ExactChanceRunouts::default(),
        MpChanceContinuationMode::SampledTurnExactRiver => {
            let turn_deal = compute_deal_buckets(deal, all_buckets, bucket_counts);
            let river_deals =
                legal_river_deals_for_turn_prefix(deal, all_buckets, bucket_counts, deal.board[3]);
            ExactChanceRunouts {
                turns: vec![ExactTurnRunout {
                    turn: deal.board[3],
                    turn_deal,
                    river_deals,
                }],
            }
        }
        MpChanceContinuationMode::SampledFlopExactTurnRiver => {
            let turns = full_deck()
                .into_iter()
                .filter(|card| !deal_uses_hole_or_board_prefix(deal, *card, 3))
                .filter_map(|turn| {
                    let river_deals =
                        legal_river_deals_for_turn_prefix(deal, all_buckets, bucket_counts, turn);
                    let turn_deal = river_deals.first().cloned()?;
                    Some(ExactTurnRunout {
                        turn,
                        turn_deal,
                        river_deals,
                    })
                })
                .collect();
            ExactChanceRunouts { turns }
        }
    }
}

fn legal_river_deals_for_turn_prefix(
    deal: &Deal,
    all_buckets: &AllBuckets,
    bucket_counts: [u16; 4],
    turn: Card,
) -> Vec<DealWithBuckets> {
    full_deck()
        .into_iter()
        .filter(|card| !deal_uses_hole_or_board_prefix(deal, *card, 3) && *card != turn)
        .map(|river| {
            let mut river_deal = deal.clone();
            river_deal.board[3] = turn;
            river_deal.board[4] = river;
            compute_deal_buckets(&river_deal, all_buckets, bucket_counts)
        })
        .collect()
}

fn deal_uses_hole_or_board_prefix(deal: &Deal, card: Card, board_len: usize) -> bool {
    deal.hole_cards
        .iter()
        .take(deal.num_players as usize)
        .flatten()
        .any(|used| *used == card)
        || deal.board[..board_len].iter().any(|used| *used == card)
}

/// Trivial fallback bucketing (for tests without cluster files).
#[cfg(test)]
fn compute_buckets_trivial(deal: &Deal, bucket_counts: [u16; 4]) -> DealWithBuckets {
    use crate::hands::CanonicalHand;
    let mut buckets = [[Bucket(0); 4]; MAX_PLAYERS];
    for (seat_buckets, hole) in buckets
        .iter_mut()
        .zip(deal.hole_cards.iter())
        .take(deal.num_players as usize)
    {
        // Preflop: canonical hand index (0-168) for 169 unique buckets
        let hand_idx = CanonicalHand::from_cards(hole[0], hole[1]).index() as u16;
        for (street, &count) in bucket_counts.iter().enumerate() {
            seat_buckets[street] = Bucket(hand_idx % count);
        }
    }
    DealWithBuckets {
        deal: deal.clone(),
        buckets,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use test_macros::timed_test;

    use super::*;
    use crate::blueprint_mp::config::{
        BlueprintMpConfig, ForcedBet, ForcedBetKind, MpActionAbstractionConfig,
        MpChanceContinuationMode, MpClusteringConfig, MpGameConfig, MpNegativeActionPurgeMode,
        MpSnapshotConfig, MpSnapshotFormat, MpStreetCluster, MpStreetSizes, MpTrainingBackend,
        MpTrainingConfig,
    };
    use crate::blueprint_mp::game_tree::MpGameTree;
    use crate::blueprint_mp::mccfr::sample_deal;
    use crate::blueprint_mp::storage::MpStorage;
    use crate::blueprint_mp::types::Street;
    use crate::blueprint_v2::Street as V2Street;
    use crate::blueprint_v2::bucket_file::{BucketFile, BucketFileHeader};
    use crate::blueprint_v2::cluster_pipeline::{canonical_key, combo_index};
    use crate::poker::{Suit, Value};
    use crate::{abstraction::isomorphism::CanonicalBoard, blueprint_v2::bucket_file};

    fn toy_config(num_players: u8, iterations: u64) -> BlueprintMpConfig {
        let blinds = vec![
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
        ];
        let game = MpGameConfig {
            name: format!("{num_players}-player trainer test"),
            num_players,
            stack_depth: 20.0,
            allow_preflop_limp: true,
            blinds,
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        let empty = MpStreetSizes {
            lead: vec![],
            raise: vec![],
        };
        let action = MpActionAbstractionConfig {
            max_flop_players: None,
            preflop: empty.clone(),
            flop: empty.clone(),
            turn: empty.clone(),
            river: empty,
        };
        let clustering = MpClusteringConfig {
            preflop: MpStreetCluster { buckets: 10 },
            flop: MpStreetCluster { buckets: 10 },
            turn: MpStreetCluster { buckets: 10 },
            river: MpStreetCluster { buckets: 10 },
        };
        let training = MpTrainingConfig {
            backend: MpTrainingBackend::Eager,
            chance_continuation_mode:
                crate::blueprint_mp::config::MpChanceContinuationMode::SampledFullDeal,
            cluster_path: None,
            iterations: Some(iterations),
            time_limit_minutes: None,
            lcfr_warmup_iterations: 0,
            lcfr_discount_interval: 50,
            prune_after_iterations: 1_000_000,
            traversal_pruning_enabled: false,
            prune_threshold: -250,
            prune_explore_pct: 0.05,
            negative_action_subtree_purge_enabled: false,
            negative_action_prune_below: -1,
            negative_action_reactivate_at: 0,
            negative_action_purge_mode: MpNegativeActionPurgeMode::ScanHistoryPrefix,
            batch_size: 10,
            dcfr_alpha: 1.5,
            dcfr_beta: 0.0,
            dcfr_gamma: 2.0,
            print_every_minutes: 999,
            purify_threshold: 0.0,
            exploitability_interval_minutes: 0,
            exploitability_samples: 0,
        };
        let snapshots = MpSnapshotConfig {
            warmup_minutes: 999,
            snapshot_every_minutes: 999,
            output_dir: "/tmp/mp_test".into(),
            resume: false,
            max_snapshots: None,
            format: MpSnapshotFormat::Legacy,
        };
        BlueprintMpConfig {
            game,
            action_abstraction: action,
            clustering,
            training,
            snapshots,
        }
    }

    fn lazy_100bb_config(iterations: u64) -> BlueprintMpConfig {
        let mut config = toy_config(6, iterations);
        config.game.name = "6-max lazy 100bb test".to_string();
        config.game.stack_depth = 200.0;
        config.action_abstraction.preflop = MpStreetSizes {
            lead: vec![serde_yaml::Value::String("2bb".to_string())],
            raise: vec![
                vec![serde_yaml::Value::String("2.0x".to_string())],
                vec![serde_yaml::Value::String("2.0x".to_string())],
            ],
        };
        config.action_abstraction.flop = MpStreetSizes {
            lead: vec![serde_yaml::Value::Number(serde_yaml::Number::from(1))],
            raise: vec![vec![serde_yaml::Value::Number(serde_yaml::Number::from(1))]],
        };
        config.action_abstraction.turn = config.action_abstraction.flop.clone();
        config.action_abstraction.river = config.action_abstraction.flop.clone();
        config.training.batch_size = 1;
        config
    }

    fn toy_training_config(iterations: u64) -> MpTrainingConfig {
        MpTrainingConfig {
            backend: MpTrainingBackend::Eager,
            chance_continuation_mode:
                crate::blueprint_mp::config::MpChanceContinuationMode::SampledFullDeal,
            cluster_path: None,
            iterations: Some(iterations),
            time_limit_minutes: None,
            lcfr_warmup_iterations: 100,
            lcfr_discount_interval: 50,
            prune_after_iterations: 1_000_000,
            traversal_pruning_enabled: false,
            prune_threshold: -250,
            prune_explore_pct: 0.05,
            negative_action_subtree_purge_enabled: false,
            negative_action_prune_below: -1,
            negative_action_reactivate_at: 0,
            negative_action_purge_mode: MpNegativeActionPurgeMode::ScanHistoryPrefix,
            batch_size: 10,
            dcfr_alpha: 1.5,
            dcfr_beta: 0.0,
            dcfr_gamma: 2.0,
            print_every_minutes: 999,
            purify_threshold: 0.0,
            exploitability_interval_minutes: 0,
            exploitability_samples: 0,
        }
    }

    fn minimal_tree(num_players: u8) -> MpGameTree {
        let blinds = vec![
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
        ];
        let game = MpGameConfig {
            name: format!("{num_players}-player trainer tree"),
            num_players,
            stack_depth: 20.0,
            allow_preflop_limp: true,
            blinds,
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        let empty = MpStreetSizes {
            lead: vec![],
            raise: vec![],
        };
        let action = MpActionAbstractionConfig {
            max_flop_players: None,
            preflop: empty.clone(),
            flop: empty.clone(),
            turn: empty.clone(),
            river: empty,
        };
        MpGameTree::build(&game, &action)
    }

    // -- train_blueprint_mp integration tests --

    #[timed_test(3)]
    fn train_2_player_toy_completes() {
        let config = toy_config(2, 100);
        let result = train_blueprint_mp(&config);
        assert_eq!(result.meta_iterations, 100);
    }

    #[timed_test(3)]
    fn train_3_player_toy_completes() {
        let config = toy_config(3, 100);
        let result = train_blueprint_mp(&config);
        assert_eq!(result.meta_iterations, 100);
    }

    #[timed_test]
    fn train_updates_storage() {
        let config = toy_config(2, 10);
        let result = train_blueprint_mp(&config);
        assert!(result.meta_iterations > 0);
    }

    #[timed_test]
    fn lazy_train_2_player_toy_completes() {
        let config = toy_config(2, 10);
        let result = train_blueprint_mp_lazy(&config);
        assert_eq!(result.meta_iterations, 10);
    }

    #[timed_test]
    fn lazy_train_sampled_turn_exact_river_completes() {
        let mut config = toy_config(2, 1);
        config.training.batch_size = 1;
        config.training.chance_continuation_mode = MpChanceContinuationMode::SampledTurnExactRiver;

        let result = train_blueprint_mp_lazy(&config);

        assert_eq!(result.meta_iterations, 1);
    }

    #[timed_test(10)]
    fn lazy_train_sampled_flop_exact_turn_river_completes() {
        let mut config = toy_config(2, 1);
        config.training.batch_size = 1;
        config.training.chance_continuation_mode =
            MpChanceContinuationMode::SampledFlopExactTurnRiver;

        let result = train_blueprint_mp_lazy(&config);

        assert_eq!(result.meta_iterations, 1);
    }

    #[timed_test]
    fn lazy_timing_snapshot_tracks_compute_components() {
        let _ = take_lazy_mp_timing_snapshot();
        let mut config = toy_config(2, 2);
        config.training.batch_size = 1;

        let result = train_blueprint_mp_lazy(&config);
        let timing = take_lazy_mp_timing_snapshot();

        assert_eq!(result.meta_iterations, 2);
        assert!(timing.batch_wall_nanos > 0);
        assert!(timing.deal_nanos > 0);
        assert!(timing.bucket_nanos > 0);
        assert!(timing.traverse_nanos > 0);
        assert!(timing.traversal_count >= 4);
        assert!(timing.max_job_nanos > 0);
        assert!(timing.max_traverser_nanos > 0);
        assert!(timing.max_traverser_seat < config.game.num_players);
        let _ = take_lazy_mp_timing_snapshot();
    }

    #[timed_test]
    fn lazy_setup_100bb_two_preflop_raise_rows_does_not_allocate_eager_tree() {
        let config = lazy_100bb_config(1);
        let ctx = setup_lazy_training(&config);

        assert_eq!(ctx.num_players, 6);
        assert_eq!(ctx.game.root_state().to_act(), Seat::from_raw(2));
        assert_eq!(ctx.storage.entry_count(), 0);
    }

    #[timed_test(2)]
    fn train_result_tracks_iterations() {
        let config = toy_config(2, 50);
        let result = train_blueprint_mp(&config);
        assert_eq!(result.meta_iterations, 50);
    }

    // -- should_discount tests --

    #[timed_test]
    fn should_discount_false_during_warmup() {
        let config = toy_training_config(1000);
        // warmup=100, interval=50 => iter 50 is in warmup
        assert!(!should_discount(50, &config));
    }

    #[timed_test]
    fn should_discount_false_at_warmup_boundary() {
        let config = toy_training_config(1000);
        // iter=100 is still in warmup (< warmup)
        assert!(!should_discount(99, &config));
    }

    #[timed_test]
    fn should_discount_true_at_interval_after_warmup() {
        let config = toy_training_config(1000);
        // warmup=100, interval=50, iter=150 => 150 >= 100 and 150 % 50 == 0
        assert!(should_discount(150, &config));
    }

    #[timed_test]
    fn should_discount_false_between_intervals() {
        let config = toy_training_config(1000);
        // iter=125 is past warmup but 125 % 50 != 0
        assert!(!should_discount(125, &config));
    }

    #[timed_test]
    fn should_discount_true_at_zero_warmup() {
        let mut config = toy_training_config(1000);
        config.lcfr_warmup_iterations = 0;
        // interval=50, iter=50 => 50 % 50 == 0
        assert!(should_discount(50, &config));
    }

    // -- should_prune tests --

    #[timed_test]
    fn should_prune_false_before_warmup() {
        let mut config = toy_training_config(1000);
        config.prune_after_iterations = 100;
        config.prune_explore_pct = 0.0;
        let mut rng = SmallRng::seed_from_u64(42);
        assert!(!should_prune(50, &config, &mut rng));
    }

    #[timed_test]
    fn should_prune_false_after_warmup_no_explore() {
        let mut config = toy_training_config(1000);
        config.prune_after_iterations = 100;
        config.prune_explore_pct = 0.0;
        let mut rng = SmallRng::seed_from_u64(42);
        assert!(!should_prune(200, &config, &mut rng));
    }

    #[timed_test]
    fn should_prune_true_after_warmup_when_enabled_no_explore() {
        let mut config = toy_training_config(1000);
        config.traversal_pruning_enabled = true;
        config.prune_after_iterations = 100;
        config.prune_explore_pct = 0.0;
        let mut rng = SmallRng::seed_from_u64(42);
        assert!(should_prune(200, &config, &mut rng));
    }

    #[timed_test]
    fn should_prune_false_when_explore_pct_is_one() {
        let mut config = toy_training_config(1000);
        config.traversal_pruning_enabled = true;
        config.prune_after_iterations = 0;
        config.prune_explore_pct = 1.0; // explore_pct=1 => never prune
        let mut rng = SmallRng::seed_from_u64(42);
        assert!(!should_prune(200, &config, &mut rng));
    }

    #[timed_test]
    fn should_prune_disabled_even_after_warmup() {
        let mut config = toy_training_config(1000);
        config.prune_after_iterations = 500;
        config.prune_explore_pct = 0.0;
        let mut rng = SmallRng::seed_from_u64(42);
        assert!(!should_prune(499, &config, &mut rng));
        assert!(!should_prune(500, &config, &mut rng));
        assert!(!should_prune(1000, &config, &mut rng));
    }

    #[timed_test]
    fn negative_action_traversal_config_respects_prune_warmup() {
        let mut config = toy_training_config(1000);
        config.negative_action_subtree_purge_enabled = true;
        config.prune_after_iterations = 500;

        assert!(!negative_action_traversal_config(&config, 499).enabled);
        assert!(negative_action_traversal_config(&config, 500).enabled);
        assert!(negative_action_traversal_config(&config, 1000).enabled);
    }

    #[timed_test]
    fn negative_action_traversal_config_still_defaults_disabled_after_warmup() {
        let mut config = toy_training_config(1000);
        config.negative_action_subtree_purge_enabled = false;
        config.prune_after_iterations = 500;

        assert!(!negative_action_traversal_config(&config, 1000).enabled);
    }

    // -- regret_discount_factors tests --

    #[timed_test]
    fn regret_discount_factors_epoch_zero() {
        let (d_pos, d_neg) = regret_discount_factors(0, 1.5, 0.5);
        // t=0: 0^a / (0^a + 1) = 0
        assert!((d_pos).abs() < 1e-10);
        assert!((d_neg).abs() < 1e-10);
    }

    #[timed_test]
    fn regret_discount_factors_epoch_one() {
        let (d_pos, d_neg) = regret_discount_factors(1, 1.5, 0.5);
        // t=1: 1^a / (1^a + 1) = 0.5
        assert!((d_pos - 0.5).abs() < 1e-10);
        assert!((d_neg - 0.5).abs() < 1e-10);
    }

    #[timed_test]
    fn regret_discount_factors_large_epoch() {
        let (d_pos, d_neg) = regret_discount_factors(100, 1.5, 0.5);
        // Both should approach 1.0 as epoch grows
        assert!(d_pos > 0.99, "d_pos={d_pos}");
        assert!(d_neg > 0.9, "d_neg={d_neg}");
    }

    #[timed_test]
    fn regret_discount_factors_alpha_gt_beta() {
        // With alpha > beta, positive discount should exceed negative at same epoch
        let (d_pos, d_neg) = regret_discount_factors(5, 2.0, 0.5);
        assert!(d_pos > d_neg, "d_pos={d_pos} should exceed d_neg={d_neg}");
    }

    // -- strategy_discount_factor tests --

    #[timed_test]
    fn strategy_discount_factor_epoch_zero() {
        let d = strategy_discount_factor(0, 2.0);
        // (0 / 1)^2 = 0
        assert!((d).abs() < 1e-10);
    }

    #[timed_test]
    fn strategy_discount_factor_epoch_ten() {
        let d = strategy_discount_factor(10, 2.0);
        let expected = (10.0_f64 / 11.0).powf(2.0);
        assert!((d - expected).abs() < 1e-10);
    }

    #[timed_test]
    fn strategy_discount_factor_large_epoch() {
        let d = strategy_discount_factor(1000, 2.0);
        // Should approach 1.0
        assert!(d > 0.99, "d={d}");
    }

    // -- compute_buckets_trivial tests --

    #[timed_test]
    fn trivial_buckets_within_range() {
        let mut rng = rand::thread_rng();
        let deal = sample_deal(2, &mut rng);
        let counts = [10u16, 20, 30, 40];
        let dwb = compute_buckets_trivial(&deal, counts);
        for seat in 0..2 {
            for street in 0..4 {
                assert!(
                    dwb.buckets[seat][street].0 < counts[street],
                    "bucket out of range: seat={seat} street={street} bucket={}",
                    dwb.buckets[seat][street].0
                );
            }
        }
    }

    #[timed_test]
    fn trivial_buckets_preserves_deal() {
        let mut rng = rand::thread_rng();
        let deal = sample_deal(3, &mut rng);
        let counts = [10u16, 10, 10, 10];
        let dwb = compute_buckets_trivial(&deal, counts);
        assert_eq!(dwb.deal.num_players, 3);
        assert_eq!(dwb.deal.board, deal.board);
    }

    #[timed_test]
    fn exact_river_runouts_enumerate_legal_rivers_for_turn_prefix() {
        let mut rng = SmallRng::seed_from_u64(0xE2AC_7A11);
        let deal = sample_deal(6, &mut rng);
        let counts = [10u16, 10, 10, 10];
        let mut all_buckets = AllBuckets::new(counts, [None, None, None, None]);
        all_buckets.equity_fallback = true;

        let runouts = exact_chance_runouts_for_deal(
            &deal,
            &all_buckets,
            counts,
            MpChanceContinuationMode::SampledTurnExactRiver,
        );

        assert_eq!(runouts.turns.len(), 1);
        assert_eq!(runouts.turns[0].turn, deal.board[3]);
        assert_eq!(runouts.turns[0].river_deals.len(), 36);
        let mut rivers = Vec::with_capacity(runouts.turns[0].river_deals.len());
        for runout in &runouts.turns[0].river_deals {
            assert_eq!(runout.deal.board[..4], deal.board[..4]);
            let river = runout.deal.board[4];
            assert!(!deal_uses_hole_or_board_prefix(&deal, river, 4));
            assert!(!rivers.contains(&river));
            rivers.push(river);
            for seat in 0..deal.num_players as usize {
                assert!(runout.buckets[seat][Street::River.index()].0 < counts[3]);
            }
        }

        let sampled = exact_chance_runouts_for_deal(
            &deal,
            &all_buckets,
            counts,
            MpChanceContinuationMode::SampledFullDeal,
        );
        assert!(sampled.turns.is_empty());
    }

    #[timed_test(10)]
    fn exact_chance_runouts_enumerate_legal_turn_rivers_for_flop_prefix() {
        let mut rng = SmallRng::seed_from_u64(0xC0FF_EE17);
        let deal = sample_deal(3, &mut rng);
        let counts = [10u16, 10, 10, 10];
        let mut all_buckets = AllBuckets::new(counts, [None, None, None, None]);
        all_buckets.equity_fallback = true;
        let expected_turns = 52usize - deal.num_players as usize * 2 - 3;
        let expected_rivers_per_turn = expected_turns - 1;

        let runouts = exact_chance_runouts_for_deal(
            &deal,
            &all_buckets,
            counts,
            MpChanceContinuationMode::SampledFlopExactTurnRiver,
        );

        assert_eq!(runouts.turns.len(), expected_turns);
        let mut turns = Vec::with_capacity(runouts.turns.len());
        let mut total_rivers = 0usize;
        for turn_runout in &runouts.turns {
            let turn = turn_runout.turn;
            assert!(!deal_uses_hole_or_board_prefix(&deal, turn, 3));
            assert!(!turns.contains(&turn));
            turns.push(turn);
            assert_eq!(turn_runout.river_deals.len(), expected_rivers_per_turn);
            assert_eq!(turn_runout.turn_deal.deal.board[3], turn);
            total_rivers += turn_runout.river_deals.len();

            let mut rivers = Vec::with_capacity(turn_runout.river_deals.len());
            for runout in &turn_runout.river_deals {
                assert_eq!(runout.deal.board[..3], deal.board[..3]);
                assert_eq!(runout.deal.board[3], turn);
                let river = runout.deal.board[4];
                assert_ne!(river, turn);
                assert!(!deal_uses_hole_or_board_prefix(&deal, river, 3));
                assert!(!rivers.contains(&river));
                rivers.push(river);
                for seat in 0..deal.num_players as usize {
                    assert!(runout.buckets[seat][Street::Turn.index()].0 < counts[2]);
                    assert!(runout.buckets[seat][Street::River.index()].0 < counts[3]);
                }
            }
        }
        assert_eq!(total_rivers, expected_turns * expected_rivers_per_turn);
    }

    #[timed_test]
    fn compute_deal_buckets_preserves_suit_isomorphic_a2_flush_draw_texture() {
        let all_buckets = a2_texture_test_buckets();

        let flush_draw_cases = [
            (
                [card(Value::Ace, Suit::Spade), card(Value::Two, Suit::Spade)],
                [
                    card(Value::King, Suit::Spade),
                    card(Value::Seven, Suit::Spade),
                    card(Value::Three, Suit::Heart),
                ],
            ),
            (
                [card(Value::Ace, Suit::Heart), card(Value::Two, Suit::Heart)],
                [
                    card(Value::King, Suit::Heart),
                    card(Value::Seven, Suit::Heart),
                    card(Value::Three, Suit::Diamond),
                ],
            ),
            (
                [
                    card(Value::Ace, Suit::Diamond),
                    card(Value::Two, Suit::Diamond),
                ],
                [
                    card(Value::King, Suit::Diamond),
                    card(Value::Seven, Suit::Diamond),
                    card(Value::Three, Suit::Club),
                ],
            ),
            (
                [card(Value::Ace, Suit::Club), card(Value::Two, Suit::Club)],
                [
                    card(Value::King, Suit::Club),
                    card(Value::Seven, Suit::Club),
                    card(Value::Three, Suit::Spade),
                ],
            ),
        ];

        for (hole, flop) in flush_draw_cases {
            assert_eq!(
                flop_bucket_for(hole, flop, &all_buckets),
                7,
                "A2s with the two-tone board suit should canonicalize to the flush-draw bucket"
            );
        }

        let no_draw_cases = [
            (
                [card(Value::Ace, Suit::Heart), card(Value::Two, Suit::Heart)],
                [
                    card(Value::King, Suit::Spade),
                    card(Value::Seven, Suit::Spade),
                    card(Value::Three, Suit::Heart),
                ],
            ),
            (
                [
                    card(Value::Ace, Suit::Diamond),
                    card(Value::Two, Suit::Diamond),
                ],
                [
                    card(Value::King, Suit::Spade),
                    card(Value::Seven, Suit::Spade),
                    card(Value::Three, Suit::Heart),
                ],
            ),
            (
                [card(Value::Ace, Suit::Club), card(Value::Two, Suit::Club)],
                [
                    card(Value::King, Suit::Spade),
                    card(Value::Seven, Suit::Spade),
                    card(Value::Three, Suit::Heart),
                ],
            ),
            (
                [card(Value::Ace, Suit::Spade), card(Value::Two, Suit::Spade)],
                [
                    card(Value::King, Suit::Heart),
                    card(Value::Seven, Suit::Heart),
                    card(Value::Three, Suit::Diamond),
                ],
            ),
        ];

        for (hole, flop) in no_draw_cases {
            assert_eq!(
                flop_bucket_for(hole, flop, &all_buckets),
                3,
                "A2s without the two-tone board suit should canonicalize to the no-draw bucket"
            );
        }
    }

    // -- apply_dcfr_discount tests --

    #[timed_test]
    fn dcfr_discount_reduces_positive_regrets() {
        let tree = minimal_tree(2);
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = MpStorage::new(&tree, bucket_counts);
        // Set a positive regret
        storage.add_regret(first_decision_node(&tree), 0, 0, 1000);
        let config = toy_training_config(1000);

        apply_dcfr_discount(&storage, 100, &config);

        let after = storage.get_regret(first_decision_node(&tree), 0, 0);
        assert!(
            after < 1000,
            "positive regret should be discounted, got {after}"
        );
        assert!(
            after > 0,
            "positive regret should stay positive, got {after}"
        );
    }

    #[timed_test]
    fn dcfr_discount_reduces_strategy_sums() {
        let tree = minimal_tree(2);
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = MpStorage::new(&tree, bucket_counts);
        let node = first_decision_node(&tree);
        storage.add_strategy_sum(node, 0, 0, 10_000);
        let config = toy_training_config(1000);

        apply_dcfr_discount(&storage, 100, &config);

        let after = storage.get_strategy_sum(node, 0, 0);
        assert!(
            after < 10_000,
            "strategy sum should be discounted, got {after}"
        );
        assert!(after > 0, "strategy sum should stay positive, got {after}");
    }

    #[timed_test]
    fn dcfr_discount_preserves_u64_strategy_sums_above_i32_max() {
        let tree = minimal_tree(2);
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = MpStorage::new(&tree, bucket_counts);
        let node = first_decision_node(&tree);
        storage.add_strategy_sum(node, 0, 0, i32::MAX);
        storage.add_strategy_sum(node, 0, 0, i32::MAX);
        let before = storage.get_strategy_sum(node, 0, 0);
        let config = toy_training_config(1000);

        apply_dcfr_discount(&storage, 50_000, &config);

        let after = storage.get_strategy_sum(node, 0, 0);
        assert!(before > i32::MAX as u64, "setup should exceed i32::MAX");
        assert!(after < before, "strategy sum should be discounted");
        assert!(
            after > i32::MAX as u64,
            "large u64 strategy sum should remain above i32::MAX, got {after}"
        );
    }

    #[timed_test(2)]
    fn dcfr_discount_handles_negative_regrets() {
        let tree = minimal_tree(2);
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = MpStorage::new(&tree, bucket_counts);
        let node = first_decision_node(&tree);
        storage.add_regret(node, 0, 0, -500);
        let config = toy_training_config(1000);

        apply_dcfr_discount(&storage, 100, &config);

        let after = storage.get_regret(node, 0, 0);
        assert!(
            after > -500,
            "negative regret should be discounted toward zero"
        );
        assert!(after <= 0, "negative regret should stay non-positive");
    }

    // -- run_batch tests --

    #[timed_test]
    fn run_batch_updates_regrets() {
        let tree = minimal_tree(2);
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = MpStorage::new(&tree, bucket_counts);

        let ab = test_all_buckets(bucket_counts);
        run_batch(
            &tree,
            &storage,
            &ab,
            2,
            bucket_counts,
            0.0,
            Chips::ZERO,
            20,
            0,
            false,
            0,
        );

        let any_nonzero = storage
            .regrets
            .iter()
            .any(|r| r.load(Ordering::Relaxed) != 0);
        assert!(any_nonzero, "run_batch should produce non-zero regrets");
    }

    #[timed_test]
    fn run_batch_updates_strategy_sums() {
        let tree = minimal_tree(2);
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = MpStorage::new(&tree, bucket_counts);

        let ab = test_all_buckets(bucket_counts);
        run_batch(
            &tree,
            &storage,
            &ab,
            2,
            bucket_counts,
            0.0,
            Chips::ZERO,
            20,
            0,
            false,
            0,
        );

        let any_nonzero = storage
            .strategy_sums
            .iter()
            .any(|s| s.load(Ordering::Relaxed) != 0);
        assert!(
            any_nonzero,
            "run_batch should produce non-zero strategy sums"
        );
    }

    #[timed_test]
    fn run_batch_3_player_updates_storage() {
        let tree = minimal_tree(3);
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = MpStorage::new(&tree, bucket_counts);

        let ab = test_all_buckets(bucket_counts);
        run_batch(
            &tree,
            &storage,
            &ab,
            3,
            bucket_counts,
            0.0,
            Chips::ZERO,
            10,
            0,
            false,
            0,
        );

        let any_nonzero = storage
            .regrets
            .iter()
            .any(|r| r.load(Ordering::Relaxed) != 0);
        assert!(
            any_nonzero,
            "3-player run_batch should produce non-zero regrets"
        );
    }

    // -- iteration count edge cases --

    #[timed_test]
    fn train_zero_iterations_returns_zero() {
        let config = toy_config(2, 0);
        let result = train_blueprint_mp(&config);
        assert_eq!(result.meta_iterations, 0);
    }

    #[timed_test]
    fn train_batch_aligns_to_iteration_limit() {
        // batch_size=10, iterations=25 => should cap at 20 or 30 depending on rounding
        let mut config = toy_config(2, 25);
        config.training.batch_size = 10;
        let result = train_blueprint_mp(&config);
        // With batch_size=10 and max=25: batches of 10,10,5 = 25
        assert_eq!(result.meta_iterations, 25);
    }

    // -- setup_training tests --

    #[timed_test]
    fn setup_training_builds_nonempty_tree() {
        let config = toy_config(2, 100);
        let ctx = setup_training(&config);
        assert!(!ctx.tree.nodes.is_empty(), "tree should have nodes");
    }

    #[timed_test]
    fn setup_training_populates_bucket_counts() {
        let config = toy_config(3, 50);
        let ctx = setup_training(&config);
        assert_eq!(ctx.bucket_counts, [10, 10, 10, 10]);
    }

    #[timed_test]
    fn setup_training_sets_num_players() {
        let config = toy_config(3, 50);
        let ctx = setup_training(&config);
        assert_eq!(ctx.num_players, 3);
    }

    #[timed_test]
    fn setup_training_iterations_start_at_zero() {
        let config = toy_config(2, 100);
        let ctx = setup_training(&config);
        assert_eq!(ctx.iterations.load(Ordering::Relaxed), 0);
    }

    // -- run_training tests --

    #[timed_test(3)]
    fn run_training_returns_correct_meta_iterations() {
        let config = toy_config(2, 50);
        let ctx = setup_training(&config);
        let result = run_training(&ctx, &config.training, &config.game);
        assert_eq!(result.meta_iterations, 50);
    }

    #[timed_test(3)]
    fn run_training_updates_shared_iterations() {
        let config = toy_config(2, 30);
        let ctx = setup_training(&config);
        let result = run_training(&ctx, &config.training, &config.game);
        let shared = ctx.iterations.load(Ordering::Relaxed);
        assert_eq!(shared, result.meta_iterations);
    }

    #[timed_test]
    fn run_training_zero_iterations() {
        let config = toy_config(2, 0);
        let ctx = setup_training(&config);
        let result = run_training(&ctx, &config.training, &config.game);
        assert_eq!(result.meta_iterations, 0);
        assert_eq!(ctx.iterations.load(Ordering::Relaxed), 0);
    }

    // -- Helper --

    fn test_all_buckets(bucket_counts: [u16; 4]) -> AllBuckets {
        let mut ab = AllBuckets::new(bucket_counts, [None, None, None, None]);
        ab.equity_fallback = true;
        ab
    }

    fn a2_texture_test_buckets() -> AllBuckets {
        let canonical_flop = canonical_a2_texture_flop();
        let mut buckets = vec![0_u16; 1326];

        set_a2_texture_bucket(
            &mut buckets,
            &canonical_flop,
            [card(Value::Ace, Suit::Spade), card(Value::Two, Suit::Spade)],
            7,
        );
        for suit in [Suit::Heart, Suit::Diamond, Suit::Club] {
            set_a2_texture_bucket(
                &mut buckets,
                &canonical_flop,
                [card(Value::Ace, suit), card(Value::Two, suit)],
                3,
            );
        }

        let bucket_file = BucketFile {
            header: BucketFileHeader {
                street: V2Street::Flop,
                bucket_count: 10,
                board_count: 1,
                combos_per_board: 1326,
                version: bucket_file::VERSION,
            },
            boards: vec![canonical_key(&canonical_flop.cards)],
            buckets,
        };
        AllBuckets::new([169, 10, 10, 10], [None, Some(bucket_file), None, None])
    }

    fn canonical_a2_texture_flop() -> CanonicalBoard {
        CanonicalBoard::from_cards(&[
            card(Value::King, Suit::Spade),
            card(Value::Seven, Suit::Spade),
            card(Value::Three, Suit::Heart),
        ])
        .expect("valid flop")
    }

    fn set_a2_texture_bucket(
        buckets: &mut [u16],
        canonical_flop: &CanonicalBoard,
        hole: [Card; 2],
        bucket: u16,
    ) {
        let (h0, h1) = canonical_flop.canonicalize_holding(hole[0], hole[1]);
        buckets[combo_index(h0, h1) as usize] = bucket;
    }

    fn flop_bucket_for(hole: [Card; 2], flop: [Card; 3], all_buckets: &AllBuckets) -> u16 {
        let deal = Deal {
            hole_cards: {
                let mut hole_cards = [[card(Value::Two, Suit::Club); 2]; MAX_PLAYERS];
                hole_cards[0] = hole;
                hole_cards[1] = [
                    card(Value::Queen, Suit::Heart),
                    card(Value::Jack, Suit::Diamond),
                ];
                hole_cards[2] = [card(Value::Ten, Suit::Club); 2];
                hole_cards[3] = [card(Value::Ten, Suit::Diamond); 2];
                hole_cards[4] = [card(Value::Nine, Suit::Club); 2];
                hole_cards[5] = [card(Value::Nine, Suit::Diamond); 2];
                hole_cards[6] = [card(Value::Eight, Suit::Club); 2];
                hole_cards[7] = [card(Value::Eight, Suit::Diamond); 2];
                hole_cards
            },
            board: [
                flop[0],
                flop[1],
                flop[2],
                card(Value::Six, Suit::Club),
                card(Value::Five, Suit::Diamond),
            ],
            num_players: 2,
        };
        compute_deal_buckets(&deal, all_buckets, [169, 10, 10, 10]).buckets[0][Street::Flop.index()]
            .0
    }

    fn card(value: Value, suit: Suit) -> Card {
        Card::new(value, suit)
    }

    fn first_decision_node(tree: &MpGameTree) -> u32 {
        tree.nodes
            .iter()
            .position(|n| {
                matches!(
                    n,
                    crate::blueprint_mp::game_tree::MpGameNode::Decision { .. }
                )
            })
            .expect("tree should have a decision node") as u32
    }
}
