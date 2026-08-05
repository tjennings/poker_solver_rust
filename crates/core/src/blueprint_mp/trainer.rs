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

use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

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
const SERIAL_DCFR_DISCOUNT_SLOTS: usize = 4_096;
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

/// Thread-safe destination for infrequent multiplayer training status events.
///
/// Runners own and install the sink. Core training never uses a global callback,
/// and preserves detailed stderr logging when no sink is present.
pub trait MpTrainingEventSink: Send + Sync {
    fn publish(&self, event: MpTrainingEvent);
}

/// Shared runner-owned multiplayer training event sink.
pub type SharedMpTrainingEventSink = Arc<dyn MpTrainingEventSink>;

/// Typed multiplayer training event suitable for terminal or UI presentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MpTrainingEvent {
    DcfrSchedule(MpDcfrScheduleEvent),
    DcfrPass(MpDcfrPassEvent),
}

impl MpTrainingEvent {
    /// Whether this status should remain visible instead of expiring.
    #[must_use]
    pub const fn is_durable(&self) -> bool {
        matches!(self, Self::DcfrPass(event) if event.max_reached)
    }
}

/// DCFR schedule selected for an MP training runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MpDcfrScheduleEvent {
    pub mode: MpDcfrScheduleMode,
    pub warmup_meta_iterations: u64,
    pub max_passes: Option<NonZeroU64>,
}

/// Unit used to schedule MP DCFR passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpDcfrScheduleMode {
    Iterations { interval: u64 },
    WallClock { interval_seconds: NonZeroU64 },
}

/// Completed MP DCFR discount pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MpDcfrPassEvent {
    pub completed_passes: u64,
    pub max_passes: Option<NonZeroU64>,
    pub epoch: u64,
    pub meta_iteration: u64,
    pub skipped_slots: u64,
    pub sweep_duration: Duration,
    pub purge_duration: Option<Duration>,
    pub max_reached: bool,
}

/// Format an MP training event for a compact single-line status display.
#[must_use]
pub fn format_mp_training_event_compact(event: &MpTrainingEvent) -> String {
    match event {
        MpTrainingEvent::DcfrSchedule(event) => {
            let cadence = match event.mode {
                MpDcfrScheduleMode::Iterations { interval } => {
                    format!("every {interval} iterations")
                }
                MpDcfrScheduleMode::WallClock { interval_seconds } => {
                    format!("every {}s", interval_seconds.get())
                }
            };
            let max = event
                .max_passes
                .map_or_else(|| "unlimited".to_string(), |max| max.get().to_string());
            format!(
                "DCFR schedule: {cadence} after warmup {} | max {max}",
                event.warmup_meta_iterations
            )
        }
        MpTrainingEvent::DcfrPass(event) => {
            let pass = event.max_passes.map_or_else(
                || event.completed_passes.to_string(),
                |max| format!("{}/{}", event.completed_passes, max.get()),
            );
            let sweep = format_duration_status(event.sweep_duration);
            let mut text = format!("DCFR pass {pass} | epoch {} | sweep {sweep}", event.epoch);
            if event.skipped_slots > 0 {
                text.push_str(&format!(" | skipped {}", event.skipped_slots));
            }
            if event.max_reached {
                text.push_str(" | cap reached; stopped");
            }
            text
        }
    }
}

fn format_duration_status(duration: Duration) -> String {
    if duration < Duration::from_secs(1) {
        format!("{:.1}ms", duration.as_secs_f64() * 1_000.0)
    } else {
        format!("{:.3}s", duration.as_secs_f64())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiscountMode {
    Iterations,
    WallClock,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiscountBoundary {
    MetaIteration(u64),
    Elapsed(Duration),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiscountLateness {
    MetaIterations(u64),
    WallClock(Duration),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingDiscount {
    mode: DiscountMode,
    epoch: u64,
    interval: u64,
    boundary: DiscountBoundary,
    lateness: DiscountLateness,
    skipped_slots: u64,
    observed_elapsed: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CompletedDiscount {
    pending: PendingDiscount,
    skipped_slots: u64,
    completed_passes: u64,
    max_passes: Option<NonZeroU64>,
    max_reached: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiscountScheduleState {
    Iterations {
        interval: u64,
        next_boundary: Option<u64>,
    },
    WallClock {
        interval_seconds: NonZeroU64,
        armed: bool,
        next_deadline: Option<Duration>,
        next_epoch: u64,
    },
}

/// Pure DCFR scheduling state shared by eager and lazy training.
///
/// Callers supply completed meta-iterations and monotonic process elapsed time.
/// This keeps tests deterministic and leaves `Instant` ownership in production
/// runners. A due pass is completed only after its table sweep succeeds, so the
/// wall-clock schedule can advance beyond deadlines crossed during the sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DiscountScheduler {
    warmup_iterations: u64,
    completed_passes: u64,
    max_passes: Option<NonZeroU64>,
    state: DiscountScheduleState,
}

impl DiscountScheduler {
    fn new(config: &MpTrainingConfig, start_meta_iter: u64) -> Self {
        let state = config.dcfr_discount_interval_seconds.map_or_else(
            || {
                let interval = config.lcfr_discount_interval.max(1);
                DiscountScheduleState::Iterations {
                    interval,
                    next_boundary: first_iteration_boundary(
                        start_meta_iter,
                        config.lcfr_warmup_iterations,
                        interval,
                    ),
                }
            },
            |interval_seconds| {
                let armed = start_meta_iter >= config.lcfr_warmup_iterations;
                let next_deadline = armed.then(|| Duration::from_secs(interval_seconds.get()));
                DiscountScheduleState::WallClock {
                    interval_seconds,
                    armed,
                    next_deadline,
                    next_epoch: 1,
                }
            },
        );
        Self {
            warmup_iterations: config.lcfr_warmup_iterations,
            completed_passes: 0,
            max_passes: config.dcfr_discount_max_passes,
            state,
        }
    }

    fn pending(&mut self, meta_iter: u64, elapsed: Duration) -> Option<PendingDiscount> {
        if self
            .max_passes
            .is_some_and(|max| self.completed_passes >= max.get())
        {
            return None;
        }
        match &mut self.state {
            DiscountScheduleState::Iterations {
                interval,
                next_boundary,
            } => {
                let boundary = (*next_boundary)?;
                if meta_iter < boundary {
                    return None;
                }
                let skipped_slots = meta_iter.saturating_sub(boundary) / *interval;
                let scheduled = boundary.checked_add(interval.checked_mul(skipped_slots)?)?;
                Some(PendingDiscount {
                    mode: DiscountMode::Iterations,
                    epoch: scheduled / *interval,
                    interval: *interval,
                    boundary: DiscountBoundary::MetaIteration(scheduled),
                    lateness: DiscountLateness::MetaIterations(meta_iter.saturating_sub(scheduled)),
                    skipped_slots,
                    observed_elapsed: elapsed,
                })
            }
            DiscountScheduleState::WallClock {
                interval_seconds,
                armed,
                next_deadline,
                next_epoch,
            } => {
                if !*armed {
                    if meta_iter < self.warmup_iterations {
                        return None;
                    }
                    *armed = true;
                    *next_deadline =
                        elapsed.checked_add(Duration::from_secs(interval_seconds.get()));
                    return None;
                }

                let deadline = (*next_deadline)?;
                if elapsed < deadline {
                    return None;
                }
                let skipped_slots = elapsed
                    .saturating_sub(deadline)
                    .as_nanos()
                    .checked_div(interval_nanos(*interval_seconds))
                    .and_then(|slots| u64::try_from(slots).ok())?;
                let scheduled =
                    checked_add_interval_slots(deadline, *interval_seconds, skipped_slots)?;
                Some(PendingDiscount {
                    mode: DiscountMode::WallClock,
                    epoch: *next_epoch,
                    interval: interval_seconds.get(),
                    boundary: DiscountBoundary::Elapsed(scheduled),
                    lateness: DiscountLateness::WallClock(elapsed.saturating_sub(scheduled)),
                    skipped_slots,
                    observed_elapsed: elapsed,
                })
            }
        }
    }

    fn complete(
        &mut self,
        pending: PendingDiscount,
        meta_iter: u64,
        fresh_elapsed: Duration,
    ) -> CompletedDiscount {
        let additional_skipped = match (&mut self.state, pending.boundary) {
            (
                DiscountScheduleState::Iterations {
                    interval,
                    next_boundary,
                },
                DiscountBoundary::MetaIteration(scheduled),
            ) => {
                let crossed = meta_iter.saturating_sub(scheduled) / *interval;
                *next_boundary = scheduled
                    .checked_add(interval.saturating_mul(crossed))
                    .and_then(|last| last.checked_add(*interval));
                crossed
            }
            (
                DiscountScheduleState::WallClock {
                    interval_seconds,
                    next_deadline,
                    next_epoch,
                    ..
                },
                DiscountBoundary::Elapsed(scheduled),
            ) => {
                let crossed = fresh_elapsed
                    .saturating_sub(scheduled)
                    .as_nanos()
                    .checked_div(interval_nanos(*interval_seconds))
                    .and_then(|slots| u64::try_from(slots).ok())
                    .unwrap_or(u64::MAX);
                *next_deadline = crossed.checked_add(1).and_then(|slots| {
                    checked_add_interval_slots(scheduled, *interval_seconds, slots)
                });
                *next_epoch = next_epoch.checked_add(1).unwrap_or(u64::MAX);
                crossed
            }
            _ => unreachable!("discount completion must match its scheduler mode"),
        };

        self.completed_passes = self.completed_passes.saturating_add(1);
        let max_reached = self
            .max_passes
            .is_some_and(|max| self.completed_passes >= max.get());

        CompletedDiscount {
            pending,
            skipped_slots: pending.skipped_slots.saturating_add(additional_skipped),
            completed_passes: self.completed_passes,
            max_passes: self.max_passes,
            max_reached,
        }
    }
}

fn first_iteration_boundary(
    start_meta_iter: u64,
    warmup_iterations: u64,
    interval: u64,
) -> Option<u64> {
    let first_after_start = start_meta_iter
        .checked_div(interval)?
        .checked_add(1)?
        .checked_mul(interval)?;
    let first_at_or_after_warmup = if warmup_iterations == 0 {
        interval
    } else {
        warmup_iterations
            .checked_add(interval - 1)?
            .checked_div(interval)?
            .checked_mul(interval)?
    };
    Some(first_after_start.max(first_at_or_after_warmup))
}

fn interval_nanos(interval_seconds: NonZeroU64) -> u128 {
    u128::from(interval_seconds.get()) * 1_000_000_000
}

fn checked_add_interval_slots(
    base: Duration,
    interval_seconds: NonZeroU64,
    slots: u64,
) -> Option<Duration> {
    let seconds = interval_seconds.get().checked_mul(slots)?;
    base.checked_add(Duration::from_secs(seconds))
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
    run_training_with_event_sink(ctx, training, game, None)
}

/// Run eager training with an optional runner-owned status-event sink.
#[must_use]
pub fn run_training_with_event_sink(
    ctx: &TrainContext,
    training: &MpTrainingConfig,
    game: &MpGameConfig,
    event_sink: Option<SharedMpTrainingEventSink>,
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
        event_sink.as_deref(),
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
    event_sink: Option<&dyn MpTrainingEventSink>,
) -> TrainResult {
    let max_iters = config.iterations.unwrap_or(u64::MAX);
    let scaled_threshold = (f64::from(config.prune_threshold) * REGRET_SCALE)
        .clamp(f64::from(i32::MIN), f64::from(i32::MAX))
        .round() as i32;
    let mut meta_iter: u64 = 0;
    let mut rng = SmallRng::seed_from_u64(0xDEAD_BEEF_CAFE_1234);
    let discount_clock_start = Instant::now();
    let mut discount_scheduler = DiscountScheduler::new(config, meta_iter);
    emit_discount_schedule(config, event_sink);

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

        if let Some(pending) = discount_scheduler.pending(meta_iter, discount_clock_start.elapsed())
        {
            let discount_started = Instant::now();
            apply_dcfr_discount(storage, pending.epoch, config);
            let sweep_duration = discount_started.elapsed();
            let completed =
                discount_scheduler.complete(pending, meta_iter, discount_clock_start.elapsed());
            emit_discount_pass(
                completed,
                meta_iter,
                config,
                sweep_duration,
                None,
                event_sink,
            );
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
    discount_clock_start: Instant,
    discount_scheduler: DiscountScheduler,
    event_sink: Option<SharedMpTrainingEventSink>,
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
        Self::from_context_with_event_sink(ctx, training, game, start_meta_iter, None)
    }

    /// Build a stepper with an optional runner-owned status-event sink.
    #[must_use]
    pub fn from_context_with_event_sink(
        ctx: &LazyTrainContext,
        training: &MpTrainingConfig,
        game: &MpGameConfig,
        start_meta_iter: u64,
        event_sink: Option<SharedMpTrainingEventSink>,
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
            event_sink,
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
        event_sink: Option<SharedMpTrainingEventSink>,
    ) -> Self {
        let scaled_threshold = (f64::from(training.prune_threshold) * REGRET_SCALE)
            .clamp(f64::from(i32::MIN), f64::from(i32::MAX))
            .round() as i32;
        let rng = pruning_rng_at_meta_iter(start_meta_iter, &training);
        let discount_clock_start = Instant::now();
        let discount_scheduler = DiscountScheduler::new(&training, start_meta_iter);
        emit_discount_schedule(&training, event_sink.as_deref());
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
            discount_clock_start,
            discount_scheduler,
            event_sink,
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

        if let Some(pending) = self
            .discount_scheduler
            .pending(self.meta_iter, self.discount_clock_start.elapsed())
        {
            let discount_started = Instant::now();
            apply_dcfr_discount_lazy(&self.storage, pending.epoch, &self.training);
            let sweep_duration = discount_started.elapsed();
            LAZY_DISCOUNT_NANOS.fetch_add(
                u64::try_from(sweep_duration.as_nanos()).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
            let purge_started = Instant::now();
            purge_negative_action_subtrees_after_discount(&self.storage, &self.training);
            let purge_duration = purge_started.elapsed();
            let completed = self.discount_scheduler.complete(
                pending,
                self.meta_iter,
                self.discount_clock_start.elapsed(),
            );
            emit_discount_pass(
                completed,
                self.meta_iter,
                &self.training,
                sweep_duration,
                Some(purge_duration),
                self.event_sink.as_deref(),
            );
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

fn emit_discount_schedule(config: &MpTrainingConfig, event_sink: Option<&dyn MpTrainingEventSink>) {
    let event = MpTrainingEvent::DcfrSchedule(MpDcfrScheduleEvent {
        mode: config.dcfr_discount_interval_seconds.map_or(
            MpDcfrScheduleMode::Iterations {
                interval: config.lcfr_discount_interval.max(1),
            },
            |interval_seconds| MpDcfrScheduleMode::WallClock { interval_seconds },
        ),
        warmup_meta_iterations: config.lcfr_warmup_iterations,
        max_passes: config.dcfr_discount_max_passes,
    });
    if let Some(event_sink) = event_sink {
        event_sink.publish(event);
        return;
    }

    let max_passes = config
        .dcfr_discount_max_passes
        .map_or_else(|| "unlimited".to_string(), |max| max.get().to_string());
    if let Some(interval) = config.dcfr_discount_interval_seconds {
        eprintln!(
            "blueprint_mp DCFR schedule: mode=wall_clock interval={}s warmup_meta_iterations={} max_passes={max_passes} overrides_lcfr_discount_interval={}",
            interval, config.lcfr_warmup_iterations, config.lcfr_discount_interval,
        );
    } else {
        eprintln!(
            "blueprint_mp DCFR schedule: mode=iterations interval={} warmup_meta_iterations={} max_passes={max_passes}",
            config.lcfr_discount_interval.max(1),
            config.lcfr_warmup_iterations,
        );
    }
}

fn emit_discount_pass(
    completed: CompletedDiscount,
    meta_iter: u64,
    config: &MpTrainingConfig,
    sweep_duration: Duration,
    purge_duration: Option<Duration>,
    event_sink: Option<&dyn MpTrainingEventSink>,
) {
    let pending = completed.pending;
    let event = MpTrainingEvent::DcfrPass(MpDcfrPassEvent {
        completed_passes: completed.completed_passes,
        max_passes: completed.max_passes,
        epoch: pending.epoch,
        meta_iteration: meta_iter,
        skipped_slots: completed.skipped_slots,
        sweep_duration,
        purge_duration,
        max_reached: completed.max_reached,
    });
    if let Some(event_sink) = event_sink {
        event_sink.publish(event);
        return;
    }

    let (d_pos, d_neg) =
        regret_discount_factors(pending.epoch, config.dcfr_alpha, config.dcfr_beta);
    let d_strat = strategy_discount_factor(pending.epoch, config.dcfr_gamma);
    let (mode, interval, scheduled, lateness) = match (pending.boundary, pending.lateness) {
        (DiscountBoundary::MetaIteration(boundary), DiscountLateness::MetaIterations(late)) => (
            "iterations",
            format!("{}meta_iterations", pending.interval),
            format!("meta_iteration:{boundary}"),
            format!("{late}meta_iterations"),
        ),
        (DiscountBoundary::Elapsed(boundary), DiscountLateness::WallClock(late)) => (
            "wall_clock",
            format!("{}s", pending.interval),
            format!("elapsed:{:.3}s", boundary.as_secs_f64()),
            format!("{:.3}s", late.as_secs_f64()),
        ),
        _ => unreachable!("discount boundary and lateness must use the same unit"),
    };
    let purge = purge_duration.map_or_else(
        || "n/a".to_string(),
        |duration| format!("{:.3}s", duration.as_secs_f64()),
    );
    let max_passes = completed
        .max_passes
        .map_or_else(|| "unlimited".to_string(), |max| max.get().to_string());
    eprintln!(
        "blueprint_mp DCFR pass: pass={}/{} mode={mode} interval={interval} scheduled={scheduled} execution_elapsed={:.3}s meta_iter={meta_iter} epoch={} d_pos={d_pos:.9} d_neg={d_neg:.9} d_strat={d_strat:.9} lateness={lateness} skipped_slots={} sweep={:.3}s purge={purge}",
        completed.completed_passes,
        max_passes,
        pending.observed_elapsed.as_secs_f64(),
        pending.epoch,
        completed.skipped_slots,
        sweep_duration.as_secs_f64(),
    );
    if completed.max_reached {
        eprintln!(
            "blueprint_mp DCFR maximum reached: completed_passes={} max_passes={}; future discount and lazy purge passes are disabled",
            completed.completed_passes, max_passes
        );
    }
}

fn apply_dcfr_discount(storage: &MpStorage, epoch: u64, config: &MpTrainingConfig) {
    let (d_pos, d_neg) = regret_discount_factors(epoch, config.dcfr_alpha, config.dcfr_beta);
    let d_strat = strategy_discount_factor(epoch, config.dcfr_gamma);

    if storage.regrets.len() <= SERIAL_DCFR_DISCOUNT_SLOTS {
        storage
            .regrets
            .iter()
            .for_each(|atom| discount_regret_atom(atom, d_pos, d_neg));
        storage
            .strategy_sums
            .iter()
            .for_each(|atom| discount_strategy_sum_atom(atom, d_strat));
        return;
    }

    storage
        .regrets
        .par_iter()
        .for_each(|atom| discount_regret_atom(atom, d_pos, d_neg));
    storage
        .strategy_sums
        .par_iter()
        .for_each(|atom| discount_strategy_sum_atom(atom, d_strat));
}

fn discount_regret_atom(atom: &AtomicI32, d_pos: f64, d_neg: f64) {
    let v = atom.load(Ordering::Relaxed);
    let d = if v >= 0 { d_pos } else { d_neg };
    let discounted = super::discount_signed_regret(v, d);
    atom.store(discounted, Ordering::Relaxed);
}

fn discount_strategy_sum_atom(atom: &AtomicU64, d_strat: f64) {
    let v = atom.load(Ordering::Relaxed);
    let discounted = ((v as f64) * d_strat).clamp(0.0, u64::MAX as f64) as u64;
    atom.store(discounted, Ordering::Relaxed);
}

fn apply_dcfr_discount_lazy(storage: &SparseMpStorage, epoch: u64, config: &MpTrainingConfig) {
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
    use std::sync::Mutex;
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
    use crate::blueprint_mp::game_tree::{MpGameNode, MpGameTree, TreeAction};
    use crate::blueprint_mp::mccfr::sample_deal;
    use crate::blueprint_mp::sparse_storage::MpInfosetKey;
    use crate::blueprint_mp::storage::MpStorage;
    use crate::blueprint_mp::types::Street;
    use crate::blueprint_v2::Street as V2Street;
    use crate::blueprint_v2::bucket_file::{BucketFile, BucketFileHeader};
    use crate::blueprint_v2::cluster_pipeline::{canonical_key, combo_index};
    use crate::poker::{Suit, Value};
    use crate::{abstraction::isomorphism::CanonicalBoard, blueprint_v2::bucket_file};

    #[derive(Default)]
    struct RecordingEventSink {
        events: Mutex<Vec<MpTrainingEvent>>,
    }

    impl MpTrainingEventSink for RecordingEventSink {
        fn publish(&self, event: MpTrainingEvent) {
            self.events.lock().unwrap().push(event);
        }
    }

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
            dcfr_discount_interval_seconds: None,
            dcfr_discount_max_passes: None,
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
            dcfr_discount_interval_seconds: None,
            dcfr_discount_max_passes: None,
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

    // -- discount scheduler tests --

    #[timed_test]
    fn compact_dcfr_pass_status_includes_required_fields_and_optional_state() {
        let event = MpTrainingEvent::DcfrPass(MpDcfrPassEvent {
            completed_passes: 40,
            max_passes: NonZeroU64::new(40),
            epoch: 77,
            meta_iteration: 12_345,
            skipped_slots: 2,
            sweep_duration: Duration::from_millis(1250),
            purge_duration: Some(Duration::from_millis(50)),
            max_reached: true,
        });

        assert_eq!(
            format_mp_training_event_compact(&event),
            "DCFR pass 40/40 | epoch 77 | sweep 1.250s | skipped 2 | cap reached; stopped"
        );
        assert!(event.is_durable());
    }

    #[timed_test]
    fn compact_dcfr_pass_status_omits_zero_skipped_and_unset_max() {
        let event = MpTrainingEvent::DcfrPass(MpDcfrPassEvent {
            completed_passes: 3,
            max_passes: None,
            epoch: 9,
            meta_iteration: 100,
            skipped_slots: 0,
            sweep_duration: Duration::from_millis(12),
            purge_duration: None,
            max_reached: false,
        });

        assert_eq!(
            format_mp_training_event_compact(&event),
            "DCFR pass 3 | epoch 9 | sweep 12.0ms"
        );
        assert!(!event.is_durable());
    }

    #[timed_test]
    fn eager_runner_routes_schedule_and_completed_pass_to_sink() {
        let mut config = toy_config(2, 1);
        config.training.batch_size = 1;
        config.training.lcfr_warmup_iterations = 0;
        config.training.lcfr_discount_interval = 1;
        config.training.dcfr_discount_max_passes = NonZeroU64::new(1);
        let ctx = setup_training(&config);
        let sink = Arc::new(RecordingEventSink::default());
        let shared_sink: SharedMpTrainingEventSink = sink.clone();

        let result =
            run_training_with_event_sink(&ctx, &config.training, &config.game, Some(shared_sink));

        assert_eq!(result.meta_iterations, 1);
        let events = sink.events.lock().unwrap();
        assert!(matches!(
            events.first(),
            Some(MpTrainingEvent::DcfrSchedule(_))
        ));
        assert!(matches!(
            events.get(1),
            Some(MpTrainingEvent::DcfrPass(MpDcfrPassEvent {
                completed_passes: 1,
                epoch: 1,
                max_reached: true,
                ..
            }))
        ));
        assert_eq!(events.len(), 2, "final pass must be a single durable event");
    }

    fn wall_clock_scheduler(warmup: u64, interval_seconds: u64) -> DiscountScheduler {
        wall_clock_scheduler_at(0, warmup, interval_seconds)
    }

    fn wall_clock_scheduler_at(
        start_meta_iter: u64,
        warmup: u64,
        interval_seconds: u64,
    ) -> DiscountScheduler {
        let mut config = toy_training_config(1_000);
        config.lcfr_warmup_iterations = warmup;
        config.dcfr_discount_interval_seconds = NonZeroU64::new(interval_seconds);
        DiscountScheduler::new(&config, start_meta_iter)
    }

    #[timed_test]
    fn wall_clock_discount_arms_at_start_when_warmup_is_zero() {
        let mut scheduler = wall_clock_scheduler(0, 10);

        assert!(scheduler.pending(0, Duration::from_secs(9)).is_none());
        let pending = scheduler
            .pending(0, Duration::from_secs(10))
            .expect("first deadline should be due exactly ten seconds after start");
        assert_eq!(pending.mode, DiscountMode::WallClock);
        assert_eq!(pending.epoch, 1);
        assert_eq!(
            pending.boundary,
            DiscountBoundary::Elapsed(Duration::from_secs(10))
        );
        assert_eq!(
            pending.lateness,
            DiscountLateness::WallClock(Duration::ZERO)
        );
        assert_eq!(pending.skipped_slots, 0);
    }

    #[timed_test]
    fn wall_clock_discount_arms_at_creation_when_start_equals_warmup() {
        let mut scheduler = wall_clock_scheduler_at(100, 100, 10);

        assert!(scheduler.pending(100, Duration::from_secs(9)).is_none());
        assert_eq!(
            scheduler
                .pending(100, Duration::from_secs(10))
                .expect("a stepper created at warmup should already be armed")
                .epoch,
            1
        );
    }

    #[timed_test]
    fn wall_clock_discount_arms_at_creation_when_start_exceeds_warmup() {
        let mut scheduler = wall_clock_scheduler_at(150, 100, 10);

        assert!(scheduler.pending(150, Duration::from_secs(9)).is_none());
        let pending = scheduler
            .pending(150, Duration::from_secs(10))
            .expect("a stepper created beyond warmup should already be armed");
        assert_eq!(pending.epoch, 1);
        assert_eq!(
            pending.boundary,
            DiscountBoundary::Elapsed(Duration::from_secs(10))
        );
    }

    #[timed_test]
    fn wall_clock_discount_arms_on_first_completed_batch_at_warmup() {
        let mut scheduler = wall_clock_scheduler(100, 10);

        assert!(scheduler.pending(99, Duration::from_secs(50)).is_none());
        assert!(scheduler.pending(100, Duration::from_secs(50)).is_none());
        assert!(scheduler.pending(110, Duration::from_secs(59)).is_none());
        let pending = scheduler
            .pending(110, Duration::from_secs(60))
            .expect("deadline should be relative to the warmup observation");
        assert_eq!(pending.epoch, 1);
        assert_eq!(
            pending.boundary,
            DiscountBoundary::Elapsed(Duration::from_secs(60))
        );
    }

    #[timed_test]
    fn wall_clock_discount_skips_missed_slots_without_catch_up() {
        let mut scheduler = wall_clock_scheduler(0, 10);
        let pending = scheduler
            .pending(1, Duration::from_secs(35))
            .expect("a late deadline should produce one pass");
        assert_eq!(pending.epoch, 1);
        assert_eq!(
            pending.boundary,
            DiscountBoundary::Elapsed(Duration::from_secs(30))
        );
        assert_eq!(
            pending.lateness,
            DiscountLateness::WallClock(Duration::from_secs(5))
        );
        assert_eq!(pending.skipped_slots, 2);

        let completed = scheduler.complete(pending, 1, Duration::from_secs(46));
        assert_eq!(completed.skipped_slots, 3);
        assert!(scheduler.pending(2, Duration::from_secs(49)).is_none());

        let next = scheduler
            .pending(2, Duration::from_secs(50))
            .expect("schedule should remain anchored at the first slot after fresh elapsed");
        assert_eq!(next.epoch, 2);
        assert_eq!(
            next.boundary,
            DiscountBoundary::Elapsed(Duration::from_secs(50))
        );
    }

    #[timed_test]
    fn wall_clock_discount_has_no_duplicate_pass_at_same_deadline() {
        let mut scheduler = wall_clock_scheduler(0, 10);
        let pending = scheduler
            .pending(0, Duration::from_secs(10))
            .expect("deadline should be due");
        let _ = scheduler.complete(pending, 0, Duration::from_secs(10));

        assert!(scheduler.pending(0, Duration::from_secs(10)).is_none());
    }

    #[timed_test]
    fn wall_clock_discount_executes_exactly_max_passes_then_stops() {
        let mut config = toy_training_config(1_000);
        config.lcfr_warmup_iterations = 0;
        config.dcfr_discount_interval_seconds = NonZeroU64::new(10);
        config.dcfr_discount_max_passes = NonZeroU64::new(2);
        let mut scheduler = DiscountScheduler::new(&config, 0);

        let first = scheduler
            .pending(1, Duration::from_secs(10))
            .expect("first pass should execute");
        let first_completed = scheduler.complete(first, 1, Duration::from_secs(10));
        assert_eq!(first_completed.completed_passes, 1);
        assert!(!first_completed.max_reached);

        let second = scheduler
            .pending(2, Duration::from_secs(20))
            .expect("final configured pass should execute");
        let second_completed = scheduler.complete(second, 2, Duration::from_secs(20));
        assert_eq!(second_completed.completed_passes, 2);
        assert!(second_completed.max_reached);

        assert!(scheduler.pending(3, Duration::from_secs(30)).is_none());
        assert_eq!(scheduler.completed_passes, 2);
        assert!(matches!(
            scheduler.state,
            DiscountScheduleState::WallClock {
                next_deadline: Some(deadline),
                ..
            } if deadline == Duration::from_secs(30)
        ));
    }

    #[timed_test]
    fn wall_clock_config_overrides_iteration_schedule() {
        let mut config = toy_training_config(1_000);
        config.lcfr_warmup_iterations = 0;
        config.lcfr_discount_interval = 1;
        config.dcfr_discount_interval_seconds = NonZeroU64::new(10);
        let mut scheduler = DiscountScheduler::new(&config, 0);

        assert!(scheduler.pending(500, Duration::from_secs(9)).is_none());
        assert_eq!(
            scheduler
                .pending(500, Duration::from_secs(10))
                .expect("wall-clock deadline should override iteration cadence")
                .mode,
            DiscountMode::WallClock
        );
    }

    #[timed_test]
    fn iteration_discount_triggers_on_nonaligned_boundary_crossing() {
        let mut config = toy_training_config(1_000);
        config.lcfr_warmup_iterations = 0;
        config.lcfr_discount_interval = 50;
        let mut scheduler = DiscountScheduler::new(&config, 0);

        assert!(scheduler.pending(40, Duration::ZERO).is_none());
        let pending = scheduler
            .pending(60, Duration::ZERO)
            .expect("crossing iteration 50 should trigger even without exact divisibility");
        assert_eq!(pending.mode, DiscountMode::Iterations);
        assert_eq!(pending.epoch, 1);
        assert_eq!(pending.boundary, DiscountBoundary::MetaIteration(50));
        assert_eq!(pending.lateness, DiscountLateness::MetaIterations(10));
        let _ = scheduler.complete(pending, 60, Duration::ZERO);
        assert!(scheduler.pending(60, Duration::ZERO).is_none());
    }

    #[timed_test]
    fn iteration_discount_preserves_aligned_boundary_epoch() {
        let mut config = toy_training_config(1_000);
        config.lcfr_warmup_iterations = 100;
        config.lcfr_discount_interval = 50;
        let mut scheduler = DiscountScheduler::new(&config, 0);

        let pending = scheduler
            .pending(100, Duration::ZERO)
            .expect("aligned warmup boundary should retain legacy behavior");
        assert_eq!(pending.epoch, 2);
        assert_eq!(pending.boundary, DiscountBoundary::MetaIteration(100));
        assert_eq!(pending.skipped_slots, 0);
    }

    #[timed_test]
    fn iteration_discount_skips_crossed_boundaries_without_catch_up() {
        let mut config = toy_training_config(1_000);
        config.lcfr_warmup_iterations = 0;
        config.lcfr_discount_interval = 50;
        let mut scheduler = DiscountScheduler::new(&config, 0);

        let pending = scheduler
            .pending(160, Duration::ZERO)
            .expect("crossing multiple boundaries should produce one pass");
        assert_eq!(pending.epoch, 3);
        assert_eq!(pending.boundary, DiscountBoundary::MetaIteration(150));
        assert_eq!(pending.skipped_slots, 2);
        let completed = scheduler.complete(pending, 160, Duration::ZERO);
        assert_eq!(completed.skipped_slots, 2);
        assert!(scheduler.pending(199, Duration::ZERO).is_none());
        assert_eq!(
            scheduler
                .pending(200, Duration::ZERO)
                .expect("next anchored boundary should be due")
                .epoch,
            4
        );
    }

    #[timed_test]
    fn iteration_discount_pass_cap_is_distinct_from_high_factor_epoch() {
        let mut config = toy_training_config(20_000);
        config.lcfr_warmup_iterations = 10_000;
        config.lcfr_discount_interval = 50;
        config.dcfr_discount_max_passes = NonZeroU64::new(1);
        let mut scheduler = DiscountScheduler::new(&config, 9_900);

        let pending = scheduler
            .pending(10_000, Duration::ZERO)
            .expect("first process-local pass should execute at the warmup boundary");
        assert_eq!(pending.epoch, 200);
        let completed = scheduler.complete(pending, 10_000, Duration::ZERO);
        assert_eq!(completed.completed_passes, 1);
        assert!(completed.max_reached);
        assert!(scheduler.pending(10_050, Duration::ZERO).is_none());
    }

    #[timed_test]
    fn iteration_discount_executes_exactly_max_passes_then_stops() {
        let mut config = toy_training_config(1_000);
        config.lcfr_warmup_iterations = 0;
        config.lcfr_discount_interval = 10;
        config.dcfr_discount_max_passes = NonZeroU64::new(2);
        let mut scheduler = DiscountScheduler::new(&config, 0);

        for pass in 1..=2 {
            let meta_iter = pass * 10;
            let pending = scheduler
                .pending(meta_iter, Duration::ZERO)
                .expect("configured pass should execute");
            assert_eq!(
                scheduler
                    .complete(pending, meta_iter, Duration::ZERO)
                    .completed_passes,
                pass
            );
        }

        assert!(scheduler.pending(30, Duration::ZERO).is_none());
    }

    #[timed_test]
    fn eager_values_remain_unchanged_after_discount_pass_cap() {
        let mut config = toy_training_config(1_000);
        config.lcfr_warmup_iterations = 0;
        config.lcfr_discount_interval = 1;
        config.dcfr_discount_max_passes = NonZeroU64::new(1);
        let tree = minimal_tree(2);
        let storage = MpStorage::new(&tree, [10; 4]);
        let node = first_decision_node(&tree);
        storage.add_regret(node, 0, 0, 1_000);
        let mut scheduler = DiscountScheduler::new(&config, 0);

        let first = scheduler
            .pending(1, Duration::ZERO)
            .expect("first discount should be due");
        apply_dcfr_discount(&storage, first.epoch, &config);
        let _ = scheduler.complete(first, 1, Duration::ZERO);
        let capped_value = storage.get_regret(node, 0, 0);

        if let Some(unexpected) = scheduler.pending(2, Duration::ZERO) {
            apply_dcfr_discount(&storage, unexpected.epoch, &config);
            let _ = scheduler.complete(unexpected, 2, Duration::ZERO);
        }
        assert_eq!(storage.get_regret(node, 0, 0), capped_value);
    }

    #[timed_test]
    fn skipped_schedule_slots_consume_only_one_pass() {
        let mut config = toy_training_config(1_000);
        config.lcfr_warmup_iterations = 0;
        config.dcfr_discount_interval_seconds = NonZeroU64::new(10);
        config.dcfr_discount_max_passes = NonZeroU64::new(2);
        let mut scheduler = DiscountScheduler::new(&config, 0);

        let late = scheduler
            .pending(1, Duration::from_secs(35))
            .expect("late observation should execute one pass");
        assert_eq!(late.skipped_slots, 2);
        let completed = scheduler.complete(late, 1, Duration::from_secs(46));
        assert_eq!(completed.skipped_slots, 3);
        assert_eq!(completed.completed_passes, 1);

        let final_pass = scheduler
            .pending(2, Duration::from_secs(50))
            .expect("one pass should remain despite skipped slots");
        assert_eq!(
            scheduler
                .complete(final_pass, 2, Duration::from_secs(50))
                .completed_passes,
            2
        );
        assert!(scheduler.pending(3, Duration::from_secs(100)).is_none());
    }

    #[timed_test]
    fn preseeded_iteration_start_does_not_preconsume_pass_cap() {
        let mut config = toy_training_config(20_000);
        config.lcfr_warmup_iterations = 0;
        config.lcfr_discount_interval = 50;
        config.dcfr_discount_max_passes = NonZeroU64::new(1);
        let mut scheduler = DiscountScheduler::new(&config, 10_000);

        assert_eq!(scheduler.completed_passes, 0);
        let pending = scheduler
            .pending(10_050, Duration::ZERO)
            .expect("preseeded progress must not consume a process-local pass");
        assert_eq!(pending.epoch, 201);
        assert_eq!(
            scheduler
                .complete(pending, 10_050, Duration::ZERO)
                .completed_passes,
            1
        );
    }

    #[timed_test]
    fn unlimited_discount_schedule_remains_unbounded() {
        let mut config = toy_training_config(1_000);
        config.lcfr_warmup_iterations = 0;
        config.lcfr_discount_interval = 10;
        let mut scheduler = DiscountScheduler::new(&config, 0);

        for pass in 1..=4 {
            let meta_iter = pass * 10;
            let pending = scheduler
                .pending(meta_iter, Duration::ZERO)
                .expect("missing cap must preserve unlimited scheduling");
            let completed = scheduler.complete(pending, meta_iter, Duration::ZERO);
            assert_eq!(completed.completed_passes, pass);
            assert_eq!(completed.max_passes, None);
            assert!(!completed.max_reached);
        }
    }

    #[timed_test]
    fn eager_and_lazy_use_identical_scheduler_decisions() {
        let mut config = toy_training_config(1_000);
        config.lcfr_warmup_iterations = 25;
        config.dcfr_discount_interval_seconds = NonZeroU64::new(10);
        config.dcfr_discount_max_passes = NonZeroU64::new(2);
        let mut eager = DiscountScheduler::new(&config, 0);
        let mut lazy = DiscountScheduler::new(&config, 0);

        for (meta_iter, elapsed) in [
            (20, Duration::from_secs(2)),
            (30, Duration::from_secs(4)),
            (40, Duration::from_secs(13)),
            (50, Duration::from_secs(14)),
            (60, Duration::from_secs(25)),
        ] {
            let eager_pending = eager.pending(meta_iter, elapsed);
            let lazy_pending = lazy.pending(meta_iter, elapsed);
            assert_eq!(eager_pending, lazy_pending);
            if let (Some(eager_pending), Some(lazy_pending)) = (eager_pending, lazy_pending) {
                assert_eq!(
                    eager.complete(eager_pending, meta_iter, elapsed),
                    lazy.complete(lazy_pending, meta_iter, elapsed)
                );
            }
        }

        assert_eq!(eager.completed_passes, 2);
        assert_eq!(lazy.completed_passes, 2);
        assert!(eager.pending(100, Duration::from_secs(100)).is_none());
        assert!(lazy.pending(100, Duration::from_secs(100)).is_none());
    }

    #[timed_test]
    fn lazy_purge_is_not_called_after_discount_pass_cap() {
        let mut config = toy_training_config(1_000);
        config.lcfr_warmup_iterations = 0;
        config.lcfr_discount_interval = 1;
        config.dcfr_discount_max_passes = NonZeroU64::new(1);
        config.negative_action_subtree_purge_enabled = true;
        let storage = SparseMpStorage::with_shards(1);
        let parent =
            MpInfosetKey::from_street_bucket(Seat::from_raw(0), Street::Flop, 1, 0, 7, 7, 1);
        storage.add_regret(parent, 4, 3, -100);
        let blocked =
            storage.transition_negative_action_edge(parent, 3, -100, -1, 0, (0, 7 | (3 << 4), 2));
        assert!(blocked.blocked);
        let mut scheduler = DiscountScheduler::new(&config, 0);

        let first = scheduler
            .pending(1, Duration::ZERO)
            .expect("first discount should be due");
        apply_dcfr_discount_lazy(&storage, first.epoch, &config);
        purge_negative_action_subtrees_after_discount(&storage, &config);
        let _ = scheduler.complete(first, 1, Duration::ZERO);
        assert_eq!(storage.negative_action_telemetry().subtree_purge_calls, 1);

        if let Some(unexpected) = scheduler.pending(2, Duration::ZERO) {
            apply_dcfr_discount_lazy(&storage, unexpected.epoch, &config);
            purge_negative_action_subtrees_after_discount(&storage, &config);
            let _ = scheduler.complete(unexpected, 2, Duration::ZERO);
        }
        assert_eq!(storage.negative_action_telemetry().subtree_purge_calls, 1);
    }

    // -- apply_dcfr_discount tests --

    #[timed_test]
    fn dcfr_discount_atom_truncates_toward_zero_symmetrically() {
        let positive = AtomicI32::new(101);
        let negative = AtomicI32::new(-101);
        let positive_endpoint = AtomicI32::new(1);
        let negative_endpoint = AtomicI32::new(-1);

        discount_regret_atom(&positive, 0.5, 0.5);
        discount_regret_atom(&negative, 0.5, 0.5);
        discount_regret_atom(&positive_endpoint, 0.5, 0.5);
        discount_regret_atom(&negative_endpoint, 0.5, 0.5);

        assert_eq!(positive.load(Ordering::Relaxed), 50);
        assert_eq!(negative.load(Ordering::Relaxed), -50);
        assert_eq!(positive_endpoint.load(Ordering::Relaxed), 0);
        assert_eq!(negative_endpoint.load(Ordering::Relaxed), 0);
    }

    #[timed_test]
    fn repeated_dcfr_discount_atom_eliminates_integer_endpoints() {
        let positive = AtomicI32::new(3);
        let negative = AtomicI32::new(-3);

        for _ in 0..3 {
            discount_regret_atom(&positive, 0.5, 0.5);
            discount_regret_atom(&negative, 0.5, 0.5);
        }

        assert_eq!(positive.load(Ordering::Relaxed), 0);
        assert_eq!(negative.load(Ordering::Relaxed), 0);
    }

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
    fn dcfr_discount_application_uses_explicit_epoch() {
        let tree = minimal_tree(2);
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = MpStorage::new(&tree, bucket_counts);
        let node = first_decision_node(&tree);
        storage.add_regret(node, 0, 0, 1_000);
        storage.add_strategy_sum(node, 0, 0, 10_000);
        let config = toy_training_config(1_000);

        apply_dcfr_discount(&storage, 1, &config);

        assert_eq!(storage.get_regret(node, 0, 0), 500);
        assert_eq!(storage.get_strategy_sum(node, 0, 0), 2_500);
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
    fn dcfr_discount_serial_path_runs_at_threshold() {
        let tree = single_decision_tree();
        let bucket_counts = [SERIAL_DCFR_DISCOUNT_SLOTS as u16, 1, 1, 1];
        let storage = MpStorage::new(&tree, bucket_counts);
        assert_eq!(storage.regrets.len(), SERIAL_DCFR_DISCOUNT_SLOTS);
        let node = tree.root;
        storage.add_strategy_sum(node, 0, 0, 10_000);
        let config = toy_training_config(1000);

        apply_dcfr_discount(&storage, 100, &config);

        let after = storage.get_strategy_sum(node, 0, 0);
        assert!(
            after < 10_000,
            "serial path should discount strategy sum, got {after}"
        );
        assert!(
            after > 0,
            "serial path should keep strategy sum positive, got {after}"
        );
    }

    #[timed_test]
    fn dcfr_discount_parallel_path_runs_above_threshold() {
        let tree = single_decision_tree();
        let bucket_counts = [(SERIAL_DCFR_DISCOUNT_SLOTS + 1) as u16, 1, 1, 1];
        let storage = MpStorage::new(&tree, bucket_counts);
        assert_eq!(storage.regrets.len(), SERIAL_DCFR_DISCOUNT_SLOTS + 1);
        let node = tree.root;
        storage.add_strategy_sum(node, 0, 0, 10_000);
        let config = toy_training_config(1000);

        apply_dcfr_discount(&storage, 100, &config);

        let after = storage.get_strategy_sum(node, 0, 0);
        assert!(
            after < 10_000,
            "parallel path should discount strategy sum, got {after}"
        );
        assert!(
            after > 0,
            "parallel path should keep strategy sum positive, got {after}"
        );
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

    fn single_decision_tree() -> MpGameTree {
        MpGameTree {
            nodes: vec![MpGameNode::Decision {
                seat: Seat::from_raw(0),
                street: Street::Preflop,
                actions: vec![TreeAction::Check],
                children: vec![],
            }],
            root: 0,
            num_players: 2,
            starting_stack: Chips(20.0),
        }
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
