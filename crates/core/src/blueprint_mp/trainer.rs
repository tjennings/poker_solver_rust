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

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;

use super::config::{BlueprintMpConfig, MpGameConfig, MpTrainingConfig};
use super::game_tree::MpGameTree;
use super::mccfr::{sample_deal, traverse_external};
use super::storage::{MpStorage, REGRET_SCALE};
use super::types::{Bucket, Chips, Deal, DealWithBuckets, Seat};
use super::MAX_PLAYERS;

/// Result of a training run.
pub struct TrainResult {
    pub meta_iterations: u64,
    pub final_strategy_delta: f64,
}

/// Shared training state accessible from outside the training loop.
pub struct TrainContext {
    pub tree: Arc<MpGameTree>,
    pub storage: Arc<MpStorage>,
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
    TrainContext {
        tree: Arc::new(tree),
        storage: Arc::new(storage),
        iterations: Arc::new(AtomicU64::new(0)),
        quit: Arc::new(AtomicBool::new(false)),
        num_players: config.game.num_players,
        bucket_counts,
    }
}

/// Run training on an existing context. Updates `ctx.iterations` atomically.
pub fn run_training(
    ctx: &TrainContext,
    training: &MpTrainingConfig,
    game: &MpGameConfig,
) -> TrainResult {
    training_loop(
        &ctx.tree,
        &ctx.storage,
        training,
        ctx.num_players,
        ctx.bucket_counts,
        game.rake_rate,
        Chips(game.rake_cap),
        &ctx.iterations,
        &ctx.quit,
    )
}

/// Train an N-player blueprint strategy (convenience wrapper).
///
/// One meta-iteration = N traversals (one per seat as traverser).
#[must_use]
pub fn train_blueprint_mp(config: &BlueprintMpConfig) -> TrainResult {
    let ctx = setup_training(config);
    run_training(&ctx, &config.training, &config.game)
}

fn training_loop(
    tree: &MpGameTree,
    storage: &MpStorage,
    config: &MpTrainingConfig,
    num_players: u8,
    bucket_counts: [u16; 4],
    rake_rate: f64,
    rake_cap: Chips,
    iterations: &AtomicU64,
    quit: &AtomicBool,
) -> TrainResult {
    let max_iters = config.iterations.unwrap_or(u64::MAX);
    let scaled_threshold = (f64::from(config.prune_threshold) * REGRET_SCALE) as i32;
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
            tree, storage, num_players, bucket_counts,
            rake_rate, rake_cap, batch, meta_iter,
            prune, scaled_threshold,
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

/// Determine whether the current batch should use pruning.
///
/// Pruning activates after `prune_after_iterations` have elapsed and
/// applies to `1 - prune_explore_pct` of batches (the rest explore
/// all actions to avoid permanently losing information).
fn should_prune(meta_iter: u64, config: &MpTrainingConfig, rng: &mut impl Rng) -> bool {
    if meta_iter < config.prune_after_iterations {
        return false;
    }
    let explore: f64 = rng.random();
    explore >= config.prune_explore_pct
}

fn run_batch(
    tree: &MpGameTree,
    storage: &MpStorage,
    num_players: u8,
    bucket_counts: [u16; 4],
    rake_rate: f64,
    rake_cap: Chips,
    batch_size: u64,
    base_iter: u64,
    prune: bool,
    prune_threshold: i32,
) {
    (0..batch_size).into_par_iter().for_each(|i| {
        let seed = base_iter.wrapping_add(i).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut rng = SmallRng::seed_from_u64(seed);
        let deal = sample_deal(num_players, &mut rng);
        let buckets = compute_buckets_trivial(&deal, bucket_counts);
        for traverser in 0..num_players {
            traverse_external(
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
        }
    });
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
        atom.store((f64::from(v) * d) as i32, Ordering::Relaxed);
    });
    storage.strategy_sums.par_iter().for_each(|atom| {
        let v = atom.load(Ordering::Relaxed);
        atom.store((v as f64 * d_strat) as i64, Ordering::Relaxed);
    });
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

fn compute_buckets_trivial(deal: &Deal, bucket_counts: [u16; 4]) -> DealWithBuckets {
    let mut buckets = [[Bucket(0); 4]; MAX_PLAYERS];
    for (seat_buckets, hole) in buckets
        .iter_mut()
        .zip(deal.hole_cards.iter())
        .take(deal.num_players as usize)
    {
        let card_idx = hole[0].value as u16;
        for (street, &count) in bucket_counts.iter().enumerate() {
            seat_buckets[street] = Bucket(card_idx % count);
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

    use test_macros::timed_test;

    use super::*;
    use crate::blueprint_mp::config::{
        BlueprintMpConfig, ForcedBet, ForcedBetKind, MpActionAbstractionConfig, MpClusteringConfig,
        MpGameConfig, MpSnapshotConfig, MpStreetCluster, MpStreetSizes, MpTrainingConfig,
    };
    use crate::blueprint_mp::game_tree::MpGameTree;
    use crate::blueprint_mp::mccfr::sample_deal;
    use crate::blueprint_mp::storage::MpStorage;

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
            blinds,
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        let empty = MpStreetSizes {
            lead: vec![],
            raise: vec![],
        };
        let action = MpActionAbstractionConfig {
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
            cluster_path: None,
            iterations: Some(iterations),
            time_limit_minutes: None,
            lcfr_warmup_iterations: 0,
            lcfr_discount_interval: 50,
            prune_after_iterations: 1_000_000,
            prune_threshold: -250,
            prune_explore_pct: 0.05,
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
        };
        BlueprintMpConfig {
            game,
            action_abstraction: action,
            clustering,
            training,
            snapshots,
        }
    }

    fn toy_training_config(iterations: u64) -> MpTrainingConfig {
        MpTrainingConfig {
            cluster_path: None,
            iterations: Some(iterations),
            time_limit_minutes: None,
            lcfr_warmup_iterations: 100,
            lcfr_discount_interval: 50,
            prune_after_iterations: 1_000_000,
            prune_threshold: -250,
            prune_explore_pct: 0.05,
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
            blinds,
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        let empty = MpStreetSizes {
            lead: vec![],
            raise: vec![],
        };
        let action = MpActionAbstractionConfig {
            preflop: empty.clone(),
            flop: empty.clone(),
            turn: empty.clone(),
            river: empty,
        };
        MpGameTree::build(&game, &action)
    }

    // -- train_blueprint_mp integration tests --

    #[timed_test]
    fn train_2_player_toy_completes() {
        let config = toy_config(2, 100);
        let result = train_blueprint_mp(&config);
        assert_eq!(result.meta_iterations, 100);
    }

    #[timed_test]
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
        config.prune_explore_pct = 0.0; // never explore => always prune if past warmup
        let mut rng = SmallRng::seed_from_u64(42);
        // iter 50 < prune_after_iterations=100 => never prune
        assert!(!should_prune(50, &config, &mut rng));
    }

    #[timed_test]
    fn should_prune_true_after_warmup_no_explore() {
        let mut config = toy_training_config(1000);
        config.prune_after_iterations = 100;
        config.prune_explore_pct = 0.0; // explore_pct=0 => always prune
        let mut rng = SmallRng::seed_from_u64(42);
        assert!(should_prune(200, &config, &mut rng));
    }

    #[timed_test]
    fn should_prune_false_when_explore_pct_is_one() {
        let mut config = toy_training_config(1000);
        config.prune_after_iterations = 0;
        config.prune_explore_pct = 1.0; // explore_pct=1 => never prune
        let mut rng = SmallRng::seed_from_u64(42);
        // rng.random() is in [0,1), always < 1.0, so always explores
        assert!(!should_prune(200, &config, &mut rng));
    }

    #[timed_test]
    fn should_prune_respects_warmup() {
        let mut config = toy_training_config(1000);
        config.prune_after_iterations = 500;
        config.prune_explore_pct = 0.0;
        let mut rng = SmallRng::seed_from_u64(42);
        // Before warmup
        assert!(!should_prune(499, &config, &mut rng));
        // At warmup boundary
        assert!(should_prune(500, &config, &mut rng));
        // After warmup
        assert!(should_prune(1000, &config, &mut rng));
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
        assert!(after < 1000, "positive regret should be discounted, got {after}");
        assert!(after > 0, "positive regret should stay positive, got {after}");
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
        assert!(after < 10_000, "strategy sum should be discounted, got {after}");
        assert!(after > 0, "strategy sum should stay positive, got {after}");
    }

    #[timed_test]
    fn dcfr_discount_handles_negative_regrets() {
        let tree = minimal_tree(2);
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = MpStorage::new(&tree, bucket_counts);
        let node = first_decision_node(&tree);
        storage.add_regret(node, 0, 0, -500);
        let config = toy_training_config(1000);

        apply_dcfr_discount(&storage, 100, &config);

        let after = storage.get_regret(node, 0, 0);
        assert!(after > -500, "negative regret should be discounted toward zero");
        assert!(after <= 0, "negative regret should stay non-positive");
    }

    // -- run_batch tests --

    #[timed_test]
    fn run_batch_updates_regrets() {
        let tree = minimal_tree(2);
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = MpStorage::new(&tree, bucket_counts);

        run_batch(&tree, &storage, 2, bucket_counts, 0.0, Chips::ZERO, 20, 0, false, 0);

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

        run_batch(&tree, &storage, 2, bucket_counts, 0.0, Chips::ZERO, 20, 0, false, 0);

        let any_nonzero = storage
            .strategy_sums
            .iter()
            .any(|s| s.load(Ordering::Relaxed) != 0);
        assert!(any_nonzero, "run_batch should produce non-zero strategy sums");
    }

    #[timed_test]
    fn run_batch_3_player_updates_storage() {
        let tree = minimal_tree(3);
        let bucket_counts = [10u16, 10, 10, 10];
        let storage = MpStorage::new(&tree, bucket_counts);

        run_batch(&tree, &storage, 3, bucket_counts, 0.0, Chips::ZERO, 10, 0, false, 0);

        let any_nonzero = storage
            .regrets
            .iter()
            .any(|r| r.load(Ordering::Relaxed) != 0);
        assert!(any_nonzero, "3-player run_batch should produce non-zero regrets");
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

    #[timed_test]
    fn run_training_returns_correct_meta_iterations() {
        let config = toy_config(2, 50);
        let ctx = setup_training(&config);
        let result = run_training(&ctx, &config.training, &config.game);
        assert_eq!(result.meta_iterations, 50);
    }

    #[timed_test]
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
