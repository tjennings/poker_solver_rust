//! Integration test: verify TUI metrics are populated during training.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use poker_solver_core::blueprint_v2::config::*;
use poker_solver_core::blueprint_v2::trainer::BlueprintTrainer;

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
        },
        training: TrainingConfig {
            cluster_path: "clusters/".into(),
            iterations: Some(100),
            time_limit_minutes: None,
            lcfr_warmup_iterations: 9999,
            lcfr_discount_interval: 1,
            prune_after_iterations: 9999,
            prune_threshold: -310_000_000,
            prune_explore_pct: 0.05,
            print_every_minutes: 9999,
            batch_size: 1,
            target_strategy_delta: None,
            purify_threshold: 0.0,
        },
        snapshots: SnapshotConfig {
            warmup_minutes: 9999,
            snapshot_every_minutes: 9999,
            output_dir: "/tmp/test_tui_integration".into(),
            resume: false,
            max_snapshots: None,
        },
    }
}

#[test]
fn shared_iterations_tracks_training() {
    let mut trainer = BlueprintTrainer::new(toy_config());
    trainer.skip_bucket_validation = true;
    assert_eq!(trainer.shared_iterations.load(Ordering::Relaxed), 0);
    trainer.train().expect("training should complete");
    assert_eq!(trainer.shared_iterations.load(Ordering::Relaxed), 100);
}

#[test]
fn quit_requested_stops_training() {
    let mut config = toy_config();
    config.training.iterations = Some(1_000_000); // would take forever
    let mut trainer = BlueprintTrainer::new(config);
    trainer.skip_bucket_validation = true;
    trainer.quit_requested.store(true, Ordering::Relaxed);
    trainer.train().expect("should exit immediately");
    assert_eq!(trainer.iterations, 0);
}

#[test]
fn strategy_refresh_callback_fires() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = Arc::clone(&call_count);

    let mut config = toy_config();
    config.training.iterations = Some(50);
    let mut trainer = BlueprintTrainer::new(config);
    trainer.skip_bucket_validation = true;
    trainer.scenario_node_indices = vec![trainer.tree.root];
    trainer.strategy_refresh_interval_secs = 0; // fire every check
    trainer.on_strategy_refresh = Some(Box::new(move |_, _, _, _| {
        cc.fetch_add(1, Ordering::Relaxed);
    }));

    trainer.train().expect("training should complete");
    // Should have fired at least once (refresh interval = 0 means every check).
    assert!(call_count.load(Ordering::Relaxed) > 0);
}
