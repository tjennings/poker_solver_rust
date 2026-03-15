//! End-to-end integration test for the Blueprint V2 pipeline.
//!
//! Exercises: cluster (river → turn → flop → preflop) → train → save/load → verify.
//! Uses tiny parameters (5 buckets, 5 BB stack, 50 iterations, 2-3 sample boards
//! per street) so the test completes in reasonable time even in debug mode.

use poker_solver_core::blueprint_v2::{
    bundle::{self, BlueprintV2Strategy},
    cluster_diagnostics,
    cluster_pipeline,
    config::*,
    trainer::BlueprintTrainer,
};

/// Build a minimal config suitable for fast integration testing.
fn tiny_config(cluster_dir: &std::path::Path, run_dir: &std::path::Path) -> BlueprintV2Config {
    BlueprintV2Config {
        game: GameConfig {
            name: "Test".to_string(),
            players: 2,
            stack_depth: 5.0,
            small_blind: 0.5,
            big_blind: 1.0,
            rake_rate: 0.0,
            rake_cap: 0.0,
        },
        clustering: ClusteringConfig {
            algorithm: ClusteringAlgorithm::PotentialAwareEmd,
            preflop: StreetClusterConfig { buckets: 5, delta_bins: None, expected_delta: false, sample_boards: None },
            flop: StreetClusterConfig { buckets: 5, delta_bins: None, expected_delta: false, sample_boards: None },
            turn: StreetClusterConfig { buckets: 5, delta_bins: None, expected_delta: false, sample_boards: None },
            river: StreetClusterConfig { buckets: 5, delta_bins: None, expected_delta: false, sample_boards: None },
            seed: 42,
            kmeans_iterations: 10,
            cfvnet_river_data: None,
        },
        action_abstraction: ActionAbstractionConfig {
            preflop: vec![vec!["2.5bb".into()]],
            flop: vec![vec![1.0]],
            turn: vec![vec![1.0]],
            river: vec![vec![1.0]],
        },
        training: TrainingConfig {
            cluster_path: Some(cluster_dir.to_string_lossy().into_owned()),
            iterations: Some(50),
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
            equity_cache_path: None,
            dcfr_alpha: 1.0,
            dcfr_beta: 1.0,
            dcfr_gamma: 1.0,
        },
        snapshots: SnapshotConfig {
            warmup_minutes: 9999,
            snapshot_every_minutes: 9999,
            output_dir: run_dir.to_string_lossy().into_owned(),
            resume: false,
            max_snapshots: None,
        },
    }
}

/// Full pipeline: cluster → train → snapshot → load → verify probabilities.
#[test]
#[ignore] // slow: full pipeline (cluster + train) takes ~50s in debug mode
fn blueprint_v2_e2e_pipeline() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let cluster_dir = dir.path().join("clusters");
    let run_dir = dir.path().join("run");
    std::fs::create_dir_all(&cluster_dir).unwrap();
    std::fs::create_dir_all(&run_dir).unwrap();

    let config = tiny_config(&cluster_dir, &run_dir);
    let seed = config.clustering.seed;
    let k = config.clustering.kmeans_iterations;

    // ── 1. Clustering (with tiny board counts) ──────────────────────────
    // Use 2 boards for equity-heavy streets to keep debug-mode runtime
    // tolerable.  Preflop needs more boards so every combo matches at
    // least one non-conflicting board (debug_assert guard).
    let few_boards = 2;
    let preflop_boards = 30;

    let river = cluster_pipeline::cluster_river_with_boards(
        config.clustering.river.buckets,
        k,
        seed,
        few_boards,
        |_, _| {},
    );
    river
        .save(&cluster_dir.join("river.buckets"))
        .expect("save river");

    let turn = cluster_pipeline::cluster_turn_with_boards(
        &river,
        config.clustering.turn.buckets,
        k,
        seed,
        few_boards,
        |_, _| {},
    );
    turn.save(&cluster_dir.join("turn.buckets"))
        .expect("save turn");

    let flop = cluster_pipeline::cluster_flop_with_boards(
        &turn,
        config.clustering.flop.buckets,
        k,
        seed,
        few_boards,
        |_, _| {},
    );
    flop.save(&cluster_dir.join("flop.buckets"))
        .expect("save flop");

    let preflop = cluster_pipeline::cluster_preflop(|_, _| {});
    preflop
        .save(&cluster_dir.join("preflop.buckets"))
        .expect("save preflop");

    // Verify all cluster files exist.
    for street in &["river", "turn", "flop", "preflop"] {
        assert!(
            cluster_dir.join(format!("{street}.buckets")).exists(),
            "{street}.buckets should exist"
        );
    }

    // ── 2. Cluster diagnostics ──────────────────────────────────────────
    let reports = cluster_diagnostics::diagnose_cluster_dir(&cluster_dir)
        .expect("diagnose clusters");
    assert_eq!(reports.len(), 4, "should have reports for all 4 streets");
    for report in &reports {
        assert!(report.bucket_count > 0, "bucket count should be positive");
        assert!(
            report.total_entries > 0,
            "total entries should be positive for {}",
            report.street
        );
    }

    // ── 3. Train ────────────────────────────────────────────────────────
    let mut trainer = BlueprintTrainer::new(config.clone());
    trainer.skip_bucket_validation = true;
    trainer.train().expect("training should complete");
    assert_eq!(trainer.iterations, 50, "should have run 50 iterations");

    // After training, some regrets should be non-zero.
    assert!(
        trainer.storage.regrets.iter().any(|r| r.load(std::sync::atomic::Ordering::Relaxed) != 0),
        "regrets should be updated after training"
    );

    // ── 4. Extract strategy and verify probability validity ─────────────
    let strategy = BlueprintV2Strategy::from_storage(&trainer.storage, &trainer.tree);
    assert!(
        strategy.num_decision_nodes() > 0,
        "strategy should contain decision nodes"
    );

    // Every (decision node, bucket) pair should have action probs summing to ~1.0.
    for dec_idx in 0..strategy.num_decision_nodes() {
        let street_idx = strategy.node_street_indices[dec_idx] as usize;
        let buckets = strategy.bucket_counts[street_idx];
        for bucket in 0..buckets {
            let probs = strategy.get_action_probs(dec_idx, bucket);
            assert!(
                !probs.is_empty(),
                "decision node {dec_idx}, bucket {bucket}: no action probs"
            );
            let sum: f32 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "decision node {dec_idx}, bucket {bucket}: probs sum to {sum}, expected ~1.0"
            );
            for (a, &p) in probs.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&p),
                    "decision node {dec_idx}, bucket {bucket}, action {a}: prob {p} out of [0,1]"
                );
            }
        }
    }

    // ── 5. Save snapshot and reload ─────────────────────────────────────
    let snapshot_dir = run_dir.join("snapshot_e2e");
    let metadata = format!(
        r#"{{"iteration": {}, "elapsed_minutes": 0}}"#,
        trainer.iterations,
    );
    bundle::save_snapshot(&snapshot_dir, &strategy, &trainer.storage, &metadata)
        .expect("save snapshot");

    assert!(
        snapshot_dir.join("strategy.bin").exists(),
        "strategy.bin should exist"
    );
    assert!(
        snapshot_dir.join("regrets.bin").exists(),
        "regrets.bin should exist"
    );
    assert!(
        snapshot_dir.join("metadata.json").exists(),
        "metadata.json should exist"
    );

    // Round-trip the strategy through save/load.
    let loaded = BlueprintV2Strategy::load(&snapshot_dir.join("strategy.bin"))
        .expect("load strategy");
    assert_eq!(
        loaded.action_probs.len(),
        strategy.action_probs.len(),
        "loaded strategy should match original length"
    );
    assert_eq!(
        loaded.node_action_counts, strategy.node_action_counts,
        "loaded action counts should match"
    );
    assert_eq!(
        loaded.bucket_counts, strategy.bucket_counts,
        "loaded bucket counts should match"
    );

    // Verify probabilities survived the round-trip.
    for (i, (&a, &b)) in strategy
        .action_probs
        .iter()
        .zip(loaded.action_probs.iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-6,
            "action prob mismatch at index {i}: {a} vs {b}"
        );
    }

    // ── 6. Config save/load round-trip ──────────────────────────────────
    bundle::save_config(&run_dir, &config).expect("save config");
    let loaded_config = bundle::load_config(&run_dir).expect("load config");
    assert!(
        (loaded_config.game.stack_depth - config.game.stack_depth).abs() < f64::EPSILON,
        "config round-trip: stack_depth mismatch"
    );
    assert_eq!(
        loaded_config.clustering.river.buckets,
        config.clustering.river.buckets,
        "config round-trip: river buckets mismatch"
    );
}
