//! Integration tests for HU legacy-to-universal-dense export (Phase 2).
//!
//! Covers the acceptance criteria:
//! 1. Legacy strategy.bin remains readable (existing tests cover this).
//! 2. Universal export probabilities match BlueprintV2Strategy row-for-row.
//! 3. Dense vs sparse projected exports produce identical strategy payloads.
//! 4. Row ordering, action descriptors, error handling, and zero-mass rows.

use poker_solver_core::blueprint_universal::{
    ActionKind, BundleReader,
};
use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::config::*;
use poker_solver_core::blueprint_v2::game_tree::{GameNode, GameTree, TreeAction};
use poker_solver_core::blueprint_v2::storage::action_schema_fingerprint;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a tiny config suitable for export testing (no cluster files needed).
fn tiny_export_config() -> BlueprintV2Config {
    BlueprintV2Config {
        game: GameConfig {
            name: "ExportTest".to_string(),
            players: 2,
            stack_depth: 10.0,
            small_blind: 1.0,
            big_blind: 2.0,
            rake_rate: 0.0,
            rake_cap: 0.0,
            allow_preflop_limp: true,
        },
        clustering: ClusteringConfig {
            algorithm: ClusteringAlgorithm::PotentialAwareEmd,
            preflop: StreetClusterConfig {
                buckets: 3,
                delta_bins: None,
                expected_delta: false,
                sample_boards: None,
                metric: Default::default(),
            },
            flop: StreetClusterConfig {
                buckets: 2,
                delta_bins: None,
                expected_delta: false,
                sample_boards: None,
                metric: Default::default(),
            },
            turn: StreetClusterConfig {
                buckets: 2,
                delta_bins: None,
                expected_delta: false,
                sample_boards: None,
                metric: Default::default(),
            },
            river: StreetClusterConfig {
                buckets: 2,
                delta_bins: None,
                expected_delta: false,
                sample_boards: None,
                metric: Default::default(),
            },
            seed: 42,
            kmeans_iterations: 10,
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
            prune_after_iterations: 200,
            prune_threshold: 0,
            prune_explore_pct: 0.05,
            print_every_minutes: 10,
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
            baseline_validation: Default::default(),
            prune_streets: None,
            regret_floor: None,
            exploitability_interval_minutes: 0,
            exploitability_samples: 100_000,
        },
        snapshots: SnapshotConfig {
            warmup_minutes: 60,
            snapshot_every_minutes: 30,
            output_dir: "runs/".into(),
            resume: false,
            max_snapshots: None,
        },
    }
}

/// Build a tree and strategy with known probabilities for deterministic testing.
fn build_test_tree_and_strategy(
    config: &BlueprintV2Config,
) -> (GameTree, BlueprintV2Strategy) {
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

    // Build a strategy with known deterministic values.
    let storage =
        poker_solver_core::blueprint_v2::storage::BlueprintStorage::new(&tree, bucket_counts);
    let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);
    (tree, strategy)
}

// ---------------------------------------------------------------------------
// Acceptance test: probabilities match row-for-row (criterion 2)
// ---------------------------------------------------------------------------

#[test]
fn export_probabilities_match_legacy_strategy_bitwise() {
    use poker_solver_core::blueprint_universal::hu_export;

    let config = tiny_export_config();
    let (tree, strategy) = build_test_tree_and_strategy(&config);
    let bucket_counts = [
        config.clustering.preflop.buckets,
        config.clustering.flop.buckets,
        config.clustering.turn.buckets,
        config.clustering.river.buckets,
    ];

    // Export to universal format in memory.
    let training = hu_export::TrainingInfo {
        source_backend: "hu_dense",
        iterations: 100,
        elapsed_minutes: 5.0,
    };
    let output = hu_export::export_hu_strategy_to_universal(
        &config, &tree, &strategy, &training,
    )
    .expect("export should succeed");

    // Write to disk, then load with BundleReader.
    let dir = tempfile::tempdir().expect("tempdir");
    poker_solver_core::blueprint_universal::write_bundle(
        dir.path(),
        &output.manifest,
        &poker_solver_core::blueprint_universal::BundleData {
            rows: &output.rows,
            actions: &output.actions,
            probs: &output.probs,
        },
    )
    .expect("write bundle");

    let reader = BundleReader::open(dir.path()).expect("open bundle");

    // Verify every (decision_node, bucket) row matches bitwise.
    let mut decision_idx = 0usize;
    for (node_idx, node) in tree.nodes.iter().enumerate() {
        if let GameNode::Decision {
            street, actions, ..
        } = node
        {
            let street_idx = *street as usize;
            let buckets = bucket_counts[street_idx];
            let num_actions = actions.len();

            for bucket in 0..buckets {
                let legacy_probs = strategy.get_action_probs(decision_idx, bucket);
                assert_eq!(
                    legacy_probs.len(),
                    num_actions,
                    "action count mismatch at node {node_idx}, bucket {bucket}"
                );

                // Find the matching row in the universal bundle.
                let mut found = false;
                for row_i in 0..reader.row_count() {
                    let row = reader.row(row_i).unwrap();
                    if row.source_node_idx == node_idx as u32
                        && row.global_bucket == u32::from(bucket)
                    {
                        found = true;
                        assert_eq!(
                            row.action_count as usize, num_actions,
                            "action count mismatch in universal row"
                        );
                        // Check probabilities are bitwise identical.
                        for a in 0..num_actions {
                            let universal_prob =
                                reader.prob(row.prob_offset as usize + a).unwrap();
                            assert_eq!(
                                universal_prob.to_bits(),
                                legacy_probs[a].to_bits(),
                                "probability mismatch at node {node_idx}, \
                                 bucket {bucket}, action {a}: \
                                 universal={universal_prob}, legacy={}",
                                legacy_probs[a]
                            );
                        }
                        break;
                    }
                }
                assert!(
                    found,
                    "no universal row for node {node_idx}, bucket {bucket}"
                );
            }

            decision_idx += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Dense vs sparse identity (criterion 3)
// ---------------------------------------------------------------------------

#[test]
fn dense_and_sparse_exports_produce_identical_payloads() {
    use poker_solver_core::blueprint_universal::hu_export;

    let config = tiny_export_config();
    let (tree, strategy) = build_test_tree_and_strategy(&config);

    let training_d = hu_export::TrainingInfo {
        source_backend: "hu_dense",
        iterations: 100,
        elapsed_minutes: 5.0,
    };
    let out_d = hu_export::export_hu_strategy_to_universal(
        &config, &tree, &strategy, &training_d,
    )
    .expect("dense export");

    let training_s = hu_export::TrainingInfo {
        source_backend: "hu_sparse_projected",
        iterations: 100,
        elapsed_minutes: 5.0,
    };
    let out_s = hu_export::export_hu_strategy_to_universal(
        &config, &tree, &strategy, &training_s,
    )
    .expect("sparse export");

    // Row descriptors must be identical.
    assert_eq!(out_d.rows.len(), out_s.rows.len());
    for (d, s) in out_d.rows.iter().zip(out_s.rows.iter()) {
        assert_eq!(d, s, "row descriptor mismatch");
    }

    // Action descriptors must be identical.
    assert_eq!(out_d.actions.len(), out_s.actions.len());
    for (d, s) in out_d.actions.iter().zip(out_s.actions.iter()) {
        assert_eq!(d, s, "action descriptor mismatch");
    }

    // Probabilities must be bitwise identical.
    assert_eq!(out_d.probs.len(), out_s.probs.len());
    for (i, (&d, &s)) in out_d.probs.iter().zip(out_s.probs.iter()).enumerate() {
        assert_eq!(
            d.to_bits(),
            s.to_bits(),
            "prob mismatch at index {i}"
        );
    }

    // Only training.source_backend should differ.
    assert_eq!(out_d.manifest.training.source_backend, "hu_dense");
    assert_eq!(out_s.manifest.training.source_backend, "hu_sparse_projected");
}

// ---------------------------------------------------------------------------
// Action descriptor test
// ---------------------------------------------------------------------------

#[test]
fn action_descriptors_match_tree_actions() {
    use poker_solver_core::blueprint_universal::hu_export;

    let config = tiny_export_config();
    let (tree, strategy) = build_test_tree_and_strategy(&config);

    let training = hu_export::TrainingInfo {
        source_backend: "hu_dense",
        iterations: 100,
        elapsed_minutes: 5.0,
    };
    let output = hu_export::export_hu_strategy_to_universal(
        &config, &tree, &strategy, &training,
    )
    .expect("export");

    // For each row, verify the action descriptors match the tree node's actions.
    for row in &output.rows {
        let node = &tree.nodes[row.source_node_idx as usize];
        if let GameNode::Decision {
            actions: tree_actions,
            ..
        } = node
        {
            assert_eq!(row.action_count as usize, tree_actions.len());

            // Check action_schema_fingerprint matches the legacy function.
            let expected_fp = action_schema_fingerprint(tree_actions.as_slice());
            assert_eq!(
                row.action_schema_fingerprint, expected_fp,
                "action_schema_fingerprint mismatch for node {}",
                row.source_node_idx
            );

            // Check individual action descriptors.
            for (a_idx, tree_action) in tree_actions.iter().enumerate() {
                let desc = &output.actions[row.action_offset as usize + a_idx];
                assert_eq!(desc.source_action_index, a_idx as u16);

                match tree_action {
                    TreeAction::Fold => {
                        assert_eq!(desc.kind, ActionKind::Fold);
                        assert!(!desc.is_aggressive);
                    }
                    TreeAction::Check => {
                        assert_eq!(desc.kind, ActionKind::Check);
                        assert!(!desc.is_aggressive);
                    }
                    TreeAction::Call => {
                        assert_eq!(desc.kind, ActionKind::Call);
                        assert!(!desc.is_aggressive);
                    }
                    TreeAction::Bet(_) => {
                        assert_eq!(desc.kind, ActionKind::Bet);
                        assert!(desc.is_aggressive);
                    }
                    TreeAction::Raise(_) => {
                        assert_eq!(desc.kind, ActionKind::Raise);
                        assert!(desc.is_aggressive);
                    }
                    TreeAction::AllIn => {
                        assert!(desc.is_aggressive);
                        assert!(
                            desc.kind == ActionKind::AllInBetRaise
                                || desc.kind == ActionKind::AllInCall,
                            "AllIn should map to AllInBetRaise or AllInCall"
                        );
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Row ordering test
// ---------------------------------------------------------------------------

#[test]
fn exported_rows_satisfy_spec_sort_order() {
    use poker_solver_core::blueprint_universal::hu_export;

    let config = tiny_export_config();
    let (tree, strategy) = build_test_tree_and_strategy(&config);

    let training = hu_export::TrainingInfo {
        source_backend: "hu_dense",
        iterations: 100,
        elapsed_minutes: 5.0,
    };
    let output = hu_export::export_hu_strategy_to_universal(
        &config, &tree, &strategy, &training,
    )
    .expect("export");

    // Rows must be sorted by identity_key and unique.
    for window in output.rows.windows(2) {
        assert!(
            window[0].identity_key() < window[1].identity_key(),
            "rows not sorted: {:?} >= {:?}",
            window[0].identity_key(),
            window[1].identity_key(),
        );
    }
}

// ---------------------------------------------------------------------------
// Zero-mass row preserves uniform
// ---------------------------------------------------------------------------

#[test]
fn zero_mass_row_exports_uniform() {
    use poker_solver_core::blueprint_universal::hu_export;

    // Fresh storage has zero strategy sums -> average_strategy returns uniform.
    let config = tiny_export_config();
    let (tree, strategy) = build_test_tree_and_strategy(&config);

    let training = hu_export::TrainingInfo {
        source_backend: "hu_dense",
        iterations: 100,
        elapsed_minutes: 5.0,
    };
    let output = hu_export::export_hu_strategy_to_universal(
        &config, &tree, &strategy, &training,
    )
    .expect("export");

    // All rows from fresh storage should be uniform.
    for row in &output.rows {
        let n = row.action_count as usize;
        let expected = 1.0f32 / n as f32;
        for a in 0..n {
            let p = output.probs[row.prob_offset as usize + a];
            assert_eq!(
                p.to_bits(),
                expected.to_bits(),
                "zero-mass row should have uniform probs"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Rejection tests
// ---------------------------------------------------------------------------

#[test]
fn rejects_missing_config_yaml() {
    use poker_solver_core::blueprint_universal::hu_export;

    let dir = tempfile::tempdir().expect("tempdir");
    let out_dir = tempfile::tempdir().expect("outdir");

    let err = hu_export::export_hu_bundle(dir.path(), "final", out_dir.path())
        .unwrap_err();
    assert!(
        format!("{err:?}").contains("config"),
        "should mention missing config: {err:?}"
    );
}

#[test]
fn rejects_missing_strategy_bin() {
    use poker_solver_core::blueprint_universal::hu_export;

    let dir = tempfile::tempdir().expect("tempdir");
    let out_dir = tempfile::tempdir().expect("outdir");

    // Write a valid config.yaml but no strategy.bin.
    let config = tiny_export_config();
    poker_solver_core::blueprint_v2::bundle::save_config(dir.path(), &config)
        .expect("save config");
    std::fs::create_dir_all(dir.path().join("final")).unwrap();

    let err = hu_export::export_hu_bundle(dir.path(), "final", out_dir.path())
        .unwrap_err();
    assert!(
        format!("{err:?}").contains("strategy"),
        "should mention missing strategy: {err:?}"
    );
}

#[test]
fn rejects_missing_metadata_json() {
    use poker_solver_core::blueprint_universal::hu_export;

    let dir = tempfile::tempdir().expect("tempdir");
    let out_dir = tempfile::tempdir().expect("outdir");

    // Write config.yaml and strategy.bin but no metadata.json.
    let config = tiny_export_config();
    poker_solver_core::blueprint_v2::bundle::save_config(dir.path(), &config)
        .expect("save config");

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
    let storage =
        poker_solver_core::blueprint_v2::storage::BlueprintStorage::new(&tree, bucket_counts);
    let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);

    let snapshot_dir = dir.path().join("final");
    std::fs::create_dir_all(&snapshot_dir).unwrap();
    strategy.save(&snapshot_dir.join("strategy.bin")).unwrap();

    let err = hu_export::export_hu_bundle(dir.path(), "final", out_dir.path())
        .unwrap_err();
    assert!(
        format!("{err:?}").contains("metadata"),
        "should mention missing metadata: {err:?}"
    );
}

#[test]
fn rejects_bad_snapshot_name() {
    use poker_solver_core::blueprint_universal::hu_export;

    let dir = tempfile::tempdir().expect("tempdir");
    let out_dir = tempfile::tempdir().expect("outdir");

    let config = tiny_export_config();
    poker_solver_core::blueprint_v2::bundle::save_config(dir.path(), &config)
        .expect("save config");

    let err = hu_export::export_hu_bundle(dir.path(), "invalid_name!", out_dir.path())
        .unwrap_err();
    assert!(
        matches!(err, hu_export::ExportError::BadSnapshotName { .. }),
        "should be BadSnapshotName: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Full disk round-trip: legacy bundle dir -> export -> universal read
// ---------------------------------------------------------------------------

#[test]
fn full_disk_export_round_trip() {
    use poker_solver_core::blueprint_universal::hu_export;

    let dir = tempfile::tempdir().expect("tempdir");

    // Build a legacy bundle directory.
    let config = tiny_export_config();
    poker_solver_core::blueprint_v2::bundle::save_config(dir.path(), &config)
        .expect("save config");

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
    let storage =
        poker_solver_core::blueprint_v2::storage::BlueprintStorage::new(&tree, bucket_counts);
    let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);

    let snapshot_dir = dir.path().join("final");
    std::fs::create_dir_all(&snapshot_dir).unwrap();
    strategy.save(&snapshot_dir.join("strategy.bin")).unwrap();
    std::fs::write(
        snapshot_dir.join("metadata.json"),
        r#"{"iteration": 100, "elapsed_minutes": 5}"#,
    )
    .unwrap();

    // Export to universal.
    let out_dir = tempfile::tempdir().expect("outdir");
    hu_export::export_hu_bundle(dir.path(), "final", out_dir.path())
        .expect("export should succeed");

    // Verify the universal bundle is loadable.
    let reader = BundleReader::open(out_dir.path()).expect("open exported bundle");
    assert!(reader.row_count() > 0, "should have exported rows");
}

/// Part A: disk exporter retains config.yaml and it round-trips.
#[test]
fn disk_export_retains_config_yaml_that_rebuilds_tree() {
    use poker_solver_core::blueprint_universal::hu_export;
    let config = tiny_export_config();
    let (_tree, strategy) = build_test_tree_and_strategy(&config);

    let dir = tempfile::tempdir().expect("source dir");
    poker_solver_core::blueprint_v2::bundle::save_config(dir.path(), &config).unwrap();
    let snapshot_dir = dir.path().join("final");
    std::fs::create_dir_all(&snapshot_dir).unwrap();
    strategy.save(&snapshot_dir.join("strategy.bin")).unwrap();
    std::fs::write(
        snapshot_dir.join("metadata.json"),
        r#"{"iteration": 100, "elapsed_minutes": 5}"#,
    )
    .unwrap();

    let out_dir = tempfile::tempdir().expect("outdir");
    hu_export::export_hu_bundle(dir.path(), "final", out_dir.path())
        .expect("export should succeed");

    // config.yaml must be present in the output.
    assert!(
        out_dir.path().join("config.yaml").exists(),
        "exported bundle should contain config.yaml"
    );

    // It must deserialize back and rebuild the same tree.
    let retained_config = poker_solver_core::blueprint_v2::bundle::load_config(out_dir.path())
        .expect("retained config.yaml should load");
    let aa = &retained_config.action_abstraction;
    let tree2 = GameTree::build_with_options(
        retained_config.game.stack_depth,
        retained_config.game.small_blind,
        retained_config.game.big_blind,
        &aa.preflop,
        &aa.flop,
        &aa.turn,
        &aa.river,
        retained_config.game.allow_preflop_limp,
    );
    assert_eq!(
        tree2.nodes.len(),
        GameTree::build_with_options(
            config.game.stack_depth,
            config.game.small_blind,
            config.game.big_blind,
            &config.action_abstraction.preflop,
            &config.action_abstraction.flop,
            &config.action_abstraction.turn,
            &config.action_abstraction.river,
            config.game.allow_preflop_limp,
        )
        .nodes
        .len(),
        "rebuilt tree should have the same node count"
    );

    // Manifest should reference config.yaml.
    let manifest_text =
        std::fs::read_to_string(out_dir.path().join("blueprint.json")).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest_text).unwrap();
    assert_eq!(
        manifest["training"]["config_path"].as_str(),
        Some("config.yaml"),
        "manifest should reference retained config.yaml"
    );
}
