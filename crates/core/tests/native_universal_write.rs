//! Byte-identity tests: native universal writes must produce payloads
//! identical to post-hoc export on the same in-memory state.
//!
//! For each backend (HU, MP eager, MP lazy), we:
//! 1. Build a tiny training state.
//! 2. Write a native universal bundle via the new snapshot-time writer.
//! 3. Write a post-hoc export bundle via the existing disk-wrapper path.
//! 4. Compare all binary payload files byte-for-byte.
//! 5. Compare manifests modulo `created_at` (per-run timestamp).

use std::path::Path;

use poker_solver_core::blueprint_universal::hu_export;
use poker_solver_core::blueprint_universal::Manifest;
use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::config::*;
use poker_solver_core::blueprint_v2::game_tree::GameTree;
use poker_solver_core::blueprint_v2::storage::BlueprintStorage;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tiny_hu_config(output_dir: &str) -> BlueprintV2Config {
    BlueprintV2Config {
        game: GameConfig {
            name: "ByteIdentity".to_string(),
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
                buckets: 3, delta_bins: None, expected_delta: false,
                sample_boards: None, metric: Default::default(),
            },
            flop: StreetClusterConfig {
                buckets: 2, delta_bins: None, expected_delta: false,
                sample_boards: None, metric: Default::default(),
            },
            turn: StreetClusterConfig {
                buckets: 2, delta_bins: None, expected_delta: false,
                sample_boards: None, metric: Default::default(),
            },
            river: StreetClusterConfig {
                buckets: 2, delta_bins: None, expected_delta: false,
                sample_boards: None, metric: Default::default(),
            },
            seed: 42, kmeans_iterations: 10, cfvnet_river_data: None,
            per_flop: None,
        },
        action_abstraction: ActionAbstractionConfig {
            preflop: vec![vec!["5bb".into()]],
            flop: vec![vec![1.0]],
            turn: vec![vec![1.0]],
            river: vec![vec![1.0]],
        },
        training: TrainingConfig {
            cluster_path: None, iterations: Some(100),
            time_limit_minutes: None, lcfr_warmup_iterations: 0,
            lcfr_discount_interval: 1, prune_after_iterations: 200,
            prune_threshold: 0, prune_explore_pct: 0.05,
            print_every_minutes: 10, batch_size: 200,
            target_strategy_delta: None, purify_threshold: 0.0,
            equity_cache_path: None, dcfr_alpha: 1.0, dcfr_beta: 1.0,
            dcfr_gamma: 1.0, dcfr_epoch_cap: None,
            optimizer: "dcfr".to_string(),
            storage_backend: "dense".to_string(),
            sapcfr_eta: 0.5, brcfr_eta: 0.6,
            brcfr_warmup_iterations: 0, brcfr_interval: 100_000_000,
            use_baselines: false, baseline_alpha: 0.01,
            baseline_validation: Default::default(),
            prune_streets: None, regret_floor: None,
            exploitability_interval_minutes: 0,
            exploitability_samples: 100_000,
        },
        snapshots: SnapshotConfig {
            warmup_minutes: 60, snapshot_every_minutes: 30,
            output_dir: output_dir.to_string(), resume: false,
            max_snapshots: None, format: SnapshotFormat::Legacy,
        },
    }
}

/// Read a binary file, return its bytes.
fn read_bytes(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Compare manifests, ignoring `created_at`.
fn manifests_equal_modulo_created_at(a: &Manifest, b: &Manifest) -> bool {
    let mut a = a.clone();
    let mut b = b.clone();
    a.created_at = String::new();
    b.created_at = String::new();
    // Compare via JSON serialization
    let a_json = serde_json::to_string(&a).unwrap();
    let b_json = serde_json::to_string(&b).unwrap();
    a_json == b_json
}

/// Load a manifest from a bundle directory.
fn load_manifest(dir: &Path) -> Manifest {
    let text = std::fs::read_to_string(dir.join("blueprint.json"))
        .expect("blueprint.json missing");
    serde_json::from_str(&text).expect("manifest parse failed")
}

// ---------------------------------------------------------------------------
// HU byte-identity test
// ---------------------------------------------------------------------------

#[test]
fn hu_native_write_payload_matches_post_hoc_export() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let output_dir = dir.path().join("output");
    let config = tiny_hu_config(&output_dir.to_string_lossy());

    // Build tree and strategy
    let tree = GameTree::build_with_options(
        config.game.stack_depth, config.game.small_blind,
        config.game.big_blind, &config.action_abstraction.preflop,
        &config.action_abstraction.flop, &config.action_abstraction.turn,
        &config.action_abstraction.river, config.game.allow_preflop_limp,
    );
    let storage = BlueprintStorage::new(&tree, [3, 2, 2, 2]);
    let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);

    // Write config.yaml (needed by both paths)
    std::fs::create_dir_all(&output_dir).unwrap();
    let config_yaml = serde_yaml::to_string(&config).unwrap();
    let config_path = output_dir.join("config.yaml");
    std::fs::write(&config_path, &config_yaml).unwrap();

    // Path A: native write
    let native_dir = dir.path().join("native");
    hu_export::write_hu_universal_snapshot(
        &config, &tree, &strategy, 42, 1.5, &config_path, &native_dir,
    )
    .expect("native write");

    // Path B: post-hoc export (reuses the same in-memory strategy)
    let posthoc_dir = dir.path().join("posthoc");
    let training = hu_export::TrainingInfo {
        source_backend: "hu_dense",
        iterations: 42,
        elapsed_minutes: 1.5,
    };
    let mut output = hu_export::export_hu_strategy_to_universal(
        &config, &tree, &strategy, &training,
    )
    .unwrap();
    poker_solver_core::blueprint_universal::export_common::retain_config_yaml(
        &config_path,
        &posthoc_dir,
        &mut output.manifest,
    )
    .unwrap();
    poker_solver_core::blueprint_universal::write_bundle(
        &posthoc_dir,
        &output.manifest,
        &poker_solver_core::blueprint_universal::BundleData {
            rows: &output.rows,
            actions: &output.actions,
            probs: &output.probs,
        },
    )
    .unwrap();

    // Compare payload files byte-for-byte
    for name in &[
        "strategy.rows.bin",
        "strategy.actions.bin",
        "strategy.probs.f32.bin",
    ] {
        let native_bytes = read_bytes(&native_dir.join(name));
        let posthoc_bytes = read_bytes(&posthoc_dir.join(name));
        assert_eq!(
            native_bytes, posthoc_bytes,
            "HU payload {name} differs between native and post-hoc"
        );
    }

    // Compare manifests modulo created_at
    let native_manifest = load_manifest(&native_dir);
    let posthoc_manifest = load_manifest(&posthoc_dir);
    assert!(
        manifests_equal_modulo_created_at(&native_manifest, &posthoc_manifest),
        "HU manifests differ (modulo created_at)"
    );

    // Verify loadable
    let reader = poker_solver_core::blueprint_universal::BundleReader::open(&native_dir)
        .expect("native bundle should load");
    assert!(reader.row_count() > 0);
}

// ---------------------------------------------------------------------------
// MP eager byte-identity test
// ---------------------------------------------------------------------------

#[test]
fn mp_eager_native_write_payload_matches_post_hoc_export() {
    use poker_solver_core::blueprint_mp::config::*;
    use poker_solver_core::blueprint_mp::game_tree::MpGameTree;
    use poker_solver_core::blueprint_mp::storage::MpStorage;
    use poker_solver_core::blueprint_universal::mp_eager_export;

    let dir = tempfile::tempdir().expect("create temp dir");

    let game = MpGameConfig {
        name: "MpByteId".to_string(),
        num_players: 2,
        stack_depth: 10.0,
        allow_preflop_limp: true,
        blinds: vec![
            ForcedBet { seat: 0, kind: ForcedBetKind::SmallBlind, amount: 1.0 },
            ForcedBet { seat: 1, kind: ForcedBetKind::BigBlind, amount: 2.0 },
        ],
        rake_rate: 0.0,
        rake_cap: 0.0,
    };
    let street_sizes = MpStreetSizes {
        lead: vec![serde_yaml::Value::Number(serde_yaml::Number::from(1_u64))],
        raise: vec![],
    };
    let action = MpActionAbstractionConfig {
        max_flop_players: None,
        preflop: MpStreetSizes {
            lead: vec![serde_yaml::Value::String("5bb".into())],
            raise: vec![],
        },
        flop: street_sizes.clone(),
        turn: street_sizes.clone(),
        river: street_sizes,
    };
    let clustering = MpClusteringConfig {
        preflop: MpStreetCluster { buckets: 3 },
        flop: MpStreetCluster { buckets: 2 },
        turn: MpStreetCluster { buckets: 2 },
        river: MpStreetCluster { buckets: 2 },
    };
    let training_cfg = MpTrainingConfig {
        backend: MpTrainingBackend::Eager,
        chance_continuation_mode: MpChanceContinuationMode::SampledFullDeal,
        cluster_path: None,
        iterations: Some(10),
        time_limit_minutes: None,
        lcfr_warmup_iterations: 0,
        lcfr_discount_interval: 1,
        prune_after_iterations: 200,
        traversal_pruning_enabled: false,
        prune_threshold: 0,
        prune_explore_pct: 0.0,
        negative_action_subtree_purge_enabled: false,
        negative_action_prune_below: -1,
        negative_action_reactivate_at: 0,
        negative_action_purge_mode: MpNegativeActionPurgeMode::ScanHistoryPrefix,
        batch_size: 200,
        dcfr_alpha: 1.0,
        dcfr_beta: 1.0,
        dcfr_gamma: 1.0,
        print_every_minutes: 10,
        purify_threshold: 0.0,
        exploitability_interval_minutes: 0,
        exploitability_samples: 0,
    };
    let snapshots = MpSnapshotConfig {
        warmup_minutes: 999,
        snapshot_every_minutes: 999,
        output_dir: dir.path().join("output").to_string_lossy().into_owned(),
        resume: false,
        max_snapshots: None,
        format: MpSnapshotFormat::Legacy,
    };
    let config = BlueprintMpConfig {
        game,
        action_abstraction: action,
        clustering,
        training: training_cfg,
        snapshots,
    };

    let tree = MpGameTree::build(&config.game, &config.action_abstraction);
    let bucket_counts = config.clustering.bucket_counts();
    let storage = MpStorage::new(&tree, bucket_counts);

    // Write config.yaml
    let output_dir = dir.path().join("output");
    std::fs::create_dir_all(&output_dir).unwrap();
    let config_yaml = serde_yaml::to_string(&config).unwrap();
    let config_path = output_dir.join("config.yaml");
    std::fs::write(&config_path, &config_yaml).unwrap();

    // Path A: native write
    let native_dir = dir.path().join("native");
    mp_eager_export::write_mp_universal_snapshot(
        &config, &tree, &storage, 42, 1.5, &config_path, &native_dir,
    )
    .expect("native write");

    // Path B: post-hoc export
    let posthoc_dir = dir.path().join("posthoc");
    let training = mp_eager_export::MpTrainingInfo {
        iterations: 42,
        elapsed_minutes: 1.5,
    };
    let mut output = mp_eager_export::export_mp_strategy_to_universal(
        &config, &tree, &storage, &training,
    )
    .unwrap();
    poker_solver_core::blueprint_universal::export_common::retain_config_yaml(
        &config_path,
        &posthoc_dir,
        &mut output.manifest,
    )
    .unwrap();
    poker_solver_core::blueprint_universal::write_bundle(
        &posthoc_dir,
        &output.manifest,
        &poker_solver_core::blueprint_universal::BundleData {
            rows: &output.rows,
            actions: &output.actions,
            probs: &output.probs,
        },
    )
    .unwrap();

    // Compare payload files
    for name in &[
        "strategy.rows.bin",
        "strategy.actions.bin",
        "strategy.probs.f32.bin",
    ] {
        let native_bytes = read_bytes(&native_dir.join(name));
        let posthoc_bytes = read_bytes(&posthoc_dir.join(name));
        assert_eq!(
            native_bytes, posthoc_bytes,
            "MP eager payload {name} differs"
        );
    }

    // Compare manifests modulo created_at
    let native_manifest = load_manifest(&native_dir);
    let posthoc_manifest = load_manifest(&posthoc_dir);
    assert!(
        manifests_equal_modulo_created_at(&native_manifest, &posthoc_manifest),
        "MP eager manifests differ (modulo created_at)"
    );
}

// ---------------------------------------------------------------------------
// MP lazy byte-identity test
// ---------------------------------------------------------------------------

#[test]
fn mp_lazy_native_write_payload_matches_post_hoc_export() {
    use poker_solver_core::blueprint_mp::sparse_storage::{
        MpInfosetKey, SparseMpStorage,
    };
    use poker_solver_core::blueprint_mp::types::{Seat, Street};
    use poker_solver_core::blueprint_universal::mp_lazy_export::{
        LazyExportConfig, LazyTrainingInfo, export_lazy_sparse_to_universal,
        write_lazy_bundle, write_lazy_universal_snapshot,
    };

    let dir = tempfile::tempdir().expect("create temp dir");

    let lazy_config = LazyExportConfig {
        num_players: 2,
        stack_depth: 10.0,
        bucket_counts: [3, 2, 2, 2],
        small_blind: 1.0,
        big_blind: 2.0,
    };

    // Build a sparse storage with a few entries
    let storage = SparseMpStorage::with_shards(4);
    let key = MpInfosetKey::from_street_bucket(
        Seat::from_raw(0), Street::Preflop, 1, 0, 0, 0, 0,
    );
    storage.add_regret(key, 2, 0, 25);
    storage.add_strategy_sum(key, 2, 1, 50);

    let entries = storage.snapshot_entries();

    // Write config.yaml
    let config_dir = dir.path().join("config");
    std::fs::create_dir_all(&config_dir).unwrap();
    let config_yaml = serde_yaml::to_string(&lazy_config).unwrap();
    let config_path = config_dir.join("config.yaml");
    std::fs::write(&config_path, &config_yaml).unwrap();

    // Path A: native write
    let native_dir = dir.path().join("native");
    write_lazy_universal_snapshot(
        &lazy_config, &entries, 42, 1.5, &config_path, &native_dir,
    )
    .expect("native write");

    // Path B: post-hoc export
    let posthoc_dir = dir.path().join("posthoc");
    let training = LazyTrainingInfo {
        iterations: 42,
        elapsed_minutes: 1.5,
    };
    let mut output = export_lazy_sparse_to_universal(
        &lazy_config, &entries, &training,
    );
    poker_solver_core::blueprint_universal::export_common::retain_config_yaml(
        &config_path,
        &posthoc_dir,
        &mut output.manifest,
    )
    .unwrap();
    write_lazy_bundle(&posthoc_dir, &output).unwrap();

    // Compare payload files
    for name in &[
        "strategy.rows.bin",
        "strategy.actions.bin",
        "strategy.probs.f32.bin",
        "strategy.semantic.bin",
    ] {
        let native_bytes = read_bytes(&native_dir.join(name));
        let posthoc_bytes = read_bytes(&posthoc_dir.join(name));
        assert_eq!(
            native_bytes, posthoc_bytes,
            "MP lazy payload {name} differs"
        );
    }

    // Compare manifests modulo created_at
    let native_manifest = load_manifest(&native_dir);
    let posthoc_manifest = load_manifest(&posthoc_dir);
    assert!(
        manifests_equal_modulo_created_at(&native_manifest, &posthoc_manifest),
        "MP lazy manifests differ (modulo created_at)"
    );
}

// ---------------------------------------------------------------------------
// format=legacy regression guard
// ---------------------------------------------------------------------------

#[test]
fn format_legacy_produces_no_universal_dir() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let output_dir = dir.path().join("output");
    let config = tiny_hu_config(&output_dir.to_string_lossy());

    // Build tree and strategy
    let tree = GameTree::build_with_options(
        config.game.stack_depth, config.game.small_blind,
        config.game.big_blind, &config.action_abstraction.preflop,
        &config.action_abstraction.flop, &config.action_abstraction.turn,
        &config.action_abstraction.river, config.game.allow_preflop_limp,
    );
    let storage = BlueprintStorage::new(&tree, [3, 2, 2, 2]);

    // Use the trainer with legacy format
    assert_eq!(config.snapshots.format, SnapshotFormat::Legacy);

    // Write legacy snapshot
    let snapshot_dir = output_dir.join("snapshot_0000");
    let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);

    std::fs::create_dir_all(&output_dir).unwrap();
    poker_solver_core::blueprint_v2::bundle::save_config(&output_dir, &config).unwrap();
    poker_solver_core::blueprint_v2::bundle::save_snapshot(
        &snapshot_dir,
        &strategy,
        &storage,
        r#"{"iteration": 0}"#,
    )
    .unwrap();

    // Verify: legacy files present, NO universal/ dir
    assert!(snapshot_dir.join("strategy.bin").exists());
    assert!(snapshot_dir.join("regrets.bin").exists());
    assert!(
        !snapshot_dir.join("universal").exists(),
        "format=legacy should not create universal/ dir"
    );
}
