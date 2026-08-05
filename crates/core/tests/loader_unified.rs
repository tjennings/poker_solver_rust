//! Integration tests for Phase 5: unified bundle detection and loading.
//!
//! Tests cover:
//! - Detection of all four bundle kinds (LegacyHu, UniversalHu, MpEager, MpLazy)
//! - Nesting resolution (final/, snapshot_NNNN/)
//! - Error cases (empty dir, garbage manifest, unknown backend, missing payload)
//! - Load + query for each kind
//! - HU equivalence: legacy vs universal loader returns identical probs
//! - MP eager: load and query matches exporter output
//! - MP lazy: load and query matches lazy normalization

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use poker_solver_core::blueprint_universal::hu_export::{self, TrainingInfo as HuTrainingInfo};
use poker_solver_core::blueprint_universal::{
    ActionKind, BundleData, BundleKind, BundleReader, FormatError, InfosetView, LoaderError,
    MpLazyKey, detect_bundle_kind, load_bundle, write_bundle,
};
use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::config::*;
use poker_solver_core::blueprint_v2::game_tree::{GameNode, GameTree};
use poker_solver_core::blueprint_v2::storage::BlueprintStorage;

use rand::SeedableRng;
use sha2::{Digest, Sha256};
use std::path::Path;
use tempfile::TempDir;

// ── Shared helpers (reused from hu_universal_export.rs pattern) ────────

fn tiny_export_config() -> BlueprintV2Config {
    BlueprintV2Config {
        game: GameConfig {
            name: "LoaderTest".to_string(),
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
            format: SnapshotFormat::Legacy,
        },
    }
}

fn build_test_tree_and_strategy(config: &BlueprintV2Config) -> (GameTree, BlueprintV2Strategy) {
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
    let storage = BlueprintStorage::new(&tree, bucket_counts);
    let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);
    (tree, strategy)
}

/// Write a legacy HU bundle (config.yaml + strategy.bin in final/).
fn write_legacy_bundle(dir: &Path) -> (BlueprintV2Config, GameTree, BlueprintV2Strategy) {
    let config = tiny_export_config();
    let (tree, strategy) = build_test_tree_and_strategy(&config);

    poker_solver_core::blueprint_v2::bundle::save_config(dir, &config).unwrap();
    let final_dir = dir.join("final");
    std::fs::create_dir_all(&final_dir).unwrap();
    strategy.save(&final_dir.join("strategy.bin")).unwrap();

    (config, tree, strategy)
}

/// Export a universal HU bundle to a directory.
fn write_universal_hu_bundle(dir: &Path) -> (BlueprintV2Config, GameTree, BlueprintV2Strategy) {
    let config = tiny_export_config();
    let (tree, strategy) = build_test_tree_and_strategy(&config);

    let training = HuTrainingInfo {
        source_backend: "hu_dense",
        iterations: 100,
        elapsed_minutes: 5.0,
    };
    let output = hu_export::export_hu_strategy_to_universal(&config, &tree, &strategy, &training)
        .expect("export should succeed");

    write_bundle(
        dir,
        &output.manifest,
        &BundleData {
            rows: &output.rows,
            actions: &output.actions,
            probs: &output.probs,
        },
    )
    .expect("write bundle");

    (config, tree, strategy)
}

// ── Detection tests ──────────────────────────────────────────────────

#[test]
fn detect_legacy_hu_bundle() {
    let tmp = TempDir::new().unwrap();
    write_legacy_bundle(tmp.path());
    let kind = detect_bundle_kind(tmp.path()).unwrap();
    assert_eq!(kind, BundleKind::LegacyHu);
}

#[test]
fn detect_universal_hu_bundle() {
    let tmp = TempDir::new().unwrap();
    write_universal_hu_bundle(tmp.path());
    let kind = detect_bundle_kind(tmp.path()).unwrap();
    assert_eq!(kind, BundleKind::UniversalHu);
}

#[test]
fn detect_universal_mp_eager_bundle() {
    let tmp = TempDir::new().unwrap();
    write_mp_eager_bundle(tmp.path());
    let kind = detect_bundle_kind(tmp.path()).unwrap();
    assert_eq!(kind, BundleKind::UniversalMpEager);
}

#[test]
fn detect_universal_mp_lazy_bundle() {
    let tmp = TempDir::new().unwrap();
    write_mp_lazy_bundle(tmp.path());
    let kind = detect_bundle_kind(tmp.path()).unwrap();
    assert_eq!(kind, BundleKind::UniversalMpLazy);
}

#[test]
fn detect_nested_final_universal() {
    let tmp = TempDir::new().unwrap();
    let final_dir = tmp.path().join("final");
    std::fs::create_dir_all(&final_dir).unwrap();
    write_universal_hu_bundle(&final_dir);
    let kind = detect_bundle_kind(tmp.path()).unwrap();
    assert_eq!(kind, BundleKind::UniversalHu);
}

#[test]
fn detect_nested_snapshot_universal() {
    let tmp = TempDir::new().unwrap();
    let snap_dir = tmp.path().join("snapshot_0042");
    std::fs::create_dir_all(&snap_dir).unwrap();
    write_universal_hu_bundle(&snap_dir);
    let kind = detect_bundle_kind(tmp.path()).unwrap();
    assert_eq!(kind, BundleKind::UniversalHu);
}

#[test]
fn root_mp_config_with_nested_snapshot_universal_loads_mp_lazy() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(
        tmp.path().join("config.yaml"),
        r#"
game:
  name: "MP lazy root config"
  num_players: 2
  stack_depth: 6
  allow_preflop_limp: true
  blinds:
    - { seat: 0, type: small_blind, amount: 1 }
    - { seat: 1, type: big_blind, amount: 2 }
"#,
    )
    .unwrap();
    let nested = tmp.path().join("snapshot_0001/universal");
    std::fs::create_dir_all(&nested).unwrap();
    write_mp_lazy_bundle(&nested);

    let kind = detect_bundle_kind(tmp.path()).unwrap();
    assert_eq!(kind, BundleKind::UniversalMpLazy);

    let bundle = load_bundle(tmp.path()).unwrap();
    assert_eq!(bundle.kind(), BundleKind::UniversalMpLazy);
    assert_eq!(bundle.source_backend(), "mp_lazy_sparse_projected");
}

#[test]
fn mixed_direct_old_and_nested_newer_universal_loads_newest_snapshot() {
    let tmp = TempDir::new().unwrap();

    let older_direct = tmp.path().join("snapshot_0001");
    std::fs::create_dir_all(&older_direct).unwrap();
    write_universal_hu_bundle(&older_direct);

    let newer_nested = tmp.path().join("snapshot_0002/universal");
    std::fs::create_dir_all(&newer_nested).unwrap();
    write_mp_lazy_bundle(&newer_nested);

    let kind = detect_bundle_kind(tmp.path()).unwrap();
    assert_eq!(kind, BundleKind::UniversalMpLazy);

    let bundle = load_bundle(tmp.path()).unwrap();
    assert_eq!(bundle.kind(), BundleKind::UniversalMpLazy);
    assert_eq!(bundle.num_players(), 6);
    assert_eq!(bundle.source_backend(), "mp_lazy_sparse_projected");
}

#[test]
fn detect_nested_final_legacy() {
    // Legacy bundle where strategy.bin is in final/ (already tested by write_legacy_bundle)
    let tmp = TempDir::new().unwrap();
    write_legacy_bundle(tmp.path());
    let kind = detect_bundle_kind(tmp.path()).unwrap();
    assert_eq!(kind, BundleKind::LegacyHu);
}

#[test]
fn detect_nested_snapshot_legacy() {
    let tmp = TempDir::new().unwrap();
    let config = tiny_export_config();
    let (_, strategy) = build_test_tree_and_strategy(&config);
    poker_solver_core::blueprint_v2::bundle::save_config(tmp.path(), &config).unwrap();
    let snap_dir = tmp.path().join("snapshot_0001");
    std::fs::create_dir_all(&snap_dir).unwrap();
    strategy.save(&snap_dir.join("strategy.bin")).unwrap();
    let kind = detect_bundle_kind(tmp.path()).unwrap();
    assert_eq!(kind, BundleKind::LegacyHu);
}

// ── Error detection tests ────────────────────────────────────────────

#[test]
fn detect_empty_dir_returns_not_a_bundle() {
    let tmp = TempDir::new().unwrap();
    let err = detect_bundle_kind(tmp.path()).unwrap_err();
    assert!(
        matches!(err, LoaderError::NotABundle { .. }),
        "expected NotABundle, got {err:?}"
    );
}

#[test]
fn detect_garbage_manifest_returns_manifest_parse() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("blueprint.json"), "not valid json!!!").unwrap();
    let err = detect_bundle_kind(tmp.path()).unwrap_err();
    assert!(
        matches!(err, LoaderError::ManifestParse { .. }),
        "expected ManifestParse, got {err:?}"
    );
}

#[test]
fn detect_unknown_source_backend() {
    let tmp = TempDir::new().unwrap();
    // Write a valid universal HU bundle first, then corrupt the source_backend.
    write_universal_hu_bundle(tmp.path());
    let manifest_path = tmp.path().join("blueprint.json");
    let text = std::fs::read_to_string(&manifest_path).unwrap();
    let corrupted = text.replace("hu_dense", "quantum_entangled");
    std::fs::write(&manifest_path, corrupted).unwrap();
    let err = detect_bundle_kind(tmp.path()).unwrap_err();
    assert!(
        matches!(err, LoaderError::UnknownSourceBackend { ref backend, .. } if backend == "quantum_entangled"),
        "expected UnknownSourceBackend with 'quantum_entangled', got {err:?}"
    );
}

#[test]
fn detect_namespace_mismatch() {
    let tmp = TempDir::new().unwrap();
    write_universal_hu_bundle(tmp.path());
    let manifest_path = tmp.path().join("blueprint.json");
    let text = std::fs::read_to_string(&manifest_path).unwrap();
    // Change namespace to mp_arena but keep hu_dense source_backend
    let corrupted = text.replace("hu_arena", "mp_arena");
    std::fs::write(&manifest_path, corrupted).unwrap();
    let err = detect_bundle_kind(tmp.path()).unwrap_err();
    assert!(
        matches!(err, LoaderError::NamespaceMismatch { .. }),
        "expected NamespaceMismatch, got {err:?}"
    );
}

#[test]
fn detect_missing_required_feature_for_mp_lazy() {
    let tmp = TempDir::new().unwrap();
    write_mp_lazy_bundle(tmp.path());
    let manifest_path = tmp.path().join("blueprint.json");
    let text = std::fs::read_to_string(&manifest_path).unwrap();
    // Parse, remove mp_semantic_rows_v1, rewrite
    let mut val: serde_json::Value = serde_json::from_str(&text).unwrap();
    val["required_features"] = serde_json::json!([]);
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&val).unwrap()).unwrap();
    let err = detect_bundle_kind(tmp.path()).unwrap_err();
    assert!(
        matches!(err, LoaderError::MissingRequiredFeature { .. }),
        "expected MissingRequiredFeature, got {err:?}"
    );
}

#[test]
fn load_missing_payload_file_error() {
    let tmp = TempDir::new().unwrap();
    write_universal_hu_bundle(tmp.path());
    // Delete a required payload file
    std::fs::remove_file(tmp.path().join("strategy.rows.bin")).unwrap();
    let err = load_bundle(tmp.path()).unwrap_err();
    // Should be either MissingPayloadFile (from detection) or Format (from reader)
    assert!(
        matches!(
            err,
            LoaderError::MissingPayloadFile { .. } | LoaderError::Format(..)
        ),
        "expected MissingPayloadFile or Format error, got {err:?}"
    );
}

// ── Load + query tests ───────────────────────────────────────────────

#[test]
fn load_legacy_hu_and_query() {
    let tmp = TempDir::new().unwrap();
    let (_config, tree, strategy) = write_legacy_bundle(tmp.path());
    let bundle = load_bundle(tmp.path()).unwrap();

    assert_eq!(bundle.kind(), BundleKind::LegacyHu);
    assert_eq!(bundle.num_players(), 2);

    // Query a known infoset (first decision node, bucket 0)
    let decision_map = tree.decision_index_map();
    let first_decision_arena_idx = decision_map
        .iter()
        .position(|&d| d == 0)
        .expect("should have decision node 0");

    let view = bundle
        .query_hu(first_decision_arena_idx as u32, 0)
        .expect("should find the infoset");
    let legacy_probs = strategy.get_action_probs(0, 0);
    assert_eq!(view.probs.len(), legacy_probs.len());
    for (a, b) in view.probs.iter().zip(legacy_probs.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "prob mismatch");
    }
}

#[test]
fn load_universal_hu_and_query() {
    let tmp = TempDir::new().unwrap();
    let (_config, tree, strategy) = write_universal_hu_bundle(tmp.path());
    let bundle = load_bundle(tmp.path()).unwrap();

    assert_eq!(bundle.kind(), BundleKind::UniversalHu);
    assert_eq!(bundle.num_players(), 2);
    assert_eq!(bundle.source_backend(), "hu_dense");

    // Query first decision node, bucket 0
    let decision_map = tree.decision_index_map();
    let first_decision_arena_idx = decision_map
        .iter()
        .position(|&d| d == 0)
        .expect("should have decision node 0");

    let view = bundle
        .query_hu(first_decision_arena_idx as u32, 0)
        .expect("should find the infoset");
    let legacy_probs = strategy.get_action_probs(0, 0);
    assert_eq!(view.probs.len(), legacy_probs.len());
    for (a, b) in view.probs.iter().zip(legacy_probs.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "prob mismatch");
    }
}

// ── HU Equivalence test (acceptance criterion 3) ─────────────────────

#[test]
fn hu_equivalence_legacy_vs_universal() {
    let tmp_legacy = TempDir::new().unwrap();
    let (config, tree, _strategy) = write_legacy_bundle(tmp_legacy.path());

    let tmp_univ = TempDir::new().unwrap();
    write_universal_hu_bundle(tmp_univ.path());

    let legacy_bundle = load_bundle(tmp_legacy.path()).unwrap();
    let universal_bundle = load_bundle(tmp_univ.path()).unwrap();

    assert_eq!(legacy_bundle.kind(), BundleKind::LegacyHu);
    assert_eq!(universal_bundle.kind(), BundleKind::UniversalHu);

    let bucket_counts = [
        config.clustering.preflop.buckets,
        config.clustering.flop.buckets,
        config.clustering.turn.buckets,
        config.clustering.river.buckets,
    ];

    let decision_map = tree.decision_index_map();
    let mut checked = 0usize;

    for (arena_idx, node) in tree.nodes.iter().enumerate() {
        if let GameNode::Decision { street, .. } = node {
            let di = decision_map[arena_idx];
            if di == u32::MAX {
                continue;
            }
            let street_idx = *street as usize;
            let buckets = bucket_counts[street_idx];

            for bucket in 0..buckets {
                let legacy_view = legacy_bundle
                    .query_hu(arena_idx as u32, bucket)
                    .expect("legacy should have this infoset");
                let universal_view = universal_bundle
                    .query_hu(arena_idx as u32, bucket)
                    .expect("universal should have this infoset");

                // Bitwise probability match
                assert_eq!(
                    legacy_view.probs.len(),
                    universal_view.probs.len(),
                    "action count mismatch at node {arena_idx}, bucket {bucket}"
                );
                for (i, (lp, up)) in legacy_view
                    .probs
                    .iter()
                    .zip(universal_view.probs.iter())
                    .enumerate()
                {
                    assert_eq!(
                        lp.to_bits(),
                        up.to_bits(),
                        "bitwise mismatch at node {arena_idx}, bucket {bucket}, action {i}: legacy={lp}, universal={up}"
                    );
                }

                // Action count match
                assert_eq!(
                    legacy_view.actions.len(),
                    universal_view.actions.len(),
                    "action descriptor count mismatch at node {arena_idx}, bucket {bucket}"
                );

                checked += 1;
            }
        }
    }

    assert!(checked > 0, "should have checked at least one infoset");
    eprintln!("HU equivalence verified for {checked} infosets (bitwise f32)");
}

// ── MP eager test ────────────────────────────────────────────────────

use poker_solver_core::blueprint_mp::config::*;
use poker_solver_core::blueprint_mp::game_tree::*;
use poker_solver_core::blueprint_mp::mccfr::{sample_deal, traverse_external};
use poker_solver_core::blueprint_mp::storage::MpStorage;
use poker_solver_core::blueprint_mp::{Bucket, Chips, DealWithBuckets, MAX_PLAYERS, Seat};
use poker_solver_core::blueprint_universal::mp_eager_export::{self, MpTrainingInfo};

const MP_BUCKET_COUNTS: [u16; 4] = [10, 10, 10, 10];

fn three_player_blinds() -> Vec<ForcedBet> {
    vec![
        ForcedBet {
            seat: 1,
            kind: ForcedBetKind::SmallBlind,
            amount: 1.0,
        },
        ForcedBet {
            seat: 2,
            kind: ForcedBetKind::BigBlind,
            amount: 2.0,
        },
    ]
}

fn build_3p_config() -> BlueprintMpConfig {
    let game = MpGameConfig {
        name: "3p-loader-test".to_string(),
        num_players: 3,
        stack_depth: 20.0,
        allow_preflop_limp: true,
        blinds: three_player_blinds(),
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
        preflop: MpStreetCluster {
            buckets: MP_BUCKET_COUNTS[0],
        },
        flop: MpStreetCluster {
            buckets: MP_BUCKET_COUNTS[1],
        },
        turn: MpStreetCluster {
            buckets: MP_BUCKET_COUNTS[2],
        },
        river: MpStreetCluster {
            buckets: MP_BUCKET_COUNTS[3],
        },
    };
    let training = MpTrainingConfig {
        backend: MpTrainingBackend::Eager,
        chance_continuation_mode: MpChanceContinuationMode::SampledFullDeal,
        cluster_path: None,
        iterations: Some(200),
        time_limit_minutes: None,
        lcfr_warmup_iterations: 0,
        lcfr_discount_interval: 50,
        dcfr_discount_interval_seconds: None,
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
        output_dir: "/tmp/mp_loader_test".into(),
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

fn trivial_buckets(deal: &poker_solver_core::blueprint_mp::Deal) -> DealWithBuckets {
    let mut buckets = [[Bucket(0); 4]; MAX_PLAYERS];
    for seat in 0..deal.num_players as usize {
        let card_idx = deal.hole_cards[seat][0].value as u16;
        for (street, &count) in MP_BUCKET_COUNTS.iter().enumerate() {
            buckets[seat][street] = Bucket(card_idx % count);
        }
    }
    DealWithBuckets {
        deal: deal.clone(),
        buckets,
    }
}

fn run_mp_iterations(tree: &MpGameTree, storage: &MpStorage, count: u64, seed: u64) {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let n = tree.num_players;
    for _ in 0..count {
        let deal = sample_deal(n, &mut rng);
        let buckets = trivial_buckets(&deal);
        for seat in 0..n {
            traverse_external(
                tree,
                storage,
                &buckets,
                Seat::from_raw(seat),
                tree.root,
                &mut rng,
                0.0,
                Chips::ZERO,
                false,
                0,
            );
        }
    }
}

fn write_mp_eager_bundle(dir: &Path) -> (BlueprintMpConfig, MpGameTree, MpStorage) {
    let config = build_3p_config();
    let tree = MpGameTree::build(&config.game, &config.action_abstraction);
    let storage = MpStorage::new(&tree, config.clustering.bucket_counts());
    run_mp_iterations(&tree, &storage, 200, 42);

    let training = MpTrainingInfo {
        iterations: 200,
        elapsed_minutes: 0.1,
    };
    let output =
        mp_eager_export::export_mp_strategy_to_universal(&config, &tree, &storage, &training)
            .expect("mp export should succeed");

    write_bundle(
        dir,
        &output.manifest,
        &BundleData {
            rows: &output.rows,
            actions: &output.actions,
            probs: &output.probs,
        },
    )
    .expect("write bundle");

    (config, tree, storage)
}

#[test]
fn load_mp_eager_and_query() {
    let tmp = TempDir::new().unwrap();
    let (_config, tree, storage) = write_mp_eager_bundle(tmp.path());
    let bundle = load_bundle(tmp.path()).unwrap();

    assert_eq!(bundle.kind(), BundleKind::UniversalMpEager);
    assert_eq!(bundle.num_players(), 3);
    assert_eq!(bundle.source_backend(), "mp_eager_dense");

    // Query a known infoset: find a decision node in the tree
    let first_decision = tree
        .nodes
        .iter()
        .enumerate()
        .find_map(|(i, n)| match n {
            MpGameNode::Decision {
                seat,
                street,
                actions,
                ..
            } => Some((i as u32, seat.index() as u8, *street as u8, actions.len())),
            _ => None,
        })
        .expect("should have at least one decision node");

    let (node_idx, seat, _street, num_actions) = first_decision;
    let view = bundle
        .query_mp_eager(node_idx, seat, 0)
        .expect("should find the infoset");
    assert_eq!(view.probs.len(), num_actions);

    // Verify probabilities match MpStorage
    let mut avg_buf = vec![0.0_f64; 32];
    storage.average_strategy(node_idx, 0, num_actions, &mut avg_buf);
    for (i, &p) in view.probs.iter().enumerate() {
        let expected = avg_buf[i] as f32;
        assert_eq!(
            p.to_bits(),
            expected.to_bits(),
            "prob mismatch at action {i}: got {p}, expected {expected}"
        );
    }
}

// ── MP lazy test ─────────────────────────────────────────────────────

use poker_solver_core::blueprint_mp::Street as MpStreet;
use poker_solver_core::blueprint_mp::sparse_storage::{
    MpInfosetKey, SparseActionDescriptor, SparseActionKind, SparseSnapshotEntry,
};
use poker_solver_core::blueprint_universal::mp_lazy_export::{
    self, LazyExportConfig, LazyTrainingInfo,
};

fn make_lazy_key(
    seat: u8,
    street: MpStreet,
    local_bucket: u16,
    history_hi: u64,
    history_lo: u64,
    history_hash: u64,
    history_len: u16,
) -> MpInfosetKey {
    MpInfosetKey::from_street_bucket(
        Seat::from_raw(seat),
        street,
        local_bucket,
        history_hi,
        history_lo,
        history_hash,
        history_len,
    )
}

fn lazy_action(
    kind: SparseActionKind,
    amount_chips: u32,
    source_action_index: u16,
) -> SparseActionDescriptor {
    SparseActionDescriptor {
        kind,
        amount_chips,
        source_action_index,
    }
}

fn build_lazy_test_entries() -> Vec<SparseSnapshotEntry> {
    vec![
        SparseSnapshotEntry {
            key: make_lazy_key(0, MpStreet::Preflop, 5, 0x1234, 0x5678, 0xAAAA, 4),
            num_actions: 3,
            action_identity: Some(vec![
                lazy_action(SparseActionKind::Fold, 0, 0),
                lazy_action(SparseActionKind::Call, 0, 1),
                lazy_action(SparseActionKind::Raise, 6, 2),
            ]),
            regrets: vec![10, -5, 3],
            strategy_sums: vec![100, 200, 300],
        },
        SparseSnapshotEntry {
            key: make_lazy_key(1, MpStreet::Flop, 10, 0xABCD, 0xEF01, 0xBBBB, 16),
            num_actions: 2,
            action_identity: Some(vec![
                lazy_action(SparseActionKind::Check, 0, 0),
                lazy_action(SparseActionKind::Lead, 20, 1),
            ]),
            regrets: vec![20, -10],
            strategy_sums: vec![500, 500],
        },
        // Zero-mass entry
        SparseSnapshotEntry {
            key: make_lazy_key(0, MpStreet::Turn, 3, 0x1111, 0x2222, 0xCCCC, 8),
            num_actions: 2,
            action_identity: Some(vec![
                lazy_action(SparseActionKind::Check, 0, 0),
                lazy_action(SparseActionKind::AllInCall, 0, 1),
            ]),
            regrets: vec![0, 0],
            strategy_sums: vec![0, 0],
        },
    ]
}

fn write_mp_lazy_bundle(dir: &Path) -> Vec<SparseSnapshotEntry> {
    let entries = build_lazy_test_entries();
    write_mp_lazy_bundle_with_entries(dir, &entries);
    entries
}

fn write_mp_lazy_bundle_with_entries(dir: &Path, entries: &[SparseSnapshotEntry]) {
    let config = LazyExportConfig {
        num_players: 6,
        stack_depth: 100.0,
        bucket_counts: [169, 100, 50, 50],
        small_blind: 1.0,
        big_blind: 2.0,
    };
    let training = LazyTrainingInfo {
        iterations: 100,
        elapsed_minutes: 1.0,
    };
    let output = mp_lazy_export::export_lazy_sparse_to_universal(&config, entries, &training);
    mp_lazy_export::write_lazy_bundle(dir, &output).unwrap();
}

fn append_trailing_payload_byte(dir: &Path, name: &str) {
    let path = dir.join(name);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.push(0xA5);
    std::fs::write(&path, &bytes).unwrap();

    let manifest_path = dir.join("blueprint.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
    let entry = manifest
        .get_mut("files")
        .and_then(|files| files.get_mut(name))
        .unwrap();
    entry["size"] = serde_json::json!(bytes.len());
    entry["sha256"] = serde_json::json!(hex::encode(Sha256::digest(&bytes)));
    std::fs::write(
        manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
}

#[test]
fn load_mp_lazy_and_query() {
    let tmp = TempDir::new().unwrap();
    let _entries = write_mp_lazy_bundle(tmp.path());
    let bundle = load_bundle(tmp.path()).unwrap();

    assert_eq!(bundle.kind(), BundleKind::UniversalMpLazy);
    assert_eq!(bundle.num_players(), 6);
    assert_eq!(bundle.source_backend(), "mp_lazy_sparse_projected");

    // Query by semantic key using MpLazyKey struct
    let key = MpLazyKey {
        seat: 0,
        street: 0,
        local_bucket: 5,
        history_hash: 0xAAAA,
        history_len: 4,
    };
    let view: InfosetView<'_> = bundle
        .query_mp_lazy(&key)
        .expect("should find the lazy infoset");
    assert_eq!(view.probs.len(), 3);

    // Verify probabilities: strategy_sums [100, 200, 300] -> normalized
    let total: f64 = 100.0 + 200.0 + 300.0;
    for (i, &p) in view.probs.iter().enumerate() {
        let expected = [100.0, 200.0, 300.0][i] / total;
        assert!(
            (p - expected as f32).abs() < 1e-6,
            "prob mismatch at action {i}: got {p}, expected {expected}"
        );
    }

    assert_eq!(view.actions[0].kind, ActionKind::Fold);
    assert_eq!(view.actions[1].kind, ActionKind::Call);
    assert_eq!(view.actions[2].kind, ActionKind::Raise);
    assert_eq!(view.actions[2].amount_chips, 6);
    assert!(view.actions[2].is_aggressive);

    let owned = bundle
        .query_mp_lazy_owned(&key)
        .expect("owned query should find the same lazy infoset");
    assert_eq!(
        view.probs
            .iter()
            .map(|prob| prob.to_bits())
            .collect::<Vec<_>>(),
        owned
            .probs
            .iter()
            .map(|prob| prob.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(view.actions, owned.actions);
}

#[test]
fn load_mp_lazy_accepts_trailing_payload_bytes() {
    let tmp = TempDir::new().unwrap();
    write_mp_lazy_bundle(tmp.path());
    append_trailing_payload_byte(tmp.path(), "strategy.probs.f32.bin");

    // BundleReader has always treated bytes after header.payload_len as
    // compatible trailing data; the lazy reader must preserve that contract.
    BundleReader::open(tmp.path()).expect("eager reader accepts trailing bytes");
    let bundle = load_bundle(tmp.path()).expect("lazy reader accepts trailing bytes");
    let view = bundle
        .query_mp_lazy(&MpLazyKey {
            seat: 0,
            street: 0,
            local_bucket: 5,
            history_hash: 0xAAAA,
            history_len: 4,
        })
        .expect("row remains queryable");
    assert_eq!(view.probs.len(), 3);
}

#[test]
fn load_mp_lazy_zero_mass_uniform() {
    let tmp = TempDir::new().unwrap();
    let _entries = write_mp_lazy_bundle(tmp.path());
    let bundle = load_bundle(tmp.path()).unwrap();

    // Zero-mass entry: seat 0, turn, bucket 3
    let key = MpLazyKey {
        seat: 0,
        street: 2,
        local_bucket: 3,
        history_hash: 0xCCCC,
        history_len: 8,
    };
    let view = bundle
        .query_mp_lazy(&key)
        .expect("should find the zero-mass infoset");
    assert_eq!(view.probs.len(), 2);

    // Should be uniform (0.5, 0.5)
    for &p in view.probs {
        assert!(
            (p - 0.5).abs() < 1e-6,
            "zero-mass row should be uniform, got {p}"
        );
    }
    assert_eq!(view.actions[1].kind, ActionKind::AllInCall);
    assert!(!view.actions[1].is_aggressive);
}

#[test]
fn load_mp_lazy_matches_multiple_histories_in_one_public_prefix() {
    let tmp = TempDir::new().unwrap();
    let entries = vec![
        SparseSnapshotEntry {
            key: make_lazy_key(0, MpStreet::Preflop, 5, 1, 2, 0xDDDD, 6),
            num_actions: 2,
            action_identity: Some(vec![
                lazy_action(SparseActionKind::Check, 0, 0),
                lazy_action(SparseActionKind::Lead, 10, 1),
            ]),
            regrets: vec![0, 0],
            strategy_sums: vec![1, 3],
        },
        SparseSnapshotEntry {
            key: make_lazy_key(0, MpStreet::Preflop, 5, 3, 4, 0xEEEE, 7),
            num_actions: 2,
            action_identity: Some(vec![
                lazy_action(SparseActionKind::Fold, 0, 0),
                lazy_action(SparseActionKind::Call, 0, 1),
            ]),
            regrets: vec![0, 0],
            strategy_sums: vec![4, 1],
        },
    ];
    write_mp_lazy_bundle_with_entries(tmp.path(), &entries);
    let bundle = load_bundle(tmp.path()).unwrap();

    let first = bundle
        .query_mp_lazy(&MpLazyKey {
            seat: 0,
            street: 0,
            local_bucket: 5,
            history_hash: 0xDDDD,
            history_len: 6,
        })
        .unwrap();
    assert_eq!(first.probs, &[0.25, 0.75]);
    assert_eq!(first.actions[1].kind, ActionKind::Bet);
    assert_eq!(first.actions[1].amount_chips, 10);

    let second = bundle
        .query_mp_lazy(&MpLazyKey {
            seat: 0,
            street: 0,
            local_bucket: 5,
            history_hash: 0xEEEE,
            history_len: 7,
        })
        .unwrap();
    assert_eq!(second.probs, &[0.8, 0.2]);
    assert_eq!(second.actions[0].kind, ActionKind::Fold);
    assert_eq!(second.actions[1].kind, ActionKind::Call);
}

#[test]
fn load_mp_lazy_duplicate_hash_and_length_returns_last_serialized_row() {
    let tmp = TempDir::new().unwrap();
    let entries = vec![
        SparseSnapshotEntry {
            key: make_lazy_key(0, MpStreet::Preflop, 5, 1, 2, 0xABCD, 9),
            num_actions: 2,
            action_identity: Some(vec![
                lazy_action(SparseActionKind::Check, 0, 0),
                lazy_action(SparseActionKind::Call, 0, 1),
            ]),
            regrets: vec![0, 0],
            strategy_sums: vec![10, 0],
        },
        SparseSnapshotEntry {
            key: make_lazy_key(0, MpStreet::Preflop, 5, 3, 4, 0xABCD, 9),
            num_actions: 2,
            action_identity: Some(vec![
                lazy_action(SparseActionKind::Fold, 0, 0),
                lazy_action(SparseActionKind::Raise, 40, 1),
            ]),
            regrets: vec![0, 0],
            strategy_sums: vec![0, 10],
        },
    ];
    write_mp_lazy_bundle_with_entries(tmp.path(), &entries);

    let reader = BundleReader::open(tmp.path()).unwrap();
    let mut last_matching_row = None;
    for i in 0..reader.row_count() {
        let row = reader.row(i).unwrap();
        let semantic = reader.semantic_record(i).unwrap();
        if row.seat == 0
            && row.street == 0
            && row.local_bucket == 5
            && semantic.history_hash == 0xABCD
            && semantic.history_len == 9
        {
            last_matching_row = Some(i);
        }
    }
    let last_matching_row = last_matching_row.unwrap();
    let expected_row = reader.row(last_matching_row).unwrap();
    let expected_probs = reader
        .prob_slice(
            expected_row.prob_offset as usize,
            expected_row.action_count as usize,
        )
        .unwrap();
    let expected_actions: Vec<_> = (0..usize::from(expected_row.action_count))
        .map(|i| {
            reader
                .action(expected_row.action_offset as usize + i)
                .unwrap()
                .clone()
        })
        .collect();

    let bundle = load_bundle(tmp.path()).unwrap();
    let view = bundle
        .query_mp_lazy(&MpLazyKey {
            seat: 0,
            street: 0,
            local_bucket: 5,
            history_hash: 0xABCD,
            history_len: 9,
        })
        .unwrap();
    assert_eq!(view.probs, expected_probs);
    assert_eq!(
        view.probs
            .iter()
            .map(|prob| prob.to_bits())
            .collect::<Vec<_>>(),
        expected_probs
            .iter()
            .map(|prob| prob.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(view.actions, expected_actions);
    assert_eq!(view.actions[0].kind, ActionKind::Fold);
    assert_eq!(view.actions[1].amount_chips, 40);
}

#[test]
fn load_mp_lazy_rejects_payload_checksum_mismatch() {
    let tmp = TempDir::new().unwrap();
    write_mp_lazy_bundle(tmp.path());

    let path = tmp.path().join("strategy.probs.f32.bin");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[48] ^= 0x01;
    std::fs::write(&path, bytes).unwrap();

    let error = load_bundle(tmp.path()).unwrap_err();
    assert!(matches!(
        error,
        LoaderError::Format(FormatError::ChecksumMismatch { .. })
    ));
}

#[test]
fn load_mp_lazy_query_returns_none_after_payload_truncation() {
    let tmp = TempDir::new().unwrap();
    write_mp_lazy_bundle(tmp.path());
    let bundle = load_bundle(tmp.path()).unwrap();
    let path = tmp.path().join("strategy.probs.f32.bin");

    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_len(0)
        .unwrap();

    assert!(
        bundle
            .query_mp_lazy(&MpLazyKey {
                seat: 0,
                street: 0,
                local_bucket: 5,
                history_hash: 0xAAAA,
                history_len: 4,
            })
            .is_none()
    );
}

// ── Detection precedence regression test ────────────────────────────

#[test]
fn universal_bundle_with_config_yaml_detected_as_universal() {
    // A universal HU bundle that also contains a retained config.yaml
    // (the spec permits an optional retained config in universal bundles)
    // must be detected as UniversalHu, NOT LegacyHu.
    // This guards the "blueprint.json wins over config.yaml" precedence.
    let tmp = TempDir::new().unwrap();
    write_universal_hu_bundle(tmp.path());

    // Now also write a config.yaml into the same bundle directory.
    let config = tiny_export_config();
    poker_solver_core::blueprint_v2::bundle::save_config(tmp.path(), &config).unwrap();

    // Detection must still return UniversalHu.
    let kind = detect_bundle_kind(tmp.path()).unwrap();
    assert_eq!(
        kind,
        BundleKind::UniversalHu,
        "blueprint.json must take precedence over config.yaml"
    );

    // Loading must also produce a universal bundle, not legacy.
    let bundle = load_bundle(tmp.path()).unwrap();
    assert_eq!(
        bundle.kind(),
        BundleKind::UniversalHu,
        "load_bundle must load as universal when blueprint.json is present"
    );
    assert_eq!(bundle.source_backend(), "hu_dense");
}

#[test]
fn universal_in_final_with_config_yaml_at_root_detected_as_universal() {
    // Universal bundle lives in final/, config.yaml at root.
    // blueprint.json in final/ must still win over config.yaml at root.
    let tmp = TempDir::new().unwrap();
    let final_dir = tmp.path().join("final");
    std::fs::create_dir_all(&final_dir).unwrap();
    write_universal_hu_bundle(&final_dir);

    // Write config.yaml at the root (not in final/).
    let config = tiny_export_config();
    poker_solver_core::blueprint_v2::bundle::save_config(tmp.path(), &config).unwrap();

    let kind = detect_bundle_kind(tmp.path()).unwrap();
    assert_eq!(
        kind,
        BundleKind::UniversalHu,
        "blueprint.json in final/ must take precedence over config.yaml at root"
    );
}

#[test]
fn mp_lazy_key_not_found_returns_none() {
    let tmp = TempDir::new().unwrap();
    write_mp_lazy_bundle(tmp.path());
    let bundle = load_bundle(tmp.path()).unwrap();

    let key = MpLazyKey {
        seat: 5,
        street: 3,
        local_bucket: 999,
        history_hash: 0xDEAD,
        history_len: 99,
    };
    assert!(bundle.query_mp_lazy(&key).is_none());
}
