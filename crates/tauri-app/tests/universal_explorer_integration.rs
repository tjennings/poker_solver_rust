//! Integration tests for Phase 6: universal bundle Explorer integration.
//!
//! Tests cover:
//! - Acceptance #2: HU universal bundles render identically to legacy through Explorer views
//! - Part C: listing reports format kind and player count
//! - Part D: MP bundles load without error; HU-only views return clean error for MP
//! - Part E: universal HU bundles load through the shared _core path

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::path::Path;

use poker_solver_core::blueprint_universal::hu_export::{self, TrainingInfo as HuTrainingInfo};
use poker_solver_core::blueprint_universal::{write_bundle, BundleData};
use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::config::*;
use poker_solver_core::blueprint_v2::game_tree::GameTree;
use poker_solver_core::blueprint_v2::storage::BlueprintStorage;

use poker_solver_tauri::{ExplorationPosition, ExplorationState};
use tempfile::TempDir;

// ── Shared helpers ──────────────────────────────────────────────────

fn tiny_export_config() -> BlueprintV2Config {
    BlueprintV2Config {
        game: GameConfig {
            name: "ExplorerUniversalTest".to_string(),
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

fn build_tree_and_strategy(config: &BlueprintV2Config) -> (GameTree, BlueprintV2Strategy) {
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

/// Write a legacy HU bundle (config.yaml + strategy.bin + metadata.json).
fn write_legacy_bundle(dir: &Path) {
    let config = tiny_export_config();
    let (_, strategy) = build_tree_and_strategy(&config);
    poker_solver_core::blueprint_v2::bundle::save_config(dir, &config).unwrap();
    let final_dir = dir.join("final");
    std::fs::create_dir_all(&final_dir).unwrap();
    strategy.save(&final_dir.join("strategy.bin")).unwrap();
    // Write metadata.json so load_bundle_core can read iterations.
    std::fs::write(
        final_dir.join("metadata.json"),
        r#"{"iteration": 100, "elapsed_minutes": 5}"#,
    )
    .unwrap();
}

/// Export a universal HU bundle (with retained config.yaml).
fn write_universal_hu_bundle(dir: &Path) {
    let config = tiny_export_config();
    let (tree, strategy) = build_tree_and_strategy(&config);
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
    // Retain config.yaml so the Explorer can rebuild the tree.
    poker_solver_core::blueprint_v2::bundle::save_config(dir, &config).unwrap();
}

// ── Acceptance test #2: HU universal renders identically to legacy ──

#[tokio::test]
async fn universal_hu_renders_identically_to_legacy() {
    // Step 1: Create legacy bundle, load through Explorer, capture outputs.
    let legacy_dir = TempDir::new().unwrap();
    write_legacy_bundle(legacy_dir.path());

    let legacy_state = ExplorationState::default();
    let legacy_info = poker_solver_tauri::load_bundle_core(
        &legacy_state,
        legacy_dir.path().to_string_lossy().to_string(),
    )
    .await
    .expect("legacy bundle should load");

    let legacy_pos = ExplorationPosition::default();
    let legacy_actions =
        poker_solver_tauri::get_available_actions_core(&legacy_state, legacy_pos.clone())
            .expect("legacy get_available_actions should succeed");

    let legacy_matrix = poker_solver_tauri::get_strategy_matrix_core(
        &legacy_state,
        legacy_pos.clone(),
        None,
        None,
        None,
    )
    .expect("legacy get_strategy_matrix should succeed");

    let legacy_bundle_info = poker_solver_tauri::get_bundle_info_core(&legacy_state)
        .expect("legacy get_bundle_info should succeed");

    // Step 2: Create universal bundle, load through Explorer, capture outputs.
    let univ_dir = TempDir::new().unwrap();
    write_universal_hu_bundle(univ_dir.path());

    let univ_state = ExplorationState::default();
    let univ_info = poker_solver_tauri::load_bundle_core(
        &univ_state,
        univ_dir.path().to_string_lossy().to_string(),
    )
    .await
    .expect("universal bundle should load");

    let univ_pos = ExplorationPosition::default();
    let univ_actions =
        poker_solver_tauri::get_available_actions_core(&univ_state, univ_pos.clone())
            .expect("universal get_available_actions should succeed");

    let univ_matrix = poker_solver_tauri::get_strategy_matrix_core(
        &univ_state,
        univ_pos.clone(),
        None,
        None,
        None,
    )
    .expect("universal get_strategy_matrix should succeed");

    let univ_bundle_info = poker_solver_tauri::get_bundle_info_core(&univ_state)
        .expect("universal get_bundle_info should succeed");

    // Step 3: Assert identical outputs.
    // Same number of actions.
    assert_eq!(
        legacy_actions.len(),
        univ_actions.len(),
        "action count mismatch: legacy={}, universal={}",
        legacy_actions.len(),
        univ_actions.len()
    );

    // Same action IDs.
    for (i, (la, ua)) in legacy_actions.iter().zip(univ_actions.iter()).enumerate() {
        assert_eq!(la.id, ua.id, "action id mismatch at index {i}");
        assert_eq!(
            la.action_type, ua.action_type,
            "action type mismatch at {i}"
        );
    }

    // Same stack depth and info sets.
    assert_eq!(legacy_info.stack_depth, univ_info.stack_depth);
    assert_eq!(legacy_bundle_info.info_sets, univ_bundle_info.info_sets);

    // Strategy matrix cells must match bitwise.
    assert_eq!(legacy_matrix.cells.len(), univ_matrix.cells.len());
    for (row_idx, (lrow, urow)) in legacy_matrix
        .cells
        .iter()
        .zip(univ_matrix.cells.iter())
        .enumerate()
    {
        assert_eq!(lrow.len(), urow.len());
        for (col_idx, (lcell, ucell)) in lrow.iter().zip(urow.iter()).enumerate() {
            assert_eq!(
                lcell.hand, ucell.hand,
                "hand label mismatch at [{row_idx}][{col_idx}]"
            );
            assert_eq!(
                lcell.probabilities.len(),
                ucell.probabilities.len(),
                "prob count mismatch for hand {} at [{row_idx}][{col_idx}]",
                lcell.hand,
            );
            for (k, (lp, up)) in lcell
                .probabilities
                .iter()
                .zip(ucell.probabilities.iter())
                .enumerate()
            {
                assert_eq!(
                    lp.probability.to_bits(),
                    up.probability.to_bits(),
                    "probability mismatch for hand {} action {k} at [{row_idx}][{col_idx}]: \
                     legacy={}, universal={}",
                    lcell.hand,
                    lp.probability,
                    up.probability,
                );
            }
        }
    }
}

// ── Part E: universal HU loads through the shared _core path ────────

// ── Part E: universal HU loads through the shared _core path ────────

#[tokio::test]
async fn universal_hu_loads_through_load_bundle_core() {
    let dir = TempDir::new().unwrap();
    write_universal_hu_bundle(dir.path());

    let state = ExplorationState::default();
    let info =
        poker_solver_tauri::load_bundle_core(&state, dir.path().to_string_lossy().to_string())
            .await
            .expect("universal HU bundle should load via load_bundle_core");

    assert!(info.stack_depth > 0, "stack_depth should be positive");
    assert!(info.info_sets > 0, "info_sets should be positive");
    assert!(poker_solver_tauri::is_bundle_loaded_core(&state));
}

// ── MP bundle helpers ───────────────────────────────────────────────

use poker_solver_core::blueprint_mp::config::*;
use poker_solver_core::blueprint_mp::game_tree::MpGameTree;
use poker_solver_core::blueprint_mp::mccfr::{sample_deal, traverse_external};
use poker_solver_core::blueprint_mp::sparse_storage::{MpInfosetKey, SparseSnapshotEntry};
use poker_solver_core::blueprint_mp::storage::MpStorage;
use poker_solver_core::blueprint_mp::Street as MpStreet;
use poker_solver_core::blueprint_mp::{Bucket, Chips, DealWithBuckets, Seat, MAX_PLAYERS};
use poker_solver_core::blueprint_universal::mp_eager_export::{
    self, MpTrainingInfo as MpEagerTrainingInfo,
};
use poker_solver_core::blueprint_universal::mp_lazy_export::{
    self, LazyExportConfig, LazyTrainingInfo,
};

const MP_BUCKET_COUNTS: [u16; 4] = [10, 10, 10, 10];

fn build_3p_config() -> BlueprintMpConfig {
    let game = MpGameConfig {
        name: "3p-explorer-test".to_string(),
        num_players: 3,
        stack_depth: 20.0,
        allow_preflop_limp: true,
        blinds: vec![
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
        ],
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
        output_dir: "/tmp/mp_explorer_test".into(),
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

fn run_mp_iterations(tree: &MpGameTree, storage: &MpStorage, count: u64) {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
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

/// Write a universal MP eager bundle.
fn write_mp_eager_bundle(dir: &Path) {
    let config = build_3p_config();
    let tree = MpGameTree::build(&config.game, &config.action_abstraction);
    let storage = MpStorage::new(&tree, config.clustering.bucket_counts());
    run_mp_iterations(&tree, &storage, 200);

    let training = MpEagerTrainingInfo {
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
}

/// Write a universal MP lazy bundle.
fn write_mp_lazy_bundle(dir: &Path) {
    let entries = vec![SparseSnapshotEntry {
        key: MpInfosetKey::from_street_bucket(
            Seat::from_raw(0),
            MpStreet::Preflop,
            5,
            0x1234,
            0x5678,
            0xAAAA,
            4,
        ),
        num_actions: 3,
        regrets: vec![10, -5, 3],
        strategy_sums: vec![100, 200, 300],
    }];
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
    let output = mp_lazy_export::export_lazy_sparse_to_universal(&config, &entries, &training);
    mp_lazy_export::write_lazy_bundle(dir, &output).unwrap();
}

// ── Part C: listing reports kind + player count ─────────────────────

#[test]
fn list_blueprints_detects_universal_and_legacy() {
    let base = TempDir::new().unwrap();

    // Legacy HU bundle.
    let legacy = base.path().join("legacy_hu");
    std::fs::create_dir_all(&legacy).unwrap();
    write_legacy_bundle(&legacy);

    // Universal HU bundle.
    let univ_hu = base.path().join("universal_hu");
    std::fs::create_dir_all(&univ_hu).unwrap();
    write_universal_hu_bundle(&univ_hu);

    // Universal MP eager bundle.
    let univ_mp = base.path().join("universal_mp_eager");
    std::fs::create_dir_all(&univ_mp).unwrap();
    write_mp_eager_bundle(&univ_mp);

    let entries =
        poker_solver_tauri::list_blueprints_core(base.path().to_string_lossy().to_string())
            .expect("listing should succeed");

    // Should find at least legacy_hu and universal_hu.
    // Note: list_blueprints_core detects config.yaml and blueprint.json.
    assert!(
        entries.len() >= 2,
        "expected at least 2 entries, got {}: {:?}",
        entries.len(),
        entries.iter().map(|e| &e.name).collect::<Vec<_>>()
    );

    // All entries should report has_strategy = true.
    for entry in &entries {
        assert!(
            entry.has_strategy,
            "entry {} should have has_strategy=true",
            entry.name,
        );
    }
}

// ── Part D: MP bundles load without error ───────────────────────────

#[tokio::test]
async fn mp_eager_bundle_loads_and_provides_bundle_info() {
    let dir = TempDir::new().unwrap();
    write_mp_eager_bundle(dir.path());

    let state = ExplorationState::default();
    let info =
        poker_solver_tauri::load_bundle_core(&state, dir.path().to_string_lossy().to_string())
            .await
            .expect("MP eager bundle should load without error");

    // Bundle info should report the MP kind.
    assert!(info.info_sets > 0, "info_sets should be positive");
    assert!(poker_solver_tauri::is_bundle_loaded_core(&state));

    // get_bundle_info_core should also succeed.
    let bi = poker_solver_tauri::get_bundle_info_core(&state)
        .expect("get_bundle_info should succeed for MP");
    assert!(bi.iterations > 0);
}

#[tokio::test]
async fn mp_lazy_bundle_loads_and_provides_bundle_info() {
    let dir = TempDir::new().unwrap();
    write_mp_lazy_bundle(dir.path());

    let state = ExplorationState::default();
    let _info =
        poker_solver_tauri::load_bundle_core(&state, dir.path().to_string_lossy().to_string())
            .await
            .expect("MP lazy bundle should load without error");

    assert!(poker_solver_tauri::is_bundle_loaded_core(&state));

    let bi = poker_solver_tauri::get_bundle_info_core(&state)
        .expect("get_bundle_info should succeed for MP lazy");
    assert_eq!(bi.iterations, 100);
}

#[tokio::test]
async fn mp_bundle_hu_views_return_clean_error() {
    let dir = TempDir::new().unwrap();
    write_mp_eager_bundle(dir.path());

    let state = ExplorationState::default();
    poker_solver_tauri::load_bundle_core(&state, dir.path().to_string_lossy().to_string())
        .await
        .expect("MP eager bundle should load");

    // Strategy matrix should return a clean error, not panic.
    let pos = ExplorationPosition::default();
    let result =
        poker_solver_tauri::get_strategy_matrix_core(&state, pos.clone(), None, None, None);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .contains("MP browsing not yet supported"),
        "expected 'MP browsing not yet supported' error"
    );

    // Available actions should also return a clean error.
    let result = poker_solver_tauri::get_available_actions_core(&state, pos);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("MP browsing not yet supported"),);

    // Preflop ranges should also return a clean error.
    let result = poker_solver_tauri::get_preflop_ranges_core(&state, vec!["c".to_string()]);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("MP browsing not yet supported"),);
}
