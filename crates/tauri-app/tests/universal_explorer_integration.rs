//! Integration tests for Phase 6: universal bundle Explorer integration.
//!
//! Tests cover:
//! - Acceptance #2: HU universal bundles render identically to legacy through Explorer views
//! - Part C: listing reports format kind and player count
//! - Part D: MP bundles load without error; HU-only views return clean error for MP
//! - Part E: universal HU bundles load through the shared _core path

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::path::Path;

use poker_solver_core::abstraction::isomorphism::CanonicalBoard;
use poker_solver_core::blueprint_mp::lazy_mccfr::{LazyMpGame, LazyResolvedSpot};
use poker_solver_core::blueprint_universal::hu_export::{self, TrainingInfo as HuTrainingInfo};
use poker_solver_core::blueprint_universal::{write_bundle, BundleData};
use poker_solver_core::blueprint_v2::bucket_file::{BucketFile, BucketFileHeader, PackedBoard};
use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::config::*;
use poker_solver_core::blueprint_v2::game_tree::GameTree;
use poker_solver_core::blueprint_v2::storage::BlueprintStorage;
use poker_solver_core::poker::{Card, Suit, Value};

use poker_solver_tauri::{
    game_back_core, game_deal_card_core, game_get_state_core, game_new_core, game_play_action_core,
    ExplorationPosition, ExplorationState, GameSessionState, PostflopState,
};
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
use poker_solver_core::blueprint_mp::game_tree::TreeAction;
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
        action_identity: None,
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

/// Write a two-player lazy bundle plus the retained trainer config needed by
/// the Tauri lazy session adapter.
fn write_mp_lazy_2p_bundle_with_config(dir: &Path) {
    write_mp_lazy_2p_bundle_with_options(dir, true, true);
}

fn write_mp_lazy_2p_bundle_with_anomaly_rows(dir: &Path) {
    write_mp_lazy_2p_bundle_with_options_and_anomaly(dir, true, true, true);
}

fn write_mp_lazy_2p_bundle_with_options(
    dir: &Path,
    include_flop_buckets: bool,
    include_flop_rows: bool,
) {
    write_mp_lazy_2p_bundle_with_options_and_anomaly(
        dir,
        include_flop_buckets,
        include_flop_rows,
        false,
    );
}

fn write_mp_lazy_2p_bundle_with_options_and_anomaly(
    dir: &Path,
    include_flop_buckets: bool,
    include_flop_rows: bool,
    include_anomaly_rows: bool,
) {
    let mut config = build_3p_config();
    config.game.name = "2p-lazy-session-test".to_string();
    config.game.num_players = 2;
    config.game.blinds = vec![
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
    config.training.backend = MpTrainingBackend::LazySparse;
    if include_anomaly_rows {
        config.action_abstraction.preflop =
            serde_yaml::from_str("lead: [\"2bb\"]\nraise: [[\"3bb\"]]")
                .expect("valid anomaly preflop action sizes");
    }
    config.action_abstraction.flop =
        serde_yaml::from_str("lead: [1.0]\nraise: [[1.0]]").expect("valid test flop action sizes");
    config.action_abstraction.turn = config.action_abstraction.flop.clone();
    config.action_abstraction.river = config.action_abstraction.flop.clone();
    std::fs::write(
        dir.join("config.yaml"),
        serde_yaml::to_string(&config).expect("serialize MP config"),
    )
    .expect("write MP config");

    let game = LazyMpGame::new(&config.game, &config.action_abstraction);
    let root = LazyResolvedSpot::root(&game);
    let mut entries = Vec::new();
    let mut add_preflop_rows = |spot: LazyResolvedSpot, sentinel: Option<&[u64]>| {
        let actions = spot.actions(&game);
        for bucket in 0..config.clustering.preflop.buckets {
            let key = spot.key_for_bucket(bucket);
            if entries
                .iter()
                .any(|entry: &SparseSnapshotEntry| entry.key == key)
            {
                continue;
            }
            let strategy_sums = if bucket == MP_BUCKET_COUNTS[0] - 1 {
                sentinel.map_or_else(
                    || {
                        (0..actions.len())
                            .map(|index| if index == 0 { u64::from(bucket) + 1 } else { 1 })
                            .collect()
                    },
                    |values| {
                        assert_eq!(values.len(), actions.len());
                        values.to_vec()
                    },
                )
            } else {
                (0..actions.len())
                    .map(|index| if index == 0 { u64::from(bucket) + 1 } else { 1 })
                    .collect()
            };
            entries.push(SparseSnapshotEntry {
                key,
                num_actions: actions.len() as u8,
                action_identity: None,
                regrets: vec![0; actions.len()],
                strategy_sums,
            });
        }
    };

    if include_anomaly_rows {
        let sb_raise_index = root
            .actions(&game)
            .iter()
            .position(|action| matches!(action, TreeAction::Lead(_)))
            .expect("anomaly root should offer an opening raise");
        let sb_raise = root
            .advance(&game, sb_raise_index)
            .expect("SB raise should reach BB decision");
        let bb_reraise_index = sb_raise
            .actions(&game)
            .iter()
            .position(|action| matches!(action, TreeAction::Raise(_)))
            .expect("BB should offer a 3bb reraise");
        let bb_reraise = sb_raise
            .advance(&game, bb_reraise_index)
            .expect("BB reraise should reach SB decision");

        add_preflop_rows(root, Some(&[1, 2, 7]));
        add_preflop_rows(sb_raise, Some(&[2, 3, 4, 1]));
        add_preflop_rows(bb_reraise, Some(&[7, 2, 1]));
    } else {
        let mut preflop_spot = root;
        for action_index in [1, 0] {
            add_preflop_rows(preflop_spot, None);
            let Some(next) = preflop_spot.advance(&game, action_index) else {
                break;
            };
            preflop_spot = next;
            if preflop_spot.street() != MpStreet::Preflop {
                break;
            }
        }
    }
    if include_flop_rows {
        let mut flop_spot = root
            .advance(&game, 1)
            .and_then(|spot| spot.advance(&game, 0))
            .expect("preflop call should reach the flop boundary");
        for _ in 0..3 {
            let flop_actions = flop_spot.actions(&game);
            for bucket in 0..=1 {
                entries.push(SparseSnapshotEntry {
                    key: flop_spot.key_for_bucket(bucket),
                    num_actions: flop_actions.len() as u8,
                    action_identity: None,
                    regrets: vec![0; flop_actions.len()],
                    strategy_sums: (0..flop_actions.len())
                        .map(|index| {
                            if bucket == 0 {
                                100 + index as u64 * 100
                            } else {
                                300 - index as u64 * 100
                            }
                        })
                        .collect(),
                });
            }
            let Some(next) = flop_spot.advance(&game, 0) else {
                break;
            };
            flop_spot = next;
        }
    }
    let export_config = LazyExportConfig {
        num_players: 2,
        stack_depth: config.game.stack_depth,
        bucket_counts: config.clustering.bucket_counts(),
        small_blind: 1.0,
        big_blind: 2.0,
    };
    let training = LazyTrainingInfo {
        iterations: 100,
        elapsed_minutes: 1.0,
    };
    let output =
        mp_lazy_export::export_lazy_sparse_to_universal(&export_config, &entries, &training);
    mp_lazy_export::write_lazy_bundle(dir, &output).expect("write 2p lazy bundle");

    if include_flop_buckets {
        write_test_flop_bucket(&dir.join("buckets"), config.clustering.flop.buckets);
    }
}

fn write_test_flop_bucket(dir: &Path, bucket_count: u16) {
    write_test_flop_bucket_with_assignment(dir, bucket_count, 0);
}

fn write_test_flop_bucket_with_assignment(dir: &Path, bucket_count: u16, bucket: u16) {
    let flop = [
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Diamond),
        Card::new(Value::Queen, Suit::Heart),
    ];
    let canonical = CanonicalBoard::from_cards(&flop).expect("valid test flop");
    std::fs::create_dir_all(dir).expect("create bucket directory");
    BucketFile {
        header: BucketFileHeader {
            street: poker_solver_core::blueprint_v2::Street::Flop,
            bucket_count,
            board_count: 1,
            combos_per_board: 1326,
            version: 2,
        },
        boards: vec![PackedBoard::from_cards(&canonical.cards)],
        buckets: vec![bucket; 1326],
    }
    .save(&dir.join("flop.buckets"))
    .expect("write test flop buckets");
}

fn write_mp_root_config(dir: &Path) {
    std::fs::write(
        dir.join("config.yaml"),
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
}

fn write_nested_mp_lazy_snapshot(root: &Path, name: &str) {
    write_mp_root_config(root);
    let snap = root.join(name);
    std::fs::create_dir_all(snap.join("universal")).unwrap();
    std::fs::write(
        snap.join("metadata.json"),
        r#"{"kind":"blueprint_mp_lazy_sparse","iterations":100,"elapsed_minutes":1}"#,
    )
    .unwrap();
    write_mp_lazy_bundle(&snap.join("universal"));
}

fn write_nested_mp_lazy_snapshot_with_relative_cluster_path(root: &Path, name: &str) {
    let universal = root.join(name).join("universal");
    std::fs::create_dir_all(&universal).unwrap();
    write_mp_lazy_2p_bundle_with_config(&universal);

    let config_path = universal.join("config.yaml");
    let mut config: BlueprintMpConfig = serde_yaml::from_str(
        &std::fs::read_to_string(&config_path).expect("read retained MP config"),
    )
    .expect("retained MP config should deserialize");
    config.training.cluster_path = Some("./local_data/buckets/test_cluster".to_string());
    std::fs::write(
        &config_path,
        serde_yaml::to_string(&config).expect("serialize retained MP config"),
    )
    .expect("write retained MP config");

    write_test_flop_bucket_with_assignment(
        &root.join("local_data/buckets/test_cluster"),
        config.clustering.flop.buckets,
        1,
    );
    write_test_flop_bucket_with_assignment(
        &universal.join("buckets"),
        config.clustering.flop.buckets + 1,
        0,
    );
}

fn write_nested_mp_lazy_snapshot_with_invalid_configured_cluster_path(root: &Path, name: &str) {
    let universal = root.join(name).join("universal");
    std::fs::create_dir_all(&universal).unwrap();
    write_mp_lazy_2p_bundle_with_config(&universal);

    let config_path = universal.join("config.yaml");
    let mut config: BlueprintMpConfig = serde_yaml::from_str(
        &std::fs::read_to_string(&config_path).expect("read retained MP config"),
    )
    .expect("retained MP config should deserialize");
    config.training.cluster_path = Some("./local_data/buckets/invalid_cluster".to_string());
    std::fs::write(
        &config_path,
        serde_yaml::to_string(&config).expect("serialize retained MP config"),
    )
    .expect("write retained MP config");

    write_test_flop_bucket(
        &root.join("local_data/buckets/invalid_cluster"),
        config.clustering.flop.buckets + 1,
    );
    write_test_flop_bucket(&universal.join("buckets"), config.clustering.flop.buckets);
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

#[test]
fn list_blueprints_uses_newest_nested_universal_snapshot_manifest() {
    let base = TempDir::new().unwrap();
    let bundle = base.path().join("mixed_universal_snapshots");
    std::fs::create_dir_all(bundle.join("snapshot_0001")).unwrap();
    write_universal_hu_bundle(&bundle.join("snapshot_0001"));

    std::fs::create_dir_all(bundle.join("snapshot_0002/universal")).unwrap();
    write_mp_lazy_bundle(&bundle.join("snapshot_0002/universal"));

    let entries =
        poker_solver_tauri::list_blueprints_core(base.path().to_string_lossy().to_string())
            .expect("listing should succeed");
    let bundle_path = bundle.to_string_lossy().to_string();
    let entry = entries
        .iter()
        .find(|entry| entry.path == bundle_path)
        .expect("mixed snapshot bundle should be listed");

    assert!(
        entry.name.contains("(6-player universal_mp_lazy)"),
        "entry name should reflect newest nested manifest, got {}",
        entry.name
    );
    assert_eq!(entry.stack_depth, 100.0);
    assert!(entry.has_strategy);
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
async fn mp_eager_game_new_is_rejected_by_lazy_session_gate() {
    let dir = TempDir::new().unwrap();
    write_mp_eager_bundle(dir.path());

    let exploration = ExplorationState::default();
    poker_solver_tauri::load_bundle_core(&exploration, dir.path().to_string_lossy().to_string())
        .await
        .expect("MP eager bundle should load for metadata");

    let sessions = GameSessionState::default();
    let error = game_new_core(&exploration, &PostflopState::default(), &sessions)
        .expect_err("eager MP bundle must not enter the lazy session");
    assert!(
        error.contains("supports only universal_mp_lazy") && error.contains("universal_mp_eager"),
        "unexpected eager backend error: {error}"
    );
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
async fn two_player_lazy_bundle_starts_game_session_and_advances_root_action() {
    let dir = TempDir::new().unwrap();
    write_mp_lazy_2p_bundle_with_config(dir.path());

    let exploration = ExplorationState::default();
    poker_solver_tauri::load_bundle_core(&exploration, dir.path().to_string_lossy().to_string())
        .await
        .expect("2p MP lazy bundle should load");

    let postflop = PostflopState::default();
    let sessions = GameSessionState::default();
    game_new_core(&exploration, &postflop, &sessions)
        .expect("configured 2p MP lazy bundle should initialize");

    let root = game_get_state_core(&sessions, None).expect("root state");
    assert_eq!(root.street, "Preflop");
    assert_eq!(root.matrix.as_ref().expect("root matrix").cells.len(), 13);
    assert_eq!(root.matrix.as_ref().unwrap().cells[0].len(), 13);
    assert!(!root.actions.is_empty());
    let high_canonical_probability = root.matrix.as_ref().unwrap().cells[12][11].probabilities[0];
    let expected_final_bucket_probability = f32::from(MP_BUCKET_COUNTS[0])
        / (f32::from(MP_BUCKET_COUNTS[0]) + (root.actions.len() - 1) as f32);
    assert!(
        (high_canonical_probability - expected_final_bucket_probability).abs() < 1e-6,
        "32o should map to final preflop bucket {}",
        MP_BUCKET_COUNTS[0] - 1
    );

    let child = game_play_action_core(&sessions, &root.actions[0].id, None)
        .expect("first root action should advance");
    assert_eq!(child.action_history.len(), 1);
}

#[tokio::test]
async fn two_player_lazy_session_uses_history_specific_sparse_row_for_72o() {
    let dir = TempDir::new().unwrap();
    write_mp_lazy_2p_bundle_with_anomaly_rows(dir.path());

    let exploration = ExplorationState::default();
    poker_solver_tauri::load_bundle_core(&exploration, dir.path().to_string_lossy().to_string())
        .await
        .expect("2p MP lazy anomaly bundle should load");
    let sessions = GameSessionState::default();
    game_new_core(&exploration, &PostflopState::default(), &sessions)
        .expect("2p MP lazy anomaly bundle should initialize");

    let assert_72o_probabilities = |state: &poker_solver_tauri::GameState, expected: &[f32]| {
        let cell = state
            .matrix
            .as_ref()
            .expect("decision state should include a matrix")
            .cells
            .iter()
            .flatten()
            .find(|cell| cell.hand == "72o")
            .expect("matrix should include 72o");
        assert_eq!(cell.probabilities, expected);
    };

    let root = game_get_state_core(&sessions, None).expect("root state");
    assert_eq!(root.position, "SB");
    assert_eq!(
        root.actions
            .iter()
            .map(|action| action.label.as_str())
            .collect::<Vec<_>>(),
        vec!["Fold", "Call", "2bb"]
    );
    assert_72o_probabilities(&root, &[0.1, 0.2, 0.7]);

    let sb_raise = root
        .actions
        .iter()
        .find(|action| action.label == "2bb")
        .expect("root should expose the 2bb SB raise");
    let after_sb_raise = game_play_action_core(&sessions, &sb_raise.id, None)
        .expect("SB 2bb raise should advance to BB");
    assert_eq!(after_sb_raise.position, "BB");
    assert_eq!(after_sb_raise.action_history.len(), 1);
    assert_eq!(after_sb_raise.action_history[0].position, "SB");
    assert_eq!(after_sb_raise.action_history[0].label, "2bb");
    assert_eq!(
        after_sb_raise
            .actions
            .iter()
            .map(|action| action.label.as_str())
            .collect::<Vec<_>>(),
        vec!["Fold", "Call", "3bb", "All-in"]
    );
    assert_72o_probabilities(&after_sb_raise, &[0.2, 0.3, 0.4, 0.1]);

    let bb_reraise = after_sb_raise
        .actions
        .iter()
        .find(|action| action.label == "3bb")
        .expect("BB should expose the 3bb reraise");
    let after_bb_reraise = game_play_action_core(&sessions, &bb_reraise.id, None)
        .expect("BB 3bb reraise should advance to SB");
    assert_eq!(after_bb_reraise.position, "SB");
    assert_eq!(after_bb_reraise.action_history.len(), 2);
    assert_eq!(
        after_bb_reraise
            .action_history
            .iter()
            .map(|record| (record.position.as_str(), record.label.as_str()))
            .collect::<Vec<_>>(),
        vec![("SB", "2bb"), ("BB", "3bb")]
    );
    assert_eq!(
        after_bb_reraise
            .actions
            .iter()
            .map(|action| action.label.as_str())
            .collect::<Vec<_>>(),
        vec!["Fold", "Call", "All-in"]
    );

    // GameState exposes one vector derived from universal lazy strategy_sums:
    // the average strategy. It has no separate current-strategy field.
    assert_72o_probabilities(&after_bb_reraise, &[0.7, 0.2, 0.1]);
}

async fn start_two_player_lazy_session(
    dir: &TempDir,
    include_flop_buckets: bool,
    include_flop_rows: bool,
) -> (ExplorationState, GameSessionState) {
    write_mp_lazy_2p_bundle_with_options(dir.path(), include_flop_buckets, include_flop_rows);
    let exploration = ExplorationState::default();
    poker_solver_tauri::load_bundle_core(&exploration, dir.path().to_string_lossy().to_string())
        .await
        .expect("2p MP lazy bundle should load");
    let sessions = GameSessionState::default();
    game_new_core(&exploration, &PostflopState::default(), &sessions)
        .expect("2p MP lazy bundle should initialize");
    (exploration, sessions)
}

fn enter_two_player_flop_chance(sessions: &GameSessionState) -> poker_solver_tauri::GameState {
    let mut state = game_get_state_core(sessions, None).expect("preflop state");
    for _ in 0..4 {
        if state.street == "Flop" && state.is_chance {
            return state;
        }
        let action = state
            .actions
            .iter()
            .find(|action| action.action_type == "call" || action.action_type == "check")
            .expect("preflop call/check action");
        state = game_play_action_core(sessions, &action.id, None).expect("advance preflop");
    }
    panic!("preflop line did not reach flop chance: {state:?}");
}

#[tokio::test]
async fn two_player_lazy_session_exposes_partial_flop_chance_and_renders_flop_matrix() {
    let dir = TempDir::new().unwrap();
    let (_exploration, sessions) = start_two_player_lazy_session(&dir, true, true).await;

    let root = game_get_state_core(&sessions, None).unwrap();
    let chance = enter_two_player_flop_chance(&sessions);
    assert!(chance.is_chance);
    assert_eq!(chance.street, "Flop");
    assert!(chance.board.is_empty());

    let one_card = game_deal_card_core(&sessions, "As").unwrap();
    assert!(one_card.is_chance);
    assert_eq!(one_card.board, vec!["As"]);
    let duplicate = game_deal_card_core(&sessions, "As").unwrap_err();
    assert!(duplicate.contains("Duplicate board card"));
    let after_duplicate = game_get_state_core(&sessions, None).unwrap();
    assert_eq!(after_duplicate.board, vec!["As"]);

    let two_cards = game_deal_card_core(&sessions, "Kd").unwrap();
    assert!(two_cards.is_chance);
    assert_eq!(two_cards.board, vec!["As", "Kd"]);
    let flop = game_deal_card_core(&sessions, "Qh").unwrap();
    assert_eq!(flop.street, "Flop");
    assert!(!flop.is_chance);
    assert_eq!(flop.board, vec!["As", "Kd", "Qh"]);
    let matrix = flop.matrix.expect("completed flop should render a matrix");
    assert_eq!(matrix.cells.len(), 13);
    assert!(matrix
        .cells
        .iter()
        .flatten()
        .any(|cell| cell.combo_count > 0));
    assert_ne!(root.actions.len(), 0);
}

#[tokio::test]
async fn two_player_lazy_session_rejects_missing_flop_bucket_source_without_mutation() {
    let dir = TempDir::new().unwrap();
    let (_exploration, sessions) = start_two_player_lazy_session(&dir, false, true).await;
    enter_two_player_flop_chance(&sessions);
    game_deal_card_core(&sessions, "As").unwrap();
    game_deal_card_core(&sessions, "Kd").unwrap();

    let error = game_deal_card_core(&sessions, "Qh").unwrap_err();
    assert!(error.contains("flop bucket source is missing"), "{error}");
    let state = game_get_state_core(&sessions, None).unwrap();
    assert_eq!(state.board, vec!["As", "Kd"]);
    assert!(state.is_chance);
}

#[tokio::test]
async fn two_player_lazy_session_rejects_missing_flop_row_with_full_key_without_mutation() {
    let dir = TempDir::new().unwrap();
    let (_exploration, sessions) = start_two_player_lazy_session(&dir, true, false).await;
    enter_two_player_flop_chance(&sessions);
    game_deal_card_core(&sessions, "As").unwrap();
    game_deal_card_core(&sessions, "Kd").unwrap();

    let error = game_deal_card_core(&sessions, "Qh").unwrap_err();
    assert!(error.contains("sparse row is missing"), "{error}");
    assert!(error.contains("history_hash") && error.contains("history_len"));
    let state = game_get_state_core(&sessions, None).unwrap();
    assert_eq!(state.board, vec!["As", "Kd"]);
    assert!(state.is_chance);
}

#[tokio::test]
async fn two_player_lazy_session_back_replays_preflop_and_flop_state() {
    let dir = TempDir::new().unwrap();
    let (_exploration, sessions) = start_two_player_lazy_session(&dir, true, true).await;
    enter_two_player_flop_chance(&sessions);
    game_deal_card_core(&sessions, "As").unwrap();
    game_deal_card_core(&sessions, "Kd").unwrap();
    let flop = game_deal_card_core(&sessions, "Qh").unwrap();

    let after_flop_action = game_play_action_core(&sessions, &flop.actions[0].id, None)
        .expect("first flop action should be replayable");
    assert_eq!(after_flop_action.board, vec!["As", "Kd", "Qh"]);
    let restored_flop = game_back_core(&sessions, None).unwrap();
    assert_eq!(restored_flop.board, vec!["As", "Kd", "Qh"]);
    assert_eq!(restored_flop.street, "Flop");
    assert!(!restored_flop.is_chance);

    let restored_preflop = game_back_core(&sessions, None).unwrap();
    assert_eq!(restored_preflop.board, Vec::<String>::new());
    assert_eq!(restored_preflop.street, "Preflop");
    assert_eq!(restored_preflop.action_history.len(), 1);
    let restored_root = game_back_core(&sessions, None).unwrap();
    assert_eq!(restored_root.board, Vec::<String>::new());
    assert_eq!(restored_root.street, "Preflop");
    assert!(restored_root.action_history.is_empty());
}

#[tokio::test]
async fn two_player_lazy_session_rejects_turn_deal_at_boundary() {
    let dir = TempDir::new().unwrap();
    let (_exploration, sessions) = start_two_player_lazy_session(&dir, true, true).await;
    enter_two_player_flop_chance(&sessions);
    game_deal_card_core(&sessions, "As").unwrap();
    game_deal_card_core(&sessions, "Kd").unwrap();
    let mut state = game_deal_card_core(&sessions, "Qh").unwrap();

    while state.street == "Flop" && !state.is_chance {
        state = game_play_action_core(&sessions, "0", None).unwrap();
    }
    assert_eq!(state.street, "Turn");
    assert!(state.is_chance);
    let error = game_deal_card_core(&sessions, "Jc").unwrap_err();
    assert!(error.contains("does not support dealing the turn"));
    let unchanged = game_get_state_core(&sessions, None).unwrap();
    assert_eq!(unchanged.board, vec!["As", "Kd", "Qh"]);
    assert_eq!(unchanged.action_history.len(), state.action_history.len());
}

#[tokio::test]
async fn nested_mp_lazy_snapshot_loads_through_root_and_snapshot_v2_entrypoint() {
    let dir = TempDir::new().unwrap();
    write_nested_mp_lazy_snapshot(dir.path(), "snapshot_0001");

    let state = ExplorationState::default();
    let root_info =
        poker_solver_tauri::load_bundle_core(&state, dir.path().to_string_lossy().to_string())
            .await
            .expect("root should resolve nested MP lazy universal bundle");
    assert_eq!(root_info.iterations, 100);
    assert!(poker_solver_tauri::is_bundle_loaded_core(&state));

    let state = ExplorationState::default();
    let snapshot_info = poker_solver_tauri::load_blueprint_v2_core(
        &state,
        dir.path().to_string_lossy().to_string(),
        Some("snapshot_0001".to_string()),
    )
    .await
    .expect("snapshot v2 entrypoint should delegate to nested universal bundle");
    assert_eq!(snapshot_info.iterations, 100);

    let snapshots =
        poker_solver_tauri::list_snapshots_core(dir.path().to_string_lossy().to_string())
            .expect("snapshot listing should succeed");
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].name, "snapshot_0001");
    assert!(snapshots[0].has_strategy);
    assert_eq!(snapshots[0].iterations, Some(100));
    assert_eq!(snapshots[0].elapsed_minutes, Some(1));
}

#[tokio::test]
async fn nested_mp_lazy_snapshot_resolves_relative_cluster_path_before_implicit_buckets() {
    let temp = TempDir::new().unwrap();
    let project = temp.path().join("project");
    write_nested_mp_lazy_snapshot_with_relative_cluster_path(&project, "snapshot_0001");

    let exploration = ExplorationState::default();
    poker_solver_tauri::load_blueprint_v2_core(
        &exploration,
        project.to_string_lossy().to_string(),
        Some("snapshot_0001".to_string()),
    )
    .await
    .expect("snapshot path should resolve nested MP lazy universal bundle");
    let sessions = GameSessionState::default();
    game_new_core(&exploration, &PostflopState::default(), &sessions)
        .expect("nested 2p MP lazy bundle should initialize");

    enter_two_player_flop_chance(&sessions);
    game_deal_card_core(&sessions, "As").unwrap();
    game_deal_card_core(&sessions, "Kd").unwrap();
    let flop = game_deal_card_core(&sessions, "Qh")
        .expect("relative retained cluster_path should resolve the flop buckets");

    assert_eq!(flop.street, "Flop");
    assert!(!flop.is_chance);
    assert_eq!(flop.board, vec!["As", "Kd", "Qh"]);
    let buckets: Vec<u16> = flop
        .matrix
        .as_ref()
        .expect("flop state should include the strategy matrix")
        .cells
        .iter()
        .flatten()
        .flat_map(|cell| cell.combos.iter())
        .filter_map(|combo| combo.bucket)
        .collect();
    assert!(!buckets.is_empty());
    assert!(
        buckets.iter().all(|&bucket| bucket == 1),
        "configured bucket source should assign every visible combo to bucket 1, got {buckets:?}"
    );
}

#[tokio::test]
async fn nested_mp_lazy_snapshot_rejects_invalid_configured_cluster_path() {
    let temp = TempDir::new().unwrap();
    let project = temp.path().join("project");
    write_nested_mp_lazy_snapshot_with_invalid_configured_cluster_path(&project, "snapshot_0001");

    let exploration = ExplorationState::default();
    poker_solver_tauri::load_blueprint_v2_core(
        &exploration,
        project.to_string_lossy().to_string(),
        Some("snapshot_0001".to_string()),
    )
    .await
    .expect("snapshot path should resolve nested MP lazy universal bundle");
    let sessions = GameSessionState::default();
    game_new_core(&exploration, &PostflopState::default(), &sessions)
        .expect("nested 2p MP lazy bundle should initialize");

    enter_two_player_flop_chance(&sessions);
    game_deal_card_core(&sessions, "As").unwrap();
    game_deal_card_core(&sessions, "Kd").unwrap();
    let error = game_deal_card_core(&sessions, "Qh").expect_err(
        "an invalid configured bucket source must not be hidden by a valid implicit source",
    );
    assert!(
        error.contains("Invalid Universal MP configured training.cluster_path flop bucket file"),
        "expected configured-source validation error, got: {error}"
    );
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
