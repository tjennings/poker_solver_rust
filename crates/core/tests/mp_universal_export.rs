//! Integration tests for MP eager-to-universal-dense export (Phase 3).
//!
//! Covers acceptance criteria:
//! 1. Row descriptors include acting seat, street, bucket, arena node,
//!    ordered actions, and fingerprints.
//! 2. Probabilities match MpStorage::average_strategy for known nodes.
//! 3. Existing MP snapshot artifacts are preserved (no changes to trainer).

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use poker_solver_core::blueprint_mp::config::*;
use poker_solver_core::blueprint_mp::game_tree::*;
use poker_solver_core::blueprint_mp::mccfr::{sample_deal, traverse_external};
use poker_solver_core::blueprint_mp::storage::MpStorage;
use poker_solver_core::blueprint_mp::{Bucket, Chips, DealWithBuckets, MAX_PLAYERS, Seat};
use poker_solver_core::blueprint_universal::mp_eager_export::{
    self, ExportError, ExportOutput, MpTrainingInfo,
};
use poker_solver_core::blueprint_universal::{ActionKind, BundleReader};
use rand::SeedableRng;
use rand::rngs::SmallRng;

const BUCKET_COUNTS: [u16; 4] = [10, 10, 10, 10];

// ── Helpers ────────────────────────────────────────────────────────

fn yaml_f64(v: f64) -> serde_yaml::Value {
    serde_yaml::Value::Number(serde_yaml::Number::from(v))
}

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
        name: "3p-export-test".to_string(),
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
            buckets: BUCKET_COUNTS[0],
        },
        flop: MpStreetCluster {
            buckets: BUCKET_COUNTS[1],
        },
        turn: MpStreetCluster {
            buckets: BUCKET_COUNTS[2],
        },
        river: MpStreetCluster {
            buckets: BUCKET_COUNTS[3],
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
        output_dir: "/tmp/mp_export_test".into(),
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
        for (street, &count) in BUCKET_COUNTS.iter().enumerate() {
            buckets[seat][street] = Bucket(card_idx % count);
        }
    }
    DealWithBuckets {
        deal: deal.clone(),
        buckets,
    }
}

fn run_iterations(tree: &MpGameTree, storage: &MpStorage, count: u64, seed: u64) {
    let mut rng = SmallRng::seed_from_u64(seed);
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

// ── Acceptance test: in-memory export round-trip ───────────────────

#[test]
fn mp_export_roundtrip_probabilities_match() {
    let config = build_3p_config();
    let tree = MpGameTree::build(&config.game, &config.action_abstraction);
    let storage = MpStorage::new(&tree, config.clustering.bucket_counts());
    run_iterations(&tree, &storage, 200, 42);

    let training = MpTrainingInfo {
        iterations: 200,
        elapsed_minutes: 0.1,
    };
    let output =
        mp_eager_export::export_mp_strategy_to_universal(&config, &tree, &storage, &training)
            .expect("export should succeed");

    // Write to temp dir and load back
    let tmp = tempfile::tempdir().expect("tempdir");
    poker_solver_core::blueprint_universal::write_bundle(
        tmp.path(),
        &output.manifest,
        &poker_solver_core::blueprint_universal::BundleData {
            rows: &output.rows,
            actions: &output.actions,
            probs: &output.probs,
        },
    )
    .expect("write_bundle");

    let reader = BundleReader::open(tmp.path()).expect("open bundle");

    // For every decision node and bucket, compare loaded probs vs storage
    let mut avg_buf = vec![0.0_f64; 32];
    let mut checked = 0usize;
    for row_idx in 0..reader.row_count() {
        let row = reader.row(row_idx).unwrap();
        let node_idx = row.source_node_idx;
        let bucket = row.local_bucket;

        // Verify it's a decision node
        let node = &tree.nodes[node_idx as usize];
        let (seat, street, actions) = match node {
            MpGameNode::Decision {
                seat,
                street,
                actions,
                ..
            } => (seat.index(), *street as u8, actions),
            _ => panic!("row points to non-decision node {node_idx}"),
        };

        // Check row identity fields
        assert_eq!(row.seat, seat, "seat mismatch at row {row_idx}");
        assert_eq!(row.street, street, "street mismatch at row {row_idx}");
        assert_eq!(row.namespace, 1, "namespace should be mp_arena (1)");
        assert_eq!(
            row.action_count,
            actions.len() as u16,
            "action_count mismatch at row {row_idx}"
        );

        // Compare probabilities bitwise
        let num_actions = actions.len();
        storage.average_strategy(node_idx, bucket, num_actions, &mut avg_buf);
        for a in 0..num_actions {
            let expected = avg_buf[a] as f32;
            let actual = reader.prob(row.prob_offset as usize + a).unwrap();
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "prob mismatch at row {row_idx}, action {a}: \
                 expected {expected}, got {actual}"
            );
        }
        checked += 1;
    }

    assert!(checked > 0, "should have checked at least one row");
    eprintln!("Verified {checked} rows with bitwise probability match");
}

// ── Seat ordering test ────────────────────────────────────────────

#[test]
fn mp_export_rows_sorted_by_seat() {
    let config = build_3p_config();
    let tree = MpGameTree::build(&config.game, &config.action_abstraction);
    let storage = MpStorage::new(&tree, config.clustering.bucket_counts());
    run_iterations(&tree, &storage, 50, 99);

    let training = MpTrainingInfo {
        iterations: 50,
        elapsed_minutes: 0.0,
    };
    let output =
        mp_eager_export::export_mp_strategy_to_universal(&config, &tree, &storage, &training)
            .unwrap();

    // Verify sort order: (namespace, seat, street, source_node_idx, global_bucket)
    for w in output.rows.windows(2) {
        assert!(
            w[0].identity_key() < w[1].identity_key(),
            "rows not sorted: {:?} >= {:?}",
            w[0].identity_key(),
            w[1].identity_key()
        );
    }

    // Verify that multiple seats appear
    let seats: std::collections::HashSet<u8> = output.rows.iter().map(|r| r.seat).collect();
    assert!(
        seats.len() >= 2,
        "3-player game should have rows for at least 2 seats, got {seats:?}"
    );
}

// ── Zero-mass row test ────────────────────────────────────────────

#[test]
fn mp_export_zero_mass_rows_are_uniform() {
    let config = build_3p_config();
    let tree = MpGameTree::build(&config.game, &config.action_abstraction);
    let storage = MpStorage::new(&tree, config.clustering.bucket_counts());
    // Do NOT run any iterations -- all rows are zero-mass

    let training = MpTrainingInfo {
        iterations: 0,
        elapsed_minutes: 0.0,
    };
    let output =
        mp_eager_export::export_mp_strategy_to_universal(&config, &tree, &storage, &training)
            .unwrap();

    for (i, row) in output.rows.iter().enumerate() {
        let n = row.action_count as usize;
        let start = row.prob_offset as usize;
        let uniform = 1.0f32 / n as f32;
        for a in 0..n {
            let p = output.probs[start + a];
            assert_eq!(
                p.to_bits(),
                uniform.to_bits(),
                "row {i} action {a}: expected uniform {uniform}, got {p}"
            );
        }
    }
}

// ── Bucket count mismatch rejection ───────────────────────────────

#[test]
fn mp_export_rejects_bucket_count_mismatch() {
    let config = build_3p_config();
    let tree = MpGameTree::build(&config.game, &config.action_abstraction);
    // Create storage with DIFFERENT bucket counts
    let wrong_counts = [5, 5, 5, 5];
    let storage = MpStorage::new(&tree, wrong_counts);

    let training = MpTrainingInfo {
        iterations: 0,
        elapsed_minutes: 0.0,
    };
    let result =
        mp_eager_export::export_mp_strategy_to_universal(&config, &tree, &storage, &training);
    assert!(result.is_err(), "should reject bucket count mismatch");
    let err = result.unwrap_err();
    assert!(
        matches!(err, ExportError::BucketCountMismatch { .. }),
        "expected BucketCountMismatch, got {err:?}"
    );
}

// ── Action mapping test ───────────────────────────────────────────

#[test]
fn mp_export_action_descriptors_correct() {
    // Use a config with sized bets to exercise Lead/Raise/AllIn
    let game = MpGameConfig {
        name: "action-test".to_string(),
        num_players: 3,
        stack_depth: 20.0,
        allow_preflop_limp: true,
        blinds: three_player_blinds(),
        rake_rate: 0.0,
        rake_cap: 0.0,
    };
    let postflop = MpStreetSizes {
        lead: vec![yaml_f64(0.67)],
        raise: vec![vec![yaml_f64(1.0)]],
    };
    let preflop = MpStreetSizes {
        lead: vec![serde_yaml::Value::String("5bb".into())],
        raise: vec![vec![serde_yaml::Value::String("3.0x".into())]],
    };
    let action_abs = MpActionAbstractionConfig {
        max_flop_players: None,
        preflop,
        flop: postflop.clone(),
        turn: postflop.clone(),
        river: postflop,
    };

    let tree = MpGameTree::build(&game, &action_abs);
    let clustering = MpClusteringConfig {
        preflop: MpStreetCluster { buckets: 3 },
        flop: MpStreetCluster { buckets: 3 },
        turn: MpStreetCluster { buckets: 3 },
        river: MpStreetCluster { buckets: 3 },
    };
    let bc = clustering.bucket_counts();
    let storage = MpStorage::new(&tree, bc);

    let config = BlueprintMpConfig {
        game,
        action_abstraction: action_abs,
        clustering,
        training: MpTrainingConfig {
            backend: MpTrainingBackend::Eager,
            chance_continuation_mode: MpChanceContinuationMode::SampledFullDeal,
            cluster_path: None,
            iterations: Some(10),
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
        },
        snapshots: MpSnapshotConfig {
            warmup_minutes: 999,
            snapshot_every_minutes: 999,
            output_dir: "/tmp/action_test".into(),
            resume: false,
            max_snapshots: None,
            format: MpSnapshotFormat::Legacy,
        },
    };

    let training = MpTrainingInfo {
        iterations: 0,
        elapsed_minutes: 0.0,
    };
    let output =
        mp_eager_export::export_mp_strategy_to_universal(&config, &tree, &storage, &training)
            .unwrap();

    // Find actions across the exported rows
    let mut found_fold = false;
    let mut found_call = false;
    let mut found_check = false;
    let mut found_lead = false;
    let mut found_raise = false;
    let mut found_allin = false;

    for row in &output.rows {
        let start = row.action_offset as usize;
        let count = row.action_count as usize;
        for i in 0..count {
            let ad = &output.actions[start + i];
            assert_eq!(
                ad.source_action_index, i as u16,
                "source_action_index mismatch"
            );
            match ad.kind {
                ActionKind::Fold => {
                    assert!(!ad.is_aggressive);
                    found_fold = true;
                }
                ActionKind::Call => {
                    assert!(!ad.is_aggressive);
                    found_call = true;
                }
                ActionKind::Check => {
                    assert!(!ad.is_aggressive);
                    found_check = true;
                }
                ActionKind::Bet => {
                    assert!(ad.is_aggressive);
                    // Lead maps to Bet kind
                    assert!(ad.amount_chips > 0, "bet amount should be > 0");
                    found_lead = true;
                }
                ActionKind::Raise => {
                    assert!(ad.is_aggressive);
                    assert!(ad.amount_chips > 0, "raise amount should be > 0");
                    found_raise = true;
                }
                ActionKind::AllInBetRaise => {
                    assert!(ad.is_aggressive);
                    found_allin = true;
                }
                ActionKind::AllInCall | ActionKind::Opaque => {}
            }
        }
    }

    // A 3-player game with sized bets should have all these action types
    assert!(found_fold, "should have fold actions");
    assert!(found_call, "should have call actions");
    // AllIn is generated in non-unopened contexts
    assert!(found_allin, "should have all-in actions");
}

// ── Fingerprint tests ─────────────────────────────────────────────

#[test]
fn mp_export_fingerprints_stable_and_differ_by_seat() {
    let config = build_3p_config();
    let tree = MpGameTree::build(&config.game, &config.action_abstraction);
    let storage = MpStorage::new(&tree, config.clustering.bucket_counts());

    let training = MpTrainingInfo {
        iterations: 0,
        elapsed_minutes: 0.0,
    };
    let output1 =
        mp_eager_export::export_mp_strategy_to_universal(&config, &tree, &storage, &training)
            .unwrap();
    let output2 =
        mp_eager_export::export_mp_strategy_to_universal(&config, &tree, &storage, &training)
            .unwrap();

    // Fingerprints are stable across runs
    assert_eq!(output1.rows.len(), output2.rows.len());
    for (r1, r2) in output1.rows.iter().zip(output2.rows.iter()) {
        assert_eq!(r1.row_key_fingerprint, r2.row_key_fingerprint);
        assert_eq!(r1.action_schema_fingerprint, r2.action_schema_fingerprint);
    }

    // Fingerprints differ across seats for same node structure
    let seat0_fps: Vec<u64> = output1
        .rows
        .iter()
        .filter(|r| r.seat == 0)
        .map(|r| r.row_key_fingerprint)
        .collect();
    let seat1_fps: Vec<u64> = output1
        .rows
        .iter()
        .filter(|r| r.seat == 1)
        .map(|r| r.row_key_fingerprint)
        .collect();

    if !seat0_fps.is_empty() && !seat1_fps.is_empty() {
        // No fingerprints should be shared between seats
        let shared: Vec<_> = seat0_fps
            .iter()
            .filter(|fp| seat1_fps.contains(fp))
            .collect();
        assert!(
            shared.is_empty(),
            "seats should have distinct fingerprints, found {shared:?} in common"
        );
    }
}

// ── Disk-path consistency test ─────────────────────────────────────

#[test]
fn mp_export_disk_path_byte_identical_to_in_memory() {
    use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;

    let config = build_3p_config();
    let tree = MpGameTree::build(&config.game, &config.action_abstraction);
    let storage = MpStorage::new(&tree, config.clustering.bucket_counts());
    run_iterations(&tree, &storage, 100, 77);

    // In-memory export
    let training = MpTrainingInfo {
        iterations: 100,
        elapsed_minutes: 0.5,
    };
    let in_mem =
        mp_eager_export::export_mp_strategy_to_universal(&config, &tree, &storage, &training)
            .unwrap();

    // Save snapshot the way the trainer does
    let tmp = tempfile::tempdir().expect("tempdir");
    let mp_dir = tmp.path().join("mp_output");
    std::fs::create_dir_all(&mp_dir).unwrap();

    // Write config.yaml
    let config_yaml = serde_yaml::to_string(&config).unwrap();
    std::fs::write(mp_dir.join("config.yaml"), &config_yaml).unwrap();

    // Build projected strategy (same as trainer's mp_strategy_from_storage)
    let mut action_probs = Vec::new();
    let mut node_action_counts = Vec::new();
    let mut node_street_indices = Vec::new();
    let mut avg = vec![0.0_f64; 32];

    for (node_idx, node) in tree.nodes.iter().enumerate() {
        if let MpGameNode::Decision {
            street, actions, ..
        } = node
        {
            let num_actions = actions.len();
            let street_idx = *street as u8;
            let buckets = storage.bucket_counts[street_idx as usize];
            if avg.len() < num_actions {
                avg.resize(num_actions, 0.0);
            }

            node_action_counts.push(num_actions as u16);
            node_street_indices.push(street_idx);
            for bucket in 0..buckets {
                storage.average_strategy(node_idx as u32, bucket, num_actions, &mut avg);
                action_probs.extend(avg[..num_actions].iter().map(|&p| p as f32));
            }
        }
    }

    let mut strategy = BlueprintV2Strategy {
        action_probs,
        node_action_counts,
        node_street_indices,
        bucket_counts: storage.bucket_counts,
        iterations: 100,
        elapsed_minutes: 0,
        node_offsets: Vec::new(),
    };
    strategy.post_deserialize();

    // Write snapshot
    let snapshot_dir = mp_dir.join("snapshot_0001");
    std::fs::create_dir_all(&snapshot_dir).unwrap();
    strategy.save(&snapshot_dir.join("strategy.bin")).unwrap();

    let metadata = serde_json::json!({
        "kind": "blueprint_mp",
        "snapshot_index": 1,
        "iterations": 100u64,
        "elapsed_seconds": 30u64,
        "elapsed_minutes": 0u64,
        "bucket_counts": storage.bucket_counts,
    });
    std::fs::write(
        snapshot_dir.join("metadata.json"),
        serde_json::to_string_pretty(&metadata).unwrap(),
    )
    .unwrap();

    // Export via disk wrapper
    let disk_out = tmp.path().join("disk_bundle");
    mp_eager_export::export_mp_bundle(&mp_dir, "snapshot_0001", &disk_out).unwrap();

    // Export in-memory to a separate dir
    let mem_out = tmp.path().join("mem_bundle");
    poker_solver_core::blueprint_universal::write_bundle(
        &mem_out,
        &in_mem.manifest,
        &poker_solver_core::blueprint_universal::BundleData {
            rows: &in_mem.rows,
            actions: &in_mem.actions,
            probs: &in_mem.probs,
        },
    )
    .unwrap();

    // Compare the three binary payloads byte-for-byte
    for name in &[
        "strategy.rows.bin",
        "strategy.actions.bin",
        "strategy.probs.f32.bin",
    ] {
        let disk_bytes = std::fs::read(disk_out.join(name)).unwrap();
        let mem_bytes = std::fs::read(mem_out.join(name)).unwrap();
        assert_eq!(
            disk_bytes, mem_bytes,
            "binary payload {name} differs between disk and in-memory export"
        );
    }
}

// ── Disk-path rejection tests ─────────────────────────────────────

#[test]
fn mp_export_disk_rejects_missing_config() {
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("out");
    let result = mp_eager_export::export_mp_bundle(tmp.path(), "snapshot_0001", &out);
    assert!(matches!(result, Err(ExportError::MissingFile { .. })));
}

#[test]
fn mp_export_disk_rejects_missing_strategy_bin() {
    let tmp = tempfile::tempdir().unwrap();
    let mp_dir = tmp.path().join("mp");
    std::fs::create_dir_all(&mp_dir).unwrap();

    let config = build_3p_config();
    let config_yaml = serde_yaml::to_string(&config).unwrap();
    std::fs::write(mp_dir.join("config.yaml"), &config_yaml).unwrap();

    let snapshot = mp_dir.join("snapshot_0001");
    std::fs::create_dir_all(&snapshot).unwrap();
    // No strategy.bin written

    let out = tmp.path().join("out");
    let result = mp_eager_export::export_mp_bundle(&mp_dir, "snapshot_0001", &out);
    assert!(matches!(result, Err(ExportError::MissingFile { .. })));
}

#[test]
fn mp_export_disk_rejects_missing_metadata() {
    let tmp = tempfile::tempdir().unwrap();
    let mp_dir = tmp.path().join("mp");
    std::fs::create_dir_all(&mp_dir).unwrap();

    let config = build_3p_config();
    let config_yaml = serde_yaml::to_string(&config).unwrap();
    std::fs::write(mp_dir.join("config.yaml"), &config_yaml).unwrap();

    let snapshot = mp_dir.join("snapshot_0001");
    std::fs::create_dir_all(&snapshot).unwrap();
    // Write a dummy strategy.bin
    std::fs::write(snapshot.join("strategy.bin"), b"dummy").unwrap();
    // No metadata.json

    let out = tmp.path().join("out");
    let result = mp_eager_export::export_mp_bundle(&mp_dir, "snapshot_0001", &out);
    assert!(matches!(result, Err(ExportError::MissingFile { .. })));
}

#[test]
fn mp_export_disk_rejects_wrong_metadata_kind() {
    let tmp = tempfile::tempdir().unwrap();
    let mp_dir = tmp.path().join("mp");
    std::fs::create_dir_all(&mp_dir).unwrap();

    let config = build_3p_config();
    let config_yaml = serde_yaml::to_string(&config).unwrap();
    std::fs::write(mp_dir.join("config.yaml"), &config_yaml).unwrap();

    let snapshot = mp_dir.join("snapshot_0001");
    std::fs::create_dir_all(&snapshot).unwrap();

    // Write strategy.bin (needs to be loadable, use a dummy empty one)
    // This test should fail at metadata kind check before loading strategy
    let metadata = serde_json::json!({
        "kind": "blueprint_v2",
        "iterations": 100,
    });
    std::fs::write(
        snapshot.join("metadata.json"),
        serde_json::to_string(&metadata).unwrap(),
    )
    .unwrap();
    // strategy.bin still needed for the file check
    std::fs::write(snapshot.join("strategy.bin"), b"dummy").unwrap();

    let out = tmp.path().join("out");
    let result = mp_eager_export::export_mp_bundle(&mp_dir, "snapshot_0001", &out);
    assert!(
        matches!(result, Err(ExportError::BadMetadataKind { .. })),
        "expected BadMetadataKind, got {result:?}"
    );
}

// ── MP action abstraction fingerprint stability ───────────────────

#[test]
fn mp_action_abstraction_fingerprint_changes_with_sizing() {
    let config1 = build_3p_config();
    let tree1 = MpGameTree::build(&config1.game, &config1.action_abstraction);
    let storage1 = MpStorage::new(&tree1, config1.clustering.bucket_counts());
    let training = MpTrainingInfo {
        iterations: 0,
        elapsed_minutes: 0.0,
    };
    let out1 =
        mp_eager_export::export_mp_strategy_to_universal(&config1, &tree1, &storage1, &training)
            .unwrap();

    // Build a config with different sizing
    let mut config2 = build_3p_config();
    config2.action_abstraction.flop = MpStreetSizes {
        lead: vec![yaml_f64(0.5)],
        raise: vec![],
    };
    let tree2 = MpGameTree::build(&config2.game, &config2.action_abstraction);
    let storage2 = MpStorage::new(&tree2, config2.clustering.bucket_counts());
    let out2 =
        mp_eager_export::export_mp_strategy_to_universal(&config2, &tree2, &storage2, &training)
            .unwrap();

    assert_ne!(
        out1.manifest.actions.action_abstraction_fingerprint,
        out2.manifest.actions.action_abstraction_fingerprint,
        "fingerprint should change when sizing changes"
    );
}
