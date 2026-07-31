//! Integration tests for MP lazy sparse-to-universal-dense export (Phase 4).
//!
//! Covers acceptance criteria:
//! 1. SparseSnapshotEntry rows sorted by semantic key, exported with VERBATIM
//!    semantic identity (seat, street, local/global bucket, history hi/lo/hash/len).
//! 2. Strategy sums normalize correctly (match sparse average_strategy semantics).
//! 3. Zero-sum rows use uniform fallback.
//! 4. Artifact marked non-resumable (analysis-only).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    dead_code
)]

use poker_solver_core::blueprint_mp::sparse_storage::{
    MpInfosetKey, SparseActionDescriptor, SparseActionKind, SparseMpStorage, SparseSnapshotEntry,
};
use poker_solver_core::blueprint_mp::{Seat, Street};
use poker_solver_core::blueprint_universal::mp_lazy_export::{self, LazyTrainingInfo};
use poker_solver_core::blueprint_universal::{ActionKind, BundleReader};
use sha2::Digest;

// ── Helpers ──────────────────────────────────────────────────────────

fn make_key(
    seat: u8,
    street: Street,
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

fn action(
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

fn universal_kind(kind: SparseActionKind) -> ActionKind {
    match kind {
        SparseActionKind::Fold => ActionKind::Fold,
        SparseActionKind::Check => ActionKind::Check,
        SparseActionKind::Call => ActionKind::Call,
        SparseActionKind::Lead => ActionKind::Bet,
        SparseActionKind::Raise => ActionKind::Raise,
        SparseActionKind::AllInCall => ActionKind::AllInCall,
        SparseActionKind::AllInBetRaise => ActionKind::AllInBetRaise,
    }
}

fn is_aggressive(kind: SparseActionKind) -> bool {
    matches!(
        kind,
        SparseActionKind::Lead | SparseActionKind::Raise | SparseActionKind::AllInBetRaise
    )
}

/// Build test entries spanning multiple seats, streets, buckets, history
/// lengths (including >32 for hash-only identity), and a zero-mass entry.
fn build_test_entries() -> Vec<SparseSnapshotEntry> {
    vec![
        // Normal entry: seat 0, preflop, bucket 5, short history
        SparseSnapshotEntry {
            key: make_key(0, Street::Preflop, 5, 0x1234, 0x5678, 0xAAAA, 4),
            num_actions: 3,
            action_identity: Some(vec![
                action(SparseActionKind::Fold, 0, 0),
                action(SparseActionKind::Call, 0, 1),
                action(SparseActionKind::Raise, 6, 2),
            ]),
            regrets: vec![10, -5, 3],
            strategy_sums: vec![100, 200, 300],
        },
        // Normal entry: seat 1, flop, bucket 10, medium history
        SparseSnapshotEntry {
            key: make_key(1, Street::Flop, 10, 0xABCD, 0xEF01, 0xBBBB, 16),
            num_actions: 2,
            action_identity: Some(vec![
                action(SparseActionKind::Check, 0, 0),
                action(SparseActionKind::Lead, 20, 1),
            ]),
            regrets: vec![20, -10],
            strategy_sums: vec![500, 500],
        },
        // Zero-mass entry: seat 0, turn, bucket 3, all strategy sums zero
        SparseSnapshotEntry {
            key: make_key(0, Street::Turn, 3, 0x1111, 0x2222, 0xCCCC, 8),
            num_actions: 2,
            action_identity: Some(vec![
                action(SparseActionKind::Check, 0, 0),
                action(SparseActionKind::AllInCall, 0, 1),
            ]),
            regrets: vec![0, 0],
            strategy_sums: vec![0, 0],
        },
        // Long history (>32 actions, hash-only identity)
        SparseSnapshotEntry {
            key: make_key(
                2,
                Street::River,
                7,
                0xDEAD_BEEF_CAFE_BABE,
                0x0123_4567_89AB_CDEF,
                0xFFFF_FFFF_FFFF_FFFF,
                40,
            ),
            num_actions: 4,
            action_identity: Some(vec![
                action(SparseActionKind::Fold, 0, 0),
                action(SparseActionKind::Call, 0, 1),
                action(SparseActionKind::Raise, 80, 2),
                action(SparseActionKind::AllInBetRaise, 0, 3),
            ]),
            regrets: vec![1, 2, 3, 4],
            strategy_sums: vec![10, 20, 30, 40],
        },
    ]
}

fn default_lazy_config() -> mp_lazy_export::LazyExportConfig {
    mp_lazy_export::LazyExportConfig {
        num_players: 6,
        stack_depth: 100.0,
        bucket_counts: [169, 100, 50, 50],
        small_blind: 1.0,
        big_blind: 2.0,
    }
}

fn default_training_info() -> LazyTrainingInfo {
    LazyTrainingInfo {
        iterations: 100,
        elapsed_minutes: 1.0,
    }
}

// ── Acceptance test: full round-trip ─────────────────────────────────

#[test]
fn acceptance_lazy_export_round_trip() {
    let entries = build_test_entries();

    let mut config = default_lazy_config();
    config.small_blind = 0.5;
    config.big_blind = 1.0;

    let training = LazyTrainingInfo {
        iterations: 1000,
        elapsed_minutes: 5.0,
    };

    // Export in memory
    let output = mp_lazy_export::export_lazy_sparse_to_universal(&config, &entries, &training);

    // Write to tempdir
    let dir = tempfile::tempdir().unwrap();
    mp_lazy_export::write_lazy_bundle(dir.path(), &output).unwrap();

    // Read back via BundleReader
    let reader = BundleReader::open(dir.path()).unwrap();

    // Verify row count matches entries
    assert_eq!(reader.row_count(), entries.len());

    // Rebuild storage from entries to get reference average_strategy
    let storage = SparseMpStorage::from_snapshot_entries(entries.clone());

    for i in 0..reader.row_count() {
        let row = reader.row(i).unwrap();
        let sem = reader.semantic_record(i).unwrap();

        // Find the matching entry by semantic identity
        let matching_entry = entries
            .iter()
            .find(|e| {
                e.key.seat == row.seat
                    && e.key.bucket_street().unwrap() as u8 == row.street
                    && e.key.local_bucket() == row.local_bucket
                    && u32::from(e.key.bucket) == row.global_bucket
                    && e.key.history_hi == sem.history_hi
                    && e.key.history_lo == sem.history_lo
                    && e.key.history_hash == sem.history_hash
                    && e.key.history_len == sem.history_len
            })
            .expect("every exported row must match an input entry verbatim");

        // Verify probabilities match average_strategy
        let num_actions = matching_entry.num_actions as usize;
        let mut expected = vec![0.0_f64; num_actions];
        storage.average_strategy(matching_entry.key, num_actions, &mut expected);

        for j in 0..num_actions {
            let prob_idx = row.prob_offset as usize + j;
            let exported = reader.prob(prob_idx).unwrap();
            let reference = expected[j] as f32;
            assert!(
                (exported - reference).abs() < 1e-7,
                "row {i} action {j}: exported {exported} != reference {reference}"
            );
        }

        // Verify zero-mass rows are uniform
        if matching_entry.strategy_sums.iter().all(|&s| s == 0) {
            let uniform = 1.0_f32 / num_actions as f32;
            for j in 0..num_actions {
                let prob_idx = row.prob_offset as usize + j;
                let exported = reader.prob(prob_idx).unwrap();
                assert!(
                    (exported - uniform).abs() < 1e-7,
                    "zero-mass row {i} action {j}: expected uniform {uniform}, got {exported}"
                );
            }
        }

        // Verify action descriptors carry real action identity when present.
        let identity = matching_entry
            .action_identity
            .as_ref()
            .expect("test fixture should carry action identity");
        for (j, expected) in identity.iter().enumerate() {
            let action_idx = row.action_offset as usize + j;
            let action = reader.action(action_idx).unwrap();
            assert_eq!(action.kind, universal_kind(expected.kind));
            assert_eq!(action.amount_chips, expected.amount_chips);
            assert_eq!(action.source_action_index, expected.source_action_index);
            assert_eq!(action.is_aggressive, is_aggressive(expected.kind));
        }

        // Verify source_node_idx is u32::MAX (semantic-only)
        assert_eq!(row.source_node_idx, u32::MAX);
    }

    // Verify non-resumable: no cfr.snapshot.bin, compatibility says not resumable
    assert!(!dir.path().join("cfr.snapshot.bin").exists());

    // Verify manifest has required features
    let manifest_text = std::fs::read_to_string(dir.path().join("blueprint.json")).unwrap();
    let manifest: serde_json::Value = serde_json::from_str(&manifest_text).unwrap();
    let required_features = manifest["required_features"].as_array().unwrap();
    let feature_strings: Vec<&str> = required_features
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(feature_strings.contains(&"mp_semantic_rows_v1"));

    // Verify training metadata
    assert_eq!(
        manifest["training"]["source_backend"],
        "mp_lazy_sparse_projected"
    );
    assert_eq!(manifest["training"]["unit_kind"], "MetaIteration");

    // Verify layout
    assert_eq!(manifest["layout"]["missing_row_policy"], "uniform_legal");
    assert_eq!(manifest["layout"]["row_namespace"][0], "mp_semantic");

    // Verify compatibility
    assert_eq!(manifest["compatibility"]["resumable"], false);
}

// ── Side-table format tests ──────────────────────────────────────────

#[test]
fn semantic_record_round_trip() {
    use poker_solver_core::blueprint_universal::mp_lazy_export::SemanticKeyRecord;

    let record = SemanticKeyRecord {
        history_hi: 0xDEAD_BEEF_CAFE_BABE,
        history_lo: 0x0123_4567_89AB_CDEF,
        history_hash: 0xFFFF_FFFF_FFFF_FFFF,
        history_len: 40,
    };

    let mut buf = [0u8; 32];
    record.write_to(&mut buf);
    let decoded = SemanticKeyRecord::from_bytes(&buf);
    assert_eq!(decoded, record);
}

// ── Feature gating tests ─────────────────────────────────────────────

#[test]
fn bundle_with_unknown_required_feature_rejected() {
    // Build a minimal valid bundle, then modify manifest to add unknown feature
    let entries = build_test_entries();
    let config = default_lazy_config();
    let training = default_training_info();
    let output = mp_lazy_export::export_lazy_sparse_to_universal(&config, &entries, &training);
    let dir = tempfile::tempdir().unwrap();
    mp_lazy_export::write_lazy_bundle(dir.path(), &output).unwrap();

    // Tamper: add unknown required feature to manifest
    let manifest_path = dir.path().join("blueprint.json");
    let mut manifest_text = std::fs::read_to_string(&manifest_path).unwrap();
    manifest_text = manifest_text.replace(
        "\"mp_semantic_rows_v1\"",
        "\"mp_semantic_rows_v1\", \"unknown_future_feature_v99\"",
    );
    std::fs::write(&manifest_path, &manifest_text).unwrap();

    let err = BundleReader::open(dir.path()).unwrap_err();
    let err_msg = format!("{err}");
    assert!(
        err_msg.contains("unknown_future_feature_v99"),
        "expected rejection mentioning unknown feature, got: {err_msg}"
    );
}

// ── Backward compatibility ───────────────────────────────────────────

#[test]
fn mp_semantic_feature_accepted_by_reader() {
    // The updated reader must accept mp_semantic_rows_v1
    let entries = build_test_entries();
    let config = default_lazy_config();
    let training = default_training_info();
    let output = mp_lazy_export::export_lazy_sparse_to_universal(&config, &entries, &training);
    let dir = tempfile::tempdir().unwrap();
    mp_lazy_export::write_lazy_bundle(dir.path(), &output).unwrap();

    // Must succeed — the reader supports mp_semantic_rows_v1
    let reader = BundleReader::open(dir.path()).unwrap();
    assert_eq!(reader.row_count(), entries.len());
}

// ── Disk wrapper test ────────────────────────────────────────────────

/// Helper: write a minimal BlueprintMpConfig-format config.yaml to dir.
fn write_real_mp_config_yaml(dir: &std::path::Path) {
    let yaml = r#"
game:
  name: "test"
  num_players: 6
  stack_depth: 100.0
  blinds:
    - seat: 4
      type: small_blind
      amount: 1
    - seat: 5
      type: big_blind
      amount: 2
clustering:
  preflop:
    buckets: 169
  flop:
    buckets: 100
  turn:
    buckets: 50
  river:
    buckets: 50
action_abstraction:
  preflop:
    lead: ["2bb"]
    raise: [["1.0x"]]
  flop:
    lead: [0.75]
    raise: [[1.0]]
  turn:
    lead: [1.0]
    raise: [[1.0]]
  river:
    lead: [1.0]
    raise: [[1.0]]
training:
  backend: lazy_sparse
  iterations: 1
snapshots:
  warmup_minutes: 0
  snapshot_every_minutes: 1
  output_dir: "."
"#;
    std::fs::write(dir.join("config.yaml"), yaml).unwrap();
}

#[test]
fn disk_wrapper_exports_with_real_directory_structure() {
    let entries = build_test_entries();

    // Create bundle_dir/snapshot_0000/ mirroring real training layout
    let bundle_dir = tempfile::tempdir().unwrap();
    let snapshot_dir = bundle_dir.path().join("snapshot_0000");
    std::fs::create_dir_all(&snapshot_dir).unwrap();

    // config.yaml lives in bundle_dir (parent), NOT snapshot subdirectory
    write_real_mp_config_yaml(bundle_dir.path());

    // sparse_entries.bin and metadata.json live in snapshot subdirectory
    let file = std::fs::File::create(snapshot_dir.join("sparse_entries.bin")).unwrap();
    let writer = std::io::BufWriter::new(file);
    bincode::serialize_into(writer, &entries).unwrap();

    let metadata = serde_json::json!({
        "kind": "blueprint_mp_lazy_sparse",
        "iterations": 500,
        "elapsed_minutes": 3,
    });
    std::fs::write(
        snapshot_dir.join("metadata.json"),
        serde_json::to_string_pretty(&metadata).unwrap(),
    )
    .unwrap();

    // Export using bundle_dir + snapshot_name (matching real CLI usage)
    let out_dir = tempfile::tempdir().unwrap();
    mp_lazy_export::export_lazy_bundle_from_disk(
        bundle_dir.path(),
        "snapshot_0000",
        out_dir.path(),
    )
    .unwrap();

    let reader = BundleReader::open(out_dir.path()).unwrap();
    assert_eq!(reader.row_count(), entries.len());

    // Verify blinds from config.yaml (SB=1, BB=2) are in the manifest
    let manifest_text = std::fs::read_to_string(out_dir.path().join("blueprint.json")).unwrap();
    let manifest: serde_json::Value = serde_json::from_str(&manifest_text).unwrap();
    assert_eq!(manifest["game"]["small_blind"], 1.0);
    assert_eq!(manifest["game"]["big_blind"], 2.0);
}

#[test]
fn disk_wrapper_rejects_wrong_kind() {
    let bundle_dir = tempfile::tempdir().unwrap();
    let snapshot_dir = bundle_dir.path().join("snapshot_0000");
    std::fs::create_dir_all(&snapshot_dir).unwrap();

    write_real_mp_config_yaml(bundle_dir.path());

    // Write empty sparse_entries.bin
    std::fs::write(snapshot_dir.join("sparse_entries.bin"), &[]).unwrap();

    // Write metadata.json with wrong kind
    let metadata = serde_json::json!({
        "kind": "blueprint_mp",  // Wrong! This is eager, not lazy
        "iterations": 100,
    });
    std::fs::write(
        snapshot_dir.join("metadata.json"),
        serde_json::to_string_pretty(&metadata).unwrap(),
    )
    .unwrap();

    let out_dir = tempfile::tempdir().unwrap();
    let err = mp_lazy_export::export_lazy_bundle_from_disk(
        bundle_dir.path(),
        "snapshot_0000",
        out_dir.path(),
    )
    .unwrap_err();
    let err_msg = format!("{err}");
    assert!(
        err_msg.contains("blueprint_mp_lazy_sparse") || err_msg.contains("kind"),
        "expected kind mismatch error, got: {err_msg}"
    );
}

// ── Truncated side table rejected ────────────────────────────────────

#[test]
fn truncated_semantic_file_rejected() {
    let entries = build_test_entries();
    let config = default_lazy_config();
    let training = default_training_info();
    let output = mp_lazy_export::export_lazy_sparse_to_universal(&config, &entries, &training);
    let dir = tempfile::tempdir().unwrap();
    mp_lazy_export::write_lazy_bundle(dir.path(), &output).unwrap();

    // Truncate the semantic file
    let sem_path = dir.path().join("strategy.semantic.bin");
    let data = std::fs::read(&sem_path).unwrap();
    // Write only the first 20 bytes (less than header)
    std::fs::write(&sem_path, &data[..20]).unwrap();

    // Update manifest with new size so length check passes but header
    // validation fails
    let manifest_path = dir.path().join("blueprint.json");
    let manifest_text = std::fs::read_to_string(&manifest_path).unwrap();
    let mut manifest: serde_json::Value = serde_json::from_str(&manifest_text).unwrap();
    manifest["files"]["strategy.semantic.bin"]["size"] = 20.into();
    // Recompute SHA-256
    let sha = hex::encode(sha2::Sha256::digest(&data[..20]));
    manifest["files"]["strategy.semantic.bin"]["sha256"] = serde_json::Value::String(sha);
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let err = BundleReader::open(dir.path()).unwrap_err();
    let err_msg = format!("{err}");
    assert!(
        err_msg.contains("truncated") || err_msg.contains("Truncated"),
        "expected truncation error, got: {err_msg}"
    );
}

// ── Bad CRC rejected ─────────────────────────────────────────────────

#[test]
fn bad_crc_semantic_file_rejected() {
    let entries = build_test_entries();
    let config = default_lazy_config();
    let training = default_training_info();
    let output = mp_lazy_export::export_lazy_sparse_to_universal(&config, &entries, &training);
    let dir = tempfile::tempdir().unwrap();
    mp_lazy_export::write_lazy_bundle(dir.path(), &output).unwrap();

    // Corrupt a payload byte in the semantic file (after header)
    let sem_path = dir.path().join("strategy.semantic.bin");
    let mut data = std::fs::read(&sem_path).unwrap();
    if data.len() > 48 {
        data[49] ^= 0xFF;
    }
    std::fs::write(&sem_path, &data).unwrap();

    // Update manifest with new SHA-256 and size
    let manifest_path = dir.path().join("blueprint.json");
    let manifest_text = std::fs::read_to_string(&manifest_path).unwrap();
    let mut manifest: serde_json::Value = serde_json::from_str(&manifest_text).unwrap();
    let sha = hex::encode(sha2::Sha256::digest(&data));
    manifest["files"]["strategy.semantic.bin"]["sha256"] = serde_json::Value::String(sha);
    manifest["files"]["strategy.semantic.bin"]["size"] =
        serde_json::Value::Number(data.len().into());
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let err = BundleReader::open(dir.path()).unwrap_err();
    let err_msg = format!("{err}");
    assert!(
        err_msg.contains("CRC") || err_msg.contains("crc"),
        "expected CRC error, got: {err_msg}"
    );
}

// ── Semantic key offset out of range rejected ────────────────────────

/// Test that the reader rejects rows with out-of-range semantic key
/// offsets. To avoid CRC validation interfering, we write a bundle
/// where the semantic side table has fewer records than the rows claim.
#[test]
fn semantic_key_offset_out_of_range_rejected() {
    // Build a single-entry export
    let entries = vec![SparseSnapshotEntry {
        key: make_key(0, Street::Preflop, 5, 1, 2, 3, 4),
        num_actions: 2,
        action_identity: None,
        regrets: vec![1, 2],
        strategy_sums: vec![10, 20],
    }];
    let config = default_lazy_config();
    let training = default_training_info();
    let mut output = mp_lazy_export::export_lazy_sparse_to_universal(&config, &entries, &training);

    // Tamper: clear semantic records so offset 0 is out of range
    output.semantic_records.clear();

    let dir = tempfile::tempdir().unwrap();
    mp_lazy_export::write_lazy_bundle(dir.path(), &output).unwrap();

    let err = BundleReader::open(dir.path()).unwrap_err();
    let err_msg = format!("{err}");
    assert!(
        err_msg.contains("semantic_key_offset") || err_msg.contains("out of range"),
        "expected semantic key offset error, got: {err_msg}"
    );
}

// ── Unknown semantic key kind rejected ───────────────────────────────

#[test]
fn unknown_semantic_key_kind_rejected() {
    let entries = vec![SparseSnapshotEntry {
        key: make_key(0, Street::Preflop, 5, 1, 2, 3, 4),
        num_actions: 2,
        action_identity: None,
        regrets: vec![1, 2],
        strategy_sums: vec![10, 20],
    }];
    let config = default_lazy_config();
    let training = default_training_info();
    let mut output = mp_lazy_export::export_lazy_sparse_to_universal(&config, &entries, &training);

    // Tamper: set an unknown semantic key kind on the row
    output.rows[0].semantic_key_kind = 99;

    let dir = tempfile::tempdir().unwrap();
    mp_lazy_export::write_lazy_bundle(dir.path(), &output).unwrap();

    let err = BundleReader::open(dir.path()).unwrap_err();
    let err_msg = format!("{err}");
    assert!(
        err_msg.contains("unknown semantic_key_kind"),
        "expected unknown kind error, got: {err_msg}"
    );
}

// ── Opaque actions without mp_semantic_rows_v1 feature gating ───────

/// Bundles whose actions contain kind=Opaque but whose manifest does NOT
/// declare `mp_semantic_rows_v1` must be rejected by the reader.
#[test]
fn opaque_actions_without_semantic_feature_rejected() {
    let mut entries = build_test_entries();
    for entry in &mut entries {
        entry.action_identity = None;
    }
    let config = default_lazy_config();
    let training = default_training_info();
    let output = mp_lazy_export::export_lazy_sparse_to_universal(&config, &entries, &training);
    let dir = tempfile::tempdir().unwrap();
    mp_lazy_export::write_lazy_bundle(dir.path(), &output).unwrap();

    // Tamper: remove mp_semantic_rows_v1 from required_features,
    // leaving the Opaque actions in the binary payload.
    let manifest_path = dir.path().join("blueprint.json");
    let manifest_text = std::fs::read_to_string(&manifest_path).unwrap();
    let mut manifest: serde_json::Value = serde_json::from_str(&manifest_text).unwrap();
    manifest["required_features"] = serde_json::Value::Array(Vec::new());
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let err = BundleReader::open(dir.path()).unwrap_err();
    let err_msg = format!("{err}");
    assert!(
        err_msg.contains("Opaque") && err_msg.contains("mp_semantic_rows_v1"),
        "expected error about Opaque actions without \
         mp_semantic_rows_v1 feature, got: {err_msg}"
    );
}

#[test]
fn entries_without_action_identity_export_opaque_actions() {
    let entries = vec![SparseSnapshotEntry {
        key: make_key(0, Street::Preflop, 5, 1, 2, 3, 4),
        num_actions: 2,
        action_identity: None,
        regrets: vec![1, 2],
        strategy_sums: vec![10, 20],
    }];
    let config = default_lazy_config();
    let training = default_training_info();
    let output = mp_lazy_export::export_lazy_sparse_to_universal(&config, &entries, &training);

    assert_eq!(output.actions.len(), 2);
    for (idx, action) in output.actions.iter().enumerate() {
        assert_eq!(action.kind, ActionKind::Opaque);
        assert_eq!(action.source_action_index, idx as u16);
        assert_eq!(action.amount_chips, 0);
    }
}
