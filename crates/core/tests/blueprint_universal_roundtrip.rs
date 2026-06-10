//! Integration tests for the universal dense blueprint format.
//!
//! Tests bundle-level write -> read round-trips with synthetic data,
//! plus deterministic rejection for every known bad-input class.

use poker_solver_core::blueprint_universal::{
    ActionDescriptor, ActionKind, BundleReader, BundleWriter, FormatError, Manifest,
    RowDescriptor,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal valid manifest for testing.
fn test_manifest(row_count: u64, action_count: u64, prob_count: u64) -> Manifest {
    Manifest::builder()
        .format_name("dense_blueprint".to_string())
        .format_version(1)
        .compat_min_reader(1)
        .producer("test-harness".to_string())
        .producer_git("unknown".to_string())
        .num_players(2)
        .row_count(row_count)
        .action_descriptor_count(action_count)
        .probability_count(prob_count)
        .normalization_tolerance(1e-4)
        .build()
}

/// Build synthetic rows, actions, and probabilities for `n` rows,
/// each with `actions_per_row` uniform actions.
fn synthetic_data(
    n: u64,
    actions_per_row: u16,
) -> (Vec<RowDescriptor>, Vec<ActionDescriptor>, Vec<f32>) {
    let mut rows = Vec::new();
    let mut actions = Vec::new();
    let mut probs = Vec::new();

    for i in 0..n {
        let action_offset = i * u64::from(actions_per_row);
        let prob_offset = action_offset;

        rows.push(RowDescriptor {
            row_id: i,
            namespace: 0, // hu_arena
            seat: 0,
            street: 0,
            local_bucket: i as u16,
            global_bucket: i as u32,
            source_node_idx: i as u32,
            action_offset,
            action_count: actions_per_row,
            prob_offset,
            row_key_fingerprint: i,
            action_schema_fingerprint: 1000 + i,
            semantic_key_kind: 0,
            semantic_key_offset: 0,
        });

        for a in 0..actions_per_row {
            actions.push(ActionDescriptor {
                kind: if a == 0 {
                    ActionKind::Fold
                } else {
                    ActionKind::Call
                },
                amount_chips: 0,
                size_key: String::new(),
                label: String::new(),
                is_aggressive: false,
                source_action_index: a,
            });
            probs.push(1.0 / f64::from(actions_per_row) as f32);
        }
    }

    (rows, actions, probs)
}

// ---------------------------------------------------------------------------
// Round-trip tests
// ---------------------------------------------------------------------------

#[test]
fn empty_bundle_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    let manifest = test_manifest(0, 0, 0);
    BundleWriter::write(dir.path(), &manifest, &[], &[], &[]).expect("write");
    let reader = BundleReader::open(dir.path()).expect("read");
    assert_eq!(reader.row_count(), 0);
}

#[test]
fn synthetic_bundle_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (rows, actions, probs) = synthetic_data(10, 3);
    let manifest = test_manifest(10, 30, 30);

    BundleWriter::write(dir.path(), &manifest, &rows, &actions, &probs).expect("write");
    let reader = BundleReader::open(dir.path()).expect("read");

    assert_eq!(reader.row_count(), 10);

    // Verify every row round-trips identically.
    for (i, original) in rows.iter().enumerate() {
        let loaded = reader.row(i).expect("row exists");
        assert_eq!(loaded.row_id, original.row_id);
        assert_eq!(loaded.namespace, original.namespace);
        assert_eq!(loaded.seat, original.seat);
        assert_eq!(loaded.street, original.street);
        assert_eq!(loaded.global_bucket, original.global_bucket);
        assert_eq!(loaded.action_count, original.action_count);
        assert_eq!(loaded.row_key_fingerprint, original.row_key_fingerprint);
    }

    // Verify actions round-trip.
    for (i, original) in actions.iter().enumerate() {
        let loaded = reader.action(i).expect("action exists");
        assert_eq!(loaded.kind, original.kind);
        assert_eq!(loaded.amount_chips, original.amount_chips);
        assert_eq!(loaded.is_aggressive, original.is_aggressive);
        assert_eq!(loaded.source_action_index, original.source_action_index);
        assert_eq!(loaded.size_key, original.size_key);
        assert_eq!(loaded.label, original.label);
    }

    // Verify probabilities round-trip.
    for (i, &original) in probs.iter().enumerate() {
        let loaded = reader.prob(i).expect("prob exists");
        assert!((loaded - original).abs() < 1e-7);
    }
}

// ---------------------------------------------------------------------------
// Rejection tests
// ---------------------------------------------------------------------------

#[test]
fn rejects_bad_magic() {
    let dir = tempfile::tempdir().expect("tempdir");
    let manifest = test_manifest(1, 2, 2);
    let (rows, actions, probs) = synthetic_data(1, 2);
    BundleWriter::write(dir.path(), &manifest, &rows, &actions, &probs).expect("write");

    // Corrupt the rows file magic bytes.
    let rows_path = dir.path().join("strategy.rows.bin");
    let mut data = std::fs::read(&rows_path).expect("read");
    data[0..4].copy_from_slice(b"JUNK");
    std::fs::write(&rows_path, &data).expect("write");

    let err = BundleReader::open(dir.path()).unwrap_err();
    assert!(
        matches!(err, FormatError::BadMagic { .. }),
        "expected BadMagic, got {err:?}"
    );
}

#[test]
fn rejects_unsupported_format_version() {
    let dir = tempfile::tempdir().expect("tempdir");
    let manifest = test_manifest(1, 2, 2);
    let (rows, actions, probs) = synthetic_data(1, 2);
    BundleWriter::write(dir.path(), &manifest, &rows, &actions, &probs).expect("write");

    // Bump the format_version in the rows binary header to 99.
    let rows_path = dir.path().join("strategy.rows.bin");
    let mut data = std::fs::read(&rows_path).expect("read");
    data[8..10].copy_from_slice(&99_u16.to_le_bytes());
    std::fs::write(&rows_path, &data).expect("write");

    let err = BundleReader::open(dir.path()).unwrap_err();
    assert!(
        matches!(err, FormatError::UnsupportedFormatVersion { .. }),
        "expected UnsupportedFormatVersion, got {err:?}"
    );
}

#[test]
fn rejects_compat_min_reader_too_new() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut manifest = test_manifest(0, 0, 0);
    manifest.compat_min_reader = 999;
    BundleWriter::write(dir.path(), &manifest, &[], &[], &[]).expect("write");

    let err = BundleReader::open(dir.path()).unwrap_err();
    assert!(
        matches!(err, FormatError::UnsupportedFormatVersion { .. }),
        "expected UnsupportedFormatVersion, got {err:?}"
    );
}

#[test]
fn rejects_unknown_required_feature() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut manifest = test_manifest(0, 0, 0);
    manifest.required_features = vec!["quantum_cfr".to_string()];
    BundleWriter::write(dir.path(), &manifest, &[], &[], &[]).expect("write");

    let err = BundleReader::open(dir.path()).unwrap_err();
    assert!(
        matches!(err, FormatError::UnsupportedRequiredFeature { .. }),
        "expected UnsupportedRequiredFeature, got {err:?}"
    );
}

#[test]
fn rejects_sha256_mismatch() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (rows, actions, probs) = synthetic_data(1, 2);
    let manifest = test_manifest(1, 2, 2);
    BundleWriter::write(dir.path(), &manifest, &rows, &actions, &probs).expect("write");

    // Corrupt a payload byte (past the header) in probs file.
    let probs_path = dir.path().join("strategy.probs.f32.bin");
    let mut data = std::fs::read(&probs_path).expect("read");
    let last = data.len() - 1;
    data[last] ^= 0xFF;
    std::fs::write(&probs_path, &data).expect("write");

    let err = BundleReader::open(dir.path()).unwrap_err();
    assert!(
        matches!(err, FormatError::ChecksumMismatch { .. }),
        "expected ChecksumMismatch, got {err:?}"
    );
}

#[test]
fn rejects_crc64_mismatch() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (rows, actions, probs) = synthetic_data(2, 3);
    let manifest = test_manifest(2, 6, 6);
    BundleWriter::write(dir.path(), &manifest, &rows, &actions, &probs).expect("write");

    // Corrupt a payload byte past the header in the rows file, but also
    // update the SHA-256 in blueprint.json so the SHA check passes but CRC fails.
    let rows_path = dir.path().join("strategy.rows.bin");
    let mut data = std::fs::read(&rows_path).expect("read");
    // The header is 48 bytes; corrupt the first payload byte.
    if data.len() > 48 {
        data[48] ^= 0xFF;
    }
    std::fs::write(&rows_path, &data).expect("write");

    // Recompute SHA-256 for the corrupted file and update manifest.
    use sha2::{Digest, Sha256};
    let new_sha = hex::encode(Sha256::digest(&data));
    let manifest_path = dir.path().join("blueprint.json");
    let manifest_text = std::fs::read_to_string(&manifest_path).expect("read");
    // Replace the old SHA in the manifest JSON.
    let manifest_json: serde_json::Value =
        serde_json::from_str(&manifest_text).expect("parse");
    let mut manifest_json = manifest_json;
    let files = manifest_json["files"].as_object_mut().unwrap();
    let entry = files
        .get_mut("strategy.rows.bin")
        .unwrap()
        .as_object_mut()
        .unwrap();
    entry.insert("sha256".to_string(), serde_json::Value::String(new_sha.clone()));
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest_json).unwrap(),
    )
    .expect("write");

    // Also update checksums.json.
    let checksums_path = dir.path().join("checksums.json");
    let checksums_text = std::fs::read_to_string(&checksums_path).expect("read");
    let mut checksums: serde_json::Value =
        serde_json::from_str(&checksums_text).expect("parse");
    checksums["strategy.rows.bin"] = serde_json::Value::String(new_sha);
    std::fs::write(
        &checksums_path,
        serde_json::to_string_pretty(&checksums).unwrap(),
    )
    .expect("write");

    let err = BundleReader::open(dir.path()).unwrap_err();
    assert!(
        matches!(err, FormatError::CrcMismatch { .. }),
        "expected CrcMismatch, got {err:?}"
    );
}

#[test]
fn rejects_truncated_mid_header() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (rows, actions, probs) = synthetic_data(1, 2);
    let manifest = test_manifest(1, 2, 2);
    BundleWriter::write(dir.path(), &manifest, &rows, &actions, &probs).expect("write");

    // Truncate the rows file mid-header (at 20 bytes, header is 48).
    // The manifest still records the original size, so LengthMismatch fires
    // before the header parse. Both are correct rejection of truncated data.
    let rows_path = dir.path().join("strategy.rows.bin");
    let data = std::fs::read(&rows_path).expect("read");
    std::fs::write(&rows_path, &data[..20]).expect("write");

    let err = BundleReader::open(dir.path()).unwrap_err();
    assert!(
        matches!(
            err,
            FormatError::Truncated { .. } | FormatError::LengthMismatch { .. }
        ),
        "expected Truncated or LengthMismatch, got {err:?}"
    );
}

#[test]
fn rejects_truncated_mid_header_consistent_manifest() {
    // Test truncation where the manifest size matches the truncated file
    // (simulates a corrupt writer that wrote wrong size). This should detect
    // the file is too short for even the header.
    let dir = tempfile::tempdir().expect("tempdir");
    let (rows, actions, probs) = synthetic_data(1, 2);
    let manifest = test_manifest(1, 2, 2);
    BundleWriter::write(dir.path(), &manifest, &rows, &actions, &probs).expect("write");

    let rows_path = dir.path().join("strategy.rows.bin");
    let data = std::fs::read(&rows_path).expect("read");
    let truncated = &data[..20];
    std::fs::write(&rows_path, truncated).expect("write");

    // Update manifest to claim the truncated size and new SHA.
    use sha2::{Digest, Sha256};
    let new_sha = hex::encode(Sha256::digest(truncated));
    let manifest_path = dir.path().join("blueprint.json");
    let text = std::fs::read_to_string(&manifest_path).expect("read");
    let mut json: serde_json::Value = serde_json::from_str(&text).expect("parse");
    let entry = json["files"]["strategy.rows.bin"].as_object_mut().unwrap();
    entry.insert("size".to_string(), serde_json::json!(20));
    entry.insert("sha256".to_string(), serde_json::json!(new_sha));
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&json).unwrap()).expect("write");

    let err = BundleReader::open(dir.path()).unwrap_err();
    assert!(
        matches!(err, FormatError::Truncated { .. }),
        "expected Truncated, got {err:?}"
    );
}

#[test]
fn rejects_truncated_mid_payload() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (rows, actions, probs) = synthetic_data(3, 2);
    let manifest = test_manifest(3, 6, 6);
    BundleWriter::write(dir.path(), &manifest, &rows, &actions, &probs).expect("write");

    // Truncate the rows file mid-payload (keep header + partial rows).
    let rows_path = dir.path().join("strategy.rows.bin");
    let data = std::fs::read(&rows_path).expect("read");
    let truncated_len = 48 + 40; // header + less than 1 full row
    if data.len() > truncated_len {
        std::fs::write(&rows_path, &data[..truncated_len]).expect("write");
    }

    let err = BundleReader::open(dir.path()).unwrap_err();
    assert!(
        matches!(
            err,
            FormatError::Truncated { .. }
                | FormatError::LengthMismatch { .. }
                | FormatError::ChecksumMismatch { .. }
        ),
        "expected Truncated or LengthMismatch, got {err:?}"
    );
}

#[test]
fn rejects_length_mismatch() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (rows, actions, probs) = synthetic_data(1, 2);
    let manifest = test_manifest(1, 2, 2);
    BundleWriter::write(dir.path(), &manifest, &rows, &actions, &probs).expect("write");

    // Modify the manifest to claim wrong file size.
    let manifest_path = dir.path().join("blueprint.json");
    let text = std::fs::read_to_string(&manifest_path).expect("read");
    let mut json: serde_json::Value = serde_json::from_str(&text).expect("parse");
    let files = json["files"].as_object_mut().unwrap();
    let entry = files
        .get_mut("strategy.rows.bin")
        .unwrap()
        .as_object_mut()
        .unwrap();
    entry.insert("size".to_string(), serde_json::Value::Number(9999.into()));
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&json).unwrap()).expect("write");

    let err = BundleReader::open(dir.path()).unwrap_err();
    assert!(
        matches!(err, FormatError::LengthMismatch { .. }),
        "expected LengthMismatch, got {err:?}"
    );
}

#[test]
fn rejects_duplicate_row_identities() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (mut rows, actions, probs) = synthetic_data(2, 2);
    // Make the two rows have identical identity keys.
    rows[1].namespace = rows[0].namespace;
    rows[1].seat = rows[0].seat;
    rows[1].street = rows[0].street;
    rows[1].source_node_idx = rows[0].source_node_idx;
    rows[1].global_bucket = rows[0].global_bucket;
    rows[1].row_key_fingerprint = rows[0].row_key_fingerprint;

    let manifest = test_manifest(2, 4, 4);
    BundleWriter::write(dir.path(), &manifest, &rows, &actions, &probs).expect("write");

    let err = BundleReader::open(dir.path()).unwrap_err();
    assert!(
        matches!(err, FormatError::DuplicateRowIdentity { .. }),
        "expected DuplicateRowIdentity, got {err:?}"
    );
}

#[test]
fn rejects_unsorted_rows() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (mut rows, actions, probs) = synthetic_data(3, 2);
    // Swap row 0 and row 2 so they're out of order.
    rows.swap(0, 2);

    let manifest = test_manifest(3, 6, 6);
    BundleWriter::write(dir.path(), &manifest, &rows, &actions, &probs).expect("write");

    let err = BundleReader::open(dir.path()).unwrap_err();
    assert!(
        matches!(err, FormatError::RowsNotSorted { .. }),
        "expected RowsNotSorted, got {err:?}"
    );
}

#[test]
fn rejects_invalid_offset() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (mut rows, actions, probs) = synthetic_data(1, 2);
    // Set action_offset past the end.
    rows[0].action_offset = 9999;

    let manifest = test_manifest(1, 2, 2);
    BundleWriter::write(dir.path(), &manifest, &rows, &actions, &probs).expect("write");

    let err = BundleReader::open(dir.path()).unwrap_err();
    assert!(
        matches!(err, FormatError::InvalidOffset { .. }),
        "expected InvalidOffset, got {err:?}"
    );
}

#[test]
fn rejects_non_normalized_nan() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (rows, actions, _) = synthetic_data(1, 2);
    let probs = vec![f32::NAN, 0.5];

    let manifest = test_manifest(1, 2, 2);
    BundleWriter::write(dir.path(), &manifest, &rows, &actions, &probs).expect("write");

    let err = BundleReader::open(dir.path()).unwrap_err();
    assert!(
        matches!(err, FormatError::NonNormalizedRow { .. }),
        "expected NonNormalizedRow, got {err:?}"
    );
}

#[test]
fn rejects_non_normalized_negative() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (rows, actions, _) = synthetic_data(1, 2);
    let probs = vec![-0.5, 1.5];

    let manifest = test_manifest(1, 2, 2);
    BundleWriter::write(dir.path(), &manifest, &rows, &actions, &probs).expect("write");

    let err = BundleReader::open(dir.path()).unwrap_err();
    assert!(
        matches!(err, FormatError::NonNormalizedRow { .. }),
        "expected NonNormalizedRow, got {err:?}"
    );
}

#[test]
fn rejects_non_normalized_bad_sum() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (rows, actions, _) = synthetic_data(1, 2);
    let probs = vec![0.3, 0.3]; // sum = 0.6, not 1.0

    let manifest = test_manifest(1, 2, 2);
    BundleWriter::write(dir.path(), &manifest, &rows, &actions, &probs).expect("write");

    let err = BundleReader::open(dir.path()).unwrap_err();
    assert!(
        matches!(err, FormatError::NonNormalizedRow { .. }),
        "expected NonNormalizedRow, got {err:?}"
    );
}

#[test]
fn rejects_missing_payload_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (rows, actions, probs) = synthetic_data(1, 2);
    let manifest = test_manifest(1, 2, 2);
    BundleWriter::write(dir.path(), &manifest, &rows, &actions, &probs).expect("write");

    // Delete the probs file.
    std::fs::remove_file(dir.path().join("strategy.probs.f32.bin")).expect("remove");

    let err = BundleReader::open(dir.path()).unwrap_err();
    assert!(
        matches!(err, FormatError::MissingFile { .. }),
        "expected MissingFile, got {err:?}"
    );
}

#[test]
fn rejects_missing_manifest() {
    let dir = tempfile::tempdir().expect("tempdir");
    let err = BundleReader::open(dir.path()).unwrap_err();
    assert!(
        matches!(err, FormatError::MissingFile { .. } | FormatError::InvalidManifest { .. }),
        "expected MissingFile or InvalidManifest, got {err:?}"
    );
}

/// Finding 3: missing file entry in manifest.files must be a hard error,
/// not silently skipped.
#[test]
fn rejects_missing_file_entry_in_manifest() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (rows, actions, probs) = synthetic_data(1, 2);
    let manifest = test_manifest(1, 2, 2);
    BundleWriter::write(dir.path(), &manifest, &rows, &actions, &probs).expect("write");

    // Remove the strategy.probs.f32.bin entry from manifest.files
    let manifest_path = dir.path().join("blueprint.json");
    let text = std::fs::read_to_string(&manifest_path).expect("read");
    let mut json: serde_json::Value = serde_json::from_str(&text).expect("parse");
    let files = json["files"].as_object_mut().unwrap();
    files.remove("strategy.probs.f32.bin");
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&json).unwrap(),
    )
    .expect("write");

    let err = BundleReader::open(dir.path()).unwrap_err();
    assert!(
        matches!(err, FormatError::MissingFileEntry { .. }),
        "expected MissingFileEntry, got {err:?}"
    );
}

/// Finding 9: reader must reject format_name != "dense_blueprint".
#[test]
fn rejects_wrong_format_name() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut manifest = test_manifest(0, 0, 0);
    manifest.format_name = "something_else".to_string();
    BundleWriter::write(dir.path(), &manifest, &[], &[], &[]).expect("write");

    let err = BundleReader::open(dir.path()).unwrap_err();
    assert!(
        matches!(err, FormatError::InvalidFormatName { .. }),
        "expected InvalidFormatName, got {err:?}"
    );
}
