//! Bundle-level writer and reader for the universal dense blueprint format.
//!
//! The writer produces a complete bundle directory from in-memory rows,
//! actions, probabilities, and metadata. The reader validates and opens an
//! existing bundle for read-only row/action/probability lookup.

use std::collections::BTreeMap;
use std::io::{BufWriter, Cursor, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

use super::descriptors::{
    ActionDescriptor, RowDescriptor, SemanticKeyRecord,
    ACTION_DESCRIPTOR_SIZE, ROW_DESCRIPTOR_SIZE, SEMANTIC_RECORD_SIZE,
};
use super::error::FormatError;
use super::export_common::{SEMANTIC_KEY_MP_HISTORY_V1, SEMANTIC_KEY_NONE};
use super::header::{
    BinaryHeader, CURRENT_FORMAT_VERSION, HEADER_SIZE, MAGIC_ACTIONS, MAGIC_PROBS,
    MAGIC_ROWS, MAGIC_SEMANTIC,
};
use super::manifest::{FileEntry, Manifest};

/// CRC-64/XZ algorithm constant from the `crc` crate.
///
/// We use CRC-64/XZ (ECMA-182, also known as CRC-64/GO-ECMA) because it is
/// a widely implemented 64-bit CRC with good error-detection properties and
/// is available as a built-in constant in the `crc` crate v3.
const CRC_ALG: crc::Algorithm<u64> = crc::CRC_64_XZ;

/// Supported features that this reader understands.
const SUPPORTED_FEATURES: &[&str] = &["mp_semantic_rows_v1"];

/// Expected value of the `format_name` field.
const EXPECTED_FORMAT_NAME: &str = "dense_blueprint";

// ---------------------------------------------------------------------------
// Bundle data
// ---------------------------------------------------------------------------

/// The three payload arrays that make up a universal dense blueprint bundle.
pub struct BundleData<'a> {
    pub rows: &'a [RowDescriptor],
    pub actions: &'a [ActionDescriptor],
    pub probs: &'a [f32],
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Write a complete universal dense blueprint bundle to a directory.
///
/// # Errors
///
/// Returns `FormatError` on I/O failure.
pub fn write_bundle(
    dir: &Path,
    manifest: &Manifest,
    data: &BundleData<'_>,
) -> Result<(), FormatError> {
    std::fs::create_dir_all(dir)?;

    let rows_entry = write_rows_file(dir, data.rows)?;
    let actions_entry = write_actions_file(dir, data.actions)?;
    let probs_entry = write_probs_file(dir, data.probs)?;

    // Start with any pre-existing file entries (e.g. semantic side table)
    // then add/overwrite with the standard payload entries.
    let mut files = manifest.files.clone();
    files.insert("strategy.rows.bin".to_string(), rows_entry);
    files.insert("strategy.actions.bin".to_string(), actions_entry);
    files.insert("strategy.probs.f32.bin".to_string(), probs_entry);

    let mut full_manifest = manifest.clone();
    full_manifest.files = files.clone();
    let manifest_json = serde_json::to_string_pretty(&full_manifest)?;
    std::fs::write(dir.join("blueprint.json"), &manifest_json)?;

    write_checksums(dir, &files)?;
    Ok(())
}

/// Legacy wrapper kept for backward compatibility during migration.
pub struct BundleWriter;

impl BundleWriter {
    /// Write a complete bundle directory.
    ///
    /// # Errors
    ///
    /// Returns `FormatError` on I/O failure.
    pub fn write(
        dir: &Path,
        manifest: &Manifest,
        rows: &[RowDescriptor],
        actions: &[ActionDescriptor],
        probs: &[f32],
    ) -> Result<(), FormatError> {
        write_bundle(dir, manifest, &BundleData { rows, actions, probs })
    }
}

/// Boxed write closure for payload serialization.
pub(super) type PayloadWriteFn<'a> =
    Box<dyn Fn(&mut Vec<u8>) -> Result<(), FormatError> + 'a>;

/// Payload spec for a single binary file to write.
pub(super) struct PayloadSpec<'a> {
    pub name: &'a str,
    pub magic: [u8; 8],
    pub record_count: usize,
    pub write_fn: PayloadWriteFn<'a>,
}

/// Write a single binary payload file and return its `FileEntry`.
#[allow(clippy::cast_possible_truncation)]
pub(super) fn write_payload_file(
    dir: &Path,
    spec: &PayloadSpec<'_>,
) -> Result<FileEntry, FormatError> {
    let mut payload_buf = Vec::new();
    (spec.write_fn)(&mut payload_buf)?;

    let crc = crc::Crc::<u64>::new(&CRC_ALG);
    let payload_crc64 = crc.checksum(&payload_buf);
    let payload_len = payload_buf.len() as u64;

    let header = BinaryHeader::new(
        spec.magic,
        spec.record_count as u64,
        payload_len,
        payload_crc64,
    );

    let path = dir.join(spec.name);
    let file = std::fs::File::create(&path)?;
    let mut writer = BufWriter::new(file);
    header.write_to(&mut writer)?;
    writer.write_all(&payload_buf)?;
    writer.flush()?;

    let file_bytes = std::fs::read(&path)?;
    let sha256 = hex::encode(Sha256::digest(&file_bytes));

    Ok(FileEntry {
        size: file_bytes.len() as u64,
        sha256,
    })
}

/// Write the rows payload file.
fn write_rows_file(dir: &Path, rows: &[RowDescriptor]) -> Result<FileEntry, FormatError> {
    write_payload_file(dir, &PayloadSpec {
        name: "strategy.rows.bin",
        magic: MAGIC_ROWS,
        record_count: rows.len(),
        write_fn: Box::new(|w| {
            for row in rows {
                row.write_to(w)?;
            }
            Ok(())
        }),
    })
}

/// Write the actions payload file.
fn write_actions_file(
    dir: &Path,
    actions: &[ActionDescriptor],
) -> Result<FileEntry, FormatError> {
    write_payload_file(dir, &PayloadSpec {
        name: "strategy.actions.bin",
        magic: MAGIC_ACTIONS,
        record_count: actions.len(),
        write_fn: Box::new(|w| {
            for action in actions {
                action.write_to(w)?;
            }
            Ok(())
        }),
    })
}

/// Write the probs payload file.
fn write_probs_file(dir: &Path, probs: &[f32]) -> Result<FileEntry, FormatError> {
    write_payload_file(dir, &PayloadSpec {
        name: "strategy.probs.f32.bin",
        magic: MAGIC_PROBS,
        record_count: probs.len(),
        write_fn: Box::new(|w| {
            for &p in probs {
                w.write_all(&p.to_le_bytes())?;
            }
            Ok(())
        }),
    })
}

fn write_checksums(
    dir: &Path,
    files: &BTreeMap<String, FileEntry>,
) -> Result<(), FormatError> {
    let checksums: BTreeMap<&str, &str> = files
        .iter()
        .map(|(k, v)| (k.as_str(), v.sha256.as_str()))
        .collect();
    let json = serde_json::to_string_pretty(&checksums)?;
    std::fs::write(dir.join("checksums.json"), json)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Read-only view into a validated universal dense blueprint bundle.
#[derive(Debug)]
pub struct BundleReader {
    rows: Vec<RowDescriptor>,
    actions: Vec<ActionDescriptor>,
    probs: Vec<f32>,
    semantic_records: Vec<SemanticKeyRecord>,
}

impl BundleReader {
    /// Open and fully validate a bundle directory.
    ///
    /// Validation order:
    /// 1. Parse and validate manifest (format, version, required features).
    /// 2. Check file existence and lengths against manifest.
    /// 3. Validate binary headers (magic, version) -- fires before SHA.
    /// 4. Validate SHA-256 checksums of entire files.
    /// 5. Validate CRC-64 of payloads and decode records.
    /// 6. Validate row sort order, uniqueness, offsets, and normalization.
    ///
    /// # Errors
    ///
    /// Returns a precise `FormatError` variant for each validation failure.
    pub fn open(dir: &Path) -> Result<Self, FormatError> {
        let manifest = read_manifest(dir)?;
        validate_manifest_meta(&manifest)?;
        validate_required_features(&manifest)?;

        let rows_bytes = read_file_checked(dir, "strategy.rows.bin", &manifest)?;
        let actions_bytes =
            read_file_checked(dir, "strategy.actions.bin", &manifest)?;
        let probs_bytes =
            read_file_checked(dir, "strategy.probs.f32.bin", &manifest)?;

        validate_binary_headers(&rows_bytes, &actions_bytes, &probs_bytes)?;
        validate_sha_checksums(
            &rows_bytes, &actions_bytes, &probs_bytes, &manifest,
        )?;

        let has_semantic = has_semantic_feature(&manifest);
        let rows = decode_rows(&rows_bytes)?;
        let actions = decode_actions_checked(&actions_bytes, has_semantic)?;
        let probs = decode_probs(&probs_bytes)?;
        let semantic_records = load_semantic_table(dir, &manifest)?;

        validate_row_order(&rows)?;
        validate_semantic_keys(&rows, &semantic_records, has_semantic)?;
        validate_offsets(&rows, actions.len(), probs.len())?;
        validate_normalization(
            &rows, &probs, manifest.strategy.normalization_tolerance,
        )?;

        Ok(Self { rows, actions, probs, semantic_records })
    }

    /// Number of strategy rows in the bundle.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Get a row descriptor by index.
    #[must_use]
    pub fn row(&self, index: usize) -> Option<&RowDescriptor> {
        self.rows.get(index)
    }

    /// Get an action descriptor by index.
    #[must_use]
    pub fn action(&self, index: usize) -> Option<&ActionDescriptor> {
        self.actions.get(index)
    }

    /// Get a probability value by index.
    #[must_use]
    pub fn prob(&self, index: usize) -> Option<f32> {
        self.probs.get(index).copied()
    }

    /// Get the semantic key record for a row by row index.
    ///
    /// Returns `None` if the row has no semantic key (kind == 0) or
    /// the row index is out of bounds.
    #[must_use]
    pub fn semantic_record(
        &self,
        row_index: usize,
    ) -> Option<&SemanticKeyRecord> {
        let row = self.rows.get(row_index)?;
        if row.semantic_key_kind == SEMANTIC_KEY_NONE {
            return None;
        }
        self.semantic_records
            .get(row.semantic_key_offset as usize)
    }
}

/// Check whether the manifest declares `mp_semantic_rows_v1`.
fn has_semantic_feature(manifest: &Manifest) -> bool {
    manifest
        .required_features
        .iter()
        .any(|f| f == "mp_semantic_rows_v1")
}

/// Validate binary headers (magic + version) before SHA so corruption
/// of magic bytes surfaces as `BadMagic`, not `ChecksumMismatch`.
fn validate_binary_headers(
    rows_bytes: &[u8],
    actions_bytes: &[u8],
    probs_bytes: &[u8],
) -> Result<(), FormatError> {
    check_header_only(rows_bytes, MAGIC_ROWS, "strategy.rows.bin")?;
    check_header_only(actions_bytes, MAGIC_ACTIONS, "strategy.actions.bin")?;
    check_header_only(probs_bytes, MAGIC_PROBS, "strategy.probs.f32.bin")
}

/// Validate SHA-256 checksums of all three standard payload files.
fn validate_sha_checksums(
    rows_bytes: &[u8],
    actions_bytes: &[u8],
    probs_bytes: &[u8],
    manifest: &Manifest,
) -> Result<(), FormatError> {
    check_sha256(rows_bytes, "strategy.rows.bin", manifest)?;
    check_sha256(actions_bytes, "strategy.actions.bin", manifest)?;
    check_sha256(probs_bytes, "strategy.probs.f32.bin", manifest)
}

/// Load the semantic side table if present in the manifest.
fn load_semantic_table(
    dir: &Path,
    manifest: &Manifest,
) -> Result<Vec<SemanticKeyRecord>, FormatError> {
    if !manifest.files.contains_key("strategy.semantic.bin") {
        return Ok(Vec::new());
    }
    let sem_bytes =
        read_file_checked(dir, "strategy.semantic.bin", manifest)?;
    check_header_only(&sem_bytes, MAGIC_SEMANTIC, "strategy.semantic.bin")?;
    check_sha256(&sem_bytes, "strategy.semantic.bin", manifest)?;
    decode_semantic(&sem_bytes)
}

// ---------------------------------------------------------------------------
// Reader helpers
// ---------------------------------------------------------------------------

fn read_manifest(dir: &Path) -> Result<Manifest, FormatError> {
    let path = dir.join("blueprint.json");
    if !path.exists() {
        return Err(FormatError::MissingFile {
            file: "blueprint.json".to_string(),
        });
    }
    let text = std::fs::read_to_string(&path)?;
    let manifest: Manifest = serde_json::from_str(&text)?;
    Ok(manifest)
}

fn validate_manifest_meta(manifest: &Manifest) -> Result<(), FormatError> {
    if manifest.format_name != EXPECTED_FORMAT_NAME {
        return Err(FormatError::InvalidFormatName {
            expected: EXPECTED_FORMAT_NAME.to_string(),
            actual: manifest.format_name.clone(),
        });
    }
    if manifest.compat_min_reader > CURRENT_FORMAT_VERSION {
        return Err(FormatError::UnsupportedFormatVersion {
            file: "blueprint.json".to_string(),
            version: manifest.compat_min_reader,
            max_supported: CURRENT_FORMAT_VERSION,
        });
    }
    Ok(())
}

fn validate_required_features(manifest: &Manifest) -> Result<(), FormatError> {
    for feature in &manifest.required_features {
        if !SUPPORTED_FEATURES.contains(&feature.as_str()) {
            return Err(FormatError::UnsupportedRequiredFeature {
                feature: feature.clone(),
            });
        }
    }
    Ok(())
}

/// Read a file and validate its length against the manifest.
///
/// A missing `manifest.files` entry for a required file is a hard error.
#[allow(clippy::cast_possible_truncation)]
fn read_file_checked(
    dir: &Path,
    name: &str,
    manifest: &Manifest,
) -> Result<Vec<u8>, FormatError> {
    let path = dir.join(name);
    if !path.exists() {
        return Err(FormatError::MissingFile {
            file: name.to_string(),
        });
    }

    let entry = manifest.files.get(name).ok_or_else(|| {
        FormatError::MissingFileEntry {
            file: name.to_string(),
        }
    })?;

    let data = std::fs::read(&path)?;
    let actual_len = data.len() as u64;
    if actual_len != entry.size {
        return Err(FormatError::LengthMismatch {
            file: name.to_string(),
            expected: entry.size,
            actual: actual_len,
        });
    }

    Ok(data)
}

/// Validate only the header (magic + format version) without full decode.
fn check_header_only(
    data: &[u8],
    expected_magic: [u8; 8],
    file_name: &str,
) -> Result<(), FormatError> {
    if data.len() < HEADER_SIZE {
        return Err(FormatError::Truncated {
            file: file_name.to_string(),
            expected: HEADER_SIZE,
            actual: data.len(),
        });
    }
    let mut cursor = Cursor::new(&data[..HEADER_SIZE]);
    BinaryHeader::read_from(&mut cursor, expected_magic, file_name)?;
    Ok(())
}

/// Validate SHA-256 of file bytes against the manifest entry.
///
/// The caller must have already validated that the file entry exists
/// via `read_file_checked`, so the lookup is defensive.
fn check_sha256(
    data: &[u8],
    name: &str,
    manifest: &Manifest,
) -> Result<(), FormatError> {
    let entry = manifest.files.get(name).ok_or_else(|| {
        FormatError::MissingFileEntry {
            file: name.to_string(),
        }
    })?;
    let actual_sha = hex::encode(Sha256::digest(data));
    if actual_sha != entry.sha256 {
        return Err(FormatError::ChecksumMismatch {
            file: name.to_string(),
            expected: entry.sha256.clone(),
            actual: actual_sha,
        });
    }
    Ok(())
}

/// Decode the rows payload: validate header, CRC, then decode row descriptors.
fn decode_rows(data: &[u8]) -> Result<Vec<RowDescriptor>, FormatError> {
    let name = "strategy.rows.bin";
    let (header, payload) = split_header_payload(data, MAGIC_ROWS, name)?;
    validate_crc(&header, payload, name)?;
    check_payload_len(payload, header.record_count, ROW_DESCRIPTOR_SIZE, name)?;

    let count = record_count_usize(header.record_count);
    let mut rows = Vec::with_capacity(count);
    for i in 0..count {
        let start = i * ROW_DESCRIPTOR_SIZE;
        let chunk: &[u8; ROW_DESCRIPTOR_SIZE] =
            payload[start..start + ROW_DESCRIPTOR_SIZE].try_into().unwrap();
        rows.push(RowDescriptor::from_bytes(chunk));
    }
    Ok(rows)
}

/// Decode the actions payload, rejecting Opaque actions unless
/// `has_semantic` is set.
fn decode_actions_checked(
    data: &[u8],
    has_semantic: bool,
) -> Result<Vec<ActionDescriptor>, FormatError> {
    let name = "strategy.actions.bin";
    let (header, payload) = split_header_payload(data, MAGIC_ACTIONS, name)?;
    validate_crc(&header, payload, name)?;
    check_payload_len(
        payload,
        header.record_count,
        ACTION_DESCRIPTOR_SIZE,
        name,
    )?;

    let count = record_count_usize(header.record_count);
    let mut actions = Vec::with_capacity(count);
    for i in 0..count {
        let start = i * ACTION_DESCRIPTOR_SIZE;
        let chunk: &[u8; ACTION_DESCRIPTOR_SIZE] = payload
            [start..start + ACTION_DESCRIPTOR_SIZE]
            .try_into()
            .unwrap();
        let action = ActionDescriptor::from_bytes(chunk)?;
        if action.kind == super::descriptors::ActionKind::Opaque
            && !has_semantic
        {
            return Err(FormatError::InvalidManifest {
                detail: format!(
                    "action {i} has Opaque kind but \
                     mp_semantic_rows_v1 feature not declared"
                ),
            });
        }
        actions.push(action);
    }
    Ok(actions)
}

/// Decode the probabilities payload.
fn decode_probs(data: &[u8]) -> Result<Vec<f32>, FormatError> {
    let name = "strategy.probs.f32.bin";
    let (header, payload) = split_header_payload(data, MAGIC_PROBS, name)?;
    validate_crc(&header, payload, name)?;
    check_payload_len(payload, header.record_count, 4, name)?;

    let count = record_count_usize(header.record_count);
    let mut probs = Vec::with_capacity(count);
    for chunk in payload.chunks_exact(4) {
        probs.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(probs)
}

/// Split raw file bytes into header + payload, validating the header.
fn split_header_payload<'a>(
    data: &'a [u8],
    expected_magic: [u8; 8],
    file_name: &str,
) -> Result<(BinaryHeader, &'a [u8]), FormatError> {
    if data.len() < HEADER_SIZE {
        return Err(FormatError::Truncated {
            file: file_name.to_string(),
            expected: HEADER_SIZE,
            actual: data.len(),
        });
    }

    let mut cursor = Cursor::new(&data[..HEADER_SIZE]);
    let header =
        BinaryHeader::read_from(&mut cursor, expected_magic, file_name)?;

    let payload_end = usize::try_from(header.payload_len)
        .ok()
        .and_then(|pl| HEADER_SIZE.checked_add(pl));
    let payload_end = match payload_end {
        Some(end) if data.len() >= end => end,
        Some(end) => {
            return Err(FormatError::Truncated {
                file: file_name.to_string(),
                expected: end,
                actual: data.len(),
            });
        }
        None => {
            return Err(FormatError::Truncated {
                file: file_name.to_string(),
                expected: usize::MAX,
                actual: data.len(),
            });
        }
    };

    Ok((header, &data[HEADER_SIZE..payload_end]))
}

/// Validate the CRC-64/XZ of the payload against the header.
fn validate_crc(
    header: &BinaryHeader,
    payload: &[u8],
    file_name: &str,
) -> Result<(), FormatError> {
    let crc = crc::Crc::<u64>::new(&CRC_ALG);
    let actual = crc.checksum(payload);
    if actual != header.payload_crc64 {
        return Err(FormatError::CrcMismatch {
            file: file_name.to_string(),
            expected: header.payload_crc64,
            actual,
        });
    }
    Ok(())
}

/// Validate payload byte length matches `record_count` * `record_size`.
fn check_payload_len(
    payload: &[u8],
    record_count: u64,
    record_size: usize,
    file_name: &str,
) -> Result<(), FormatError> {
    let expected = usize::try_from(record_count)
        .ok()
        .and_then(|rc| rc.checked_mul(record_size));
    match expected {
        Some(exp) if payload.len() >= exp => Ok(()),
        Some(exp) => Err(FormatError::Truncated {
            file: file_name.to_string(),
            expected: HEADER_SIZE + exp,
            actual: HEADER_SIZE + payload.len(),
        }),
        None => Err(FormatError::Truncated {
            file: file_name.to_string(),
            expected: usize::MAX,
            actual: HEADER_SIZE + payload.len(),
        }),
    }
}

/// Convert `u64` record count to `usize`, truncating on 32-bit targets.
#[allow(clippy::cast_possible_truncation)]
fn record_count_usize(count: u64) -> usize {
    count as usize
}

/// Validate that rows are sorted by identity key and contain no duplicates.
fn validate_row_order(rows: &[RowDescriptor]) -> Result<(), FormatError> {
    for i in 1..rows.len() {
        let prev = rows[i - 1].identity_key();
        let curr = rows[i].identity_key();
        match prev.cmp(&curr) {
            std::cmp::Ordering::Greater => {
                return Err(FormatError::RowsNotSorted { index: i });
            }
            std::cmp::Ordering::Equal => {
                return Err(FormatError::DuplicateRowIdentity { index: i });
            }
            std::cmp::Ordering::Less => {}
        }
    }
    Ok(())
}

/// Validate that all row offsets point within bounds.
///
/// Inlines range checking (finding 13: `check_offset_range` had one caller).
fn validate_offsets(
    rows: &[RowDescriptor],
    total_actions: usize,
    total_probs: usize,
) -> Result<(), FormatError> {
    for (i, row) in rows.iter().enumerate() {
        let count = u64::from(row.action_count);
        for &(label, offset, total) in &[
            ("action", row.action_offset, total_actions),
            ("prob", row.prob_offset, total_probs),
        ] {
            let end = offset.checked_add(count).ok_or_else(|| {
                FormatError::InvalidOffset {
                    row_index: i,
                    detail: format!("{label} offset overflow"),
                }
            })?;
            if end > total as u64 {
                return Err(FormatError::InvalidOffset {
                    row_index: i,
                    detail: format!(
                        "{label} range {offset}..{end} exceeds total {total}"
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Decode the semantic side table payload.
fn decode_semantic(
    data: &[u8],
) -> Result<Vec<SemanticKeyRecord>, FormatError> {
    let name = "strategy.semantic.bin";
    let (header, payload) =
        split_header_payload(data, MAGIC_SEMANTIC, name)?;
    validate_crc(&header, payload, name)?;
    check_payload_len(
        payload,
        header.record_count,
        SEMANTIC_RECORD_SIZE,
        name,
    )?;

    let count = record_count_usize(header.record_count);
    let mut records = Vec::with_capacity(count);
    for i in 0..count {
        let start = i * SEMANTIC_RECORD_SIZE;
        let chunk: &[u8; SEMANTIC_RECORD_SIZE] = payload
            [start..start + SEMANTIC_RECORD_SIZE]
            .try_into()
            .unwrap();
        records.push(SemanticKeyRecord::from_bytes(chunk));
    }
    Ok(records)
}

/// Validate semantic key kinds and offsets for all rows.
///
/// Rules:
/// - `semantic_key_kind == 0` (none) is always valid.
/// - `semantic_key_kind == 1` (mp_history_v1) requires the
///   `mp_semantic_rows_v1` feature to be declared.
/// - Any other kind is rejected.
/// - `semantic_key_offset` must be in range when kind != 0.
fn validate_semantic_keys(
    rows: &[RowDescriptor],
    semantic_records: &[SemanticKeyRecord],
    has_semantic_feature: bool,
) -> Result<(), FormatError> {
    for (i, row) in rows.iter().enumerate() {
        match row.semantic_key_kind {
            SEMANTIC_KEY_NONE => {}
            SEMANTIC_KEY_MP_HISTORY_V1 => {
                if !has_semantic_feature {
                    return Err(FormatError::InvalidManifest {
                        detail: format!(
                            "row {i} has semantic_key_kind {} but \
                             mp_semantic_rows_v1 feature not declared",
                            SEMANTIC_KEY_MP_HISTORY_V1
                        ),
                    });
                }
                let offset = row.semantic_key_offset as usize;
                if offset >= semantic_records.len() {
                    return Err(FormatError::InvalidOffset {
                        row_index: i,
                        detail: format!(
                            "semantic_key_offset {offset} out of range \
                             (table has {} records)",
                            semantic_records.len()
                        ),
                    });
                }
            }
            unknown => {
                return Err(FormatError::InvalidManifest {
                    detail: format!(
                        "row {i} has unknown semantic_key_kind {unknown}"
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Validate that each row's probabilities are finite, non-negative, and
/// normalized to sum 1 within the given tolerance.
#[allow(clippy::cast_possible_truncation)]
fn validate_normalization(
    rows: &[RowDescriptor],
    probs: &[f32],
    tolerance: f64,
) -> Result<(), FormatError> {
    for (i, row) in rows.iter().enumerate() {
        let start = row.prob_offset as usize;
        let end = start + usize::from(row.action_count);
        validate_row_probs(i, &probs[start..end], tolerance)?;
    }
    Ok(())
}

/// Validate a single row's probability slice.
fn validate_row_probs(
    row_index: usize,
    probs: &[f32],
    tolerance: f64,
) -> Result<(), FormatError> {
    if probs.is_empty() {
        return Ok(());
    }

    let mut sum: f64 = 0.0;
    for (j, &p) in probs.iter().enumerate() {
        if !p.is_finite() {
            return Err(FormatError::NonNormalizedRow {
                row_index,
                detail: format!("prob[{j}] is not finite: {p}"),
            });
        }
        if p < 0.0 {
            return Err(FormatError::NonNormalizedRow {
                row_index,
                detail: format!("prob[{j}] is negative: {p}"),
            });
        }
        sum += f64::from(p);
    }

    if (sum - 1.0).abs() > tolerance {
        return Err(FormatError::NonNormalizedRow {
            row_index,
            detail: format!(
                "sum = {sum}, expected 1.0 (tolerance {tolerance})"
            ),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::header::MAGIC_ROWS;

    /// Finding 1: payload_len near u64::MAX causes overflow in
    /// HEADER_SIZE + payload_len as usize, wrapping on 64-bit targets.
    #[test]
    fn split_header_payload_rejects_overflow_payload_len() {
        let mut buf = [0u8; HEADER_SIZE];
        let header = BinaryHeader::new(MAGIC_ROWS, 0, u64::MAX - 10, 0);
        header.write_to(&mut buf.as_mut_slice()).unwrap();

        let err =
            split_header_payload(&buf, MAGIC_ROWS, "test.bin").unwrap_err();
        assert!(
            matches!(err, FormatError::Truncated { .. }),
            "expected Truncated on overflow, got {err:?}"
        );
    }

    /// Finding 2: record_count near u64::MAX causes overflow in
    /// record_count as usize * record_size.
    #[test]
    fn check_payload_len_rejects_overflow_record_count() {
        let payload = &[0u8; 100];
        let err =
            check_payload_len(payload, u64::MAX, 96, "test.bin").unwrap_err();
        assert!(
            matches!(err, FormatError::Truncated { .. }),
            "expected Truncated on overflow, got {err:?}"
        );
    }
}
