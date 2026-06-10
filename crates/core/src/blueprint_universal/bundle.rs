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
    ActionDescriptor, RowDescriptor, ACTION_DESCRIPTOR_SIZE, ROW_DESCRIPTOR_SIZE,
};
use super::error::FormatError;
use super::header::{
    BinaryHeader, CURRENT_FORMAT_VERSION, HEADER_SIZE, MAGIC_ACTIONS, MAGIC_PROBS,
    MAGIC_ROWS,
};
use super::manifest::{FileEntry, Manifest};

/// CRC-64/XZ algorithm constant from the `crc` crate.
///
/// We use CRC-64/XZ (ECMA-182, also known as CRC-64/GO-ECMA) because it is
/// a widely implemented 64-bit CRC with good error-detection properties and
/// is available as a built-in constant in the `crc` crate v3.
const CRC_ALG: crc::Algorithm<u64> = crc::CRC_64_XZ;

/// Supported features that this reader understands.
const SUPPORTED_FEATURES: &[&str] = &[];

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Writes a complete universal dense blueprint bundle to a directory.
pub struct BundleWriter;

impl BundleWriter {
    /// Write a complete bundle directory.
    ///
    /// # Errors
    ///
    /// Returns `FormatError` on I/O failure.
    #[allow(clippy::cast_possible_truncation)]
    pub fn write(
        dir: &Path,
        manifest: &Manifest,
        rows: &[RowDescriptor],
        actions: &[ActionDescriptor],
        probs: &[f32],
    ) -> Result<(), FormatError> {
        std::fs::create_dir_all(dir)?;
        let mut files = BTreeMap::new();

        write_rows_file(dir, rows, &mut files)?;
        write_actions_file(dir, actions, &mut files)?;
        write_probs_file(dir, probs, &mut files)?;

        let mut full_manifest = manifest.clone();
        full_manifest.files.clone_from(&files);
        let manifest_json = serde_json::to_string_pretty(&full_manifest)?;
        std::fs::write(dir.join("blueprint.json"), &manifest_json)?;

        write_checksums(dir, &files)?;
        Ok(())
    }
}

/// Write the rows payload file.
fn write_rows_file(
    dir: &Path,
    rows: &[RowDescriptor],
    files: &mut BTreeMap<String, FileEntry>,
) -> Result<(), FormatError> {
    write_payload_file(dir, "strategy.rows.bin", MAGIC_ROWS, rows.len(), &|w| {
        for row in rows {
            row.write_to(w)?;
        }
        Ok(())
    }, files)
}

/// Write the actions payload file.
fn write_actions_file(
    dir: &Path,
    actions: &[ActionDescriptor],
    files: &mut BTreeMap<String, FileEntry>,
) -> Result<(), FormatError> {
    write_payload_file(dir, "strategy.actions.bin", MAGIC_ACTIONS, actions.len(), &|w| {
        for action in actions {
            action.write_to(w)?;
        }
        Ok(())
    }, files)
}

/// Write the probs payload file.
fn write_probs_file(
    dir: &Path,
    probs: &[f32],
    files: &mut BTreeMap<String, FileEntry>,
) -> Result<(), FormatError> {
    write_payload_file(dir, "strategy.probs.f32.bin", MAGIC_PROBS, probs.len(), &|w| {
        for &p in probs {
            w.write_all(&p.to_le_bytes())?;
        }
        Ok(())
    }, files)
}

/// Write a single binary payload file: header + payload, computing CRC and SHA.
#[allow(clippy::cast_possible_truncation)]
fn write_payload_file(
    dir: &Path,
    name: &str,
    magic: [u8; 8],
    record_count: usize,
    write_payload: &dyn Fn(&mut Vec<u8>) -> Result<(), FormatError>,
    files: &mut BTreeMap<String, FileEntry>,
) -> Result<(), FormatError> {
    let mut payload_buf = Vec::new();
    write_payload(&mut payload_buf)?;

    let crc = crc::Crc::<u64>::new(&CRC_ALG);
    let payload_crc64 = crc.checksum(&payload_buf);
    let payload_len = payload_buf.len() as u64;

    let header = BinaryHeader::new(magic, record_count as u64, payload_len, payload_crc64);

    let path = dir.join(name);
    let file = std::fs::File::create(&path)?;
    let mut writer = BufWriter::new(file);
    header.write_to(&mut writer)?;
    writer.write_all(&payload_buf)?;
    writer.flush()?;

    let file_bytes = std::fs::read(&path)?;
    let sha256 = hex::encode(Sha256::digest(&file_bytes));

    files.insert(name.to_string(), FileEntry {
        size: file_bytes.len() as u64,
        sha256,
    });

    Ok(())
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
        let actions_bytes = read_file_checked(dir, "strategy.actions.bin", &manifest)?;
        let probs_bytes = read_file_checked(dir, "strategy.probs.f32.bin", &manifest)?;

        // Validate headers (magic + version) before SHA so corruption of magic
        // bytes surfaces as BadMagic, not ChecksumMismatch.
        check_header_only(&rows_bytes, MAGIC_ROWS, "strategy.rows.bin")?;
        check_header_only(&actions_bytes, MAGIC_ACTIONS, "strategy.actions.bin")?;
        check_header_only(&probs_bytes, MAGIC_PROBS, "strategy.probs.f32.bin")?;

        check_sha256(&rows_bytes, "strategy.rows.bin", &manifest)?;
        check_sha256(&actions_bytes, "strategy.actions.bin", &manifest)?;
        check_sha256(&probs_bytes, "strategy.probs.f32.bin", &manifest)?;

        let rows = decode_rows(&rows_bytes)?;
        let actions = decode_actions(&actions_bytes)?;
        let probs = decode_probs(&probs_bytes)?;

        validate_row_order(&rows)?;
        validate_offsets(&rows, actions.len(), probs.len())?;
        validate_normalization(&rows, &probs, manifest.strategy.normalization_tolerance)?;

        Ok(Self { rows, actions, probs })
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

    let data = std::fs::read(&path)?;

    if let Some(entry) = manifest.files.get(name) {
        let actual_len = data.len() as u64;
        if actual_len != entry.size {
            return Err(FormatError::LengthMismatch {
                file: name.to_string(),
                expected: entry.size,
                actual: actual_len,
            });
        }
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
fn check_sha256(
    data: &[u8],
    name: &str,
    manifest: &Manifest,
) -> Result<(), FormatError> {
    if let Some(entry) = manifest.files.get(name) {
        let actual_sha = hex::encode(Sha256::digest(data));
        if actual_sha != entry.sha256 {
            return Err(FormatError::ChecksumMismatch {
                file: name.to_string(),
                expected: entry.sha256.clone(),
                actual: actual_sha,
            });
        }
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

/// Decode the actions payload.
fn decode_actions(data: &[u8]) -> Result<Vec<ActionDescriptor>, FormatError> {
    let name = "strategy.actions.bin";
    let (header, payload) = split_header_payload(data, MAGIC_ACTIONS, name)?;
    validate_crc(&header, payload, name)?;
    check_payload_len(payload, header.record_count, ACTION_DESCRIPTOR_SIZE, name)?;

    let count = record_count_usize(header.record_count);
    let mut actions = Vec::with_capacity(count);
    for i in 0..count {
        let start = i * ACTION_DESCRIPTOR_SIZE;
        let chunk: &[u8; ACTION_DESCRIPTOR_SIZE] =
            payload[start..start + ACTION_DESCRIPTOR_SIZE].try_into().unwrap();
        actions.push(ActionDescriptor::from_bytes(chunk)?);
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
#[allow(clippy::cast_possible_truncation)]
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
    let header = BinaryHeader::read_from(&mut cursor, expected_magic, file_name)?;

    let payload_end = HEADER_SIZE + header.payload_len as usize;
    if data.len() < payload_end {
        return Err(FormatError::Truncated {
            file: file_name.to_string(),
            expected: payload_end,
            actual: data.len(),
        });
    }

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
#[allow(clippy::cast_possible_truncation)]
fn check_payload_len(
    payload: &[u8],
    record_count: u64,
    record_size: usize,
    file_name: &str,
) -> Result<(), FormatError> {
    let expected = record_count as usize * record_size;
    if payload.len() < expected {
        return Err(FormatError::Truncated {
            file: file_name.to_string(),
            expected: HEADER_SIZE + expected,
            actual: HEADER_SIZE + payload.len(),
        });
    }
    Ok(())
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
fn validate_offsets(
    rows: &[RowDescriptor],
    total_actions: usize,
    total_probs: usize,
) -> Result<(), FormatError> {
    for (i, row) in rows.iter().enumerate() {
        check_offset_range(
            i, "action", row.action_offset, row.action_count, total_actions,
        )?;
        check_offset_range(
            i, "prob", row.prob_offset, row.action_count, total_probs,
        )?;
    }
    Ok(())
}

/// Check a single offset+count range against a total bound.
fn check_offset_range(
    row_index: usize,
    label: &str,
    offset: u64,
    count: u16,
    total: usize,
) -> Result<(), FormatError> {
    let end = offset
        .checked_add(u64::from(count))
        .ok_or_else(|| FormatError::InvalidOffset {
            row_index,
            detail: format!("{label} offset overflow"),
        })?;
    if end > total as u64 {
        return Err(FormatError::InvalidOffset {
            row_index,
            detail: format!(
                "{label} range {offset}..{end} exceeds total {total}"
            ),
        });
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
            detail: format!("sum = {sum}, expected 1.0 (tolerance {tolerance})"),
        });
    }

    Ok(())
}
