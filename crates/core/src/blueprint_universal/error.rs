//! Error types for the universal dense blueprint format.

use thiserror::Error;

/// Errors raised during bundle read/write and validation.
///
/// Each variant maps to a distinct bad-input class so that tests can assert
/// on the specific rejection reason.
#[derive(Debug, Error)]
pub enum FormatError {
    #[error("bad magic in {file}: expected {expected}, got {actual}")]
    BadMagic {
        file: String,
        expected: String,
        actual: String,
    },

    #[error("unsupported format version in {file}: got {version}, reader supports up to {max_supported}")]
    UnsupportedFormatVersion {
        file: String,
        version: u16,
        max_supported: u16,
    },

    #[error("unsupported required feature: {feature}")]
    UnsupportedRequiredFeature { feature: String },

    #[error("SHA-256 checksum mismatch for {file}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        file: String,
        expected: String,
        actual: String,
    },

    #[error("CRC-64 mismatch for {file}: expected {expected:#018x}, got {actual:#018x}")]
    CrcMismatch {
        file: String,
        expected: u64,
        actual: u64,
    },

    #[error("truncated {file}: expected at least {expected} bytes, got {actual}")]
    Truncated {
        file: String,
        expected: usize,
        actual: usize,
    },

    #[error("file length mismatch for {file}: manifest says {expected} bytes, actual is {actual}")]
    LengthMismatch {
        file: String,
        expected: u64,
        actual: u64,
    },

    #[error("duplicate row identity at index {index}")]
    DuplicateRowIdentity { index: usize },

    #[error("rows not sorted: row {index} is out of order")]
    RowsNotSorted { index: usize },

    #[error("invalid offset in row {row_index}: {detail}")]
    InvalidOffset { row_index: usize, detail: String },

    #[error("non-normalized probabilities for row {row_index}: {detail}")]
    NonNormalizedRow { row_index: usize, detail: String },

    #[error("invalid manifest: {detail}")]
    InvalidManifest { detail: String },

    #[error("missing file: {file}")]
    MissingFile { file: String },

    #[error("missing file entry in manifest for required file: {file}")]
    MissingFileEntry { file: String },

    #[error("invalid format_name: expected \"{expected}\", got \"{actual}\"")]
    InvalidFormatName { expected: String, actual: String },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
