//! Universal dense blueprint format.
//!
//! Provides read/write primitives for the universal dense blueprint bundle
//! format, which supports HU and N-player exported strategies in a single
//! directory layout.
//!
//! See `docs/blueprint_format.md` for the full specification.

mod error;
pub(crate) mod hash;
mod header;
mod manifest;
mod descriptors;
mod bundle;
pub(crate) mod export_common;
pub mod hu_export;
pub mod mp_eager_export;

pub use error::FormatError;
pub use header::{BinaryHeader, HEADER_SIZE};
pub use manifest::{
    Manifest, GameMetadata, TrainingMetadata, StrategyMetadata, LayoutMetadata,
    ActionsMetadata, BucketsMetadata, CompatibilityMetadata, FileEntry,
    SeatDescriptor, RakeConfig, BucketFileRef, PerFlopBucketConfig,
};
pub use descriptors::{ActionDescriptor, ActionKind, RowDescriptor, ROW_DESCRIPTOR_SIZE};
pub use bundle::{BundleData, BundleReader, BundleWriter, write_bundle};
