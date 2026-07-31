//! Universal dense blueprint format.
//!
//! Provides read/write primitives for the universal dense blueprint bundle
//! format, which supports HU and N-player exported strategies in a single
//! directory layout.
//!
//! See `docs/blueprint_format.md` for the full specification.

mod bundle;
mod descriptors;
mod error;
pub mod export_common;
pub(crate) mod hash;
mod header;
pub mod hu_export;
pub mod loader;
mod manifest;
pub mod mp_eager_export;
pub mod mp_lazy_export;

pub use bundle::{BundleData, BundleReader, BundleWriter, write_bundle};
pub use descriptors::{
    ActionDescriptor, ActionKind, ROW_DESCRIPTOR_SIZE, RowDescriptor, SEMANTIC_RECORD_SIZE,
    SemanticKeyRecord,
};
pub use error::FormatError;
pub use header::{BinaryHeader, HEADER_SIZE};
pub use loader::{
    BundleKind, InfosetView, LoadedBundle, LoaderError, MpLazyInfosetView, MpLazyKey,
    detect_bundle_kind, load_bundle,
};
pub use manifest::{
    ActionsMetadata, BucketFileRef, BucketsMetadata, CompatibilityMetadata, FileEntry,
    GameMetadata, LayoutMetadata, Manifest, PerFlopBucketConfig, RakeConfig, SeatDescriptor,
    StrategyMetadata, TrainingMetadata,
};
