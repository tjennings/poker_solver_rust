//! Universal dense blueprint format.
//!
//! Provides read/write primitives for the universal dense blueprint bundle
//! format, which supports HU and N-player exported strategies in a single
//! directory layout.
//!
//! See `docs/blueprint_format.md` for the full specification.

mod error;
mod header;
mod manifest;
mod descriptors;
mod bundle;

pub use error::FormatError;
pub use header::{BinaryHeader, HEADER_SIZE};
pub use manifest::Manifest;
pub use descriptors::{ActionDescriptor, ActionKind, RowDescriptor, ROW_DESCRIPTOR_SIZE};
pub use bundle::{BundleReader, BundleWriter};
