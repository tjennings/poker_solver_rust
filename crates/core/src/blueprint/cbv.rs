//! Precomputed continuation-value (CBV) lookup table.
//!
//! Stores one `f32` per `(boundary_node, bucket)` pair in a flat array with
//! per-node offset indexing. Supports bincode serialization for fast
//! save/load during real-time subgame solving.

use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;

/// Flat lookup table mapping `(boundary_node, bucket)` pairs to precomputed
/// continuation values.
///
/// The `values` array is partitioned into contiguous slices, one per boundary
/// node. `node_offsets[i]` gives the starting index of node `i`'s slice, and
/// `buckets_per_node[i]` gives its length.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CbvTable {
    /// Flat storage of continuation values, indexed by
    /// `node_offsets[node] + bucket`.
    pub values: Vec<f32>,
    /// Starting index in `values` for each boundary node.
    pub node_offsets: Vec<usize>,
    /// Number of buckets for each boundary node.
    pub buckets_per_node: Vec<u16>,
}

impl CbvTable {
    /// Look up the CBV for a specific boundary node and bucket.
    ///
    /// # Panics
    ///
    /// Panics if `boundary_node >= self.num_boundary_nodes()` or
    /// `bucket >= self.buckets_per_node[boundary_node]`.
    #[inline]
    #[must_use]
    pub fn lookup(&self, boundary_node: usize, bucket: usize) -> f32 {
        debug_assert!(
            boundary_node < self.node_offsets.len(),
            "boundary_node {boundary_node} out of range (num_nodes = {})",
            self.node_offsets.len(),
        );
        debug_assert!(
            bucket < self.buckets_per_node[boundary_node] as usize,
            "bucket {bucket} out of range (buckets = {})",
            self.buckets_per_node[boundary_node],
        );
        self.values[self.node_offsets[boundary_node] + bucket]
    }

    /// Number of boundary nodes in this table.
    #[inline]
    #[must_use]
    pub fn num_boundary_nodes(&self) -> usize {
        self.node_offsets.len()
    }

    /// Save to file using bincode.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or serialization fails.
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let file = std::fs::File::create(path)?;
        let writer = io::BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    /// Load from file using bincode.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or deserialization fails.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let reader = io::BufReader::new(file);
        let table: Self = bincode::deserialize_from(reader)?;
        Ok(table)
    }

    /// Save to any writer (for testing without touching the filesystem).
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn save_to_writer<W: io::Write>(
        &self,
        writer: &mut W,
    ) -> Result<(), bincode::Error> {
        bincode::serialize_into(writer, self)
    }

    /// Load from any reader (for testing without touching the filesystem).
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails.
    pub fn load_from_reader<R: io::Read>(reader: R) -> Result<Self, bincode::Error> {
        bincode::deserialize_from(reader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uniform_table() -> CbvTable {
        // 2 boundary nodes, each with 3 buckets
        CbvTable {
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            node_offsets: vec![0, 3],
            buckets_per_node: vec![3, 3],
        }
    }

    #[test]
    fn test_cbv_lookup() {
        let table = make_uniform_table();

        assert_eq!(table.lookup(0, 0), 1.0);
        assert_eq!(table.lookup(0, 1), 2.0);
        assert_eq!(table.lookup(0, 2), 3.0);
        assert_eq!(table.lookup(1, 0), 4.0);
        assert_eq!(table.lookup(1, 1), 5.0);
        assert_eq!(table.lookup(1, 2), 6.0);
    }

    #[test]
    fn test_cbv_roundtrip_serialization() {
        let table = make_uniform_table();

        let mut buf = Vec::new();
        table.save_to_writer(&mut buf).expect("serialize");

        let loaded = CbvTable::load_from_reader(buf.as_slice()).expect("deserialize");
        assert_eq!(table, loaded);
    }

    #[test]
    fn test_cbv_num_boundary_nodes() {
        let table = make_uniform_table();
        assert_eq!(table.num_boundary_nodes(), 2);

        let empty = CbvTable {
            values: vec![],
            node_offsets: vec![],
            buckets_per_node: vec![],
        };
        assert_eq!(empty.num_boundary_nodes(), 0);
    }

    #[test]
    fn test_cbv_different_bucket_counts() {
        // Node 0 has 3 buckets, node 1 has 2 buckets
        let table = CbvTable {
            values: vec![10.0, 20.0, 30.0, 40.0, 50.0],
            node_offsets: vec![0, 3],
            buckets_per_node: vec![3, 2],
        };

        assert_eq!(table.lookup(0, 0), 10.0);
        assert_eq!(table.lookup(0, 1), 20.0);
        assert_eq!(table.lookup(0, 2), 30.0);
        assert_eq!(table.lookup(1, 0), 40.0);
        assert_eq!(table.lookup(1, 1), 50.0);
    }
}
