//! Precomputed continuation-value (CBV) lookup table.
//!
//! Stores one `f32` per `(boundary_node, bucket)` pair in a flat array with
//! per-node offset indexing. Supports bincode serialization for fast
//! save/load during real-time subgame solving.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io;
use std::path::Path;

use super::game_tree::{GameNode, GameTree};

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

    /// Build a mapping from abstract tree arena index to dense CBV ordinal.
    ///
    /// The ordinal is the position of a `Chance` node in the tree's node
    /// list when filtering for `GameNode::Chance` variants and iterating
    /// in arena-index order. This is the same ordering used by
    /// `cbv_compute::compute_cbvs` when populating the table.
    ///
    /// Only `Chance` nodes appear in the returned map.
    #[must_use]
    pub fn build_node_to_ordinal_map(tree: &GameTree) -> HashMap<u32, usize> {
        tree.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| matches!(n, GameNode::Chance { .. }))
            .enumerate()
            .map(|(ordinal, (arena_idx, _))| (arena_idx as u32, ordinal))
            .collect()
    }

    /// Look up the dense CBV ordinal for an abstract tree arena index,
    /// panicking if the node is not a CBV boundary (chance node).
    ///
    /// # Panics
    ///
    /// Panics with a descriptive message if `arena_idx` is not a chance
    /// node in the map.
    #[must_use]
    pub fn require_ordinal(map: &HashMap<u32, usize>, arena_idx: u32) -> usize {
        *map.get(&arena_idx).unwrap_or_else(|| {
            panic!(
                "abstract tree node {arena_idx} is not a CBV boundary \
                 (chance node); cannot look up CBV ordinal"
            )
        })
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

    #[test]
    fn cbv_ordinal_of_node_maps_sparse_arena_to_dense() {
        use crate::blueprint_v2::game_tree::{GameNode, GameTree, TerminalKind, TreeAction};
        use crate::blueprint_v2::Street;

        // Tree with chance nodes at arena indices 2 and 6 (NOT 0 and 1).
        // Ordinals should be 0 and 1 respectively.
        let nodes = vec![
            // 0: Root decision
            GameNode::Decision {
                player: 0,
                street: Street::Flop,
                actions: vec![TreeAction::Check, TreeAction::Bet(2.0)],
                children: vec![1, 4],
                blueprint_decision_idx: None,
            },
            // 1: Player 1 decision
            GameNode::Decision {
                player: 1,
                street: Street::Flop,
                actions: vec![TreeAction::Check],
                children: vec![2],
                blueprint_decision_idx: None,
            },
            // 2: Chance node (ordinal 0)
            GameNode::Chance {
                next_street: Street::Turn,
                child: 3,
            },
            // 3: Terminal
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 2.0,
                stacks: [49.0, 49.0],
            },
            // 4: Player 1 decision
            GameNode::Decision {
                player: 1,
                street: Street::Flop,
                actions: vec![TreeAction::Fold, TreeAction::Call],
                children: vec![5, 6],
                blueprint_decision_idx: None,
            },
            // 5: Fold terminal
            GameNode::Terminal {
                kind: TerminalKind::Fold { winner: 0 },
                pot: 2.0,
                stacks: [49.0, 49.0],
            },
            // 6: Chance node (ordinal 1)
            GameNode::Chance {
                next_street: Street::Turn,
                child: 7,
            },
            // 7: Terminal
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 4.0,
                stacks: [48.0, 48.0],
            },
        ];
        let tree = GameTree {
            nodes,
            root: 0,
            dealer: 0,
            starting_stack: 50.0,
        };

        let map = CbvTable::build_node_to_ordinal_map(&tree);

        // Chance nodes at arena index 2 -> ordinal 0, arena index 6 -> ordinal 1
        assert_eq!(map.get(&2), Some(&0));
        assert_eq!(map.get(&6), Some(&1));

        // Non-chance nodes should not be in the map
        assert_eq!(map.get(&0), None);
        assert_eq!(map.get(&1), None);
        assert_eq!(map.get(&3), None);
        assert_eq!(map.get(&5), None);
        assert_eq!(map.get(&7), None);
    }

    #[test]
    #[should_panic(expected = "not a CBV boundary")]
    fn cbv_ordinal_of_node_panics_for_non_chance_node() {
        use crate::blueprint_v2::game_tree::{GameNode, GameTree, TerminalKind, TreeAction};
        use crate::blueprint_v2::Street;

        let nodes = vec![
            GameNode::Decision {
                player: 0,
                street: Street::Flop,
                actions: vec![TreeAction::Check],
                children: vec![1],
                blueprint_decision_idx: None,
            },
            GameNode::Chance {
                next_street: Street::Turn,
                child: 2,
            },
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 2.0,
                stacks: [49.0, 49.0],
            },
        ];
        let tree = GameTree {
            nodes,
            root: 0,
            dealer: 0,
            starting_stack: 50.0,
        };

        let map = CbvTable::build_node_to_ordinal_map(&tree);

        // Should panic: node 0 is a Decision, not a Chance node
        CbvTable::require_ordinal(&map, 0);
    }

    #[test]
    fn cbv_require_ordinal_returns_correct_value() {
        use crate::blueprint_v2::game_tree::{GameNode, GameTree, TerminalKind, TreeAction};
        use crate::blueprint_v2::Street;

        let nodes = vec![
            GameNode::Decision {
                player: 0,
                street: Street::Flop,
                actions: vec![TreeAction::Check],
                children: vec![1],
                blueprint_decision_idx: None,
            },
            GameNode::Chance {
                next_street: Street::Turn,
                child: 2,
            },
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 2.0,
                stacks: [49.0, 49.0],
            },
        ];
        let tree = GameTree {
            nodes,
            root: 0,
            dealer: 0,
            starting_stack: 50.0,
        };

        let map = CbvTable::build_node_to_ordinal_map(&tree);
        assert_eq!(CbvTable::require_ordinal(&map, 1), 0);
    }
}
