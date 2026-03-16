//! BucketedTree: a flat game tree with bucket-level terminal data.
//!
//! Reuses the same BFS walk topology as `FlatTree::from_postflop_game()`,
//! but instead of per-hand card data, stores per-terminal bucket equity
//! tables and fold payoffs.

// Implementation in Task A2 below.
