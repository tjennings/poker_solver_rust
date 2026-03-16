//! Bucketed GPU solver for Supremus-style abstracted postflop solving.
//!
//! Instead of per-hand-combo data, this module operates on bucket-level
//! abstractions where each of 1326 combos is mapped to one of ~500 buckets.
//! This dramatically reduces the dimensionality of the solver state.

pub mod equity;
pub mod mapping;
#[cfg(feature = "training")]
pub mod sampler;
pub mod solver;
pub mod tree;
