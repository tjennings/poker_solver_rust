//! Blueprint Strategy Module
//!
//! Provides types and functionality for blueprint strategies computed by CFR.
//! Blueprint strategies represent the precomputed equilibrium strategies that
//! can be used as a starting point for real-time subgame solving.
//!
//! # Submodules
//!
//! - `error` - Error types for blueprint operations

mod error;

pub use error::BlueprintError;
