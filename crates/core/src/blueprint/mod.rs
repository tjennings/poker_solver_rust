//! Blueprint Strategy Module
//!
//! Provides types and functionality for blueprint strategies computed by CFR.
//! Blueprint strategies represent the precomputed equilibrium strategies that
//! can be used as a starting point for real-time subgame solving.
//!
//! # Submodules
//!
//! - `error` - Error types for blueprint operations
//! - `strategy` - Strategy storage with save/load functionality
//! - `cache` - Two-tier caching for subgame solver results
//! - `bundle` - Directory-based strategy bundle format

mod bundle;
mod cache;
mod error;
mod strategy;
mod subgame;
mod subgame_cfr;
mod subgame_tree;

pub use bundle::{AbstractionModeConfig, BundleConfig, StrategyBundle};
pub use cache::{CacheConfig, SubgameCache, SubgameKey};
pub use error::BlueprintError;
pub use strategy::{BlueprintStrategy, ReachDecision, ReachQuery};
pub use subgame::{SubgameConfig, SubgameSolver};
pub use subgame_cfr::{SubgameCfrSolver, SubgameStrategy, solve_subgame};
pub use subgame_tree::{SubgameHands, SubgameNode, SubgameTree, SubgameTreeBuilder};
