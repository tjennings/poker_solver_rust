//! GPU-resident training pipelines for river, turn, flop, and preflop CFVNet.
//!
//! Modules for sampling situations, evaluating hand strengths, building
//! batch solvers, storing training examples in a reservoir buffer,
//! training the CFVNet model, and running the full pipeline.

pub mod builder;
pub mod hand_eval;
pub mod reservoir;
pub mod sampler;

#[cfg(feature = "training")]
pub mod trainer;
#[cfg(feature = "training")]
pub mod validation;
#[cfg(feature = "training")]
pub mod pipeline;
#[cfg(feature = "training")]
pub mod leaf_eval;
#[cfg(feature = "training")]
pub mod cuda_net;
#[cfg(feature = "training")]
pub mod turn_solver;
#[cfg(feature = "training")]
pub mod turn_pipeline;
#[cfg(feature = "training")]
pub mod flop_solver;
#[cfg(feature = "training")]
pub mod flop_pipeline;
#[cfg(feature = "training")]
pub mod preflop_pipeline;
pub mod stack_config;
