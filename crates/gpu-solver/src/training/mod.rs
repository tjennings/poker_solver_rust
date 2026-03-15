//! GPU-resident training pipeline for river CFVNet.
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
