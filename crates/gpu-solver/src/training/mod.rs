//! GPU-resident training pipeline for river CFVNet.
//!
//! Modules for sampling situations, evaluating hand strengths, building
//! batch solvers, and storing training examples in a reservoir buffer.

pub mod builder;
pub mod hand_eval;
pub mod reservoir;
