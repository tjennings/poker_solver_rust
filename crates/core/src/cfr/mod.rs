#[cfg(feature = "gpu")]
pub mod batched;
#[cfg(feature = "gpu")]
pub mod compiled;
pub mod exploitability;
pub mod mccfr;
pub mod regret;
#[cfg(feature = "gpu")]
pub mod tensor_ops;
pub mod vanilla;

#[cfg(feature = "gpu")]
pub use batched::BatchedCfr;
#[cfg(feature = "gpu")]
pub use compiled::{CompiledGame, compile};
pub use exploitability::calculate_exploitability;
pub use mccfr::{MccfrConfig, MccfrSolver};
#[cfg(feature = "gpu")]
pub use tensor_ops::{
    accumulate_strategy, compute_average_strategy, regret_match as regret_match_tensor,
    update_regrets_cfr_plus,
};
pub use vanilla::VanillaCfr;
