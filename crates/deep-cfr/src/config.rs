/// Configuration for the Single Deep CFR solver.
#[derive(Debug, Clone)]
pub struct SdCfrConfig {
    /// Number of CFR iterations (T)
    pub cfr_iterations: u32,
    /// Number of traversals per player per iteration (K)
    pub traversals_per_iter: u32,
    /// Maximum capacity of advantage memory reservoir buffer
    pub advantage_memory_cap: usize,
    /// Hidden dimension for neural network layers
    pub hidden_dim: usize,
    /// Maximum number of actions in the game
    pub num_actions: usize,
    /// Number of SGD steps when training the value network
    pub sgd_steps: usize,
    /// Training batch size
    pub batch_size: usize,
    /// Adam learning rate
    pub learning_rate: f64,
    /// Gradient norm clipping threshold
    pub grad_clip_norm: f64,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for SdCfrConfig {
    fn default() -> Self {
        Self {
            cfr_iterations: 100,
            traversals_per_iter: 1_000,
            advantage_memory_cap: 100_000,
            hidden_dim: 64,
            num_actions: 14,
            sgd_steps: 4_000,
            batch_size: 10_000,
            learning_rate: 0.001,
            grad_clip_norm: 1.0,
            seed: 42,
        }
    }
}

impl SdCfrConfig {
    pub fn validate(&self) -> Result<(), crate::SdCfrError> {
        if self.cfr_iterations == 0 {
            return Err(crate::SdCfrError::Config(
                "cfr_iterations must be > 0".into(),
            ));
        }
        if self.hidden_dim == 0 {
            return Err(crate::SdCfrError::Config("hidden_dim must be > 0".into()));
        }
        if self.num_actions == 0 {
            return Err(crate::SdCfrError::Config("num_actions must be > 0".into()));
        }
        if self.batch_size == 0 {
            return Err(crate::SdCfrError::Config("batch_size must be > 0".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        SdCfrConfig::default().validate().unwrap();
    }

    #[test]
    fn zero_iterations_is_invalid() {
        let config = SdCfrConfig {
            cfr_iterations: 0,
            ..SdCfrConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn zero_hidden_dim_is_invalid() {
        let config = SdCfrConfig {
            hidden_dim: 0,
            ..SdCfrConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
