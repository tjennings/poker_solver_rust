//! Batched CFR+ solver using tensor operations.
//!
//! This module provides a GPU-accelerated CFR+ implementation that processes
//! all info sets in parallel using tensor operations.

// Allow doc_markdown for tensor shape notation like [num_info_sets, max_actions]
// Allow casts for tensor index conversions (i32/i64/usize interop)
#![allow(
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::collections::HashMap;

use burn::prelude::*;

use super::compiled::CompiledGame;
use super::tensor_ops::{
    accumulate_strategy, compute_average_strategy, regret_match, update_regrets_cfr_plus,
};

/// Batched CFR+ solver state.
///
/// Stores regrets and strategy sums as tensors for efficient GPU processing.
#[derive(Debug)]
pub struct BatchedCfr<B: Backend> {
    /// Cumulative regrets [num_info_sets, max_actions]
    regrets: Tensor<B, 2>,
    /// Cumulative strategy sums [num_info_sets, max_actions]
    strategy_sums: Tensor<B, 2>,
    /// Action mask for valid actions [num_info_sets, max_actions]
    action_mask: Tensor<B, 2, Bool>,
    /// Number of info sets
    num_info_sets: usize,
    /// Maximum actions per info set
    max_actions: usize,
    /// Iteration count
    iteration: usize,
}

impl<B: Backend> BatchedCfr<B> {
    /// Create a new batched CFR+ solver from a compiled game.
    ///
    /// Initializes regrets and strategy sums to zero.
    pub fn new(compiled: &CompiledGame<B>, device: &B::Device) -> Self {
        let num_info_sets = compiled.num_info_sets;
        let max_actions = compiled.max_actions;

        // Initialize regrets to zero
        let regrets = Tensor::zeros([num_info_sets, max_actions], device);
        let strategy_sums = Tensor::zeros([num_info_sets, max_actions], device);

        // Build action mask per info set from the compiled game's action mask
        // We need to extract the action mask for each info set
        let action_mask = build_info_set_action_mask(compiled, device);

        Self {
            regrets,
            strategy_sums,
            action_mask,
            num_info_sets,
            max_actions,
            iteration: 0,
        }
    }

    /// Get the current strategy via regret matching.
    ///
    /// Returns tensor of shape [num_info_sets, max_actions].
    pub fn current_strategy(&self) -> Tensor<B, 2> {
        regret_match(self.regrets.clone(), self.action_mask.clone())
    }

    /// Get the average strategy computed from accumulated strategy sums.
    ///
    /// Returns tensor of shape [num_info_sets, max_actions].
    pub fn average_strategy(&self) -> Tensor<B, 2> {
        compute_average_strategy(self.strategy_sums.clone(), self.action_mask.clone())
    }

    /// Update regrets with deltas (CFR+ style with flooring).
    pub fn update_regrets(&mut self, deltas: Tensor<B, 2>) {
        self.regrets = update_regrets_cfr_plus(self.regrets.clone(), deltas);
    }

    /// Accumulate strategy weighted by reach probabilities.
    pub fn accumulate_strategy(&mut self, reach: Tensor<B, 1>) {
        let strategy = self.current_strategy();
        self.strategy_sums = accumulate_strategy(self.strategy_sums.clone(), strategy, reach);
        self.iteration += 1;
    }

    /// Get the current iteration count.
    #[must_use]
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Get the number of info sets.
    #[must_use]
    pub fn num_info_sets(&self) -> usize {
        self.num_info_sets
    }

    /// Get strategy for a specific info set as a HashMap.
    ///
    /// Returns None if the info set index is out of bounds.
    #[must_use]
    pub fn get_strategy(&self, info_set_idx: usize) -> Option<HashMap<usize, f64>> {
        if info_set_idx >= self.num_info_sets {
            return None;
        }

        let avg = self.average_strategy();
        let data: Vec<f32> = avg.to_data().to_vec().ok()?;
        let mask_data: Vec<bool> = self.action_mask.clone().to_data().to_vec().ok()?;

        let start = info_set_idx * self.max_actions;
        let mut strategy = HashMap::new();

        for action in 0..self.max_actions {
            if mask_data[start + action] {
                strategy.insert(action, f64::from(data[start + action]));
            }
        }

        Some(strategy)
    }
}

/// Build action mask per info set from compiled game.
///
/// This extracts the valid actions for each info set by examining
/// nodes that belong to each info set.
fn build_info_set_action_mask<B: Backend>(
    compiled: &CompiledGame<B>,
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    let num_info_sets = compiled.num_info_sets;
    let max_actions = compiled.max_actions;

    // Get info set indices for each node
    let info_set_data: Vec<i64> = compiled.node_info_set.to_data().to_vec().unwrap();
    let num_actions_data: Vec<i64> = compiled.node_num_actions.to_data().to_vec().unwrap();

    // For each info set, find the number of actions (all nodes in same info set have same actions)
    let mut info_set_num_actions = vec![0usize; num_info_sets];

    for (node_idx, &info_set_idx) in info_set_data.iter().enumerate() {
        if info_set_idx >= 0 {
            let idx = info_set_idx as usize;
            if idx < num_info_sets {
                info_set_num_actions[idx] = num_actions_data[node_idx] as usize;
            }
        }
    }

    // Build mask: action is valid if action_idx < num_actions for that info set
    let mut mask_int = Vec::with_capacity(num_info_sets * max_actions);
    for &num_actions in &info_set_num_actions {
        for action in 0..max_actions {
            if action < num_actions {
                mask_int.push(1i32);
            } else {
                mask_int.push(0i32);
            }
        }
    }

    Tensor::<B, 1, Int>::from_ints(mask_int.as_slice(), device)
        .reshape([num_info_sets, max_actions])
        .equal_elem(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::compile;
    use crate::game::KuhnPoker;
    use burn::backend::ndarray::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn batched_cfr_initializes() {
        let game = KuhnPoker::new();
        let device = Default::default();
        let compiled: CompiledGame<TestBackend> = compile(&game, &device);
        let solver = BatchedCfr::new(&compiled, &device);

        assert_eq!(solver.num_info_sets(), 12);
        assert_eq!(solver.iteration(), 0);
    }

    #[test]
    fn initial_strategy_is_uniform() {
        let game = KuhnPoker::new();
        let device = Default::default();
        let compiled: CompiledGame<TestBackend> = compile(&game, &device);
        let solver = BatchedCfr::new(&compiled, &device);

        let strategy = solver.current_strategy();
        let data: Vec<f32> = strategy.to_data().to_vec().unwrap();

        // Each info set should have uniform strategy over valid actions
        // Kuhn has 2 actions at each info set
        for info_set in 0..solver.num_info_sets() {
            let start = info_set * solver.max_actions;
            let sum: f32 = data[start..start + solver.max_actions].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Info set {} strategy sums to {}, expected 1.0",
                info_set,
                sum
            );
        }
    }

    #[test]
    fn regret_update_changes_strategy() {
        let game = KuhnPoker::new();
        let device = Default::default();
        let compiled: CompiledGame<TestBackend> = compile(&game, &device);
        let mut solver = BatchedCfr::new(&compiled, &device);

        let initial_strategy = solver.current_strategy();

        // Add positive regret to first action for first info set
        let mut deltas_data = vec![0.0f32; solver.num_info_sets() * solver.max_actions];
        deltas_data[0] = 1.0; // First action of first info set
        let deltas = Tensor::<TestBackend, 1>::from_floats(deltas_data.as_slice(), &device)
            .reshape([solver.num_info_sets(), solver.max_actions]);

        solver.update_regrets(deltas);

        let new_strategy = solver.current_strategy();

        // Strategy should have changed for first info set
        let initial_data: Vec<f32> = initial_strategy.to_data().to_vec().unwrap();
        let new_data: Vec<f32> = new_strategy.to_data().to_vec().unwrap();

        // First action should have higher probability now
        assert!(
            new_data[0] > initial_data[0],
            "Expected first action probability to increase: {} -> {}",
            initial_data[0],
            new_data[0]
        );
    }

    #[test]
    fn strategy_accumulation_works() {
        let game = KuhnPoker::new();
        let device = Default::default();
        let compiled: CompiledGame<TestBackend> = compile(&game, &device);
        let mut solver = BatchedCfr::new(&compiled, &device);

        // Uniform reach probabilities
        let reach = Tensor::<TestBackend, 1>::ones([solver.num_info_sets()], &device);

        solver.accumulate_strategy(reach);

        assert_eq!(solver.iteration(), 1);

        // Average strategy should be uniform since we only accumulated once with uniform
        let avg = solver.average_strategy();
        let data: Vec<f32> = avg.to_data().to_vec().unwrap();

        for info_set in 0..solver.num_info_sets() {
            let start = info_set * solver.max_actions;
            let sum: f32 = data[start..start + solver.max_actions].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Info set {} average strategy sums to {}",
                info_set,
                sum
            );
        }
    }

    #[test]
    fn get_strategy_returns_hashmap() {
        let game = KuhnPoker::new();
        let device = Default::default();
        let compiled: CompiledGame<TestBackend> = compile(&game, &device);
        let mut solver = BatchedCfr::new(&compiled, &device);

        // Accumulate once to have non-zero strategy sums
        let reach = Tensor::<TestBackend, 1>::ones([solver.num_info_sets()], &device);
        solver.accumulate_strategy(reach);

        let strategy = solver
            .get_strategy(0)
            .expect("Should get strategy for info set 0");

        // Should have 2 actions (Kuhn Poker)
        assert_eq!(strategy.len(), 2);

        // Probabilities should sum to 1
        let sum: f64 = strategy.values().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn get_strategy_out_of_bounds_returns_none() {
        let game = KuhnPoker::new();
        let device = Default::default();
        let compiled: CompiledGame<TestBackend> = compile(&game, &device);
        let solver = BatchedCfr::new(&compiled, &device);

        assert!(solver.get_strategy(100).is_none());
    }
}
