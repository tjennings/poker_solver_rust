//! GPU-accelerated CFR+ solver using tensor operations.
//!
//! This module provides a complete CFR solver that runs on GPU using the Burn
//! framework. It uses level-based tree traversal to maximize parallelism.

#![allow(
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

use std::collections::HashMap;

use burn::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

use super::compiled::{GameMetadata, TreeBuilder, build_game_tree};
use super::tensor_ops::regret_match;
use crate::game::Game;

/// Cached tree structure for efficient tensor-based iteration.
/// Precomputed indices enable parallel processing by depth level.
///
/// Built directly from `TreeBuilder` to avoid redundant tensor allocation.
struct TreeCache {
    /// Node indices grouped by depth level: levels[d] contains indices of nodes at depth d
    levels: Vec<Vec<usize>>,
    /// For each node: [parent_idx, parent_action, parent_player, parent_info_set]
    /// Stores PARENT's player and info set for reach computation
    /// -1 values indicate root nodes or terminals
    node_info: Vec<[i32; 4]>,
    /// For each node: [player, info_set_idx] - the node's own player and info set
    node_self_info: Vec<[i32; 2]>,
    /// For each node: indices of children for each action, -1 if no child
    /// Flattened: node_children[node * max_actions + action]
    node_children: Vec<i32>,
    /// Terminal utilities [num_nodes * 2] - utils[node*2] for P1, utils[node*2+1] for P2
    terminal_utils: Vec<f32>,
    /// Terminal mask
    terminal: Vec<bool>,
    /// Initial reach probabilities
    initial_reach: Vec<f32>,
    /// Root node indices
    root_nodes: Vec<usize>,
    /// Basic dimensions
    num_nodes: usize,
    max_actions: usize,
    num_info_sets: usize,
}

impl TreeCache {
    /// Build a `TreeCache` directly from a `TreeBuilder`, skipping tensors.
    ///
    /// This eliminates the redundant path: TreeBuilder → tensors → extract back to Vecs.
    fn from_builder(builder: &TreeBuilder, root_nodes: &[i32]) -> Self {
        let num_nodes = builder.node_player.len();
        let max_actions = builder.max_actions.max(1);
        let num_info_sets = builder.info_set_to_idx.len();
        let max_depth = builder.max_depth;

        // Group nodes by depth level
        let mut levels: Vec<Vec<usize>> = vec![Vec::new(); max_depth + 1];
        for (node, &depth) in builder.node_depth.iter().enumerate() {
            if depth >= 0 {
                levels[depth as usize].push(node);
            }
        }

        // Build node info array: [parent_idx, parent_action, parent_player, parent_info_set]
        let mut node_info = Vec::with_capacity(num_nodes);
        let mut node_self_info = Vec::with_capacity(num_nodes);
        for node in 0..num_nodes {
            let parent_idx = builder.node_parent[node];
            let (parent_player, parent_is) = if parent_idx >= 0 {
                let p = parent_idx as usize;
                (builder.node_player[p], builder.node_info_set[p])
            } else {
                (-1, -1)
            };
            node_info.push([
                parent_idx,
                builder.node_parent_action[node],
                parent_player,
                parent_is,
            ]);
            node_self_info.push([builder.node_player[node], builder.node_info_set[node]]);
        }

        // Flatten action_child to [node * max_actions + action]
        let mut node_children = Vec::with_capacity(num_nodes * max_actions);
        for children in &builder.action_child {
            node_children.extend_from_slice(children);
        }

        // Flatten terminal_utils to [node*2 + player]
        let mut terminal_utils = Vec::with_capacity(num_nodes * 2);
        for utils in &builder.terminal_utils {
            terminal_utils.push(utils[0]);
            terminal_utils.push(utils[1]);
        }

        // Compute initial reach: uniform over initial states
        let num_initial = root_nodes.len();
        let initial_reach_val = 1.0 / num_initial as f32;
        let initial_reach = vec![initial_reach_val; num_initial];

        let root_nodes_usize: Vec<usize> = root_nodes.iter().map(|&x| x as usize).collect();

        Self {
            levels,
            node_info,
            node_self_info,
            node_children,
            terminal_utils,
            terminal: builder.terminal.clone(),
            initial_reach,
            root_nodes: root_nodes_usize,
            num_nodes,
            max_actions,
            num_info_sets,
        }
    }
}

/// GPU-accelerated CFR+ solver.
///
/// Uses tensor operations to process the game tree in parallel, with
/// level-based traversal for computing counterfactual values.
pub struct GpuCfrSolver<B: Backend> {
    /// Game metadata (info set mappings, dimensions)
    metadata: GameMetadata,
    /// Cached tree structure (built directly from TreeBuilder, no tensor overhead)
    cache: TreeCache,
    /// Cumulative regrets [num_info_sets, max_actions]
    regrets: Tensor<B, 2>,
    /// Cumulative strategy sums [num_info_sets, max_actions]
    strategy_sums: Tensor<B, 2>,
    /// Action mask per info set [num_info_sets, max_actions]
    info_set_action_mask: Tensor<B, 2, Bool>,
    /// Device
    device: B::Device,
    /// Iteration count
    iteration: u64,

    // Persistent per-iteration buffers (allocated once, reused every iteration)
    reach_p1: Vec<f32>,
    reach_p2: Vec<f32>,
    values_p1: Vec<f32>,
    values_p2: Vec<f32>,
    regret_updates: Vec<f32>,
    strategy_updates: Vec<f32>,
}

impl<B: Backend> GpuCfrSolver<B> {
    /// Create a new GPU solver for the given game.
    #[allow(clippy::missing_panics_doc)] // progress bar templates are static
    pub fn new<G: Game>(game: &G, device: B::Device) -> Self {
        let (builder, metadata, root_nodes) = build_game_tree(game);

        let num_info_sets = metadata.num_info_sets;
        let max_actions = metadata.max_actions;
        let num_nodes = metadata.num_nodes;

        let sp = ProgressBar::new_spinner();
        sp.set_style(
            ProgressStyle::with_template("  {spinner} {msg}")
                .unwrap(),
        );
        sp.enable_steady_tick(std::time::Duration::from_millis(100));

        sp.set_message("allocating solver tensors...");
        let regrets = Tensor::zeros([num_info_sets, max_actions], &device);
        let strategy_sums = Tensor::zeros([num_info_sets, max_actions], &device);

        sp.set_message("building action mask...");
        let info_set_action_mask =
            build_info_set_action_mask(&builder, num_info_sets, max_actions, &device);

        sp.set_message("building tree cache...");
        let cache = TreeCache::from_builder(&builder, &root_nodes);

        sp.finish_and_clear();

        Self {
            metadata,
            cache,
            regrets,
            strategy_sums,
            info_set_action_mask,
            device,
            iteration: 0,
            // Allocate persistent buffers once
            reach_p1: vec![0.0f32; num_nodes],
            reach_p2: vec![0.0f32; num_nodes],
            values_p1: vec![0.0f32; num_nodes],
            values_p2: vec![0.0f32; num_nodes],
            regret_updates: vec![0.0f32; num_info_sets * max_actions],
            strategy_updates: vec![0.0f32; num_info_sets * max_actions],
        }
    }

    /// Train for the specified number of iterations.
    #[allow(clippy::missing_panics_doc)] // progress bar templates are static
    pub fn train(&mut self, iterations: u64) {
        let pb = ProgressBar::new(iterations);
        pb.set_style(ProgressStyle::with_template(
            "  training [{bar:40}] {pos}/{len} iters [{elapsed} < {eta}, {per_sec}]"
        ).unwrap());

        for _ in 0..iterations {
            self.iterate();
            pb.inc(1);
        }

        pb.finish_and_clear();
    }

    /// Train with a progress callback.
    pub fn train_with_callback<F>(&mut self, iterations: u64, mut callback: F)
    where
        F: FnMut(u64),
    {
        for i in 0..iterations {
            self.iterate();
            if (i + 1) % 100 == 0 || i == iterations - 1 {
                callback(self.iteration);
            }
        }
    }

    /// Perform one CFR iteration using tensor operations.
    fn iterate(&mut self) {
        let c = &self.cache;
        let num_nodes = c.num_nodes;
        let max_actions = c.max_actions;
        let num_info_sets = c.num_info_sets;

        // Get current strategy using tensor regret matching
        let strategy = regret_match(self.regrets.clone(), self.info_set_action_mask.clone());
        let strategy_data: Vec<f32> = strategy.to_data().to_vec().unwrap();

        // ========================================
        // TOP-DOWN PASS: Compute reach probabilities
        // ========================================

        // Zero out persistent buffers
        self.reach_p1[..num_nodes].fill(0.0);
        self.reach_p2[..num_nodes].fill(0.0);

        // Initialize root nodes
        for (i, &root) in c.root_nodes.iter().enumerate() {
            let prob = c.initial_reach[i];
            self.reach_p1[root] = prob;
            self.reach_p2[root] = prob;
        }

        // Process level by level (top-down)
        for level in &c.levels {
            for &node in level {
                let [parent_idx, parent_action, parent_player, parent_info_set] = c.node_info[node];
                if parent_idx < 0 {
                    continue;
                }

                let parent = parent_idx as usize;
                let action_idx = parent_action as usize;

                // Get action probability from PARENT's strategy
                let action_prob = if parent_info_set >= 0 {
                    let is_idx = parent_info_set as usize;
                    strategy_data[is_idx * max_actions + action_idx]
                } else {
                    1.0
                };

                // Update reach based on which player acted (PARENT's player)
                if parent_player == 0 {
                    // Parent was P1's node, so P1's reach gets multiplied
                    self.reach_p1[node] = self.reach_p1[parent] * action_prob;
                    self.reach_p2[node] = self.reach_p2[parent];
                } else {
                    // Parent was P2's node, so P2's reach gets multiplied
                    self.reach_p1[node] = self.reach_p1[parent];
                    self.reach_p2[node] = self.reach_p2[parent] * action_prob;
                }
            }
        }

        // ========================================
        // BOTTOM-UP PASS: Compute counterfactual values
        // ========================================

        // Zero out persistent buffers
        self.values_p1[..num_nodes].fill(0.0);
        self.values_p2[..num_nodes].fill(0.0);

        // Initialize terminal values
        for node in 0..num_nodes {
            if c.terminal[node] {
                self.values_p1[node] = c.terminal_utils[node * 2];
                self.values_p2[node] = c.terminal_utils[node * 2 + 1];
            }
        }

        // Process level by level (bottom-up)
        for level in c.levels.iter().rev() {
            for &node in level {
                if c.terminal[node] {
                    continue;
                }

                let [_player, info_set] = c.node_self_info[node];
                if info_set < 0 {
                    continue;
                }
                let is_idx = info_set as usize;

                let mut val_p1 = 0.0f32;
                let mut val_p2 = 0.0f32;

                for action in 0..max_actions {
                    let child_idx = c.node_children[node * max_actions + action];
                    if child_idx < 0 {
                        continue;
                    }
                    let child = child_idx as usize;
                    let action_prob = strategy_data[is_idx * max_actions + action];

                    val_p1 += action_prob * self.values_p1[child];
                    val_p2 += action_prob * self.values_p2[child];
                }

                self.values_p1[node] = val_p1;
                self.values_p2[node] = val_p2;
            }
        }

        // ========================================
        // COMPUTE REGRET AND STRATEGY UPDATES
        // ========================================

        // Zero out persistent buffers
        self.regret_updates[..num_info_sets * max_actions].fill(0.0);
        self.strategy_updates[..num_info_sets * max_actions].fill(0.0);

        for &node in c.levels.iter().flatten() {
            if c.terminal[node] {
                continue;
            }

            let [player, info_set] = c.node_self_info[node];
            if info_set < 0 {
                continue;
            }
            let is_idx = info_set as usize;
            let player_idx = player as usize;

            let (player_reach, opponent_reach) = if player_idx == 0 {
                (self.reach_p1[node], self.reach_p2[node])
            } else {
                (self.reach_p2[node], self.reach_p1[node])
            };

            let node_value = if player_idx == 0 {
                self.values_p1[node]
            } else {
                self.values_p2[node]
            };

            for action in 0..max_actions {
                let child_idx = c.node_children[node * max_actions + action];
                if child_idx < 0 {
                    continue;
                }
                let child = child_idx as usize;

                let action_value = if player_idx == 0 {
                    self.values_p1[child]
                } else {
                    self.values_p2[child]
                };

                // Counterfactual regret
                let regret = (action_value - node_value) * opponent_reach;
                self.regret_updates[is_idx * max_actions + action] += regret;

                // Strategy accumulation
                let action_prob = strategy_data[is_idx * max_actions + action];
                self.strategy_updates[is_idx * max_actions + action] += player_reach * action_prob;
            }
        }

        // Apply updates as tensors
        let regret_delta =
            Tensor::<B, 1>::from_floats(self.regret_updates.as_slice(), &self.device)
                .reshape([num_info_sets, max_actions]);
        self.regrets = self.regrets.clone().add(regret_delta).clamp_min(0.0);

        let strategy_delta =
            Tensor::<B, 1>::from_floats(self.strategy_updates.as_slice(), &self.device)
                .reshape([num_info_sets, max_actions]);
        self.strategy_sums = self.strategy_sums.clone().add(strategy_delta);

        self.iteration += 1;
    }

    /// Get the average strategy for an info set by key.
    pub fn get_strategy(&self, info_set: &str) -> Option<Vec<f64>> {
        let idx = *self.metadata.info_set_to_idx.get(info_set)?;
        self.get_strategy_by_idx(idx)
    }

    /// Get the average strategy for an info set by index.
    ///
    /// # Panics
    ///
    /// Panics if tensor data conversion fails (should not happen in practice).
    pub fn get_strategy_by_idx(&self, idx: usize) -> Option<Vec<f64>> {
        if idx >= self.metadata.num_info_sets {
            return None;
        }

        let max_actions = self.metadata.max_actions;
        let strategy_sums_data: Vec<f32> = self.strategy_sums.to_data().to_vec().unwrap();
        let mask_data: Vec<bool> = self.info_set_action_mask.to_data().to_vec().unwrap();

        let start = idx * max_actions;
        let mut probs = Vec::new();
        let mut total = 0.0f32;

        for action in 0..max_actions {
            if mask_data[start + action] {
                let sum = strategy_sums_data[start + action];
                probs.push(sum);
                total += sum;
            }
        }

        if total > 0.0 {
            for p in &mut probs {
                *p /= total;
            }
        } else {
            let uniform = 1.0 / probs.len() as f32;
            probs.fill(uniform);
        }

        Some(probs.into_iter().map(f64::from).collect())
    }

    /// Get all strategies as a HashMap.
    ///
    /// This is optimized to extract tensor data once rather than per info set.
    ///
    /// # Panics
    ///
    /// Panics if tensor data extraction fails.
    pub fn all_strategies(&self) -> HashMap<String, Vec<f64>> {
        let max_actions = self.metadata.max_actions;

        // Extract tensor data once (not per info set!)
        let strategy_sums_data: Vec<f32> = self.strategy_sums.to_data().to_vec().unwrap();
        let mask_data: Vec<bool> = self.info_set_action_mask.to_data().to_vec().unwrap();

        let mut result = HashMap::with_capacity(self.metadata.info_set_to_idx.len());

        for (key, &idx) in &self.metadata.info_set_to_idx {
            let start = idx * max_actions;
            let mut probs = Vec::new();
            let mut total = 0.0f32;

            for action in 0..max_actions {
                if mask_data[start + action] {
                    let sum = strategy_sums_data[start + action];
                    probs.push(sum);
                    total += sum;
                }
            }

            if total > 0.0 {
                for p in &mut probs {
                    *p /= total;
                }
            } else {
                let uniform = 1.0 / probs.len() as f32;
                probs.fill(uniform);
            }

            result.insert(key.clone(), probs.into_iter().map(f64::from).collect());
        }

        result
    }

    /// Get the info set to index mapping.
    pub fn info_set_to_idx(&self) -> &HashMap<String, usize> {
        &self.metadata.info_set_to_idx
    }

    /// Current iteration count.
    pub fn iterations(&self) -> u64 {
        self.iteration
    }
}

/// Build action mask per info set directly from `TreeBuilder` data.
fn build_info_set_action_mask<B: Backend>(
    builder: &TreeBuilder,
    num_info_sets: usize,
    max_actions: usize,
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    let mut info_set_num_actions = vec![0usize; num_info_sets];

    for (node_idx, &info_set_idx) in builder.node_info_set.iter().enumerate() {
        if info_set_idx >= 0 {
            let idx = info_set_idx as usize;
            if idx < num_info_sets {
                info_set_num_actions[idx] = builder.node_num_actions[node_idx] as usize;
            }
        }
    }

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
    use crate::game::KuhnPoker;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use test_macros::timed_test;

    type TestBackend = NdArray;

    #[timed_test]
    fn solver_initializes() {

        let game = KuhnPoker::new();
        let device = NdArrayDevice::default();
        let solver = GpuCfrSolver::<TestBackend>::new(&game, device);

        assert_eq!(solver.iterations(), 0);
        assert_eq!(solver.metadata.num_info_sets, 12);
    }

    #[timed_test]
    fn initial_strategy_is_uniform() {

        let game = KuhnPoker::new();
        let device = NdArrayDevice::default();
        let solver = GpuCfrSolver::<TestBackend>::new(&game, device);

        let strategy = solver.get_strategy("K").expect("Should have K");
        assert_eq!(strategy.len(), 2);

        let sum: f64 = strategy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[timed_test]
    fn training_increases_iterations() {

        let game = KuhnPoker::new();
        let device = NdArrayDevice::default();
        let mut solver = GpuCfrSolver::<TestBackend>::new(&game, device);

        solver.train(10);
        assert_eq!(solver.iterations(), 10);
    }

    #[timed_test]
    fn strategy_changes_after_training() {

        let game = KuhnPoker::new();
        let device = NdArrayDevice::default();
        let mut solver = GpuCfrSolver::<TestBackend>::new(&game, device);

        let initial = solver.get_strategy("K").unwrap();
        solver.train(100);
        let trained = solver.get_strategy("K").unwrap();

        let diff: f64 = initial
            .iter()
            .zip(trained.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.01, "Strategy should change after training");
    }
}
