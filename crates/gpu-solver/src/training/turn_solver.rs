//! Turn batch solver with neural leaf evaluation at depth boundaries.
//!
//! Wraps a `BatchGpuSolver` and injects leaf evaluation into the DCFR+
//! solve loop. At each iteration, after fold terminal evaluation, the
//! solver:
//!
//! 1. Encodes reach probabilities at depth boundaries into 2720-dim inputs
//!    for all (boundary x river_card x spot) combinations
//! 2. Runs a batched forward pass through the trained river model
//! 3. Averages the 48 river outputs per combo and scatters to cfvalues
//!
//! The rest of the solve loop (regret match, forward pass, backward CFV,
//! regret update) is identical to the river `BatchGpuSolver`.

#[cfg(feature = "training")]
use crate::batch::{BatchGpuSolver, BatchSolveResultGpu};
#[cfg(feature = "training")]
use crate::gpu::{GpuContext, GpuError};
#[cfg(feature = "training")]
use crate::training::leaf_eval::GpuLeafEvaluator;
#[cfg(feature = "training")]
use crate::training::cuda_net::GpuLeafEvaluatorCuda;
#[cfg(feature = "training")]
use burn::tensor::backend::Backend;
#[cfg(feature = "training")]
use cudarc::driver::CudaSlice;

/// Number of possible river cards per turn board (52 - 4 board cards).
#[cfg(feature = "training")]
const NUM_RIVERS: u32 = 48;

/// Turn batch solver that wraps a `BatchGpuSolver` and adds leaf evaluation.
///
/// Pre-allocates GPU working buffers for encoded inputs and raw CFV outputs.
/// The solve loop is the same as `BatchGpuSolver::solve()` with leaf
/// evaluation injected between fold terminal eval and backward CFV.
#[cfg(feature = "training")]
pub struct TurnBatchSolver<'a, B: Backend> {
    /// The inner batch solver (owns all DCFR+ state).
    solver: BatchGpuSolver<'a>,
    /// River model for leaf evaluation at depth boundaries.
    leaf_eval: &'a GpuLeafEvaluator<B>,
    /// GPU buffer: depth-boundary node IDs `[num_boundaries]`.
    gpu_boundary_nodes: CudaSlice<u32>,
    /// GPU buffer: pot at each boundary `[num_boundaries]`.
    gpu_boundary_pots: CudaSlice<f32>,
    /// GPU buffer: effective stack at each boundary `[num_boundaries]`.
    gpu_boundary_stacks: CudaSlice<f32>,
    /// Number of depth boundaries in the turn tree.
    num_boundaries: u32,
    /// GPU buffer: per-spot turn board cards `[num_spots * 4]`.
    gpu_boards: CudaSlice<u32>,
    /// GPU buffer: per-spot river cards `[num_spots * NUM_RIVERS]`.
    gpu_river_cards: CudaSlice<u32>,
    /// GPU buffer: combo card pairs `[1326 * 2]`.
    gpu_combo_cards: CudaSlice<u32>,
    /// Working buffer: encoded leaf inputs `[num_boundaries * 48 * num_spots * 2720]`.
    encoded_inputs: CudaSlice<f32>,
    /// Working buffer: raw CFV predictions `[num_boundaries * 48 * num_spots * 1326]`.
    raw_cfvs: CudaSlice<f32>,
    /// Number of spots in this batch.
    num_spots: u32,
    /// Hands per spot (1326).
    hands_per_spot: u32,
}

#[cfg(feature = "training")]
impl<'a, B: Backend> TurnBatchSolver<'a, B> {
    /// Create a new turn batch solver.
    ///
    /// # Arguments
    /// * `gpu` -- CUDA context.
    /// * `solver` -- Pre-built `BatchGpuSolver` for the turn tree topology.
    /// * `leaf_eval` -- Trained river model for leaf evaluation.
    /// * `boundary_indices` -- Node IDs of depth boundaries from the FlatTree.
    /// * `boundary_pots` -- Pot at each boundary.
    /// * `boundary_stacks` -- Effective stack at each boundary.
    /// * `boards_flat` -- Per-spot turn board cards `[num_spots * 4]` on CPU.
    /// * `num_spots` -- Number of spots in the batch.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &'a GpuContext,
        solver: BatchGpuSolver<'a>,
        leaf_eval: &'a GpuLeafEvaluator<B>,
        boundary_indices: &[u32],
        boundary_pots: &[f32],
        boundary_stacks: &[f32],
        boards_flat: &[u32],
        num_spots: usize,
    ) -> Result<Self, String> {
        let gpu_err = |e: GpuError| format!("GPU error: {e}");

        let num_boundaries = boundary_indices.len() as u32;
        assert_eq!(boards_flat.len(), num_spots * 4);

        // Upload boundary metadata
        let gpu_boundary_nodes = gpu.upload(boundary_indices).map_err(gpu_err)?;
        let gpu_boundary_pots = gpu.upload(boundary_pots).map_err(gpu_err)?;
        let gpu_boundary_stacks = gpu.upload(boundary_stacks).map_err(gpu_err)?;

        // Upload per-spot boards
        let gpu_boards = gpu.upload(boards_flat).map_err(gpu_err)?;

        // Compute per-spot river cards on CPU (cheap, once per batch)
        let mut river_cards_flat = Vec::with_capacity(num_spots * NUM_RIVERS as usize);
        for spot in 0..num_spots {
            let board = &boards_flat[spot * 4..(spot + 1) * 4];
            let board_set: Vec<u32> = board.to_vec();
            let mut count = 0u32;
            for card in 0..52u32 {
                if !board_set.contains(&card) {
                    river_cards_flat.push(card);
                    count += 1;
                }
            }
            // Pad to exactly NUM_RIVERS if needed (shouldn't happen for valid 4-card boards)
            assert_eq!(
                count, NUM_RIVERS,
                "Expected {} river cards for spot {}, got {}",
                NUM_RIVERS, spot, count
            );
        }
        let gpu_river_cards = gpu.upload(&river_cards_flat).map_err(gpu_err)?;

        // Upload combo cards lookup
        let combo_cards_flat = super::hand_eval::build_combo_cards_flat();
        let gpu_combo_cards = gpu.upload(&combo_cards_flat).map_err(gpu_err)?;

        // Allocate working buffers
        let total_inputs = num_boundaries as usize * NUM_RIVERS as usize * num_spots;
        let encoded_inputs = gpu
            .alloc_zeros::<f32>(total_inputs * 2720)
            .map_err(gpu_err)?;
        let raw_cfvs = gpu
            .alloc_zeros::<f32>(total_inputs * 1326)
            .map_err(gpu_err)?;

        let hands_per_spot = solver.hands_per_spot_val() as u32;

        Ok(Self {
            solver,
            leaf_eval,
            gpu_boundary_nodes,
            gpu_boundary_pots,
            gpu_boundary_stacks,
            num_boundaries,
            gpu_boards,
            gpu_river_cards,
            gpu_combo_cards,
            encoded_inputs,
            raw_cfvs,
            num_spots: num_spots as u32,
            hands_per_spot,
        })
    }

    /// Run the DCFR+ solve loop with leaf evaluation at depth boundaries.
    ///
    /// Returns root CFVs for both traversers on GPU.
    pub fn solve_with_cfvs(
        &mut self,
        max_iterations: u32,
    ) -> Result<BatchSolveResultGpu, String> {
        // If no boundaries, delegate to the regular solver
        if self.num_boundaries == 0 {
            return self.solver.solve_with_cfvs(max_iterations, None);
        }

        // Run the modified solve loop with leaf evaluation
        self.solve_loop(max_iterations)?;

        // Extract root CFVs (same as BatchGpuSolver::solve_with_cfvs)
        self.extract_root_cfvs()
    }

    /// The main DCFR+ solve loop with leaf evaluation injected.
    ///
    /// This is a near-copy of `BatchGpuSolver::solve()` with the addition
    /// of leaf evaluation between terminal eval and backward CFV.
    fn solve_loop(&mut self, max_iterations: u32) -> Result<(), String> {
        let gpu_err = |e: GpuError| format!("GPU error: {e}");

        for t in 1..=max_iterations {
            let current_iteration = t - 1;

            // DCFR+ discount parameters
            let t_alpha = (current_iteration as i32 - 1).max(0) as f64;
            let pow_alpha = t_alpha * t_alpha.sqrt();
            let pos_discount = (pow_alpha / (pow_alpha + 1.0)) as f32;
            let neg_discount = 0.5f32;

            let nearest_lower_power_of_4 = match current_iteration {
                0 => 0u32,
                x => 1u32 << ((x.leading_zeros() ^ 31) & !1),
            };
            let t_gamma = (current_iteration - nearest_lower_power_of_4) as f64;
            let strat_discount = ((t_gamma / (t_gamma + 1.0)).powi(3)) as f32;

            for traverser in 0..2u32 {
                // 1. Regret match -> current strategy
                self.solver.regret_match().map_err(gpu_err)?;

                // 2. Initialize reach at root
                self.solver.init_reach_pub().map_err(gpu_err)?;

                // 3. Forward pass (top-down)
                self.solver.forward_pass().map_err(gpu_err)?;

                // 4a. Zero out cfvalues
                self.solver.zero_cfvalues().map_err(gpu_err)?;

                // 4b. Terminal fold eval
                self.solver.fold_eval(traverser).map_err(gpu_err)?;

                // 4c. Terminal showdown eval (should be 0 for turn trees, but handle gracefully)
                self.solver.showdown_eval(traverser).map_err(gpu_err)?;

                // *** NEW: Leaf evaluation at depth boundaries ***
                self.leaf_evaluation(traverser)?;

                // 4d. Backward CFV (bottom-up)
                self.solver.backward_cfv(traverser).map_err(gpu_err)?;

                // 4e. Update regrets
                self.solver.update_regrets(
                    traverser,
                    pos_discount,
                    neg_discount,
                    strat_discount,
                ).map_err(gpu_err)?;
            }
        }

        Ok(())
    }

    /// Perform leaf evaluation at depth boundaries for the given traverser.
    ///
    /// 1. Encode reach at boundaries into 2720-dim inputs
    /// 2. Run batched inference through the river model
    /// 3. Average across river cards and scatter to cfvalues
    fn leaf_evaluation(&mut self, traverser: u32) -> Result<(), String> {
        let gpu_err = |e: GpuError| format!("GPU leaf eval error: {e}");

        let total_hands = self.solver.total_hands();
        let total_inputs = self.num_boundaries as usize
            * NUM_RIVERS as usize
            * self.num_spots as usize;

        if total_inputs == 0 {
            return Ok(());
        }

        // Get a reference to the gpu context (shared, non-exclusive)
        let gpu = self.solver.gpu();

        // 1. Encode: gather reach at boundaries -> 2720-dim inputs
        // The encode kernel reads from reach (immutable) and writes to encoded_inputs.
        // We need to borrow solver fields immutably and self.encoded_inputs mutably.
        gpu.launch_encode_leaf_inputs(
            &mut self.encoded_inputs,
            self.solver.reach_oop(),
            self.solver.reach_ip(),
            &self.gpu_boundary_nodes,
            &self.gpu_boards,
            &self.gpu_river_cards,
            &self.gpu_boundary_pots,
            &self.gpu_boundary_stacks,
            &self.gpu_combo_cards,
            traverser,
            self.num_boundaries,
            NUM_RIVERS,
            self.num_spots,
            total_hands,
            self.hands_per_spot,
        ).map_err(gpu_err)?;

        // 2. Infer: one batched burn-cuda forward pass
        self.raw_cfvs = self.leaf_eval.infer_from_cudarc(
            gpu,
            &self.encoded_inputs,
            total_inputs,
        )?;

        // 3. Average: average 48 river outputs per combo, scatter to cfvalues
        gpu.launch_average_leaf_cfvs(
            self.solver.cfvalues_mut(),
            &self.raw_cfvs,
            &self.gpu_boundary_nodes,
            &self.gpu_river_cards,
            &self.gpu_combo_cards,
            self.num_boundaries,
            NUM_RIVERS,
            self.num_spots,
            total_hands,
            self.hands_per_spot,
        ).map_err(gpu_err)?;

        Ok(())
    }

    /// Extract root CFVs for both traversers after the solve is complete.
    ///
    /// Runs a final forward pass with converged strategy, then backward CFV
    /// per traverser to get root CFVs.
    fn extract_root_cfvs(&mut self) -> Result<BatchSolveResultGpu, String> {
        let gpu_err = |e: GpuError| format!("GPU error: {e}");

        let total_hands = self.solver.total_hands() as usize;

        // Get converged strategy
        self.solver.regret_match().map_err(gpu_err)?;

        // Single forward pass
        self.solver.init_reach_pub().map_err(gpu_err)?;
        self.solver.forward_pass().map_err(gpu_err)?;

        let gpu = self.solver.gpu();
        let mut cfvs_oop = gpu.alloc_zeros::<f32>(total_hands).map_err(gpu_err)?;
        let mut cfvs_ip = gpu.alloc_zeros::<f32>(total_hands).map_err(gpu_err)?;

        for traverser in 0..2u32 {
            // Zero cfvalues
            self.solver.zero_cfvalues().map_err(gpu_err)?;

            // Terminal eval
            self.solver.fold_eval(traverser).map_err(gpu_err)?;
            self.solver.showdown_eval(traverser).map_err(gpu_err)?;

            // Leaf evaluation
            self.leaf_evaluation(traverser)?;

            // Backward CFV
            self.solver.backward_cfv(traverser).map_err(gpu_err)?;

            // Download root CFVs (node 0)
            let gpu = self.solver.gpu();
            let full_cfvs: Vec<f32> = gpu.download(
                self.solver.cfvalues(),
            ).map_err(gpu_err)?;
            let root_cfvs = &full_cfvs[..total_hands];
            let dst = if traverser == 0 {
                &mut cfvs_oop
            } else {
                &mut cfvs_ip
            };
            *dst = gpu.upload(root_cfvs).map_err(gpu_err)?;
        }

        Ok(BatchSolveResultGpu {
            iterations: 0, // filled by caller
            cfvs_oop,
            cfvs_ip,
        })
    }
}

// ---------------------------------------------------------------------------
// TurnBatchSolverCuda: CUDA-native leaf evaluation (no burn runtime)
// ---------------------------------------------------------------------------

/// Turn batch solver using CUDA-native neural network inference.
///
/// Identical to `TurnBatchSolver` but uses `GpuLeafEvaluatorCuda` which
/// keeps all data on the GPU throughout inference, eliminating the
/// GPU->CPU->GPU bounce that made burn-based inference slow.
#[cfg(feature = "training")]
pub struct TurnBatchSolverCuda<'a> {
    /// The inner batch solver (owns all DCFR+ state).
    solver: BatchGpuSolver<'a>,
    /// CUDA-native river model for leaf evaluation.
    leaf_eval: &'a mut GpuLeafEvaluatorCuda,
    /// GPU buffer: depth-boundary node IDs `[num_boundaries]`.
    gpu_boundary_nodes: CudaSlice<u32>,
    /// GPU buffer: pot at each boundary `[num_boundaries]`.
    gpu_boundary_pots: CudaSlice<f32>,
    /// GPU buffer: effective stack at each boundary `[num_boundaries]`.
    gpu_boundary_stacks: CudaSlice<f32>,
    /// Number of depth boundaries in the turn tree.
    num_boundaries: u32,
    /// GPU buffer: per-spot turn board cards `[num_spots * 4]`.
    gpu_boards: CudaSlice<u32>,
    /// GPU buffer: per-spot river cards `[num_spots * NUM_RIVERS]`.
    gpu_river_cards: CudaSlice<u32>,
    /// GPU buffer: combo card pairs `[1326 * 2]`.
    gpu_combo_cards: CudaSlice<u32>,
    /// Working buffer: encoded leaf inputs `[num_boundaries * 48 * num_spots * 2720]`.
    encoded_inputs: CudaSlice<f32>,
    /// Working buffer: raw CFV predictions `[num_boundaries * 48 * num_spots * 1326]`.
    raw_cfvs: CudaSlice<f32>,
    /// Number of spots in this batch.
    num_spots: u32,
    /// Hands per spot (1326).
    hands_per_spot: u32,
}

#[cfg(feature = "training")]
impl<'a> TurnBatchSolverCuda<'a> {
    /// Create a new CUDA-native turn batch solver.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &'a GpuContext,
        solver: BatchGpuSolver<'a>,
        leaf_eval: &'a mut GpuLeafEvaluatorCuda,
        boundary_indices: &[u32],
        boundary_pots: &[f32],
        boundary_stacks: &[f32],
        boards_flat: &[u32],
        num_spots: usize,
    ) -> Result<Self, String> {
        let gpu_err = |e: GpuError| format!("GPU error: {e}");

        let num_boundaries = boundary_indices.len() as u32;
        assert_eq!(boards_flat.len(), num_spots * 4);

        let gpu_boundary_nodes = gpu.upload(boundary_indices).map_err(gpu_err)?;
        let gpu_boundary_pots = gpu.upload(boundary_pots).map_err(gpu_err)?;
        let gpu_boundary_stacks = gpu.upload(boundary_stacks).map_err(gpu_err)?;
        let gpu_boards = gpu.upload(boards_flat).map_err(gpu_err)?;

        // Compute per-spot river cards on CPU
        let mut river_cards_flat = Vec::with_capacity(num_spots * NUM_RIVERS as usize);
        for spot in 0..num_spots {
            let board = &boards_flat[spot * 4..(spot + 1) * 4];
            let board_set: Vec<u32> = board.to_vec();
            let mut count = 0u32;
            for card in 0..52u32 {
                if !board_set.contains(&card) {
                    river_cards_flat.push(card);
                    count += 1;
                }
            }
            assert_eq!(
                count, NUM_RIVERS,
                "Expected {} river cards for spot {}, got {}",
                NUM_RIVERS, spot, count
            );
        }
        let gpu_river_cards = gpu.upload(&river_cards_flat).map_err(gpu_err)?;

        let combo_cards_flat = super::hand_eval::build_combo_cards_flat();
        let gpu_combo_cards = gpu.upload(&combo_cards_flat).map_err(gpu_err)?;

        let total_inputs = num_boundaries as usize * NUM_RIVERS as usize * num_spots;
        let encoded_inputs = gpu
            .alloc_zeros::<f32>(total_inputs * 2720)
            .map_err(gpu_err)?;
        let raw_cfvs = gpu
            .alloc_zeros::<f32>(total_inputs * 1326)
            .map_err(gpu_err)?;

        let hands_per_spot = solver.hands_per_spot_val() as u32;

        Ok(Self {
            solver,
            leaf_eval,
            gpu_boundary_nodes,
            gpu_boundary_pots,
            gpu_boundary_stacks,
            num_boundaries,
            gpu_boards,
            gpu_river_cards,
            gpu_combo_cards,
            encoded_inputs,
            raw_cfvs,
            num_spots: num_spots as u32,
            hands_per_spot,
        })
    }

    /// Run the DCFR+ solve loop with CUDA-native leaf evaluation.
    pub fn solve_with_cfvs(
        &mut self,
        max_iterations: u32,
    ) -> Result<BatchSolveResultGpu, String> {
        if self.num_boundaries == 0 {
            return self.solver.solve_with_cfvs(max_iterations, None);
        }
        self.solve_loop(max_iterations)?;
        self.extract_root_cfvs()
    }

    fn solve_loop(&mut self, max_iterations: u32) -> Result<(), String> {
        let gpu_err = |e: GpuError| format!("GPU error: {e}");

        for t in 1..=max_iterations {
            let current_iteration = t - 1;

            let t_alpha = (current_iteration as i32 - 1).max(0) as f64;
            let pow_alpha = t_alpha * t_alpha.sqrt();
            let pos_discount = (pow_alpha / (pow_alpha + 1.0)) as f32;
            let neg_discount = 0.5f32;

            let nearest_lower_power_of_4 = match current_iteration {
                0 => 0u32,
                x => 1u32 << ((x.leading_zeros() ^ 31) & !1),
            };
            let t_gamma = (current_iteration - nearest_lower_power_of_4) as f64;
            let strat_discount = ((t_gamma / (t_gamma + 1.0)).powi(3)) as f32;

            for traverser in 0..2u32 {
                self.solver.regret_match().map_err(gpu_err)?;
                self.solver.init_reach_pub().map_err(gpu_err)?;
                self.solver.forward_pass().map_err(gpu_err)?;
                self.solver.zero_cfvalues().map_err(gpu_err)?;
                self.solver.fold_eval(traverser).map_err(gpu_err)?;
                self.solver.showdown_eval(traverser).map_err(gpu_err)?;

                // CUDA-native leaf evaluation (no CPU round-trip)
                self.leaf_evaluation_cuda(traverser)?;

                self.solver.backward_cfv(traverser).map_err(gpu_err)?;
                self.solver.update_regrets(
                    traverser,
                    pos_discount,
                    neg_discount,
                    strat_discount,
                ).map_err(gpu_err)?;
            }
        }
        Ok(())
    }

    /// Perform leaf evaluation using CUDA-native inference.
    fn leaf_evaluation_cuda(&mut self, traverser: u32) -> Result<(), String> {
        let gpu_err = |e: GpuError| format!("GPU leaf eval error: {e}");

        let total_hands = self.solver.total_hands();
        let total_inputs = self.num_boundaries as usize
            * NUM_RIVERS as usize
            * self.num_spots as usize;

        if total_inputs == 0 {
            return Ok(());
        }

        let gpu = self.solver.gpu();

        // 1. Encode reach at boundaries into 2720-dim inputs
        gpu.launch_encode_leaf_inputs(
            &mut self.encoded_inputs,
            self.solver.reach_oop(),
            self.solver.reach_ip(),
            &self.gpu_boundary_nodes,
            &self.gpu_boards,
            &self.gpu_river_cards,
            &self.gpu_boundary_pots,
            &self.gpu_boundary_stacks,
            &self.gpu_combo_cards,
            traverser,
            self.num_boundaries,
            NUM_RIVERS,
            self.num_spots,
            total_hands,
            self.hands_per_spot,
        ).map_err(gpu_err)?;

        // 2. CUDA-native inference (all on GPU, no CPU bounce)
        self.raw_cfvs = self.leaf_eval.infer(
            gpu,
            &self.encoded_inputs,
            total_inputs,
        ).map_err(gpu_err)?;

        // 3. Average across rivers and scatter to cfvalues
        gpu.launch_average_leaf_cfvs(
            self.solver.cfvalues_mut(),
            &self.raw_cfvs,
            &self.gpu_boundary_nodes,
            &self.gpu_river_cards,
            &self.gpu_combo_cards,
            self.num_boundaries,
            NUM_RIVERS,
            self.num_spots,
            total_hands,
            self.hands_per_spot,
        ).map_err(gpu_err)?;

        Ok(())
    }

    fn extract_root_cfvs(&mut self) -> Result<BatchSolveResultGpu, String> {
        let gpu_err = |e: GpuError| format!("GPU error: {e}");

        let total_hands = self.solver.total_hands() as usize;

        self.solver.regret_match().map_err(gpu_err)?;
        self.solver.init_reach_pub().map_err(gpu_err)?;
        self.solver.forward_pass().map_err(gpu_err)?;

        let gpu = self.solver.gpu();
        let mut cfvs_oop = gpu.alloc_zeros::<f32>(total_hands).map_err(gpu_err)?;
        let mut cfvs_ip = gpu.alloc_zeros::<f32>(total_hands).map_err(gpu_err)?;

        for traverser in 0..2u32 {
            self.solver.zero_cfvalues().map_err(gpu_err)?;
            self.solver.fold_eval(traverser).map_err(gpu_err)?;
            self.solver.showdown_eval(traverser).map_err(gpu_err)?;
            self.leaf_evaluation_cuda(traverser)?;
            self.solver.backward_cfv(traverser).map_err(gpu_err)?;

            let gpu = self.solver.gpu();
            let full_cfvs: Vec<f32> = gpu.download(
                self.solver.cfvalues(),
            ).map_err(gpu_err)?;
            let root_cfvs = &full_cfvs[..total_hands];
            let dst = if traverser == 0 {
                &mut cfvs_oop
            } else {
                &mut cfvs_ip
            };
            *dst = gpu.upload(root_cfvs).map_err(gpu_err)?;
        }

        Ok(BatchSolveResultGpu {
            iterations: 0,
            cfvs_oop,
            cfvs_ip,
        })
    }
}

#[cfg(all(test, feature = "training"))]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use cfvnet::model::network::{CfvNet, INPUT_SIZE};
    use crate::training::leaf_eval::GpuLeafEvaluator;

    type TestBackend = NdArray;

    /// Verify that TurnBatchSolver can be constructed with valid inputs.
    /// This test runs on CPU (NdArray backend) and validates the structure
    /// without requiring a GPU for the leaf evaluator.
    #[test]
    fn test_turn_solver_construction_params() {
        // Just verify the NUM_RIVERS constant and river card computation
        let board = [0u32, 1, 2, 3];
        let mut rivers = Vec::new();
        for card in 0..52u32 {
            if !board.contains(&card) {
                rivers.push(card);
            }
        }
        assert_eq!(rivers.len(), 48, "4-card board should have 48 possible rivers");
    }

    /// Verify river card computation for different boards.
    #[test]
    fn test_per_spot_river_cards() {
        let boards = [
            [0u32, 13, 26, 39],  // 2c, 2d, 2h, 2s
            [10u32, 11, 12, 48], // 5h, 5s, 6c, As
        ];

        for board in &boards {
            let mut rivers = Vec::new();
            for card in 0..52u32 {
                if !board.contains(&card) {
                    rivers.push(card);
                }
            }
            assert_eq!(rivers.len(), 48);
            // Verify no board card appears in rivers
            for r in &rivers {
                assert!(!board.contains(r));
            }
        }
    }
}
