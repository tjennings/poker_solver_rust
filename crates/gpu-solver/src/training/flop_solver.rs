//! Flop batch solver with neural leaf evaluation at depth boundaries.
//!
//! Wraps a `BatchGpuSolver` and injects leaf evaluation into the DCFR+
//! solve loop. At each iteration, after fold terminal evaluation, the
//! solver:
//!
//! 1. Encodes reach probabilities at depth boundaries into 2720-dim inputs
//!    for all (boundary x turn_card x spot) combinations
//! 2. Runs a batched forward pass through the trained turn model
//! 3. Averages the 49 turn outputs per combo and scatters to cfvalues
//!
//! The rest of the solve loop (regret match, forward pass, backward CFV,
//! regret update) is identical to the river `BatchGpuSolver`.

#[cfg(feature = "training")]
use crate::batch::{BatchGpuSolver, BatchSolveResultGpu};
#[cfg(feature = "training")]
use crate::gpu::{GpuContext, GpuError};
#[cfg(feature = "training")]
use crate::training::cuda_net::GpuLeafEvaluatorCuda;
#[cfg(feature = "training")]
use cudarc::driver::CudaSlice;

/// Number of possible turn cards per flop board (52 - 3 board cards).
#[cfg(feature = "training")]
const NUM_TURN_CARDS: u32 = 49;

/// Flop batch solver using CUDA-native neural network inference.
///
/// Identical in structure to `TurnBatchSolverCuda` but uses 3-card flop
/// boards and enumerates 49 possible turn cards at leaf boundaries.
#[cfg(feature = "training")]
pub struct FlopBatchSolverCuda<'a> {
    /// The inner batch solver (owns all DCFR+ state).
    solver: BatchGpuSolver<'a>,
    /// CUDA-native turn model for leaf evaluation.
    leaf_eval: &'a mut GpuLeafEvaluatorCuda,
    /// GPU buffer: depth-boundary node IDs `[num_boundaries]`.
    gpu_boundary_nodes: CudaSlice<u32>,
    /// GPU buffer: pot at each boundary `[num_boundaries]`.
    gpu_boundary_pots: CudaSlice<f32>,
    /// GPU buffer: effective stack at each boundary `[num_boundaries]`.
    gpu_boundary_stacks: CudaSlice<f32>,
    /// Number of depth boundaries in the flop tree.
    num_boundaries: u32,
    /// GPU buffer: per-spot flop board cards `[num_spots * 3]`.
    gpu_boards: CudaSlice<u32>,
    /// GPU buffer: per-spot turn cards `[num_spots * NUM_TURN_CARDS]`.
    gpu_turn_cards: CudaSlice<u32>,
    /// GPU buffer: combo card pairs `[1326 * 2]`.
    gpu_combo_cards: CudaSlice<u32>,
    /// Working buffer: encoded leaf inputs `[num_boundaries * 49 * num_spots * 2720]`.
    encoded_inputs: CudaSlice<f32>,
    /// Working buffer: raw CFV predictions `[num_boundaries * 49 * num_spots * 1326]`.
    raw_cfvs: CudaSlice<f32>,
    /// Number of spots in this batch.
    num_spots: u32,
    /// Hands per spot (1326).
    hands_per_spot: u32,
}

#[cfg(feature = "training")]
impl<'a> FlopBatchSolverCuda<'a> {
    /// Create a new CUDA-native flop batch solver.
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
        assert_eq!(boards_flat.len(), num_spots * 3);

        let gpu_boundary_nodes = gpu.upload(boundary_indices).map_err(gpu_err)?;
        let gpu_boundary_pots = gpu.upload(boundary_pots).map_err(gpu_err)?;
        let gpu_boundary_stacks = gpu.upload(boundary_stacks).map_err(gpu_err)?;
        let gpu_boards = gpu.upload(boards_flat).map_err(gpu_err)?;

        // Compute per-spot turn cards on CPU (52 - 3 board cards = 49)
        let mut turn_cards_flat = Vec::with_capacity(num_spots * NUM_TURN_CARDS as usize);
        for spot in 0..num_spots {
            let board = &boards_flat[spot * 3..(spot + 1) * 3];
            let board_set: Vec<u32> = board.to_vec();
            let mut count = 0u32;
            for card in 0..52u32 {
                if !board_set.contains(&card) {
                    turn_cards_flat.push(card);
                    count += 1;
                }
            }
            assert_eq!(
                count, NUM_TURN_CARDS,
                "Expected {} turn cards for spot {}, got {}",
                NUM_TURN_CARDS, spot, count
            );
        }
        let gpu_turn_cards = gpu.upload(&turn_cards_flat).map_err(gpu_err)?;

        let combo_cards_flat = super::hand_eval::build_combo_cards_flat();
        let gpu_combo_cards = gpu.upload(&combo_cards_flat).map_err(gpu_err)?;

        let total_inputs = num_boundaries as usize * NUM_TURN_CARDS as usize * num_spots;
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
            gpu_turn_cards,
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

    /// Perform leaf evaluation using CUDA-native inference with flop-specific encoding.
    fn leaf_evaluation_cuda(&mut self, traverser: u32) -> Result<(), String> {
        let gpu_err = |e: GpuError| format!("GPU leaf eval error: {e}");

        let total_hands = self.solver.total_hands();
        let total_inputs = self.num_boundaries as usize
            * NUM_TURN_CARDS as usize
            * self.num_spots as usize;

        if total_inputs == 0 {
            return Ok(());
        }

        let gpu = self.solver.gpu();

        // 1. Encode reach at boundaries into 2720-dim inputs (flop variant: 3-card boards)
        gpu.launch_encode_leaf_inputs_flop(
            &mut self.encoded_inputs,
            self.solver.reach_oop(),
            self.solver.reach_ip(),
            &self.gpu_boundary_nodes,
            &self.gpu_boards,
            &self.gpu_turn_cards,
            &self.gpu_boundary_pots,
            &self.gpu_boundary_stacks,
            &self.gpu_combo_cards,
            traverser,
            self.num_boundaries,
            NUM_TURN_CARDS,
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

        // 3. Average across turn cards and scatter to cfvalues.
        // Reuses the same average_leaf_cfvs kernel (parameterized by num_rivers/num_next_cards).
        gpu.launch_average_leaf_cfvs(
            self.solver.cfvalues_mut(),
            &self.raw_cfvs,
            &self.gpu_boundary_nodes,
            &self.gpu_turn_cards,
            &self.gpu_combo_cards,
            self.num_boundaries,
            NUM_TURN_CARDS,
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

    /// Verify turn card computation for flop boards.
    #[test]
    fn test_per_spot_turn_cards() {
        let boards = [
            [0u32, 13, 26],     // 2c, 2d, 2h
            [10u32, 11, 12],    // 5h, 5s, 6c
        ];

        for board in &boards {
            let mut turns = Vec::new();
            for card in 0..52u32 {
                if !board.contains(&card) {
                    turns.push(card);
                }
            }
            assert_eq!(turns.len(), 49, "3-card board should have 49 possible turn cards");
            // Verify no board card appears in turn cards
            for t in &turns {
                assert!(!board.contains(t));
            }
        }
    }
}
