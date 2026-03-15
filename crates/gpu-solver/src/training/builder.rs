//! GPU-native per-hand data builder for batch solving.
//!
//! Builds all per-hand arrays directly on the GPU without downloading
//! sampled data to the CPU. The builder takes GPU-resident sampled situation
//! data (boards, ranges, pots, stacks) and precomputed hand strengths,
//! assembles them into the format required by `BatchGpuSolver`, and
//! produces a solver ready to run.
//!
//! The tree TOPOLOGY (node_types, child_offsets, children, parent_nodes,
//! etc.) is shared across all spots and uploaded once. Only per-hand data
//! (reach, strengths, payoffs, card pairs) varies per batch.
//!
//! # GPU-native pipeline
//!
//! 1. Hand strengths: computed on GPU via `GpuHandEvaluator` (Task 4)
//! 2. Initial reach: already on GPU from sampler
//! 3. Hand cards: from combo_cards lookup (static, uploaded once)
//! 4. Same-hand index: identity mapping (same 1326 combos for both players)
//! 5. Payoffs: broadcast from reference tree's terminal payoffs

#[cfg(feature = "cuda")]
use crate::batch::{BatchConfig, BatchGpuSolver, RiverSpot};
#[cfg(feature = "cuda")]
use crate::gpu::{GpuContext, GpuError};
#[cfg(feature = "cuda")]
use crate::training::hand_eval::GpuHandEvaluator;
#[cfg(feature = "cuda")]
use crate::tree::{FlatTree, NodeType};
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

/// Build a [`BatchGpuSolver`] from CPU-side sampled situation data.
///
/// This is the original CPU-round-trip builder, kept for reference and
/// testing. For the GPU-native path, use `GpuBatchBuilder`.
///
/// # Arguments
///
/// * `gpu` -- CUDA context.
/// * `boards` -- Board cards: `[num_situations * 5]` u32 values.
/// * `ranges_oop` -- OOP range weights: `[num_situations * 1326]` f32 values.
/// * `ranges_ip` -- IP range weights: `[num_situations * 1326]` f32 values.
/// * `pots` -- Pot sizes: `[num_situations]` f32 values.
/// * `stacks` -- Effective stacks: `[num_situations]` f32 values.
/// * `num_situations` -- Number of situations in the batch.
/// * `config` -- Shared bet-size configuration for the batch.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn build_batch_from_samples<'a>(
    gpu: &'a GpuContext,
    boards: &[u32],
    ranges_oop: &[f32],
    ranges_ip: &[f32],
    pots: &[f32],
    stacks: &[f32],
    num_situations: usize,
    config: &BatchConfig,
) -> Result<BatchGpuSolver<'a>, String> {
    assert_eq!(boards.len(), num_situations * 5);
    assert_eq!(ranges_oop.len(), num_situations * 1326);
    assert_eq!(ranges_ip.len(), num_situations * 1326);
    assert_eq!(pots.len(), num_situations);
    assert_eq!(stacks.len(), num_situations);

    let spots: Vec<RiverSpot> = (0..num_situations)
        .map(|i| RiverSpot {
            flop: [
                boards[i * 5] as u8,
                boards[i * 5 + 1] as u8,
                boards[i * 5 + 2] as u8,
            ],
            turn: boards[i * 5 + 3] as u8,
            river: boards[i * 5 + 4] as u8,
            oop_range: ranges_oop[i * 1326..(i + 1) * 1326].to_vec(),
            ip_range: ranges_ip[i * 1326..(i + 1) * 1326].to_vec(),
            pot: pots[i] as i32,
            effective_stack: stacks[i] as i32,
        })
        .collect();

    BatchGpuSolver::new(gpu, &spots, config)
}

/// GPU-native batch builder that assembles per-hand solver data directly
/// on the GPU without any CPU round-trip.
///
/// Reuses a shared tree topology (uploaded once) and pre-allocated scratch
/// buffers across batches.
#[cfg(feature = "cuda")]
pub struct GpuBatchBuilder {
    /// Shared tree topology from a reference spot.
    shared_topology: SharedTopology,
    /// GPU hand evaluator (combo_cards + HAND_TABLE on GPU).
    hand_evaluator: GpuHandEvaluator,
    /// Hands per spot (always 1326 for GPU-native path).
    hands_per_spot: usize,
}

/// Shared tree topology data uploaded once and reused for all batches.
///
/// Everything from the reference `FlatTree` that doesn't change per spot:
/// node types, child offsets, children, parent nodes, etc.
#[cfg(feature = "cuda")]
pub struct SharedTopology {
    /// Reference FlatTree for building the solver (CPU-side, needed by `BatchGpuSolver`).
    pub ref_tree: FlatTree,
    /// The config used to build the reference tree.
    pub config: BatchConfig,
    /// Reference pot size used for the topology.
    pub ref_pot: i32,
    /// Reference effective stack used for the topology.
    pub ref_stack: i32,
}

#[cfg(feature = "cuda")]
impl GpuBatchBuilder {
    /// Create a new GPU batch builder from a reference spot configuration.
    ///
    /// The reference spot is used to build the shared tree topology. All
    /// subsequent batches must use the same pot/stack values (since the tree
    /// topology depends on pot/stack via allin thresholds).
    ///
    /// # Arguments
    /// * `gpu` - CUDA context.
    /// * `config` - Bet size configuration shared by all spots.
    /// * `ref_pot` - Reference pot size.
    /// * `ref_stack` - Reference effective stack.
    pub fn new(
        gpu: &GpuContext,
        config: &BatchConfig,
        ref_pot: i32,
        ref_stack: i32,
    ) -> Result<Self, String> {
        // Build a reference spot to get the tree topology
        let ref_spot = RiverSpot {
            flop: [0, 1, 2], // dummy board cards
            turn: 3,
            river: 4,
            oop_range: vec![1.0; 1326],
            ip_range: vec![1.0; 1326],
            pot: ref_pot,
            effective_stack: ref_stack,
        };

        let mut game = crate::batch::build_game_from_spot(&ref_spot, config)
            .map_err(|e| format!("Failed to build reference tree: {e}"))?;
        let ref_tree = FlatTree::from_postflop_game(&mut game);

        let hand_evaluator = GpuHandEvaluator::new(gpu)
            .map_err(|e| format!("Failed to create GPU hand evaluator: {e}"))?;

        let shared_topology = SharedTopology {
            ref_tree,
            config: config.clone(),
            ref_pot,
            ref_stack,
        };

        Ok(Self {
            shared_topology,
            hand_evaluator,
            hands_per_spot: 1326,
        })
    }

    /// Create a new GPU batch builder using a **turn** reference tree
    /// with `depth_limit: Some(0)`.
    ///
    /// The turn tree has fold terminals and depth-boundary leaves
    /// (no showdowns, no chance nodes). The topology depends on bet sizes
    /// and pot/stack via allin thresholds.
    pub fn new_turn(
        gpu: &GpuContext,
        config: &BatchConfig,
        ref_pot: i32,
        ref_stack: i32,
    ) -> Result<Self, String> {
        use range_solver::bet_size::BetSizeOptions;

        // Build a reference turn game with depth_limit=0
        let bet_sizes = BetSizeOptions::try_from(("50%,a", "")).unwrap();
        let mut game = crate::tree::build_turn_game(
            [0, 1, 2], // dummy flop
            3,          // dummy turn
            ref_pot,
            ref_stack,
            &bet_sizes,
        );
        let ref_tree = FlatTree::from_postflop_game(&mut game);

        let hand_evaluator = GpuHandEvaluator::new(gpu)
            .map_err(|e| format!("Failed to create GPU hand evaluator: {e}"))?;

        let shared_topology = SharedTopology {
            ref_tree,
            config: config.clone(),
            ref_pot,
            ref_stack,
        };

        Ok(Self {
            shared_topology,
            hand_evaluator,
            hands_per_spot: 1326,
        })
    }

    /// Create a new GPU batch builder using a **flop** reference tree
    /// with `depth_limit: Some(0)`.
    ///
    /// The flop tree has fold terminals and depth-boundary leaves
    /// (no showdowns, no chance nodes). The topology depends on bet sizes
    /// and pot/stack via allin thresholds.
    pub fn new_flop(
        gpu: &GpuContext,
        config: &BatchConfig,
        ref_pot: i32,
        ref_stack: i32,
    ) -> Result<Self, String> {
        use range_solver::bet_size::BetSizeOptions;

        // Build a reference flop game with depth_limit=0
        let bet_sizes = BetSizeOptions::try_from(("50%,a", "")).unwrap();
        let mut game = crate::tree::build_flop_game(
            [0, 1, 2], // dummy flop
            ref_pot,
            ref_stack,
            &bet_sizes,
        );
        let ref_tree = FlatTree::from_postflop_game(&mut game);

        let hand_evaluator = GpuHandEvaluator::new(gpu)
            .map_err(|e| format!("Failed to create GPU hand evaluator: {e}"))?;

        let shared_topology = SharedTopology {
            ref_tree,
            config: config.clone(),
            ref_pot,
            ref_stack,
        };

        Ok(Self {
            shared_topology,
            hand_evaluator,
            hands_per_spot: 1326,
        })
    }

    /// Reference to the shared topology.
    pub fn topology(&self) -> &SharedTopology {
        &self.shared_topology
    }

    /// Reference to the GPU hand evaluator.
    pub fn hand_evaluator(&self) -> &GpuHandEvaluator {
        &self.hand_evaluator
    }

    /// Hands per spot (always 1326).
    pub fn hands_per_spot(&self) -> usize {
        self.hands_per_spot
    }

    /// Build a `BatchGpuSolver` from GPU-resident data.
    ///
    /// All inputs are already on the GPU. No data is downloaded to the CPU.
    ///
    /// # Arguments
    /// * `gpu` - CUDA context.
    /// * `boards` - GPU buffer `[num_spots * 5]` board card indices.
    /// * `ranges_oop` - GPU buffer `[num_spots * 1326]` OOP range weights.
    /// * `ranges_ip` - GPU buffer `[num_spots * 1326]` IP range weights.
    /// * `num_spots` - Number of spots in the batch.
    ///
    /// # Returns
    /// A `BatchGpuSolver` ready to call `.solve()` on.
    #[allow(clippy::too_many_arguments)]
    pub fn build_from_gpu_data<'a>(
        &self,
        gpu: &'a GpuContext,
        boards: &CudaSlice<u32>,
        ranges_oop: &CudaSlice<f32>,
        ranges_ip: &CudaSlice<f32>,
        num_spots: usize,
    ) -> Result<BatchGpuSolver<'a>, String> {
        if num_spots == 0 {
            return Err("No spots provided".to_string());
        }

        let gpu_err = |e: GpuError| format!("GPU error: {e}");
        let hps = self.hands_per_spot;
        let total_hands = num_spots * hps;

        // 1. Evaluate hand strengths on GPU
        let gpu_strengths = self.hand_evaluator
            .evaluate_on_gpu(gpu, boards, num_spots)
            .map_err(gpu_err)?;

        // For river with same 1326 combos, both players use the same strengths
        // gpu_strengths is [num_spots * 1326] — this is used for both OOP and IP

        // 2. Build hand_cards arrays — same for both players since we use all 1326 combos
        // combo_cards is already on GPU from the evaluator: [1326 * 2]
        // We need [total_hands * 2] where each spot's 1326 hands map to the same combos.
        // Build on GPU: for hand h, cards are combo_cards[(h % 1326) * 2] and combo_cards[(h % 1326) * 2 + 1]
        let hand_cards = self.build_hand_cards_gpu(gpu, total_hands).map_err(gpu_err)?;

        // 3. Build same_hand_index: identity mapping since both players use same 1326 combos
        // For hand h in spot s, same_hand_index[h] = h (same global index)
        let same_hand_index = self.build_same_hand_index_gpu(gpu, total_hands).map_err(gpu_err)?;

        // 4. Build fold/showdown payoffs
        // Since all spots share the same pot/stack, the payoffs are the same for all spots.
        // We broadcast the reference tree's payoffs to all hands.
        let ref_tree = &self.shared_topology.ref_tree;

        let mut fold_nodes = Vec::new();
        let mut fold_payoff_win_ref = Vec::new();
        let mut fold_payoff_lose_ref = Vec::new();
        let mut fold_pl = Vec::new();

        let mut showdown_nodes = Vec::new();
        let mut showdown_payoff_win_ref = Vec::new();
        let mut showdown_payoff_lose_ref = Vec::new();

        for (term_idx, &node_id) in ref_tree.terminal_indices.iter().enumerate() {
            match ref_tree.node_types[node_id as usize] {
                NodeType::TerminalFold => {
                    fold_nodes.push(node_id);
                    let payoff = &ref_tree.fold_payoffs[term_idx];
                    fold_payoff_win_ref.push(payoff[0]);
                    fold_payoff_lose_ref.push(payoff[1]);
                    fold_pl.push(payoff[2] as u32);
                }
                NodeType::TerminalShowdown => {
                    showdown_nodes.push(node_id);
                    let eq_id = ref_tree.showdown_equity_ids[term_idx] as usize;
                    let eq = &ref_tree.equity_tables[eq_id];
                    showdown_payoff_win_ref.push(eq[0]);
                    showdown_payoff_lose_ref.push(eq[1]);
                }
                _ => {}
            }
        }

        let num_fold_terminals = fold_nodes.len();
        let num_showdown_terminals = showdown_nodes.len();

        // Build per-hand payoffs by broadcasting reference payoffs (parallel)
        use rayon::prelude::*;

        let mut fold_win_per_hand = vec![0.0f32; num_fold_terminals * total_hands];
        let mut fold_lose_per_hand = vec![0.0f32; num_fold_terminals * total_hands];
        fold_win_per_hand
            .par_chunks_exact_mut(total_hands)
            .zip(fold_lose_per_hand.par_chunks_exact_mut(total_hands))
            .enumerate()
            .for_each(|(term_idx, (win_chunk, lose_chunk))| {
                win_chunk.fill(fold_payoff_win_ref[term_idx]);
                lose_chunk.fill(fold_payoff_lose_ref[term_idx]);
            });

        let mut showdown_win_per_hand = vec![0.0f32; num_showdown_terminals * total_hands];
        let mut showdown_lose_per_hand = vec![0.0f32; num_showdown_terminals * total_hands];
        showdown_win_per_hand
            .par_chunks_exact_mut(total_hands)
            .zip(showdown_lose_per_hand.par_chunks_exact_mut(total_hands))
            .enumerate()
            .for_each(|(term_idx, (win_chunk, lose_chunk))| {
                win_chunk.fill(showdown_payoff_win_ref[term_idx]);
                lose_chunk.fill(showdown_payoff_lose_ref[term_idx]);
            });

        // Ensure non-empty buffers
        if fold_nodes.is_empty() {
            fold_nodes.push(0);
            fold_win_per_hand.push(0.0);
            fold_lose_per_hand.push(0.0);
            fold_pl.push(0);
        }
        if showdown_nodes.is_empty() {
            showdown_nodes.push(0);
            showdown_win_per_hand.push(0.0);
            showdown_lose_per_hand.push(0.0);
        }

        // Now assemble into BatchGpuSolver using from_gpu_data
        BatchGpuSolver::from_gpu_data(
            gpu,
            ref_tree,
            &gpu_strengths,
            &gpu_strengths,
            ranges_oop,
            ranges_ip,
            &hand_cards,
            &hand_cards,
            &same_hand_index,
            &same_hand_index,
            &fold_nodes,
            &fold_win_per_hand,
            &fold_lose_per_hand,
            &fold_pl,
            num_fold_terminals as u32,
            &showdown_nodes,
            &showdown_win_per_hand,
            &showdown_lose_per_hand,
            num_showdown_terminals as u32,
            num_spots,
            hps,
        )
    }

    /// Build a `BatchGpuSolver` for depth-limited trees (no showdowns).
    ///
    /// Unlike `build_from_gpu_data`, this method does NOT evaluate hand
    /// strengths (which requires 5-card boards). Instead it uses zero
    /// strengths, which is correct because depth-limited trees have no
    /// showdown terminals.
    ///
    /// This is the correct method to use for turn and flop training
    /// where boards have fewer than 5 cards.
    ///
    /// # Arguments
    /// * `gpu` - CUDA context.
    /// * `ranges_oop` - GPU buffer `[num_spots * 1326]` OOP range weights.
    /// * `ranges_ip` - GPU buffer `[num_spots * 1326]` IP range weights.
    /// * `num_spots` - Number of spots in the batch.
    #[allow(clippy::too_many_arguments)]
    pub fn build_depth_limited<'a>(
        &self,
        gpu: &'a GpuContext,
        ranges_oop: &CudaSlice<f32>,
        ranges_ip: &CudaSlice<f32>,
        num_spots: usize,
    ) -> Result<BatchGpuSolver<'a>, String> {
        if num_spots == 0 {
            return Err("No spots provided".to_string());
        }

        let gpu_err = |e: GpuError| format!("GPU error: {e}");
        let hps = self.hands_per_spot;
        let total_hands = num_spots * hps;

        // No hand strength evaluation -- depth-limited trees have no showdowns.
        // Use zero strengths (they won't be read since there are no showdown terminals).
        let gpu_strengths = gpu.alloc_zeros::<u32>(total_hands).map_err(gpu_err)?;

        let hand_cards = self.build_hand_cards_gpu(gpu, total_hands).map_err(gpu_err)?;
        let same_hand_index = self.build_same_hand_index_gpu(gpu, total_hands).map_err(gpu_err)?;

        let ref_tree = &self.shared_topology.ref_tree;

        let mut fold_nodes = Vec::new();
        let mut fold_payoff_win_ref = Vec::new();
        let mut fold_payoff_lose_ref = Vec::new();
        let mut fold_pl = Vec::new();

        // No showdown terminals expected in depth-limited trees
        let showdown_nodes_dummy = vec![0u32];
        let showdown_win_dummy = vec![0.0f32];
        let showdown_lose_dummy = vec![0.0f32];

        for (term_idx, &node_id) in ref_tree.terminal_indices.iter().enumerate() {
            match ref_tree.node_types[node_id as usize] {
                NodeType::TerminalFold => {
                    fold_nodes.push(node_id);
                    let payoff = &ref_tree.fold_payoffs[term_idx];
                    fold_payoff_win_ref.push(payoff[0]);
                    fold_payoff_lose_ref.push(payoff[1]);
                    fold_pl.push(payoff[2] as u32);
                }
                NodeType::TerminalShowdown => {
                    // Should not occur in depth-limited trees
                    panic!("Unexpected showdown terminal in depth-limited tree");
                }
                _ => {}
            }
        }

        let num_fold_terminals = fold_nodes.len();

        use rayon::prelude::*;

        let mut fold_win_per_hand = vec![0.0f32; num_fold_terminals * total_hands];
        let mut fold_lose_per_hand = vec![0.0f32; num_fold_terminals * total_hands];
        fold_win_per_hand
            .par_chunks_exact_mut(total_hands)
            .zip(fold_lose_per_hand.par_chunks_exact_mut(total_hands))
            .enumerate()
            .for_each(|(term_idx, (win_chunk, lose_chunk))| {
                win_chunk.fill(fold_payoff_win_ref[term_idx]);
                lose_chunk.fill(fold_payoff_lose_ref[term_idx]);
            });

        if fold_nodes.is_empty() {
            fold_nodes.push(0);
            fold_win_per_hand.push(0.0);
            fold_lose_per_hand.push(0.0);
            fold_pl.push(0);
        }

        BatchGpuSolver::from_gpu_data(
            gpu,
            ref_tree,
            &gpu_strengths,
            &gpu_strengths,
            ranges_oop,
            ranges_ip,
            &hand_cards,
            &hand_cards,
            &same_hand_index,
            &same_hand_index,
            &fold_nodes,
            &fold_win_per_hand,
            &fold_lose_per_hand,
            &fold_pl,
            num_fold_terminals as u32,
            &showdown_nodes_dummy,
            &showdown_win_dummy,
            &showdown_lose_dummy,
            0, // num_showdown_terminals = 0
            num_spots,
            hps,
        )
    }

    /// Build hand_cards GPU buffer [total_hands * 2] by tiling combo_cards.
    fn build_hand_cards_gpu(
        &self,
        gpu: &GpuContext,
        total_hands: usize,
    ) -> Result<CudaSlice<u32>, GpuError> {
        use rayon::prelude::*;

        // Build on CPU and upload — this is a one-time static lookup.
        // combo_cards_flat is [1326 * 2].
        let combo_cards_flat = super::hand_eval::build_combo_cards_flat();
        let mut hand_cards = vec![0u32; total_hands * 2];
        hand_cards.par_chunks_exact_mut(2).enumerate().for_each(|(h, pair)| {
            let combo = h % 1326;
            pair[0] = combo_cards_flat[combo * 2];
            pair[1] = combo_cards_flat[combo * 2 + 1];
        });
        gpu.upload(&hand_cards)
    }

    /// Build same_hand_index GPU buffer [total_hands] — identity mapping.
    fn build_same_hand_index_gpu(
        &self,
        gpu: &GpuContext,
        total_hands: usize,
    ) -> Result<CudaSlice<u32>, GpuError> {
        // For river with same 1326 combos for both players,
        // same_hand_index[h] = h (within each spot, the same combo index)
        let same_hand: Vec<u32> = (0..total_hands).map(|h| h as u32).collect();
        gpu.upload(&same_hand)
    }
}

#[cfg(test)]
mod tests {
    use crate::batch::RiverSpot;
    use range_solver::card::card_from_str;

    /// Helper: convert card name to u32.
    fn card(s: &str) -> u32 {
        u32::from(card_from_str(s).unwrap())
    }

    /// Build a valid board as `[u32; 5]`.
    fn test_board() -> [u32; 5] {
        [card("2c"), card("5d"), card("8h"), card("Js"), card("Ac")]
    }

    /// Build a uniform range (all combos equal weight, respecting board blocking).
    fn uniform_range(board: &[u32; 5]) -> Vec<f32> {
        let board_set: Vec<u8> = board.iter().map(|&c| c as u8).collect();
        let mut range = vec![0.0f32; 1326];
        for (i, r) in range.iter_mut().enumerate() {
            let (c1, c2) = range_solver::card::index_to_card_pair(i);
            if !board_set.contains(&c1) && !board_set.contains(&c2) {
                *r = 1.0;
            }
        }
        range
    }

    #[test]
    fn test_build_spots_from_samples() {
        let board = test_board();
        let range = uniform_range(&board);

        let num_situations = 3;
        let mut boards = Vec::new();
        let mut ranges_oop = Vec::new();
        let mut ranges_ip = Vec::new();
        let mut pots = Vec::new();
        let mut stacks = Vec::new();

        for _ in 0..num_situations {
            boards.extend_from_slice(&board);
            ranges_oop.extend_from_slice(&range);
            ranges_ip.extend_from_slice(&range);
            pots.push(200.0f32);
            stacks.push(400.0f32);
        }

        // Verify spot construction (doesn't need GPU)
        let spots: Vec<RiverSpot> = (0..num_situations)
            .map(|i| RiverSpot {
                flop: [
                    boards[i * 5] as u8,
                    boards[i * 5 + 1] as u8,
                    boards[i * 5 + 2] as u8,
                ],
                turn: boards[i * 5 + 3] as u8,
                river: boards[i * 5 + 4] as u8,
                oop_range: ranges_oop[i * 1326..(i + 1) * 1326].to_vec(),
                ip_range: ranges_ip[i * 1326..(i + 1) * 1326].to_vec(),
                pot: pots[i] as i32,
                effective_stack: stacks[i] as i32,
            })
            .collect();

        assert_eq!(spots.len(), 3);
        assert_eq!(spots[0].flop, [board[0] as u8, board[1] as u8, board[2] as u8]);
        assert_eq!(spots[0].turn, board[3] as u8);
        assert_eq!(spots[0].river, board[4] as u8);
        assert_eq!(spots[0].pot, 200);
        assert_eq!(spots[0].effective_stack, 400);
        assert_eq!(spots[0].oop_range.len(), 1326);
        assert_eq!(spots[0].ip_range.len(), 1326);

        let as_card = card_from_str("Ac").unwrap();
        let ah_card = card_from_str("Ah").unwrap();
        let blocked_idx = range_solver::card::card_pair_to_index(as_card, ah_card);
        assert_eq!(spots[0].oop_range[blocked_idx], 0.0);

        let c3h = card_from_str("3h").unwrap();
        let c4s = card_from_str("4s").unwrap();
        let unblocked_idx = range_solver::card::card_pair_to_index(c3h, c4s);
        assert_eq!(spots[0].oop_range[unblocked_idx], 1.0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_gpu_batch_builder_creates_solver() {
        use crate::batch::BatchConfig;
        use crate::gpu::GpuContext;
        use super::GpuBatchBuilder;

        let gpu = GpuContext::new(0).expect("CUDA device required");
        let config = BatchConfig::default();

        let builder = GpuBatchBuilder::new(&gpu, &config, 100, 100).unwrap();

        let board = test_board();
        let range = uniform_range(&board);

        let num_spots = 2;
        let mut boards = Vec::new();
        let mut ranges_oop_flat = Vec::new();
        let mut ranges_ip_flat = Vec::new();

        // Use two different boards (both with same pot/stack)
        boards.extend_from_slice(&board);
        boards.extend_from_slice(&[card("3c"), card("6d"), card("9h"), card("Qs"), card("Kc")]);

        let range2 = uniform_range(&[card("3c"), card("6d"), card("9h"), card("Qs"), card("Kc")]);

        ranges_oop_flat.extend_from_slice(&range);
        ranges_oop_flat.extend_from_slice(&range2);
        ranges_ip_flat.extend_from_slice(&range);
        ranges_ip_flat.extend_from_slice(&range2);

        // Upload to GPU
        let gpu_boards = gpu.upload(&boards).unwrap();
        let gpu_ranges_oop = gpu.upload(&ranges_oop_flat).unwrap();
        let gpu_ranges_ip = gpu.upload(&ranges_ip_flat).unwrap();

        let mut solver = builder
            .build_from_gpu_data(&gpu, &gpu_boards, &gpu_ranges_oop, &gpu_ranges_ip, num_spots)
            .unwrap();

        // Verify it can solve
        let result = solver.solve(100, None).unwrap();
        assert_eq!(result.strategies.len(), num_spots);

        // Each strategy should have non-zero entries
        for (i, strat) in result.strategies.iter().enumerate() {
            let non_zero = strat.iter().filter(|&&v| v > 0.0).count();
            assert!(non_zero > 0, "Strategy {i} should have non-zero entries");
        }
    }
}
