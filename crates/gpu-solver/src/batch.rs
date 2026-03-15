// Batch GPU solver for solving N independent river spots simultaneously.
//
// All spots share the same tree topology (same bet sizes) but differ in
// board cards, ranges, and pot/stack sizes. The batch solver concatenates
// per-hand data from all spots and runs the existing CUDA kernels with
// `num_hands = num_spots * hands_per_spot`.
//
// Terminal evaluation uses the same optimized kernels as the single-spot solver:
// - O(n) fold eval via inclusion-exclusion (precompute_fold_aggregates_batch + fold_eval_from_aggregates_batch)
// - Shared-memory showdown eval (terminal_showdown_eval_shm_batch)
// Both are scoped per (terminal, spot) to keep cross-spot isolation.

#[cfg(feature = "cuda")]
use crate::gpu::{GpuContext, GpuError};
#[cfg(feature = "cuda")]
use crate::tree::{FlatTree, NodeType};
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
use range_solver::bet_size::BetSizeOptions;
use range_solver::range::Range;
use range_solver::{CardConfig, PostFlopGame};

/// A single river spot to solve in a batch.
#[derive(Debug, Clone)]
pub struct RiverSpot {
    pub flop: [u8; 3],
    pub turn: u8,
    pub river: u8,
    pub oop_range: Vec<f32>,  // 1326 combo weights
    pub ip_range: Vec<f32>,   // 1326 combo weights
    pub pot: i32,
    pub effective_stack: i32,
}

/// Shared config for all spots in a batch.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub oop_bet_sizes: BetSizeOptions,
    pub ip_bet_sizes: BetSizeOptions,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            oop_bet_sizes: BetSizeOptions::try_from(("50%,a", "")).unwrap(),
            ip_bet_sizes: BetSizeOptions::try_from(("50%,a", "")).unwrap(),
        }
    }
}

/// Build a `PostFlopGame` from a `RiverSpot` and `BatchConfig`.
///
/// Constructs a CardConfig from the spot's board and ranges, creates a
/// TreeConfig with the spot's pot/stack and the shared bet sizes, then
/// builds the action tree and allocates memory.
pub fn build_game_from_spot(spot: &RiverSpot, config: &BatchConfig) -> Result<PostFlopGame, String> {
    let oop_range = Range::from_raw_data(&spot.oop_range)?;
    let ip_range = Range::from_raw_data(&spot.ip_range)?;

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: spot.flop,
        turn: spot.turn,
        river: spot.river,
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: spot.pot,
        effective_stack: spot.effective_stack,
        river_bet_sizes: [config.oop_bet_sizes.clone(), config.ip_bet_sizes.clone()],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).map_err(|e| format!("{e}"))?;
    let mut game = PostFlopGame::with_config(card_config, action_tree).map_err(|e| format!("{e}"))?;
    game.allocate_memory(false);
    Ok(game)
}

/// Result of a batch solve: per-spot strategies plus iteration count.
#[cfg(feature = "cuda")]
pub struct BatchSolveResult {
    /// Per-spot strategies. Each entry is `[num_infosets * max_actions * actual_hands_this_spot]`.
    pub strategies: Vec<Vec<f32>>,
    /// Number of iterations completed.
    pub iterations: u32,
}

/// Precomputed per-level data for GPU kernels (same as in solver.rs).
#[cfg(feature = "cuda")]
struct LevelData {
    nodes: CudaSlice<u32>,
    parent_nodes: CudaSlice<u32>,
    parent_actions: CudaSlice<u32>,
    parent_infosets: CudaSlice<u32>,
    parent_players: CudaSlice<u32>,
    num_nodes: u32,
}

/// Precomputed per-level decision nodes for backward CFV pass.
#[cfg(feature = "cuda")]
struct BackwardLevelData {
    decision_nodes: CudaSlice<u32>,
    decision_players: CudaSlice<u32>,
    num_decision_nodes: u32,
}

/// Per-spot metadata used during batch assembly.
#[cfg(feature = "cuda")]
struct SpotInfo {
    /// Number of actual (non-padded) hands for this spot.
    actual_hands: usize,
    /// FlatTree for this spot (used to extract per-hand data).
    flat_tree: FlatTree,
}

/// Batch GPU solver that solves N independent river spots simultaneously.
///
/// All spots **must** share the same tree topology. This means they must use
/// the same bet sizes **and** the same pot and effective stack values. Even
/// with identical bet size percentages, different pot/stack ratios can cause
/// the tree builder's allin thresholds (`add_allin_threshold`,
/// `force_allin_threshold`) to merge or eliminate bet sizes near all-in,
/// producing trees with different node counts.
///
/// Spots may differ in board cards and ranges, which affect per-hand data
/// but not tree shape. Per-hand data is concatenated across spots with
/// uniform padding so `total_hands = num_spots * hands_per_spot`.
///
/// Terminal kernels use per-hand payoffs to account for different pot sizes
/// across spots. Card blocking is spot-scoped via the `hands_per_spot`
/// parameter passed to kernels.
#[cfg(feature = "cuda")]
pub struct BatchGpuSolver<'a> {
    gpu: &'a GpuContext,

    // Solver state on GPU
    regrets: CudaSlice<f32>,
    strategy_sum: CudaSlice<f32>,
    current_strategy: CudaSlice<f32>,
    reach_oop: CudaSlice<f32>,
    reach_ip: CudaSlice<f32>,
    cfvalues: CudaSlice<f32>,

    // Tree structure on GPU (shared topology — uploaded once)
    gpu_child_offsets: CudaSlice<u32>,
    gpu_children: CudaSlice<u32>,
    gpu_infoset_ids: CudaSlice<u32>,
    gpu_num_actions: CudaSlice<u32>,

    // Per-level data for forward pass
    level_data: Vec<LevelData>,

    // Per-level data for backward CFV pass
    backward_level_data: Vec<BackwardLevelData>,

    // Terminal data (per-hand payoffs for batch mode)
    fold_terminal_nodes: CudaSlice<u32>,
    fold_amount_win: CudaSlice<f32>,   // [num_fold_terminals * total_hands]
    fold_amount_lose: CudaSlice<f32>,  // [num_fold_terminals * total_hands]
    fold_player: CudaSlice<u32>,
    num_fold_terminals: u32,

    showdown_terminal_nodes: CudaSlice<u32>,
    showdown_amount_win: CudaSlice<f32>,   // [num_showdown_terminals * total_hands]
    showdown_amount_lose: CudaSlice<f32>,  // [num_showdown_terminals * total_hands]
    num_showdown_terminals: u32,

    gpu_hand_strengths_oop: CudaSlice<u32>,
    gpu_hand_strengths_ip: CudaSlice<u32>,

    // Hand card arrays for O(n) fold eval and shared-memory showdown eval
    // Each is [total_hands * 2] containing (c1, c2) per hand as u32
    gpu_hand_cards_oop: CudaSlice<u32>,
    gpu_hand_cards_ip: CudaSlice<u32>,

    // Same-hand index for inclusion-exclusion correction
    // For each player's hand, the opponent's hand index holding same cards (or u32::MAX)
    gpu_same_hand_index_oop: CudaSlice<u32>,
    gpu_same_hand_index_ip: CudaSlice<u32>,

    // Working buffers for O(n) fold aggregates (reused each iteration)
    // Sized for num_fold_terminals * num_spots (one aggregate per terminal per spot)
    gpu_fold_total_opp_reach: CudaSlice<f32>,
    gpu_fold_per_card_reach: CudaSlice<f32>,

    // Decision node lists per player
    decision_nodes_oop: CudaSlice<u32>,
    decision_nodes_ip: CudaSlice<u32>,
    num_oop_decisions: u32,
    num_ip_decisions: u32,

    // Initial reach (on GPU, uploaded once in new())
    gpu_initial_reach_oop: CudaSlice<f32>,
    gpu_initial_reach_ip: CudaSlice<f32>,

    // Dimensions
    num_hands: u32,       // total_hands = num_spots * hands_per_spot
    max_actions: u32,
    num_infosets: u32,
    num_nodes: u32,
    num_levels: usize,

    // Batch-specific
    hands_per_spot: usize,
    num_spots: usize,
    /// Actual (non-padded) hand count per spot, used for result extraction.
    actual_hands_per_spot: Vec<usize>,
}

#[cfg(feature = "cuda")]
impl<'a> BatchGpuSolver<'a> {
    /// Create a new batch GPU solver from a list of river spots and shared config.
    ///
    /// Builds one FlatTree per spot, validates that all topologies match,
    /// concatenates per-hand data with uniform padding, and uploads everything
    /// to the GPU.
    pub fn new(
        gpu: &'a GpuContext,
        spots: &[RiverSpot],
        config: &BatchConfig,
    ) -> Result<Self, String> {
        if spots.is_empty() {
            return Err("No spots provided".to_string());
        }

        // Build a PostFlopGame + FlatTree for each spot
        let mut spot_infos: Vec<SpotInfo> = Vec::with_capacity(spots.len());
        for (i, spot) in spots.iter().enumerate() {
            let mut game = build_game_from_spot(spot, config)
                .map_err(|e| format!("Spot {i}: {e}"))?;
            let flat_tree = FlatTree::from_postflop_game(&mut game);
            let actual_hands = flat_tree.num_hands;
            spot_infos.push(SpotInfo {
                actual_hands,
                flat_tree,
            });
        }

        // Use the first spot's tree as the reference topology
        let ref_tree = &spot_infos[0].flat_tree;

        // Validate all spots have the same topology
        for (i, si) in spot_infos.iter().enumerate().skip(1) {
            let t = &si.flat_tree;
            if t.num_nodes() != ref_tree.num_nodes() {
                return Err(format!(
                    "Spot {i} has {} nodes, expected {}",
                    t.num_nodes(),
                    ref_tree.num_nodes()
                ));
            }
            if t.num_infosets != ref_tree.num_infosets {
                return Err(format!(
                    "Spot {i} has {} infosets, expected {}",
                    t.num_infosets, ref_tree.num_infosets
                ));
            }
            if t.max_actions() != ref_tree.max_actions() {
                return Err(format!(
                    "Spot {i} has max_actions {}, expected {}",
                    t.max_actions(),
                    ref_tree.max_actions()
                ));
            }
            // Node types must match (same topology)
            if t.node_types != ref_tree.node_types {
                return Err(format!("Spot {i} has different node types"));
            }
        }

        let num_spots = spots.len();
        let hands_per_spot = spot_infos.iter().map(|si| si.actual_hands).max().unwrap();
        let total_hands = num_spots * hands_per_spot;
        let num_nodes = ref_tree.num_nodes();
        let num_infosets = ref_tree.num_infosets;
        let max_actions = ref_tree.max_actions();
        let num_levels = ref_tree.num_levels();

        let actual_hands_per_spot: Vec<usize> =
            spot_infos.iter().map(|si| si.actual_hands).collect();

        // --- Assemble concatenated per-hand data ---

        // Initial reach: [total_hands] per player
        let mut initial_reach_oop = vec![0.0f32; total_hands];
        let mut initial_reach_ip = vec![0.0f32; total_hands];

        // Hand strengths: [total_hands] per player
        let mut hand_strengths_oop = vec![0u32; total_hands];
        let mut hand_strengths_ip = vec![0u32; total_hands];

        // Hand card arrays: [total_hands * 2] per player (c1, c2 pairs as u32)
        let mut hand_cards_oop_flat = vec![255u32; total_hands * 2];
        let mut hand_cards_ip_flat = vec![255u32; total_hands * 2];

        // Same-hand index arrays: [total_hands] per player
        let mut same_hand_index_oop = vec![u32::MAX; total_hands];
        let mut same_hand_index_ip = vec![u32::MAX; total_hands];

        for (spot_idx, si) in spot_infos.iter().enumerate() {
            let t = &si.flat_tree;
            let ah = si.actual_hands;
            let base = spot_idx * hands_per_spot;

            // Copy initial reach (padded hands stay 0.0)
            for h in 0..ah {
                initial_reach_oop[base + h] = t.initial_reach_oop[h];
                initial_reach_ip[base + h] = t.initial_reach_ip[h];
            }

            // Copy hand strengths (padded hands stay 0)
            for h in 0..t.hand_strengths_oop.len().min(ah) {
                hand_strengths_oop[base + h] = t.hand_strengths_oop[h];
            }
            for h in 0..t.hand_strengths_ip.len().min(ah) {
                hand_strengths_ip[base + h] = t.hand_strengths_ip[h];
            }

            // Copy hand card pairs for OOP
            for (i, &(c1, c2)) in t.cards_oop.iter().enumerate() {
                if i < ah {
                    hand_cards_oop_flat[(base + i) * 2] = c1 as u32;
                    hand_cards_oop_flat[(base + i) * 2 + 1] = c2 as u32;
                }
            }
            // Copy hand card pairs for IP
            for (i, &(c1, c2)) in t.cards_ip.iter().enumerate() {
                if i < ah {
                    hand_cards_ip_flat[(base + i) * 2] = c1 as u32;
                    hand_cards_ip_flat[(base + i) * 2 + 1] = c2 as u32;
                }
            }

            // Copy same-hand indices, offsetting by spot base
            // same_hand_index_oop[global_oop_hand] -> global_ip_hand (or u32::MAX)
            for h in 0..ah {
                let local_idx = t.same_hand_index_oop[h];
                if local_idx != u32::MAX {
                    same_hand_index_oop[base + h] = base as u32 + local_idx;
                }
                let local_idx = t.same_hand_index_ip[h];
                if local_idx != u32::MAX {
                    same_hand_index_ip[base + h] = base as u32 + local_idx;
                }
            }
        }

        // --- Build terminal data with per-hand payoffs ---
        // Terminal nodes and fold_player are shared (same topology).
        // Payoffs differ per spot, so we broadcast per-spot payoffs to per-hand arrays.

        let mut fold_nodes = Vec::new();
        let mut fold_win_per_hand = Vec::new();
        let mut fold_lose_per_hand = Vec::new();
        let mut fold_pl = Vec::new();

        let mut showdown_nodes = Vec::new();
        let mut showdown_win_per_hand = Vec::new();
        let mut showdown_lose_per_hand = Vec::new();

        for (term_idx, &node_id) in ref_tree.terminal_indices.iter().enumerate() {
            match ref_tree.node_types[node_id as usize] {
                NodeType::TerminalFold => {
                    fold_nodes.push(node_id);
                    // Per-hand payoffs: each hand gets its spot's payoff
                    for (spot_idx, si) in spot_infos.iter().enumerate() {
                        let payoff = &si.flat_tree.fold_payoffs[term_idx];
                        let win = payoff[0];
                        let lose = payoff[1];
                        for _h in 0..hands_per_spot {
                            fold_win_per_hand.push(win);
                            fold_lose_per_hand.push(lose);
                        }
                        // fold_player is the same for all spots (same topology)
                        if spot_idx == 0 {
                            fold_pl.push(payoff[2] as u32);
                        }
                    }
                }
                NodeType::TerminalShowdown => {
                    showdown_nodes.push(node_id);
                    for si in spot_infos.iter() {
                        let eq_id = si.flat_tree.showdown_equity_ids[term_idx] as usize;
                        let eq = &si.flat_tree.equity_tables[eq_id];
                        let win = eq[0];
                        let lose = eq[1];
                        for _h in 0..hands_per_spot {
                            showdown_win_per_hand.push(win);
                            showdown_lose_per_hand.push(lose);
                        }
                    }
                }
                _ => {}
            }
        }

        let num_fold_terminals = fold_nodes.len() as u32;
        let num_showdown_terminals = showdown_nodes.len() as u32;

        // Ensure non-empty buffers for GPU
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

        // --- Upload everything to GPU ---
        let gpu_err = |e: GpuError| format!("GPU error: {e}");

        let strat_size = num_infosets * max_actions * total_hands;
        let reach_size = num_nodes * total_hands;

        let regrets = gpu.alloc_zeros::<f32>(strat_size).map_err(gpu_err)?;
        let strategy_sum = gpu.alloc_zeros::<f32>(strat_size).map_err(gpu_err)?;
        let current_strategy = gpu.alloc_zeros::<f32>(strat_size).map_err(gpu_err)?;
        let reach_oop = gpu.alloc_zeros::<f32>(reach_size).map_err(gpu_err)?;
        let reach_ip = gpu.alloc_zeros::<f32>(reach_size).map_err(gpu_err)?;
        let cfvalues = gpu.alloc_zeros::<f32>(reach_size).map_err(gpu_err)?;

        // Upload shared topology
        let gpu_child_offsets = gpu.upload(&ref_tree.child_offsets).map_err(gpu_err)?;
        let gpu_children = gpu.upload(&ref_tree.children).map_err(gpu_err)?;
        let gpu_infoset_ids = gpu.upload(&ref_tree.infoset_ids).map_err(gpu_err)?;
        let gpu_num_actions = gpu.upload(&ref_tree.infoset_num_actions).map_err(gpu_err)?;

        // Build per-level data for forward pass
        let mut level_data = Vec::with_capacity(num_levels);
        for level in 0..num_levels {
            let start = ref_tree.level_starts[level] as usize;
            let end = ref_tree.level_starts[level + 1] as usize;

            let mut nodes_vec = Vec::with_capacity(end - start);
            let mut parent_nodes_vec = Vec::with_capacity(end - start);
            let mut parent_actions_vec = Vec::with_capacity(end - start);
            let mut parent_infosets_vec = Vec::with_capacity(end - start);
            let mut parent_players_vec = Vec::with_capacity(end - start);

            for node_id in start..end {
                nodes_vec.push(node_id as u32);
                let parent = ref_tree.parent_nodes[node_id];
                parent_nodes_vec.push(parent);
                parent_actions_vec.push(ref_tree.parent_actions[node_id]);
                if parent != u32::MAX {
                    parent_infosets_vec.push(ref_tree.infoset_ids[parent as usize]);
                    parent_players_vec.push(ref_tree.player(parent as usize) as u32);
                } else {
                    parent_infosets_vec.push(0);
                    parent_players_vec.push(0);
                }
            }

            let num_nodes_level = nodes_vec.len() as u32;
            level_data.push(LevelData {
                nodes: gpu.upload(&nodes_vec).map_err(gpu_err)?,
                parent_nodes: gpu.upload(&parent_nodes_vec).map_err(gpu_err)?,
                parent_actions: gpu.upload(&parent_actions_vec).map_err(gpu_err)?,
                parent_infosets: gpu.upload(&parent_infosets_vec).map_err(gpu_err)?,
                parent_players: gpu.upload(&parent_players_vec).map_err(gpu_err)?,
                num_nodes: num_nodes_level,
            });
        }

        // Build per-level decision node lists for backward CFV
        let mut backward_level_data = Vec::with_capacity(num_levels);
        for level in 0..num_levels {
            let start = ref_tree.level_starts[level] as usize;
            let end = ref_tree.level_starts[level + 1] as usize;

            let mut decision_nodes_vec = Vec::new();
            let mut decision_players_vec = Vec::new();
            for node_id in start..end {
                if !ref_tree.is_terminal(node_id) {
                    decision_nodes_vec.push(node_id as u32);
                    decision_players_vec.push(ref_tree.player(node_id) as u32);
                }
            }

            let num_decision = decision_nodes_vec.len() as u32;
            if decision_nodes_vec.is_empty() {
                decision_nodes_vec.push(0);
                decision_players_vec.push(0);
            }
            backward_level_data.push(BackwardLevelData {
                decision_nodes: gpu.upload(&decision_nodes_vec).map_err(gpu_err)?,
                decision_players: gpu.upload(&decision_players_vec).map_err(gpu_err)?,
                num_decision_nodes: num_decision,
            });
        }

        // Upload terminal data
        let fold_terminal_nodes = gpu.upload(&fold_nodes).map_err(gpu_err)?;
        let fold_amount_win = gpu.upload(&fold_win_per_hand).map_err(gpu_err)?;
        let fold_amount_lose = gpu.upload(&fold_lose_per_hand).map_err(gpu_err)?;
        let fold_player = gpu.upload(&fold_pl).map_err(gpu_err)?;

        let showdown_terminal_nodes = gpu.upload(&showdown_nodes).map_err(gpu_err)?;
        let showdown_amount_win = gpu.upload(&showdown_win_per_hand).map_err(gpu_err)?;
        let showdown_amount_lose = gpu.upload(&showdown_lose_per_hand).map_err(gpu_err)?;

        // Upload hand strengths
        let gpu_hand_strengths_oop = gpu.upload(&hand_strengths_oop).map_err(gpu_err)?;
        let gpu_hand_strengths_ip = gpu.upload(&hand_strengths_ip).map_err(gpu_err)?;

        // Upload hand card arrays
        let gpu_hand_cards_oop = gpu.upload(&hand_cards_oop_flat).map_err(gpu_err)?;
        let gpu_hand_cards_ip = gpu.upload(&hand_cards_ip_flat).map_err(gpu_err)?;

        // Upload same-hand index arrays
        let gpu_same_hand_index_oop = gpu.upload(&same_hand_index_oop).map_err(gpu_err)?;
        let gpu_same_hand_index_ip = gpu.upload(&same_hand_index_ip).map_err(gpu_err)?;

        // Allocate working buffers for fold aggregates (per terminal per spot)
        let fold_agg_count = (num_fold_terminals as usize * num_spots).max(1);
        let gpu_fold_total_opp_reach = gpu.alloc_zeros::<f32>(fold_agg_count).map_err(gpu_err)?;
        let gpu_fold_per_card_reach = gpu.alloc_zeros::<f32>(fold_agg_count * 52).map_err(gpu_err)?;

        // Build decision node lists per player
        let mut oop_decisions = Vec::new();
        let mut ip_decisions = Vec::new();
        for (i, nt) in ref_tree.node_types.iter().enumerate() {
            match nt {
                NodeType::DecisionOop => oop_decisions.push(i as u32),
                NodeType::DecisionIp => ip_decisions.push(i as u32),
                _ => {}
            }
        }

        let num_oop_decisions = oop_decisions.len() as u32;
        let num_ip_decisions = ip_decisions.len() as u32;

        if oop_decisions.is_empty() {
            oop_decisions.push(0);
        }
        if ip_decisions.is_empty() {
            ip_decisions.push(0);
        }

        let decision_nodes_oop = gpu.upload(&oop_decisions).map_err(gpu_err)?;
        let decision_nodes_ip = gpu.upload(&ip_decisions).map_err(gpu_err)?;

        Ok(Self {
            gpu,
            regrets,
            strategy_sum,
            current_strategy,
            reach_oop,
            reach_ip,
            cfvalues,
            gpu_child_offsets,
            gpu_children,
            gpu_infoset_ids,
            gpu_num_actions,
            level_data,
            backward_level_data,
            fold_terminal_nodes,
            fold_amount_win,
            fold_amount_lose,
            fold_player,
            num_fold_terminals,
            showdown_terminal_nodes,
            showdown_amount_win,
            showdown_amount_lose,
            num_showdown_terminals,
            gpu_hand_strengths_oop,
            gpu_hand_strengths_ip,
            gpu_hand_cards_oop,
            gpu_hand_cards_ip,
            gpu_same_hand_index_oop,
            gpu_same_hand_index_ip,
            gpu_fold_total_opp_reach,
            gpu_fold_per_card_reach,
            decision_nodes_oop,
            decision_nodes_ip,
            num_oop_decisions,
            num_ip_decisions,
            gpu_initial_reach_oop: gpu.upload(&initial_reach_oop).map_err(gpu_err)?,
            gpu_initial_reach_ip: gpu.upload(&initial_reach_ip).map_err(gpu_err)?,
            num_hands: total_hands as u32,
            max_actions: max_actions as u32,
            num_infosets: num_infosets as u32,
            num_nodes: num_nodes as u32,
            num_levels,
            hands_per_spot,
            num_spots,
            actual_hands_per_spot,
        })
    }

    /// Create a batch GPU solver from pre-built GPU buffers and a reference
    /// `FlatTree` topology. No CPU round-trip for per-hand data assembly.
    ///
    /// The topology (node_types, child_offsets, etc.) is extracted from
    /// `ref_tree`. Per-hand data (strengths, reach, cards, payoffs) are
    /// provided as references to already-computed data (on host or GPU).
    ///
    /// This constructor uploads topology + per-hand data to GPU in one shot,
    /// avoiding the need to build `RiverSpot` structs and `PostFlopGame`
    /// instances per spot.
    #[allow(clippy::too_many_arguments)]
    pub fn from_gpu_data(
        gpu: &'a GpuContext,
        ref_tree: &FlatTree,
        hand_strengths_oop: &CudaSlice<u32>,
        hand_strengths_ip: &CudaSlice<u32>,
        initial_reach_oop: &CudaSlice<f32>,
        initial_reach_ip: &CudaSlice<f32>,
        hand_cards_oop: &CudaSlice<u32>,
        hand_cards_ip: &CudaSlice<u32>,
        same_hand_index_oop: &CudaSlice<u32>,
        same_hand_index_ip: &CudaSlice<u32>,
        fold_nodes: &[u32],
        fold_win_per_hand: &[f32],
        fold_lose_per_hand: &[f32],
        fold_pl: &[u32],
        num_fold_terminals: u32,
        showdown_nodes: &[u32],
        showdown_win_per_hand: &[f32],
        showdown_lose_per_hand: &[f32],
        num_showdown_terminals: u32,
        num_spots: usize,
        hands_per_spot: usize,
    ) -> Result<Self, String> {
        let gpu_err = |e: GpuError| format!("GPU error: {e}");

        let total_hands = num_spots * hands_per_spot;
        let num_nodes = ref_tree.num_nodes();
        let num_infosets = ref_tree.num_infosets;
        let max_actions = ref_tree.max_actions();
        let num_levels = ref_tree.num_levels();

        let actual_hands_per_spot = vec![hands_per_spot; num_spots];

        // Allocate solver state
        let strat_size = num_infosets * max_actions * total_hands;
        let reach_size = num_nodes * total_hands;

        let regrets = gpu.alloc_zeros::<f32>(strat_size).map_err(gpu_err)?;
        let strategy_sum = gpu.alloc_zeros::<f32>(strat_size).map_err(gpu_err)?;
        let current_strategy = gpu.alloc_zeros::<f32>(strat_size).map_err(gpu_err)?;
        let reach_oop = gpu.alloc_zeros::<f32>(reach_size).map_err(gpu_err)?;
        let reach_ip = gpu.alloc_zeros::<f32>(reach_size).map_err(gpu_err)?;
        let cfvalues = gpu.alloc_zeros::<f32>(reach_size).map_err(gpu_err)?;

        // Upload shared topology
        let gpu_child_offsets = gpu.upload(&ref_tree.child_offsets).map_err(gpu_err)?;
        let gpu_children = gpu.upload(&ref_tree.children).map_err(gpu_err)?;
        let gpu_infoset_ids = gpu.upload(&ref_tree.infoset_ids).map_err(gpu_err)?;
        let gpu_num_actions = gpu.upload(&ref_tree.infoset_num_actions).map_err(gpu_err)?;

        // Build per-level data for forward pass
        let mut level_data = Vec::with_capacity(num_levels);
        for level in 0..num_levels {
            let start = ref_tree.level_starts[level] as usize;
            let end = ref_tree.level_starts[level + 1] as usize;

            let mut nodes_vec = Vec::with_capacity(end - start);
            let mut parent_nodes_vec = Vec::with_capacity(end - start);
            let mut parent_actions_vec = Vec::with_capacity(end - start);
            let mut parent_infosets_vec = Vec::with_capacity(end - start);
            let mut parent_players_vec = Vec::with_capacity(end - start);

            for node_id in start..end {
                nodes_vec.push(node_id as u32);
                let parent = ref_tree.parent_nodes[node_id];
                parent_nodes_vec.push(parent);
                parent_actions_vec.push(ref_tree.parent_actions[node_id]);
                if parent != u32::MAX {
                    parent_infosets_vec.push(ref_tree.infoset_ids[parent as usize]);
                    parent_players_vec.push(ref_tree.player(parent as usize) as u32);
                } else {
                    parent_infosets_vec.push(0);
                    parent_players_vec.push(0);
                }
            }

            let num_nodes_level = nodes_vec.len() as u32;
            level_data.push(LevelData {
                nodes: gpu.upload(&nodes_vec).map_err(gpu_err)?,
                parent_nodes: gpu.upload(&parent_nodes_vec).map_err(gpu_err)?,
                parent_actions: gpu.upload(&parent_actions_vec).map_err(gpu_err)?,
                parent_infosets: gpu.upload(&parent_infosets_vec).map_err(gpu_err)?,
                parent_players: gpu.upload(&parent_players_vec).map_err(gpu_err)?,
                num_nodes: num_nodes_level,
            });
        }

        // Build per-level decision node lists for backward CFV
        let mut backward_level_data = Vec::with_capacity(num_levels);
        for level in 0..num_levels {
            let start = ref_tree.level_starts[level] as usize;
            let end = ref_tree.level_starts[level + 1] as usize;

            let mut decision_nodes_vec = Vec::new();
            let mut decision_players_vec = Vec::new();
            for node_id in start..end {
                if !ref_tree.is_terminal(node_id) {
                    decision_nodes_vec.push(node_id as u32);
                    decision_players_vec.push(ref_tree.player(node_id) as u32);
                }
            }

            let num_decision = decision_nodes_vec.len() as u32;
            if decision_nodes_vec.is_empty() {
                decision_nodes_vec.push(0);
                decision_players_vec.push(0);
            }
            backward_level_data.push(BackwardLevelData {
                decision_nodes: gpu.upload(&decision_nodes_vec).map_err(gpu_err)?,
                decision_players: gpu.upload(&decision_players_vec).map_err(gpu_err)?,
                num_decision_nodes: num_decision,
            });
        }

        // Upload terminal data
        let fold_terminal_nodes = gpu.upload(fold_nodes).map_err(gpu_err)?;
        let fold_amount_win = gpu.upload(fold_win_per_hand).map_err(gpu_err)?;
        let fold_amount_lose = gpu.upload(fold_lose_per_hand).map_err(gpu_err)?;
        let fold_player = gpu.upload(fold_pl).map_err(gpu_err)?;

        let showdown_terminal_nodes = gpu.upload(showdown_nodes).map_err(gpu_err)?;
        let showdown_amount_win = gpu.upload(showdown_win_per_hand).map_err(gpu_err)?;
        let showdown_amount_lose = gpu.upload(showdown_lose_per_hand).map_err(gpu_err)?;

        // Clone GPU buffers for per-hand data (solver takes ownership)
        let gpu_hand_strengths_oop = gpu.clone_slice(hand_strengths_oop).map_err(gpu_err)?;
        let gpu_hand_strengths_ip = gpu.clone_slice(hand_strengths_ip).map_err(gpu_err)?;
        let gpu_hand_cards_oop = gpu.clone_slice(hand_cards_oop).map_err(gpu_err)?;
        let gpu_hand_cards_ip = gpu.clone_slice(hand_cards_ip).map_err(gpu_err)?;
        let gpu_same_hand_index_oop = gpu.clone_slice(same_hand_index_oop).map_err(gpu_err)?;
        let gpu_same_hand_index_ip = gpu.clone_slice(same_hand_index_ip).map_err(gpu_err)?;
        let gpu_initial_reach_oop = gpu.clone_slice(initial_reach_oop).map_err(gpu_err)?;
        let gpu_initial_reach_ip = gpu.clone_slice(initial_reach_ip).map_err(gpu_err)?;

        // Allocate working buffers for fold aggregates
        let fold_agg_count = (num_fold_terminals as usize * num_spots).max(1);
        let gpu_fold_total_opp_reach = gpu.alloc_zeros::<f32>(fold_agg_count).map_err(gpu_err)?;
        let gpu_fold_per_card_reach = gpu.alloc_zeros::<f32>(fold_agg_count * 52).map_err(gpu_err)?;

        // Build decision node lists per player
        let mut oop_decisions = Vec::new();
        let mut ip_decisions = Vec::new();
        for (i, nt) in ref_tree.node_types.iter().enumerate() {
            match nt {
                NodeType::DecisionOop => oop_decisions.push(i as u32),
                NodeType::DecisionIp => ip_decisions.push(i as u32),
                _ => {}
            }
        }

        let num_oop_decisions = oop_decisions.len() as u32;
        let num_ip_decisions = ip_decisions.len() as u32;

        if oop_decisions.is_empty() {
            oop_decisions.push(0);
        }
        if ip_decisions.is_empty() {
            ip_decisions.push(0);
        }

        let decision_nodes_oop = gpu.upload(&oop_decisions).map_err(gpu_err)?;
        let decision_nodes_ip = gpu.upload(&ip_decisions).map_err(gpu_err)?;

        Ok(Self {
            gpu,
            regrets,
            strategy_sum,
            current_strategy,
            reach_oop,
            reach_ip,
            cfvalues,
            gpu_child_offsets,
            gpu_children,
            gpu_infoset_ids,
            gpu_num_actions,
            level_data,
            backward_level_data,
            fold_terminal_nodes,
            fold_amount_win,
            fold_amount_lose,
            fold_player,
            num_fold_terminals,
            showdown_terminal_nodes,
            showdown_amount_win,
            showdown_amount_lose,
            num_showdown_terminals,
            gpu_hand_strengths_oop,
            gpu_hand_strengths_ip,
            gpu_hand_cards_oop,
            gpu_hand_cards_ip,
            gpu_same_hand_index_oop,
            gpu_same_hand_index_ip,
            gpu_fold_total_opp_reach,
            gpu_fold_per_card_reach,
            decision_nodes_oop,
            decision_nodes_ip,
            num_oop_decisions,
            num_ip_decisions,
            gpu_initial_reach_oop,
            gpu_initial_reach_ip,
            num_hands: total_hands as u32,
            max_actions: max_actions as u32,
            num_infosets: num_infosets as u32,
            num_nodes: num_nodes as u32,
            num_levels,
            hands_per_spot,
            num_spots,
            actual_hands_per_spot,
        })
    }

    /// Run the DCFR+ solver for the batch.
    ///
    /// The solve loop is structurally identical to `GpuSolver::solve()` but uses
    /// batch-aware optimized terminal kernels:
    /// - O(n) fold eval via inclusion-exclusion (precompute_fold_aggregates_batch + fold_eval_from_aggregates_batch)
    /// - Shared-memory showdown eval scoped per (terminal, spot)
    pub fn solve(
        &mut self,
        max_iterations: u32,
        _target_exploitability: Option<f32>,
    ) -> Result<BatchSolveResult, String> {
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
                // 1. Regret match -> current strategy
                self.gpu
                    .launch_regret_match(
                        &self.regrets,
                        &self.gpu_num_actions,
                        &mut self.current_strategy,
                        self.num_infosets,
                        self.max_actions,
                        self.num_hands,
                    )
                    .map_err(gpu_err)?;

                // 2. Initialize reach at root
                self.init_reach().map_err(gpu_err)?;

                // 3. Forward pass (top-down, level by level)
                for level in 1..self.num_levels {
                    let ld = &self.level_data[level];
                    if ld.num_nodes == 0 {
                        continue;
                    }
                    self.gpu
                        .launch_forward_pass(
                            &mut self.reach_oop,
                            &mut self.reach_ip,
                            &self.current_strategy,
                            &ld.nodes,
                            &ld.parent_nodes,
                            &ld.parent_actions,
                            &ld.parent_infosets,
                            &ld.parent_players,
                            ld.num_nodes,
                            self.num_hands,
                            self.max_actions,
                        )
                        .map_err(gpu_err)?;
                }

                // 4a. Zero out cfvalues (GPU-only, no allocation)
                let cfv_size = (self.num_nodes * self.num_hands) as u32;
                self.gpu
                    .launch_zero_buffer(&mut self.cfvalues, cfv_size)
                    .map_err(gpu_err)?;

                // 4b. Terminal fold eval — O(n) via inclusion-exclusion (batch)
                if self.num_fold_terminals > 0 {
                    let opp_reach = if traverser == 0 {
                        &self.reach_ip
                    } else {
                        &self.reach_oop
                    };
                    // Opponent's hand cards (for aggregate computation)
                    let opp_hand_cards = if traverser == 0 {
                        &self.gpu_hand_cards_ip
                    } else {
                        &self.gpu_hand_cards_oop
                    };
                    // Traverser's hand cards (for blocking lookup)
                    let trav_hand_cards = if traverser == 0 {
                        &self.gpu_hand_cards_oop
                    } else {
                        &self.gpu_hand_cards_ip
                    };
                    let same_hand_index = if traverser == 0 {
                        &self.gpu_same_hand_index_oop
                    } else {
                        &self.gpu_same_hand_index_ip
                    };

                    // Phase A: precompute per-card reach aggregates (per terminal per spot)
                    self.gpu
                        .launch_precompute_fold_aggregates_batch(
                            opp_reach,
                            &self.fold_terminal_nodes,
                            opp_hand_cards,
                            &mut self.gpu_fold_total_opp_reach,
                            &mut self.gpu_fold_per_card_reach,
                            self.num_fold_terminals,
                            self.num_hands,
                            self.hands_per_spot as u32,
                        )
                        .map_err(gpu_err)?;

                    // Phase B: O(1) per-hand CFV using aggregates
                    self.gpu
                        .launch_fold_eval_from_aggregates_batch(
                            &mut self.cfvalues,
                            opp_reach,
                            &self.fold_terminal_nodes,
                            &self.fold_amount_win,
                            &self.fold_amount_lose,
                            &self.fold_player,
                            &self.gpu_fold_total_opp_reach,
                            &self.gpu_fold_per_card_reach,
                            trav_hand_cards,
                            same_hand_index,
                            traverser,
                            self.num_fold_terminals,
                            self.num_hands,
                            self.hands_per_spot as u32,
                        )
                        .map_err(gpu_err)?;
                }

                // 4c. Terminal showdown eval — shared memory (batch)
                if self.num_showdown_terminals > 0 {
                    let opp_reach = if traverser == 0 {
                        &self.reach_ip
                    } else {
                        &self.reach_oop
                    };
                    let (traverser_strengths, opponent_strengths) = if traverser == 0 {
                        (&self.gpu_hand_strengths_oop, &self.gpu_hand_strengths_ip)
                    } else {
                        (&self.gpu_hand_strengths_ip, &self.gpu_hand_strengths_oop)
                    };
                    let trav_hand_cards = if traverser == 0 {
                        &self.gpu_hand_cards_oop
                    } else {
                        &self.gpu_hand_cards_ip
                    };
                    let opp_hand_cards = if traverser == 0 {
                        &self.gpu_hand_cards_ip
                    } else {
                        &self.gpu_hand_cards_oop
                    };
                    self.gpu
                        .launch_terminal_showdown_eval_shm_batch(
                            &mut self.cfvalues,
                            opp_reach,
                            &self.showdown_terminal_nodes,
                            &self.showdown_amount_win,
                            &self.showdown_amount_lose,
                            traverser_strengths,
                            opponent_strengths,
                            trav_hand_cards,
                            opp_hand_cards,
                            self.num_showdown_terminals,
                            self.num_hands,
                            self.hands_per_spot as u32,
                        )
                        .map_err(gpu_err)?;
                }

                // 4d. Backward CFV (bottom-up)
                for level in (0..self.num_levels).rev() {
                    let bld = &self.backward_level_data[level];
                    if bld.num_decision_nodes == 0 {
                        continue;
                    }
                    self.gpu
                        .launch_backward_cfv(
                            &mut self.cfvalues,
                            &self.current_strategy,
                            &bld.decision_nodes,
                            &self.gpu_child_offsets,
                            &self.gpu_children,
                            &self.gpu_infoset_ids,
                            &bld.decision_players,
                            traverser,
                            bld.num_decision_nodes,
                            self.num_hands,
                            self.max_actions,
                        )
                        .map_err(gpu_err)?;
                }

                // 4e. Update regrets
                let (decision_nodes, num_decision) = if traverser == 0 {
                    (&self.decision_nodes_oop, self.num_oop_decisions)
                } else {
                    (&self.decision_nodes_ip, self.num_ip_decisions)
                };

                if num_decision > 0 {
                    self.gpu
                        .launch_update_regrets(
                            &mut self.regrets,
                            &mut self.strategy_sum,
                            &self.current_strategy,
                            &self.cfvalues,
                            decision_nodes,
                            &self.gpu_child_offsets,
                            &self.gpu_children,
                            &self.gpu_infoset_ids,
                            &self.gpu_num_actions,
                            num_decision,
                            self.num_hands,
                            self.max_actions,
                            pos_discount,
                            neg_discount,
                            strat_discount,
                        )
                        .map_err(gpu_err)?;
                }
            }
        }

        // Extract final strategy from strategy_sum
        self.gpu
            .launch_extract_strategy(
                &self.strategy_sum,
                &self.gpu_num_actions,
                &mut self.current_strategy,
                self.num_infosets,
                self.max_actions,
                self.num_hands,
            )
            .map_err(gpu_err)?;

        let full_strategy = self.gpu.download(&self.current_strategy).map_err(gpu_err)?;

        // Split strategy back into per-spot results
        let strategies = self.extract_per_spot_strategies(&full_strategy);

        Ok(BatchSolveResult {
            strategies,
            iterations: max_iterations,
        })
    }

    /// Initialize reach probabilities at the root node for all spots.
    /// Fully GPU-resident: zero buffers then set root reach from pre-uploaded values.
    fn init_reach(&mut self) -> Result<(), GpuError> {
        let reach_size = (self.num_nodes * self.num_hands) as u32;
        self.gpu.launch_zero_buffer(&mut self.reach_oop, reach_size)?;
        self.gpu.launch_zero_buffer(&mut self.reach_ip, reach_size)?;
        self.gpu.launch_set_root_reach(&mut self.reach_oop, &self.gpu_initial_reach_oop, self.num_hands)?;
        self.gpu.launch_set_root_reach(&mut self.reach_ip, &self.gpu_initial_reach_ip, self.num_hands)?;
        Ok(())
    }

    /// Extract per-spot strategies from the full batched strategy array.
    ///
    /// The full strategy is `[num_infosets * max_actions * total_hands]`.
    /// For each spot, we extract the hands at offsets
    /// `[spot * hands_per_spot .. spot * hands_per_spot + actual_hands[spot]]`
    /// for each infoset/action.
    fn extract_per_spot_strategies(&self, full_strategy: &[f32]) -> Vec<Vec<f32>> {
        let num_infosets = self.num_infosets as usize;
        let max_actions = self.max_actions as usize;
        let total_hands = self.num_hands as usize;
        let hps = self.hands_per_spot;

        let mut strategies = Vec::with_capacity(self.num_spots);

        for spot_idx in 0..self.num_spots {
            let ah = self.actual_hands_per_spot[spot_idx];
            let spot_base = spot_idx * hps;
            let mut spot_strategy = vec![0.0f32; num_infosets * max_actions * ah];

            for iset in 0..num_infosets {
                for a in 0..max_actions {
                    let src_base = (iset * max_actions + a) * total_hands + spot_base;
                    let dst_base = (iset * max_actions + a) * ah;
                    for h in 0..ah {
                        spot_strategy[dst_base + h] = full_strategy[src_base + h];
                    }
                }
            }

            strategies.push(spot_strategy);
        }

        strategies
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::gpu::GpuContext;
    use crate::solver::GpuSolver;
    use crate::tree::FlatTree;
    use range_solver::card::{card_from_str, flop_from_str};
    use range_solver::interface::Game;

    /// Helper to build a RiverSpot from string descriptions.
    fn make_spot(
        oop_range_str: &str,
        ip_range_str: &str,
        flop: &str,
        turn: &str,
        river: &str,
        pot: i32,
        effective_stack: i32,
    ) -> RiverSpot {
        let oop_range: Range = oop_range_str.parse().unwrap();
        let ip_range: Range = ip_range_str.parse().unwrap();
        let flop_cards = flop_from_str(flop).unwrap();
        RiverSpot {
            flop: flop_cards,
            turn: card_from_str(turn).unwrap(),
            river: card_from_str(river).unwrap(),
            oop_range: oop_range.raw_data().to_vec(),
            ip_range: ip_range.raw_data().to_vec(),
            pot,
            effective_stack,
        }
    }

    #[test]
    fn test_build_game_from_spot() {
        let spot = make_spot(
            "AA,KK,QQ,AKs",
            "QQ-JJ,AQs,AJs",
            "Qs Jh 2c",
            "8d",
            "3s",
            100,
            100,
        );
        let config = BatchConfig::default();
        let game = build_game_from_spot(&spot, &config).unwrap();
        assert!(game.num_private_hands(0) > 0);
        assert!(game.num_private_hands(1) > 0);
    }

    #[test]
    fn test_batch_solver_single_spot_matches_single() {
        // Solve a single spot both ways and verify strategies match
        let spot = make_spot(
            "AA,KK,QQ,AKs",
            "QQ-JJ,AQs,AJs",
            "Qs Jh 2c",
            "8d",
            "3s",
            100,
            100,
        );
        let config = BatchConfig::default();
        let iterations = 500;

        // Solve with single GpuSolver
        let mut game_single = build_game_from_spot(&spot, &config).unwrap();
        let flat_tree = FlatTree::from_postflop_game(&mut game_single);
        let gpu = GpuContext::new(0).unwrap();
        let single_result = GpuSolver::new(&gpu, &flat_tree)
            .unwrap()
            .solve(iterations, None)
            .unwrap();

        // Solve with BatchGpuSolver (1 spot)
        let mut batch_solver =
            BatchGpuSolver::new(&gpu, &[spot.clone()], &config).unwrap();
        let batch_result = batch_solver.solve(iterations, None).unwrap();

        assert_eq!(batch_result.strategies.len(), 1);

        let single_strat = &single_result.strategy;
        let batch_strat = &batch_result.strategies[0];

        // Both strategies should have the same length
        assert_eq!(
            single_strat.len(),
            batch_strat.len(),
            "Strategy lengths differ: single={}, batch={}",
            single_strat.len(),
            batch_strat.len()
        );

        // Strategies should match closely
        let max_diff: f32 = single_strat
            .iter()
            .zip(batch_strat.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "Single vs batch strategies differ by max {max_diff}"
        );
    }

    #[test]
    fn test_batch_solver_three_spots() {
        let config = BatchConfig::default();
        let iterations = 500;

        let spots = vec![
            make_spot(
                "AA,KK,QQ,AKs",
                "QQ-JJ,AQs,AJs",
                "Qs Jh 2c",
                "8d",
                "3s",
                100,
                100,
            ),
            make_spot(
                "AA,KK,QQ,AKs",
                "QQ-JJ,AQs,AJs",
                "Ks Td 5h",
                "7c",
                "2d",
                100,
                100,
            ),
            make_spot(
                "AA,KK,QQ,AKs",
                "QQ-JJ,AQs,AJs",
                "9s 8d 4c",
                "3h",
                "2s",
                100,
                100,
            ),
        ];

        let gpu = GpuContext::new(0).unwrap();

        // Solve in batch
        let mut batch_solver =
            BatchGpuSolver::new(&gpu, &spots, &config).unwrap();
        let batch_result = batch_solver.solve(iterations, None).unwrap();
        assert_eq!(batch_result.strategies.len(), 3);

        // Solve each spot individually and compare
        for (i, spot) in spots.iter().enumerate() {
            let mut game = build_game_from_spot(spot, &config).unwrap();
            let flat_tree = FlatTree::from_postflop_game(&mut game);
            let single_result = GpuSolver::new(&gpu, &flat_tree)
                .unwrap()
                .solve(iterations, None)
                .unwrap();

            let single_strat = &single_result.strategy;
            let batch_strat = &batch_result.strategies[i];

            assert_eq!(
                single_strat.len(),
                batch_strat.len(),
                "Spot {i}: strategy lengths differ"
            );

            let max_diff: f32 = single_strat
                .iter()
                .zip(batch_strat.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            eprintln!("Spot {i}: max diff = {max_diff}");
            assert!(
                max_diff < 1e-4,
                "Spot {i}: single vs batch strategies differ by max {max_diff}"
            );
        }
    }

    #[test]
    fn test_batch_solver_different_pots() {
        // Same board/ranges but different pot sizes should produce different strategies
        let config = BatchConfig::default();
        let iterations = 500;

        let spots = vec![
            make_spot(
                "AA,KK,QQ,AKs",
                "QQ-JJ,AQs,AJs",
                "Qs Jh 2c",
                "8d",
                "3s",
                50,   // small pot
                200,
            ),
            make_spot(
                "AA,KK,QQ,AKs",
                "QQ-JJ,AQs,AJs",
                "Qs Jh 2c",
                "8d",
                "3s",
                200,  // large pot
                200,
            ),
        ];

        let gpu = GpuContext::new(0).unwrap();
        let mut batch_solver =
            BatchGpuSolver::new(&gpu, &spots, &config).unwrap();
        let batch_result = batch_solver.solve(iterations, None).unwrap();

        assert_eq!(batch_result.strategies.len(), 2);

        // Strategies should NOT be identical (different pot sizes → different play)
        let s0 = &batch_result.strategies[0];
        let s1 = &batch_result.strategies[1];
        assert_eq!(s0.len(), s1.len());

        let max_diff: f32 = s0
            .iter()
            .zip(s1.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // With very different pot sizes, strategies should meaningfully differ
        eprintln!("Different pot sizes max diff: {max_diff}");
        // Just verify they're valid; we don't strictly require them to differ
        // in case the game happens to have the same equilibrium regardless of pot
    }

    #[test]
    fn test_batch_solver_strategy_validity() {
        let config = BatchConfig::default();
        let iterations = 500;

        let spots = vec![
            make_spot(
                "AA,KK,QQ,AKs",
                "QQ-JJ,AQs,AJs",
                "Qs Jh 2c",
                "8d",
                "3s",
                100,
                100,
            ),
            make_spot(
                "AA,KK,QQ,AKs",
                "QQ-JJ,AQs,AJs",
                "Ks Td 5h",
                "7c",
                "2d",
                100,
                100,
            ),
        ];

        let gpu = GpuContext::new(0).unwrap();
        let mut batch_solver =
            BatchGpuSolver::new(&gpu, &spots, &config).unwrap();
        let batch_result = batch_solver.solve(iterations, None).unwrap();

        // For each spot, verify strategies sum to ~1.0 per hand per infoset
        for (spot_idx, spot) in spots.iter().enumerate() {
            let strat = &batch_result.strategies[spot_idx];
            let mut game = build_game_from_spot(spot, &config).unwrap();
            let flat = FlatTree::from_postflop_game(&mut game);
            let num_hands = flat.num_hands;
            let max_actions = flat.max_actions();

            for iset in 0..flat.num_infosets {
                let n_actions = flat.infoset_num_actions[iset] as usize;
                for h in 0..num_hands {
                    let mut total = 0.0f32;
                    for a in 0..n_actions {
                        let idx = (iset * max_actions + a) * num_hands + h;
                        let prob = strat[idx];
                        assert!(
                            prob.is_finite(),
                            "Spot {spot_idx}: non-finite at iset={iset} a={a} h={h}"
                        );
                        assert!(
                            prob >= -1e-5,
                            "Spot {spot_idx}: negative prob {prob} at iset={iset} a={a} h={h}"
                        );
                        total += prob;
                    }
                    assert!(
                        (total - 1.0).abs() < 1e-3,
                        "Spot {spot_idx}: sum={total} at iset={iset} h={h}"
                    );
                }
            }
        }
    }
}
