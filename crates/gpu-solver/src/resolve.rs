//! GPU resolver for interactive single-position solving.
//!
//! Provides `GpuModelStack` which loads a set of trained CFV networks
//! (river, turn, flop, preflop) and resolver methods that build game
//! trees, dispatch to the appropriate solver, and extract strategies at
//! the root for the Explorer UI.
//!
//! Supports progressive resolving via `resolve_progressive()` which
//! yields intermediate strategy snapshots at configurable iteration
//! checkpoints.

#[cfg(feature = "training")]
use std::path::Path;

#[cfg(feature = "training")]
use crate::batch::BatchGpuSolver;
#[cfg(feature = "training")]
use crate::gpu::{GpuContext, GpuError};
#[cfg(feature = "training")]
use crate::solver::GpuSolver;
#[cfg(feature = "training")]
use crate::training::cuda_net::GpuLeafEvaluatorCuda;
#[cfg(feature = "training")]
use crate::training::flop_solver::FlopBatchSolverCuda;
#[cfg(feature = "training")]
use crate::training::turn_solver::TurnBatchSolverCuda;
#[cfg(feature = "training")]
use crate::tree::{FlatTree, NodeType};

#[cfg(feature = "training")]
use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
#[cfg(feature = "training")]
use range_solver::bet_size::BetSizeOptions;
#[cfg(feature = "training")]
use range_solver::range::Range;
#[cfg(feature = "training")]
use range_solver::{CardConfig, PostFlopGame};

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Configuration for the neural network architecture of each street model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelStackConfig {
    pub river_layers: usize,
    pub river_hidden: usize,
    pub turn_layers: usize,
    pub turn_hidden: usize,
    pub flop_layers: usize,
    pub flop_hidden: usize,
    pub preflop_layers: usize,
    pub preflop_hidden: usize,
}

impl Default for ModelStackConfig {
    fn default() -> Self {
        Self {
            river_layers: 7,
            river_hidden: 500,
            turn_layers: 7,
            turn_hidden: 500,
            flop_layers: 7,
            flop_hidden: 500,
            preflop_layers: 7,
            preflop_hidden: 500,
        }
    }
}

/// Description of a game state to resolve.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GameState {
    /// Board cards: 0 cards (preflop), 3 (flop), 4 (turn), or 5 (river).
    pub board: Vec<u8>,
    /// OOP range weights (1326 combos).
    pub oop_range: Vec<f32>,
    /// IP range weights (1326 combos).
    pub ip_range: Vec<f32>,
    /// Total pot size at the start of this street.
    pub pot: i32,
    /// Effective stack remaining.
    pub effective_stack: i32,
    /// OOP bet sizes, e.g. "50%,100%,a".
    pub oop_bet_sizes: String,
    /// OOP raise sizes, e.g. "50%,100%".
    pub oop_raise_sizes: String,
    /// IP bet sizes.
    pub ip_bet_sizes: String,
    /// IP raise sizes.
    pub ip_raise_sizes: String,
}

/// Result of resolving a single position.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResolveResult {
    /// Strategy for the acting player: `[num_actions * num_hands]`.
    /// Layout: action-major, i.e. `strategy[a * num_hands + h]` is the
    /// probability of action `a` for hand `h`.
    pub strategy: Vec<f32>,
    /// Human-readable action labels.
    pub action_names: Vec<String>,
    /// Per-hand EVs (empty if not available).
    pub evs: Vec<f32>,
    /// Number of DCFR+ iterations completed.
    pub iterations: u32,
    /// Player who acts at the root: 0 = OOP, 1 = IP.
    pub player: u8,
    /// Number of hands for the acting player.
    pub num_hands: usize,
    /// Number of actions at the root.
    pub num_actions: usize,
}

// ---------------------------------------------------------------------------
// GpuModelStack
// ---------------------------------------------------------------------------

/// A set of four trained CFV networks loaded on the GPU for interactive
/// resolving across all streets.
#[cfg(feature = "training")]
pub struct GpuModelStack {
    gpu: GpuContext,
    river_eval: GpuLeafEvaluatorCuda,
    turn_eval: GpuLeafEvaluatorCuda,
    #[allow(dead_code)]
    flop_eval: GpuLeafEvaluatorCuda,
    #[allow(dead_code)]
    preflop_eval: GpuLeafEvaluatorCuda,
    #[allow(dead_code)]
    config: ModelStackConfig,
}

#[cfg(feature = "training")]
impl GpuModelStack {
    /// Load four models from a directory.
    ///
    /// Expects burn model files at: `river.mpk.gz`, `turn.mpk.gz`,
    /// `flop.mpk.gz`, `preflop.mpk.gz`. Burn's recorder adds the
    /// `.mpk.gz` extension, so the paths we pass omit it.
    pub fn load(dir: &Path, config: &ModelStackConfig) -> Result<Self, String> {
        let gpu = GpuContext::new(0).map_err(|e| format!("GPU init error: {e}"))?;
        let max_batch: usize = 50_000;

        let river_eval = GpuLeafEvaluatorCuda::load(
            &dir.join("river"),
            &gpu,
            config.river_layers,
            config.river_hidden,
            max_batch,
        )?;

        let turn_eval = GpuLeafEvaluatorCuda::load(
            &dir.join("turn"),
            &gpu,
            config.turn_layers,
            config.turn_hidden,
            max_batch,
        )?;

        let flop_eval = GpuLeafEvaluatorCuda::load(
            &dir.join("flop"),
            &gpu,
            config.flop_layers,
            config.flop_hidden,
            max_batch,
        )?;

        let preflop_eval = GpuLeafEvaluatorCuda::load(
            &dir.join("preflop"),
            &gpu,
            config.preflop_layers,
            config.preflop_hidden,
            max_batch,
        )?;

        Ok(Self {
            gpu,
            river_eval,
            turn_eval,
            flop_eval,
            preflop_eval,
            config: config.clone(),
        })
    }

    /// Reference to the GPU context.
    pub fn gpu(&self) -> &GpuContext {
        &self.gpu
    }

    // -----------------------------------------------------------------------
    // Single-position resolving
    // -----------------------------------------------------------------------

    /// Resolve a single position, dispatching by board length.
    pub fn resolve(
        &mut self,
        state: &GameState,
        max_iterations: u32,
    ) -> Result<ResolveResult, String> {
        match state.board.len() {
            0 => Err(
                "Preflop resolving is not yet supported (BoardState::Preflop \
                 does not exist in range-solver)"
                    .to_string(),
            ),
            3 => self.resolve_flop(state, max_iterations),
            4 => self.resolve_turn(state, max_iterations),
            5 => self.resolve_river(state, max_iterations),
            n => Err(format!("Invalid board size: {n} (expected 0, 3, 4, or 5)")),
        }
    }

    /// Resolve with progressive checkpoints.
    ///
    /// Returns a `ResolveResult` at each iteration checkpoint. The solver
    /// state (regrets, strategy_sum) persists between checkpoints so later
    /// results are strictly more converged.
    pub fn resolve_progressive(
        &mut self,
        state: &GameState,
        checkpoints: &[u32],
    ) -> Result<Vec<ResolveResult>, String> {
        match state.board.len() {
            0 => Err("Preflop resolving is not yet supported".to_string()),
            3 => self.resolve_flop_progressive(state, checkpoints),
            4 => self.resolve_turn_progressive(state, checkpoints),
            5 => self.resolve_river_progressive(state, checkpoints),
            n => Err(format!("Invalid board size: {n}")),
        }
    }

    // -----------------------------------------------------------------------
    // River resolving (5 board cards, no leaf model needed)
    // -----------------------------------------------------------------------

    fn resolve_river(
        &self,
        state: &GameState,
        max_iterations: u32,
    ) -> Result<ResolveResult, String> {
        let (_game, flat_tree, action_names, root_player) = build_river_game(state)?;

        let mut solver = GpuSolver::new(&self.gpu, &flat_tree)
            .map_err(|e| format!("GpuSolver init: {e}"))?;

        let result = solver
            .solve(max_iterations, None)
            .map_err(|e| format!("GpuSolver solve: {e}"))?;

        extract_root_strategy(
            &result.strategy,
            &flat_tree,
            &action_names,
            root_player,
            result.iterations,
        )
    }

    fn resolve_river_progressive(
        &self,
        state: &GameState,
        checkpoints: &[u32],
    ) -> Result<Vec<ResolveResult>, String> {
        let (_game, flat_tree, action_names, root_player) = build_river_game(state)?;

        let mut solver = GpuSolver::new(&self.gpu, &flat_tree)
            .map_err(|e| format!("GpuSolver init: {e}"))?;

        let mut results = Vec::with_capacity(checkpoints.len());
        let mut done = 0u32;

        for &target in checkpoints {
            if target <= done {
                continue;
            }
            let additional = target - done;

            solver
                .solve_iterations(additional, done)
                .map_err(|e| format!("GpuSolver solve: {e}"))?;
            done = target;

            let strategy = solver
                .extract_current_strategy()
                .map_err(|e| format!("extract strategy: {e}"))?;

            let r = extract_root_strategy(
                &strategy,
                &flat_tree,
                &action_names,
                root_player,
                done,
            )?;
            results.push(r);
        }

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Turn resolving (4 board cards, river model at leaves)
    // -----------------------------------------------------------------------

    fn resolve_turn(
        &mut self,
        state: &GameState,
        max_iterations: u32,
    ) -> Result<ResolveResult, String> {
        let (_game, flat_tree, action_names, root_player, boards_flat) =
            build_turn_game(state)?;

        let batch_solver = build_single_spot_solver(&self.gpu, &flat_tree)?;

        let mut turn_solver = TurnBatchSolverCuda::new(
            &self.gpu,
            batch_solver,
            &mut self.river_eval,
            &flat_tree.boundary_indices,
            &flat_tree.boundary_pots,
            &flat_tree.boundary_stacks,
            &boards_flat,
            1, // single spot
            flat_tree.num_combinations,
        )?;

        turn_solver.solve_with_cfvs(max_iterations)?;
        let strategy = turn_solver.extract_strategy(&self.gpu)?;

        extract_root_strategy(
            &strategy,
            &flat_tree,
            &action_names,
            root_player,
            max_iterations,
        )
    }

    fn resolve_turn_progressive(
        &mut self,
        state: &GameState,
        checkpoints: &[u32],
    ) -> Result<Vec<ResolveResult>, String> {
        let (_game, flat_tree, action_names, root_player, boards_flat) =
            build_turn_game(state)?;

        let batch_solver = build_single_spot_solver(&self.gpu, &flat_tree)?;

        let mut turn_solver = TurnBatchSolverCuda::new(
            &self.gpu,
            batch_solver,
            &mut self.river_eval,
            &flat_tree.boundary_indices,
            &flat_tree.boundary_pots,
            &flat_tree.boundary_stacks,
            &boards_flat,
            1,
            flat_tree.num_combinations,
        )?;

        let mut results = Vec::with_capacity(checkpoints.len());
        let mut done = 0u32;

        for &target in checkpoints {
            if target <= done {
                continue;
            }
            let additional = target - done;

            turn_solver.solve_iterations(additional, done)?;
            done = target;

            let strategy = turn_solver.extract_strategy(&self.gpu)?;
            let r = extract_root_strategy(
                &strategy,
                &flat_tree,
                &action_names,
                root_player,
                done,
            )?;
            results.push(r);
        }

        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Flop resolving (3 board cards, turn model at leaves)
    // -----------------------------------------------------------------------

    fn resolve_flop(
        &mut self,
        state: &GameState,
        max_iterations: u32,
    ) -> Result<ResolveResult, String> {
        let (_game, flat_tree, action_names, root_player, boards_flat) =
            build_flop_game(state)?;

        let batch_solver = build_single_spot_solver(&self.gpu, &flat_tree)?;

        let mut flop_solver = FlopBatchSolverCuda::new(
            &self.gpu,
            batch_solver,
            &mut self.turn_eval,
            &flat_tree.boundary_indices,
            &flat_tree.boundary_pots,
            &flat_tree.boundary_stacks,
            &boards_flat,
            1,
            flat_tree.num_combinations,
        )?;

        flop_solver.solve_with_cfvs(max_iterations)?;
        let strategy = flop_solver.extract_strategy(&self.gpu)?;

        extract_root_strategy(
            &strategy,
            &flat_tree,
            &action_names,
            root_player,
            max_iterations,
        )
    }

    fn resolve_flop_progressive(
        &mut self,
        state: &GameState,
        checkpoints: &[u32],
    ) -> Result<Vec<ResolveResult>, String> {
        let (_game, flat_tree, action_names, root_player, boards_flat) =
            build_flop_game(state)?;

        let batch_solver = build_single_spot_solver(&self.gpu, &flat_tree)?;

        let mut flop_solver = FlopBatchSolverCuda::new(
            &self.gpu,
            batch_solver,
            &mut self.turn_eval,
            &flat_tree.boundary_indices,
            &flat_tree.boundary_pots,
            &flat_tree.boundary_stacks,
            &boards_flat,
            1,
            flat_tree.num_combinations,
        )?;

        let mut results = Vec::with_capacity(checkpoints.len());
        let mut done = 0u32;

        for &target in checkpoints {
            if target <= done {
                continue;
            }
            let additional = target - done;

            flop_solver.solve_iterations(additional, done)?;
            done = target;

            let strategy = flop_solver.extract_strategy(&self.gpu)?;
            let r = extract_root_strategy(
                &strategy,
                &flat_tree,
                &action_names,
                root_player,
                done,
            )?;
            results.push(r);
        }

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Game building helpers (free functions)
// ---------------------------------------------------------------------------

/// Parse bet size strings into `BetSizeOptions`.
#[cfg(feature = "training")]
fn parse_bet_sizes(state: &GameState) -> Result<(BetSizeOptions, BetSizeOptions), String> {
    let oop = BetSizeOptions::try_from((
        state.oop_bet_sizes.as_str(),
        state.oop_raise_sizes.as_str(),
    ))
    .map_err(|e| format!("OOP bet sizes: {e}"))?;
    let ip = BetSizeOptions::try_from((
        state.ip_bet_sizes.as_str(),
        state.ip_raise_sizes.as_str(),
    ))
    .map_err(|e| format!("IP bet sizes: {e}"))?;
    Ok((oop, ip))
}

/// Build a river PostFlopGame + FlatTree from a GameState.
#[cfg(feature = "training")]
fn build_river_game(
    state: &GameState,
) -> Result<(PostFlopGame, FlatTree, Vec<String>, u8), String> {
    let oop_range = Range::from_raw_data(&state.oop_range)?;
    let ip_range = Range::from_raw_data(&state.ip_range)?;
    let (oop_sizes, ip_sizes) = parse_bet_sizes(state)?;

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: [state.board[0], state.board[1], state.board[2]],
        turn: state.board[3],
        river: state.board[4],
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: state.pot,
        effective_stack: state.effective_stack,
        river_bet_sizes: [oop_sizes, ip_sizes],
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).map_err(|e| format!("ActionTree: {e}"))?;
    let action_names: Vec<String> = action_tree
        .available_actions()
        .iter()
        .map(|a| format!("{a}"))
        .collect();

    let mut game =
        PostFlopGame::with_config(card_config, action_tree).map_err(|e| format!("{e}"))?;
    game.allocate_memory(false);
    let flat_tree = FlatTree::from_postflop_game(&mut game);

    Ok((game, flat_tree, action_names, 0))
}

/// Build a depth-limited turn PostFlopGame + FlatTree from a GameState.
#[cfg(feature = "training")]
fn build_turn_game(
    state: &GameState,
) -> Result<(PostFlopGame, FlatTree, Vec<String>, u8, Vec<u32>), String> {
    let oop_range = Range::from_raw_data(&state.oop_range)?;
    let ip_range = Range::from_raw_data(&state.ip_range)?;
    let (oop_sizes, ip_sizes) = parse_bet_sizes(state)?;

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: [state.board[0], state.board[1], state.board[2]],
        turn: state.board[3],
        river: range_solver::card::NOT_DEALT,
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: state.pot,
        effective_stack: state.effective_stack,
        turn_bet_sizes: [oop_sizes, ip_sizes],
        depth_limit: Some(0),
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).map_err(|e| format!("ActionTree: {e}"))?;
    let action_names: Vec<String> = action_tree
        .available_actions()
        .iter()
        .map(|a| format!("{a}"))
        .collect();

    let mut game =
        PostFlopGame::with_config(card_config, action_tree).map_err(|e| format!("{e}"))?;
    game.allocate_memory(false);
    let flat_tree = FlatTree::from_postflop_game(&mut game);

    let boards_flat: Vec<u32> = state.board[..4].iter().map(|&c| c as u32).collect();

    Ok((game, flat_tree, action_names, 0, boards_flat))
}

/// Build a depth-limited flop PostFlopGame + FlatTree from a GameState.
#[cfg(feature = "training")]
fn build_flop_game(
    state: &GameState,
) -> Result<(PostFlopGame, FlatTree, Vec<String>, u8, Vec<u32>), String> {
    let oop_range = Range::from_raw_data(&state.oop_range)?;
    let ip_range = Range::from_raw_data(&state.ip_range)?;
    let (oop_sizes, ip_sizes) = parse_bet_sizes(state)?;

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: [state.board[0], state.board[1], state.board[2]],
        turn: range_solver::card::NOT_DEALT,
        river: range_solver::card::NOT_DEALT,
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: state.pot,
        effective_stack: state.effective_stack,
        flop_bet_sizes: [oop_sizes, ip_sizes],
        depth_limit: Some(0),
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).map_err(|e| format!("ActionTree: {e}"))?;
    let action_names: Vec<String> = action_tree
        .available_actions()
        .iter()
        .map(|a| format!("{a}"))
        .collect();

    let mut game =
        PostFlopGame::with_config(card_config, action_tree).map_err(|e| format!("{e}"))?;
    game.allocate_memory(false);
    let flat_tree = FlatTree::from_postflop_game(&mut game);

    let boards_flat: Vec<u32> = state.board[..3].iter().map(|&c| c as u32).collect();

    Ok((game, flat_tree, action_names, 0, boards_flat))
}

// ---------------------------------------------------------------------------
// Solver building helper
// ---------------------------------------------------------------------------

/// Build a `BatchGpuSolver` for a single spot from a FlatTree.
///
/// Uses batch_size=1. The FlatTree provides all structural info; this
/// function uploads per-hand data and terminal payoffs.
#[cfg(feature = "training")]
fn build_single_spot_solver<'a>(
    gpu: &'a GpuContext,
    flat_tree: &FlatTree,
) -> Result<BatchGpuSolver<'a>, String> {
    let gpu_err = |e: GpuError| format!("GPU error: {e}");
    let num_hands = flat_tree.num_hands;

    // Upload hand strengths (padded to num_hands)
    let mut hs_oop = flat_tree.hand_strengths_oop.clone();
    hs_oop.resize(num_hands, 0);
    let mut hs_ip = flat_tree.hand_strengths_ip.clone();
    hs_ip.resize(num_hands, 0);
    let gpu_strengths_oop = gpu.upload(&hs_oop).map_err(gpu_err)?;
    let gpu_strengths_ip = gpu.upload(&hs_ip).map_err(gpu_err)?;

    // Upload initial reach
    let gpu_initial_oop = gpu.upload(&flat_tree.initial_reach_oop).map_err(gpu_err)?;
    let gpu_initial_ip = gpu.upload(&flat_tree.initial_reach_ip).map_err(gpu_err)?;

    // Upload hand cards as flat u32 arrays
    let mut hand_cards_oop_flat = vec![255u32; num_hands * 2];
    for (i, &(c1, c2)) in flat_tree.cards_oop.iter().enumerate() {
        hand_cards_oop_flat[i * 2] = c1 as u32;
        hand_cards_oop_flat[i * 2 + 1] = c2 as u32;
    }
    let mut hand_cards_ip_flat = vec![255u32; num_hands * 2];
    for (i, &(c1, c2)) in flat_tree.cards_ip.iter().enumerate() {
        hand_cards_ip_flat[i * 2] = c1 as u32;
        hand_cards_ip_flat[i * 2 + 1] = c2 as u32;
    }
    let gpu_hand_cards_oop = gpu.upload(&hand_cards_oop_flat).map_err(gpu_err)?;
    let gpu_hand_cards_ip = gpu.upload(&hand_cards_ip_flat).map_err(gpu_err)?;

    // Upload same-hand index
    let gpu_same_oop = gpu.upload(&flat_tree.same_hand_index_oop).map_err(gpu_err)?;
    let gpu_same_ip = gpu.upload(&flat_tree.same_hand_index_ip).map_err(gpu_err)?;

    // Build terminal payoff arrays (per-hand for batch API, single spot)
    let mut fold_nodes = Vec::new();
    let mut fold_win = Vec::new();
    let mut fold_lose = Vec::new();
    let mut fold_pl = Vec::new();

    let mut showdown_nodes = Vec::new();
    let mut showdown_win = Vec::new();
    let mut showdown_lose = Vec::new();

    for (term_idx, &node_id) in flat_tree.terminal_indices.iter().enumerate() {
        match flat_tree.node_types[node_id as usize] {
            NodeType::TerminalFold => {
                fold_nodes.push(node_id);
                let payoff = &flat_tree.fold_payoffs[term_idx];
                // Broadcast scalar payoff to per-hand (single spot)
                for _ in 0..num_hands {
                    fold_win.push(payoff[0]);
                    fold_lose.push(payoff[1]);
                }
                fold_pl.push(payoff[2] as u32);
            }
            NodeType::TerminalShowdown => {
                showdown_nodes.push(node_id);
                let eq_id = flat_tree.showdown_equity_ids[term_idx] as usize;
                let eq = &flat_tree.equity_tables[eq_id];
                for _ in 0..num_hands {
                    showdown_win.push(eq[0]);
                    showdown_lose.push(eq[1]);
                }
            }
            _ => {}
        }
    }

    let num_fold = fold_nodes.len() as u32;
    let num_showdown = showdown_nodes.len() as u32;

    // Ensure non-empty
    if fold_nodes.is_empty() {
        fold_nodes.push(0);
        fold_win.push(0.0);
        fold_lose.push(0.0);
        fold_pl.push(0);
    }
    if showdown_nodes.is_empty() {
        showdown_nodes.push(0);
        showdown_win.push(0.0);
        showdown_lose.push(0.0);
    }

    BatchGpuSolver::from_gpu_data(
        gpu,
        flat_tree,
        &gpu_strengths_oop,
        &gpu_strengths_ip,
        &gpu_initial_oop,
        &gpu_initial_ip,
        &gpu_hand_cards_oop,
        &gpu_hand_cards_ip,
        &gpu_same_oop,
        &gpu_same_ip,
        &fold_nodes,
        &fold_win,
        &fold_lose,
        &fold_pl,
        num_fold,
        &showdown_nodes,
        &showdown_win,
        &showdown_lose,
        num_showdown,
        1,         // num_spots = 1
        num_hands, // hands_per_spot = num_hands
    )
}

// ---------------------------------------------------------------------------
// Strategy extraction helpers
// ---------------------------------------------------------------------------

/// Extract the root node's strategy from the full strategy array.
///
/// The full strategy has layout `[num_infosets * max_actions * num_hands]`.
/// The root is infoset 0. We extract `[num_root_actions * num_hands]`.
#[cfg(feature = "training")]
fn extract_root_strategy(
    full_strategy: &[f32],
    flat_tree: &FlatTree,
    action_names: &[String],
    root_player: u8,
    iterations: u32,
) -> Result<ResolveResult, String> {
    let num_hands = flat_tree.num_hands;
    let num_root_actions = action_names.len();

    if full_strategy.is_empty() {
        return Err("Empty strategy array".to_string());
    }

    // Root is infoset 0 (the first decision node in BFS order)
    let mut strategy = Vec::with_capacity(num_root_actions * num_hands);
    for a in 0..num_root_actions {
        let base = a * num_hands;
        for h in 0..num_hands {
            strategy.push(full_strategy[base + h]);
        }
    }

    Ok(ResolveResult {
        strategy,
        action_names: action_names.to_vec(),
        evs: Vec::new(),
        iterations,
        player: root_player,
        num_hands,
        num_actions: num_root_actions,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "training")]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
    use cfvnet::model::network::{CfvNet, INPUT_SIZE};
    use tempfile::TempDir;

    /// Save a randomly-initialized CfvNet to disk at the given path.
    fn save_random_model(dir: &Path, name: &str, layers: usize, hidden: usize) {
        type B = NdArray;
        let device = Default::default();
        let model = CfvNet::<B>::new(&device, layers, hidden, INPUT_SIZE);
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
        model
            .save_file(dir.join(name), &recorder)
            .expect("Failed to save model");
    }

    fn make_test_config() -> ModelStackConfig {
        ModelStackConfig {
            river_layers: 2,
            river_hidden: 64,
            turn_layers: 2,
            turn_hidden: 64,
            flop_layers: 2,
            flop_hidden: 64,
            preflop_layers: 2,
            preflop_hidden: 64,
        }
    }

    fn save_all_models(dir: &Path) {
        save_random_model(dir, "river", 2, 64);
        save_random_model(dir, "turn", 2, 64);
        save_random_model(dir, "flop", 2, 64);
        save_random_model(dir, "preflop", 2, 64);
    }

    #[test]
    fn test_load_model_stack() {
        let tmp = TempDir::new().unwrap();
        save_all_models(tmp.path());

        let result = GpuModelStack::load(tmp.path(), &make_test_config());
        assert!(
            result.is_ok(),
            "Failed to load model stack: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_resolve_river() {
        let tmp = TempDir::new().unwrap();
        save_all_models(tmp.path());
        let mut stack = GpuModelStack::load(tmp.path(), &make_test_config()).unwrap();

        // Qs=10, Jh=35, 2c=0, 8d=19, 3s=3
        let state = GameState {
            board: vec![10, 35, 0, 19, 3],
            oop_range: vec![1.0f32; 1326],
            ip_range: vec![1.0f32; 1326],
            pot: 100,
            effective_stack: 100,
            oop_bet_sizes: "50%,a".to_string(),
            oop_raise_sizes: "".to_string(),
            ip_bet_sizes: "50%,a".to_string(),
            ip_raise_sizes: "".to_string(),
        };

        let result = stack.resolve(&state, 200).unwrap();

        assert!(result.num_actions > 0);
        assert!(result.num_hands > 0);
        assert_eq!(result.strategy.len(), result.num_actions * result.num_hands);

        // Verify strategy sums to ~1.0 per hand
        for h in 0..result.num_hands {
            let mut total = 0.0f32;
            for a in 0..result.num_actions {
                let prob = result.strategy[a * result.num_hands + h];
                assert!(prob >= -1e-5, "Negative probability: {prob}");
                total += prob;
            }
            assert!(
                (total - 1.0).abs() < 1e-3,
                "Strategy sum for hand {h}: {total}"
            );
        }
    }

    #[test]
    fn test_resolve_river_progressive() {
        let tmp = TempDir::new().unwrap();
        save_all_models(tmp.path());
        let mut stack = GpuModelStack::load(tmp.path(), &make_test_config()).unwrap();

        let state = GameState {
            board: vec![10, 35, 0, 19, 3],
            oop_range: vec![1.0f32; 1326],
            ip_range: vec![1.0f32; 1326],
            pot: 100,
            effective_stack: 100,
            oop_bet_sizes: "50%,a".to_string(),
            oop_raise_sizes: "".to_string(),
            ip_bet_sizes: "50%,a".to_string(),
            ip_raise_sizes: "".to_string(),
        };

        let checkpoints = vec![100, 500, 1000];
        let results = stack
            .resolve_progressive(&state, &checkpoints)
            .unwrap();

        assert_eq!(results.len(), 3);

        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.iterations, checkpoints[i]);

            for h in 0..r.num_hands {
                let mut total = 0.0f32;
                for a in 0..r.num_actions {
                    total += r.strategy[a * r.num_hands + h];
                }
                assert!(
                    (total - 1.0).abs() < 1e-3,
                    "Checkpoint {}: hand {h} sum={total}",
                    checkpoints[i]
                );
            }
        }
    }

    #[test]
    fn test_resolve_turn() {
        let tmp = TempDir::new().unwrap();
        save_all_models(tmp.path());
        let mut stack = GpuModelStack::load(tmp.path(), &make_test_config()).unwrap();

        // Turn: 4 board cards
        let state = GameState {
            board: vec![10, 35, 0, 19],
            oop_range: vec![1.0f32; 1326],
            ip_range: vec![1.0f32; 1326],
            pot: 100,
            effective_stack: 100,
            oop_bet_sizes: "50%,a".to_string(),
            oop_raise_sizes: "".to_string(),
            ip_bet_sizes: "50%,a".to_string(),
            ip_raise_sizes: "".to_string(),
        };

        let result = stack.resolve(&state, 100).unwrap();

        assert!(result.num_actions > 0);
        assert!(result.num_hands > 0);
        assert_eq!(result.strategy.len(), result.num_actions * result.num_hands);

        for h in 0..result.num_hands {
            let mut total = 0.0f32;
            for a in 0..result.num_actions {
                total += result.strategy[a * result.num_hands + h];
            }
            assert!(
                (total - 1.0).abs() < 1e-3,
                "Turn strategy sum for hand {h}: {total}"
            );
        }
    }

    #[test]
    fn test_resolve_turn_progressive() {
        let tmp = TempDir::new().unwrap();
        save_all_models(tmp.path());
        let mut stack = GpuModelStack::load(tmp.path(), &make_test_config()).unwrap();

        let state = GameState {
            board: vec![10, 35, 0, 19],
            oop_range: vec![1.0f32; 1326],
            ip_range: vec![1.0f32; 1326],
            pot: 100,
            effective_stack: 100,
            oop_bet_sizes: "50%,a".to_string(),
            oop_raise_sizes: "".to_string(),
            ip_bet_sizes: "50%,a".to_string(),
            ip_raise_sizes: "".to_string(),
        };

        let checkpoints = vec![100, 500, 1000];
        let results = stack.resolve_progressive(&state, &checkpoints).unwrap();

        assert_eq!(results.len(), 3);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.iterations, checkpoints[i]);
        }
    }
}
