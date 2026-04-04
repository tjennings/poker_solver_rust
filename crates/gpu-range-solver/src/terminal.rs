//! Terminal evaluation: launch fold_eval and showdown_eval CUDA kernels.

use crate::gpu::{grid_size, CfrKernels, GpuSolverState, BLOCK_SIZE};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// Pre-uploaded GPU data for fold terminal nodes.
pub struct FoldGpuData {
    pub player_card1: CudaSlice<i32>,
    pub player_card2: CudaSlice<i32>,
    pub opp_card1: CudaSlice<i32>,
    pub opp_card2: CudaSlice<i32>,
    pub same_hand_idx: CudaSlice<i32>,
    pub num_player_hands: usize,
    pub num_opp_hands: usize,
}

/// Pre-uploaded GPU data for showdown terminal nodes.
pub struct ShowdownGpuData {
    pub outcome_gpu: CudaSlice<f32>,
    pub num_player_hands: usize,
    pub num_opp_hands: usize,
}

/// Upload fold terminal data for one perspective (player as hero).
pub fn upload_fold_data(
    stream: &Arc<CudaStream>,
    player_cards: &[(u8, u8)],
    opp_cards: &[(u8, u8)],
    same_hand_index: &[u16],
) -> Result<FoldGpuData, Box<dyn std::error::Error>> {
    let pc1: Vec<i32> = player_cards.iter().map(|&(c1, _)| c1 as i32).collect();
    let pc2: Vec<i32> = player_cards.iter().map(|&(_, c2)| c2 as i32).collect();
    let oc1: Vec<i32> = opp_cards.iter().map(|&(c1, _)| c1 as i32).collect();
    let oc2: Vec<i32> = opp_cards.iter().map(|&(_, c2)| c2 as i32).collect();
    let shi: Vec<i32> = same_hand_index
        .iter()
        .map(|&idx| if idx == u16::MAX { -1i32 } else { idx as i32 })
        .collect();

    Ok(FoldGpuData {
        player_card1: stream.clone_htod(&pc1)?,
        player_card2: stream.clone_htod(&pc2)?,
        opp_card1: stream.clone_htod(&oc1)?,
        opp_card2: stream.clone_htod(&oc2)?,
        same_hand_idx: stream.clone_htod(&shi)?,
        num_player_hands: player_cards.len(),
        num_opp_hands: opp_cards.len(),
    })
}

/// Upload showdown outcome matrix for one perspective.
pub fn upload_showdown_data(
    stream: &Arc<CudaStream>,
    outcome_f64: &[f64],
    num_player_hands: usize,
    num_opp_hands: usize,
) -> Result<ShowdownGpuData, Box<dyn std::error::Error>> {
    let outcome_f32: Vec<f32> = outcome_f64.iter().map(|&v| v as f32).collect();
    Ok(ShowdownGpuData {
        outcome_gpu: stream.clone_htod(&outcome_f32)?,
        num_player_hands,
        num_opp_hands,
    })
}

/// Launch fold evaluation kernel for one terminal node.
pub fn launch_fold_eval(
    stream: &Arc<CudaStream>,
    kernels: &CfrKernels,
    state: &mut GpuSolverState,
    node_id: usize,
    payoff: f32,
    fold_data: &FoldGpuData,
) -> Result<(), Box<dyn std::error::Error>> {
    let block = fold_data
        .num_opp_hands
        .max(fold_data.num_player_hands)
        .clamp(64, 1024) as u32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    let node_id_i32 = node_id as i32;
    let num_player_i32 = fold_data.num_player_hands as i32;
    let num_opp_i32 = fold_data.num_opp_hands as i32;
    let h_i32 = state.num_hands as i32;

    unsafe {
        let mut b = stream.launch_builder(&kernels.fold_eval);
        b.arg(&mut state.cfv);
        b.arg(&state.reach);
        b.arg(&node_id_i32);
        b.arg(&payoff);
        b.arg(&fold_data.player_card1);
        b.arg(&fold_data.player_card2);
        b.arg(&fold_data.opp_card1);
        b.arg(&fold_data.opp_card2);
        b.arg(&fold_data.same_hand_idx);
        b.arg(&num_player_i32);
        b.arg(&num_opp_i32);
        b.arg(&h_i32);
        b.launch(cfg)?;
    }

    Ok(())
}

/// Launch showdown evaluation kernel for one terminal node.
pub fn launch_showdown_eval(
    stream: &Arc<CudaStream>,
    kernels: &CfrKernels,
    state: &mut GpuSolverState,
    node_id: usize,
    sd_gpu: &ShowdownGpuData,
    amount_win: f32,
    amount_lose: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = LaunchConfig {
        grid_dim: (grid_size(sd_gpu.num_player_hands, BLOCK_SIZE), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let node_id_i32 = node_id as i32;
    let num_player_i32 = sd_gpu.num_player_hands as i32;
    let num_opp_i32 = sd_gpu.num_opp_hands as i32;
    let h_i32 = state.num_hands as i32;

    unsafe {
        let mut b = stream.launch_builder(&kernels.showdown_eval);
        b.arg(&mut state.cfv);
        b.arg(&state.reach);
        b.arg(&node_id_i32);
        b.arg(&sd_gpu.outcome_gpu);
        b.arg(&amount_win);
        b.arg(&amount_lose);
        b.arg(&num_player_i32);
        b.arg(&num_opp_i32);
        b.arg(&h_i32);
        b.launch(cfg)?;
    }

    Ok(())
}
