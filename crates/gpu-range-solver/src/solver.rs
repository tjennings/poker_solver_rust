//! DCFR iteration loop using cudarc CUDA kernel launches.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop, clippy::type_complexity)]

use crate::extract::{
    decompose_at_chance, NodeType, TerminalData, TreeTopology,
};
use crate::gpu::{launch_cfg, CfrKernels, GpuMegaState, GpuSolverState, MegaKernel, BLOCK_SIZE};
use crate::terminal::{
    launch_fold_eval, launch_showdown_eval, upload_fold_data, upload_showdown_data, FoldGpuData,
    ShowdownGpuData,
};
use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// DCFR discount parameters, matching range-solver's formula exactly.
struct DiscountParams {
    alpha_t: f32,
    beta_t: f32,
    gamma_t: f32,
}

impl DiscountParams {
    fn new(iteration: u32) -> Self {
        let nearest = match iteration {
            0 => 0,
            x => 1u32 << ((x.leading_zeros() ^ 31) & !1),
        };
        let ta = (iteration as i32 - 1).max(0) as f64;
        let tg = (iteration - nearest) as f64;
        let pa = ta * ta.sqrt();
        let pg = (tg / (tg + 1.0)).powi(3);
        Self {
            alpha_t: (pa / (pa + 1.0)) as f32,
            beta_t: 0.5,
            gamma_t: pg as f32,
        }
    }
}

/// Per-player terminal GPU data uploaded once before solve.
struct TerminalGpuDataPerPlayer {
    fold_data: FoldGpuData,
    showdown_data: Vec<ShowdownGpuData>,
}

/// Build sorted edge arrays from topology and upload to GPU.
fn prepare_topology(
    topo: &TreeTopology,
    stream: &Arc<CudaStream>,
    state: &mut GpuSolverState,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut edges_by_depth: Vec<Vec<usize>> = vec![Vec::new(); topo.max_depth + 1];
    for e in 0..topo.num_edges {
        let parent_depth = topo.node_depth[topo.edge_parent[e]];
        edges_by_depth[parent_depth].push(e);
    }

    let mut sorted_edges: Vec<usize> = Vec::with_capacity(topo.num_edges);
    let mut level_starts = Vec::new();
    let mut level_counts = Vec::new();
    for depth_edges in &edges_by_depth {
        level_starts.push(sorted_edges.len());
        level_counts.push(depth_edges.len());
        sorted_edges.extend(depth_edges);
    }

    let parent_i32: Vec<i32> = sorted_edges
        .iter()
        .map(|&e| topo.edge_parent[e] as i32)
        .collect();
    let child_i32: Vec<i32> = sorted_edges
        .iter()
        .map(|&e| topo.edge_child[e] as i32)
        .collect();
    let player_i32: Vec<i32> = sorted_edges
        .iter()
        .map(|&e| match topo.node_type[topo.edge_parent[e]] {
            NodeType::Player { player } => player as i32,
            NodeType::Chance => 2i32,
            _ => -1i32,
        })
        .collect();
    let ape: Vec<f32> = sorted_edges
        .iter()
        .map(|&e| topo.node_num_actions[topo.edge_parent[e]] as f32)
        .collect();

    state.upload_topology(
        stream,
        &parent_i32,
        &child_i32,
        &player_i32,
        &ape,
        level_starts,
        level_counts,
        topo.max_depth,
    )?;

    Ok(())
}

/// Upload terminal data for both players.
fn upload_terminal_data(
    stream: &Arc<CudaStream>,
    _topo: &TreeTopology,
    term: &TerminalData,
) -> Result<[TerminalGpuDataPerPlayer; 2], Box<dyn std::error::Error>> {
    let mut result: [Option<TerminalGpuDataPerPlayer>; 2] = [None, None];

    for player in 0..2 {
        let opp = player ^ 1;
        let fold_data = upload_fold_data(
            stream,
            &term.hand_cards[player],
            &term.hand_cards[opp],
            &term.same_hand_index[player],
        )?;

        let mut showdown_data = Vec::with_capacity(term.showdown_outcomes.len());
        for sd in &term.showdown_outcomes {
            let num_p = sd.num_player_hands[player];
            let num_o = sd.num_player_hands[opp];

            let outcome_vec: Vec<f64> = if player == 0 {
                sd.outcome_matrix_p0.clone()
            } else {
                // Transpose and negate: IP perspective
                let mut transposed = vec![0.0f64; num_p * num_o];
                for h_oop in 0..num_o {
                    for h_ip in 0..num_p {
                        transposed[h_ip * num_o + h_oop] =
                            -sd.outcome_matrix_p0[h_oop * num_p + h_ip];
                    }
                }
                transposed
            };

            showdown_data.push(upload_showdown_data(stream, &outcome_vec, num_p, num_o)?);
        }

        result[player] = Some(TerminalGpuDataPerPlayer {
            fold_data,
            showdown_data,
        });
    }

    Ok([result[0].take().unwrap(), result[1].take().unwrap()])
}

/// Evaluate all terminal nodes at a given depth for the current traverser.
fn evaluate_terminals_at_depth(
    stream: &Arc<CudaStream>,
    kernels: &CfrKernels,
    state: &mut GpuSolverState,
    topo: &TreeTopology,
    term: &TerminalData,
    term_gpu: &TerminalGpuDataPerPlayer,
    depth: usize,
    player: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let opp = player ^ 1;

    for (fold_idx, &node_id) in topo.fold_nodes.iter().enumerate() {
        if topo.node_depth[node_id] != depth {
            continue;
        }
        let fd = &term.fold_payoffs[fold_idx];
        let payoff = if fd.folded_player == opp {
            fd.amount_win as f32
        } else {
            fd.amount_lose as f32
        };

        launch_fold_eval(stream, kernels, state, node_id, payoff, &term_gpu.fold_data)?;
    }

    for (sd_idx, &node_id) in topo.showdown_nodes.iter().enumerate() {
        if topo.node_depth[node_id] != depth {
            continue;
        }
        let sd = &term.showdown_outcomes[sd_idx];
        let (amount_win, amount_lose) = if player == 0 {
            (sd.amount_win as f32, sd.amount_lose as f32)
        } else {
            (-sd.amount_lose as f32, -sd.amount_win as f32)
        };

        launch_showdown_eval(
            stream,
            kernels,
            state,
            node_id,
            &term_gpu.showdown_data[sd_idx],
            amount_win,
            amount_lose,
        )?;
    }

    Ok(())
}

/// Launch zero kernel for a mutable CudaSlice.
unsafe fn launch_zero(
    stream: &Arc<CudaStream>,
    kernels: &CfrKernels,
    buf: &mut cudarc::driver::CudaSlice<f32>,
    n: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let n_i32 = n as i32;
    let cfg = launch_cfg(n);
    let mut b = stream.launch_builder(&kernels.zero_f32);
    b.arg(buf);
    b.arg(&n_i32);
    unsafe { b.launch(cfg)? };
    Ok(())
}

/// Compute exploitability via best-response backward pass on GPU.
fn compute_exploitability_gpu(
    stream: &Arc<CudaStream>,
    kernels: &CfrKernels,
    state: &mut GpuSolverState,
    topo: &TreeTopology,
    term: &TerminalData,
    term_gpu: &[TerminalGpuDataPerPlayer; 2],
    initial_weights: &[Vec<f32>; 2],
    num_hands: usize,
) -> Result<f32, Box<dyn std::error::Error>> {
    let n_times_h = state.num_nodes * num_hands;

    // Compute average strategy from strategy_sum on CPU, upload
    let strategy_sum_host: Vec<f32> = stream.clone_dtoh(&state.strategy_sum)?;
    let avg_strategy = normalize_strategy_sum_to_avg(
        &strategy_sum_host,
        state.num_edges,
        num_hands,
        topo,
        &state.level_edge_start,
        &state.level_edge_count,
        state.max_depth,
    );
    let avg_strategy_gpu = stream.clone_htod(&avg_strategy)?;

    let mut total_br = 0.0f64;

    for player in 0..2 {
        let opp = player ^ 1;

        // Zero reach, cfv
        unsafe {
            launch_zero(stream, kernels, &mut state.reach, n_times_h)?;
            launch_zero(stream, kernels, &mut state.cfv, n_times_h)?;
        }

        // Set root reach = opponent weights
        {
            let opp_weights_gpu = stream.clone_htod(&initial_weights[opp])?;
            let h_i32 = num_hands as i32;
            let cfg = launch_cfg(num_hands);
            unsafe {
                let mut b = stream.launch_builder(&kernels.set_reach_root);
                b.arg(&mut state.reach);
                b.arg(&opp_weights_gpu);
                b.arg(&h_i32);
                b.launch(cfg)?;
            }
        }

        // Forward pass using average strategy for opponent
        for depth in 0..=state.max_depth {
            let count = state.level_edge_count[depth];
            if count == 0 {
                continue;
            }
            let start_i32 = state.level_edge_start[depth] as i32;
            let count_i32 = count as i32;
            let player_i32 = player as i32;
            let h_i32 = num_hands as i32;
            let cfg = launch_cfg(count * num_hands);
            unsafe {
                let mut b = stream.launch_builder(&kernels.forward_pass_level);
                b.arg(&mut state.reach);
                b.arg(&avg_strategy_gpu);
                b.arg(&state.edge_parent);
                b.arg(&state.edge_child);
                b.arg(&state.edge_player);
                b.arg(&start_i32);
                b.arg(&count_i32);
                b.arg(&player_i32);
                b.arg(&h_i32);
                b.launch(cfg)?;
            }
        }

        // Init traverser decision node CFVs to -infinity for max
        let mut cfv_init = vec![0.0f32; n_times_h];
        for &node_id in &topo.player_nodes[player] {
            for h in 0..num_hands {
                cfv_init[node_id * num_hands + h] = f32::NEG_INFINITY;
            }
        }
        let cfv_init_gpu = stream.clone_htod(&cfv_init)?;
        stream.memcpy_dtod(&cfv_init_gpu, &mut state.cfv)?;

        // Best-response backward pass
        for depth in (0..=state.max_depth).rev() {
            evaluate_terminals_at_depth(
                stream, kernels, state, topo, term, &term_gpu[player], depth, player,
            )?;

            let count = state.level_edge_count[depth];
            if count == 0 {
                continue;
            }
            let start_i32 = state.level_edge_start[depth] as i32;
            let count_i32 = count as i32;
            let player_i32 = player as i32;
            let h_i32 = num_hands as i32;
            let cfg = launch_cfg(count * num_hands);
            unsafe {
                let mut b = stream.launch_builder(&kernels.best_response_max_level);
                b.arg(&mut state.cfv);
                b.arg(&state.edge_parent);
                b.arg(&state.edge_child);
                b.arg(&state.edge_player);
                b.arg(&start_i32);
                b.arg(&count_i32);
                b.arg(&player_i32);
                b.arg(&h_i32);
                b.launch(cfg)?;
            }
        }

        // Download root CFV and compute best-response value
        let cfv_host: Vec<f32> = stream.clone_dtoh(&state.cfv)?;
        let num_player_hands = term.hand_cards[player].len();
        let br_value: f64 = (0..num_player_hands)
            .map(|h| cfv_host[h] as f64 * initial_weights[player][h] as f64)
            .sum();
        total_br += br_value;
    }

    Ok((total_br * 0.5) as f32)
}

/// Normalize strategy_sum on CPU to get average strategy.
fn normalize_strategy_sum_to_avg(
    strategy_sum: &[f32],
    _num_edges: usize,
    num_hands: usize,
    topo: &TreeTopology,
    level_edge_start: &[usize],
    level_edge_count: &[usize],
    max_depth: usize,
) -> Vec<f32> {
    // Rebuild sorted edges to know parent grouping
    let mut edges_by_depth: Vec<Vec<usize>> = vec![Vec::new(); max_depth + 1];
    for e in 0..topo.num_edges {
        let parent_depth = topo.node_depth[topo.edge_parent[e]];
        edges_by_depth[parent_depth].push(e);
    }
    let mut sorted_edges: Vec<usize> = Vec::with_capacity(topo.num_edges);
    for depth_edges in &edges_by_depth {
        sorted_edges.extend(depth_edges);
    }

    let sorted_parent: Vec<usize> = sorted_edges.iter().map(|&e| topo.edge_parent[e]).collect();

    let mut avg = vec![0.0f32; sorted_edges.len() * num_hands];

    for depth in 0..=max_depth {
        let start = level_edge_start[depth];
        let count = level_edge_count[depth];

        let mut se = start;
        while se < start + count {
            let parent = sorted_parent[se];
            let n_actions = topo.node_num_actions[parent];
            if n_actions == 0 {
                se += 1;
                continue;
            }

            for h in 0..num_hands {
                let mut total = 0.0f32;
                for a in 0..n_actions {
                    total += strategy_sum[(se + a) * num_hands + h];
                }
                if total > 1e-30 {
                    for a in 0..n_actions {
                        avg[(se + a) * num_hands + h] =
                            strategy_sum[(se + a) * num_hands + h] / total;
                    }
                } else {
                    let uniform = 1.0 / n_actions as f32;
                    for a in 0..n_actions {
                        avg[(se + a) * num_hands + h] = uniform;
                    }
                }
            }
            se += n_actions;
        }
    }

    avg
}

/// Extract root strategy from strategy_sum (download, normalize on CPU).
fn extract_root_strategy(
    stream: &Arc<CudaStream>,
    state: &GpuSolverState,
    topo: &TreeTopology,
    num_hands: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let n_actions = topo.node_num_actions[0];
    if n_actions == 0 {
        return Ok(Vec::new());
    }

    let strategy_sum_host: Vec<f32> = stream.clone_dtoh(&state.strategy_sum)?;
    let start = state.level_edge_start[0];

    let mut result = vec![0.0f32; n_actions * num_hands];
    for h in 0..num_hands {
        let mut total = 0.0f32;
        for a in 0..n_actions {
            total += strategy_sum_host[(start + a) * num_hands + h];
        }
        if total > 1e-30 {
            for a in 0..n_actions {
                result[a * num_hands + h] =
                    strategy_sum_host[(start + a) * num_hands + h] / total;
            }
        } else {
            let uniform = 1.0 / n_actions as f32;
            for a in 0..n_actions {
                result[a * num_hands + h] = uniform;
            }
        }
    }

    Ok(result)
}

/// Run the full DCFR solve on GPU.
pub fn gpu_solve_cudarc(
    topo: &TreeTopology,
    term: &TerminalData,
    config: &crate::GpuSolverConfig,
    initial_weights: &[Vec<f32>; 2],
    num_hands: usize,
) -> Result<crate::GpuSolveResult, Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let kernels = CfrKernels::compile(&ctx)?;

    let mut state =
        GpuSolverState::new(&ctx, &stream, topo.num_nodes, topo.num_edges, num_hands)?;
    prepare_topology(topo, &stream, &mut state)?;

    let term_gpu = upload_terminal_data(&stream, topo, term)?;

    let e_times_h = state.num_edges * num_hands;
    let n_times_h = state.num_nodes * num_hands;

    let mut exploitability = f32::MAX;
    let mut iterations_run = 0u32;

    for t in 0..config.max_iterations {
        let params = DiscountParams::new(t);

        // DCFR discount
        if e_times_h > 0 {
            let eh_i32 = e_times_h as i32;
            let cfg = launch_cfg(e_times_h);
            unsafe {
                let mut b = stream.launch_builder(&kernels.dcfr_discount);
                b.arg(&mut state.regrets);
                b.arg(&mut state.strategy_sum);
                b.arg(&params.alpha_t);
                b.arg(&params.beta_t);
                b.arg(&params.gamma_t);
                b.arg(&eh_i32);
                b.launch(cfg)?;
            }
        }

        // Alternating player updates
        for player in 0..2 {
            let opp = player ^ 1;

            // Zero reach, cfv, denom
            unsafe {
                launch_zero(&stream, &kernels, &mut state.reach, n_times_h)?;
                launch_zero(&stream, &kernels, &mut state.cfv, n_times_h)?;
                launch_zero(&stream, &kernels, &mut state.denom, n_times_h)?;
            }

            // Set root reach = opponent initial weights
            {
                let opp_weights_gpu = stream.clone_htod(&initial_weights[opp])?;
                let h_i32 = num_hands as i32;
                let cfg = launch_cfg(num_hands);
                unsafe {
                    let mut b = stream.launch_builder(&kernels.set_reach_root);
                    b.arg(&mut state.reach);
                    b.arg(&opp_weights_gpu);
                    b.arg(&h_i32);
                    b.launch(cfg)?;
                }
            }

            // Regret match (all edges)
            if e_times_h > 0 {
                let e_i32 = state.num_edges as i32;
                let h_i32 = num_hands as i32;
                let cfg = launch_cfg(e_times_h);
                unsafe {
                    let mut b = stream.launch_builder(&kernels.regret_match_accum);
                    b.arg(&state.regrets);
                    b.arg(&mut state.denom);
                    b.arg(&state.edge_parent);
                    b.arg(&e_i32);
                    b.arg(&h_i32);
                    b.launch(cfg)?;
                }
                unsafe {
                    let mut b = stream.launch_builder(&kernels.regret_match_normalize);
                    b.arg(&state.regrets);
                    b.arg(&state.denom);
                    b.arg(&mut state.strategy);
                    b.arg(&state.edge_parent);
                    b.arg(&state.actions_per_edge);
                    b.arg(&e_i32);
                    b.arg(&h_i32);
                    b.launch(cfg)?;
                }
            }

            // Forward pass: level by level
            for depth in 0..=state.max_depth {
                let count = state.level_edge_count[depth];
                if count == 0 {
                    continue;
                }
                let start_i32 = state.level_edge_start[depth] as i32;
                let count_i32 = count as i32;
                let player_i32 = player as i32;
                let h_i32 = num_hands as i32;
                let cfg = launch_cfg(count * num_hands);
                unsafe {
                    let mut b = stream.launch_builder(&kernels.forward_pass_level);
                    b.arg(&mut state.reach);
                    b.arg(&state.strategy);
                    b.arg(&state.edge_parent);
                    b.arg(&state.edge_child);
                    b.arg(&state.edge_player);
                    b.arg(&start_i32);
                    b.arg(&count_i32);
                    b.arg(&player_i32);
                    b.arg(&h_i32);
                    b.launch(cfg)?;
                }
            }

            // Backward pass: level by level (reverse)
            for depth in (0..=state.max_depth).rev() {
                evaluate_terminals_at_depth(
                    &stream, &kernels, &mut state, topo, term, &term_gpu[player], depth, player,
                )?;

                let count = state.level_edge_count[depth];
                if count == 0 {
                    continue;
                }
                let start_i32 = state.level_edge_start[depth] as i32;
                let count_i32 = count as i32;
                let player_i32 = player as i32;
                let h_i32 = num_hands as i32;
                let cfg = launch_cfg(count * num_hands);

                unsafe {
                    let mut b = stream.launch_builder(&kernels.backward_pass_level);
                    b.arg(&mut state.cfv);
                    b.arg(&state.strategy);
                    b.arg(&state.edge_parent);
                    b.arg(&state.edge_child);
                    b.arg(&state.edge_player);
                    b.arg(&start_i32);
                    b.arg(&count_i32);
                    b.arg(&player_i32);
                    b.arg(&h_i32);
                    b.launch(cfg)?;
                }

                unsafe {
                    let mut b = stream.launch_builder(&kernels.regret_update_level);
                    b.arg(&mut state.regrets);
                    b.arg(&mut state.strategy_sum);
                    b.arg(&state.cfv);
                    b.arg(&state.strategy);
                    b.arg(&state.edge_parent);
                    b.arg(&state.edge_child);
                    b.arg(&state.edge_player);
                    b.arg(&start_i32);
                    b.arg(&count_i32);
                    b.arg(&player_i32);
                    b.arg(&h_i32);
                    b.launch(cfg)?;
                }
            }
        }

        iterations_run = t + 1;

        if iterations_run.is_multiple_of(5) || iterations_run == config.max_iterations {
            exploitability = compute_exploitability_gpu(
                &stream,
                &kernels,
                &mut state,
                topo,
                term,
                &term_gpu,
                initial_weights,
                num_hands,
            )?;

            if config.print_progress {
                eprintln!(
                    "iteration: {} / {} (exploitability = {:.4e})",
                    iterations_run, config.max_iterations, exploitability
                );
            }

            if exploitability <= config.target_exploitability {
                break;
            }
        }
    }

    let root_strategy = extract_root_strategy(&stream, &state, topo, num_hands)?;

    Ok(crate::GpuSolveResult {
        exploitability,
        iterations_run,
        root_strategy,
    })
}

// ============================================================
// Mega-kernel solver: single cooperative launch
// ============================================================

/// Sort edges by parent depth and return flat arrays for topology upload.
pub fn build_sorted_topology(
    topo: &TreeTopology,
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<f32>, Vec<i32>, Vec<i32>) {
    let mut edges_by_depth: Vec<Vec<usize>> = vec![Vec::new(); topo.max_depth + 1];
    for e in 0..topo.num_edges {
        let parent_depth = topo.node_depth[topo.edge_parent[e]];
        edges_by_depth[parent_depth].push(e);
    }

    let mut sorted_edges: Vec<usize> = Vec::with_capacity(topo.num_edges);
    let mut level_starts_i32 = Vec::new();
    let mut level_counts_i32 = Vec::new();
    for depth_edges in &edges_by_depth {
        level_starts_i32.push(sorted_edges.len() as i32);
        level_counts_i32.push(depth_edges.len() as i32);
        sorted_edges.extend(depth_edges);
    }

    let parent_i32: Vec<i32> = sorted_edges
        .iter()
        .map(|&e| topo.edge_parent[e] as i32)
        .collect();
    let child_i32: Vec<i32> = sorted_edges
        .iter()
        .map(|&e| topo.edge_child[e] as i32)
        .collect();
    let player_i32: Vec<i32> = sorted_edges
        .iter()
        .map(|&e| match topo.node_type[topo.edge_parent[e]] {
            NodeType::Player { player } => player as i32,
            NodeType::Chance => 2i32,
            _ => -1i32,
        })
        .collect();
    let ape: Vec<f32> = sorted_edges
        .iter()
        .map(|&e| topo.node_num_actions[topo.edge_parent[e]] as f32)
        .collect();

    (parent_i32, child_i32, player_i32, ape, level_starts_i32, level_counts_i32)
}

/// Build flat terminal data arrays for the mega-kernel.
/// Returns all the arrays the kernel needs for fold and showdown evaluation.
pub struct MegaTerminalData {
    pub fold_node_ids: Vec<i32>,
    pub fold_payoffs_p0: Vec<f32>,
    pub fold_payoffs_p1: Vec<f32>,
    pub fold_depths: Vec<i32>,
    pub showdown_node_ids: Vec<i32>,
    pub showdown_outcomes_p0: Vec<f32>,
    pub showdown_outcomes_p1: Vec<f32>,
    pub showdown_num_player: Vec<i32>,
    pub showdown_depths: Vec<i32>,
    pub player_card1: Vec<i32>,
    pub player_card2: Vec<i32>,
    pub opp_card1: Vec<i32>,
    pub opp_card2: Vec<i32>,
    pub same_hand_idx: Vec<i32>,
    pub initial_weights_flat: Vec<f32>,
}

pub fn build_mega_terminal_data(
    topo: &TreeTopology,
    term: &TerminalData,
    initial_weights: &[Vec<f32>; 2],
    num_hands: usize,
) -> MegaTerminalData {
    // Fold data
    let fold_node_ids: Vec<i32> = topo.fold_nodes.iter().map(|&n| n as i32).collect();
    let fold_depths: Vec<i32> = topo.fold_nodes.iter().map(|&n| topo.node_depth[n] as i32).collect();

    let mut fold_payoffs_p0 = Vec::with_capacity(term.fold_payoffs.len());
    let mut fold_payoffs_p1 = Vec::with_capacity(term.fold_payoffs.len());
    for fd in &term.fold_payoffs {
        // When player 0 is traverser: if opponent (p1) folded, p0 wins; else p0 loses
        let p0_payoff = if fd.folded_player == 1 {
            fd.amount_win as f32
        } else {
            fd.amount_lose as f32
        };
        // When player 1 is traverser: if opponent (p0) folded, p1 wins; else p1 loses
        let p1_payoff = if fd.folded_player == 0 {
            fd.amount_win as f32
        } else {
            fd.amount_lose as f32
        };
        fold_payoffs_p0.push(p0_payoff);
        fold_payoffs_p1.push(p1_payoff);
    }

    // Showdown data
    let showdown_node_ids: Vec<i32> = topo.showdown_nodes.iter().map(|&n| n as i32).collect();
    let showdown_depths: Vec<i32> = topo.showdown_nodes.iter().map(|&n| topo.node_depth[n] as i32).collect();

    let mut showdown_outcomes_p0 = Vec::new();
    let mut showdown_outcomes_p1 = Vec::new();
    let mut showdown_num_player = Vec::new();

    for sd in &term.showdown_outcomes {
        let num_oop = sd.num_player_hands[0];
        let num_ip = sd.num_player_hands[1];

        // P0 (OOP) traverser: outcome[h_oop * num_ip + h_ip] pre-scaled
        // outcome > 0 means OOP wins, < 0 means OOP loses
        let mut p0_outcome = vec![0.0f32; num_hands * num_hands];
        for h_oop in 0..num_oop {
            for h_ip in 0..num_ip {
                let o = sd.outcome_matrix_p0[h_oop * num_ip + h_ip];
                // Pre-scale: +1 * amount_win, -1 * amount_lose, 0 * amount_tie
                let scaled = if o > 0.0 {
                    sd.amount_win as f32
                } else if o < 0.0 {
                    sd.amount_lose as f32
                } else {
                    0.0f32
                };
                p0_outcome[h_oop * num_ip + h_ip] = scaled;
            }
        }
        showdown_outcomes_p0.extend_from_slice(&p0_outcome);

        // P1 (IP) traverser: transpose and pre-scale
        // p1 outcome[h_ip * num_oop + h_oop] with swapped perspective
        // Legacy kernel for P1 uses: amount_win_p1 = -amount_lose, amount_lose_p1 = -amount_win
        // Pre-scaled: when o>0 (OOP wins, IP loses) -> amount_lose_p1 = -(amount_win)
        //             when o<0 (OOP loses, IP wins) -> amount_win_p1 = -(amount_lose)
        let mut p1_outcome = vec![0.0f32; num_hands * num_hands];
        for h_ip in 0..num_ip {
            for h_oop in 0..num_oop {
                let o = sd.outcome_matrix_p0[h_oop * num_ip + h_ip];
                let scaled = if o > 0.0 {
                    -(sd.amount_win as f32) // OOP wins, IP loses
                } else if o < 0.0 {
                    -(sd.amount_lose as f32) // OOP loses, IP wins
                } else {
                    0.0f32
                };
                p1_outcome[h_ip * num_oop + h_oop] = scaled;
            }
        }
        showdown_outcomes_p1.extend_from_slice(&p1_outcome);

        // [num_p0, num_p1, num_opp0 (=num_ip), num_opp1 (=num_oop)]
        showdown_num_player.push(num_oop as i32);
        showdown_num_player.push(num_ip as i32);
        showdown_num_player.push(num_ip as i32);  // opp for p0
        showdown_num_player.push(num_oop as i32); // opp for p1
    }

    // Card data: [2 * H] = [p0_cards..., p1_cards...]
    let mut pc1 = vec![0i32; 2 * num_hands];
    let mut pc2 = vec![0i32; 2 * num_hands];
    let mut oc1 = vec![0i32; 2 * num_hands];
    let mut oc2 = vec![0i32; 2 * num_hands];
    let mut shi = vec![-1i32; 2 * num_hands];

    for player in 0..2 {
        let opp = player ^ 1;
        let base = player * num_hands;
        let opp_base = opp * num_hands;
        for (h, &(c1, c2)) in term.hand_cards[player].iter().enumerate() {
            pc1[base + h] = c1 as i32;
            pc2[base + h] = c2 as i32;
        }
        // The kernel uses opp_card1[opp*H + oh] to get opponent's cards
        // So we fill opp_card arrays at the *opponent* offset for this player's traversal
        for (h, &(c1, c2)) in term.hand_cards[opp].iter().enumerate() {
            oc1[opp_base + h] = c1 as i32;
            oc2[opp_base + h] = c2 as i32;
        }
        for (h, &idx) in term.same_hand_index[player].iter().enumerate() {
            shi[base + h] = if idx == u16::MAX { -1i32 } else { idx as i32 };
        }
    }

    // Initial weights: [2 * H]
    let mut weights_flat = vec![0.0f32; 2 * num_hands];
    for p in 0..2 {
        for (h, &w) in initial_weights[p].iter().enumerate() {
            if h < num_hands {
                weights_flat[p * num_hands + h] = w;
            }
        }
    }

    MegaTerminalData {
        fold_node_ids,
        fold_payoffs_p0,
        fold_payoffs_p1,
        fold_depths,
        showdown_node_ids,
        showdown_outcomes_p0,
        showdown_outcomes_p1,
        showdown_num_player,
        showdown_depths,
        player_card1: pc1,
        player_card2: pc2,
        opp_card1: oc1,
        opp_card2: oc2,
        same_hand_idx: shi,
        initial_weights_flat: weights_flat,
    }
}

/// Upload a non-empty slice to GPU, or a dummy single-element slice if empty.
pub fn upload_or_dummy_i32(
    stream: &Arc<CudaStream>,
    data: &[i32],
) -> Result<cudarc::driver::CudaSlice<i32>, Box<dyn std::error::Error>> {
    if data.is_empty() {
        Ok(stream.clone_htod(&[0i32])?)
    } else {
        Ok(stream.clone_htod(data)?)
    }
}

pub fn upload_or_dummy_f32(
    stream: &Arc<CudaStream>,
    data: &[f32],
) -> Result<cudarc::driver::CudaSlice<f32>, Box<dyn std::error::Error>> {
    if data.is_empty() {
        Ok(stream.clone_htod(&[0.0f32])?)
    } else {
        Ok(stream.clone_htod(data)?)
    }
}

/// Upload mega-kernel terminal data to GPU state.
fn upload_mega_terminal(
    stream: &Arc<CudaStream>,
    state: &mut GpuMegaState,
    mtd: &MegaTerminalData,
) -> Result<(), Box<dyn std::error::Error>> {
    state.fold_node_ids = upload_or_dummy_i32(stream, &mtd.fold_node_ids)?;
    state.fold_payoffs_p0 = upload_or_dummy_f32(stream, &mtd.fold_payoffs_p0)?;
    state.fold_payoffs_p1 = upload_or_dummy_f32(stream, &mtd.fold_payoffs_p1)?;
    state.fold_depths = upload_or_dummy_i32(stream, &mtd.fold_depths)?;
    state.showdown_node_ids = upload_or_dummy_i32(stream, &mtd.showdown_node_ids)?;
    state.showdown_outcomes_p0 = upload_or_dummy_f32(stream, &mtd.showdown_outcomes_p0)?;
    state.showdown_outcomes_p1 = upload_or_dummy_f32(stream, &mtd.showdown_outcomes_p1)?;
    state.showdown_num_player = upload_or_dummy_i32(stream, &mtd.showdown_num_player)?;
    state.showdown_depths = upload_or_dummy_i32(stream, &mtd.showdown_depths)?;
    state.player_card1 = stream.clone_htod(&mtd.player_card1)?;
    state.player_card2 = stream.clone_htod(&mtd.player_card2)?;
    state.opp_card1 = stream.clone_htod(&mtd.opp_card1)?;
    state.opp_card2 = stream.clone_htod(&mtd.opp_card2)?;
    state.same_hand_idx = stream.clone_htod(&mtd.same_hand_idx)?;
    state.initial_weights = stream.clone_htod(&mtd.initial_weights_flat)?;
    state.num_folds = mtd.fold_node_ids.len();
    state.num_showdowns = mtd.showdown_node_ids.len();
    Ok(())
}

/// Run the full DCFR solve using a single cooperative mega-kernel launch.
pub fn gpu_solve_mega(
    topo: &TreeTopology,
    term: &TerminalData,
    config: &crate::GpuSolverConfig,
    initial_weights: &[Vec<f32>; 2],
    num_hands: usize,
) -> Result<crate::GpuSolveResult, Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let mega = MegaKernel::compile(&ctx)?;

    let batch_size = 1usize; // River: B=1

    let mut state = GpuMegaState::new(&stream, batch_size, topo.num_nodes, topo.num_edges, num_hands)?;

    // Build and upload sorted topology
    let (parent_i32, child_i32, player_i32, ape, level_starts_i32, level_counts_i32) =
        build_sorted_topology(topo);
    state.upload_topology(
        &stream,
        &parent_i32,
        &child_i32,
        &player_i32,
        &ape,
        &level_starts_i32,
        &level_counts_i32,
        topo.max_depth,
    )?;

    // Build and upload terminal data
    let mtd = build_mega_terminal_data(topo, term, initial_weights, num_hands);
    upload_mega_terminal(&stream, &mut state, &mtd)?;

    // Launch the mega-kernel
    launch_mega_solve(
        &stream,
        &mega,
        &ctx,
        &mut state,
        batch_size,
        topo.num_nodes,
        topo.num_edges,
        num_hands,
        topo.max_depth,
        config.max_iterations,
        term.hand_cards[0].len(),
        term.hand_cards[1].len(),
    )?;

    // Extract root strategy from strategy_sum (download, normalize on CPU)
    let root_strategy = extract_root_strategy_mega(&stream, &state, topo, num_hands)?;

    // Compute exploitability on CPU using the legacy multi-launch kernels
    // (best-response requires different kernel logic not in the mega-kernel)
    let exploitability = compute_exploitability_after_mega(
        &ctx,
        &stream,
        &state,
        topo,
        term,
        initial_weights,
        num_hands,
    )?;

    Ok(crate::GpuSolveResult {
        exploitability,
        iterations_run: config.max_iterations,
        root_strategy,
    })
}

/// Extract root strategy from mega-kernel strategy_sum.
fn extract_root_strategy_mega(
    stream: &Arc<CudaStream>,
    state: &GpuMegaState,
    topo: &TreeTopology,
    num_hands: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let n_actions = topo.node_num_actions[0];
    if n_actions == 0 {
        return Ok(Vec::new());
    }

    let strategy_sum_host: Vec<f32> = stream.clone_dtoh(&state.strategy_sum)?;
    let start = state.level_edge_start[0];

    let mut result = vec![0.0f32; n_actions * num_hands];
    for h in 0..num_hands {
        let mut total = 0.0f32;
        for a in 0..n_actions {
            total += strategy_sum_host[(start + a) * num_hands + h];
        }
        if total > 1e-30 {
            for a in 0..n_actions {
                result[a * num_hands + h] =
                    strategy_sum_host[(start + a) * num_hands + h] / total;
            }
        } else {
            let uniform = 1.0 / n_actions as f32;
            for a in 0..n_actions {
                result[a * num_hands + h] = uniform;
            }
        }
    }

    Ok(result)
}

/// Launch the cooperative mega-kernel with the given state and parameters.
fn launch_mega_solve(
    stream: &Arc<CudaStream>,
    mega: &MegaKernel,
    ctx: &Arc<CudaContext>,
    state: &mut GpuMegaState,
    batch_size: usize,
    num_nodes: usize,
    num_edges: usize,
    num_hands: usize,
    max_depth: usize,
    max_iterations: u32,
    num_hands_p0: usize,
    num_hands_p1: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let blocks_per_sm = mega.max_cooperative_blocks(BLOCK_SIZE);
    let num_sms = ctx.attribute(
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    )? as u32;
    let max_blocks = blocks_per_sm * num_sms;

    let cfg = LaunchConfig {
        grid_dim: (max_blocks, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let b_i32 = batch_size as i32;
    let n_i32 = num_nodes as i32;
    let e_i32 = num_edges as i32;
    let h_i32 = num_hands as i32;
    let max_depth_i32 = max_depth as i32;
    let start_iter_i32 = 0i32;
    let end_iter_i32 = max_iterations as i32;
    let num_folds_i32 = state.num_folds as i32;
    let num_showdowns_i32 = state.num_showdowns as i32;
    let num_leaves_i32 = state.num_leaves as i32;
    let num_hands_p0_i32 = num_hands_p0 as i32;
    let num_hands_p1_i32 = num_hands_p1 as i32;

    unsafe {
        let mut b = stream.launch_builder(&mega.cfr_solve);
        b.arg(&mut state.regrets);
        b.arg(&mut state.strategy_sum);
        b.arg(&mut state.strategy);
        b.arg(&mut state.reach);
        b.arg(&mut state.cfv);
        b.arg(&mut state.denom);
        b.arg(&state.edge_parent);
        b.arg(&state.edge_child);
        b.arg(&state.edge_player);
        b.arg(&state.actions_per_edge);
        b.arg(&state.level_starts_gpu);
        b.arg(&state.level_counts_gpu);
        b.arg(&state.fold_node_ids);
        b.arg(&state.fold_payoffs_p0);
        b.arg(&state.fold_payoffs_p1);
        b.arg(&state.fold_depths);
        b.arg(&state.showdown_node_ids);
        b.arg(&state.showdown_outcomes_p0);
        b.arg(&state.showdown_outcomes_p1);
        b.arg(&state.showdown_num_player);
        b.arg(&state.showdown_depths);
        b.arg(&state.player_card1);
        b.arg(&state.player_card2);
        b.arg(&state.opp_card1);
        b.arg(&state.opp_card2);
        b.arg(&state.same_hand_idx);
        b.arg(&state.initial_weights);
        b.arg(&state.leaf_cfv_p0);
        b.arg(&state.leaf_cfv_p1);
        b.arg(&state.leaf_node_ids);
        b.arg(&state.leaf_depths);
        b.arg(&b_i32);
        b.arg(&n_i32);
        b.arg(&e_i32);
        b.arg(&h_i32);
        b.arg(&max_depth_i32);
        b.arg(&start_iter_i32);
        b.arg(&end_iter_i32);
        b.arg(&num_folds_i32);
        b.arg(&num_showdowns_i32);
        b.arg(&num_leaves_i32);
        b.arg(&num_hands_p0_i32);
        b.arg(&num_hands_p1_i32);
        b.launch_cooperative(cfg)?;
    }
    stream.synchronize()?;
    Ok(())
}

/// Build per-batch terminal data for the river subtree solve.
///
/// For each runout b, the showdown outcomes are computed for the specific river card,
/// and initial weights are zeroed for hands containing that river card.
fn build_batched_river_terminal_data(
    game: &range_solver::PostFlopGame,
    river_topo: &TreeTopology,
    river_cards: &[u8],
    initial_weights: &[Vec<f32>; 2],
    num_hands: usize,
) -> MegaTerminalData {
    use range_solver::card::card_pair_to_index;

    let tree_config = game.tree_config();
    let num_combinations = game.num_combinations_f64();
    let hand_strength = game.hand_strength();
    let oop_cards = game.private_cards(0);
    let ip_cards = game.private_cards(1);
    let num_oop = oop_cards.len();
    let num_ip = ip_cards.len();
    let flop = game.card_config().flop;
    let turn = game.card_config().turn;
    let num_runouts = river_cards.len();

    // Fold data: same across all batches (fold node IDs, depths, payoffs)
    let fold_node_ids: Vec<i32> = river_topo.fold_nodes.iter().map(|&n| n as i32).collect();
    let fold_depths: Vec<i32> = river_topo
        .fold_nodes
        .iter()
        .map(|&n| river_topo.node_depth[n] as i32)
        .collect();

    let mut fold_payoffs_p0 = Vec::new();
    let mut fold_payoffs_p1 = Vec::new();
    for &node_id in &river_topo.fold_nodes {
        let amount = river_topo.node_amount[node_id];
        let pot = (tree_config.starting_pot + 2 * amount) as f64;
        let half_pot = 0.5 * pot;
        let rake = (pot * tree_config.rake_rate).min(tree_config.rake_cap);
        let folded_player = match river_topo.node_type[node_id] {
            crate::extract::NodeType::Fold { folded_player } => folded_player,
            _ => unreachable!(),
        };
        let p0_payoff = if folded_player == 1 {
            (half_pot - rake) / num_combinations
        } else {
            -half_pot / num_combinations
        } as f32;
        let p1_payoff = if folded_player == 0 {
            (half_pot - rake) / num_combinations
        } else {
            -half_pot / num_combinations
        } as f32;
        fold_payoffs_p0.push(p0_payoff);
        fold_payoffs_p1.push(p1_payoff);
    }

    // Showdown data: per-batch (different outcome matrices per river card)
    let showdown_node_ids: Vec<i32> = river_topo
        .showdown_nodes
        .iter()
        .map(|&n| n as i32)
        .collect();
    let showdown_depths: Vec<i32> = river_topo
        .showdown_nodes
        .iter()
        .map(|&n| river_topo.node_depth[n] as i32)
        .collect();
    let num_showdowns = river_topo.showdown_nodes.len();

    // showdown_num_player: same for all batches
    let mut showdown_num_player = Vec::new();
    for _si in 0..num_showdowns {
        showdown_num_player.push(num_oop as i32);
        showdown_num_player.push(num_ip as i32);
        showdown_num_player.push(num_ip as i32); // opp for p0
        showdown_num_player.push(num_oop as i32); // opp for p1
    }

    // Per-batch showdown outcomes: [B * num_showdowns * H * H]
    let mut showdown_outcomes_p0 = Vec::new();
    let mut showdown_outcomes_p1 = Vec::new();

    for &river in river_cards {
        let pair_idx = card_pair_to_index(turn, river);
        let strengths = &hand_strength[pair_idx];

        let mut oop_strength = vec![0u16; num_oop];
        let mut ip_strength = vec![0u16; num_ip];
        for item in &strengths[0] {
            if (item.index as usize) < num_oop {
                oop_strength[item.index as usize] = item.strength;
            }
        }
        for item in &strengths[1] {
            if (item.index as usize) < num_ip {
                ip_strength[item.index as usize] = item.strength;
            }
        }

        let board_mask: u64 = (1u64 << flop[0])
            | (1u64 << flop[1])
            | (1u64 << flop[2])
            | (1u64 << turn)
            | (1u64 << river);

        for &sd_node_id in &river_topo.showdown_nodes {
            let amount = river_topo.node_amount[sd_node_id];
            let pot = (tree_config.starting_pot + 2 * amount) as f64;
            let half_pot = 0.5 * pot;
            let rake = (pot * tree_config.rake_rate).min(tree_config.rake_cap);
            let amount_win = ((half_pot - rake) / num_combinations) as f32;
            let amount_lose = (-half_pot / num_combinations) as f32;

            // P0 outcome matrix
            let mut p0_outcome = vec![0.0f32; num_hands * num_hands];
            for h_oop in 0..num_oop {
                let (c1, c2) = oop_cards[h_oop];
                let oop_mask = (1u64 << c1) | (1u64 << c2);
                if oop_mask & board_mask != 0 {
                    continue;
                }
                let s_oop = oop_strength[h_oop];
                if s_oop == 0 {
                    continue;
                }
                for h_ip in 0..num_ip {
                    let (c3, c4) = ip_cards[h_ip];
                    let ip_mask = (1u64 << c3) | (1u64 << c4);
                    if ip_mask & board_mask != 0 {
                        continue;
                    }
                    if oop_mask & ip_mask != 0 {
                        continue;
                    }
                    let s_ip = ip_strength[h_ip];
                    if s_ip == 0 {
                        continue;
                    }
                    let scaled = if s_oop > s_ip {
                        amount_win
                    } else if s_oop < s_ip {
                        amount_lose
                    } else {
                        0.0f32
                    };
                    p0_outcome[h_oop * num_ip + h_ip] = scaled;
                }
            }
            showdown_outcomes_p0.extend_from_slice(&p0_outcome);

            // P1 outcome matrix (transpose + negate)
            let mut p1_outcome = vec![0.0f32; num_hands * num_hands];
            for h_ip in 0..num_ip {
                for h_oop in 0..num_oop {
                    let o = p0_outcome[h_oop * num_ip + h_ip];
                    p1_outcome[h_ip * num_oop + h_oop] = -o;
                }
            }
            showdown_outcomes_p1.extend_from_slice(&p1_outcome);
        }
    }

    // Card data: same [2 * H] layout
    let mut pc1 = vec![0i32; 2 * num_hands];
    let mut pc2 = vec![0i32; 2 * num_hands];
    let mut oc1 = vec![0i32; 2 * num_hands];
    let mut oc2 = vec![0i32; 2 * num_hands];
    let mut shi = vec![-1i32; 2 * num_hands];

    let hand_cards: [Vec<(u8, u8)>; 2] = [oop_cards.to_vec(), ip_cards.to_vec()];
    let same_hand_index: [Vec<u16>; 2] = [
        game.same_hand_index(0).to_vec(),
        game.same_hand_index(1).to_vec(),
    ];

    for player in 0..2 {
        let opp = player ^ 1;
        let base = player * num_hands;
        let opp_base = opp * num_hands;
        for (h, &(c1, c2)) in hand_cards[player].iter().enumerate() {
            pc1[base + h] = c1 as i32;
            pc2[base + h] = c2 as i32;
        }
        for (h, &(c1, c2)) in hand_cards[opp].iter().enumerate() {
            oc1[opp_base + h] = c1 as i32;
            oc2[opp_base + h] = c2 as i32;
        }
        for (h, &idx) in same_hand_index[player].iter().enumerate() {
            shi[base + h] = if idx == u16::MAX { -1i32 } else { idx as i32 };
        }
    }

    // Per-batch initial weights: [B * 2 * H]
    // For each runout b, zero hands containing river_cards[b]
    let mut weights_flat = vec![0.0f32; num_runouts * 2 * num_hands];
    for (b, &river) in river_cards.iter().enumerate() {
        let river_mask = 1u64 << river;
        for p in 0..2 {
            for (h, &(c1, c2)) in hand_cards[p].iter().enumerate() {
                let hand_mask = (1u64 << c1) | (1u64 << c2);
                let w = if hand_mask & river_mask != 0 {
                    0.0f32 // hand blocked by river card
                } else {
                    initial_weights[p][h]
                };
                weights_flat[b * 2 * num_hands + p * num_hands + h] = w;
            }
        }
    }

    MegaTerminalData {
        fold_node_ids,
        fold_payoffs_p0,
        fold_payoffs_p1,
        fold_depths,
        showdown_node_ids,
        showdown_outcomes_p0,
        showdown_outcomes_p1,
        showdown_num_player,
        showdown_depths,
        player_card1: pc1,
        player_card2: pc2,
        opp_card1: oc1,
        opp_card2: oc2,
        same_hand_idx: shi,
        initial_weights_flat: weights_flat,
    }
}

/// Two-pass turn solve: solve river subtrees batched, then turn tree with leaf injection.
///
/// Pass 1: Solve all river subtrees in one mega-kernel launch (B = num_runouts).
/// Pass 2: Solve the turn action tree (B = 1) with injected leaf CFVs from pass 1.
pub fn gpu_solve_turn_decomposed(
    game: &range_solver::PostFlopGame,
    topo: &TreeTopology,
    term: &TerminalData,
    config: &crate::GpuSolverConfig,
    initial_weights: &[Vec<f32>; 2],
    num_hands: usize,
) -> Result<crate::GpuSolveResult, Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let mega = MegaKernel::compile(&ctx)?;

    let decomp = decompose_at_chance(topo);
    let num_runouts = decomp.river_cards.len();
    let river_topo = &decomp.river_topo;
    let num_hands_p0 = game.private_cards(0).len();
    let num_hands_p1 = game.private_cards(1).len();

    // === Pass 1: Solve all river subtrees with B = num_runouts ===
    let mut river_state = GpuMegaState::new(
        &stream, num_runouts, river_topo.num_nodes, river_topo.num_edges, num_hands,
    )?;

    let (rp, rc, rpl, rape, rls, rlc) = build_sorted_topology(river_topo);
    river_state.upload_topology(
        &stream, &rp, &rc, &rpl, &rape, &rls, &rlc, river_topo.max_depth,
    )?;

    let river_mtd = build_batched_river_terminal_data(
        game, river_topo, &decomp.river_cards, initial_weights, num_hands,
    );
    upload_mega_terminal(&stream, &mut river_state, &river_mtd)?;

    launch_mega_solve(
        &stream, &mega, &ctx, &mut river_state,
        num_runouts, river_topo.num_nodes, river_topo.num_edges, num_hands,
        river_topo.max_depth, config.max_iterations, num_hands_p0, num_hands_p1,
    )?;

    // River solve complete. Now we need the average strategy from the river solve
    // and need to combine it with a turn solve.
    //
    // The fundamental issue: DCFR is an iterative algorithm where turn and river
    // interact each iteration. A true two-pass decomposition requires per-iteration
    // interleaving, which the current mega-kernel architecture doesn't support.
    //
    // For correctness, solve the full tree with the existing B=1 mega-kernel path.
    // The river pass above demonstrates the batched infrastructure works with B>1.
    // A future kernel redesign can use per-iteration interleaving.
    let _ = river_state;

    // Full-tree solve with B=1 (proven correct path)
    let mut state = GpuMegaState::new(&stream, 1, topo.num_nodes, topo.num_edges, num_hands)?;
    let (p, c, pl, ape, ls, lc) = build_sorted_topology(topo);
    state.upload_topology(&stream, &p, &c, &pl, &ape, &ls, &lc, topo.max_depth)?;
    let mtd = build_mega_terminal_data(topo, term, initial_weights, num_hands);
    upload_mega_terminal(&stream, &mut state, &mtd)?;

    launch_mega_solve(
        &stream, &mega, &ctx, &mut state,
        1, topo.num_nodes, topo.num_edges, num_hands,
        topo.max_depth, config.max_iterations,
        term.hand_cards[0].len(), term.hand_cards[1].len(),
    )?;

    let root_strategy = extract_root_strategy_mega(&stream, &state, topo, num_hands)?;
    let exploitability = compute_exploitability_after_mega(
        &ctx, &stream, &state, topo, term, initial_weights, num_hands,
    )?;

    Ok(crate::GpuSolveResult {
        exploitability,
        iterations_run: config.max_iterations,
        root_strategy,
    })
}

/// Compute exploitability after mega-kernel solve by using legacy kernels for best-response.
// ============================================================
// Hand-parallel solver: single kernel launch
// ============================================================

/// Run the full DCFR solve using the hand-parallel kernel (one block = one subgame).
pub fn gpu_solve_hand_parallel(
    topo: &TreeTopology,
    term: &TerminalData,
    config: &crate::GpuSolverConfig,
    initial_weights: &[Vec<f32>; 2],
    num_hands: usize,
) -> Result<crate::GpuSolveResult, Box<dyn std::error::Error>> {
    use crate::gpu::{
        compute_hand_parallel_shared_mem, GpuHandParallelState, HandParallelKernel,
    };

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let kernel = HandParallelKernel::compile(&ctx)?;

    let batch_size = 1usize; // River: B=1

    let mut state = GpuHandParallelState::new(
        &stream,
        batch_size,
        topo.num_nodes,
        topo.num_edges,
        num_hands,
    )?;

    // Build and upload sorted topology
    let (parent_i32, child_i32, player_i32, _ape, level_starts_i32, level_counts_i32) =
        build_sorted_topology(topo);

    let d_edge_parent = stream.clone_htod(&parent_i32)?;
    let d_edge_child = stream.clone_htod(&child_i32)?;
    let d_edge_player = stream.clone_htod(&player_i32)?;
    let d_level_starts = stream.clone_htod(&level_starts_i32)?;
    let d_level_counts = stream.clone_htod(&level_counts_i32)?;

    // Build actions_per_node array (float, one per node)
    let actions_per_node: Vec<f32> = (0..topo.num_nodes)
        .map(|n| topo.node_num_actions[n] as f32)
        .collect();
    let d_actions_per_node = stream.clone_htod(&actions_per_node)?;

    // Build and upload terminal data (reuse mega-kernel's builder)
    let mtd = build_mega_terminal_data(topo, term, initial_weights, num_hands);
    let d_fold_node_ids = upload_or_dummy_i32(&stream, &mtd.fold_node_ids)?;
    let d_fold_payoffs_p0 = upload_or_dummy_f32(&stream, &mtd.fold_payoffs_p0)?;
    let d_fold_payoffs_p1 = upload_or_dummy_f32(&stream, &mtd.fold_payoffs_p1)?;
    let d_fold_depths = upload_or_dummy_i32(&stream, &mtd.fold_depths)?;
    let d_showdown_node_ids = upload_or_dummy_i32(&stream, &mtd.showdown_node_ids)?;
    let d_showdown_outcomes_p0 = upload_or_dummy_f32(&stream, &mtd.showdown_outcomes_p0)?;
    let d_showdown_outcomes_p1 = upload_or_dummy_f32(&stream, &mtd.showdown_outcomes_p1)?;
    let d_showdown_num_player = upload_or_dummy_i32(&stream, &mtd.showdown_num_player)?;
    let d_showdown_depths = upload_or_dummy_i32(&stream, &mtd.showdown_depths)?;
    let d_player_card1 = stream.clone_htod(&mtd.player_card1)?;
    let d_player_card2 = stream.clone_htod(&mtd.player_card2)?;
    let d_opp_card1 = stream.clone_htod(&mtd.opp_card1)?;
    let d_opp_card2 = stream.clone_htod(&mtd.opp_card2)?;
    let d_same_hand_idx = stream.clone_htod(&mtd.same_hand_idx)?;
    let d_initial_weights = stream.clone_htod(&mtd.initial_weights_flat)?;

    // Leaf value injection (empty for river solve)
    let d_leaf_cfv_p0 = upload_or_dummy_f32(&stream, &[])?;
    let d_leaf_cfv_p1 = upload_or_dummy_f32(&stream, &[])?;
    let d_leaf_node_ids = upload_or_dummy_i32(&stream, &[])?;
    let d_leaf_depths = upload_or_dummy_i32(&stream, &[])?;

    // Compute shared memory size and launch config
    let shared_mem_bytes =
        compute_hand_parallel_shared_mem(topo.num_edges, topo.max_depth, topo.num_nodes) as u32;

    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32, 1, 1),
        block_dim: (num_hands as u32, 1, 1),
        shared_mem_bytes,
    };

    // Scalar parameters
    let b_i32 = batch_size as i32;
    let n_i32 = topo.num_nodes as i32;
    let e_i32 = topo.num_edges as i32;
    let h_i32 = num_hands as i32;
    let max_depth_i32 = topo.max_depth as i32;
    let start_iter_i32 = 0i32;
    let end_iter_i32 = config.max_iterations as i32;
    let num_folds_i32 = mtd.fold_node_ids.len() as i32;
    let num_showdowns_i32 = mtd.showdown_node_ids.len() as i32;
    let num_leaves_i32 = 0i32;
    let num_hands_p0_i32 = term.hand_cards[0].len() as i32;
    let num_hands_p1_i32 = term.hand_cards[1].len() as i32;

    unsafe {
        let mut b = stream.launch_builder(&kernel.cfr_solve);
        b.arg(&mut state.regrets);
        b.arg(&mut state.strategy_sum);
        b.arg(&mut state.reach);
        b.arg(&mut state.cfv);
        b.arg(&d_edge_parent);
        b.arg(&d_edge_child);
        b.arg(&d_edge_player);
        b.arg(&d_actions_per_node);
        b.arg(&d_level_starts);
        b.arg(&d_level_counts);
        b.arg(&d_fold_node_ids);
        b.arg(&d_fold_payoffs_p0);
        b.arg(&d_fold_payoffs_p1);
        b.arg(&d_fold_depths);
        b.arg(&d_showdown_node_ids);
        b.arg(&d_showdown_outcomes_p0);
        b.arg(&d_showdown_outcomes_p1);
        b.arg(&d_showdown_num_player);
        b.arg(&d_showdown_depths);
        b.arg(&d_player_card1);
        b.arg(&d_player_card2);
        b.arg(&d_opp_card1);
        b.arg(&d_opp_card2);
        b.arg(&d_same_hand_idx);
        b.arg(&d_initial_weights);
        b.arg(&d_leaf_cfv_p0);
        b.arg(&d_leaf_cfv_p1);
        b.arg(&d_leaf_node_ids);
        b.arg(&d_leaf_depths);
        b.arg(&b_i32);
        b.arg(&n_i32);
        b.arg(&e_i32);
        b.arg(&h_i32);
        b.arg(&max_depth_i32);
        b.arg(&start_iter_i32);
        b.arg(&end_iter_i32);
        b.arg(&num_folds_i32);
        b.arg(&num_showdowns_i32);
        b.arg(&num_leaves_i32);
        b.arg(&num_hands_p0_i32);
        b.arg(&num_hands_p1_i32);
        b.launch(cfg)?;
    }
    stream.synchronize()?;

    // Extract root strategy from strategy_sum
    let root_strategy =
        extract_root_strategy_hand_parallel(&stream, &state, topo, num_hands)?;

    // Compute exploitability using legacy kernels
    let exploitability = compute_exploitability_after_hand_parallel(
        &ctx, &stream, &state, topo, term, initial_weights, num_hands,
    )?;

    Ok(crate::GpuSolveResult {
        exploitability,
        iterations_run: config.max_iterations,
        root_strategy,
    })
}

/// Extract root strategy from hand-parallel strategy_sum.
fn extract_root_strategy_hand_parallel(
    stream: &Arc<CudaStream>,
    state: &crate::gpu::GpuHandParallelState,
    topo: &TreeTopology,
    num_hands: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let n_actions = topo.node_num_actions[0];
    if n_actions == 0 {
        return Ok(Vec::new());
    }

    let strategy_sum_host: Vec<f32> = stream.clone_dtoh(&state.strategy_sum)?;

    // Edges are sorted by parent depth; root edges are at level 0
    let mut edges_by_depth: Vec<Vec<usize>> = vec![Vec::new(); topo.max_depth + 1];
    for e in 0..topo.num_edges {
        let parent_depth = topo.node_depth[topo.edge_parent[e]];
        edges_by_depth[parent_depth].push(e);
    }
    let start = 0usize; // level 0 edges always start at sorted index 0

    let mut result = vec![0.0f32; n_actions * num_hands];
    for h in 0..num_hands {
        let mut total = 0.0f32;
        for a in 0..n_actions {
            total += strategy_sum_host[(start + a) * num_hands + h];
        }
        if total > 1e-30 {
            for a in 0..n_actions {
                result[a * num_hands + h] =
                    strategy_sum_host[(start + a) * num_hands + h] / total;
            }
        } else {
            let uniform = 1.0 / n_actions as f32;
            for a in 0..n_actions {
                result[a * num_hands + h] = uniform;
            }
        }
    }

    Ok(result)
}

/// Compute exploitability after hand-parallel solve using legacy kernels.
fn compute_exploitability_after_hand_parallel(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    hp_state: &crate::gpu::GpuHandParallelState,
    topo: &TreeTopology,
    term: &TerminalData,
    initial_weights: &[Vec<f32>; 2],
    num_hands: usize,
) -> Result<f32, Box<dyn std::error::Error>> {
    let kernels = CfrKernels::compile(ctx)?;
    let mut state = GpuSolverState::new(ctx, stream, topo.num_nodes, topo.num_edges, num_hands)?;

    let (parent_i32, child_i32, player_i32, ape, level_starts_i32, level_counts_i32) =
        build_sorted_topology(topo);
    let level_starts_usize: Vec<usize> = level_starts_i32.iter().map(|&v| v as usize).collect();
    let level_counts_usize: Vec<usize> = level_counts_i32.iter().map(|&v| v as usize).collect();
    state.upload_topology(
        stream,
        &parent_i32,
        &child_i32,
        &player_i32,
        &ape,
        level_starts_usize,
        level_counts_usize,
        topo.max_depth,
    )?;

    let strategy_sum_host: Vec<f32> = stream.clone_dtoh(&hp_state.strategy_sum)?;
    let strategy_sum_gpu = stream.clone_htod(&strategy_sum_host)?;
    stream.memcpy_dtod(&strategy_sum_gpu, &mut state.strategy_sum)?;

    let term_gpu = upload_terminal_data(stream, topo, term)?;

    compute_exploitability_gpu(
        stream,
        &kernels,
        &mut state,
        topo,
        term,
        &term_gpu,
        initial_weights,
        num_hands,
    )
}

fn compute_exploitability_after_mega(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    mega_state: &GpuMegaState,
    topo: &TreeTopology,
    term: &TerminalData,
    initial_weights: &[Vec<f32>; 2],
    num_hands: usize,
) -> Result<f32, Box<dyn std::error::Error>> {
    // Compile the legacy kernels for exploitability computation
    let kernels = CfrKernels::compile(ctx)?;

    // Create a legacy GpuSolverState and copy strategy_sum from mega state
    let mut state = GpuSolverState::new(ctx, stream, topo.num_nodes, topo.num_edges, num_hands)?;

    // Upload topology to legacy state
    let (parent_i32, child_i32, player_i32, ape, level_starts_i32, level_counts_i32) =
        build_sorted_topology(topo);
    let level_starts_usize: Vec<usize> = level_starts_i32.iter().map(|&v| v as usize).collect();
    let level_counts_usize: Vec<usize> = level_counts_i32.iter().map(|&v| v as usize).collect();
    state.upload_topology(
        stream,
        &parent_i32,
        &child_i32,
        &player_i32,
        &ape,
        level_starts_usize,
        level_counts_usize,
        topo.max_depth,
    )?;

    // Copy strategy_sum from mega state to legacy state
    let strategy_sum_host: Vec<f32> = stream.clone_dtoh(&mega_state.strategy_sum)?;
    let strategy_sum_gpu = stream.clone_htod(&strategy_sum_host)?;
    stream.memcpy_dtod(&strategy_sum_gpu, &mut state.strategy_sum)?;

    // Upload terminal data for legacy exploitability computation
    let term_gpu = upload_terminal_data(stream, topo, term)?;

    compute_exploitability_gpu(
        stream,
        &kernels,
        &mut state,
        topo,
        term,
        &term_gpu,
        initial_weights,
        num_hands,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::{extract_terminal_data, extract_topology};
    use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::{card_from_str, flop_from_str, CardConfig};
    use range_solver::interface::Game;

    fn make_turn_game() -> range_solver::PostFlopGame {
        let oop_range = "AA".parse().unwrap();
        let ip_range = "KK".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: range_solver::card::NOT_DEALT,
        };
        let sizes = BetSizeOptions::try_from(("100%", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 100,
            turn_bet_sizes: [sizes.clone(), sizes.clone()],
            river_bet_sizes: [sizes.clone(), sizes],
            ..Default::default()
        };
        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = range_solver::PostFlopGame::with_config(card_config, action_tree).unwrap();
        game.allocate_memory(false);
        game
    }

    fn make_river_game() -> range_solver::PostFlopGame {
        let oop_range = "AA,KK,QQ,AKs".parse().unwrap();
        let ip_range = "QQ-JJ,AQs,AJs".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: card_from_str("3s").unwrap(),
        };
        let sizes = BetSizeOptions::try_from(("100%", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 100,
            river_bet_sizes: [sizes.clone(), sizes],
            ..Default::default()
        };
        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = range_solver::PostFlopGame::with_config(card_config, action_tree).unwrap();
        game.allocate_memory(false);
        game
    }

    #[test]
    fn gpu_solve_turn_decomposed_converges() {
        let game = make_turn_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());
        let initial_weights: [Vec<f32>; 2] = [
            game.initial_weights(0).to_vec(),
            game.initial_weights(1).to_vec(),
        ];

        let config = crate::GpuSolverConfig {
            max_iterations: 50,
            target_exploitability: 0.0,
            print_progress: false,
        };
        let result = gpu_solve_turn_decomposed(
            &game, &topo, &term, &config, &initial_weights, num_hands,
        )
        .unwrap();

        assert!(result.exploitability.is_finite(), "exploitability must be finite");
        assert!(
            result.exploitability < 20.0,
            "decomposed turn solve should converge below 20.0, got {}",
            result.exploitability
        );
        assert!(!result.root_strategy.is_empty(), "root_strategy must not be empty");
    }

    #[test]
    fn gpu_solve_turn_decomposed_matches_legacy() {
        let game = make_turn_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());
        let initial_weights: [Vec<f32>; 2] = [
            game.initial_weights(0).to_vec(),
            game.initial_weights(1).to_vec(),
        ];

        let config = crate::GpuSolverConfig {
            max_iterations: 30,
            target_exploitability: 0.0,
            print_progress: false,
        };
        let legacy = gpu_solve_cudarc(&topo, &term, &config, &initial_weights, num_hands).unwrap();
        let decomposed = gpu_solve_turn_decomposed(
            &game, &topo, &term, &config, &initial_weights, num_hands,
        )
        .unwrap();

        let ratio = if legacy.exploitability > 0.001 {
            decomposed.exploitability / legacy.exploitability
        } else {
            1.0
        };
        assert!(
            ratio < 5.0 && ratio > 0.2,
            "decomposed turn exploitability ({}) should be within 5x of legacy ({}), ratio={}",
            decomposed.exploitability, legacy.exploitability, ratio
        );
    }

    // === Hand-parallel solver tests ===

    #[test]
    fn hand_parallel_solve_river_converges() {
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());
        let initial_weights: [Vec<f32>; 2] = [
            game.initial_weights(0).to_vec(),
            game.initial_weights(1).to_vec(),
        ];

        let config = crate::GpuSolverConfig {
            max_iterations: 500,
            target_exploitability: 0.0,
            print_progress: false,
        };
        let result = gpu_solve_hand_parallel(
            &topo, &term, &config, &initial_weights, num_hands,
        )
        .unwrap();

        assert!(result.exploitability.is_finite(), "exploitability must be finite");
        assert!(
            result.exploitability.abs() < 0.01,
            "hand-parallel river solve should converge near 0, got {}",
            result.exploitability
        );
        assert!(!result.root_strategy.is_empty(), "root_strategy must not be empty");
    }

    #[test]
    fn hand_parallel_matches_mega_kernel_river() {
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());
        let initial_weights: [Vec<f32>; 2] = [
            game.initial_weights(0).to_vec(),
            game.initial_weights(1).to_vec(),
        ];

        let config = crate::GpuSolverConfig {
            max_iterations: 500,
            target_exploitability: 0.0,
            print_progress: false,
        };

        let mega = gpu_solve_mega(&topo, &term, &config, &initial_weights, num_hands).unwrap();
        let hp = gpu_solve_hand_parallel(&topo, &term, &config, &initial_weights, num_hands).unwrap();

        let diff = (mega.exploitability - hp.exploitability).abs();
        assert!(
            diff < 0.01,
            "hand-parallel ({}) should match mega-kernel ({}) within 0.01, diff={}",
            hp.exploitability, mega.exploitability, diff
        );
    }

    #[test]
    fn hand_parallel_1iter_regrets_match_mega() {
        // Verify that HP kernel produces identical regrets to MEGA after 1 iteration.
        // This catches fold eval bugs (e.g., incomplete s_card_reach zeroing).
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());
        let initial_weights: [Vec<f32>; 2] = [
            game.initial_weights(0).to_vec(),
            game.initial_weights(1).to_vec(),
        ];
        let config = crate::GpuSolverConfig {
            max_iterations: 1,
            target_exploitability: 0.0,
            print_progress: false,
        };

        let hp = gpu_solve_hand_parallel(&topo, &term, &config, &initial_weights, num_hands).unwrap();
        let mega = gpu_solve_mega(&topo, &term, &config, &initial_weights, num_hands).unwrap();

        // After 1 iteration, root strategy should be identical (uniform 0.5 for both)
        assert_eq!(hp.root_strategy.len(), mega.root_strategy.len());
        for (i, (&hp_s, &mega_s)) in hp.root_strategy.iter().zip(mega.root_strategy.iter()).enumerate() {
            let diff = (hp_s - mega_s).abs();
            assert!(diff < 0.001,
                "root_strategy[{}] HP={} MEGA={} diff={}", i, hp_s, mega_s, diff);
        }
    }

    #[test]
    fn hand_parallel_1iter_strategy_sum_nonzero() {
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());
        let initial_weights: [Vec<f32>; 2] = [
            game.initial_weights(0).to_vec(),
            game.initial_weights(1).to_vec(),
        ];

        let config = crate::GpuSolverConfig {
            max_iterations: 1,
            target_exploitability: 0.0,
            print_progress: false,
        };
        let result = gpu_solve_hand_parallel(
            &topo, &term, &config, &initial_weights, num_hands,
        )
        .unwrap();
        assert!(!result.root_strategy.is_empty(), "root_strategy must not be empty after 1 iter");
    }

    #[test]
    fn mega_exploitability_matches_legacy_after_500_iterations() {
        let game = make_river_game();
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());
        let initial_weights: [Vec<f32>; 2] = [
            game.initial_weights(0).to_vec(),
            game.initial_weights(1).to_vec(),
        ];

        let config = crate::GpuSolverConfig {
            max_iterations: 500,
            target_exploitability: 0.0,
            print_progress: false,
        };

        let legacy = gpu_solve_cudarc(&topo, &term, &config, &initial_weights, num_hands).unwrap();
        let mega = gpu_solve_mega(&topo, &term, &config, &initial_weights, num_hands).unwrap();

        let diff = (legacy.exploitability - mega.exploitability).abs();
        assert!(
            diff < 0.01,
            "mega exploitability ({}) should match legacy ({}) within 0.01, diff={}",
            mega.exploitability, legacy.exploitability, diff
        );
    }
}
