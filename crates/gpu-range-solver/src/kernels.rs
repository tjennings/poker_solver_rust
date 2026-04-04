//! CUDA kernel source for the CFR solver, compiled at runtime via nvrtc.

/// All CUDA kernels for the CFR solver.
pub const CFR_KERNELS_SOURCE: &str = r#"
extern "C" {

// ============================================================
// Utility: zero a float array
// ============================================================
__global__ void zero_f32(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = 0.0f;
}

// ============================================================
// Regret matching pass 1: accumulate clipped regrets to denom
// ============================================================
__global__ void regret_match_accum(
    const float* regrets,
    float* denom,
    const int* edge_parent,
    int E, int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E * H) return;
    int e = idx / H;
    int h = idx % H;
    float clipped = fmaxf(regrets[e * H + h], 0.0f);
    if (clipped > 0.0f) {
        atomicAdd(&denom[edge_parent[e] * H + h], clipped);
    }
}

// ============================================================
// Regret matching pass 2: normalize to get strategy
// ============================================================
__global__ void regret_match_normalize(
    const float* regrets,
    const float* denom,
    float* strategy,
    const int* edge_parent,
    const float* actions_per_edge,
    int E, int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E * H) return;
    int e = idx / H;
    int h = idx % H;
    float clipped = fmaxf(regrets[e * H + h], 0.0f);
    float d = denom[edge_parent[e] * H + h];
    strategy[e * H + h] = (d > 1e-30f) ? (clipped / d) : (1.0f / actions_per_edge[e]);
}

// ============================================================
// Forward pass: propagate reach for one level's edges
// ============================================================
__global__ void forward_pass_level(
    float* reach,
    const float* strategy,
    const int* edge_parent,
    const int* edge_child,
    const int* edge_player,
    int level_start,
    int level_count,
    int traverser_player,
    int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= level_count * H) return;
    int local_e = idx / H;
    int h = idx % H;
    int e = level_start + local_e;

    int p = edge_parent[e];
    int c = edge_child[e];
    int player = edge_player[e];

    float parent_reach = reach[p * H + h];

    if (player == traverser_player) {
        reach[c * H + h] = parent_reach;
    } else if (player == 2) {
        reach[c * H + h] = parent_reach;
    } else {
        reach[c * H + h] = parent_reach * strategy[e * H + h];
    }
}

// ============================================================
// Backward pass: scatter-add weighted child CFVs to parents
// ============================================================
__global__ void backward_pass_level(
    float* cfv,
    const float* strategy,
    const int* edge_parent,
    const int* edge_child,
    const int* edge_player,
    int level_start,
    int level_count,
    int traverser_player,
    int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= level_count * H) return;
    int local_e = idx / H;
    int h = idx % H;
    int e = level_start + local_e;

    int p = edge_parent[e];
    int c = edge_child[e];
    int player = edge_player[e];

    float child_cfv = cfv[c * H + h];

    if (player == traverser_player) {
        atomicAdd(&cfv[p * H + h], strategy[e * H + h] * child_cfv);
    } else {
        atomicAdd(&cfv[p * H + h], child_cfv);
    }
}

// ============================================================
// Regret update: after backward_pass_level, parent CFVs are final
// ============================================================
__global__ void regret_update_level(
    float* regrets,
    float* strategy_sum,
    const float* cfv,
    const float* strategy,
    const int* edge_parent,
    const int* edge_child,
    const int* edge_player,
    int level_start,
    int level_count,
    int traverser_player,
    int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= level_count * H) return;
    int local_e = idx / H;
    int h = idx % H;
    int e = level_start + local_e;

    if (edge_player[e] != traverser_player) return;

    int p = edge_parent[e];
    int c = edge_child[e];

    float instant_regret = cfv[c * H + h] - cfv[p * H + h];
    regrets[e * H + h] += instant_regret;
    strategy_sum[e * H + h] += strategy[e * H + h];
}

// ============================================================
// DCFR discount: scale regrets and strategy_sum
// ============================================================
__global__ void dcfr_discount(
    float* regrets,
    float* strategy_sum,
    float alpha,
    float beta,
    float gamma,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float r = regrets[idx];
    regrets[idx] = (r >= 0.0f) ? r * alpha : r * beta;
    strategy_sum[idx] *= gamma;
}

// ============================================================
// Fold evaluation: compute CFV for fold terminal nodes
// ============================================================
__global__ void fold_eval(
    float* cfv,
    const float* reach,
    int node_id,
    float payoff,
    const int* player_card1,
    const int* player_card2,
    const int* opp_card1,
    const int* opp_card2,
    const int* same_hand_idx,
    int num_player_hands,
    int num_opp_hands,
    int H
) {
    __shared__ float card_reach[52];
    __shared__ float total_reach;

    int tid = threadIdx.x;
    if (tid < 52) card_reach[tid] = 0.0f;
    if (tid == 0) total_reach = 0.0f;
    __syncthreads();

    if (tid < num_opp_hands) {
        float r = reach[node_id * H + tid];
        atomicAdd(&total_reach, r);
        atomicAdd(&card_reach[opp_card1[tid]], r);
        atomicAdd(&card_reach[opp_card2[tid]], r);
    }
    __syncthreads();

    if (tid < num_player_hands) {
        int c1 = player_card1[tid];
        int c2 = player_card2[tid];
        float blocking = card_reach[c1] + card_reach[c2];
        int same = same_hand_idx[tid];
        if (same >= 0) {
            blocking -= reach[node_id * H + same];
        }
        cfv[node_id * H + tid] = payoff * (total_reach - blocking);
    }
}

// ============================================================
// Showdown evaluation: matmul outcome * opp_reach
// ============================================================
__global__ void showdown_eval(
    float* cfv,
    const float* reach,
    int node_id,
    const float* outcome,
    float amount_win,
    float amount_lose,
    int num_player_hands,
    int num_opp_hands,
    int H
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_player_hands) return;

    float win_sum = 0.0f;
    float lose_sum = 0.0f;
    for (int opp = 0; opp < num_opp_hands; opp++) {
        float opp_r = reach[node_id * H + opp];
        float o = outcome[h * num_opp_hands + opp];
        if (o > 0.0f) win_sum += opp_r;
        else if (o < 0.0f) lose_sum += opp_r;
    }
    cfv[node_id * H + h] = win_sum * amount_win + lose_sum * amount_lose;
}

// ============================================================
// Best-response backward: max over actions at traverser nodes
// ============================================================
__global__ void best_response_max_level(
    float* cfv,
    const int* edge_parent,
    const int* edge_child,
    const int* edge_player,
    int level_start,
    int level_count,
    int traverser_player,
    int H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= level_count * H) return;
    int local_e = idx / H;
    int h = idx % H;
    int e = level_start + local_e;

    if (edge_player[e] != traverser_player) {
        atomicAdd(&cfv[edge_parent[e] * H + h], cfv[edge_child[e] * H + h]);
        return;
    }

    int p = edge_parent[e];
    float val = cfv[edge_child[e] * H + h];
    int* addr = (int*)&cfv[p * H + h];
    int old = *addr, assumed;
    do {
        assumed = old;
        float old_f = __int_as_float(assumed);
        float new_f = fmaxf(old_f, val);
        old = atomicCAS(addr, assumed, __float_as_int(new_f));
    } while (assumed != old);
}

// ============================================================
// Copy initial weights into reach at root node
// ============================================================
__global__ void set_reach_root(
    float* reach,
    const float* initial_weights,
    int H
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < H) reach[h] = initial_weights[h];
}

} // extern "C"
"#;

/// Compile options for the hand-parallel kernel.
/// Requires sm_89 (Ada architecture) for optimal performance.
pub fn hand_parallel_compile_opts() -> cudarc::nvrtc::CompileOptions {
    cudarc::nvrtc::CompileOptions {
        arch: Some("sm_89"),
        ..Default::default()
    }
}

/// Hand-parallel CUDA kernel: one thread block = one subgame, threads = hands.
/// No cooperative groups, no grid.sync(). Only __syncthreads() for fold eval.
pub const HAND_PARALLEL_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void cfr_solve(
    // Per-block solver state in global memory
    float* regrets,           // [B * E * H]
    float* strategy_sum,      // [B * E * H]
    float* reach,             // [B * N * H]
    float* cfv,               // [B * N * H]

    // Topology (shared across blocks, loaded into shared mem)
    const int* edge_parent,       // [E]
    const int* edge_child,        // [E]
    const int* edge_player,       // [E]  (0=p0, 1=p1, 2=chance)
    const float* actions_per_node, // [N]
    const int* level_starts,      // [max_depth+1]
    const int* level_counts,      // [max_depth+1]

    // Terminal node data
    const int* fold_node_ids,     // [num_folds]
    const float* fold_payoffs_p0, // [num_folds]
    const float* fold_payoffs_p1, // [num_folds]
    const int* fold_depths,       // [num_folds]

    // Showdown terminal data
    const int* showdown_node_ids,     // [num_showdowns]
    const float* showdown_outcomes_p0, // [B * num_showdowns * H * H]
    const float* showdown_outcomes_p1, // [B * num_showdowns * H * H]
    const int* showdown_num_player,   // [num_showdowns * 4]
    const int* showdown_depths,       // [num_showdowns]

    // Card data for fold eval — [2 * H]
    const int* player_card1,
    const int* player_card2,
    const int* opp_card1,
    const int* opp_card2,
    const int* same_hand_idx,

    // Initial weights — [B * 2 * H]
    const float* initial_weights,

    // Leaf value injection (for two-pass turn solve)
    const float* leaf_cfv_p0,     // [num_leaves * H]
    const float* leaf_cfv_p1,     // [num_leaves * H]
    const int* leaf_node_ids,     // [num_leaves]
    const int* leaf_depths,       // [num_leaves]

    // Scalar dimensions
    int B, int N, int E, int H,
    int max_depth,
    int max_iterations,
    int num_folds,
    int num_showdowns,
    int num_leaves,
    int num_hands_p0,
    int num_hands_p1
) {
    int bid = blockIdx.x;    // which board/subgame
    int tid = threadIdx.x;   // which hand

    if (bid >= B) return;

    int EH = E * H;
    int NH = N * H;
    int player_num_hands[2] = {num_hands_p0, num_hands_p1};
    int showdown_stride = num_showdowns * H * H;

    // Dynamic shared memory for topology + fold eval scratch
    extern __shared__ char shared_raw[];

    // Layout shared memory:
    // [0..E*4] edge_parent as int
    // [E*4..2*E*4] edge_child as int
    // [2*E*4..3*E*4] edge_player as int
    // [3*E*4..3*E*4+(max_depth+1)*4] level_starts as int
    // [then (max_depth+1)*4] level_counts as int
    // [then N*4] actions_per_node as float
    // Then fold eval scratch: card_reach[52] + total_reach
    int* s_parent = (int*)shared_raw;
    int* s_child = s_parent + E;
    int* s_player = s_child + E;
    int* s_level_starts = s_player + E;
    int* s_level_counts = s_level_starts + (max_depth + 1);
    float* s_actions = (float*)(s_level_counts + (max_depth + 1));

    // Cooperative load of topology into shared memory
    for (int i = tid; i < E; i += blockDim.x) {
        s_parent[i] = edge_parent[i];
        s_child[i] = edge_child[i];
        s_player[i] = edge_player[i];
    }
    for (int i = tid; i < max_depth + 1; i += blockDim.x) {
        s_level_starts[i] = level_starts[i];
        s_level_counts[i] = level_counts[i];
    }
    for (int i = tid; i < N; i += blockDim.x) {
        s_actions[i] = actions_per_node[i];
    }
    __syncthreads();

    // Fold eval scratch in shared memory (after topology)
    __shared__ float s_card_reach[52];
    __shared__ float s_total_reach;

    // ============================================
    // DCFR iteration loop
    // ============================================
    for (int iter = 0; iter < max_iterations; iter++) {

        // --- Compute DCFR discount params ---
        float alpha, beta, gamma;
        {
            float ta = (float)((iter > 0) ? (iter - 1) : 0);
            float pa = ta * sqrtf(ta);
            alpha = pa / (pa + 1.0f);
            beta = 0.5f;
            int nearest_p4 = (iter == 0) ? 0 : (1 << ((31 - __clz(iter)) & ~1));
            float tg = (float)(iter - nearest_p4);
            float gb = tg / (tg + 1.0f);
            gamma = gb * gb * gb;
        }

        // --- DCFR discount: per-thread, no sync ---
        for (int e = 0; e < E; e++) {
            int idx = bid * EH + e * H + tid;
            if (tid < H) {
                float r = regrets[idx];
                regrets[idx] = (r >= 0.0f) ? r * alpha : r * beta;
                strategy_sum[idx] *= gamma;
            }
        }

        // --- Alternating player updates ---
        for (int player = 0; player < 2; player++) {
            int opp = 1 - player;
            int num_ph = player_num_hands[player];
            int num_oh = player_num_hands[opp];

            // --- Zero reach and cfv: per-thread ---
            if (tid < H) {
                for (int n = 0; n < N; n++) {
                    reach[bid * NH + n * H + tid] = 0.0f;
                    cfv[bid * NH + n * H + tid] = 0.0f;
                }
            }

            // --- Set root reach = opponent initial weights ---
            if (tid < H) {
                reach[bid * NH + tid] = initial_weights[bid * 2 * H + opp * H + tid];
            }

            // --- Forward pass: sequential over levels, parallel over hands ---
            for (int depth = 0; depth <= max_depth; depth++) {
                int start = s_level_starts[depth];
                int count = s_level_counts[depth];

                int e = start;
                while (e < start + count) {
                    int parent = s_parent[e];
                    int n_actions = (int)s_actions[parent];
                    if (n_actions == 0) { e++; continue; }

                    if (tid < H) {
                        // Regret match for this node
                        float denom = 0.0f;
                        for (int a = 0; a < n_actions; a++) {
                            denom += fmaxf(regrets[bid * EH + (e + a) * H + tid], 0.0f);
                        }
                        float uniform = 1.0f / (float)n_actions;

                        for (int a = 0; a < n_actions; a++) {
                            int edge = e + a;
                            int child = s_child[edge];
                            float clipped = fmaxf(regrets[bid * EH + edge * H + tid], 0.0f);
                            float strat = (denom > 1e-30f) ? clipped / denom : uniform;
                            float pr = reach[bid * NH + parent * H + tid];

                            if (s_player[edge] == player) {
                                reach[bid * NH + child * H + tid] = pr;
                            } else {
                                reach[bid * NH + child * H + tid] = pr * strat;
                            }
                        }
                    }
                    e += n_actions;
                }
            }

            // --- Backward pass: sequential over levels (reverse) ---
            for (int depth = max_depth; depth >= 0; depth--) {

                // --- Fold terminal evaluation ---
                for (int fi = 0; fi < num_folds; fi++) {
                    if (fold_depths[fi] != depth) continue;
                    int node_id = fold_node_ids[fi];
                    float payoff = (player == 0) ? fold_payoffs_p0[fi] : fold_payoffs_p1[fi];

                    // Phase 1: zero accumulators (cooperative loop for H < 52)
                    for (int i = tid; i < 52; i += blockDim.x) s_card_reach[i] = 0.0f;
                    if (tid == 0) s_total_reach = 0.0f;
                    __syncthreads();

                    // Phase 2: opponent hands contribute reach
                    if (tid < num_oh) {
                        float r = reach[bid * NH + node_id * H + tid];
                        atomicAdd(&s_total_reach, r);
                        int opp_base = opp * H;
                        atomicAdd(&s_card_reach[opp_card1[opp_base + tid]], r);
                        atomicAdd(&s_card_reach[opp_card2[opp_base + tid]], r);
                    }
                    __syncthreads();

                    // Phase 3: player hands compute cfv
                    if (tid < num_ph) {
                        int player_base = player * H;
                        int c1 = player_card1[player_base + tid];
                        int c2 = player_card2[player_base + tid];
                        float blocking = s_card_reach[c1] + s_card_reach[c2];
                        int same = same_hand_idx[player_base + tid];
                        if (same >= 0)
                            blocking -= reach[bid * NH + node_id * H + same];
                        cfv[bid * NH + node_id * H + tid] = payoff * (s_total_reach - blocking);
                    }
                    __syncthreads();
                }

                // --- Showdown terminal evaluation ---
                for (int si = 0; si < num_showdowns; si++) {
                    if (showdown_depths[si] != depth) continue;
                    int node_id = showdown_node_ids[si];
                    int s_num_ph = showdown_num_player[si * 4 + player];
                    int s_num_oh = showdown_num_player[si * 4 + 2 + player];

                    if (tid < s_num_ph) {
                        const float* outcome = (player == 0)
                            ? &showdown_outcomes_p0[bid * showdown_stride + si * H * H]
                            : &showdown_outcomes_p1[bid * showdown_stride + si * H * H];
                        float val = 0.0f;
                        for (int oh = 0; oh < s_num_oh; oh++) {
                            float opp_r = reach[bid * NH + node_id * H + oh];
                            float o = outcome[tid * s_num_oh + oh];
                            val += o * opp_r;
                        }
                        cfv[bid * NH + node_id * H + tid] = val;
                    }
                }

                // --- Leaf value injection ---
                for (int li = 0; li < num_leaves; li++) {
                    if (leaf_depths[li] != depth) continue;
                    int node_id = leaf_node_ids[li];
                    const float* leaf_cfv = (player == 0) ? leaf_cfv_p0 : leaf_cfv_p1;
                    if (tid < player_num_hands[player]) {
                        cfv[bid * NH + node_id * H + tid] = leaf_cfv[li * H + tid];
                    }
                }

                // --- CFV accumulation + regret update ---
                int start = s_level_starts[depth];
                int count = s_level_counts[depth];

                int e = start;
                while (e < start + count) {
                    int parent = s_parent[e];
                    int n_actions = (int)s_actions[parent];
                    if (n_actions == 0) { e++; continue; }

                    if (tid < H) {
                        // Recompute strategy
                        float denom = 0.0f;
                        for (int a = 0; a < n_actions; a++)
                            denom += fmaxf(regrets[bid * EH + (e + a) * H + tid], 0.0f);
                        float uniform = 1.0f / (float)n_actions;

                        if (s_player[e] == player) {
                            // Traverser: cfv[parent] = sum(strat * child_cfv)
                            float node_cfv = 0.0f;
                            for (int a = 0; a < n_actions; a++) {
                                float clipped = fmaxf(regrets[bid * EH + (e + a) * H + tid], 0.0f);
                                float strat = (denom > 1e-30f) ? clipped / denom : uniform;
                                float child_cfv = cfv[bid * NH + s_child[e + a] * H + tid];
                                node_cfv += strat * child_cfv;
                            }
                            cfv[bid * NH + parent * H + tid] = node_cfv;

                            // Regret + strategy_sum update
                            for (int a = 0; a < n_actions; a++) {
                                float clipped = fmaxf(regrets[bid * EH + (e + a) * H + tid], 0.0f);
                                float strat = (denom > 1e-30f) ? clipped / denom : uniform;
                                float child_cfv = cfv[bid * NH + s_child[e + a] * H + tid];
                                int idx = bid * EH + (e + a) * H + tid;
                                regrets[idx] += child_cfv - node_cfv;
                                strategy_sum[idx] += strat;
                            }
                        } else {
                            // Opponent/chance: cfv[parent] = sum(child_cfv)
                            float node_cfv = 0.0f;
                            for (int a = 0; a < n_actions; a++)
                                node_cfv += cfv[bid * NH + s_child[e + a] * H + tid];
                            cfv[bid * NH + parent * H + tid] = node_cfv;
                        }
                    }
                    e += n_actions;
                }
            } // end backward depth loop
        } // end player loop
    } // end iteration loop
}
"#;

/// Compile options for the cooperative mega-kernel.
/// Requires sm_89 (Ada architecture) and CUDA include path for cooperative_groups.h.
pub fn mega_kernel_compile_opts() -> cudarc::nvrtc::CompileOptions {
    cudarc::nvrtc::CompileOptions {
        arch: Some("sm_89"),
        include_paths: cuda_include_paths(),
        ..Default::default()
    }
}

/// Discover CUDA include paths for nvrtc compilation of cooperative_groups.h.
fn cuda_include_paths() -> Vec<String> {
    let candidates = [
        "/usr/local/cuda/targets/x86_64-linux/include",
        "/usr/local/cuda-12.1/targets/x86_64-linux/include",
        "/usr/local/cuda-12/targets/x86_64-linux/include",
        "/usr/local/cuda/include",
    ];
    for path in &candidates {
        if std::path::Path::new(path).join("cooperative_groups.h").exists() {
            return vec![path.to_string()];
        }
    }
    // Fallback: try to find via CUDA_HOME env var
    if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
        let p = format!("{cuda_home}/targets/x86_64-linux/include");
        if std::path::Path::new(&p).join("cooperative_groups.h").exists() {
            return vec![p];
        }
        let p = format!("{cuda_home}/include");
        if std::path::Path::new(&p).join("cooperative_groups.h").exists() {
            return vec![p];
        }
    }
    Vec::new()
}

/// Cooperative mega-kernel CUDA source: single `cfr_solve` function that
/// runs the entire DCFR solve loop. Requires cooperative groups (sm_70+).
pub const CFR_MEGA_KERNEL_SOURCE: &str = r#"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__ void cfr_solve(
    // Mutable state arrays
    float* regrets,           // [B * E * H]
    float* strategy_sum,      // [B * E * H]
    float* strategy,          // [B * E * H]
    float* reach,             // [B * N * H]
    float* cfv,               // [B * N * H]
    float* denom,             // [B * N * H]
    // Topology (constant, shared across batch)
    const int* edge_parent,       // [E]
    const int* edge_child,        // [E]
    const int* edge_player,       // [E]  (0=p0, 1=p1, 2=chance)
    const float* actions_per_edge, // [E]
    const int* level_starts,      // [max_depth+1]
    const int* level_counts,      // [max_depth+1]
    // Terminal data — fold
    const int* fold_node_ids,     // [num_folds]
    const float* fold_payoffs_p0, // [num_folds]
    const float* fold_payoffs_p1, // [num_folds]
    const int* fold_depths,       // [num_folds]
    // Terminal data — showdown (per-batch: B copies of outcome matrices)
    const int* showdown_node_ids,     // [num_showdowns]
    const float* showdown_outcomes_p0, // [B * num_showdowns * H * H]
    const float* showdown_outcomes_p1, // [B * num_showdowns * H * H]
    const int* showdown_num_player,   // [num_showdowns * 4] — (num_p0, num_p1, num_opp0, num_opp1)
    const int* showdown_depths,       // [num_showdowns]
    // Card data for fold eval — [2 * H]: [p0_cards..., p1_cards...]
    const int* player_card1,
    const int* player_card2,
    const int* opp_card1,
    const int* opp_card2,
    const int* same_hand_idx,
    // Initial weights — [B * 2 * H]: per-batch [p0_weights..., p1_weights...]
    const float* initial_weights,
    // Leaf value injection — pre-computed CFVs at leaf nodes (for two-pass turn solve)
    const float* leaf_cfv_p0,     // [num_leaves * H]
    const float* leaf_cfv_p1,     // [num_leaves * H]
    const int* leaf_node_ids,     // [num_leaves]
    const int* leaf_depths,       // [num_leaves]
    // Scalar dimensions
    int B,
    int N,
    int E,
    int H,
    int max_depth,
    int max_iterations,
    int num_folds,
    int num_showdowns,
    int num_leaves,
    int num_hands_p0,   // actual hand count for player 0
    int num_hands_p1    // actual hand count for player 1
) {
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int EH = E * H;
    int NH = N * H;
    int BEH = B * EH;
    int BNH = B * NH;
    int showdown_stride = num_showdowns * H * H;
    int player_num_hands[2] = {num_hands_p0, num_hands_p1};

    for (int iter = 0; iter < max_iterations; iter++) {

        // === DCFR Discount ===
        // Compute alpha, beta, gamma from iteration number
        float t_alpha_f = (float)(iter > 0 ? iter - 1 : 0);
        float pow_alpha = t_alpha_f * sqrtf(t_alpha_f);
        float alpha = pow_alpha / (pow_alpha + 1.0f);
        float beta = 0.5f;
        // gamma: nearest lower power of 4
        int nearest_p4 = (iter == 0) ? 0 : (1 << ((31 - __clz(iter)) & ~1));
        float t_gamma = (float)(iter - nearest_p4);
        float gamma_base = t_gamma / (t_gamma + 1.0f);
        float gamma = gamma_base * gamma_base * gamma_base;

        for (int i = tid; i < BEH; i += stride) {
            float r = regrets[i];
            regrets[i] = (r >= 0.0f) ? r * alpha : r * beta;
            strategy_sum[i] *= gamma;
        }
        grid.sync();

        // === Alternating player updates ===
        for (int player = 0; player < 2; player++) {
            int opp = 1 - player;

            // Zero scratch arrays
            for (int i = tid; i < BNH; i += stride) {
                reach[i] = 0.0f;
                cfv[i] = 0.0f;
                denom[i] = 0.0f;
            }
            for (int i = tid; i < BEH; i += stride) {
                strategy[i] = 0.0f;
            }
            grid.sync();

            // Set root reach = per-batch opponent initial weights
            for (int i = tid; i < B * H; i += stride) {
                int b = i / H;
                int h = i % H;
                reach[b * NH + h] = initial_weights[b * 2 * H + opp * H + h];
            }
            grid.sync();

            // === Regret match phase 1: accumulate ===
            for (int i = tid; i < BEH; i += stride) {
                int b = i / EH;
                int local = i % EH;
                int e = local / H;
                int h = local % H;
                float clipped = fmaxf(regrets[i], 0.0f);
                if (clipped > 0.0f) {
                    atomicAdd(&denom[b * NH + edge_parent[e] * H + h], clipped);
                }
            }
            grid.sync();

            // === Regret match phase 2: normalize ===
            for (int i = tid; i < BEH; i += stride) {
                int b = i / EH;
                int local = i % EH;
                int e = local / H;
                int h = local % H;
                float clipped = fmaxf(regrets[i], 0.0f);
                float d = denom[b * NH + edge_parent[e] * H + h];
                strategy[i] = (d > 1e-30f) ? (clipped / d) : (1.0f / actions_per_edge[e]);
            }
            grid.sync();

            // === Forward pass: level by level ===
            for (int depth = 0; depth <= max_depth; depth++) {
                int start = level_starts[depth];
                int count = level_counts[depth];
                int total = B * count * H;
                for (int i = tid; i < total; i += stride) {
                    int b = i / (count * H);
                    int local = i % (count * H);
                    int le = local / H;
                    int h = local % H;
                    int e = start + le;
                    int p = edge_parent[e];
                    int c = edge_child[e];
                    float pr = reach[b * NH + p * H + h];

                    if (edge_player[e] == player) {
                        reach[b * NH + c * H + h] = pr;
                    } else if (edge_player[e] == 2) {
                        reach[b * NH + c * H + h] = pr;
                    } else {
                        reach[b * NH + c * H + h] = pr * strategy[b * EH + e * H + h];
                    }
                }
                grid.sync();
            }

            // === Backward pass: level by level (reverse) ===
            for (int depth = max_depth; depth >= 0; depth--) {

                // --- Fold terminal evaluation (inline) ---
                for (int fi = 0; fi < num_folds; fi++) {
                    if (fold_depths[fi] != depth) continue;
                    int node_id = fold_node_ids[fi];
                    float payoff = (player == 0) ? fold_payoffs_p0[fi] : fold_payoffs_p1[fi];

                    // Per-batch fold eval using registers for card_reach
                    int num_ph_fold = player_num_hands[player];
                    int num_oh_fold = player_num_hands[opp];
                    for (int bi = tid; bi < B; bi += stride) {
                        float total_r = 0.0f;
                        float card_r[52];
                        for (int c = 0; c < 52; c++) card_r[c] = 0.0f;

                        int opp_base = opp * H;
                        for (int oh = 0; oh < num_oh_fold; oh++) {
                            float r = reach[bi * NH + node_id * H + oh];
                            if (r != 0.0f) {
                                total_r += r;
                                card_r[opp_card1[opp_base + oh]] += r;
                                card_r[opp_card2[opp_base + oh]] += r;
                            }
                        }

                        int player_base = player * H;
                        for (int ph = 0; ph < num_ph_fold; ph++) {
                            int c1 = player_card1[player_base + ph];
                            int c2 = player_card2[player_base + ph];
                            float blocking = card_r[c1] + card_r[c2];
                            int same = same_hand_idx[player_base + ph];
                            if (same >= 0) {
                                blocking -= reach[bi * NH + node_id * H + same];
                            }
                            cfv[bi * NH + node_id * H + ph] = payoff * (total_r - blocking);
                        }
                    }
                }
                grid.sync();

                // --- Showdown terminal evaluation (inline, per-batch outcomes) ---
                for (int si = 0; si < num_showdowns; si++) {
                    if (showdown_depths[si] != depth) continue;
                    int node_id = showdown_node_ids[si];
                    // showdown_num_player: [si*4+0]=num_p0, [si*4+1]=num_p1, [si*4+2]=num_opp0, [si*4+3]=num_opp1
                    int num_ph = showdown_num_player[si * 4 + player];
                    int num_oh = showdown_num_player[si * 4 + 2 + player];

                    int total = B * num_ph;
                    for (int i = tid; i < total; i += stride) {
                        int b = i / num_ph;
                        int h = i % num_ph;
                        // Per-batch outcome: index by b * showdown_stride
                        const float* outcome = (player == 0)
                            ? &showdown_outcomes_p0[b * showdown_stride + si * H * H]
                            : &showdown_outcomes_p1[b * showdown_stride + si * H * H];
                        float val = 0.0f;
                        for (int oh = 0; oh < num_oh; oh++) {
                            float opp_r = reach[b * NH + node_id * H + oh];
                            float o = outcome[h * num_oh + oh];
                            val += o * opp_r;
                        }
                        cfv[b * NH + node_id * H + h] = val;
                    }
                }
                grid.sync();

                // --- Leaf value injection (pre-computed CFVs for two-pass solve) ---
                for (int li = 0; li < num_leaves; li++) {
                    if (leaf_depths[li] != depth) continue;
                    int node_id = leaf_node_ids[li];
                    const float* leaf_cfv = (player == 0) ? leaf_cfv_p0 : leaf_cfv_p1;
                    int num_ph = player_num_hands[player];
                    int total = B * num_ph;
                    for (int i = tid; i < total; i += stride) {
                        int b = i / num_ph;
                        int h = i % num_ph;
                        cfv[b * NH + node_id * H + h] = leaf_cfv[li * H + h];
                    }
                }
                if (num_leaves > 0) grid.sync();

                // --- Scatter-add child CFVs to parents ---
                {
                    int start = level_starts[depth];
                    int count = level_counts[depth];
                    int total = B * count * H;
                    for (int i = tid; i < total; i += stride) {
                        int b = i / (count * H);
                        int local = i % (count * H);
                        int le = local / H;
                        int h = local % H;
                        int e = start + le;
                        int p = edge_parent[e];
                        int c = edge_child[e];
                        float cv = cfv[b * NH + c * H + h];

                        if (edge_player[e] == player) {
                            atomicAdd(&cfv[b * NH + p * H + h], strategy[b * EH + e * H + h] * cv);
                        } else {
                            atomicAdd(&cfv[b * NH + p * H + h], cv);
                        }
                    }
                }
                grid.sync();

                // --- Regret update (traverser edges only) ---
                {
                    int start = level_starts[depth];
                    int count = level_counts[depth];
                    int total = B * count * H;
                    for (int i = tid; i < total; i += stride) {
                        int b = i / (count * H);
                        int local = i % (count * H);
                        int le = local / H;
                        int h = local % H;
                        int e = start + le;
                        if (edge_player[e] != player) continue;
                        int p = edge_parent[e];
                        int c = edge_child[e];
                        int idx = b * EH + e * H + h;
                        regrets[idx] += cfv[b * NH + c * H + h] - cfv[b * NH + p * H + h];
                        strategy_sum[idx] += strategy[idx];
                    }
                }
                grid.sync();
            } // end backward depth loop
        } // end player loop
    } // end iteration loop
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_source_compiles_with_nvrtc() {
        let ptx = cudarc::nvrtc::compile_ptx(CFR_KERNELS_SOURCE);
        assert!(ptx.is_ok(), "CUDA source must compile: {:?}", ptx.err());
    }

    #[test]
    fn kernel_source_contains_all_required_kernels() {
        let required = [
            "zero_f32",
            "regret_match_accum",
            "regret_match_normalize",
            "forward_pass_level",
            "backward_pass_level",
            "regret_update_level",
            "dcfr_discount",
            "fold_eval",
            "showdown_eval",
            "best_response_max_level",
            "set_reach_root",
        ];
        for name in &required {
            assert!(
                CFR_KERNELS_SOURCE.contains(name),
                "kernel source must contain '{name}'"
            );
        }
    }

    #[test]
    fn mega_kernel_source_contains_cfr_solve() {
        assert!(
            CFR_MEGA_KERNEL_SOURCE.contains("cfr_solve"),
            "mega-kernel source must contain cfr_solve function"
        );
    }

    #[test]
    fn mega_kernel_source_uses_cooperative_groups() {
        assert!(
            CFR_MEGA_KERNEL_SOURCE.contains("cooperative_groups"),
            "mega-kernel must use cooperative_groups"
        );
        assert!(
            CFR_MEGA_KERNEL_SOURCE.contains("grid.sync()"),
            "mega-kernel must use grid.sync() for synchronization"
        );
    }

    #[test]
    fn mega_kernel_source_compiles_with_nvrtc() {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            CFR_MEGA_KERNEL_SOURCE,
            mega_kernel_compile_opts(),
        );
        assert!(ptx.is_ok(), "mega-kernel CUDA source must compile: {:?}", ptx.err());
    }

    #[test]
    fn mega_kernel_source_has_batch_dimension_params() {
        // The kernel must accept B (batch size), N (nodes), E (edges), H (hands)
        for param in ["int B,", "int N,", "int E,", "int H,"] {
            assert!(
                CFR_MEGA_KERNEL_SOURCE.contains(param),
                "mega-kernel must accept parameter '{param}'"
            );
        }
    }

    #[test]
    fn mega_kernel_source_has_iteration_loop() {
        assert!(
            CFR_MEGA_KERNEL_SOURCE.contains("max_iterations"),
            "mega-kernel must have max_iterations parameter"
        );
        assert!(
            CFR_MEGA_KERNEL_SOURCE.contains("for (int iter"),
            "mega-kernel must contain the iteration loop"
        );
    }

    #[test]
    fn mega_kernel_source_has_grid_stride_loops() {
        // Grid-stride pattern: for (int i = tid; i < total; i += stride)
        assert!(
            CFR_MEGA_KERNEL_SOURCE.contains("i += stride"),
            "mega-kernel must use grid-stride loops"
        );
    }

    #[test]
    fn mega_kernel_source_has_dcfr_discount() {
        // Must compute alpha, beta, gamma inline
        assert!(
            CFR_MEGA_KERNEL_SOURCE.contains("alpha") && CFR_MEGA_KERNEL_SOURCE.contains("beta")
                && CFR_MEGA_KERNEL_SOURCE.contains("gamma"),
            "mega-kernel must compute DCFR discount params inline"
        );
    }

    #[test]
    fn mega_kernel_source_has_terminal_eval() {
        // Must have inline fold and showdown evaluation
        assert!(
            CFR_MEGA_KERNEL_SOURCE.contains("fold_node_ids"),
            "mega-kernel must reference fold_node_ids for inline fold eval"
        );
        assert!(
            CFR_MEGA_KERNEL_SOURCE.contains("showdown_node_ids"),
            "mega-kernel must reference showdown_node_ids for inline showdown eval"
        );
    }

    #[test]
    fn mega_kernel_has_per_batch_initial_weights() {
        // Root reach must index initial_weights by batch: b * 2 * H + opp * H + h
        assert!(
            CFR_MEGA_KERNEL_SOURCE.contains("b * 2 * H + opp * H + h"),
            "mega-kernel must index initial_weights per-batch: b * 2 * H + opp * H + h"
        );
    }

    #[test]
    fn mega_kernel_has_per_batch_showdown_outcomes() {
        // Showdown outcome must index by batch: b * showdown_stride
        assert!(
            CFR_MEGA_KERNEL_SOURCE.contains("b * showdown_stride"),
            "mega-kernel must index showdown outcomes per-batch"
        );
    }

    #[test]
    fn mega_kernel_has_leaf_injection_params() {
        assert!(
            CFR_MEGA_KERNEL_SOURCE.contains("leaf_cfv_p0"),
            "mega-kernel must have leaf_cfv_p0 parameter"
        );
        assert!(
            CFR_MEGA_KERNEL_SOURCE.contains("leaf_cfv_p1"),
            "mega-kernel must have leaf_cfv_p1 parameter"
        );
        assert!(
            CFR_MEGA_KERNEL_SOURCE.contains("num_leaves"),
            "mega-kernel must have num_leaves parameter"
        );
    }

    // === Hand-parallel kernel tests ===

    #[test]
    fn hand_parallel_kernel_compiles() {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            HAND_PARALLEL_KERNEL_SOURCE,
            hand_parallel_compile_opts(),
        );
        assert!(ptx.is_ok(), "Hand-parallel kernel must compile: {:?}", ptx.err());
    }

    #[test]
    fn hand_parallel_kernel_has_cfr_solve() {
        assert!(
            HAND_PARALLEL_KERNEL_SOURCE.contains("cfr_solve"),
            "must contain cfr_solve entry point"
        );
    }

    #[test]
    fn hand_parallel_kernel_uses_syncthreads_not_grid_sync() {
        assert!(
            HAND_PARALLEL_KERNEL_SOURCE.contains("__syncthreads()"),
            "must use __syncthreads() for block-level sync"
        );
        assert!(
            !HAND_PARALLEL_KERNEL_SOURCE.contains("grid.sync()"),
            "must NOT use grid.sync() — hand-parallel uses block sync only"
        );
        assert!(
            !HAND_PARALLEL_KERNEL_SOURCE.contains("cooperative_groups"),
            "must NOT use cooperative_groups"
        );
    }

    #[test]
    fn hand_parallel_kernel_has_iteration_loop() {
        assert!(
            HAND_PARALLEL_KERNEL_SOURCE.contains("max_iterations"),
            "must have max_iterations parameter"
        );
        assert!(
            HAND_PARALLEL_KERNEL_SOURCE.contains("for (int iter"),
            "must contain the iteration loop"
        );
    }

    #[test]
    fn hand_parallel_kernel_has_dcfr_discount() {
        assert!(
            HAND_PARALLEL_KERNEL_SOURCE.contains("alpha")
                && HAND_PARALLEL_KERNEL_SOURCE.contains("beta")
                && HAND_PARALLEL_KERNEL_SOURCE.contains("gamma"),
            "must compute DCFR discount params inline"
        );
    }

    #[test]
    fn hand_parallel_kernel_has_shared_memory_topology() {
        assert!(
            HAND_PARALLEL_KERNEL_SOURCE.contains("__shared__"),
            "must use shared memory for topology and fold eval scratch"
        );
        assert!(
            HAND_PARALLEL_KERNEL_SOURCE.contains("extern __shared__"),
            "must use extern __shared__ for dynamic shared memory"
        );
    }

    #[test]
    fn hand_parallel_kernel_has_block_level_indexing() {
        assert!(
            HAND_PARALLEL_KERNEL_SOURCE.contains("blockIdx.x"),
            "must use blockIdx.x for batch/board index"
        );
        assert!(
            HAND_PARALLEL_KERNEL_SOURCE.contains("threadIdx.x"),
            "must use threadIdx.x for hand index"
        );
    }

    #[test]
    fn hand_parallel_kernel_has_fold_eval_with_card_blocking() {
        assert!(
            HAND_PARALLEL_KERNEL_SOURCE.contains("s_card_reach"),
            "must have shared card_reach for fold eval card blocking"
        );
        assert!(
            HAND_PARALLEL_KERNEL_SOURCE.contains("s_total_reach"),
            "must have shared total_reach for fold eval"
        );
    }

    #[test]
    fn hand_parallel_kernel_has_batch_dimension_params() {
        for param in ["int B,", "int N,", "int E,", "int H,"] {
            assert!(
                HAND_PARALLEL_KERNEL_SOURCE.contains(param),
                "must accept parameter '{param}'"
            );
        }
    }

    #[test]
    fn hand_parallel_kernel_has_terminal_params() {
        assert!(
            HAND_PARALLEL_KERNEL_SOURCE.contains("fold_node_ids"),
            "must have fold_node_ids param"
        );
        assert!(
            HAND_PARALLEL_KERNEL_SOURCE.contains("showdown_node_ids"),
            "must have showdown_node_ids param"
        );
    }
}
