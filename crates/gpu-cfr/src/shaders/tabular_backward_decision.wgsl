// Tabular CFR: backward pass for decision nodes.
//
// Propagates utilities from children and accumulates regrets.
// Thread per (decision_node_in_level, trajectory).
//
// At P1 decision node:
//   util_p1[node][t1] = Σ_a strategy[a] * util_p1[child_a][t1]
//   regret[info_id][a] += util_p1[child_a][t1] - util_p1[node][t1]
//   util_p2[node][t2] = Σ_a util_p2[child_a][t2]  (simple sum, P2 reach already embedded)
//
// At P2 decision node:
//   util_p2[node][t2] = Σ_a strategy[a] * util_p2[child_a][t2]
//   regret[info_id][a] += util_p2[child_a][t2] - util_p2[node][t2]
//   util_p1[node][t1] = Σ_a util_p1[child_a][t1]  (simple sum)

struct Uniforms {
    level_start: u32,
    level_count: u32,
    num_nodes: u32,
    n_traj_p1: u32,
    n_traj_p2: u32,
    max_traj: u32,
    num_info_sets: u32,
    max_actions: u32,
    iteration: u32,
    dcfr_alpha: f32,
    dcfr_beta: f32,
    dcfr_gamma: f32,
    strategy_discount: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct GpuNode {
    node_type: u32,
    position_index: u32,
    first_child: u32,
    num_children: u32,
    p1_invested_bb: f32,
    p2_invested_bb: f32,
    street: u32,
    decision_idx: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> nodes: array<GpuNode>;
@group(0) @binding(2) var<storage, read_write> children_arr: array<u32>;
@group(0) @binding(3) var<storage, read_write> level_nodes: array<u32>;

@group(1) @binding(1) var<storage, read_write> strategy: array<f32>;
@group(1) @binding(3) var<storage, read_write> regret_delta: array<atomic<u32>>;
@group(1) @binding(4) var<storage, read_write> strat_sum_delta: array<atomic<u32>>;

@group(2) @binding(0) var<storage, read_write> reach_p1: array<f32>;
@group(2) @binding(1) var<storage, read_write> reach_p2: array<f32>;
@group(2) @binding(2) var<storage, read_write> util_p1: array<f32>;
@group(2) @binding(3) var<storage, read_write> util_p2: array<f32>;
@group(2) @binding(4) var<storage, read_write> info_id_table: array<u32>;
@group(2) @binding(5) var<storage, read_write> weight_sum: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_id = gid.x;
    let total_threads = uniforms.level_count * uniforms.max_traj;
    if thread_id >= total_threads {
        return;
    }

    let node_local = thread_id / uniforms.max_traj;
    let traj_id = thread_id % uniforms.max_traj;
    let node_idx = level_nodes[uniforms.level_start + node_local];
    let node = nodes[node_idx];

    // Skip terminals
    if node.node_type >= 2u {
        return;
    }

    let player = node.node_type; // 0 = P1, 1 = P2
    let n1 = uniforms.n_traj_p1;
    let n2 = uniforms.n_traj_p2;

    if player == 0u {
        // ---- P1 decision node ----

        // P1 utility: weighted average by P1's strategy
        if traj_id < n1 {
            let info_id = info_id_table[node.decision_idx * uniforms.max_traj + traj_id];
            let strat_base = info_id * uniforms.max_actions;

            // Compute node utility
            var node_util: f32 = 0.0;
            for (var a = 0u; a < node.num_children; a++) {
                let child_idx = children_arr[node.first_child + a];
                node_util += strategy[strat_base + a] * util_p1[child_idx * n1 + traj_id];
            }
            util_p1[node_idx * n1 + traj_id] = node_util;

            // Accumulate regrets (inline atomic CAS)
            for (var a = 0u; a < node.num_children; a++) {
                let child_idx = children_arr[node.first_child + a];
                let child_util = util_p1[child_idx * n1 + traj_id];
                let cf_regret = child_util - node_util;
                let rd_idx = strat_base + a;
                var rd_old = atomicLoad(&regret_delta[rd_idx]);
                loop {
                    let rd_new = bitcast<f32>(rd_old) + cf_regret;
                    let rd_result = atomicCompareExchangeWeak(&regret_delta[rd_idx], rd_old, bitcast<u32>(rd_new));
                    if rd_result.exchanged { break; }
                    rd_old = rd_result.old_value;
                }
            }

            // Accumulate strategy sum
            if uniforms.strategy_discount > 0.0 {
                let my_reach = reach_p1[node_idx * n1 + traj_id];
                let w = weight_sum[traj_id]; // weight_sum_p1
                for (var a = 0u; a < node.num_children; a++) {
                    let contrib = my_reach * strategy[strat_base + a] * uniforms.strategy_discount * w;
                    let ss_idx = strat_base + a;
                    var ss_old = atomicLoad(&strat_sum_delta[ss_idx]);
                    loop {
                        let ss_new = bitcast<f32>(ss_old) + contrib;
                        let ss_result = atomicCompareExchangeWeak(&strat_sum_delta[ss_idx], ss_old, bitcast<u32>(ss_new));
                        if ss_result.exchanged { break; }
                        ss_old = ss_result.old_value;
                    }
                }
            }
        }

        // P2 utility: simple sum (P2's strategy already embedded in reach)
        if traj_id < n2 {
            var sum_util: f32 = 0.0;
            for (var a = 0u; a < node.num_children; a++) {
                let child_idx = children_arr[node.first_child + a];
                sum_util += util_p2[child_idx * n2 + traj_id];
            }
            util_p2[node_idx * n2 + traj_id] = sum_util;
        }
    } else {
        // ---- P2 decision node ----

        // P2 utility: weighted average by P2's strategy
        if traj_id < n2 {
            let info_id = info_id_table[node.decision_idx * uniforms.max_traj + traj_id];
            let strat_base = info_id * uniforms.max_actions;

            // Compute node utility
            var node_util: f32 = 0.0;
            for (var a = 0u; a < node.num_children; a++) {
                let child_idx = children_arr[node.first_child + a];
                node_util += strategy[strat_base + a] * util_p2[child_idx * n2 + traj_id];
            }
            util_p2[node_idx * n2 + traj_id] = node_util;

            // Accumulate regrets (inline atomic CAS)
            for (var a = 0u; a < node.num_children; a++) {
                let child_idx = children_arr[node.first_child + a];
                let child_util = util_p2[child_idx * n2 + traj_id];
                let cf_regret = child_util - node_util;
                let rd_idx = strat_base + a;
                var rd_old = atomicLoad(&regret_delta[rd_idx]);
                loop {
                    let rd_new = bitcast<f32>(rd_old) + cf_regret;
                    let rd_result = atomicCompareExchangeWeak(&regret_delta[rd_idx], rd_old, bitcast<u32>(rd_new));
                    if rd_result.exchanged { break; }
                    rd_old = rd_result.old_value;
                }
            }

            // Accumulate strategy sum
            if uniforms.strategy_discount > 0.0 {
                let my_reach = reach_p2[node_idx * n2 + traj_id];
                let w = weight_sum[n1 + traj_id]; // weight_sum_p2 starts at offset n_traj_p1
                for (var a = 0u; a < node.num_children; a++) {
                    let contrib = my_reach * strategy[strat_base + a] * uniforms.strategy_discount * w;
                    let ss_idx = strat_base + a;
                    var ss_old = atomicLoad(&strat_sum_delta[ss_idx]);
                    loop {
                        let ss_new = bitcast<f32>(ss_old) + contrib;
                        let ss_result = atomicCompareExchangeWeak(&strat_sum_delta[ss_idx], ss_old, bitcast<u32>(ss_new));
                        if ss_result.exchanged { break; }
                        ss_old = ss_result.old_value;
                    }
                }
            }
        }

        // P1 utility: simple sum (P1's strategy already embedded in reach)
        if traj_id < n1 {
            var sum_util: f32 = 0.0;
            for (var a = 0u; a < node.num_children; a++) {
                let child_idx = children_arr[node.first_child + a];
                sum_util += util_p1[child_idx * n1 + traj_id];
            }
            util_p1[node_idx * n1 + traj_id] = sum_util;
        }
    }
}
