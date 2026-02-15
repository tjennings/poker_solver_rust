// Tiled tabular CFR: backward decision pass for one player direction.
//
// At own-player decision nodes:
//   util_own[node][lt] = SUM_a strategy[a] * util_own[child_a][lt]
//   regret_delta[info_id][a] += child_util - node_util
//   strat_sum_delta[info_id][a] += reach_own * strategy[a] * discount * weight
//
// At opponent decision nodes:
//   util_own[node][lt] = SUM_a util_own[child_a][lt]  (simple sum)
//
// Thread per (decision_node_in_level, local_trajectory).
// Dispatch: ceil(level_count * tile_size / 256) workgroups.

struct Uniforms {
    level_start: u32,
    level_count: u32,
    num_nodes: u32,
    n_traj_own: u32,
    n_traj_opp: u32,
    max_traj: u32,
    num_info_sets: u32,
    max_actions: u32,
    iteration: u32,
    dcfr_alpha: f32,
    dcfr_beta: f32,
    dcfr_gamma: f32,
    strategy_discount: f32,
    player: u32,
    tile_offset: u32,
    tile_size: u32,
    opp_tile_size: u32,
    n_decision: u32,
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

@group(2) @binding(0) var<storage, read_write> reach_own: array<f32>;
@group(2) @binding(2) var<storage, read_write> util_own: array<f32>;
@group(2) @binding(4) var<storage, read_write> info_id_table: array<u32>;
@group(2) @binding(5) var<storage, read_write> weight_sum: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_traj = gid.x;
    let node_local = gid.y;
    if local_traj >= uniforms.tile_size || node_local >= uniforms.level_count {
        return;
    }
    let global_traj = uniforms.tile_offset + local_traj;
    let node_idx = level_nodes[uniforms.level_start + node_local];
    let node = nodes[node_idx];

    // Skip terminals
    if node.node_type >= 2u {
        return;
    }

    let ts = uniforms.tile_size;

    if node.node_type == uniforms.player {
        // ---- Own player's decision node ----

        let info_id = info_id_table[local_traj * uniforms.n_decision + node.decision_idx];
        let strat_base = info_id * uniforms.max_actions;

        // Compute node utility (weighted average by strategy)
        var node_util: f32 = 0.0;
        for (var a = 0u; a < node.num_children; a++) {
            let child_idx = children_arr[node.first_child + a];
            node_util += strategy[strat_base + a] * util_own[child_idx * ts + local_traj];
        }
        util_own[node_idx * ts + local_traj] = node_util;

        // Accumulate regrets (atomic CAS)
        for (var a = 0u; a < node.num_children; a++) {
            let child_idx = children_arr[node.first_child + a];
            let child_util = util_own[child_idx * ts + local_traj];
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
            let my_reach = reach_own[node_idx * ts + local_traj];
            // weight_sum layout: [p1_weights..., p2_weights...]
            // When player=0: offset = 0, when player=1: offset = n_traj_opp (which is n1)
            var ws_base: u32 = 0u;
            if uniforms.player == 1u {
                ws_base = uniforms.n_traj_opp;
            }
            let w = weight_sum[ws_base + global_traj];
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
    } else {
        // ---- Opponent's decision node: simple sum ----
        var sum_util: f32 = 0.0;
        for (var a = 0u; a < node.num_children; a++) {
            let child_idx = children_arr[node.first_child + a];
            sum_util += util_own[child_idx * ts + local_traj];
        }
        util_own[node_idx * ts + local_traj] = sum_util;
    }
}
