// Tabular CFR: forward pass — propagate factored reach probabilities.
//
// Dispatched per tree level (root to leaves).
// Thread per (decision_node_in_level, trajectory).
//
// For P1 decision nodes:
//   reach_p1[child][t1] = reach_p1[parent][t1] * strategy[a]  (t1 < n_traj_p1)
//   reach_p2[child][t2] = reach_p2[parent][t2]                (t2 < n_traj_p2, copy)
//
// For P2 decision nodes:
//   reach_p2[child][t2] = reach_p2[parent][t2] * strategy[a]  (t2 < n_traj_p2)
//   reach_p1[child][t1] = reach_p1[parent][t1]                (t1 < n_traj_p1, copy)

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

@group(2) @binding(0) var<storage, read_write> reach_p1: array<f32>;
@group(2) @binding(1) var<storage, read_write> reach_p2: array<f32>;
@group(2) @binding(4) var<storage, read_write> info_id_table: array<u32>;

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

    // Skip terminals (shouldn't be in decision level_nodes, but guard anyway)
    if node.node_type >= 2u {
        return;
    }

    let player = node.node_type; // 0 = P1, 1 = P2

    if player == 0u {
        // P1 decision node
        // Update reach_p1 for P1 trajectories
        if traj_id < uniforms.n_traj_p1 {
            let info_id = info_id_table[node.decision_idx * uniforms.max_traj + traj_id];
            let strat_base = info_id * uniforms.max_actions;
            let parent_r1 = reach_p1[node_idx * uniforms.n_traj_p1 + traj_id];
            for (var a = 0u; a < node.num_children; a++) {
                let child_idx = children_arr[node.first_child + a];
                let prob = strategy[strat_base + a];
                reach_p1[child_idx * uniforms.n_traj_p1 + traj_id] = parent_r1 * prob;
            }
        }
        // Copy reach_p2 unchanged to all children
        if traj_id < uniforms.n_traj_p2 {
            let parent_r2 = reach_p2[node_idx * uniforms.n_traj_p2 + traj_id];
            for (var a = 0u; a < node.num_children; a++) {
                let child_idx = children_arr[node.first_child + a];
                reach_p2[child_idx * uniforms.n_traj_p2 + traj_id] = parent_r2;
            }
        }
    } else {
        // P2 decision node
        // Update reach_p2 for P2 trajectories
        if traj_id < uniforms.n_traj_p2 {
            let info_id = info_id_table[node.decision_idx * uniforms.max_traj + traj_id];
            let strat_base = info_id * uniforms.max_actions;
            let parent_r2 = reach_p2[node_idx * uniforms.n_traj_p2 + traj_id];
            for (var a = 0u; a < node.num_children; a++) {
                let child_idx = children_arr[node.first_child + a];
                let prob = strategy[strat_base + a];
                reach_p2[child_idx * uniforms.n_traj_p2 + traj_id] = parent_r2 * prob;
            }
        }
        // Copy reach_p1 unchanged to all children
        if traj_id < uniforms.n_traj_p1 {
            let parent_r1 = reach_p1[node_idx * uniforms.n_traj_p1 + traj_id];
            for (var a = 0u; a < node.num_children; a++) {
                let child_idx = children_arr[node.first_child + a];
                reach_p1[child_idx * uniforms.n_traj_p1 + traj_id] = parent_r1;
            }
        }
    }
}
