// Tiled tabular CFR: forward pass for one player's reach tile.
//
// Propagates reach probabilities through the tree for tile_size trajectories
// of one player. At own-player decision nodes, applies strategy. At opponent
// decision nodes, copies reach unchanged to children.
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

@group(2) @binding(0) var<storage, read_write> reach_own: array<f32>;
@group(2) @binding(4) var<storage, read_write> info_id_table: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_traj = gid.x;
    let node_local = gid.y;
    if local_traj >= uniforms.tile_size || node_local >= uniforms.level_count {
        return;
    }
    let node_idx = level_nodes[uniforms.level_start + node_local];
    let node = nodes[node_idx];

    // Skip terminals (shouldn't be in decision level_nodes, but guard)
    if node.node_type >= 2u {
        return;
    }

    let ts = uniforms.tile_size;
    let parent_reach = reach_own[node_idx * ts + local_traj];

    if node.node_type == uniforms.player {
        // Own player's decision node: apply strategy
        let info_id = info_id_table[local_traj * uniforms.n_decision + node.decision_idx];
        let strat_base = info_id * uniforms.max_actions;
        for (var a = 0u; a < node.num_children; a++) {
            let child_idx = children_arr[node.first_child + a];
            let prob = strategy[strat_base + a];
            reach_own[child_idx * ts + local_traj] = parent_reach * prob;
        }
    } else {
        // Opponent's decision node: copy reach unchanged
        for (var a = 0u; a < node.num_children; a++) {
            let child_idx = children_arr[node.first_child + a];
            reach_own[child_idx * ts + local_traj] = parent_reach;
        }
    }
}
