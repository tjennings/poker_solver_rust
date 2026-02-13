// Forward pass: propagate reach probabilities from parent to children.
//
// Dispatched once per tree level (root to leaves). Each thread handles
// one (deal, node_in_level) pair.
//
// Bind groups:
//   group 0: uniforms (0), nodes (1), children (2), level_nodes (3)
//   group 1: strategy (1)
//   group 2: info_id_lookup (3), reach_p1 (4), reach_p2 (5)

struct Uniforms {
    level_start: u32,
    level_count: u32,
    num_deals: u32,
    num_nodes: u32,
    num_info_sets: u32,
    max_actions: u32,
    num_hand_classes: u32,
    iteration: u32,
    dcfr_alpha: f32,
    dcfr_beta: f32,
    dcfr_gamma: f32,
    strategy_discount: f32,
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

@group(2) @binding(3) var<storage, read_write> info_id_lookup: array<u32>;
@group(2) @binding(4) var<storage, read_write> reach_p1: array<f32>;
@group(2) @binding(5) var<storage, read_write> reach_p2: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_id = gid.x;
    let total_threads = uniforms.num_deals * uniforms.level_count;
    if thread_id >= total_threads {
        return;
    }

    let deal_idx = thread_id / uniforms.level_count;
    let level_idx = thread_id % uniforms.level_count;
    let node_idx = level_nodes[uniforms.level_start + level_idx];
    let node = nodes[node_idx];

    // Skip terminals (node_type >= 2)
    if node.node_type >= 2u {
        return;
    }

    let info_id = info_id_lookup[deal_idx * uniforms.num_nodes + node_idx];
    let strat_base = info_id * uniforms.max_actions;
    let deal_base = deal_idx * uniforms.num_nodes;
    let player = node.node_type; // 0 = P1, 1 = P2

    let parent_reach_p1 = reach_p1[deal_base + node_idx];
    let parent_reach_p2 = reach_p2[deal_base + node_idx];

    for (var a = 0u; a < node.num_children; a++) {
        let child_idx = children_arr[node.first_child + a];
        let prob = strategy[strat_base + a];
        let child_off = deal_base + child_idx;

        if player == 0u {
            reach_p1[child_off] = parent_reach_p1 * prob;
            reach_p2[child_off] = parent_reach_p2;
        } else {
            reach_p1[child_off] = parent_reach_p1;
            reach_p2[child_off] = parent_reach_p2 * prob;
        }
    }
}
