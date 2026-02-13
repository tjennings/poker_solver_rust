// Backward pass: compute utilities at terminals, propagate up, accumulate regrets.
//
// Dispatched once per tree level (leaves to root). Each thread handles
// one (deal, node_in_level) pair.
//
// Bind groups:
//   group 0: uniforms (0), nodes (1), children (2), level_nodes (3)
//   group 1: strategy (1), regret_delta (3), strat_sum_delta (4)
//   group 2: deal_p1_wins (2), info_id_lookup (3), reach_p1 (4), reach_p2 (5), utility (6)

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
@group(1) @binding(3) var<storage, read_write> regret_delta: array<atomic<u32>>;
@group(1) @binding(4) var<storage, read_write> strat_sum_delta: array<atomic<u32>>;

@group(2) @binding(2) var<storage, read_write> deal_p1_equity: array<f32>;
@group(2) @binding(3) var<storage, read_write> info_id_lookup: array<u32>;
@group(2) @binding(4) var<storage, read_write> reach_p1: array<f32>;
@group(2) @binding(5) var<storage, read_write> reach_p2: array<f32>;
@group(2) @binding(6) var<storage, read_write> utility_p1: array<f32>;
@group(2) @binding(7) var<storage, read_write> deal_weight: array<f32>;

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
    let deal_base = deal_idx * uniforms.num_nodes;

    // Terminal node: compute utility
    if node.node_type >= 2u {
        var util: f32 = 0.0;
        if node.node_type == 2u {
            // P1 folded: P1 loses investment
            util = -node.p1_invested_bb;
        } else if node.node_type == 3u {
            // P2 folded: P1 gains P2's investment
            util = node.p2_invested_bb;
        } else {
            // Showdown: interpolate between win/lose utility using equity
            let equity = deal_p1_equity[deal_idx];
            let win_util = node.p2_invested_bb;     // P1 wins: gains P2's investment
            let lose_util = -node.p1_invested_bb;   // P1 loses: loses own investment
            util = equity * win_util + (1.0 - equity) * lose_util;
        }
        utility_p1[deal_base + node_idx] = util;
        return;
    }

    // Decision node
    let info_id = info_id_lookup[deal_idx * uniforms.num_nodes + node_idx];
    let strat_base = info_id * uniforms.max_actions;
    let player = node.node_type; // 0 = P1, 1 = P2

    // Compute node utility as weighted sum of child utilities
    var node_util: f32 = 0.0;
    for (var a = 0u; a < node.num_children; a++) {
        let child_idx = children_arr[node.first_child + a];
        node_util += strategy[strat_base + a] * utility_p1[deal_base + child_idx];
    }
    utility_p1[deal_base + node_idx] = node_util;

    // Opponent and own reach probabilities
    var opp_reach: f32;
    var my_reach: f32;
    if player == 0u {
        opp_reach = reach_p2[deal_base + node_idx];
        my_reach = reach_p1[deal_base + node_idx];
    } else {
        opp_reach = reach_p1[deal_base + node_idx];
        my_reach = reach_p2[deal_base + node_idx];
    }

    // Deal weight for scaling regrets and strategy sums
    let w = deal_weight[deal_idx];

    // Accumulate counterfactual regrets via CAS-based atomic f32 add
    for (var a = 0u; a < node.num_children; a++) {
        let child_idx = children_arr[node.first_child + a];
        let child_util = utility_p1[deal_base + child_idx];
        var cf_regret: f32;
        if player == 0u {
            cf_regret = opp_reach * (child_util - node_util) * w;
        } else {
            cf_regret = opp_reach * (node_util - child_util) * w;
        }

        // Inline atomic f32 add for regret_delta
        let rd_idx = strat_base + a;
        var rd_old = atomicLoad(&regret_delta[rd_idx]);
        loop {
            let rd_new = bitcast<f32>(rd_old) + cf_regret;
            let rd_result = atomicCompareExchangeWeak(&regret_delta[rd_idx], rd_old, bitcast<u32>(rd_new));
            if rd_result.exchanged {
                break;
            }
            rd_old = rd_result.old_value;
        }
    }

    // Accumulate strategy sum (for average strategy)
    if uniforms.strategy_discount > 0.0 {
        for (var a = 0u; a < node.num_children; a++) {
            let contribution = my_reach * strategy[strat_base + a] * uniforms.strategy_discount * w;

            // Inline atomic f32 add for strat_sum_delta
            let ss_idx = strat_base + a;
            var ss_old = atomicLoad(&strat_sum_delta[ss_idx]);
            loop {
                let ss_new = bitcast<f32>(ss_old) + contribution;
                let ss_result = atomicCompareExchangeWeak(&strat_sum_delta[ss_idx], ss_old, bitcast<u32>(ss_new));
                if ss_result.exchanged {
                    break;
                }
                ss_old = ss_result.old_value;
            }
        }
    }
}
