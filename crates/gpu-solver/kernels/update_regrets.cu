// DCFR+ regret update kernel.
//
// For each (decision_node, action, hand) triple:
//   1. Computes instantaneous regret: cfv[child_a][hand] - cfv[node][hand]
//   2. Updates cumulative regret with DCFR discounting:
//        new_regret = old_regret * discount + instant_regret
//      where discount = pos_discount if old_regret >= 0, else neg_discount
//   3. Updates strategy sum: strategy_sum = strategy_sum * strat_discount + strategy
//
// One thread per (decision_node, action, hand) linearized triple.

extern "C" __global__ void update_regrets(
    float* regrets,
    float* strategy_sum,
    const float* strategy,
    const float* cfvalues,
    const unsigned int* decision_nodes,
    const unsigned int* child_offsets,
    const unsigned int* children_arr,
    const unsigned int* infoset_ids,
    const unsigned int* num_actions_arr,
    unsigned int num_decision_nodes,
    unsigned int num_hands,
    unsigned int max_actions,
    float pos_discount,
    float neg_discount,
    float strat_discount
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Linearize: (decision_node_local, action, hand)
    unsigned int total_per_node = max_actions * num_hands;
    unsigned int dec_local = tid / total_per_node;
    unsigned int remainder = tid % total_per_node;
    unsigned int action = remainder / num_hands;
    unsigned int hand = remainder % num_hands;

    if (dec_local >= num_decision_nodes) return;

    unsigned int node = decision_nodes[dec_local];
    unsigned int infoset = infoset_ids[node];

    // Skip if infoset is invalid (should not happen for decision nodes)
    if (infoset == 0xFFFFFFFF) return;

    unsigned int n_actions = num_actions_arr[infoset];
    if (action >= n_actions) return;

    unsigned int first_child = child_offsets[node];
    unsigned int child = children_arr[first_child + action];

    float child_cfv = cfvalues[child * num_hands + hand];
    float node_cfv = cfvalues[node * num_hands + hand];
    float inst_regret = child_cfv - node_cfv;

    unsigned int reg_idx = (infoset * max_actions + action) * num_hands + hand;

    // DCFR+ regret update with discounting
    float old_regret = regrets[reg_idx];
    float discount = (old_regret >= 0.0f) ? pos_discount : neg_discount;
    float new_regret = old_regret * discount + inst_regret;
    regrets[reg_idx] = new_regret;

    // Strategy sum update: discount old sum, add current strategy
    strategy_sum[reg_idx] = strategy_sum[reg_idx] * strat_discount + strategy[reg_idx];
}
