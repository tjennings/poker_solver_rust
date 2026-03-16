// Supremus DCFR+ regret update kernel.
//
// Differences from the standard DCFR+ `update_regrets` kernel:
//
// 1. **Regret discounting**: uses `t` directly (not `t-1`).
//      pos_discount = t^1.5 / (t^1.5 + 1)
//      neg_discount = 0.5 (unchanged)
//
// 2. **Strategy weighting**: additive linear with delay instead of multiplicative.
//      strategy_sum += max(0, t - delay) * strategy
//    This means early iterations (t <= delay) do not contribute to the
//    average strategy, and later iterations contribute proportional to
//    (t - delay).
//
// One thread per (decision_node, action, hand) linearized triple.

extern "C" __global__ void update_regrets_supremus(
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
    unsigned int iteration,
    unsigned int delay
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

    // Supremus DCFR+ regret discounting: uses t directly (not t-1)
    float t_f = (float)iteration;
    float t_alpha = t_f * sqrtf(t_f);  // t^1.5
    float pos_discount = t_alpha / (t_alpha + 1.0f);
    float neg_discount = 0.5f;

    float old_regret = regrets[reg_idx];
    float discount = (old_regret >= 0.0f) ? pos_discount : neg_discount;
    float new_regret = old_regret * discount + inst_regret;
    regrets[reg_idx] = new_regret;

    // Strategy sum: additive linear with delay
    // weight = max(0, iteration - delay)
    int strat_weight = (int)iteration - (int)delay;
    if (strat_weight > 0) {
        strategy_sum[reg_idx] += (float)strat_weight * strategy[reg_idx];
    }
}
