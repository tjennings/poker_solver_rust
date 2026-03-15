// Terminal fold evaluation kernel.
//
// For each fold terminal node, computes the traverser's counterfactual value.
// One thread per (fold_terminal, hand) pair.
//
// If the traverser folded, they receive amount_lose * total_opp_reach.
// If the opponent folded, the traverser receives amount_win * total_opp_reach.
//
// Card blocking: only sum opponent reach for hands that don't share cards
// with the traverser's hand (valid_matchups).

extern "C" __global__ void terminal_fold_eval(
    float* cfvalues,
    const float* opp_reach,
    const unsigned int* terminal_nodes,
    const float* fold_amount_win,
    const float* fold_amount_lose,
    const unsigned int* fold_player,
    const float* valid_matchups,
    unsigned int traverser,
    unsigned int num_fold_terminals,
    unsigned int num_hands
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int term_idx = tid / num_hands;
    unsigned int hand = tid % num_hands;
    if (term_idx >= num_fold_terminals) return;

    unsigned int node = terminal_nodes[term_idx];

    // Sum opponent reach at this node, excluding blocked hands
    float opp_reach_sum = 0.0f;
    for (unsigned int h = 0; h < num_hands; h++) {
        float valid = valid_matchups[hand * num_hands + h];
        if (valid < 0.5f) continue;
        opp_reach_sum += opp_reach[node * num_hands + h];
    }

    // Determine payoff based on who folded
    float payoff;
    if (fold_player[term_idx] == traverser) {
        payoff = fold_amount_lose[term_idx];
    } else {
        payoff = fold_amount_win[term_idx];
    }

    cfvalues[node * num_hands + hand] = payoff * opp_reach_sum;
}
