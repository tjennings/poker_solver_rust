// Batch terminal fold evaluation kernel.
//
// Like terminal_fold_eval but with:
// 1. Per-hand payoffs: fold_amount_win[term_idx * num_hands + hand]
// 2. Spot-scoped card blocking: hands_per_spot determines opponent range
//    valid_matchups is [num_spots * hps * hps], indexed as
//    spot * hps * hps + local_hand * hps + local_opp

extern "C" __global__ void terminal_fold_eval_batch(
    float* cfvalues,
    const float* opp_reach,
    const unsigned int* terminal_nodes,
    const float* fold_amount_win,
    const float* fold_amount_lose,
    const unsigned int* fold_player,
    const float* valid_matchups,
    unsigned int traverser,
    unsigned int num_fold_terminals,
    unsigned int num_hands,
    unsigned int hands_per_spot
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int term_idx = tid / num_hands;
    unsigned int hand = tid % num_hands;
    if (term_idx >= num_fold_terminals) return;

    unsigned int node = terminal_nodes[term_idx];

    // Compute spot and local hand index
    unsigned int spot = hand / hands_per_spot;
    unsigned int local_hand = hand % hands_per_spot;
    unsigned int spot_start = spot * hands_per_spot;

    // Sum opponent reach within the same spot only
    float opp_reach_sum = 0.0f;
    for (unsigned int local_opp = 0; local_opp < hands_per_spot; local_opp++) {
        float valid = valid_matchups[spot * hands_per_spot * hands_per_spot + local_hand * hands_per_spot + local_opp];
        if (valid < 0.5f) continue;
        unsigned int opp_global = spot_start + local_opp;
        opp_reach_sum += opp_reach[node * num_hands + opp_global];
    }

    // Per-hand payoff lookup
    float payoff;
    if (fold_player[term_idx] == traverser) {
        payoff = fold_amount_lose[term_idx * num_hands + hand];
    } else {
        payoff = fold_amount_win[term_idx * num_hands + hand];
    }

    cfvalues[node * num_hands + hand] = payoff * opp_reach_sum;
}
