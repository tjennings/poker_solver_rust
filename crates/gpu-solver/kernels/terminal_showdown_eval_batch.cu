// Batch terminal showdown evaluation kernel.
//
// Like terminal_showdown_eval but with:
// 1. Per-hand payoffs: amount_win[term_idx * num_hands + hand]
// 2. Spot-scoped card blocking: hands_per_spot determines opponent range
//    valid_matchups is [num_spots * hps * hps], indexed as
//    spot * hps * hps + local_hand * hps + local_opp

extern "C" __global__ void terminal_showdown_eval_batch(
    float* cfvalues,
    const float* opp_reach,
    const unsigned int* terminal_nodes,
    const float* amount_win,
    const float* amount_lose,
    const unsigned int* traverser_strengths,
    const unsigned int* opponent_strengths,
    const float* valid_matchups,
    unsigned int num_showdown_terminals,
    unsigned int num_hands,
    unsigned int hands_per_spot
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int term_idx = tid / num_hands;
    unsigned int hand = tid % num_hands;
    if (term_idx >= num_showdown_terminals) return;

    unsigned int node = terminal_nodes[term_idx];
    unsigned int my_strength = traverser_strengths[hand];

    // Compute spot and local hand index
    unsigned int spot = hand / hands_per_spot;
    unsigned int local_hand = hand % hands_per_spot;
    unsigned int spot_start = spot * hands_per_spot;

    // Per-hand payoff lookup
    float win = amount_win[term_idx * num_hands + hand];
    float lose = amount_lose[term_idx * num_hands + hand];

    float cfv = 0.0f;
    for (unsigned int local_opp = 0; local_opp < hands_per_spot; local_opp++) {
        float valid = valid_matchups[spot * hands_per_spot * hands_per_spot + local_hand * hands_per_spot + local_opp];
        if (valid < 0.5f) continue;

        unsigned int opp_global = spot_start + local_opp;
        float opp_r = opp_reach[node * num_hands + opp_global];
        unsigned int opp_strength = opponent_strengths[opp_global];

        if (my_strength > opp_strength) {
            cfv += win * opp_r;
        } else if (my_strength < opp_strength) {
            cfv += lose * opp_r;
        }
        // tie: no contribution
    }

    cfvalues[node * num_hands + hand] = cfv;
}
