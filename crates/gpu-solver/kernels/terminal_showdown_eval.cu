// Terminal showdown evaluation kernel.
//
// For each showdown terminal node, computes the traverser's counterfactual value
// by comparing hand strengths against all opponent hands.
// One thread per (showdown_terminal, hand) pair.
//
// cfv[h] = sum over opponent hands h':
//   if strength[h] > strength[h']: amount_win * opp_reach[h']
//   if strength[h] < strength[h']: amount_lose * opp_reach[h']
//   if tied: 0 (no contribution)

extern "C" __global__ void terminal_showdown_eval(
    float* cfvalues,
    const float* opp_reach,
    const unsigned int* terminal_nodes,
    const float* amount_win,
    const float* amount_lose,
    const unsigned int* hand_strengths,
    unsigned int num_showdown_terminals,
    unsigned int num_hands
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int term_idx = tid / num_hands;
    unsigned int hand = tid % num_hands;
    if (term_idx >= num_showdown_terminals) return;

    unsigned int node = terminal_nodes[term_idx];
    unsigned int my_strength = hand_strengths[hand];

    float cfv = 0.0f;
    float win = amount_win[term_idx];
    float lose = amount_lose[term_idx];

    for (unsigned int opp = 0; opp < num_hands; opp++) {
        float opp_r = opp_reach[node * num_hands + opp];
        unsigned int opp_strength = hand_strengths[opp];

        if (my_strength > opp_strength) {
            cfv += win * opp_r;
        } else if (my_strength < opp_strength) {
            cfv += lose * opp_r;
        }
        // tie: no contribution
    }

    cfvalues[node * num_hands + hand] = cfv;
}
