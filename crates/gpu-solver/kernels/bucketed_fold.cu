// Bucketed fold evaluation kernel.
//
// In the bucketed solver, fold payoffs are simplified: no card blocking.
// Every bucket gets the same payoff structure:
//
//   cfv[bucket_i] = payoff * sum_j(opp_reach[j])
//
// where payoff = +half_pot if opponent folded, -half_pot if traverser folded,
// and the sum is over all opponent buckets (no card blocking in bucket space).
//
// For single-spot mode: num_spots = 1, total_hands = num_buckets.
//
// One thread per (terminal, hand) pair where hand = spot * num_buckets + bucket_i.

extern "C" __global__ void bucketed_fold_eval(
    float* cfvalues,                    // [num_nodes * total_hands]
    const float* opp_reach,             // [num_nodes * total_hands]
    const unsigned int* terminal_nodes, // [num_fold_terminals]
    const float* half_pots,             // [num_fold_terminals]
    const unsigned int* fold_player,    // [num_fold_terminals] - which player folded (0=OOP, 1=IP)
    unsigned int traverser,
    unsigned int num_fold_terminals,
    unsigned int total_hands,           // = num_buckets for single-spot
    unsigned int num_buckets
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_fold_terminals * total_hands;
    if (tid >= total) return;

    unsigned int term_idx = tid / total_hands;
    unsigned int hand = tid % total_hands;
    unsigned int spot = hand / num_buckets;

    unsigned int node = terminal_nodes[term_idx];
    float hp = half_pots[term_idx];

    // Sum ALL opponent bucket reach (no card blocking in bucket space)
    float opp_sum = 0.0f;
    unsigned int reach_base = node * total_hands + spot * num_buckets;
    for (unsigned int j = 0; j < num_buckets; j++) {
        opp_sum += opp_reach[reach_base + j];
    }

    // Payoff: positive if opponent folded (traverser wins),
    //         negative if traverser folded (traverser loses)
    float payoff = (fold_player[term_idx] == traverser) ? -hp : hp;
    cfvalues[node * total_hands + hand] = payoff * opp_sum;
}
