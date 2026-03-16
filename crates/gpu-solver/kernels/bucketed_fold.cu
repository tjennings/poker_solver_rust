// Bucketed fold evaluation kernel.
//
// cfv[bucket_i] = (payoff / reach_normalizer) * sum_j(opp_reach[j])
//
// reach_normalizer = total initial opponent reach (e.g., ~1081 for uniform ranges).
// This normalizes the CFV so fold and showdown payoffs are comparable
// (analogous to dividing by num_combinations in the concrete solver).

extern "C" __global__ void bucketed_fold_eval(
    float* cfvalues,
    const float* opp_reach,
    const unsigned int* terminal_nodes,
    const float* half_pots,
    const unsigned int* fold_player,
    unsigned int traverser,
    unsigned int num_fold_terminals,
    unsigned int total_hands,
    unsigned int num_buckets,
    float reach_normalizer
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_fold_terminals * total_hands;
    if (tid >= total) return;

    unsigned int term_idx = tid / total_hands;
    unsigned int hand = tid % total_hands;
    unsigned int spot = hand / num_buckets;

    unsigned int node = terminal_nodes[term_idx];
    float hp = half_pots[term_idx];

    float opp_sum = 0.0f;
    unsigned int reach_base = node * total_hands + spot * num_buckets;
    for (unsigned int j = 0; j < num_buckets; j++) {
        opp_sum += opp_reach[reach_base + j];
    }

    float payoff = (fold_player[term_idx] == traverser) ? -hp : hp;
    cfvalues[node * total_hands + hand] = (payoff / reach_normalizer) * opp_sum;
}
