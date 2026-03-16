// Bucketed showdown evaluation kernel.
//
// cfv[bucket_i] = (half_pot / reach_normalizer) * sum_j(equity[i][j] * opp_reach[j])
//
// reach_normalizer = total initial opponent reach (e.g., ~1081 for uniform ranges).
// This normalizes the CFV so fold and showdown payoffs are comparable.

extern "C" __global__ void bucketed_showdown_eval(
    float* cfvalues,
    const float* opp_reach,
    const unsigned int* terminal_nodes,
    const float* equity_tables,
    const float* half_pots,
    unsigned int num_sd_terminals,
    unsigned int total_hands,
    unsigned int num_buckets,
    float reach_normalizer
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_sd_terminals * total_hands;
    if (tid >= total) return;

    unsigned int term_idx = tid / total_hands;
    unsigned int hand = tid % total_hands;
    unsigned int bucket_i = hand % num_buckets;

    unsigned int node = terminal_nodes[term_idx];
    float hp = half_pots[term_idx];

    unsigned int eq_base = term_idx * num_buckets * num_buckets + bucket_i * num_buckets;

    float cfv = 0.0f;
    unsigned int reach_base = node * total_hands + (hand / num_buckets) * num_buckets;
    for (unsigned int j = 0; j < num_buckets; j++) {
        cfv += equity_tables[eq_base + j] * opp_reach[reach_base + j];
    }

    cfvalues[node * total_hands + hand] = (hp / reach_normalizer) * cfv;
}
