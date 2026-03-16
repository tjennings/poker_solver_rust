// Bucketed showdown evaluation kernel.
//
// In the bucketed solver, showdown payoffs are computed as a matrix-vector
// multiply: for each showdown terminal and traverser bucket i,
//
//   cfv[bucket_i] = half_pot * sum_j(equity[i][j] * opp_reach[j])
//
// where equity[i][j] is the precomputed average payoff for bucket i vs bucket j
// (in [-1, 1]) and opp_reach[j] is the opponent's reach probability for bucket j.
//
// No card blocking is needed -- blocking is already accounted for in the equity
// table precomputation.
//
// For single-spot mode: num_spots = 1, total_hands = num_buckets.
// For batch mode: different spots have different boards and equity tables.
//
// One thread per (terminal, hand) pair where hand = spot * num_buckets + bucket_i.

extern "C" __global__ void bucketed_showdown_eval(
    float* cfvalues,                    // [num_nodes * total_hands]
    const float* opp_reach,             // [num_nodes * total_hands]
    const unsigned int* terminal_nodes, // [num_sd_terminals]
    const float* equity_tables,         // [num_sd_terminals * num_buckets * num_buckets]
                                        // (single-spot: one table per terminal)
    const float* half_pots,             // [num_sd_terminals]
    unsigned int num_sd_terminals,
    unsigned int total_hands,           // = num_buckets for single-spot
    unsigned int num_buckets
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_sd_terminals * total_hands;
    if (tid >= total) return;

    unsigned int term_idx = tid / total_hands;
    unsigned int hand = tid % total_hands;
    unsigned int bucket_i = hand % num_buckets;

    unsigned int node = terminal_nodes[term_idx];
    float hp = half_pots[term_idx];

    // Equity table for this terminal:
    // equity_tables[term_idx * nb * nb + bucket_i * nb + j]
    unsigned int eq_base = term_idx * num_buckets * num_buckets + bucket_i * num_buckets;

    // Matrix-vector multiply: cfv = hp * sum_j(equity[i][j] * opp_reach_j)
    float cfv = 0.0f;
    unsigned int reach_base = node * total_hands + (hand / num_buckets) * num_buckets;
    for (unsigned int j = 0; j < num_buckets; j++) {
        cfv += equity_tables[eq_base + j] * opp_reach[reach_base + j];
    }

    cfvalues[node * total_hands + hand] = hp * cfv;
}
