// Bucketed showdown evaluation kernel — batch variant.
//
// Like bucketed_showdown.cu but with per-spot equity tables and half-pots.
// Each spot has a different board (different equity table) and pot size.
//
// Data layout:
//   equity_tables: [num_sd_terminals * num_spots * num_buckets * num_buckets]
//   half_pots:     [num_sd_terminals * num_spots]
//
// total_hands = num_spots * num_buckets.
// One thread per (terminal, hand) pair where hand = spot * num_buckets + bucket_i.

extern "C" __global__ void bucketed_showdown_eval_batch(
    float* cfvalues,                    // [num_nodes * total_hands]
    const float* opp_reach,             // [num_nodes * total_hands]
    const unsigned int* terminal_nodes, // [num_sd_terminals]
    const float* equity_tables,         // [num_sd_terminals * num_spots * num_buckets * num_buckets]
    const float* half_pots,             // [num_sd_terminals * num_spots]
    unsigned int num_sd_terminals,
    unsigned int total_hands,           // = num_spots * num_buckets
    unsigned int num_buckets,
    unsigned int num_spots
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_sd_terminals * total_hands;
    if (tid >= total) return;

    unsigned int term_idx = tid / total_hands;
    unsigned int hand = tid % total_hands;
    unsigned int spot = hand / num_buckets;
    unsigned int bucket_i = hand % num_buckets;

    unsigned int node = terminal_nodes[term_idx];
    float hp = half_pots[term_idx * num_spots + spot];

    // Equity table for this terminal AND this spot:
    // equity_tables[(term_idx * num_spots + spot) * nb * nb + bucket_i * nb + j]
    unsigned int nb2 = num_buckets * num_buckets;
    unsigned int eq_base = (term_idx * num_spots + spot) * nb2 + bucket_i * num_buckets;

    // Matrix-vector multiply: cfv = hp * sum_j(equity[i][j] * opp_reach_j)
    float cfv = 0.0f;
    unsigned int reach_base = node * total_hands + spot * num_buckets;
    for (unsigned int j = 0; j < num_buckets; j++) {
        cfv += equity_tables[eq_base + j] * opp_reach[reach_base + j];
    }

    cfvalues[node * total_hands + hand] = hp * cfv;
}
