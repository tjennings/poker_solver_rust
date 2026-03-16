// Bucketed fold evaluation kernel — batch variant.
//
// Like bucketed_fold.cu but with per-spot half-pots.
// Each spot has a different pot size from stratified sampling.
//
// Data layout:
//   half_pots:    [num_fold_terminals * num_spots]
//   fold_player:  [num_fold_terminals]  (same for all spots — topology shared)
//
// total_hands = num_spots * num_buckets.
// One thread per (terminal, hand) pair where hand = spot * num_buckets + bucket_i.

extern "C" __global__ void bucketed_fold_eval_batch(
    float* cfvalues,                    // [num_nodes * total_hands]
    const float* opp_reach,             // [num_nodes * total_hands]
    const unsigned int* terminal_nodes, // [num_fold_terminals]
    const float* half_pots,             // [num_fold_terminals * num_spots]
    const unsigned int* fold_player,    // [num_fold_terminals]
    unsigned int traverser,
    unsigned int num_fold_terminals,
    unsigned int total_hands,           // = num_spots * num_buckets
    unsigned int num_buckets,
    unsigned int num_spots
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_fold_terminals * total_hands;
    if (tid >= total) return;

    unsigned int term_idx = tid / total_hands;
    unsigned int hand = tid % total_hands;
    unsigned int spot = hand / num_buckets;

    unsigned int node = terminal_nodes[term_idx];
    float hp = half_pots[term_idx * num_spots + spot];

    // Sum ALL opponent bucket reach for this spot (no card blocking in bucket space)
    float opp_sum = 0.0f;
    unsigned int reach_base = node * total_hands + spot * num_buckets;
    for (unsigned int j = 0; j < num_buckets; j++) {
        opp_sum += opp_reach[reach_base + j];
    }

    // Payoff: positive if opponent folded, negative if traverser folded
    float payoff = (fold_player[term_idx] == traverser) ? -hp : hp;
    cfvalues[node * total_hands + hand] = payoff * opp_sum;
}
