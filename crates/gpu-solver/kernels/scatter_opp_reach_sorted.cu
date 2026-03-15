// Kernel 1 of 3-kernel O(n log n) showdown evaluation.
//
// Scatters opponent reach probabilities from natural hand order into
// strength-sorted order. After this kernel, sorted_reach[seg][i] contains
// the reach of the i-th weakest opponent hand in that segment.
//
// Each segment is one (terminal, spot) pair with hands_per_spot elements.
//
// Launch: ceil(num_sd_terminals * num_hands / 256) blocks, 256 threads.
// Fully parallel — one thread per (terminal, hand).

extern "C" __global__ void scatter_opp_reach_sorted(
    float* sorted_reach,                    // [num_sd_terminals * num_hands] output
    const float* opp_reach,                 // [num_nodes * num_hands]
    const unsigned int* terminal_nodes,     // [num_sd_terminals]
    const unsigned int* sorted_indices,     // [num_spots * hands_per_spot] — sorted_indices[spot*hps + rank] = local hand index of rank-th weakest
    unsigned int num_sd_terminals,
    unsigned int num_hands,
    unsigned int hands_per_spot
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_sd_terminals * num_hands;
    if (tid >= total) return;

    unsigned int term_idx = tid / num_hands;
    unsigned int sorted_pos = tid % num_hands;  // position in sorted order (global across all spots)

    unsigned int node = terminal_nodes[term_idx];
    unsigned int spot = sorted_pos / hands_per_spot;
    unsigned int local_sorted_pos = sorted_pos % hands_per_spot;

    // sorted_indices[spot * hps + local_sorted_pos] = local hand index of the i-th weakest hand
    unsigned int local_hand = sorted_indices[spot * hands_per_spot + local_sorted_pos];
    unsigned int global_hand = spot * hands_per_spot + local_hand;

    sorted_reach[term_idx * num_hands + sorted_pos] = opp_reach[node * num_hands + global_hand];
}
