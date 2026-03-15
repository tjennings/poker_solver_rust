// Precompute per-card reach aggregates for O(n) fold evaluation.
//
// One thread per (fold_terminal, hand) pair accumulates opponent reach
// into per-card buckets using atomicAdd into shared memory.
//
// After this kernel, fold_eval_from_aggregates.cu can compute each
// hand's CFV in O(1) via inclusion-exclusion instead of O(n).
//
// Outputs:
//   total_opp_reach[term_idx]  — total opponent reach at this terminal
//   per_card_reach[term_idx * 52 + card] — opponent reach for hands containing `card`

extern "C" __global__ void precompute_fold_aggregates(
    const float* opp_reach,                // [num_nodes * num_hands]
    const unsigned int* terminal_nodes,    // [num_fold_terminals]
    const unsigned int* opp_hand_cards,    // [num_hands * 2] — opponent's (c1, c2) per hand
    float* total_opp_reach,                // [num_fold_terminals] output
    float* per_card_reach,                 // [num_fold_terminals * 52] output
    unsigned int num_fold_terminals,
    unsigned int num_hands
) {
    // Each block handles one fold terminal.
    // Threads within the block cooperate to sum over all opponent hands.
    unsigned int term_idx = blockIdx.x;
    if (term_idx >= num_fold_terminals) return;

    unsigned int node = terminal_nodes[term_idx];

    // Shared memory: 52 floats for per-card accumulation + 1 float for total
    __shared__ float shm_cards[52];
    __shared__ float shm_total[1];

    // Initialize shared memory
    for (unsigned int i = threadIdx.x; i < 52; i += blockDim.x) {
        shm_cards[i] = 0.0f;
    }
    if (threadIdx.x == 0) {
        shm_total[0] = 0.0f;
    }
    __syncthreads();

    // Each thread processes a subset of opponent hands
    for (unsigned int h = threadIdx.x; h < num_hands; h += blockDim.x) {
        float r = opp_reach[node * num_hands + h];
        if (r != 0.0f) {
            atomicAdd(&shm_total[0], r);
            unsigned int c1 = opp_hand_cards[h * 2];
            unsigned int c2 = opp_hand_cards[h * 2 + 1];
            if (c1 < 52) atomicAdd(&shm_cards[c1], r);
            if (c2 < 52) atomicAdd(&shm_cards[c2], r);
        }
    }
    __syncthreads();

    // Write results to global memory
    if (threadIdx.x == 0) {
        total_opp_reach[term_idx] = shm_total[0];
    }
    unsigned int base = term_idx * 52;
    for (unsigned int i = threadIdx.x; i < 52; i += blockDim.x) {
        per_card_reach[base + i] = shm_cards[i];
    }
}
