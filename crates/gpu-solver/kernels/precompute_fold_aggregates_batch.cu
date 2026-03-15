// Batch precompute per-card reach aggregates for O(n) fold evaluation.
//
// Like precompute_fold_aggregates but for batch mode: one block per
// (terminal, spot) pair. Each block only iterates over its spot's hands,
// keeping shared memory atomics contention low.
//
// Outputs:
//   total_opp_reach[term_idx * num_spots + spot]  -- total opponent reach for this (terminal, spot)
//   per_card_reach[(term_idx * num_spots + spot) * 52 + card] -- opponent reach for hands containing `card`

extern "C" __global__ void precompute_fold_aggregates_batch(
    const float* opp_reach,                // [num_nodes * num_hands]
    const unsigned int* terminal_nodes,    // [num_fold_terminals]
    const unsigned int* opp_hand_cards,    // [num_hands * 2] -- opponent's (c1, c2) per hand
    float* total_opp_reach,                // [num_fold_terminals * num_spots] output
    float* per_card_reach,                 // [num_fold_terminals * num_spots * 52] output
    unsigned int num_fold_terminals,
    unsigned int num_hands,
    unsigned int hands_per_spot
) {
    // One block per (terminal, spot)
    unsigned int num_spots = num_hands / hands_per_spot;
    unsigned int term_idx = blockIdx.x / num_spots;
    unsigned int spot = blockIdx.x % num_spots;
    if (term_idx >= num_fold_terminals) return;

    unsigned int node = terminal_nodes[term_idx];
    unsigned int spot_start = spot * hands_per_spot;

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

    // Each thread processes a subset of this spot's opponent hands
    for (unsigned int i = threadIdx.x; i < hands_per_spot; i += blockDim.x) {
        unsigned int h = spot_start + i;
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

    // Write results to global memory -- indexed by (terminal, spot)
    unsigned int out_idx = term_idx * num_spots + spot;
    if (threadIdx.x == 0) {
        total_opp_reach[out_idx] = shm_total[0];
    }
    unsigned int card_base = out_idx * 52;
    for (unsigned int i = threadIdx.x; i < 52; i += blockDim.x) {
        per_card_reach[card_base + i] = shm_cards[i];
    }
}
