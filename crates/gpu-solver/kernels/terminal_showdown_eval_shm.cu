// Shared-memory showdown evaluation kernel.
//
// Still O(n^2) in work per (terminal, block) but loads opponent reach into
// shared memory once per block, reducing global memory bandwidth by ~n×.
//
// Launch: one block per showdown terminal, blockDim.x = num_hands (or next
// power of 2). Shared memory = num_hands * sizeof(float).
//
// Each thread handles one traverser hand and iterates over all opponent hands
// in shared memory, comparing strengths and applying card blocking.

extern "C" __global__ void terminal_showdown_eval_shm(
    float* cfvalues,
    const float* opp_reach,
    const unsigned int* terminal_nodes,
    const float* amount_win,
    const float* amount_lose,
    const unsigned int* traverser_strengths,
    const unsigned int* opponent_strengths,
    const unsigned int* trav_hand_cards,    // [num_hands * 2]
    const unsigned int* opp_hand_cards,     // [num_hands * 2]
    unsigned int num_showdown_terminals,
    unsigned int num_hands
) {
    extern __shared__ float shm_opp_reach[];

    unsigned int term_idx = blockIdx.x;
    if (term_idx >= num_showdown_terminals) return;

    unsigned int node = terminal_nodes[term_idx];

    // Cooperatively load opponent reach into shared memory
    for (unsigned int i = threadIdx.x; i < num_hands; i += blockDim.x) {
        shm_opp_reach[i] = opp_reach[node * num_hands + i];
    }
    __syncthreads();

    // Each thread handles one traverser hand
    unsigned int hand = threadIdx.x;
    if (hand >= num_hands) return;

    unsigned int my_strength = traverser_strengths[hand];
    float win = amount_win[term_idx];
    float lose = amount_lose[term_idx];

    unsigned int c1 = trav_hand_cards[hand * 2];
    unsigned int c2 = trav_hand_cards[hand * 2 + 1];

    float cfv = 0.0f;

    for (unsigned int opp = 0; opp < num_hands; opp++) {
        // Card blocking: skip if any cards overlap
        unsigned int opp_c1 = opp_hand_cards[opp * 2];
        unsigned int opp_c2 = opp_hand_cards[opp * 2 + 1];

        if (c1 == opp_c1 || c1 == opp_c2 || c2 == opp_c1 || c2 == opp_c2) continue;

        float opp_r = shm_opp_reach[opp];
        unsigned int opp_strength = opponent_strengths[opp];

        if (my_strength > opp_strength) {
            cfv += win * opp_r;
        } else if (my_strength < opp_strength) {
            cfv += lose * opp_r;
        }
        // tie: no contribution
    }

    cfvalues[node * num_hands + hand] = cfv;
}
