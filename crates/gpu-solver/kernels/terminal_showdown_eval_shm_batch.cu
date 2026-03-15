// Batch shared-memory showdown evaluation kernel.
//
// Like terminal_showdown_eval_shm but for batch mode:
// - Per-hand payoffs: amount_win[term_idx * num_hands + hand]
// - Spot-scoped blocking: one block per (terminal, spot) pair
//
// Launch: num_showdown_terminals * num_spots blocks, each with
// min(hands_per_spot, 1024) threads. Shared memory = hands_per_spot * sizeof(float).
//
// Each block loads one spot's opponent reach into shared memory, then
// each thread computes its traverser hand's CFV by iterating over
// opponent hands within the same spot only.

extern "C" __global__ void terminal_showdown_eval_shm_batch(
    float* cfvalues,
    const float* opp_reach,
    const unsigned int* terminal_nodes,
    const float* amount_win,               // [num_terminals * num_hands]
    const float* amount_lose,              // [num_terminals * num_hands]
    const unsigned int* traverser_strengths,
    const unsigned int* opponent_strengths,
    const unsigned int* trav_hand_cards,   // [num_hands * 2]
    const unsigned int* opp_hand_cards,    // [num_hands * 2]
    unsigned int num_showdown_terminals,
    unsigned int num_hands,
    unsigned int hands_per_spot
) {
    extern __shared__ float shm_opp_reach[];

    // blockIdx.x = term_idx * num_spots + spot
    unsigned int num_spots = num_hands / hands_per_spot;
    unsigned int term_idx = blockIdx.x / num_spots;
    unsigned int spot = blockIdx.x % num_spots;
    if (term_idx >= num_showdown_terminals) return;

    unsigned int node = terminal_nodes[term_idx];
    unsigned int spot_start = spot * hands_per_spot;

    // Cooperatively load this spot's opponent reach into shared memory
    for (unsigned int i = threadIdx.x; i < hands_per_spot; i += blockDim.x) {
        shm_opp_reach[i] = opp_reach[node * num_hands + spot_start + i];
    }
    __syncthreads();

    // Each thread handles one traverser hand within this spot.
    // Use strided access if hands_per_spot > blockDim.x.
    for (unsigned int local_hand = threadIdx.x; local_hand < hands_per_spot; local_hand += blockDim.x) {
        unsigned int hand = spot_start + local_hand;
        unsigned int my_strength = traverser_strengths[hand];
        float win = amount_win[term_idx * num_hands + hand];
        float lose = amount_lose[term_idx * num_hands + hand];
        unsigned int c1 = trav_hand_cards[hand * 2];
        unsigned int c2 = trav_hand_cards[hand * 2 + 1];

        float cfv = 0.0f;

        for (unsigned int opp_local = 0; opp_local < hands_per_spot; opp_local++) {
            unsigned int opp = spot_start + opp_local;
            unsigned int opp_c1 = opp_hand_cards[opp * 2];
            unsigned int opp_c2 = opp_hand_cards[opp * 2 + 1];

            if (c1 == opp_c1 || c1 == opp_c2 || c2 == opp_c1 || c2 == opp_c2) continue;

            float opp_r = shm_opp_reach[opp_local];
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
}
