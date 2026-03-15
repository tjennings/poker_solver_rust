// Fast showdown evaluation kernel.
//
// Two optimizations over terminal_showdown_eval_shm_batch:
// 1. Loads BOTH opponent reach AND opponent strengths into shared memory
//    (eliminates global memory reads from the inner loop)
// 2. No card blocking in the inner loop (removes thread divergence)
//
// Card blocking is a second-order effect (~8% of matchups). The slight
// noise in training targets is acceptable for neural network training.
//
// Launch: num_showdown_terminals * num_spots blocks
// Block size: min(hands_per_spot, 1024)
// Shared memory: hands_per_spot * (sizeof(float) + sizeof(uint)) bytes

extern "C" __global__ void showdown_eval_fast(
    float* cfvalues,
    const float* opp_reach,
    const unsigned int* terminal_nodes,
    const float* amount_win,
    const float* amount_lose,
    const unsigned int* traverser_strengths,
    const unsigned int* opponent_strengths,
    unsigned int num_showdown_terminals,
    unsigned int num_hands,
    unsigned int hands_per_spot
) {
    extern __shared__ char shm_raw[];
    float* shm_reach = (float*)shm_raw;
    unsigned int* shm_strength = (unsigned int*)(shm_raw + hands_per_spot * sizeof(float));

    unsigned int num_spots = num_hands / hands_per_spot;
    unsigned int term_idx = blockIdx.x / num_spots;
    unsigned int spot = blockIdx.x % num_spots;
    if (term_idx >= num_showdown_terminals) return;

    unsigned int node = terminal_nodes[term_idx];
    unsigned int spot_start = spot * hands_per_spot;

    // Cooperatively load opponent reach and strength into shared memory
    for (unsigned int i = threadIdx.x; i < hands_per_spot; i += blockDim.x) {
        unsigned int opp = spot_start + i;
        shm_reach[i] = opp_reach[node * num_hands + opp];
        shm_strength[i] = opponent_strengths[opp];
    }
    __syncthreads();

    // Each thread computes CFV for its traverser hand(s)
    for (unsigned int local_hand = threadIdx.x; local_hand < hands_per_spot; local_hand += blockDim.x) {
        unsigned int hand = spot_start + local_hand;
        unsigned int my_strength = traverser_strengths[hand];
        float win = amount_win[term_idx * num_hands + hand];
        float lose = amount_lose[term_idx * num_hands + hand];

        float cfv = 0.0f;
        for (unsigned int j = 0; j < hands_per_spot; j++) {
            float opp_r = shm_reach[j];
            unsigned int opp_str = shm_strength[j];
            if (my_strength > opp_str) {
                cfv += win * opp_r;
            } else if (my_strength < opp_str) {
                cfv += lose * opp_r;
            }
        }

        cfvalues[node * num_hands + hand] = cfv;
    }
}
