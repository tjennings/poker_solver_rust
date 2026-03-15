// Block-parallel showdown evaluation kernel using cooperative prefix-sum.
//
// Replaces the O(n) sorted kernel that launched one THREAD per (terminal, spot)
// with a kernel that launches one BLOCK per (terminal, spot). This massively
// increases GPU occupancy (from ~5000 threads to ~5000 blocks × 1024 threads).
//
// Algorithm per block:
//   1. All threads cooperatively load opponent reach into shared memory (coalesced).
//   2. All threads cooperatively accumulate per-card totals via atomicAdd in shared mem.
//   3. Thread 0 performs TWO sequential scans on shared memory:
//      - Ascending: for each opponent hand (weakest to strongest), record prefix sum
//        and per-card prefix sums at each traverser hand's rank position.
//      - Descending: same but strongest to weakest (suffix sum).
//      These scans interleave traverser and opponent hands by strength (two-pointer merge)
//      and maintain 53-element per-card accumulators for inclusion-exclusion blocking.
//      Results are written to shared memory arrays indexed by traverser local hand index.
//   4. All threads read their win/lose reach from shared memory and compute CFV.
//
// The serial scan by thread 0 takes ~2μs for 1326 hands (shared memory, fast).
// The parallel lookup/write by 1024 threads takes ~1μs.
// Total: ~3μs per block. With 5000 blocks on 142 SMs: 5000/142 ≈ 35 waves × 3μs ≈ 0.1ms.
//
// Launch: num_showdown_terminals * num_spots blocks
// Block: min(1024, hands_per_spot) threads
// Shared memory: see layout below

// Clamp card index to [0, 52] — real cards pass through, padded cards (255) go to bucket 52.
__device__ __forceinline__ unsigned int card_idx(unsigned int c) {
    return (c < 52u) ? c : 52u;
}

extern "C" __global__ void showdown_eval_block_scan(
    float* cfvalues,                        // [num_nodes * num_hands] output
    const float* opp_reach,                 // [num_nodes * num_hands]
    const unsigned int* terminal_nodes,     // [num_sd_terminals]
    const float* amount_win,                // [num_sd_terminals * num_hands]
    const float* amount_lose,               // [num_sd_terminals * num_hands]
    const unsigned int* trav_sorted,        // [num_spots * hands_per_spot] traverser hand indices sorted by strength (ascending)
    const unsigned int* opp_sorted,         // [num_spots * hands_per_spot] opponent hand indices sorted by strength (ascending)
    const unsigned int* trav_strengths,     // [num_hands] traverser hand strengths
    const unsigned int* opp_strengths,      // [num_hands] opponent hand strengths
    const unsigned int* trav_hand_cards,    // [num_hands * 2] traverser (c1, c2) per hand
    const unsigned int* opp_hand_cards,     // [num_hands * 2] opponent (c1, c2) per hand
    unsigned int num_sd_terminals,
    unsigned int num_hands,
    unsigned int hands_per_spot
) {
    // Shared memory layout:
    //   [0 .. hps):              sorted opponent reach (loaded cooperatively)
    //   [hps .. 2*hps):          win_reach per traverser local hand (written by thread 0, read by all)
    //   [2*hps .. 3*hps):        lose_reach per traverser local hand (written by thread 0, read by all)
    // Total: 3 * hps floats
    extern __shared__ float shm[];

    unsigned int hps = hands_per_spot;
    unsigned int num_spots = num_hands / hps;
    unsigned int term_idx = blockIdx.x / num_spots;
    unsigned int spot = blockIdx.x % num_spots;
    if (term_idx >= num_sd_terminals) return;

    unsigned int node = terminal_nodes[term_idx];
    unsigned int spot_start = spot * hps;
    unsigned int sort_base = spot * hps;

    float* sorted_reach = &shm[0];        // [hps]
    float* win_reach    = &shm[hps];       // [hps] — per traverser local hand
    float* lose_reach   = &shm[2u * hps];  // [hps] — per traverser local hand

    // === Step 1: Cooperatively load opponent reach in SORTED order ===
    // sorted_reach[i] = reach of the i-th weakest opponent hand
    for (unsigned int i = threadIdx.x; i < hps; i += blockDim.x) {
        unsigned int opp_local = opp_sorted[sort_base + i];
        unsigned int opp_global = spot_start + opp_local;
        sorted_reach[i] = opp_reach[node * num_hands + opp_global];
    }
    // Zero out win_reach and lose_reach
    for (unsigned int i = threadIdx.x; i < hps; i += blockDim.x) {
        win_reach[i] = 0.0f;
        lose_reach[i] = 0.0f;
    }
    __syncthreads();

    // === Step 2: Thread 0 performs the two-pass scan ===
    // This is the same algorithm as the sorted O(n) kernel, but operating
    // entirely on shared memory. It writes results indexed by traverser LOCAL hand
    // (not sorted position) so all threads can look up their result in O(1).
    if (threadIdx.x == 0) {
        // Per-card accumulator for inclusion-exclusion blocking (register/local memory).
        float cfreach_minus[53];

        // ========= ASCENDING PASS: weaker opponents → win reach =========
        float cfreach_sum = 0.0f;
        for (int c = 0; c < 53; c++) cfreach_minus[c] = 0.0f;

        unsigned int oi = 0; // opponent pointer into sorted order
        for (unsigned int ti = 0; ti < hps; ti++) {
            // Get traverser hand at this rank position (sorted ascending by strength)
            unsigned int trav_local = trav_sorted[sort_base + ti];
            unsigned int trav_global = spot_start + trav_local;
            unsigned int trav_strength = trav_strengths[trav_global];

            // Advance opponent pointer: accumulate all opponents strictly weaker
            while (oi < hps) {
                unsigned int opp_local = opp_sorted[sort_base + oi];
                unsigned int opp_global = spot_start + opp_local;
                unsigned int opp_str = opp_strengths[opp_global];
                if (opp_str >= trav_strength) break;

                float r = sorted_reach[oi]; // already in sorted order
                if (r != 0.0f) {
                    cfreach_sum += r;
                    unsigned int oc1 = card_idx(opp_hand_cards[opp_global * 2]);
                    unsigned int oc2 = card_idx(opp_hand_cards[opp_global * 2 + 1]);
                    cfreach_minus[oc1] += r;
                    cfreach_minus[oc2] += r;
                }
                oi++;
            }

            // Compute blocked win reach for this traverser hand
            unsigned int tc1 = card_idx(trav_hand_cards[trav_global * 2]);
            unsigned int tc2 = card_idx(trav_hand_cards[trav_global * 2 + 1]);
            float blocked = cfreach_sum - cfreach_minus[tc1] - cfreach_minus[tc2];

            // Store indexed by traverser LOCAL hand (not sorted position)
            win_reach[trav_local] = blocked;
        }

        // ========= DESCENDING PASS: stronger opponents → lose reach =========
        cfreach_sum = 0.0f;
        for (int c = 0; c < 53; c++) cfreach_minus[c] = 0.0f;

        int oi_desc = (int)hps - 1;
        for (int ti = (int)hps - 1; ti >= 0; ti--) {
            unsigned int trav_local = trav_sorted[sort_base + ti];
            unsigned int trav_global = spot_start + trav_local;
            unsigned int trav_strength = trav_strengths[trav_global];

            // Advance opponent pointer (descending): accumulate all opponents strictly stronger
            while (oi_desc >= 0) {
                unsigned int opp_local = opp_sorted[sort_base + oi_desc];
                unsigned int opp_global = spot_start + opp_local;
                unsigned int opp_str = opp_strengths[opp_global];
                if (opp_str <= trav_strength) break;

                float r = sorted_reach[oi_desc];
                if (r != 0.0f) {
                    cfreach_sum += r;
                    unsigned int oc1 = card_idx(opp_hand_cards[opp_global * 2]);
                    unsigned int oc2 = card_idx(opp_hand_cards[opp_global * 2 + 1]);
                    cfreach_minus[oc1] += r;
                    cfreach_minus[oc2] += r;
                }
                oi_desc--;
            }

            unsigned int tc1 = card_idx(trav_hand_cards[trav_global * 2]);
            unsigned int tc2 = card_idx(trav_hand_cards[trav_global * 2 + 1]);
            float blocked = cfreach_sum - cfreach_minus[tc1] - cfreach_minus[tc2];

            lose_reach[trav_local] = blocked;
        }
    }
    __syncthreads();

    // === Step 3: All threads compute CFV from shared memory results ===
    for (unsigned int local_hand = threadIdx.x; local_hand < hps; local_hand += blockDim.x) {
        unsigned int hand = spot_start + local_hand;
        float win = amount_win[term_idx * num_hands + hand];
        float lose = amount_lose[term_idx * num_hands + hand];
        float w = win_reach[local_hand];
        float l = lose_reach[local_hand];

        cfvalues[node * num_hands + hand] = win * w + lose * l;
    }
}
