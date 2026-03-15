// O(n) showdown evaluation kernel using sorted prefix-sum approach.
//
// Replaces the O(n^2) shared-memory showdown kernel. Instead of each thread
// iterating over all opponent hands, a single thread per (terminal, spot)
// does two linear scans (ascending and descending) through interleaved
// sorted traverser/opponent hand lists, maintaining running per-card
// accumulators for inclusion-exclusion card blocking.
//
// Matches the CPU range-solver's 2-pass algorithm exactly:
// - Ascending pass: accumulate weaker opponents' reach -> compute win contribution
// - Descending pass: accumulate stronger opponents' reach -> compute lose contribution
//
// Launch: num_showdown_terminals * num_spots threads total.
// Each thread does O(hands_per_spot) work => O(n) total.
//
// NOTE: Uses 53-element per-card array. Cards 0-51 map to themselves; padded
// cards (value >= 52, typically 255) are clamped to index 52 as a catch-all.
// This correctly handles the inclusion-exclusion for padded hands: since both
// cards map to the same bucket (52), the subtraction works as if they share a
// card with themselves, matching the O(n^2) kernel's behavior.

// Clamp card index to [0, 52] — real cards pass through, padded cards (255) go to bucket 52.
__device__ __forceinline__ unsigned int card_idx(unsigned int c) {
    return (c < 52) ? c : 52;
}

extern "C" __global__ void showdown_eval_sorted_batch(
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
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int num_spots = num_hands / hands_per_spot;
    unsigned int term_idx = tid / num_spots;
    unsigned int spot = tid % num_spots;
    if (term_idx >= num_sd_terminals) return;

    unsigned int node = terminal_nodes[term_idx];
    unsigned int spot_start = spot * hands_per_spot;
    unsigned int sort_base = spot * hands_per_spot;

    // Per-card accumulator for inclusion-exclusion blocking.
    // 53 elements: indices 0-51 for real cards, index 52 for padded cards.
    float cfreach_minus[53];

    // =========================================================
    // ASCENDING PASS: accumulate weaker opponents -> win amount
    // =========================================================
    float cfreach_sum = 0.0f;
    for (int c = 0; c < 53; c++) cfreach_minus[c] = 0.0f;

    // Two pointers: ti into traverser sorted, oi into opponent sorted
    unsigned int oi = 0; // opponent pointer
    for (unsigned int ti = 0; ti < hands_per_spot; ti++) {
        // Get traverser hand at this rank position
        unsigned int trav_local = trav_sorted[sort_base + ti];
        unsigned int trav_global = spot_start + trav_local;
        unsigned int trav_strength = trav_strengths[trav_global];

        // Advance opponent pointer: accumulate all opponents weaker than traverser
        while (oi < hands_per_spot) {
            unsigned int opp_local = opp_sorted[sort_base + oi];
            unsigned int opp_global = spot_start + opp_local;
            unsigned int opp_str = opp_strengths[opp_global];
            if (opp_str >= trav_strength) break;

            float r = opp_reach[node * num_hands + opp_global];
            if (r != 0.0f) {
                cfreach_sum += r;
                unsigned int oc1 = card_idx(opp_hand_cards[opp_global * 2]);
                unsigned int oc2 = card_idx(opp_hand_cards[opp_global * 2 + 1]);
                cfreach_minus[oc1] += r;
                cfreach_minus[oc2] += r;
            }
            oi++;
        }

        // Compute win contribution with inclusion-exclusion card blocking
        unsigned int tc1 = card_idx(trav_hand_cards[trav_global * 2]);
        unsigned int tc2 = card_idx(trav_hand_cards[trav_global * 2 + 1]);
        float blocked = cfreach_sum - cfreach_minus[tc1] - cfreach_minus[tc2];

        float win = amount_win[term_idx * num_hands + trav_global];
        cfvalues[node * num_hands + trav_global] = win * blocked;
    }

    // ===========================================================
    // DESCENDING PASS: accumulate stronger opponents -> lose amount
    // ===========================================================
    cfreach_sum = 0.0f;
    for (int c = 0; c < 53; c++) cfreach_minus[c] = 0.0f;

    int oi_desc = (int)hands_per_spot - 1; // opponent pointer (descending)
    for (int ti = (int)hands_per_spot - 1; ti >= 0; ti--) {
        // Get traverser hand at this rank position
        unsigned int trav_local = trav_sorted[sort_base + ti];
        unsigned int trav_global = spot_start + trav_local;
        unsigned int trav_strength = trav_strengths[trav_global];

        // Advance opponent pointer (descending): accumulate all opponents stronger
        while (oi_desc >= 0) {
            unsigned int opp_local = opp_sorted[sort_base + oi_desc];
            unsigned int opp_global = spot_start + opp_local;
            unsigned int opp_str = opp_strengths[opp_global];
            if (opp_str <= trav_strength) break;

            float r = opp_reach[node * num_hands + opp_global];
            if (r != 0.0f) {
                cfreach_sum += r;
                unsigned int oc1 = card_idx(opp_hand_cards[opp_global * 2]);
                unsigned int oc2 = card_idx(opp_hand_cards[opp_global * 2 + 1]);
                cfreach_minus[oc1] += r;
                cfreach_minus[oc2] += r;
            }
            oi_desc--;
        }

        // Compute lose contribution with inclusion-exclusion card blocking
        unsigned int tc1 = card_idx(trav_hand_cards[trav_global * 2]);
        unsigned int tc2 = card_idx(trav_hand_cards[trav_global * 2 + 1]);
        float blocked = cfreach_sum - cfreach_minus[tc1] - cfreach_minus[tc2];

        float lose = amount_lose[term_idx * num_hands + trav_global];
        // Add to existing win contribution from ascending pass
        cfvalues[node * num_hands + trav_global] += lose * blocked;
    }
}
