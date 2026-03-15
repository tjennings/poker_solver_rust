// Kernel 3 of 3-kernel O(n) showdown evaluation.
//
// For each traverser hand, uses precomputed exclusive prefix sums to determine:
//   win_reach  = prefix_excl[rank_win]            (reach of all strictly weaker opponents)
//   lose_reach = total - prefix_excl[rank_next]   (reach of all strictly stronger opponents)
//
// rank_win[h]  = # of opponent hands with strength < traverser_strength[h]
// rank_next[h] = # of opponent hands with strength <= traverser_strength[h]
//
// The difference (rank_next - rank_win) is the number of tied opponents,
// which contribute nothing to CFV (same as the O(n^2) kernel).
//
// No card blocking (same approximation as the fast shared-memory kernel).
//
// Launch: ceil(num_sd_terminals * num_hands / 256) blocks, 256 threads.
// Fully parallel — one thread per (terminal, hand).

extern "C" __global__ void showdown_lookup_cfv(
    float* cfvalues,                        // [num_nodes * num_hands] output
    const float* prefix_excl,               // [num_sd_terminals * num_hands]
    const float* segment_totals,            // [num_sd_terminals * num_spots]
    const float* amount_win,                // [num_sd_terminals * num_hands]
    const float* amount_lose,               // [num_sd_terminals * num_hands]
    const unsigned int* terminal_nodes,     // [num_sd_terminals]
    const unsigned int* rank_win,           // [num_spots * hps] — # opp hands with strength < trav strength
    const unsigned int* rank_next,          // [num_spots * hps] — # opp hands with strength <= trav strength
    unsigned int num_sd_terminals,
    unsigned int num_hands,
    unsigned int hands_per_spot
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_sd_terminals * num_hands;
    if (tid >= total) return;

    unsigned int term_idx = tid / num_hands;
    unsigned int hand = tid % num_hands;

    unsigned int node = terminal_nodes[term_idx];
    unsigned int spot = hand / hands_per_spot;
    unsigned int local_hand = hand % hands_per_spot;
    unsigned int num_spots = num_hands / hands_per_spot;

    // Base offset into the prefix_excl array for this (terminal, spot) segment
    unsigned int seg_base = term_idx * num_hands + spot * hands_per_spot;

    // Total opponent reach for this segment
    float total_reach = segment_totals[term_idx * num_spots + spot];

    // Win reach = sum of reach of opponent hands strictly weaker than this traverser hand.
    // prefix_excl[rw] = sum of reach at sorted positions 0..rw-1.
    // When rw == hands_per_spot, all opponents are weaker, so win_reach = total_reach.
    // When rw == 0, no opponents are weaker, so win_reach = 0 (= prefix_excl[0]).
    unsigned int rw = rank_win[spot * hands_per_spot + local_hand];
    float win_reach;
    if (rw < hands_per_spot) {
        win_reach = prefix_excl[seg_base + rw];
    } else {
        win_reach = total_reach;
    }

    // Lose reach = sum of reach of opponent hands strictly stronger than this traverser hand.
    // prefix_excl[rn] = sum of reach at sorted positions 0..rn-1 (= weaker + tied opponents).
    // total - prefix_excl[rn] = reach of opponents at positions rn..end = strictly stronger.
    // When rn == hands_per_spot, no opponents are stronger, so lose_reach = 0.
    unsigned int rn = rank_next[spot * hands_per_spot + local_hand];
    float prefix_at_next;
    if (rn < hands_per_spot) {
        prefix_at_next = prefix_excl[seg_base + rn];
    } else {
        prefix_at_next = total_reach;
    }
    float lose_reach = total_reach - prefix_at_next;

    float win = amount_win[term_idx * num_hands + hand];
    float lose = amount_lose[term_idx * num_hands + hand];

    cfvalues[node * num_hands + hand] = win * win_reach + lose * lose_reach;
}
