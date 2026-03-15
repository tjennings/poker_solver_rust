// Batch O(1) per-hand fold CFV computation using precomputed aggregates.
//
// Like fold_eval_from_aggregates but for batch mode:
// - Per-hand payoffs: fold_amount_win[term_idx * num_hands + hand]
// - Aggregates indexed by (terminal, spot): total_opp_reach[term_idx * num_spots + spot]
//
// Uses inclusion-exclusion: for each traverser hand (c1, c2) in spot s:
//   blocked_reach = per_card_reach[c1] + per_card_reach[c2] - same_hand_reach
//   valid_opp_reach = total_opp_reach - blocked_reach
//   cfv = payoff * valid_opp_reach
//
// One thread per (fold_terminal, hand) pair.

extern "C" __global__ void fold_eval_from_aggregates_batch(
    float* cfvalues,                       // [num_nodes * num_hands]
    const float* opp_reach,                // [num_nodes * num_hands] -- for same_hand lookup
    const unsigned int* terminal_nodes,    // [num_fold_terminals]
    const float* fold_amount_win,          // [num_fold_terminals * num_hands] -- per-hand payoff
    const float* fold_amount_lose,         // [num_fold_terminals * num_hands] -- per-hand payoff
    const unsigned int* fold_player,       // [num_fold_terminals]
    const float* total_opp_reach,          // [num_fold_terminals * num_spots]
    const float* per_card_reach,           // [num_fold_terminals * num_spots * 52]
    const unsigned int* trav_hand_cards,   // [num_hands * 2] -- traverser's (c1, c2) per hand
    const unsigned int* same_hand_index,   // [num_hands] -- opponent hand index with same cards (0xFFFFFFFF if none)
    unsigned int traverser,
    unsigned int num_fold_terminals,
    unsigned int num_hands,
    unsigned int hands_per_spot
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int term_idx = tid / num_hands;
    unsigned int hand = tid % num_hands;
    if (term_idx >= num_fold_terminals) return;

    unsigned int node = terminal_nodes[term_idx];
    unsigned int num_spots = num_hands / hands_per_spot;
    unsigned int spot = hand / hands_per_spot;

    // Per-hand payoff lookup
    float payoff;
    if (fold_player[term_idx] == traverser) {
        payoff = fold_amount_lose[term_idx * num_hands + hand];
    } else {
        payoff = fold_amount_win[term_idx * num_hands + hand];
    }

    // Index into aggregates for this (terminal, spot)
    unsigned int agg_idx = term_idx * num_spots + spot;
    float total = total_opp_reach[agg_idx];

    unsigned int c1 = trav_hand_cards[hand * 2];
    unsigned int c2 = trav_hand_cards[hand * 2 + 1];

    // Inclusion-exclusion: subtract reach of opponent hands sharing our cards
    unsigned int card_base = agg_idx * 52;
    float blocked = 0.0f;
    if (c1 < 52) blocked += per_card_reach[card_base + c1];
    if (c2 < 52) blocked += per_card_reach[card_base + c2];

    // Add back same-hand reach (double-subtracted by per-card terms)
    unsigned int same_idx = same_hand_index[hand];
    float same_reach = 0.0f;
    if (same_idx != 0xFFFFFFFF) {
        same_reach = opp_reach[node * num_hands + same_idx];
    }

    cfvalues[node * num_hands + hand] = payoff * (total - blocked + same_reach);
}
