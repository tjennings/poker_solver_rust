// Showdown card-blocking correction kernel.
//
// The fast showdown kernel and the 3-kernel O(n) pipeline skip card blocking:
// they include opponent hands that share cards with the traverser. This kernel
// subtracts those incorrect contributions as an additive correction pass.
//
// For each traverser hand with cards (c1, c2), the blocked opponents are all
// opponent hands containing c1 or c2. These are found via a precomputed CSR
// (compressed sparse row) lookup: card_hand_offsets / card_hands.
//
// Since we iterate c1's and c2's blocked lists separately, any opponent hand
// containing BOTH c1 and c2 gets double-subtracted. We detect this by checking
// opp_hand_cards and add back one copy.
//
// Launch: ceil(num_sd_terminals * num_hands / 256) blocks, 256 threads.
// One thread per (terminal, traverser_hand).

extern "C" __global__ void showdown_block_correction(
    float* cfvalues,                       // [num_nodes * num_hands] — in-place correction
    const float* opp_reach,                // [num_nodes * num_hands]
    const unsigned int* terminal_nodes,    // [num_sd_terminals]
    const float* amount_win,               // [num_sd_terminals * num_hands] per-hand payoffs
    const float* amount_lose,              // [num_sd_terminals * num_hands]
    const unsigned int* traverser_strengths, // [num_hands]
    const unsigned int* opponent_strengths,  // [num_hands]
    const unsigned int* trav_hand_cards,   // [num_hands * 2] traverser's (c1, c2)
    const unsigned int* opp_hand_cards,    // [num_hands * 2] opponent's (c1, c2)
    const unsigned int* card_hand_offsets, // [num_spots * 53] CSR offsets into card_hands
    const unsigned int* card_hands,        // [total_entries] local opponent hand indices per card per spot
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
    unsigned int spot_start = spot * hands_per_spot;

    unsigned int my_strength = traverser_strengths[hand];
    float win = amount_win[term_idx * num_hands + hand];
    float lose = amount_lose[term_idx * num_hands + hand];

    unsigned int c1 = trav_hand_cards[hand * 2];
    unsigned int c2 = trav_hand_cards[hand * 2 + 1];

    // CSR base for this spot
    unsigned int csr_base = spot * 53;

    float correction = 0.0f;

    // Subtract contributions of opponent hands containing c1
    if (c1 < 52) {
        unsigned int start = card_hand_offsets[csr_base + c1];
        unsigned int end = card_hand_offsets[csr_base + c1 + 1];

        for (unsigned int idx = start; idx < end; idx++) {
            unsigned int opp_local = card_hands[idx];
            unsigned int opp = spot_start + opp_local;

            float opp_r = opp_reach[node * num_hands + opp];
            if (opp_r == 0.0f) continue;

            unsigned int opp_str = opponent_strengths[opp];

            if (my_strength > opp_str) {
                correction -= win * opp_r;
            } else if (my_strength < opp_str) {
                correction -= lose * opp_r;
            }
        }
    }

    // Subtract contributions of opponent hands containing c2
    if (c2 < 52) {
        unsigned int start = card_hand_offsets[csr_base + c2];
        unsigned int end = card_hand_offsets[csr_base + c2 + 1];

        for (unsigned int idx = start; idx < end; idx++) {
            unsigned int opp_local = card_hands[idx];
            unsigned int opp = spot_start + opp_local;

            float opp_r = opp_reach[node * num_hands + opp];
            if (opp_r == 0.0f) continue;

            unsigned int opp_str = opponent_strengths[opp];

            if (my_strength > opp_str) {
                correction -= win * opp_r;
            } else if (my_strength < opp_str) {
                correction -= lose * opp_r;
            }
        }
    }

    // Fix double-subtraction: opponent hands containing BOTH c1 and c2 were
    // subtracted twice. Add back one copy.
    //
    // Iterate c1's blocked list and check if each opponent also contains c2.
    // Since each card appears in ~50 hands and only ONE hand can contain both
    // c1 and c2 (the combo (c1,c2)), this loop finds at most one match.
    if (c1 < 52 && c2 < 52) {
        unsigned int start = card_hand_offsets[csr_base + c1];
        unsigned int end = card_hand_offsets[csr_base + c1 + 1];

        for (unsigned int idx = start; idx < end; idx++) {
            unsigned int opp_local = card_hands[idx];
            unsigned int opp = spot_start + opp_local;

            // Check if this opponent hand also contains c2
            unsigned int oc1 = opp_hand_cards[opp * 2];
            unsigned int oc2 = opp_hand_cards[opp * 2 + 1];

            if (oc1 == c2 || oc2 == c2) {
                // This opponent hand contains both c1 and c2 — double-subtracted.
                // Add back one subtraction.
                float opp_r = opp_reach[node * num_hands + opp];
                if (opp_r != 0.0f) {
                    unsigned int opp_str = opponent_strengths[opp];
                    if (my_strength > opp_str) {
                        correction += win * opp_r;
                    } else if (my_strength < opp_str) {
                        correction += lose * opp_r;
                    }
                }
                break; // at most one such hand
            }
        }
    }

    cfvalues[node * num_hands + hand] += correction;
}
