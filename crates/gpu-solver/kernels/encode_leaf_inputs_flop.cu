// GPU kernel for encoding reach probabilities at depth boundaries into
// 2720-dim neural network inputs for the turn CFV model.
//
// Variant of encode_leaf_inputs for FLOP training: boards have 3 cards
// (flop only) and `next_street_cards` are turn cards (49 possible).
//
// One thread per (boundary_idx, turn_idx, spot_idx) triple.
// Each thread writes one INPUT_SIZE=2720 element feature vector.
//
// Input layout (2720 floats):
//   [0..1326)      -- OOP reach at this boundary (zeroed for turn conflicts)
//   [1326..2652)   -- IP reach at this boundary (zeroed for turn conflicts)
//   [2652..2704)   -- board one-hot (52 elements: 3 flop + 1 turn card)
//   [2704..2717)   -- rank presence (13 elements)
//   [2717]         -- pot / 400.0
//   [2718]         -- effective_stack / 400.0
//   [2719]         -- traverser indicator (0.0=OOP, 1.0=IP)
//
// Supports per-spot boards: each spot may have a different 3-card flop board.
// `boards` is [num_spots * 3] containing the board cards for each spot.
// `next_street_cards` is [num_spots * num_next_cards] containing possible
// turn cards per spot (since different boards yield different turn card sets).

extern "C" __global__ void encode_leaf_inputs_flop(
    float* output,                          // [total_inputs * 2720]
    const float* reach_oop,                 // [num_nodes * total_hands]
    const float* reach_ip,                  // [num_nodes * total_hands]
    const unsigned int* boundary_nodes,     // [num_boundaries] -- node IDs
    const unsigned int* boards,             // [num_spots * 3] -- per-spot flop boards
    const unsigned int* next_street_cards,  // [num_spots * num_next_cards] -- per-spot turn cards
    const float* boundary_pots,             // [num_boundaries]
    const float* boundary_stacks,           // [num_boundaries]
    const unsigned int* combo_cards,        // [1326 * 2]
    unsigned int traverser,                 // 0=OOP, 1=IP
    unsigned int num_boundaries,
    unsigned int num_next_cards,            // 49 for flop
    unsigned int num_spots,
    unsigned int total_hands,               // num_nodes dimension for reach
    unsigned int hands_per_spot
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_inputs = num_boundaries * num_next_cards * num_spots;
    if (tid >= total_inputs) return;

    // Decompose tid into (boundary_idx, next_card_idx, spot_idx)
    unsigned int spot_idx = tid % num_spots;
    unsigned int tmp = tid / num_spots;
    unsigned int next_card_idx = tmp % num_next_cards;
    unsigned int boundary_idx = tmp / num_next_cards;

    unsigned int node_id = boundary_nodes[boundary_idx];
    unsigned int next_card = next_street_cards[spot_idx * num_next_cards + next_card_idx];

    // Per-spot board cards (3 cards for flop)
    const unsigned int* spot_board = &boards[spot_idx * 3u];

    // Output pointer for this thread's 2720-dim vector
    float* out = &output[tid * 2720u];

    // Reach base offset for this boundary node + this spot
    unsigned int reach_base = node_id * total_hands + spot_idx * hands_per_spot;

    // Board card values
    unsigned int board0 = spot_board[0];
    unsigned int board1 = spot_board[1];
    unsigned int board2 = spot_board[2];

    // Write OOP range [0..1326), zeroing combos that conflict with turn card or board
    for (unsigned int c = 0; c < 1326u; c++) {
        unsigned int c1 = combo_cards[c * 2];
        unsigned int c2 = combo_cards[c * 2 + 1];
        int conflict = (c1 == next_card || c2 == next_card ||
                        c1 == board0 || c2 == board0 ||
                        c1 == board1 || c2 == board1 ||
                        c1 == board2 || c2 == board2);
        if (conflict) {
            out[c] = 0.0f;
        } else if (c < hands_per_spot) {
            out[c] = reach_oop[reach_base + c];
        } else {
            out[c] = 0.0f;
        }
    }

    // Write IP range [1326..2652), zeroing combos that conflict with turn card or board
    for (unsigned int c = 0; c < 1326u; c++) {
        unsigned int c1 = combo_cards[c * 2];
        unsigned int c2 = combo_cards[c * 2 + 1];
        int conflict = (c1 == next_card || c2 == next_card ||
                        c1 == board0 || c2 == board0 ||
                        c1 == board1 || c2 == board1 ||
                        c1 == board2 || c2 == board2);
        if (conflict) {
            out[1326u + c] = 0.0f;
        } else if (c < hands_per_spot) {
            out[1326u + c] = reach_ip[reach_base + c];
        } else {
            out[1326u + c] = 0.0f;
        }
    }

    // Board one-hot [2652..2704): 52 elements, set 3 flop cards + 1 turn card
    for (unsigned int i = 0; i < 52u; i++) {
        out[2652u + i] = 0.0f;
    }
    out[2652u + board0] = 1.0f;
    out[2652u + board1] = 1.0f;
    out[2652u + board2] = 1.0f;
    out[2652u + next_card] = 1.0f;

    // Rank presence [2704..2717): 13 elements
    for (unsigned int i = 0; i < 13u; i++) {
        out[2704u + i] = 0.0f;
    }
    out[2704u + board0 / 4u] = 1.0f;
    out[2704u + board1 / 4u] = 1.0f;
    out[2704u + board2 / 4u] = 1.0f;
    out[2704u + next_card / 4u] = 1.0f;

    // Pot, stack, traverser
    out[2717u] = boundary_pots[boundary_idx] / 400.0f;
    out[2718u] = boundary_stacks[boundary_idx] / 400.0f;
    out[2719u] = (traverser == 0) ? 0.0f : 1.0f;
}
