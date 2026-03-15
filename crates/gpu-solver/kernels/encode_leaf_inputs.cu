// GPU kernel for encoding reach probabilities at depth boundaries into
// 2720-dim neural network inputs for the river CFV model.
//
// One thread per (boundary_idx, river_idx, spot_idx) triple.
// Each thread writes one INPUT_SIZE=2720 element feature vector.
//
// Input layout (2720 floats):
//   [0..1326)      — OOP reach at this boundary (zeroed for river conflicts)
//   [1326..2652)   — IP reach at this boundary (zeroed for river conflicts)
//   [2652..2704)   — board one-hot (52 elements: 4 turn + 1 river card)
//   [2704..2717)   — rank presence (13 elements)
//   [2717]         — pot / 400.0
//   [2718]         — effective_stack / 400.0
//   [2719]         — traverser indicator (0.0=OOP, 1.0=IP)

extern "C" __global__ void encode_leaf_inputs(
    float* output,                      // [total_inputs * 2720]
    const float* reach_oop,             // [num_nodes * total_hands]
    const float* reach_ip,              // [num_nodes * total_hands]
    const unsigned int* boundary_nodes, // [num_boundaries] — node IDs
    const unsigned int* turn_board,     // [4] — the 4 turn board cards
    const unsigned int* river_cards,    // [num_rivers] — possible river cards
    const float* boundary_pots,         // [num_boundaries]
    const float* boundary_stacks,       // [num_boundaries]
    const unsigned int* combo_cards,    // [1326 * 2]
    unsigned int traverser,             // 0=OOP, 1=IP
    unsigned int num_boundaries,
    unsigned int num_rivers,
    unsigned int num_spots,
    unsigned int total_hands,           // num_nodes dimension for reach
    unsigned int hands_per_spot
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_inputs = num_boundaries * num_rivers * num_spots;
    if (tid >= total_inputs) return;

    // Decompose tid into (boundary_idx, river_idx, spot_idx)
    unsigned int spot_idx = tid % num_spots;
    unsigned int tmp = tid / num_spots;
    unsigned int river_idx = tmp % num_rivers;
    unsigned int boundary_idx = tmp / num_rivers;

    unsigned int node_id = boundary_nodes[boundary_idx];
    unsigned int river_card = river_cards[river_idx];

    // Output pointer for this thread's 2720-dim vector
    float* out = &output[tid * 2720u];

    // Reach base offset for this boundary node + this spot
    unsigned int reach_base = node_id * total_hands + spot_idx * hands_per_spot;

    // Write OOP range [0..1326), zeroing combos that conflict with river card
    for (unsigned int c = 0; c < 1326u; c++) {
        unsigned int c1 = combo_cards[c * 2];
        unsigned int c2 = combo_cards[c * 2 + 1];
        if (c1 == river_card || c2 == river_card) {
            out[c] = 0.0f;
        } else if (c < hands_per_spot) {
            out[c] = reach_oop[reach_base + c];
        } else {
            out[c] = 0.0f;
        }
    }

    // Write IP range [1326..2652), zeroing combos that conflict with river card
    for (unsigned int c = 0; c < 1326u; c++) {
        unsigned int c1 = combo_cards[c * 2];
        unsigned int c2 = combo_cards[c * 2 + 1];
        if (c1 == river_card || c2 == river_card) {
            out[1326u + c] = 0.0f;
        } else if (c < hands_per_spot) {
            out[1326u + c] = reach_ip[reach_base + c];
        } else {
            out[1326u + c] = 0.0f;
        }
    }

    // Board one-hot [2652..2704): 52 elements, set 4 turn cards + 1 river card
    for (unsigned int i = 0; i < 52u; i++) {
        out[2652u + i] = 0.0f;
    }
    for (unsigned int i = 0; i < 4u; i++) {
        out[2652u + turn_board[i]] = 1.0f;
    }
    out[2652u + river_card] = 1.0f;

    // Rank presence [2704..2717): 13 elements
    for (unsigned int i = 0; i < 13u; i++) {
        out[2704u + i] = 0.0f;
    }
    for (unsigned int i = 0; i < 4u; i++) {
        out[2704u + turn_board[i] / 4u] = 1.0f;
    }
    out[2704u + river_card / 4u] = 1.0f;

    // Pot, stack, traverser
    out[2717u] = boundary_pots[boundary_idx] / 400.0f;
    out[2718u] = boundary_stacks[boundary_idx] / 400.0f;
    out[2719u] = (traverser == 0) ? 0.0f : 1.0f;
}
