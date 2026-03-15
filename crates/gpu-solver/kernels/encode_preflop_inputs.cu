// GPU kernel for encoding preflop training inputs for the flop model.
//
// For each (flop_idx, spot_idx), encode a 2720-dim input using the spot's
// ranges + the flop board. This allows batched flop model inference across
// all 22,100 possible flops for a given preflop situation.
//
// One thread per (flop_idx, spot_idx) pair.
// Each thread writes one INPUT_SIZE=2720 element feature vector.
//
// Input layout (2720 floats):
//   [0..1326)      -- OOP range (zeroed for flop-conflicting combos)
//   [1326..2652)   -- IP range (zeroed for flop-conflicting combos)
//   [2652..2704)   -- board one-hot (52 elements: 3 flop cards)
//   [2704..2717)   -- rank presence (13 elements)
//   [2717]         -- pot / 400.0
//   [2718]         -- effective_stack / 400.0
//   [2719]         -- traverser indicator (0.0=OOP, 1.0=IP)

extern "C" __global__ void encode_preflop_inputs(
    float* output,                    // [num_flops * num_spots * 2720]
    const float* ranges_oop,          // [num_spots * 1326]
    const float* ranges_ip,           // [num_spots * 1326]
    const unsigned int* all_flops,    // [num_flops * 3] -- all C(52,3) flop cards
    const unsigned int* combo_cards,  // [1326 * 2]
    const float* pots,                // [num_spots]
    const float* stacks,              // [num_spots]
    unsigned int traverser,
    unsigned int num_flops,           // 22100
    unsigned int num_spots
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_flops * num_spots;
    if (tid >= total) return;

    unsigned int spot_idx = tid % num_spots;
    unsigned int flop_idx = tid / num_spots;

    // Get this flop's 3 cards
    unsigned int f0 = all_flops[flop_idx * 3];
    unsigned int f1 = all_flops[flop_idx * 3 + 1];
    unsigned int f2 = all_flops[flop_idx * 3 + 2];

    float pot = pots[spot_idx];
    float stack = stacks[spot_idx];

    // Output pointer for this thread's 2720-dim vector
    float* out = &output[tid * 2720u];

    // Write OOP range [0..1326), zeroing combos that conflict with flop cards
    for (unsigned int c = 0; c < 1326u; c++) {
        unsigned int c1 = combo_cards[c * 2];
        unsigned int c2 = combo_cards[c * 2 + 1];
        int conflict = (c1 == f0 || c2 == f0 ||
                        c1 == f1 || c2 == f1 ||
                        c1 == f2 || c2 == f2);
        if (conflict) {
            out[c] = 0.0f;
        } else {
            out[c] = ranges_oop[spot_idx * 1326u + c];
        }
    }

    // Write IP range [1326..2652), zeroing combos that conflict with flop cards
    for (unsigned int c = 0; c < 1326u; c++) {
        unsigned int c1 = combo_cards[c * 2];
        unsigned int c2 = combo_cards[c * 2 + 1];
        int conflict = (c1 == f0 || c2 == f0 ||
                        c1 == f1 || c2 == f1 ||
                        c1 == f2 || c2 == f2);
        if (conflict) {
            out[1326u + c] = 0.0f;
        } else {
            out[1326u + c] = ranges_ip[spot_idx * 1326u + c];
        }
    }

    // Board one-hot [2652..2704): 52 elements, set 3 flop cards
    for (unsigned int i = 0; i < 52u; i++) {
        out[2652u + i] = 0.0f;
    }
    out[2652u + f0] = 1.0f;
    out[2652u + f1] = 1.0f;
    out[2652u + f2] = 1.0f;

    // Rank presence [2704..2717): 13 elements
    for (unsigned int i = 0; i < 13u; i++) {
        out[2704u + i] = 0.0f;
    }
    out[2704u + f0 / 4u] = 1.0f;
    out[2704u + f1 / 4u] = 1.0f;
    out[2704u + f2 / 4u] = 1.0f;

    // Pot, stack, traverser
    out[2717u] = pot / 400.0f;
    out[2718u] = stack / 400.0f;
    out[2719u] = (traverser == 0) ? 0.0f : 1.0f;
}
