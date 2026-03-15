// GPU kernel to encode training records and insert into the reservoir buffer.
//
// Each situation produces two records: one from OOP's perspective (player=0)
// and one from IP's perspective (player=1).
//
// Input encoding (2720 floats):
//   [0..1326]:     OOP range weights
//   [1326..2652]:  IP range weights
//   [2652..2704]:  Board one-hot (52 cards)
//   [2704..2717]:  Rank presence (13 ranks)
//   [2717]:        Pot (normalised by 400)
//   [2718]:        Effective stack (normalised by 400)
//   [2719]:        Player indicator (0.0 = OOP, 1.0 = IP)
//
// Launch: ceil(num_spots * 2 / blockDim.x) blocks.
// One thread per record (num_spots * 2 total records).

extern "C" __global__ void reservoir_encode_insert(
    // Reservoir destination arrays
    float* res_inputs,           // [capacity * 2720]
    float* res_targets,          // [capacity * 1326]
    float* res_masks,            // [capacity * 1326]
    float* res_ranges,           // [capacity * 1326]
    float* res_game_values,      // [capacity]
    // Source data (all GPU-resident)
    const float* ranges_oop,     // [num_spots * 1326]
    const float* ranges_ip,      // [num_spots * 1326]
    const unsigned int* boards,  // [num_spots * 5]
    const float* pots,           // [num_spots]
    const float* stacks,         // [num_spots]
    const float* cfvs_oop,       // [num_spots * 1326] root CFVs
    const float* cfvs_ip,        // [num_spots * 1326]
    const unsigned int* combo_cards, // [1326 * 2] precomputed card pairs
    // Indexing
    unsigned int write_start,    // circular write position
    unsigned int capacity,
    unsigned int num_spots
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int num_records = num_spots * 2;
    if (tid >= num_records) return;

    unsigned int spot = tid / 2;
    unsigned int player = tid % 2;  // 0=OOP, 1=IP
    unsigned int dest = (write_start + tid) % capacity;

    // Load board cards for this spot
    unsigned int b[5];
    for (int i = 0; i < 5; i++) {
        b[i] = boards[spot * 5 + i];
    }

    float pot = pots[spot];
    float stack = stacks[spot];

    // === Encode input (2720 features) ===
    unsigned int in_base = dest * 2720;

    // [0..1326]: OOP range
    for (int i = 0; i < 1326; i++) {
        res_inputs[in_base + i] = ranges_oop[spot * 1326 + i];
    }

    // [1326..2652]: IP range
    for (int i = 0; i < 1326; i++) {
        res_inputs[in_base + 1326 + i] = ranges_ip[spot * 1326 + i];
    }

    // [2652..2704]: Board one-hot (52 cards) — first zero out
    for (int i = 0; i < 52; i++) {
        res_inputs[in_base + 2652 + i] = 0.0f;
    }
    for (int i = 0; i < 5; i++) {
        if (b[i] < 52) {
            res_inputs[in_base + 2652 + b[i]] = 1.0f;
        }
    }

    // [2704..2717]: Rank presence (13 ranks) — first zero out
    for (int i = 0; i < 13; i++) {
        res_inputs[in_base + 2704 + i] = 0.0f;
    }
    for (int i = 0; i < 5; i++) {
        if (b[i] < 52) {
            res_inputs[in_base + 2704 + b[i] / 4] = 1.0f;
        }
    }

    // [2717]: Pot normalised
    res_inputs[in_base + 2717] = pot / 400.0f;

    // [2718]: Stack normalised
    res_inputs[in_base + 2718] = stack / 400.0f;

    // [2719]: Player indicator
    res_inputs[in_base + 2719] = (float)player;

    // === Set target CFVs ===
    unsigned int tgt_base = dest * 1326;
    const float* cfvs = (player == 0) ? cfvs_oop : cfvs_ip;
    for (int i = 0; i < 1326; i++) {
        res_targets[tgt_base + i] = cfvs[spot * 1326 + i];
    }

    // === Set mask (1.0 for non-blocked combos, 0.0 for blocked) ===
    for (int i = 0; i < 1326; i++) {
        unsigned int c1 = combo_cards[i * 2];
        unsigned int c2 = combo_cards[i * 2 + 1];
        float mask = 1.0f;
        if (c1 == b[0] || c1 == b[1] || c1 == b[2] || c1 == b[3] || c1 == b[4] ||
            c2 == b[0] || c2 == b[1] || c2 == b[2] || c2 == b[3] || c2 == b[4]) {
            mask = 0.0f;
        }
        res_masks[tgt_base + i] = mask;
    }

    // === Set range (acting player's range for aux loss) ===
    const float* range = (player == 0) ? ranges_oop : ranges_ip;
    for (int i = 0; i < 1326; i++) {
        res_ranges[tgt_base + i] = range[spot * 1326 + i];
    }

    // === Compute game value = sum(range[i] * cfv[i]) ===
    float gv = 0.0f;
    for (int i = 0; i < 1326; i++) {
        gv += range[spot * 1326 + i] * cfvs[spot * 1326 + i];
    }
    res_game_values[dest] = gv;
}
