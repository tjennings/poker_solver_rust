// GPU kernel for averaging flop model CFV predictions across all flops
// to produce preflop CFV training targets.
//
// After the flop model returns CFVs for all (flop x spot) inputs,
// this kernel averages across the 22,100 flops per combo, skipping
// flops where the combo conflicts with board cards.
//
// One thread per (spot_idx, hand) pair.
// Each thread loops over all flops, accumulates non-conflicting CFVs,
// and writes the average.

extern "C" __global__ void average_preflop_cfvs(
    float* output,                    // [num_spots * 1326] averaged CFVs
    const float* raw_cfvs,            // [num_flops * num_spots * 1326]
    const unsigned int* all_flops,    // [num_flops * 3] -- all C(52,3) flop cards
    const unsigned int* combo_cards,  // [1326 * 2] -- card pairs per combo
    unsigned int num_flops,           // 22100
    unsigned int num_spots
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = num_spots * 1326u;
    if (tid >= total_threads) return;

    unsigned int hand = tid % 1326u;
    unsigned int spot_idx = tid / 1326u;

    // Get this hand's cards for conflict checking
    unsigned int c1 = combo_cards[hand * 2];
    unsigned int c2 = combo_cards[hand * 2 + 1];

    // Accumulate CFVs across all flops, skipping conflicts
    float sum = 0.0f;
    unsigned int count = 0;

    for (unsigned int f = 0; f < num_flops; f++) {
        unsigned int f0 = all_flops[f * 3];
        unsigned int f1 = all_flops[f * 3 + 1];
        unsigned int f2 = all_flops[f * 3 + 2];

        // Skip flops where the combo conflicts with board cards
        if (c1 == f0 || c1 == f1 || c1 == f2 ||
            c2 == f0 || c2 == f1 || c2 == f2) {
            continue;
        }

        // Input index: flop_idx * num_spots + spot_idx
        unsigned int input_idx = f * num_spots + spot_idx;

        // The model output for this input is raw_cfvs[input_idx * 1326 + hand]
        sum += raw_cfvs[input_idx * 1326u + hand];
        count++;
    }

    // Write averaged CFV
    float avg = (count > 0) ? (sum / (float)count) : 0.0f;
    output[spot_idx * 1326u + hand] = avg;
}
