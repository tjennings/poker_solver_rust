// GPU kernel for weighted averaging of flop model CFV predictions across
// canonical flops to produce preflop CFV training targets.
//
// After the flop model returns CFVs for all (canonical_flop x spot) inputs,
// this kernel computes a weighted average across the 1,755 canonical flops
// per combo, skipping flops where the combo conflicts with board cards.
//
// Weights are pre-normalized to sum to 1.0, so the weighted average for
// non-conflicting flops is: sum(cfv * weight) / sum(weight) over valid flops.
//
// One thread per (spot_idx, hand) pair.
// Each thread loops over all canonical flops, accumulates weighted CFVs,
// and writes the weighted average.

extern "C" __global__ void average_preflop_cfvs(
    float* output,                    // [num_spots * 1326] averaged CFVs
    const float* raw_cfvs,            // [num_flops * num_spots * 1326]
    const unsigned int* all_flops,    // [num_flops * 3] -- canonical flop cards
    const unsigned int* combo_cards,  // [1326 * 2] -- card pairs per combo
    const float* weights,             // [num_flops] -- pre-normalized weights
    unsigned int num_flops,           // 1755 canonical flops
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

    // Accumulate weighted CFVs across all canonical flops, skipping conflicts
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;

    for (unsigned int f = 0; f < num_flops; f++) {
        unsigned int f0 = all_flops[f * 3];
        unsigned int f1 = all_flops[f * 3 + 1];
        unsigned int f2 = all_flops[f * 3 + 2];

        // Skip flops where the combo conflicts with board cards
        if (c1 == f0 || c1 == f1 || c1 == f2 ||
            c2 == f0 || c2 == f1 || c2 == f2) {
            continue;
        }

        float w = weights[f];

        // Input index: flop_idx * num_spots + spot_idx
        unsigned int input_idx = f * num_spots + spot_idx;

        // The model output for this input is raw_cfvs[input_idx * 1326 + hand]
        weighted_sum += raw_cfvs[input_idx * 1326u + hand] * w;
        weight_sum += w;
    }

    // Write weighted-averaged CFV (re-normalize to account for skipped conflicts)
    float avg = (weight_sum > 0.0f) ? (weighted_sum / weight_sum) : 0.0f;
    output[spot_idx * 1326u + hand] = avg;
}
