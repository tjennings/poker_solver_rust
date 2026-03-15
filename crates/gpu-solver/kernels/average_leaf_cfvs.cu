// GPU kernel for averaging river model CFV predictions across river cards
// and scattering the results back to the cfvalues buffer at depth-boundary nodes.
//
// After the river model returns CFVs for all (boundary x river x spot) inputs,
// this kernel averages across the 48 river cards per combo and writes the
// averaged values to cfvalues at the correct node/hand positions.
//
// One thread per (boundary_idx, spot_idx, hand) triple.
// Each thread loops over all river cards, accumulates non-conflicting CFVs,
// and writes the average to cfvalues.

extern "C" __global__ void average_leaf_cfvs(
    float* cfvalues,                    // [num_nodes * total_hands] output (scattered writes)
    const float* raw_cfvs,              // [num_boundaries * num_rivers * num_spots * 1326]
    const unsigned int* boundary_nodes, // [num_boundaries] — node IDs
    const unsigned int* river_cards,    // [num_rivers] — possible river cards
    const unsigned int* combo_cards,    // [1326 * 2] — card pairs per combo
    unsigned int num_boundaries,
    unsigned int num_rivers,
    unsigned int num_spots,
    unsigned int total_hands,           // num_nodes * num_spots * hands_per_spot stride
    unsigned int hands_per_spot         // typically 1326
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = num_boundaries * num_spots * hands_per_spot;
    if (tid >= total_threads) return;

    // Decompose tid into (boundary_idx, spot_idx, hand)
    unsigned int hand = tid % hands_per_spot;
    unsigned int tmp = tid / hands_per_spot;
    unsigned int spot_idx = tmp % num_spots;
    unsigned int boundary_idx = tmp / num_spots;

    unsigned int node_id = boundary_nodes[boundary_idx];

    // Get this hand's cards for conflict checking
    unsigned int c1 = combo_cards[hand * 2];
    unsigned int c2 = combo_cards[hand * 2 + 1];

    // Accumulate CFVs across river cards, skipping conflicts
    float sum = 0.0f;
    unsigned int count = 0;

    for (unsigned int r = 0; r < num_rivers; r++) {
        unsigned int river_card = river_cards[r];

        // Skip river cards that conflict with this hand
        if (river_card == c1 || river_card == c2) {
            continue;
        }

        // Input index: boundary_idx * num_rivers * num_spots + river_idx * num_spots + spot_idx
        unsigned int input_idx = boundary_idx * num_rivers * num_spots
                               + r * num_spots
                               + spot_idx;

        // The model output for this input is raw_cfvs[input_idx * 1326 + hand]
        sum += raw_cfvs[input_idx * 1326u + hand];
        count++;
    }

    // Write averaged CFV to cfvalues at the boundary node position
    float avg = (count > 0) ? (sum / (float)count) : 0.0f;
    cfvalues[node_id * total_hands + spot_idx * hands_per_spot + hand] = avg;
}
