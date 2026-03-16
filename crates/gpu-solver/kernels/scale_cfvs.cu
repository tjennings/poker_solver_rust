// Scale raw DCFR+ counterfactual values to pot-relative EVs.
//
// Formula: pot_relative[h] = raw_cfv[h] * num_combinations / pot + 0.5
//
// This matches the cfvnet datagen format where EVs are expressed as
// fractions of the pot (0.5 = break even, >0.5 = winning, <0.5 = losing).

extern "C" __global__ void scale_cfvs_to_pot_relative(
    float* cfvs,
    const float* pots,
    float num_combinations,
    unsigned int total_hands,
    unsigned int hands_per_spot
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_hands) return;
    unsigned int spot = tid / hands_per_spot;
    float pot = pots[spot];
    if (pot > 0.0f) {
        cfvs[tid] = cfvs[tid] * num_combinations / pot + 0.5f;
    }
}
