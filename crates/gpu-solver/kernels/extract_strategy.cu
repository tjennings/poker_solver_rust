// Extract strategy kernel: normalizes cumulative strategy sums into final
// action probabilities.
//
// One thread per (infoset, hand) pair. For each pair, normalizes strategy
// sums into probabilities. Falls back to uniform when all sums are zero.
//
// Layout:
//   strategy_sum[(infoset * max_actions + action) * num_hands + hand]
//   output_strategy[(infoset * max_actions + action) * num_hands + hand]

extern "C" __global__ void extract_strategy(
    const float* strategy_sum,
    const unsigned int* num_actions,
    float* output_strategy,
    unsigned int num_infosets,
    unsigned int max_actions,
    unsigned int num_hands
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iset = tid / num_hands;
    unsigned int hand = tid % num_hands;
    if (iset >= num_infosets) return;

    unsigned int n = num_actions[iset];

    float total = 0.0f;
    for (unsigned int a = 0; a < n; a++) {
        total += strategy_sum[(iset * max_actions + a) * num_hands + hand];
    }

    if (total > 0.0f) {
        for (unsigned int a = 0; a < n; a++) {
            unsigned int idx = (iset * max_actions + a) * num_hands + hand;
            output_strategy[idx] = strategy_sum[idx] / total;
        }
    } else {
        float uniform = 1.0f / (float)n;
        for (unsigned int a = 0; a < n; a++) {
            output_strategy[(iset * max_actions + a) * num_hands + hand] = uniform;
        }
    }

    // Zero padding beyond valid actions
    for (unsigned int a = n; a < max_actions; a++) {
        output_strategy[(iset * max_actions + a) * num_hands + hand] = 0.0f;
    }
}
