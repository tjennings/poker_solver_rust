// Regret matching kernel: converts per-hand cumulative regrets into strategy.
//
// One thread per (infoset, hand) pair. For each pair, normalizes positive
// regrets into action probabilities. Falls back to uniform when all regrets
// are non-positive.
//
// Layout:
//   regrets[infoset * max_actions * num_hands + action * num_hands + hand]
//   strategy[infoset * max_actions * num_hands + action * num_hands + hand]

extern "C" __global__ void regret_match(
    const float* regrets,
    const unsigned int* num_actions,
    float* strategy,
    unsigned int num_infosets,
    unsigned int max_actions,
    unsigned int num_hands
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iset = tid / num_hands;
    unsigned int hand = tid % num_hands;
    if (iset >= num_infosets) return;

    unsigned int n = num_actions[iset];

    float pos_sum = 0.0f;
    for (unsigned int a = 0; a < n; a++) {
        float r = regrets[(iset * max_actions + a) * num_hands + hand];
        if (r > 0.0f) pos_sum += r;
    }

    if (pos_sum > 0.0f) {
        for (unsigned int a = 0; a < n; a++) {
            float r = regrets[(iset * max_actions + a) * num_hands + hand];
            strategy[(iset * max_actions + a) * num_hands + hand] = (r > 0.0f) ? (r / pos_sum) : 0.0f;
        }
    } else {
        float uniform = 1.0f / (float)n;
        for (unsigned int a = 0; a < n; a++) {
            strategy[(iset * max_actions + a) * num_hands + hand] = uniform;
        }
    }

    // Zero out padding slots beyond num_actions
    for (unsigned int a = n; a < max_actions; a++) {
        strategy[(iset * max_actions + a) * num_hands + hand] = 0.0f;
    }
}
