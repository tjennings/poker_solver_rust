extern "C" __global__ void regret_match(
    const float* regrets,
    const unsigned int* num_actions,
    float* strategy,
    unsigned int num_infosets,
    unsigned int max_actions
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_infosets) return;

    unsigned int n = num_actions[i];
    unsigned int base = i * max_actions;

    float pos_sum = 0.0f;
    for (unsigned int a = 0; a < n; a++) {
        float r = regrets[base + a];
        if (r > 0.0f) pos_sum += r;
    }

    if (pos_sum > 0.0f) {
        for (unsigned int a = 0; a < n; a++) {
            float r = regrets[base + a];
            strategy[base + a] = (r > 0.0f) ? (r / pos_sum) : 0.0f;
        }
    } else {
        float uniform = 1.0f / (float)n;
        for (unsigned int a = 0; a < n; a++) {
            strategy[base + a] = uniform;
        }
    }

    // Zero out padding slots beyond num_actions
    for (unsigned int a = n; a < max_actions; a++) {
        strategy[base + a] = 0.0f;
    }
}
