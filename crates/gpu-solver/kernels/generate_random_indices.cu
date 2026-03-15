// GPU kernel to generate random indices for reservoir sampling.
//
// Uses xorshift32 per-thread PRNG seeded from a base seed plus thread id.
//
// Launch: ceil(batch_size / blockDim.x) blocks.

extern "C" __global__ void generate_random_indices(
    unsigned int* indices,       // [batch_size] output
    unsigned int batch_size,
    unsigned int max_index,      // reservoir size (exclusive upper bound)
    unsigned int seed            // base seed
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    // Xorshift32 PRNG with per-thread seed
    unsigned int state = seed ^ (tid * 2654435761u + 1);
    // Ensure non-zero state
    if (state == 0) state = 1;

    // Run a few rounds to warm up
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;

    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;

    indices[tid] = state % max_index;
}
