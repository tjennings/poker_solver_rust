extern "C" __global__ void set_root_reach(
    float* reach,
    const float* initial_values,
    unsigned int num_hands
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_hands) {
        reach[tid] = initial_values[tid];
    }
}
