extern "C" __global__ void zero_buffer(float* buf, unsigned int len) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) buf[tid] = 0.0f;
}
