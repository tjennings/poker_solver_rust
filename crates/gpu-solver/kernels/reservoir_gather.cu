// GPU kernel to gather random samples from the reservoir into contiguous
// mini-batch tensors.
//
// Launch: ceil(batch_size * feature_size / blockDim.x) blocks.
// One thread per output element.

extern "C" __global__ void reservoir_gather(
    float* batch_output,         // [batch_size * feature_size] output
    const float* reservoir,      // [capacity * feature_size] source
    const unsigned int* indices, // [batch_size] random indices into reservoir
    unsigned int batch_size,
    unsigned int feature_size
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch_size * feature_size;
    if (tid >= total) return;

    unsigned int batch_idx = tid / feature_size;
    unsigned int feat_idx = tid % feature_size;
    unsigned int src_idx = indices[batch_idx];

    batch_output[tid] = reservoir[src_idx * feature_size + feat_idx];
}
