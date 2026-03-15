// Neural network forward pass helper kernels for CudaNetInference.
//
// Two kernels:
// 1. bn_prelu: Fused BatchNorm (inference mode) + PReLU activation
// 2. add_bias: Bias addition after cuBLAS GEMM

/// Fused BatchNorm (inference) + PReLU activation.
///
/// For each element at index (batch_idx, feature_idx):
///   x_hat = (x - running_mean[f]) / sqrt(running_var[f] + eps)
///   y = bn_weight[f] * x_hat + bn_bias[f]
///   output = y >= 0 ? y : prelu_alpha[f] * y
///
/// `data` is modified in-place: [batch_size * features], row-major.
extern "C" __global__ void bn_prelu(
    float* data,
    const float* running_mean,
    const float* running_var,
    const float* bn_weight,
    const float* bn_bias,
    const float* prelu_alpha,
    float eps,
    unsigned int total,     // batch_size * features
    unsigned int features
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    unsigned int f = tid % features;

    float x = data[tid];
    float mean = running_mean[f];
    float var = running_var[f];
    float w = bn_weight[f];
    float b = bn_bias[f];
    float alpha = prelu_alpha[f];

    // BatchNorm inference: normalize then scale+shift
    float inv_std = rsqrtf(var + eps);
    float y = w * (x - mean) * inv_std + b;

    // PReLU: y >= 0 ? y : alpha * y
    data[tid] = (y >= 0.0f) ? y : alpha * y;
}

/// Bias addition after GEMM.
///
/// Adds per-feature bias to each element: data[i] += bias[i % features].
/// `data` is [batch_size * features], row-major.
extern "C" __global__ void add_bias(
    float* data,
    const float* bias,
    unsigned int total,     // batch_size * features
    unsigned int features
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    data[tid] += bias[tid % features];
}
