// Kernel 2 of 3-kernel O(n log n) showdown evaluation.
//
// Performs exclusive prefix sum on opponent reach values within each segment.
// A segment is one (terminal, spot) pair with segment_len = hands_per_spot
// elements, already in strength-sorted order from kernel 1 (scatter).
//
// After this kernel:
//   prefix_excl[seg][i] = sum of sorted_reach[seg][0..i]  (exclusive)
//   segment_totals[seg] = sum of all reach in the segment
//
// The exclusive prefix gives "reach of all weaker opponents" when indexed
// by a hand's rank. The total minus prefix_excl[rank+1] gives "reach of
// all stronger opponents".
//
// Uses blockDim=32 (one warp) for high occupancy. Thread 0 performs a
// serial scan on shared memory. All 32 threads cooperate on global
// memory loads and stores.
//
// Launch: num_segments blocks (= num_sd_terminals * num_spots), 32 threads.
// Shared memory: segment_len * sizeof(float) bytes.

extern "C" __global__ void segmented_prefix_sum(
    const float* sorted_reach,      // [num_segments * segment_len] input
    float* prefix_excl,             // [num_segments * segment_len] output (exclusive prefix)
    float* segment_totals,          // [num_segments] output
    unsigned int num_segments,
    unsigned int segment_len
) {
    unsigned int seg = blockIdx.x;
    if (seg >= num_segments) return;

    // Shared memory for the segment
    extern __shared__ float shm[];

    unsigned int base = seg * segment_len;

    // Load into shared memory (all 32 threads cooperate)
    for (unsigned int i = threadIdx.x; i < segment_len; i += blockDim.x) {
        shm[i] = sorted_reach[base + i];
    }
    __syncthreads();

    // Thread 0: serial exclusive prefix sum
    if (threadIdx.x == 0) {
        float running = 0.0f;
        for (unsigned int i = 0; i < segment_len; i++) {
            float val = shm[i];
            shm[i] = running;  // exclusive: prefix before this element
            running += val;
        }
        segment_totals[seg] = running;
    }
    __syncthreads();

    // Write back to global memory (all threads cooperate)
    for (unsigned int i = threadIdx.x; i < segment_len; i += blockDim.x) {
        prefix_excl[base + i] = shm[i];
    }
}
