// GPU kernel for sampling random turn situations.
//
// Generates `batch_size` random turn boards (4 unique cards each) and
// uniform ranges (1.0 for unblocked combos, 0.0 for blocked) entirely
// on the GPU. Uses per-thread xorshift32 PRNG.
//
// Phase 1: Sample boards -- one thread per situation.
// Phase 2: Build ranges -- one thread per (situation, combo) pair.

// ------------------------------------------------------------------
// Phase 1: Sample 4 unique board cards per situation (flop + turn)
// ------------------------------------------------------------------
extern "C" __global__ void sample_turn_boards(
    unsigned int* boards,       // [batch_size * 4] output
    unsigned int batch_size,
    unsigned int seed
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    // Xorshift32 PRNG with per-thread seed
    unsigned int state = seed ^ (tid * 2654435761u + 1);
    if (state == 0) state = 1;

    // Warm up
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;

    // Fisher-Yates partial shuffle to pick 4 unique cards from 0..52
    unsigned char deck[52];
    for (int i = 0; i < 52; i++) {
        deck[i] = (unsigned char)i;
    }

    for (int pick = 0; pick < 4; pick++) {
        // Advance PRNG
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;

        unsigned int remaining = 52 - pick;
        unsigned int idx = pick + (state % remaining);

        // Swap
        unsigned char tmp = deck[pick];
        deck[pick] = deck[idx];
        deck[idx] = tmp;

        boards[tid * 4 + pick] = (unsigned int)deck[pick];
    }
}

// ------------------------------------------------------------------
// Phase 2: Build blocking-aware ranges for turn (4-card board)
// ------------------------------------------------------------------
extern "C" __global__ void build_turn_ranges(
    float* ranges_oop,              // [batch_size * 1326] output
    float* ranges_ip,               // [batch_size * 1326] output
    const unsigned int* boards,     // [batch_size * 4] input (from phase 1)
    const unsigned int* combo_cards, // [1326 * 2] static lookup
    unsigned int batch_size,
    unsigned int seed               // different seed for range weights
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = batch_size * 1326u;
    if (tid >= total) return;

    unsigned int sit = tid / 1326u;
    unsigned int combo = tid % 1326u;

    unsigned int c1 = combo_cards[combo * 2];
    unsigned int c2 = combo_cards[combo * 2 + 1];

    // Check if either hole card collides with any of the 4 board cards
    const unsigned int* board = &boards[sit * 4];
    int blocked = 0;
    for (int i = 0; i < 4; i++) {
        if (board[i] == c1 || board[i] == c2) {
            blocked = 1;
            break;
        }
    }

    float weight = blocked ? 0.0f : 1.0f;
    ranges_oop[tid] = weight;
    ranges_ip[tid] = weight;
}
