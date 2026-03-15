// GPU hand strength evaluation kernel.
//
// Evaluates 7-card poker hands (5 board + 2 hole cards) and produces a
// u16-equivalent strength ranking. The algorithm exactly mirrors the CPU
// implementation in range-solver/src/hand.rs:
//   1. Compute the internal i32 hand value using rank/suit analysis
//   2. Binary search into HAND_TABLE to get the u16 strength index
//
// Launch: ceil(num_situations * 1326 / blockDim.x) blocks.
// One thread per (situation, combo) pair.

// keep_n_msb: keeps the n most-significant set bits of x, clearing all others.
__device__ int keep_n_msb(int x, int n) {
    int ret = 0;
    for (int i = 0; i < n; i++) {
        // Find highest set bit
        int bit = 1 << (31 - __clz(x));
        x ^= bit;
        ret |= bit;
    }
    return ret;
}

// find_straight: detects a 5-card straight in a rank bitset.
// Returns the top-rank bit of the straight, or 0 if none.
// Handles the wheel (A-2-3-4-5).
__device__ int find_straight(int rankset) {
    const int WHEEL = 0x100F; // 0b1_0000_0000_1111
    int is_straight = rankset & (rankset << 1) & (rankset << 2) & (rankset << 3) & (rankset << 4);
    if (is_straight != 0) {
        return keep_n_msb(is_straight, 1);
    } else if ((rankset & WHEEL) == WHEEL) {
        return 1 << 3;
    } else {
        return 0;
    }
}

// evaluate_7_cards: compute the internal i32 hand value for 7 cards.
// Cards are indices 0..51 where rank = card/4, suit = card%4.
// This exactly mirrors Hand::evaluate_internal() from hand.rs.
__device__ int evaluate_7_cards(unsigned int cards[7]) {
    int rankset = 0;
    int rankset_suit[4] = {0, 0, 0, 0};
    int rankset_of_count[5] = {0, 0, 0, 0, 0};
    int rank_count[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < 7; i++) {
        int rank = cards[i] / 4;
        int suit = cards[i] % 4;
        rankset |= 1 << rank;
        rankset_suit[suit] |= 1 << rank;
        rank_count[rank]++;
    }

    for (int rank = 0; rank < 13; rank++) {
        rankset_of_count[rank_count[rank]] |= 1 << rank;
    }

    int flush_suit = -1;
    for (int suit = 0; suit < 4; suit++) {
        if (__popc(rankset_suit[suit]) >= 5) {
            flush_suit = suit;
        }
    }

    int is_straight = find_straight(rankset);

    if (flush_suit >= 0) {
        int is_straight_flush = find_straight(rankset_suit[flush_suit]);
        if (is_straight_flush != 0) {
            // straight flush
            return (8 << 26) | is_straight_flush;
        } else {
            // flush
            return (5 << 26) | keep_n_msb(rankset_suit[flush_suit], 5);
        }
    } else if (rankset_of_count[4] != 0) {
        // four of a kind
        int remaining = keep_n_msb(rankset ^ rankset_of_count[4], 1);
        return (7 << 26) | (rankset_of_count[4] << 13) | remaining;
    } else if (__popc(rankset_of_count[3]) == 2) {
        // full house (two trips -> best trips + second as pair)
        int trips = keep_n_msb(rankset_of_count[3], 1);
        int pair = rankset_of_count[3] ^ trips;
        return (6 << 26) | (trips << 13) | pair;
    } else if (rankset_of_count[3] != 0 && rankset_of_count[2] != 0) {
        // full house (trips + pair)
        int pair = keep_n_msb(rankset_of_count[2], 1);
        return (6 << 26) | (rankset_of_count[3] << 13) | pair;
    } else if (is_straight != 0) {
        // straight
        return (4 << 26) | is_straight;
    } else if (rankset_of_count[3] != 0) {
        // three of a kind
        int remaining = keep_n_msb(rankset_of_count[1], 2);
        return (3 << 26) | (rankset_of_count[3] << 13) | remaining;
    } else if (__popc(rankset_of_count[2]) >= 2) {
        // two pair
        int pairs = keep_n_msb(rankset_of_count[2], 2);
        int remaining = keep_n_msb(rankset ^ pairs, 1);
        return (2 << 26) | (pairs << 13) | remaining;
    } else if (rankset_of_count[2] != 0) {
        // one pair
        int remaining = keep_n_msb(rankset_of_count[1], 3);
        return (1 << 26) | (rankset_of_count[2] << 13) | remaining;
    } else {
        // high card
        return keep_n_msb(rankset, 5);
    }
}

// Binary search for val in sorted array. Returns the index, or 0 if not found.
__device__ unsigned int binary_search(const int* table, unsigned int table_len, int val) {
    unsigned int lo = 0;
    unsigned int hi = table_len;
    while (lo < hi) {
        unsigned int mid = lo + (hi - lo) / 2;
        if (table[mid] < val) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

extern "C" __global__ void evaluate_hand_strengths(
    unsigned int* strengths,           // [num_situations * 1326] output
    const unsigned int* boards,        // [num_situations * 5]
    const unsigned int* combo_cards,   // [1326 * 2] precomputed lookup table
    const int* hand_table,             // [4824] sorted hand values
    unsigned int hand_table_len,       // 4824
    unsigned int num_situations
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_situations * 1326;
    if (tid >= total) return;

    unsigned int sit = tid / 1326;
    unsigned int combo = tid % 1326;

    unsigned int c1 = combo_cards[combo * 2];
    unsigned int c2 = combo_cards[combo * 2 + 1];

    // Load board cards
    unsigned int b0 = boards[sit * 5];
    unsigned int b1 = boards[sit * 5 + 1];
    unsigned int b2 = boards[sit * 5 + 2];
    unsigned int b3 = boards[sit * 5 + 3];
    unsigned int b4 = boards[sit * 5 + 4];

    // Check if combo conflicts with board
    if (c1 == b0 || c1 == b1 || c1 == b2 || c1 == b3 || c1 == b4 ||
        c2 == b0 || c2 == b1 || c2 == b2 || c2 == b3 || c2 == b4) {
        strengths[tid] = 0;
        return;
    }

    unsigned int cards[7] = {b0, b1, b2, b3, b4, c1, c2};
    int raw_value = evaluate_7_cards(cards);
    unsigned int rank = binary_search(hand_table, hand_table_len, raw_value);
    strengths[tid] = rank;
}
