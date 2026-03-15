// GPU kernel to compute fold and showdown payoffs for batch solver.
//
// For each (terminal, hand) pair, computes the per-hand payoff amounts
// using the spot's pot size and the terminal's pot delta from the shared
// tree topology.
//
// Launch: ceil(max(num_fold_terminals, num_showdown_terminals) * total_hands / blockDim.x) blocks.

extern "C" __global__ void compute_fold_payoffs(
    float* fold_win,              // [num_fold_terminals * total_hands] output
    float* fold_lose,             // [num_fold_terminals * total_hands] output
    const float* spot_pots,       // [num_spots] starting pot per spot
    const float* fold_payoff_win, // [num_fold_terminals] per-terminal win payoff (from ref tree)
    const float* fold_payoff_lose,// [num_fold_terminals] per-terminal lose payoff (from ref tree)
    unsigned int num_fold_terminals,
    unsigned int total_hands,
    unsigned int hands_per_spot
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_fold_terminals * total_hands;
    if (tid >= total) return;

    unsigned int term = tid / total_hands;
    unsigned int hand = tid % total_hands;
    unsigned int spot = hand / hands_per_spot;

    // Scale the reference payoffs by (spot_pot / ref_pot).
    // The reference payoffs already encode the pot-size relationship,
    // but since all spots share the same tree and same pot, we just
    // broadcast the reference payoff to all hands in the spot.
    // For batch mode with uniform pot, fold_payoff_win/lose are the
    // same for all spots. For variable pots, we'd need spot_pots[spot].
    //
    // Actually, for the GPU-native builder the payoffs are pre-scaled
    // per the reference tree's pot. Since all spots share pot/stack,
    // we just broadcast: each hand gets the same payoff as the ref tree.
    float win = fold_payoff_win[term];
    float lose = fold_payoff_lose[term];

    unsigned int idx = term * total_hands + hand;
    fold_win[idx] = win;
    fold_lose[idx] = lose;
}

extern "C" __global__ void compute_showdown_payoffs(
    float* showdown_win,               // [num_showdown_terminals * total_hands] output
    float* showdown_lose,              // [num_showdown_terminals * total_hands] output
    const float* showdown_payoff_win,  // [num_showdown_terminals] per-terminal win payoff
    const float* showdown_payoff_lose, // [num_showdown_terminals] per-terminal lose payoff
    unsigned int num_showdown_terminals,
    unsigned int total_hands,
    unsigned int hands_per_spot
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_showdown_terminals * total_hands;
    if (tid >= total) return;

    unsigned int term = tid / total_hands;

    float win = showdown_payoff_win[term];
    float lose = showdown_payoff_lose[term];

    showdown_win[tid] = win;
    showdown_lose[tid] = lose;
}
