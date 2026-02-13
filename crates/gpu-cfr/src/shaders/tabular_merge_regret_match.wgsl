// Tabular CFR: fused merge_discount + regret_match.
//
// One thread per info set. Each thread:
// 1. Loops over actions: atomicExchange deltas, merge into regret/strategy_sum, DCFR discount
// 2. Regret-matches to compute current strategy
//
// Runs at the TOP of each iteration. For iteration 0, deltas are zero → uniform strategy.

struct Uniforms {
    level_start: u32,
    level_count: u32,
    num_nodes: u32,
    n_traj_p1: u32,
    n_traj_p2: u32,
    max_traj: u32,
    num_info_sets: u32,
    max_actions: u32,
    iteration: u32,
    dcfr_alpha: f32,
    dcfr_beta: f32,
    dcfr_gamma: f32,
    strategy_discount: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<storage, read_write> info_set_num_actions: array<u32>;

@group(1) @binding(0) var<storage, read_write> regret: array<f32>;
@group(1) @binding(1) var<storage, read_write> strategy: array<f32>;
@group(1) @binding(2) var<storage, read_write> strategy_sum: array<f32>;
@group(1) @binding(3) var<storage, read_write> regret_delta: array<atomic<u32>>;
@group(1) @binding(4) var<storage, read_write> strat_sum_delta: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let info_id = gid.x;
    if info_id >= uniforms.num_info_sets {
        return;
    }

    let num_actions = info_set_num_actions[info_id];
    let base = info_id * uniforms.max_actions;
    let t = f32(uniforms.iteration + 1u);

    // Phase 1: merge deltas + DCFR discount
    var pos_sum: f32 = 0.0;
    for (var a = 0u; a < num_actions; a++) {
        let idx = base + a;

        // Merge regret delta
        let r_bits = atomicExchange(&regret_delta[idx], 0u);
        var r = regret[idx] + bitcast<f32>(r_bits);

        // DCFR discount on regret
        if r > 0.0 {
            let t_alpha = pow(t, uniforms.dcfr_alpha);
            r = r * t_alpha / (t_alpha + 1.0);
        } else if r < 0.0 {
            let t_beta = pow(t, uniforms.dcfr_beta);
            r = r * t_beta / (t_beta + 1.0);
        }
        regret[idx] = r;

        // Merge strategy sum delta
        let s_bits = atomicExchange(&strat_sum_delta[idx], 0u);
        strategy_sum[idx] += bitcast<f32>(s_bits);

        // Track positive regret sum for regret matching
        if r > 0.0 {
            pos_sum += r;
        }
    }

    // Phase 2: regret matching → compute strategy
    if pos_sum > 0.0 {
        for (var a = 0u; a < num_actions; a++) {
            let r = regret[base + a];
            strategy[base + a] = select(0.0, r / pos_sum, r > 0.0);
        }
    } else {
        let uniform_prob = 1.0 / f32(num_actions);
        for (var a = 0u; a < num_actions; a++) {
            strategy[base + a] = uniform_prob;
        }
    }

    // Zero unused action slots
    for (var a = num_actions; a < uniforms.max_actions; a++) {
        strategy[base + a] = 0.0;
    }
}
