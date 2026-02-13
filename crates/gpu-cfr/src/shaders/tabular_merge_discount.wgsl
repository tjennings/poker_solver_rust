// Tabular CFR: merge atomic deltas and apply DCFR discounting in one pass.
//
// One thread per (info_set * max_actions) entry.
// 1. Merge regret_delta and strat_sum_delta into regret/strategy_sum
// 2. Apply DCFR discount to regrets

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

@group(1) @binding(0) var<storage, read_write> regret: array<f32>;
@group(1) @binding(2) var<storage, read_write> strategy_sum: array<f32>;
@group(1) @binding(3) var<storage, read_write> regret_delta: array<atomic<u32>>;
@group(1) @binding(4) var<storage, read_write> strat_sum_delta: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = uniforms.num_info_sets * uniforms.max_actions;
    if idx >= total {
        return;
    }

    // Merge regret delta
    let r_bits = atomicExchange(&regret_delta[idx], 0u);
    let r_delta = bitcast<f32>(r_bits);
    var r = regret[idx] + r_delta;

    // Merge strategy sum delta
    let s_bits = atomicExchange(&strat_sum_delta[idx], 0u);
    let s_delta = bitcast<f32>(s_bits);
    strategy_sum[idx] += s_delta;

    // DCFR discount
    let t = f32(uniforms.iteration + 1u);
    if r > 0.0 {
        let t_alpha = pow(t, uniforms.dcfr_alpha);
        r = r * t_alpha / (t_alpha + 1.0);
    } else if r < 0.0 {
        let t_beta = pow(t, uniforms.dcfr_beta);
        r = r * t_beta / (t_beta + 1.0);
    }
    regret[idx] = r;
}
