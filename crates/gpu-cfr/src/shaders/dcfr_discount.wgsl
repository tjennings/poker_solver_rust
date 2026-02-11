// Apply DCFR discounting to cumulative regrets.
//
// One thread per (info_set * max_actions) entry.
// Positive regrets: multiply by t^alpha / (t^alpha + 1)
// Negative regrets: multiply by t^beta / (t^beta + 1)
//
// Bind groups:
//   group 0: uniforms (0)
//   group 1: regret (0)

struct Uniforms {
    level_start: u32,
    level_count: u32,
    num_deals: u32,
    num_nodes: u32,
    num_info_sets: u32,
    max_actions: u32,
    num_hand_classes: u32,
    iteration: u32,
    dcfr_alpha: f32,
    dcfr_beta: f32,
    dcfr_gamma: f32,
    strategy_discount: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var<storage, read_write> regret: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = uniforms.num_info_sets * uniforms.max_actions;
    if idx >= total {
        return;
    }

    let t = f32(uniforms.iteration + 1u);
    let r = regret[idx];

    if r > 0.0 {
        let t_alpha = pow(t, uniforms.dcfr_alpha);
        regret[idx] = r * t_alpha / (t_alpha + 1.0);
    } else if r < 0.0 {
        let t_beta = pow(t, uniforms.dcfr_beta);
        regret[idx] = r * t_beta / (t_beta + 1.0);
    }
}
