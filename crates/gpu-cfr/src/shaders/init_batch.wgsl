// Batch initialization: set root reach = 1.0 for each deal.
//
// Called after encoder.clear_buffer() zeros reach and utility buffers.
// Each thread handles one deal, setting reach_p1[root] = reach_p2[root] = 1.0.

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

@group(2) @binding(4) var<storage, read_write> reach_p1: array<f32>;
@group(2) @binding(5) var<storage, read_write> reach_p2: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let deal_id = gid.x;
    if (deal_id >= uniforms.num_deals) {
        return;
    }
    let root_offset = deal_id * uniforms.num_nodes;
    reach_p1[root_offset] = 1.0;
    reach_p2[root_offset] = 1.0;
}
