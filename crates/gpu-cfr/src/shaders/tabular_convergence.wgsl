// Tabular CFR: GPU-side convergence metric via max positive regret.
//
// Two-pass workgroup reduction:
// Pass 1 (pass_id=0): Each workgroup processes 256 info sets,
//   finds local max positive regret, writes to scratch[workgroup_id].
// Pass 2 (pass_id=1): Single workgroup reduces scratch[] to scalar at scratch[0].

struct ConvergenceUniforms {
    num_info_sets: u32,
    max_actions: u32,
    pass_id: u32,       // 0=per-infoset, 1=final reduction
    num_workgroups: u32, // from pass 1 (used in pass 2)
};

@group(0) @binding(0) var<uniform> uniforms: ConvergenceUniforms;
@group(0) @binding(1) var<storage, read_write> regret: array<f32>;
@group(0) @binding(2) var<storage, read_write> scratch: array<f32>;

var<workgroup> shared_max: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    if uniforms.pass_id == 0u {
        // Pass 1: each thread finds max positive regret for one info set
        let info_id = gid.x;
        var local_max: f32 = 0.0;
        if info_id < uniforms.num_info_sets {
            let base = info_id * uniforms.max_actions;
            for (var a = 0u; a < uniforms.max_actions; a++) {
                local_max = max(local_max, regret[base + a]);
            }
        }
        shared_max[lid.x] = local_max;
        workgroupBarrier();

        // Tree reduction within workgroup
        for (var s = 128u; s > 0u; s >>= 1u) {
            if lid.x < s {
                shared_max[lid.x] = max(shared_max[lid.x], shared_max[lid.x + s]);
            }
            workgroupBarrier();
        }

        if lid.x == 0u {
            scratch[wid.x] = shared_max[0];
        }
    } else {
        // Pass 2: reduce scratch[] to a single value
        let n = uniforms.num_workgroups;
        var local_max: f32 = 0.0;
        var idx = lid.x;
        while idx < n {
            local_max = max(local_max, scratch[idx]);
            idx += 256u;
        }
        shared_max[lid.x] = local_max;
        workgroupBarrier();

        for (var s = 128u; s > 0u; s >>= 1u) {
            if lid.x < s {
                shared_max[lid.x] = max(shared_max[lid.x], shared_max[lid.x + s]);
            }
            workgroupBarrier();
        }

        if lid.x == 0u {
            scratch[0] = shared_max[0];
        }
    }
}
