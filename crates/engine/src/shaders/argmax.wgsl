// Greedy sampling: tok_out[m] = argmax(logits[m]), one workgroup per row m. With MAPPED, logits
// cover a vocabulary subset and the winning index is mapped to its token id.

override V: u32;
override MAPPED: bool = false;

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> tok_out: array<u32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;  // [V] with MAPPED, else unused

var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;

@compute @workgroup_size(256)
fn argmax(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let lb = wid.y * V;
    var best = -3.0e38;
    var idx = 0u;
    for (var i = lid; i < V; i += 256u) {
        let v = logits[lb + i];
        if (v > best) {
            best = v;
            idx = i;
        }
    }
    bv[lid] = best;
    bi[lid] = idx;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid < s) {
            let ov = bv[lid + s];
            let oi = bi[lid + s];
            if (ov > bv[lid] || (ov == bv[lid] && oi < bi[lid])) {
                bv[lid] = ov;
                bi[lid] = oi;
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        tok_out[wid.y] = select(bi[0], ids[bi[0]], MAPPED);
    }
}
