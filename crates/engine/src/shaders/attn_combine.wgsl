// Merge flash-decoding chunks for one query head and emit the head output as int8 groups
// (same format as quant.wgsl), ready for the o projection. With GATED (Qwen3.5), the output is
// multiplied by sigmoid(gate) first; gate rows are laid out [head: q (D) | gate (D)] in `gsrc`.
// One workgroup per (query head, token), thread j owns dims 2j and 2j + 1.

struct Params {
    pos: u32,
    seq_len: u32,
    m: u32,
    slot: u32,
}

override H: u32;
override D: u32;
override MAXCTX: u32;
override KVH: u32;
override GATED: bool = false;

const CH: u32 = 32u;

@group(0) @binding(0) var<storage, read> part: array<f32>;  // [M, H, MAXCTX / CH, D + 2]
@group(0) @binding(1) var<storage, read_write> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> xs: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> p: Params;
@group(0) @binding(4) var<storage, read> gsrc: array<f32>;  // rows of (2 H + 2 KVH) D = 2 (H + KVH) D

var<workgroup> o: array<f32, D>;
var<workgroup> gs: array<f32, D / 32u>;

@compute @workgroup_size(D / 2u)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    let m = wid.y;
    let nch = MAXCTX / CH;
    let n = (p.pos + m + CH) / CH;  // chunks covering positions 0 ..= p.pos + m
    let hb = (m * H + h) * nch * (D + 2u);

    var mx = -3.0e38;
    for (var c = 0u; c < n; c++) {
        mx = max(mx, part[hb + c * (D + 2u) + D]);
    }
    var l = 0.0;
    var acc = vec2<f32>(0.0);
    for (var c = 0u; c < n; c++) {
        let cb = hb + c * (D + 2u);
        let w = exp(part[cb + D] - mx);
        l += w * part[cb + D + 1u];
        acc += w * vec2<f32>(part[cb + 2u * lid], part[cb + 2u * lid + 1u]);
    }
    acc /= l;
    if (GATED) {
        let gb = m * 2u * (H + KVH) * D + h * 2u * D + D + 2u * lid;
        acc /= 1.0 + exp(-vec2<f32>(gsrc[gb], gsrc[gb + 1u]));
    }
    o[2u * lid] = acc.x;
    o[2u * lid + 1u] = acc.y;
    workgroupBarrier();

    // groups of 32 within this head
    if (lid < D / 32u) {
        var amax = 0.0;
        for (var i = 0u; i < 32u; i++) {
            amax = max(amax, abs(o[lid * 32u + i]));
        }
        gs[lid] = amax / 127.0;
    }
    workgroupBarrier();
    if (lid < D / 4u) {
        let dx = gs[lid / 8u];
        let inv = select(0.0, 1.0 / dx, dx > 0.0);
        let q = vec4<i32>(round(vec4<f32>(o[4u * lid], o[4u * lid + 1u], o[4u * lid + 2u], o[4u * lid + 3u]) * inv));
        xq[(m * H + h) * (D / 4u) + lid] = pack4xI8(q);
        o[4u * lid] = f32(q.x + q.y + q.z + q.w);
    }
    workgroupBarrier();
    if (lid < D / 32u) {
        var sum = 0.0;
        for (var i = 0u; i < 8u; i++) {
            sum += o[(lid * 8u + i) * 4u];
        }
        xs[(m * H + h) * (D / 32u) + lid] = vec2<f32>(gs[lid], gs[lid] * sum);
    }
}
