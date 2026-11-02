// Decode attention, flash-decoding style.
//   attn    : one workgroup per (chunk of CH positions, kv head, token). Computes scores for
//             the GQ query heads sharing that kv head, a chunk-local softmax and the partial
//             P V. Writes (o_partial[D], max, sum) per (token, head, chunk).
//   combine : (attn_combine.wgsl) merges the chunks per query head.
// Token m (of the p.m tokens of a step, > 1 when verifying a draft) sits at position p.pos + m
// and attends to positions 0 ..= p.pos + m. The host may dispatch more chunks than needed (when
// the position is only known on the GPU); those exit at once.
// Assumes GQ <= 16, checked on the host.

struct Params {
    pos: u32,
    seq_len: u32,
    m: u32,
    slot: u32,
}

override H: u32;
override KVH: u32;
override D: u32;
override MAXCTX: u32;

const CH: u32 = 32u;
const WG: u32 = 256u;

@group(0) @binding(0) var<storage, read> qkv: array<vec4<f32>>;  // rows [H + 2 KVH, D] per token
@group(0) @binding(1) var<storage, read> kc: array<vec2<u32>>;  // f16 [KVH, MAXCTX, D], 4 values per entry
@group(0) @binding(2) var<storage, read> vc: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;  // [M, H, MAXCTX / CH, D + 2]
@group(0) @binding(4) var<uniform> p: Params;

// q rows padded by one vec4 so that different heads hit different banks
var<workgroup> qs: array<vec4<f32>, (H / KVH) * (D / 4u + 1u)>;
var<workgroup> ps: array<f32, (H / KVH) * CH>;
var<workgroup> ms: array<vec2<f32>, H / KVH>;

fn h4(u: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(u.x), unpack2x16float(u.y));
}

@compute @workgroup_size(256)
fn attn(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let gq = H / KVH;
    let d4 = D / 4u;
    let qstride = d4 + 1u;
    let chunk = wid.x;
    let kh = wid.y % KVH;
    let m = wid.y / KVH;
    let seq_len = p.pos + m + 1u;
    let t0 = chunk * CH;
    if (t0 >= seq_len) {
        return;
    }
    let qb = m * (H + 2u * KVH) * d4 + kh * gq * d4;
    for (var i = lid; i < gq * d4; i += WG) {
        qs[(i / d4) * qstride + i % d4] = qkv[qb + i];
    }
    workgroupBarrier();

    // scores; consecutive threads share a position (k row is broadcast) and differ in head
    let scale = inverseSqrt(f32(D));
    for (var e = lid; e < gq * CH; e += WG) {
        let h = e % gq;
        let tt = e / gq;
        var s = -3.0e38;
        if (t0 + tt < seq_len) {
            let kb = (kh * MAXCTX + t0 + tt) * d4;
            s = 0.0;
            for (var i = 0u; i < d4; i++) {
                s += dot(qs[h * qstride + i], h4(kc[kb + i]));
            }
            s *= scale;
        }
        ps[h * CH + tt] = s;
    }
    workgroupBarrier();

    if (lid < gq) {
        var m = -3.0e38;
        for (var i = 0u; i < CH; i++) {
            m = max(m, ps[lid * CH + i]);
        }
        var l = 0.0;
        for (var i = 0u; i < CH; i++) {
            l += exp(ps[lid * CH + i] - m);
        }
        ms[lid] = vec2<f32>(m, l);
    }
    workgroupBarrier();
    for (var e = lid; e < gq * CH; e += WG) {
        ps[e] = exp(ps[e] - ms[e / CH].x);
    }
    workgroupBarrier();

    let n = min(CH, seq_len - t0);
    let nch = MAXCTX / CH;
    for (var e = lid; e < gq * d4; e += WG) {
        let h = e / d4;
        let dv = e % d4;
        var acc = vec4<f32>(0.0);
        let vb = (kh * MAXCTX + t0) * d4 + dv;
        for (var i = 0u; i < n; i++) {
            acc += ps[h * CH + i] * h4(vc[vb + i * d4]);
        }
        let base = ((m * H + kh * gq + h) * nch + chunk) * (D + 2u);
        part[base + 4u * dv] = acc.x;
        part[base + 4u * dv + 1u] = acc.y;
        part[base + 4u * dv + 2u] = acc.z;
        part[base + 4u * dv + 3u] = acc.w;
        if (dv == 0u) {
            part[base + D] = ms[h].x;
            part[base + D + 1u] = ms[h].y;
        }
    }
}
