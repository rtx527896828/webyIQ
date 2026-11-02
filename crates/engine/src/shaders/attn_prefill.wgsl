// Prefill attention (causal). One workgroup per (query token m, kv head): walks the keys
// 0..=pos+m in chunks of CH with an online softmax, for the GQ query heads that share the
// kv head. Output: att[m, h * D + d] (f32, quantized by a following quant pass). With GATED
// (Qwen3.5) the output is multiplied by sigmoid(gate), gate rows [head: q (D) | gate (D)] x H + k, v.
// Assumes D == 128 and GQ <= 16 (checked on the host).

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
override GATED: bool = false;

const CH: u32 = 32u;
const WG: u32 = 256u;

@group(0) @binding(0) var<storage, read> qkv: array<vec4<f32>>;  // [M, (H + 2 KVH) D]
@group(0) @binding(1) var<storage, read> kc: array<vec2<u32>>;  // f16 [KVH, MAXCTX, D], 4 values per entry
@group(0) @binding(2) var<storage, read> vc: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> att: array<vec4<f32>>;  // [M, H D]
@group(0) @binding(4) var<uniform> p: Params;
@group(0) @binding(5) var<storage, read> gsrc: array<vec4<f32>>;

var<workgroup> qs: array<vec4<f32>, (H / KVH) * (D / 4u + 1u)>;
var<workgroup> ps: array<f32, (H / KVH) * CH>;
var<workgroup> ms: array<vec2<f32>, H / KVH>;  // (running max, rescale factor of this chunk)
var<workgroup> ls: array<f32, H / KVH>;        // running denominator

fn h4(u: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(u.x), unpack2x16float(u.y));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let gq = H / KVH;
    let d4 = D / 4u;
    let qstride = d4 + 1u;
    let mi = wid.x;
    let kh = wid.y;
    let n_keys = p.pos + mi + 1u;
    let qrow = mi * (H + 2u * KVH) * d4;
    for (var i = lid; i < gq * d4; i += WG) {
        qs[(i / d4) * qstride + i % d4] = qkv[qrow + kh * gq * d4 + i];
    }
    if (lid < gq) {
        ms[lid] = vec2<f32>(-3.0e38, 0.0);
        ls[lid] = 0.0;
    }
    // each thread owns up to two (head, dim4) outputs
    var acc: array<vec4<f32>, 2>;
    let scale = inverseSqrt(f32(D));
    workgroupBarrier();

    for (var t0 = 0u; t0 < n_keys; t0 += CH) {
        for (var e = lid; e < gq * CH; e += WG) {
            let h = e % gq;
            let tt = e / gq;
            var s = -3.0e38;
            if (t0 + tt < n_keys) {
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
            var mc = ms[lid].x;
            for (var i = 0u; i < CH; i++) {
                mc = max(mc, ps[lid * CH + i]);
            }
            let alpha = exp(ms[lid].x - mc);
            var l = ls[lid] * alpha;
            for (var i = 0u; i < CH; i++) {
                l += exp(ps[lid * CH + i] - mc);
            }
            ls[lid] = l;
            ms[lid] = vec2<f32>(mc, alpha);
        }
        workgroupBarrier();
        for (var e = lid; e < gq * CH; e += WG) {
            ps[e] = exp(ps[e] - ms[e / CH].x);
        }
        workgroupBarrier();

        let n = min(CH, n_keys - t0);
        for (var k = 0u; k < 2u; k++) {
            let e = lid + k * WG;
            if (e < gq * d4) {
                let h = e / d4;
                let vb = (kh * MAXCTX + t0) * d4 + e % d4;
                var a = acc[k] * ms[h].y;
                for (var i = 0u; i < n; i++) {
                    a += ps[h * CH + i] * h4(vc[vb + i * d4]);
                }
                acc[k] = a;
            }
        }
        workgroupBarrier();
    }

    for (var k = 0u; k < 2u; k++) {
        let e = lid + k * WG;
        if (e < gq * d4) {
            var o = acc[k] / ls[e / d4];
            if (GATED) {
                let hq = kh * gq + e / d4;
                o /= 1.0 + exp(-gsrc[mi * (2u * H + 2u * KVH) * d4 + hq * 2u * d4 + d4 + e % d4]);
            }
            att[mi * H * d4 + kh * gq * d4 + e] = o;
        }
    }
}
