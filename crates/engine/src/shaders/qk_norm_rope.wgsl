// Qwen3.5 full-attention input stage. Source rows (from the fused q/gate/k/v projection):
//   [ head 0: q (D) | gate (D) | head 1: q | gate | ... | k (KVH x D) | v (KVH x D) ]
// For q and k heads: per-head RMSNorm (weights stored as 1 + w), then rotary embedding on the
// first ROT dims (rotate-half within those dims). q goes to `qout` with row layout
// [q (H x D) | unused (2 KVH x D)], k and v go to the f16 cache [KVH, MAXCTX, D / 2].
// One workgroup per (head, token); head in [0, H) = q, [H, H + KVH) = k, rest = v.
// Thread j owns dims 2j and 2j + 1 so that it writes whole packed f16 pairs.

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
override ROT: u32;
override EPS: f32 = 1e-6;

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> qout: array<f32>;
@group(0) @binding(2) var<storage, read_write> kc: array<u32>;
@group(0) @binding(3) var<storage, read_write> vc: array<u32>;
@group(0) @binding(4) var<storage, read> rope: array<vec2<f32>>;  // [MAXCTX, ROT / 2] (cos, sin)
@group(0) @binding(5) var<storage, read> qn: array<f32>;
@group(0) @binding(6) var<storage, read> kn: array<f32>;
@group(0) @binding(7) var<uniform> p: Params;

var<workgroup> v: array<f32, D>;
var<workgroup> red: array<f32, D / 2u>;

@compute @workgroup_size(D / 2u)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) j: u32) {
    let head = wid.x;
    let m = wid.y;
    let pos = p.pos + m;
    let row = m * (2u * H + 2u * KVH) * D;
    let is_q = head < H;
    let is_v = head >= H + KVH;
    var base: u32;
    if (is_q) {
        base = row + head * 2u * D;
    } else if (!is_v) {
        base = row + 2u * H * D + (head - H) * D;
    } else {
        base = row + 2u * H * D + KVH * D + (head - H - KVH) * D;
    }
    var x = vec2<f32>(src[base + 2u * j], src[base + 2u * j + 1u]);

    if (!is_v) {
        red[j] = dot(x, x);
        workgroupBarrier();
        for (var s = D / 4u; s > 0u; s >>= 1u) {
            if (j < s) {
                red[j] += red[j + s];
            }
            workgroupBarrier();
        }
        let inv = inverseSqrt(red[0] / f32(D) + EPS);
        var w: vec2<f32>;
        if (is_q) {
            w = vec2<f32>(qn[2u * j], qn[2u * j + 1u]);
        } else {
            w = vec2<f32>(kn[2u * j], kn[2u * j + 1u]);
        }
        x *= inv * w;
        v[2u * j] = x.x;
        v[2u * j + 1u] = x.y;
        workgroupBarrier();
        let half = ROT / 2u;
        for (var k = 0u; k < 2u; k++) {
            let d = 2u * j + k;
            if (d < ROT) {
                let cs = rope[pos * half + d % half];
                if (d < half) {
                    x[k] = v[d] * cs.x - v[d + half] * cs.y;
                } else {
                    x[k] = v[d] * cs.x + v[d - half] * cs.y;
                }
            }
        }
    }

    if (is_q) {
        let o = m * (H + 2u * KVH) * D + head * D + 2u * j;
        qout[o] = x.x;
        qout[o + 1u] = x.y;
    } else if (!is_v) {
        kc[((head - H) * MAXCTX + pos) * (D / 2u) + j] = pack2x16float(x);
    } else {
        vc[((head - H - KVH) * MAXCTX + pos) * (D / 2u) + j] = pack2x16float(x);
    }
}
