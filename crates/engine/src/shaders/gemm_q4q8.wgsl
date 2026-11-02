// Prefill: Y[M, N] = X W^T for M tokens. int4 weights (asymmetric, group 32) x int8
// activations (symmetric, group 32), integer dot products via dot4I8Packed.
//   MODE : 0 = store, 1 = Y += XW^T (residual), 2 = Y = silu(X W[n]) * (X W[n + N])
// Workgroup tile: TM tokens x TN output rows; K is staged GK groups at a time.
// Thread (tn, tm) owns rows n0 + 2tn + {0,1} and tokens t0 + 4tm + {0..3}.
// N % TN == 0 and (K / 32) % GK == 0 are checked on the host.

struct Params {
    pos: u32,
    seq_len: u32,
    m: u32,
    slot: u32,
}

override K: u32;
override N: u32;
override MODE: u32 = 0u;

const TM: u32 = 32u;
const TN: u32 = 32u;
const GK: u32 = 4u;
const WG: u32 = 128u;

@group(0) @binding(0) var<storage, read> wq: array<vec4<u32>>;  // [N(*2), K/32]
@group(0) @binding(1) var<storage, read> ws: array<u32>;        // [N(*2), K/32]
@group(0) @binding(2) var<storage, read> xq: array<vec4<u32>>;  // [M, K/32, 2]
@group(0) @binding(3) var<storage, read> xs: array<vec2<f32>>;  // [M, K/32]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;   // [M, N]
@group(0) @binding(5) var<uniform> p: Params;

// [g][half][row] so that threads with consecutive tn read consecutive entries
var<workgroup> sw: array<vec4<u32>, GK * 2u * TN>;
var<workgroup> ss: array<vec2<f32>, GK * 2u * TN>;
// [g][token][2]
var<workgroup> sx: array<vec4<u32>, GK * TM * 2u>;
var<workgroup> sxs: array<vec2<f32>, GK * TM>;

// one group: 8 nibble-unpacked weight words (lo, hi) against 8 int8x4 activation words
fn gdot(lo: vec4<u32>, hi: vec4<u32>, a: vec4<u32>, b: vec4<u32>) -> i32 {
    return dot4I8Packed(lo.x, a.x) + dot4I8Packed(hi.x, a.y)
         + dot4I8Packed(lo.y, a.z) + dot4I8Packed(hi.y, a.w)
         + dot4I8Packed(lo.z, b.x) + dot4I8Packed(hi.z, b.y)
         + dot4I8Packed(lo.w, b.z) + dot4I8Packed(hi.w, b.w);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let G = K / 32u;
    let n0 = wid.x * TN;
    let t0 = wid.y * TM;
    let tn = lid % 16u;
    let tm = lid / 16u;
    let halves = select(1u, 2u, MODE == 2u);

    var acc: array<f32, 16>;  // [half][r][t] = half * 8 + r * 4 + t
    for (var g0 = 0u; g0 < G; g0 += GK) {
        for (var i = lid; i < GK * halves * TN; i += WG) {
            let g = i / (halves * TN);
            let hf = (i / TN) % halves;
            let r = i % TN;
            let idx = (n0 + r + hf * N) * G + g0 + g;
            let si = (g * 2u + hf) * TN + r;
            sw[si] = wq[idx];
            ss[si] = unpack2x16float(ws[idx]);
        }
        for (var i = lid; i < GK * TM; i += WG) {
            let g = i / TM;
            let t = i % TM;
            let tok = min(t0 + t, p.m - 1u);  // rows past M compute garbage that is never stored
            let xi = tok * (K / 16u) + 2u * (g0 + g);
            sx[2u * i] = xq[xi];
            sx[2u * i + 1u] = xq[xi + 1u];
            sxs[i] = xs[tok * G + g0 + g];
        }
        workgroupBarrier();

        for (var g = 0u; g < GK; g++) {
            // this thread's 4 tokens stay in registers; each weight group is unpacked once
            var a: array<vec4<u32>, 4>;
            var b: array<vec4<u32>, 4>;
            var xsc: array<vec2<f32>, 4>;
            for (var t = 0u; t < 4u; t++) {
                let xi = g * TM + 4u * tm + t;
                a[t] = sx[2u * xi];
                b[t] = sx[2u * xi + 1u];
                xsc[t] = sxs[xi];
            }
            for (var hf = 0u; hf < halves; hf++) {
                for (var r = 0u; r < 2u; r++) {
                    let wi = (g * 2u + hf) * TN + 2u * tn + r;
                    let w = sw[wi];
                    let s = ss[wi];
                    let lo = w & vec4<u32>(0x0F0F0F0Fu);
                    let hi = (w >> vec4<u32>(4u)) & vec4<u32>(0x0F0F0F0Fu);
                    for (var t = 0u; t < 4u; t++) {
                        acc[hf * 8u + r * 4u + t] += s.x * xsc[t].x * f32(gdot(lo, hi, a[t], b[t])) + s.y * xsc[t].y;
                    }
                }
            }
        }
        workgroupBarrier();
    }

    for (var t = 0u; t < 4u; t++) {
        let tok = t0 + 4u * tm + t;
        if (tok >= p.m) {
            break;
        }
        for (var r = 0u; r < 2u; r++) {
            let o = tok * N + n0 + 2u * tn + r;
            let a = acc[r * 4u + t];
            if (MODE == 0u) {
                y[o] = a;
            } else if (MODE == 1u) {
                y[o] += a;
            } else {
                y[o] = a / (1.0 + exp(-a)) * acc[8u + r * 4u + t];
            }
        }
    }
}
