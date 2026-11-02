// Qwen3.5 MTP input stage for row m (one workgroup per (half, m)):
//   half 0: norm_e(emb[m])                   -> int8 groups of input row m, dims [0, K)
//   half 1: norm_h(final_norm(x[m]))         -> dims [K, 2K)
// i.e. the int8-quantized cat(...) that feeds the MTP fc projection (K -> 2K wide rows).
// x is the main model's last hidden state (before the final norm), emb the embedding of the
// token after it (embed.wgsl).

override K: u32;  // hidden size
override EPS: f32 = 1e-6;

const WG: u32 = 256u;

@group(0) @binding(0) var<storage, read> emb: array<f32>;  // [M, K]
@group(0) @binding(1) var<storage, read> x: array<f32>;    // [M, K]
@group(0) @binding(2) var<storage, read> nf: array<f32>;   // final norm
@group(0) @binding(3) var<storage, read> ne: array<f32>;   // norm_e
@group(0) @binding(4) var<storage, read> nh: array<f32>;   // norm_h
@group(0) @binding(5) var<storage, read_write> xq: array<u32>;
@group(0) @binding(6) var<storage, read_write> xs: array<vec2<f32>>;

var<workgroup> v: array<f32, K>;
var<workgroup> red: array<f32, WG>;

// 1 / rms(v)
fn inv_rms(tid: u32) -> f32 {
    var ss = 0.0;
    for (var i = tid; i < K; i += WG) {
        ss += v[i] * v[i];
    }
    red[tid] = ss;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            red[tid] += red[tid + s];
        }
        workgroupBarrier();
    }
    let r = inverseSqrt(red[0] / f32(K) + EPS);
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let half = wid.x;
    let m = wid.y;
    if (half == 0u) {
        for (var i = tid; i < K; i += WG) {
            v[i] = emb[m * K + i];
        }
        workgroupBarrier();
        let r = inv_rms(tid);
        for (var i = tid; i < K; i += WG) {
            v[i] *= r * ne[i];
        }
    } else {
        for (var i = tid; i < K; i += WG) {
            v[i] = x[m * K + i];
        }
        workgroupBarrier();
        let r0 = inv_rms(tid);
        for (var i = tid; i < K; i += WG) {
            v[i] *= r0 * nf[i];
        }
        workgroupBarrier();
        let r1 = inv_rms(tid);
        for (var i = tid; i < K; i += WG) {
            v[i] *= r1 * nh[i];
        }
    }
    workgroupBarrier();

    let row = 2u * m + half;  // K-wide half rows of the 2K-wide input row m
    for (var g = tid; g < K / 32u; g += WG) {
        var amax = 0.0;
        for (var i = 0u; i < 32u; i++) {
            amax = max(amax, abs(v[g * 32u + i]));
        }
        let dx = amax / 127.0;
        let inv = select(0.0, 1.0 / dx, amax > 0.0);
        var sum = 0;
        for (var i = 0u; i < 8u; i++) {
            let e = vec4<f32>(v[g * 32u + 4u * i], v[g * 32u + 4u * i + 1u], v[g * 32u + 4u * i + 2u], v[g * 32u + 4u * i + 3u]);
            let q = vec4<i32>(round(e * inv));
            sum += q.x + q.y + q.z + q.w;
            xq[row * (K / 4u) + g * 8u + i] = pack4xI8(q);
        }
        xs[row * (K / 32u) + g] = vec2<f32>(dx, dx * f32(sum));
    }
}
