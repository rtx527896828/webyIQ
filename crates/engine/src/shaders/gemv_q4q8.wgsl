// y = W x for T decode tokens (T = 2 when verifying a speculative draft): int4 weights
// (asymmetric, group 32) x int8 activations (symmetric, group 32), integer dot products via
// dot4I8Packed. Each weight group is loaded and unpacked once for all T tokens.
//   MODE : 0 = store, 1 = y += Wx (residual), 2 = y = silu(W[n] x) * W[n + N] x
// Rows of x / y are tokens (stride K / N).
// Per group:  sum_i (q_i s + m)(xq_i dx) = s dx sum_i q_i xq_i + m (dx sum_i xq_i)
// N must be a multiple of R (checked on the host).
// The `//#if SUBGROUPS` blocks are selected by the host (see shader_source in lib.rs).

override K: u32;
override N: u32;
override MODE: u32 = 0u;
override R: u32 = 8u;   // output rows per workgroup
override T: u32 = 1u;   // tokens; (MODE 2 ? 2 : 1) * R * T <= 32

const MAXT: u32 = 2u;   // T <= MAXT

const WG: u32 = 64u;

@group(0) @binding(0) var<storage, read> wq: array<vec4<u32>>;  // [N(*2), K/32]
@group(0) @binding(1) var<storage, read> ws: array<u32>;        // [N(*2), K/32] pack2x16float(scale, min)
@group(0) @binding(2) var<storage, read> xq: array<vec4<u32>>;  // [T, K/32, 2] int8 x4 packed
@group(0) @binding(3) var<storage, read> xs: array<vec2<f32>>;  // [T, K/32] (dx, dx * sum xq)
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

var<workgroup> red: array<f32, 2048>;  // 32 * WG

// lo / hi: low and high nibbles of a weight group (see embed.wgsl for the packing)
fn gdot(lo: vec4<u32>, hi: vec4<u32>, a: vec4<u32>, b: vec4<u32>) -> i32 {
    return dot4I8Packed(lo.x, a.x) + dot4I8Packed(hi.x, a.y)
         + dot4I8Packed(lo.y, a.z) + dot4I8Packed(hi.y, a.w)
         + dot4I8Packed(lo.z, b.x) + dot4I8Packed(hi.z, b.y)
         + dot4I8Packed(lo.w, b.z) + dot4I8Packed(hi.w, b.w);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>,
//#if SUBGROUPS
        @builtin(subgroup_invocation_id) sg_lane: u32,
        @builtin(subgroup_size) sg_size: u32,
//#endif
        @builtin(local_invocation_index) tid: u32) {
    let G = K / 32u;
    let row0 = (wid.y * nwg.x + wid.x) * R;
    let nv = select(R, 2u * R, MODE == 2u);  // accumulators per token: acc[t * nv + v]
    let m4 = vec4<u32>(0x0F0F0F0Fu);

    var acc: array<f32, 32>;
    for (var g = tid; g < G; g += WG) {
        // this group's activations of every token
        var xa: array<vec4<u32>, MAXT>;
        var xb: array<vec4<u32>, MAXT>;
        var xc: array<vec2<f32>, MAXT>;
        for (var t = 0u; t < T; t++) {
            let xg = t * G + g;
            xa[t] = xq[2u * xg];
            xb[t] = xq[2u * xg + 1u];
            xc[t] = xs[xg];
        }
        for (var r = 0u; r < R; r++) {
            let idx = (row0 + r) * G + g;
            let w = wq[idx];
            let lo = w & m4;
            let hi = (w >> vec4<u32>(4u)) & m4;
            let s = unpack2x16float(ws[idx]);
            for (var t = 0u; t < T; t++) {
                acc[t * nv + r] += s.x * xc[t].x * f32(gdot(lo, hi, xa[t], xb[t])) + s.y * xc[t].y;
            }
            if (MODE == 2u) {
                let iu = idx + N * G;
                let wu = wq[iu];
                let lou = wu & m4;
                let hiu = (wu >> vec4<u32>(4u)) & m4;
                let su = unpack2x16float(ws[iu]);
                for (var t = 0u; t < T; t++) {
                    acc[t * nv + R + r] += su.x * xc[t].x * f32(gdot(lou, hiu, xa[t], xb[t])) + su.y * xc[t].y;
                }
            }
        }
    }
    let na = nv * T;

    // reduce acc[v] over the workgroup into red[v * WG]
//#if SUBGROUPS
    let nsg = (WG + sg_size - 1u) / sg_size;
    for (var v = 0u; v < na; v++) {
        let t = subgroupAdd(acc[v]);
        if (sg_lane == 0u) {
            red[v * WG + tid / sg_size] = t;
        }
    }
    workgroupBarrier();
    if (tid < na) {
        var t = 0.0;
        for (var i = 0u; i < nsg; i++) {
            t += red[tid * WG + i];
        }
        red[tid * WG] = t;
    }
    workgroupBarrier();
//#else
    for (var v = 0u; v < na; v++) {
        red[v * WG + tid] = acc[v];
    }
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            for (var v = 0u; v < na; v++) {
                red[v * WG + tid] += red[v * WG + tid + s];
            }
        }
        workgroupBarrier();
    }
//#endif

    if (tid < R * T) {
        let t = tid / R;
        let r = tid % R;
        let row = t * N + row0 + r;
        let a = red[(t * nv + r) * WG];
        if (MODE == 0u) {
            y[row] = a;
        } else if (MODE == 1u) {
            y[row] += a;
        } else {
            let u = red[(t * nv + R + r) * WG];
            y[row] = a / (1.0 + exp(-a)) * u;
        }
    }
}
