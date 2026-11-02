// Activation quantization for the int8 GEMV input, optionally fused with RMSNorm:
//   v = NORM ? x * nw / rms(x) : x
//   per group of 32: dx = max|v| / 127, xq = round(v / dx), xs = (dx, dx * sum xq)
// One workgroup per row (token); K is at most a few thousand.

override K: u32;
override NORM: bool = false;
override EPS: f32 = 1e-6;

const WG: u32 = 256u;

@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> nw: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> xq: array<u32>;
@group(0) @binding(3) var<storage, read_write> xs: array<vec2<f32>>;

var<workgroup> red: array<f32, WG>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let xb = wid.x * (K / 4u);
    let gb = wid.x * (K / 32u);
    var scale = 1.0;
    if (NORM) {
        var ss = 0.0;
        for (var i = tid; i < K / 4u; i += WG) {
            let v = x[xb + i];
            ss += dot(v, v);
        }
        red[tid] = ss;
        workgroupBarrier();
        for (var s = WG / 2u; s > 0u; s >>= 1u) {
            if (tid < s) {
                red[tid] += red[tid + s];
            }
            workgroupBarrier();
        }
        scale = inverseSqrt(red[0] / f32(K) + EPS);
    }

    for (var g = tid; g < K / 32u; g += WG) {
        var v: array<vec4<f32>, 8>;
        var amax = 0.0;
        for (var i = 0u; i < 8u; i++) {
            var e = x[xb + g * 8u + i] * scale;
            if (NORM) {
                e *= nw[g * 8u + i];
            }
            v[i] = e;
            amax = max(amax, max(max(abs(e.x), abs(e.y)), max(abs(e.z), abs(e.w))));
        }
        let dx = amax / 127.0;
        let inv = select(0.0, 1.0 / dx, amax > 0.0);
        var sum = 0;
        for (var i = 0u; i < 8u; i++) {
            let q = vec4<i32>(round(v[i] * inv));
            sum += q.x + q.y + q.z + q.w;
            xq[xb + g * 8u + i] = pack4xI8(q);
        }
        xs[gb + g] = vec2<f32>(dx, dx * f32(sum));
    }
}
