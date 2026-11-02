// Embedding gather with int4 dequant: x[m] = embed[tok_in[m]], m = global_invocation_id.y.
// Packing (see convert.py): u32 i holds weights 8i..8i+7, byte b = w[8i+b] | w[8i+4+b] << 4.

override K: u32;  // hidden size

@group(0) @binding(0) var<storage, read> eq: array<u32>;   // [V, K/8]
@group(0) @binding(1) var<storage, read> es: array<u32>;   // [V, K/32]
@group(0) @binding(2) var<storage, read> tok_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(64)
fn embed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= K / 8u) {
        return;
    }
    let m = gid.y;
    let t = tok_in[m];
    let xb = m * K;
    let w = eq[t * (K / 8u) + i];
    let s = unpack2x16float(es[t * (K / 32u) + i / 4u]);
    for (var b = 0u; b < 4u; b++) {
        x[xb + i * 8u + b] = f32((w >> (8u * b)) & 15u) * s.x + s.y;
        x[xb + i * 8u + 4u + b] = f32((w >> (8u * b + 4u)) & 15u) * s.x + s.y;
    }
}
