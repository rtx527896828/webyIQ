// Rotary embedding (rotate-half convention) on q and k, then append k, v to the cache.
// qkv row layout: q [H, D] | k [KVH, D] | v [KVH, D] (f32); row m = global_invocation_id.y is the
// token at position pos + m. The cache is f16, two values per u32: [KVH, MAXCTX, D / 2].
// Each rotation thread handles dims (j, j+1) and (j+half, j+half+1), so packed f16 pairs are
// written by a single thread.

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

@group(0) @binding(0) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> kc: array<u32>;
@group(0) @binding(2) var<storage, read_write> vc: array<u32>;
@group(0) @binding(3) var<storage, read> rope: array<vec2<f32>>;  // [MAXCTX, D/2] (cos, sin)
@group(0) @binding(4) var<uniform> p: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = D / 2u;
    let i = gid.x;
    let pos = p.pos + gid.y;
    let qb = gid.y * (H + 2u * KVH) * D;
    let n_rot = (H + KVH) * (half / 2u);
    if (i < n_rot) {
        let head = i / (half / 2u);
        let j = 2u * (i % (half / 2u));
        let base = qb + head * D;
        let c0 = rope[pos * half + j];
        let c1 = rope[pos * half + j + 1u];
        let x0 = vec2<f32>(qkv[base + j], qkv[base + j + 1u]);
        let x1 = vec2<f32>(qkv[base + j + half], qkv[base + j + half + 1u]);
        let cs = vec2<f32>(c0.x, c1.x);
        let sn = vec2<f32>(c0.y, c1.y);
        let o0 = x0 * cs - x1 * sn;
        let o1 = x1 * cs + x0 * sn;
        if (head < H) {
            qkv[base + j] = o0.x;
            qkv[base + j + 1u] = o0.y;
            qkv[base + j + half] = o1.x;
            qkv[base + j + half + 1u] = o1.y;
        } else {
            let dst = ((head - H) * MAXCTX + pos) * half;
            kc[dst + j / 2u] = pack2x16float(o0);
            kc[dst + (j + half) / 2u] = pack2x16float(o1);
        }
    } else if (i < n_rot + KVH * half) {
        let k = i - n_rot;  // packed pair index within the v row
        let src = qb + (H + KVH) * D + 2u * k;
        vc[((k / half) * MAXCTX + pos) * half + k % half] = pack2x16float(vec2<f32>(qkv[src], qkv[src + 1u]));
    }
}
