// One-time gather of int4 embedding rows (packed weights + scales) for the reduced draft
// lm_head: dst row r = src row ids[r]. One row of workgroups per output row (gid.y).

override W: u32;   // u32 words per packed row (K / 8)
override WS: u32;  // u32 words per scale row (K / 32)

@group(0) @binding(0) var<storage, read> sq: array<u32>;
@group(0) @binding(1) var<storage, read> ss: array<u32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> dq: array<u32>;
@group(0) @binding(4) var<storage, read_write> ds: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let r = gid.y;
    let src = ids[r];
    if (i < W) {
        dq[r * W + i] = sq[src * W + i];
    }
    if (i < WS) {
        ds[r * WS + i] = ss[src * WS + i];
    }
}
