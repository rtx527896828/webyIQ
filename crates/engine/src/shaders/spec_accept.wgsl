// End of a speculative decode step (one thread). The step ran the main model on
// tok = [x, d] (x at p.pos, draft d at p.pos + 1), giving greedy y[0] (after x) and y[1]
// (after d), and the MTP head on (y[0], y[1]), giving drafts e[0] (for after y[0]) and e[1]
// (for after y[1]).
//   accepted (d == y[0]): emit y[0], y[1]; next step [y[1], e[1]] at p.pos + 2
//   rejected:             emit y[0];       next step [y[0], e[0]] at p.pos + 1
// The next step's Params go to `ctl` (the host copies them into the uniform buffer); the
// DeltaNet state of the last accepted token is in slot (p.slot + 1 + accepted) % 3.

struct Params {
    pos: u32,
    seq_len: u32,
    m: u32,
    slot: u32,
}

@group(0) @binding(0) var<storage, read_write> tok: array<u32>;  // [2]
@group(0) @binding(1) var<storage, read> y: array<u32>;          // [2]
@group(0) @binding(2) var<storage, read> e: array<u32>;          // [2]
@group(0) @binding(3) var<storage, read_write> ctl: Params;
@group(0) @binding(4) var<storage, read_write> out: array<u32>;  // [count, t0, t1, pos, slot]
@group(0) @binding(5) var<uniform> p: Params;

@compute @workgroup_size(1)
fn main() {
    let acc = tok[1] == y[0];
    let a = u32(acc);
    let pos = p.pos + 1u + a;
    let slot = (p.slot + 1u + a) % 3u;
    tok[0] = y[a];
    tok[1] = e[a];
    ctl = Params(pos, pos + 2u, 2u, slot);
    out[0] = 1u + a;
    out[1] = y[0];
    out[2] = y[1];
    out[3] = pos;
    out[4] = slot;
}
