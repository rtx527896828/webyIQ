// Qwen3.5 Gated DeltaNet token mixer, fused: causal depthwise conv (+ SiLU) on the q/k/v
// channels, L2-normalized q and k, the gated delta rule recurrence and the gated RMSNorm.
// One workgroup per head, thread j owns column j of the head's state S [KD, VD] (in registers)
// and output dim j. Tokens of a submission (p.m, 1 when decoding) are processed in order.
//
// Per token, with decay = exp(g), g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b):
//   S <- decay * S;  delta = (v - S^T k) * beta;  S <- S + k delta^T;  o = S^T q
// computed in one pass over S: S^T k and S^T q of the old state, then
//   o = decay * (S^T q) + delta * (k . q).
//
// proj rows: [ q (NH x KD) | k (NH x KD) | v (NH x VD) | z (NH x VD) | b (NH) | a (NH) ]
// Assumes KD == VD == 128 (checked on the host).
//
// State (conv window and S) lives in three slots; p.slot holds the current one. Normally it is
// updated in place. With SPEC (draft verification), the state after token m goes to slot
// (p.slot + 1 + m) % 3, so after a rejected draft the state after the first token is still there
// (the step's accept kernel picks the next p.slot).

struct Params {
    pos: u32,
    seq_len: u32,
    m: u32,
    slot: u32,
}

override NH: u32;
override EPS: f32 = 1e-6;
override SPEC: bool = false;

const KD: u32 = 128u;
const VD: u32 = 128u;

@group(0) @binding(0) var<storage, read> proj: array<f32>;
@group(0) @binding(1) var<storage, read> conv_w: array<vec4<f32>>;         // [C] taps, oldest first
@group(0) @binding(2) var<storage, read_write> conv_state: array<f32>;  // [3 slots, C, 3] previous inputs, oldest first
@group(0) @binding(3) var<storage, read_write> state: array<f32>;       // [3 slots, NH, KD, VD]
@group(0) @binding(4) var<storage, read_write> out: array<f32>;         // [M, NH x VD]
@group(0) @binding(5) var<storage, read> neg_a: array<f32>;             // -exp(A_log) [NH]
@group(0) @binding(6) var<storage, read> dt_bias: array<f32>;           // [NH]
@group(0) @binding(7) var<storage, read> gate_norm: array<f32>;         // [VD]
@group(0) @binding(8) var<uniform> p: Params;

var<workgroup> qs: array<f32, KD>;
var<workgroup> ks: array<f32, KD>;
var<workgroup> red: array<vec2<f32>, KD>;

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

// causal conv over (window of 3 previous inputs, current) for channel c; shifts the window
fn conv(c: u32, win: ptr<function, vec3<f32>>, x: f32) -> f32 {
    let w = conv_w[c];
    let s = *win;
    *win = vec3<f32>(s.y, s.z, x);
    return silu(w.x * s.x + w.y * s.y + w.z * s.z + w.w * x);
}

fn load_win(slot: u32, C: u32, c: u32) -> vec3<f32> {
    let b = (slot * C + c) * 3u;
    return vec3<f32>(conv_state[b], conv_state[b + 1u], conv_state[b + 2u]);
}

fn store_win(slot: u32, C: u32, c: u32, w: vec3<f32>) {
    let b = (slot * C + c) * 3u;
    conv_state[b] = w.x;
    conv_state[b + 1u] = w.y;
    conv_state[b + 2u] = w.z;
}

// sum of (a, b) over the workgroup
fn reduce2(j: u32, a: f32, b: f32) -> vec2<f32> {
    red[j] = vec2<f32>(a, b);
    workgroupBarrier();
    for (var s = KD / 2u; s > 0u; s >>= 1u) {
        if (j < s) {
            red[j] += red[j + s];
        }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) j: u32) {
    let h = wid.x;
    let C = NH * (2u * KD + VD);
    let row_len = C + NH * VD + 2u * NH;
    let slot_size = NH * KD * VD;
    let sb = h * KD * VD + j;
    let cq = h * KD + j;
    let ck = NH * KD + h * KD + j;
    let cv = 2u * NH * KD + h * VD + j;

    var s: array<f32, KD>;
    for (var i = 0u; i < KD; i++) {
        s[i] = state[p.slot * slot_size + sb + i * VD];
    }
    var wq = load_win(p.slot, C, cq);
    var wk = load_win(p.slot, C, ck);
    var wv = load_win(p.slot, C, cv);

    for (var m = 0u; m < p.m; m++) {
        let row = m * row_len;
        var q = conv(cq, &wq, proj[row + cq]);
        var k = conv(ck, &wk, proj[row + ck]);
        let v = conv(cv, &wv, proj[row + cv]);

        let n = reduce2(j, q * q, k * k);
        q *= inverseSqrt(n.x + 1e-6) * inverseSqrt(f32(KD));
        k *= inverseSqrt(n.y + 1e-6);
        qs[j] = q;
        ks[j] = k;
        workgroupBarrier();

        let b = proj[row + C + NH * VD + h];
        let a = proj[row + C + NH * VD + NH + h] + dt_bias[h];
        let softplus = select(log(1.0 + exp(a)), a, a > 20.0);
        let decay = exp(neg_a[h] * softplus);
        let beta = 1.0 / (1.0 + exp(-b));

        var sk = 0.0;
        var sq = 0.0;
        var kq = 0.0;
        for (var i = 0u; i < KD; i++) {
            sk += s[i] * ks[i];
            sq += s[i] * qs[i];
            kq += ks[i] * qs[i];
        }
        let delta = (v - decay * sk) * beta;
        var o = decay * sq + delta * kq;
        for (var i = 0u; i < KD; i++) {
            s[i] = decay * s[i] + ks[i] * delta;
        }

        let z = proj[row + C + h * VD + j];
        let ms = reduce2(j, o * o, 0.0).x / f32(VD);
        o *= inverseSqrt(ms + EPS) * gate_norm[j] * silu(z);
        out[m * NH * VD + h * VD + j] = o;

        if (SPEC) {
            store_state(&s, &wq, &wk, &wv, (p.slot + 1u + m) % 3u, slot_size, sb, C, cq, ck, cv);
        }
    }
    if (!SPEC) {
        store_state(&s, &wq, &wk, &wv, p.slot, slot_size, sb, C, cq, ck, cv);
    }
}

fn store_state(s: ptr<function, array<f32, KD>>, wq: ptr<function, vec3<f32>>, wk: ptr<function, vec3<f32>>,
               wv: ptr<function, vec3<f32>>, slot: u32, slot_size: u32, sb: u32, C: u32, cq: u32, ck: u32, cv: u32) {
    for (var i = 0u; i < KD; i++) {
        state[slot * slot_size + sb + i * VD] = (*s)[i];
    }
    store_win(slot, C, cq, *wq);
    store_win(slot, C, ck, *wk);
    store_win(slot, C, cv, *wv);
}
