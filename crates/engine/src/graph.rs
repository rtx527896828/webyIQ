//! Model graphs: the dispatch lists of one decode step and one prefill chunk, built once at load
//! time per architecture. Every op is (pipeline, bind group, rule for its workgroup counts).

use std::collections::HashMap;

use wgpu::util::DeviceExt;

use crate::gpu::{Bind, ShaderCaps, bind_group, bind_layout, grid, pipeline, shader_source};
use crate::manifest::{Config, Manifest};

pub(crate) const GEMV_ROWS: u32 = 8; // rows per workgroup (R in gemv_q4q8.wgsl)
pub(crate) const ATTN_CHUNK: u32 = 32; // positions per flash-decoding chunk (CH in the attn shaders)
pub(crate) const PREFILL_CHUNK: u32 = 128; // tokens per prefill submission
pub(crate) const SPEC_TOKENS: u32 = 2; // tokens per speculative step: last token + one draft
pub(crate) const STATE_SLOTS: u64 = 3; // recurrent state slots (see deltanet.wgsl)
const GEMM_TN: u32 = 32; // output rows per GEMM workgroup (TN in gemm_q4q8.wgsl)
const GEMM_TM: u32 = 32; // tokens per GEMM workgroup (TM)

const SHADERS: &[(&str, &str)] = &[
    ("gemv_q4q8", include_str!("shaders/gemv_q4q8.wgsl")),
    ("gemm_q4q8", include_str!("shaders/gemm_q4q8.wgsl")),
    ("quant", include_str!("shaders/quant.wgsl")),
    ("embed", include_str!("shaders/embed.wgsl")),
    ("rope_kv", include_str!("shaders/rope_kv.wgsl")),
    ("qk_norm_rope", include_str!("shaders/qk_norm_rope.wgsl")),
    ("attn", include_str!("shaders/attn.wgsl")),
    ("attn_combine", include_str!("shaders/attn_combine.wgsl")),
    ("attn_prefill", include_str!("shaders/attn_prefill.wgsl")),
    ("deltanet", include_str!("shaders/deltanet.wgsl")),
    ("argmax", include_str!("shaders/argmax.wgsl")),
    ("mtp_in", include_str!("shaders/mtp_in.wgsl")),
    ("spec_accept", include_str!("shaders/spec_accept.wgsl")),
    ("gather_rows", include_str!("shaders/gather_rows.wgsl")),
];

/// How an op's workgroup counts follow the step: `m` tokens in the submission, `seq_len` total
/// (an upper bound when the position is only known on the GPU).
#[derive(Clone, Copy)]
pub(crate) enum Dims {
    Fixed(u32, u32),
    /// 1D count, split to respect the per-dimension limit
    Grid(u32),
    /// (x, m)
    PerToken(u32),
    /// (m, y)
    Tokens(u32),
    /// (x, ceil(m / GEMM_TM))
    GemmTiles(u32),
    /// (ceil(seq_len / ATTN_CHUNK), y * m)
    AttnChunks(u32),
}

impl Dims {
    pub(crate) fn resolve(self, seq_len: u32, m: u32) -> (u32, u32) {
        match self {
            Dims::Fixed(x, y) => (x, y),
            Dims::Grid(n) => grid(n),
            Dims::PerToken(x) => (x, m),
            Dims::Tokens(y) => (m, y),
            Dims::GemmTiles(x) => (x, m.div_ceil(GEMM_TM)),
            Dims::AttnChunks(y) => (seq_len.div_ceil(ATTN_CHUNK), y * m),
        }
    }
}

pub(crate) struct OpDef {
    pub name: &'static str,
    pub pipe: usize,
    pub bg: wgpu::BindGroup,
    pub dims: Dims,
}

/// Speculative decoding with the MTP head (Qwen3.5): each step verifies one draft token.
pub(crate) struct Spec {
    /// one step: main model on `tok` = [last token, draft], lm_head + argmax of both, the MTP
    /// head on the two greedy tokens, and `spec_accept` (writes `tok`, `ctl` and `out`)
    pub step: Vec<OpDef>,
    /// after a prefill chunk's layers: the MTP layer over the chunk (fills its KV cache);
    /// input tokens are the chunk's tokens shifted by one, in `ntok`
    pub prefill: Vec<OpDef>,
    /// MTP norm + lm_head + argmax of the hidden state in `x` (decode buffers) into `draft`
    pub tail: Vec<OpDef>,
    /// one-time setup after the weights are uploaded (gathers the reduced draft lm_head)
    pub init: Vec<OpDef>,
    pub tok: wgpu::Buffer,
    pub ntok: wgpu::Buffer,
    pub draft: wgpu::Buffer,
    /// Params of the next step, written by `spec_accept`
    pub ctl: wgpu::Buffer,
    /// [accepted tokens (1 or 2), t0, t1, pos, slot] of the last step
    pub out: wgpu::Buffer,
}

pub(crate) struct Graph {
    pub pipelines: Vec<wgpu::ComputePipeline>,
    /// one decode step (embedding of `tok` .. last layer), followed by `tail`
    pub decode: Vec<OpDef>,
    /// one prefill chunk (embedding of `ptok` .. last layer, hidden states in `px`)
    pub prefill: Vec<OpDef>,
    /// final norm of `x`, lm_head, argmax into `tok`
    pub tail: Vec<OpDef>,
    pub spec: Option<Spec>,
    pub x: wgpu::Buffer,
    pub px: wgpu::Buffer,
    pub tok: wgpu::Buffer,
    pub ptok: wgpu::Buffer,
    pub params: wgpu::Buffer,
    pub logits: wgpu::Buffer,
    /// recurrent state (conv windows, DeltaNet matrices), zeroed on reset
    pub states: Vec<wgpu::Buffer>,
    pub max_ctx: u32,
    _keep: Vec<wgpu::Buffer>,
}

use Bind::{Read as R, ReadWrite as W, Uniform as U};

struct Builder<'a> {
    device: &'a wgpu::Device,
    weights: &'a HashMap<String, wgpu::Buffer>,
    caps: ShaderCaps,
    modules: HashMap<&'static str, wgpu::ShaderModule>,
    layouts: HashMap<Vec<Bind>, wgpu::BindGroupLayout>,
    pipelines: Vec<wgpu::ComputePipeline>,
    pipeline_ids: HashMap<String, usize>,
}

/// Which dispatch list an op is built for.
#[derive(Clone, Copy, PartialEq)]
enum Path {
    /// one token, GEMV
    Decode,
    /// SPEC_TOKENS tokens (draft verification), one GEMV over all of them
    Spec,
    /// up to PREFILL_CHUNK tokens, GEMM
    Prefill,
}

impl Path {
    fn tokens(self) -> u32 {
        match self {
            Path::Decode => 1,
            Path::Spec => SPEC_TOKENS,
            Path::Prefill => PREFILL_CHUNK,
        }
    }
}

impl<'a> Builder<'a> {
    fn op(
        &mut self,
        name: &'static str,
        shader: &'static str,
        entry: &str,
        consts: &[(&str, f64)],
        binds: &[(Bind, &wgpu::Buffer)],
        dims: Dims,
    ) -> OpDef {
        let kinds: Vec<Bind> = binds.iter().map(|b| b.0).collect();
        let device = self.device;
        let layout = self.layouts.entry(kinds.clone()).or_insert_with(|| bind_layout(device, &kinds)).clone();
        let key = format!("{shader}:{entry}:{consts:?}:{kinds:?}");
        let pipe = match self.pipeline_ids.get(&key) {
            Some(&i) => i,
            None => {
                let caps = self.caps;
                let module = self.modules.entry(shader).or_insert_with(|| {
                    let src = SHADERS.iter().find(|s| s.0 == shader).expect("unknown shader").1;
                    let desc = wgpu::ShaderModuleDescriptor {
                        label: Some(shader),
                        source: wgpu::ShaderSource::Wgsl(shader_source(src, caps).into()),
                    };
                    // All kernel loops are bounded by construction; naga's injected loop guards
                    // block unrolling (and crash DXC on the GEMV). Bounds checks stay on, as in browsers.
                    let checks = wgpu::ShaderRuntimeChecks { force_loop_bounding: false, ..wgpu::ShaderRuntimeChecks::checked() };
                    unsafe { device.create_shader_module_trusted(desc, checks) }
                });
                self.pipelines.push(pipeline(device, &layout, module, shader, entry, consts));
                self.pipeline_ids.insert(key, self.pipelines.len() - 1);
                self.pipelines.len() - 1
            }
        };
        let bufs: Vec<&wgpu::Buffer> = binds.iter().map(|b| b.1).collect();
        OpDef { name, pipe, bg: bind_group(self.device, &layout, &bufs), dims }
    }

    fn w(&self, name: &str) -> Result<&'a wgpu::Buffer, String> {
        self.weights.get(name).ok_or_else(|| format!("missing tensor {name}"))
    }

    /// y (+)= W x for `t` tokens; `w` names the q4 tensor pair, input is (xq, xs).
    #[allow(clippy::too_many_arguments)]
    fn gemv(&mut self, name: &'static str, w: &str, k: u32, n: u32, mode: u32, t: u32, xq: &wgpu::Buffer, xs: &wgpu::Buffer, y: &wgpu::Buffer) -> Result<OpDef, String> {
        let (wq, ws) = (self.w(&format!("{w}.q"))?, self.w(&format!("{w}.s"))?);
        // several tokens: half the rows per workgroup (same accumulator count, twice the workgroups)
        let rows = if t > 1 { GEMV_ROWS / 2 } else { GEMV_ROWS };
        let c = [("K", k as f64), ("N", n as f64), ("MODE", mode as f64), ("R", rows as f64), ("T", t as f64)];
        Ok(self.op(name, "gemv_q4q8", "main", &c, &[(R, wq), (R, ws), (R, xq), (R, xs), (W, y)], Dims::Grid(n / rows)))
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(&mut self, name: &'static str, w: &str, k: u32, n: u32, mode: u32, xq: &wgpu::Buffer, xs: &wgpu::Buffer, y: &wgpu::Buffer, params: &wgpu::Buffer) -> Result<OpDef, String> {
        let (wq, ws) = (self.w(&format!("{w}.q"))?, self.w(&format!("{w}.s"))?);
        let c = [("K", k as f64), ("N", n as f64), ("MODE", mode as f64)];
        Ok(self.op(name, "gemm_q4q8", "main", &c, &[(R, wq), (R, ws), (R, xq), (R, xs), (W, y), (U, params)], Dims::GemmTiles(n / GEMM_TN)))
    }

    /// Product of the path's activations (xq, xs) with q4 tensor `w`: GEMM for prefill, GEMV otherwise.
    #[allow(clippy::too_many_arguments)]
    fn matmul(&mut self, path: Path, name: &'static str, w: &str, k: u32, n: u32, mode: u32, a: &Acts, y: &wgpu::Buffer, params: &wgpu::Buffer) -> Result<OpDef, String> {
        match path {
            Path::Prefill => self.gemm(name, w, k, n, mode, &a.xq, &a.xs, y, params),
            _ => self.gemv(name, w, k, n, mode, path.tokens(), &a.xq, &a.xs, y),
        }
    }

    /// int8 activation quantization of the rows of `x` (K wide), optionally RMSNorm'ed with `norm`.
    #[allow(clippy::too_many_arguments)]
    fn quant(&mut self, name: &'static str, k: u32, eps: f64, x: &wgpu::Buffer, norm: Option<&wgpu::Buffer>, dummy: &wgpu::Buffer, xq: &wgpu::Buffer, xs: &wgpu::Buffer) -> OpDef {
        let c = [("K", k as f64), ("NORM", norm.is_some() as u32 as f64), ("EPS", eps)];
        // the unused norm-weight slot takes any read-only buffer
        self.op(name, "quant", "main", &c, &[(R, x), (R, norm.unwrap_or(dummy)), (W, xq), (W, xs)], Dims::Tokens(1))
    }
}

/// Activation buffers of one path (decode: rows = 1, prefill and spec: rows = PREFILL_CHUNK).
struct Acts {
    x: wgpu::Buffer,
    qkv: wgpu::Buffer,
    proj: wgpu::Buffer,
    att: wgpu::Buffer,
    h: wgpu::Buffer,
    xq: wgpu::Buffer,
    xs: wgpu::Buffer,
    tok: wgpu::Buffer,
}

/// Model-wide values and shared buffers for the per-layer builders.
struct Ctx<'a> {
    cfg: &'a Config,
    qwen: bool,
    eps: f64,
    qkvg_dim: u32,
    inproj_dim: u32,
    max_ctx: u32,
    params: &'a wgpu::Buffer,
    rope: &'a wgpu::Buffer,
    part: &'a wgpu::Buffer,
}

impl Ctx<'_> {
    fn attn_c(&self) -> [(&'static str, f64); 4] {
        let c = self.cfg;
        [("H", c.num_heads as f64), ("KVH", c.num_kv_heads as f64), ("D", c.head_dim as f64), ("MAXCTX", self.max_ctx as f64)]
    }
}

/// Full-attention block of layer `p` (weights `{p}.*`), pre-norm to o projection (+ residual).
#[allow(clippy::too_many_arguments)]
fn attention(b: &mut Builder, cx: &Ctx, p: &str, path: Path, a: &Acts, kc: &wgpu::Buffer, vc: &wgpu::Buffer, ops: &mut Vec<OpDef>) -> Result<(), String> {
    let cfg = cx.cfg;
    let (hd, nh, nkv, d) = (cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    let lw = |s: &str| format!("{p}.{s}");
    let attn_c = cx.attn_c();
    let (params, rope) = (cx.params, cx.rope);
    let norm = b.w(&lw("attn_norm"))?;
    ops.push(b.quant("norm_quant", hd, cx.eps, &a.x, Some(norm), rope, &a.xq, &a.xs));
    if cx.qwen {
        let (qn, kn) = (b.w(&lw("q_norm"))?, b.w(&lw("k_norm"))?);
        ops.push(b.matmul(path, "qkv", &lw("qkv"), hd, cx.qkvg_dim, 0, a, &a.proj, params)?);
        let c = [attn_c[0], attn_c[1], attn_c[2], attn_c[3], ("ROT", cfg.rotary_dim() as f64), ("EPS", cx.eps)];
        let binds = [(R, &a.proj), (W, &a.qkv), (W, kc), (W, vc), (R, rope), (R, qn), (R, kn), (U, params)];
        ops.push(b.op("qk_norm_rope", "qk_norm_rope", "main", &c, &binds, Dims::PerToken(nh + 2 * nkv)));
    } else {
        ops.push(b.matmul(path, "qkv", &lw("qkv"), hd, (nh + 2 * nkv) * d, 0, a, &a.qkv, params)?);
        let rope_threads = (nh + nkv) * d / 4 + nkv * d / 2;
        let binds = [(W, &a.qkv), (W, kc), (W, vc), (R, rope), (U, params)];
        ops.push(b.op("rope_kv", "rope_kv", "main", &attn_c, &binds, Dims::PerToken(rope_threads.div_ceil(64))));
    }
    // the gate (Qwen) is read from the fused projection; Llama binds any read-only buffer
    let gsrc = if cx.qwen { &a.proj } else { rope };
    if path == Path::Prefill {
        let c = [attn_c[0], attn_c[1], attn_c[2], attn_c[3], ("GATED", cx.qwen as u32 as f64)];
        let binds = [(R, &a.qkv), (R, kc), (R, vc), (W, &a.att), (U, params), (R, gsrc)];
        ops.push(b.op("attn", "attn_prefill", "main", &c, &binds, Dims::Tokens(nkv)));
        ops.push(b.quant("quant", nh * d, cx.eps, &a.att, None, rope, &a.xq, &a.xs));
    } else {
        let binds = [(R, &a.qkv), (R, kc), (R, vc), (W, cx.part), (U, params)];
        ops.push(b.op("attn", "attn", "attn", &attn_c, &binds, Dims::AttnChunks(nkv)));
        let c = [attn_c[0], attn_c[2], attn_c[3], attn_c[1], ("GATED", cx.qwen as u32 as f64)];
        let binds = [(R, cx.part), (W, &a.xq), (W, &a.xs), (U, params), (R, gsrc)];
        ops.push(b.op("attn_combine", "attn_combine", "main", &c, &binds, Dims::PerToken(nh)));
    }
    ops.push(b.matmul(path, "o", &lw("o"), nh * d, hd, 1, a, &a.x, params)?);
    Ok(())
}

/// Gated DeltaNet block of layer `p`, pre-norm to o projection (+ residual).
#[allow(clippy::too_many_arguments)]
fn deltanet(b: &mut Builder, cx: &Ctx, p: &str, path: Path, a: &Acts, conv_state: &wgpu::Buffer, state: &wgpu::Buffer, ops: &mut Vec<OpDef>) -> Result<(), String> {
    let cfg = cx.cfg;
    let (hd, lnh, lvd) = (cfg.hidden_size, cfg.linear_num_heads, cfg.linear_value_head_dim);
    let lw = |s: &str| format!("{p}.{s}");
    let norm = b.w(&lw("attn_norm"))?;
    ops.push(b.quant("norm_quant", hd, cx.eps, &a.x, Some(norm), cx.rope, &a.xq, &a.xs));
    ops.push(b.matmul(path, "in_proj", &lw("in_proj"), hd, cx.inproj_dim, 0, a, &a.proj, cx.params)?);
    let (conv_w, neg_a, dt_bias, gate_norm) = (b.w(&lw("conv_w"))?, b.w(&lw("neg_a"))?, b.w(&lw("dt_bias"))?, b.w(&lw("gate_norm"))?);
    let binds = [
        (R, &a.proj), (R, conv_w), (W, conv_state), (W, state), (W, &a.att),
        (R, neg_a), (R, dt_bias), (R, gate_norm), (U, cx.params),
    ];
    let c = [("NH", lnh as f64), ("EPS", cx.eps), ("SPEC", (path == Path::Spec) as u32 as f64)];
    ops.push(b.op("deltanet", "deltanet", "main", &c, &binds, Dims::Fixed(lnh, 1)));
    ops.push(b.quant("quant", lnh * lvd, cx.eps, &a.att, None, cx.rope, &a.xq, &a.xs));
    ops.push(b.matmul(path, "o", &lw("o"), lnh * lvd, hd, 1, a, &a.x, cx.params)?);
    Ok(())
}

/// SwiGLU MLP of layer `p` with its pre-norm (+ residual).
fn mlp(b: &mut Builder, cx: &Ctx, p: &str, path: Path, a: &Acts, ops: &mut Vec<OpDef>) -> Result<(), String> {
    let (hd, ffn) = (cx.cfg.hidden_size, cx.cfg.intermediate_size);
    let lw = |s: &str| format!("{p}.{s}");
    let norm = b.w(&lw("ffn_norm"))?;
    ops.push(b.quant("norm_quant", hd, cx.eps, &a.x, Some(norm), cx.rope, &a.xq, &a.xs));
    ops.push(b.matmul(path, "gate_up", &lw("gate_up"), hd, ffn, 2, a, &a.h, cx.params)?);
    ops.push(b.quant("quant", ffn, cx.eps, &a.h, None, cx.rope, &a.xq, &a.xs));
    ops.push(b.matmul(path, "down", &lw("down"), ffn, hd, 1, a, &a.x, cx.params)?);
    Ok(())
}

/// cos/sin table [max_ctx, rotary_dim / 2], computed the way HF does (f32 inv_freq and angle).
fn rope_table(cfg: &Config, max_ctx: u32) -> Vec<f32> {
    let rot = cfg.rotary_dim();
    let inv: Vec<f32> = (0..rot / 2).map(|j| (1.0 / cfg.rope_theta.powf((2 * j) as f64 / rot as f64)) as f32).collect();
    let mut out = Vec::with_capacity((max_ctx * rot) as usize);
    for p in 0..max_ctx {
        for f in &inv {
            let a = (p as f32 * f) as f64;
            out.push(a.cos() as f32);
            out.push(a.sin() as f32);
        }
    }
    out
}

pub(crate) fn build(
    device: &wgpu::Device,
    manifest: &Manifest,
    weights: &HashMap<String, wgpu::Buffer>,
    max_ctx: u32,
    caps: ShaderCaps,
) -> Result<Graph, String> {
    let cfg = &manifest.config;
    let qwen = manifest.arch == "qwen3_5";
    let (hd, ffn, nh, nkv, d) = (cfg.hidden_size, cfg.intermediate_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    let (lnh, lkd, lvd) = (cfg.linear_num_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim);
    let linear_layers = (0..cfg.num_layers as usize).filter(|&l| cfg.is_linear(l)).count();
    let use_mtp = qwen && cfg.mtp_layers > 0 && weights.contains_key("mtp.fc.q");
    // fused projection widths: Qwen full attention [q|gate per head, k, v]; DeltaNet [q k v | z | b | a]
    let qkvg_dim = (2 * nh + 2 * nkv) * d;
    let inproj_dim = lnh * (2 * lkd + 2 * lvd) + 2 * lnh;

    let mut gemv_rows = vec![hd, ffn, (nh + 2 * nkv) * d, cfg.vocab_size];
    let mut gemm_rows = vec![hd, ffn, (nh + 2 * nkv) * d];
    if qwen {
        gemv_rows.extend([qkvg_dim, inproj_dim]);
        gemm_rows.extend([qkvg_dim, inproj_dim]);
    }
    let shapes_ok = d.is_power_of_two()
        && (64..=256).contains(&d)
        && nh % nkv == 0
        && nh / nkv <= 16
        && (nh / nkv) * d / 4 <= 512
        && cfg.rotary_dim() <= d
        && gemv_rows.iter().all(|n| n % GEMV_ROWS == 0)
        && gemm_rows.iter().all(|n| n % GEMM_TN == 0)
        && [hd, ffn, nh * d].iter().all(|k| k % 128 == 0)
        && (linear_layers == 0 || (lkd == 128 && lvd == 128 && cfg.linear_conv_kernel == 4 && (lnh * lvd) % 128 == 0));
    if !shapes_ok {
        return Err("unsupported model shape".into());
    }
    let max_ctx = max_ctx.min(cfg.max_position_embeddings) / ATTN_CHUNK * ATTN_CHUNK;

    let storage = |label: &str, floats: u64| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: floats.max(1) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let kmax = hd.max(ffn).max(nh * d).max(lnh * lvd).max(2 * hd) as u64;
    let att_dim = (nh * d).max(lnh * lvd) as u64;
    let acts = |rows: u64| Acts {
        x: storage("x", rows * hd as u64),
        qkv: storage("qkv", rows * ((nh + 2 * nkv) * d) as u64),
        proj: storage("proj", if qwen { rows * qkvg_dim.max(inproj_dim) as u64 } else { 1 }),
        att: storage("att", rows * att_dim),
        h: storage("h", rows * ffn as u64),
        xq: storage("xq", rows * kmax / 4),
        xs: storage("xs", rows * kmax / 32 * 2),
        tok: storage("tok", rows),
    };
    // the spec path runs on the prefill buffers (the two are never in flight together)
    let dec = acts(1);
    let pre = acts(PREFILL_CHUNK as u64);
    let part = storage("attn_part", (SPEC_TOKENS * nh * (max_ctx / ATTN_CHUNK) * (d + 2)) as u64);
    let logits = storage("logits", (SPEC_TOKENS * cfg.vocab_size) as u64);
    let rope = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rope"),
        contents: bytemuck::cast_slice(&rope_table(cfg, max_ctx)),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut b = Builder {
        device,
        weights,
        caps,
        modules: HashMap::new(),
        layouts: HashMap::new(),
        pipelines: Vec::new(),
        pipeline_ids: HashMap::new(),
    };
    let cx = Ctx { cfg, qwen, eps: cfg.rms_norm_eps, qkvg_dim, inproj_dim, max_ctx, params: &params, rope: &rope, part: &part };
    let mut keep = Vec::new();
    let mut states = Vec::new();
    let mut paths = vec![(Path::Decode, &dec), (Path::Prefill, &pre)];
    if use_mtp {
        paths.push((Path::Spec, &pre));
    }
    let mut lists: Vec<Vec<OpDef>> = paths.iter().map(|_| Vec::new()).collect();
    let spec_tok = storage("spec_tok", SPEC_TOKENS as u64);

    let (eq, es) = (b.w("embed.q")?, b.w("embed.s")?);
    let embed_wg = (hd / 8).div_ceil(64);
    for ((path, a), ops) in paths.iter().zip(&mut lists) {
        let tok = if *path == Path::Spec { &spec_tok } else { &a.tok };
        ops.push(b.op("embed", "embed", "embed", &[("K", hd as f64)], &[(R, eq), (R, es), (R, tok), (W, &a.x)], Dims::PerToken(embed_wg)));
    }

    for l in 0..cfg.num_layers as usize {
        let p = format!("l{l}");
        if cfg.is_linear(l) {
            let conv_state = storage("conv_state", STATE_SLOTS * (lnh * (2 * lkd + lvd) * 3) as u64);
            let state = storage("deltanet_state", STATE_SLOTS * (lnh * lkd * lvd) as u64);
            for ((path, a), ops) in paths.iter().zip(&mut lists) {
                deltanet(&mut b, &cx, &p, *path, a, &conv_state, &state, ops)?;
            }
            states.extend([conv_state, state]);
        } else {
            let kv_words = (nkv * max_ctx * d / 2) as u64; // f16 cache, two values per u32
            let (kc, vc) = (storage("k_cache", kv_words), storage("v_cache", kv_words));
            for ((path, a), ops) in paths.iter().zip(&mut lists) {
                attention(&mut b, &cx, &p, *path, a, &kc, &vc, ops)?;
            }
            keep.extend([kc, vc]);
        }
        for ((path, a), ops) in paths.iter().zip(&mut lists) {
            mlp(&mut b, &cx, &p, *path, a, ops)?;
        }
    }

    let head = if cfg.tie_word_embeddings { "embed" } else { "lm_head" };
    let final_norm = b.w("norm")?;
    let v = cfg.vocab_size;
    // the draft lm_head: a subset of the (tied) embedding rows, `mtp.head_ids`, if present
    let draft_ids = weights.get("mtp.head_ids").filter(|_| use_mtp && cfg.tie_word_embeddings);
    let draft_v = draft_ids.map_or(0, |ids| (ids.size() / 4) as u32);
    // the draft head has its own logits, so `logits` always holds the main model's
    let dlogits = storage("draft_logits", if use_mtp { (SPEC_TOKENS * v) as u64 } else { 1 });
    // norm + lm_head (tensor `head`, `v` rows; `ids` maps rows to tokens) + argmax of the rows of `a.x` into `out`
    #[allow(clippy::too_many_arguments)]
    let head_ops = |b: &mut Builder, path: Path, a: &Acts, norm: &wgpu::Buffer, head: &str, v: u32, ids: Option<&wgpu::Buffer>, out: &wgpu::Buffer| -> Result<Vec<OpDef>, String> {
        let c = [("V", v as f64), ("MAPPED", ids.is_some() as u32 as f64)];
        let lg = if std::ptr::eq(norm, final_norm) { &logits } else { &dlogits };
        Ok(vec![
            b.quant("norm_quant", hd, cx.eps, &a.x, Some(norm), &rope, &a.xq, &a.xs),
            b.gemv("lm_head", head, hd, v, 0, path.tokens(), &a.xq, &a.xs, lg)?,
            b.op("argmax", "argmax", "argmax", &c, &[(R, lg), (W, out), (R, ids.unwrap_or(&rope))], Dims::PerToken(1)),
        ])
    };
    let tail = head_ops(&mut b, Path::Decode, &dec, final_norm, head, v, None, &dec.tok)?;

    let spec = if use_mtp {
        let ntok = storage("mtp_tok", PREFILL_CHUNK as u64);
        let (y, e) = (storage("spec_y", SPEC_TOKENS as u64), storage("spec_e", SPEC_TOKENS as u64));
        let draft = storage("draft", 1);
        let ctl = storage("spec_ctl", 4);
        let out = storage("spec_out", 8);
        let kv_words = (nkv * max_ctx * d / 2) as u64;
        let (kc, vc) = (storage("mtp_k_cache", kv_words), storage("mtp_v_cache", kv_words));
        let (ne, nhid, mnorm) = (b.w("mtp.norm_e")?, b.w("mtp.norm_h")?, b.w("mtp.norm")?);

        // the MTP layer on rows of `a.x` (main hidden states, before the final norm), next tokens in `tok`
        let mtp_layer = |b: &mut Builder, path: Path, a: &Acts, tok: &wgpu::Buffer| -> Result<Vec<OpDef>, String> {
            // embeddings of the next tokens go to the (free at this point) attention output buffer
            let mut ops = vec![b.op("embed", "embed", "embed", &[("K", hd as f64)], &[(R, eq), (R, es), (R, tok), (W, &a.att)], Dims::PerToken(embed_wg))];
            let binds = [(R, &a.att), (R, &a.x), (R, final_norm), (R, ne), (R, nhid), (W, &a.xq), (W, &a.xs)];
            ops.push(b.op("mtp_in", "mtp_in", "main", &[("K", hd as f64), ("EPS", cx.eps)], &binds, Dims::PerToken(2)));
            ops.push(b.matmul(path, "mtp_fc", "mtp.fc", 2 * hd, hd, 0, a, &a.x, &params)?);
            attention(b, &cx, "mtp", path, a, &kc, &vc, &mut ops)?;
            mlp(b, &cx, "mtp", path, a, &mut ops)?;
            Ok(ops)
        };
        let mut init = Vec::new();
        let (dhead, dv) = match draft_ids {
            Some(ids) => {
                let (dq, ds) = (b.w("mtp.head.q")?, b.w("mtp.head.s")?);
                let c = [("W", (hd / 8) as f64), ("WS", (hd / 32) as f64)];
                let binds = [(R, eq), (R, es), (R, ids), (W, dq), (W, ds)];
                init.push(b.op("gather_rows", "gather_rows", "main", &c, &binds, Dims::Fixed((hd / 8).div_ceil(64), draft_v)));
                ("mtp.head", draft_v)
            }
            None => (head, v),
        };
        let prefill = mtp_layer(&mut b, Path::Prefill, &pre, &ntok)?;
        let tail = head_ops(&mut b, Path::Decode, &dec, mnorm, dhead, dv, draft_ids, &draft)?;

        let mut step = lists.pop().expect("spec path");
        step.extend(head_ops(&mut b, Path::Spec, &pre, final_norm, head, v, None, &y)?);
        step.extend(mtp_layer(&mut b, Path::Spec, &pre, &y)?);
        step.extend(head_ops(&mut b, Path::Spec, &pre, mnorm, dhead, dv, draft_ids, &e)?);
        let binds = [(W, &spec_tok), (R, &y), (R, &e), (W, &ctl), (W, &out), (U, &params)];
        step.push(b.op("spec_accept", "spec_accept", "main", &[], &binds, Dims::Fixed(1, 1)));
        keep.extend([y, e, kc, vc, dlogits]);
        Some(Spec { step, prefill, tail, init, tok: spec_tok, ntok, draft, ctl, out })
    } else {
        keep.extend([spec_tok, dlogits]);
        None
    };

    let prefill = lists.pop().expect("prefill path");
    let decode = lists.pop().expect("decode path");
    keep.extend([part, rope, dec.qkv, dec.proj, dec.att, dec.h, dec.xq, dec.xs, pre.qkv, pre.proj, pre.att, pre.h, pre.xq, pre.xs]);
    Ok(Graph {
        pipelines: b.pipelines,
        decode,
        prefill,
        tail,
        spec,
        x: dec.x,
        px: pre.x,
        tok: dec.tok,
        ptok: pre.tok,
        params,
        logits,
        states,
        max_ctx,
        _keep: keep,
    })
}
