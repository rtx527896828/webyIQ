//! webyIQ: architecture-specialized WebGPU inference for small LLMs.
//! The same engine runs natively (wgpu over DX12/Vulkan/Metal) and in the browser (WebGPU).
//!
//! `graph` builds the per-architecture dispatch lists once at load time; this module runs them:
//! decode steps (pipelined, greedy token stays on the GPU), batched prefill, profiling, and
//! speculative decoding with the model's MTP head (Qwen3.5), whose step control stays on the GPU.

mod gpu;
mod graph;
mod manifest;

use std::collections::{HashMap, VecDeque};

pub use gpu::ShaderCaps;
pub use manifest::{Config, Manifest, TensorInfo};

use graph::{Graph, OpDef, PREFILL_CHUNK, SPEC_TOKENS};

const READBACK_RING: usize = 8; // staging buffers for pipelined token readback
const READBACK_BYTES: u64 = 32; // one plain token or a spec step's `out`

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    pos: u32,
    seq_len: u32,
    /// tokens in this submission
    m: u32,
    /// current recurrent-state slot (see deltanet.wgsl)
    slot: u32,
}

/// One dispatch, resolved for a given step.
struct Op<'a> {
    name: &'static str,
    pipeline: &'a wgpu::ComputePipeline,
    bg: &'a wgpu::BindGroup,
    wg: (u32, u32),
}

pub struct Engine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    manifest: Manifest,
    pub max_ctx: u32,
    pos: u32,
    weights: HashMap<String, wgpu::Buffer>,
    g: Graph,

    ring: Vec<wgpu::Buffer>,
    ring_next: usize,
    // submitted steps whose token is not read yet: (ring slot, submission)
    inflight: VecDeque<(usize, wgpu::SubmissionIndex)>,
    last_token: u32,
    // recurrent state ran ahead of the consumed tokens (cannot be rolled back)
    state_dirty: bool,
    slot: u32,

    // speculative decoding: enabled by the caller, active after a prefill that ran the MTP head
    spec_enabled: bool,
    spec_active: bool,
    // the next spec step needs its Params from the host (after a prefill)
    spec_sync: bool,
    // position of the next step's first token as of the last step read back
    spec_pos: u32,
    // tokens read back but not handed out yet
    pending: VecDeque<u32>,
    spec_steps: u64,
    spec_tokens: u64,
    // Spec::init has run (it needs the uploaded weights)
    spec_ready: bool,
}

impl Engine {
    /// Limits the engine needs from the adapter (pass to `request_device`).
    pub fn required_limits(adapter: &wgpu::Limits) -> wgpu::Limits {
        wgpu::Limits {
            max_storage_buffer_binding_size: adapter.max_storage_buffer_binding_size,
            max_buffer_size: adapter.max_buffer_size,
            ..wgpu::Limits::default()
        }
    }

    pub fn new(device: wgpu::Device, queue: wgpu::Queue, manifest: Manifest, max_ctx: u32, caps: ShaderCaps) -> Result<Self, String> {
        let limit = device.limits().max_storage_buffer_binding_size as u64;
        let mut weights = HashMap::new();
        for (name, t) in &manifest.tensors {
            if t.bytes > limit {
                return Err(format!("{name}: {} bytes exceeds max_storage_buffer_binding_size {limit}", t.bytes));
            }
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size: t.bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            weights.insert(name.clone(), buf);
        }
        // reduced draft lm_head, gathered from the embedding on the GPU (Spec::init)
        if let Some(ids) = manifest.tensors.get("mtp.head_ids") {
            let rows = ids.bytes / 4;
            let hd = manifest.config.hidden_size as u64;
            for (name, bytes) in [("mtp.head.q", rows * hd / 2), ("mtp.head.s", rows * hd / 32 * 4)] {
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(name),
                    size: bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                weights.insert(name.to_string(), buf);
            }
        }
        let g = graph::build(&device, &manifest, &weights, max_ctx, caps)?;
        let ring = (0..READBACK_RING)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("tok_readback"),
                    size: READBACK_BYTES,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        Ok(Self {
            device,
            queue,
            manifest,
            max_ctx: g.max_ctx,
            pos: 0,
            spec_enabled: g.spec.is_some(),
            weights,
            g,
            ring,
            ring_next: 0,
            inflight: VecDeque::new(),
            last_token: 0,
            state_dirty: false,
            slot: 0,
            spec_active: false,
            spec_sync: false,
            spec_pos: 0,
            pending: VecDeque::new(),
            spec_steps: 0,
            spec_tokens: 0,
            spec_ready: false,
        })
    }

    /// Whether the model has a draft head for speculative decoding.
    pub fn has_speculative(&self) -> bool {
        self.g.spec.is_some()
    }

    /// Use speculative decoding in `next_token` (from the next prefill on). Output is unchanged:
    /// every draft token is verified by the main model.
    pub fn set_speculative(&mut self, on: bool) {
        self.spec_enabled = on && self.g.spec.is_some();
    }

    /// (speculative steps, tokens they produced) since the last reset
    pub fn spec_stats(&self) -> (u64, u64) {
        (self.spec_steps, self.spec_tokens)
    }

    pub fn config(&self) -> &Config {
        &self.manifest.config
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn pos(&self) -> u32 {
        self.pos
    }

    /// Upload every tensor stored in shard `idx`; `data` is the whole shard file.
    pub fn upload_shard(&self, idx: usize, data: &[u8]) {
        for (name, t) in &self.manifest.tensors {
            if t.shard == idx {
                let (s, e) = (t.offset as usize, (t.offset + t.bytes) as usize);
                self.queue.write_buffer(&self.weights[name], 0, &data[s..e]);
            }
        }
        // flush staged uploads so peak memory stays at ~one shard
        self.queue.submit([]);
    }

    /// Start a new sequence (clears recurrent state).
    pub fn reset(&mut self) {
        self.pos = 0;
        self.inflight.clear();
        self.state_dirty = false;
        self.slot = 0;
        self.spec_active = false;
        self.pending.clear();
        (self.spec_steps, self.spec_tokens) = (0, 0);
        if !self.g.states.is_empty() {
            let mut enc = self.device.create_command_encoder(&Default::default());
            for s in &self.g.states {
                enc.clear_buffer(s, 0, None);
            }
            self.queue.submit([enc.finish()]);
        }
    }

    fn resolve<'a>(&'a self, defs: &'a [OpDef], seq_len: u32, m: u32) -> impl Iterator<Item = Op<'a>> {
        defs.iter().map(move |d| Op {
            name: d.name,
            pipeline: &self.g.pipelines[d.pipe],
            bg: &d.bg,
            wg: d.dims.resolve(seq_len, m),
        })
    }

    /// The dispatch list for one decode step at the current position.
    fn decode_ops(&self) -> Vec<Op<'_>> {
        let seq_len = self.pos + 1;
        self.resolve(&self.g.decode, seq_len, 1).chain(self.resolve(&self.g.tail, seq_len, 1)).collect()
    }

    fn encode<'a>(pass: &mut wgpu::ComputePass<'_>, ops: impl IntoIterator<Item = Op<'a>>) {
        for op in ops {
            pass.set_pipeline(op.pipeline);
            pass.set_bind_group(0, op.bg, &[]);
            pass.dispatch_workgroups(op.wg.0, op.wg.1, 1);
        }
    }

    fn check_state(&self) -> Result<(), String> {
        if self.state_dirty {
            return Err("recurrent state is ahead of the consumed tokens; reset and prefill again".into());
        }
        Ok(())
    }

    fn begin_step(&self, token: Option<u32>) -> Result<(), String> {
        self.check_state()?;
        if self.pos >= self.max_ctx {
            return Err("context full".into());
        }
        if let Some(t) = token {
            self.queue.write_buffer(&self.g.tok, 0, bytemuck::bytes_of(&t));
        }
        let params = Params { pos: self.pos, seq_len: self.pos + 1, m: 1, slot: self.slot };
        self.queue.write_buffer(&self.g.params, 0, bytemuck::bytes_of(&params));
        Ok(())
    }

    fn take_slot(&mut self, enc: &mut wgpu::CommandEncoder) -> usize {
        let slot = self.ring_next;
        self.ring_next = (slot + 1) % READBACK_RING;
        enc.copy_buffer_to_buffer(&self.g.tok, 0, &self.ring[slot], 0, 4);
        slot
    }

    fn take_ring(&mut self) -> usize {
        let slot = self.ring_next;
        self.ring_next = (slot + 1) % READBACK_RING;
        slot
    }

    /// Run one token through the model at the current position. The greedy next token
    /// stays on the GPU (and is the input of the next step if `token` is None).
    pub fn submit_step(&mut self, token: Option<u32>, readback: bool) -> Result<(), String> {
        self.begin_step(token)?;
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            Self::encode(&mut pass, self.decode_ops());
        }
        let slot = readback.then(|| self.take_slot(&mut enc));
        let idx = self.queue.submit([enc.finish()]);
        if let Some(slot) = slot {
            self.inflight.push_back((slot, idx));
        }
        self.pos += 1;
        Ok(())
    }

    async fn read(&self, buf: &wgpu::Buffer, size: u64) -> Result<Vec<u8>, String> {
        self.read_after(buf, size, None).await
    }

    async fn read_after(&self, buf: &wgpu::Buffer, size: u64, sub: Option<wgpu::SubmissionIndex>) -> Result<Vec<u8>, String> {
        let slice = buf.slice(0..size);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::Wait { submission_index: sub, timeout: None })
            .map_err(|e| e.to_string())?;
        #[cfg(target_arch = "wasm32")]
        let _ = sub;
        rx.await.map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        let data = slice.get_mapped_range().map_err(|e| e.to_string())?.to_vec();
        buf.unmap();
        Ok(data)
    }

    async fn pop_token(&mut self) -> Result<u32, String> {
        let (slot, idx) = self.inflight.pop_front().ok_or("no step in flight")?;
        let b = self.read_after(&self.ring[slot], 4, Some(idx)).await?;
        self.last_token = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        Ok(self.last_token)
    }

    /// Queue one speculative step; `seq_bound` bounds the attention length (the exact position
    /// is only known on the GPU).
    fn submit_spec(&mut self, seq_bound: u32) {
        if self.spec_sync {
            let params = Params { pos: self.spec_pos, seq_len: self.spec_pos + SPEC_TOKENS, m: SPEC_TOKENS, slot: self.slot };
            self.queue.write_buffer(&self.g.params, 0, bytemuck::bytes_of(&params));
            self.spec_sync = false;
        }
        let slot = self.take_ring();
        let spec = self.g.spec.as_ref().expect("spec graph");
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            Self::encode(&mut pass, self.resolve(&spec.step, seq_bound, SPEC_TOKENS));
        }
        enc.copy_buffer_to_buffer(&spec.ctl, 0, &self.g.params, 0, 16);
        enc.copy_buffer_to_buffer(&spec.out, 0, &self.ring[slot], 0, 20);
        let idx = self.queue.submit([enc.finish()]);
        self.inflight.push_back((slot, idx));
    }

    async fn next_token_spec(&mut self, depth: usize) -> Result<u32, String> {
        if self.pending.is_empty() {
            while self.inflight.len() < depth {
                // each queued step advances by at most SPEC_TOKENS
                let start = self.spec_pos + SPEC_TOKENS * self.inflight.len() as u32;
                if start + SPEC_TOKENS > self.max_ctx {
                    break;
                }
                self.submit_spec(start + SPEC_TOKENS);
            }
            let (slot, idx) = self.inflight.pop_front().ok_or("context full")?;
            let b = self.read_after(&self.ring[slot], 20, Some(idx)).await?;
            let o: Vec<u32> = bytemuck::pod_collect_to_vec(&b);
            let n = o[0] as usize;
            self.pending.extend(&o[1..1 + n]);
            self.spec_pos = o[3];
            self.slot = o[4];
            self.spec_steps += 1;
            self.spec_tokens += n as u64;
        }
        let t = self.pending.pop_front().expect("pending token");
        self.last_token = t;
        self.pos = self.spec_pos - self.pending.len() as u32;
        Ok(t)
    }

    /// Feed `token` (None: the last generated token, still on the GPU), return the greedy next token.
    pub async fn step(&mut self, token: Option<u32>) -> Result<u32, String> {
        self.discard_ahead();
        self.submit_step(token, true)?;
        self.pop_token().await
    }

    /// Feed a prompt; returns the greedy token following it.
    pub async fn prefill(&mut self, tokens: &[u32]) -> Result<u32, String> {
        if tokens.is_empty() {
            return Err("empty prompt".into());
        }
        self.discard_ahead();
        let mtp = self.spec_enabled;
        if mtp && !self.spec_ready {
            let spec = self.g.spec.as_ref().expect("spec graph");
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                Self::encode(&mut pass, self.resolve(&spec.init, 0, 1));
            }
            self.queue.submit([enc.finish()]);
            self.spec_ready = true;
        }
        let n = tokens.chunks(PREFILL_CHUNK as usize).count();
        for (i, chunk) in tokens.chunks(PREFILL_CHUNK as usize).enumerate() {
            let next = tokens.get((i + 1) * PREFILL_CHUNK as usize).copied();
            self.submit_prefill(chunk, i + 1 == n, mtp.then_some(next))?;
        }
        let t = self.pop_token().await?;
        if mtp {
            self.spec_active = true;
            self.spec_sync = true;
            self.spec_pos = self.pos;
        }
        Ok(t)
    }

    fn begin_prefill(&self, tokens: &[u32]) -> Result<u32, String> {
        self.check_state()?;
        let m = tokens.len() as u32;
        if m == 0 || m > PREFILL_CHUNK || self.pos + m > self.max_ctx {
            return Err("bad prefill chunk or context full".into());
        }
        self.queue.write_buffer(&self.g.ptok, 0, bytemuck::cast_slice(tokens));
        let params = Params { pos: self.pos, seq_len: self.pos + m, m, slot: self.slot };
        self.queue.write_buffer(&self.g.params, 0, bytemuck::bytes_of(&params));
        Ok(m)
    }

    /// Run up to PREFILL_CHUNK tokens at once. With `last`, the hidden state of the final
    /// token goes through lm_head + argmax and the next token is queued for readback.
    /// With `mtp` (Some(token after the chunk, None for the last chunk)), the MTP layer runs over
    /// the chunk too, and after the last chunk the first draft is set up for speculative steps.
    fn submit_prefill(&mut self, tokens: &[u32], last: bool, mtp: Option<Option<u32>>) -> Result<(), String> {
        let m = self.begin_prefill(tokens)?;
        let seq_len = self.pos + m;
        if let (Some(spec), Some(next)) = (&self.g.spec, mtp) {
            // MTP row i takes the token after token i; for the last prompt token that is the
            // first generated one, copied in on the GPU below
            let mut shifted = tokens[1..].to_vec();
            shifted.push(next.unwrap_or(0));
            self.queue.write_buffer(&spec.ntok, 0, bytemuck::cast_slice(&shifted));
        }
        let row = (self.manifest.config.hidden_size * 4) as u64;
        let last_row = (m as u64 - 1) * row;
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            Self::encode(&mut pass, self.resolve(&self.g.prefill, seq_len, m));
        }
        let mut slot = None;
        if last {
            enc.copy_buffer_to_buffer(&self.g.px, last_row, &self.g.x, 0, row);
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                Self::encode(&mut pass, self.resolve(&self.g.tail, seq_len, 1));
            }
            slot = Some(self.take_slot(&mut enc));
        }
        if let (Some(spec), Some(_)) = (&self.g.spec, mtp) {
            if last {
                enc.copy_buffer_to_buffer(&self.g.tok, 0, &spec.ntok, (m as u64 - 1) * 4, 4);
            }
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                Self::encode(&mut pass, self.resolve(&spec.prefill, seq_len, m));
            }
            if last {
                // first draft from the MTP output at the last prompt position
                enc.copy_buffer_to_buffer(&self.g.px, last_row, &self.g.x, 0, row);
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    Self::encode(&mut pass, self.resolve(&spec.tail, seq_len, 1));
                }
                enc.copy_buffer_to_buffer(&self.g.tok, 0, &spec.tok, 0, 4);
                enc.copy_buffer_to_buffer(&spec.draft, 0, &spec.tok, 4, 4);
            }
        }
        let idx = self.queue.submit([enc.finish()]);
        if let Some(s) = slot {
            self.inflight.push_back((s, idx));
        }
        self.pos += m;
        Ok(())
    }

    /// Token-by-token prefill through the decode path (reference for testing the batched one).
    pub async fn prefill_sequential(&mut self, tokens: &[u32]) -> Result<u32, String> {
        let (last, rest) = tokens.split_last().ok_or("empty prompt")?;
        self.discard_ahead();
        for &t in rest {
            self.submit_step(Some(t), false)?;
        }
        self.step(Some(*last)).await
    }

    /// Pipelined greedy decoding: keeps up to `depth` steps queued on the GPU so the token
    /// readback latency overlaps with compute. Returns the next generated token; steps
    /// submitted beyond what the caller consumes are dropped by `discard_ahead`.
    pub async fn next_token(&mut self, depth: usize) -> Result<u32, String> {
        let depth = depth.clamp(1, READBACK_RING - 1);
        if self.spec_active {
            return self.next_token_spec(depth).await;
        }
        while self.inflight.len() < depth && self.pos < self.max_ctx {
            self.submit_step(None, true)?;
        }
        self.pop_token().await
    }

    /// Forget queued steps whose tokens were not consumed. Attention KV entries are simply
    /// overwritten later; recurrent state cannot be rolled back, so for such models the
    /// sequence must be reset before it is continued.
    pub fn discard_ahead(&mut self) {
        if self.spec_active {
            // back to plain steps, which continue from the last token handed out
            self.spec_active = false;
            if !self.inflight.is_empty() || !self.pending.is_empty() {
                self.inflight.clear();
                self.pending.clear();
                self.state_dirty = true;
            }
            self.queue.write_buffer(&self.g.tok, 0, bytemuck::bytes_of(&self.last_token));
            return;
        }
        if !self.inflight.is_empty() {
            self.pos -= self.inflight.len() as u32;
            self.inflight.clear();
            // the GPU token slot already holds a later token; restore the last one handed out
            self.queue.write_buffer(&self.g.tok, 0, bytemuck::bytes_of(&self.last_token));
            self.state_dirty |= !self.g.states.is_empty();
        }
    }

    /// Run one decode step with every dispatch in its own timestamped pass (needs
    /// `Features::TIMESTAMP_QUERY`). Returns (kernel, total microseconds, dispatch count), slowest first.
    pub async fn profile_step(&mut self, token: Option<u32>) -> Result<Vec<(&'static str, f64, u32)>, String> {
        self.discard_ahead();
        self.begin_step(token)?;
        let agg = self.profile_ops(self.decode_ops()).await?;
        self.pos += 1;
        Ok(agg)
    }

    /// Like `profile_step` for one speculative step (right after a prefill, nothing queued);
    /// its tokens are handed out by the following `next_token` calls.
    pub async fn profile_spec(&mut self) -> Result<Vec<(&'static str, f64, u32)>, String> {
        if !self.spec_active || !self.inflight.is_empty() || !self.pending.is_empty() {
            return Err("profile_spec needs a speculative sequence with nothing queued".into());
        }
        let params = Params { pos: self.spec_pos, seq_len: self.spec_pos + SPEC_TOKENS, m: SPEC_TOKENS, slot: self.slot };
        self.queue.write_buffer(&self.g.params, 0, bytemuck::bytes_of(&params));
        self.spec_sync = false;
        let spec = self.g.spec.as_ref().expect("spec graph");
        let agg = self.profile_ops(self.resolve(&spec.step, self.spec_pos + SPEC_TOKENS, SPEC_TOKENS).collect()).await?;
        let slot = self.take_ring();
        let spec = self.g.spec.as_ref().expect("spec graph");
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&spec.ctl, 0, &self.g.params, 0, 16);
        enc.copy_buffer_to_buffer(&spec.out, 0, &self.ring[slot], 0, 20);
        let idx = self.queue.submit([enc.finish()]);
        self.inflight.push_back((slot, idx));
        Ok(agg)
    }

    /// Like `profile_step` for one batched prefill chunk (the model state is left as after the chunk).
    pub async fn profile_prefill(&mut self, tokens: &[u32]) -> Result<Vec<(&'static str, f64, u32)>, String> {
        self.discard_ahead();
        let m = self.begin_prefill(tokens)?;
        let ops = self.resolve(&self.g.prefill, self.pos + m, m).collect();
        let agg = self.profile_ops(ops).await?;
        self.pos += m;
        Ok(agg)
    }

    async fn profile_ops(&self, ops: Vec<Op<'_>>) -> Result<Vec<(&'static str, f64, u32)>, String> {
        let n = ops.len() as u32 * 2;
        let qs = self.device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: n });
        let size = n as u64 * 8;
        let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        for (i, op) in ops.iter().enumerate() {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &qs,
                    beginning_of_pass_write_index: Some(2 * i as u32),
                    end_of_pass_write_index: Some(2 * i as u32 + 1),
                }),
            });
            pass.set_pipeline(op.pipeline);
            pass.set_bind_group(0, op.bg, &[]);
            pass.dispatch_workgroups(op.wg.0, op.wg.1, 1);
        }
        enc.resolve_query_set(&qs, 0..n, &resolve, 0);
        enc.copy_buffer_to_buffer(&resolve, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        let names: Vec<&'static str> = ops.iter().map(|o| o.name).collect();

        let ts: Vec<u64> = bytemuck::pod_collect_to_vec(&self.read(&staging, size).await?);
        let period = self.queue.get_timestamp_period() as f64;
        let mut agg: Vec<(&'static str, f64, u32)> = Vec::new();
        for (i, &name) in names.iter().enumerate() {
            let us = ts[2 * i + 1].saturating_sub(ts[2 * i]) as f64 * period / 1e3;
            match agg.iter_mut().find(|a| a.0 == name) {
                Some(a) => {
                    a.1 += us;
                    a.2 += 1;
                }
                None => agg.push((name, us, 1)),
            }
        }
        agg.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(agg)
    }

    /// Logits of the most recent plain step or prefill (debug / verification).
    pub async fn read_logits(&self) -> Result<Vec<f32>, String> {
        let size = self.manifest.config.vocab_size as u64 * 4;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.g.logits, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        let b = self.read(&staging, size).await?;
        Ok(bytemuck::cast_slice(&b).to_vec())
    }
}
