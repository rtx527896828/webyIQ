//! Browser bindings: thin wasm-bindgen wrapper over the engine.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use webyiq_engine::{Engine, Manifest, ShaderCaps};

fn js_err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

#[wasm_bindgen]
pub struct WebEngine {
    // shared so async methods can hold it across awaits without borrowing `self`
    engine: Rc<RefCell<Engine>>,
    info: String,
}

#[wasm_bindgen]
impl WebEngine {
    /// Create the engine on the browser's WebGPU adapter (weights are uploaded afterwards).
    /// `packed_int8`: `navigator.gpu.wgslLanguageFeatures.has("packed_4x8_integer_dot_product")`.
    pub async fn create(
        manifest_json: String,
        max_ctx: u32,
        high_performance: bool,
        packed_int8: bool,
        use_subgroups: bool,
    ) -> Result<WebEngine, JsError> {
        let manifest = Manifest::parse(&manifest_json).map_err(js_err)?;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: if high_performance {
                    wgpu::PowerPreference::HighPerformance
                } else {
                    wgpu::PowerPreference::LowPower
                },
                ..Default::default()
            })
            .await
            .map_err(js_err)?;
        let ai = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: adapter.features() & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SUBGROUP),
                required_limits: Engine::required_limits(&adapter.limits()),
                ..Default::default()
            })
            .await
            .map_err(js_err)?;
        let info = format!("{} {} ({})", ai.vendor, ai.name, ai.driver_info).trim().to_string();
        let caps = ShaderCaps { packed_int8, subgroups: use_subgroups && device.features().contains(wgpu::Features::SUBGROUP) };
        let info = format!("{info} {caps:?}");
        let engine = Engine::new(device, queue, manifest, max_ctx, caps).map_err(js_err)?;
        Ok(WebEngine { engine: Rc::new(RefCell::new(engine)), info })
    }

    #[wasm_bindgen(getter)]
    pub fn adapter_info(&self) -> String {
        self.info.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn pos(&self) -> u32 {
        self.engine.borrow().pos()
    }

    #[wasm_bindgen(getter)]
    pub fn max_ctx(&self) -> u32 {
        self.engine.borrow().max_ctx
    }

    pub fn upload_shard(&self, idx: usize, data: &[u8]) {
        self.engine.borrow().upload_shard(idx, data);
    }

    pub fn reset(&self) {
        self.engine.borrow_mut().reset();
    }

    /// Whether the model has an MTP draft head (speculative decoding in `next_token`).
    #[wasm_bindgen(getter)]
    pub fn has_speculative(&self) -> bool {
        self.engine.borrow().has_speculative()
    }

    /// Enable / disable speculative decoding (takes effect at the next prefill; output is identical).
    pub fn set_speculative(&self, on: bool) {
        self.engine.borrow_mut().set_speculative(on);
    }

    /// [speculative steps, tokens they produced] since the last reset.
    pub fn spec_stats(&self) -> Vec<f64> {
        let (s, t) = self.engine.borrow().spec_stats();
        vec![s as f64, t as f64]
    }

    /// Feed prompt tokens, resolve to the greedy next token.
    pub fn prefill(&self, tokens: Vec<u32>) -> js_sys::Promise {
        let engine = self.engine.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let mut e = engine.borrow_mut();
            e.prefill(&tokens).await.map(JsValue::from).map_err(|s| JsError::new(&s).into())
        })
    }

    /// Feed the previously generated token (kept on the GPU), resolve to the next one.
    pub fn step(&self) -> js_sys::Promise {
        let engine = self.engine.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let mut e = engine.borrow_mut();
            e.step(None).await.map(JsValue::from).map_err(|s| JsError::new(&s).into())
        })
    }

    /// Pipelined decoding: keeps `depth` steps queued on the GPU; resolves to the next token.
    pub fn next_token(&self, depth: usize) -> js_sys::Promise {
        let engine = self.engine.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let mut e = engine.borrow_mut();
            e.next_token(depth).await.map(JsValue::from).map_err(|s| JsError::new(&s).into())
        })
    }

    /// Roll back steps queued by `next_token` that were not consumed (e.g. after EOS).
    pub fn discard_ahead(&self) {
        self.engine.borrow_mut().discard_ahead();
    }

    /// Per-kernel GPU timings of one prefill chunk: JSON [[name, us, count], ...].
    pub fn profile_prefill(&self, tokens: Vec<u32>) -> js_sys::Promise {
        let engine = self.engine.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let mut e = engine.borrow_mut();
            let rows = e.profile_prefill(&tokens).await.map_err(|s| JsValue::from(JsError::new(&s)))?;
            let json: Vec<String> = rows.iter().map(|(n, us, c)| format!("[\"{n}\",{us:.1},{c}]")).collect();
            Ok(JsValue::from(format!("[{}]", json.join(","))))
        })
    }

    /// Per-kernel GPU timings of one speculative step (right after a prefill): JSON [[name, us, count], ...].
    pub fn profile_spec(&self) -> js_sys::Promise {
        let engine = self.engine.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let mut e = engine.borrow_mut();
            let rows = e.profile_spec().await.map_err(|s| JsValue::from(JsError::new(&s)))?;
            let json: Vec<String> = rows.iter().map(|(n, us, c)| format!("[\"{n}\",{us:.1},{c}]")).collect();
            Ok(JsValue::from(format!("[{}]", json.join(","))))
        })
    }

    /// Per-kernel GPU timings of one decode step: JSON [[name, us, count], ...].
    pub fn profile_step(&self) -> js_sys::Promise {
        let engine = self.engine.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            let mut e = engine.borrow_mut();
            let rows = e.profile_step(None).await.map_err(|s| JsValue::from(JsError::new(&s)))?;
            let json: Vec<String> = rows.iter().map(|(n, us, c)| format!("[\"{n}\",{us:.1},{c}]")).collect();
            Ok(JsValue::from(format!("[{}]", json.join(","))))
        })
    }
}
