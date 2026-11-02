use std::collections::HashMap;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub arch: String,
    pub config: Config,
    pub quant: Quant,
    pub shards: Vec<String>,
    pub tensors: HashMap<String, TensorInfo>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub vocab_size: u32,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: u32,
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// dims of each head that get rotary embedding (default: all)
    #[serde(default)]
    pub rotary_dim: Option<u32>,
    /// qwen3_5: "linear" (Gated DeltaNet) or "full" per layer
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub linear_num_heads: u32,
    #[serde(default)]
    pub linear_key_head_dim: u32,
    #[serde(default)]
    pub linear_value_head_dim: u32,
    #[serde(default)]
    pub linear_conv_kernel: u32,
    /// qwen3_5: multi-token-prediction layers (draft head for speculative decoding)
    #[serde(default)]
    pub mtp_layers: u32,
}

impl Config {
    pub fn rotary_dim(&self) -> u32 {
        self.rotary_dim.unwrap_or(self.head_dim)
    }

    pub fn is_linear(&self, layer: usize) -> bool {
        self.layer_types.get(layer).is_some_and(|t| t == "linear")
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quant {
    pub bits: u32,
    pub group: u32,
    pub scheme: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TensorInfo {
    pub shard: usize,
    pub offset: u64,
    pub bytes: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
}

impl Manifest {
    pub fn parse(json: &str) -> Result<Self, String> {
        let m: Manifest = serde_json::from_str(json).map_err(|e| e.to_string())?;
        if m.format != "webyiq-v2" || !matches!(m.arch.as_str(), "llama" | "qwen3_5") {
            return Err(format!("unsupported model: {} / {}", m.format, m.arch));
        }
        if m.quant.bits != 4 || m.quant.group != 32 {
            return Err(format!("unsupported quant: {:?}", m.quant));
        }
        Ok(m)
    }

    pub fn shard_bytes(&self) -> Vec<u64> {
        let mut sizes = vec![0u64; self.shards.len()];
        for t in self.tensors.values() {
            sizes[t.shard] = sizes[t.shard].max(t.offset + t.bytes);
        }
        sizes
    }
}
