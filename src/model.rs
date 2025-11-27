//! Core model structures and weights.

use crate::config::Config;
use crate::error::{Error, Result};
use crate::operations::{matmul, rmsnorm, softmax, swish, add_scaled, apply_rotary_emb, precompute_freqs_cis};
use memmap2::Mmap;
use std::fs::File;
use std::mem;
use std::path::Path;

/// Transformer weights containing all model parameters.
///
/// This struct holds all the learned parameters of the Llama 2 model,
/// organized as pointers into memory-mapped weight data.
#[derive(Debug)]
pub struct TransformerWeights {
    /// Memory-mapped weight data
    data: Mmap,
    /// Pointer to token embedding table (vocab_size, dim)
    pub token_embedding_table: *const f32,
    /// RMSNorm weights for attention (layer, dim)
    pub rms_att_weight: *const f32,
    /// RMSNorm weights for FFN (layer, dim)
    pub rms_ffn_weight: *const f32,
    /// Query weights (layer, dim, dim)
    pub wq: *const f32,
    /// Key weights (layer, dim, kv_dim)
    pub wk: *const f32,
    /// Value weights (layer, dim, kv_dim)
    pub wv: *const f32,
    /// Output weights (layer, dim, dim)
    pub wo: *const f32,
    /// FFN gate weights (layer, dim, hidden_dim)
    pub w1: *const f32,
    /// FFN down projection weights (layer, hidden_dim, dim)
    pub w2: *const f32,
    /// FFN up projection weights (layer, dim, hidden_dim)
    pub w3: *const f32,
    /// Final RMSNorm weights (dim,)
    pub rms_final_weight: *const f32,
    /// Classifier weights (optional, shared with embedding by default)
    pub wcls: *const f32,
    /// Precomputed RoPE frequencies
    pub freqs_cis: Vec<(f32, f32)>,
}

impl TransformerWeights {
    /// Load weights from a checkpoint file.
    ///
    /// This function memory-maps the checkpoint file and sets up all the weight pointers.
    ///
    /// # Arguments
    /// * `checkpoint_path` - Path to the model checkpoint file
    /// * `config` - Model configuration
    ///
    /// # Returns
    /// Loaded transformer weights
    pub fn from_checkpoint<P: AsRef<Path>>(checkpoint_path: P, config: &Config) -> Result<Self> {
        let file = File::open(checkpoint_path)?;
        let data = unsafe { Mmap::map(&file)? };

        if data.len() < mem::size_of::<Config>() {
            return Err(Error::ModelLoad("Checkpoint file too small".to_string()));
        }

        // Read config from the beginning of the file
        let config_ptr = data.as_ptr() as *const Config;
        let file_config = unsafe { &*config_ptr };

        // Validate config matches what we expect
        if file_config.dim != config.dim ||
           file_config.hidden_dim != config.hidden_dim ||
           file_config.n_layers != config.n_layers ||
           file_config.n_heads != config.n_heads ||
           file_config.n_kv_heads != config.n_kv_heads ||
           file_config.vocab_size != config.vocab_size ||
           file_config.seq_len != config.seq_len {
            return Err(Error::ModelLoad("Model config mismatch".to_string()));
        }

        // Check if vocab_size is negative (indicates unshared weights)
        let mut vocab_size = file_config.vocab_size as isize;
        let shared_weights = vocab_size > 0;
        vocab_size = vocab_size.abs() as isize;

        // Set up weight pointers
        let mut ptr = data.as_ptr() as *const f32;
        ptr = unsafe { ptr.add(mem::size_of::<Config>() / mem::size_of::<f32>()) };

        let token_embedding_table = ptr;
        ptr = unsafe { ptr.add((vocab_size as usize) * config.dim) };

        let rms_att_weight = ptr;
        ptr = unsafe { ptr.add(config.n_layers * config.dim) };

        let wq = ptr;
        ptr = unsafe { ptr.add(config.n_layers * config.dim * config.dim) };

        let wk = ptr;
        ptr = unsafe { ptr.add(config.n_layers * config.dim * config.kv_dim()) };

        let wv = ptr;
        ptr = unsafe { ptr.add(config.n_layers * config.dim * config.kv_dim()) };

        let wo = ptr;
        ptr = unsafe { ptr.add(config.n_layers * config.dim * config.dim) };

        let rms_ffn_weight = ptr;
        ptr = unsafe { ptr.add(config.n_layers * config.dim) };

        let w1 = ptr;
        ptr = unsafe { ptr.add(config.n_layers * config.dim * config.hidden_dim) };

        let w2 = ptr;
        ptr = unsafe { ptr.add(config.n_layers * config.hidden_dim * config.dim) };

        let w3 = ptr;
        ptr = unsafe { ptr.add(config.n_layers * config.dim * config.hidden_dim) };

        let rms_final_weight = ptr;
        ptr = unsafe { ptr.add(config.dim) };

        // Skip RoPE frequencies (they were removed in newer checkpoints)
        ptr = unsafe { ptr.add(config.seq_len * config.head_size() / 2) };
        ptr = unsafe { ptr.add(config.seq_len * config.head_size() / 2) };

        let wcls = if shared_weights {
            token_embedding_table
        } else {
            ptr
        };

        // Precompute RoPE frequencies
        let freqs_cis = precompute_freqs_cis(config.dim, config.seq_len, 10000.0);

        Ok(Self {
            data,
            token_embedding_table,
            rms_att_weight,
            rms_ffn_weight,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            rms_final_weight,
            wcls,
            freqs_cis,
        })
    }

    /// Get token embedding for a given token ID.
    ///
    /// # Arguments
    /// * `token` - Token ID
    /// * `config` - Model configuration
    ///
    /// # Returns
    /// Slice containing the token embedding
    pub fn get_token_embedding(&self, token: usize, config: &Config) -> &[f32] {
        debug_assert!(token < config.vocab_size);
        unsafe {
            let ptr = self.token_embedding_table.add(token * config.dim);
            std::slice::from_raw_parts(ptr, config.dim)
        }
    }

    /// Get RMSNorm weights for attention at a specific layer.
    ///
    /// # Arguments
    /// * `layer` - Layer index
    /// * `config` - Model configuration
    ///
    /// # Returns
    /// Slice containing the RMSNorm weights
    pub fn get_rms_att_weight(&self, layer: usize, config: &Config) -> &[f32] {
        debug_assert!(layer < config.n_layers);
        unsafe {
            let ptr = self.rms_att_weight.add(layer * config.dim);
            std::slice::from_raw_parts(ptr, config.dim)
        }
    }

    /// Get query weights for a specific layer.
    ///
    /// # Arguments
    /// * `layer` - Layer index
    /// * `config` - Model configuration
    ///
    /// # Returns
    /// Slice containing the query weights
    pub fn get_wq(&self, layer: usize, config: &Config) -> &[f32] {
        debug_assert!(layer < config.n_layers);
        unsafe {
            let ptr = self.wq.add(layer * config.dim * config.dim);
            std::slice::from_raw_parts(ptr, config.dim * config.dim)
        }
    }

    /// Get key weights for a specific layer.
    pub fn get_wk(&self, layer: usize, config: &Config) -> &[f32] {
        debug_assert!(layer < config.n_layers);
        unsafe {
            let ptr = self.wk.add(layer * config.dim * config.kv_dim());
            std::slice::from_raw_parts(ptr, config.dim * config.kv_dim())
        }
    }

    /// Get value weights for a specific layer.
    pub fn get_wv(&self, layer: usize, config: &Config) -> &[f32] {
        debug_assert!(layer < config.n_layers);
        unsafe {
            let ptr = self.wv.add(layer * config.dim * config.kv_dim());
            std::slice::from_raw_parts(ptr, config.dim * config.kv_dim())
        }
    }

    /// Get output weights for a specific layer.
    pub fn get_wo(&self, layer: usize, config: &Config) -> &[f32] {
        debug_assert!(layer < config.n_layers);
        unsafe {
            let ptr = self.wo.add(layer * config.dim * config.dim);
            std::slice::from_raw_parts(ptr, config.dim * config.dim)
        }
    }

    /// Get RMSNorm weights for FFN at a specific layer.
    pub fn get_rms_ffn_weight(&self, layer: usize, config: &Config) -> &[f32] {
        debug_assert!(layer < config.n_layers);
        unsafe {
            let ptr = self.rms_ffn_weight.add(layer * config.dim);
            std::slice::from_raw_parts(ptr, config.dim)
        }
    }

    /// Get FFN gate weights for a specific layer.
    pub fn get_w1(&self, layer: usize, config: &Config) -> &[f32] {
        debug_assert!(layer < config.n_layers);
        unsafe {
            let ptr = self.w1.add(layer * config.dim * config.hidden_dim);
            std::slice::from_raw_parts(ptr, config.dim * config.hidden_dim)
        }
    }

    /// Get FFN down projection weights for a specific layer.
    pub fn get_w2(&self, layer: usize, config: &Config) -> &[f32] {
        debug_assert!(layer < config.n_layers);
        unsafe {
            let ptr = self.w2.add(layer * config.hidden_dim * config.dim);
            std::slice::from_raw_parts(ptr, config.hidden_dim * config.dim)
        }
    }

    /// Get FFN up projection weights for a specific layer.
    pub fn get_w3(&self, layer: usize, config: &Config) -> &[f32] {
        debug_assert!(layer < config.n_layers);
        unsafe {
            let ptr = self.w3.add(layer * config.dim * config.hidden_dim);
            std::slice::from_raw_parts(ptr, config.dim * config.hidden_dim)
        }
    }

    /// Get final RMSNorm weights.
    pub fn get_rms_final_weight(&self, config: &Config) -> &[f32] {
        unsafe {
            std::slice::from_raw_parts(self.rms_final_weight, config.dim)
        }
    }

    /// Get classifier weights.
    pub fn get_wcls(&self, config: &Config) -> &[f32] {
        unsafe {
            std::slice::from_raw_parts(self.wcls, config.dim * config.vocab_size)
        }
    }

    /// Get the configuration from the loaded weights.
    pub fn get_config(&self) -> Config {
        // We need to reconstruct the config from the file
        // This is a bit hacky since we don't store it directly
        // In a real implementation, we'd store the config alongside the weights
        Config {
            dim: 0, // These would need to be read from the file
            hidden_dim: 0,
            n_layers: 0,
            n_heads: 0,
            n_kv_heads: 0,
            vocab_size: 0,
            seq_len: 0,
        }
    }
}

/// Runtime state for transformer inference.
///
/// Contains all the buffers needed for forward pass computation,
/// including activations, KV cache, and temporary storage.
#[derive(Debug)]
pub struct RunState {
    /// Current activation at timestep (dim,)
    pub x: Vec<f32>,
    /// Same as x but inside residual branch (dim,)
    pub xb: Vec<f32>,
    /// Additional buffer for convenience (dim,)
    pub xb2: Vec<f32>,
    /// Buffer for hidden dimension in FFN (hidden_dim,)
    pub hb: Vec<f32>,
    /// Buffer for hidden dimension in FFN (hidden_dim,)
    pub hb2: Vec<f32>,
    /// Query vector (dim,)
    pub q: Vec<f32>,
    /// Key vector (dim,)
    pub k: Vec<f32>,
    /// Value vector (dim,)
    pub v: Vec<f32>,
    /// Buffer for attention scores (n_heads, seq_len)
    pub att: Vec<f32>,
    /// Output logits (vocab_size,)
    pub logits: Vec<f32>,
    /// Key cache (layer, seq_len, kv_dim)
    pub key_cache: Vec<f32>,
    /// Value cache (layer, seq_len, kv_dim)
    pub value_cache: Vec<f32>,
}

impl RunState {
    /// Create a new RunState with buffers allocated for the given config.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    ///
    /// # Returns
    /// New RunState instance
    pub fn new(config: &Config) -> Self {
        let kv_dim = config.kv_dim();

        Self {
            x: vec![0.0; config.dim],
            xb: vec![0.0; config.dim],
            xb2: vec![0.0; config.dim],
            hb: vec![0.0; config.hidden_dim],
            hb2: vec![0.0; config.hidden_dim],
            q: vec![0.0; config.dim],
            att: vec![0.0; config.n_heads * config.seq_len],
            logits: vec![0.0; config.vocab_size],
            key_cache: vec![0.0; config.n_layers * config.seq_len * kv_dim],
            value_cache: vec![0.0; config.n_layers * config.seq_len * kv_dim],
        }
    }
}

/// Complete transformer model with weights and runtime state.
#[derive(Debug)]
pub struct Transformer {
    /// Model configuration
    pub config: Config,
    /// Model weights (memory-mapped)
    pub weights: TransformerWeights,
    /// Runtime state for inference
    pub state: RunState,
}

impl Transformer {
    /// Load a transformer from a checkpoint file.
    ///
    /// This is the main constructor for loading a trained model.
    ///
    /// # Arguments
    /// * `checkpoint_path` - Path to the model checkpoint file
    ///
    /// # Returns
    /// Loaded transformer ready for inference
    pub fn from_checkpoint<P: AsRef<Path>>(checkpoint_path: P) -> Result<Self> {
        // First, read just the config from the file to validate
        let file = File::open(&checkpoint_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < mem::size_of::<Config>() {
            return Err(Error::ModelLoad("Checkpoint file too small for config".to_string()));
        }

        let config_ptr = mmap.as_ptr() as *const Config;
        let file_config = unsafe { &*config_ptr };

        // Create a temporary config for validation
        let config = Config {
            dim: file_config.dim,
            hidden_dim: file_config.hidden_dim,
            n_layers: file_config.n_layers,
            n_heads: file_config.n_heads,
            n_kv_heads: file_config.n_kv_heads,
            vocab_size: file_config.vocab_size.abs() as usize, // Handle negative vocab_size
            seq_len: file_config.seq_len,
        };

        config.validate()?;

        // Now load the full weights
        let weights = TransformerWeights::from_checkpoint(checkpoint_path, &config)?;
        let state = RunState::new(&config);

        Ok(Self {
            config,
            weights,
            state,
        })
    }

    /// Forward pass through the transformer.
    ///
    /// Computes the next token logits given the current token and position.
    ///
    /// # Arguments
    /// * `token` - Current token ID
    /// * `pos` - Position in sequence (0-based)
    ///
    /// # Returns
    /// Slice containing the output logits
    pub fn forward(&mut self, token: usize, pos: usize) -> &[f32] {
        debug_assert!(token < self.config.vocab_size);
        debug_assert!(pos < self.config.seq_len);

        let config = &self.config;
        let weights = &self.weights;
        let state = &mut self.state;

        // Copy token embedding into x
        let content_row = weights.get_token_embedding(token, config);
        state.x.copy_from_slice(content_row);

        // Forward through all layers
        for l in 0..config.n_layers {
            // Attention RMSNorm
            let rms_att_weight = weights.get_rms_att_weight(l, config);
            rmsnorm(&mut state.xb, &state.x, rms_att_weight, config.dim);

            // Key and value point to KV cache
            let loff = l * config.seq_len * config.kv_dim();
            let k_cache = &mut state.key_cache[loff + pos * config.kv_dim()..loff + (pos + 1) * config.kv_dim()];
            let v_cache = &mut state.value_cache[loff + pos * config.seq_len * config.kv_dim()..loff + (pos + 1) * config.kv_dim()];

            // QKV matmuls
            let wq = weights.get_wq(l, config);
            matmul(&mut state.q, &state.xb, wq, config.dim, config.dim);

            let wk = weights.get_wk(l, config);
            matmul(k_cache, &state.xb, wk, config.dim, config.kv_dim());

            let wv = weights.get_wv(l, config);
            matmul(v_cache, &state.xb, wv, config.dim, config.kv_dim());

            // RoPE rotary positional encoding
            apply_rotary_emb(&mut state.q, &weights.freqs_cis, pos, config.dim);
            apply_rotary_emb(k_cache, &weights.freqs_cis, pos, config.kv_dim());

            // Multi-head attention
            self.multi_head_attention(l, pos);

            // Final matmul to get output of attention
            let wo = weights.get_wo(l, config);
            matmul(&mut state.xb2, &state.xb, wo, config.dim, config.dim);

            // Residual connection
            add_scaled(&mut state.x, &state.xb2, 1.0);

            // FFN RMSNorm
            let rms_ffn_weight = weights.get_rms_ffn_weight(l, config);
            rmsnorm(&mut state.xb, &state.x, rms_ffn_weight, config.dim);

            // FFN: SwiGLU activation
            self.feed_forward(l);

            // Residual connection
            add_scaled(&mut state.x, &state.xb, 1.0);
        }

        // Final RMSNorm
        let rms_final_weight = weights.get_rms_final_weight(config);
        rmsnorm(&mut state.x, &state.x, rms_final_weight, config.dim);

        // Classifier
        let wcls = weights.get_wcls(config);
        matmul(&mut state.logits, &state.x, wcls, config.dim, config.vocab_size);

        &state.logits
    }

    /// Multi-head attention computation.
    fn multi_head_attention(&mut self, layer: usize, pos: usize) {
        let config = &self.config;
        let state = &mut self.state;

        let loff = layer * config.seq_len * config.kv_dim();

        // Iterate over all heads in parallel
        for h in 0..config.n_heads {
            let q_offset = h * config.head_size();
            let q = &state.q[q_offset..q_offset + config.head_size()];

            let att_offset = h * config.seq_len;
            let att = &mut state.att[att_offset..att_offset + config.seq_len];

            // Attention scores for this head
            for t in 0..=pos {
                let k_offset = loff + t * config.kv_dim() + (h / config.kv_mul()) * config.head_size();
                let k = &state.key_cache[k_offset..k_offset + config.head_size()];

                let mut score = 0.0;
                for i in 0..config.head_size() {
                    score += q[i] * k[i];
                }
                score /= (config.head_size() as f32).sqrt();
                att[t] = score;
            }

            // Softmax the scores
            softmax(&mut att[..=pos], pos + 1);

            // Weighted sum of values
            let xb_offset = h * config.head_size();
            let xb = &mut state.xb[xb_offset..xb_offset + config.head_size()];
            xb.fill(0.0);

            for t in 0..=pos {
                let v_offset = loff + t * config.kv_dim() + (h / config.kv_mul()) * config.head_size();
                let v = &state.value_cache[v_offset..v_offset + config.head_size()];
                let a = att[t];

                for i in 0..config.head_size() {
                    xb[i] += a * v[i];
                }
            }
        }
    }

    /// Feed-forward network computation.
    fn feed_forward(&mut self, layer: usize) {
        let config = &self.config;
        let state = &mut self.state;
        let weights = &self.weights;

        // Get FFN weights
        let w1 = weights.get_w1(layer, config);
        let w3 = weights.get_w3(layer, config);
        let w2 = weights.get_w2(layer, config);

        // w1(x) and w3(x)
        matmul(&mut state.hb, &state.xb, w1, config.dim, config.hidden_dim);
        matmul(&mut state.hb2, &state.xb, w3, config.dim, config.hidden_dim);

        // SwiGLU: hb * silu(hb2)
        for i in 0..config.hidden_dim {
            let val = state.hb[i] * swish(state.hb2[i]);
            state.hb[i] = val;
        }

        // Final projection
        matmul(&mut state.xb, &state.hb, w2, config.hidden_dim, config.dim);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // Note: These tests would require actual model files to run
    // For now, we just test the structure

    #[test]
    fn test_config_sizes() {
        let config = Config {
            dim: 768,
            hidden_dim: 3072,
            n_layers: 12,
            n_heads: 12,
            n_kv_heads: 12,
            vocab_size: 32000,
            seq_len: 1024,
        };

        assert_eq!(config.kv_dim(), 768);
        assert_eq!(config.head_size(), 64);
        assert_eq!(config.kv_mul(), 1);
    }

    #[test]
    fn test_runstate_allocation() {
        let config = Config {
            dim: 64,
            hidden_dim: 256,
            n_layers: 2,
            n_heads: 8,
            n_kv_heads: 8,
            vocab_size: 1000,
            seq_len: 128,
        };

        let state = RunState::new(&config);

        assert_eq!(state.x.len(), 64);
        assert_eq!(state.xb.len(), 64);
        assert_eq!(state.hb.len(), 256);
        assert_eq!(state.q.len(), 64);
        assert_eq!(state.att.len(), 8 * 128);
        assert_eq!(state.logits.len(), 1000);
        assert_eq!(state.key_cache.len(), 2 * 128 * 64);
        assert_eq!(state.value_cache.len(), 2 * 128 * 64);
    }
}
