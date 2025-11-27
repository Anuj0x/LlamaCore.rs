//! Complete transformer inference implementation.

use crate::config::Config;
use crate::error::{Error, Result};
use crate::model::TransformerWeights;
use crate::operations::{matmul, rmsnorm, softmax, swish, add_scaled, apply_rotary_emb, precompute_freqs_cis};
use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;
use std::path::Path;

/// Runtime state for transformer inference.
///
/// Contains all the buffers needed for forward passes through the model.
#[derive(Debug)]
pub struct RunState {
    /// Activation at current timestep (dim,)
    pub x: Vec<f32>,
    /// Activation in residual branch (dim,)
    pub xb: Vec<f32>,
    /// Additional buffer for convenience (dim,)
    pub xb2: Vec<f32>,
    /// Buffer for hidden dimension in FFN (hidden_dim,)
    pub hb: Vec<f32>,
    /// Additional buffer for hidden dimension in FFN (hidden_dim,)
    pub hb2: Vec<f32>,
    /// Query vector (dim,)
    pub q: Vec<f32>,
    /// Attention buffer (n_heads, seq_len)
    pub att: Vec<f32>,
    /// Output logits (vocab_size,)
    pub logits: Vec<f32>,
    /// Key cache (layer, seq_len, kv_dim)
    pub key_cache: Vec<f32>,
    /// Value cache (layer, seq_len, kv_dim)
    pub value_cache: Vec<f32>,
    /// Precomputed RoPE frequencies
    pub freqs_cis: Vec<(f32, f32)>,
}

impl RunState {
    /// Create a new RunState with buffers allocated for the given config.
    pub fn new(config: &Config) -> Self {
        let kv_dim = config.kv_dim();
        let kv_cache_size = config.n_layers * config.seq_len * kv_dim;

        Self {
            x: vec![0.0; config.dim],
            xb: vec![0.0; config.dim],
            xb2: vec![0.0; config.dim],
            hb: vec![0.0; config.hidden_dim],
            hb2: vec![0.0; config.hidden_dim],
            q: vec![0.0; config.dim],
            att: vec![0.0; config.n_heads * config.seq_len],
            logits: vec![0.0; config.vocab_size],
            key_cache: vec![0.0; kv_cache_size],
            value_cache: vec![0.0; kv_cache_size],
            freqs_cis: precompute_freqs_cis(config.dim, config.seq_len, 10000.0),
        }
    }
}

/// Complete transformer model with weights and runtime state.
pub struct Transformer {
    /// Model configuration
    pub config: Config,
    /// Model weights
    pub weights: TransformerWeights,
    /// Runtime state buffers
    pub state: RunState,
}

impl Transformer {
    /// Load a transformer from a checkpoint file.
    ///
    /// # Arguments
    /// * `checkpoint_path` - Path to the model checkpoint file
    ///
    /// # Returns
    /// Loaded transformer ready for inference
    pub fn from_checkpoint<P: AsRef<Path>>(checkpoint_path: P) -> Result<Self> {
        // First read the config from the file
        let file = std::fs::File::open(&checkpoint_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        if mmap.len() < std::mem::size_of::<Config>() {
            return Err(Error::ModelLoad("Checkpoint file too small".to_string()));
        }

        let config_ptr = mmap.as_ptr() as *const Config;
        let config = unsafe { &*config_ptr }.clone();

        // Validate config
        config.validate()?;

        // Load weights
        let weights = TransformerWeights::from_checkpoint(checkpoint_path, &config)?;

        // Create runtime state
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
    /// Slice of logits for all vocabulary tokens
    pub fn forward(&mut self, token: usize, pos: usize) -> Result<&[f32]> {
        if pos >= self.config.seq_len {
            return Err(Error::Inference(format!(
                "Position {} exceeds maximum sequence length {}",
                pos, self.config.seq_len
            )));
        }

        // Convenience references
        let config = &self.config;
        let weights = &self.weights;
        let state = &mut self.state;

        let dim = config.dim;
        let kv_dim = config.kv_dim();
        let kv_mul = config.kv_mul();
        let hidden_dim = config.hidden_dim;
        let head_size = config.head_size();

        // Copy token embedding into x
        let token_embedding = weights.get_token_embedding(token, config);
        state.x.copy_from_slice(token_embedding);

        // Forward all layers
        for layer in 0..config.n_layers {
            // Attention RMSNorm
            let rms_att_weight = weights.get_rms_att_weight(layer, config);
            rmsnorm(&mut state.xb, &state.x, rms_att_weight, dim);

            // Key and value point to KV cache
            let loff = layer * config.seq_len * kv_dim;

            // Get pointers to current KV cache positions
            let k_start = loff + pos * kv_dim;
            let v_start = loff + pos * kv_dim;
            let k_cache = &mut state.key_cache[k_start..k_start + kv_dim];
            let v_cache = &mut state.value_cache[v_start..v_start + kv_dim];

            // QKV matmuls for this position
            let wq = weights.get_wq(layer, config);
            matmul(&mut state.q, &state.xb, wq, dim, dim);

            let wk = weights.get_wk(layer, config);
            matmul(k_cache, &state.xb, wk, dim, kv_dim);

            let wv = weights.get_wv(layer, config);
            matmul(v_cache, &state.xb, wv, dim, kv_dim);

            // RoPE relative positional encoding
            apply_rotary_emb(&mut state.q, &state.freqs_cis, pos, dim);
            apply_rotary_emb(k_cache, &state.freqs_cis, pos, kv_dim);

            // Multihead attention
            for h in 0..config.n_heads {
                // Get query vector for this head
                let q_start = h * head_size;
                let q_end = q_start + head_size;
                let q = &state.q[q_start..q_end];

                // Attention scores for this head
                let att_start = h * config.seq_len;
                let att_end = att_start + config.seq_len;
                let att = &mut state.att[att_start..att_end];

                // Iterate over all timesteps
                for t in 0..=pos {
                    // Get key vector for this head and timestep
                    let k_start = loff + t * kv_dim + (h / kv_mul) * head_size;
                    let k_end = k_start + head_size;
                    let k = &state.key_cache[k_start..k_end];

                    // Calculate attention score
                    let mut score = 0.0;
                    for i in 0..head_size {
                        score += q[i] * k[i];
                    }
                    score /= (head_size as f32).sqrt();
                    att[t] = score;
                }

                // Softmax the scores
                softmax(&mut att[..=pos], pos + 1);

                // Weighted sum of values
                let xb_start = h * head_size;
                let xb_end = xb_start + head_size;
                let xb = &mut state.xb[xb_start..xb_end];
                xb.fill(0.0);

                for t in 0..=pos {
                    let v_start = loff + t * kv_dim + (h / kv_mul) * head_size;
                    let v_end = v_start + head_size;
                    let v = &state.value_cache[v_start..v_end];
                    let a = att[t];

                    for i in 0..head_size {
                        xb[i] += a * v[i];
                    }
                }
            }

            // Final matmul to get output of attention
            let wo = weights.get_wo(layer, config);
            matmul(&mut state.xb2, &state.xb, wo, dim, dim);

            // Residual connection back into x
            add_scaled(&mut state.x, &state.xb2, 1.0);

            // FFN RMSNorm
            let rms_ffn_weight = weights.get_rms_ffn_weight(layer, config);
            rmsnorm(&mut state.xb, &state.x, rms_ffn_weight, dim);

            // FFN: SwiGLU
            let w1 = weights.get_w1(layer, config);
            matmul(&mut state.hb, &state.xb, w1, dim, hidden_dim);

            let w3 = weights.get_w3(layer, config);
            matmul(&mut state.hb2, &state.xb, w3, dim, hidden_dim);

            // SwiGLU activation: SiLU(w1*x) * w3*x
            for i in 0..hidden_dim {
                let val = swish(state.hb[i]) * state.hb2[i];
                state.hb[i] = val;
            }

            // Final FFN matmul
            let w2 = weights.get_w2(layer, config);
            matmul(&mut state.xb, &state.hb, w2, hidden_dim, dim);

            // Residual connection
            add_scaled(&mut state.x, &state.xb, 1.0);
        }

        // Final RMSNorm
        let rms_final_weight = weights.get_rms_final_weight(config);
        rmsnorm(&mut state.x, &state.x, rms_final_weight, dim);

        // Classifier into logits
        let wcls = weights.get_wcls(config);
        matmul(&mut state.logits, &state.x, wcls, dim, config.vocab_size);

        Ok(&state.logits)
    }

    /// Generate text using the transformer model.
    ///
    /// # Arguments
    /// * `tokenizer` - Tokenizer for encoding/decoding text
    /// * `sampler` - Sampler for token selection
    /// * `prompt` - Input prompt text
    /// * `steps` - Maximum number of tokens to generate
    ///
    /// # Returns
    /// Generated text
    pub fn generate(
        &mut self,
        tokenizer: &mut Tokenizer,
        sampler: &mut Sampler,
        prompt: &str,
        steps: usize,
    ) -> Result<String> {
        let mut result = String::new();

        // Encode prompt
        let tokens = tokenizer.encode(prompt, true, false);

        if tokens.is_empty() {
            return Err(Error::Inference("Failed to encode prompt".to_string()));
        }

        // Start with first token
        let mut token = tokens[0];
        let mut pos = 0;

        // Process prompt tokens
        for &prompt_token in &tokens[1..] {
            self.forward(token, pos)?;
            pos += 1;
            token = prompt_token;
        }

        // Generate new tokens
        while pos < steps {
            // Forward pass
            let logits = self.forward(token, pos)?;
            let mut logits_copy = logits.to_vec();

            // Sample next token
            let next_token = sampler.sample(&mut logits_copy)?;

            // Decode and append to result
            let piece = tokenizer.decode(token, next_token)?;
            let safe_piece = tokenizer.safe_print(&piece);
            result.push_str(&safe_piece);

            // Update for next iteration
            token = next_token;
            pos += 1;

            // Stop at EOS token
            if next_token == 2 {
                break;
            }
        }

        Ok(result)
    }

    /// Get model configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // Note: These tests would require actual model files to run
    // For now, they serve as documentation of the API

    #[test]
    fn test_runstate_creation() {
        let config = Config {
            dim: 768,
            hidden_dim: 3072,
            n_layers: 12,
            n_heads: 12,
            n_kv_heads: 12,
            vocab_size: 32000,
            seq_len: 1024,
        };

        let state = RunState::new(&config);

        assert_eq!(state.x.len(), 768);
        assert_eq!(state.xb.len(), 768);
        assert_eq!(state.hb.len(), 3072);
        assert_eq!(state.logits.len(), 32000);
        assert_eq!(state.key_cache.len(), 12 * 1024 * 768);
    }

    #[test]
    fn test_config_validation() {
        let config = Config {
            dim: 768,
            hidden_dim: 3072,
            n_layers: 12,
            n_heads: 12,
            n_kv_heads: 12,
            vocab_size: 32000,
            seq_len: 1024,
        };

        assert!(config.validate().is_ok());
    }
}
