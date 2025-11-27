//! Model configuration structures.

use serde::{Deserialize, Serialize};

/// Configuration for a Llama 2 transformer model.
///
/// This struct contains all the hyperparameters that define the architecture
/// of a Llama 2 model, including dimensions, number of layers, heads, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Transformer dimension (embedding dimension)
    pub dim: usize,

    /// Hidden dimension for feed-forward layers
    pub hidden_dim: usize,

    /// Number of transformer layers
    pub n_layers: usize,

    /// Number of query heads
    pub n_heads: usize,

    /// Number of key/value heads (can be less than query heads for multi-query)
    pub n_kv_heads: usize,

    /// Vocabulary size
    pub vocab_size: usize,

    /// Maximum sequence length
    pub seq_len: usize,
}

impl Config {
    /// Calculate the size of the key/value dimension (accounting for multi-query attention)
    pub fn kv_dim(&self) -> usize {
        (self.dim * self.n_kv_heads) / self.n_heads
    }

    /// Calculate the head size
    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }

    /// Calculate the number of KV repetitions for multi-query attention
    pub fn kv_mul(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// Validate the configuration for consistency
    pub fn validate(&self) -> crate::Result<()> {
        if self.dim == 0 {
            return Err(crate::Error::InvalidConfig("dim must be > 0".to_string()));
        }
        if self.hidden_dim == 0 {
            return Err(crate::Error::InvalidConfig("hidden_dim must be > 0".to_string()));
        }
        if self.n_layers == 0 {
            return Err(crate::Error::InvalidConfig("n_layers must be > 0".to_string()));
        }
        if self.n_heads == 0 {
            return Err(crate::Error::InvalidConfig("n_heads must be > 0".to_string()));
        }
        if self.n_kv_heads == 0 {
            return Err(crate::Error::InvalidConfig("n_kv_heads must be > 0".to_string()));
        }
        if self.vocab_size == 0 {
            return Err(crate::Error::InvalidConfig("vocab_size must be > 0".to_string()));
        }
        if self.seq_len == 0 {
            return Err(crate::Error::InvalidConfig("seq_len must be > 0".to_string()));
        }
        if self.dim % self.n_heads != 0 {
            return Err(crate::Error::InvalidConfig(
                "dim must be divisible by n_heads".to_string(),
            ));
        }
        if self.n_heads % self.n_kv_heads != 0 {
            return Err(crate::Error::InvalidConfig(
                "n_heads must be divisible by n_kv_heads".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(config.kv_dim(), 768);
        assert_eq!(config.head_size(), 64);
        assert_eq!(config.kv_mul(), 1);
    }

    #[test]
    fn test_invalid_config() {
        let config = Config {
            dim: 0, // Invalid
            hidden_dim: 3072,
            n_layers: 12,
            n_heads: 12,
            n_kv_heads: 12,
            vocab_size: 32000,
            seq_len: 1024,
        };

        assert!(config.validate().is_err());
    }
}
