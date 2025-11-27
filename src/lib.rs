//! # llama2-rs
//!
//! A modern, safe, and efficient Rust implementation of Llama 2 inference.
//!
//! This crate provides a complete transformer implementation with memory-mapped weights,
//! optimized operations using Rayon for parallelism, and a clean, safe API.

pub mod config;
pub mod error;
pub mod model;
pub mod operations;
pub mod sampler;
pub mod tokenizer;
pub mod transformer;

// Re-export main types
pub use config::Config;
pub use error::{Error, Result};
pub use model::Transformer;
pub use sampler::Sampler;
pub use tokenizer::Tokenizer;

use std::path::Path;

/// Load a transformer model from a checkpoint file.
///
/// This is the main entry point for loading and using a trained Llama 2 model.
/// The model weights are memory-mapped for efficient loading and minimal memory usage.
///
/// # Arguments
/// * `checkpoint_path` - Path to the model checkpoint file (.bin format)
///
/// # Returns
/// A fully loaded transformer ready for inference
///
/// # Example
/// ```no_run
/// use llama2_rs::{load_transformer, Sampler, Tokenizer};
///
/// // Load model
/// let mut transformer = load_transformer("stories15M.bin").unwrap();
///
/// // Load tokenizer
/// let mut tokenizer = Tokenizer::from_file("tokenizer.bin").unwrap();
///
/// // Create sampler
/// let mut sampler = Sampler::new(32000, 1.0, 0.9, 42);
///
/// // Generate text
/// let text = transformer.generate(&mut tokenizer, &mut sampler, "Once upon a time", 100).unwrap();
/// println!("{}", text);
/// ```
pub fn load_transformer<P: AsRef<Path>>(checkpoint_path: P) -> Result<Transformer> {
    Transformer::from_checkpoint(checkpoint_path)
}
