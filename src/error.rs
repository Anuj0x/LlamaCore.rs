//! Error types for the llama2-rs crate.

use std::fmt;

/// Result type alias for operations that can fail.
pub type Result<T> = std::result::Result<T, Error>;

/// Comprehensive error type for all llama2-rs operations.
#[derive(Debug)]
pub enum Error {
    /// I/O errors (file operations, memory mapping, etc.)
    Io(std::io::Error),

    /// Invalid model configuration or parameters
    InvalidConfig(String),

    /// Tokenizer-related errors
    Tokenizer(String),

    /// Model loading/parsing errors
    ModelLoad(String),

    /// Runtime inference errors
    Inference(String),

    /// Quantization-related errors
    Quantization(String),

    /// Memory allocation errors
    Memory(String),

    /// Invalid arguments or parameters
    InvalidArgument(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(err) => write!(f, "I/O error: {}", err),
            Error::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            Error::Tokenizer(msg) => write!(f, "Tokenizer error: {}", msg),
            Error::ModelLoad(msg) => write!(f, "Model loading error: {}", msg),
            Error::Inference(msg) => write!(f, "Inference error: {}", msg),
            Error::Quantization(msg) => write!(f, "Quantization error: {}", msg),
            Error::Memory(msg) => write!(f, "Memory error: {}", msg),
            Error::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        Error::Inference(err.to_string())
    }
}
