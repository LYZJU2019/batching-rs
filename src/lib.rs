//! # Continuous Batching for Llama Models
//!
//! This library implements core continuous batching logic for Llama-family models
//! (including Llama 3 with Grouped Query Attention) using Rust and MLX.
//!
//! ## Scope
//! - Core batching logic only
//! - No HTTP/SSE serving layer
//! - No tokenizer or text processing
//! - No advanced optimizations (paged KV, speculative decoding, etc.)
//!
//! ## Architecture
//! - `config`: Model configuration (layers, heads, GQA parameters)
//! - `sequence`: Sequence state management and lifecycle
//! - `kv_cache`: Per-sequence KV cache storage
//! - `scheduler`: TGI-style continuous batching scheduler
//! - `model`: Llama forward pass (prefill + decode)
//! - `worker`: Single-threaded model worker and execution loop

// Error handling
use thiserror::Error;

/// Error types for the continuous batching system
#[derive(Debug, Error)]
pub enum BatchingError {
    /// Error from MLX operations
    #[error("MLX error: {0}")]
    MlxError(String),

    /// Sequence not found in scheduler
    #[error("Sequence not found: {0}")]
    SequenceNotFound(u64),

    /// Invalid configuration provided
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Model error during forward pass
    #[error("Model error: {0}")]
    ModelError(String),

    /// Scheduler error
    #[error("Scheduler error: {0}")]
    SchedulerError(String),
}

// Convert MLX errors when the feature is enabled
#[cfg(feature = "mlx")]
impl From<mlx_rs::error::Exception> for BatchingError {
    fn from(err: mlx_rs::error::Exception) -> Self {
        BatchingError::MlxError(err.to_string())
    }
}

/// Result type alias for continuous batching operations
pub type Result<T> = std::result::Result<T, BatchingError>;

// Module declarations
pub mod config;
pub mod kv_cache;
pub mod model;
pub mod scheduler;
pub mod sequence;
// pub mod worker;

// Re-exports
pub use config::ModelConfig;
pub use kv_cache::{LayerKVCache, SequenceKVCache};
pub use model::LlamaModel;
pub use scheduler::{Scheduler, SchedulerDecision};
pub use sequence::{SeqId, Sequence, SequenceStatus};
// pub use worker::ModelWorker;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_types() {
        // Test that our error types can be created and formatted
        let err = BatchingError::SequenceNotFound(42);
        assert_eq!(err.to_string(), "Sequence not found: 42");

        let err = BatchingError::InvalidConfig("test error".to_string());
        assert_eq!(err.to_string(), "Invalid configuration: test error");

        let err = BatchingError::ModelError("test model error".to_string());
        assert_eq!(err.to_string(), "Model error: test model error");

        let err = BatchingError::SchedulerError("test scheduler error".to_string());
        assert_eq!(err.to_string(), "Scheduler error: test scheduler error");

        let err = BatchingError::MlxError("test mlx error".to_string());
        assert_eq!(err.to_string(), "MLX error: test mlx error");
    }

    #[test]
    fn test_result_type() {
        // Test that our Result type works correctly
        let success: Result<i32> = Ok(42);
        assert!(success.is_ok());
        assert_eq!(success.unwrap(), 42);

        let failure: Result<i32> = Err(BatchingError::SequenceNotFound(1));
        assert!(failure.is_err());
    }
}
