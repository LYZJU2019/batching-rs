//! Model configuration for Llama-family models
//!
//! This module defines the architecture parameters for Llama models,
//! including support for Grouped Query Attention (GQA) as used in Llama 3.

use crate::{BatchingError, Result};

/// Configuration for Llama model architecture
///
/// This struct contains all the hyperparameters needed to define a Llama model,
/// including support for Grouped Query Attention (GQA).
///
/// # Example
///
/// ```
/// use batching_rs::ModelConfig;
///
/// // Llama 3 8B configuration
/// let config = ModelConfig::llama3_8b();
/// assert_eq!(config.n_heads, 32);
/// assert_eq!(config.n_kv_heads, 8); // GQA with 4 query heads per KV head
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    // Model architecture dimensions
    /// Size of the vocabulary
    pub vocab_size: usize,
    
    /// Number of transformer layers
    pub n_layers: usize,
    
    /// Hidden dimension (model dimension)
    pub hidden_dim: usize,
    
    /// Number of attention heads for queries
    pub n_heads: usize,
    
    /// Number of attention heads for keys and values (for GQA)
    ///
    /// For standard Multi-Head Attention (MHA): n_kv_heads == n_heads
    /// For Grouped Query Attention (GQA): n_kv_heads < n_heads
    /// For Multi-Query Attention (MQA): n_kv_heads == 1
    pub n_kv_heads: usize,
    
    /// Dimension of each attention head
    pub head_dim: usize,
    
    /// Dimension of the feedforward intermediate layer
    pub intermediate_dim: usize,
    
    // RoPE (Rotary Position Embedding) configuration
    /// Base frequency for RoPE
    pub rope_base: f32,
    
    /// Scaling factor for RoPE (usually 1.0)
    pub rope_scale: f32,
    
    // Normalization
    /// Epsilon for RMS normalization
    pub rms_norm_eps: f32,
}

impl ModelConfig {
    /// Create a new ModelConfig with validation
    ///
    /// # Arguments
    ///
    /// * `vocab_size` - Size of the vocabulary
    /// * `n_layers` - Number of transformer layers
    /// * `hidden_dim` - Hidden dimension
    /// * `n_heads` - Number of query heads
    /// * `n_kv_heads` - Number of key/value heads (for GQA)
    /// * `head_dim` - Dimension per head
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `rope_base` - RoPE base frequency
    /// * `rope_scale` - RoPE scaling factor
    /// * `rms_norm_eps` - RMS norm epsilon
    ///
    /// # Errors
    ///
    /// Returns an error if configuration is invalid:
    /// - n_heads must be divisible by n_kv_heads (GQA requirement)
    /// - hidden_dim must equal n_heads * head_dim
    /// - All dimensions must be positive
    pub fn new(
        vocab_size: usize,
        n_layers: usize,
        hidden_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
        rope_base: f32,
        rope_scale: f32,
        rms_norm_eps: f32,
    ) -> Result<Self> {
        let config = Self {
            vocab_size,
            n_layers,
            hidden_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            intermediate_dim,
            rope_base,
            rope_scale,
            rms_norm_eps,
        };
        
        config.validate()?;
        Ok(config)
    }
    
    /// Validate the configuration
    ///
    /// Checks that all parameters are valid and consistent.
    pub fn validate(&self) -> Result<()> {
        // Check positive dimensions
        if self.vocab_size == 0 {
            return Err(BatchingError::InvalidConfig(
                "vocab_size must be positive".to_string()
            ));
        }
        if self.n_layers == 0 {
            return Err(BatchingError::InvalidConfig(
                "n_layers must be positive".to_string()
            ));
        }
        if self.hidden_dim == 0 {
            return Err(BatchingError::InvalidConfig(
                "hidden_dim must be positive".to_string()
            ));
        }
        if self.n_heads == 0 {
            return Err(BatchingError::InvalidConfig(
                "n_heads must be positive".to_string()
            ));
        }
        if self.n_kv_heads == 0 {
            return Err(BatchingError::InvalidConfig(
                "n_kv_heads must be positive".to_string()
            ));
        }
        if self.head_dim == 0 {
            return Err(BatchingError::InvalidConfig(
                "head_dim must be positive".to_string()
            ));
        }
        if self.intermediate_dim == 0 {
            return Err(BatchingError::InvalidConfig(
                "intermediate_dim must be positive".to_string()
            ));
        }
        
        // GQA requirement: n_heads must be divisible by n_kv_heads
        if self.n_heads % self.n_kv_heads != 0 {
            return Err(BatchingError::InvalidConfig(
                format!(
                    "n_heads ({}) must be divisible by n_kv_heads ({}) for GQA",
                    self.n_heads, self.n_kv_heads
                )
            ));
        }
        
        // Check that hidden_dim matches n_heads * head_dim
        if self.hidden_dim != self.n_heads * self.head_dim {
            return Err(BatchingError::InvalidConfig(
                format!(
                    "hidden_dim ({}) must equal n_heads ({}) * head_dim ({})",
                    self.hidden_dim, self.n_heads, self.head_dim
                )
            ));
        }
        
        // Check positive floats
        if self.rope_base <= 0.0 {
            return Err(BatchingError::InvalidConfig(
                "rope_base must be positive".to_string()
            ));
        }
        if self.rope_scale <= 0.0 {
            return Err(BatchingError::InvalidConfig(
                "rope_scale must be positive".to_string()
            ));
        }
        if self.rms_norm_eps <= 0.0 {
            return Err(BatchingError::InvalidConfig(
                "rms_norm_eps must be positive".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Get the number of query heads per KV head (for GQA)
    ///
    /// This is the grouping factor in Grouped Query Attention.
    /// For MHA (Multi-Head Attention), this returns 1.
    /// For GQA, this returns n_heads / n_kv_heads > 1.
    pub fn n_queries_per_kv(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
    
    /// Llama 3 8B configuration
    ///
    /// Standard configuration for Llama 3 8B model with GQA.
    pub fn llama3_8b() -> Self {
        Self {
            vocab_size: 128_256,
            n_layers: 32,
            hidden_dim: 4096,
            n_heads: 32,
            n_kv_heads: 8, // GQA: 4 query heads per KV head
            head_dim: 128,
            intermediate_dim: 14_336,
            rope_base: 500_000.0,
            rope_scale: 1.0,
            rms_norm_eps: 1e-5,
        }
    }
    
    /// Llama 3 70B configuration
    ///
    /// Standard configuration for Llama 3 70B model with GQA.
    pub fn llama3_70b() -> Self {
        Self {
            vocab_size: 128_256,
            n_layers: 80,
            hidden_dim: 8192,
            n_heads: 64,
            n_kv_heads: 8, // GQA: 8 query heads per KV head
            head_dim: 128,
            intermediate_dim: 28_672,
            rope_base: 500_000.0,
            rope_scale: 1.0,
            rms_norm_eps: 1e-5,
        }
    }
    
    /// Llama 2 7B configuration (without GQA, standard MHA)
    ///
    /// For comparison: Llama 2 uses standard Multi-Head Attention.
    pub fn llama2_7b() -> Self {
        Self {
            vocab_size: 32_000,
            n_layers: 32,
            hidden_dim: 4096,
            n_heads: 32,
            n_kv_heads: 32, // MHA: same number of KV heads as query heads
            head_dim: 128,
            intermediate_dim: 11_008,
            rope_base: 10_000.0,
            rope_scale: 1.0,
            rms_norm_eps: 1e-5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_valid_config() {
        let config = ModelConfig::new(
            32_000,  // vocab_size
            32,      // n_layers
            4096,    // hidden_dim
            32,      // n_heads
            8,       // n_kv_heads (GQA)
            128,     // head_dim
            11_008,  // intermediate_dim
            10_000.0,
            1.0,
            1e-5,
        );
        assert!(config.is_ok());
        
        let config = config.unwrap();
        assert_eq!(config.n_queries_per_kv(), 4);
    }
    
    #[test]
    fn test_invalid_gqa_grouping() {
        // n_heads not divisible by n_kv_heads
        let config = ModelConfig::new(
            32_000,
            32,
            4096,
            32,
            7, // Not a divisor of 32
            128,
            11_008,
            10_000.0,
            1.0,
            1e-5,
        );
        assert!(config.is_err());
    }
    
    #[test]
    fn test_invalid_hidden_dim() {
        // hidden_dim != n_heads * head_dim
        let config = ModelConfig::new(
            32_000,
            32,
            4000, // Wrong: should be 32 * 128 = 4096
            32,
            8,
            128,
            11_008,
            10_000.0,
            1.0,
            1e-5,
        );
        assert!(config.is_err());
    }
    
    #[test]
    fn test_zero_dimensions() {
        let config = ModelConfig::new(
            0, // Invalid: zero vocab_size
            32,
            4096,
            32,
            8,
            128,
            11_008,
            10_000.0,
            1.0,
            1e-5,
        );
        assert!(config.is_err());
    }
    
    #[test]
    fn test_negative_floats() {
        let config = ModelConfig::new(
            32_000,
            32,
            4096,
            32,
            8,
            128,
            11_008,
            -10_000.0, // Invalid: negative rope_base
            1.0,
            1e-5,
        );
        assert!(config.is_err());
    }
    
    #[test]
    fn test_llama3_8b_preset() {
        let config = ModelConfig::llama3_8b();
        assert_eq!(config.vocab_size, 128_256);
        assert_eq!(config.n_layers, 32);
        assert_eq!(config.n_heads, 32);
        assert_eq!(config.n_kv_heads, 8);
        assert_eq!(config.n_queries_per_kv(), 4);
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_llama3_70b_preset() {
        let config = ModelConfig::llama3_70b();
        assert_eq!(config.vocab_size, 128_256);
        assert_eq!(config.n_layers, 80);
        assert_eq!(config.n_heads, 64);
        assert_eq!(config.n_kv_heads, 8);
        assert_eq!(config.n_queries_per_kv(), 8);
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_llama2_7b_preset() {
        let config = ModelConfig::llama2_7b();
        assert_eq!(config.n_heads, 32);
        assert_eq!(config.n_kv_heads, 32); // MHA: same number
        assert_eq!(config.n_queries_per_kv(), 1); // No grouping
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_config_clone() {
        let config1 = ModelConfig::llama3_8b();
        let config2 = config1.clone();
        assert_eq!(config1, config2);
    }
}
