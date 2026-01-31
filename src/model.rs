//! Llama model interface and stub implementation
//!
//! This module defines the LlamaModel interface for prefill and decode operations.
//! The actual model implementation with GQA and RoPE will be completed in PR-008.
//! For now, this provides a stub that returns dummy tensors with correct shapes.

use crate::{
    config::ModelConfig,
    kv_cache::SequenceKVCache,
    BatchingError, Result,
};

/// Llama model for text generation
///
/// This model implements the forward pass for Llama-family models with support
/// for Grouped Query Attention (GQA). It provides two main operations:
/// - `prefill`: Process the full prompt and initialize KV cache
/// - `decode_step`: Generate one token for a batch of sequences
///
/// # Architecture Notes
///
/// ## Grouped Query Attention (GQA)
/// - Query heads: `config.n_heads` (e.g., 32)
/// - KV heads: `config.n_kv_heads` (e.g., 8 for Llama 3)
/// - Each KV head serves `n_heads / n_kv_heads` query heads
/// - KV tensors shape: `[n_kv_heads, seq_len, head_dim]`
/// - During attention, KV heads are repeated to match query heads
///
/// ## RoPE (Rotary Position Embedding)
/// - Applied to Q and K before attention
/// - Uses absolute position indices for each token
/// - Frequencies: `freq_i = base^(-2i/head_dim)`
///
/// ## Prefill vs Decode
/// - **Prefill**: Full causal attention over prompt (compute all KV pairs)
/// - **Decode**: Single-token forward with existing KV cache (append 1 KV pair)
///
/// # Example
///
/// ```
/// use batching_rs::{LlamaModel, ModelConfig};
///
/// let config = ModelConfig::llama3_8b();
/// let model = LlamaModel::new(config).unwrap();
///
/// // This is a stub implementation - returns dummy tensors
/// // Actual model implementation will be in PR-008
/// ```
pub struct LlamaModel {
    /// Model configuration
    config: ModelConfig,
}

impl LlamaModel {
    /// Create a new Llama model
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration with architecture parameters
    ///
    /// # Returns
    ///
    /// A new LlamaModel instance
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid
    pub fn new(config: ModelConfig) -> Result<Self> {
        // Validate configuration
        config.validate()?;
        
        Ok(Self { config })
    }
    
    /// Prefill: compute KV cache for entire prompt
    ///
    /// This performs a forward pass over the full prompt sequence, computing
    /// all key-value pairs and storing them in the KV cache. Returns logits
    /// at the last position for sampling the first generated token.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Prompt tokens `[seq_len]`
    /// * `start_pos` - Starting position index (usually 0 for prefill)
    ///
    /// # Returns
    ///
    /// A tuple of:
    /// - `logits`: Logits at last position `[vocab_size]`
    /// - `kv_cache`: Populated SequenceKVCache with K/V for all layers
    ///
    /// # Implementation Notes (for PR-008)
    ///
    /// 1. Embed tokens: `[seq_len] -> [seq_len, hidden_dim]`
    /// 2. For each layer:
    ///    - Apply RMSNorm
    ///    - Compute Q, K, V with GQA (K/V use n_kv_heads)
    ///    - Apply RoPE to Q and K (positions: start_pos..start_pos+seq_len)
    ///    - Causal self-attention with triangular mask
    ///    - Store K/V in cache: `[n_kv_heads, seq_len, head_dim]`
    ///    - Feedforward (SwiGLU)
    /// 3. Final RMSNorm + output projection
    /// 4. Return logits at last position
    ///
    /// # Stub Implementation
    ///
    /// Currently returns dummy tensors with correct shapes for testing.
    pub fn prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<(Vec<f32>, SequenceKVCache)> {
        if tokens.is_empty() {
            return Err(BatchingError::ModelError(
                "Cannot prefill empty token sequence".to_string()
            ));
        }
        
        let seq_len = tokens.len();
        
        // TODO (PR-008): Actual model forward pass
        // For now, return dummy tensors with correct shapes
        
        // Dummy logits: [vocab_size]
        let logits = vec![0.0; self.config.vocab_size];
        
        // Populate KV cache with dummy values
        let mut kv_cache = SequenceKVCache::new(self.config.n_layers);
        for layer_idx in 0..self.config.n_layers {
            let cache_size = self.config.n_kv_heads * seq_len * self.config.head_dim;
            let k = vec![1.0; cache_size];
            let v = vec![1.0; cache_size];
            kv_cache.set_layer(
                layer_idx,
                k,
                v,
                (self.config.n_kv_heads, seq_len, self.config.head_dim),
            );
        }
        
        // Suppress unused variable warning
        let _ = start_pos;
        
        Ok((logits, kv_cache))
    }
    
    /// Decode: forward pass for batched sequences (1 token each)
    ///
    /// This performs a single-token forward pass for each sequence in the batch.
    /// Each sequence has its own KV cache at a different length. The function
    /// appends new K/V pairs to each sequence's cache.
    ///
    /// # Arguments
    ///
    /// * `tokens` - One token per sequence `[batch_size]`
    /// * `positions` - Position index for each sequence `[batch_size]`
    /// * `kv_caches` - Mutable references to each sequence's KV cache
    ///
    /// # Returns
    ///
    /// Logits for each sequence: `[batch_size, vocab_size]` (flattened)
    ///
    /// # Side Effects
    ///
    /// Updates each sequence's KV cache by appending new K/V pairs.
    ///
    /// # Implementation Notes (for PR-008)
    ///
    /// 1. Embed tokens: `[batch_size] -> [batch_size, hidden_dim]`
    /// 2. For each layer:
    ///    - Apply RMSNorm
    ///    - Compute Q, K, V with GQA (K/V use n_kv_heads)
    ///    - Apply RoPE to Q and K (use positions array)
    ///    - For each sequence i:
    ///      - Attend to cached K/V plus new K/V
    ///      - Cache length varies per sequence
    ///    - Append new K/V to cache: `[n_kv_heads, 1, head_dim]`
    ///    - Feedforward (SwiGLU)
    /// 3. Final RMSNorm + output projection
    /// 4. Return logits: `[batch_size, vocab_size]`
    ///
    /// # Batching Challenge
    ///
    /// Each sequence has different KV cache length. Options:
    /// - Process sequences independently (simple, implemented here)
    /// - Use padding + attention mask (more efficient)
    /// - Use paged attention (most efficient, out of scope)
    ///
    /// # Stub Implementation
    ///
    /// Currently returns dummy logits and updates KV caches with dummy values.
    pub fn decode_step(
        &self,
        tokens: &[u32],
        positions: &[usize],
        kv_caches: &mut [&mut SequenceKVCache],
    ) -> Result<Vec<f32>> {
        let batch_size = tokens.len();
        
        if batch_size == 0 {
            return Err(BatchingError::ModelError(
                "Cannot decode empty batch".to_string()
            ));
        }
        
        if positions.len() != batch_size {
            return Err(BatchingError::ModelError(
                format!(
                    "Positions length ({}) must match batch size ({})",
                    positions.len(),
                    batch_size
                )
            ));
        }
        
        if kv_caches.len() != batch_size {
            return Err(BatchingError::ModelError(
                format!(
                    "KV caches length ({}) must match batch size ({})",
                    kv_caches.len(),
                    batch_size
                )
            ));
        }
        
        // TODO (PR-008): Actual model forward pass
        // For now, return dummy tensors and update caches
        
        // Dummy logits: [batch_size, vocab_size] (flattened)
        let logits = vec![0.0; batch_size * self.config.vocab_size];
        
        // Append dummy K/V to each sequence's cache
        for (i, cache) in kv_caches.iter_mut().enumerate() {
            let cache_size = self.config.n_kv_heads * 1 * self.config.head_dim;
            let k = vec![1.0; cache_size];
            let v = vec![1.0; cache_size];
            
            for layer_idx in 0..self.config.n_layers {
                cache.append_layer(
                    layer_idx,
                    k.clone(),
                    v.clone(),
                    (self.config.n_kv_heads, 1, self.config.head_dim),
                );
            }
            
            // Suppress unused variable warning
            let _ = (tokens[i], positions[i]);
        }
        
        Ok(logits)
    }
    
    /// Get the model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new_model() {
        let config = ModelConfig::llama3_8b();
        let model = LlamaModel::new(config.clone());
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert_eq!(model.config().vocab_size, config.vocab_size);
    }
    
    #[test]
    fn test_new_model_invalid_config() {
        // Create invalid config (n_heads not divisible by n_kv_heads)
        let result = ModelConfig::new(
            32_000, 32, 4096, 32, 7, 128, 11_008, 10_000.0, 1.0, 1e-5
        );
        assert!(result.is_err());
    }
    
    #[test]
    fn test_prefill_stub() {
        let config = ModelConfig::llama3_8b();
        let model = LlamaModel::new(config.clone()).unwrap();
        
        let tokens = vec![1, 2, 3, 4, 5];
        let result = model.prefill(&tokens, 0);
        
        assert!(result.is_ok());
        let (logits, kv_cache) = result.unwrap();
        
        // Check logits shape
        assert_eq!(logits.len(), config.vocab_size);
        
        // Check KV cache was populated
        assert_eq!(kv_cache.num_layers(), config.n_layers);
        assert_eq!(kv_cache.current_length(), tokens.len());
        
        // Verify each layer has correct shape
        for layer_idx in 0..config.n_layers {
            let layer = kv_cache.get_layer(layer_idx);
            assert_eq!(layer.seq_len(), tokens.len());
            assert_eq!(
                layer.shape,
                Some((config.n_kv_heads, tokens.len(), config.head_dim))
            );
        }
    }
    
    #[test]
    fn test_prefill_empty_tokens() {
        let config = ModelConfig::llama3_8b();
        let model = LlamaModel::new(config).unwrap();
        
        let result = model.prefill(&[], 0);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_decode_step_stub() {
        let config = ModelConfig::llama3_8b();
        let model = LlamaModel::new(config.clone()).unwrap();
        
        // Create two sequences with different cache lengths
        let mut cache1 = SequenceKVCache::new(config.n_layers);
        let mut cache2 = SequenceKVCache::new(config.n_layers);
        
        // Initialize cache1 with length 5
        for layer_idx in 0..config.n_layers {
            let size = config.n_kv_heads * 5 * config.head_dim;
            cache1.set_layer(layer_idx, vec![1.0; size], vec![1.0; size],
                           (config.n_kv_heads, 5, config.head_dim));
        }
        
        // Initialize cache2 with length 3
        for layer_idx in 0..config.n_layers {
            let size = config.n_kv_heads * 3 * config.head_dim;
            cache2.set_layer(layer_idx, vec![1.0; size], vec![1.0; size],
                           (config.n_kv_heads, 3, config.head_dim));
        }
        
        let tokens = vec![10, 20];
        let positions = vec![6, 4];
        let mut caches = vec![&mut cache1, &mut cache2];
        
        let result = model.decode_step(&tokens, &positions, &mut caches);
        assert!(result.is_ok());
        
        let logits = result.unwrap();
        
        // Check logits shape: [batch_size, vocab_size]
        assert_eq!(logits.len(), 2 * config.vocab_size);
        
        // Check caches were updated (each grew by 1)
        assert_eq!(cache1.current_length(), 6);
        assert_eq!(cache2.current_length(), 4);
    }
    
    #[test]
    fn test_decode_step_empty_batch() {
        let config = ModelConfig::llama3_8b();
        let model = LlamaModel::new(config).unwrap();
        
        let result = model.decode_step(&[], &[], &mut []);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_decode_step_mismatched_lengths() {
        let config = ModelConfig::llama3_8b();
        let model = LlamaModel::new(config.clone()).unwrap();
        
        let mut cache = SequenceKVCache::new(config.n_layers);
        
        // Mismatch: 2 tokens, 1 position
        let result = model.decode_step(&[1, 2], &[0], &mut [&mut cache]);
        assert!(result.is_err());
        
        // Mismatch: 1 token, 2 positions
        let result = model.decode_step(&[1], &[0, 1], &mut [&mut cache]);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_prefill_then_decode() {
        let config = ModelConfig::llama3_8b();
        let model = LlamaModel::new(config.clone()).unwrap();
        
        // 1. Prefill
        let prompt = vec![1, 2, 3];
        let (logits, mut kv_cache) = model.prefill(&prompt, 0).unwrap();
        assert_eq!(logits.len(), config.vocab_size);
        assert_eq!(kv_cache.current_length(), 3);
        
        // 2. Decode step 1
        let tokens = vec![10];
        let positions = vec![3];
        let mut caches = vec![&mut kv_cache];
        let logits = model.decode_step(&tokens, &positions, &mut caches).unwrap();
        assert_eq!(logits.len(), config.vocab_size);
        assert_eq!(kv_cache.current_length(), 4);
        
        // 3. Decode step 2
        let tokens = vec![11];
        let positions = vec![4];
        let mut caches = vec![&mut kv_cache];
        let logits = model.decode_step(&tokens, &positions, &mut caches).unwrap();
        assert_eq!(logits.len(), config.vocab_size);
        assert_eq!(kv_cache.current_length(), 5);
    }
}
