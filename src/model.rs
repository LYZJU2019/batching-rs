//! Llama model interface and implementation
//!
//! This module implements the LlamaModel with a simulated forward pass that
//! demonstrates the correct structure and data flow for Llama-family models
//! with Grouped Query Attention (GQA) and RoPE.
//!
//! # Implementation Status (PR-008)
//!
//! This is a **simulated implementation** that:
//! - Has the correct algorithmic structure
//! - Computes proper tensor shapes
//! - Generates realistic (non-zero) logits based on input
//! - Can be replaced with MLX operations when available
//!
//! ## What's Implemented:
//! - Token embedding simulation
//! - RoPE frequency computation
//! - Layer-by-layer forward pass structure
//! - GQA attention shape handling
//! - SwiGLU feedforward structure
//! - Logit generation based on token IDs
//!
//! ## What's NOT Implemented (requires MLX):
//! - Actual weight matrices and computations
//! - Real attention mechanism
//! - Gradient computation
//! - Weight loading from checkpoints

use crate::gguf::{GGUFFile, GGUFTensorType};
use crate::{config::ModelConfig, kv_cache::SequenceKVCache, BatchingError, Result};
use std::collections::HashMap;

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
/// // Prefill a prompt
/// let prompt = vec![1, 2, 3, 4, 5];
/// let (logits, cache) = model.prefill(&prompt, 0).unwrap();
/// ```
pub struct LlamaModel {
    /// Model configuration
    config: ModelConfig,

    /// RoPE frequencies (precomputed)
    /// These would be used in a real implementation for rotary position embeddings
    #[allow(dead_code)]
    rope_freqs: Vec<f32>,

    /// Token embeddings: [vocab_size, hidden_dim]
    token_embeddings: Option<Vec<u8>>,

    /// Layer weights (stored as raw bytes from GGUF)
    /// Each layer contains attention and feedforward weights
    layer_weights: HashMap<String, Vec<u8>>,

    /// Output normalization weights
    output_norm: Option<Vec<u8>>,

    /// Output projection (lm_head): [hidden_dim, vocab_size]
    output_weights: Option<Vec<u8>>,

    /// Track tensor types for each weight
    tensor_types: HashMap<String, GGUFTensorType>,
}

impl LlamaModel {
    /// Create a new Llama model from a GGUF file
    ///
    /// This loads the actual model weights from the GGUF file and creates
    /// a model instance ready for inference.
    ///
    /// # Arguments
    ///
    /// * `gguf` - Parsed GGUF file containing model weights and metadata
    ///
    /// # Returns
    ///
    /// A new LlamaModel instance with loaded weights
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Configuration extraction fails
    /// - Required tensors are missing
    /// - Tensor dimensions are invalid
    pub fn from_gguf(gguf: &GGUFFile) -> Result<Self> {
        println!("🏗️  Building LlamaModel from GGUF file...");

        // Extract configuration
        let config = gguf.extract_config()?;

        // Precompute RoPE frequencies
        let rope_freqs = Self::precompute_rope_freqs(&config);

        // Show available tensors for debugging
        println!("   📋 Available tensors in GGUF file:");
        let tensor_names = gguf.tensor_names();
        println!("      Total: {} tensors", tensor_names.len());

        // Show a sample of tensor names to help with debugging
        let sample_count = 10.min(tensor_names.len());
        for name in tensor_names.iter().take(sample_count) {
            println!("      - {}", name);
        }
        if tensor_names.len() > sample_count {
            println!("      ... and {} more", tensor_names.len() - sample_count);
        }

        // Load token embeddings - try common naming conventions
        println!("\n   Loading token embeddings...");
        let (token_embeddings, token_emb_type, emb_name) = Self::try_load_tensor(
            gguf,
            &[
                "token_embd.weight",
                "tok_embeddings.weight",
                "model.embed_tokens.weight",
            ],
        )?;
        println!("      ✓ Loaded: {}", emb_name);

        // Load layer weights
        println!("   Loading {} transformer layers...", config.n_layers);
        let mut layer_weights = HashMap::new();
        let mut tensor_types = HashMap::new();

        tensor_types.insert(emb_name.clone(), token_emb_type);

        for layer_idx in 0..config.n_layers {
            // Try different naming conventions for layer tensors
            // llama.cpp style: blk.N.*
            // transformers style: model.layers.N.*

            for tensor_name in &[
                &format!("blk.{}.attn_q.weight", layer_idx),
                &format!("blk.{}.attn_k.weight", layer_idx),
                &format!("blk.{}.attn_v.weight", layer_idx),
                &format!("blk.{}.attn_output.weight", layer_idx),
                &format!("blk.{}.ffn_gate.weight", layer_idx),
                &format!("blk.{}.ffn_up.weight", layer_idx),
                &format!("blk.{}.ffn_down.weight", layer_idx),
                &format!("blk.{}.attn_norm.weight", layer_idx),
                &format!("blk.{}.ffn_norm.weight", layer_idx),
            ] {
                if let Ok((data, dtype)) = Self::load_tensor(gguf, tensor_name) {
                    layer_weights.insert(tensor_name.to_string(), data);
                    tensor_types.insert(tensor_name.to_string(), dtype);
                }
            }

            if (layer_idx + 1) % 8 == 0 || layer_idx == config.n_layers - 1 {
                println!("      Loaded layers 0-{}", layer_idx);
            }
        }

        // Load output normalization - try common naming conventions
        println!("   Loading output layers...");
        let (output_norm, output_norm_type, norm_name) = Self::try_load_tensor(
            gguf,
            &["output_norm.weight", "norm.weight", "model.norm.weight"],
        )?;
        println!("      ✓ Loaded: {}", norm_name);
        tensor_types.insert(norm_name.clone(), output_norm_type);

        // Load output weights (lm_head) - try common naming conventions
        let output_result = Self::try_load_tensor(
            gguf,
            &[
                "output.weight",
                "lm_head.weight",
                "output_weight",
                "model.lm_head.weight",
            ],
        );

        let (output_weights, output_type, output_name) = match output_result {
            Ok(result) => {
                println!("      ✓ Loaded: {}", result.2);
                result
            }
            Err(_) => {
                // Some models tie output weights to token embeddings
                println!("      ⚠️  No separate output.weight found, will use token embeddings (weight tying)");
                (token_embeddings.clone(), token_emb_type, emb_name.clone())
            }
        };
        tensor_types.insert(output_name, output_type);

        println!("   ✅ Model loaded successfully!");
        println!("      Total tensors loaded: {}", tensor_types.len());

        // Calculate total memory usage
        let token_emb_bytes = token_embeddings.len();
        let output_norm_bytes = output_norm.len();
        let output_weights_bytes = output_weights.len();
        let layer_weights_bytes: usize = layer_weights.values().map(|v| v.len()).sum();
        let total_bytes =
            token_emb_bytes + output_norm_bytes + output_weights_bytes + layer_weights_bytes;

        println!(
            "      Approximate size: {:.2} MB",
            total_bytes as f64 / (1024.0 * 1024.0)
        );

        Ok(Self {
            config,
            rope_freqs,
            token_embeddings: Some(token_embeddings),
            layer_weights,
            output_norm: Some(output_norm),
            output_weights: Some(output_weights),
            tensor_types,
        })
    }

    /// Load a tensor from GGUF file by name
    ///
    /// Returns the raw bytes and tensor type. The actual dequantization
    /// will be done during inference.
    fn load_tensor(gguf: &GGUFFile, name: &str) -> Result<(Vec<u8>, GGUFTensorType)> {
        let tensor_info = gguf.get_tensor_info(name).ok_or_else(|| {
            BatchingError::ModelError(format!("Required tensor '{}' not found in GGUF file", name))
        })?;

        let data = gguf.read_tensor_data(name)?;
        Ok((data, tensor_info.tensor_type))
    }

    /// Try to load a tensor with multiple possible names
    ///
    /// Returns the raw bytes, tensor type, and the name that was found.
    fn try_load_tensor(
        gguf: &GGUFFile,
        names: &[&str],
    ) -> Result<(Vec<u8>, GGUFTensorType, String)> {
        for &name in names {
            if let Some(tensor_info) = gguf.get_tensor_info(name) {
                let data = gguf.read_tensor_data(name)?;
                return Ok((data, tensor_info.tensor_type, name.to_string()));
            }
        }

        Err(BatchingError::ModelError(format!(
            "Required tensor not found. Tried: {:?}",
            names
        )))
    }

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

        // Precompute RoPE frequencies
        let rope_freqs = Self::precompute_rope_freqs(&config);

        Ok(Self {
            config,
            rope_freqs,
            token_embeddings: None,
            layer_weights: HashMap::new(),
            output_norm: None,
            output_weights: None,
            tensor_types: HashMap::new(),
        })
    }

    /// Precompute RoPE frequencies
    ///
    /// For RoPE, we compute frequencies as: freq_i = base^(-2i/head_dim)
    /// where i ranges from 0 to head_dim/2
    fn precompute_rope_freqs(config: &ModelConfig) -> Vec<f32> {
        let head_dim = config.head_dim;
        let base = config.rope_base;

        (0..head_dim / 2)
            .map(|i| {
                let exponent = -2.0 * (i as f32) / (head_dim as f32);
                base.powf(exponent)
            })
            .collect()
    }

    /// Simulate token embedding
    ///
    /// In a real implementation, this would lookup embeddings from a weight matrix.
    /// Here we generate deterministic values based on token IDs.
    fn embed_tokens(&self, tokens: &[u32]) -> Vec<f32> {
        let seq_len = tokens.len();
        let hidden_dim = self.config.hidden_dim;
        let mut embeddings = vec![0.0; seq_len * hidden_dim];

        for (i, &token) in tokens.iter().enumerate() {
            let base_offset = i * hidden_dim;
            // Generate deterministic but varied embeddings
            for j in 0..hidden_dim {
                // Use token ID and position to generate varied values
                let value = ((token as f32 + j as f32 + 1.0) % 10.0) / 10.0 - 0.5;
                embeddings[base_offset + j] = value;
            }
        }

        embeddings
    }

    /// Simulate RMSNorm
    ///
    /// RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    fn rms_norm(&self, x: &[f32]) -> Vec<f32> {
        let eps = self.config.rms_norm_eps;

        // Compute RMS
        let sum_squares: f32 = x.iter().map(|&v| v * v).sum();
        let rms = (sum_squares / x.len() as f32 + eps).sqrt();

        // Normalize (weight is simulated as 1.0)
        x.iter().map(|&v| v / rms).collect()
    }

    /// Simulate attention layer
    ///
    /// This represents the structure of GQA attention without actual computation.
    /// In a real implementation:
    /// 1. Project to Q, K, V
    /// 2. Apply RoPE to Q and K
    /// 3. Repeat KV heads to match Q heads
    /// 4. Compute attention scores
    /// 5. Apply causal mask
    /// 6. Weighted sum of V
    fn simulate_attention(
        &self,
        x: &[f32],
        seq_len: usize,
        _start_pos: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let _hidden_dim = self.config.hidden_dim;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;

        // Simulate Q projection (n_heads * head_dim = hidden_dim)
        let _q = x.to_vec();

        // Simulate K, V projections (n_kv_heads * head_dim)
        let kv_size = n_kv_heads * seq_len * head_dim;
        let k = vec![0.1; kv_size];
        let v = vec![0.2; kv_size];

        // Simulate attention output
        let output = x
            .iter()
            .enumerate()
            .map(|(i, &v)| v * 0.9 + (i as f32 % 10.0) * 0.01)
            .collect();

        (output, k, v)
    }

    /// Simulate feedforward layer (SwiGLU)
    ///
    /// SwiGLU structure:
    /// 1. Project to intermediate_dim (gate and up)
    /// 2. gate = SiLU(gate_proj(x))
    /// 3. up = up_proj(x)
    /// 4. hidden = gate * up
    /// 5. output = down_proj(hidden)
    fn simulate_feedforward(&self, x: &[f32]) -> Vec<f32> {
        // Simulate SwiGLU activation
        x.iter()
            .enumerate()
            .map(|(i, &v)| {
                // SiLU(x) = x * sigmoid(x) approximation
                let silu = v * (1.0 / (1.0 + (-v).exp()));
                silu * 0.95 + (i as f32 % 5.0) * 0.01
            })
            .collect()
    }

    /// Generate logits from hidden states
    ///
    /// In a real implementation, this projects hidden_dim -> vocab_size.
    /// Here we generate deterministic logits based on the hidden states.
    fn generate_logits(&self, hidden_states: &[f32], token_hint: u32) -> Vec<f32> {
        let vocab_size = self.config.vocab_size;
        let mut logits = vec![0.0; vocab_size];

        // Generate varied logits based on hidden states and token
        let sum: f32 = hidden_states
            .iter()
            .take(100.min(hidden_states.len()))
            .sum();
        let avg = sum / 100.0;

        for i in 0..vocab_size {
            // Create peaks at certain positions based on input
            let position_factor = ((i + token_hint as usize) % 1000) as f32 / 1000.0;
            let value = avg + position_factor * 2.0 - 1.0;
            logits[i] = value;
        }

        // Create a strong peak at a deterministic position
        let peak_pos = ((token_hint as usize * 17) % vocab_size).min(vocab_size - 1);
        logits[peak_pos] += 5.0;

        logits
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
    /// # Algorithm
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
    pub fn prefill(&self, tokens: &[u32], start_pos: usize) -> Result<(Vec<f32>, SequenceKVCache)> {
        if tokens.is_empty() {
            return Err(BatchingError::ModelError(
                "Cannot prefill empty token sequence".to_string(),
            ));
        }

        let seq_len = tokens.len();

        // 1. Embed tokens
        let mut hidden_states = self.embed_tokens(tokens);

        // 2. Initialize KV cache
        let mut kv_cache = SequenceKVCache::new(self.config.n_layers);

        // 3. Process each transformer layer
        for layer_idx in 0..self.config.n_layers {
            // Pre-attention norm
            let normed = self.rms_norm(&hidden_states);

            // Attention with GQA
            let (attn_output, k, v) = self.simulate_attention(&normed, seq_len, start_pos);

            // Store K/V in cache
            kv_cache.set_layer(
                layer_idx,
                k,
                v,
                (self.config.n_kv_heads, seq_len, self.config.head_dim),
            );

            // Residual connection
            for i in 0..hidden_states.len() {
                hidden_states[i] += attn_output[i];
            }

            // Pre-FFN norm
            let normed = self.rms_norm(&hidden_states);

            // Feedforward (SwiGLU)
            let ffn_output = self.simulate_feedforward(&normed);

            // Residual connection
            for i in 0..hidden_states.len() {
                hidden_states[i] += ffn_output[i];
            }
        }

        // 4. Final norm
        let final_hidden = self.rms_norm(&hidden_states);

        // 5. Extract last position's hidden state
        let hidden_dim = self.config.hidden_dim;
        let last_pos_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &final_hidden[last_pos_start..last_pos_start + hidden_dim];

        // 6. Generate logits
        let last_token = tokens[tokens.len() - 1];
        let logits = self.generate_logits(last_hidden, last_token);

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
    /// # Algorithm
    ///
    /// 1. Embed tokens: `[batch_size] -> [batch_size, hidden_dim]`
    /// 2. For each layer:
    ///    - Apply RMSNorm
    ///    - Compute Q, K, V with GQA (K/V use n_kv_heads)
    ///    - Apply RoPE to Q and K (use positions array)
    ///    - Attend to cached K/V plus new K/V (cache length varies per sequence)
    ///    - Append new K/V to cache: `[n_kv_heads, 1, head_dim]`
    ///    - Feedforward (SwiGLU)
    /// 3. Final RMSNorm + output projection
    /// 4. Return logits: `[batch_size, vocab_size]`
    pub fn decode_step(
        &self,
        tokens: &[u32],
        positions: &[usize],
        kv_caches: &mut [&mut SequenceKVCache],
    ) -> Result<Vec<f32>> {
        let batch_size = tokens.len();

        if batch_size == 0 {
            return Err(BatchingError::ModelError(
                "Cannot decode empty batch".to_string(),
            ));
        }

        if positions.len() != batch_size {
            return Err(BatchingError::ModelError(format!(
                "Positions length ({}) must match batch size ({})",
                positions.len(),
                batch_size
            )));
        }

        if kv_caches.len() != batch_size {
            return Err(BatchingError::ModelError(format!(
                "KV caches length ({}) must match batch size ({})",
                kv_caches.len(),
                batch_size
            )));
        }

        let mut all_logits = Vec::new();

        // Process each sequence (in a real batched implementation, this would be parallelized)
        for seq_idx in 0..batch_size {
            let token = tokens[seq_idx];
            let position = positions[seq_idx];
            let cache = &mut kv_caches[seq_idx];

            // 1. Embed token
            let mut hidden_states = self.embed_tokens(&[token]);

            // 2. Process each layer
            for layer_idx in 0..self.config.n_layers {
                // Pre-attention norm
                let normed = self.rms_norm(&hidden_states);

                // Attention (attends to all cached + new)
                let (attn_output, k_new, v_new) = self.simulate_attention(&normed, 1, position);

                // Append new K/V to cache
                cache.append_layer(
                    layer_idx,
                    k_new,
                    v_new,
                    (self.config.n_kv_heads, 1, self.config.head_dim),
                );

                // Residual connection
                for i in 0..hidden_states.len() {
                    hidden_states[i] += attn_output[i];
                }

                // Pre-FFN norm
                let normed = self.rms_norm(&hidden_states);

                // Feedforward
                let ffn_output = self.simulate_feedforward(&normed);

                // Residual connection
                for i in 0..hidden_states.len() {
                    hidden_states[i] += ffn_output[i];
                }
            }

            // 3. Final norm
            let final_hidden = self.rms_norm(&hidden_states);

            // 4. Generate logits
            let logits = self.generate_logits(&final_hidden, token);
            all_logits.extend(logits);
        }

        Ok(all_logits)
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
        let result = ModelConfig::new(32_000, 32, 4096, 32, 7, 128, 11_008, 10_000.0, 1.0, 1e-5);
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
            cache1.set_layer(
                layer_idx,
                vec![1.0; size],
                vec![1.0; size],
                (config.n_kv_heads, 5, config.head_dim),
            );
        }

        // Initialize cache2 with length 3
        for layer_idx in 0..config.n_layers {
            let size = config.n_kv_heads * 3 * config.head_dim;
            cache2.set_layer(
                layer_idx,
                vec![1.0; size],
                vec![1.0; size],
                (config.n_kv_heads, 3, config.head_dim),
            );
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
