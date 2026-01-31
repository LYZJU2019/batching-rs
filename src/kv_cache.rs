//! KV cache data structures for per-sequence storage
//!
//! This module implements the key-value cache that stores attention keys and values
//! for each sequence. Each sequence maintains its own independent cache that grows
//! as tokens are generated.
//!
//! The cache is organized as a vector of LayerKVCache (one per transformer layer),
//! where each layer stores K and V tensors with shape [n_kv_heads, seq_len, head_dim].

/// KV cache for a single transformer layer
///
/// Stores the key and value tensors for one layer of the model.
/// During generation, these tensors grow along the sequence dimension.
///
/// # Shape
/// - K: `[n_kv_heads, seq_len, head_dim]`
/// - V: `[n_kv_heads, seq_len, head_dim]`
///
/// where:
/// - `n_kv_heads`: Number of key/value heads (for GQA)
/// - `seq_len`: Current sequence length (grows during generation)
/// - `head_dim`: Dimension of each attention head
#[derive(Debug, Clone)]
pub struct LayerKVCache {
    /// Key tensor: [n_kv_heads, seq_len, head_dim]
    pub k: Option<Vec<f32>>,
    
    /// Value tensor: [n_kv_heads, seq_len, head_dim]
    pub v: Option<Vec<f32>>,
    
    /// Shape metadata: (n_kv_heads, seq_len, head_dim)
    pub shape: Option<(usize, usize, usize)>,
}

impl LayerKVCache {
    /// Create a new empty layer KV cache
    pub fn new() -> Self {
        Self {
            k: None,
            v: None,
            shape: None,
        }
    }
    
    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.k.is_none() && self.v.is_none()
    }
    
    /// Get the current sequence length stored in this cache
    ///
    /// Returns 0 if cache is empty.
    pub fn seq_len(&self) -> usize {
        self.shape.map(|(_, seq_len, _)| seq_len).unwrap_or(0)
    }
}

impl Default for LayerKVCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-sequence KV cache storing keys and values for all layers
///
/// This structure maintains the KV cache for a single sequence across all
/// transformer layers. Each layer has its own LayerKVCache.
///
/// # Usage
///
/// ```
/// use batching_rs::SequenceKVCache;
///
/// // Create cache for a model with 32 layers
/// let mut cache = SequenceKVCache::new(32);
/// assert_eq!(cache.num_layers(), 32);
/// assert_eq!(cache.current_length(), 0);
/// ```
#[derive(Debug, Clone)]
pub struct SequenceKVCache {
    /// Vector of layer caches (one per transformer layer)
    pub layers: Vec<LayerKVCache>,
}

impl SequenceKVCache {
    /// Create a new empty KV cache for a model with `n_layers` layers
    ///
    /// # Arguments
    ///
    /// * `n_layers` - Number of transformer layers in the model
    ///
    /// # Returns
    ///
    /// A new SequenceKVCache with empty LayerKVCache for each layer
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers: vec![LayerKVCache::new(); n_layers],
        }
    }
    
    /// Get the number of layers in this cache
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
    
    /// Set the K and V tensors for a specific layer
    ///
    /// This is typically used during the prefill phase to initialize the cache
    /// with the full prompt's KV pairs.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Index of the layer (0 to n_layers-1)
    /// * `k` - Key tensor data (flattened)
    /// * `v` - Value tensor data (flattened)
    /// * `shape` - Shape tuple (n_kv_heads, seq_len, head_dim)
    ///
    /// # Panics
    ///
    /// Panics if layer_idx is out of bounds
    pub fn set_layer(
        &mut self,
        layer_idx: usize,
        k: Vec<f32>,
        v: Vec<f32>,
        shape: (usize, usize, usize),
    ) {
        assert!(layer_idx < self.layers.len(), "Layer index out of bounds");
        
        let (n_kv_heads, seq_len, head_dim) = shape;
        let expected_len = n_kv_heads * seq_len * head_dim;
        
        assert_eq!(
            k.len(),
            expected_len,
            "K tensor size mismatch: expected {}, got {}",
            expected_len,
            k.len()
        );
        assert_eq!(
            v.len(),
            expected_len,
            "V tensor size mismatch: expected {}, got {}",
            expected_len,
            v.len()
        );
        
        self.layers[layer_idx].k = Some(k);
        self.layers[layer_idx].v = Some(v);
        self.layers[layer_idx].shape = Some(shape);
    }
    
    /// Get a reference to the K and V tensors for a specific layer
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Index of the layer (0 to n_layers-1)
    ///
    /// # Returns
    ///
    /// A reference to the LayerKVCache for the specified layer
    ///
    /// # Panics
    ///
    /// Panics if layer_idx is out of bounds
    pub fn get_layer(&self, layer_idx: usize) -> &LayerKVCache {
        assert!(layer_idx < self.layers.len(), "Layer index out of bounds");
        &self.layers[layer_idx]
    }
    
    /// Get a mutable reference to the K and V tensors for a specific layer
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Index of the layer (0 to n_layers-1)
    ///
    /// # Returns
    ///
    /// A mutable reference to the LayerKVCache for the specified layer
    ///
    /// # Panics
    ///
    /// Panics if layer_idx is out of bounds
    pub fn get_layer_mut(&mut self, layer_idx: usize) -> &mut LayerKVCache {
        assert!(layer_idx < self.layers.len(), "Layer index out of bounds");
        &mut self.layers[layer_idx]
    }
    
    /// Append new K and V tensors to a specific layer
    ///
    /// This is used during the decode phase to append a single token's KV pairs
    /// to the existing cache. The new tensors are concatenated along the sequence
    /// dimension.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Index of the layer (0 to n_layers-1)
    /// * `new_k` - New key tensor to append (flattened)
    /// * `new_v` - New value tensor to append (flattened)
    /// * `new_shape` - Shape of new tensors (n_kv_heads, new_seq_len, head_dim)
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - layer_idx is out of bounds
    /// - The cache is empty (use set_layer for initialization)
    /// - Shape dimensions don't match (n_kv_heads and head_dim must match)
    pub fn append_layer(
        &mut self,
        layer_idx: usize,
        new_k: Vec<f32>,
        new_v: Vec<f32>,
        new_shape: (usize, usize, usize),
    ) {
        assert!(layer_idx < self.layers.len(), "Layer index out of bounds");
        
        let layer = &mut self.layers[layer_idx];
        
        // If cache is empty, just set it
        if layer.is_empty() {
            self.set_layer(layer_idx, new_k, new_v, new_shape);
            return;
        }
        
        let (n_kv_heads, new_seq_len, head_dim) = new_shape;
        let (cached_n_kv_heads, cached_seq_len, cached_head_dim) = 
            layer.shape.expect("Cache shape should be set");
        
        // Validate shape compatibility
        assert_eq!(
            n_kv_heads, cached_n_kv_heads,
            "n_kv_heads mismatch: cached={}, new={}",
            cached_n_kv_heads, n_kv_heads
        );
        assert_eq!(
            head_dim, cached_head_dim,
            "head_dim mismatch: cached={}, new={}",
            cached_head_dim, head_dim
        );
        
        // Concatenate along sequence dimension
        let mut cached_k = layer.k.take().expect("K tensor should be present");
        let mut cached_v = layer.v.take().expect("V tensor should be present");
        
        cached_k.extend(new_k);
        cached_v.extend(new_v);
        
        let new_total_seq_len = cached_seq_len + new_seq_len;
        
        layer.k = Some(cached_k);
        layer.v = Some(cached_v);
        layer.shape = Some((n_kv_heads, new_total_seq_len, head_dim));
    }
    
    /// Get the current sequence length stored in this cache
    ///
    /// Returns 0 if cache is empty, otherwise returns the sequence length
    /// from the first layer (all layers should have the same sequence length).
    pub fn current_length(&self) -> usize {
        if self.layers.is_empty() {
            return 0;
        }
        self.layers[0].seq_len()
    }
    
    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty() || self.layers[0].is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_layer_kv_cache_new() {
        let cache = LayerKVCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);
    }
    
    #[test]
    fn test_sequence_kv_cache_new() {
        let cache = SequenceKVCache::new(32);
        assert_eq!(cache.num_layers(), 32);
        assert_eq!(cache.current_length(), 0);
        assert!(cache.is_empty());
    }
    
    #[test]
    fn test_set_layer() {
        let mut cache = SequenceKVCache::new(4);
        
        // Create dummy K and V tensors
        // Shape: [n_kv_heads=2, seq_len=3, head_dim=4]
        let n_kv_heads = 2;
        let seq_len = 3;
        let head_dim = 4;
        let size = n_kv_heads * seq_len * head_dim;
        
        let k = vec![1.0; size];
        let v = vec![2.0; size];
        
        cache.set_layer(0, k.clone(), v.clone(), (n_kv_heads, seq_len, head_dim));
        
        assert_eq!(cache.current_length(), seq_len);
        assert!(!cache.is_empty());
        
        let layer = cache.get_layer(0);
        assert_eq!(layer.seq_len(), seq_len);
        assert_eq!(layer.shape, Some((n_kv_heads, seq_len, head_dim)));
    }
    
    #[test]
    fn test_append_layer() {
        let mut cache = SequenceKVCache::new(2);
        
        let n_kv_heads = 2;
        let head_dim = 4;
        
        // Initial cache: seq_len = 3
        let initial_seq_len = 3;
        let initial_size = n_kv_heads * initial_seq_len * head_dim;
        let k1 = vec![1.0; initial_size];
        let v1 = vec![2.0; initial_size];
        
        cache.set_layer(0, k1, v1, (n_kv_heads, initial_seq_len, head_dim));
        assert_eq!(cache.current_length(), 3);
        
        // Append: seq_len = 1 (single token)
        let append_seq_len = 1;
        let append_size = n_kv_heads * append_seq_len * head_dim;
        let k2 = vec![3.0; append_size];
        let v2 = vec![4.0; append_size];
        
        cache.append_layer(0, k2, v2, (n_kv_heads, append_seq_len, head_dim));
        assert_eq!(cache.current_length(), 4);
        
        let layer = cache.get_layer(0);
        assert_eq!(layer.seq_len(), 4);
        assert_eq!(layer.shape, Some((n_kv_heads, 4, head_dim)));
    }
    
    #[test]
    fn test_append_to_empty_cache() {
        let mut cache = SequenceKVCache::new(2);
        
        let n_kv_heads = 2;
        let seq_len = 1;
        let head_dim = 4;
        let size = n_kv_heads * seq_len * head_dim;
        
        let k = vec![1.0; size];
        let v = vec![2.0; size];
        
        // Appending to empty cache should work like set_layer
        cache.append_layer(0, k, v, (n_kv_heads, seq_len, head_dim));
        
        assert_eq!(cache.current_length(), seq_len);
        assert!(!cache.is_empty());
    }
    
    #[test]
    #[should_panic(expected = "n_kv_heads mismatch")]
    fn test_append_shape_mismatch_heads() {
        let mut cache = SequenceKVCache::new(1);
        
        // Initial cache with n_kv_heads=2
        cache.set_layer(0, vec![1.0; 24], vec![2.0; 24], (2, 3, 4));
        
        // Try to append with n_kv_heads=3 (should panic)
        cache.append_layer(0, vec![1.0; 12], vec![2.0; 12], (3, 1, 4));
    }
    
    #[test]
    #[should_panic(expected = "head_dim mismatch")]
    fn test_append_shape_mismatch_dim() {
        let mut cache = SequenceKVCache::new(1);
        
        // Initial cache with head_dim=4
        cache.set_layer(0, vec![1.0; 24], vec![2.0; 24], (2, 3, 4));
        
        // Try to append with head_dim=8 (should panic)
        cache.append_layer(0, vec![1.0; 16], vec![2.0; 16], (2, 1, 8));
    }
    
    #[test]
    fn test_multiple_layers() {
        let mut cache = SequenceKVCache::new(3);
        
        let n_kv_heads = 2;
        let seq_len = 5;
        let head_dim = 4;
        let size = n_kv_heads * seq_len * head_dim;
        
        // Set different layers
        for layer_idx in 0..3 {
            let k = vec![layer_idx as f32; size];
            let v = vec![(layer_idx + 10) as f32; size];
            cache.set_layer(layer_idx, k, v, (n_kv_heads, seq_len, head_dim));
        }
        
        // Verify each layer
        for layer_idx in 0..3 {
            let layer = cache.get_layer(layer_idx);
            assert_eq!(layer.seq_len(), seq_len);
            assert!(!layer.is_empty());
        }
        
        assert_eq!(cache.current_length(), seq_len);
    }
    
    #[test]
    #[should_panic(expected = "Layer index out of bounds")]
    fn test_out_of_bounds_access() {
        let cache = SequenceKVCache::new(4);
        cache.get_layer(4); // Should panic
    }
    
    #[test]
    #[should_panic(expected = "K tensor size mismatch")]
    fn test_set_layer_wrong_size() {
        let mut cache = SequenceKVCache::new(1);
        
        // Declare shape (2, 3, 4) but provide wrong size
        let k = vec![1.0; 10]; // Should be 24
        let v = vec![2.0; 24];
        
        cache.set_layer(0, k, v, (2, 3, 4));
    }
}
