# Continuous Batching Implementation Plan for Llama Models (Rust + MLX)

## Overview

This document outlines the implementation plan for continuous batching core logic for Llama-family models (including Llama 3 with GQA) using Rust and MLX (mlx-rs).

**Scope**: Core batching logic only. No HTTP/SSE, no tokenizer, no text processing, no advanced optimizations.

---

## Module Breakdown

```
src/
├── lib.rs                 // Public API and re-exports
├── config.rs             // Model configuration (layers, heads, dims, GQA params)
├── sequence.rs           // Sequence state management and lifecycle
├── kv_cache.rs           // Per-sequence KV cache storage
├── model.rs              // Llama forward pass (prefill + decode with GQA)
├── scheduler.rs          // Continuous batching scheduler logic
└── worker.rs             // Single-threaded model worker + execution loop
```

---

## Data Structures

### 1. Model Configuration (`config.rs`)

```rust
pub struct ModelConfig {
    // Model architecture
    pub vocab_size: usize,
    pub n_layers: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,          // Number of query heads
    pub n_kv_heads: usize,       // Number of KV heads (for GQA)
    pub head_dim: usize,
    pub intermediate_dim: usize,
    
    // RoPE configuration
    pub rope_base: f32,
    pub rope_scale: f32,
    
    // Normalization
    pub rms_norm_eps: f32,
}
```

**Key features**:
- `n_kv_heads` for Grouped Query Attention (GQA) support
- Standard Llama 3 architecture parameters
- RoPE configuration for positional encoding

---

### 2. Sequence State (`sequence.rs`)

```rust
pub type SeqId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,      // In prefill queue
    Running,      // In decode set
    Completed,    // Finished (EOS or max_tokens)
}

pub struct Sequence {
    pub id: SeqId,
    pub status: SequenceStatus,
    
    // Token tracking
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    
    // Position tracking (for RoPE)
    pub next_pos: usize,
    
    // KV cache
    pub kv_cache: SequenceKVCache,
    
    // Stopping conditions
    pub eos_token_id: u32,
}
```

**Key responsibilities**:
- Track all tokens (prompt + generated)
- Maintain RoPE position index (`next_pos`)
- Store per-sequence KV cache
- Manage lifecycle status
- Check stopping conditions

**Stopping conditions**:
1. Generated token is EOS token
2. `generated_tokens.len() >= max_new_tokens`

---

### 3. KV Cache (`kv_cache.rs`)

```rust
pub struct LayerKVCache {
    // K: [n_kv_heads, seq_len, head_dim]
    pub k: Option<mlx_rs::Array>,
    
    // V: [n_kv_heads, seq_len, head_dim]
    pub v: Option<mlx_rs::Array>,
}

pub struct SequenceKVCache {
    // One LayerKVCache per layer
    pub layers: Vec<LayerKVCache>,
}
```

**Operations**:
- `new(n_layers)`: Initialize empty cache
- `append_layer(layer_idx, k, v)`: Append new K/V tensors for one layer
- `get_layer(layer_idx)`: Retrieve K/V for a layer
- `current_length()`: Get current sequence length in cache

**Shape evolution**:
- **Prefill**: Create full K/V for prompt length
  - K: `[n_kv_heads, prompt_len, head_dim]`
  - V: `[n_kv_heads, prompt_len, head_dim]`
- **Decode**: Concatenate 1-token K/V at each step
  - New K: `[n_kv_heads, 1, head_dim]`
  - Result: `[n_kv_heads, seq_len+1, head_dim]`

---

### 4. Model (`model.rs`)

```rust
pub struct LlamaModel {
    config: ModelConfig,
    // Model weights would be loaded here
    // For this implementation, we focus on structure
}

impl LlamaModel {
    /// Prefill: compute KV cache for entire prompt
    /// Returns: (logits, kv_cache)
    /// logits: [prompt_len, vocab_size] (or just last position)
    pub fn prefill(
        &self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<(mlx_rs::Array, SequenceKVCache)>;
    
    /// Decode: forward pass for batched sequences (1 token each)
    /// Returns: logits [batch_size, vocab_size]
    /// Side effect: updates each sequence's KV cache in-place
    pub fn decode_step(
        &self,
        tokens: &[u32],              // [batch_size]
        positions: &[usize],          // [batch_size]
        kv_caches: &mut [&mut SequenceKVCache],
    ) -> Result<mlx_rs::Array>;
}
```

**Attention implementation notes**:
- **GQA (Grouped Query Attention)**:
  - `n_heads` query heads
  - `n_kv_heads` key/value heads
  - Each KV head serves `n_heads / n_kv_heads` query heads
  - Repeat KV tensors: `k.repeat(n_heads / n_kv_heads, axis=1)`

- **RoPE (Rotary Position Embedding)**:
  - Applied to Q and K before attention
  - Uses `positions` array for each token's absolute position
  - Standard Llama implementation

**Prefill vs Decode differences**:
- **Prefill**: Full causal attention over prompt
  - Input: `[seq_len]` tokens
  - Attention mask: lower triangular
  - Returns all KV pairs
  
- **Decode**: Attention over full cached sequence
  - Input: `[batch_size]` tokens (one per sequence)
  - Each sequence has different cache length
  - Use existing KV cache + append new KV

---

### 5. Scheduler (`scheduler.rs`)

```rust
pub struct Scheduler {
    // Configuration
    max_batch_size: usize,        // Maximum total batch size
    max_waiting_tokens: usize,    // Maximum tokens in waiting queue
    
    // Queues
    waiting_queue: VecDeque<SeqId>,  // Sequences waiting for prefill
    running_batch: Vec<SeqId>,       // Currently running sequences
    
    // State
    sequences: HashMap<SeqId, Sequence>,
    next_seq_id: SeqId,
}
```

**Core operations**:

```rust
impl Scheduler {
    /// Add a new sequence to waiting queue
    pub fn add_sequence(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        eos_token_id: u32,
    ) -> SeqId;
    
    /// Schedule next batch operation
    /// Returns: SchedulerDecision
    pub fn schedule(&mut self) -> SchedulerDecision;
    
    /// Mark sequence as completed and remove from running batch
    pub fn complete_sequence(&mut self, seq_id: SeqId);
    
    /// Check if any work remains
    pub fn has_work(&self) -> bool;
}

pub enum SchedulerDecision {
    /// Run a mixed batch: prefill new sequences + decode running sequences
    Batch {
        prefill_seq_ids: Vec<SeqId>,  // Sequences to prefill this iteration
        decode_seq_ids: Vec<SeqId>,   // Sequences to decode (continue)
    },
    
    /// No work to do
    Idle,
}
```

**Scheduling algorithm (TGI-style)**:

```
function schedule():
    if waiting_queue is empty and running_batch is empty:
        return Idle
    
    decode_seq_ids = running_batch.clone()
    prefill_seq_ids = []
    
    current_batch_size = running_batch.len()
    
    // TGI Key Insight: Add prefill sequences to fill up the batch dynamically
    while current_batch_size < max_batch_size and waiting_queue not empty:
        seq_id = waiting_queue.pop_front()
        prefill_seq_ids.push(seq_id)
        current_batch_size += 1
    
    // Move prefilled sequences to running batch (done after execution)
    // This happens in the worker after prefill completes
    
    return Batch { prefill_seq_ids, decode_seq_ids }
```

**Key properties (TGI approach)**:
- **Dynamic batch composition**: Each iteration builds a batch that fills up to `max_batch_size`
- **Continuous batching**: Prefill and decode happen in the same forward pass when possible
- **No artificial separation**: Unlike the previous design, we don't limit to "one prefill per tick"
- **Batch filling**: Aggressively fill the batch with new sequences as space becomes available
- **Running batch**: Tracks all sequences currently in decode phase

---

### 6. Model Worker (`worker.rs`)

```rust
pub struct ModelWorker {
    model: LlamaModel,
    scheduler: Scheduler,
}

pub struct GeneratedToken {
    pub seq_id: SeqId,
    pub token_id: u32,
    pub is_finished: bool,
}
```

**Main execution loop**:

```rust
impl ModelWorker {
    /// Execute one scheduler tick
    /// Returns: Vec<GeneratedToken> for all sequences that advanced
    pub fn step(&mut self) -> Result<Vec<GeneratedToken>>;
    
    /// Run until all sequences complete
    pub fn run_until_complete(&mut self) -> Result<Vec<Vec<u32>>>;
}
```

**Step execution pseudocode (TGI-style)**:

```
function step():
    decision = scheduler.schedule()
    
    match decision:
        Idle:
            return empty
        
        Batch { prefill_seq_ids, decode_seq_ids }:
            results = []
            
            // 1. Process all prefill sequences
            for seq_id in prefill_seq_ids:
                seq = get_sequence(seq_id)
                (logits, kv_cache) = model.prefill(seq.prompt_tokens, 0)
                token = sample(logits[-1])  // Sample from last position
                
                seq.generated_tokens.push(token)
                seq.next_pos = seq.prompt_tokens.len() + 1
                seq.kv_cache = kv_cache
                seq.status = Running
                
                // Add to running batch immediately
                scheduler.add_to_running_batch(seq_id)
                
                is_finished = token == seq.eos_token_id 
                           or seq.generated_tokens.len() >= seq.max_new_tokens
                
                if is_finished:
                    scheduler.complete_sequence(seq_id)
                
                results.push(GeneratedToken { seq_id, token, is_finished })
            
            // 2. Process all decode sequences (if any)
            if decode_seq_ids not empty:
                results.extend(decode_batch(decode_seq_ids))
            
            return results

function decode_batch(seq_ids):
    // Gather inputs
    tokens = [seq.generated_tokens.last() for seq in seq_ids]
    positions = [seq.next_pos for seq in seq_ids]
    kv_caches = [&mut seq.kv_cache for seq in seq_ids]
    
    // Forward pass
    logits = model.decode_step(tokens, positions, kv_caches)
    
    // Sample and update
    results = []
    for (i, seq_id) in enumerate(seq_ids):
        token = sample(logits[i])
        seq = get_sequence(seq_id)
        seq.generated_tokens.push(token)
        seq.next_pos += 1
        
        // Check stopping conditions
        is_finished = token == seq.eos_token_id 
                   or seq.generated_tokens.len() >= seq.max_new_tokens
        
        if is_finished:
            scheduler.complete_sequence(seq_id)
        
        results.push(GeneratedToken { seq_id, token, is_finished })
    
    return results
```

**Key differences from previous design**:
- **Multiple prefills per tick**: Process all sequences that fit in the batch
- **Immediate integration**: Prefilled sequences are added to running batch right away
- **Batch-first approach**: Always try to fill the batch to maximum capacity

---

## Execution Flow Example

### Scenario: 3 sequences arrive, max_batch_size = 4

```
Initial state:
  waiting_queue: [seq1, seq2, seq3]
  running_batch: []

Tick 1:
  Schedule: Batch { prefill: [seq1, seq2, seq3], decode: [] }
  - Prefill seq1 (prompt length 10) -> generates token
  - Prefill seq2 (prompt length 5) -> generates token
  - Prefill seq3 (prompt length 8) -> generates token
  - All three moved to running_batch
  Result: seq1@pos=11, seq2@pos=6, seq3@pos=9

State after tick 1:
  waiting_queue: []
  running_batch: [seq1, seq2, seq3]

Tick 2:
  Schedule: Batch { prefill: [], decode: [seq1, seq2, seq3] }
  - Decode all three sequences (1 token each)
  Result: seq1@pos=12, seq2@pos=7, seq3@pos=10

Tick 3:
  Schedule: Batch { prefill: [], decode: [seq1, seq2, seq3] }
  - Decode all three
  - Assume seq2 hits EOS
  Result: seq1@pos=13, seq2@COMPLETED, seq3@pos=11

State after tick 3:
  waiting_queue: []
  running_batch: [seq1, seq3]  // seq2 removed

Tick 4:
  // New sequence arrives
  Add seq4 to waiting_queue
  
  Schedule: Batch { prefill: [seq4], decode: [seq1, seq3] }
  - Prefill seq4 (prompt length 15) -> generates token
  - Decode seq1 and seq3
  - seq4 moved to running_batch
  Result: seq1@pos=14, seq3@pos=12, seq4@pos=16

State after tick 4:
  waiting_queue: []
  running_batch: [seq1, seq3, seq4]

... continues until all sequences complete
```

**Key observations**:
- **Batch is filled maximally**: When seq2 completes, seq4 is immediately prefilled
- **No artificial limits**: All 3 sequences prefilled in tick 1 (not just 1)
- **Dynamic throughput**: Batch size varies as sequences complete/arrive

---

## Key Implementation Details

### GQA (Grouped Query Attention)

Llama 3 uses GQA where:
- **Query heads**: `n_heads` (e.g., 32)
- **KV heads**: `n_kv_heads` (e.g., 8)
- **Grouping**: Each KV head serves `n_heads / n_kv_heads` query heads (e.g., 4)

**Implementation**:
```rust
// After computing K and V with n_kv_heads
let n_repeats = config.n_heads / config.n_kv_heads;

// Repeat KV to match query heads
// k: [n_kv_heads, seq_len, head_dim] -> [n_heads, seq_len, head_dim]
let k_expanded = k.repeat(n_repeats, axis=1);
let v_expanded = v.repeat(n_repeats, axis=1);

// Now perform attention with expanded K, V
```

### RoPE Position Handling

- Each sequence maintains `next_pos` (absolute position index)
- **Prefill**: positions = `[0, 1, 2, ..., prompt_len-1]`
- **Decode**: positions = `[next_pos]` for each sequence
- RoPE frequencies: `freq = base^(-2i/d)` where `d = head_dim`

### KV Cache Concatenation

**MLX concatenation**:
```rust
// Append new K/V to existing cache
let new_k = /* computed from current token */;  // [n_kv_heads, 1, head_dim]
let cached_k = kv_cache.layers[layer].k.as_ref().unwrap();
let updated_k = mlx_rs::ops::concatenate(&[cached_k, &new_k], 1)?; // axis=1 (seq_len)

kv_cache.layers[layer].k = Some(updated_k);
```

### Batching Decode Step

**Challenge**: Each sequence has different KV cache length

**Solution**: 
- Each sequence maintains its own KV cache
- During attention, each sequence attends to its own cache
- Output logits are independent per sequence
- This is **not** the most memory-efficient (paged KV would be better), but it's simple and correct

---

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum BatchingError {
    #[error("MLX error: {0}")]
    MlxError(#[from] mlx_rs::error::Exception),
    
    #[error("Sequence not found: {0}")]
    SequenceNotFound(SeqId),
    
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    
    #[error("Model error: {0}")]
    ModelError(String),
}

pub type Result<T> = std::result::Result<T, BatchingError>;
```

---

## Testing Strategy

### Unit Tests

1. **Sequence lifecycle**:
   - Create sequence
   - Check stopping conditions
   - Validate token generation

2. **KV cache operations**:
   - Initialize cache
   - Append K/V tensors
   - Verify shapes

3. **Scheduler logic**:
   - Add sequences
   - Verify scheduling decisions
   - Test queue/set management

### Integration Tests

1. **Single sequence end-to-end**:
   - Add one sequence
   - Run until completion
   - Verify output tokens

2. **Multiple sequences with batching**:
   - Add multiple sequences
   - Verify interleaved execution
   - Check all sequences complete

3. **Stopping conditions**:
   - EOS token
   - Max tokens reached

---

## Example Usage

```rust
use batching_rs::{ModelConfig, ModelWorker, Scheduler};

fn main() -> anyhow::Result<()> {
    // Configure model (Llama 3 8B example)
    let config = ModelConfig {
        vocab_size: 128256,
        n_layers: 32,
        hidden_dim: 4096,
        n_heads: 32,
        n_kv_heads: 8,  // GQA
        head_dim: 128,
        intermediate_dim: 14336,
        rope_base: 500000.0,
        rope_scale: 1.0,
        rms_norm_eps: 1e-5,
    };
    
    // Create scheduler
    let scheduler = Scheduler::new(2); // DECODE_BATCH_MAX = 2
    
    // Create worker
    let mut worker = ModelWorker::new(config, scheduler)?;
    
    // Add sequences (token IDs already preprocessed)
    let seq1 = worker.add_sequence(
        vec![1, 2, 3, 4, 5],  // prompt tokens
        100,                   // max_new_tokens
        2,                     // eos_token_id
    );
    
    let seq2 = worker.add_sequence(
        vec![1, 10, 20, 30],
        50,
        2,
    );
    
    // Run until all complete
    let results = worker.run_until_complete()?;
    
    println!("Sequence 1 generated: {:?}", results[0]);
    println!("Sequence 2 generated: {:?}", results[1]);
    
    Ok(())
}
```

---

## Implementation Phases

### Phase 1: Core Data Structures
- [ ] `config.rs`: ModelConfig
- [ ] `sequence.rs`: Sequence, SeqId, SequenceStatus
- [ ] `kv_cache.rs`: LayerKVCache, SequenceKVCache

### Phase 2: Scheduler
- [ ] `scheduler.rs`: Scheduler with prefill_queue and decode_set
- [ ] Implement scheduling algorithm
- [ ] Add sequence management

### Phase 3: Model Stub
- [ ] `model.rs`: LlamaModel structure
- [ ] Prefill method signature
- [ ] Decode step method signature
- [ ] (Actual weights/computation can be stubbed initially)

### Phase 4: Worker
- [ ] `worker.rs`: ModelWorker
- [ ] Implement step() method
- [ ] Implement run_until_complete()
- [ ] Token sampling (greedy)

### Phase 5: Model Implementation
- [ ] Implement actual Llama forward pass
- [ ] GQA attention
- [ ] RoPE embeddings
- [ ] KV cache management

### Phase 6: Testing & Validation
- [ ] Unit tests for each module
- [ ] Integration tests
- [ ] End-to-end example

---

## Performance Considerations (Future)

While not required for this implementation, these optimizations could be added later:

- **Paged KV cache**: Share memory across sequences
- **Chunked prefill**: Break long prompts into chunks
- **Dynamic batching**: Adjust batch size based on sequence lengths
- **Memory pooling**: Reuse KV cache memory
- **Prefix caching**: Share common prompt prefixes

---

## Dependencies

```toml
[dependencies]
mlx-rs = "0.18"          # MLX bindings for Rust
anyhow = "1.0"           # Error handling
thiserror = "1.0"        # Error types
```

---

## Conclusion

This implementation plan provides a minimal, correct implementation of continuous batching for Llama models. The focus is on:

1. **Correctness**: Proper KV cache management and position tracking
2. **Simplicity**: No advanced optimizations, single-threaded execution
3. **GQA support**: Handle Llama 3 grouped query attention
4. **Core batching**: Interleave prefill and decode to maximize throughput

The result is a foundation that can be extended with optimizations later, but demonstrates the essential mechanics of continuous batching.
