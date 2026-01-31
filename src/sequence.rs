//! Sequence state management and lifecycle
//!
//! This module defines the Sequence structure that tracks the state of each
//! generation request throughout its lifecycle, from waiting in the queue
//! through active generation to completion.

use crate::kv_cache::SequenceKVCache;

/// Unique identifier for a sequence
pub type SeqId = u64;

/// Status of a sequence in the batching system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    /// Sequence is waiting in the queue for prefill
    Waiting,
    
    /// Sequence is actively running (being decoded)
    Running,
    
    /// Sequence has completed generation
    Completed,
}

/// A sequence represents a single generation request
///
/// Each sequence tracks its prompt tokens, generated tokens, position in the
/// sequence, KV cache, and completion status. Sequences move through the
/// lifecycle: Waiting -> Running -> Completed.
///
/// # Example
///
/// ```
/// use batching_rs::{Sequence, SequenceStatus, SequenceKVCache};
///
/// // Create a new sequence
/// let prompt = vec![1, 2, 3, 4, 5];
/// let cache = SequenceKVCache::new(32); // 32 layers
/// let mut seq = Sequence::new(0, prompt, 100, 2, cache);
///
/// assert_eq!(seq.status, SequenceStatus::Waiting);
/// assert_eq!(seq.current_length(), 5); // prompt length
/// assert!(!seq.is_finished());
/// ```
#[derive(Debug, Clone)]
pub struct Sequence {
    /// Unique identifier for this sequence
    pub id: SeqId,
    
    /// Current status of the sequence
    pub status: SequenceStatus,
    
    /// Original prompt tokens
    pub prompt_tokens: Vec<u32>,
    
    /// Tokens generated so far
    pub generated_tokens: Vec<u32>,
    
    /// Maximum number of new tokens to generate
    pub max_new_tokens: usize,
    
    /// Next position index (for RoPE)
    ///
    /// This tracks the absolute position in the sequence for positional embeddings.
    /// It equals prompt_tokens.len() + generated_tokens.len()
    pub next_pos: usize,
    
    /// Per-sequence KV cache
    pub kv_cache: SequenceKVCache,
    
    /// EOS (End of Sequence) token ID
    pub eos_token_id: u32,
}

impl Sequence {
    /// Create a new sequence
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this sequence
    /// * `prompt_tokens` - Initial prompt tokens
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `eos_token_id` - Token ID that signals end of generation
    /// * `kv_cache` - Pre-initialized KV cache (empty)
    ///
    /// # Returns
    ///
    /// A new Sequence in Waiting status with next_pos set to prompt length
    pub fn new(
        id: SeqId,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        eos_token_id: u32,
        kv_cache: SequenceKVCache,
    ) -> Self {
        let next_pos = prompt_tokens.len();
        
        Self {
            id,
            status: SequenceStatus::Waiting,
            prompt_tokens,
            generated_tokens: Vec::new(),
            max_new_tokens,
            next_pos,
            kv_cache,
            eos_token_id,
        }
    }
    
    /// Get the total current length of the sequence
    ///
    /// Returns prompt_tokens.len() + generated_tokens.len()
    pub fn current_length(&self) -> usize {
        self.prompt_tokens.len() + self.generated_tokens.len()
    }
    
    /// Get the number of tokens generated so far
    pub fn num_generated(&self) -> usize {
        self.generated_tokens.len()
    }
    
    /// Check if the sequence has finished generation
    ///
    /// A sequence is finished if:
    /// 1. The last generated token is the EOS token, OR
    /// 2. We have generated max_new_tokens
    ///
    /// # Returns
    ///
    /// `true` if generation should stop, `false` otherwise
    pub fn is_finished(&self) -> bool {
        // Check if we've hit max tokens
        if self.generated_tokens.len() >= self.max_new_tokens {
            return true;
        }
        
        // Check if last token is EOS
        if let Some(&last_token) = self.generated_tokens.last() {
            if last_token == self.eos_token_id {
                return true;
            }
        }
        
        false
    }
    
    /// Append a newly generated token to the sequence
    ///
    /// This updates:
    /// - generated_tokens (adds the new token)
    /// - next_pos (increments by 1)
    ///
    /// # Arguments
    ///
    /// * `token` - The newly generated token ID
    pub fn append_token(&mut self, token: u32) {
        self.generated_tokens.push(token);
        self.next_pos += 1;
    }
    
    /// Get the last generated token
    ///
    /// Returns None if no tokens have been generated yet.
    pub fn last_token(&self) -> Option<u32> {
        self.generated_tokens.last().copied()
    }
    
    /// Mark the sequence as running
    ///
    /// Called when the sequence moves from waiting queue to running batch
    /// after prefill is complete.
    pub fn mark_running(&mut self) {
        self.status = SequenceStatus::Running;
    }
    
    /// Mark the sequence as completed
    ///
    /// Called when generation finishes (EOS or max_tokens reached).
    pub fn mark_completed(&mut self) {
        self.status = SequenceStatus::Completed;
    }
    
    /// Get all tokens (prompt + generated)
    ///
    /// Returns a vector containing the full sequence.
    pub fn all_tokens(&self) -> Vec<u32> {
        let mut tokens = self.prompt_tokens.clone();
        tokens.extend_from_slice(&self.generated_tokens);
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_sequence(
        id: SeqId,
        prompt_len: usize,
        max_new_tokens: usize,
    ) -> Sequence {
        let prompt_tokens: Vec<u32> = (1..=prompt_len as u32).collect();
        let kv_cache = SequenceKVCache::new(32);
        Sequence::new(id, prompt_tokens, max_new_tokens, 2, kv_cache)
    }
    
    #[test]
    fn test_new_sequence() {
        let seq = create_test_sequence(0, 5, 100);
        
        assert_eq!(seq.id, 0);
        assert_eq!(seq.status, SequenceStatus::Waiting);
        assert_eq!(seq.prompt_tokens.len(), 5);
        assert_eq!(seq.generated_tokens.len(), 0);
        assert_eq!(seq.max_new_tokens, 100);
        assert_eq!(seq.next_pos, 5);
        assert_eq!(seq.eos_token_id, 2);
        assert!(!seq.is_finished());
    }
    
    #[test]
    fn test_current_length() {
        let mut seq = create_test_sequence(0, 5, 100);
        assert_eq!(seq.current_length(), 5);
        
        seq.append_token(10);
        assert_eq!(seq.current_length(), 6);
        
        seq.append_token(11);
        assert_eq!(seq.current_length(), 7);
    }
    
    #[test]
    fn test_append_token() {
        let mut seq = create_test_sequence(0, 5, 100);
        
        assert_eq!(seq.num_generated(), 0);
        assert_eq!(seq.next_pos, 5);
        
        seq.append_token(10);
        assert_eq!(seq.num_generated(), 1);
        assert_eq!(seq.next_pos, 6);
        assert_eq!(seq.last_token(), Some(10));
        
        seq.append_token(11);
        assert_eq!(seq.num_generated(), 2);
        assert_eq!(seq.next_pos, 7);
        assert_eq!(seq.last_token(), Some(11));
    }
    
    #[test]
    fn test_is_finished_eos() {
        let mut seq = create_test_sequence(0, 5, 100);
        assert!(!seq.is_finished());
        
        seq.append_token(10);
        assert!(!seq.is_finished());
        
        seq.append_token(2); // EOS token
        assert!(seq.is_finished());
    }
    
    #[test]
    fn test_is_finished_max_tokens() {
        let mut seq = create_test_sequence(0, 5, 3);
        assert!(!seq.is_finished());
        
        seq.append_token(10);
        assert!(!seq.is_finished());
        
        seq.append_token(11);
        assert!(!seq.is_finished());
        
        seq.append_token(12);
        assert!(seq.is_finished()); // Reached max_new_tokens
    }
    
    #[test]
    fn test_is_finished_empty_generated() {
        let seq = create_test_sequence(0, 5, 100);
        assert!(!seq.is_finished()); // No generated tokens yet
    }
    
    #[test]
    fn test_last_token() {
        let mut seq = create_test_sequence(0, 5, 100);
        assert_eq!(seq.last_token(), None);
        
        seq.append_token(10);
        assert_eq!(seq.last_token(), Some(10));
        
        seq.append_token(11);
        assert_eq!(seq.last_token(), Some(11));
    }
    
    #[test]
    fn test_status_transitions() {
        let mut seq = create_test_sequence(0, 5, 100);
        
        assert_eq!(seq.status, SequenceStatus::Waiting);
        
        seq.mark_running();
        assert_eq!(seq.status, SequenceStatus::Running);
        
        seq.mark_completed();
        assert_eq!(seq.status, SequenceStatus::Completed);
    }
    
    #[test]
    fn test_all_tokens() {
        let mut seq = create_test_sequence(0, 3, 100);
        
        // Initial: just prompt tokens [1, 2, 3]
        assert_eq!(seq.all_tokens(), vec![1, 2, 3]);
        
        // After generating one token
        seq.append_token(10);
        assert_eq!(seq.all_tokens(), vec![1, 2, 3, 10]);
        
        // After generating another token
        seq.append_token(11);
        assert_eq!(seq.all_tokens(), vec![1, 2, 3, 10, 11]);
    }
    
    #[test]
    fn test_sequence_lifecycle() {
        let mut seq = create_test_sequence(0, 5, 10);
        
        // 1. Created in Waiting state
        assert_eq!(seq.status, SequenceStatus::Waiting);
        assert_eq!(seq.num_generated(), 0);
        assert!(!seq.is_finished());
        
        // 2. Prefill happens (simulated), move to Running
        seq.mark_running();
        assert_eq!(seq.status, SequenceStatus::Running);
        
        // 3. Generate tokens until finished
        for i in 0..10 {
            seq.append_token(100 + i);
            assert_eq!(seq.num_generated(), (i + 1) as usize);
        }
        
        // 4. Check finished
        assert!(seq.is_finished()); // max_new_tokens reached
        
        // 5. Mark completed
        seq.mark_completed();
        assert_eq!(seq.status, SequenceStatus::Completed);
        assert_eq!(seq.all_tokens().len(), 15); // 5 prompt + 10 generated
    }
    
    #[test]
    fn test_multiple_sequences_independent() {
        let seq1 = create_test_sequence(1, 5, 100);
        let seq2 = create_test_sequence(2, 10, 50);
        
        assert_eq!(seq1.id, 1);
        assert_eq!(seq2.id, 2);
        assert_eq!(seq1.prompt_tokens.len(), 5);
        assert_eq!(seq2.prompt_tokens.len(), 10);
        assert_eq!(seq1.max_new_tokens, 100);
        assert_eq!(seq2.max_new_tokens, 50);
    }
}
