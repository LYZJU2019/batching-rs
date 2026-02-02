//! Model worker and execution loop
//!
//! This module implements the ModelWorker that orchestrates the scheduler and model,
//! executing the TGI-style continuous batching algorithm. The worker is single-threaded
//! and handles all prefill and decode operations.

use crate::{
    config::ModelConfig,
    model::LlamaModel,
    scheduler::{Scheduler, SchedulerDecision},
    sequence::SeqId,
    BatchingError, Result,
};

/// Result of generating one token
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedToken {
    /// Sequence ID
    pub seq_id: SeqId,

    /// Generated token ID
    pub token_id: u32,

    /// Whether this sequence has finished generation
    pub is_finished: bool,
}

/// Model worker that executes the continuous batching loop
///
/// The worker combines the scheduler and model to implement the full
/// continuous batching pipeline. It handles:
/// - Prefilling new sequences
/// - Decoding running sequences
/// - Sampling tokens from logits
/// - Updating sequence state
/// - Managing the execution loop
///
/// # Example
///
/// ```
/// use batching_rs::{ModelConfig, ModelWorker};
///
/// let config = ModelConfig::llama3_8b();
/// let mut worker = ModelWorker::new(config, 4).unwrap();
///
/// // Add sequences
/// let seq1 = worker.add_sequence(vec![1, 2, 3], 10, 2);
/// let seq2 = worker.add_sequence(vec![4, 5, 6], 10, 2);
///
/// // Run until all complete
/// let results = worker.run_until_complete().unwrap();
/// ```
pub struct ModelWorker {
    /// The model for forward passes
    model: LlamaModel,

    /// The scheduler managing batches
    scheduler: Scheduler,
}

impl ModelWorker {
    /// Create a new model worker
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    /// * `max_batch_size` - Maximum number of sequences per batch
    ///
    /// # Returns
    ///
    /// A new ModelWorker instance
    ///
    /// # Errors
    ///
    /// Returns an error if the model configuration is invalid
    pub fn new(config: ModelConfig, max_batch_size: usize) -> Result<Self> {
        let model = LlamaModel::new(config)?;
        let scheduler = Scheduler::new(max_batch_size);

        Ok(Self { model, scheduler })
    }

    /// Add a new sequence to the waiting queue
    ///
    /// # Arguments
    ///
    /// * `prompt_tokens` - Prompt tokens for the sequence
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `eos_token_id` - End-of-sequence token ID
    ///
    /// # Returns
    ///
    /// The unique SeqId assigned to this sequence
    pub fn add_sequence(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        eos_token_id: u32,
    ) -> SeqId {
        self.scheduler.add_sequence(
            prompt_tokens,
            max_new_tokens,
            eos_token_id,
            self.model.config().n_layers,
        )
    }

    /// Execute one scheduler tick
    ///
    /// This implements the core TGI continuous batching logic:
    /// 1. Get scheduling decision (which sequences to prefill/decode)
    /// 2. Prefill any new sequences
    /// 3. Decode running sequences in batch
    /// 4. Sample tokens and update sequence state
    /// 5. Remove completed sequences
    ///
    /// # Returns
    ///
    /// A vector of GeneratedToken for all sequences that advanced this tick
    ///
    /// # Errors
    ///
    /// Returns an error if model forward pass or scheduler operations fail
    pub fn step(&mut self) -> Result<Vec<GeneratedToken>> {
        let decision = self.scheduler.schedule();

        match decision {
            SchedulerDecision::Idle => {
                // No work to do
                Ok(Vec::new())
            }

            SchedulerDecision::Batch {
                prefill_seq_ids,
                decode_seq_ids,
            } => {
                let mut results = Vec::new();

                // 1. Process all prefill sequences
                for seq_id in prefill_seq_ids {
                    let token = self.process_prefill(seq_id)?;
                    results.push(token);
                }

                // 2. Process decode batch (if any)
                if !decode_seq_ids.is_empty() {
                    let tokens = self.process_decode_batch(&decode_seq_ids)?;
                    results.extend(tokens);
                }

                Ok(results)
            }
        }
    }

    /// Process prefill for a single sequence
    ///
    /// # Arguments
    ///
    /// * `seq_id` - The sequence to prefill
    ///
    /// # Returns
    ///
    /// GeneratedToken for the first token after prefill
    ///
    /// # Errors
    ///
    /// Returns an error if prefill fails or sequence not found
    fn process_prefill(&mut self, seq_id: SeqId) -> Result<GeneratedToken> {
        // Get sequence and clone data we need
        let prompt_tokens = {
            let seq = self.scheduler.get_sequence(seq_id)?;
            seq.prompt_tokens.clone()
        };

        // Run prefill
        let (logits, kv_cache) = self.model.prefill(&prompt_tokens, 0)?;

        // Sample token (greedy for now)
        let token_id = self.sample_token(&logits);

        // Update sequence
        let seq = self.scheduler.get_sequence_mut(seq_id)?;
        seq.kv_cache = kv_cache;
        seq.append_token(token_id);
        let is_finished = seq.is_finished();

        // Move to running batch
        self.scheduler.add_to_running_batch(seq_id)?;

        // Complete if finished
        if is_finished {
            self.scheduler.complete_sequence(seq_id)?;
        }

        Ok(GeneratedToken {
            seq_id,
            token_id,
            is_finished,
        })
    }

    /// Process decode batch for multiple sequences
    ///
    /// # Arguments
    ///
    /// * `seq_ids` - The sequences to decode
    ///
    /// # Returns
    ///
    /// Vector of GeneratedToken for all sequences
    ///
    /// # Errors
    ///
    /// Returns an error if decode fails or sequences not found
    fn process_decode_batch(&mut self, seq_ids: &[SeqId]) -> Result<Vec<GeneratedToken>> {
        // Gather inputs for batched decode
        let mut tokens = Vec::new();
        let mut positions = Vec::new();

        for &seq_id in seq_ids {
            let seq = self.scheduler.get_sequence(seq_id)?;

            // Last generated token (or error if none - shouldn't happen in running batch)
            let last_token = seq.last_token().ok_or_else(|| {
                BatchingError::ModelError(format!(
                    "Sequence {} in running batch has no generated tokens",
                    seq_id
                ))
            })?;

            tokens.push(last_token);
            positions.push(seq.next_pos);
        }

        // Process each sequence individually to avoid borrow checker issues
        // This is less efficient than true batching but works with our simple design
        let mut results = Vec::new();
        let vocab_size = self.model.config().vocab_size;

        for (i, &seq_id) in seq_ids.iter().enumerate() {
            // Get mutable reference to this sequence's cache
            let seq = self.scheduler.get_sequence_mut(seq_id)?;
            let cache = &mut seq.kv_cache;

            // Run decode for this sequence
            let logits = self
                .model
                .decode_step(&[tokens[i]], &[positions[i]], &mut [cache])?;

            // Extract logits (first vocab_size elements since batch_size=1)
            let seq_logits = &logits[0..vocab_size];

            // Sample token
            let token_id = self.sample_token(seq_logits);

            // Update sequence (get mutable reference again)
            let seq = self.scheduler.get_sequence_mut(seq_id)?;
            seq.append_token(token_id);

            // Check if finished
            let is_finished = seq.is_finished();
            if is_finished {
                self.scheduler.complete_sequence(seq_id)?;
            }

            results.push(GeneratedToken {
                seq_id,
                token_id,
                is_finished,
            });
        }

        Ok(results)
    }

    /// Sample a token from logits (greedy sampling)
    ///
    /// # Arguments
    ///
    /// * `logits` - Logits vector [vocab_size]
    ///
    /// # Returns
    ///
    /// The token ID with highest logit value
    ///
    /// # Note
    ///
    /// This is a simple greedy sampler. PR-010 will add temperature,
    /// top-k, and top-p sampling.
    fn sample_token(&self, logits: &[f32]) -> u32 {
        // Greedy sampling: argmax
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0)
    }

    /// Run the execution loop until all sequences complete
    ///
    /// This repeatedly calls step() until there's no more work to do.
    ///
    /// # Returns
    ///
    /// A vector of generated token sequences, one per input sequence,
    /// indexed by the order they were added.
    ///
    /// # Errors
    ///
    /// Returns an error if any step fails
    pub fn run_until_complete(&mut self) -> Result<Vec<Vec<u32>>> {
        // Track all generated tokens per sequence
        let mut all_results: Vec<Vec<GeneratedToken>> = Vec::new();

        while self.scheduler.has_work() {
            let tokens = self.step()?;

            // Group by sequence
            for token in tokens {
                // Extend the results vector if needed
                while all_results.len() <= token.seq_id as usize {
                    all_results.push(Vec::new());
                }
                all_results[token.seq_id as usize].push(token);
            }
        }

        // Extract just the token IDs for each sequence
        let generated_sequences: Vec<Vec<u32>> = all_results
            .iter()
            .map(|tokens| tokens.iter().map(|t| t.token_id).collect())
            .collect();

        Ok(generated_sequences)
    }

    /// Check if there's any work remaining
    pub fn has_work(&self) -> bool {
        self.scheduler.has_work()
    }

    /// Get the number of sequences waiting for prefill
    pub fn waiting_count(&self) -> usize {
        self.scheduler.waiting_count()
    }

    /// Get the number of sequences currently running (being decoded)
    pub fn running_count(&self) -> usize {
        self.scheduler.running_count()
    }

    /// Get the total number of sequences (including completed)
    pub fn total_sequences(&self) -> usize {
        self.scheduler.total_sequences()
    }

    /// Get the number of completed sequences
    pub fn completed_count(&self) -> usize {
        self.scheduler.completed_count()
    }

    /// Remove all completed sequences from memory
    ///
    /// Essential for long-running workers processing tens of thousands of sequences.
    /// Call this periodically (e.g., after every batch or every N sequences) to prevent
    /// memory buildup from completed sequences and their KV caches.
    ///
    /// # Returns
    ///
    /// The number of sequences removed
    ///
    /// # Example
    ///
    /// ```
    /// use batching_rs::{ModelConfig, ModelWorker};
    ///
    /// let config = ModelConfig::llama3_8b();
    /// let mut worker = ModelWorker::new(config, 4).unwrap();
    ///
    /// // Process sequences...
    /// // worker.add_sequence(...);
    /// // worker.run_until_complete().unwrap();
    ///
    /// // Clean up completed sequences
    /// let removed = worker.clear_completed_sequences();
    /// println!("Removed {} completed sequences", removed);
    /// ```
    pub fn clear_completed_sequences(&mut self) -> usize {
        self.scheduler.clear_completed_sequences()
    }

    /// Reset the worker to its initial state
    ///
    /// This clears all sequences and resets the scheduler. Use this to completely
    /// reset the worker for a fresh batch of work.
    ///
    /// # Warning
    ///
    /// This will remove all sequence data. Make sure to retrieve results before calling this.
    pub fn reset(&mut self) {
        self.scheduler.reset();
    }

    /// Get a reference to a sequence by ID
    ///
    /// Useful for inspecting sequence state and retrieving generated tokens.
    ///
    /// # Arguments
    ///
    /// * `seq_id` - The sequence ID
    ///
    /// # Errors
    ///
    /// Returns an error if the sequence is not found
    pub fn get_sequence(&self, seq_id: SeqId) -> Result<&crate::sequence::Sequence> {
        self.scheduler.get_sequence(seq_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_worker() {
        let config = ModelConfig::llama3_8b();
        let worker = ModelWorker::new(config, 4);
        assert!(worker.is_ok());
    }

    #[test]
    fn test_add_sequence() {
        let config = ModelConfig::llama3_8b();
        let mut worker = ModelWorker::new(config, 4).unwrap();

        let seq1 = worker.add_sequence(vec![1, 2, 3], 10, 2);
        assert_eq!(seq1, 0);

        let seq2 = worker.add_sequence(vec![4, 5, 6], 10, 2);
        assert_eq!(seq2, 1);

        assert!(worker.has_work());
    }

    #[test]
    fn test_step_idle() {
        let config = ModelConfig::llama3_8b();
        let mut worker = ModelWorker::new(config, 4).unwrap();

        // No sequences, should return empty
        let result = worker.step().unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_step_single_prefill() {
        let config = ModelConfig::llama3_8b();
        let mut worker = ModelWorker::new(config, 4).unwrap();

        let seq_id = worker.add_sequence(vec![1, 2, 3], 10, 2);

        // First step should prefill
        let result = worker.step().unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].seq_id, seq_id);
        assert!(!result[0].is_finished);
    }

    #[test]
    fn test_step_multiple_prefills() {
        let config = ModelConfig::llama3_8b();
        let mut worker = ModelWorker::new(config, 4).unwrap();

        worker.add_sequence(vec![1, 2, 3], 10, 2);
        worker.add_sequence(vec![4, 5, 6], 10, 2);
        worker.add_sequence(vec![7, 8, 9], 10, 2);

        // First step should prefill all 3 (batch size allows it)
        let result = worker.step().unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_step_prefill_then_decode() {
        let config = ModelConfig::llama3_8b();
        let mut worker = ModelWorker::new(config, 4).unwrap();

        worker.add_sequence(vec![1, 2, 3], 10, 2);

        // Step 1: Prefill
        let result = worker.step().unwrap();
        assert_eq!(result.len(), 1);

        // Step 2: Decode
        let result = worker.step().unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].seq_id, 0);
    }

    #[test]
    fn test_step_mixed_batch() {
        let config = ModelConfig::llama3_8b();
        let mut worker = ModelWorker::new(config, 3).unwrap();

        // Add and prefill first sequence
        worker.add_sequence(vec![1, 2, 3], 10, 2);
        worker.step().unwrap(); // Prefill seq 0

        // Add two more sequences
        worker.add_sequence(vec![4, 5, 6], 10, 2);
        worker.add_sequence(vec![7, 8, 9], 10, 2);

        // Next step should: decode seq 0 + prefill seq 1 and seq 2
        let result = worker.step().unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_sample_token_greedy() {
        let config = ModelConfig::llama3_8b();
        let worker = ModelWorker::new(config, 4).unwrap();

        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let token = worker.sample_token(&logits);
        assert_eq!(token, 3); // Index of max value (0.9)
    }

    #[test]
    fn test_run_until_complete_single_sequence() {
        let config = ModelConfig::llama3_8b();
        let mut worker = ModelWorker::new(config, 4).unwrap();

        // Add sequence that will generate 3 tokens (max_new_tokens=3)
        worker.add_sequence(vec![1, 2, 3], 3, 999); // EOS won't trigger

        let results = worker.run_until_complete().unwrap();

        // Should have 1 sequence with 3 generated tokens
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 3);
    }

    #[test]
    fn test_run_until_complete_multiple_sequences() {
        let config = ModelConfig::llama3_8b();
        let mut worker = ModelWorker::new(config, 4).unwrap();

        worker.add_sequence(vec![1, 2, 3], 2, 999);
        worker.add_sequence(vec![4, 5, 6], 3, 999);

        let results = worker.run_until_complete().unwrap();

        // Should have 2 sequences
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2); // max_new_tokens=2
        assert_eq!(results[1].len(), 3); // max_new_tokens=3
    }

    #[test]
    fn test_run_until_complete_with_eos() {
        let config = ModelConfig::llama3_8b();
        let mut worker = ModelWorker::new(config, 4).unwrap();

        // With the enhanced model, logits are generated deterministically based on token input
        // The greedy sampler will pick the highest logit position
        // We can't predict exactly which token will be EOS, so we just verify it completes
        worker.add_sequence(vec![1, 2, 3], 10, 999999); // High EOS unlikely to be generated

        let results = worker.run_until_complete().unwrap();

        // Should generate exactly max_new_tokens since EOS won't be hit
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 10);
    }

    #[test]
    fn test_batch_size_limit() {
        let config = ModelConfig::llama3_8b();
        let mut worker = ModelWorker::new(config, 2).unwrap(); // Max batch = 2

        worker.add_sequence(vec![1], 10, 999);
        worker.add_sequence(vec![2], 10, 999);
        worker.add_sequence(vec![3], 10, 999);

        // First step: prefill 2 sequences (limited by batch size)
        let result = worker.step().unwrap();
        assert_eq!(result.len(), 2);

        // Second step: decode 2 + prefill 1
        // Note: decode processes sequences one-by-one in current implementation
        let result = worker.step().unwrap();
        // Should have 3 results: 2 decodes + 1 prefill
        assert!(result.len() >= 2); // At least the 2 decodes + maybe prefill
    }

    #[test]
    fn test_clear_completed_sequences() {
        let config = ModelConfig::llama3_8b();
        let mut worker = ModelWorker::new(config, 4).unwrap();

        // Add sequences with small max_new_tokens so they complete quickly
        worker.add_sequence(vec![1, 2, 3], 2, 999);
        worker.add_sequence(vec![4, 5, 6], 2, 999);

        // Run until complete
        worker.run_until_complete().unwrap();

        // All should be completed
        assert_eq!(worker.completed_count(), 2);
        assert_eq!(worker.total_sequences(), 2);

        // Clear completed sequences
        let removed = worker.clear_completed_sequences();
        assert_eq!(removed, 2);
        assert_eq!(worker.completed_count(), 0);
        assert_eq!(worker.total_sequences(), 0);
    }

    #[test]
    fn test_worker_reset() {
        let config = ModelConfig::llama3_8b();
        let mut worker = ModelWorker::new(config, 4).unwrap();

        // Add and process sequences
        worker.add_sequence(vec![1, 2, 3], 2, 999);
        worker.add_sequence(vec![4, 5, 6], 3, 999);
        worker.step().unwrap();

        assert!(worker.has_work());
        assert_eq!(worker.total_sequences(), 2);

        // Reset everything
        worker.reset();

        assert!(!worker.has_work());
        assert_eq!(worker.total_sequences(), 0);
        assert_eq!(worker.waiting_count(), 0);
        assert_eq!(worker.running_count(), 0);
        assert_eq!(worker.completed_count(), 0);

        // Should be able to add new sequences after reset
        let new_seq = worker.add_sequence(vec![100], 1, 999);
        assert_eq!(new_seq, 0); // IDs restart from 0
    }

    #[test]
    fn test_high_volume_with_cleanup() {
        let config = ModelConfig::llama3_8b();
        let mut worker = ModelWorker::new(config, 4).unwrap();

        // Simulate processing batches with periodic cleanup
        for batch in 0..10 {
            // Add 10 sequences per batch
            for i in 0..10 {
                worker.add_sequence(vec![batch * 10 + i], 1, 999);
            }

            // Process the batch
            worker.run_until_complete().unwrap();

            // Verify all completed
            assert_eq!(worker.completed_count(), 10);

            // Clean up after each batch
            let removed = worker.clear_completed_sequences();
            assert_eq!(removed, 10);
            assert_eq!(worker.total_sequences(), 0);
        }

        // After 10 batches of 10 sequences each (100 total), memory should be clean
        assert_eq!(worker.total_sequences(), 0);
        assert_eq!(worker.completed_count(), 0);
    }

    #[test]
    fn test_get_sequence() {
        let config = ModelConfig::llama3_8b();
        let mut worker = ModelWorker::new(config, 4).unwrap();

        let seq_id = worker.add_sequence(vec![1, 2, 3], 5, 999);

        // Should be able to get sequence
        let seq = worker.get_sequence(seq_id).unwrap();
        assert_eq!(seq.id, seq_id);
        assert_eq!(seq.prompt_tokens, vec![1, 2, 3]);

        // After prefill, should have generated tokens
        worker.step().unwrap();
        let seq = worker.get_sequence(seq_id).unwrap();
        assert_eq!(seq.generated_tokens.len(), 1);

        // Invalid seq_id should error
        assert!(worker.get_sequence(999).is_err());
    }

    #[test]
    fn test_worker_counters() {
        let config = ModelConfig::llama3_8b();
        let mut worker = ModelWorker::new(config, 4).unwrap();

        // Initially empty
        assert_eq!(worker.waiting_count(), 0);
        assert_eq!(worker.running_count(), 0);
        assert_eq!(worker.completed_count(), 0);
        assert_eq!(worker.total_sequences(), 0);

        // Add 3 sequences
        worker.add_sequence(vec![1], 2, 999);
        worker.add_sequence(vec![2], 2, 999);
        worker.add_sequence(vec![3], 2, 999);

        assert_eq!(worker.waiting_count(), 3);
        assert_eq!(worker.total_sequences(), 3);

        // Prefill all 3
        worker.step().unwrap();

        assert_eq!(worker.waiting_count(), 0);
        assert_eq!(worker.running_count(), 3);
        assert_eq!(worker.total_sequences(), 3);

        // Complete all
        worker.run_until_complete().unwrap();

        assert_eq!(worker.running_count(), 0);
        assert_eq!(worker.completed_count(), 3);
        assert_eq!(worker.total_sequences(), 3);
    }
}
