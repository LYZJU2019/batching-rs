//! TGI-style continuous batching scheduler
//!
//! This module implements the core scheduling logic that dynamically composes
//! batches by filling available slots with prefills while maintaining all
//! running sequences for decode. This follows the Text Generation Inference (TGI)
//! continuous batching approach.

use std::collections::{HashMap, VecDeque};

use crate::{
    kv_cache::SequenceKVCache,
    sequence::{SeqId, Sequence},
    BatchingError, Result,
};

/// Decision made by the scheduler for the next execution step
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerDecision {
    /// Run a mixed batch: prefill new sequences + decode running sequences
    Batch {
        /// Sequences to prefill this iteration (from waiting queue)
        prefill_seq_ids: Vec<SeqId>,

        /// Sequences to decode (continue generation)
        decode_seq_ids: Vec<SeqId>,
    },

    /// No work to do
    Idle,
}

/// TGI-style continuous batching scheduler
///
/// The scheduler manages two main data structures:
/// - `waiting_queue`: FIFO queue of sequences waiting for prefill
/// - `running_batch`: Vector of sequences currently generating tokens
///
/// Each scheduling tick, the scheduler:
/// 1. Takes all running sequences for decode
/// 2. Fills remaining batch slots with prefills from waiting queue
/// 3. Returns a batch decision for the worker to execute
///
/// # Example
///
/// ```
/// use batching_rs::{Scheduler, SequenceKVCache};
///
/// let mut scheduler = Scheduler::new(4); // max_batch_size = 4
///
/// // Add sequences
/// let seq1 = scheduler.add_sequence(vec![1, 2, 3], 10, 2, 32);
/// let seq2 = scheduler.add_sequence(vec![4, 5, 6], 10, 2, 32);
///
/// // Schedule
/// let decision = scheduler.schedule();
/// ```
pub struct Scheduler {
    /// Maximum number of sequences in a batch
    max_batch_size: usize,

    /// Sequences waiting for prefill (FIFO queue)
    waiting_queue: VecDeque<SeqId>,

    /// Sequences currently running (being decoded)
    running_batch: Vec<SeqId>,

    /// All sequences (waiting, running, and completed)
    sequences: HashMap<SeqId, Sequence>,

    /// Next sequence ID to assign
    next_seq_id: SeqId,
}

impl Scheduler {
    /// Create a new scheduler
    ///
    /// # Arguments
    ///
    /// * `max_batch_size` - Maximum number of sequences to process in one batch
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            max_batch_size,
            waiting_queue: VecDeque::new(),
            running_batch: Vec::new(),
            sequences: HashMap::new(),
            next_seq_id: 0,
        }
    }

    /// Add a new sequence to the waiting queue
    ///
    /// # Arguments
    ///
    /// * `prompt_tokens` - Initial prompt tokens
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `eos_token_id` - Token ID that signals end of generation
    /// * `n_layers` - Number of layers in the model (for KV cache initialization)
    ///
    /// # Returns
    ///
    /// The unique SeqId assigned to this sequence
    pub fn add_sequence(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        eos_token_id: u32,
        n_layers: usize,
    ) -> SeqId {
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        let kv_cache = SequenceKVCache::new(n_layers);
        let sequence = Sequence::new(
            seq_id,
            prompt_tokens,
            max_new_tokens,
            eos_token_id,
            kv_cache,
        );

        self.sequences.insert(seq_id, sequence);
        self.waiting_queue.push_back(seq_id);

        seq_id
    }

    /// Schedule the next batch operation
    ///
    /// This implements the TGI continuous batching algorithm:
    /// 1. Include all running sequences for decode
    /// 2. Fill remaining slots with prefills from waiting queue
    ///
    /// # Returns
    ///
    /// SchedulerDecision indicating what to execute next
    pub fn schedule(&self) -> SchedulerDecision {
        // Check if there's any work to do
        if self.waiting_queue.is_empty() && self.running_batch.is_empty() {
            return SchedulerDecision::Idle;
        }

        // All running sequences will be decoded
        let decode_seq_ids = self.running_batch.clone();

        // Calculate how many prefills we can fit
        let current_batch_size = self.running_batch.len();
        let available_slots = self.max_batch_size.saturating_sub(current_batch_size);

        // Fill available slots with prefills from waiting queue
        let num_prefills = available_slots.min(self.waiting_queue.len());
        let prefill_seq_ids: Vec<SeqId> = self
            .waiting_queue
            .iter()
            .take(num_prefills)
            .copied()
            .collect();

        // Return batch decision
        SchedulerDecision::Batch {
            prefill_seq_ids,
            decode_seq_ids,
        }
    }

    /// Move a sequence from waiting queue to running batch
    ///
    /// Called after prefill completes to transition the sequence to running state.
    ///
    /// # Arguments
    ///
    /// * `seq_id` - The sequence ID to move
    ///
    /// # Errors
    ///
    /// Returns an error if the sequence is not found or not in waiting queue
    pub fn add_to_running_batch(&mut self, seq_id: SeqId) -> Result<()> {
        // Remove from waiting queue
        let position = self.waiting_queue.iter().position(|&id| id == seq_id);
        if position.is_none() {
            return Err(BatchingError::SchedulerError(format!(
                "Sequence {} not found in waiting queue",
                seq_id
            )));
        }
        self.waiting_queue.remove(position.unwrap());

        // Update sequence status
        let sequence = self
            .sequences
            .get_mut(&seq_id)
            .ok_or_else(|| BatchingError::SequenceNotFound(seq_id))?;
        sequence.mark_running();

        // Add to running batch
        self.running_batch.push(seq_id);

        Ok(())
    }

    /// Mark a sequence as completed and remove it from running batch
    ///
    /// # Arguments
    ///
    /// * `seq_id` - The sequence ID to complete
    ///
    /// # Errors
    ///
    /// Returns an error if the sequence is not found
    pub fn complete_sequence(&mut self, seq_id: SeqId) -> Result<()> {
        // Remove from running batch
        self.running_batch.retain(|&id| id != seq_id);

        // Update sequence status
        let sequence = self
            .sequences
            .get_mut(&seq_id)
            .ok_or_else(|| BatchingError::SequenceNotFound(seq_id))?;
        sequence.mark_completed();

        Ok(())
    }

    /// Check if there's any work remaining
    ///
    /// Returns true if there are sequences waiting or running
    pub fn has_work(&self) -> bool {
        !self.waiting_queue.is_empty() || !self.running_batch.is_empty()
    }

    /// Get a reference to a sequence
    ///
    /// # Arguments
    ///
    /// * `seq_id` - The sequence ID
    ///
    /// # Errors
    ///
    /// Returns an error if the sequence is not found
    pub fn get_sequence(&self, seq_id: SeqId) -> Result<&Sequence> {
        self.sequences
            .get(&seq_id)
            .ok_or_else(|| BatchingError::SequenceNotFound(seq_id))
    }

    /// Get a mutable reference to a sequence
    ///
    /// # Arguments
    ///
    /// * `seq_id` - The sequence ID
    ///
    /// # Errors
    ///
    /// Returns an error if the sequence is not found
    pub fn get_sequence_mut(&mut self, seq_id: SeqId) -> Result<&mut Sequence> {
        self.sequences
            .get_mut(&seq_id)
            .ok_or_else(|| BatchingError::SequenceNotFound(seq_id))
    }

    /// Get the number of sequences in the waiting queue
    pub fn waiting_count(&self) -> usize {
        self.waiting_queue.len()
    }

    /// Get the number of sequences in the running batch
    pub fn running_count(&self) -> usize {
        self.running_batch.len()
    }

    /// Get the total number of sequences (including completed)
    pub fn total_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Get the number of completed sequences
    pub fn completed_count(&self) -> usize {
        self.sequences
            .iter()
            .filter(|(_, seq)| seq.status == crate::sequence::SequenceStatus::Completed)
            .count()
    }

    /// Remove all completed sequences from memory
    ///
    /// This is essential for long-running workers processing thousands of sequences.
    /// Call this periodically to prevent memory buildup from completed sequences.
    ///
    /// # Returns
    ///
    /// The number of sequences removed
    pub fn clear_completed_sequences(&mut self) -> usize {
        let initial_count = self.sequences.len();
        self.sequences
            .retain(|_, seq| seq.status != crate::sequence::SequenceStatus::Completed);
        initial_count - self.sequences.len()
    }

    /// Reset the scheduler to its initial state
    ///
    /// This clears all sequences (waiting, running, and completed) and resets
    /// the sequence ID counter. Use this to completely reset the scheduler
    /// for a fresh batch of work.
    ///
    /// # Warning
    ///
    /// This will remove all sequence data. Make sure to retrieve results before calling this.
    pub fn reset(&mut self) {
        self.waiting_queue.clear();
        self.running_batch.clear();
        self.sequences.clear();
        self.next_seq_id = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SequenceStatus;

    #[test]
    fn test_new_scheduler() {
        let scheduler = Scheduler::new(4);
        assert_eq!(scheduler.max_batch_size, 4);
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(scheduler.running_count(), 0);
        assert!(!scheduler.has_work());
    }

    #[test]
    fn test_add_sequence() {
        let mut scheduler = Scheduler::new(4);

        let seq1 = scheduler.add_sequence(vec![1, 2, 3], 10, 2, 32);
        assert_eq!(seq1, 0);
        assert_eq!(scheduler.waiting_count(), 1);
        assert!(scheduler.has_work());

        let seq2 = scheduler.add_sequence(vec![4, 5, 6], 20, 2, 32);
        assert_eq!(seq2, 1);
        assert_eq!(scheduler.waiting_count(), 2);

        // Verify sequence was created correctly
        let sequence = scheduler.get_sequence(seq1).unwrap();
        assert_eq!(sequence.id, seq1);
        assert_eq!(sequence.status, SequenceStatus::Waiting);
        assert_eq!(sequence.prompt_tokens, vec![1, 2, 3]);
        assert_eq!(sequence.max_new_tokens, 10);
    }

    #[test]
    fn test_schedule_idle() {
        let scheduler = Scheduler::new(4);
        let decision = scheduler.schedule();
        assert_eq!(decision, SchedulerDecision::Idle);
    }

    #[test]
    fn test_schedule_prefill_only() {
        let mut scheduler = Scheduler::new(4);

        scheduler.add_sequence(vec![1, 2, 3], 10, 2, 32);
        scheduler.add_sequence(vec![4, 5, 6], 10, 2, 32);

        let decision = scheduler.schedule();
        match decision {
            SchedulerDecision::Batch {
                prefill_seq_ids,
                decode_seq_ids,
            } => {
                assert_eq!(prefill_seq_ids.len(), 2);
                assert_eq!(decode_seq_ids.len(), 0);
                assert_eq!(prefill_seq_ids, vec![0, 1]);
            }
            _ => panic!("Expected Batch decision"),
        }
    }

    #[test]
    fn test_schedule_respects_max_batch_size() {
        let mut scheduler = Scheduler::new(2);

        // Add 3 sequences, but max_batch_size is 2
        scheduler.add_sequence(vec![1], 10, 2, 32);
        scheduler.add_sequence(vec![2], 10, 2, 32);
        scheduler.add_sequence(vec![3], 10, 2, 32);

        let decision = scheduler.schedule();
        match decision {
            SchedulerDecision::Batch {
                prefill_seq_ids,
                decode_seq_ids,
            } => {
                assert_eq!(prefill_seq_ids.len(), 2); // Limited by max_batch_size
                assert_eq!(decode_seq_ids.len(), 0);
            }
            _ => panic!("Expected Batch decision"),
        }
    }

    #[test]
    fn test_add_to_running_batch() {
        let mut scheduler = Scheduler::new(4);

        let seq_id = scheduler.add_sequence(vec![1, 2, 3], 10, 2, 32);
        assert_eq!(scheduler.waiting_count(), 1);
        assert_eq!(scheduler.running_count(), 0);

        scheduler.add_to_running_batch(seq_id).unwrap();
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(scheduler.running_count(), 1);

        let sequence = scheduler.get_sequence(seq_id).unwrap();
        assert_eq!(sequence.status, SequenceStatus::Running);
    }

    #[test]
    fn test_complete_sequence() {
        let mut scheduler = Scheduler::new(4);

        let seq_id = scheduler.add_sequence(vec![1, 2, 3], 10, 2, 32);
        scheduler.add_to_running_batch(seq_id).unwrap();
        assert_eq!(scheduler.running_count(), 1);

        scheduler.complete_sequence(seq_id).unwrap();
        assert_eq!(scheduler.running_count(), 0);

        let sequence = scheduler.get_sequence(seq_id).unwrap();
        assert_eq!(sequence.status, SequenceStatus::Completed);
    }

    #[test]
    fn test_schedule_mixed_batch() {
        let mut scheduler = Scheduler::new(4);

        // Add 2 sequences and move them to running
        let seq1 = scheduler.add_sequence(vec![1], 10, 2, 32);
        let seq2 = scheduler.add_sequence(vec![2], 10, 2, 32);
        scheduler.add_to_running_batch(seq1).unwrap();
        scheduler.add_to_running_batch(seq2).unwrap();

        // Add 3 more sequences to waiting queue
        let seq3 = scheduler.add_sequence(vec![3], 10, 2, 32);
        let seq4 = scheduler.add_sequence(vec![4], 10, 2, 32);
        let _seq5 = scheduler.add_sequence(vec![5], 10, 2, 32);

        // Schedule: should include 2 running + 2 prefills (limited by max_batch_size)
        let decision = scheduler.schedule();
        match decision {
            SchedulerDecision::Batch {
                prefill_seq_ids,
                decode_seq_ids,
            } => {
                assert_eq!(decode_seq_ids.len(), 2);
                assert_eq!(decode_seq_ids, vec![seq1, seq2]);
                assert_eq!(prefill_seq_ids.len(), 2); // 4 - 2 = 2 available slots
                assert_eq!(prefill_seq_ids, vec![seq3, seq4]);
            }
            _ => panic!("Expected Batch decision"),
        }
    }

    #[test]
    fn test_schedule_decode_only() {
        let mut scheduler = Scheduler::new(2);

        // Add 2 sequences and move to running (batch is full)
        let seq1 = scheduler.add_sequence(vec![1], 10, 2, 32);
        let seq2 = scheduler.add_sequence(vec![2], 10, 2, 32);
        scheduler.add_to_running_batch(seq1).unwrap();
        scheduler.add_to_running_batch(seq2).unwrap();

        // Add one more to waiting queue
        scheduler.add_sequence(vec![3], 10, 2, 32);

        // Schedule: batch is full, so only decode
        let decision = scheduler.schedule();
        match decision {
            SchedulerDecision::Batch {
                prefill_seq_ids,
                decode_seq_ids,
            } => {
                assert_eq!(decode_seq_ids.len(), 2);
                assert_eq!(prefill_seq_ids.len(), 0); // No room for prefills
            }
            _ => panic!("Expected Batch decision"),
        }
    }

    #[test]
    fn test_sequence_lifecycle_full() {
        let mut scheduler = Scheduler::new(4);

        // 1. Add sequence (Waiting)
        let seq_id = scheduler.add_sequence(vec![1, 2, 3], 3, 2, 32);
        assert_eq!(scheduler.waiting_count(), 1);
        assert!(scheduler.has_work());

        // 2. Schedule prefill
        let decision = scheduler.schedule();
        match decision {
            SchedulerDecision::Batch {
                prefill_seq_ids, ..
            } => {
                assert_eq!(prefill_seq_ids, vec![seq_id]);
            }
            _ => panic!("Expected Batch decision"),
        }

        // 3. Move to running (simulating prefill completion)
        scheduler.add_to_running_batch(seq_id).unwrap();
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(scheduler.running_count(), 1);

        // 4. Generate tokens
        {
            let seq = scheduler.get_sequence_mut(seq_id).unwrap();
            seq.append_token(10);
            seq.append_token(11);
            seq.append_token(12);
        }

        // 5. Check if finished
        {
            let seq = scheduler.get_sequence(seq_id).unwrap();
            assert!(seq.is_finished());
        }

        // 6. Complete sequence
        scheduler.complete_sequence(seq_id).unwrap();
        assert_eq!(scheduler.running_count(), 0);
        assert!(!scheduler.has_work());

        let seq = scheduler.get_sequence(seq_id).unwrap();
        assert_eq!(seq.status, SequenceStatus::Completed);
    }

    #[test]
    fn test_error_sequence_not_found() {
        let scheduler = Scheduler::new(4);
        let result = scheduler.get_sequence(999);
        assert!(result.is_err());
        assert!(matches!(result, Err(BatchingError::SequenceNotFound(999))));
    }

    #[test]
    fn test_error_add_to_running_batch_not_in_queue() {
        let mut scheduler = Scheduler::new(4);
        let result = scheduler.add_to_running_batch(999);
        assert!(result.is_err());
    }

    #[test]
    fn test_clear_completed_sequences() {
        let mut scheduler = Scheduler::new(4);

        // Add and complete 3 sequences
        let seq1 = scheduler.add_sequence(vec![1], 1, 2, 32);
        let seq2 = scheduler.add_sequence(vec![2], 1, 2, 32);
        let seq3 = scheduler.add_sequence(vec![3], 1, 2, 32);

        scheduler.add_to_running_batch(seq1).unwrap();
        scheduler.add_to_running_batch(seq2).unwrap();
        scheduler.add_to_running_batch(seq3).unwrap();

        scheduler.complete_sequence(seq1).unwrap();
        scheduler.complete_sequence(seq2).unwrap();

        // seq3 is still running
        assert_eq!(scheduler.total_sequences(), 3);
        assert_eq!(scheduler.completed_count(), 2);
        assert_eq!(scheduler.running_count(), 1);

        // Clear completed sequences
        let removed = scheduler.clear_completed_sequences();
        assert_eq!(removed, 2);
        assert_eq!(scheduler.total_sequences(), 1);
        assert_eq!(scheduler.completed_count(), 0);

        // seq3 should still be accessible
        assert!(scheduler.get_sequence(seq3).is_ok());

        // seq1 and seq2 should be gone
        assert!(scheduler.get_sequence(seq1).is_err());
        assert!(scheduler.get_sequence(seq2).is_err());
    }

    #[test]
    fn test_reset() {
        let mut scheduler = Scheduler::new(4);

        // Add sequences in various states
        let seq1 = scheduler.add_sequence(vec![1], 1, 2, 32);
        let seq2 = scheduler.add_sequence(vec![2], 1, 2, 32);
        scheduler.add_to_running_batch(seq1).unwrap();
        scheduler.complete_sequence(seq1).unwrap();

        assert_eq!(scheduler.total_sequences(), 2);
        assert_eq!(scheduler.waiting_count(), 1);
        assert_eq!(scheduler.completed_count(), 1);

        // Reset everything
        scheduler.reset();

        assert_eq!(scheduler.total_sequences(), 0);
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(scheduler.running_count(), 0);
        assert_eq!(scheduler.completed_count(), 0);
        assert!(!scheduler.has_work());

        // Sequence IDs should restart from 0
        let new_seq = scheduler.add_sequence(vec![100], 1, 2, 32);
        assert_eq!(new_seq, 0);
    }

    #[test]
    fn test_memory_management_workflow() {
        let mut scheduler = Scheduler::new(4);

        // Simulate processing batches over time
        for batch in 0..3 {
            // Add 4 sequences
            let mut seq_ids = Vec::new();
            for i in 0..4 {
                let seq_id = scheduler.add_sequence(vec![batch * 4 + i], 1, 2, 32);
                seq_ids.push(seq_id);
            }

            // Process them
            for seq_id in &seq_ids {
                scheduler.add_to_running_batch(*seq_id).unwrap();
                scheduler.complete_sequence(*seq_id).unwrap();
            }

            // After each batch, clear completed sequences
            let removed = scheduler.clear_completed_sequences();
            assert_eq!(removed, 4);
            assert_eq!(scheduler.total_sequences(), 0);
        }

        // After 3 batches, we should have processed 12 sequences total
        // but memory should be clean
        assert_eq!(scheduler.total_sequences(), 0);
    }
}
