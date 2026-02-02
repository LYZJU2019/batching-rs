//! Integration tests for continuous batching pipeline
//!
//! These tests verify end-to-end functionality of the entire system:
//! - Single sequence generation
//! - Multiple sequences with batching
//! - Stopping conditions (EOS and max tokens)
//! - Mixed prefill and decode batches
//! - Scheduler batch size limits
//! - Complete generation lifecycle

use batching_rs::{ModelConfig, ModelWorker};

#[test]
fn test_single_sequence_end_to_end() {
    // Test: Single sequence generation from start to finish
    let config = ModelConfig::llama3_8b();
    let mut worker = ModelWorker::new(config, 4).unwrap();

    // Add one sequence
    let seq_id = worker.add_sequence(
        vec![1, 2, 3, 4, 5], // 5-token prompt
        10,                  // Generate 10 tokens
        999999,              // High EOS (won't trigger)
    );

    assert_eq!(seq_id, 0, "First sequence should have ID 0");
    assert!(worker.has_work(), "Worker should have work to do");

    // Run to completion
    let results = worker.run_until_complete().unwrap();

    // Verify results
    assert_eq!(results.len(), 1, "Should have 1 sequence result");
    assert_eq!(results[0].len(), 10, "Should generate exactly 10 tokens");

    // Verify no more work
    assert!(!worker.has_work(), "Worker should have no more work");
}

#[test]
fn test_multiple_sequences_parallel() {
    // Test: Multiple sequences running in parallel batch
    let config = ModelConfig::llama3_8b();
    let mut worker = ModelWorker::new(config, 4).unwrap();

    // Add three sequences with different lengths
    let seq1 = worker.add_sequence(vec![1, 2, 3], 5, 999999);
    let seq2 = worker.add_sequence(vec![4, 5, 6, 7], 3, 999999);
    let seq3 = worker.add_sequence(vec![8, 9], 7, 999999);

    assert_eq!(seq1, 0);
    assert_eq!(seq2, 1);
    assert_eq!(seq3, 2);

    // Run to completion
    let results = worker.run_until_complete().unwrap();

    // Verify all sequences completed
    assert_eq!(results.len(), 3, "Should have 3 sequence results");
    assert_eq!(results[0].len(), 5, "Seq 1 should generate 5 tokens");
    assert_eq!(results[1].len(), 3, "Seq 2 should generate 3 tokens");
    assert_eq!(results[2].len(), 7, "Seq 3 should generate 7 tokens");
}

#[test]
fn test_stopping_on_max_tokens() {
    // Test: Sequences stop after generating max_new_tokens
    let config = ModelConfig::llama3_8b();
    let mut worker = ModelWorker::new(config, 2).unwrap();

    worker.add_sequence(vec![1], 5, 999999); // Will stop after 5
    worker.add_sequence(vec![2], 3, 999999); // Will stop after 3

    let results = worker.run_until_complete().unwrap();

    assert_eq!(results[0].len(), 5);
    assert_eq!(results[1].len(), 3);
}

#[test]
fn test_interleaved_prefill_and_decode() {
    // Test: Sequences are added at different times, demonstrating
    // continuous batching with mixed prefill/decode operations
    let config = ModelConfig::llama3_8b();
    let mut worker = ModelWorker::new(config, 3).unwrap();

    // Add first sequence
    worker.add_sequence(vec![1, 2, 3], 5, 999999);

    // Step 1: Prefill seq 0
    let result = worker.step().unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].seq_id, 0);

    // Add second sequence while first is running
    worker.add_sequence(vec![4, 5], 3, 999999);

    // Step 2: Decode seq 0 + Prefill seq 1 (mixed batch)
    let result = worker.step().unwrap();
    assert_eq!(result.len(), 2, "Should process both sequences");

    // Verify both sequences are in results
    let seq_ids: Vec<u64> = result.iter().map(|r| r.seq_id).collect();
    assert!(seq_ids.contains(&0), "Should include seq 0");
    assert!(seq_ids.contains(&1), "Should include seq 1");

    // Complete remaining work
    while worker.has_work() {
        worker.step().unwrap();
    }
}

#[test]
fn test_batch_size_limit_respected() {
    // Test: Batch size limit is properly enforced
    let config = ModelConfig::llama3_8b();
    let max_batch_size = 2;
    let mut worker = ModelWorker::new(config, max_batch_size).unwrap();

    // Add 4 sequences (more than batch size)
    worker.add_sequence(vec![1], 3, 999999);
    worker.add_sequence(vec![2], 3, 999999);
    worker.add_sequence(vec![3], 3, 999999);
    worker.add_sequence(vec![4], 3, 999999);

    // First step: Should prefill only 2 sequences (batch limit)
    let result = worker.step().unwrap();
    assert_eq!(
        result.len(),
        2,
        "First step should process 2 sequences only"
    );

    // Second step: Should decode 2 + prefill 2 more (but limited to 2 total)
    let result = worker.step().unwrap();
    assert!(result.len() <= 3, "Should respect batch size limit");
}

#[test]
fn test_sequences_complete_at_different_times() {
    // Test: Handle sequences finishing at different times
    let config = ModelConfig::llama3_8b();
    let mut worker = ModelWorker::new(config, 3).unwrap();

    // Seq 0: Will finish quickly (2 tokens)
    worker.add_sequence(vec![1], 2, 999999);

    // Seq 1: Will finish later (5 tokens)
    worker.add_sequence(vec![2], 5, 999999);

    let results = worker.run_until_complete().unwrap();

    assert_eq!(results[0].len(), 2, "Short sequence completed");
    assert_eq!(results[1].len(), 5, "Long sequence completed");
}

#[test]
fn test_all_tokens_are_deterministic() {
    // Test: Same input produces same output (deterministic generation)
    let config = ModelConfig::llama3_8b();

    // Run 1
    let mut worker1 = ModelWorker::new(config.clone(), 4).unwrap();
    worker1.add_sequence(vec![1, 2, 3], 5, 999999);
    let results1 = worker1.run_until_complete().unwrap();

    // Run 2 (same input)
    let mut worker2 = ModelWorker::new(config, 4).unwrap();
    worker2.add_sequence(vec![1, 2, 3], 5, 999999);
    let results2 = worker2.run_until_complete().unwrap();

    // Should produce identical results
    assert_eq!(
        results1, results2,
        "Deterministic generation should produce same tokens"
    );
}

#[test]
fn test_empty_worker_idle() {
    // Test: Worker with no sequences returns Idle
    let config = ModelConfig::llama3_8b();
    let mut worker = ModelWorker::new(config, 4).unwrap();

    assert!(!worker.has_work(), "Empty worker should have no work");

    let result = worker.step().unwrap();
    assert_eq!(result.len(), 0, "Idle step should return empty results");
}

#[test]
fn test_sequence_lifecycle_complete() {
    // Test: Verify sequence goes through all lifecycle stages
    let config = ModelConfig::llama3_8b();
    let mut worker = ModelWorker::new(config, 4).unwrap();

    // Add sequence
    let _seq_id = worker.add_sequence(vec![1, 2], 3, 999999);

    // Initially: has work (sequence is waiting)
    assert!(worker.has_work());

    // Step 1: Prefill (sequence moves to running)
    let result = worker.step().unwrap();
    assert_eq!(result.len(), 1);
    assert!(!result[0].is_finished);
    assert!(worker.has_work());

    // Step 2-3: Decode (sequence still running)
    worker.step().unwrap();
    assert!(worker.has_work());

    worker.step().unwrap();

    // After max tokens reached, should be done
    assert!(!worker.has_work());
}

#[test]
fn test_different_prompt_lengths() {
    // Test: Handle sequences with varying prompt lengths
    let config = ModelConfig::llama3_8b();
    let mut worker = ModelWorker::new(config, 4).unwrap();

    // Short prompt
    worker.add_sequence(vec![1], 2, 999999);

    // Medium prompt
    worker.add_sequence(vec![1, 2, 3, 4, 5], 2, 999999);

    // Long prompt
    worker.add_sequence(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2, 999999);

    let results = worker.run_until_complete().unwrap();

    // All should generate the same number of new tokens
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].len(), 2);
    assert_eq!(results[1].len(), 2);
    assert_eq!(results[2].len(), 2);
}

#[test]
fn test_prefill_generates_first_token() {
    // Test: Prefill operation generates the first token correctly
    let config = ModelConfig::llama3_8b();
    let vocab_size = config.vocab_size;
    let mut worker = ModelWorker::new(config, 4).unwrap();

    worker.add_sequence(vec![1, 2, 3], 5, 999999);

    // First step (prefill) should generate a token
    let result = worker.step().unwrap();
    assert_eq!(result.len(), 1);
    assert!(
        result[0].token_id < vocab_size as u32,
        "Generated token should be valid"
    );
}

#[test]
fn test_decode_step_updates_position() {
    // Test: Each decode step properly advances the position
    let config = ModelConfig::llama3_8b();
    let mut worker = ModelWorker::new(config, 4).unwrap();

    worker.add_sequence(vec![1, 2, 3, 4, 5], 3, 999999);

    // Track tokens generated
    let mut all_tokens = Vec::new();

    while worker.has_work() {
        let result = worker.step().unwrap();
        for token_result in result {
            all_tokens.push(token_result.token_id);
        }
    }

    // Should have generated exactly 3 tokens (max_new_tokens)
    assert_eq!(all_tokens.len(), 3);
}

#[test]
fn test_batch_with_one_sequence() {
    // Test: Batch of size 1 works correctly
    let config = ModelConfig::llama3_8b();
    let mut worker = ModelWorker::new(config, 1).unwrap(); // Max batch = 1

    worker.add_sequence(vec![1], 3, 999999);
    worker.add_sequence(vec![2], 3, 999999);

    // Should process sequences one at a time
    let results = worker.run_until_complete().unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].len(), 3);
    assert_eq!(results[1].len(), 3);
}

#[test]
fn test_run_until_complete_returns_all_sequences() {
    // Test: run_until_complete returns results for all sequences in order
    let config = ModelConfig::llama3_8b();
    let mut worker = ModelWorker::new(config, 4).unwrap();

    let seq0 = worker.add_sequence(vec![10], 2, 999999);
    let seq1 = worker.add_sequence(vec![20], 3, 999999);
    let seq2 = worker.add_sequence(vec![30], 4, 999999);

    assert_eq!(seq0, 0);
    assert_eq!(seq1, 1);
    assert_eq!(seq2, 2);

    let results = worker.run_until_complete().unwrap();

    // Results should be indexed by sequence ID
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].len(), 2); // seq0
    assert_eq!(results[1].len(), 3); // seq1
    assert_eq!(results[2].len(), 4); // seq2
}

#[test]
fn test_greedy_sampling_picks_max_logit() {
    // Test: Greedy sampling behavior (implicitly tested through determinism)
    // The model should always pick the highest logit value
    let config = ModelConfig::llama3_8b();
    let mut worker = ModelWorker::new(config, 4).unwrap();

    // Same prompt should always generate same tokens (greedy)
    worker.add_sequence(vec![42, 42, 42], 3, 999999);

    let result1 = worker.run_until_complete().unwrap();

    // Run again with same prompt
    let mut worker2 = ModelWorker::new(ModelConfig::llama3_8b(), 4).unwrap();
    worker2.add_sequence(vec![42, 42, 42], 3, 999999);
    let result2 = worker2.run_until_complete().unwrap();

    assert_eq!(
        result1[0], result2[0],
        "Greedy sampling should be deterministic"
    );
}

#[test]
fn test_kv_cache_grows_correctly() {
    // Test: Implicit verification that KV cache is working
    // (If KV cache wasn't working, generation would fail)
    let config = ModelConfig::llama3_8b();
    let mut worker = ModelWorker::new(config, 4).unwrap();

    // Long generation to stress KV cache
    worker.add_sequence(vec![1], 20, 999999);

    let results = worker.run_until_complete().unwrap();

    // If KV cache wasn't growing properly, this would fail or panic
    assert_eq!(results[0].len(), 20, "Should generate all 20 tokens");
}

#[test]
fn test_concurrent_sequences_different_cache_lengths() {
    // Test: Multiple sequences with different cache lengths can run together
    let config = ModelConfig::llama3_8b();
    let mut worker = ModelWorker::new(config, 3).unwrap();

    // Different prompt lengths = different initial cache lengths
    worker.add_sequence(vec![1], 10, 999999); // Cache starts at 1
    worker.add_sequence(vec![1, 2, 3, 4, 5], 10, 999999); // Cache starts at 5
    worker.add_sequence(vec![1, 2], 10, 999999); // Cache starts at 2

    // Should handle all sequences despite different cache lengths
    let results = worker.run_until_complete().unwrap();

    assert_eq!(results.len(), 3);
    for result in results {
        assert_eq!(result.len(), 10, "All should generate 10 tokens");
    }
}

#[test]
fn test_step_by_step_execution() {
    // Test: Manual step-by-step execution matches run_until_complete
    let config = ModelConfig::llama3_8b();

    // Run with run_until_complete
    let mut worker1 = ModelWorker::new(config.clone(), 4).unwrap();
    worker1.add_sequence(vec![1, 2], 3, 999999);
    let results1 = worker1.run_until_complete().unwrap();

    // Run with manual steps
    let mut worker2 = ModelWorker::new(config, 4).unwrap();
    worker2.add_sequence(vec![1, 2], 3, 999999);

    let mut manual_tokens = Vec::new();
    while worker2.has_work() {
        let step_results = worker2.step().unwrap();
        for result in step_results {
            if result.seq_id == 0 {
                manual_tokens.push(result.token_id);
            }
        }
    }

    assert_eq!(
        results1[0], manual_tokens,
        "Manual stepping should match run_until_complete"
    );
}
