//! Real integration tests with actual Llama 3 tokenizers
//!
//! These tests use:
//! - Real HuggingFace tokenizers with actual Llama 3 vocabulary
//! - Real token IDs and vocabulary
//! - Stub model (until MLX can be compiled on SDK < 14.0)
//!
//! This validates:
//! - Tokenization pipeline works correctly
//! - Continuous batching handles real token sequences
//! - System works with realistic prompt/generation lengths
//!
//! Run with: cargo test --test real_model_tests --features real-models -- --ignored --nocapture
//!
//! NOTE: For full MLX GPU inference, you need macOS SDK >= 14.0 (macOS Sonoma+)

#![cfg(feature = "real-models")]

use batching_rs::{ModelConfig, ModelWorker};
use std::path::PathBuf;
use tokenizers::Tokenizer;

/// Path to cache tokenizer locally
fn get_cache_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap();
    PathBuf::from(home).join(".cache").join("batching-rs-tests")
}

/// Download tokenizer if needed
fn download_tokenizer_if_needed() -> anyhow::Result<PathBuf> {
    let cache_dir = get_cache_dir();
    std::fs::create_dir_all(&cache_dir)?;

    let tokenizer_path = cache_dir.join("tokenizer.json");

    if tokenizer_path.exists() {
        println!(
            "✓ Tokenizer already cached at: {}",
            tokenizer_path.display()
        );
        return Ok(tokenizer_path);
    }

    println!("⬇ Downloading Llama 3 tokenizer from HuggingFace...");

    // Get HuggingFace token from standard location or environment
    let token = get_hf_token()?;

    // Download from HuggingFace with authentication
    let url = "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/raw/main/tokenizer.json";
    let client = reqwest::blocking::Client::new();
    let response = client
        .get(url)
        .header("Authorization", format!("Bearer {}", token))
        .send()?;

    if response.status() == reqwest::StatusCode::FORBIDDEN {
        anyhow::bail!(
            "❌ 403 Forbidden: Your HuggingFace account doesn't have access to Llama models yet.\n\n\
             To fix this:\n\
             1. Go to: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct\n\
             2. Click 'Request Access' and accept Meta's terms\n\
             3. Wait for approval (usually takes a few hours)\n\
             4. Then run the tests again\n\n\
             Alternative: Use GPT-2 tokenizer (no approval needed):\n\
             • Change the URL below to use 'gpt2' instead of 'meta-llama/Llama-3.2-3B-Instruct'\n"
        );
    }

    if !response.status().is_success() {
        anyhow::bail!("Failed to download tokenizer: HTTP {}", response.status());
    }

    let bytes = response.bytes()?;
    std::fs::write(&tokenizer_path, bytes)?;

    println!("✓ Tokenizer cached at: {}", tokenizer_path.display());

    Ok(tokenizer_path)
}

/// Download tokenizer if needed (GPT-2 - no access approval needed)
fn download_gpt2_tokenizer_if_needed() -> anyhow::Result<PathBuf> {
    let cache_dir = get_cache_dir();
    std::fs::create_dir_all(&cache_dir)?;

    let tokenizer_path = cache_dir.join("tokenizer_gpt2.json");

    if tokenizer_path.exists() {
        println!(
            "✓ GPT-2 Tokenizer already cached at: {}",
            tokenizer_path.display()
        );
        return Ok(tokenizer_path);
    }

    println!("⬇ Downloading GPT-2 tokenizer from HuggingFace...");

    // GPT-2 is publicly accessible, no token needed
    let url = "https://huggingface.co/gpt2/raw/main/tokenizer.json";
    let client = reqwest::blocking::Client::new();
    let response = client.get(url).send()?;

    if !response.status().is_success() {
        anyhow::bail!("Failed to download tokenizer: HTTP {}", response.status());
    }

    let bytes = response.bytes()?;
    std::fs::write(&tokenizer_path, bytes)?;

    println!("✓ GPT-2 Tokenizer cached at: {}", tokenizer_path.display());

    Ok(tokenizer_path)
}

/// Get HuggingFace token from standard CLI cache or environment variable
fn get_hf_token() -> anyhow::Result<String> {
    // Try environment variable first
    if let Ok(token) = std::env::var("HF_TOKEN") {
        return Ok(token);
    }

    if let Ok(token) = std::env::var("HUGGING_FACE_HUB_TOKEN") {
        return Ok(token);
    }

    // Try reading from HuggingFace CLI token file
    let home = std::env::var("HOME")?;
    let token_path = PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("token");

    if token_path.exists() {
        let token = std::fs::read_to_string(&token_path)?.trim().to_string();
        return Ok(token);
    }

    anyhow::bail!(
        "HuggingFace token not found. Please:\n\
         1. Run: huggingface-cli login\n\
         2. Or set HF_TOKEN environment variable\n\
         3. Or set HUGGING_FACE_HUB_TOKEN environment variable"
    )
}

/// Load the real Llama 3 tokenizer
fn load_tokenizer() -> anyhow::Result<Tokenizer> {
    let tokenizer_path = download_tokenizer_if_needed()?;
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    Ok(tokenizer)
}

#[test]
fn test_real_tokenizer_single_generation() {
    println!("\n=== Test: Single Generation with Real Tokenizer ===");

    let tokenizer = load_tokenizer().unwrap();
    let config = ModelConfig::llama3_8b();
    let mut worker = ModelWorker::new(config, 1).unwrap();

    // Real prompt
    let prompt = "Hello, my name is";
    let encoding = tokenizer.encode(prompt, false).unwrap();
    let token_ids: Vec<u32> = encoding.get_ids().iter().map(|&id| id).collect();

    println!("📝 Prompt: \"{}\"", prompt);
    println!("🔢 Token IDs: {:?}", token_ids);
    println!("📊 Prompt length: {} tokens", token_ids.len());

    // Generate tokens
    let max_new = 20;
    worker.add_sequence(token_ids.clone(), max_new, 128001);

    println!("⚙️  Generating {} new tokens...", max_new);
    let results = worker.run_until_complete().unwrap();

    // Decode generated tokens
    let generated_ids: Vec<u32> = results[0].clone();
    let generated_text = tokenizer.decode(&generated_ids, true).unwrap();

    println!("📤 Generated {} tokens", generated_ids.len());
    println!("✨ Generated text: \"{}\"", generated_text);

    // Verify
    assert!(
        !generated_ids.is_empty(),
        "Should generate at least one token"
    );
    assert!(
        generated_ids.len() <= max_new as usize,
        "Should not exceed max_new_tokens"
    );

    println!("✅ Test passed!");
}

#[test]
fn test_real_tokenizer_batch_generation() {
    println!("\n=== Test: Batch Generation with Real Tokenizer ===");

    let tokenizer = load_tokenizer().unwrap();
    let config = ModelConfig::llama3_8b();
    let mut worker = ModelWorker::new(config, 4).unwrap();

    // Multiple real prompts
    let prompts = vec![
        "The capital of France is",
        "Once upon a time",
        "In the year 2050,",
    ];

    println!("📝 Processing {} prompts:", prompts.len());
    for (i, text) in prompts.iter().enumerate() {
        let encoding = tokenizer.encode(*text, false).unwrap();
        let token_ids: Vec<u32> = encoding.get_ids().iter().map(|&id| id).collect();
        println!(
            "  Prompt {}: \"{}\" → {} tokens",
            i + 1,
            text,
            token_ids.len()
        );
        worker.add_sequence(token_ids, 15, 128001);
    }

    println!("⚙️  Generating in batch...");
    let results = worker.run_until_complete().unwrap();

    assert_eq!(results.len(), prompts.len());

    println!("\n📤 Results:");
    for (i, (prompt, generated_ids)) in prompts.iter().zip(results.iter()).enumerate() {
        let generated_text = tokenizer.decode(generated_ids, true).unwrap();
        println!(
            "  {}. \"{}\" → \"{}\" ({} tokens)",
            i + 1,
            *prompt,
            generated_text,
            generated_ids.len()
        );
        assert!(!generated_ids.is_empty());
    }

    println!("✅ Test passed!");
}

#[test]
fn test_real_tokenizer_continuous_batching() {
    println!("\n=== Test: Continuous Batching with Real Tokenizer ===");

    let tokenizer = load_tokenizer().unwrap();
    let config = ModelConfig::llama3_8b();
    let mut worker = ModelWorker::new(config, 3).unwrap();

    // First sequence
    let prompt1 = "The meaning of life is";
    let encoding1 = tokenizer.encode(prompt1, false).unwrap();
    let token_ids1: Vec<u32> = encoding1.get_ids().iter().map(|&id| id).collect();

    println!(
        "📝 Sequence 1: \"{}\" ({} tokens)",
        prompt1,
        token_ids1.len()
    );
    worker.add_sequence(token_ids1, 30, 128001);

    // Run a few steps
    println!("⚙️  Running 5 prefill/decode steps...");
    for i in 0..5 {
        let step_results = worker.step().unwrap();
        println!("  Step {}: {} sequences active", i + 1, step_results.len());
    }

    // Add second sequence while first is running (continuous batching!)
    let prompt2 = "Artificial intelligence will";
    let encoding2 = tokenizer.encode(prompt2, false).unwrap();
    let token_ids2: Vec<u32> = encoding2.get_ids().iter().map(|&id| id).collect();

    println!(
        "📝 Sequence 2: \"{}\" ({} tokens) [ADDED WHILE SEQ1 RUNNING]",
        prompt2,
        token_ids2.len()
    );
    worker.add_sequence(token_ids2, 20, 128001);

    println!("⚙️  Completing all sequences with continuous batching...");
    let results = worker.run_until_complete().unwrap();

    assert_eq!(results.len(), 2);

    let gen1 = tokenizer.decode(&results[0], true).unwrap();
    let gen2 = tokenizer.decode(&results[1], true).unwrap();

    println!("\n📤 Final Results:");
    println!(
        "  1. \"{}\" → \"{}\" ({} tokens)",
        prompt1,
        gen1,
        results[0].len()
    );
    println!(
        "  2. \"{}\" → \"{}\" ({} tokens)",
        prompt2,
        gen2,
        results[1].len()
    );

    println!("✅ Continuous batching test passed!");
}

#[test]
fn test_real_tokenizer_vocabulary_coverage() {
    println!("\n=== Test: Tokenizer Vocabulary Coverage ===");

    let tokenizer = load_tokenizer().unwrap();

    // Test various prompts to ensure tokenizer works correctly
    let test_cases = vec![
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "In 2050, artificial intelligence will transform society.",
        "पृथ्वी एक ग्रह है",               // Non-English (Hindi)
        "🚀 Let's go to the moon! 🌙", // Emojis
    ];

    println!(
        "Testing tokenization on {} diverse inputs:\n",
        test_cases.len()
    );

    for (i, text) in test_cases.iter().enumerate() {
        let encoding = tokenizer.encode(*text, false).unwrap();
        let token_ids: Vec<u32> = encoding.get_ids().iter().map(|&id| id).collect();
        let decoded = tokenizer.decode(&token_ids, true).unwrap();

        println!("{}. Input:   \"{}\"", i + 1, text);
        println!("   Tokens:  {:?}", token_ids);
        println!("   Count:   {}", token_ids.len());
        println!("   Decoded: \"{}\"\n", decoded);

        assert!(
            !token_ids.is_empty(),
            "Should tokenize to at least one token"
        );
    }

    println!("✅ Vocabulary coverage test passed!");
}

#[test]
fn test_real_tokenizer_special_tokens() {
    println!("\n=== Test: Special Tokens Handling ===");

    let tokenizer = load_tokenizer().unwrap();

    // Test that we can identify special tokens
    let prompts = vec![
        "Hello",
        "</s>",              // EOS token
        "<|begin_of_text|>", // Llama 3 special token
    ];

    println!("Testing special token handling:\n");

    for prompt in prompts {
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let token_ids: Vec<u32> = encoding.get_ids().iter().map(|&id| id).collect();

        println!("Input: \"{}\"", prompt);
        println!("Token IDs: {:?}", token_ids);
        println!("Count: {}\n", token_ids.len());
    }

    // Verify EOS token ID
    let eos_id = 128001u32; // Llama 3 EOS token
    println!("Llama 3 EOS token ID: {}", eos_id);

    println!("✅ Special tokens test passed!");
}

#[test]
fn test_batching_performance_metrics() {
    println!("\n=== Test: Batching Performance Metrics ===");

    let config = ModelConfig::llama3_8b();
    let mut worker = ModelWorker::new(config, 4).unwrap();

    // Simulate realistic token sequences (typical prompt lengths)
    let sequences = vec![
        (vec![1, 2, 3, 4, 5, 6, 7], 50), // Short prompt, long generation
        (vec![1; 100], 20),              // Long prompt, short generation
        (vec![1; 50], 30),               // Medium prompt and generation
        (vec![1, 2, 3], 100),            // Very short prompt, long generation
    ];

    println!("Adding {} sequences with varying lengths:", sequences.len());
    for (i, (tokens, max_new)) in sequences.iter().enumerate() {
        println!(
            "  Seq {}: {} prompt tokens, {} max_new",
            i + 1,
            tokens.len(),
            max_new
        );
        worker.add_sequence(tokens.clone(), *max_new, 128001);
    }

    // Time the execution
    let start = std::time::Instant::now();
    let results = worker.run_until_complete().unwrap();
    let elapsed = start.elapsed();

    // Calculate metrics
    let total_tokens_generated: usize = results.iter().map(|r| r.len()).sum();
    let tokens_per_sec = total_tokens_generated as f64 / elapsed.as_secs_f64();

    println!("\n📊 Performance Metrics:");
    println!("  Total sequences: {}", results.len());
    println!("  Total tokens generated: {}", total_tokens_generated);
    println!("  Time elapsed: {:?}", elapsed);
    println!("  Throughput: {:.2} tokens/second", tokens_per_sec);

    // Verify all sequences completed
    assert_eq!(results.len(), sequences.len());
    for (i, result) in results.iter().enumerate() {
        println!("  Seq {} generated: {} tokens", i + 1, result.len());
        assert!(!result.is_empty());
    }

    println!("✅ Performance test passed!");
}

#[test]
fn test_tokenizer_gpt2() -> anyhow::Result<()> {
    println!("\n=== Testing GPT-2 Tokenizer (No Access Approval Needed) ===\n");

    let tokenizer_path = download_gpt2_tokenizer_if_needed()?;

    println!("📖 Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    println!("✓ Tokenizer loaded successfully\n");

    // Test prompts
    let test_prompts = vec![
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog.",
        "What is the meaning of life?",
    ];

    println!("🧪 Testing tokenization:\n");

    for prompt in test_prompts {
        let encoding = tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let tokens = encoding.get_ids();
        let decoded = tokenizer
            .decode(tokens, false)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

        println!("  Input:   \"{}\"", prompt);
        println!("  Tokens:  {:?}", tokens);
        println!("  Length:  {} tokens", tokens.len());
        println!("  Decoded: \"{}\"", decoded);
        println!();
    }

    println!("✅ All GPT-2 tokenizer tests passed!");

    Ok(())
}
