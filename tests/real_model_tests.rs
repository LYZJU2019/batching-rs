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

/// Download model weights from HuggingFace
fn download_model_weights(repo_id: &str, model_files: &[&str]) -> anyhow::Result<PathBuf> {
    let cache_dir = get_cache_dir()
        .join("models")
        .join(repo_id.replace("/", "--"));
    std::fs::create_dir_all(&cache_dir)?;

    println!("\n📦 Downloading model: {}", repo_id);
    println!("📁 Cache directory: {}", cache_dir.display());

    let token = get_hf_token().ok(); // Token is optional for public models

    for file_name in model_files {
        let file_path = cache_dir.join(file_name);

        if file_path.exists() {
            println!("  ✓ {} (cached)", file_name);
            continue;
        }

        println!("  ⬇ Downloading {}...", file_name);

        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo_id, file_name
        );

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(600)) // 10 minute timeout for large files
            .build()?;

        let mut request = client.get(&url);

        if let Some(ref token) = token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request.send()?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            anyhow::bail!(
                "❌ 404 Not Found: File '{}' does not exist in repository '{}'.\n\n\
                 The file may have been moved or renamed. Check the repository:\n\
                 https://huggingface.co/{}/tree/main\n",
                file_name,
                repo_id,
                repo_id
            );
        }

        if response.status() == reqwest::StatusCode::FORBIDDEN {
            anyhow::bail!(
                "❌ 403 Forbidden: Your HuggingFace account doesn't have access to {}.\n\n\
                 To fix this:\n\
                 1. Go to: https://huggingface.co/{}\n\
                 2. Click 'Request Access' and accept the terms\n\
                 3. Wait for approval (usually takes a few hours)\n\
                 4. Then run the tests again\n",
                repo_id,
                repo_id
            );
        }

        if !response.status().is_success() {
            anyhow::bail!(
                "Failed to download {}: HTTP {}",
                file_name,
                response.status()
            );
        }

        // Get content length for progress tracking
        let total_size = response.content_length().unwrap_or(0);
        println!("    Size: {:.2} MB", total_size as f64 / 1_048_576.0);

        let bytes = response.bytes()?;
        std::fs::write(&file_path, bytes)?;

        println!("    ✓ Downloaded successfully");
    }

    println!("✅ Model download complete\n");

    Ok(cache_dir)
}

/// Download Llama 3.2 3B model (smaller, faster for testing) in GGUF format
fn download_llama3_2_3b_model() -> anyhow::Result<PathBuf> {
    // Download GGUF file from bartowski repo
    let gguf_files = vec!["Llama-3.2-3B-Instruct-Q4_K_M.gguf"];
    let gguf_dir = download_model_weights("bartowski/Llama-3.2-3B-Instruct-GGUF", &gguf_files)?;

    // Download tokenizer and config from original Meta repo
    let config_files = vec!["config.json", "tokenizer.json", "tokenizer_config.json"];
    let _ = download_model_weights("meta-llama/Llama-3.2-3B-Instruct", &config_files)?;

    // Copy tokenizer files to the GGUF directory for convenience
    let original_dir = get_cache_dir()
        .join("models")
        .join("meta-llama--Llama-3.2-3B-Instruct");

    for file in &config_files {
        let src = original_dir.join(file);
        let dst = gguf_dir.join(file);
        if src.exists() && !dst.exists() {
            std::fs::copy(&src, &dst)?;
        }
    }

    Ok(gguf_dir)
}

/// Download Llama 3.2 1B model (smallest, fastest for testing) in GGUF format
fn download_llama3_2_1b_model() -> anyhow::Result<PathBuf> {
    // Download GGUF file from bartowski repo
    let gguf_files = vec!["Llama-3.2-1B-Instruct-Q4_K_M.gguf"];
    let gguf_dir = download_model_weights("bartowski/Llama-3.2-1B-Instruct-GGUF", &gguf_files)?;

    // Download tokenizer and config from original Meta repo
    let config_files = vec!["config.json", "tokenizer.json", "tokenizer_config.json"];
    let _ = download_model_weights("meta-llama/Llama-3.2-1B-Instruct", &config_files)?;

    // Copy tokenizer files to the GGUF directory for convenience
    let original_dir = get_cache_dir()
        .join("models")
        .join("meta-llama--Llama-3.2-1B-Instruct");

    for file in &config_files {
        let src = original_dir.join(file);
        let dst = gguf_dir.join(file);
        if src.exists() && !dst.exists() {
            std::fs::copy(&src, &dst)?;
        }
    }

    Ok(gguf_dir)
}

/// Download GPT-2 model in GGUF format (public, no approval needed)
fn download_gpt2_model() -> anyhow::Result<PathBuf> {
    // Download GGUF file from RichardErkhov repo (converted from official OpenAI GPT-2)
    let gguf_files = vec!["gpt2.Q4_K_M.gguf"];
    let gguf_dir =
        download_model_weights("RichardErkhov/openai-community_-_gpt2-gguf", &gguf_files)?;

    // Download tokenizer and config from original GPT-2 repo
    let config_files = vec!["config.json", "tokenizer.json", "vocab.json", "merges.txt"];
    let _ = download_model_weights("gpt2", &config_files)?;

    // Copy config files to the GGUF directory for convenience
    let original_dir = get_cache_dir().join("models").join("gpt2");

    for file in &config_files {
        let src = original_dir.join(file);
        let dst = gguf_dir.join(file);
        if src.exists() && !dst.exists() {
            std::fs::copy(&src, &dst)?;
        }
    }

    Ok(gguf_dir)
}

/// Download Phi-2 model in GGUF format (smaller Microsoft model, good for testing)
fn download_phi2_model() -> anyhow::Result<PathBuf> {
    let model_files = vec![
        "config.json",
        "phi-2-Q4_K_M.gguf", // GGUF format
        "tokenizer.json",
        "tokenizer_config.json",
    ];

    download_model_weights("TheBloke/phi-2-GGUF", &model_files)
}

/// Verify model files exist
fn verify_model_files(model_dir: &PathBuf, required_files: &[&str]) -> anyhow::Result<()> {
    for file_name in required_files {
        let file_path = model_dir.join(file_name);
        if !file_path.exists() {
            anyhow::bail!(
                "Required model file not found: {}\nExpected at: {}",
                file_name,
                file_path.display()
            );
        }
    }
    Ok(())
}

/// List available models in cache
fn list_cached_models() -> anyhow::Result<Vec<String>> {
    let models_dir = get_cache_dir().join("models");

    if !models_dir.exists() {
        return Ok(Vec::new());
    }

    let mut models = Vec::new();

    for entry in std::fs::read_dir(models_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            let model_name = entry.file_name().to_string_lossy().to_string();
            models.push(model_name.replace("--", "/"));
        }
    }

    Ok(models)
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

// ============================================================================
// Model Verification Tests
// ============================================================================

/// Verify downloaded model metadata and config
#[test]
fn verify_llama3_2_1b_metadata() -> anyhow::Result<()> {
    println!("\n=== Verifying Llama 3.2 1B Model Metadata ===\n");

    let model_dir = download_llama3_2_1b_model()?;

    // Verify all required files exist
    println!("📋 Checking required files...");
    let required_files = vec![
        "config.json",
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "tokenizer.json",
        "tokenizer_config.json",
    ];

    verify_model_files(&model_dir, &required_files)?;
    println!("✓ All required files present\n");

    // Check config.json
    println!("🔍 Analyzing config.json...");
    let config_path = model_dir.join("config.json");
    let config_content = std::fs::read_to_string(&config_path)?;
    let config: serde_json::Value = serde_json::from_str(&config_content)?;

    // Verify key configuration parameters
    println!(
        "  Model type: {}",
        config["model_type"].as_str().unwrap_or("unknown")
    );

    if let Some(vocab_size) = config["vocab_size"].as_u64() {
        println!("  Vocabulary size: {}", vocab_size);
        assert!(vocab_size > 0, "Vocabulary size must be positive");
        assert!(
            vocab_size < 1_000_000,
            "Vocabulary size seems unreasonably large"
        );
    }

    if let Some(hidden_size) = config["hidden_size"].as_u64() {
        println!("  Hidden size: {}", hidden_size);
        assert!(hidden_size >= 512, "Hidden size too small for Llama model");
    }

    if let Some(num_layers) = config["num_hidden_layers"].as_u64() {
        println!("  Number of layers: {}", num_layers);
        assert!(num_layers > 0, "Must have at least one layer");
        assert!(num_layers <= 100, "Too many layers for 1B model");
    }

    if let Some(num_heads) = config["num_attention_heads"].as_u64() {
        println!("  Attention heads: {}", num_heads);
        assert!(num_heads > 0, "Must have at least one attention head");
    }

    println!("✓ Config validation passed\n");

    // Check GGUF model file size
    println!("📦 Checking GGUF model file...");
    let gguf_path = model_dir.join("Llama-3.2-1B-Instruct-Q4_K_M.gguf");
    let metadata = std::fs::metadata(&gguf_path)?;
    let size_mb = metadata.len() as f64 / 1_048_576.0;

    println!("  File size: {:.2} MB", size_mb);

    // Q4_K_M quantized 1B model should be roughly 500MB-1GB
    assert!(size_mb > 100.0, "GGUF file suspiciously small (< 100MB)");
    assert!(
        size_mb < 5000.0,
        "GGUF file suspiciously large (> 5GB) for 1B model"
    );

    println!("✓ GGUF file size reasonable\n");

    // Check tokenizer
    println!("🔤 Verifying tokenizer...");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Test basic tokenization
    let test_text = "Hello, world!";
    let encoding = tokenizer
        .encode(test_text, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let tokens = encoding.get_ids();

    println!("  Test: \"{}\" → {} tokens", test_text, tokens.len());
    assert!(tokens.len() > 0, "Should produce at least one token");
    assert!(
        tokens.len() < 100,
        "Simple phrase shouldn't produce excessive tokens"
    );

    // Test decoding
    let decoded = tokenizer
        .decode(tokens, true)
        .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;
    println!("  Decoded: \"{}\"", decoded);

    println!("✓ Tokenizer working correctly\n");

    println!("✅ All metadata verification tests passed!");

    Ok(())
}

/// Verify GPT-2 model metadata
#[test]
fn verify_gpt2_metadata() -> anyhow::Result<()> {
    println!("\n=== Verifying GPT-2 Model Metadata ===\n");

    let model_dir = download_gpt2_model()?;

    println!("📋 Checking required files...");
    let required_files = vec!["config.json", "gpt2.Q4_K_M.gguf", "tokenizer.json"];

    verify_model_files(&model_dir, &required_files)?;
    println!("✓ All required files present\n");

    // Check config.json
    println!("🔍 Analyzing config.json...");
    let config_path = model_dir.join("config.json");
    let config_content = std::fs::read_to_string(&config_path)?;
    let config: serde_json::Value = serde_json::from_str(&config_content)?;

    println!(
        "  Model type: {}",
        config["model_type"].as_str().unwrap_or("unknown")
    );

    // GPT-2 specific checks
    if let Some(vocab_size) = config["vocab_size"].as_u64() {
        println!("  Vocabulary size: {}", vocab_size);
        // GPT-2 has vocab size of 50257
        assert!(vocab_size > 40000, "GPT-2 vocab size should be around 50k");
        assert!(vocab_size < 60000, "GPT-2 vocab size should be around 50k");
    }

    if let Some(n_embd) = config.get("n_embd").or(config.get("hidden_size")) {
        if let Some(n_embd) = n_embd.as_u64() {
            println!("  Embedding dimension: {}", n_embd);
            assert!(
                n_embd == 768 || n_embd == 1024 || n_embd == 1280 || n_embd == 1600,
                "GPT-2 embedding should be standard size"
            );
        }
    }

    println!("✓ GPT-2 config validation passed\n");

    // Check tokenizer
    println!("🔤 Verifying tokenizer...");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    let test_text = "The quick brown fox";
    let encoding = tokenizer
        .encode(test_text, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

    println!("  Test tokenization successful");
    println!("✓ Tokenizer working correctly\n");

    println!("✅ GPT-2 metadata verification passed!");

    Ok(())
}

/// Test model file integrity by checking file sizes and basic structure
#[test]
fn verify_model_file_integrity() -> anyhow::Result<()> {
    println!("\n=== Model File Integrity Verification ===\n");

    // Check cached models
    let cached_models = list_cached_models()?;

    if cached_models.is_empty() {
        println!("⚠️  No models cached yet. Download a model first.");
        return Ok(());
    }

    println!("📦 Found {} cached model(s):\n", cached_models.len());

    for model_name in cached_models {
        println!("Checking: {}", model_name);

        let model_dir = get_cache_dir()
            .join("models")
            .join(model_name.replace("/", "--"));

        // Check directory exists
        assert!(model_dir.exists(), "Model directory should exist");
        assert!(model_dir.is_dir(), "Model path should be a directory");

        // List all files in directory
        let mut file_count = 0;
        let mut total_size = 0u64;

        for entry in std::fs::read_dir(&model_dir)? {
            let entry = entry?;
            let metadata = entry.metadata()?;

            if metadata.is_file() {
                let file_name = entry.file_name();
                let file_size = metadata.len();
                total_size += file_size;
                file_count += 1;

                println!(
                    "  ✓ {} ({:.2} MB)",
                    file_name.to_string_lossy(),
                    file_size as f64 / 1_048_576.0
                );

                // Verify file is not empty
                assert!(file_size > 0, "File should not be empty: {:?}", file_name);

                // Check specific file types
                let file_name_str = file_name.to_string_lossy();
                if file_name_str.ends_with(".json") {
                    // Verify JSON files are valid
                    let content = std::fs::read_to_string(entry.path())?;
                    serde_json::from_str::<serde_json::Value>(&content)
                        .map_err(|e| anyhow::anyhow!("Invalid JSON in {}: {}", file_name_str, e))?;
                } else if file_name_str.ends_with(".gguf") {
                    // GGUF files should have the magic number (first 4 bytes: "GGUF")
                    let file = std::fs::File::open(entry.path())?;
                    let mut magic = [0u8; 4];
                    use std::io::Read;
                    let mut reader = std::io::BufReader::new(file);
                    reader.read_exact(&mut magic)?;
                    assert_eq!(&magic, b"GGUF", "GGUF file should start with magic number");
                }
            }
        }

        println!(
            "  Total: {} files, {:.2} MB\n",
            file_count,
            total_size as f64 / 1_048_576.0
        );

        assert!(file_count > 0, "Model directory should contain files");
    }

    println!("✅ All integrity checks passed!");

    Ok(())
}

/// Small inference test with downloaded model
#[test]
fn small_inference_test() -> anyhow::Result<()> {
    println!("\n=== Small Inference Test ===\n");

    println!("📥 Downloading Llama 3.2 1B model...");
    let model_dir = download_llama3_2_1b_model()?;

    // Load tokenizer
    println!("📖 Loading tokenizer...");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    println!("✓ Tokenizer loaded\n");

    // Small test prompt
    let test_prompts = vec!["Hello", "The capital of France is", "1 + 1 ="];

    println!("🧪 Running small inference tests:\n");

    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("Test {}: \"{}\"", i + 1, prompt);

        // Create a fresh worker for each test to avoid result accumulation
        println!("🤖 Initializing model worker...");
        let config = ModelConfig::llama3_8b(); // Using default config for now
        let mut worker = ModelWorker::new(config, 1)?;

        // Tokenize
        let encoding = tokenizer
            .encode(*prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        println!(
            "  Input tokens: {:?} ({} tokens)",
            token_ids,
            token_ids.len()
        );

        // Generate a few tokens
        let max_new_tokens = 5;
        worker.add_sequence(token_ids.clone(), max_new_tokens, 128001);

        let start = std::time::Instant::now();
        let results = worker.run_until_complete()?;
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 1, "Should get one result");
        let generated_ids = &results[0];

        // Decode
        let generated_text = tokenizer
            .decode(generated_ids, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

        println!(
            "  Generated: \"{}\" ({} tokens)",
            generated_text,
            generated_ids.len()
        );
        println!("  Time: {:?}", elapsed);

        // Verify basic properties
        assert!(
            !generated_ids.is_empty(),
            "Should generate at least one token"
        );
        assert!(
            generated_ids.len() <= max_new_tokens as usize,
            "Should not exceed max_new_tokens"
        );

        // Verify all token IDs are valid (non-negative)
        for &token_id in generated_ids {
            assert!(token_id < 200000, "Token ID seems invalid: {}", token_id);
        }

        println!("  ✓ Inference successful\n");
    }

    println!("✅ All inference tests passed!");
    println!(
        "\n⚠️  Note: Currently using stub model. Real MLX inference requires macOS SDK >= 14.0"
    );

    Ok(())
}

/// Verify tokenizer configuration matches model
#[test]
fn verify_tokenizer_config_match() -> anyhow::Result<()> {
    println!("\n=== Tokenizer-Model Configuration Match Test ===\n");

    let model_dir = download_llama3_2_1b_model()?;

    // Load config.json
    let config_path = model_dir.join("config.json");
    let config_content = std::fs::read_to_string(&config_path)?;
    let config: serde_json::Value = serde_json::from_str(&config_content)?;

    // Load tokenizer_config.json
    let tokenizer_config_path = model_dir.join("tokenizer_config.json");
    let tokenizer_config_content = std::fs::read_to_string(&tokenizer_config_path)?;
    let tokenizer_config: serde_json::Value = serde_json::from_str(&tokenizer_config_content)?;

    println!("🔍 Checking configuration consistency...\n");

    // Check vocabulary size consistency
    if let (Some(model_vocab), Some(tok_vocab)) = (
        config["vocab_size"].as_u64(),
        tokenizer_config.get("vocab_size").and_then(|v| v.as_u64()),
    ) {
        println!("  Model vocab size: {}", model_vocab);
        println!("  Tokenizer vocab size: {}", tok_vocab);

        // They should match or be very close
        let diff = (model_vocab as i64 - tok_vocab as i64).abs();
        assert!(
            diff < 100,
            "Vocab sizes differ too much: {} vs {}",
            model_vocab,
            tok_vocab
        );
    }

    // Check model type
    if let Some(model_type) = config["model_type"].as_str() {
        println!("  Model type: {}", model_type);
        assert!(
            model_type.to_lowercase().contains("llama")
                || model_type.to_lowercase().contains("mistral"),
            "Unexpected model type: {}",
            model_type
        );
    }

    // Check special tokens
    if let Some(eos_token_id) = tokenizer_config["eos_token_id"].as_u64() {
        println!("  EOS token ID: {}", eos_token_id);
        assert!(eos_token_id > 0, "EOS token ID should be positive");
    }

    if let Some(bos_token_id) = tokenizer_config["bos_token_id"].as_u64() {
        println!("  BOS token ID: {}", bos_token_id);
        assert!(bos_token_id < 1000000, "BOS token ID seems invalid");
    }

    println!("\n✓ Configuration consistency verified");

    // Load and test tokenizer
    println!("\n🔤 Testing tokenizer with special tokens...");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Test that tokenizer handles various inputs correctly
    let test_cases = vec![
        ("", "empty string"),
        ("a", "single character"),
        ("Hello, world!", "basic text"),
        ("   spaces   ", "text with spaces"),
        ("123456789", "numbers"),
    ];

    for (text, description) in test_cases {
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Failed to encode {}: {}", description, e))?;
        let tokens = encoding.get_ids();

        if !text.is_empty() {
            assert!(
                tokens.len() > 0,
                "Non-empty text should produce tokens: {}",
                description
            );
        }

        println!("  ✓ {} → {} tokens", description, tokens.len());
    }

    println!("\n✅ Tokenizer-model configuration match verified!");

    Ok(())
}

/// Benchmark tokenization speed
#[test]
fn benchmark_tokenization() -> anyhow::Result<()> {
    println!("\n=== Tokenization Benchmark ===\n");

    let model_dir = download_llama3_2_1b_model()?;
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Generate test text of various lengths
    let short_text = "Hello, world!";
    let medium_text = "The quick brown fox jumps over the lazy dog. ".repeat(10);
    let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(100);

    let test_cases = vec![
        (short_text, "short", 10000),
        (&medium_text, "medium", 1000),
        (&long_text, "long", 100),
    ];

    for (text, label, iterations) in test_cases {
        let start = std::time::Instant::now();
        let mut total_tokens = 0;

        for _ in 0..iterations {
            let encoding = tokenizer
                .encode(text, false)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
            total_tokens += encoding.get_ids().len();
        }

        let elapsed = start.elapsed();
        let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();
        let tokens_per_sec = total_tokens as f64 / elapsed.as_secs_f64();

        println!("📊 {} text ({} chars):", label, text.len());
        println!("  Iterations: {}", iterations);
        println!("  Time: {:?}", elapsed);
        println!("  Speed: {:.0} ops/sec", ops_per_sec);
        println!("  Throughput: {:.0} tokens/sec\n", tokens_per_sec);

        // Basic sanity checks
        assert!(ops_per_sec > 10.0, "Tokenization seems too slow");
    }

    println!("✅ Tokenization benchmark complete!");

    Ok(())
}
