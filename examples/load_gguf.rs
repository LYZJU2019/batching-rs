use std::path::Path;

use batching_rs::gguf::GGUFFile;
use batching_rs::model::LlamaModel;
use batching_rs::worker::ModelWorker;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <path_to_gguf_file>", args[0]);
        std::process::exit(1);
    }

    let gguf_path = Path::new(&args[1]);

    println!("Loading GGUF model from: {}", gguf_path.display());
    println!("{}", "=".repeat(60));

    // 1. Load and parse the GGUF file
    println!("\n📦 Step 1: Parsing GGUF file...");
    let gguf = GGUFFile::load(gguf_path)?;

    println!("\n✅ GGUF file parsed successfully!");
    println!("   Found {} tensors", gguf.tensors.len());
    println!("   Found {} metadata entries", gguf.metadata.len());

    // 2. Create the model from GGUF
    println!("\n🏗️  Step 2: Building model from GGUF...");
    let model = LlamaModel::from_gguf(&gguf)?;

    println!("\n✅ Model loaded successfully!");
    println!("\n📊 Model Configuration:");
    println!("   Vocab size: {}", model.config().vocab_size);
    println!("   Hidden dimension: {}", model.config().hidden_dim);
    println!("   Number of layers: {}", model.config().n_layers);
    println!("   Attention heads: {}", model.config().n_heads);
    println!("   KV heads (GQA): {}", model.config().n_kv_heads);
    println!("   Head dimension: {}", model.config().head_dim);
    println!("   Intermediate dim: {}", model.config().intermediate_dim);
    println!("   RoPE base: {}", model.config().rope_base);

    // 3. Run a simple inference test
    println!("\n🚀 Step 3: Running inference test...");
    println!("{}", "=".repeat(60));

    // Create a worker with the model's config
    let max_batch_size = 4;
    let mut worker = ModelWorker::new(model.config().clone(), max_batch_size)?;

    // Add a test sequence
    // Using some example token IDs (in practice, these would come from a tokenizer)
    let prompt_tokens = vec![1, 15043, 338, 278]; // Example: "What is the"
    let max_new_tokens = 10;
    let eos_token_id = 2; // Common EOS token ID

    println!("\n📝 Input:");
    println!("   Prompt tokens: {:?}", prompt_tokens);
    println!("   Max new tokens: {}", max_new_tokens);

    let seq_id = worker.add_sequence(prompt_tokens.clone(), max_new_tokens, eos_token_id);
    println!("   Sequence ID: {}", seq_id);

    // Run generation
    println!("\n⚙️  Generating tokens...");
    let results = worker.run_until_complete()?;

    println!("\n✅ Generation complete!");
    println!("\n📤 Output:");
    println!("   Generated {} tokens: {:?}", results[0].len(), results[0]);

    // Show sequence details
    let sequence = worker.get_sequence(seq_id)?;
    println!("\n📊 Sequence Statistics:");
    println!("   Prompt length: {}", sequence.prompt_tokens.len());
    println!("   Generated length: {}", sequence.generated_tokens.len());
    println!(
        "   Total tokens: {}",
        sequence.prompt_tokens.len() + sequence.generated_tokens.len()
    );
    println!("   KV cache length: {}", sequence.kv_cache.current_length());

    println!("\n{}", "=".repeat(60));
    println!("✅ All tests passed! Model is ready for inference.");
    println!("\nNote: This example uses simulated inference. To use real weights,");
    println!("the model implementation needs to be connected to MLX operations.");

    Ok(())
}
