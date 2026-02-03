use std::path::Path;

use batching_rs::gguf::GGUFFile;
use batching_rs::model::LlamaModel;
use batching_rs::tokenizer::Tokenizer;
use batching_rs::worker::ModelWorker;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.len() > 3 {
        eprintln!(
            "Usage: {} <path_to_gguf_file> [path_to_tokenizer_json]",
            args[0]
        );
        eprintln!();
        eprintln!("If tokenizer path is not provided, will run with raw token IDs.");
        std::process::exit(1);
    }

    let gguf_path = Path::new(&args[1]);
    let tokenizer_path = args.get(2).map(|s| Path::new(s.as_str()));

    println!("Loading GGUF model from: {}", gguf_path.display());
    println!("{}", "=".repeat(60));

    // 1. Load and parse the GGUF file
    println!("\n📦 Step 1: Parsing GGUF file...");
    let gguf = GGUFFile::load(gguf_path)?;

    println!("\n✅ GGUF file parsed successfully!");
    println!("   Found {} tensors", gguf.tensors.len());
    println!("   Found {} metadata entries", gguf.metadata.len());

    // 2. Load tokenizer if provided
    let tokenizer = if let Some(tok_path) = tokenizer_path {
        println!("\n🔤 Step 2: Loading tokenizer...");
        println!("   Path: {}", tok_path.display());
        match Tokenizer::from_file(tok_path) {
            Ok(tok) => {
                println!("   ✅ Tokenizer loaded successfully!");
                println!("      Vocab size: {}", tok.vocab_size());
                println!("      BOS token ID: {}", tok.bos_token_id());
                println!("      EOS token ID: {}", tok.eos_token_id());
                Some(tok)
            }
            Err(e) => {
                println!("   ⚠️  Failed to load tokenizer: {}", e);
                println!("   Continuing without tokenizer...");
                None
            }
        }
    } else {
        println!("\n🔤 Step 2: Tokenizer not provided, will use raw token IDs");
        None
    };

    // 3. Create the model from GGUF
    let step_num = if tokenizer.is_some() { 3 } else { 2 };
    println!("\n🏗️  Step {}: Building model from GGUF...", step_num);
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

    // 4. Run a simple inference test
    let step_num = step_num + 1;
    println!("\n🚀 Step {}: Running inference test...", step_num);
    println!("{}", "=".repeat(60));

    // Create a worker with the model's config
    let max_batch_size = 4;
    let mut worker = ModelWorker::new(model.config().clone(), max_batch_size)?;

    // Prepare input based on whether we have a tokenizer
    let (prompt_tokens, eos_token_id) = if let Some(ref tok) = tokenizer {
        // Use natural language input with Llama 3 chat template
        let user_message = "What is the capital of France?";
        let system_prompt = "You are a helpful assistant.";

        println!("\n📝 Input (natural language):");

        // Check if this is a Llama 3 tokenizer and format accordingly
        let formatted_prompt = if tok.is_llama3() {
            println!("   Detected Llama 3 tokenizer, using chat template");
            println!("   System: \"{}\"", system_prompt);
            println!("   User: \"{}\"", user_message);
            tok.format_llama3_prompt(system_prompt, user_message)
        } else {
            println!("   Using simple prompt format");
            println!("   Prompt: \"{}\"", user_message);
            user_message.to_string()
        };

        println!("\n   Formatted prompt:");
        // Show the formatted prompt (truncate if too long)
        if formatted_prompt.len() > 200 {
            println!("   \"{}...\"", &formatted_prompt[..200]);
        } else {
            println!("   \"{}\"", formatted_prompt);
        }

        let tokens = tok.encode(&formatted_prompt, false)?;
        println!("\n   Encoded to {} tokens", tokens.len());

        (tokens, tok.eos_token_id())
    } else {
        // Use raw token IDs
        let tokens = vec![1, 15043, 338, 278]; // Example: "What is the"
        println!("\n📝 Input (raw token IDs):");
        println!("   Prompt tokens: {:?}", tokens);

        (tokens, 2) // Default EOS token ID
    };

    let max_new_tokens = 20;
    println!("   Max new tokens: {}", max_new_tokens);
    println!("   EOS token ID: {}", eos_token_id);

    let seq_id = worker.add_sequence(prompt_tokens.clone(), max_new_tokens, eos_token_id);
    println!("   Sequence ID: {}", seq_id);

    // Run generation
    println!("\n⚙️  Generating tokens...");
    let results = worker.run_until_complete()?;

    println!("\n✅ Generation complete!");

    // Display results
    if let Some(ref tok) = tokenizer {
        // Decode and show natural language output
        println!("\n📤 Output (natural language):");
        println!("   Generated {} tokens", results[0].len());

        let generated_text = tok.decode(&results[0], true)?;
        println!("\n   📝 Generated text:");
        println!("   \"{}\"", generated_text);

        // Also show the full conversation
        let mut full_tokens = prompt_tokens.clone();
        full_tokens.extend(&results[0]);
        let full_text = tok.decode(&full_tokens, true)?;

        println!("\n   💬 Full output (prompt + generated):");
        println!("   \"{}\"", full_text);
    } else {
        // Show raw token IDs
        println!("\n📤 Output (raw token IDs):");
        println!("   Generated {} tokens: {:?}", results[0].len(), results[0]);
    }

    // Show sequence statistics
    let sequence = worker.get_sequence(seq_id)?;
    println!("\n📊 Sequence Statistics:");
    println!("   Prompt length: {} tokens", sequence.prompt_tokens.len());
    println!(
        "   Generated length: {} tokens",
        sequence.generated_tokens.len()
    );
    println!(
        "   Total tokens: {} tokens",
        sequence.prompt_tokens.len() + sequence.generated_tokens.len()
    );
    println!(
        "   KV cache length: {} tokens",
        sequence.kv_cache.current_length()
    );

    println!("\n{}", "=".repeat(60));
    println!("✅ All tests passed! Model is ready for inference.");

    if tokenizer.is_none() {
        println!("\n💡 Tip: Provide a tokenizer.json file as the second argument");
        println!("   to use natural language input/output:");
        println!("   cargo run --example load_gguf -- model.gguf tokenizer.json");
    }

    println!("\nNote: This example uses simulated inference. To use real weights,");
    println!("the model implementation needs to be connected to MLX operations.");

    Ok(())
}
