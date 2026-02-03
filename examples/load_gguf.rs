use batching_rs::gguf::GGUFFile;
use std::path::PathBuf;

fn main() -> batching_rs::Result<()> {
    println!("🚀 GGUF Loader Example\n");

    // Path to the model
    let model_path = PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/batching-rs-tests/models/bartowski--Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf");

    println!("📂 Loading model from: {}\n", model_path.display());

    // Load the GGUF file
    let gguf = GGUFFile::load(&model_path)?;

    // Extract model configuration
    let config = gguf.extract_config()?;
    println!("\n✅ Model configuration extracted successfully!");
    println!("{:#?}", config);

    // List some tensors
    println!("\n📊 Sample tensors in the model:");
    let tensor_names = gguf.tensor_names();
    for (i, name) in tensor_names.iter().take(10).enumerate() {
        if let Some(info) = gguf.get_tensor_info(name) {
            println!(
                "  {}. {} - {:?} - dims: {:?} - {} bytes",
                i + 1,
                name,
                info.tensor_type,
                info.dimensions,
                info.size_bytes()
            );
        }
    }

    println!(
        "\n... and {} more tensors",
        tensor_names.len().saturating_sub(10)
    );

    // Show some metadata
    println!("\n🔍 Key metadata:");
    let metadata_keys = [
        "general.name",
        "general.architecture",
        "llama.context_length",
        "llama.rope.freq_base",
        "tokenizer.ggml.model",
    ];

    for key in metadata_keys {
        if let Some(value) = gguf.metadata.get(key) {
            println!("  {}: {:?}", key, value);
        }
    }

    // Try reading one tensor as a demonstration
    println!("\n💾 Reading a sample tensor (token_embd.weight)...");
    if let Ok(data) = gguf.read_tensor_data("token_embd.weight") {
        println!("  ✓ Successfully read {} bytes of tensor data", data.len());
    } else {
        println!("  ℹ️  Tensor not found or different naming convention");
    }

    println!("\n✨ GGUF loading completed successfully!");

    Ok(())
}
