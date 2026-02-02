# Real Model Integration Tests

These tests run the continuous batching system with **actual Llama 3 models** on **Metal GPU**.

## ⚠️ Current Status

The real model integration test framework is in place, but requires:
1. **macOS SDK >= 14.0** for MLX compilation
2. **Xcode Command Line Tools** updated to latest version

To check your SDK version:
```bash
xcrun --show-sdk-version
```

To update (if needed):
```bash
xcode-select --install
# Or update Xcode from App Store
```

## Prerequisites

1. **macOS with Metal GPU** (Apple Silicon M1/M2/M3 recommended)
2. **macOS SDK >= 14.0** (Sonoma or later)
3. **Llama 3 model** in MLX format (~2GB for quantized 3B model)
4. **Tokenizer** (auto-downloaded from HuggingFace)

## Setting Up Model

```bash
# Create cache directory
mkdir -p ~/.cache/batching-rs-tests

# Download Llama 3.2 3B Instruct (4-bit quantized, ~2GB)
cd ~/.cache/batching-rs-tests
huggingface-cli download mlx-community/Llama-3.2-3B-Instruct-4bit \
  --local-dir Llama-3.2-3B-Instruct-4bit

# Or use git:
git clone https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-4bit
```

## Running the Tests

Once MLX is properly installed:

```bash
# Compile and run all real model tests
cargo test --test real_model_tests --features real-models -- --ignored --nocapture

# Run specific test with verbose output
cargo test --test real_model_tests --features real-models test_real_model_single_generation -- --ignored --nocapture --test-threads=1

# Run continuous batching test
cargo test --test real_model_tests --features real-models test_real_model_continuous_batching -- --ignored --nocapture
```

## Test Cases Implemented

### 1. `test_real_model_single_generation`
- **Purpose**: Basic end-to-end test with a real model
- **What it does**:
  - Loads Llama 3 model and tokenizer
  - Tokenizes: "Hello, my name is"
  - Generates 20 tokens with Metal GPU
  - Decodes and prints the output
- **Verifies**: Real text generation works end-to-end

### 2. `test_real_model_batch_generation`
- **Purpose**: Test parallel batch processing
- **What it does**:
  - Runs 3 prompts simultaneously:
    - "The capital of France is"
    - "Once upon a time"
    - "In the year 2050,"
  - Max batch size: 4
  - Generates 15 tokens per sequence
  - Prints all outputs
- **Verifies**: Multi-sequence batching with real models

### 3. `test_real_model_continuous_batching` ⭐
- **Purpose**: Test true continuous batching (the core feature!)
- **What it does**:
  - Adds first sequence: "The meaning of life is"
  - Runs 5 prefill/decode steps
  - **Adds second sequence WHILE first is running**: "Artificial intelligence will"
  - Completes both sequences
  - Prints both outputs
- **Verifies**: Interleaved prefill + decode (TGI-style continuous batching)

### 4. `test_metal_gpu_performance`
- **Purpose**: Verify Metal GPU acceleration
- **What it does**:
  - Generates 50 tokens
  - Measures tokens/second
  - Asserts completion time < 30 seconds
- **Verifies**: GPU acceleration is actually working
- **Expected**: >5 tokens/sec on Apple Silicon

## Model Directory Structure

```
~/.cache/batching-rs-tests/
├── Llama-3.2-3B-Instruct-4bit/
│   ├── config.json           # Model architecture config
│   ├── tokenizer.json        # Tokenizer vocabulary
│   ├── tokenizer_config.json # Tokenizer settings
│   ├── weights.npz           # MLX model weights
│   └── ...
└── tokenizer.json            # (auto-downloaded fallback)
```

## What These Tests Verify

✅ **Real MLX inference** - Actual model forward passes, not stubs  
✅ **Metal GPU acceleration** - Uses Apple's Metal framework  
✅ **Real tokenization** - HuggingFace tokenizers with actual Llama vocab  
✅ **KV cache correctness** - Maintains attention cache across decode steps  
✅ **Continuous batching** - Interleaves prefill and decode (core innovation!)  
✅ **Multi-sequence handling** - Multiple prompts in single batch  
✅ **Performance** - Measures tokens/second on GPU  
✅ **End-to-end pipeline** - Tokenization → Batching → Generation → Decoding  

## Troubleshooting

### "MLX requires macOS SDK >= 14.0"
Update Xcode Command Line Tools:
```bash
xcode-select --install
xcrun --show-sdk-version  # Should show >= 14.0
```

### "Model not found"
Download the model to `~/.cache/batching-rs-tests/` as shown above.

### "Metal device not available"
Ensure you're on macOS with Apple Silicon (M1/M2/M3) or Intel Mac with discrete GPU.

### "Out of memory"
- Use quantized 3B model (not 8B)
- Reduce batch size
- Close other GPU-intensive apps

### Compilation is slow
First compilation downloads and builds MLX (~10-15 min). Subsequent builds are much faster.

## Comparison: Unit Tests vs Real Integration Tests

| Feature | Unit Tests (`integration_tests.rs`) | Real Tests (`real_model_tests.rs`) |
|---------|-------------------------------------|-------------------------------------|
| Model | Stub (random weights) | Real Llama 3 (2-8B params) |
| GPU | No | Yes (Metal) |
| Tokenizer | Mock token IDs | Real HuggingFace tokenizer |
| Output | Synthetic | Real text generation |
| Speed | Fast (~1s) | Realistic (~10-30s) |
| Purpose | Logic verification | End-to-end validation |

## Why Both Are Important

- **Unit Tests**: Fast feedback, verify logic correctness, CI/CD friendly
- **Real Tests**: Validate actual model inference, catch GPU/memory issues, production confidence

## Next Steps After Tests Pass

Once these tests pass, you can be confident that:
1. ✅ Continuous batching works with real models
2. ✅ Metal GPU is properly utilized
3. ✅ System can handle production workloads
4. ✅ Performance is acceptable
5. ✅ Ready to build HTTP API layer (PR-010+)

## Example Output

When working correctly, you'll see output like:
```
test test_real_model_single_generation ... 
Prompt: Hello, my name is
Token IDs: [9906, 11, 856, 836, 374]
Generated:  John and I am a software engineer...
ok

test test_real_model_continuous_batching ...
Running continuous batching with interleaved sequences...
Sequence 1 (The meaning of life is):  to find happiness and fulfillment...
Sequence 2 (Artificial intelligence will):  revolutionize how we work...
ok
```
