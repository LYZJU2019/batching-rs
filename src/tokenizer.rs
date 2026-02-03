//! Tokenizer wrapper for Llama models
//!
//! This module provides a simple interface to the HuggingFace tokenizers library
//! for encoding text to tokens and decoding tokens back to text.

use crate::{BatchingError, Result};
use std::path::Path;
use tokenizers::Tokenizer as HFTokenizer;

/// Wrapper around HuggingFace tokenizer for Llama models
pub struct Tokenizer {
    tokenizer: HFTokenizer,
    bos_token_id: u32,
    eos_token_id: u32,
}

impl Tokenizer {
    /// Load a tokenizer from a tokenizer.json file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the tokenizer.json file
    ///
    /// # Returns
    ///
    /// A new Tokenizer instance
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer file cannot be loaded or parsed
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer = HFTokenizer::from_file(path)
            .map_err(|e| BatchingError::ModelError(format!("Failed to load tokenizer: {}", e)))?;

        // Get special token IDs
        let bos_token_id = tokenizer
            .token_to_id("<|begin_of_text|>")
            .or_else(|| tokenizer.token_to_id("<s>"))
            .or_else(|| tokenizer.token_to_id("<|startoftext|>"))
            .unwrap_or(1); // Default BOS token ID

        let eos_token_id = tokenizer
            .token_to_id("<|end_of_text|>")
            .or_else(|| tokenizer.token_to_id("</s>"))
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .unwrap_or(2); // Default EOS token ID

        Ok(Self {
            tokenizer,
            bos_token_id,
            eos_token_id,
        })
    }

    /// Encode text into token IDs
    ///
    /// # Arguments
    ///
    /// * `text` - The text to encode
    /// * `add_special_tokens` - Whether to add BOS/EOS tokens
    ///
    /// # Returns
    ///
    /// A vector of token IDs
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| BatchingError::ModelError(format!("Failed to encode text: {}", e)))?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back into text
    ///
    /// # Arguments
    ///
    /// * `token_ids` - The token IDs to decode
    /// * `skip_special_tokens` - Whether to skip special tokens in output
    ///
    /// # Returns
    ///
    /// The decoded text string
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| BatchingError::ModelError(format!("Failed to decode tokens: {}", e)))
    }

    /// Get the BOS (Beginning of Sequence) token ID
    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    /// Get the EOS (End of Sequence) token ID
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }

    /// Format a prompt using Llama 3 chat template
    ///
    /// This creates a properly formatted prompt for Llama 3 models with
    /// system message and user message.
    ///
    /// # Arguments
    ///
    /// * `system_prompt` - System instructions (can be empty)
    /// * `user_message` - The user's message/prompt
    ///
    /// # Returns
    ///
    /// A formatted string ready for tokenization
    ///
    /// # Example
    ///
    /// ```ignore
    /// let prompt = tokenizer.format_llama3_prompt(
    ///     "You are a helpful assistant.",
    ///     "What is the capital of France?"
    /// );
    /// ```
    pub fn format_llama3_prompt(&self, system_prompt: &str, user_message: &str) -> String {
        let date = "December 2023"; // Knowledge cutoff date
        let today = chrono::Local::now().format("%d %b %Y").to_string();

        let system_content = if system_prompt.is_empty() {
            String::new()
        } else {
            system_prompt.to_string()
        };

        format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
             Cutting Knowledge Date: {}\n\
             Today Date: {}\n\n\
             {}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\
             {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            date, today, system_content, user_message
        )
    }

    /// Format a simple prompt with just the begin_of_text token
    ///
    /// This is useful for simple completion tasks without chat formatting.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The text to format
    ///
    /// # Returns
    ///
    /// A formatted string with begin_of_text token
    pub fn format_simple_prompt(&self, prompt: &str) -> String {
        format!("<|begin_of_text|>{}", prompt)
    }

    /// Check if this is a Llama 3 tokenizer
    ///
    /// Returns true if the tokenizer has Llama 3 special tokens
    pub fn is_llama3(&self) -> bool {
        self.tokenizer.token_to_id("<|begin_of_text|>").is_some()
            && self.tokenizer.token_to_id("<|start_header_id|>").is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_basic() {
        // This test requires a real tokenizer file
        // Skip if not available
        let tokenizer_path = "tokenizer.json";
        if !std::path::Path::new(tokenizer_path).exists() {
            println!("Skipping test: tokenizer.json not found");
            return;
        }

        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
        let text = "Hello, world!";
        let tokens = tokenizer.encode(text, false).unwrap();
        let decoded = tokenizer.decode(&tokens, false).unwrap();

        assert!(!tokens.is_empty());
        assert!(decoded.contains("Hello"));
    }
}
