//! GGUF file format parser and model loader
//!
//! This module implements loading Llama models from GGUF (GPT-Generated Unified Format) files.
//! GGUF is a binary format used by llama.cpp for storing quantized LLM weights.
//!
//! # File Structure
//!
//! 1. Header: magic number (GGUF), version, tensor count, metadata count
//! 2. Metadata: key-value pairs (model config, rope settings, etc.)
//! 3. Tensor Info: name, dimensions, data type, offset for each tensor
//! 4. Alignment padding (typically 32 bytes)
//! 5. Tensor Data: actual weight data (may be quantized)

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::{BatchingError, ModelConfig, Result};

/// GGUF magic number: "GGUF" in ASCII (little-endian)
const GGUF_MAGIC: u32 = 0x46554747;

/// GGUF version we support (version 3 is current standard)
const GGUF_VERSION: u32 = 3;

/// Default alignment for tensor data
const DEFAULT_ALIGNMENT: u64 = 32;

/// Metadata value types in GGUF format
#[derive(Debug, Clone)]
pub enum GGUFMetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFMetadataValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GGUFMetadataValue {
    /// Try to convert to u32
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GGUFMetadataValue::U32(v) => Some(*v),
            GGUFMetadataValue::I32(v) => Some(*v as u32),
            GGUFMetadataValue::U64(v) => Some(*v as u32),
            GGUFMetadataValue::I64(v) => Some(*v as u32),
            _ => None,
        }
    }

    /// Try to convert to f32
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GGUFMetadataValue::F32(v) => Some(*v),
            GGUFMetadataValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Try to convert to string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            GGUFMetadataValue::String(s) => Some(s),
            _ => None,
        }
    }
}

/// GGUF tensor data types (including quantized formats)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GGUFTensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2, // 4-bit quantization (32 values per block)
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10, // K-quants (256 values per block)
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
}

impl GGUFTensorType {
    /// Parse from u32 value
    pub fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(GGUFTensorType::F32),
            1 => Ok(GGUFTensorType::F16),
            2 => Ok(GGUFTensorType::Q4_0),
            3 => Ok(GGUFTensorType::Q4_1),
            6 => Ok(GGUFTensorType::Q5_0),
            7 => Ok(GGUFTensorType::Q5_1),
            8 => Ok(GGUFTensorType::Q8_0),
            9 => Ok(GGUFTensorType::Q8_1),
            10 => Ok(GGUFTensorType::Q2_K),
            11 => Ok(GGUFTensorType::Q3_K),
            12 => Ok(GGUFTensorType::Q4_K),
            13 => Ok(GGUFTensorType::Q5_K),
            14 => Ok(GGUFTensorType::Q6_K),
            15 => Ok(GGUFTensorType::Q8_K),
            16 => Ok(GGUFTensorType::I8),
            17 => Ok(GGUFTensorType::I16),
            18 => Ok(GGUFTensorType::I32),
            _ => Err(BatchingError::ModelError(format!(
                "Unknown GGUF tensor type: {}",
                value
            ))),
        }
    }

    /// Get the block size (number of elements encoded together)
    pub fn block_size(&self) -> usize {
        match self {
            GGUFTensorType::F32 | GGUFTensorType::F16 => 1,
            GGUFTensorType::I8 | GGUFTensorType::I16 | GGUFTensorType::I32 => 1,
            GGUFTensorType::Q4_0 | GGUFTensorType::Q4_1 => 32,
            GGUFTensorType::Q5_0 | GGUFTensorType::Q5_1 => 32,
            GGUFTensorType::Q8_0 | GGUFTensorType::Q8_1 => 32,
            // K-quants use 256 element blocks
            GGUFTensorType::Q2_K | GGUFTensorType::Q3_K => 256,
            GGUFTensorType::Q4_K | GGUFTensorType::Q5_K => 256,
            GGUFTensorType::Q6_K | GGUFTensorType::Q8_K => 256,
        }
    }

    /// Get the size in bytes per block
    pub fn block_bytes(&self) -> usize {
        match self {
            GGUFTensorType::F32 => 4,
            GGUFTensorType::F16 => 2,
            GGUFTensorType::I8 => 1,
            GGUFTensorType::I16 => 2,
            GGUFTensorType::I32 => 4,
            GGUFTensorType::Q4_0 => 18, // 16 nibbles + 2 bytes scale
            GGUFTensorType::Q4_1 => 20,
            GGUFTensorType::Q5_0 => 22,
            GGUFTensorType::Q5_1 => 24,
            GGUFTensorType::Q8_0 => 34,
            GGUFTensorType::Q8_1 => 36,
            GGUFTensorType::Q2_K => 80,
            GGUFTensorType::Q3_K => 112,
            GGUFTensorType::Q4_K => 144,
            GGUFTensorType::Q5_K => 176,
            GGUFTensorType::Q6_K => 210,
            GGUFTensorType::Q8_K => 292,
        }
    }
}

/// Information about a tensor in the GGUF file
#[derive(Debug, Clone)]
pub struct GGUFTensorInfo {
    pub name: String,
    pub tensor_type: GGUFTensorType,
    pub dimensions: Vec<u64>,
    pub offset: u64,
}

impl GGUFTensorInfo {
    /// Calculate total number of elements
    pub fn num_elements(&self) -> u64 {
        self.dimensions.iter().product()
    }

    /// Calculate size in bytes
    pub fn size_bytes(&self) -> u64 {
        let block_size = self.tensor_type.block_size() as u64;
        let block_bytes = self.tensor_type.block_bytes() as u64;
        let num_blocks = (self.num_elements() + block_size - 1) / block_size;
        num_blocks * block_bytes
    }
}

/// Parsed GGUF file
pub struct GGUFFile {
    pub metadata: HashMap<String, GGUFMetadataValue>,
    pub tensors: HashMap<String, GGUFTensorInfo>,
    pub tensor_data_offset: u64,
    file_path: std::path::PathBuf,
}

impl GGUFFile {
    /// Load and parse a GGUF file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| {
            BatchingError::ModelError(format!(
                "Failed to open GGUF file '{}': {}",
                path.display(),
                e
            ))
        })?;
        let mut reader = BufReader::new(file);

        // Read and validate header
        let magic = Self::read_u32(&mut reader)?;
        if magic != GGUF_MAGIC {
            return Err(BatchingError::ModelError(format!(
                "Invalid GGUF magic number: 0x{:08X} (expected 0x{:08X})",
                magic, GGUF_MAGIC
            )));
        }

        let version = Self::read_u32(&mut reader)?;
        if version != GGUF_VERSION {
            return Err(BatchingError::ModelError(format!(
                "Unsupported GGUF version: {} (expected {})",
                version, GGUF_VERSION
            )));
        }

        let tensor_count = Self::read_u64(&mut reader)?;
        let metadata_kv_count = Self::read_u64(&mut reader)?;

        println!("📦 Loading GGUF file: {}", path.display());
        println!(
            "   Version: {}, Tensors: {}, Metadata entries: {}",
            version, tensor_count, metadata_kv_count
        );

        // Read metadata key-value pairs
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = Self::read_string(&mut reader)?;
            let value = Self::read_metadata_value(&mut reader)?;
            metadata.insert(key, value);
        }

        // Read tensor information
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name = Self::read_string(&mut reader)?;
            let n_dimensions = Self::read_u32(&mut reader)?;

            let mut dimensions = Vec::new();
            for _ in 0..n_dimensions {
                dimensions.push(Self::read_u64(&mut reader)?);
            }

            let tensor_type_value = Self::read_u32(&mut reader)?;
            let tensor_type = GGUFTensorType::from_u32(tensor_type_value)?;
            let offset = Self::read_u64(&mut reader)?;

            tensors.insert(
                name.clone(),
                GGUFTensorInfo {
                    name,
                    tensor_type,
                    dimensions,
                    offset,
                },
            );
        }

        // Calculate tensor data offset (aligned)
        let current_pos = reader.stream_position().map_err(|e| {
            BatchingError::ModelError(format!("Failed to get stream position: {}", e))
        })?;

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(DEFAULT_ALIGNMENT as u32) as u64;

        let tensor_data_offset = ((current_pos + alignment - 1) / alignment) * alignment;

        println!(
            "   ✓ Parsed successfully. Tensor data starts at offset: {}",
            tensor_data_offset
        );

        Ok(Self {
            metadata,
            tensors,
            tensor_data_offset,
            file_path: path.to_path_buf(),
        })
    }

    /// Extract ModelConfig from GGUF metadata
    pub fn extract_config(&self) -> Result<ModelConfig> {
        println!("🔧 Extracting model configuration from GGUF metadata...");

        // Read required fields
        let vocab_size = self
            .get_metadata_u32("llama.vocab_size")
            .or_else(|| self.get_metadata_u32("tokenizer.ggml.vocab_size"))
            .ok_or_else(|| {
                BatchingError::ModelError("Missing vocab_size in metadata".to_string())
            })?;

        let n_layers = self.get_metadata_u32("llama.block_count").ok_or_else(|| {
            BatchingError::ModelError("Missing block_count in metadata".to_string())
        })?;

        let hidden_dim = self
            .get_metadata_u32("llama.embedding_length")
            .ok_or_else(|| {
                BatchingError::ModelError("Missing embedding_length in metadata".to_string())
            })?;

        let n_heads = self
            .get_metadata_u32("llama.attention.head_count")
            .ok_or_else(|| {
                BatchingError::ModelError("Missing head_count in metadata".to_string())
            })?;

        let n_kv_heads = self
            .get_metadata_u32("llama.attention.head_count_kv")
            .unwrap_or(n_heads);

        let head_dim = hidden_dim / n_heads;

        let intermediate_dim = self
            .get_metadata_u32("llama.feed_forward_length")
            .unwrap_or((hidden_dim * 8) / 3);

        let rope_base = self
            .get_metadata_f32("llama.rope.freq_base")
            .unwrap_or(10000.0);

        let rope_scale = self
            .get_metadata_f32("llama.rope.scale_linear")
            .unwrap_or(1.0);

        let rms_norm_eps = self
            .get_metadata_f32("llama.attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-5);

        println!("   Vocab size: {}", vocab_size);
        println!("   Layers: {}", n_layers);
        println!("   Hidden dim: {}", hidden_dim);
        println!("   Heads: {} (KV heads: {})", n_heads, n_kv_heads);
        println!("   Intermediate dim: {}", intermediate_dim);

        ModelConfig::new(
            vocab_size as usize,
            n_layers as usize,
            hidden_dim as usize,
            n_heads as usize,
            n_kv_heads as usize,
            head_dim as usize,
            intermediate_dim as usize,
            rope_base,
            rope_scale,
            rms_norm_eps,
        )
    }

    /// Read raw tensor data from file
    pub fn read_tensor_data(&self, tensor_name: &str) -> Result<Vec<u8>> {
        let tensor_info = self.tensors.get(tensor_name).ok_or_else(|| {
            BatchingError::ModelError(format!("Tensor '{}' not found in GGUF file", tensor_name))
        })?;

        let mut file = File::open(&self.file_path)
            .map_err(|e| BatchingError::ModelError(format!("Failed to open GGUF file: {}", e)))?;

        let absolute_offset = self.tensor_data_offset + tensor_info.offset;
        file.seek(SeekFrom::Start(absolute_offset)).map_err(|e| {
            BatchingError::ModelError(format!("Failed to seek to tensor data: {}", e))
        })?;

        let size = tensor_info.size_bytes() as usize;
        let mut buffer = vec![0u8; size];
        file.read_exact(&mut buffer)
            .map_err(|e| BatchingError::ModelError(format!("Failed to read tensor data: {}", e)))?;

        Ok(buffer)
    }

    /// Get tensor info by name
    pub fn get_tensor_info(&self, name: &str) -> Option<&GGUFTensorInfo> {
        self.tensors.get(name)
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.tensors.keys().cloned().collect();
        names.sort();
        names
    }

    /// Helper to get u32 from metadata
    fn get_metadata_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key)?.as_u32()
    }

    /// Helper to get f32 from metadata
    fn get_metadata_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key)?.as_f32()
    }

    // Binary reading helpers

    fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
        let mut buf = [0u8; 4];
        reader
            .read_exact(&mut buf)
            .map_err(|e| BatchingError::ModelError(format!("Failed to read u32: {}", e)))?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
        let mut buf = [0u8; 8];
        reader
            .read_exact(&mut buf)
            .map_err(|e| BatchingError::ModelError(format!("Failed to read u64: {}", e)))?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i32<R: Read>(reader: &mut R) -> Result<i32> {
        let mut buf = [0u8; 4];
        reader
            .read_exact(&mut buf)
            .map_err(|e| BatchingError::ModelError(format!("Failed to read i32: {}", e)))?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_f32<R: Read>(reader: &mut R) -> Result<f32> {
        let mut buf = [0u8; 4];
        reader
            .read_exact(&mut buf)
            .map_err(|e| BatchingError::ModelError(format!("Failed to read f32: {}", e)))?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_f64<R: Read>(reader: &mut R) -> Result<f64> {
        let mut buf = [0u8; 8];
        reader
            .read_exact(&mut buf)
            .map_err(|e| BatchingError::ModelError(format!("Failed to read f64: {}", e)))?;
        Ok(f64::from_le_bytes(buf))
    }

    fn read_string<R: Read>(reader: &mut R) -> Result<String> {
        let len = Self::read_u64(reader)? as usize;
        let mut buf = vec![0u8; len];
        reader
            .read_exact(&mut buf)
            .map_err(|e| BatchingError::ModelError(format!("Failed to read string: {}", e)))?;
        String::from_utf8(buf)
            .map_err(|e| BatchingError::ModelError(format!("Invalid UTF-8 in string: {}", e)))
    }

    fn read_metadata_value<R: Read>(reader: &mut R) -> Result<GGUFMetadataValue> {
        let value_type = Self::read_u32(reader)?;

        match value_type {
            0 => {
                // U8
                let mut buf = [0u8; 1];
                reader
                    .read_exact(&mut buf)
                    .map_err(|e| BatchingError::ModelError(format!("Failed to read u8: {}", e)))?;
                Ok(GGUFMetadataValue::U8(buf[0]))
            }
            1 => {
                // I8
                let mut buf = [0u8; 1];
                reader
                    .read_exact(&mut buf)
                    .map_err(|e| BatchingError::ModelError(format!("Failed to read i8: {}", e)))?;
                Ok(GGUFMetadataValue::I8(buf[0] as i8))
            }
            2 => {
                // U16
                let mut buf = [0u8; 2];
                reader
                    .read_exact(&mut buf)
                    .map_err(|e| BatchingError::ModelError(format!("Failed to read u16: {}", e)))?;
                Ok(GGUFMetadataValue::U16(u16::from_le_bytes(buf)))
            }
            3 => {
                // I16
                let mut buf = [0u8; 2];
                reader
                    .read_exact(&mut buf)
                    .map_err(|e| BatchingError::ModelError(format!("Failed to read i16: {}", e)))?;
                Ok(GGUFMetadataValue::I16(i16::from_le_bytes(buf)))
            }
            4 => Ok(GGUFMetadataValue::U32(Self::read_u32(reader)?)),
            5 => Ok(GGUFMetadataValue::I32(Self::read_i32(reader)?)),
            6 => Ok(GGUFMetadataValue::F32(Self::read_f32(reader)?)),
            7 => {
                // Bool
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf).map_err(|e| {
                    BatchingError::ModelError(format!("Failed to read bool: {}", e))
                })?;
                Ok(GGUFMetadataValue::Bool(buf[0] != 0))
            }
            8 => Ok(GGUFMetadataValue::String(Self::read_string(reader)?)),
            9 => {
                // Array
                let element_type = Self::read_u32(reader)?;
                let len = Self::read_u64(reader)? as usize;
                let mut array = Vec::with_capacity(len);

                for _ in 0..len {
                    let value = match element_type {
                        4 => GGUFMetadataValue::U32(Self::read_u32(reader)?),
                        5 => GGUFMetadataValue::I32(Self::read_i32(reader)?),
                        6 => GGUFMetadataValue::F32(Self::read_f32(reader)?),
                        8 => GGUFMetadataValue::String(Self::read_string(reader)?),
                        10 => GGUFMetadataValue::U64(Self::read_u64(reader)?),
                        12 => GGUFMetadataValue::F64(Self::read_f64(reader)?),
                        _ => {
                            return Err(BatchingError::ModelError(format!(
                                "Unsupported array element type: {}",
                                element_type
                            )))
                        }
                    };
                    array.push(value);
                }
                Ok(GGUFMetadataValue::Array(array))
            }
            10 => Ok(GGUFMetadataValue::U64(Self::read_u64(reader)?)),
            11 => {
                // I64
                let mut buf = [0u8; 8];
                reader
                    .read_exact(&mut buf)
                    .map_err(|e| BatchingError::ModelError(format!("Failed to read i64: {}", e)))?;
                Ok(GGUFMetadataValue::I64(i64::from_le_bytes(buf)))
            }
            12 => Ok(GGUFMetadataValue::F64(Self::read_f64(reader)?)),
            _ => Err(BatchingError::ModelError(format!(
                "Unknown metadata value type: {}",
                value_type
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_type_sizes() {
        assert_eq!(GGUFTensorType::F32.block_size(), 1);
        assert_eq!(GGUFTensorType::F32.block_bytes(), 4);
        assert_eq!(GGUFTensorType::Q4_K.block_size(), 256);
        assert_eq!(GGUFTensorType::Q4_K.block_bytes(), 144);
    }

    #[test]
    fn test_tensor_info_calculations() {
        let info = GGUFTensorInfo {
            name: "test".to_string(),
            tensor_type: GGUFTensorType::F32,
            dimensions: vec![10, 20],
            offset: 0,
        };
        assert_eq!(info.num_elements(), 200);
        assert_eq!(info.size_bytes(), 800); // 200 * 4 bytes
    }
}
