//! Byte Pair Encoding (BPE) tokenizer for text processing.

use crate::error::{Error, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Token index entry for efficient lookup.
#[derive(Debug, Clone)]
struct TokenIndex {
    /// Token string
    str: String,
    /// Token ID
    id: usize,
}

/// Byte Pair Encoding (BPE) tokenizer.
///
/// Implements tokenization and detokenization using a trained BPE vocabulary.
/// Compatible with the tokenizers trained by the Python training code.
#[derive(Debug)]
pub struct Tokenizer {
    /// Token strings
    vocab: Vec<String>,
    /// Token scores (for BPE merges)
    vocab_scores: Vec<f32>,
    /// Sorted vocabulary for efficient lookup
    sorted_vocab: Option<Vec<TokenIndex>>,
    /// Maximum token length
    max_token_length: usize,
    /// Single-byte token representations
    byte_pieces: [String; 256],
}

impl Tokenizer {
    /// Load a tokenizer from a binary file.
    ///
    /// The binary format is created by the Python tokenizer.py export script.
    ///
    /// # Arguments
    /// * `tokenizer_path` - Path to the tokenizer binary file
    ///
    /// # Returns
    /// Loaded tokenizer
    pub fn from_file<P: AsRef<Path>>(tokenizer_path: P) -> Result<Self> {
        let file = File::open(tokenizer_path)?;
        let data = unsafe { Mmap::map(&file)? };

        let mut offset = 0;

        // Read max_token_length
        if offset + std::mem::size_of::<usize>() > data.len() {
            return Err(Error::Tokenizer("File too small for max_token_length".to_string()));
        }
        let max_token_length = unsafe {
            let ptr = data.as_ptr().add(offset) as *const usize;
            *ptr
        };
        offset += std::mem::size_of::<usize>();

        // Initialize vocab and scores
        let vocab_size = 32000; // Default vocab size, could be made configurable
        let mut vocab = Vec::with_capacity(vocab_size);
        let mut vocab_scores = Vec::with_capacity(vocab_size);

        // Read vocab entries
        for _ in 0..vocab_size {
            // Read score
            if offset + std::mem::size_of::<f32>() > data.len() {
                return Err(Error::Tokenizer("Unexpected end of file reading vocab scores".to_string()));
            }
            let score = unsafe {
                let ptr = data.as_ptr().add(offset) as *const f32;
                *ptr
            };
            vocab_scores.push(score);
            offset += std::mem::size_of::<f32>();

            // Read string length
            if offset + std::mem::size_of::<usize>() > data.len() {
                return Err(Error::Tokenizer("Unexpected end of file reading string length".to_string()));
            }
            let len = unsafe {
                let ptr = data.as_ptr().add(offset) as *const usize;
                *ptr
            };
            offset += std::mem::size_of::<usize>();

            // Read string
            if offset + len > data.len() {
                return Err(Error::Tokenizer("Unexpected end of file reading string".to_string()));
            }
            let string_bytes = &data[offset..offset + len];
            let token_string = String::from_utf8_lossy(string_bytes).to_string();
            vocab.push(token_string);
            offset += len;
        }

        // Initialize byte pieces
        let mut byte_pieces = [const { String::new() }; 256];
        for i in 0..256 {
            byte_pieces[i] = format!("<0x{:02X}>", i);
        }

        Ok(Self {
            vocab,
            vocab_scores,
            sorted_vocab: None,
            max_token_length,
            byte_pieces,
        })
    }

    /// Encode text into tokens.
    ///
    /// Converts input text into a sequence of token IDs using BPE encoding.
    ///
    /// # Arguments
    /// * `text` - Input text to tokenize
    /// * `bos` - Whether to prepend BOS (beginning of sequence) token
    /// * `eos` - Whether to append EOS (end of sequence) token
    ///
    /// # Returns
    /// Vector of token IDs
    pub fn encode(&mut self, text: &str, bos: bool, eos: bool) -> Vec<usize> {
        let mut tokens = Vec::new();

        if text.is_empty() {
            if bos {
                tokens.push(1); // BOS token
            }
            if eos {
                tokens.push(2); // EOS token
            }
            return tokens;
        }

        // Initialize sorted vocab for lookup
        if self.sorted_vocab.is_none() {
            let mut sorted_vocab = Vec::with_capacity(self.vocab.len());
            for (id, token) in self.vocab.iter().enumerate() {
                sorted_vocab.push(TokenIndex {
                    str: token.clone(),
                    id,
                });
            }
            sorted_vocab.sort_by(|a, b| a.str.cmp(&b.str));
            self.sorted_vocab = Some(sorted_vocab);
        }

        // Add BOS token if requested
        if bos {
            tokens.push(1);
        }

        // Add dummy prefix token (following sentencepiece convention)
        if let Some(dummy_id) = self.str_lookup(" ") {
            tokens.push(dummy_id);
        }

        // Process UTF-8 bytes
        let mut str_buffer = String::with_capacity(self.max_token_length * 2 + 2);
        let mut str_len = 0;
        let text_bytes = text.as_bytes();

        for &byte in text_bytes {
            // Reset buffer if current byte is ASCII or leading UTF-8 byte
            if (byte & 0xC0) != 0x80 {
                str_len = 0;
            }

            // Append byte to buffer
            if str_len < self.max_token_length * 2 + 1 {
                str_buffer.push(byte as char);
                str_len += 1;
            }

            // Process complete UTF-8 sequence
            if (byte & 0xC0) != 0x80 {
                // Try to find this sequence in vocab
                if let Some(id) = self.str_lookup(&str_buffer) {
                    tokens.push(id);
                } else {
                    // Byte fallback: encode each byte as separate token
                    for &b in str_buffer.as_bytes() {
                        tokens.push(b as usize + 3); // +3 because first 3 tokens are special
                    }
                }
                str_buffer.clear();
                str_len = 0;
            }
        }

        // BPE merge loop
        let mut i = 0;
        while i < tokens.len() - 1 {
            let mut best_score = -f32::INFINITY;
            let mut best_id = None;
            let mut best_idx = None;

            // Look for merge candidates
            for j in i..(tokens.len() - 1).min(i + 20) { // Limit search window for efficiency
                let token1 = &self.vocab[tokens[j]];
                let token2 = &self.vocab[tokens[j + 1]];
                let merged = format!("{}{}", token1, token2);

                if let Some(id) = self.str_lookup(&merged) {
                    let score = self.vocab_scores[id];
                    if score > best_score {
                        best_score = score;
                        best_id = Some(id);
                        best_idx = Some(j);
                    }
                }
            }

            // Apply best merge if found
            if let (Some(id), Some(idx)) = (best_id, best_idx) {
                tokens[idx] = id;
                tokens.remove(idx + 1);
            } else {
                i += 1;
            }
        }

        // Add EOS token if requested
        if eos {
            tokens.push(2);
        }

        tokens
    }

    /// Decode tokens back into text.
    ///
    /// Converts a sequence of token IDs back into human-readable text.
    ///
    /// # Arguments
    /// * `tokens` - Token IDs to decode
    ///
    /// # Returns
    /// Decoded text
    pub fn decode(&self, tokens: &[usize]) -> String {
        let mut result = String::new();

        for (i, &token) in tokens.iter().enumerate() {
            let piece = if token < self.vocab.len() {
                let vocab_token = &self.vocab[token];

                // Handle BOS token stripping
                if i == 0 && token == 1 && vocab_token.starts_with(' ') {
                    &vocab_token[1..]
                } else {
                    vocab_token
                }
            } else {
                // Handle byte fallback tokens
                let byte_val = token - 3;
                if byte_val < 256 {
                    &self.byte_pieces[byte_val]
                } else {
                    "<?>"
                }
            };

            self.safe_print(piece, &mut result);
        }

        result
    }

    /// Efficient string lookup in sorted vocabulary.
    fn str_lookup(&self, s: &str) -> Option<usize> {
        self.sorted_vocab.as_ref()?.binary_search_by(|ti| ti.str.as_str().cmp(s))
            .ok()
            .map(|idx| self.sorted_vocab.as_ref().unwrap()[idx].id)
    }

    /// Safe printing that handles control characters and unprintable bytes.
    fn safe_print(&self, piece: &str, output: &mut String) {
        if piece.is_empty() {
            return;
        }

        // Handle single-byte tokens (may be control characters)
        if piece.len() == 1 {
            let byte = piece.as_bytes()[0];
            if byte.is_ascii_graphic() || byte.is_ascii_whitespace() {
                output.push(byte as char);
            }
            // Skip unprintable characters
            return;
        }

        // Handle multi-byte UTF-8 sequences
        output.push_str(piece);
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get maximum token length.
    pub fn max_token_length(&self) -> usize {
        self.max_token_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_tokenizer_creation() {
        // Create a minimal test tokenizer
        let vocab = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "a".to_string(),
            "b".to_string(),
            "ab".to_string(),
        ];
        let vocab_scores = vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0];
        let max_token_length = 10;

        let mut byte_pieces = [const { String::new() }; 256];
        for i in 0..256 {
            byte_pieces[i] = format!("<0x{:02X}>", i);
        }

        let mut tokenizer = Tokenizer {
            vocab,
            vocab_scores,
            sorted_vocab: None,
            max_token_length,
            byte_pieces,
        };

        // Test encoding
        let tokens = tokenizer.encode("ab", false, false);
        assert!(!tokens.is_empty());

        // Test decoding
        let text = tokenizer.decode(&tokens);
        assert!(!text.is_empty());
    }

    #[test]
    fn test_basic_encoding_decoding() {
        let vocab = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
        ];
        let vocab_scores = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut byte_pieces = [const { String::new() }; 256];
        for i in 0..256 {
            byte_pieces[i] = format!("<0x{:02X}>", i);
        }

        let mut tokenizer = Tokenizer {
            vocab,
            vocab_scores,
            sorted_vocab: None,
            max_token_length: 10,
            byte_pieces,
        };

        let test_text = "abc";
        let tokens = tokenizer.encode(test_text, false, false);
        let decoded = tokenizer.decode(&tokens);

        // Basic check that encoding/decoding works
        assert!(!tokens.is_empty());
        assert!(!decoded.is_empty());
    }
}
