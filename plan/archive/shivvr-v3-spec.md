# shivvr v3: Semantic Memory Service with Embedding Inversion

## Overview

A containerized memory service that uses **embedding inversion** for both boundary detection and text cleaning. Instead of heuristic boundary detection, we use reconstruction quality as the signal: when an embedding inverts cleanly, the window is coherent; when it struggles, we're spanning a topic boundary.

**You throw messy text at it. It figures out boundaries. It stores clean text.**

---

## The Core Insight

```
┌─────────────────────────────────────────────────────────────────────┐
│  TRADITIONAL: Detect boundaries → Chunk → Embed → Store            │
│                                                                     │
│  CHONK v3:    Slide window → Embed → Invert → Measure quality      │
│               → Quality drops = boundary → Store clean inversions   │
└─────────────────────────────────────────────────────────────────────┘
```

The inversion model was trained on clean, coherent text. When you embed messy or boundary-spanning content:
- Coherent content → high-quality inversion → store it
- Boundary-spanning → confused inversion → split here

---

## Architecture

```
                              INGEST FLOW
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Raw Text                                                            │
│  "Uh, so the the vectors are like 300 numbers... [TOPIC] stocks fell"│
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Sliding Window Extractor                                            │
│  - Window size: ~512 tokens                                          │
│  - Stride: ~256 tokens (50% overlap)                                 │
│  - Outputs: [(start, end, text), ...]                                │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Embedder (bge-small-en-v1.5)                                        │
│  - Batch embed all windows                                           │
│  - Output: [(window, embedding), ...]                                │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Inverter (vec2text)                                                 │
│  - Hypothesis model: embedding → rough text                          │
│  - Corrector model: iterative refinement (N steps)                   │
│  - Output: [(window, embedding, inverted_text), ...]                 │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Quality Scorer                                                      │
│  - Re-embed inverted text                                            │
│  - Score = cosine_similarity(original_emb, reconstructed_emb)        │
│  - Output: [(window, embedding, inverted_text, score), ...]          │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Boundary Detector                                                   │
│  - Find windows with LOW reconstruction score                        │
│  - These span topic boundaries                                       │
│  - Merge adjacent HIGH-score windows into chunks                     │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Chunk Builder                                                       │
│  - For each coherent region: use inverted (clean) text               │
│  - Store: { clean_text, embedding, original_text (optional) }        │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  sled Storage                                                        │
│  - Persist chunks with embeddings                                    │
│  - Index by session                                                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Runtime | Rust + Tokio | Fast, safe, async |
| HTTP | axum | Modern, fast |
| Embedder | bge-small-en-v1.5 (ONNX) | 384 dims, fast, good quality |
| Inverter | vec2text (ONNX) | Hypothesis + Corrector |
| Storage | sled + bincode | Embedded, Rust-native |
| Vector ops | simsimd | SIMD cosine similarity |

---

## Models Required

| Model | Architecture | Size | Purpose |
|-------|--------------|------|---------|
| bge-small-en-v1.5 | BERT | 127MB | Embedding |
| bge-vec2text-hypothesis | T5-base | ~500MB | Initial inversion |
| bge-vec2text-corrector | T5-base | ~500MB | Iterative refinement |

**Note**: The vec2text models must be trained for bge-small. See Training section.

---

## Project Structure

```
shivvr/
├── Cargo.toml
├── Dockerfile
├── docker-compose.yml
├── src/
│   ├── main.rs              # Entry point
│   ├── api.rs               # HTTP handlers
│   ├── embedder.rs          # bge-small ONNX wrapper
│   ├── inverter.rs          # vec2text hypothesis + corrector
│   ├── chunker.rs           # Sliding window + boundary detection
│   ├── store.rs             # sled storage
│   ├── similarity.rs        # SIMD vector ops
│   └── types.rs             # Shared data structures
├── models/
│   ├── bge-small-en-v1.5.onnx
│   ├── bge-vec2text-hypothesis.onnx
│   └── bge-vec2text-corrector.onnx
└── scripts/
    └── train_inverter.py    # Training script for vec2text models
```

---

## Cargo.toml

```toml
[package]
name = "shivvr"
version = "3.0.0"
edition = "2021"

[dependencies]
# Async runtime
tokio = { version = "1", features = ["full"] }

# HTTP
axum = "0.7"
tower-http = { version = "0.5", features = ["cors", "trace"] }

# ML Inference
ort = { version = "2.0.0-rc.9", features = ["download-binaries"] }
tokenizers = "0.21"
ndarray = "0.16"

# Vector ops
simsimd = "0.5"

# Storage
sled = "0.34"
bincode = "1.3"

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Utils
uuid = { version = "1", features = ["v4", "v7"] }
chrono = { version = "0.4", features = ["serde"] }
anyhow = "1"
thiserror = "2"
tracing = "0.1"
tracing-subscriber = "0.3"
fastrand = "2"

[profile.release]
lto = true
codegen-units = 1
panic = "abort"
```

---

## src/types.rs

```rust
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A processed chunk with clean inverted text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    
    /// Clean text from inversion (what we search/return)
    pub text: String,
    
    /// Original raw text (optional, for debugging)
    pub original_text: Option<String>,
    
    /// Embedding of the clean text
    pub embedding: Vec<f32>,
    
    /// Reconstruction quality score (0.0 - 1.0)
    pub quality_score: f32,
    
    /// Token count of clean text
    pub token_count: usize,
    
    /// Byte range in original document
    pub start_byte: usize,
    pub end_byte: usize,
    
    /// Source identifier
    pub source: Option<String>,
    
    /// User metadata
    pub metadata: serde_json::Value,
    
    pub created_at: DateTime<Utc>,
}

/// Session metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub last_ingested: DateTime<Utc>,
    pub chunk_count: usize,
    pub total_tokens: usize,
}

/// A window extracted from the document
#[derive(Debug, Clone)]
pub struct Window {
    pub start_byte: usize,
    pub end_byte: usize,
    pub text: String,
    pub token_count: usize,
}

/// Window with embedding
#[derive(Debug, Clone)]
pub struct EmbeddedWindow {
    pub window: Window,
    pub embedding: Vec<f32>,
}

/// Window with inversion results
#[derive(Debug, Clone)]
pub struct InvertedWindow {
    pub window: Window,
    pub original_embedding: Vec<f32>,
    pub inverted_text: String,
    pub reconstructed_embedding: Vec<f32>,
    pub quality_score: f32,
}

/// Configuration for the chunking pipeline
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Target window size in tokens
    pub window_tokens: usize,
    
    /// Stride between windows (overlap = window_tokens - stride)
    pub stride_tokens: usize,
    
    /// Number of correction steps for inversion
    pub correction_steps: usize,
    
    /// Quality threshold - windows below this are boundaries
    pub quality_threshold: f32,
    
    /// Minimum chunk size in tokens
    pub min_chunk_tokens: usize,
    
    /// Whether to store original text alongside clean
    pub store_original: bool,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            window_tokens: 512,
            stride_tokens: 256,
            correction_steps: 10,
            quality_threshold: 0.85,
            min_chunk_tokens: 50,
            store_original: false,
        }
    }
}

/// Search result
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub chunk_id: String,
    pub score: f32,
    pub text: String,
    pub quality_score: f32,
    pub source: Option<String>,
    pub metadata: serde_json::Value,
}
```

---

## src/embedder.rs

```rust
use anyhow::Result;
use ndarray::{Array2, Axis};
use ort::{GraphOptimizationLevel, Session};
use tokenizers::Tokenizer;

pub struct Embedder {
    session: Session,
    tokenizer: Tokenizer,
    pub dimensions: usize,
}

impl Embedder {
    pub fn new(model_path: &str, tokenizer_name: &str) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        let tokenizer = Tokenizer::from_pretrained(tokenizer_name, None)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            session,
            tokenizer,
            dimensions: 384,
        })
    }

    /// Embed single text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let batch = self.embed_batch(&[text])?;
        Ok(batch.into_iter().next().unwrap())
    }

    /// Embed batch of texts
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(512);

        let batch_size = texts.len();

        let mut input_ids = vec![0i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];
        let mut token_type_ids = vec![0i64; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            for (j, (&id, &m)) in ids.iter().zip(mask.iter()).enumerate() {
                if j >= max_len {
                    break;
                }
                input_ids[i * max_len + j] = id as i64;
                attention_mask[i * max_len + j] = m as i64;
            }
        }

        let input_ids = Array2::from_shape_vec((batch_size, max_len), input_ids)?;
        let attention_mask = Array2::from_shape_vec((batch_size, max_len), attention_mask)?;
        let token_type_ids = Array2::from_shape_vec((batch_size, max_len), token_type_ids)?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
            "token_type_ids" => token_type_ids,
        ]?)?;

        // Mean pooling
        let hidden = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()?
            .to_owned();
        
        let shape = hidden.shape();
        let hidden = hidden.into_shape((shape[0], shape[1], shape[2]))?;

        let pooled: Vec<Vec<f32>> = hidden
            .axis_iter(Axis(0))
            .map(|seq| {
                let mean = seq.mean_axis(Axis(0)).unwrap();
                l2_normalize(mean.to_vec())
            })
            .collect();

        Ok(pooled)
    }

    /// Count tokens
    pub fn count_tokens(&self, text: &str) -> usize {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().len())
            .unwrap_or(0)
    }

    /// Tokenize and return token boundaries for window extraction
    pub fn tokenize_with_offsets(&self, text: &str) -> Result<Vec<(usize, usize)>> {
        let encoding = self.tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        
        Ok(encoding
            .get_offsets()
            .iter()
            .map(|&(start, end)| (start, end))
            .collect())
    }
}

fn l2_normalize(v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.into_iter().map(|x| x / norm).collect()
    } else {
        v
    }
}
```

---

## src/inverter.rs

```rust
use anyhow::Result;
use ndarray::Array2;
use ort::{GraphOptimizationLevel, Session};
use tokenizers::Tokenizer;

/// Vec2Text inverter with hypothesis and corrector models
pub struct Inverter {
    hypothesis_session: Session,
    corrector_session: Session,
    tokenizer: Tokenizer,
    embedding_dim: usize,
}

impl Inverter {
    pub fn new(
        hypothesis_path: &str,
        corrector_path: &str,
        tokenizer_name: &str,
        embedding_dim: usize,
    ) -> Result<Self> {
        let hypothesis_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(hypothesis_path)?;

        let corrector_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(corrector_path)?;

        // T5 tokenizer for the inverter models
        let tokenizer = Tokenizer::from_pretrained(tokenizer_name, None)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            hypothesis_session,
            corrector_session,
            tokenizer,
            embedding_dim,
        })
    }

    /// Invert a single embedding to text
    pub fn invert(&self, embedding: &[f32], num_corrections: usize) -> Result<String> {
        // Step 0: Generate initial hypothesis
        let mut hypothesis = self.generate_hypothesis(embedding)?;

        // Steps 1-N: Iterative correction
        for _ in 0..num_corrections {
            hypothesis = self.correct(&hypothesis, embedding)?;
        }

        Ok(hypothesis)
    }

    /// Invert batch of embeddings
    pub fn invert_batch(
        &self,
        embeddings: &[Vec<f32>],
        num_corrections: usize,
    ) -> Result<Vec<String>> {
        // Generate hypotheses for all embeddings
        let mut hypotheses = self.generate_hypothesis_batch(embeddings)?;

        // Iterative correction
        for _ in 0..num_corrections {
            hypotheses = self.correct_batch(&hypotheses, embeddings)?;
        }

        Ok(hypotheses)
    }

    /// Generate initial hypothesis from embedding
    fn generate_hypothesis(&self, embedding: &[f32]) -> Result<String> {
        let batch = self.generate_hypothesis_batch(&[embedding.to_vec()])?;
        Ok(batch.into_iter().next().unwrap())
    }

    fn generate_hypothesis_batch(&self, embeddings: &[Vec<f32>]) -> Result<Vec<String>> {
        let batch_size = embeddings.len();
        
        // Prepare embedding input [batch_size, embedding_dim]
        let flat: Vec<f32> = embeddings.iter().flatten().copied().collect();
        let embedding_input = Array2::from_shape_vec(
            (batch_size, self.embedding_dim),
            flat,
        )?;

        // Run hypothesis model
        let outputs = self.hypothesis_session.run(ort::inputs![
            "embeddings" => embedding_input,
        ]?)?;

        // Decode output tokens
        let output_ids = outputs["output_ids"]
            .try_extract_tensor::<i64>()?
            .to_owned();

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let ids: Vec<u32> = output_ids
                .slice(ndarray::s![i, ..])
                .iter()
                .map(|&x| x as u32)
                .filter(|&x| x != 0) // Remove padding
                .collect();
            
            let text = self.tokenizer
                .decode(&ids, true)
                .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?;
            
            results.push(text);
        }

        Ok(results)
    }

    /// Correct hypothesis given target embedding
    fn correct(&self, hypothesis: &str, target_embedding: &[f32]) -> Result<String> {
        let batch = self.correct_batch(&[hypothesis.to_string()], &[target_embedding.to_vec()])?;
        Ok(batch.into_iter().next().unwrap())
    }

    fn correct_batch(
        &self,
        hypotheses: &[String],
        target_embeddings: &[Vec<f32>],
    ) -> Result<Vec<String>> {
        let batch_size = hypotheses.len();

        // Tokenize hypotheses
        let hypothesis_encodings = self.tokenizer
            .encode_batch(hypotheses.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let max_len = hypothesis_encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(128);

        // Prepare hypothesis input IDs
        let mut hypothesis_ids = vec![0i64; batch_size * max_len];
        let mut hypothesis_mask = vec![0i64; batch_size * max_len];

        for (i, encoding) in hypothesis_encodings.iter().enumerate() {
            for (j, (&id, &m)) in encoding.get_ids().iter()
                .zip(encoding.get_attention_mask().iter())
                .enumerate()
            {
                if j >= max_len { break; }
                hypothesis_ids[i * max_len + j] = id as i64;
                hypothesis_mask[i * max_len + j] = m as i64;
            }
        }

        let hypothesis_ids = Array2::from_shape_vec((batch_size, max_len), hypothesis_ids)?;
        let hypothesis_mask = Array2::from_shape_vec((batch_size, max_len), hypothesis_mask)?;

        // Prepare target embeddings
        let flat: Vec<f32> = target_embeddings.iter().flatten().copied().collect();
        let target_emb = Array2::from_shape_vec((batch_size, self.embedding_dim), flat)?;

        // Run corrector
        let outputs = self.corrector_session.run(ort::inputs![
            "hypothesis_ids" => hypothesis_ids,
            "hypothesis_mask" => hypothesis_mask,
            "target_embedding" => target_emb,
        ]?)?;

        // Decode outputs
        let output_ids = outputs["output_ids"]
            .try_extract_tensor::<i64>()?
            .to_owned();

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let ids: Vec<u32> = output_ids
                .slice(ndarray::s![i, ..])
                .iter()
                .map(|&x| x as u32)
                .filter(|&x| x != 0)
                .collect();
            
            let text = self.tokenizer
                .decode(&ids, true)
                .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?;
            
            results.push(text);
        }

        Ok(results)
    }
}
```

---

## src/similarity.rs

```rust
use simsimd::SpatialSimilarity;

/// Cosine similarity (SIMD accelerated)
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    f32::cosine(a, b).unwrap_or(0.0)
}

/// Find top-k most similar
pub fn top_k(query: &[f32], embeddings: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut scores: Vec<(usize, f32)> = embeddings
        .iter()
        .enumerate()
        .map(|(i, emb)| (i, cosine_similarity(query, emb)))
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(k);
    scores
}

/// Calculate reconstruction quality
/// High score = embedding inverts cleanly = coherent content
/// Low score = inversion struggles = boundary or noise
#[inline]
pub fn reconstruction_quality(original: &[f32], reconstructed: &[f32]) -> f32 {
    cosine_similarity(original, reconstructed)
}
```

---

## src/chunker.rs

```rust
use crate::embedder::Embedder;
use crate::inverter::Inverter;
use crate::similarity::reconstruction_quality;
use crate::types::{Chunk, ChunkConfig, EmbeddedWindow, InvertedWindow, Window};
use anyhow::Result;
use std::sync::Arc;

pub struct Chunker {
    embedder: Arc<Embedder>,
    inverter: Arc<Inverter>,
    config: ChunkConfig,
}

impl Chunker {
    pub fn new(
        embedder: Arc<Embedder>,
        inverter: Arc<Inverter>,
        config: ChunkConfig,
    ) -> Self {
        Self {
            embedder,
            inverter,
            config,
        }
    }

    /// Main entry point: chunk text using inversion-based boundary detection
    pub fn chunk(
        &self,
        text: &str,
        source: Option<String>,
        metadata: serde_json::Value,
    ) -> Result<Vec<Chunk>> {
        // Handle tiny inputs
        let token_count = self.embedder.count_tokens(text);
        if token_count <= self.config.window_tokens {
            return self.single_chunk(text, source, metadata);
        }

        // Step 1: Extract sliding windows
        let windows = self.extract_windows(text)?;
        
        if windows.is_empty() {
            return self.single_chunk(text, source, metadata);
        }

        // Step 2: Embed all windows
        let embedded = self.embed_windows(&windows)?;

        // Step 3: Invert all windows
        let inverted = self.invert_windows(&embedded)?;

        // Step 4: Detect boundaries using quality scores
        let boundaries = self.detect_boundaries(&inverted);

        // Step 5: Build chunks from coherent regions
        self.build_chunks(&inverted, &boundaries, text, source, metadata)
    }

    /// Extract sliding windows from text
    fn extract_windows(&self, text: &str) -> Result<Vec<Window>> {
        let offsets = self.embedder.tokenize_with_offsets(text)?;
        
        if offsets.is_empty() {
            return Ok(vec![]);
        }

        let mut windows = Vec::new();
        let mut start_token = 0;

        while start_token < offsets.len() {
            let end_token = (start_token + self.config.window_tokens).min(offsets.len());
            
            let start_byte = offsets[start_token].0;
            let end_byte = offsets[end_token - 1].1;
            
            let window_text = &text[start_byte..end_byte];
            
            windows.push(Window {
                start_byte,
                end_byte,
                text: window_text.to_string(),
                token_count: end_token - start_token,
            });

            start_token += self.config.stride_tokens;
            
            if end_token >= offsets.len() {
                break;
            }
        }

        Ok(windows)
    }

    /// Embed all windows
    fn embed_windows(&self, windows: &[Window]) -> Result<Vec<EmbeddedWindow>> {
        let texts: Vec<&str> = windows.iter().map(|w| w.text.as_str()).collect();
        let embeddings = self.embedder.embed_batch(&texts)?;

        Ok(windows
            .iter()
            .zip(embeddings.into_iter())
            .map(|(window, embedding)| EmbeddedWindow {
                window: window.clone(),
                embedding,
            })
            .collect())
    }

    /// Invert all windows and calculate quality scores
    fn invert_windows(&self, embedded: &[EmbeddedWindow]) -> Result<Vec<InvertedWindow>> {
        let embeddings: Vec<Vec<f32>> = embedded.iter()
            .map(|e| e.embedding.clone())
            .collect();

        // Batch invert
        let inverted_texts = self.inverter.invert_batch(
            &embeddings,
            self.config.correction_steps,
        )?;

        // Re-embed inverted texts to get reconstruction quality
        let inverted_refs: Vec<&str> = inverted_texts.iter().map(|s| s.as_str()).collect();
        let reconstructed_embeddings = self.embedder.embed_batch(&inverted_refs)?;

        // Calculate quality scores
        let mut results = Vec::with_capacity(embedded.len());
        for (i, ew) in embedded.iter().enumerate() {
            let quality = reconstruction_quality(
                &ew.embedding,
                &reconstructed_embeddings[i],
            );

            results.push(InvertedWindow {
                window: ew.window.clone(),
                original_embedding: ew.embedding.clone(),
                inverted_text: inverted_texts[i].clone(),
                reconstructed_embedding: reconstructed_embeddings[i].clone(),
                quality_score: quality,
            });
        }

        Ok(results)
    }

    /// Detect boundaries using quality score drops
    fn detect_boundaries(&self, inverted: &[InvertedWindow]) -> Vec<usize> {
        if inverted.is_empty() {
            return vec![];
        }

        let mut boundaries = vec![0];

        for i in 1..inverted.len() {
            let prev_quality = inverted[i - 1].quality_score;
            let curr_quality = inverted[i].quality_score;

            let is_low_quality = curr_quality < self.config.quality_threshold;
            let quality_dropped = prev_quality - curr_quality > 0.1;

            if is_low_quality || quality_dropped {
                if i + 1 < inverted.len() && inverted[i + 1].quality_score >= self.config.quality_threshold {
                    boundaries.push(i + 1);
                }
            }
        }

        boundaries
    }

    /// Build chunks from coherent regions
    fn build_chunks(
        &self,
        inverted: &[InvertedWindow],
        boundaries: &[usize],
        original_text: &str,
        source: Option<String>,
        metadata: serde_json::Value,
    ) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();

        for (i, &start_idx) in boundaries.iter().enumerate() {
            let end_idx = boundaries.get(i + 1).copied().unwrap_or(inverted.len());
            
            if start_idx >= end_idx {
                continue;
            }

            let best_window = inverted[start_idx..end_idx]
                .iter()
                .max_by(|a, b| a.quality_score.partial_cmp(&b.quality_score).unwrap())
                .unwrap();

            let start_byte = inverted[start_idx].window.start_byte;
            let end_byte = inverted[end_idx - 1].window.end_byte;

            let original_slice = if self.config.store_original {
                Some(original_text[start_byte..end_byte].to_string())
            } else {
                None
            };

            let clean_text = &best_window.inverted_text;
            let embedding = best_window.reconstructed_embedding.clone();

            chunks.push(Chunk {
                id: format!("chunk-{}", uuid::Uuid::now_v7()),
                text: clean_text.clone(),
                original_text: original_slice,
                embedding,
                quality_score: best_window.quality_score,
                token_count: self.embedder.count_tokens(clean_text),
                start_byte,
                end_byte,
                source: source.clone(),
                metadata: metadata.clone(),
                created_at: chrono::Utc::now(),
            });
        }

        let chunks: Vec<Chunk> = chunks
            .into_iter()
            .filter(|c| c.token_count >= self.config.min_chunk_tokens)
            .collect();

        Ok(chunks)
    }

    /// Handle small documents
    fn single_chunk(
        &self,
        text: &str,
        source: Option<String>,
        metadata: serde_json::Value,
    ) -> Result<Vec<Chunk>> {
        let embedding = self.embedder.embed(text)?;
        let inverted = self.inverter.invert(&embedding, self.config.correction_steps)?;
        let reconstructed_emb = self.embedder.embed(&inverted)?;
        let quality = reconstruction_quality(&embedding, &reconstructed_emb);

        Ok(vec![Chunk {
            id: format!("chunk-{}", uuid::Uuid::now_v7()),
            text: inverted,
            original_text: if self.config.store_original {
                Some(text.to_string())
            } else {
                None
            },
            embedding: reconstructed_emb,
            quality_score: quality,
            token_count: self.embedder.count_tokens(text),
            start_byte: 0,
            end_byte: text.len(),
            source,
            metadata,
            created_at: chrono::Utc::now(),
        }])
    }
}
```

---

## src/store.rs

```rust
use crate::types::{Chunk, SessionMeta};
use anyhow::Result;
use sled::Db;

pub struct Store {
    db: Db,
}

impl Store {
    pub fn open(path: &str) -> Result<Self> {
        let db = sled::open(path)?;
        Ok(Self { db })
    }

    pub fn add_chunks(&self, session_id: &str, chunks: Vec<Chunk>) -> Result<()> {
        let batch_size = chunks.len();
        let tokens: usize = chunks.iter().map(|c| c.token_count).sum();

        for chunk in &chunks {
            let key = format!("chunks:{}:{}", session_id, chunk.id);
            let value = bincode::serialize(chunk)?;
            self.db.insert(key.as_bytes(), value)?;
        }

        let meta_key = format!("meta:{}", session_id);
        let meta = match self.db.get(meta_key.as_bytes())? {
            Some(bytes) => {
                let mut meta: SessionMeta = bincode::deserialize(&bytes)?;
                meta.last_ingested = chrono::Utc::now();
                meta.chunk_count += batch_size;
                meta.total_tokens += tokens;
                meta
            }
            None => {
                self.add_session_to_list(session_id)?;
                SessionMeta {
                    id: session_id.to_string(),
                    created_at: chrono::Utc::now(),
                    last_ingested: chrono::Utc::now(),
                    chunk_count: batch_size,
                    total_tokens: tokens,
                }
            }
        };
        self.db.insert(meta_key.as_bytes(), bincode::serialize(&meta)?)?;
        self.db.flush()?;
        Ok(())
    }

    pub fn get_chunks(&self, session_id: &str) -> Result<Vec<Chunk>> {
        let prefix = format!("chunks:{}:", session_id);
        let mut chunks = Vec::new();
        for item in self.db.scan_prefix(prefix.as_bytes()) {
            let (_, value) = item?;
            let chunk: Chunk = bincode::deserialize(&value)?;
            chunks.push(chunk);
        }
        Ok(chunks)
    }

    pub fn get_session_meta(&self, session_id: &str) -> Result<Option<SessionMeta>> {
        let key = format!("meta:{}", session_id);
        match self.db.get(key.as_bytes())? {
            Some(bytes) => Ok(Some(bincode::deserialize(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn delete_session(&self, session_id: &str) -> Result<usize> {
        let prefix = format!("chunks:{}:", session_id);
        let mut deleted = 0;
        for item in self.db.scan_prefix(prefix.as_bytes()) {
            let (key, _) = item?;
            self.db.remove(key)?;
            deleted += 1;
        }
        let meta_key = format!("meta:{}", session_id);
        self.db.remove(meta_key.as_bytes())?;
        self.remove_session_from_list(session_id)?;
        self.db.flush()?;
        Ok(deleted)
    }

    pub fn list_sessions(&self) -> Result<Vec<SessionMeta>> {
        let session_ids = self.get_session_list()?;
        let mut sessions = Vec::new();
        for id in session_ids {
            if let Some(meta) = self.get_session_meta(&id)? {
                sessions.push(meta);
            }
        }
        Ok(sessions)
    }

    pub fn get_sources(&self, session_id: &str) -> Result<Vec<String>> {
        let chunks = self.get_chunks(session_id)?;
        let sources: std::collections::HashSet<String> = chunks
            .iter()
            .filter_map(|c| c.source.clone())
            .collect();
        Ok(sources.into_iter().collect())
    }

    pub fn total_chunks(&self) -> Result<usize> {
        let sessions = self.list_sessions()?;
        Ok(sessions.iter().map(|s| s.chunk_count).sum())
    }

    fn get_session_list(&self) -> Result<Vec<String>> {
        match self.db.get(b"sessions")? {
            Some(bytes) => Ok(bincode::deserialize(&bytes)?),
            None => Ok(Vec::new()),
        }
    }

    fn add_session_to_list(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.get_session_list()?;
        if !sessions.contains(&session_id.to_string()) {
            sessions.push(session_id.to_string());
            self.db.insert(b"sessions", bincode::serialize(&sessions)?)?;
        }
        Ok(())
    }

    fn remove_session_from_list(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.get_session_list()?;
        sessions.retain(|s| s != session_id);
        self.db.insert(b"sessions", bincode::serialize(&sessions)?)?;
        Ok(())
    }
}
```

---

## src/api.rs

```rust
use crate::chunker::Chunker;
use crate::embedder::Embedder;
use crate::similarity::top_k;
use crate::store::Store;
use crate::types::SearchResult;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub struct AppState {
    pub store: Arc<Store>,
    pub chunker: Arc<Chunker>,
    pub embedder: Arc<Embedder>,
    pub start_time: std::time::Instant,
}

#[derive(Deserialize)]
pub struct IngestRequest {
    pub text: String,
    pub source: Option<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Serialize)]
pub struct IngestResponse {
    pub chunks_created: usize,
    pub tokens_processed: usize,
    pub avg_quality: f32,
    pub time_ms: u64,
}

#[derive(Deserialize)]
pub struct SearchQuery {
    pub q: String,
    #[serde(default = "default_n")]
    pub n: usize,
    #[serde(default)]
    pub min_quality: Option<f32>,
}

fn default_n() -> usize { 5 }

#[derive(Serialize)]
pub struct SearchResponse {
    pub query: String,
    pub results: Vec<SearchResult>,
    pub time_ms: u64,
}

#[derive(Serialize)]
pub struct SessionInfoResponse {
    pub session: String,
    pub chunks: usize,
    pub total_tokens: usize,
    pub sources: Vec<String>,
    pub created_at: String,
    pub last_ingested: String,
}

#[derive(Serialize)]
pub struct DeleteResponse {
    pub deleted_chunks: usize,
    pub session: String,
}

#[derive(Serialize)]
pub struct ListSessionsResponse {
    pub sessions: Vec<SessionItem>,
}

#[derive(Serialize)]
pub struct SessionItem {
    pub id: String,
    pub chunks: usize,
    pub last_active: String,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
    pub sessions: usize,
    pub total_chunks: usize,
    pub uptime_seconds: u64,
}

async fn ingest(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
    Json(req): Json<IngestRequest>,
) -> Result<Json<IngestResponse>, StatusCode> {
    let start = std::time::Instant::now();

    let chunks = state.chunker
        .chunk(&req.text, req.source, req.metadata)
        .map_err(|e| {
            tracing::error!("Chunking failed: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let chunks_created = chunks.len();
    let tokens_processed: usize = chunks.iter().map(|c| c.token_count).sum();
    let avg_quality: f32 = if chunks.is_empty() {
        0.0
    } else {
        chunks.iter().map(|c| c.quality_score).sum::<f32>() / chunks.len() as f32
    };

    state.store.add_chunks(&session_id, chunks).map_err(|e| {
        tracing::error!("Storage failed: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(IngestResponse {
        chunks_created,
        tokens_processed,
        avg_quality,
        time_ms: start.elapsed().as_millis() as u64,
    }))
}

async fn search(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
    Query(query): Query<SearchQuery>,
) -> Result<Json<SearchResponse>, StatusCode> {
    let start = std::time::Instant::now();

    let query_embedding = state.embedder.embed(&query.q).map_err(|e| {
        tracing::error!("Embed failed: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let mut chunks = state.store.get_chunks(&session_id).map_err(|e| {
        tracing::error!("Storage read failed: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    if let Some(min_q) = query.min_quality {
        chunks.retain(|c| c.quality_score >= min_q);
    }

    if chunks.is_empty() {
        return Ok(Json(SearchResponse {
            query: query.q,
            results: vec![],
            time_ms: start.elapsed().as_millis() as u64,
        }));
    }

    let embeddings: Vec<Vec<f32>> = chunks.iter().map(|c| c.embedding.clone()).collect();
    let top = top_k(&query_embedding, &embeddings, query.n);

    let results: Vec<SearchResult> = top
        .into_iter()
        .map(|(idx, score)| SearchResult {
            chunk_id: chunks[idx].id.clone(),
            score,
            text: chunks[idx].text.clone(),
            quality_score: chunks[idx].quality_score,
            source: chunks[idx].source.clone(),
            metadata: chunks[idx].metadata.clone(),
        })
        .collect();

    Ok(Json(SearchResponse {
        query: query.q,
        results,
        time_ms: start.elapsed().as_millis() as u64,
    }))
}

async fn session_info(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> Result<Json<SessionInfoResponse>, StatusCode> {
    let meta = state.store
        .get_session_meta(&session_id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let sources = state.store
        .get_sources(&session_id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(SessionInfoResponse {
        session: meta.id,
        chunks: meta.chunk_count,
        total_tokens: meta.total_tokens,
        sources,
        created_at: meta.created_at.to_rfc3339(),
        last_ingested: meta.last_ingested.to_rfc3339(),
    }))
}

async fn delete_session(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> Result<Json<DeleteResponse>, StatusCode> {
    let deleted = state.store
        .delete_session(&session_id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(DeleteResponse {
        deleted_chunks: deleted,
        session: session_id,
    }))
}

async fn list_sessions(State(state): State<Arc<AppState>>) -> Json<ListSessionsResponse> {
    let sessions = state.store.list_sessions().unwrap_or_default();
    Json(ListSessionsResponse {
        sessions: sessions
            .into_iter()
            .map(|s| SessionItem {
                id: s.id,
                chunks: s.chunk_count,
                last_active: s.last_ingested.to_rfc3339(),
            })
            .collect(),
    })
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let sessions = state.store.list_sessions().unwrap_or_default();
    let total_chunks = state.store.total_chunks().unwrap_or(0);
    Json(HealthResponse {
        status: "ok".to_string(),
        model: "bge-small-en-v1.5 + vec2text".to_string(),
        sessions: sessions.len(),
        total_chunks,
        uptime_seconds: state.start_time.elapsed().as_secs(),
    })
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/memory", get(list_sessions))
        .route("/memory/{session_id}/ingest", post(ingest))
        .route("/memory/{session_id}/search", get(search))
        .route("/memory/{session_id}/info", get(session_info))
        .route("/memory/{session_id}", delete(delete_session))
        .with_state(state)
}
```

---

## src/main.rs

```rust
mod api;
mod chunker;
mod embedder;
mod inverter;
mod similarity;
mod store;
mod types;

use crate::api::AppState;
use crate::chunker::Chunker;
use crate::embedder::Embedder;
use crate::inverter::Inverter;
use crate::store::Store;
use crate::types::ChunkConfig;
use std::sync::Arc;
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let data_path = std::env::var("DATA_PATH").unwrap_or_else(|_| "/data/shivvr".to_string());
    
    let embedder_model = std::env::var("EMBEDDER_MODEL")
        .unwrap_or_else(|_| "/models/bge-small-en-v1.5.onnx".to_string());
    let hypothesis_model = std::env::var("HYPOTHESIS_MODEL")
        .unwrap_or_else(|_| "/models/bge-vec2text-hypothesis.onnx".to_string());
    let corrector_model = std::env::var("CORRECTOR_MODEL")
        .unwrap_or_else(|_| "/models/bge-vec2text-corrector.onnx".to_string());

    let config = ChunkConfig {
        window_tokens: std::env::var("WINDOW_TOKENS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(512),
        stride_tokens: std::env::var("STRIDE_TOKENS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(256),
        correction_steps: std::env::var("CORRECTION_STEPS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(10),
        quality_threshold: std::env::var("QUALITY_THRESHOLD")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(0.85),
        min_chunk_tokens: std::env::var("MIN_CHUNK_TOKENS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(50),
        store_original: std::env::var("STORE_ORIGINAL")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(false),
    };

    tracing::info!("Loading embedder...");
    let embedder = Arc::new(Embedder::new(&embedder_model, "BAAI/bge-small-en-v1.5")?);

    tracing::info!("Loading inverter...");
    let inverter = Arc::new(Inverter::new(
        &hypothesis_model,
        &corrector_model,
        "t5-base",
        embedder.dimensions,
    )?);

    tracing::info!("Opening database...");
    let store = Arc::new(Store::open(&data_path)?);

    let chunker = Arc::new(Chunker::new(embedder.clone(), inverter, config));

    let state = Arc::new(AppState {
        store,
        chunker,
        embedder,
        start_time: std::time::Instant::now(),
    });

    let app = api::router(state);
    let addr = format!("0.0.0.0:{}", port);
    tracing::info!("Starting shivvr v3 on {}", addr);

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
```

---

## Dockerfile

```dockerfile
FROM rust:1.83-slim-bookworm AS builder

RUN apt-get update && apt-get install -y pkg-config libssl-dev curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

COPY src ./src
RUN touch src/main.rs
RUN cargo build --release

RUN mkdir -p /models && \
    curl -L -o /models/bge-small-en-v1.5.onnx \
    "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx"

# NOTE: vec2text models must be trained and hosted separately
# curl -L -o /models/bge-vec2text-hypothesis.onnx "YOUR_URL"
# curl -L -o /models/bge-vec2text-corrector.onnx "YOUR_URL"

FROM gcr.io/distroless/cc-debian12

COPY --from=builder /app/target/release/shivvr /shivvr
COPY --from=builder /models /models

ENV PORT=8080
ENV DATA_PATH=/data/shivvr
ENV EMBEDDER_MODEL=/models/bge-small-en-v1.5.onnx
ENV HYPOTHESIS_MODEL=/models/bge-vec2text-hypothesis.onnx
ENV CORRECTOR_MODEL=/models/bge-vec2text-corrector.onnx

EXPOSE 8080
VOLUME ["/data"]
ENTRYPOINT ["/shivvr"]
```

---

## Training Vec2Text for bge-small

### Prerequisites

```bash
pip install vec2text transformers sentence-transformers torch
```

### Step 1: Train Hypothesis Model

```bash
python -m vec2text.run \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 128 \
  --max_seq_length 128 \
  --model_name_or_path t5-base \
  --dataset_name msmarco \
  --embedder_model_name BAAI/bge-small-en-v1.5 \
  --num_repeat_tokens 16 \
  --embedder_no_grad True \
  --num_train_epochs 100 \
  --experiment inversion \
  --learning_rate 0.001 \
  --output_dir ./models/bge-hypothesis \
  --bf16 1
```

### Step 2: Train Corrector Model

```bash
python -m vec2text.run \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --max_seq_length 128 \
  --model_name_or_path t5-base \
  --dataset_name msmarco \
  --embedder_model_name BAAI/bge-small-en-v1.5 \
  --experiment corrector \
  --corrector_model_alias ./models/bge-hypothesis \
  --learning_rate 0.001 \
  --output_dir ./models/bge-corrector \
  --bf16 1
```

### Step 3: Export to ONNX

```bash
python -c "
from transformers import T5ForConditionalGeneration
import torch

# Load and export (simplified - actual export needs custom handling)
model = T5ForConditionalGeneration.from_pretrained('./models/bge-hypothesis')
# ... export logic
"
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| PORT | 8080 | HTTP port |
| DATA_PATH | /data/shivvr | sled database directory |
| EMBEDDER_MODEL | /models/bge-small-en-v1.5.onnx | Embedder path |
| HYPOTHESIS_MODEL | /models/bge-vec2text-hypothesis.onnx | Hypothesis model |
| CORRECTOR_MODEL | /models/bge-vec2text-corrector.onnx | Corrector model |
| WINDOW_TOKENS | 512 | Window size |
| STRIDE_TOKENS | 256 | Window stride |
| CORRECTION_STEPS | 10 | Inversion iterations |
| QUALITY_THRESHOLD | 0.85 | Boundary threshold |
| STORE_ORIGINAL | false | Keep original text |

---

## What's Novel

1. **Inversion as boundary detection** — Quality drops at topic changes
2. **Automatic text cleaning** — Filler words removed by inversion
3. **Quality scores as metadata** — Filter low-confidence chunks
4. **No heuristics** — Model learned coherence from data

---

## Fallback: Start Without Inverter

If models aren't ready, implement fallback to simple chunking:

```rust
impl Chunker {
    pub fn chunk(&self, text: &str, ...) -> Result<Vec<Chunk>> {
        match &self.inverter {
            Some(inv) => self.chunk_with_inversion(text, inv, ...),
            None => self.chunk_simple(text, ...),  // sentence-based fallback
        }
    }
}
```

This lets you deploy and test before training completes.
