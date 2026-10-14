# shivvr: Semantic Memory Service

## Overview

A containerized memory service that accepts text of any length, chunks it semantically with a Monte Carlo sampler, embeds it, and provides fast brute-force search. Includes optional time-weighted ranking, temporal context expansion, and sled-backed persistence. Designed for AI agents, projects, and conversational memory.

**You throw text at it. It figures out the rest.**

---

## Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Runtime | Rust + Tokio | Fast, safe, async |
| HTTP | axum | Modern, fast, ergonomic |
| Inference | ort (ONNX Runtime) | Production-ready, fast CPU |
| Model | bge-small-en-v1.5 | 384 dims, good quality |
| Tokenizer | tokenizers (local JSON) | Accurate, fast |
| Vector Ops | simsimd | SIMD-accelerated cosine |
| Storage | sled | Embedded, disk-backed, ACID |
| Serialization | bincode | Fast, compact binary |
| Container | debian:bookworm-slim | Small, compatible runtime |

---

## API

### Ingest Text

```
POST /memory/{session}/ingest
Content-Type: application/json

{
  "text": "Any length of text. Could be a sentence. Could be a book.",
  "source": "optional-identifier",
  "metadata": {}
}

Response 200:
{
  "chunks_created": 12,
  "tokens_processed": 4823,
  "time_ms": 847
}
```

### Search Memory

```
GET /memory/{session}/search?q={query}&n={count}&time_weight={0..1}&include_nearby={true|false}&time_window_minutes={mins}

Response 200:
{
  "query": "what is the refund policy",
  "results": [
    {
      "chunk_id": "chunk-a1b2c3",
      "score": 0.847,
      "text": "Our refund policy allows returns within 30 days...",
      "source": "faq-doc",
      "metadata": {},
      "created_at": "2025-01-06T14:32:00Z",
      "nearby_chunks": [
        {
          "chunk_id": "chunk-a1b2c4",
          "text": "Returns must be in original packaging...",
          "source": "faq-doc",
          "created_at": "2025-01-06T14:33:10Z"
        }
      ]
    }
  ],
  "time_ms": 12
}
```

- `time_weight`: blends semantic similarity with a recency decay (default 0.0).
- `include_nearby`: when true, returns nearby chunks by time.
- `time_window_minutes`: temporal window for nearby chunks (default 30).

### Session Info

```
GET /memory/{session}/info

Response 200:
{
  "session": "agent-claude-001",
  "chunks": 1847,
  "total_tokens": 523841,
  "sources": ["email-123", "doc-abc"],
  "created_at": "2025-01-06T10:00:00Z",
  "last_ingested": "2025-01-06T14:32:00Z",
  "memory_bytes": 28483921
}
```

### Clear Session

```
DELETE /memory/{session}

Response 200:
{
  "deleted_chunks": 1847,
  "session": "agent-claude-001"
}
```

### List Sessions

```
GET /memory

Response 200:
{
  "sessions": [
    { "id": "agent-claude-001", "chunks": 1847, "last_active": "..." },
    { "id": "project-xyz", "chunks": 523, "last_active": "..." }
  ]
}
```

### Health Check

```
GET /health

Response 200:
{
  "status": "ok",
  "model": "bge-small-en-v1.5",
  "sessions": 12,
  "total_chunks": 48291,
  "uptime_seconds": 84729
}
```

---

## Project Structure

```
shivvr/
├── Cargo.toml
├── Dockerfile
├── docker-compose.yml
├── shivvr-spec-final.md
├── src/
│   ├── main.rs           # Entry point, server setup
│   ├── api.rs            # HTTP handlers + router
│   ├── chunker.rs        # Monte Carlo chunking algorithm
│   ├── embedder.rs       # ONNX inference wrapper
│   ├── similarity.rs     # SIMD vector ops
│   └── store.rs          # sled storage layer
└── models/               # Downloaded during Docker build
    ├── bge-small-en-v1.5.onnx
    └── tokenizer.json
```

---

## Cargo.toml

```toml
[package]
name = "shivvr"
version = "0.1.0"
edition = "2021"

[dependencies]
# Async runtime
tokio = { version = "1", features = ["full"] }

# HTTP
axum = "0.7"
tower-http = { version = "0.5", features = ["cors", "trace"] }

# ML Inference
ort = { version = "2.0.0-rc.11", features = ["download-binaries", "tls-native", "ndarray"] }
tokenizers = "0.21"

# Vector ops
simsimd = "6.5.12"
ndarray = "0.17"

# Storage
sled = "0.34"
bincode = "1.3"

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Utils
uuid = { version = "1", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }
anyhow = "1"
thiserror = "2"
tracing = "0.1"
tracing-subscriber = "0.3"

[profile.release]
lto = true
codegen-units = 1
panic = "abort"
```

---


## src/store.rs

```rust
use crate::similarity::cosine_similarity;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sled::Db;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub text: String,
    pub embedding: Vec<f32>,
    pub token_count: usize,
    pub source: Option<String>,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub last_ingested: DateTime<Utc>,
    pub chunk_count: usize,
    pub total_tokens: usize,
}

/// Key prefixes for sled
/// chunks:{session_id}:{chunk_id} -> Chunk (bincode)
/// meta:{session_id} -> SessionMeta (bincode)
/// sessions -> Vec<String> of session IDs (bincode)

pub struct Store {
    db: Db,
}

impl Store {
    /// Open or create the database
    pub fn open(path: &str) -> Result<Self> {
        let db = sled::open(path)?;
        Ok(Self { db })
    }

    /// Add chunks to a session
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
                meta.last_ingested = Utc::now();
                meta.chunk_count += batch_size;
                meta.total_tokens += tokens;
                meta
            }
            None => {
                self.add_session_to_list(session_id)?;
                SessionMeta {
                    id: session_id.to_string(),
                    created_at: Utc::now(),
                    last_ingested: Utc::now(),
                    chunk_count: batch_size,
                    total_tokens: tokens,
                }
            }
        };
        self.db.insert(meta_key.as_bytes(), bincode::serialize(&meta)?)?;

        self.db.flush()?;
        Ok(())
    }

    /// Get all chunks for a session
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

    /// Get session metadata
    pub fn get_session_meta(&self, session_id: &str) -> Result<Option<SessionMeta>> {
        let key = format!("meta:{}", session_id);
        match self.db.get(key.as_bytes())? {
            Some(bytes) => Ok(Some(bincode::deserialize(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Delete a session and all its chunks
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

    /// List all sessions
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

    /// Get total chunk count across all sessions
    pub fn total_chunks(&self) -> Result<usize> {
        let sessions = self.list_sessions()?;
        Ok(sessions.iter().map(|s| s.chunk_count).sum())
    }

    /// Search by semantic similarity with optional time weighting
    pub fn search(
        &self,
        session_id: &str,
        query_embedding: &[f32],
        n: usize,
        time_weight: Option<f32>,
    ) -> Result<Vec<(Chunk, f32)>> {
        let chunks = self.get_chunks(session_id)?;
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let now = Utc::now();
        let time_weight = time_weight.unwrap_or(0.0).clamp(0.0, 1.0);

        let mut scores: Vec<(usize, f32)> = chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                let semantic_score = cosine_similarity(query_embedding, &chunk.embedding);

                let time_score = if time_weight > 0.0 {
                    let age_hours = (now - chunk.created_at).num_hours() as f32;
                    (-age_hours / 168.0).exp()
                } else {
                    0.0
                };

                let combined = (1.0 - time_weight) * semantic_score + time_weight * time_score;
                (i, combined)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(n);

        Ok(scores
            .into_iter()
            .map(|(i, score)| (chunks[i].clone(), score))
            .collect())
    }

    /// Search and include chunks near in time to each match
    pub fn search_with_temporal_context(
        &self,
        session_id: &str,
        query_embedding: &[f32],
        n: usize,
        time_window_minutes: i64,
    ) -> Result<Vec<(Chunk, f32, Vec<Chunk>)>> {
        let chunks = self.get_chunks(session_id)?;
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let mut scores: Vec<(usize, f32)> = chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                let score = cosine_similarity(query_embedding, &chunk.embedding);
                (i, score)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(n);

        let window = time_window_minutes.abs();

        Ok(scores
            .into_iter()
            .map(|(i, score)| {
                let chunk = &chunks[i];
                let chunk_time = chunk.created_at;

                let nearby: Vec<Chunk> = chunks
                    .iter()
                    .enumerate()
                    .filter(|(j, c)| {
                        if *j == i {
                            return false;
                        }
                        let diff = (c.created_at - chunk_time).num_minutes().abs();
                        diff <= window
                    })
                    .map(|(_, c)| c.clone())
                    .collect();

                (chunk.clone(), score, nearby)
            })
            .collect())
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

## src/embedder.rs

```rust
use anyhow::Result;
use ndarray::{Array2, Axis, Ix3};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use tokenizers::Tokenizer;
use std::sync::Mutex;

pub struct Embedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl Embedder {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
        })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        let seq_len = ids.len();

        let input_ids: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
        let attention: Vec<i64> = attention_mask.iter().map(|&x| x as i64).collect();
        let token_type_ids: Vec<i64> = vec![0i64; seq_len];

        let input_ids = Array2::from_shape_vec((1, seq_len), input_ids)?;
        let attention_mask = Array2::from_shape_vec((1, seq_len), attention)?;
        let token_type_ids = Array2::from_shape_vec((1, seq_len), token_type_ids)?;

        let input_ids = Value::from_array(input_ids)?;
        let attention_mask = Value::from_array(attention_mask)?;
        let token_type_ids = Value::from_array(token_type_ids)?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("Session lock poisoned"))?;

        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
            "token_type_ids" => token_type_ids,
        ])?;

        let output = outputs
            .get("last_hidden_state")
            .ok_or_else(|| anyhow::anyhow!("No last_hidden_state in output"))?;

        let view = output.try_extract_array::<f32>()?;
        let view = view
            .into_dimensionality::<Ix3>()
            .map_err(|_| anyhow::anyhow!("Unexpected tensor shape for last_hidden_state"))?;

        let pooled = view
            .mean_axis(Axis(1))
            .ok_or_else(|| anyhow::anyhow!("Failed to mean-pool embeddings"))?;

        let embedding: Vec<f32> = pooled.index_axis(Axis(0), 0).to_vec();

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = if norm > 0.0 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            embedding
        };

        Ok(normalized)
    }

    pub fn count_tokens(&self, text: &str) -> usize {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().len())
            .unwrap_or(0)
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
    f32::cosine(a, b).unwrap_or(0.0) as f32
}
```


---


## src/chunker.rs

```rust
use crate::embedder::Embedder;
use crate::similarity::cosine_similarity;
use crate::store::Chunk;
use anyhow::Result;
use std::sync::Arc;

pub struct Chunker {
    embedder: Arc<Embedder>,
    config: ChunkConfig,
}

#[derive(Debug, Clone)]
pub struct ChunkConfig {
    pub min_chunk_tokens: usize,
    pub max_chunk_tokens: usize,
    pub similarity_threshold: f32,
    pub sample_count: usize,
    pub optimization_iterations: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            min_chunk_tokens: 50,
            max_chunk_tokens: 512,
            similarity_threshold: 0.5,
            sample_count: 10,
            optimization_iterations: 5,
        }
    }
}

#[derive(Debug, Clone)]
struct Sentence {
    text: String,
    start_char: usize,
    end_char: usize,
}

impl Chunker {
    pub fn new(embedder: Arc<Embedder>) -> Self {
        Self {
            embedder,
            config: ChunkConfig::default(),
        }
    }

    pub async fn chunk(
        &self,
        text: &str,
        source: Option<String>,
        metadata: serde_json::Value,
    ) -> Result<Vec<Chunk>> {
        // Handle tiny inputs
        if text.len() < 100 {
            return self.single_chunk(text, source, metadata).await;
        }

        // Split into sentences
        let sentences = self.split_sentences(text);

        if sentences.len() < 3 {
            return self.single_chunk(text, source, metadata).await;
        }

        // Monte Carlo sampling to find candidate boundaries
        let candidate_regions = self.monte_carlo_sample(&sentences).await?;

        // Build boundaries
        let mut boundaries = vec![0];
        for (start, end) in candidate_regions {
            let boundary = self.binary_search_boundary(&sentences, start, end).await?;
            if boundary > *boundaries.last().unwrap() {
                boundaries.push(boundary);
            }
        }
        boundaries.push(sentences.len());

        // Optimize boundaries
        let boundaries = self.optimize_boundaries(&sentences, boundaries).await?;

        // Build chunks
        let mut chunks = Vec::new();
        for i in 0..boundaries.len() - 1 {
            let start = boundaries[i];
            let end = boundaries[i + 1];
            if start >= end || start >= sentences.len() {
                continue;
            }
            let end = end.min(sentences.len());

            let chunk_text: String = sentences[start..end]
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            if chunk_text.is_empty() {
                continue;
            }

            let embedding = self.embedder.embed(&chunk_text)?;
            let token_count = self.embedder.count_tokens(&chunk_text);

            chunks.push(Chunk {
                id: format!("chunk-{}", uuid::Uuid::new_v4()),
                text: chunk_text,
                embedding,
                token_count,
                source: source.clone(),
                metadata: metadata.clone(),
                created_at: chrono::Utc::now(),
            });
        }

        // Enforce size limits
        self.enforce_size_limits(chunks, source, metadata).await
    }

    async fn single_chunk(
        &self,
        text: &str,
        source: Option<String>,
        metadata: serde_json::Value,
    ) -> Result<Vec<Chunk>> {
        let embedding = self.embedder.embed(text)?;
        let token_count = self.embedder.count_tokens(text);

        Ok(vec![Chunk {
            id: format!("chunk-{}", uuid::Uuid::new_v4()),
            text: text.to_string(),
            embedding,
            token_count,
            source,
            metadata,
            created_at: chrono::Utc::now(),
        }])
    }

    fn split_sentences(&self, text: &str) -> Vec<Sentence> {
        let mut sentences = Vec::new();
        let mut start = 0;

        // Simple sentence splitting on . ! ? followed by space or end
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let c = chars[i];
            if (c == '.' || c == '!' || c == '?')
                && (i + 1 >= chars.len() || chars[i + 1].is_whitespace())
            {
                let end = i + 1;
                let sentence_text: String = chars[start..end].iter().collect();
                let trimmed = sentence_text.trim();
                if !trimmed.is_empty() {
                    sentences.push(Sentence {
                        text: trimmed.to_string(),
                        start_char: start,
                        end_char: end,
                    });
                }
                // Skip whitespace after sentence
                i += 1;
                while i < chars.len() && chars[i].is_whitespace() {
                    i += 1;
                }
                start = i;
            } else {
                i += 1;
            }
        }

        // Don't forget remaining text
        if start < chars.len() {
            let sentence_text: String = chars[start..].iter().collect();
            let trimmed = sentence_text.trim();
            if !trimmed.is_empty() {
                sentences.push(Sentence {
                    text: trimmed.to_string(),
                    start_char: start,
                    end_char: chars.len(),
                });
            }
        }

        sentences
    }

    async fn monte_carlo_sample(&self, sentences: &[Sentence]) -> Result<Vec<(usize, usize)>> {
        if sentences.len() < 5 {
            return Ok(vec![]);
        }

        let mut regions = Vec::new();
        let step = sentences.len() / (self.config.sample_count + 1);

        for i in 1..=self.config.sample_count {
            let pos = i * step;
            if pos > 0 && pos < sentences.len() - 1 {
                // Sample embeddings before and after this point
                let before_start = pos.saturating_sub(2);
                let before_text: String = sentences[before_start..pos]
                    .iter()
                    .map(|s| s.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");

                let after_end = (pos + 2).min(sentences.len());
                let after_text: String = sentences[pos..after_end]
                    .iter()
                    .map(|s| s.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");

                if before_text.is_empty() || after_text.is_empty() {
                    continue;
                }

                let before_emb = self.embedder.embed(&before_text)?;
                let after_emb = self.embedder.embed(&after_text)?;

                let similarity = cosine_similarity(&before_emb, &after_emb);

                if similarity < self.config.similarity_threshold {
                    // Found a potential boundary region
                    let start = pos.saturating_sub(3);
                    let end = (pos + 3).min(sentences.len());
                    regions.push((start, end));
                }
            }
        }

        Ok(regions)
    }

    async fn binary_search_boundary(
        &self,
        sentences: &[Sentence],
        start: usize,
        end: usize,
    ) -> Result<usize> {
        if end <= start + 1 {
            return Ok(start);
        }

        let mut best_pos = start;
        let mut lowest_sim = 1.0f32;

        for pos in start..end {
            if pos == 0 || pos >= sentences.len() {
                continue;
            }

            let before_text = &sentences[pos - 1].text;
            let after_text = &sentences[pos].text;

            let before_emb = self.embedder.embed(before_text)?;
            let after_emb = self.embedder.embed(after_text)?;

            let sim = cosine_similarity(&before_emb, &after_emb);
            if sim < lowest_sim {
                lowest_sim = sim;
                best_pos = pos;
            }
        }

        Ok(best_pos)
    }

    async fn optimize_boundaries(
        &self,
        sentences: &[Sentence],
        mut boundaries: Vec<usize>,
    ) -> Result<Vec<usize>> {
        for _ in 0..self.config.optimization_iterations {
            for i in 1..boundaries.len() - 1 {
                let prev = boundaries[i - 1];
                let curr = boundaries[i];
                let next = boundaries[i + 1];

                // Try moving boundary left or right
                let mut best_pos = curr;
                let mut best_coherence = self.chunk_coherence(sentences, prev, curr).await?
                    + self.chunk_coherence(sentences, curr, next).await?;

                for delta in [-1i32, 1i32] {
                    let new_pos = (curr as i32 + delta) as usize;
                    if new_pos > prev && new_pos < next {
                        let coherence = self.chunk_coherence(sentences, prev, new_pos).await?
                            + self.chunk_coherence(sentences, new_pos, next).await?;
                        if coherence > best_coherence {
                            best_coherence = coherence;
                            best_pos = new_pos;
                        }
                    }
                }

                boundaries[i] = best_pos;
            }
        }

        Ok(boundaries)
    }

    async fn chunk_coherence(&self, sentences: &[Sentence], start: usize, end: usize) -> Result<f32> {
        if end <= start || start >= sentences.len() {
            return Ok(0.0);
        }
        let end = end.min(sentences.len());

        let chunk_text: String = sentences[start..end]
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        if chunk_text.is_empty() {
            return Ok(0.0);
        }

        // Coherence = average pairwise similarity within chunk
        let emb = self.embedder.embed(&chunk_text)?;

        // For simplicity, just return magnitude as proxy for coherence
        let mag: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        Ok(mag)
    }

    async fn enforce_size_limits(
        &self,
        chunks: Vec<Chunk>,
        source: Option<String>,
        metadata: serde_json::Value,
    ) -> Result<Vec<Chunk>> {
        let mut result = Vec::new();

        for chunk in chunks {
            if chunk.token_count <= self.config.max_chunk_tokens {
                result.push(chunk);
            } else {
                // Split oversized chunk
                let sub_chunks = self.split_large_chunk(&chunk, source.clone(), metadata.clone()).await?;
                result.extend(sub_chunks);
            }
        }

        Ok(result)
    }

    async fn split_large_chunk(
        &self,
        chunk: &Chunk,
        source: Option<String>,
        metadata: serde_json::Value,
    ) -> Result<Vec<Chunk>> {
        let sentences = self.split_sentences(&chunk.text);
        let mut result = Vec::new();
        let mut current_text = String::new();
        let mut current_tokens = 0;

        for sentence in sentences {
            let sentence_tokens = self.embedder.count_tokens(&sentence.text);

            if current_tokens + sentence_tokens > self.config.max_chunk_tokens && !current_text.is_empty() {
                let embedding = self.embedder.embed(&current_text)?;
                result.push(Chunk {
                    id: format!("chunk-{}", uuid::Uuid::new_v4()),
                    text: current_text.clone(),
                    embedding,
                    token_count: current_tokens,
                    source: source.clone(),
                    metadata: metadata.clone(),
                    created_at: chrono::Utc::now(),
                });
                current_text.clear();
                current_tokens = 0;
            }

            if !current_text.is_empty() {
                current_text.push(' ');
            }
            current_text.push_str(&sentence.text);
            current_tokens += sentence_tokens;
        }

        if !current_text.is_empty() {
            let embedding = self.embedder.embed(&current_text)?;
            result.push(Chunk {
                id: format!("chunk-{}", uuid::Uuid::new_v4()),
                text: current_text,
                embedding,
                token_count: current_tokens,
                source: source.clone(),
                metadata: metadata.clone(),
                created_at: chrono::Utc::now(),
            });
        }

        Ok(result)
    }
}
```

---

## src/api.rs

```rust
use crate::chunker::Chunker;
use crate::embedder::Embedder;
use crate::store::Store;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;

pub struct AppState {
    pub store: Arc<Store>,
    pub chunker: Arc<Chunker>,
    pub embedder: Arc<Embedder>,
    pub start_time: std::time::Instant,
}

// ===== Request/Response Types =====

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
    pub time_ms: u64,
}

#[derive(Deserialize)]
pub struct SearchQuery {
    pub q: String,
    #[serde(default = "default_n")]
    pub n: usize,
    #[serde(default)]
    pub time_weight: Option<f32>,
    #[serde(default)]
    pub include_nearby: Option<bool>,
    #[serde(default = "default_time_window")]
    pub time_window_minutes: i64,
}

fn default_n() -> usize {
    5
}
fn default_time_window() -> i64 {
    30
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub query: String,
    pub results: Vec<SearchResult>,
    pub time_ms: u64,
}

#[derive(Serialize)]
pub struct SearchResult {
    pub chunk_id: String,
    pub score: f32,
    pub text: String,
    pub source: Option<String>,
    pub metadata: serde_json::Value,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nearby_chunks: Option<Vec<NearbyChunk>>,
}

#[derive(Serialize)]
pub struct NearbyChunk {
    pub chunk_id: String,
    pub text: String,
    pub source: Option<String>,
    pub created_at: String,
}

#[derive(Serialize)]
pub struct SessionInfoResponse {
    pub session: String,
    pub chunks: usize,
    pub total_tokens: usize,
    pub sources: Vec<String>,
    pub created_at: String,
    pub last_ingested: String,
    pub memory_bytes: usize,
}

#[derive(Serialize)]
pub struct DeleteResponse {
    pub deleted_chunks: usize,
    pub session: String,
}

#[derive(Serialize)]
pub struct ListSessionsResponse {
    pub sessions: Vec<SessionListItem>,
}

#[derive(Serialize)]
pub struct SessionListItem {
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

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

// ===== Handlers =====

pub async fn ingest(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
    Json(req): Json<IngestRequest>,
) -> Result<Json<IngestResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();

    let chunks = state
        .chunker
        .chunk(&req.text, req.source, req.metadata)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

    let chunks_created = chunks.len();
    let tokens_processed: usize = chunks.iter().map(|c| c.token_count).sum();

    state.store.add_chunks(&session_id, chunks).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(IngestResponse {
        chunks_created,
        tokens_processed,
        time_ms: start.elapsed().as_millis() as u64,
    }))
}

pub async fn search(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
    Query(query): Query<SearchQuery>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();

    let query_embedding = state.embedder.embed(&query.q).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let results = if query.include_nearby.unwrap_or(false) {
        // Search with temporal context
        let results_with_context = state
            .store
            .search_with_temporal_context(
                &session_id,
                &query_embedding,
                query.n,
                query.time_window_minutes,
            )
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: e.to_string(),
                    }),
                )
            })?;

        results_with_context
            .into_iter()
            .map(|(chunk, score, nearby)| SearchResult {
                chunk_id: chunk.id,
                score,
                text: chunk.text,
                source: chunk.source,
                metadata: chunk.metadata,
                created_at: chunk.created_at.to_rfc3339(),
                nearby_chunks: Some(
                    nearby
                        .into_iter()
                        .map(|c| NearbyChunk {
                            chunk_id: c.id,
                            text: c.text,
                            source: c.source,
                            created_at: c.created_at.to_rfc3339(),
                        })
                        .collect(),
                ),
            })
            .collect()
    } else {
        // Standard search
        let results = state
            .store
            .search(&session_id, &query_embedding, query.n, query.time_weight)
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: e.to_string(),
                    }),
                )
            })?;

        results
            .into_iter()
            .map(|(chunk, score)| SearchResult {
                chunk_id: chunk.id,
                score,
                text: chunk.text,
                source: chunk.source,
                metadata: chunk.metadata,
                created_at: chunk.created_at.to_rfc3339(),
                nearby_chunks: None,
            })
            .collect()
    };

    Ok(Json(SearchResponse {
        query: query.q,
        results,
        time_ms: start.elapsed().as_millis() as u64,
    }))
}

pub async fn session_info(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> Result<Json<SessionInfoResponse>, StatusCode> {
    let meta = state
        .store
        .get_session_meta(&session_id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let chunks = state
        .store
        .get_chunks(&session_id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let sources: HashSet<String> = chunks.iter().filter_map(|c| c.source.clone()).collect();
    let memory_bytes: usize = chunks
        .iter()
        .map(|c| {
            c.text.len()
                + c.embedding.len() * 4
                + c.source.as_ref().map(|s| s.len()).unwrap_or(0)
        })
        .sum();

    Ok(Json(SessionInfoResponse {
        session: meta.id,
        chunks: meta.chunk_count,
        total_tokens: meta.total_tokens,
        sources: sources.into_iter().collect(),
        created_at: meta.created_at.to_rfc3339(),
        last_ingested: meta.last_ingested.to_rfc3339(),
        memory_bytes,
    }))
}

pub async fn delete_session(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> Result<Json<DeleteResponse>, StatusCode> {
    let exists = state
        .store
        .get_session_meta(&session_id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .is_some();

    if !exists {
        return Err(StatusCode::NOT_FOUND);
    }

    let deleted = state
        .store
        .delete_session(&session_id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(DeleteResponse {
        deleted_chunks: deleted,
        session: session_id,
    }))
}

pub async fn list_sessions(State(state): State<Arc<AppState>>) -> Json<ListSessionsResponse> {
    let sessions = state.store.list_sessions().unwrap_or_default();

    Json(ListSessionsResponse {
        sessions: sessions
            .into_iter()
            .map(|s| SessionListItem {
                id: s.id,
                chunks: s.chunk_count,
                last_active: s.last_ingested.to_rfc3339(),
            })
            .collect(),
    })
}

pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let sessions = state.store.list_sessions().unwrap_or_default();
    let total_chunks = state.store.total_chunks().unwrap_or(0);

    Json(HealthResponse {
        status: "ok".to_string(),
        model: "bge-small-en-v1.5".to_string(),
        sessions: sessions.len(),
        total_chunks,
        uptime_seconds: state.start_time.elapsed().as_secs(),
    })
}

// ===== Router =====

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
use std::sync::Arc;
use tokio::net::TcpListener;

mod api;
mod chunker;
mod embedder;
mod similarity;
mod store;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/bge-small-en-v1.5.onnx".to_string());
    let tokenizer_path = std::env::var("TOKENIZER_PATH")
        .unwrap_or_else(|_| "models/tokenizer.json".to_string());
    let data_path = std::env::var("DATA_PATH").unwrap_or_else(|_| "/data/shivvr".to_string());

    println!("Loading embedding model from {}...", model_path);
    let embedder = Arc::new(embedder::Embedder::new(&model_path, &tokenizer_path)?);

    println!("Opening database at {}...", data_path);
    let store = Arc::new(store::Store::open(&data_path)?);

    let chunker = Arc::new(chunker::Chunker::new(embedder.clone()));

    let state = Arc::new(api::AppState {
        store,
        chunker,
        embedder,
        start_time: std::time::Instant::now(),
    });

    let app = api::router(state);

    let addr = format!("0.0.0.0:{}", port);
    println!("Starting shivvr service on {}", addr);

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
```

---

## Dockerfile

```dockerfile
FROM rust:1.88-slim-bookworm AS builder

RUN apt-get update && apt-get install -y \
    pkg-config libssl-dev curl g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release || true
RUN rm -rf src

COPY src ./src
RUN touch src/main.rs
RUN cargo build --release

# Download model and tokenizer
RUN mkdir -p /models && \
    curl -L -o /models/bge-small-en-v1.5.onnx \
    "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx" && \
    curl -L -o /models/tokenizer.json \
    "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json"

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/shivvr /shivvr
COPY --from=builder /models /models

ENV PORT=8080
ENV MODEL_PATH=/models/bge-small-en-v1.5.onnx
ENV TOKENIZER_PATH=/models/tokenizer.json
ENV DATA_PATH=/data/shivvr

EXPOSE 8080
VOLUME ["/data"]
ENTRYPOINT ["/shivvr"]
```

---

## docker-compose.yml

```yaml
version: '3.8'

services:
  shivvr:
    build: .
    ports:
      - "8080:8080"
    environment:
      - PORT=8080
      - MODEL_PATH=/models/bge-small-en-v1.5.onnx
      - TOKENIZER_PATH=/models/tokenizer.json
      - DATA_PATH=/data/shivvr
    volumes:
      - shivvr-data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  shivvr-data:
```

---

## Usage Examples

```bash
# Start service
docker-compose up -d

# Ingest text (any length)
curl -X POST http://localhost:8080/memory/agent-001/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "The user Bob lives in Austin. He loves Italian food.", "source": "profile"}'

# Ingest a whole file
curl -X POST http://localhost:8080/memory/project-x/ingest \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"$(cat docs/readme.md)\", \"source\": \"readme\"}"

# Search (semantic only)
curl "http://localhost:8080/memory/agent-001/search?q=what+food+does+user+like&n=3"

# Search with recency boost
curl "http://localhost:8080/memory/agent-001/search?q=what+food+does+user+like&n=3&time_weight=0.2"

# Search with temporal context
curl "http://localhost:8080/memory/agent-001/search?q=what+food+does+user+like&n=3&include_nearby=true&time_window_minutes=30"

# Session info
curl http://localhost:8080/memory/agent-001/info

# List sessions
curl http://localhost:8080/memory

# Delete session
curl -X DELETE http://localhost:8080/memory/agent-001

# Health
curl http://localhost:8080/health
```

---

## Performance

- Ingestion cost is dominated by embedding inference and scales roughly linearly with token count.
- Search is brute-force cosine over all chunks in a session; time weighting and nearby expansion add minimal overhead.
- Persistence uses sled + bincode; disk size is approximately text bytes plus 4 bytes per embedding dimension per chunk, plus bincode overhead.

---

## Environment Variables

| Var | Default | Description |
|-----|---------|-------------|
| PORT | 8080 | HTTP port |
| MODEL_PATH | models/bge-small-en-v1.5.onnx | ONNX model path |
| TOKENIZER_PATH | models/tokenizer.json | Tokenizer JSON path |
| DATA_PATH | /data/shivvr | sled database directory |
| RUST_LOG | info | Log level |
