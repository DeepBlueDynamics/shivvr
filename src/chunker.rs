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
                embedding_retrieve: None,
                token_count,
                source: source.clone(),
                metadata: metadata.clone(),
                created_at: chrono::Utc::now(),
                emotion_primary: None,
                emotion_secondary: None,
                encrypted: false,
                agent_id: None,
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
            embedding_retrieve: None,
            token_count,
            source,
            metadata,
            created_at: chrono::Utc::now(),
            emotion_primary: None,
            emotion_secondary: None,
            encrypted: false,
            agent_id: None,
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
                    embedding_retrieve: None,
                    token_count: current_tokens,
                    source: source.clone(),
                    metadata: metadata.clone(),
                    created_at: chrono::Utc::now(),
                    emotion_primary: None,
                    emotion_secondary: None,
                    encrypted: false,
                    agent_id: None,
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
                embedding_retrieve: None,
                token_count: current_tokens,
                source: source.clone(),
                metadata: metadata.clone(),
                created_at: chrono::Utc::now(),
                emotion_primary: None,
                emotion_secondary: None,
                encrypted: false,
                agent_id: None,
            });
        }

        Ok(result)
    }
}
