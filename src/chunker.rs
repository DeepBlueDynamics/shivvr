use crate::embedder::Embedder;
use crate::similarity::cosine_similarity;
use crate::store::Chunk;
use anyhow::Result;
use std::collections::HashMap;
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
        let mut embedding_cache = HashMap::new();

        // Handle tiny inputs
        if text.len() < 100 {
            return self
                .single_chunk(text, source, metadata, &mut embedding_cache)
                .await;
        }

        // Split into sentences
        let sentences = Self::split_sentences(text);

        if sentences.is_empty() || sentences.len() == 1 {
            return self
                .word_window_chunks(text, source, metadata, &mut embedding_cache)
                .await;
        }

        if sentences.len() < 3 {
            return self
                .greedy_sentence_chunks(&sentences, source, metadata, &mut embedding_cache)
                .await;
        }

        // Monte Carlo sampling to find candidate boundaries
        let candidate_regions = self
            .monte_carlo_sample(&sentences, &mut embedding_cache)
            .await?;

        if candidate_regions.is_empty() {
            return self
                .greedy_sentence_chunks(&sentences, source, metadata, &mut embedding_cache)
                .await;
        }

        // Build boundaries
        let mut boundaries = vec![0];
        for (start, end) in candidate_regions {
            let boundary = self
                .binary_search_boundary(&sentences, start, end, &mut embedding_cache)
                .await?;
            if boundary > *boundaries.last().unwrap() {
                boundaries.push(boundary);
            }
        }
        boundaries.push(sentences.len());

        // Optimize boundaries
        let boundaries = self
            .optimize_boundaries(&sentences, boundaries, &mut embedding_cache)
            .await?;

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

            let embedding = self.embed_cached(&chunk_text, &mut embedding_cache)?;
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
        embedding_cache: &mut HashMap<String, Vec<f32>>,
    ) -> Result<Vec<Chunk>> {
        let embedding = self.embed_cached(text, embedding_cache)?;
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

    fn split_sentences(text: &str) -> Vec<Sentence> {
        let mut sentences = Vec::new();
        let mut start = 0;

        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let c = chars[i];
            let sentence_boundary = (c == '.' || c == '!' || c == '?')
                && (i + 1 >= chars.len() || chars[i + 1].is_whitespace());
            let paragraph_boundary = c == '\n' && Self::is_double_newline(&chars, i);
            let line_boundary = c == '\n' && Self::next_line_starts_boundary(&chars, i + 1);

            if sentence_boundary || paragraph_boundary || line_boundary {
                let end = if sentence_boundary { i + 1 } else { i };
                Self::push_sentence(&mut sentences, &chars, start, end);
                i = if sentence_boundary { i + 1 } else { i };
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
            Self::push_sentence(&mut sentences, &chars, start, chars.len());
        }

        sentences
    }

    fn push_sentence(sentences: &mut Vec<Sentence>, chars: &[char], start: usize, end: usize) {
        if start >= end || end > chars.len() {
            return;
        }

        let sentence_text: String = chars[start..end].iter().collect();
        let trimmed = sentence_text.trim();
        if !trimmed.is_empty() {
            sentences.push(Sentence {
                text: trimmed.to_string(),
                start_char: start,
                end_char: end,
            });
        }
    }

    fn is_double_newline(chars: &[char], newline_index: usize) -> bool {
        let mut i = newline_index + 1;
        if i < chars.len() && chars[i] == '\r' {
            i += 1;
        }
        i < chars.len() && chars[i] == '\n'
    }

    fn next_line_starts_boundary(chars: &[char], mut i: usize) -> bool {
        if i < chars.len() && chars[i] == '\r' {
            i += 1;
        }

        let mut j = i;
        while j < chars.len() && (chars[j] == ' ' || chars[j] == '\t') {
            j += 1;
        }

        if j >= chars.len() || chars[j] == '\n' || chars[j] == '\r' {
            return true;
        }

        if matches!(chars[j], '#' | '-' | '*' | '>') {
            return true;
        }

        if chars[j].is_ascii_digit() {
            j += 1;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            return j < chars.len() && chars[j] == '.';
        }

        false
    }

    async fn greedy_sentence_chunks(
        &self,
        sentences: &[Sentence],
        source: Option<String>,
        metadata: serde_json::Value,
        embedding_cache: &mut HashMap<String, Vec<f32>>,
    ) -> Result<Vec<Chunk>> {
        let texts = Self::sentence_text_windows(sentences, self.config.max_chunk_tokens, |text| {
            self.embedder.count_tokens(text)
        });
        self.make_chunks_from_texts(texts, source, metadata, embedding_cache)
            .await
    }

    fn sentence_text_windows<F>(
        sentences: &[Sentence],
        max_tokens: usize,
        mut count_tokens: F,
    ) -> Vec<String>
    where
        F: FnMut(&str) -> usize,
    {
        let mut chunks = Vec::new();
        let mut current_text = String::new();
        let mut current_tokens = 0;
        let max_tokens = max_tokens.max(1);

        for sentence in sentences {
            let sentence_tokens = count_tokens(&sentence.text);

            if current_tokens + sentence_tokens > max_tokens && !current_text.is_empty() {
                chunks.push(current_text.clone());
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
            chunks.push(current_text);
        }

        chunks
    }

    async fn word_window_chunks(
        &self,
        text: &str,
        source: Option<String>,
        metadata: serde_json::Value,
        embedding_cache: &mut HashMap<String, Vec<f32>>,
    ) -> Result<Vec<Chunk>> {
        let texts = Self::word_text_windows(text, self.config.max_chunk_tokens);
        self.make_chunks_from_texts(texts, source, metadata, embedding_cache)
            .await
    }

    fn word_text_windows(text: &str, max_words: usize) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Vec::new();
        }

        words
            .chunks(max_words.max(1))
            .map(|window| window.join(" "))
            .collect()
    }

    async fn make_chunks_from_texts(
        &self,
        texts: Vec<String>,
        source: Option<String>,
        metadata: serde_json::Value,
        embedding_cache: &mut HashMap<String, Vec<f32>>,
    ) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();
        for text in texts {
            chunks.push(
                self.make_chunk(text, source.clone(), metadata.clone(), embedding_cache)
                    .await?,
            );
        }
        Ok(chunks)
    }

    async fn make_chunk(
        &self,
        text: String,
        source: Option<String>,
        metadata: serde_json::Value,
        embedding_cache: &mut HashMap<String, Vec<f32>>,
    ) -> Result<Chunk> {
        let embedding = self.embed_cached(&text, embedding_cache)?;
        let token_count = self.embedder.count_tokens(&text);

        Ok(Chunk {
            id: format!("chunk-{}", uuid::Uuid::new_v4()),
            text,
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
        })
    }

    async fn monte_carlo_sample(
        &self,
        sentences: &[Sentence],
        embedding_cache: &mut HashMap<String, Vec<f32>>,
    ) -> Result<Vec<(usize, usize)>> {
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

                let before_emb = self.embed_cached(&before_text, embedding_cache)?;
                let after_emb = self.embed_cached(&after_text, embedding_cache)?;

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
        embedding_cache: &mut HashMap<String, Vec<f32>>,
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

            let before_emb = self.embed_cached(before_text, embedding_cache)?;
            let after_emb = self.embed_cached(after_text, embedding_cache)?;

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
        embedding_cache: &mut HashMap<String, Vec<f32>>,
    ) -> Result<Vec<usize>> {
        for _ in 0..self.config.optimization_iterations {
            for i in 1..boundaries.len() - 1 {
                let prev = boundaries[i - 1];
                let curr = boundaries[i];
                let next = boundaries[i + 1];

                // Try moving boundary left or right
                let mut best_pos = curr;
                let mut best_coherence = self
                    .chunk_coherence(sentences, prev, curr, embedding_cache)
                    .await?
                    + self
                        .chunk_coherence(sentences, curr, next, embedding_cache)
                        .await?;

                for delta in [-1i32, 1i32] {
                    let new_pos = (curr as i32 + delta) as usize;
                    if new_pos > prev && new_pos < next {
                        let coherence = self
                            .chunk_coherence(sentences, prev, new_pos, embedding_cache)
                            .await?
                            + self
                                .chunk_coherence(sentences, new_pos, next, embedding_cache)
                                .await?;
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

    async fn chunk_coherence(
        &self,
        sentences: &[Sentence],
        start: usize,
        end: usize,
        embedding_cache: &mut HashMap<String, Vec<f32>>,
    ) -> Result<f32> {
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
        let emb = self.embed_cached(&chunk_text, embedding_cache)?;

        // For simplicity, just return magnitude as proxy for coherence
        let mag: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        Ok(mag)
    }

    fn embed_cached(
        &self,
        text: &str,
        embedding_cache: &mut HashMap<String, Vec<f32>>,
    ) -> Result<Vec<f32>> {
        if let Some(embedding) = embedding_cache.get(text) {
            return Ok(embedding.clone());
        }

        let embedding = self.embedder.embed(text)?;
        embedding_cache.insert(text.to_string(), embedding.clone());
        Ok(embedding)
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
        let mut embedding_cache = HashMap::new();
        let sentences = Self::split_sentences(&chunk.text);

        if sentences.is_empty() || sentences.len() == 1 {
            return self
                .word_window_chunks(&chunk.text, source, metadata, &mut embedding_cache)
                .await;
        }

        self.greedy_sentence_chunks(&sentences, source, metadata, &mut embedding_cache)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::Chunker;

    #[test]
    fn split_sentences_finds_markdown_boundaries() {
        let text = "# Intro\n\
Some setup text without punctuation\n\
- first bullet has words\n\
- second bullet has words\n\n\
## Next\n\
1. numbered item\n\
> quoted item";

        let sentences = Chunker::split_sentences(text);
        let parts: Vec<&str> = sentences.iter().map(|s| s.text.as_str()).collect();

        assert!(parts.len() > 1, "markdown should split into multiple units");
        assert!(parts.iter().any(|part| part.starts_with("# Intro")));
        assert!(parts.iter().any(|part| part.starts_with("- first")));
        assert!(parts.iter().any(|part| part.starts_with("## Next")));
        assert!(parts.iter().any(|part| part.starts_with("1.")));
    }

    #[test]
    fn greedy_sentence_windows_split_markdown_by_size() {
        let text = "# Intro\n\
alpha beta gamma delta\n\
- epsilon zeta eta theta\n\
- iota kappa lambda mu\n\
## Next\n\
nu xi omicron pi";
        let sentences = Chunker::split_sentences(text);

        let chunks = Chunker::sentence_text_windows(&sentences, 8, |text| {
            text.split_whitespace().count()
        });

        assert!(
            chunks.len() > 1,
            "markdown headers and bullets should produce more than one chunk"
        );
    }

    #[test]
    fn word_windows_split_text_without_sentence_boundaries() {
        let text = "one two three four five six seven eight nine ten";
        let chunks = Chunker::word_text_windows(text, 4);

        assert_eq!(chunks, vec!["one two three four", "five six seven eight", "nine ten"]);
    }
}
