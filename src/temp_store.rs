use crate::similarity::cosine_similarity;
use crate::store::Chunk;
use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;
use std::sync::RwLock;

const TEMP_STORE_TTL_HOURS: i64 = 2;

#[derive(Debug, Clone)]
pub struct TempStoreMeta {
    pub name: String,
    pub chunk_count: usize,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

struct TempSession {
    chunks: Vec<Chunk>,
    bm25_index: Option<lume_hybrid::bm25::Bm25Index>,
    created_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
}

pub struct TempStore {
    stores: RwLock<HashMap<String, TempSession>>,
}

impl TempStore {
    pub fn new() -> Self {
        Self {
            stores: RwLock::new(HashMap::new()),
        }
    }

    pub fn add_chunks(&self, name: &str, chunks: Vec<Chunk>) -> Result<()> {
        let now = Utc::now();
        let expires_at = now + Duration::hours(TEMP_STORE_TTL_HOURS);

        let mut stores = self.stores.write().unwrap();
        let session = stores.entry(name.to_string()).or_insert_with(|| TempSession {
            chunks: Vec::new(),
            bm25_index: None,
            created_at: now,
            expires_at,
        });

        session.chunks.extend(chunks);
        session.expires_at = expires_at;

        // Rebuild the local BM25 index with the combined session chunks. See
        // store.rs for the same pattern + Section/Tagger explanation.
        let sections: Vec<lume_hybrid::bm25::Section> = session.chunks.iter().enumerate().map(|(i, c)| {
            lume_hybrid::bm25::Section {
                title: c.source.clone().unwrap_or_else(|| c.id.clone()),
                body: c.text.clone(),
                line_number: i,
                filename: c.source.clone(),
                entities: Vec::new(),
            }
        }).collect();
        session.bm25_index = Some(lume_hybrid::bm25::Bm25Index::build(sections, None));

        Ok(())
    }

    pub fn get_chunks(&self, name: &str) -> Result<Vec<Chunk>> {
        let stores = self.stores.read().unwrap();
        Ok(stores
            .get(name)
            .map(|s| s.chunks.clone())
            .unwrap_or_default())
    }

    pub fn delete_store(&self, name: &str) -> Result<usize> {
        let mut stores = self.stores.write().unwrap();
        let deleted = stores.remove(name).map(|s| s.chunks.len()).unwrap_or(0);
        Ok(deleted)
    }

    pub fn list_stores(&self) -> Vec<TempStoreMeta> {
        let stores = self.stores.read().unwrap();
        stores
            .iter()
            .map(|(name, session)| TempStoreMeta {
                name: name.clone(),
                chunk_count: session.chunks.len(),
                created_at: session.created_at,
                expires_at: session.expires_at,
            })
            .collect()
    }

    pub fn sweep_expired(&self) -> usize {
        let now = Utc::now();
        let mut stores = self.stores.write().unwrap();
        let before = stores.len();
        stores.retain(|_, session| session.expires_at > now);
        before.saturating_sub(stores.len())
    }

    pub fn search(
        &self,
        name: &str,
        query_embedding: &[f32],
        n: usize,
        time_weight: Option<f32>,
        decay_halflife_hours: f32,
        role: &str,
    ) -> Result<Vec<(Chunk, f32)>> {
        let chunks = self.get_chunks(name)?;
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let now = Utc::now();
        let time_weight = time_weight.unwrap_or(0.0).clamp(0.0, 1.0);

        let mut scores: Vec<(usize, f32)> = chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                let chunk_emb = Self::get_embedding_for_role(chunk, role);
                let semantic_score = cosine_similarity(query_embedding, chunk_emb);

                let time_score = if time_weight > 0.0 {
                    let age_hours = (now - chunk.created_at).num_hours() as f32;
                    (-age_hours / decay_halflife_hours).exp()
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

    /// Search hybrid combining semantic vector search and BM25F lexical search with RRF & FST boosts
    pub fn search_hybrid(
        &self,
        name: &str,
        query: &str,
        query_embedding: Option<&[f32]>,
        n: usize,
        time_weight: Option<f32>,
        decay_halflife_hours: f32,
        role: &str,
        tags: &[lume_hybrid::Tag],
        _params: &lume_hybrid::bm25::Bm25Params,
    ) -> Result<Vec<(Chunk, f32)>> {
        let chunks = self.get_chunks(name)?;
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Compute Semantic Scores (if embedding is available)
        let mut vector_ranks: HashMap<String, usize> = HashMap::new();
        let mut vector_scores: HashMap<String, f32> = HashMap::new();

        if let Some(emb) = query_embedding {
            let semantic_hits = self.search(
                name,
                emb,
                chunks.len(),
                time_weight,
                decay_halflife_hours,
                role,
            )?;
            for (rank, (chunk, score)) in semantic_hits.into_iter().enumerate() {
                vector_ranks.insert(chunk.id.clone(), rank);
                vector_scores.insert(chunk.id.clone(), score);
            }
        }

        // 2. Compute Lexical BM25 Scores
        let mut lexical_ranks: HashMap<String, usize> = HashMap::new();
        let mut lexical_scores: HashMap<String, f32> = HashMap::new();

        let stores = self.stores.read().unwrap();
        let session = stores.get(name);
        if let Some(s) = session {
            if let Some(ref idx) = s.bm25_index {
                // See store.rs for why we use Bm25Index::search instead of the
                // old manual postings iteration.
                let hits = idx.search(
                    query,
                    lume_hybrid::bm25::SearchVariant::Classic,
                    _params,
                    None,
                );

                for (rank, hit) in hits.iter().enumerate() {
                    if hit.section_index < s.chunks.len() {
                        let chunk_id = &s.chunks[hit.section_index].id;
                        lexical_ranks.insert(chunk_id.clone(), rank);
                        lexical_scores.insert(chunk_id.clone(), hit.score as f32);
                    }
                }
            }
        }

        // 3. Reciprocal Rank Fusion (RRF) & FST Intent Boosting
        let k = 60.0f32; // RRF parameter
        let mut final_scores: Vec<(usize, f32)> = chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                let chunk_id = &chunk.id;
                let rrf_sem = vector_ranks.get(chunk_id)
                    .map(|&rank| 1.0 / (k + rank as f32))
                    .unwrap_or(0.0);
                let rrf_lex = lexical_ranks.get(chunk_id)
                    .map(|&rank| 1.0 / (k + rank as f32))
                    .unwrap_or(0.0);
                let mut combined_score = rrf_sem + rrf_lex;

                // FST dynamic intent boosting
                let normalized_text = chunk.text.to_lowercase();
                for tag in tags {
                    let surface_matched = normalized_text.contains(&tag.surface.to_lowercase());
                    let output_matched = normalized_text.contains(&tag.output.to_lowercase());
                    if surface_matched || output_matched {
                        combined_score += 0.05f32; // Boost rank dynamically in the RRF domain
                    }
                }

                (i, combined_score)
            })
            .collect();

        final_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        final_scores.truncate(n);

        Ok(final_scores
            .into_iter()
            .map(|(i, score)| (chunks[i].clone(), score))
            .collect())
    }

    pub fn search_with_temporal_context(
        &self,
        name: &str,
        query_embedding: &[f32],
        n: usize,
        time_window_minutes: i64,
        role: &str,
    ) -> Result<Vec<(Chunk, f32, Vec<Chunk>)>> {
        let chunks = self.get_chunks(name)?;
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let mut scores: Vec<(usize, f32)> = chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                let chunk_emb = Self::get_embedding_for_role(chunk, role);
                let score = cosine_similarity(query_embedding, chunk_emb);
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

    fn get_embedding_for_role<'a>(chunk: &'a Chunk, role: &str) -> &'a [f32] {
        if role == "retrieve" {
            if let Some(ref emb) = chunk.embedding_retrieve {
                return emb;
            }
        }
        &chunk.embedding
    }
}
