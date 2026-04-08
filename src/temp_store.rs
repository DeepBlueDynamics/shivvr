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
            created_at: now,
            expires_at,
        });

        session.chunks.extend(chunks);
        session.expires_at = expires_at;
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
