use crate::similarity::cosine_similarity;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub text: String,
    /// Organize embedding (768d, gtr-t5-base)
    pub embedding: Vec<f32>,
    /// Retrieve embedding (1536d, ada-002) — optional
    #[serde(default)]
    pub embedding_retrieve: Option<Vec<f32>>,
    pub token_count: usize,
    pub source: Option<String>,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
    /// Vedanā — feeling tone at time of memory creation (Abhidharma)
    #[serde(default)]
    pub emotion_primary: Option<String>,
    #[serde(default)]
    pub emotion_secondary: Option<String>,
    /// Whether embeddings are encrypted with agent identity key
    #[serde(default)]
    pub encrypted: bool,
    /// Agent that owns this chunk (for encryption key lookup)
    #[serde(default)]
    pub agent_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SessionMeta {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub last_ingested: DateTime<Utc>,
    pub chunk_count: usize,
    pub total_tokens: usize,
}

struct Session {
    chunks: Vec<Chunk>,
    created_at: DateTime<Utc>,
    last_ingested: DateTime<Utc>,
    total_tokens: usize,
    /// Identity of the caller who created this session (user_id from auth claims).
    /// None = created in open dev mode (no auth) — accessible to anyone.
    owner: Option<String>,
}

/// Ephemeral in-memory store. No persistence. All state is lost on restart.
pub struct Store {
    sessions: RwLock<HashMap<String, Session>>,
}

impl Store {
    pub fn new() -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
        }
    }

    /// Add chunks to a session (creates session if it doesn't exist).
    /// `owner` is set only on creation — subsequent ingests don't change it.
    pub fn add_chunks(&self, session_id: &str, chunks: Vec<Chunk>, owner: Option<&str>) -> Result<()> {
        let tokens: usize = chunks.iter().map(|c| c.token_count).sum();
        let now = Utc::now();

        let mut sessions = self.sessions.write().unwrap();
        let session = sessions.entry(session_id.to_string()).or_insert_with(|| Session {
            chunks: Vec::new(),
            created_at: now,
            last_ingested: now,
            total_tokens: 0,
            owner: owner.map(|s| s.to_string()),
        });

        session.chunks.extend(chunks);
        session.total_tokens += tokens;
        session.last_ingested = now;

        Ok(())
    }

    /// Check if a caller is allowed to access a session.
    /// Returns true if session has no owner (dev mode) or owner matches caller.
    pub fn caller_owns_session(&self, session_id: &str, caller_id: Option<&str>) -> bool {
        let sessions = self.sessions.read().unwrap();
        match sessions.get(session_id) {
            None => true, // session doesn't exist yet — allow (will be created)
            Some(s) => match (&s.owner, caller_id) {
                (None, _) => true,           // no owner = open dev mode, anyone can access
                (Some(_), None) => false,    // owned session, unauthenticated caller
                (Some(o), Some(c)) => o == c,
            },
        }
    }

    /// Get all chunks for a session
    pub fn get_chunks(&self, session_id: &str) -> Result<Vec<Chunk>> {
        let sessions = self.sessions.read().unwrap();
        Ok(sessions
            .get(session_id)
            .map(|s| s.chunks.clone())
            .unwrap_or_default())
    }

    /// Get session metadata
    pub fn get_session_meta(&self, session_id: &str) -> Result<Option<SessionMeta>> {
        let sessions = self.sessions.read().unwrap();
        Ok(sessions.get(session_id).map(|s| SessionMeta {
            id: session_id.to_string(),
            created_at: s.created_at,
            last_ingested: s.last_ingested,
            chunk_count: s.chunks.len(),
            total_tokens: s.total_tokens,
        }))
    }

    /// Delete a session and all its chunks
    pub fn delete_session(&self, session_id: &str) -> Result<usize> {
        let mut sessions = self.sessions.write().unwrap();
        let deleted = sessions
            .remove(session_id)
            .map(|s| s.chunks.len())
            .unwrap_or(0);
        Ok(deleted)
    }

    /// List sessions visible to the caller.
    /// With auth: returns only sessions owned by caller_id.
    /// Without auth (caller_id = None): returns only unowned sessions.
    pub fn list_sessions(&self, caller_id: Option<&str>) -> Result<Vec<SessionMeta>> {
        let sessions = self.sessions.read().unwrap();
        Ok(sessions
            .iter()
            .filter(|(_, s)| match (&s.owner, caller_id) {
                (None, None) => true,        // both unowned — dev mode sees unowned sessions
                (Some(o), Some(c)) => o == c, // owned session matches caller
                _ => false,
            })
            .map(|(id, s)| SessionMeta {
                id: id.clone(),
                created_at: s.created_at,
                last_ingested: s.last_ingested,
                chunk_count: s.chunks.len(),
                total_tokens: s.total_tokens,
            })
            .collect())
    }

    /// Get total chunk count across all sessions
    pub fn total_chunks(&self) -> Result<usize> {
        let sessions = self.sessions.read().unwrap();
        Ok(sessions.values().map(|s| s.chunks.len()).sum())
    }

    /// Get the embedding for a chunk by role ("organize" or "retrieve")
    fn get_embedding_for_role<'a>(chunk: &'a Chunk, role: &str) -> &'a [f32] {
        if role == "retrieve" {
            if let Some(ref emb) = chunk.embedding_retrieve {
                return emb;
            }
        }
        &chunk.embedding
    }

    /// Search by semantic similarity with optional time weighting
    pub fn search(
        &self,
        session_id: &str,
        query_embedding: &[f32],
        n: usize,
        time_weight: Option<f32>,
        decay_halflife_hours: f32,
        role: &str,
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

    /// Search and include chunks near in time to each match
    pub fn search_with_temporal_context(
        &self,
        session_id: &str,
        query_embedding: &[f32],
        n: usize,
        time_window_minutes: i64,
        role: &str,
    ) -> Result<Vec<(Chunk, f32, Vec<Chunk>)>> {
        let chunks = self.get_chunks(session_id)?;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> Store {
        Store::new()
    }

    fn make_chunk(id: &str, embedding: Vec<f32>, retrieve: Option<Vec<f32>>) -> Chunk {
        Chunk {
            id: id.to_string(),
            text: format!("text for {}", id),
            embedding,
            embedding_retrieve: retrieve,
            token_count: 10,
            source: Some("test".to_string()),
            metadata: serde_json::json!({}),
            created_at: Utc::now(),
            emotion_primary: None,
            emotion_secondary: None,
            encrypted: false,
            agent_id: None,
        }
    }

    fn l2_normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v.to_vec()
        }
    }

    // ===== Serialization tests =====

    #[test]
    fn chunk_serde_with_all_fields() {
        let chunk = Chunk {
            id: "chunk-123".to_string(),
            text: "hello world".to_string(),
            embedding: vec![0.1, 0.2, 0.3],
            embedding_retrieve: Some(vec![0.4, 0.5, 0.6, 0.7]),
            token_count: 2,
            source: Some("test".to_string()),
            metadata: serde_json::json!({"key": "value"}),
            created_at: Utc::now(),
            emotion_primary: Some("curious".to_string()),
            emotion_secondary: None,
            encrypted: true,
            agent_id: Some("acala".to_string()),
        };

        let json = serde_json::to_string(&chunk).unwrap();
        let deserialized: Chunk = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, "chunk-123");
        assert_eq!(deserialized.embedding_retrieve.unwrap().len(), 4);
        assert!(deserialized.encrypted);
        assert_eq!(deserialized.agent_id.unwrap(), "acala");
    }

    #[test]
    fn chunk_backward_compat_missing_new_fields() {
        let json = r#"{
            "id": "chunk-old",
            "text": "old chunk",
            "embedding": [0.1, 0.2],
            "token_count": 5,
            "source": null,
            "metadata": {},
            "created_at": "2025-01-01T00:00:00Z"
        }"#;

        let chunk: Chunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.id, "chunk-old");
        assert!(chunk.embedding_retrieve.is_none());
        assert!(!chunk.encrypted);
        assert!(chunk.agent_id.is_none());
        assert!(chunk.emotion_primary.is_none());
    }

    // ===== Role-based embedding selection =====

    #[test]
    fn get_embedding_organize_role() {
        let chunk = make_chunk("c1", vec![1.0, 0.0], Some(vec![0.0, 1.0]));
        let emb = Store::get_embedding_for_role(&chunk, "organize");
        assert_eq!(emb, &[1.0, 0.0]);
    }

    #[test]
    fn get_embedding_retrieve_role_with_retrieve() {
        let chunk = make_chunk("c1", vec![1.0, 0.0], Some(vec![0.0, 1.0]));
        let emb = Store::get_embedding_for_role(&chunk, "retrieve");
        assert_eq!(emb, &[0.0, 1.0]);
    }

    #[test]
    fn get_embedding_retrieve_role_fallback() {
        let chunk = make_chunk("c1", vec![1.0, 0.0], None);
        let emb = Store::get_embedding_for_role(&chunk, "retrieve");
        assert_eq!(emb, &[1.0, 0.0]);
    }

    // ===== Store CRUD tests =====

    #[test]
    fn add_and_get_chunks() {
        let store = make_store();
        let chunks = vec![
            make_chunk("c1", vec![1.0, 0.0], None),
            make_chunk("c2", vec![0.0, 1.0], Some(vec![0.5, 0.5])),
        ];

        store.add_chunks("session1", chunks, None).unwrap();

        let retrieved = store.get_chunks("session1").unwrap();
        assert_eq!(retrieved.len(), 2);

        let c2 = retrieved.iter().find(|c| c.id == "c2").unwrap();
        assert!(c2.embedding_retrieve.is_some());
        assert_eq!(c2.embedding_retrieve.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn session_meta_tracks_chunks() {
        let store = make_store();
        store
            .add_chunks("s1", vec![make_chunk("c1", vec![1.0], None)], None)
            .unwrap();
        store
            .add_chunks("s1", vec![make_chunk("c2", vec![0.5], None)], None)
            .unwrap();

        let meta = store.get_session_meta("s1").unwrap().unwrap();
        assert_eq!(meta.chunk_count, 2);
        assert_eq!(meta.total_tokens, 20);
    }

    #[test]
    fn delete_session_removes_everything() {
        let store = make_store();
        store
            .add_chunks(
                "s1",
                vec![
                    make_chunk("c1", vec![1.0], None),
                    make_chunk("c2", vec![0.5], None),
                ],
                None,
            )
            .unwrap();

        let deleted = store.delete_session("s1").unwrap();
        assert_eq!(deleted, 2);
        assert!(store.get_session_meta("s1").unwrap().is_none());
        assert!(store.get_chunks("s1").unwrap().is_empty());
    }

    #[test]
    fn list_sessions() {
        let store = make_store();
        store
            .add_chunks("alpha", vec![make_chunk("c1", vec![1.0], None)], None)
            .unwrap();
        store
            .add_chunks("beta", vec![make_chunk("c2", vec![0.5], None)], None)
            .unwrap();

        let sessions = store.list_sessions(None).unwrap();
        assert_eq!(sessions.len(), 2);
    }

    // ===== Search tests with synthetic embeddings =====

    #[test]
    fn search_organize_role() {
        let store = make_store();

        let chunks = vec![
            make_chunk("match", l2_normalize(&[1.0, 0.0, 0.0, 0.0]), None),
            make_chunk("partial", l2_normalize(&[0.7, 0.7, 0.0, 0.0]), None),
            make_chunk("orthogonal", l2_normalize(&[0.0, 0.0, 1.0, 0.0]), None),
        ];
        store.add_chunks("s1", chunks, None).unwrap();

        let query = l2_normalize(&[1.0, 0.0, 0.0, 0.0]);
        let results = store.search("s1", &query, 3, None, 168.0, "organize").unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0.id, "match");
        assert!(results[0].1 > 0.99, "best match should be ~1.0");
        assert_eq!(results[2].0.id, "orthogonal");
        assert!(results[2].1.abs() < 0.1, "orthogonal should be ~0.0");
    }

    #[test]
    fn search_retrieve_role_uses_retrieve_embedding() {
        let store = make_store();

        let chunks = vec![
            make_chunk(
                "c1",
                l2_normalize(&[1.0, 0.0, 0.0, 0.0]),
                Some(l2_normalize(&[0.0, 1.0, 0.0, 0.0])),
            ),
            make_chunk(
                "c2",
                l2_normalize(&[0.0, 1.0, 0.0, 0.0]),
                Some(l2_normalize(&[1.0, 0.0, 0.0, 0.0])),
            ),
        ];
        store.add_chunks("s1", chunks, None).unwrap();

        let query = l2_normalize(&[1.0, 0.0, 0.0, 0.0]);

        let org_results = store.search("s1", &query, 2, None, 168.0, "organize").unwrap();
        assert_eq!(org_results[0].0.id, "c1");

        let ret_results = store.search("s1", &query, 2, None, 168.0, "retrieve").unwrap();
        assert_eq!(ret_results[0].0.id, "c2");
    }

    #[test]
    fn search_retrieve_fallback_when_no_retrieve_embedding() {
        let store = make_store();

        let chunks = vec![
            make_chunk("c1", l2_normalize(&[1.0, 0.0, 0.0, 0.0]), None),
            make_chunk("c2", l2_normalize(&[0.0, 1.0, 0.0, 0.0]), None),
        ];
        store.add_chunks("s1", chunks, None).unwrap();

        let query = l2_normalize(&[1.0, 0.0, 0.0, 0.0]);
        let results = store.search("s1", &query, 2, None, 168.0, "retrieve").unwrap();
        assert_eq!(results[0].0.id, "c1");
    }

    #[test]
    fn search_empty_session() {
        let store = make_store();
        let query = vec![1.0, 0.0];
        let results = store.search("empty", &query, 5, None, 168.0, "organize").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_with_temporal_context_role() {
        let store = make_store();

        let chunks = vec![
            make_chunk("c1", l2_normalize(&[1.0, 0.0, 0.0, 0.0]), None),
            make_chunk("c2", l2_normalize(&[0.9, 0.1, 0.0, 0.0]), None),
        ];
        store.add_chunks("s1", chunks, None).unwrap();

        let query = l2_normalize(&[1.0, 0.0, 0.0, 0.0]);
        let results = store
            .search_with_temporal_context("s1", &query, 1, 60, "organize")
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, "c1");
        assert_eq!(results[0].2.len(), 1);
        assert_eq!(results[0].2[0].id, "c2");
    }

    #[test]
    fn encrypted_chunk_stored_in_memory() {
        let store = make_store();

        let mut chunk = make_chunk("enc1", vec![0.5, 0.5], None);
        chunk.encrypted = true;
        chunk.agent_id = Some("vajrayaksa".to_string());

        store.add_chunks("s1", vec![chunk], None).unwrap();

        let retrieved = store.get_chunks("s1").unwrap();
        assert_eq!(retrieved.len(), 1);
        assert!(retrieved[0].encrypted);
        assert_eq!(retrieved[0].agent_id.as_deref(), Some("vajrayaksa"));
    }
}
