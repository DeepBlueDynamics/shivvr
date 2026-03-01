use crate::chunker::Chunker;
use crate::crypto::CryptoManager;
use crate::embedder::Embedder;
use crate::inverter::Inverter;
use crate::openai::OpenAIEmbedder;
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
    pub openai_embedder: Option<Arc<OpenAIEmbedder>>,
    pub crypto: Arc<CryptoManager>,
    pub inverter: Option<Arc<Inverter>>,
    pub start_time: std::time::Instant,
}

// ===== Request/Response Types =====

#[derive(Deserialize)]
pub struct IngestRequest {
    pub text: String,
    pub source: Option<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
    /// Vedanā — emotion at time of memory creation
    pub emotion_primary: Option<String>,
    pub emotion_secondary: Option<String>,
    /// Agent ID for per-agent encryption
    pub agent_id: Option<String>,
}

#[derive(Serialize)]
pub struct IngestResponse {
    pub chunks_created: usize,
    pub tokens_processed: usize,
    pub time_ms: u64,
    pub chunks: Vec<IngestChunk>,
}

#[derive(Serialize)]
pub struct IngestChunk {
    pub chunk_id: String,
    pub text: String,
    pub embedding: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_retrieve: Option<Vec<f32>>,
    pub token_count: usize,
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotion_primary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotion_secondary: Option<String>,
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
    /// Embedding role: "organize" (768d gtr-t5-base) or "retrieve" (1536d ada-002)
    #[serde(default = "default_role")]
    pub role: String,
    /// Agent ID for encrypted search
    pub agent_id: Option<String>,
}

fn default_n() -> usize {
    5
}
fn default_time_window() -> i64 {
    30
}
fn default_role() -> String {
    "organize".to_string()
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
    pub emotion_primary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotion_secondary: Option<String>,
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
pub struct ModelInfo {
    pub name: String,
    pub role: String,
    pub dimension: usize,
    pub status: String,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub models: Vec<ModelInfo>,
    pub sessions: usize,
    pub total_chunks: usize,
    pub uptime_seconds: u64,
    pub encryption_available: bool,
    pub inversion_available: bool,
    pub gpu: bool,
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

    let emotion_primary = req.emotion_primary.clone();
    let emotion_secondary = req.emotion_secondary.clone();
    let agent_id = req.agent_id.clone();

    let mut chunks = state
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

    // Batch-embed with ada-002 if available (graceful degradation)
    if let Some(ref openai) = state.openai_embedder {
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        match openai.embed_batch(&texts).await {
            Ok(embeddings) => {
                for (chunk, emb) in chunks.iter_mut().zip(embeddings.into_iter()) {
                    chunk.embedding_retrieve = Some(emb);
                }
            }
            Err(e) => {
                tracing::warn!("OpenAI ada-002 embedding failed, continuing without: {}", e);
            }
        }
    }

    // Stamp emotion and agent_id onto every chunk
    for chunk in &mut chunks {
        chunk.emotion_primary = emotion_primary.clone();
        chunk.emotion_secondary = emotion_secondary.clone();
        chunk.agent_id = agent_id.clone();
    }

    // Encrypt embeddings if agent_id provided and keys exist
    if let Some(ref aid) = agent_id {
        if let Some(keys) = state.crypto.get_keys(aid) {
            for chunk in &mut chunks {
                chunk.embedding = keys.encrypt(&chunk.embedding, "organize");
                if let Some(ref emb) = chunk.embedding_retrieve {
                    chunk.embedding_retrieve = Some(keys.encrypt(emb, "retrieve"));
                }
                chunk.encrypted = true;
            }
        }
    }

    let chunks_created = chunks.len();
    let tokens_processed: usize = chunks.iter().map(|c| c.token_count).sum();

    let response_chunks: Vec<IngestChunk> = chunks
        .iter()
        .map(|c| IngestChunk {
            chunk_id: c.id.clone(),
            text: c.text.clone(),
            embedding: c.embedding.clone(),
            embedding_retrieve: c.embedding_retrieve.clone(),
            token_count: c.token_count,
            source: c.source.clone(),
            emotion_primary: c.emotion_primary.clone(),
            emotion_secondary: c.emotion_secondary.clone(),
        })
        .collect();

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
        chunks: response_chunks,
    }))
}

pub async fn search(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
    Query(query): Query<SearchQuery>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();
    let role = &query.role;

    // Embed query with the appropriate model for the role
    let mut query_embedding = if role == "retrieve" {
        // Try ada-002 first for retrieve role, fall back to gtr-base
        if let Some(ref openai) = state.openai_embedder {
            openai.embed(&query.q).await.map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("OpenAI embed failed: {}", e),
                    }),
                )
            })?
        } else {
            state.embedder.embed(&query.q).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: e.to_string(),
                    }),
                )
            })?
        }
    } else {
        state.embedder.embed(&query.q).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?
    };

    // Encrypt query embedding if agent_id provided
    if let Some(ref aid) = query.agent_id {
        if let Some(keys) = state.crypto.get_keys(aid) {
            query_embedding = keys.encrypt(&query_embedding, role);
        }
    }

    let results = if query.include_nearby.unwrap_or(false) {
        let results_with_context = state
            .store
            .search_with_temporal_context(
                &session_id,
                &query_embedding,
                query.n,
                query.time_window_minutes,
                role,
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
                emotion_primary: chunk.emotion_primary,
                emotion_secondary: chunk.emotion_secondary,
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
        let results = state
            .store
            .search(&session_id, &query_embedding, query.n, query.time_weight, role)
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
                emotion_primary: chunk.emotion_primary,
                emotion_secondary: chunk.emotion_secondary,
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

    let mut models = vec![ModelInfo {
        name: "gtr-t5-base".to_string(),
        role: "organize".to_string(),
        dimension: 768,
        status: "active".to_string(),
    }];

    if state.openai_embedder.is_some() {
        models.push(ModelInfo {
            name: "text-embedding-ada-002".to_string(),
            role: "retrieve".to_string(),
            dimension: 1536,
            status: "active".to_string(),
        });
    }

    Json(HealthResponse {
        status: "ok".to_string(),
        models,
        sessions: sessions.len(),
        total_chunks,
        uptime_seconds: state.start_time.elapsed().as_secs(),
        encryption_available: true,
        inversion_available: state.inverter.is_some(),
        gpu: cfg!(feature = "cuda"),
    })
}

// ===== Phase 2: Crypto Endpoints =====

#[derive(Deserialize)]
pub struct RegisterAgentRequest {
    /// Flattened row-major orthogonal matrix for organize role (dim^2 floats)
    pub organize_key: Vec<f32>,
    /// Dimension of the organize key matrix
    pub organize_dim: usize,
    /// Optional flattened retrieve key matrix
    pub retrieve_key: Option<Vec<f32>>,
    /// Dimension of the retrieve key matrix
    pub retrieve_dim: Option<usize>,
}

#[derive(Serialize)]
pub struct RegisterAgentResponse {
    pub agent_id: String,
    pub organize_dim: usize,
    pub retrieve_dim: Option<usize>,
}

pub async fn register_agent(
    State(state): State<Arc<AppState>>,
    Path(agent_id): Path<String>,
    Json(req): Json<RegisterAgentRequest>,
) -> Result<Json<RegisterAgentResponse>, (StatusCode, Json<ErrorResponse>)> {
    let retrieve_dim = req.retrieve_dim;

    state
        .crypto
        .register_keys(
            &agent_id,
            &req.organize_key,
            req.organize_dim,
            req.retrieve_key.as_deref(),
            req.retrieve_dim,
        )
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

    Ok(Json(RegisterAgentResponse {
        agent_id,
        organize_dim: req.organize_dim,
        retrieve_dim,
    }))
}

#[derive(Deserialize)]
pub struct CryptoRequest {
    pub embeddings: Vec<Vec<f32>>,
    #[serde(default = "default_role")]
    pub role: String,
}

#[derive(Serialize)]
pub struct CryptoResponse {
    pub embeddings: Vec<Vec<f32>>,
}

pub async fn decrypt_embeddings(
    State(state): State<Arc<AppState>>,
    Path(agent_id): Path<String>,
    Json(req): Json<CryptoRequest>,
) -> Result<Json<CryptoResponse>, (StatusCode, Json<ErrorResponse>)> {
    let keys = state.crypto.get_keys(&agent_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("No keys registered for agent {}", agent_id),
            }),
        )
    })?;

    let decrypted: Vec<Vec<f32>> = req
        .embeddings
        .iter()
        .map(|emb| keys.decrypt(emb, &req.role))
        .collect();

    Ok(Json(CryptoResponse {
        embeddings: decrypted,
    }))
}

pub async fn encrypt_embeddings(
    State(state): State<Arc<AppState>>,
    Path(agent_id): Path<String>,
    Json(req): Json<CryptoRequest>,
) -> Result<Json<CryptoResponse>, (StatusCode, Json<ErrorResponse>)> {
    let keys = state.crypto.get_keys(&agent_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("No keys registered for agent {}", agent_id),
            }),
        )
    })?;

    let encrypted: Vec<Vec<f32>> = req
        .embeddings
        .iter()
        .map(|emb| keys.encrypt(emb, &req.role))
        .collect();

    Ok(Json(CryptoResponse {
        embeddings: encrypted,
    }))
}

// ===== Phase 3: Inversion Endpoint =====

#[derive(Deserialize)]
pub struct InvertRequest {
    pub embedding: Vec<f32>,
    #[serde(default = "default_role")]
    pub role: String,
}

#[derive(Serialize)]
pub struct InvertResponse {
    pub text: String,
    pub similarity: f32,
}

pub async fn invert(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InvertRequest>,
) -> Result<Json<InvertResponse>, (StatusCode, Json<ErrorResponse>)> {
    let inverter = state.inverter.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Vec2text inverter not loaded (no ONNX models available)".to_string(),
            }),
        )
    })?;

    let result = inverter.invert(&req.embedding).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Inversion failed: {}", e),
            }),
        )
    })?;

    // Re-embed the reconstructed text and compute similarity
    let re_embedded = state.embedder.embed(&result).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Re-embedding failed: {}", e),
            }),
        )
    })?;

    let similarity = crate::similarity::cosine_similarity(&req.embedding, &re_embedded);

    Ok(Json(InvertResponse {
        text: result,
        similarity,
    }))
}

// ===== Router =====

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/memory", get(list_sessions))
        .route("/memory/:session_id/ingest", post(ingest))
        .route("/memory/:session_id/search", get(search))
        .route("/memory/:session_id/info", get(session_info))
        .route("/memory/:session_id", delete(delete_session))
        // Phase 2: Crypto endpoints
        .route("/agent/:agent_id/register", post(register_agent))
        .route("/agent/:agent_id/decrypt", post(decrypt_embeddings))
        .route("/agent/:agent_id/encrypt", post(encrypt_embeddings))
        // Phase 3: Inversion endpoint
        .route("/invert", post(invert))
        .with_state(state)
}
