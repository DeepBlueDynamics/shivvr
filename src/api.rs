use crate::chunker::Chunker;
use crate::crypto::CryptoManager;
use crate::embedder::Embedder;
use crate::inverter::Inverter;
use crate::openai::OpenAIEmbedder;
use crate::auth::NutsAuth;
use crate::store::Store;
use crate::temp_store::TempStore;
use axum::{
    extract::{Path, Query, Request, State},
    http::{Method, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Json, Response},
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, HashMap};
use std::sync::Arc;

pub struct AppState {
    pub store: Arc<Store>,
    pub temp_store: Arc<TempStore>,
    pub chunker: Arc<Chunker>,
    pub embedder: Arc<Embedder>,
    pub openai_embedder: Option<Arc<OpenAIEmbedder>>,
    pub crypto: Arc<CryptoManager>,
    pub inverter: Option<Arc<Inverter>>,
    pub start_time: std::time::Instant,
    pub nuts_auth: Option<Arc<NutsAuth>>,
    pub openai_auth_required: bool,
    pub guardrail_tagger: Arc<std::sync::RwLock<lume_hybrid::Tagger>>,
    /// BM25 tuning params (k1/b/delta/title_weight/body_weight). Lume's
    /// successor to the old SearchConfig — variant + tagger get selected
    /// per call rather than living on the global config.
    pub search_params: lume_hybrid::bm25::Bm25Params,
    pub mcp_connections: Arc<tokio::sync::RwLock<HashMap<String, tokio::sync::mpsc::UnboundedSender<Result<axum::response::sse::Event, std::convert::Infallible>>>>>,
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
    /// Caller-supplied OpenAI API key — enables retrieve embedding without server key
    pub openai_api_key: Option<String>,
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
    /// Half-life for temporal decay in hours (default 168 = one week).
    /// Only used when time_weight > 0. Smaller values favour recent chunks more aggressively.
    #[serde(default = "default_decay_halflife")]
    pub decay_halflife_hours: f32,
    #[serde(default)]
    pub include_nearby: Option<bool>,
    #[serde(default = "default_time_window")]
    pub time_window_minutes: i64,
    /// Embedding role: "organize" (768d gtr-t5-base) or "retrieve" (1536d ada-002)
    #[serde(default = "default_role")]
    pub role: String,
    /// Agent ID for encrypted search
    pub agent_id: Option<String>,
    /// Caller-supplied OpenAI API key — enables retrieve role without server key
    pub openai_api_key: Option<String>,
    
    // NEW FIELDS
    #[serde(default)]
    pub hybrid: Option<bool>,
    #[serde(default)]
    pub lexical_only: Option<bool>,
    #[serde(default = "default_true")]
    pub guardrail: Option<bool>,
}

fn default_true() -> Option<bool> {
    Some(true)
}

fn default_n() -> usize {
    5
}
fn default_time_window() -> i64 {
    30
}
fn default_decay_halflife() -> f32 {
    168.0
}
fn default_role() -> String {
    "organize".to_string()
}
fn default_max_length() -> usize {
    64
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
pub struct TempStoreListResponse {
    pub stores: Vec<TempStoreListItem>,
}

#[derive(Serialize)]
pub struct TempStoreListItem {
    pub name: String,
    pub chunks: usize,
    pub expires_at: String,
}

#[derive(Serialize)]
pub struct TempDumpResponse {
    pub name: String,
    pub chunks: Vec<crate::store::Chunk>,
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
    pub version: String,
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
    claims: Option<axum::Extension<crate::auth::NutsAuthClaims>>,
    Json(req): Json<IngestRequest>,
) -> Result<Json<IngestResponse>, (StatusCode, Json<ErrorResponse>)> {
    let caller_id = claims.as_ref().map(|c| c.user_id.as_str());
    if !state.store.caller_owns_session(&session_id, caller_id) {
        return Err((StatusCode::NOT_FOUND, Json(ErrorResponse { error: "session not found".to_string() })));
    }
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

    // Batch-embed with ada-002 — prefer caller-supplied key, fall back to server key
    let openai_for_request = req.openai_api_key
        .as_deref()
        .map(|k| crate::openai::OpenAIEmbedder::new(k.to_string()))
        .transpose()
        .ok()
        .flatten()
        .map(Arc::new);
    let effective_openai = openai_for_request.as_ref().or(state.openai_embedder.as_ref());

    if let Some(openai) = effective_openai {
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

    state.store.add_chunks(&session_id, chunks, caller_id).map_err(|e| {
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

pub async fn temp_ingest(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
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

    let openai_for_request = req.openai_api_key
        .as_deref()
        .map(|k| crate::openai::OpenAIEmbedder::new(k.to_string()))
        .transpose()
        .ok()
        .flatten()
        .map(Arc::new);
    let effective_openai = openai_for_request.as_ref().or(state.openai_embedder.as_ref());

    if let Some(openai) = effective_openai {
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        match openai.embed_batch(&texts).await {
            Ok(embeddings) => {
                for (chunk, emb) in chunks.iter_mut().zip(embeddings.into_iter()) {
                    chunk.embedding_retrieve = Some(emb);
                }
            }
            Err(e) => {
                tracing::warn!("OpenAI retrieve embedding failed, continuing without: {}", e);
            }
        }
    }

    for chunk in &mut chunks {
        chunk.emotion_primary = emotion_primary.clone();
        chunk.emotion_secondary = emotion_secondary.clone();
        chunk.agent_id = agent_id.clone();
    }

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

    state.temp_store.add_chunks(&name, chunks).map_err(|e| {
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
    claims: Option<axum::Extension<crate::auth::NutsAuthClaims>>,
    Query(query): Query<SearchQuery>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();
    let role = &query.role;
    let caller_id = claims.as_ref().map(|c| c.user_id.as_str());

    if !state.store.caller_owns_session(&session_id, caller_id) {
        return Err((StatusCode::NOT_FOUND, Json(ErrorResponse { error: "session not found".to_string() })));
    }

    // 1. Run FST Tagger Guardrail
    let tags = state.guardrail_tagger.read().unwrap().tag(&query.q);
    if query.guardrail.unwrap_or(true) {
        let blocked_terms: Vec<String> = tags
            .iter()
            .filter(|t| t.kind.contains("offensive") || t.output == "BLOCK")
            .map(|t| t.surface.clone())
            .collect();
        if !blocked_terms.is_empty() {
            return Err((
                StatusCode::FORBIDDEN,
                Json(ErrorResponse {
                    error: format!("Query blocked by safety guardrail. Blocked terms: {:?}", blocked_terms),
                }),
            ));
        }
    }

    // 2. Generate Query Embedding (unless lexical-only requested)
    let query_embedding = if query.lexical_only.unwrap_or(false) {
        None
    } else {
        let openai_for_request = query.openai_api_key
            .as_deref()
            .map(|k| crate::openai::OpenAIEmbedder::new(k.to_string()))
            .transpose()
            .ok()
            .flatten()
            .map(Arc::new);
        let effective_openai = openai_for_request.as_ref().or(state.openai_embedder.as_ref());

        if role == "retrieve" && effective_openai.is_none() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "role=retrieve requires an OpenAI API key (set OPENAI_API_KEY or pass openai_api_key in the request)".to_string(),
                }),
            ));
        }

        let mut emb = if role == "retrieve" {
            let openai = effective_openai.as_ref().expect("checked above");
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
        };

        if let Some(ref aid) = query.agent_id {
            if let Some(keys) = state.crypto.get_keys(aid) {
                emb = keys.encrypt(&emb, role);
            }
        }
        Some(emb)
    };

    // 3. Routing & Execution
    let results = if query.hybrid.unwrap_or(false) || query.lexical_only.unwrap_or(false) {
        let raw_results = state
            .store
            .search_hybrid(
                &session_id,
                &query.q,
                query_embedding.as_deref(),
                query.n,
                query.time_weight,
                query.decay_halflife_hours,
                role,
                &tags,
                &state.search_params,
            )
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: e.to_string(),
                    }),
                )
            })?;

        raw_results
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
    } else {
        let emb = query_embedding.as_ref().unwrap();
        if query.include_nearby.unwrap_or(false) {
            let results_with_context = state
                .store
                .search_with_temporal_context(
                    &session_id,
                    emb,
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
                .search(&session_id, emb, query.n, query.time_weight, query.decay_halflife_hours, role)
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
        }
    };

    Ok(Json(SearchResponse {
        query: query.q,
        results,
        time_ms: start.elapsed().as_millis() as u64,
    }))
}

pub async fn temp_search(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(query): Query<SearchQuery>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();
    let role = &query.role;

    // 1. Run FST Tagger Guardrail
    let tags = state.guardrail_tagger.read().unwrap().tag(&query.q);
    if query.guardrail.unwrap_or(true) {
        let blocked_terms: Vec<String> = tags
            .iter()
            .filter(|t| t.kind.contains("offensive") || t.output == "BLOCK")
            .map(|t| t.surface.clone())
            .collect();
        if !blocked_terms.is_empty() {
            return Err((
                StatusCode::FORBIDDEN,
                Json(ErrorResponse {
                    error: format!("Query blocked by safety guardrail. Blocked terms: {:?}", blocked_terms),
                }),
            ));
        }
    }

    // 2. Generate Query Embedding (unless lexical-only requested)
    let query_embedding = if query.lexical_only.unwrap_or(false) {
        None
    } else {
        if role == "retrieve" && state.openai_embedder.is_none() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "role=retrieve requires OPENAI embedder configuration".to_string(),
                }),
            ));
        }

        let mut emb = if role == "retrieve" {
            let openai = state.openai_embedder.as_ref().expect("checked above");
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
        };

        if let Some(ref aid) = query.agent_id {
            if let Some(keys) = state.crypto.get_keys(aid) {
                emb = keys.encrypt(&emb, role);
            }
        }
        Some(emb)
    };

    // 3. Routing & Execution
    let results = if query.hybrid.unwrap_or(false) || query.lexical_only.unwrap_or(false) {
        let raw_results = state
            .temp_store
            .search_hybrid(
                &name,
                &query.q,
                query_embedding.as_deref(),
                query.n,
                query.time_weight,
                query.decay_halflife_hours,
                role,
                &tags,
                &state.search_params,
            )
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: e.to_string(),
                    }),
                )
            })?;

        raw_results
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
    } else {
        let emb = query_embedding.as_ref().unwrap();
        if query.include_nearby.unwrap_or(false) {
            let results_with_context = state
                .temp_store
                .search_with_temporal_context(
                    &name,
                    emb,
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
                .temp_store
                .search(
                    &name,
                    emb,
                    query.n,
                    query.time_weight,
                    query.decay_halflife_hours,
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
        }
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
    claims: Option<axum::Extension<crate::auth::NutsAuthClaims>>,
) -> Result<Json<SessionInfoResponse>, StatusCode> {
    let caller_id = claims.as_ref().map(|c| c.user_id.as_str());
    if !state.store.caller_owns_session(&session_id, caller_id) {
        return Err(StatusCode::NOT_FOUND); // 404 not 403 — don't confirm existence
    }
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
    claims: Option<axum::Extension<crate::auth::NutsAuthClaims>>,
) -> Result<Json<DeleteResponse>, StatusCode> {
    let caller_id = claims.as_ref().map(|c| c.user_id.as_str());
    if !state.store.caller_owns_session(&session_id, caller_id) {
        return Err(StatusCode::NOT_FOUND);
    }
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

pub async fn list_sessions(
    State(state): State<Arc<AppState>>,
    claims: Option<axum::Extension<crate::auth::NutsAuthClaims>>,
) -> Json<ListSessionsResponse> {
    let caller_id = claims.as_ref().map(|c| c.user_id.as_str());
    let sessions = state.store.list_sessions(caller_id).unwrap_or_default();

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

pub async fn list_temp_stores(State(state): State<Arc<AppState>>) -> Json<TempStoreListResponse> {
    let stores = state.temp_store.list_stores();

    Json(TempStoreListResponse {
        stores: stores
            .into_iter()
            .map(|s| TempStoreListItem {
                name: s.name,
                chunks: s.chunk_count,
                expires_at: s.expires_at.to_rfc3339(),
            })
            .collect(),
    })
}

pub async fn temp_dump(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Json<TempDumpResponse> {
    let chunks = state.temp_store.get_chunks(&name).unwrap_or_default();

    Json(TempDumpResponse { name, chunks })
}

pub async fn delete_temp_store(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<DeleteResponse>, StatusCode> {
    let deleted = state
        .temp_store
        .delete_store(&name)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if deleted == 0 {
        return Err(StatusCode::NOT_FOUND);
    }

    Ok(Json(DeleteResponse {
        deleted_chunks: deleted,
        session: name,
    }))
}

pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let sessions = state.store.list_sessions(None).unwrap_or_default();
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
        version: env!("CARGO_PKG_VERSION").to_string(),
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
    /// Maximum tokens to generate (default 64)
    #[serde(default = "default_max_length")]
    pub max_length: usize,
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

    let result = inverter.invert(&req.embedding, req.max_length).map_err(|e| {
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

// ===== Homepage =====

pub async fn homepage(State(state): State<Arc<AppState>>) -> Html<String> {
    let sessions = state.store.list_sessions(None).unwrap_or_default().len();
    let chunks = state.store.total_chunks().unwrap_or(0);
    let uptime = state.start_time.elapsed().as_secs();
    let gpu = if cfg!(feature = "cuda") { "CUDA" } else { "CPU" };
    let inversion = if state.inverter.is_some() { "enabled" } else { "disabled" };
    let version = env!("CARGO_PKG_VERSION");

    Html(format!(r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>shivvr — semantic embedding & cognitive agent service</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%230a0a0a'/%3E%3Cpath d='M10 11 Q16 8 22 11 Q16 14 10 17 Q16 20 22 23' stroke='%2300c2a8' stroke-width='2.2' fill='none' stroke-linecap='round'/%3E%3C/svg%3E">
<meta name="description" content="Ephemeral semantic embedding & cognitive agent service. Chunk text, embed with GTR-T5-base (768d), search with hybrid BM25 + FST routers, and run autonomous tool reasoning loops.">
<meta property="og:type" content="website">
<meta property="og:url" content="https://shivvr.nuts.services">
<meta property="og:title" content="shivvr — semantic embedding & cognitive agent service">
<meta property="og:description" content="Chunk. Embed. Hybrid Search. Autonomous GhostAgent. Ephemeral RAG + MCP server. Rust + ONNX Runtime on GPU. No disk. No state.">
<meta name="twitter:card" content="summary">
<meta name="twitter:site" content="@deepbluedynamic">
<meta name="twitter:title" content="shivvr — semantic embedding & cognitive agent service">
<meta name="twitter:description" content="Chunk. Embed. Hybrid Search. Autonomous GhostAgent. Ephemeral RAG + MCP server. Rust + ONNX Runtime on GPU. No disk. No state.">
<style>
  :root {{
    --bg: #0a0a0f; --fg: #c8c8d0; --accent: #00c2a8; --accent2: #7b68ee;
    --dim: #555568; --card: #12121a; --border: #1e1e2e; --green: #50fa7b;
    --red: #ff5555;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    background: var(--bg); color: var(--fg);
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 14px; line-height: 1.75;
    padding: 2.5rem 2rem; max-width: 900px; margin: 0 auto;
  }}

  /* ── Hero ── */
  .hero {{ margin-bottom: 3rem; }}
  .hero h1 {{
    font-size: clamp(2rem, 5vw, 3rem);
    color: var(--accent); letter-spacing: -0.03em;
    line-height: 1.1; margin-bottom: 0.4rem;
  }}
  .hero h1 .ver {{ color: var(--dim); font-weight: 300; font-size: 0.42em; vertical-align: middle; }}
  .hero .sub {{
    color: var(--fg); font-size: 1.05rem; margin-bottom: 0.5rem;
  }}
  .hero .desc {{ color: var(--dim); font-size: 0.85rem; margin-bottom: 1.75rem; max-width: 560px; }}
  .ctas {{ display: flex; gap: 0.75rem; flex-wrap: wrap; }}
  .cta {{
    display: inline-block;
    padding: 0.55rem 1.25rem; border-radius: 6px; font-size: 0.82rem;
    text-decoration: none; transition: all 0.15s;
    border: 1px solid var(--accent); color: var(--accent);
  }}
  .cta:hover {{ background: var(--accent); color: #fff; text-decoration: none; }}
  .cta.dim {{ border-color: var(--border); color: var(--dim); }}
  .cta.dim:hover {{ border-color: var(--dim); background: transparent; color: var(--fg); }}

  /* ── Stats strip ── */
  .stats {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 1rem; margin-bottom: 3rem;
  }}
  .stat {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem 1.2rem;
  }}
  .stat .val {{ font-size: 1.6rem; color: var(--green); font-weight: 700; line-height: 1.2; }}
  .stat .lbl {{
    font-size: 0.7rem; color: var(--dim);
    text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.15rem;
  }}

  /* ── Section headings ── */
  h2 {{
    color: var(--accent); font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin: 2.5rem 0 0.9rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }}

  /* ── Capability cards ── */
  .features {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.9rem; margin-bottom: 0.5rem;
  }}
  .feature {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem 1.2rem;
    transition: border-color 0.15s;
  }}
  .feature:hover {{ border-color: var(--accent); }}
  .feature .name {{ color: var(--green); font-weight: 600; margin-bottom: 0.3rem; }}
  .feature p {{ color: var(--dim); font-size: 0.82rem; line-height: 1.6; }}

  /* ── Tables ── */
  .tbl-wrap {{ overflow-x: auto; margin: 0.25rem 0 0.5rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th, td {{ text-align: left; padding: 0.45rem 0.8rem; border-bottom: 1px solid var(--border); }}
  th {{
    color: var(--dim); font-weight: 600; font-size: 0.68rem;
    text-transform: uppercase; letter-spacing: 0.1em;
  }}
  td:first-child {{ color: var(--dim); white-space: nowrap; }}
  td code {{ color: var(--green); }}
  .method {{ color: var(--accent2); font-size: 0.75rem; }}
  tr:last-child td {{ border-bottom: none; }}

  /* ── Code blocks ── */
  pre {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 1.1rem 1.25rem;
    overflow-x: auto; font-size: 0.8rem; line-height: 1.7;
    margin: 0.25rem 0;
  }}
  .cm {{ color: var(--dim); }}
  .kw {{ color: var(--accent2); }}

  /* ── Inline ── */
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  code {{ color: var(--green); font-size: 0.82rem; }}

  /* ── Footer links ── */
  .links {{ display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 3rem; }}
  .links a {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 6px; padding: 0.5rem 1.1rem; font-size: 0.8rem;
    transition: border-color 0.15s;
  }}
  .links a:hover {{ border-color: var(--accent); text-decoration: none; }}

  .footer {{
    color: var(--dim); font-size: 0.72rem; margin-top: 1.5rem;
    padding-top: 1.5rem; border-top: 1px solid var(--border);
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;
  }}
  .dot {{ color: var(--border); }}
</style>
</head>
<body>

<!-- Hero -->
<div class="hero">
  <h1>shivvr 🔪 <span class="ver">v{version}</span></h1>
  <p class="sub">Ephemeral semantic embedding & cognitive agent service.</p>
  <p class="desc">
    Chunk text. Embed with GTR-T5-base. Search via hybrid FST-BM25 vector rank fusion. Stream turn-based cognitive agent reasoning via native Model Context Protocol (MCP).
  </p>
  <div class="ctas">
    <a class="cta" href="#api">View API</a>
    <a class="cta" href="#quickstart">Quick Start</a>
    <a class="cta" href="https://github.com/DeepBlueDynamics/shivvr">GitHub</a>
    <a class="cta dim" href="/health">Health</a>
  </div>
</div>

<!-- Live stats -->
<div class="stats">
  <div class="stat"><div class="val" id="s-uptime">{uptime}s</div><div class="lbl">Uptime</div></div>
  <div class="stat"><div class="val" id="s-gpu">{gpu}</div><div class="lbl">Compute</div></div>
  <div class="stat"><div class="val" id="s-sessions">{sessions}</div><div class="lbl">Sessions</div></div>
  <div class="stat"><div class="val" id="s-chunks">{chunks}</div><div class="lbl">Chunks</div></div>
  <div class="stat"><div class="val" id="s-enc">&#x2713;</div><div class="lbl">Encryption</div></div>
  <div class="stat"><div class="val" id="s-inv">{inversion}</div><div class="lbl">Inversion</div></div>
</div>

<!-- Capabilities -->
<h2>Capabilities</h2>
<div class="features">
  <div class="feature">
    <div class="name">Ingest</div>
    <p>Sentence-boundary chunking + GTR-T5-base embeddings (768d). Stores in RwLock&lt;HashMap&gt; — pure ephemeral compute.</p>
  </div>
  <div class="feature">
    <div class="name">FST-BM25 Hybrid Search</div>
    <p>RRF blending dense vectors and sparse lexical indices. FST dictionary scanning provides microsecond safe query guardrails and intent-entity score boosting.</p>
  </div>
  <div class="feature">
    <div class="name">Cognitive GhostAgent</div>
    <p>Turn-based agent loop with integrated memory search, document ingestion, session indexing, and sandboxed command execution. Powered by OpenAI/Anthropic.</p>
  </div>
  <div class="feature">
    <div class="name">Native MCP Server</div>
    <p>Complete Model Context Protocol HTTP/SSE server endpoint (<code>/mcp/sse</code>) allowing immediate, zero-config integration with Claude Code, Antigravity, and Codex.</p>
  </div>
  <div class="feature">
    <div class="name">Crypto</div>
    <p>Per-agent orthogonal matrix rotation on embeddings. Cosine similarity preserved under encryption. Keys are in-memory only.</p>
  </div>
  <div class="feature">
    <div class="name">Dual embedding</div>
    <p><code>organize</code> role uses GTR-T5-base (768d). <code>retrieve</code> role uses OpenAI text-embedding-ada-002 (1536d) — pass your own key or set server-side.</p>
  </div>
</div>

<!-- API -->
<h2 id="api">API</h2>
<div class="tbl-wrap">
<table>
  <tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
  <tr><td class="method">GET</td><td><code>/health</code></td><td>Status, model info, live counts</td></tr>
  <tr><td class="method">GET</td><td><code>/mcp/sse</code></td><td>MCP Server SSE handshake</td></tr>
  <tr><td class="method">POST</td><td><code>/mcp/message</code></td><td>MCP Server JSON-RPC message router</td></tr>
  <tr><td class="method">POST</td><td><code>/sessions/:id/agent/chat</code></td><td>Stream non-blocking GhostAgent cognitive turns (SSE)</td></tr>
  <tr><td class="method">POST</td><td><code>/sessions/:id/ingest</code></td><td>Chunk + embed text into session</td></tr>
  <tr><td class="method">GET</td><td><code>/sessions/:id/search?q=...</code></td><td>Semantic search (supports RRF <code>hybrid</code> &amp; <code>lexical_only</code>)</td></tr>
  <tr><td class="method">GET</td><td><code>/sessions/:id</code></td><td>Session metadata</td></tr>
  <tr><td class="method">DELETE</td><td><code>/sessions/:id</code></td><td>Delete session</td></tr>
  <tr><td class="method">GET</td><td><code>/temp</code></td><td>List temp stores with TTL</td></tr>
  <tr><td class="method">POST</td><td><code>/temp/:name/ingest</code></td><td>Ingest into temp store (2 hr TTL)</td></tr>
  <tr><td class="method">GET</td><td><code>/temp/:name/search?q=...</code></td><td>Search temp store</td></tr>
  <tr><td class="method">DELETE</td><td><code>/temp/:name</code></td><td>Delete temp store</td></tr>
  <tr><td class="method">POST</td><td><code>/agent/:id/register</code></td><td>Register per-agent orthogonal key</td></tr>
  <tr><td class="method">POST</td><td><code>/agent/:id/encrypt</code></td><td>Encrypt embeddings</td></tr>
  <tr><td class="method">POST</td><td><code>/agent/:id/decrypt</code></td><td>Decrypt embeddings</td></tr>
  <tr><td class="method">POST</td><td><code>/invert</code></td><td>Reconstruct text from embedding vector</td></tr>
</table>
</div>

<!-- Quick start -->
<h2 id="quickstart">Quick start</h2>
<pre><code><span class="cm"># Ingest into session</span>
curl -X POST https://shivvr.nuts.services/sessions/my-session/ingest \
  -H "Content-Type: application/json" \
  -d '{{"text": "Supreme Raven is protected by Known Opossum.", "source": "vault_specs"}}'

<span class="cm"># Autonomous Agent Conversational Chat (Streams Thoughts, ToolCalls, &amp; Answer via SSE)</span>
curl -i -X POST http://localhost:8085/sessions/my-session/agent/chat \
  -H "Content-Type: application/json" \
  -d '{{"message": "Who protects the Supreme Raven?"}}'

<span class="cm"># Vector-Lexical Hybrid RRF Search</span>
curl "http://localhost:8085/sessions/my-session/search?q=Known+Opossum&amp;hybrid=true"

<span class="cm"># High-speed Lexical-Only BM25 Search (Bypasses ONNX embedder)</span>
curl "http://localhost:8085/sessions/my-session/search?q=Opossum&amp;lexical_only=true"

<span class="cm"># Synchronize Claude Code or Antigravity with shivvr's Native MCP Server</span>
nemesis8 mcp add http://localhost:8085/mcp/sse
</code></pre>

<!-- Search params -->
<h2>Search parameters</h2>
<div class="tbl-wrap">
<table>
  <tr><th>Param</th><th>Default</th><th>Description</th></tr>
  <tr><td><code>q</code></td><td>required</td><td>Query text</td></tr>
  <tr><td><code>n</code></td><td>5</td><td>Number of results</td></tr>
  <tr><td><code>hybrid</code></td><td>false</td><td>Blend semantic vectors + BM25 scores (Reciprocal Rank Fusion)</td></tr>
  <tr><td><code>lexical_only</code></td><td>false</td><td>Bypass vector embedder, execute pure BM25 search</td></tr>
  <tr><td><code>guardrail</code></td><td>true</td><td>Enable FST toxic term scanning and automatic query blocking</td></tr>
  <tr><td><code>role</code></td><td>organize</td><td><code>organize</code> (768d local) or <code>retrieve</code> (1536d OpenAI)</td></tr>
  <tr><td><code>time_weight</code></td><td>0.0</td><td>Blend semantic + recency score (0–1)</td></tr>
  <tr><td><code>decay_halflife_hours</code></td><td>168</td><td>Recency decay half-life in hours</td></tr>
  <tr><td><code>include_nearby</code></td><td>false</td><td>Return temporally adjacent chunks</td></tr>
  <tr><td><code>agent_id</code></td><td>—</td><td>Agent ID for encrypted search</td></tr>
  <tr><td><code>openai_api_key</code></td><td>—</td><td>Per-request OpenAI key for <code>retrieve</code> role (overrides server key)</td></tr>
</table>
</div>

<!-- Environment -->
<h2>Environment</h2>
<div class="tbl-wrap">
<table>
  <tr><th>Variable</th><th>Default</th><th>Description</th></tr>
  <tr><td><code>PORT</code></td><td>8080</td><td>Listen port</td></tr>
  <tr><td><code>MODEL_PATH</code></td><td>models/gtr-t5-base.onnx</td><td>GTR-T5-base ONNX embedder</td></tr>
  <tr><td><code>TOKENIZER_PATH</code></td><td>models/tokenizer.json</td><td>Tokenizer</td></tr>
  <tr><td><code>OPENAI_API_KEY</code></td><td>—</td><td>Enables OpenAI completions and retrieve embeddings</td></tr>
  <tr><td><code>ANTHROPIC_API_KEY</code></td><td>—</td><td>Enables Anthropic completions and GhostAgent loops</td></tr>
  <tr><td><code>NUTS_AUTH_JWKS_URL</code></td><td>—</td><td>Enable auth (open dev mode if unset)</td></tr>
  <tr><td><code>NUTS_AUTH_VALIDATE_URL</code></td><td>https://auth.nuts.services/api/validate</td><td>API token validation endpoint</td></tr>
</table>
</div>

<!-- Stack -->
<h2>Stack</h2>
<div class="tbl-wrap">
<table>
  <tr><th>Layer</th><th>Choice</th></tr>
  <tr><td>Runtime</td><td>Rust + Tokio + axum</td></tr>
  <tr><td>Cognition</td><td>GhostAgent cognitive RAG turn loop (OpenAI / Anthropic compat)</td></tr>
  <tr><td>MCP Server</td><td>HTTP/SSE JSON-RPC 2.0 Model Context Protocol transport layer</td></tr>
  <tr><td>Hybrid Index</td><td>Tantivy FST deterministic phrase engine + BM25F field indexer</td></tr>
  <tr><td>Embedding</td><td>GTR-T5-base (768d) via ONNX Runtime 2.0 — local, required</td></tr>
  <tr><td>Storage</td><td>Ephemeral RwLock&lt;HashMap&gt; — no disk, no volume mounts</td></tr>
  <tr><td>GPU</td><td>CUDA 12.6 via ort EP on Cloud Run L4 — CPU fallback automatic</td></tr>
  <tr><td>Inversion</td><td>vec2text gtr-base (projection + T5 enc/dec) — optional</td></tr>
</table>
</div>

<!-- Footer links -->
<div class="links">
  <a href="https://github.com/DeepBlueDynamics/shivvr">GitHub</a>
  <a href="/health">Health JSON</a>
  <a href="/sessions">Sessions</a>
</div>

<div class="footer">
  <span>shivvr <span class="dot">&middot;</span> Rust + ONNX Runtime</span>
  <span>
    <a href="https://deepbluedynamics.com">DeepBlueDynamics</a>
    <span class="dot">&middot;</span>
    <a href="https://nuts.services">nuts.services</a>
    <span class="dot">&middot;</span>
    <a href="https://hyperia.nuts.services">hyperia</a>
    <span class="dot">&middot;</span>
    <a href="https://twitter.com/deepbluedynamic">@deepbluedynamic</a>
  </span>
</div>

<script>
(function() {{
  function fmt(n) {{
    if (typeof n !== 'number') return n;
    if (n >= 1e6) return (n/1e6).toFixed(1) + 'M';
    if (n >= 1e3) return (n/1e3).toFixed(1) + 'k';
    return String(n);
  }}
  function counter(el, target, suffix) {{
    if (!el) return;
    if (typeof target !== 'number' || target === 0) {{ el.textContent = fmt(target) + (suffix||''); return; }}
    var start = 0, dur = 800, step = 16;
    var t = setInterval(function() {{
      start += step;
      var pct = Math.min(start/dur, 1);
      var val = Math.round(pct * target);
      el.textContent = fmt(val) + (suffix||'');
      if (pct >= 1) clearInterval(t);
    }}, step);
  }}
  fetch('/health').then(function(r) {{ return r.json(); }}).then(function(d) {{
    counter(document.getElementById('s-sessions'), d.sessions || 0, '');
    counter(document.getElementById('s-chunks'), d.total_chunks || 0, '');
    counter(document.getElementById('s-uptime'), d.uptime_seconds || 0, 's');
    document.getElementById('s-gpu').textContent = d.gpu ? 'CUDA' : 'CPU';
    document.getElementById('s-enc').textContent = d.encryption_available ? '\u2713' : '\u2715';
    document.getElementById('s-inv').textContent = d.inversion_available ? 'on' : 'off';
  }}).catch(function() {{}});
}})();
</script>

</body>
</html>"##))
}

// ===== Auth Middleware =====

fn extract_bearer(req: &Request) -> Option<String> {
    req.headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|s| s.to_string())
}

fn query_role(query: Option<&str>) -> String {
    query
        .and_then(|q| {
            q.split('&').find_map(|pair| {
                let (key, value) = pair.split_once('=')?;
                if key == "role" {
                    Some(value.to_string())
                } else {
                    None
                }
            })
        })
        .unwrap_or_else(|| "organize".to_string())
}

fn is_free_operation(path: &str, method: &Method, query: Option<&str>, state: &AppState) -> bool {
    match (method, path) {
        (&Method::GET, "/") => true,
        (&Method::GET, "/health") => true,
        (&Method::POST, p) if p.starts_with("/sessions/") && p.ends_with("/ingest") => {
            !state.openai_auth_required
        }
        (&Method::GET, p) if p.starts_with("/sessions/") && p.ends_with("/search") => {
            query_role(query) == "organize"
        }
        (&Method::POST, p) if p.starts_with("/temp/") && p.ends_with("/ingest") => {
            !state.openai_auth_required
        }
        (&Method::GET, p) if p.starts_with("/temp/") && p.ends_with("/search") => {
            query_role(query) == "organize"
        }
        _ => false,
    }
}

async fn nuts_auth_gate(
    State(state): State<Arc<AppState>>,
    mut req: Request,
    next: Next,
) -> Response {
    let is_free = is_free_operation(
        req.uri().path(),
        req.method(),
        req.uri().query(),
        &state,
    );
    let token = extract_bearer(&req);

    match (is_free, &state.nuts_auth, token.as_deref()) {
        (_, None, _) => next.run(req).await,
        (true, Some(_), None) => next.run(req).await,
        (true, Some(auth), Some(tok)) => {
            if let Ok(claims) = auth.verify(tok).await {
                req.extensions_mut().insert(claims);
            }
            next.run(req).await
        }
        (false, Some(_), None) => (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "authentication required".to_string(),
            }),
        )
            .into_response(),
        (false, Some(auth), Some(tok)) => match auth.verify(tok).await {
            Ok(claims) => {
                req.extensions_mut().insert(claims);
                next.run(req).await
            }
            Err(_) => (
                StatusCode::UNAUTHORIZED,
                Json(ErrorResponse {
                    error: "invalid token".to_string(),
                }),
            )
                .into_response(),
        },
    }
}

// ===== Router =====

pub fn router(state: Arc<AppState>) -> Router {
    let r = Router::new()
        .route("/", get(homepage))
        .route("/health", get(health))
        .route("/sessions/:session_id/ingest", post(ingest))
        .route("/sessions/:session_id/search", get(search))
        .route("/sessions/:session_id", get(session_info).delete(delete_session))
        .route("/temp", get(list_temp_stores))
        .route("/temp/:name/ingest", post(temp_ingest))
        .route("/temp/:name/search", get(temp_search))
        .route("/temp/:name/dump", get(temp_dump))
        .route("/temp/:name", delete(delete_temp_store))
        // Phase 2: Crypto endpoints
        .route("/agent/:agent_id/register", post(register_agent))
        .route("/agent/:agent_id/decrypt", post(decrypt_embeddings))
        .route("/agent/:agent_id/encrypt", post(encrypt_embeddings))
        // Phase 3: Inversion endpoint
        .route("/invert", post(invert))
        .layer(middleware::from_fn_with_state(state.clone(), nuts_auth_gate));

    // Register MCP & Agent endpoints without the authentication gate for seamless integration
    r.route("/mcp/sse", get(mcp_sse))
        .route("/mcp/message", post(mcp_message))
        .route("/sessions/:session_id/agent/chat", post(agent_chat))
        .with_state(state)
}

// ===== MCP & Agent Chat Implementation =====

#[derive(Deserialize)]
pub struct AgentChatRequest {
    pub message: String,
}

pub async fn agent_chat(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
    Json(payload): Json<AgentChatRequest>,
) -> impl IntoResponse {
    use axum::response::sse::Sse;
    use tokio_stream::wrappers::UnboundedReceiverStream;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    
    let agent = crate::agent::GhostAgent::new(
        state.store.clone(),
        state.temp_store.clone(),
        state.chunker.clone(),
        state.embedder.clone(),
    );

    tokio::spawn(async move {
        agent.run_loop(session_id, payload.message, tx).await;
    });

    let stream = UnboundedReceiverStream::new(rx);
    Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default())
}

#[derive(Deserialize)]
pub struct McpMessageQuery {
    pub id: String,
}

#[derive(Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    pub method: String,
    pub params: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Serialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

pub async fn mcp_sse(
    State(state): State<Arc<AppState>>,
    req: axum::extract::Request,
) -> impl IntoResponse {
    use axum::response::sse::{Event, Sse};
    use tokio_stream::wrappers::UnboundedReceiverStream;

    let client_id = uuid::Uuid::new_v4().to_string();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    // Dynamically retrieve the request host header
    let host = req.headers()
        .get("host")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("localhost:8085");

    let post_url = format!("http://{}/mcp/message?id={}", host, client_id);

    {
        let mut conns = state.mcp_connections.write().await;
        conns.insert(client_id.clone(), tx.clone());
    }

    // Emit the initial "endpoint" event informing client where to send POSTs
    let endpoint_event = Event::default().event("endpoint").data(post_url);
    let _ = tx.send(Ok(endpoint_event));

    let stream = UnboundedReceiverStream::new(rx);
    Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default())
}

pub async fn mcp_message(
    State(state): State<Arc<AppState>>,
    Query(query): Query<McpMessageQuery>,
    Json(payload): Json<JsonRpcRequest>,
) -> impl IntoResponse {
    let id_val = payload.id.clone().unwrap_or(serde_json::Value::Null);

    let response = match process_mcp_request(&state, payload).await {
        Ok(result) => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: id_val,
            result: Some(result),
            error: None,
        },
        Err(err) => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: id_val,
            result: None,
            error: Some(err),
        },
    };

    let tx = {
        let conns = state.mcp_connections.read().await;
        conns.get(&query.id).cloned()
    };

    if let Some(tx) = tx {
        if let Ok(data_str) = serde_json::to_string(&response) {
            let sse_event = axum::response::sse::Event::default().event("message").data(data_str);
            let _ = tx.send(Ok(sse_event));
        }
        StatusCode::ACCEPTED.into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

async fn process_mcp_request(
    state: &Arc<AppState>,
    req: JsonRpcRequest,
) -> Result<serde_json::Value, JsonRpcError> {
    match req.method.as_str() {
        "initialize" => {
            Ok(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {}
                },
                "serverInfo": {
                    "name": "shivvr-mcp",
                    "version": "0.2.0"
                }
            }))
        }
        "notifications/initialized" => {
            Ok(serde_json::json!({}))
        }
        "tools/list" => {
            Ok(serde_json::json!({
                "tools": [
                    {
                        "name": "search_memory",
                        "description": "Query high-performance hybrid memory search.",
                        "inputSchema": {
                          "type": "object",
                          "properties": {
                            "session_id": { "type": "string", "description": "The memory session ID to search." },
                            "query": { "type": "string", "description": "The semantic search query." },
                            "lexical_only": { "type": "boolean", "description": "Whether to perform keyword-only search (default: false)." }
                          },
                          "required": ["session_id", "query"]
                        }
                    },
                    {
                        "name": "ingest_memory",
                        "description": "Index a new memory string.",
                        "inputSchema": {
                          "type": "object",
                          "properties": {
                            "session_id": { "type": "string", "description": "The session ID to store the memory in." },
                            "text": { "type": "string", "description": "The text content of the memory fragment to index." },
                            "source": { "type": "string", "description": "Optional source filename or metadata label." }
                          },
                          "required": ["session_id", "text"]
                        }
                    },
                    {
                        "name": "list_sessions",
                        "description": "Return active memory sessions.",
                        "inputSchema": {
                          "type": "object",
                          "properties": {}
                        }
                    },
                    {
                        "name": "run_command",
                        "description": "Execute a shell command inside the linux container.",
                        "inputSchema": {
                          "type": "object",
                          "properties": {
                            "command": { "type": "string", "description": "The command string to execute." }
                          },
                          "required": ["command"]
                        }
                    }
                ]
            }))
        }
        "tools/call" => {
            let params = req.params.ok_or_else(|| JsonRpcError {
                code: -32602,
                message: "Missing params for tools/call".to_string(),
                data: None,
            })?;
            let name = params.get("name").and_then(|v| v.as_str()).ok_or_else(|| JsonRpcError {
                code: -32602,
                message: "Missing tool name in tools/call".to_string(),
                data: None,
            })?;
            let arguments = params.get("arguments").cloned().unwrap_or(serde_json::Value::Null);

            let result_str = match name {
                "search_memory" => {
                    let session_id = arguments.get("session_id").and_then(|v| v.as_str()).unwrap_or("default");
                    let query = arguments.get("query").and_then(|v| v.as_str()).unwrap_or("");
                    let lexical_only = arguments.get("lexical_only").and_then(|v| v.as_bool()).unwrap_or(false);

                    let query_embedding = if lexical_only {
                        None
                    } else {
                        state.embedder.embed(query).ok()
                    };

                    match state.store.search_hybrid(
                        session_id,
                        query,
                        query_embedding.as_deref(),
                        5,
                        None,
                        168.0,
                        "organize",
                        &[],
                        &state.search_params,
                    ) {
                        Ok(results) => {
                            let mapped: Vec<serde_json::Value> = results.into_iter().map(|(chunk, score)| {
                                serde_json::json!({
                                    "chunk_id": chunk.id,
                                    "score": score,
                                    "text": chunk.text,
                                    "source": chunk.source,
                                    "metadata": chunk.metadata,
                                    "created_at": chunk.created_at.to_rfc3339()
                                })
                            }).collect();
                            serde_json::to_string_pretty(&mapped).unwrap_or_else(|_| "[]".to_string())
                        }
                        Err(e) => format!("Error searching hybrid store: {}", e),
                    }
                }
                "ingest_memory" => {
                    let session_id = arguments.get("session_id").and_then(|v| v.as_str()).unwrap_or("default");
                    let text = arguments.get("text").and_then(|v| v.as_str()).unwrap_or("");
                    let source = arguments.get("source").and_then(|v| v.as_str()).map(|s| s.to_string());

                    match state.chunker.chunk(text, source, serde_json::Value::Null).await {
                        Ok(chunks) => {
                            if chunks.is_empty() {
                                "Success: Text was too short or no chunks produced.".to_string()
                            } else {
                                let first_id = chunks[0].id.clone();
                                match state.store.add_chunks(session_id, chunks, None) {
                                    Ok(_) => format!("Success: Memory chunk indexed under ID {}", first_id),
                                    Err(e) => format!("Error indexing chunk in store: {}", e),
                                }
                            }
                        }
                        Err(e) => format!("Error chunking text: {}", e),
                    }
                }
                "list_sessions" => {
                    let sessions = state.store.list_sessions(None).unwrap_or_default();
                    let mapped: Vec<serde_json::Value> = sessions.into_iter().map(|s| {
                        serde_json::json!({
                            "id": s.id,
                            "chunks": s.chunk_count,
                            "last_active": s.last_ingested.to_rfc3339()
                        })
                    }).collect();
                    serde_json::to_string_pretty(&mapped).unwrap_or_else(|_| "[]".to_string())
                }
                "run_command" => {
                    let command = arguments.get("command").and_then(|v| v.as_str()).unwrap_or("");
                    if command.is_empty() {
                        "Error: Empty command".to_string()
                    } else {
                        #[cfg(unix)]
                        {
                            match std::process::Command::new("sh")
                                .arg("-c")
                                .arg(command)
                                .output()
                            {
                                Ok(output) => {
                                    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                                    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                                    format!("Exit Code: {}\nSTDOUT:\n{}\nSTDERR:\n{}", output.status.code().unwrap_or(-1), stdout, stderr)
                                }
                                Err(e) => format!("Error executing command: {}", e),
                            }
                        }
                        #[cfg(not(unix))]
                        {
                            "Error: run_command is only supported in Unix/Linux container environments".to_string()
                        }
                    }
                }
                _ => return Err(JsonRpcError {
                    code: -32601,
                    message: format!("Method not found: {}", name),
                    data: None,
                }),
            };

            Ok(serde_json::json!({
                "content": [
                    {
                        "type": "text",
                        "text": result_str
                    }
                ]
            }))
        }
        "resources/list" => {
            let sessions = state.store.list_sessions(None).unwrap_or_default();
            let mut resources = vec![
                serde_json::json!({
                    "uri": "shivvr://sessions",
                    "name": "Memory Sessions Overview",
                    "description": "List of all active memory sessions in shivvr",
                    "mimeType": "application/json"
                })
            ];

            for s in sessions {
                resources.push(serde_json::json!({
                    "uri": format!("shivvr://sessions/{}", s.id),
                    "name": format!("Session '{}' Memory Dump", s.id),
                    "description": format!("Memory chunks indexed under session '{}'", s.id),
                    "mimeType": "application/json"
                }));
            }

            Ok(serde_json::json!({ "resources": resources }))
        }
        "resources/read" => {
            let params = req.params.ok_or_else(|| JsonRpcError {
                code: -32602,
                message: "Missing params for resources/read".to_string(),
                data: None,
            })?;
            let uri = params.get("uri").and_then(|v| v.as_str()).ok_or_else(|| JsonRpcError {
                code: -32602,
                message: "Missing uri in resources/read".to_string(),
                data: None,
            })?;

            if uri == "shivvr://sessions" {
                let sessions = state.store.list_sessions(None).unwrap_or_default();
                let mapped: Vec<serde_json::Value> = sessions.into_iter().map(|s| {
                    serde_json::json!({
                        "id": s.id,
                        "chunks": s.chunk_count,
                        "last_active": s.last_ingested.to_rfc3339()
                    })
                }).collect();
                let dump = serde_json::to_string_pretty(&mapped).unwrap_or_default();
                Ok(serde_json::json!({
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": dump
                        }
                    ]
                }))
            } else if uri.starts_with("shivvr://sessions/") {
                let session_id = &uri["shivvr://sessions/".len()..];
                let chunks = state.store.get_chunks(session_id).unwrap_or_default();
                let mapped: Vec<serde_json::Value> = chunks.into_iter().map(|chunk| {
                    serde_json::json!({
                        "chunk_id": chunk.id,
                        "text": chunk.text,
                        "source": chunk.source,
                        "metadata": chunk.metadata,
                        "created_at": chunk.created_at.to_rfc3339()
                    })
                }).collect();
                let dump = serde_json::to_string_pretty(&mapped).unwrap_or_default();
                Ok(serde_json::json!({
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": dump
                        }
                    ]
                }))
            } else {
                Err(JsonRpcError {
                    code: -32602,
                    message: format!("Unsupported resource URI: {}", uri),
                    data: None,
                })
            }
        }
        _ => Err(JsonRpcError {
            code: -32601,
            message: format!("Method not found: {}", req.method),
            data: None,
        }),
    }
}
