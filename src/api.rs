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
    pub audio_client: Arc<crate::audio::AudioClient>,
    pub vision_embedder: Option<Arc<crate::vision::VisionEmbedder>>,
    /// SigLIP text tower (same space as `vision_embedder`); `None` when the
    /// ONNX or tokenizer file is missing.
    pub siglip_text: Option<Arc<crate::vision::SiglipTextEmbedder>>,
    /// EmbeddingGemma 2 text path; `None` when the ONNX or tokenizer file is
    /// missing.
    pub embeddinggemma2: Option<Arc<crate::embeddinggemma::EmbeddingGemma2Embedder>>,
    /// Text models for sessions, temp stores, search, MCP and the agent
    /// (EmbeddingGemma 2 by default, GTR-T5-base selectable).
    pub text_models: Arc<crate::text_model::TextEmbedders>,
    pub start_time: std::time::Instant,
    pub nuts_auth: Option<Arc<NutsAuth>>,
    pub openai_auth_required: bool,
    pub guardrail_tagger: Arc<std::sync::RwLock<lume::Tagger>>,
    /// BM25 tuning params (k1/b/delta/title_weight/body_weight). Lume's
    /// successor to the old SearchConfig — variant + tagger get selected
    /// per call rather than living on the global config.
    pub search_params: lume::bm25::Bm25Params,
    pub mcp_connections: Arc<tokio::sync::RwLock<HashMap<String, tokio::sync::mpsc::UnboundedSender<Result<axum::response::sse::Event, std::convert::Infallible>>>>>,
}

// ===== Request/Response Types =====

#[derive(Deserialize)]
pub struct IngestRequest {
    #[serde(default)]
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
    /// If true, mean-pool all chunk embeddings into a single normalized vector.
    /// Returns one IngestChunk regardless of how many internal chunks were produced.
    /// Use this for sliding-window / bhavanga context embeddings where the caller
    /// wants one vector representing the whole input window.
    #[serde(default)]
    pub mean_pool: bool,
    /// Base64 encoded audio to transcribe via Docker service and embed
    pub audio_base64: Option<String>,
    /// Base64 encoded image to embed via SigLIP vision model
    pub image_base64: Option<String>,
    /// Text model for this session or temp store: `embeddinggemma-2` (the
    /// default) or `gtr-t5-base`. Fixed on the first ingest; omit it on later
    /// ingests to use the store's model.
    #[serde(default)]
    pub model: Option<String>,
    /// Extra vectors to return per chunk, by model name. Today only
    /// `"gtr-t5-base"` (the invertible vector kept alongside the search
    /// vector). Returned under each chunk's `embeddings` map.
    #[serde(default)]
    pub return_vectors: Vec<String>,
}

#[derive(Serialize)]
pub struct IngestResponse {
    pub chunks_created: usize,
    pub tokens_processed: usize,
    pub time_ms: u64,
    /// Text model that produced the chunk embeddings.
    pub model: String,
    pub chunks: Vec<IngestChunk>,
}

#[derive(Serialize)]
pub struct IngestChunk {
    pub chunk_id: String,
    pub text: String,
    pub embedding: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_retrieve: Option<Vec<f32>>,
    /// Vectors requested with `return_vectors`, keyed by model name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embeddings: Option<std::collections::HashMap<String, Vec<f32>>>,
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
    /// Number of results to return. Accepts `limit` as an alias —
    /// qdrant/weaviate/milvus/chroma all use `limit`, so clients coming
    /// from those reach for it first. `n` remains canonical.
    #[serde(default = "default_n", alias = "limit")]
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
    /// Embedding role: "organize" (768d, the store's text model) or "retrieve" (1536d ada-002)
    #[serde(default = "default_role")]
    pub role: String,
    /// Optional check: the text model the caller expects the store to hold.
    /// The query is always embedded with the store's own model; a different
    /// value here is a 400.
    #[serde(default)]
    pub model: Option<String>,
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
    /// Text model used when a request names none (`/embed`, new sessions and
    /// temp stores): `embeddinggemma-2` when loaded, else `gtr-t5-base`.
    pub default_model: String,
    pub models: Vec<ModelInfo>,
    pub sessions: usize,
    pub total_chunks: usize,
    pub uptime_seconds: u64,
    pub encryption_available: bool,
    pub inversion_available: bool,
    pub audio_available: bool,
    pub vision_available: bool,
    pub gpu: bool,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Deserialize)]
pub struct AudioRequest {
    pub audio_base64: String,
    /// `/audio/embed` only: text model for the transcript
    /// (`embeddinggemma-2` by default, or `gtr-t5-base`).
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Serialize)]
pub struct AudioTranscribeResponse {
    pub transcript: String,
}

#[derive(Serialize)]
pub struct AudioEmbedResponse {
    pub transcript: String,
    pub embedding: Vec<f32>,
    pub dimension: usize,
    pub model: String,
}

#[derive(Deserialize)]
pub struct ImageEmbedRequest {
    pub image_base64: String,
}

#[derive(Serialize)]
pub struct ImageEmbedResponse {
    pub embedding: Vec<f32>,
    pub dimension: usize,
    pub model: String,
}


// ===== Text model helpers =====

type ApiError = (StatusCode, Json<ErrorResponse>);

fn api_error(status: StatusCode, error: impl Into<String>) -> ApiError {
    (status, Json(ErrorResponse { error: error.into() }))
}

fn resolve_error(e: crate::text_model::ResolveError) -> ApiError {
    let status = match e {
        crate::text_model::ResolveError::Unknown(_) => StatusCode::BAD_REQUEST,
        crate::text_model::ResolveError::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
    };
    api_error(status, e.to_string())
}

/// Text model for an ingest into a store that may already have one: the
/// requested model, else the store's, else the server default. Naming a
/// different model than the store holds is a 400.
fn ingest_model(state: &AppState, requested: Option<&str>, existing: Option<String>) -> Result<&'static str, ApiError> {
    let requested = requested.map(str::trim).filter(|m| !m.is_empty());
    let model = state
        .text_models
        .resolve(requested.or(existing.as_deref()))
        .map_err(resolve_error)?;
    if let Some(existing) = existing {
        if existing != model {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                crate::store::ModelMismatch { store_model: existing, requested: model.to_string() }.to_string(),
            ));
        }
    }
    Ok(model)
}

/// Text model for a search over a store: the store's model (or the server
/// default when the store doesn't exist). `expected` is the optional `model`
/// query parameter; a mismatch is a 400.
fn search_model(state: &AppState, expected: Option<&str>, existing: Option<String>) -> Result<&'static str, ApiError> {
    let expected = expected.map(str::trim).filter(|m| !m.is_empty());
    let model = state
        .text_models
        .resolve(existing.as_deref().or(expected))
        .map_err(resolve_error)?;
    if let Some(expected) = expected {
        if expected != model {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                format!("this store holds {model} vectors, not {expected}"),
            ));
        }
    }
    Ok(model)
}

/// Re-embed chunk texts with `model` (`document` prefix). The chunker already
/// embedded them with GTR-T5-base, so GTR needs no second pass; for any other
/// model the GTR vector moves to `embedding_gtr` so the chunk stays invertible.
async fn embed_chunks(state: &AppState, model: &'static str, chunks: &mut [crate::store::Chunk]) -> Result<(), ApiError> {
    if model == crate::embed_api::EMBED_MODEL || chunks.is_empty() {
        return Ok(());
    }
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let models = state.text_models.clone();
    let vectors = tokio::task::spawn_blocking(move || {
        texts
            .iter()
            .map(|t| models.embed(model, crate::text_model::Purpose::Document, t))
            .collect::<anyhow::Result<Vec<Vec<f32>>>>()
    })
    .await
    .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("embedding task failed: {e}")))?
    .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("Embedding failed: {e}")))?;
    for (chunk, vector) in chunks.iter_mut().zip(vectors) {
        chunk.embedding_gtr = Some(std::mem::replace(&mut chunk.embedding, vector));
    }
    Ok(())
}

/// Parse an ingest's `return_vectors`. Only `gtr-t5-base` can be requested
/// today (the store's own vector is always returned as `embedding`).
fn parse_return_vectors(names: &[String]) -> Result<bool, ApiError> {
    let mut gtr = false;
    for name in names {
        match name.trim() {
            crate::embed_api::EMBED_MODEL => gtr = true,
            other => {
                return Err(api_error(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "unsupported return_vectors entry {other:?}; available: [{:?}]",
                        crate::embed_api::EMBED_MODEL
                    ),
                ))
            }
        }
    }
    Ok(gtr)
}

/// The `embeddings` map for an ingest response chunk: the GTR vector when
/// requested (the stored one, or `embedding` itself for GTR stores).
fn extra_vectors(
    want_gtr: bool,
    store_model: &str,
    embedding: &[f32],
    embedding_gtr: Option<&Vec<f32>>,
) -> Option<std::collections::HashMap<String, Vec<f32>>> {
    if !want_gtr {
        return None;
    }
    let gtr = match embedding_gtr {
        Some(v) => v.clone(),
        None if store_model == crate::embed_api::EMBED_MODEL => embedding.to_vec(),
        None => return None,
    };
    Some(std::collections::HashMap::from([(crate::embed_api::EMBED_MODEL.to_string(), gtr)]))
}

/// Embed a search query with `model` (`query` prefix).
async fn embed_query(state: &AppState, model: &'static str, q: &str) -> Result<Vec<f32>, ApiError> {
    let models = state.text_models.clone();
    let q = q.to_string();
    tokio::task::spawn_blocking(move || models.embed(model, crate::text_model::Purpose::Query, &q))
        .await
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("embedding task failed: {e}")))?
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
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
    let model = ingest_model(&state, req.model.as_deref(), state.store.session_model(&session_id))?;
    let want_gtr = parse_return_vectors(&req.return_vectors)?;

    let emotion_primary = req.emotion_primary.clone();
    let emotion_secondary = req.emotion_secondary.clone();
    let agent_id = req.agent_id.clone();

    // 1. Process audio if audio_base64 provided
    let mut text = req.text.clone();
    let mut source = req.source.clone();

    if let Some(ref audio_b64) = req.audio_base64 {
        use base64::Engine;
        let audio_bytes = base64::engine::general_purpose::STANDARD
            .decode(audio_b64.trim())
            .map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: format!("Invalid audio_base64: {}", e),
                    }),
                )
            })?;
        let transcript = state.audio_client.transcribe(audio_bytes, None).await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Audio transcription failed: {}", e),
                }),
            )
        })?;
        if text.is_empty() {
            text = transcript;
        } else {
            text = format!("{}\n{}", text, transcript);
        }
        if source.is_none() {
            source = Some("audio".to_string());
        }
    }

    let mut chunks = if !text.trim().is_empty() {
        state
            .chunker
            .chunk(&text, source.clone(), req.metadata.clone())
            .await
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: e.to_string(),
                    }),
                )
            })?
    } else {
        Vec::new()
    };
    embed_chunks(&state, model, &mut chunks).await?;

    // 2. Process image if image_base64 provided
    if let Some(ref img_b64) = req.image_base64 {
        let vision = state.vision_embedder.as_ref().ok_or_else(|| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    error: "Vision embedder not loaded (models/siglip-vision.onnx not available)".to_string(),
                }),
            )
        })?;
        use base64::Engine;
        let img_bytes = base64::engine::general_purpose::STANDARD
            .decode(img_b64.trim())
            .map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        error: format!("Invalid image_base64: {}", e),
                    }),
                )
            })?;
        let img_emb = vision.embed_image(&img_bytes).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Vision embedding failed: {}", e),
                }),
            )
        })?;
        let img_chunk_id = format!("img_{}", uuid::Uuid::new_v4());
        let img_text = if text.is_empty() { "[image]".to_string() } else { format!("[image] {}", text) };
        let img_chunk = crate::store::Chunk {
            id: img_chunk_id,
            text: img_text,
            embedding: img_emb,
            embedding_retrieve: None,
            embedding_gtr: None,
            token_count: 1,
            source: Some("image".to_string()),
            metadata: req.metadata.clone(),
            created_at: chrono::Utc::now(),
            emotion_primary: emotion_primary.clone(),
            emotion_secondary: emotion_secondary.clone(),
            encrypted: false,
            agent_id: agent_id.clone(),
        };
        chunks.push(img_chunk);
    }
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
                if let Some(ref emb) = chunk.embedding_gtr {
                    chunk.embedding_gtr = Some(keys.encrypt(emb, "organize"));
                }
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
            // Image chunks hold a SigLIP vector and have no GTR vector.
            embeddings: if c.id.starts_with("img_") {
                None
            } else {
                extra_vectors(want_gtr, model, &c.embedding, c.embedding_gtr.as_ref())
            },
            token_count: c.token_count,
            source: c.source.clone(),
            emotion_primary: c.emotion_primary.clone(),
            emotion_secondary: c.emotion_secondary.clone(),
        })
        .collect();

    state
        .store
        .claim_model(&session_id, model, caller_id)
        .map_err(|e| api_error(StatusCode::BAD_REQUEST, e.to_string()))?;
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
        model: model.to_string(),
        chunks: response_chunks,
    }))
}

pub async fn temp_ingest(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<IngestRequest>,
) -> Result<Json<IngestResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();
    let model = ingest_model(&state, req.model.as_deref(), state.temp_store.store_model(&name))?;
    let want_gtr = parse_return_vectors(&req.return_vectors)?;

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
    embed_chunks(&state, model, &mut chunks).await?;

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
                if let Some(ref emb) = chunk.embedding_gtr {
                    chunk.embedding_gtr = Some(keys.encrypt(emb, "organize"));
                }
                if let Some(ref emb) = chunk.embedding_retrieve {
                    chunk.embedding_retrieve = Some(keys.encrypt(emb, "retrieve"));
                }
                chunk.encrypted = true;
            }
        }
    }

    let chunks_created = chunks.len();
    let tokens_processed: usize = chunks.iter().map(|c| c.token_count).sum();

    // Mean-pool: collapse all chunk embeddings into one normalized vector.
    let response_chunks: Vec<IngestChunk> = if req.mean_pool && !chunks.is_empty() {
        let dim = chunks[0].embedding.len();
        let mut pooled = vec![0.0f32; dim];
        for chunk in &chunks {
            for (p, v) in pooled.iter_mut().zip(&chunk.embedding) {
                *p += v;
            }
        }
        let n = chunks.len() as f32;
        for p in &mut pooled { *p /= n; }
        let norm = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 { for p in &mut pooled { *p /= norm; } }

        let pooled_retrieve = if chunks.iter().all(|c| c.embedding_retrieve.is_some()) {
            let rdim = chunks[0].embedding_retrieve.as_ref().unwrap().len();
            let mut rp = vec![0.0f32; rdim];
            for chunk in &chunks {
                for (p, v) in rp.iter_mut().zip(chunk.embedding_retrieve.as_ref().unwrap()) {
                    *p += v;
                }
            }
            for p in &mut rp { *p /= n; }
            let rnorm = rp.iter().map(|v| v * v).sum::<f32>().sqrt();
            if rnorm > 0.0 { for p in &mut rp { *p /= rnorm; } }
            Some(rp)
        } else {
            None
        };

        // Pooled GTR vector, when requested: mean of each chunk's GTR vector
        // (the stored one, or `embedding` itself in a GTR store).
        let pooled_gtr = if want_gtr {
            let vectors: Vec<&[f32]> = chunks
                .iter()
                .filter_map(|c| match &c.embedding_gtr {
                    Some(v) => Some(v.as_slice()),
                    None if model == crate::embed_api::EMBED_MODEL => Some(c.embedding.as_slice()),
                    None => None,
                })
                .collect();
            if vectors.len() == chunks.len() {
                let mut gp = vec![0.0f32; vectors[0].len()];
                for v in &vectors {
                    for (p, x) in gp.iter_mut().zip(v.iter()) {
                        *p += x;
                    }
                }
                let gnorm = gp.iter().map(|v| v * v).sum::<f32>().sqrt();
                if gnorm > 0.0 { for p in &mut gp { *p /= gnorm; } }
                Some(std::collections::HashMap::from([(crate::embed_api::EMBED_MODEL.to_string(), gp)]))
            } else {
                None
            }
        } else {
            None
        };

        vec![IngestChunk {
            chunk_id: chunks[0].id.clone(),
            text: req.text.clone(),
            embedding: pooled,
            embedding_retrieve: pooled_retrieve,
            embeddings: pooled_gtr,
            token_count: tokens_processed,
            source: chunks[0].source.clone(),
            emotion_primary: emotion_primary.clone(),
            emotion_secondary: emotion_secondary.clone(),
        }]
    } else {
        chunks
            .iter()
            .map(|c| IngestChunk {
                chunk_id: c.id.clone(),
                text: c.text.clone(),
                embedding: c.embedding.clone(),
                embedding_retrieve: c.embedding_retrieve.clone(),
                embeddings: extra_vectors(want_gtr, model, &c.embedding, c.embedding_gtr.as_ref()),
                token_count: c.token_count,
                source: c.source.clone(),
                emotion_primary: c.emotion_primary.clone(),
                emotion_secondary: c.emotion_secondary.clone(),
            })
            .collect()
    };

    state
        .temp_store
        .claim_model(&name, model)
        .map_err(|e| api_error(StatusCode::BAD_REQUEST, e.to_string()))?;
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
        model: model.to_string(),
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
    let model = search_model(&state, query.model.as_deref(), state.store.session_model(&session_id))?;

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
            embed_query(&state, model, &query.q).await?
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
    let model = search_model(&state, query.model.as_deref(), state.temp_store.store_model(&name))?;

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
            embed_query(&state, model, &query.q).await?
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

    if state.vision_embedder.is_some() {
        models.push(ModelInfo {
            name: "siglip-base-patch16-224".to_string(),
            role: "multimodal".to_string(),
            dimension: 768,
            status: "active".to_string(),
        });
    }

    if state.siglip_text.is_some() {
        models.push(ModelInfo {
            name: "siglip-base-patch16-224".to_string(),
            role: "multimodal-text".to_string(),
            dimension: 768,
            status: "active".to_string(),
        });
    }

    if state.embeddinggemma2.is_some() {
        models.push(ModelInfo {
            name: crate::embed_api::EMBEDDINGGEMMA2_MODEL.to_string(),
            role: "text".to_string(),
            dimension: crate::embed_api::EMBEDDINGGEMMA2_DIM,
            status: "active".to_string(),
        });
    }

    let audio_available = state.audio_client.health().await;
    let vision_available = state.vision_embedder.is_some();

    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        default_model: state.text_models.default_model().to_string(),
        models,
        sessions: sessions.len(),
        total_chunks,
        uptime_seconds: state.start_time.elapsed().as_secs(),
        encryption_available: true,
        inversion_available: state.inverter.is_some(),
        audio_available,
        vision_available,
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
    /// Model that produced `embedding`. Only `gtr-t5-base` can be inverted;
    /// naming another model is a 400. Omitted = assumed GTR.
    #[serde(default)]
    pub model: Option<String>,
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
    if let Some(model) = req.model.as_deref().map(str::trim).filter(|m| !m.is_empty()) {
        if model != crate::embed_api::EMBED_MODEL {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "/invert only reconstructs {} vectors, not {model}; ingest with \"model\": \"{}\" to get invertible vectors",
                    crate::embed_api::EMBED_MODEL,
                    crate::embed_api::EMBED_MODEL
                ),
            ));
        }
    }
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

// One page for the backend, landing-only mode and the gateway (src/landing.html);
// the version comes from Cargo.toml and a script fills the live stats from /health.
pub async fn homepage() -> Html<String> {
    landing_page()
}

// ===== Landing-only mode =====
//
// A lightweight "front door" deployment: serves only the static homepage
// and a stub /health, with no model loading, no embedder, no AppState —
// safe to run on a GPU-less, minimal-resource Cloud Run instance. Used to
// keep shivvr.nuts.services reachable while the real GPU-backed backend
// (separate service, started on demand) is scaled down between uses.
//
// TODO: once login-gated activation is wired up, this is where a request
// that needs real embedding/search would proxy through to the backend
// service rather than 404ing.

pub fn landing_router() -> Router {
    Router::new()
        .route("/", get(landing_homepage))
        .route("/health", get(landing_health))
}

async fn landing_homepage() -> Html<String> {
    landing_page()
}

/// src/landing.html with the Cargo.toml version filled in.
pub fn landing_page() -> Html<String> {
    Html(include_str!("landing.html").replace("{{VERSION}}", env!("CARGO_PKG_VERSION")))
}

async fn landing_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "mode": "landing_only",
        "backend": "not running -- static landing page only"
    }))
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
        (&Method::POST, "/audio/transcribe") => true,
        (&Method::POST, "/audio/embed") => true,
        (&Method::POST, "/image/embed") => true,
        // Local model compute only (GTR-T5, SigLIP text, EmbeddingGemma 2), like /image/embed and organize-role ingest.
        (&Method::POST, "/embed") => true,
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

// ===== Audio & Image Handlers =====

pub async fn audio_transcribe_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AudioRequest>,
) -> Result<Json<AudioTranscribeResponse>, (StatusCode, Json<ErrorResponse>)> {
    use base64::Engine;
    let audio_bytes = base64::engine::general_purpose::STANDARD
        .decode(req.audio_base64.trim())
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Invalid audio_base64: {}", e),
                }),
            )
        })?;

    let transcript = state.audio_client.transcribe(audio_bytes, None).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Audio transcription failed: {}", e),
            }),
        )
    })?;

    Ok(Json(AudioTranscribeResponse { transcript }))
}

pub async fn audio_embed_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AudioRequest>,
) -> Result<Json<AudioEmbedResponse>, (StatusCode, Json<ErrorResponse>)> {
    use base64::Engine;
    let audio_bytes = base64::engine::general_purpose::STANDARD
        .decode(req.audio_base64.trim())
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Invalid audio_base64: {}", e),
                }),
            )
        })?;

    let transcript = state.audio_client.transcribe(audio_bytes, None).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Audio transcription failed: {}", e),
            }),
        )
    })?;

    let model = state.text_models.resolve(req.model.as_deref()).map_err(resolve_error)?;
    let models = state.text_models.clone();
    let text = transcript.clone();
    let embedding = tokio::task::spawn_blocking(move || {
        models.embed(model, crate::text_model::Purpose::Document, &text)
    })
    .await
    .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("embedding task failed: {e}")))?
    .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("Embedding failed: {e}")))?;

    let dimension = embedding.len();
    Ok(Json(AudioEmbedResponse {
        transcript,
        embedding,
        dimension,
        model: model.to_string(),
    }))
}

pub async fn image_embed_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ImageEmbedRequest>,
) -> Result<Json<ImageEmbedResponse>, (StatusCode, Json<ErrorResponse>)> {
    let vision = state.vision_embedder.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Vision embedder not loaded (models/siglip-vision.onnx not available)".to_string(),
            }),
        )
    })?;

    use base64::Engine;
    let img_bytes = base64::engine::general_purpose::STANDARD
        .decode(req.image_base64.trim())
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Invalid image_base64: {}", e),
                }),
            )
        })?;

    let embedding = vision.embed_image(&img_bytes).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Vision embedding failed: {}", e),
            }),
        )
    })?;

    let dimension = embedding.len();
    Ok(Json(ImageEmbedResponse {
        embedding,
        dimension,
        model: "siglip-base-patch16-224".to_string(),
    }))
}

// ===== Plain text embedding =====

/// `POST /embed`: vectors for a batch of texts. No session, store, or cache
/// side effects. `model` picks the space:
/// - `embeddinggemma-2` (default): EmbeddingGemma 2's text path. `task`
///   prepends the matching prompt prefix and `dimensions` truncates
///   (Matryoshka) and re-normalizes. 503 when named but not loaded; with no
///   `model`, `task` or `dimensions` and EmbeddingGemma 2 not loaded, the
///   request falls back to `gtr-t5-base`.
/// - `gtr-t5-base`: texts are embedded one at a time inside a single
///   blocking task: the embedder mean-pools without a padding mask, so padded
///   batch inference would change the vectors. The only space `/invert` reads.
/// - `siglip-base-patch16-224`: the SigLIP text tower, same space as
///   `POST /image/embed`, for text-to-image comparison. 503 when the tower or
///   its tokenizer is not loaded.
pub async fn embed_texts(
    State(state): State<Arc<AppState>>,
    Json(mut req): Json<crate::embed_api::EmbedRequest>,
) -> Result<Json<crate::embed_api::EmbedResponse>, (StatusCode, Json<ErrorResponse>)> {
    use crate::embed_api::{truncate_normalize, validate, EmbedModel, EmbedResponse};
    let unspecified = req.model.as_deref().map(str::trim).unwrap_or("").is_empty();
    if unspecified && state.embeddinggemma2.is_none() && req.task.is_none() && req.dimensions.is_none() {
        req.model = Some(crate::embed_api::EMBED_MODEL.to_string());
    }
    let plan = validate(&req).map_err(|error| (StatusCode::BAD_REQUEST, Json(ErrorResponse { error })))?;
    let model = plan.model;
    let texts = req.texts;
    let expected_dim = model.dim();

    let joined = match model {
        EmbedModel::GtrT5Base => {
            let embedder = state.embedder.clone();
            tokio::task::spawn_blocking(move || {
                texts
                    .iter()
                    .map(|t| embedder.embed(t))
                    .collect::<anyhow::Result<Vec<Vec<f32>>>>()
            })
            .await
        }
        EmbedModel::SiglipText => {
            let siglip = state.siglip_text.clone().ok_or_else(|| {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(ErrorResponse {
                        error: "SigLIP text tower not loaded (models/siglip-text.onnx or models/siglip-tokenizer.json not available)".to_string(),
                    }),
                )
            })?;
            tokio::task::spawn_blocking(move || {
                texts
                    .iter()
                    .map(|t| siglip.embed_text(t))
                    .collect::<anyhow::Result<Vec<Vec<f32>>>>()
            })
            .await
        }
        EmbedModel::EmbeddingGemma2 => {
            let gemma = state.embeddinggemma2.clone().ok_or_else(|| {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(ErrorResponse {
                        error: "EmbeddingGemma 2 not loaded (models/embeddinggemma2-text.onnx or models/embeddinggemma2-tokenizer.json not available)".to_string(),
                    }),
                )
            })?;
            let prefix = plan.prefix;
            tokio::task::spawn_blocking(move || {
                texts
                    .iter()
                    .map(|t| gemma.embed_text(prefix, t))
                    .collect::<anyhow::Result<Vec<Vec<f32>>>>()
            })
            .await
        }
    };

    let vectors = joined
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: format!("embedding task failed: {e}") }),
            )
        })?
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: format!("Embedding failed: {e}") }),
            )
        })?;

    if let Some(bad) = vectors.iter().find(|v| v.len() != expected_dim) {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("model returned {} dims, expected {expected_dim}", bad.len()),
            }),
        ));
    }

    let vectors: Vec<Vec<f32>> = if plan.dim < expected_dim {
        vectors.into_iter().map(|v| truncate_normalize(v, plan.dim)).collect()
    } else {
        vectors
    };

    Ok(Json(EmbedResponse {
        model: model.name().to_string(),
        dim: plan.dim,
        vectors,
    }))
}

// ===== Request timeouts =====

/// Bound how long a handler may take to produce a response, so a hung
/// inference cannot hold a connection forever. Streaming bodies (SSE) are not
/// covered: the budget ends once the response starts. A blocking task that
/// already started keeps running to completion; only the connection is
/// released.
async fn enforce_timeout(
    State(budget): State<std::time::Duration>,
    req: Request,
    next: Next,
) -> Response {
    match tokio::time::timeout(budget, next.run(req)).await {
        Ok(response) => response,
        Err(_) => (
            StatusCode::REQUEST_TIMEOUT,
            Json(ErrorResponse {
                error: format!(
                    "request exceeded the {}s server timeout",
                    budget.as_secs()
                ),
            }),
        )
            .into_response(),
    }
}

// ===== Router =====

pub fn router(state: Arc<AppState>) -> Router {
    use crate::limits::{request_timeouts, EMBED_BODY_LIMIT_BYTES, MEDIA_BODY_LIMIT_BYTES};
    use axum::extract::DefaultBodyLimit;

    let (heavy_timeout, light_timeout) = request_timeouts();

    // Routes carrying base64 media need more than axum's 2 MiB default, or a
    // single rendered page / a minute of audio is rejected with 413.
    let media_routes = Router::new()
        .route("/sessions/:session_id/ingest", post(ingest))
        .route("/temp/:name/ingest", post(temp_ingest))
        .route("/audio/transcribe", post(audio_transcribe_handler))
        .route("/audio/embed", post(audio_embed_handler))
        .route("/image/embed", post(image_embed_handler))
        .route_layer(DefaultBodyLimit::max(MEDIA_BODY_LIMIT_BYTES))
        .route_layer(middleware::from_fn_with_state(heavy_timeout, enforce_timeout));

    let embed_routes = Router::new()
        .route("/embed", post(embed_texts))
        .route_layer(DefaultBodyLimit::max(EMBED_BODY_LIMIT_BYTES))
        .route_layer(middleware::from_fn_with_state(heavy_timeout, enforce_timeout));

    // Inversion is a decode loop; it shares the heavy budget.
    let invert_routes = Router::new()
        .route("/invert", post(invert))
        .route_layer(middleware::from_fn_with_state(heavy_timeout, enforce_timeout));

    let r = Router::new()
        .route("/", get(homepage))
        .route("/health", get(health))
        .route("/sessions/:session_id/search", get(search))
        .route("/sessions/:session_id", get(session_info).delete(delete_session))
        .route("/temp", get(list_temp_stores))
        .route("/temp/:name/search", get(temp_search))
        .route("/temp/:name/dump", get(temp_dump))
        .route("/temp/:name", delete(delete_temp_store))
        // Phase 2: Crypto endpoints
        .route("/agent/:agent_id/register", post(register_agent))
        .route("/agent/:agent_id/decrypt", post(decrypt_embeddings))
        .route("/agent/:agent_id/encrypt", post(encrypt_embeddings))
        // route_layer only covers routes added above; the merged groups keep
        // their own (heavy) budget. MCP/SSE and agent chat stay unbounded.
        .route_layer(middleware::from_fn_with_state(light_timeout, enforce_timeout))
        .merge(media_routes)
        .merge(embed_routes)
        .merge(invert_routes)
        .layer(middleware::from_fn_with_state(state.clone(), nuts_auth_gate));

    // Register MCP & Agent endpoints without the authentication gate for seamless integration
    r.route("/mcp/sse", get(mcp_sse))
        .route("/mcp/message", post(mcp_message))
        .route("/sessions/:session_id/agent/chat", post(agent_chat))
        .with_state(state)
}

// ===== MCP & Agent Chat Implementation =====

/// `run_command` executes arbitrary shell inside the container, so it is off
/// unless `SHIVVR_ENABLE_RUN_COMMAND=true` (or `1`) is set explicitly. When
/// off it is omitted from `tools/list` and refused by `tools/call`.
pub fn run_command_enabled() -> bool {
    std::env::var("SHIVVR_ENABLE_RUN_COMMAND")
        .map(|v| {
            let v = v.trim();
            v.eq_ignore_ascii_case("true") || v == "1"
        })
        .unwrap_or(false)
}

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
        state.text_models.clone(),
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
                    // Single source of truth: Cargo.toml [package].version.
                    // Health endpoint (line ~1048) and homepage (line ~1240)
                    // already read it the same way. To rev, bump Cargo.toml
                    // and re-deploy — nothing else needs editing.
                    "version": env!("CARGO_PKG_VERSION")
                }
            }))
        }
        "notifications/initialized" => {
            Ok(serde_json::json!({}))
        }
        "tools/list" => {
            let mut tools = serde_json::json!({
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
                    }
                ]
            });
            if run_command_enabled() {
                if let Some(list) = tools.get_mut("tools").and_then(|t| t.as_array_mut()) {
                    list.push(serde_json::json!({
                        "name": "run_command",
                        "description": "Execute a shell command inside the linux container.",
                        "inputSchema": {
                          "type": "object",
                          "properties": {
                            "command": { "type": "string", "description": "The command string to execute." }
                          },
                          "required": ["command"]
                        }
                    }));
                }
            }
            Ok(tools)
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
                        let model = state
                            .store
                            .session_model(session_id)
                            .and_then(|m| state.text_models.resolve(Some(&m)).ok())
                            .unwrap_or_else(|| state.text_models.default_model());
                        embed_query(&state, model, query).await.ok()
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

                    match ingest_model(&state, None, state.store.session_model(session_id)) {
                        Err((_, Json(e))) => format!("Error: {}", e.error),
                        Ok(model) => match state.chunker.chunk(text, source, serde_json::Value::Null).await {
                            Ok(mut chunks) => {
                                if chunks.is_empty() {
                                    "Success: Text was too short or no chunks produced.".to_string()
                                } else if let Err((_, Json(e))) = embed_chunks(&state, model, &mut chunks).await {
                                    format!("Error embedding chunks: {}", e.error)
                                } else if let Err(e) = state.store.claim_model(session_id, model, None) {
                                    format!("Error: {}", e)
                                } else {
                                    let first_id = chunks[0].id.clone();
                                    match state.store.add_chunks(session_id, chunks, None) {
                                        Ok(_) => format!("Success: Memory chunk indexed under ID {}", first_id),
                                        Err(e) => format!("Error indexing chunk in store: {}", e),
                                    }
                                }
                            }
                            Err(e) => format!("Error chunking text: {}", e),
                        },
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
                "run_command" if !run_command_enabled() => {
                    "Error: run_command is disabled on this server. Set SHIVVR_ENABLE_RUN_COMMAND=true to enable it.".to_string()
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
