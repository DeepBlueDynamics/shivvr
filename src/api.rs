use crate::chunker::Chunker;
use crate::crypto::CryptoManager;
use crate::embedder::Embedder;
use crate::inverter::Inverter;
use crate::openai::OpenAIEmbedder;
use crate::store::Store;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{Html, Json},
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

// ===== Homepage =====

pub async fn homepage(State(state): State<Arc<AppState>>) -> Html<String> {
    let sessions = state.store.list_sessions().unwrap_or_default().len();
    let chunks = state.store.total_chunks().unwrap_or(0);
    let uptime = state.start_time.elapsed().as_secs();
    let gpu = if cfg!(feature = "cuda") { "CUDA" } else { "CPU" };
    let inversion = if state.inverter.is_some() { "enabled" } else { "disabled" };

    Html(format!(r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>shivvr</title>
<style>
  :root {{ --bg: #0a0a0f; --fg: #c8c8d0; --accent: #7b68ee; --dim: #555568;
           --card: #12121a; --border: #1e1e2e; --green: #50fa7b; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--fg); font-family: 'SF Mono', 'Fira Code', monospace;
          line-height: 1.7; padding: 2rem; max-width: 860px; margin: 0 auto; }}
  h1 {{ color: var(--accent); font-size: 2.4rem; letter-spacing: -0.02em; margin-bottom: 0.25rem; }}
  h1 span {{ color: var(--dim); font-weight: 300; font-size: 0.5em; }}
  .tagline {{ color: var(--dim); font-size: 1rem; margin-bottom: 2.5rem; }}
  h2 {{ color: var(--accent); font-size: 1.1rem; margin: 2rem 0 0.75rem; letter-spacing: 0.05em;
        text-transform: uppercase; }}
  .stats {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 2rem; }}
  .stat {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
           padding: 1rem 1.25rem; min-width: 140px; }}
  .stat .val {{ font-size: 1.5rem; color: var(--green); font-weight: 700; }}
  .stat .lbl {{ font-size: 0.75rem; color: var(--dim); text-transform: uppercase; letter-spacing: 0.08em; }}
  .features {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
               gap: 1rem; margin-bottom: 1.5rem; }}
  .feature {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
              padding: 1rem 1.25rem; }}
  .feature b {{ color: var(--accent); }}
  .feature p {{ color: var(--dim); font-size: 0.85rem; margin-top: 0.25rem; }}
  pre {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
         padding: 1rem 1.25rem; overflow-x: auto; font-size: 0.85rem; line-height: 1.6; }}
  code {{ color: var(--fg); }}
  .comment {{ color: var(--dim); }}
  table {{ width: 100%; border-collapse: collapse; margin: 0.5rem 0; font-size: 0.85rem; }}
  th, td {{ text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--accent); font-weight: 600; font-size: 0.75rem; text-transform: uppercase;
       letter-spacing: 0.08em; }}
  td code {{ color: var(--green); }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .links {{ display: flex; gap: 1rem; margin-top: 2.5rem; margin-bottom: 1rem; }}
  .links a {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
              padding: 0.6rem 1.25rem; font-size: 0.85rem; transition: border-color 0.2s; }}
  .links a:hover {{ border-color: var(--accent); text-decoration: none; }}
  .footer {{ color: var(--dim); font-size: 0.75rem; margin-top: 3rem; padding-top: 1.5rem;
             border-top: 1px solid var(--border); }}
</style>
</head>
<body>

<h1>shivvr <span>v0.1</span></h1>
<p class="tagline">Semantic memory service. You throw text at it. It figures out the rest.</p>

<div class="stats">
  <div class="stat"><div class="val">{sessions}</div><div class="lbl">Sessions</div></div>
  <div class="stat"><div class="val">{chunks}</div><div class="lbl">Chunks</div></div>
  <div class="stat"><div class="val">{uptime}s</div><div class="lbl">Uptime</div></div>
  <div class="stat"><div class="val">{gpu}</div><div class="lbl">Compute</div></div>
</div>

<h2>What it does</h2>
<div class="features">
  <div class="feature"><b>Ingest</b><p>Chunks text by sentence boundaries, embeds each chunk with GTR-T5-base (768d), stores in sled.</p></div>
  <div class="feature"><b>Search</b><p>Cosine similarity over embeddings with optional time-weighting and temporal context expansion.</p></div>
  <div class="feature"><b>Crypto</b><p>Per-agent orthogonal matrix encryption on embeddings. Your vectors, your keys.</p></div>
  <div class="feature"><b>Inversion</b><p>Vec2text: reconstruct approximate text from embedding vectors. T5-based, currently {inversion}.</p></div>
</div>

<h2>API</h2>
<table>
  <tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
  <tr><td>GET</td><td><code>/health</code></td><td>Status, models, session/chunk counts</td></tr>
  <tr><td>GET</td><td><code>/memory</code></td><td>List all sessions</td></tr>
  <tr><td>POST</td><td><code>/memory/:session/ingest</code></td><td>Ingest text (auto-chunk + embed)</td></tr>
  <tr><td>GET</td><td><code>/memory/:session/search?q=...</code></td><td>Semantic search</td></tr>
  <tr><td>GET</td><td><code>/memory/:session/info</code></td><td>Session metadata</td></tr>
  <tr><td>DELETE</td><td><code>/memory/:session</code></td><td>Delete a session</td></tr>
  <tr><td>POST</td><td><code>/agent/:id/register</code></td><td>Register encryption keys</td></tr>
  <tr><td>POST</td><td><code>/agent/:id/encrypt</code></td><td>Encrypt embeddings</td></tr>
  <tr><td>POST</td><td><code>/agent/:id/decrypt</code></td><td>Decrypt embeddings</td></tr>
  <tr><td>POST</td><td><code>/invert</code></td><td>Reconstruct text from embedding</td></tr>
</table>

<h2>Quick start</h2>
<pre><code><span class="comment"># Clone and run with Docker Compose</span>
git clone git@github.com:DeepBlueDynamics/shivvr.git
cd shivvr
docker compose up -d

<span class="comment"># Ingest some text</span>
curl -X POST http://localhost:8080/memory/my-session/ingest \
  -H "Content-Type: application/json" \
  -d '{{"text": "The harbor was quiet at dawn. Only the sound of halyards against aluminum masts."}}'

<span class="comment"># Search it</span>
curl "http://localhost:8080/memory/my-session/search?q=morning+at+the+marina&amp;n=5"</code></pre>

<h2>Deploy on Docker</h2>
<pre><code><span class="comment"># CPU only (no NVIDIA GPU required)</span>
docker compose up -d

<span class="comment"># With GPU (requires nvidia-container-toolkit)</span>
docker compose up -d    <span class="comment"># compose.yml reserves 1 GPU by default</span>

<span class="comment"># Data persists in the shivvr-data volume</span>
docker volume inspect shivvr-data</code></pre>

<h2>Deploy on Cloud Run (GCP)</h2>
<pre><code><span class="comment"># Build, push, and deploy with L4 GPU</span>
docker compose build
docker tag gnosis-chunk-shivvr gcr.io/YOUR_PROJECT/shivvr:latest
docker push gcr.io/YOUR_PROJECT/shivvr:latest

gcloud run deploy shivvr \
  --image gcr.io/YOUR_PROJECT/shivvr:latest \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 16Gi --cpu 4 \
  --gpu 1 --gpu-type nvidia-l4 \
  --max-instances 1 \
  --port 8080</code></pre>

<h2>Environment</h2>
<table>
  <tr><th>Variable</th><th>Default</th><th>Description</th></tr>
  <tr><td><code>PORT</code></td><td>8080</td><td>Listen port</td></tr>
  <tr><td><code>DATA_PATH</code></td><td>/data/shivvr</td><td>sled database directory</td></tr>
  <tr><td><code>MODEL_PATH</code></td><td>/models/gtr-t5-base.onnx</td><td>Embedding model</td></tr>
  <tr><td><code>TOKENIZER_PATH</code></td><td>/models/tokenizer.json</td><td>Tokenizer</td></tr>
  <tr><td><code>OPENAI_API_KEY</code></td><td>&mdash;</td><td>Enables ada-002 retrieve embeddings</td></tr>
</table>

<h2>Stack</h2>
<table>
  <tr><th>Layer</th><th>Choice</th></tr>
  <tr><td>Runtime</td><td>Rust + Tokio + axum</td></tr>
  <tr><td>Embedding</td><td>GTR-T5-base (768d) via ONNX Runtime</td></tr>
  <tr><td>Vector ops</td><td>simsimd (SIMD-accelerated cosine)</td></tr>
  <tr><td>Storage</td><td>sled (embedded, ACID, persistent)</td></tr>
  <tr><td>GPU</td><td>CUDA 12.6 via ort EP (optional)</td></tr>
</table>

<div class="links">
  <a href="https://github.com/DeepBlueDynamics/shivvr">GitHub</a>
  <a href="/health">Health Check</a>
  <a href="/memory">Sessions</a>
</div>

<div class="footer">shivvr &middot; Rust + ONNX + sled &middot; DeepBlueDynamics</div>

</body>
</html>"##))
}

// ===== Router =====

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(homepage))
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
