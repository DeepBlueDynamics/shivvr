# SHIVVR — Codebase Review (April 2026)

## What it is

Shivvr is an embedded semantic memory service written in Rust. It accepts text, chunks it semantically, embeds it, and stores the resulting vectors in a local sled database. It exposes an HTTP API for ingesting and searching memory, with optional per-agent encryption, dual-role embeddings, and a T5-based vec2text inversion pipeline.

---

## Stack

| Layer | Choice | Notes |
|---|---|---|
| HTTP | axum + tokio | Async, concurrent |
| Embeddings (organize) | GTR-T5-base via ONNX Runtime | 768d, local, required |
| Embeddings (retrieve) | OpenAI ada-002 | 1536d, optional, via API key |
| Vector store | sled (embedded KV) | ACID, on-disk, all vectors stored |
| Similarity | simsimd (SIMD cosine) | Feature-gated, f32 |
| Encryption | Per-agent orthogonal matrix | Preserves cosine similarity |
| Inversion | T5-base hypothesis + corrector | Optional, ONNX, vec→text |
| GPU | CUDA 12.6 via `ort/cuda` feature | Falls back to CPU gracefully |

---

## How search works

### Ingest (`POST /memory/:session/ingest`)

1. **Chunk**: Monte Carlo sampling identifies semantic discontinuities between sentences. Binary search and iterative refinement pin the boundaries. Token limits (50–512) are enforced; oversized chunks are split recursively.
2. **Embed**: Each chunk is embedded via GTR-T5-base ONNX → 768d, L2-normalized. If `OPENAI_API_KEY` is set, ada-002 is also called → 1536d, stored separately as `embedding_retrieve`.
3. **Encrypt** (optional): If an `agent_id` is provided and registered, the embedding is multiplied by the agent's orthogonal matrix (`v @ Q`). This preserves cosine similarity while making vectors uninterpretable without the key.
4. **Store**: Every chunk, including its full embedding vector(s), is written to sled under key `chunks:{session_id}:{chunk_id}`.

### Search (`GET /memory/:session/search`)

1. **Embed query**: Using the same model as was selected at ingest (GTR or ada-002 based on `role` param).
2. **Encrypt query** (optional): Same matrix applied to query as was applied at ingest.
3. **Brute-force scan**: All chunks for the session are loaded, cosine similarity is computed (SIMD-accelerated) against the query, and optionally blended with a temporal decay score (`age_hours / 168.0`).
4. **Return top-k**: Results include text, score, metadata, emotion tags. If `include_nearby=true`, chunks created within ±N minutes of each result are also included.

### Vector storage: ALL vectors are stored

Sled persists 100% of chunk embeddings — no pruning, no sampling, no indexing structure like HNSW or IVF. Every ingest writes bincode/JSON-encoded vectors to disk. Search is an O(n) linear scan across all chunks in the session. This is fine at small scale but becomes a bottleneck as session chunk counts grow into the thousands.

---

## Deployment modes

### Local Docker (GPU passthrough)
- `docker-compose.yml` passes one NVIDIA GPU to the container.
- The Dockerfile builds with `--features cuda` against CUDA 12.6.3 + cuDNN.
- sled data is volume-mounted at `/data`.
- Models are baked into the image during build (copied from `/models/`).

### Cloud Run (GCP with L4 GPU)
- `deploy.sh` tags the local image, pushes to GCR, and deploys with `--gpu=1 --gpu-type=nvidia-l4`.
- Region: `us-central1`, 16 GB RAM, 4 CPUs, max 1 instance, min 0.
- **Data is ephemeral** — Cloud Run has no volume mount in the current deploy script. Each cold start begins with an empty sled database. This is a significant issue for a memory service.
- Models need to be baked into the container image (correct) or loaded from GCS on startup (not yet implemented).

### CPU-only mode
- Build without `cuda` feature: `cargo build --release --features ml`.
- ONNX Runtime falls back to CPU if CUDA is unavailable at runtime.
- No code changes needed; the feature flag handles it.

---

## Environment variables

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `PORT` | `8080` | No | Listen port |
| `DATA_PATH` | `/data/shivvr` | No | sled database directory |
| `MODEL_PATH` | `/models/gtr-t5-base.onnx` | **Yes** | Organize embedding model |
| `TOKENIZER_PATH` | `/models/tokenizer.json` | **Yes** | Tokenizer for above |
| `OPENAI_API_KEY` | — | No | Enables ada-002 retrieve embeddings |
| `API_TOKEN` | — | No | Enables bearer auth on all endpoints |
| `INVERTER_*_PATH` | `/models/inverter/*.onnx` | No | vec2text model paths |
| `INVERTER_TOKENIZER_PATH` | `/models/inverter/tokenizer.json` | No | Inversion tokenizer |

---

## Training pipeline (Python, `training/`)

Trains two T5-base models to invert embeddings back to text:

1. **Generate embeddings**: Encode MSMARCO corpus with BGE-small-en-v1.5 → `.npy` + `.txt`.
2. **Train hypothesis model**: (embedding → text) fine-tuned T5-base.
3. **Generate hypotheses**: Run hypothesis model on all embeddings.
4. **Train corrector model**: (embedding + hypothesis + hypothesis_embedding → refined text) T5-base.
5. **Export to ONNX**: `export_onnx_full.py` → encoder, decoder, projection ONNXs.
6. **Test**: BLEU/similarity eval.

Timeline: 5–7 days on a 12 GB GPU, 3–5 days on 24 GB+. The inverter in the Rust service uses greedy decoding capped at 64 tokens.

---

## Issues and update priorities

### Critical

**1. Dimension mismatch bug in retrieve fallback**
When `role=retrieve` is requested and `openai_embedder` is `None` (key not set), the search path
falls back to GTR-T5-base and produces a 768d query vector. But stored `embedding_retrieve` values
are 1536d (written at ingest time when ada-002 was available). The store's role-based field
selector will compare a 768d query against 1536d stored vectors, producing either a panic or
silently wrong cosine scores.

Fix: if `role=retrieve` is requested but ada-002 is unavailable, the handler should either:
(a) return a 400 with a clear message (`retrieve role requires OpenAI API key`), or
(b) fall back to `role=organize` explicitly and document the downgrade in the response.
The current silent fallback to GTR without changing the comparison field is the bug.

**2. Cloud Run data is ephemeral**
`deploy.sh` does not mount persistent storage. Every cold start loses all ingested memory. Options:
- Mount a Cloud Filestore NFS share (expensive, always-on)
- Replace sled with Cloud Spanner, Firestore, or Cloud SQL and serialize vectors via pgvector
- Use GCS as a snapshot backend: dump/restore sled on startup/shutdown (acceptable for single-instance deployments with infrequent cold starts)
- Switch to a managed vector DB (Pinecone, Qdrant, Weaviate Cloud) for pure retrieval and keep metadata elsewhere

**2. Linear scan doesn't scale**
Search is O(n) over all chunks in a session. At ~10K chunks (typical for a long-running agent), each query reads and scores every vector. For larger deployments, replace with an ANN index (HNSW via `usearch` or `instant-distance` crates, or offload to Qdrant).

### Moderate

**3. No reproducible model artifact path**
`models/` is excluded from version control (`.gitignore`). The Docker build copies model files from
the local filesystem during `COPY models/ /models/`. There is no documented procedure for obtaining
or re-creating the GTR-T5-base ONNX file, tokenizer, or the inverter ONNX assets. Anyone who
clones the repo cannot build the Docker image. Options:
- Store model artifacts in GCS and fetch at build time or on container startup
- Commit a model download script (`scripts/fetch_models.sh`) that pulls from a known GCS/HuggingFace URI
- Document exact export steps and add them to the Dockerfile as a build stage

**4. ada-002 is deprecated (model sunsetting)**
OpenAI has moved to `text-embedding-3-small` (1536d) and `text-embedding-3-large` (3072d). ada-002 will eventually be removed. The `openai.rs` module should be updated to use `text-embedding-3-small` with the same dimension, or made configurable.

**5. Cloud Run concurrency vs ONNX session locking**
`deploy.sh` sets `--concurrency 80` (80 simultaneous requests per instance). ONNX Runtime GPU
sessions are not thread-safe by default — concurrent inference calls serialize on a lock or error.
With `--max-instances 1`, all 80 concurrent requests share one ONNX session. This will produce
latency spikes or errors under real load. Either:
- Lower concurrency to match expected ONNX throughput (likely 1–4 for GPU inference)
- Create a pool of ONNX sessions (one per tokio worker thread) so inference can proceed in parallel
- Benchmark actual behavior before leaving 80 in production

**6. Chunker is embedding-expensive**
The Monte Carlo boundary search calls the embedding model 10–50 times per document. For long documents this is the dominant ingest cost. Could be improved by:
- Caching sentence embeddings during the boundary search
- Using a lightweight proxy model for candidate sampling (then final embed with GTR)

**7. Inverter max length is hardcoded**
`max_length = 64` tokens in `inverter.rs`. Should be a configurable query param or env var.

**8. No ANN index for retrieval**
Currently sled stores raw vectors and search is a flat scan. Adding even a simple HNSW index would make retrieval O(log n) instead of O(n).

**9. Temporal decay is hardcoded**
`age_hours / 168.0` (one-week half-life) is hardcoded. Should be configurable per-request or per-session.

### Minor / Quality

**10. Warning if `API_TOKEN` unset**
Superseded by nuts-auth plan — see AUTH_PLAN.md.

**11. No integration tests**
Unit tests exist in `store.rs`, `crypto.rs`, `openai.rs`. No end-to-end test covers ingest → search → result validation. Worth adding a small test harness that runs against a real (tempdir) sled instance.

**12. Training/production stack drift**
The `training/` directory targets BGE-small-en-v1.5 (384d embeddings). The production Docker image
expects GTR-T5-base (768d) and GTR-compatible inverter ONNX assets. These two are incompatible —
the inverter models produced by the training pipeline cannot be dropped into `/models/inverter/` and
used by the running service without re-exporting against GTR-T5-base instead of BGE-small. Older
specs and comments still reference bge-small throughout. Before running the training pipeline to
produce new inverter models, the training scripts need to be updated to use GTR-T5-base as the
embedding model, not BGE-small.

**13. Session isolation is soft**
Sessions are separated only by key prefix in sled. There is no access control between sessions — any caller with the API token can read any session. If multi-tenant isolation matters, session-level tokens are needed.

**14. GPU warm-up latency**
First inference after a cold start takes several seconds (ONNX graph compilation + CUDA context init). Cloud Run min-instances=0 means every cold start pays this penalty. Consider min-instances=1 or a lightweight `/warmup` endpoint that triggers a dummy inference.

---

## Update sequence (suggested order)

1. **Fix the retrieve fallback dimension bug** — either error on unavailable ada-002 or explicitly downgrade role
2. **Add model artifact fetch path** — GCS download script or documented export steps so the Docker build is reproducible
3. **Fix Cloud Run persistence** — GCS snapshot backend or swap sled for a managed store
4. **Add nuts-auth integration** — see AUTH_PLAN.md
5. **Lower Cloud Run concurrency** — benchmark ONNX session throughput, set `--concurrency` accordingly
6. **Update ada-002 → text-embedding-3-small** in `openai.rs`
7. **Update training pipeline to target GTR-T5-base** — align `training/` scripts with the production embedding model before doing any new inverter training run
8. **Add ANN index** (usearch or qdrant-client) for sessions above a configurable threshold
9. **Make temporal decay and inverter max_length configurable**
10. **Add integration test harness** (tempdir sled, mock embedder or real model)
11. **Cache sentence embeddings in chunker** to reduce ingest cost
12. **Retrain inverter on domain data** once the pipeline is aligned and stable
