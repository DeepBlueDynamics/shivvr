# shivvr

Ephemeral semantic embedding service. Ingests text, chunks it, embeds it with GTR-T5-base (768d), and returns ranked results. `POST /embed` also serves SigLIP text vectors (same space as image vectors) and EmbeddingGemma 2 text vectors, and an OpenAI embedder can be added for the `retrieve` role. No persistence — all state is in-process and lost on restart.

Rust + ONNX Runtime. Runs in Docker on port 8080.

## What it does

- **Ingest** — chunks text by sentence boundaries and embeds each chunk with GTR-T5-base (768d, local, always on). Optional `audio_base64` is transcribed first; optional `image_base64` adds an image chunk embedded with SigLIP
- **Search** — cosine similarity, optional temporal decay, optional BM25 hybrid or lexical-only ranking, and a query guardrail
- **Embedders**
  - GTR-T5-base (768d) — ingest, search (`organize` role), and the default for `/embed`. Required
  - SigLIP base patch16-224 (768d) — image vectors (`/image/embed`, image ingest) and text vectors in the same space (`/embed`), for text-to-image comparison. Optional
  - EmbeddingGemma 2, text only (768d, Matryoshka 128/256/512) — `/embed` with task prefixes. Optional
  - OpenAI text-embedding-ada-002 (1536d) — `retrieve` role on ingest and search, when `OPENAI_API_KEY` is set or the caller passes `openai_api_key`. Optional
- Vectors from different models are not comparable with each other, and `/invert` only reconstructs GTR vectors
- **Per-agent encryption** — orthogonal matrix rotation on embeddings; preserves cosine similarity, keys ephemeral
- **Vec2text inversion** — reconstruct approximate text from a GTR vector (optional, requires inverter models)
- **Temp store** — named ephemeral vector stores with 2 hr TTL, separate from session store
- **Auth** — nuts-auth JWT + API token verification (optional; open dev mode if `NUTS_AUTH_JWKS_URL` unset)

## Prerequisites

### NVIDIA Driver

The Docker image uses CUDA 12.6. Host driver must be **545+**.

```bash
# Check
nvidia-smi

# Ubuntu/Debian upgrade
sudo apt-get install -y nvidia-driver-590 && sudo reboot
```

Windows: download from nvidia.com, run installer, reboot.

### Docker + NVIDIA Container Toolkit

**Linux:**
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

**Windows:** Docker Desktop with WSL 2 backend includes GPU support automatically.

### Models

Models are not in the repo: GTR-T5-base, the vec2text inverter, the SigLIP vision and
text towers, the SigLIP tokenizer JSON, and the EmbeddingGemma 2 text model and
tokenizer JSON. Everything but EmbeddingGemma 2 is about 2.2 GB; the EmbeddingGemma 2
text model (270M parameters, float32) adds roughly another 1 GB, not measured yet. They come
from a prebuilt image, `gcr.io/gnosis-459403/shivvr-models:latest`, which
`Dockerfile.models` produces and the app `Dockerfile` copies `/models` from. Rebuild
that image only when the export scripts change:

```bash
bash deploy.sh --rebuild-models    # Cloud Build
```

`docker compose build` pulls the models image from GCR, so authenticate first
(`gcloud auth configure-docker`) or point the build elsewhere with
`docker compose build --build-arg MODELS_IMAGE=<image>`.

Only GTR-T5-base is required at startup. A missing inverter, SigLIP or EmbeddingGemma 2
file logs a line and disables that feature (`/invert`, `/image/embed`, or that `/embed`
model returns `503`).

To run the binary outside Docker, export into `models/` locally instead:

```bash
bash scripts/fetch_models.sh                            # GTR-T5-base + vec2text, with verification
pip install torch transformers sentencepiece protobuf onnx onnxruntime
python scripts/export_siglip.py --output_dir models/    # SigLIP towers + siglip-tokenizer.json
```

`export_siglip.py` builds the fast tokenizer JSON straight from `spiece.model`
(transformers has no converter for `SiglipTokenizer`) and verifies it against the
slow tokenizer before writing it, so a mismatch fails the export instead of shipping
wrong vectors.

EmbeddingGemma 2 needs a newer stack than the two exports above (`Dockerfile.models`
runs it in its own Python 3.12 stage), so use a separate virtualenv:

```bash
pip install "torch>=2.9" "transformers>=5.18" "sentence-transformers>=6.1" onnx onnxscript onnxruntime
python scripts/export_embeddinggemma2.py --output_dir models/   # embeddinggemma2-text.onnx + embeddinggemma2-tokenizer.json
```

The script exports the text backbone only, in float32 (the model produces NaN in
float16), with masked mean pooling and L2 normalization in the graph. It fails unless
the tokenizer JSON reproduces SentenceTransformer's token ids and the ONNX output
reaches cosine >= 0.9999 (`--min_cosine`) against `SentenceTransformer.encode()`.

**Not verified yet:** the EmbeddingGemma 2 export script and its `Dockerfile.models`
stage have not been run end to end, so no real EmbeddingGemma 2 vector has been
produced or checked against Python. That is WP1 in `plan/EMBEDDINGGEMMA2_PLAN.md`.

## Quick start

```bash
docker compose up -d
```

Builds from source with CUDA, with models copied in from the prebuilt models image (see above).
Listens on `:8080` inside the container; compose maps it to `:8085`. No volume needed.
The examples below use `localhost:8080`; through compose, use `localhost:8085`.

## API

### Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Status, version, loaded models, session/chunk counts |
| POST | `/sessions/:id/ingest` | Ingest text (chunk + embed) |
| GET | `/sessions/:id/search?q=...` | Semantic search |
| GET | `/sessions/:id` | Session metadata |
| DELETE | `/sessions/:id` | Delete session |

### Temp store

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/temp` | List named temp stores (with expiry) |
| POST | `/temp/:name/ingest` | Ingest into ephemeral named store (2 hr TTL) |
| GET | `/temp/:name/search?q=...` | Search temp store |
| GET | `/temp/:name/dump` | Dump all chunks |
| DELETE | `/temp/:name` | Delete temp store |

### Crypto

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/agent/:id/register` | Register per-agent orthogonal key |
| POST | `/agent/:id/encrypt` | Encrypt embeddings |
| POST | `/agent/:id/decrypt` | Decrypt embeddings |

### MCP and agent chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/mcp/sse` | MCP over SSE |
| POST | `/mcp/message` | MCP messages |
| POST | `/sessions/:id/agent/chat` | Agent loop over a session's memory (uses `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) |

These three routes sit outside the auth gate and the request timeout.

### Health

`GET /health` returns `status`, `version`, `models`, `sessions`, `total_chunks`,
`uptime_seconds`, `encryption_available`, `inversion_available`, `audio_available`,
`vision_available` and `gpu` (whether the binary was built with CUDA). Each `models`
entry is `{"name", "role", "dimension", "status"}`, listed only when loaded:

| `name` | `role` | `dimension` | Present when |
|--------|--------|-------------|--------------|
| `gtr-t5-base` | `organize` | 768 | always |
| `text-embedding-ada-002` | `retrieve` | 1536 | `OPENAI_API_KEY` set (name and dimension are fixed even if `OPENAI_EMBEDDING_MODEL` overrides the model) |
| `siglip-base-patch16-224` | `multimodal` | 768 | SigLIP vision tower loaded (`/image/embed`) |
| `siglip-base-patch16-224` | `multimodal-text` | 768 | SigLIP text tower and tokenizer loaded (`/embed`) |
| `embeddinggemma-2` | `text` | 768 | EmbeddingGemma 2 model and tokenizer loaded (`/embed`) |

Check this list before calling `/embed` with a non-default model.

### Inversion

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/invert` | Reconstruct approximate text from a GTR embedding |

`{"embedding": [...768 floats], "max_length": 64}` returns `{"text", "similarity"}`,
where `similarity` is the cosine between your vector and the GTR embedding of `text`.
`max_length` (default 64) bounds the decoded tokens. The inverter is trained on
GTR-T5-base vectors only; SigLIP, EmbeddingGemma 2 or OpenAI vectors produce
meaningless text.

Inversion is **approximate, not lossless**. Embedding is many-to-one — many
different sentences map to nearly the same 768d vector — so a single `/invert`
call returns text that is *semantically close* to the source but rarely the same
words. Invert the embedding of "Rust is a systems programming language" and you
might get back "the Rust language is used for systems programming": same meaning,
different tokens.

To get closer, **iterate** (vec2text-style correction loop):

1. Start with the target vector `v`.
2. `/invert` → candidate text `t` and its `similarity` to `v`.
3. If `similarity` is not close enough, re-embed or adjust `t` and invert again.

Each round nudges the wording so its embedding moves toward `v`; cosine
similarity climbs and the text stabilizes. The fixed point is text whose
embedding reproduces the target vector — text and vector have converged, even if
the final wording still differs from the original. (Requires the inverter models — see the
`INVERTER_*` env vars; otherwise `/invert` returns `503`.)

### Embed

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/embed` | Batched text vectors (GTR, SigLIP text or EmbeddingGemma 2), no store side effects |
| POST | `/image/embed` | SigLIP image vector (768d) from `image_base64` |
| POST | `/audio/embed` | Transcribe `audio_base64`, then embed the transcript (GTR, 768d) |
| POST | `/audio/transcribe` | Transcript only |

`POST /embed` request fields:

| Field | Required | Description |
|-------|----------|-------------|
| `texts` | yes | 1–256 strings, each non-blank and at most 32 KiB (UTF-8 bytes) |
| `model` | no | `gtr-t5-base` (default; same vectors ingest stores), `siglip-base-patch16-224` (same space as `/image/embed`), or `embeddinggemma-2` |
| `task` | no | `embeddinggemma-2` only. Prompt name whose prefix is prepended to every text (see below). Omitted or blank means no prefix |
| `dimensions` | no | `embeddinggemma-2` only: `128`, `256`, `512` or `768` (default). Smaller sizes are the leading components re-normalized (Matryoshka) |

The response is `{"model", "dim", "vectors"}`: one L2-normalized vector per text, in
order, with `dim` the length returned.

`task` names for `embeddinggemma-2`, matched case-insensitively: use `query` for
search queries and `document` for the passages they search. The others are the
model's prompt names: `Retrieval-query`, `Retrieval-document`, `Retrieval`,
`SearchQuery`, `Reranking`, `BitextMining`, `QuestionAnswering`, `FactChecking`,
`Classification`, `MultilabelClassification`, `Clustering`, `STS`,
`SentenceSimilarity`, `PairClassification`, `Summarization`, `CodeRetrieval`,
`InstructionRetrieval`. No prefix matches `SentenceTransformer.encode()` without a
prompt. EmbeddingGemma 2 input is truncated to 2,048 tokens by default (`EMBEDDINGGEMMA2_MAX_TOKENS`, up to the model's 8,192).

```bash
# GTR-T5-base (default)
curl -X POST http://localhost:8080/embed -H "Content-Type: application/json" \
  -d '{"texts": ["The harbor was quiet at dawn."]}'

# SigLIP text, comparable with /image/embed vectors
curl -X POST http://localhost:8080/embed -H "Content-Type: application/json" \
  -d '{"texts": ["a photo of a sailboat"], "model": "siglip-base-patch16-224"}'

# EmbeddingGemma 2: query and documents with matching tasks, 256d
curl -X POST http://localhost:8080/embed -H "Content-Type: application/json" \
  -d '{"texts": ["waterproof trail shoes"], "model": "embeddinggemma-2", "task": "query", "dimensions": 256}'
curl -X POST http://localhost:8080/embed -H "Content-Type: application/json" \
  -d '{"texts": ["Gore-Tex lined hiking shoe with a lugged sole."], "model": "embeddinggemma-2", "task": "document", "dimensions": 256}'
```

Errors are `{"error": "..."}`:

- `400` — unknown `model`; unknown `task`; `dimensions` not in 128/256/512/768;
  `task` or a non-768 `dimensions` sent with `gtr-t5-base` or
  `siglip-base-patch16-224`; empty `texts`; more than 256 texts; a blank text; a
  text over 32 KiB.
- `503` — `siglip-base-patch16-224` requested but the SigLIP text tower or its
  tokenizer is not loaded, or `embeddinggemma-2` requested but its model or
  tokenizer is not loaded. `/health` shows which are loaded.
- `500` — inference failed. `413` and `408` as in [Request limits](#request-limits).

Vectors from the three models are in different spaces: compare a vector only with
vectors from the same model (and, for EmbeddingGemma 2, the same `dimensions`).
Image, audio and video inputs for EmbeddingGemma 2 are not served yet (see
`plan/EMBEDDINGGEMMA2_PLAN.md`).

### Request limits

| Routes | Max body | Notes |
|--------|----------|-------|
| `/sessions/:id/ingest`, `/temp/:name/ingest`, `/image/embed`, `/audio/*` | 32 MiB | base64 media payloads |
| `/embed` | 8 MiB | 256 texts × 32 KiB |
| everything else | 2 MiB | axum default |

Requests over the limit are rejected with `413 Payload Too Large`. Handlers
that exceed the server timeout (see `SHIVVR_REQUEST_TIMEOUT_SECS`) get `408`.

### Ingest

```bash
curl -X POST http://localhost:8080/sessions/my-session/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The harbor was quiet at dawn. Only the sound of halyards against aluminum masts.",
    "source": "journal",
    "emotion_primary": "calm"
  }'
```

Fields: `text`, `source`, `metadata` (any JSON), `emotion_primary`,
`emotion_secondary`, `agent_id` (encrypt with that agent's keys), `openai_api_key`
(per-request key for the `retrieve` vector), `mean_pool` (return one pooled vector
instead of one per chunk), `audio_base64`, `image_base64`. The response is
`{"chunks_created", "tokens_processed", "time_ms", "chunks"}`.

An `image_base64` chunk stores a SigLIP vector in the same session as GTR text
chunks, and `organize` search scores it against a GTR query vector, so its score is
not meaningful.

### Search

```bash
# Basic
curl "http://localhost:8080/sessions/my-session/search?q=morning+at+the+marina&n=5"

# With temporal decay (half-life 24 hours, 30% time weight)
curl "http://localhost:8080/sessions/my-session/search?q=marina&n=5&time_weight=0.3&decay_halflife_hours=24"

# BM25 + vector rank fusion
curl "http://localhost:8080/sessions/my-session/search?q=marina&hybrid=true"

# retrieve role (requires OPENAI_API_KEY or openai_api_key)
curl "http://localhost:8080/sessions/my-session/search?q=marina&role=retrieve"
```

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Listen port |
| `BIND_ADDR` | `127.0.0.1` | Listen address. The Docker images set `0.0.0.0`; set it yourself only to expose a non-Docker run on the network |
| `MODEL_PATH` | `models/gtr-t5-base.onnx` | GTR-T5-base ONNX embedder (required) |
| `TOKENIZER_PATH` | `models/tokenizer.json` | GTR tokenizer |
| `OPENAI_API_KEY` | — | Enables the text-embedding-ada-002 `retrieve` role; also used by agent chat |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-ada-002` | Override OpenAI embedding model |
| `NUTS_AUTH_JWKS_URL` | — | Enable auth (open dev mode if unset) |
| `NUTS_AUTH_VALIDATE_URL` | `https://auth.nuts.services/api/validate` | API token validation |
| `INVERTER_PROJECTION_PATH` | `models/inverter/projection.onnx` | Vec2text projection |
| `INVERTER_ENCODER_PATH` | `models/inverter/encoder.onnx` | Vec2text T5 encoder |
| `INVERTER_DECODER_PATH` | `models/inverter/decoder.onnx` | Vec2text T5 decoder |
| `INVERTER_TOKENIZER_PATH` | `models/inverter/tokenizer.json` | Vec2text tokenizer |
| `VISION_MODEL_PATH` | `models/siglip-vision.onnx` | SigLIP vision tower (`/image/embed`) |
| `SIGLIP_TEXT_MODEL_PATH` | `models/siglip-text.onnx` | SigLIP text tower (`/embed` model=siglip-base-patch16-224) |
| `SIGLIP_TOKENIZER_PATH` | `models/siglip-tokenizer.json` | Fast-tokenizer JSON for the SigLIP text tower |
| `EMBEDDINGGEMMA2_MODEL_PATH` | `models/embeddinggemma2-text.onnx` | EmbeddingGemma 2 text path (`/embed` model=embeddinggemma-2) |
| `EMBEDDINGGEMMA2_TOKENIZER_PATH` | `models/embeddinggemma2-tokenizer.json` | Tokenizer JSON for EmbeddingGemma 2 |
| `EMBEDDINGGEMMA2_MAX_TOKENS` | `2048` | Input token cap for EmbeddingGemma 2 (max 8192). Cost grows faster than linearly: about 5 s per text at 2,048 tokens, and 70-90 s with ~8 GB RSS at 8,192, on CPU |
| `TRANSCRIPTION_URL` | `http://localhost:8765` | Transcription service for `/audio/*` and `audio_base64` ingest |
| `GUARDRAILS_DIR` | `guardrails` | Guardrail phrase CSVs for search (a default `offensive.csv` is written if missing) |
| `ANTHROPIC_API_KEY` | — | Agent chat LLM (preferred over OpenAI when set) |
| `ANTHROPIC_MODEL` | `claude-3-5-haiku-20241022` | Agent chat model (Anthropic) |
| `OPENAI_MODEL` | `gpt-4o-mini` | Agent chat model (OpenAI) |
| `OPENAI_API_URL` | `https://api.openai.com/v1/chat/completions` | Agent chat endpoint (OpenAI-compatible) |
| `LANDING_ONLY` | unset | `true` serves only the static homepage and a stub `/health`, loading no models |
| `SHIVVR_ENABLE_RUN_COMMAND` | unset (off) | **Danger.** Exposes the MCP `run_command` tool, which runs arbitrary shell inside the container. Leave unset anywhere the MCP endpoint is reachable by untrusted clients |
| `SHIVVR_REQUEST_TIMEOUT_SECS` | `120` | Server-side timeout for inference routes (`/embed`, `/image/embed`, `/audio/*`, ingest, `/invert`); other routes use `min(30, this)`. Timed-out requests get `408` with `{"error": ...}` |

## Search query parameters

Same for `/sessions/:id/search` and `/temp/:name/search`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `q` | required | Search query text |
| `n` (alias `limit`) | `5` | Number of results |
| `role` | `organize` | `organize` (768d GTR) or `retrieve` (1536d OpenAI) |
| `openai_api_key` | — | Per-request OpenAI key for `retrieve` (sessions only) |
| `time_weight` | `0.0` | Blend semantic score with recency (0–1) |
| `decay_halflife_hours` | `168` | Recency decay half-life in hours |
| `include_nearby` | — | Include temporally adjacent chunks in results |
| `time_window_minutes` | `30` | Window for nearby chunks |
| `agent_id` | — | Agent ID for encrypted search |
| `hybrid` | `false` | Fuse vector and BM25 rankings |
| `lexical_only` | `false` | BM25 only, no query embedding |
| `guardrail` | `true` | Reject queries containing blocked guardrail phrases |

## Auth

If `NUTS_AUTH_JWKS_URL` is set, the service enforces nuts-auth. Without a token
(`Authorization: Bearer <jwt>` or `Authorization: ahp_<token>`) these stay open:

- `GET /`, `GET /health`
- search with `role=organize` (the default)
- `/embed`, `/image/embed`, `/audio/*` (local compute only)
- session and temp ingest, but only when no server `OPENAI_API_KEY` is configured

Everything else behind the gate needs a token, including `role=retrieve` search,
session metadata and delete, temp list/dump/delete, the crypto routes and `/invert`.
MCP and agent chat routes are not gated.

Leave `NUTS_AUTH_JWKS_URL` unset for open dev mode.

## Stack

- **Rust** — axum, tokio, ort (ONNX Runtime 2.0)
- **Embedding** — GTR-T5-base (sentence-transformers), 768d, L2-normalized
- **Multimodal** — SigLIP base patch16-224 vision and text towers, 768d, one shared space
- **Text embedding** — EmbeddingGemma 2 text path, 768d with Matryoshka truncation (optional, export not yet verified)
- **Search** — cosine + BM25 rank fusion and FST guardrails (lume_hybrid)
- **Storage** — ephemeral `RwLock<HashMap>`, no disk persistence
- **Inversion** — vec2text gtr-base (projection + T5 encoder/decoder, optional)
- **Auth** — nuts-auth RS256 JWT + `ahp_` API tokens

## Versioning & releases

`version` in `Cargo.toml` is the single source of truth; `/health` reports it. Every release
is tagged `v<version>`. `.github/workflows/release.yml` builds the gateway image on tag push to
`ghcr.io/deepbluedynamics/shivvr-gateway` (and to Docker Hub as `deepbluedynamics/shivvr-gateway`
when the repo has `DOCKER_USERNAME` / `DOCKER_TOKEN` secrets), and refuses a tag that does not
match `Cargo.toml`. The CUDA app image is built on Cloud Build by `deploy.sh` as
`gcr.io/gnosis-459403/shivvr:v<version>`. Full procedure: [OPERATIONS.md](OPERATIONS.md#versioning--releases).

## License

BSD 3-Clause. See [LICENSE](LICENSE).
