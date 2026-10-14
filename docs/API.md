# shivvr — API Reference

All routes are served on `:8080` by default (`PORT=8080`). Responses and error payloads are JSON (`Content-Type: application/json`), except `GET /` (HTML) and the SSE streaming routes (`/mcp/sse`, `/sessions/:id/agent/chat`).

## Endpoints overview

### Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Status, version, loaded models, session/chunk counts, feature flags |
| `POST` | `/sessions/:id/ingest` | Ingest text, audio or image into session (chunk + embed) |
| `GET` | `/sessions/:id/search?q=...` | Semantic, BM25 lexical or hybrid RRF search |
| `GET` | `/sessions/:id` | Session metadata (`chunks`, `total_tokens`, `sources`, `memory_bytes`, timestamps) |
| `DELETE` | `/sessions/:id` | Delete session and all its in-memory chunks |

### Temp store

Named ephemeral vector stores with a 2-hour TTL (swept every 10 minutes), separate from the session store:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/temp` | List active named temp stores (`name`, `chunks`, `expires_at`) |
| `POST` | `/temp/:name/ingest` | Ingest into an ephemeral named store (resets 2 hr TTL) |
| `GET` | `/temp/:name/search?q=...` | Search temp store (same query parameters as session search) |
| `GET` | `/temp/:name/dump` | Dump all chunks in a temp store |
| `DELETE` | `/temp/:name` | Delete temp store |

### Direct embedding & transcription

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/embed` | Batched text vectors (`embeddinggemma-2` default, `gtr-t5-base` or `siglip-base-patch16-224`), no store side effects |
| `POST` | `/image/embed` | SigLIP image vector (768d) from `image_base64` |
| `POST` | `/audio/embed` | Transcribe `audio_base64`, then embed the transcript (768d) |
| `POST` | `/audio/transcribe` | Transcribe `audio_base64` via the transcription service (transcript only) |

### Inversion

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/invert` | Reconstruct approximate text from a 768d `gtr-t5-base` embedding |

### Crypto

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/agent/:id/register` | Register per-agent orthogonal rotation key matrix (`dim²` floats) |
| `POST` | `/agent/:id/encrypt` | Multiply embeddings by the agent's orthogonal matrix |
| `POST` | `/agent/:id/decrypt` | Multiply embeddings by the transpose of the agent's orthogonal matrix |

### MCP and agent chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/mcp/sse` | Model Context Protocol (MCP) over Server-Sent Events |
| `POST` | `/mcp/message` | MCP JSON-RPC 2.0 message endpoint (`?id=<client_id>`) |
| `POST` | `/sessions/:id/agent/chat` | Stream GhostAgent cognitive turns over a session's memory (SSE; uses `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) |

These three routes sit outside the auth gate and the request timeout.

---

## Health (`GET /health`)

`GET /health` returns `status`, `version`, `default_model` (the text model used when a request names none: `embeddinggemma-2` when loaded, otherwise `gtr-t5-base`), `models`, `sessions`, `total_chunks`, `uptime_seconds`, `encryption_available`, `inversion_available`, `audio_available`, `vision_available` and `gpu` (whether the binary was built with the `cuda` feature). Each `models` entry is `{"name", "role", "dimension", "status"}`, listed only when loaded:

| `name` | `role` | `dimension` | Present when |
|--------|--------|-------------|--------------|
| `embeddinggemma-2` | `text` | 768 | EmbeddingGemma 2 model and tokenizer loaded; then it is the `default_model` for `/embed`, ingest and search |
| `gtr-t5-base` | `organize` | 768 | always (required at startup); used for sessions created with `model=gtr-t5-base`, and the only space `/invert` reads |
| `text-embedding-ada-002` | `retrieve` | 1536 | `OPENAI_API_KEY` set (name and dimension are fixed in `/health` even if `OPENAI_EMBEDDING_MODEL` overrides the model) |
| `siglip-base-patch16-224` | `multimodal` | 768 | SigLIP vision tower loaded (`/image/embed`) |
| `siglip-base-patch16-224` | `multimodal-text` | 768 | SigLIP text tower and tokenizer loaded (`/embed`) |

Check this list before calling `/embed` or ingest with a non-default model. When hitting the lightweight gateway (`shivvr-gateway` or `LANDING_ONLY=true`), `GET /health` returns `{"status": "ok", "mode": "gateway", "backend": "on demand"}`.

---

## Embed (`POST /embed`, `/image/embed`, `/audio/*`)

`POST /embed` request fields:

| Field | Required | Description |
|-------|----------|-------------|
| `texts` | yes | 1–256 strings, each non-blank and at most 32 KiB (UTF-8 bytes) |
| `model` | no | `embeddinggemma-2` (default, 768d), `gtr-t5-base` (768d; invertible via `/invert`), or `siglip-base-patch16-224` (same space as `/image/embed`) |
| `task` | no | `embeddinggemma-2` only. Prompt name whose prefix is prepended to every text (see below). Omitted or blank means no prefix |
| `dimensions` | no | `embeddinggemma-2` only: `128`, `256`, `512` or `768` (default). Smaller sizes are the leading components re-normalized (Matryoshka) |

The response is `{"model", "dim", "vectors"}`: one L2-normalized vector per text, in order, with `dim` the length returned.

`task` names for `embeddinggemma-2`, matched case-insensitively: use `query` for search queries and `document` for the passages they search (session/temp ingest and search apply `document` and `query` automatically). The full list of supported prompt names is:
- `query`, `Retrieval-query`, `Retrieval`, `SearchQuery`, `Reranking`, `BitextMining` → `"task: search result | query: "`
- `document`, `Retrieval-document` → `"title: none | text: "`
- `QuestionAnswering` → `"task: question answering | query: "`
- `FactChecking` → `"task: fact checking | query: "`
- `Classification`, `MultilabelClassification` → `"task: classification | query: "`
- `Clustering` → `"task: clustering | query: "`
- `STS`, `SentenceSimilarity`, `PairClassification`, `Summarization` → `"task: sentence similarity | query: "`
- `CodeRetrieval`, `InstructionRetrieval` → `"task: code retrieval | query: "`

No prefix matches `SentenceTransformer.encode()` without a prompt. EmbeddingGemma 2 input is truncated to 2,048 tokens by default (`EMBEDDINGGEMMA2_MAX_TOKENS`, up to the model's 8,192).

```bash
# EmbeddingGemma 2 (default, 768d; optional task + Matryoshka dimensions)
curl -X POST http://localhost:8080/embed -H "Content-Type: application/json" \
  -d '{"texts": ["waterproof trail shoes"], "task": "query", "dimensions": 256}'
curl -X POST http://localhost:8080/embed -H "Content-Type: application/json" \
  -d '{"texts": ["Gore-Tex lined hiking shoe with a lugged sole."], "task": "document", "dimensions": 256}'

# GTR-T5-base (768d; compatible with /invert)
curl -X POST http://localhost:8080/embed -H "Content-Type: application/json" \
  -d '{"texts": ["The harbor was quiet at dawn."], "model": "gtr-t5-base"}'

# SigLIP text, comparable with /image/embed vectors
curl -X POST http://localhost:8080/embed -H "Content-Type: application/json" \
  -d '{"texts": ["a photo of a sailboat"], "model": "siglip-base-patch16-224"}'
```

Errors are `{"error": "..."}`:

- `400` — unknown `model`; unknown `task`; `dimensions` not in 128/256/512/768; `task` or a non-768 `dimensions` sent with `gtr-t5-base` or `siglip-base-patch16-224`; empty `texts`; more than 256 texts; a blank text; a text over 32 KiB.
- `503` — the requested `model` (`embeddinggemma-2`, `gtr-t5-base` or `siglip-base-patch16-224`) or its tokenizer is not loaded. `/health` shows which are loaded.
- `500` — inference failed. `413` and `408` as in [Request limits](#request-limits).

Vectors from the three models are in different spaces: compare a vector only with vectors from the same model (and, for EmbeddingGemma 2, the same `dimensions`).

### Image & audio embedding

- `POST /image/embed` takes `{"image_base64": "<base64 PNG/JPEG/WebP>"}` and returns `{"embedding": [...768 floats], "dimension": 768, "model": "siglip-base-patch16-224"}`. Returns `503` if `VISION_MODEL_PATH` is not loaded.
- `POST /audio/transcribe` takes `{"audio_base64": "<base64 audio>"}` and returns `{"transcript": "..."}` via `TRANSCRIPTION_URL`.
- `POST /audio/embed` takes `{"audio_base64": "<base64 audio>", "model": "..."}` (`model` optional: `embeddinggemma-2` default, or `gtr-t5-base`) and returns `{"transcript": "...", "embedding": [...768 floats], "dimension": 768, "model": "..."}`.

---

## Request limits

| Routes | Max body | Timeout | Notes |
|--------|----------|---------|-------|
| `/sessions/:id/ingest`, `/temp/:name/ingest`, `/image/embed`, `/audio/*` | 32 MiB | `120s` (`SHIVVR_REQUEST_TIMEOUT_SECS`) | base64 media payloads |
| `/embed` | 8 MiB | `120s` (`SHIVVR_REQUEST_TIMEOUT_SECS`) | up to 256 texts × 32 KiB |
| `/invert` | 2 MiB | `120s` (`SHIVVR_REQUEST_TIMEOUT_SECS`) | autoregressive decode loop |
| everything else (except MCP & agent chat) | 2 MiB | `min(30s, SHIVVR_REQUEST_TIMEOUT_SECS)` | axum default body limit |

Requests over the body limit are rejected with `413 Payload Too Large`. Handlers that exceed the server timeout get `408 Request Timeout` with `{"error": "request exceeded the <N>s server timeout"}`.

---

## Ingest (`POST /sessions/:id/ingest` & `POST /temp/:name/ingest`)

```bash
curl -X POST http://localhost:8080/sessions/my-session/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The harbor was quiet at dawn. Only the sound of halyards against aluminum masts.",
    "source": "journal",
    "emotion_primary": "calm"
  }'
```

Request fields:

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Text to chunk and embed (768d) |
| `model` | string (optional) | `embeddinggemma-2` (default; applies the `document` prefix on ingest and `query` prefix on search automatically) or `gtr-t5-base` (required if you plan to call `/invert` on this session's vectors). Chosen on the first ingest for a session/temp store and locked for subsequent ingests |
| `source` | string (optional) | Source identifier stored on every chunk |
| `metadata` | JSON (optional) | Arbitrary JSON metadata attached to every chunk |
| `emotion_primary` | string (optional) | Primary emotion tag (`vedanā`) |
| `emotion_secondary` | string (optional) | Secondary emotion tag |
| `agent_id` | string (optional) | Encrypt chunk vectors with that agent's registered orthogonal key |
| `openai_api_key` | string (optional) | Per-request OpenAI key for generating 1536d `retrieve` vectors |
| `mean_pool` | bool (default `false`) | Mean-pool all chunk embeddings into a single normalized vector and return one chunk |
| `audio_base64` | string (optional) | Base64 audio transcribed via `TRANSCRIPTION_URL` and appended to `text` before chunking |
| `image_base64` | string (optional) | Base64 image embedded with SigLIP vision (768d) and added as a chunk |

Omit `model` on later ingests to use the store's model. Naming a different model than the store holds returns `400`; use a new session for another model. Chunk boundaries and `token_count` come from the GTR tokenizer whichever model embeds the chunks.

The response is `{"chunks_created", "tokens_processed", "time_ms", "model", "chunks"}` (`model` is the store's text model), where each item in `chunks` contains `chunk_id`, `text`, `embedding`, `embedding_retrieve` (when `retrieve` is active), `token_count`, `source`, `emotion_primary` and `emotion_secondary`.

> **Note:** An `image_base64` chunk stores a SigLIP vector, whereas text chunks use the session's text model (`embeddinggemma-2` or `gtr-t5-base`). Use `POST /image/embed` and `POST /embed` (`model=siglip-base-patch16-224`) for direct text-to-image comparison.

---

## Search (`GET /sessions/:id/search` & `GET /temp/:name/search`)

Search automatically uses the embedding model chosen when the session or temp store was created (applying the `query` prefix when the session uses `embeddinggemma-2`).

```bash
# Basic semantic search
curl "http://localhost:8080/sessions/my-session/search?q=morning+at+the+marina&n=5"

# With temporal decay (half-life 24 hours, 30% time weight)
curl "http://localhost:8080/sessions/my-session/search?q=marina&n=5&time_weight=0.3&decay_halflife_hours=24"

# BM25 + vector Reciprocal Rank Fusion
curl "http://localhost:8080/sessions/my-session/search?q=marina&hybrid=true"

# Lexical-only BM25 (skips query embedding)
curl "http://localhost:8080/sessions/my-session/search?q=marina&lexical_only=true"

# retrieve role (requires OPENAI_API_KEY or openai_api_key)
curl "http://localhost:8080/sessions/my-session/search?q=marina&role=retrieve"
```

### Search query parameters

Same for `/sessions/:id/search` and `/temp/:name/search`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `q` | required | Search query text |
| `n` (alias `limit`) | `5` | Number of results to return |
| `role` | `organize` | `organize` (768d local session model: `embeddinggemma-2` or `gtr-t5-base`) or `retrieve` (1536d OpenAI) |
| `model` | — | Optional check: the text model you expect the store to hold. The query is always embedded with the store's model; a different value returns `400` |
| `openai_api_key` | — | Per-request OpenAI key for `retrieve` (sessions only) |
| `time_weight` | `0.0` | Blend semantic score with recency (`0.0`–`1.0`) |
| `decay_halflife_hours` | `168` | Recency decay half-life in hours (default 1 week) |
| `include_nearby` | `false` | Include temporally adjacent chunks in results (`nearby_chunks`) |
| `time_window_minutes` | `30` | Window (±minutes) for nearby chunks |
| `agent_id` | — | Agent ID for encrypted search (rotates query vector with agent's key) |
| `hybrid` | `false` | Fuse dense vector and BM25 lexical rankings (Reciprocal Rank Fusion) |
| `lexical_only` | `false` | BM25 lexical search only, no query embedding |
| `guardrail` | `true` | Reject queries containing blocked guardrail phrases (`403 Forbidden`) |

The response is `{"query", "results", "time_ms"}`, where each result has `chunk_id`, `score`, `text`, `source`, `metadata`, `created_at`, and optional `emotion_primary`, `emotion_secondary`, `nearby_chunks`.

---

## Inversion (`POST /invert`)

`POST /invert` takes `{"embedding": [...768 floats], "max_length": 64, "model": "gtr-t5-base"}` (`model` optional; any value other than `gtr-t5-base` returns `400`, so pass the model your vector came from to catch mistakes) and returns `{"text", "similarity"}`, where `similarity` is the cosine between your vector and the `gtr-t5-base` embedding of `text`. `max_length` (default `64`) bounds the decoded tokens.

> **Important:** The inverter is trained on **`gtr-t5-base` vectors only**. If you want to invert vectors from a session or `/embed`, pass `"model": "gtr-t5-base"` when ingesting or embedding. EmbeddingGemma 2, SigLIP or OpenAI vectors cannot be inverted by `/invert`.

Inversion is **approximate, not lossless**. Embedding is many-to-one — many different sentences map to nearly the same 768d vector — so a single `/invert` call returns text that is *semantically close* to the source but rarely the exact same words. Invert the embedding of `"Rust is a systems programming language"` and you might get back `"the Rust language is used for systems programming"`: same meaning, different tokens.

To get closer, **iterate** (vec2text-style correction loop):

1. Start with the target vector `v` (from `gtr-t5-base`).
2. `/invert` → candidate text `t` and its `similarity` to `v`.
3. If `similarity` is not close enough, re-embed or adjust `t` and invert again.

Each round nudges the wording so its embedding moves toward `v`; cosine similarity climbs and the text stabilizes. The fixed point is text whose embedding reproduces the target vector — text and vector have converged, even if the final wording still differs from the original. (Requires the inverter models — see `INVERTER_*` in [CONFIGURATION.md](CONFIGURATION.md); otherwise `/invert` returns `503`.)

---

## Per-agent encryption (`/agent/:id/*`)

Shivvr supports per-agent orthogonal transformation (`v @ Q` where `Q` is an orthogonal matrix of `dim × dim` floats). Because orthogonal transformations preserve inner products, cosine similarity between two encrypted vectors is identical to cosine similarity between the unencrypted vectors. Keys live in process memory only.

- `POST /agent/:id/register`:
  ```json
  {
    "organize_key": [/* 768 * 768 = 589824 floats, row-major orthogonal */],
    "organize_dim": 768,
    "retrieve_key": [/* optional 1536 * 1536 floats */],
    "retrieve_dim": 1536
  }
  ```
- `POST /agent/:id/encrypt` and `POST /agent/:id/decrypt`:
  ```json
  {
    "embeddings": [[/* 768 floats */]],
    "role": "organize"
  }
  ```

---

## MCP server & GhostAgent chat

- `GET /mcp/sse`: opens an SSE stream and emits an `endpoint` event with `/mcp/message?id=<client_id>`.
- `POST /mcp/message?id=<client_id>`: accepts JSON-RPC 2.0 requests (`initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`).
  - Exposed tools: `search_memory`, `ingest_memory`, `list_sessions`, and (only when `SHIVVR_ENABLE_RUN_COMMAND=true`) `run_command`.
- `POST /sessions/:id/agent/chat`: takes `{"message": "..."}` and streams SSE `agent` events (`Status`, `Thought`, `ToolCall`, `ToolResult`, `Answer`, `Error`) using Anthropic (`ANTHROPIC_API_KEY`) or OpenAI (`OPENAI_API_KEY`).

---

## Auth

If `NUTS_AUTH_JWKS_URL` is set on the backend, shivvr enforces `nuts-auth` verification. Send tokens in the `Authorization` header with the `Bearer ` prefix:

- JWT (RS256): `Authorization: Bearer <jwt>`
- API token (`ahp_`): `Authorization: Bearer ahp_<token>`

When `NUTS_AUTH_JWKS_URL` is set on the backend (`shivvr`), these routes remain open without a token:

- `GET /`, `GET /health`
- `GET /sessions/:id/search` and `GET /temp/:name/search` with `role=organize` (the default)
- `POST /embed`, `POST /image/embed`, `POST /audio/*` (local compute only)
- `POST /sessions/:id/ingest` and `POST /temp/:name/ingest`, **only** when no server `OPENAI_API_KEY` is configured

Everything else behind the gate requires a valid token, including `role=retrieve` search, session metadata (`GET /sessions/:id`) and delete (`DELETE /sessions/:id`), temp store list/dump/delete, `/agent/:id/*` crypto routes, and `POST /invert`. When a caller authenticates on session ingest, the session is bound to that caller's `user_id` and hidden from other callers (`404`).

> **Gateway note:** When requests go through `shivvr-gateway` (such as `https://shivvr.nuts.services`), only `GET /` and `GET /health` are public; the gateway requires `Authorization: Bearer <jwt or ahp_token>` before forwarding any other route to the GPU backend.

Leave `NUTS_AUTH_JWKS_URL` unset for unauthenticated local dev mode.
