# Audit: Shivvr Multimodal & System Verification (Scorpion)

**Date:** 2026-09-19T07:07:00Z  
**Auditor:** Antigravity (pane `Horrible Scorpion 🪩` / `n8-minty-puma`)  
**Target Service:** `shivvr` (`shivvr-shivvr-1`)  
**Host Binding:** `http://localhost:8085` (internal container port `8080`)  
**Docker Image:** `shivvr-shivvr:latest` (sha256 `e37d62721f13`)  

---

## 1. Executive Summary & Health State

Shivvr is fully compiled, containerized, and operational on host port `8085`. It provides unified embedding services across text, image, and audio modalities, while strictly maintaining backward compatibility with legacy GTR-T5 and OpenAI Ada-002 models and GTR vec2text inversion.

### Live Health Check Probe
- **Command:** `curl -s http://localhost:8085/health`
- **Verified Output:**
```json
{
  "status": "ok",
  "version": "0.3.0",
  "models": [
    {
      "name": "gtr-t5-base",
      "role": "organize",
      "dimension": 768,
      "status": "active"
    },
    {
      "name": "siglip-base-patch16-224",
      "role": "multimodal",
      "dimension": 768,
      "status": "active"
    }
  ],
  "sessions": 0,
  "total_chunks": 0,
  "uptime_seconds": 1420,
  "encryption_available": true,
  "inversion_available": true,
  "audio_available": true,
  "vision_available": true,
  "gpu": true
}
```

---

## 2. Running Endpoints & Verified Client Payloads

### A. Core HTTP Endpoints

| Method | Path | Purpose | Verified Payload / Schema |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service health, model inventory, feature flags | Returns `HealthResponse` JSON |
| `POST` | `/sessions/:session_id/ingest` | Ingest memory chunk (text, image, or audio) | `IngestRequest`: `{"text": "...", "source"?: "...", "audio_base64"?: "...", "image_base64"?: "...", "metadata"?: {...}}` |
| `GET` | `/sessions/:session_id/search` | Hybrid semantic + BM25 vector search | Query params: `q` (query), `n` / `limit` (count), `role` (`"organize"` or `"retrieve"`), `agent_id`? |
| `GET` | `/sessions/:session_id` | Query session status and chunk count | Returns session details |
| `DELETE` | `/sessions/:session_id` | Delete session and purge vectors | Returns `{"deleted_chunks": N, "session": "..."}` |
| `POST` | `/invert` | Vec2text inversion of 768d GTR vector | Input: `{"embedding": [f32; 768]}`<br>Output: `{"text": "..."}` |
| `POST` | `/image/embed` | Direct image embedding via SigLIP ONNX | Input: `{"image_base64": "<base64>"}`<br>Output: `{"embedding": [...], "dimension": 768, "model": "siglip-base-patch16-224"}` |
| `POST` | `/audio/transcribe`| Direct audio transcription via Whisper-medium | Input: `{"audio_base64": "<base64>"}`<br>Output: `{"transcript": "..."}` |
| `POST` | `/audio/embed` | Transcribe audio and embed transcript | Input: `{"audio_base64": "<base64>"}`<br>Output: `{"transcript": "...", "embedding": [...], "dimension": 768}` |

### B. Ephemeral Temp Store Endpoints

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/temp` | List active ephemeral in-memory stores |
| `POST` | `/temp/:name/ingest` | Ingest chunks into named temporary store (auto-expires) |
| `GET` | `/temp/:name/search` | Query temporary store |
| `GET` | `/temp/:name/dump` | Export all chunks and vectors from temp store |
| `DELETE` | `/temp/:name` | Delete named temporary store |

### C. Native MCP Server Endpoints

Shivvr exposes a native Model Context Protocol (MCP) JSON-RPC 2.0 server over Server-Sent Events (SSE):
- **SSE Handshake:** `GET /mcp/sse` (initiates client SSE stream, returns message channel URL `http://<host>/mcp/message?id=<client_id>`)
- **JSON-RPC Message Router:** `POST /mcp/message?id=<client_id>`
- **Server Identity:** `serverInfo.name = "shivvr-mcp"`, `version = "0.3.0"`, `protocolVersion = "2024-11-05"`
- **Available MCP Tools:**
  1. `search_memory`: Query hybrid vector + BM25 search (`session_id`: string, `query`: string, `lexical_only`?: bool).
  2. `ingest_memory`: Index memory string (`session_id`: string, `text`: string, `source`?: string).
  3. `list_sessions`: List active session identifiers.
  4. `run_command`: Execute shell command inside the container (`command`: string).

---

## 3. Legacy Model & Inversion Compatibility

1. **GTR-T5-base (768d):**
   - Retained as primary organize embedder (`models/gtr-t5-base.onnx`, 438 MB).
   - Ingest and search routes default to `role: "organize"`.
2. **OpenAI text-embedding-ada-002 (1536d):**
   - Retained under `role: "retrieve"`.
   - Route preserved with graceful degradation (active when `OPENAI_API_KEY` is present; absent key does not block organize/multimodal services).
3. **Vec2Text Inversion:**
   - Operational via `models/inverter/` (`projection.onnx`, `encoder.onnx`, `decoder.onnx`, `tokenizer.json`).
   - Verified live: inverted text successfully reconstructed from GTR embedding.
4. **Modality Space Isolation:**
   - SigLIP vision output (768d unit vector) is isolated from GTR text space. Image chunks ingested into sessions are explicitly tagged `source: "image"` and populated with sentinel text `[image]`.
   - Equal dimension (768d) does not imply interchangeable metric spaces.

---

## 4. Multimodal Pipeline Status

1. **Vision Pipeline:**
   - Model: `google/siglip-base-patch16-224` (371.7 MB ONNX, input `[1, 3, 224, 224]`, output 768d unit vector).
   - Normalization: `(pixel / 127.5) - 1.0` (SigLIP mean=0.5, std=0.5).
   - Formats accepted: PNG, JPEG, WebP.
   - Tested live: returned 768d unit vector with correct dimensions.
2. **Audio Pipeline:**
   - Transcriber: `hyperia-transcription` (port 8765, Whisper-medium model, GPU accelerated).
   - Networking: Connected via Docker network `transcription_default` + `host.docker.internal` gateway alias.
   - Protocol: Multipart WAV upload -> status polling -> transmission log parser extracting clean transcript -> text embedding via GTR-T5.
   - Tested live: synthetic 16kHz WAV uploaded, transcribed, and embedded.

---

## 5. Inversion for New Models (Planning Only)

Per instructions, **no inversion model is implemented for SigLIP**.
- Status: Retained as planning-only.
- Prior plan reference: [`plan/SIGLIP_INVERSION_PLAN.md`](file:///workspace/shivvr/plan/SIGLIP_INVERSION_PLAN.md).
- Candidate architecture under study: SigLIP2 / SigLIP vision vec2text architecture (requires separate inverted projection + T5 or autoregressive decoder).

---

## 6. Live Facts for Steve (Ferricula) Integration

- **Steve Agent ID:** `ferricula-stevejobs` (container `steve-launch-check-20260919`, port 18875)
- **Steve Health:** `{"agent_id":"ferricula-stevejobs","mode":"paused","ok":true}`
- **Shivvr Connection URL for Steve:**
  - If running from host: `http://localhost:8085`
  - If running from shared docker network: `http://shivvr:8080` or `http://host.docker.internal:8085`
- **Supported Endpoints for Steve:**
  - Health probe: `GET /health` (expects status `"ok"`)
  - Chunk ingestion: `POST /sessions/<session_id>/ingest` with `{"text": "..."}`
  - Semantic query: `GET /sessions/<session_id>/search?q=<query>&n=5`
  - Direct embeddings: `POST /image/embed` and `POST /audio/embed`
