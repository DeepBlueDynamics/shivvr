# shivvr

Semantic memory service. Cuts text into pieces, embeds them, stores them, finds them again.

Rust + ONNX + sled. Runs in Docker on port 8080.

## What it does

- **Ingest** — chunks text by sentence boundaries, embeds each chunk with BGE-small-en-v1.5 (384d), stores in sled
- **Search** — cosine similarity over organize embeddings, returns ranked chunks
- **Dual embedding** — local BGE-small for organize (384d), optional OpenAI ada-002 for retrieve (1536d)
- **Per-agent encryption** — identity-keyed orthogonal matrix encryption on embeddings
- **Vec2text inversion** — reconstruct approximate text from embeddings (T5-based, optional)
- **Emotion tagging** — vedana/feeling-tone metadata on each chunk at ingest time

## Quick start

```
docker compose up -d
```

Builds from source, downloads the BGE-small ONNX model, starts on `:8080`.

## API

### Memory

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Status, model info, session/chunk counts |
| GET | `/memory` | List all sessions |
| POST | `/memory/:session/ingest` | Ingest text (auto-chunks + embeds) |
| GET | `/memory/:session/search?q=...` | Semantic search |
| GET | `/memory/:session/info` | Session metadata |
| DELETE | `/memory/:session` | Delete a session |

### Crypto

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/agent/:id/register` | Register agent encryption keys |
| POST | `/agent/:id/encrypt` | Encrypt embeddings with agent key |
| POST | `/agent/:id/decrypt` | Decrypt embeddings with agent key |

### Inversion

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/invert` | Reconstruct text from embedding vector |

### Ingest example

```bash
curl -X POST http://localhost:8080/memory/my-session/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "The harbor was quiet at dawn. Only the sound of halyards against aluminum masts."}'
```

### Search example

```bash
curl "http://localhost:8080/memory/my-session/search?q=morning+at+the+marina&top_k=5"
```

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Listen port |
| `DATA_PATH` | `/data/shivvr` | sled database directory |
| `MODEL_PATH` | `/models/bge-small-en-v1.5.onnx` | BGE-small ONNX model |
| `TOKENIZER_PATH` | `/models/tokenizer.json` | BGE tokenizer |
| `OPENAI_API_KEY` | — | Enables ada-002 retrieve embeddings |

## Stack

- **Rust** — axum, tokio, sled, ort (ONNX Runtime)
- **Embedding** — BGE-small-en-v1.5 (BAAI), 384 dimensions, L2-normalized
- **Storage** — sled embedded database, persistent Docker volume
- **Inversion** — T5-base hypothesis + corrector trained on MSMARCO (optional)

## License

Proprietary. All rights reserved.
