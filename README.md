# shivvr

Ephemeral semantic embedding service. Ingests text, chunks it, embeds it with GTR-T5-base (768d), and returns ranked results. No persistence — all state is in-process and lost on restart.

Rust + ONNX Runtime. Runs in Docker on port 8080.

## What it does

- **Ingest** — chunks text by sentence boundaries, embeds each chunk with GTR-T5-base (768d local, always on)
- **Search** — cosine similarity with optional temporal decay weighting
- **Dual embedding** — local GTR-T5-base for `organize` role (768d), optional OpenAI text-embedding-ada-002 for `retrieve` role (1536d)
- **Per-agent encryption** — orthogonal matrix rotation on embeddings; preserves cosine similarity, keys ephemeral
- **Vec2text inversion** — reconstruct approximate text from an embedding vector (optional, requires inverter models)
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

### Export ONNX Models

Models are not in the repo. Run once before building the Docker image:

```bash
bash scripts/fetch_models.sh
```

This checks for Python deps, installs them if needed, exports GTR-T5-base + vec2text to `models/` (~280 MB), and runs a verification pass. Pass `--force` to re-export.

Manual export:
```bash
pip install torch transformers sentence-transformers vec2text onnx onnxruntime
python scripts/export_gtr_models.py --output_dir models/ --verify
```

## Quick start

```bash
docker compose up -d
```

Builds from source with CUDA, starts on `:8080`. No volume needed.

## API

### Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Status, model info, session/chunk counts |
| GET | `/sessions` | List all sessions |
| POST | `/sessions/:id/ingest` | Ingest text (chunk + embed) |
| GET | `/sessions/:id/search?q=...` | Semantic search |
| GET | `/sessions/:id` | Session metadata |
| DELETE | `/sessions/:id` | Delete session |

### Temp store

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/temp` | List named temp stores (with TTL) |
| POST | `/temp/:name/ingest` | Ingest into ephemeral named store (2 hr TTL) |
| GET | `/temp/:name/search?q=...` | Search temp store |
| GET | `/temp/:name` | Dump all chunks |
| DELETE | `/temp/:name` | Delete temp store |

### Crypto

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/agent/:id/register` | Register per-agent orthogonal key |
| POST | `/agent/:id/encrypt` | Encrypt embeddings |
| POST | `/agent/:id/decrypt` | Decrypt embeddings |

### Inversion

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/invert` | Reconstruct approximate text from embedding |

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

### Search

```bash
# Basic
curl "http://localhost:8080/sessions/my-session/search?q=morning+at+the+marina&n=5"

# With temporal decay (half-life 24 hours, 30% time weight)
curl "http://localhost:8080/sessions/my-session/search?q=marina&n=5&time_weight=0.3&decay_halflife_hours=24"

# retrieve role (requires OPENAI_API_KEY)
curl "http://localhost:8080/sessions/my-session/search?q=marina&role=retrieve"
```

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Listen port |
| `MODEL_PATH` | `models/gtr-t5-base.onnx` | GTR-T5-base ONNX embedder |
| `TOKENIZER_PATH` | `models/tokenizer.json` | GTR tokenizer |
| `OPENAI_API_KEY` | — | Enables text-embedding-ada-002 retrieve role |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-ada-002` | Override OpenAI model |
| `NUTS_AUTH_JWKS_URL` | — | Enable auth (open dev mode if unset) |
| `NUTS_AUTH_VALIDATE_URL` | `https://auth.nuts.services/api/validate` | API token validation |
| `INVERTER_PROJECTION_PATH` | `models/inverter/projection.onnx` | Vec2text projection |
| `INVERTER_ENCODER_PATH` | `models/inverter/encoder.onnx` | Vec2text T5 encoder |
| `INVERTER_DECODER_PATH` | `models/inverter/decoder.onnx` | Vec2text T5 decoder |
| `INVERTER_TOKENIZER_PATH` | `models/inverter/tokenizer.json` | Vec2text tokenizer |

## Search query parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `q` | required | Search query text |
| `n` | `5` | Number of results |
| `role` | `organize` | `organize` (768d local) or `retrieve` (1536d OpenAI) |
| `time_weight` | `0.0` | Blend semantic score with recency (0–1) |
| `decay_halflife_hours` | `168` | Recency decay half-life in hours |
| `include_nearby` | — | Include temporally adjacent chunks in results |
| `time_window_minutes` | `30` | Window for nearby chunks |
| `agent_id` | — | Agent ID for encrypted search |
| `max_length` | `64` | Max tokens for inverter output |

## Auth

If `NUTS_AUTH_JWKS_URL` is set, the service enforces nuts-auth:

- **organize role** (local GTR) — always free, no token required
- **retrieve role** (OpenAI) — requires `Authorization: Bearer <jwt>` or `Authorization: ahp_<token>`

Leave `NUTS_AUTH_JWKS_URL` unset for open dev mode.

## Stack

- **Rust** — axum, tokio, ort (ONNX Runtime 2.0)
- **Embedding** — GTR-T5-base (sentence-transformers), 768d, L2-normalized
- **Storage** — ephemeral `RwLock<HashMap>`, no disk persistence
- **Inversion** — vec2text gtr-base (projection + T5 encoder/decoder, optional)
- **Auth** — nuts-auth RS256 JWT + `ahp_` API tokens

## License

Proprietary. All rights reserved.
