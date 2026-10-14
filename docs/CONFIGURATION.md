# shivvr — Configuration

shivvr is configured entirely via environment variables. Outside Docker, the binary defaults to loopback (`127.0.0.1:8080`) and looks for models under `./models/`.

## Environment variables (`shivvr` backend)

| Variable | Binary Default | Description |
|----------|----------------|-------------|
| `PORT` | `8080` | HTTP listen port |
| `BIND_ADDR` | `127.0.0.1` | Listen address. All Docker images override this to `0.0.0.0`; set it yourself only to expose a native binary run on the network |
| `EMBEDDINGGEMMA2_MODEL_PATH` | `models/embeddinggemma2-text.onnx` | EmbeddingGemma 2 text path (768d default model for `/embed`, ingest and search) |
| `EMBEDDINGGEMMA2_TOKENIZER_PATH` | `models/embeddinggemma2-tokenizer.json` | Fast-tokenizer JSON for EmbeddingGemma 2 |
| `EMBEDDINGGEMMA2_MAX_TOKENS` | `2048` | Input token cap for EmbeddingGemma 2 (clamped to `1..=8192`). Cost grows faster than linearly: ~5 s per text at 2,048 tokens, and 70–90 s with ~8 GB RSS at 8,192 tokens on CPU |
| `MODEL_PATH` | `models/gtr-t5-base.onnx` | GTR-T5-base ONNX embedder (768d; selectable via `model=gtr-t5-base` and required for `/invert`) |
| `TOKENIZER_PATH` | `models/tokenizer.json` | GTR-T5-base fast tokenizer JSON |
| `VISION_MODEL_PATH` | `models/siglip-vision.onnx` | SigLIP base patch16-224 vision tower (`POST /image/embed` and `image_base64` ingest). Optional |
| `SIGLIP_TEXT_MODEL_PATH` | `models/siglip-text.onnx` | SigLIP base patch16-224 text tower (`POST /embed` with `model=siglip-base-patch16-224`). Optional |
| `SIGLIP_TOKENIZER_PATH` | `models/siglip-tokenizer.json` | Fast-tokenizer JSON for the SigLIP text tower. Optional |
| `INVERTER_PROJECTION_PATH` | `models/inverter/projection.onnx` | Vec2text projection ONNX model (`POST /invert`, GTR-T5 vectors only). Optional |
| `INVERTER_ENCODER_PATH` | `models/inverter/encoder.onnx` | Vec2text T5 encoder ONNX model. Optional |
| `INVERTER_DECODER_PATH` | `models/inverter/decoder.onnx` | Vec2text T5 decoder ONNX model. Optional |
| `INVERTER_TOKENIZER_PATH` | `models/inverter/tokenizer.json` | Vec2text T5 tokenizer JSON. Optional |
| `GUARDRAILS_DIR` | `guardrails` | Directory of phrase CSVs for the FST query guardrail tagger (writes a minimal `offensive.csv` if missing) |
| `TRANSCRIPTION_URL` | `http://localhost:8765` | Whisper/transcription HTTP service for `/audio/*` and `audio_base64` ingest |
| `OPENAI_API_KEY` | — | Enables the `retrieve` role (1536d OpenAI embeddings) and OpenAI GhostAgent chat |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-ada-002` | Override the OpenAI embedding model used by the `retrieve` role |
| `ANTHROPIC_API_KEY` | — | Enables Anthropic LLM for GhostAgent chat (`/sessions/:id/agent/chat`; preferred over OpenAI when set) |
| `ANTHROPIC_MODEL` | `claude-3-5-haiku-20241022` | Anthropic model for GhostAgent chat |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model for GhostAgent chat |
| `OPENAI_API_URL` | `https://api.openai.com/v1/chat/completions` | Chat completions endpoint for GhostAgent (OpenAI-compatible) |
| `NUTS_AUTH_JWKS_URL` | — | Enables `nuts-auth` RS256 JWT + `ahp_` token verification (unauthenticated dev mode if unset) |
| `NUTS_AUTH_VALIDATE_URL` | `https://auth.nuts.services/api/validate` | Remote endpoint used to validate `ahp_` API tokens |
| `SHIVVR_REQUEST_TIMEOUT_SECS` | `120` | Server-side timeout (seconds) for heavy inference routes (`/embed`, `/image/embed`, `/audio/*`, ingest, `/invert`); light routes use `min(30, this)` |
| `SHIVVR_ENABLE_RUN_COMMAND` | unset (`false`) | **Danger.** Set to `true` or `1` to expose the MCP `run_command` tool (executes arbitrary shell inside the container). Leave unset anywhere `/mcp/*` is reachable by untrusted clients |
| `LANDING_ONLY` | unset (`false`) | When `true`, skips all model/store init and serves only the static homepage (`GET /`) and stub `GET /health` |

## Environment variables (`shivvr-gateway`)

The `gateway` binary (`src/bin/gateway.rs`, built by `Dockerfile.gateway`) is a lightweight front door for Cloud Run that verifies `nuts-auth` tokens before waking the GPU backend:

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_URL` | **required** | HTTPS origin of the backend `shivvr` Cloud Run service (e.g. `https://shivvr-<hash>-uc.a.run.app`) |
| `NUTS_AUTH_JWKS_URL` | **required** | JWKS URL for RS256 JWT verification (`https://auth.nuts.services/.well-known/jwks.json`) |
| `NUTS_AUTH_VALIDATE_URL` | `https://auth.nuts.services/api/validate` | Endpoint for `ahp_` token validation |
| `PORT` | `8080` | HTTP listen port |
| `BIND_ADDR` | `127.0.0.1` | Listen address (`Dockerfile.gateway` sets `0.0.0.0`) |

---

## Defaults overridden by Docker images

Both `deepbluedynamics/shivvr` (CUDA, `Dockerfile`) and `deepbluedynamics/shivvr:latest-cpu` (CPU, `Dockerfile.cpu`) set `BIND_ADDR=0.0.0.0`, `PORT=8080`, and point all model paths to `/models/*`:

```env
PORT=8080
BIND_ADDR=0.0.0.0
MODEL_PATH=/models/gtr-t5-base.onnx
TOKENIZER_PATH=/models/tokenizer.json
VISION_MODEL_PATH=/models/siglip-vision.onnx
SIGLIP_TEXT_MODEL_PATH=/models/siglip-text.onnx
SIGLIP_TOKENIZER_PATH=/models/siglip-tokenizer.json
EMBEDDINGGEMMA2_MODEL_PATH=/models/embeddinggemma2-text.onnx
EMBEDDINGGEMMA2_TOKENIZER_PATH=/models/embeddinggemma2-tokenizer.json
INVERTER_PROJECTION_PATH=/models/inverter/projection.onnx
INVERTER_ENCODER_PATH=/models/inverter/encoder.onnx
INVERTER_DECODER_PATH=/models/inverter/decoder.onnx
INVERTER_TOKENIZER_PATH=/models/inverter/tokenizer.json
```

Image-specific differences:

| Setting | CUDA Image (`Dockerfile`) | CPU Image (`Dockerfile.cpu`) | HaLOS Pi Compose (`deploy/halos/shivvr`) |
|---------|---------------------------|------------------------------|------------------------------------------|
| `LD_LIBRARY_PATH` | `/usr/local/cuda-12.6/compat:/usr/lib/onnxruntime` | `/usr/lib/onnxruntime` | `/usr/lib/onnxruntime` |
| `TRANSCRIPTION_URL` | `http://hyperia-transcription:8765` | `http://localhost:8765` (binary default) | `http://localhost:8765` |
| `GUARDRAILS_DIR` | `guardrails` (binary default) | `/data/guardrails` | `/data/guardrails` (mounted volume) |
| `EMBEDDINGGEMMA2_MAX_TOKENS` | `2048` (binary default) | `2048` (binary default) | `512` (set in compose) |
| Models included | All fp32 models baked into `/models` | None baked in (`VOLUME ["/models"]`; use `shivvr-models:int8` or bind-mount) | `shivvr-models:int8` (EmbeddingGemma 2 + GTR-T5 in `int8`) |
