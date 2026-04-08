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

## Prerequisites

### NVIDIA Driver

The Docker image uses CUDA 12.6. Your host NVIDIA driver must be **545+** to support this.

**Linux (Ubuntu/Debian):**
```bash
# Check current driver
nvidia-smi

# Install/upgrade driver
sudo apt-get update
sudo apt-get install -y nvidia-driver-590

# Reboot required after install
sudo reboot
```

**Windows:**
- Download the latest Game Ready or Studio driver from https://www.nvidia.com/Download/index.aspx
- Run the installer, reboot when prompted
- Verify with `nvidia-smi` in PowerShell

### Docker + NVIDIA Container Toolkit

**Linux:**
```bash
# Install Docker (if not already installed)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in for group change

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Windows:**
- Install Docker Desktop from https://www.docker.com/products/docker-desktop/
- Docker Desktop includes GPU support automatically when WSL 2 backend is enabled
- Ensure WSL 2 is installed: `wsl --install` in PowerShell (admin)

### Export ONNX Models

Models are not included in the repo. Export them before building:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install torch transformers==4.44.2 sentence-transformers vec2text onnx onnxruntime onnxscript
python scripts/export_gtr_models.py --output_dir models/ --verify
```

This produces ~280MB of model files in `models/`.

## Quick start

```
docker compose up -d
```

Builds from source with CUDA support, starts on `:8080`.

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
