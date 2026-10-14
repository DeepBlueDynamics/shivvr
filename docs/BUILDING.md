# shivvr — Building & Exporting Models

If you only want to run shivvr, use the prebuilt Docker Hub images in [README.md](../README.md#quick-start). Read on if you are building the Rust binary from source, exporting or quantizing ONNX models locally, or building the Docker images yourself.

---

## 1. Host GPU setup (for CUDA builds & containers)

### NVIDIA Driver

The CUDA Docker image (`Dockerfile`) uses CUDA 12.6 (`nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` with `cuda-compat-12-6`). The host driver must be **545+**.

```bash
# Check installed driver
nvidia-smi

# Ubuntu/Debian upgrade
sudo apt-get install -y nvidia-driver-590 && sudo reboot
```

Windows: install the Game Ready or Studio driver from nvidia.com and reboot. Docker Desktop with the WSL 2 backend passes the GPU through automatically.

### Docker + NVIDIA Container Toolkit (Linux)

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

---

## 2. Exporting ONNX models locally

Models are gitignored (`models/` and `models-int8/`). Only `gtr-t5-base.onnx` and `tokenizer.json` are required at startup; missing inverter, SigLIP, or EmbeddingGemma 2 files log a line at boot and disable only that route/model (`503`).

### GTR-T5-base + vec2text inverter + SigLIP (Python 3.11)

`scripts/fetch_models.sh` exports GTR-T5-base (768d) and the `vec2text` GTR inverter (`projection.onnx`, `encoder.onnx`, `decoder.onnx`, `tokenizer.json`) via `scripts/export_gtr_models.py`. `scripts/export_siglip.py` exports the SigLIP base patch16-224 vision and text towers plus `siglip-tokenizer.json`:

```bash
python3 -m venv .venv-gtr && source .venv-gtr/bin/activate
pip install "torch==2.5.1" "transformers==4.44.2" "sentence-transformers==3.2.1" \
  vec2text "onnx==1.16.2" "onnxruntime==1.19.2" "sentencepiece>=0.2.0" "protobuf>=3.20,<5"

bash scripts/fetch_models.sh                            # GTR-T5-base + vec2text inverter
python scripts/export_siglip.py --output_dir models/    # SigLIP vision + text + siglip-tokenizer.json
```

`export_siglip.py` builds the fast tokenizer JSON straight from `spiece.model` (`transformers` has no converter for `SiglipTokenizer`) and verifies it against the slow tokenizer before writing it, so a mismatch fails the export instead of shipping wrong vectors.

### EmbeddingGemma 2 text path (Python 3.11 / 3.12, separate venv)

EmbeddingGemma 2 requires `transformers >= 5.18` and `sentence-transformers >= 6.1`, which conflict with `vec2text`'s pins above (`Dockerfile.models` runs it in its own Python 3.12 stage). Use a separate virtualenv:

```bash
python3 -m venv .venv-gemma && source .venv-gemma/bin/activate
pip install "torch>=2.9" torchvision "transformers>=5.18" "sentence-transformers>=6.1.0" \
  "pillow>=12.0" "onnx>=1.18" onnxscript "onnxruntime>=1.22"

python scripts/export_embeddinggemma2.py --output_dir models/
```

The script exports the text backbone only (`models/embeddinggemma2-text.onnx`, 1.09 GB float32, and `models/embeddinggemma2-tokenizer.json`) with masked mean pooling and L2 normalization in the graph. Never export or run EmbeddingGemma 2 in float16 — it produces NaN or degraded vectors. The export script fails unless:
1. The exported `embeddinggemma2-tokenizer.json` reproduces `SentenceTransformer`'s exact token IDs across 18 strict test inputs.
2. The ONNX model output reaches cosine similarity `>= 0.9999` (`--min_cosine`) against `SentenceTransformer.encode()`.

### Quantizing to `int8` for CPU / Raspberry Pi (`models-int8/`)

`scripts/quantize_models.py` turns the fp32 exports in `models/` into the 440 MB `int8` set in `models-int8/` (GTR-T5-base at 110 MB and EmbeddingGemma 2 text at 295 MB, keeping EmbeddingGemma 2's final layer 23 and output projection in fp32 to preserve cosine quality):

```bash
python scripts/quantize_models.py --input_dir models --output_dir models-int8
```

Each quantized model is verified against its fp32 original on sample texts and the script fails if worst cosine drops below `0.98` (`--min_cosine`). It also writes a content hash to `models-int8/.models-version`.

---

## 3. Building the Rust binary from source

Requires **Rust 1.88+**. `ort` (`=2.0.0-rc.11`) automatically downloads prebuilt ONNX Runtime binaries during compilation (`download-binaries` feature). On Linux, the prebuilt ONNX Runtime binary links against `glibc >= 2.38` (Debian trixie / Ubuntu 24.04+).

```bash
# CPU build (default 'ml' feature)
cargo build --release --locked --bin shivvr

# CUDA build (enables ort/cuda execution provider)
cargo build --release --locked --features cuda --bin shivvr

# Lightweight gateway binary (no ONNX Runtime / ML dependencies)
cargo build --release --locked --no-default-features --bin gateway
```

### Windows note (`build.rs`)

On Windows MSVC targets, `ort-sys` links against `DXCORE.lib`, `DirectML.lib`, and `D3D12.lib`, which may be absent in older Windows SDKs. [`build.rs`](../build.rs) automatically locates `lib.exe` via `vswhere.exe` (or generates a minimal valid COFF import archive as a fallback) so CPU and CUDA builds link cleanly on Windows without extra SDK setup.

### Running tests

```bash
# Fast unit tests (embed_api validation, limits, gateway) without loading ONNX Runtime
cargo test --lib --no-default-features
cargo test --no-default-features --bin gateway

# Full unit + integration test suite (synthetic vectors, store, chunker, crypto)
cargo test

# End-to-end functional test suite against a running shivvr instance
python tests/functional.py http://localhost:8080
```

---

## 4. Dockerfiles & Docker Compose

### Docker Compose

[`docker-compose.yml`](../docker-compose.yml) runs `deepbluedynamics/shivvr:latest` with GPU passthrough on host port `127.0.0.1:8085`:

```bash
# Standard GPU run (on 127.0.0.1:8085)
docker compose up -d

# Attach to an external hyperia-transcription network (transcription_default)
docker compose -f docker-compose.yml -f docker-compose.transcription.yml up -d
```

### Building the Docker images

| Dockerfile | Target Image | Build Command |
|------------|--------------|---------------|
| `Dockerfile` | `deepbluedynamics/shivvr:latest` (CUDA + fp32 models) | `docker build --build-arg MODELS_IMAGE=gcr.io/gnosis-459403/shivvr-models:latest -t shivvr .` |
| `Dockerfile.cpu` | `deepbluedynamics/shivvr:latest-cpu` (amd64/arm64 CPU, no models baked in) | `docker buildx build --platform linux/amd64,linux/arm64 -f Dockerfile.cpu -t shivvr-cpu .` |
| `Dockerfile.models` | `gcr.io/gnosis-459403/shivvr-models:latest` (fp32 exporter image, ~3.3 GB of `/models`) | `docker build -f Dockerfile.models -t shivvr-models .` (or `bash deploy.sh --rebuild-models`) |
| `Dockerfile.models-int8` | `deepbluedynamics/shivvr-models:int8` (440 MB multi-arch `int8` volume installer) | `docker buildx build --platform linux/amd64,linux/arm64 -f Dockerfile.models-int8 -t deepbluedynamics/shivvr-models:int8 models-int8` |
| `Dockerfile.gateway` | `deepbluedynamics/shivvr-gateway:latest` (GPU-less front door) | `docker build -f Dockerfile.gateway -t shivvr-gateway .` |

 Note that `Dockerfile.models-int8` takes `models-int8/` (the output directory of `scripts/quantize_models.py`) as its build context, not the repository root.
