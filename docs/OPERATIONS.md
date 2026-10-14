# shivvr — Operations Manual

## Service

- **Production URL**: https://shivvr.nuts.services (mapped via Cloud Run domain to `shivvr-landing`, the GPU-less `shivvr-gateway` front door)
- **Gateway Service**: `shivvr-landing` (1 CPU, 512 MiB, no GPU; verifies `nuts-auth` tokens and proxies authorized requests to the `shivvr` GPU backend)
- **Backend Service**: `shivvr` (`https://shivvr-<PROJECT_NUMBER>.us-central1.run.app`, IAM-restricted to the gateway service account)
- **GCP Project**: `gnosis-459403` (hard-coded in [`deploy.sh`](../deploy.sh) and [`cloudbuild-models.yaml`](../cloudbuild-models.yaml); see [Google Cloud project](#google-cloud-project) to adapt for another project)
- **Region**: `us-central1`
- **GPU**: NVIDIA L4 (1x), CUDA 12.6
- **Concurrency**: 4 on `shivvr` (serialized GPU inference); 8 on `shivvr-landing`
- **Instances**: 0 min, 1 max on `shivvr` (scales to zero when idle)

---

## Google Cloud project

[`deploy.sh`](../deploy.sh) and [`cloudbuild-models.yaml`](../cloudbuild-models.yaml) hard-code `PROJECT_ID="gnosis-459403"`. For ad-hoc `gcloud` commands in your shell, set `PROJECT_ID`:

```bash
export PROJECT_ID="gnosis-459403"   # or $(gcloud config get-value project)
```

Don't know your project ID, or need to pick one?

```bash
gcloud projects list                       # all projects you can access (ID + number)
gcloud config get-value project            # the currently active project ID
gcloud config set project YOUR_PROJECT_ID  # switch the active project
```

Starting from scratch in a new project? Create a project, update `PROJECT_ID` in `deploy.sh` and `cloudbuild-models.yaml`, and enable the APIs this service uses:

```bash
gcloud projects create YOUR_PROJECT_ID --name="shivvr"
gcloud config set project YOUR_PROJECT_ID
# Link billing in the console, then:
gcloud services enable run.googleapis.com cloudbuild.googleapis.com containerregistry.googleapis.com
```

> **ID vs. number:** the project **ID** (e.g. `gnosis-459403`) is the
> human-readable string you choose and is what almost every `gcloud` command
> wants. The project **number** (e.g. `123456789012`) is auto-assigned and shows
> up in the default Cloud Run URL. Look up your number with:
> ```bash
> gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)"
> ```

Once `$PROJECT_ID` is set, you can find the live Cloud Run URLs anytime with:

```bash
gcloud run services describe shivvr-landing --region us-central1 \
  --project "$PROJECT_ID" --format="value(status.url)"
gcloud run services describe shivvr --region us-central1 \
  --project "$PROJECT_ID" --format="value(status.url)"
```

---

## Versioning & releases

- **Single source of truth:** `version` in `Cargo.toml`. `/health`, the homepage, and MCP `serverInfo` report it via `CARGO_PKG_VERSION`.
- **Bump with [`scripts/bump-version.sh`](../scripts/bump-version.sh):** updates `Cargo.toml`, refreshes `Cargo.lock`, and verifies `cargo check --release --locked`.
- **Tag every release** with `v<version>` on the commit you shipped (`git tag -a v0.5.3`). `.github/workflows/release.yml` refuses a tag that does not match `Cargo.toml`.
- **Images per release:**
  - `gcr.io/gnosis-459403/shivvr:v<version>` (+ `:latest`) — the CUDA app image (~4.4 GB compressed), built by `deploy.sh` on Cloud Build. Copies `/models` from `gcr.io/gnosis-459403/shivvr-models:latest` (~3.3 GB of fp32 ONNX files: EmbeddingGemma 2 text at 1.09 GB, GTR-T5-base, SigLIP vision + text, and the vec2text inverter).
  - `deepbluedynamics/shivvr:v<version>` (+ `:<version>`, `:latest`) — the public Docker Hub CUDA image, copied from `gcr.io/gnosis-459403/shivvr:v<version>` with `docker buildx imagetools create` (see below).
  - `ghcr.io/deepbluedynamics/shivvr:<version>-cpu` and `deepbluedynamics/shivvr:<version>-cpu` (+ `:v<version>-cpu`, `:latest-cpu`) — multi-arch (`linux/amd64`, `linux/arm64`) CPU image with no models baked in, built automatically by `.github/workflows/release.yml` (`cpu-build` + `cpu-merge`) on every `v*` tag push.
  - `ghcr.io/deepbluedynamics/shivvr-gateway:<version>` and `deepbluedynamics/shivvr-gateway:<version>` (+ `:v<version>`, `:latest`) — the GPU-less gateway, built automatically by `.github/workflows/release.yml` (`release` job) on every `v*` tag push (`DOCKER_USERNAME` and `DOCKER_TOKEN` repository secrets are configured).
  - `deepbluedynamics/shivvr-models:int8` — multi-arch (`linux/amd64`, `linux/arm64`) 440 MB `int8` model volume installer (EmbeddingGemma 2 text + GTR-T5-base), rebuilt only when quantized models change (`Dockerfile.models-int8`).

Cutting a release:

```bash
# 1. bump the version in Cargo.toml + Cargo.lock
bash scripts/bump-version.sh patch   # or: minor | major | X.Y.Z
git commit -am "release: v0.5.3"
git tag -a v0.5.3 -m "shivvr v0.5.3"
git push origin main v0.5.3          # triggers release.yml (gateway + multi-arch -cpu on GHCR & Docker Hub)

# 2. when the tag's workflow has finished: build + deploy + publish the rest
bash deploy.sh
```

`deploy.sh` then:
1. builds the CUDA app image on Cloud Build and pushes `gcr.io/gnosis-459403/shivvr:v0.5.3` and `:latest`;
2. deploys it to Cloud Run `shivvr`;
3. copies it to Docker Hub as `deepbluedynamics/shivvr:0.5.3`, `v0.5.3` and `latest` (registry to registry,
   about 10 minutes for 4.4 GB);
4. copies the gateway that CI published (`deepbluedynamics/shivvr-gateway:0.5.3`) into GCR;
5. deploys it to Cloud Run `shivvr-landing` (shivvr.nuts.services).

With the tag's CI (gateway and `-cpu` images), that publishes all four images. It needs `docker login` to Docker Hub as
`deepbluedynamics` and `gcloud auth configure-docker`. `SKIP_PUBLISH=1 bash deploy.sh` stops after step 2. If the gateway
isn't on Docker Hub yet, the script stops at step 4 and prints the two commands to finish by hand.

On Windows, run it with Git Bash (`& "C:\Program Files\Git\bin\bash.exe" deploy.sh` from PowerShell). Plain `bash` in
PowerShell starts WSL, which usually has no `docker`; the script checks for `gcloud` and `docker` before building and
stops with a message if either is missing.

---

## Deploy

Full build + deploy from source (runs in Cloud Build — no local Docker or models needed):

```bash
bash deploy.sh                    # app image + Cloud Run deploy
bash deploy.sh --rebuild-models   # rebuild the models image first (only when export scripts change)
```

What it does:
1. (`--rebuild-models` only) `gcloud builds submit --config cloudbuild-models.yaml --timeout 60m` runs `Dockerfile.models`:
   exports the EmbeddingGemma 2 text model (1.09 GB float32) and tokenizer in a Python 3.12 stage, then exports GTR-T5-base, the vec2text inverter, and the SigLIP vision and text towers + tokenizer in a Python 3.11 stage, and pushes `gcr.io/gnosis-459403/shivvr-models:latest`. The EmbeddingGemma 2 stage fails the build if its tokenizer or ONNX output drifts from `SentenceTransformer.encode()` (cosine below 0.9999). `cloudbuild-models.yaml` sets a `3600s` timeout and `deploy.sh` passes `--timeout 60m`.

   What the models image holds under `/models`:

   | File | Model | Served by |
   |------|-------|-----------|
   | `embeddinggemma2-text.onnx`, `embeddinggemma2-tokenizer.json` | EmbeddingGemma 2, text path (768d, float32, 1.09 GB) | Default model for `POST /embed`, session/temp ingest, and search |
   | `gtr-t5-base.onnx`, `tokenizer.json` | GTR-T5-base (768d, float32, 230 MB) | `POST /embed` and ingest/search with `model=gtr-t5-base`; required for `/invert` |
   | `inverter/*.onnx`, `inverter/tokenizer.json` | vec2text gtr-base (~1.0 GB) | `POST /invert` (GTR-T5-base vectors only) |
   | `siglip-vision.onnx` | SigLIP base patch16-224, vision tower | `POST /image/embed` and `image_base64` ingest |
   | `siglip-text.onnx`, `siglip-tokenizer.json` | SigLIP base patch16-224, text tower | `POST /embed` with `model=siglip-base-patch16-224` |

2. `gcloud builds submit --tag gcr.io/gnosis-459403/shivvr:v<version> --timeout 40m` compiles Rust with CUDA and copies `/models` from `shivvr-models:latest`. Cloud Build never uploads the gitignored `models/` directory, which is why the app image copies from the prebuilt models image. About 17 minutes before v0.6.0; the v0.6.0 build ran past 20 minutes (lume is now fetched and compiled from git), so the limit is 40. The image is then also tagged `:latest`.
3. `gcloud run deploy shivvr --image ...:v<version>` creates the new revision with `NUTS_AUTH_JWKS_URL` and `NUTS_AUTH_VALIDATE_URL` set and routes 100% of traffic.

The gateway (`shivvr-landing`, the public `shivvr.nuts.services` front door) is deployed separately using the image built by `.github/workflows/release.yml`:

```bash
gcloud run deploy shivvr-landing \
  --image ghcr.io/deepbluedynamics/shivvr-gateway:v<version> \
  --region us-central1 \
  --project "$PROJECT_ID"
```

---

## Redeploy existing image (no code change)

```bash
gcloud run deploy shivvr \
  --image gcr.io/$PROJECT_ID/shivvr:latest \
  --region us-central1 \
  --project "$PROJECT_ID" \
  --platform managed \
  --allow-unauthenticated \
  --memory 16Gi \
  --cpu 4 \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --max-instances 1 \
  --min-instances 0 \
  --concurrency 4 \
  --execution-environment gen2 \
  --no-gpu-zonal-redundancy \
  --port 8080 \
  --set-env-vars "NUTS_AUTH_JWKS_URL=https://auth.nuts.services/.well-known/jwks.json,NUTS_AUTH_VALIDATE_URL=https://auth.nuts.services/api/validate"
```

---

## Check service status

```bash
# Public gateway health endpoint
curl https://shivvr.nuts.services/health

# Text embedding smoke test (through the gateway, pass a nuts-auth Bearer token)
curl -s -X POST https://shivvr.nuts.services/embed \
  -H "Authorization: Bearer $SHIVVR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["hello"], "model": "embeddinggemma-2", "task": "query", "dimensions": 256}' | head -c 200

# Cloud Run revision list
gcloud run revisions list --service shivvr --region us-central1 --project "$PROJECT_ID"
gcloud run revisions list --service shivvr-landing --region us-central1 --project "$PROJECT_ID"

# Logs (last 100 lines)
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=shivvr" \
  --project "$PROJECT_ID" --limit 100 --format "value(textPayload)"
```

On the `shivvr` backend, `/health` lists each loaded model under `models` (`name`, `role`, `dimension`, `status`). With every model file present:

| `name` | `role` | `dimension` |
|--------|--------|-------------|
| `embeddinggemma-2` | `text` / `organize` | 768 (`/embed` can truncate to 512/256/128) |
| `gtr-t5-base` | `organize` | 768 |
| `text-embedding-ada-002` | `retrieve` | 1536 (only with `OPENAI_API_KEY`) |
| `siglip-base-patch16-224` | `multimodal` | 768 (vision tower) |
| `siglip-base-patch16-224` | `multimodal-text` | 768 (text tower) |

An optional model that failed to load is absent from the list; its startup log line says why.

### Troubleshooting

**`EmbeddingGemma 2 text embedder not available: ...`** in the startup log, and `/embed` with `model=embeddinggemma-2` answers `503`.
The ONNX file or the tokenizer JSON was missing or failed to load.
- Check the paths: `EMBEDDINGGEMMA2_MODEL_PATH` (`/models/embeddinggemma2-text.onnx` in the image, default `models/embeddinggemma2-text.onnx` when unset) and `EMBEDDINGGEMMA2_TOKENIZER_PATH` (`/models/embeddinggemma2-tokenizer.json`).
- Files missing from the image means `shivvr-models:latest` predates EmbeddingGemma 2: run `bash deploy.sh --rebuild-models`. Locally, run `python scripts/export_embeddinggemma2.py --output_dir models/` in its own venv with `transformers>=5.18` and `sentence-transformers>=6.1` (see [BUILDING.md](BUILDING.md)).
- A tokenizer load error usually means the JSON was not written by the export script.
- Never export or run it in float16: the model returns NaN or silently degraded vectors. The export is float32 (or `int8` via `scripts/quantize_models.py` on CPU).

`/embed` answers `400` for an unknown `task`, for `dimensions` other than 128/256/512/768, and for a `task` or a non-native `dimensions` on `gtr-t5-base` or `siglip-base-patch16-224`.

---

## Rollback

```bash
# List revisions
gcloud run revisions list --service shivvr --region us-central1 --project "$PROJECT_ID"

# Route traffic to a previous revision
gcloud run services update-traffic shivvr \
  --region us-central1 \
  --project "$PROJECT_ID" \
  --to-revisions shivvr-00014-kt6=100
```

---

## Domain mapping

Domain `shivvr.nuts.services` is mapped to the `shivvr-landing` gateway service. To update or verify:

```bash
gcloud run domain-mappings list --region us-central1 --project "$PROJECT_ID"

gcloud run domain-mappings create \
  --service shivvr-landing \
  --domain shivvr.nuts.services \
  --region us-central1 \
  --project "$PROJECT_ID"
```

DNS: CNAME `shivvr.nuts.services` → `ghs.googlehosted.com.`  
SSL is automatic (provisioned by GCP, takes 15–30 min on first mapping).

---

## Environment variables

See [CONFIGURATION.md](CONFIGURATION.md) for the complete environment variable reference. In Cloud Run, `deploy.sh` sets `NUTS_AUTH_JWKS_URL` and `NUTS_AUTH_VALIDATE_URL` automatically.

Set an additional secret or env var on the backend service:
```bash
gcloud run services update shivvr \
  --region us-central1 \
  --project "$PROJECT_ID" \
  --update-env-vars OPENAI_API_KEY=sk-...
```

---

## Architecture notes

- **No persistence**: all embeddings live in process memory. Restart or scale-to-zero = clean slate.
- **GPU inference is serialized**: concurrency is capped at 4 on `shivvr`. Raising it risks OOM on the L4.
- **Scale to zero**: cold start takes ~30–60s (ONNX model load). `shivvr-landing` stays lightweight and only wakes `shivvr` when an authenticated API request arrives.
- **Models baked into CUDA image**: `Dockerfile.models` exports EmbeddingGemma 2 (`scripts/export_embeddinggemma2.py`), GTR-T5-base and the vec2text inverter (`scripts/export_gtr_models.py`), and SigLIP (`scripts/export_siglip.py`) into `shivvr-models:latest`, and `Dockerfile` copies `/models` from it. No runtime download needed.
- **Inverter**: `/invert` is live in production (`inversion_available: true` in `/health`) and reconstructs text from 768d `gtr-t5-base` vectors.
- **Auth**: `deploy.sh` configures `NUTS_AUTH_JWKS_URL` and `NUTS_AUTH_VALIDATE_URL` on `shivvr`, and `shivvr-landing` enforces Bearer token authentication (`Authorization: Bearer <jwt or ahp_token>`) on all non-health routes before waking the GPU backend.

---

## Cloud Build history

```bash
gcloud builds list --project "$PROJECT_ID" --limit 10
```

View a specific build:
```bash
gcloud builds log BUILD_ID --project "$PROJECT_ID"
```

---

## Costs (approximate)

- Cloud Run GPU (L4): ~$0.85/hr when running, $0 when scaled to zero
- Cloud Build: ~$0.003/build-minute (free tier: 120 min/day)
- GCR storage: ~$0.02/GB/month for the CUDA image (~4.4 GB compressed)
