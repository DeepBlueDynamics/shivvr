# shivvr — Operations Manual

## Service

- **Production URL**: https://shivvr.nuts.services (mapped via Cloud Run domain)
- **Cloud Run URL**: https://shivvr-949870462453.us-central1.run.app
- **GCP Project**: gnosis-459403
- **Region**: us-central1
- **Service name**: shivvr
- **GPU**: NVIDIA L4 (1x), CUDA 12.6
- **Concurrency**: 4 (serialized GPU inference)
- **Instances**: 0 min, 1 max (scales to zero)

---

## Deploy

Full build + deploy from source (runs in Cloud Build — no local Docker or models needed):

```bash
cd shivvr
bash deploy.sh
```

What it does:
1. `gcloud builds submit` — uploads source, builds Docker image in GCP (downloads PyTorch + HuggingFace models, compiles Rust with CUDA), pushes to `gcr.io/gnosis-459403/shivvr:latest`
2. `gcloud run deploy` — deploys new revision, routes 100% traffic

Build time: ~10–15 min first run (model download + Rust compile), ~8 min with partial cache.

---

## Redeploy existing image (no code change)

```bash
gcloud run deploy shivvr \
  --image gcr.io/gnosis-459403/shivvr:latest \
  --region us-central1 \
  --project gnosis-459403 \
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
  --port 8080
```

---

## Check service status

```bash
# Health endpoint
curl https://shivvr.nuts.services/health

# Cloud Run revision list
gcloud run revisions list --service shivvr --region us-central1 --project gnosis-459403

# Current service URL
gcloud run services describe shivvr --region us-central1 --project gnosis-459403 --format="value(status.url)"

# Logs (last 100 lines)
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=shivvr" \
  --project gnosis-459403 --limit 100 --format "value(textPayload)"
```

---

## Rollback

```bash
# List revisions
gcloud run revisions list --service shivvr --region us-central1 --project gnosis-459403

# Route traffic to a previous revision
gcloud run services update-traffic shivvr \
  --region us-central1 \
  --project gnosis-459403 \
  --to-revisions shivvr-00001-xxx=100
```

---

## Domain mapping

Domain `shivvr.nuts.services` is mapped to the Cloud Run service. To update or verify:

```bash
gcloud run domain-mappings list --region us-central1 --project gnosis-459403

gcloud run domain-mappings create \
  --service shivvr \
  --domain shivvr.nuts.services \
  --region us-central1 \
  --project gnosis-459403
```

DNS: CNAME `shivvr.nuts.services` → `ghs.googlehosted.com.`  
SSL is automatic (provisioned by GCP, takes 15–30 min on first mapping).

---

## Environment variables

Set at deploy time via `--set-env-vars` or the Cloud Run console. Current required vars:

| Variable | Value | Notes |
|----------|-------|-------|
| `PORT` | `8080` | Set by Cloud Run automatically |
| `MODEL_PATH` | `/models/gtr-t5-base.onnx` | Baked into image |
| `TOKENIZER_PATH` | `/models/tokenizer.json` | Baked into image |
| `LD_LIBRARY_PATH` | `/usr/local/cuda-12.6/compat:/usr/lib/onnxruntime` | Set in Dockerfile |

Optional vars (set in Cloud Run console or via `--set-env-vars`):

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Enables `retrieve` role (text-embedding-3-small) |
| `OPENAI_EMBEDDING_MODEL` | Override OpenAI model (default: text-embedding-3-small) |
| `NUTS_AUTH_JWKS_URL` | Enable nuts-auth JWT verification |
| `NUTS_AUTH_VALIDATE_URL` | API token validation endpoint |

Set a secret env var:
```bash
gcloud run services update shivvr \
  --region us-central1 \
  --project gnosis-459403 \
  --set-env-vars OPENAI_API_KEY=sk-...
```

---

## Architecture notes

- **No persistence**: all embeddings live in process memory. Restart = clean slate.
- **GPU inference is serialized**: concurrency is capped at 4. Raising it risks OOM on the L4.
- **Scale to zero**: cold start takes ~30–60s (ONNX model load). Expected for this use case.
- **Models baked into image**: GTR-T5-base ONNX exported during `docker build` via `scripts/export_gtr_models.py`. No runtime download needed.
- **Inverter disabled**: vec2text inverter models not included in current build. `/invert` returns 503.
- **Auth**: `NUTS_AUTH_JWKS_URL` not set in current deployment → open dev mode (no token required).

---

## Cloud Build history

```bash
gcloud builds list --project gnosis-459403 --limit 10
```

View a specific build:
```bash
gcloud builds log BUILD_ID --project gnosis-459403
```

---

## Costs (approximate)

- Cloud Run GPU (L4): ~$0.85/hr when running, $0 when scaled to zero
- Cloud Build: ~$0.003/build-minute (free tier: 120 min/day)
- GCR storage: ~$0.02/GB/month for the image (~3 GB)
