#!/bin/bash
set -e

PROJECT_ID="gnosis-459403"
REGION="us-central1"
SERVICE="shivvr"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE}"
MODELS_IMAGE="gcr.io/${PROJECT_ID}/shivvr-models"

# Version comes from Cargo.toml, the single source of truth (see OPERATIONS.md -> Versioning).
# Images are pushed as :v<version> and :latest so a revision can always be traced to a tag.
VERSION="$(sed -nE 's/^version *= *"([^"]+)".*/\1/p' Cargo.toml | head -1)"
TAG="v${VERSION}"
echo "==> Releasing ${SERVICE} ${TAG}"

# Rebuild models image only when needed (export script or pip deps changed).
# Normal code deploys skip this — models are already baked into shivvr-models:latest.
if [[ "$1" == "--rebuild-models" ]]; then
  echo "==> Rebuilding model image (slow — downloads PyTorch + HuggingFace)..."
  gcloud builds submit \
    --config cloudbuild-models.yaml \
    --project "${PROJECT_ID}" \
    --timeout 40m \
    .
  echo "==> Models image updated: ${MODELS_IMAGE}:latest"
fi

echo "==> Building and pushing app image via Cloud Build..."
gcloud builds submit \
  --tag "${IMAGE}:${TAG}" \
  --project "${PROJECT_ID}" \
  --timeout 20m \
  .

gcloud container images add-tag -q "${IMAGE}:${TAG}" "${IMAGE}:latest" --project "${PROJECT_ID}"

echo "==> Deploying ${SERVICE} ${TAG} to Cloud Run (L4 GPU)..."
# NUTS_AUTH_JWKS_URL turns on the auth gate. Without it shivvr boots in
# "dev mode" with all endpoints public — fine locally, dangerous in prod.
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}:${TAG}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
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

echo ""
echo "==> Service URL:"
gcloud run services describe "${SERVICE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --format="value(status.url)"
