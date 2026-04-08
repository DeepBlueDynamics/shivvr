#!/bin/bash
set -e

PROJECT_ID="gnosis-459403"
REGION="us-central1"
SERVICE="shivvr"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE}"
MODELS_IMAGE="gcr.io/${PROJECT_ID}/shivvr-models"

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
  --tag "${IMAGE}:latest" \
  --project "${PROJECT_ID}" \
  --timeout 20m \
  .

echo "==> Deploying ${SERVICE} to Cloud Run (L4 GPU)..."
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}:latest" \
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
  --port 8080

echo ""
echo "==> Service URL:"
gcloud run services describe "${SERVICE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --format="value(status.url)"
