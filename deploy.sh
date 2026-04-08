#!/bin/bash
set -e

PROJECT_ID="gnosis-459403"
REGION="us-central1"
SERVICE="shivvr"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE}"

echo "Tagging and pushing local image to ${IMAGE}..."
docker tag gnosis-chunk-shivvr "${IMAGE}:latest"
docker push "${IMAGE}:latest"

# concurrency: GPU inference serializes — benchmark before raising above 4
echo "Deploying ${SERVICE} to Cloud Run with L4 GPU..."
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
  --port 8080

echo "Service URL:"
gcloud run services describe "${SERVICE}" --region "${REGION}" --project "${PROJECT_ID}" --format="value(status.url)"
