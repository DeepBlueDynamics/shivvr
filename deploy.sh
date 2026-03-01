#!/bin/bash
set -e

PROJECT_ID="gnosis-459403"
REGION="us-central1"
SERVICE="shivvr"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE}"

echo "Building ${IMAGE}..."
gcloud builds submit --tag "${IMAGE}" --project "${PROJECT_ID}" --timeout=1800

echo "Deploying ${SERVICE} to Cloud Run with L4 GPU..."
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 8Gi \
  --cpu 4 \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --max-instances 1 \
  --min-instances 0 \
  --concurrency 80 \
  --execution-environment gen2 \
  --port 8080 \
  --set-env-vars "DATA_PATH=/data/shivvr"

echo "Deployed: https://${SERVICE}-$(gcloud run services describe ${SERVICE} --region ${REGION} --project ${PROJECT_ID} --format 'value(status.url)' 2>/dev/null | sed 's|https://||')"
