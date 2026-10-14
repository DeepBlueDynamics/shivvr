#!/bin/bash
set -e

PROJECT_ID="gnosis-459403"
REGION="us-central1"
SERVICE="shivvr"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE}"
MODELS_IMAGE="gcr.io/${PROJECT_ID}/shivvr-models"
GATEWAY_SERVICE="shivvr-landing"
GATEWAY_IMAGE="gcr.io/${PROJECT_ID}/shivvr-gateway"
HUB_IMAGE="docker.io/deepbluedynamics/shivvr"
HUB_GATEWAY="docker.io/deepbluedynamics/shivvr-gateway"

# Version comes from Cargo.toml, the single source of truth (see docs/OPERATIONS.md -> Versioning).
# Images are pushed as :v<version> and :latest so a revision can always be traced to a tag.
VERSION="$(sed -nE 's/^version *= *"([^"]+)".*/\1/p' Cargo.toml | head -1)"
TAG="v${VERSION}"
echo "==> Releasing ${SERVICE} ${TAG}"

# Check the tools before the 25-minute build, not after it. On Windows, plain
# `bash` from PowerShell is WSL, which usually has no docker; use Git Bash.
NEEDED="gcloud"
[[ "${SKIP_PUBLISH:-}" == "1" ]] || NEEDED="gcloud docker"
for tool in ${NEEDED}; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "==> ${tool} not found in this shell. On Windows, run: & \"C:\\Program Files\\Git\\bin\\bash.exe\" deploy.sh" >&2
    exit 1
  fi
done

# Rebuild models image only when needed (export script or pip deps changed).
# Normal code deploys skip this — models are already baked into shivvr-models:latest.
if [[ "$1" == "--rebuild-models" ]]; then
  echo "==> Rebuilding model image (slow — downloads PyTorch + HuggingFace)..."
  gcloud builds submit \
    --config cloudbuild-models.yaml \
    --project "${PROJECT_ID}" \
    --timeout 60m \
    .
  echo "==> Models image updated: ${MODELS_IMAGE}:latest"
fi

echo "==> Building and pushing app image via Cloud Build..."
gcloud builds submit \
  --tag "${IMAGE}:${TAG}" \
  --project "${PROJECT_ID}" \
  --timeout 40m \
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

# Publish the rest of the release so one tag + deploy.sh covers all four images
# (the tag's CI publishes the gateway and the -cpu images). Needs `docker login`
# to Docker Hub as deepbluedynamics and `gcloud auth configure-docker`.
# SKIP_PUBLISH=1 stops here.
if [[ "${SKIP_PUBLISH:-}" == "1" ]]; then
  echo "==> SKIP_PUBLISH=1: not copying images to Docker Hub or deploying the gateway."
  exit 0
fi

echo "==> Copying the CUDA image to Docker Hub as ${HUB_IMAGE}:${VERSION}, ${TAG} and latest (registry to registry, ~10 min for 4.4 GB)..."
docker buildx imagetools create \
  --tag "${HUB_IMAGE}:${VERSION}" \
  --tag "${HUB_IMAGE}:${TAG}" \
  --tag "${HUB_IMAGE}:latest" \
  "${IMAGE}:${TAG}"

echo "==> Copying gateway ${HUB_GATEWAY}:${VERSION} (published by the tag's CI) to ${GATEWAY_IMAGE}:${TAG}..."
if ! docker buildx imagetools inspect "${HUB_GATEWAY}:${VERSION}" >/dev/null 2>&1; then
  echo "    ${HUB_GATEWAY}:${VERSION} is not on Docker Hub yet; wait for the tag's release workflow, then copy and deploy it with:" >&2
  echo "    docker buildx imagetools create --tag ${GATEWAY_IMAGE}:${TAG} ${HUB_GATEWAY}:${VERSION}" >&2
  echo "    gcloud run deploy ${GATEWAY_SERVICE} --image ${GATEWAY_IMAGE}:${TAG} --region ${REGION} --project ${PROJECT_ID}" >&2
  exit 1
fi
docker buildx imagetools create --tag "${GATEWAY_IMAGE}:${TAG}" "${HUB_GATEWAY}:${VERSION}"

echo "==> Deploying gateway ${TAG} to Cloud Run service ${GATEWAY_SERVICE} (shivvr.nuts.services)..."
gcloud run deploy "${GATEWAY_SERVICE}" \
  --image "${GATEWAY_IMAGE}:${TAG}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}"

echo "==> Release ${TAG} published: Cloud Run ${SERVICE} + ${GATEWAY_SERVICE}, Docker Hub ${HUB_IMAGE}:{${VERSION},${TAG},latest}."
