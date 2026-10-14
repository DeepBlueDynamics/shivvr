# Shivvr token gateway and sleeping GPU backend

## Verified live layout (2026-09-16)

- `shivvr-landing` is the public Cloud Run service serving `shivvr.nuts.services`. It uses `LANDING_ONLY` and the GPU image despite doing no inference; its configured resources are 1 CPU and 512 MiB with no GPU.
- `shivvr` is the existing GPU service. Its ingress is `internal`, its service IAM grants `allUsers` invoker, its maximum is 1 instance, and its minimum is unset (Cloud Run default 0). It has nuts-auth env var names configured. The service uses image `gcr.io/gnosis-459403/shivvr:latest` and a 300-second timeout.
- Both currently run as the project default Compute service account. The backend holds embeddings in memory, so scaling to zero discards sessions.

## Architecture

1. Build the small CPU `gateway` image from `Dockerfile.gateway` using `cloudbuild-gateway.yaml`. It serves the existing landing page and public `/health` without starting GPU resources.
2. Set `shivvr-landing` to use that image, with `BACKEND_URL` set to the canonical `shivvr` run.app URL and nuts-auth JWKS/validation URLs. The gateway requires `Authorization: Bearer <JWT or ahp_ token>` for every API route. Missing or invalid tokens return 401 without contacting `shivvr`.
3. Reuse the existing `shivvr` GPU service as backend. Grant a dedicated gateway service account only `roles/run.invoker` on `shivvr`. Remove `allUsers` invoker from `shivvr`, then change its ingress to `all` so the gateway can call its run.app URL. IAM still prevents direct public invocation. Google documents that Cloud Run-to-Cloud Run calls to an `internal` ingress service need VPC routing, which is not configured in this repository.
4. The gateway obtains a Google ID token from the Cloud Run metadata server and forwards it as `X-Serverless-Authorization`; the caller's nuts-auth token remains in `Authorization`. The backend keeps its own token gate. With minimum instances 0, the first authorized request starts the GPU instance, which scales down after idle time.

## Rollout after user confirmation

1. Record current revisions, image, service accounts, IAM, domain mapping, scaling, and env var names for rollback. Confirm the backend has a real GPU allocation and automatic scaling.
2. Create a dedicated `shivvr-gateway` service account if absent; build a uniquely tagged gateway image. Grant that account invoker on `shivvr`.
3. Remove `allUsers` invoker from `shivvr` before setting ingress to `all`. Keep min instances 0 and max instances 1. Do not rebuild the GPU image.
4. Deploy the gateway image to `shivvr-landing` with 1 CPU, 512 MiB, concurrency 8, a 300-second timeout, the dedicated service account, `BACKEND_URL`, nuts-auth URLs, and no `LANDING_ONLY` variable. Preserve the custom domain mapping and keep 100% traffic on the new revision only after ready.
5. Verify public `/health` reports gateway mode; unauthenticated API returns 401; direct backend without Google IAM returns 403; a real valid nuts-auth token reaches the backend. Confirm min instances 0 and observe startup latency/cost. Roll back traffic and ingress/IAM if verification fails.

## Limits

A real valid token is needed for the final authenticated smoke test. The backend stores embeddings in process memory; a scale-to-zero event loses them. Persistent memory needs separate storage work.

## Approval boundary

Do not run Cloud Build, change GCP IAM or service settings, or shift live traffic until the user confirms this rollout.
