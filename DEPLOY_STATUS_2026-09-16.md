# Shivvr gateway deployment — 2026-09-16

## Live revisions

- Public domain `shivvr.nuts.services` maps to `shivvr-landing`.
- `shivvr-landing` revision `shivvr-landing-00002-87l` serves 100% traffic from `gcr.io/gnosis-459403/shivvr-gateway:c6b9e2aa-00e9-4878-8df6-7df250daa383` (Cloud Build `c6b9e2aa-00e9-4878-8df6-7df250daa383`, SUCCESS). It runs as `shivvr-gateway@gnosis-459403.iam.gserviceaccount.com`, concurrency 8, maximum 3 instances, minimum 0/default. Env var names: `BACKEND_URL`, `NUTS_AUTH_JWKS_URL`, `NUTS_AUTH_VALIDATE_URL`; `LANDING_ONLY` is gone.
- Backend `shivvr` revision `shivvr-00014-kt6` serves 100% traffic. It retains one L4 GPU, maximum 1 instance, minimum 0/default, and its prior model image. Ingress is `all`; service IAM grants `roles/run.invoker` only to the gateway service account. The earlier backend revision was `shivvr-00013-2vb`.
- Previous landing revision: `shivvr-landing-00001-t25`.

## Checks performed

- Public `GET https://shivvr.nuts.services/health`: HTTP 200, `mode: gateway`, `backend: on demand`.
- Public unauthenticated `POST /temp/probe/ingest`: HTTP 401 `authentication required`.
- Public `POST /temp/probe/ingest` with an invalid `ahp_` token: HTTP 401 `invalid token`.
- Direct unauthenticated `GET https://shivvr-ugcdy6vw7a-uc.a.run.app/health`: HTTP 403 after IAM propagation.
- Gateway build: Cloud Build SUCCESS. Local `cargo test --locked --no-default-features --bin gateway`: 3 passed. Local `cargo check --locked --bin shivvr`: passed.

## Remaining verification

- A real nuts-auth token was not available. The `nutnews-shivvr-token` Secret Manager secret had no version at the time of verification. A valid-token ingest through the public gateway and the nuts-news integration have not yet been run. The nuts-news agent is arranging a dedicated `ahp_` token version and holding its deployment until then.
- The backend stores embeddings in process memory. Scale-to-zero still discards sessions.
