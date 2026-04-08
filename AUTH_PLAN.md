# Shivvr × Nuts-Auth Integration Plan

## Summary

Replace the current single `API_TOKEN` env var with nuts-auth JWT/API-token verification.
The guiding rule: **operations that are purely local (GTR-T5-base) are free. Operations that
require an external paid token (OpenAI ada-002) require a nuts-auth token from the caller.**

---

## Auth policy by endpoint

| Endpoint | Free? | Condition |
|---|---|---|
| `GET /` | Yes | Always |
| `GET /health` | Yes | Always |
| `POST /memory/:session/ingest` | Yes | Only when ada-002 is NOT configured (`OPENAI_API_KEY` absent) |
| `POST /memory/:session/ingest` | **No** | When ada-002 IS configured — every ingest triggers an ada-002 API call |
| `GET /memory/:session/search?role=organize` | Yes | GTR-T5-base only, no external cost |
| `GET /memory/:session/search?role=retrieve` | **No** | Calls ada-002 |
| `GET /memory/:session/search` (no role) | Yes | Defaults to organize |
| `GET /memory` | **No** | Admin — list all sessions |
| `GET /memory/:session/info` | **No** | Session metadata |
| `DELETE /memory/:session` | **No** | Destructive |
| `POST /agent/:id/register` | **No** | Key management |
| `POST /agent/:id/encrypt` | **No** | Crypto op |
| `POST /agent/:id/decrypt` | **No** | Crypto op |
| `POST /invert` | **No** | Local but expensive, premium feature |

---

## How nuts-auth tokens work

The nuts-auth service at `https://auth.nuts.services` issues two kinds of tokens:

**JWT** (short-lived, 30 min, RS256 signed)
```
Authorization: Bearer eyJ...
```
Claims: `sub` (email), `user_id` (UUID), `name`, optional `scopes`, `exp`

**API Token** (long-lived, prefix `ahp_`)
```
Authorization: Bearer ahp_<48 chars>
```
Validated by calling `POST https://auth.nuts.services/api/validate`

Shivvr should accept both in the same `Authorization: Bearer` header and auto-detect by prefix.

---

## What to build

### 1. New crate dependency: `jsonwebtoken`

Add to `Cargo.toml`:
```toml
jsonwebtoken = "9"
```

`reqwest` is already present (used for OpenAI) — reuse it for JWKS fetch and API token validation.

---

### 2. New file: `src/auth.rs`

```rust
// NutsAuth — the client that verifies tokens against nuts-auth

pub struct NutsAuthClaims {
    pub user_id: String,      // UUID from JWT claim "user_id"
    pub email: String,        // "sub" claim
    pub name: Option<String>, // "name" claim
}

pub struct NutsAuth {
    jwks_url: String,              // https://auth.nuts.services/.well-known/jwks.json
    validate_url: String,          // https://auth.nuts.services/api/validate
    decoding_key: RwLock<Option<DecodingKey>>,  // cached RSA public key
    client: reqwest::Client,       // reuse existing reqwest
}

impl NutsAuth {
    pub fn new(jwks_url: String, validate_url: String) -> Self { ... }

    // Fetch JWKS and cache the public key. Called at startup and on key-not-found.
    pub async fn refresh_jwks(&self) -> Result<()> {
        // GET jwks_url
        // Parse JWKS JSON: keys[0].n, keys[0].e (base64url RSA components)
        // DecodingKey::from_rsa_components(n, e) → store in self.decoding_key
    }

    // Verify a JWT token locally using cached public key.
    pub async fn verify_jwt(&self, token: &str) -> Result<NutsAuthClaims> {
        // jsonwebtoken::decode::<Claims>(token, &key, &Validation::new(Algorithm::RS256))
        // If key missing → refresh_jwks() then retry once
        // Map claims to NutsAuthClaims
    }

    // Validate an ahp_ API token by calling the remote /api/validate endpoint.
    pub async fn validate_api_token(&self, token: &str) -> Result<NutsAuthClaims> {
        // POST validate_url with {"token": token}
        // Parse response: {valid, subject, user_uid, actor, token_uid}
        // If valid=false → Err(AuthError::InvalidToken)
        // Map to NutsAuthClaims {user_id: user_uid, email: subject, name: actor}
    }

    // Auto-detect token type and verify.
    pub async fn verify(&self, bearer: &str) -> Result<NutsAuthClaims> {
        if bearer.starts_with("ahp_") {
            self.validate_api_token(bearer).await
        } else {
            self.verify_jwt(bearer).await
        }
    }
}
```

---

### 3. Update `AppState`

In `src/api.rs`, add:

```rust
pub struct AppState {
    pub store: Arc<Store>,
    pub chunker: Arc<Chunker>,
    pub embedder: Arc<Embedder>,
    pub openai_embedder: Option<Arc<OpenAIEmbedder>>,
    pub crypto: Arc<CryptoManager>,
    pub inverter: Option<Arc<Inverter>>,
    pub start_time: std::time::Instant,
    // Replace: api_token: Option<String>
    // With:
    pub nuts_auth: Option<Arc<NutsAuth>>,       // None → unauthenticated dev mode
    pub openai_auth_required: bool,             // true if openai_embedder.is_some()
}
```

`openai_auth_required` is set at startup: `openai_embedder.is_some()`. The middleware reads it to decide whether ingest needs auth.

---

### 4. Replace `require_token` middleware

The new middleware `nuts_auth_gate` in `src/api.rs`:

```rust
async fn nuts_auth_gate(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path();
    let method = req.method();

    // Determine if this operation is free
    let is_free = is_free_operation(path, method, req.uri().query(), &state);

    // Extract token if present
    let token = extract_bearer(&req);

    match (is_free, &state.nuts_auth, token) {
        // Free op, no auth configured or no token: allow
        (true, None, _) => next.run(req).await,
        (true, _, None) => next.run(req).await,

        // Free op, token provided: verify and attach claims (optional enrichment)
        (true, Some(auth), Some(tok)) => {
            match auth.verify(tok).await {
                Ok(claims) => {
                    // Attach to extensions for handlers that want it
                    // req.extensions_mut().insert(claims);
                    next.run(req).await
                }
                Err(_) => next.run(req).await, // Free op: bad token doesn't block
            }
        }

        // Paid op, no auth configured: allow (dev mode warning at startup)
        (false, None, _) => next.run(req).await,

        // Paid op, no token: 401
        (false, Some(_), None) => {
            (StatusCode::UNAUTHORIZED, Json(json!({"error": "authentication required"}))).into_response()
        }

        // Paid op, token present: verify
        (false, Some(auth), Some(tok)) => {
            match auth.verify(tok).await {
                Ok(_claims) => next.run(req).await,
                Err(_) => {
                    (StatusCode::UNAUTHORIZED, Json(json!({"error": "invalid token"}))).into_response()
                }
            }
        }
    }
}

fn is_free_operation(path: &str, method: &Method, query: Option<&str>, state: &AppState) -> bool {
    match (method, path) {
        (&Method::GET, "/") => true,
        (&Method::GET, "/health") => true,

        // Ingest: free only when ada-002 is not configured
        (&Method::POST, p) if p.ends_with("/ingest") => !state.openai_auth_required,

        // Search: free only when role=organize (or no role → defaults to organize)
        (&Method::GET, p) if p.ends_with("/search") => {
            let role = query
                .and_then(|q| {
                    url::form_urlencoded::parse(q.as_bytes())
                        .find(|(k, _)| k == "role")
                        .map(|(_, v)| v.into_owned())
                })
                .unwrap_or_else(|| "organize".to_string());
            role == "organize"
        }

        _ => false,
    }
}

fn extract_bearer(req: &Request) -> Option<&str> {
    req.headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
}
```

Note: the `url` crate is needed for query string parsing, or use a simpler manual parse.
Alternative: use `form_urlencoded` from the `percent-encoding` or `url` crate already in the tree
if present, otherwise a simple manual `split('&').find(|s| s.starts_with("role="))` works fine.

---

### 5. Update `main.rs` startup

```rust
// After loading openai_embedder:
let nuts_auth = match std::env::var("NUTS_AUTH_JWKS_URL") {
    Ok(jwks_url) => {
        let validate_url = std::env::var("NUTS_AUTH_VALIDATE_URL")
            .unwrap_or_else(|_| "https://auth.nuts.services/api/validate".to_string());
        let auth = Arc::new(NutsAuth::new(jwks_url, validate_url));
        // Fetch JWKS at startup so first request doesn't pay the fetch cost
        if let Err(e) = auth.refresh_jwks().await {
            println!("WARNING: Could not fetch JWKS from nuts-auth: {} — JWT verification unavailable until resolved", e);
        } else {
            println!("Nuts-auth: JWKS loaded, JWT+API token verification active");
        }
        Some(auth)
    }
    Err(_) => {
        println!("WARNING: NUTS_AUTH_JWKS_URL not set — running unauthenticated dev mode");
        None
    }
};

let openai_auth_required = openai_embedder.is_some();

let state = Arc::new(api::AppState {
    store,
    chunker,
    embedder,
    openai_embedder,
    crypto,
    inverter,
    start_time: std::time::Instant::now(),
    nuts_auth,
    openai_auth_required,
});
```

Remove the old `api_token` field and `API_TOKEN` env var reading.

---

### 6. New environment variables

| Variable | Default | Required | Description |
|---|---|---|---|
| `NUTS_AUTH_JWKS_URL` | — | For auth | `https://auth.nuts.services/.well-known/jwks.json` |
| `NUTS_AUTH_VALIDATE_URL` | `https://auth.nuts.services/api/validate` | No | API token validation endpoint |

Remove:
| Variable | Status |
|---|---|
| `API_TOKEN` | **Remove** — replaced by nuts-auth |

---

### 7. Update `deploy.sh`

Remove `--set-env-vars API_TOKEN=...` and add:
```bash
--set-env-vars "NUTS_AUTH_JWKS_URL=https://auth.nuts.services/.well-known/jwks.json"
```

For Cloud Run, `OPENAI_API_KEY` stays as-is (if used). When it's set, the Cloud Run deploy
automatically gates ingest behind nuts-auth.

---

### 8. Update Dockerfile ENV defaults

Remove:
```dockerfile
# ENV API_TOKEN=  ← remove
```

Add (can be set at runtime):
```dockerfile
ENV NUTS_AUTH_JWKS_URL=https://auth.nuts.services/.well-known/jwks.json
ENV NUTS_AUTH_VALIDATE_URL=https://auth.nuts.services/api/validate
```

---

## JWKS structure (for implementation reference)

```json
{
  "keys": [{
    "kty": "RSA",
    "use": "sig",
    "kid": "nuts-auth-key-1",
    "alg": "RS256",
    "n": "<base64url-encoded RSA modulus>",
    "e": "AQAB"
  }]
}
```

Parse with `jsonwebtoken::DecodingKey::from_rsa_components(n, e)` where n and e are the
base64url-decoded byte slices.

---

## API token validation response (for implementation reference)

`POST https://auth.nuts.services/api/validate`
Body: `{"token": "ahp_..."}`

```json
{
  "valid": true,
  "subject": "user@example.com",
  "actor": "my-ai-agent",
  "user_uid": "f6e58154-12c8-4ed1-8c02-6194c0eb6473",
  "token_uid": "abc123"
}
```

Map to `NutsAuthClaims`:
- `user_id` ← `user_uid`
- `email` ← `subject`
- `name` ← `actor` (may be None)

---

## Caller flow (how a client gets access)

**For human users:**
1. Log in at `https://auth.nuts.services/login` (magic link or OAuth)
2. Create an API token at `GET /api/tokens`
3. Use the `ahp_` token directly in `Authorization: Bearer ahp_...`
   — OR — exchange it for a JWT: `POST https://auth.nuts.services/auth` with `{"token": "ahp_..."}`

**For AI agents / backend services:**
1. An admin creates an API token with an `actor` field naming the agent
2. Agent includes `Authorization: Bearer ahp_...` on all paid requests to shivvr
3. Shivvr validates it live against `/api/validate` on first use (or pass through JWT if pre-exchanged)

**For local Docker dev (no nuts-auth):**
- Don't set `NUTS_AUTH_JWKS_URL`
- All requests pass through (dev mode, same as today without `API_TOKEN`)

---

## Error response shape (standardized)

All 401/403 responses should return JSON (already the pattern in api.rs):
```json
{"error": "authentication required"}
{"error": "invalid token"}
{"error": "token expired"}
```

---

## Implementation order

1. Add `jsonwebtoken = "9"` to `Cargo.toml`
2. Write `src/auth.rs` (NutsAuth struct, verify_jwt, validate_api_token, refresh_jwks)
3. Update `AppState` (add `nuts_auth`, `openai_auth_required`, remove `api_token`)
4. Replace `require_token` with `nuts_auth_gate` in `src/api.rs`
5. Write `is_free_operation` helper
6. Update `main.rs` startup (load nuts-auth, remove API_TOKEN logic)
7. Update `deploy.sh` and `Dockerfile`
8. Test locally with `NUTS_AUTH_JWKS_URL` unset (dev mode), then with a real `ahp_` token

---

## Notes and edge cases

**JWKS caching**: Fetch at startup. On JWT verification failure due to signature mismatch (could
indicate key rotation), attempt one `refresh_jwks()` then retry. Don't retry infinitely.

**API token latency**: Each `ahp_` token validation is a round-trip to auth.nuts.services
(~50–150ms). JWT verification is local and fast (~<1ms). Encourage clients to pre-exchange
API tokens for JWTs and cache them for the 30-minute window. Document this.

**No `scopes` check yet**: The JWT includes a `scopes` field. Currently nuts-auth doesn't
enforce scopes on shivvr specifically. For now, any valid token authorizes any paid op.
Future work: require `scopes: ["shivvr:write"]` for ingest, `scopes: ["shivvr:read"]` for search.

**Session ownership**: Currently any authenticated user can access any session by ID. This is
fine for a single-tenant deployment. For multi-tenant use, session IDs should be prefixed by
`user_id` from claims, or session ownership stored in sled.

**OpenAI key absent + nuts-auth present**: If `NUTS_AUTH_JWKS_URL` is set but `OPENAI_API_KEY`
is not, ingest and organize-search are free (no external cost), retrieve-search returns an error
(ada-002 unavailable), and auth is still required for management endpoints.
