use axum::{
    body::{to_bytes, Body},
    extract::State,
    http::{header, HeaderMap, Request, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde_json::json;
use shivvr::auth::NutsAuth;
use std::{sync::Arc, time::Duration};

const MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;
const METADATA_IDENTITY_URL: &str = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity";

struct GatewayState {
    auth: NutsAuth,
    backend_url: String,
    backend_client: reqwest::Client,
    metadata_client: reqwest::Client,
    metadata_url: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let jwks_url = required_env("NUTS_AUTH_JWKS_URL")?;
    let backend_url = required_env("BACKEND_URL")?.trim_end_matches('/').to_string();
    let parsed = reqwest::Url::parse(&backend_url)?;
    anyhow::ensure!(parsed.scheme() == "https" && parsed.host_str().is_some(), "BACKEND_URL must be HTTPS");
    anyhow::ensure!(parsed.path() == "/" && parsed.query().is_none(), "BACKEND_URL must be a service origin");
    let validate_url = std::env::var("NUTS_AUTH_VALIDATE_URL")
        .unwrap_or_else(|_| "https://auth.nuts.services/api/validate".to_string());
    let auth = NutsAuth::new(jwks_url, validate_url);
    auth.refresh_jwks().await?;

    let state = Arc::new(GatewayState {
        auth,
        backend_url,
        backend_client: reqwest::Client::builder().timeout(Duration::from_secs(300)).build()?,
        metadata_client: reqwest::Client::builder().timeout(Duration::from_secs(5)).build()?,
        metadata_url: METADATA_IDENTITY_URL.to_string(),
    });
    let app = router(state);
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    // Loopback unless BIND_ADDR says otherwise; Dockerfile.gateway sets 0.0.0.0.
    let bind_addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1".to_string());
    let listener = tokio::net::TcpListener::bind(format!("{bind_addr}:{port}")).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn required_env(name: &str) -> anyhow::Result<String> {
    let value = std::env::var(name)?;
    anyhow::ensure!(!value.trim().is_empty(), "{name} must not be empty");
    Ok(value)
}

fn router(state: Arc<GatewayState>) -> Router {
    Router::new()
        .route("/", get(home))
        .route("/health", get(health))
        .fallback(proxy)
        .with_state(state)
}

async fn home() -> Html<String> {
    Html(include_str!("../landing.html").replace("{{VERSION}}", env!("CARGO_PKG_VERSION")))
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({"status":"ok","mode":"gateway","backend":"on demand"}))
}

fn bearer(headers: &HeaderMap) -> Option<&str> {
    let value = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    let token = value.strip_prefix("Bearer ")?.trim();
    if token.is_empty() { None } else { Some(token) }
}

fn error(status: StatusCode, message: &'static str) -> Response {
    (status, Json(json!({"error": message}))).into_response()
}

fn forward_header(name: &header::HeaderName) -> bool {
    !matches!(name.as_str(),
        "host" | "connection" | "keep-alive" | "proxy-authenticate" |
        "proxy-authorization" | "te" | "trailer" | "transfer-encoding" |
        "upgrade" | "content-length" | "x-serverless-authorization")
}

async fn proxy(State(state): State<Arc<GatewayState>>, request: Request<Body>) -> Response {
    let token = match bearer(request.headers()) {
        Some(token) => token,
        None => return error(StatusCode::UNAUTHORIZED, "authentication required"),
    };
    if state.auth.verify(token).await.is_err() {
        return error(StatusCode::UNAUTHORIZED, "invalid token");
    }

    let (parts, body) = request.into_parts();
    let body = match to_bytes(body, MAX_REQUEST_BYTES).await {
        Ok(body) => body,
        Err(_) => return error(StatusCode::PAYLOAD_TOO_LARGE, "request too large"),
    };
    let id_token = match backend_id_token(&state).await {
        Ok(token) => token,
        Err(err) => {
            tracing::error!("backend identity failed: {err}");
            return error(StatusCode::BAD_GATEWAY, "backend authentication unavailable");
        }
    };

    let url = format!("{}{}", state.backend_url, parts.uri);
    let mut upstream = state.backend_client.request(parts.method, url);
    for (name, value) in parts.headers.iter() {
        if forward_header(name) {
            upstream = upstream.header(name, value);
        }
    }
    upstream = upstream.header("X-Serverless-Authorization", format!("Bearer {id_token}")).body(body);
    let upstream = match upstream.send().await {
        Ok(response) => response,
        Err(err) => {
            tracing::error!("backend request failed: {err}");
            return error(StatusCode::BAD_GATEWAY, "backend unavailable");
        }
    };

    let status = upstream.status();
    let headers = upstream.headers().clone();
    let mut response = Response::new(Body::from_stream(upstream.bytes_stream()));
    *response.status_mut() = status;
    for (name, value) in headers.iter() {
        if forward_header(name) {
            response.headers_mut().insert(name, value.clone());
        }
    }
    response
}

async fn backend_id_token(state: &GatewayState) -> anyhow::Result<String> {
    let mut url = reqwest::Url::parse(&state.metadata_url)?;
    url.query_pairs_mut().append_pair("audience", &state.backend_url);
    let response = state.metadata_client
        .get(url)
        .header("Metadata-Flavor", "Google")
        .send().await?
        .error_for_status()?;
    let token = response.text().await?;
    anyhow::ensure!(!token.trim().is_empty(), "metadata server returned an empty ID token");
    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authorization_header_requires_bearer_token() {
        let mut headers = HeaderMap::new();
        assert_eq!(bearer(&headers), None);
        headers.insert(header::AUTHORIZATION, "Bearer ".parse().unwrap());
        assert_eq!(bearer(&headers), None);
        headers.insert(header::AUTHORIZATION, "Bearer ahp_example".parse().unwrap());
        assert_eq!(bearer(&headers), Some("ahp_example"));
    }

    #[test]
    fn caller_cannot_supply_cloud_run_identity() {
        assert!(!forward_header(&"x-serverless-authorization".parse().unwrap()));
        assert!(!forward_header(&header::HOST));
        assert!(forward_header(&header::AUTHORIZATION));
    }

    #[tokio::test]
    async fn only_valid_tokens_wake_and_reach_backend() {
        use axum::{body::Bytes, extract::Json as ExtractJson, routing::post};
        use std::sync::atomic::{AtomicUsize, Ordering};

        async fn serve(app: Router) -> String {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            format!("http://127.0.0.1:{port}")
        }

        let auth_url = serve(Router::new().route(
            "/validate",
            post(|ExtractJson(value): ExtractJson<serde_json::Value>| async move {
                Json(json!({
                    "valid": value["token"] == "ahp_good",
                    "user_uid": "user",
                    "subject": "user@example.test"
                }))
            }),
        )).await;
        let metadata_url = serve(Router::new().route(
            "/identity",
            get(|| async { "cloud-run-id-token" }),
        )).await;
        let calls = Arc::new(AtomicUsize::new(0));
        let backend_calls = calls.clone();
        let backend_url = serve(Router::new().route(
            "/probe",
            post(move |headers: HeaderMap, uri: axum::http::Uri, body: Bytes| {
                let calls = backend_calls.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Json(json!({
                        "path_query": uri.to_string(),
                        "body": String::from_utf8(body.to_vec()).unwrap(),
                        "caller": headers.get(header::AUTHORIZATION).unwrap().to_str().unwrap(),
                        "iam": headers.get("x-serverless-authorization").unwrap().to_str().unwrap()
                    }))
                }
            }),
        )).await;
        let state = Arc::new(GatewayState {
            auth: NutsAuth::new("unused".into(), format!("{auth_url}/validate")),
            backend_url,
            backend_client: reqwest::Client::new(),
            metadata_client: reqwest::Client::new(),
            metadata_url: format!("{metadata_url}/identity"),
        });
        let gateway_url = serve(router(state)).await;
        let client = reqwest::Client::new();
        let endpoint = format!("{gateway_url}/probe?session=one");

        let missing = client.post(&endpoint).send().await.unwrap();
        assert_eq!(missing.status(), StatusCode::UNAUTHORIZED);
        let invalid = client.post(&endpoint).header(header::AUTHORIZATION, "Bearer ahp_bad").send().await.unwrap();
        assert_eq!(invalid.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        let valid = client.post(&endpoint)
            .header(header::AUTHORIZATION, "Bearer ahp_good")
            .header("x-serverless-authorization", "forged")
            .body("hello")
            .send().await.unwrap();
        assert_eq!(valid.status(), StatusCode::OK);
        let body: serde_json::Value = valid.json().await.unwrap();
        assert_eq!(body["path_query"], "/probe?session=one");
        assert_eq!(body["body"], "hello");
        assert_eq!(body["caller"], "Bearer ahp_good");
        assert_eq!(body["iam"], "Bearer cloud-run-id-token");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
