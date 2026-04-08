#[cfg(feature = "ml")]
use shivvr::{api, auth, chunker, crypto, embedder, inverter, openai, store, temp_store};
#[cfg(feature = "ml")]
use std::sync::Arc;
#[cfg(feature = "ml")]
use tokio::net::TcpListener;

#[cfg(feature = "ml")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/gtr-t5-base.onnx".to_string());
    let tokenizer_path = std::env::var("TOKENIZER_PATH")
        .unwrap_or_else(|_| "models/tokenizer.json".to_string());
    // Phase 1: GTR-T5-base embedder (required)
    println!("Loading embedding model from {}...", model_path);
    let embedder = Arc::new(embedder::Embedder::new(&model_path, &tokenizer_path)?);

    // Phase 1: OpenAI retrieve embedder (optional, graceful degradation)
    let openai_embedder = match std::env::var("OPENAI_API_KEY") {
        Ok(key) if !key.is_empty() => {
            println!("OpenAI API key found — retrieve embeddings enabled");
            Some(Arc::new(openai::OpenAIEmbedder::new(key)?))
        }
        _ => {
            println!("No OPENAI_API_KEY — organize-only mode (retrieve role unavailable)");
            None
        }
    };
    let openai_auth_required = openai_embedder.is_some();

    let nuts_auth = match std::env::var("NUTS_AUTH_JWKS_URL") {
        Ok(jwks_url) if !jwks_url.is_empty() => {
            let validate_url = std::env::var("NUTS_AUTH_VALIDATE_URL")
                .unwrap_or_else(|_| "https://auth.nuts.services/api/validate".to_string());
            let auth = Arc::new(auth::NutsAuth::new(jwks_url, validate_url));
            if let Err(e) = auth.refresh_jwks().await {
                println!(
                    "WARNING: Could not fetch JWKS from nuts-auth: {} — JWT verification unavailable until resolved",
                    e
                );
            } else {
                println!("Nuts-auth: JWKS loaded, JWT+API token verification active");
            }
            Some(auth)
        }
        _ => {
            println!("WARNING: NUTS_AUTH_JWKS_URL not set — running unauthenticated dev mode");
            None
        }
    };

    let store = Arc::new(store::Store::new());
    let temp_store = Arc::new(temp_store::TempStore::new());

    let chunker = Arc::new(chunker::Chunker::new(embedder.clone()));

    // Phase 2: Crypto manager (in-memory, keys lost on restart)
    let crypto = Arc::new(crypto::CryptoManager::new());

    // Phase 3: Vec2text inverter (optional, needs ONNX models)
    let inverter = {
        let projection_path = std::env::var("INVERTER_PROJECTION_PATH")
            .unwrap_or_else(|_| "models/inverter/projection.onnx".to_string());
        let encoder_path = std::env::var("INVERTER_ENCODER_PATH")
            .unwrap_or_else(|_| "models/inverter/encoder.onnx".to_string());
        let decoder_path = std::env::var("INVERTER_DECODER_PATH")
            .unwrap_or_else(|_| "models/inverter/decoder.onnx".to_string());
        let t5_tokenizer_path = std::env::var("INVERTER_TOKENIZER_PATH")
            .unwrap_or_else(|_| "models/inverter/tokenizer.json".to_string());

        match inverter::Inverter::new(
            &projection_path,
            &encoder_path,
            &decoder_path,
            &t5_tokenizer_path,
        ) {
            Ok(inv) => {
                println!("Vec2text inverter loaded");
                Some(Arc::new(inv))
            }
            Err(e) => {
                println!("Vec2text inverter not available: {} — /invert disabled", e);
                None
            }
        }
    };

    let state = Arc::new(api::AppState {
        store,
        temp_store: temp_store.clone(),
        chunker,
        embedder,
        openai_embedder,
        crypto,
        inverter,
        start_time: std::time::Instant::now(),
        nuts_auth,
        openai_auth_required,
    });

    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(600));
        loop {
            interval.tick().await;
            let removed = temp_store.sweep_expired();
            if removed > 0 {
                println!("Temp store sweeper removed {} expired stores", removed);
            }
        }
    });

    let app = api::router(state);

    let addr = format!("0.0.0.0:{}", port);
    if cfg!(feature = "cuda") {
        println!("GPU: CUDA execution provider enabled");
    } else {
        println!("GPU: none (CPU only)");
    }
    println!("Starting shivvr on {}", addr);

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(not(feature = "ml"))]
fn main() {
    eprintln!("shivvr binary requires the 'ml' feature. Build with: cargo build --features ml");
    std::process::exit(1);
}
