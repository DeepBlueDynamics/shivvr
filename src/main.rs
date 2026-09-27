#[cfg(feature = "ml")]
use shivvr::{api, audio, auth, chunker, crypto, embedder, inverter, openai, store, temp_store, vision};
#[cfg(feature = "ml")]
use std::sync::Arc;
#[cfg(feature = "ml")]
use tokio::net::TcpListener;

#[cfg(feature = "ml")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());

    // Landing-only mode: skip all model/backend initialization entirely and
    // serve just the static homepage + a stub /health. Used for a lightweight,
    // GPU-less "front door" instance -- the real backend runs as a separate
    // service, started on demand, and is not touched here.
    if std::env::var("LANDING_ONLY").map(|v| v == "true").unwrap_or(false) {
        println!("LANDING_ONLY=true -- serving static homepage only, no backend/model init");
        let app = api::landing_router();
        let addr = format!("0.0.0.0:{}", port);
        println!("Starting shivvr (landing-only) on {}", addr);
        let listener = TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;
        return Ok(());
    }

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

    // Phase 4: FST Tagger Guardrails & Search Config
    let guardrails_dir = std::env::var("GUARDRAILS_DIR").unwrap_or_else(|_| "guardrails".to_string());
    let _ = std::fs::create_dir_all(&guardrails_dir);
    let default_csv_path = std::path::Path::new(&guardrails_dir).join("offensive.csv");
    if !default_csv_path.exists() {
        if let Err(e) = std::fs::write(&default_csv_path, "phrase,action\nbullshit,BLOCK\ndamn,BLOCK\n") {
            println!("WARNING: Could not write default guardrail file: {}", e);
        }
    }

    println!("Loading guardrails from {}...", guardrails_dir);
    let tagger = match lume_hybrid::Tagger::from_data_dir(&guardrails_dir) {
        Ok(t) => {
            println!("Guardrails loaded successfully. Active entries: {}", t.record_count());
            t
        }
        Err(e) => {
            println!("WARNING: Failed to load guardrails from {}: {} — fallback to empty tagger", guardrails_dir, e);
            lume_hybrid::Tagger::build(Vec::<lume_hybrid::Entry>::new()).unwrap()
        }
    };
        let guardrail_tagger = Arc::new(std::sync::RwLock::new(tagger));
    // Was `SearchConfig::default()` in the old rust-hybrid-search crate. Lume
    // split that into Bm25Params (tuning) + SearchVariant (selector). We hold
    // the params; variant is selected per call in the search handlers.
    let search_params = lume_hybrid::bm25::Bm25Params::default();
    let mcp_connections = Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new()));


    let transcription_url = std::env::var("TRANSCRIPTION_URL").ok();
    let audio_client = Arc::new(audio::AudioClient::new(transcription_url));

    let vision_model_path = std::env::var("VISION_MODEL_PATH")
        .unwrap_or_else(|_| "models/siglip-vision.onnx".to_string());
    let vision_embedder = match vision::VisionEmbedder::new(&vision_model_path) {
        Ok(v) => {
            println!("SigLIP vision embedder loaded from {}", vision_model_path);
            Some(Arc::new(v))
        }
        Err(e) => {
            println!("Vision embedder not available: {} — image embedding disabled", e);
            None
        }
    };

    // SigLIP text tower: same space as the vision embedder, served through
    // `POST /embed` with model=siglip-base-patch16-224.
    let siglip_text_model_path = std::env::var("SIGLIP_TEXT_MODEL_PATH")
        .unwrap_or_else(|_| "models/siglip-text.onnx".to_string());
    let siglip_tokenizer_path = std::env::var("SIGLIP_TOKENIZER_PATH")
        .unwrap_or_else(|_| "models/siglip-tokenizer.json".to_string());
    let siglip_text = match vision::SiglipTextEmbedder::new(&siglip_text_model_path, &siglip_tokenizer_path) {
        Ok(t) => {
            println!(
                "SigLIP text embedder loaded from {} (tokenizer {})",
                siglip_text_model_path, siglip_tokenizer_path
            );
            Some(Arc::new(t))
        }
        Err(e) => {
            println!(
                "SigLIP text embedder not available: {} — /embed model=siglip-base-patch16-224 disabled",
                e
            );
            None
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
        audio_client,
        vision_embedder,
        siglip_text,
        start_time: std::time::Instant::now(),
        nuts_auth,
        openai_auth_required,
        guardrail_tagger,
        search_params,
        mcp_connections,
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
