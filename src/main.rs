use std::sync::Arc;
use tokio::net::TcpListener;

#[cfg(feature = "ml")]
mod api;
#[cfg(feature = "ml")]
mod chunker;
mod crypto;
#[cfg(feature = "ml")]
mod embedder;
#[cfg(feature = "ml")]
mod inverter;
mod openai;
mod similarity;
mod store;

#[cfg(feature = "ml")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/gtr-t5-base.onnx".to_string());
    let tokenizer_path = std::env::var("TOKENIZER_PATH")
        .unwrap_or_else(|_| "models/tokenizer.json".to_string());
    let data_path = std::env::var("DATA_PATH").unwrap_or_else(|_| "/data/shivvr".to_string());

    // Phase 1: BGE-small embedder (required)
    println!("Loading embedding model from {}...", model_path);
    let embedder = Arc::new(embedder::Embedder::new(&model_path, &tokenizer_path)?);

    // Phase 1: OpenAI ada-002 embedder (optional, graceful degradation)
    let openai_embedder = match std::env::var("OPENAI_API_KEY") {
        Ok(key) if !key.is_empty() => {
            println!("OpenAI API key found — ada-002 retrieve embeddings enabled");
            Some(Arc::new(openai::OpenAIEmbedder::new(key)?))
        }
        _ => {
            println!("No OPENAI_API_KEY — running organize-only mode");
            None
        }
    };

    println!("Opening database at {}...", data_path);
    let store = Arc::new(store::Store::open(&data_path)?);

    let chunker = Arc::new(chunker::Chunker::new(embedder.clone()));

    // Phase 2: Crypto manager (shares sled db)
    let crypto = Arc::new(crypto::CryptoManager::new(store.db()));

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
        chunker,
        embedder,
        openai_embedder,
        crypto,
        inverter,
        start_time: std::time::Instant::now(),
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
