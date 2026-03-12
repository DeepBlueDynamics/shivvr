use anyhow::Result;
use ndarray::{Array2, Axis, Ix3};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use tokenizers::Tokenizer;
use std::sync::Mutex;

pub struct Embedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    /// True if the ONNX model accepts token_type_ids (BERT-based models only)
    has_token_type_ids: bool,
}

impl Embedder {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?;

        #[cfg(feature = "cuda")]
        {
            use ort::ep::CUDA;
            builder = builder.with_execution_providers([CUDA::default().build()])?;
            println!("CUDA execution provider registered");
        }

        let session = builder.commit_from_file(model_path)?;

        // Check if model accepts token_type_ids (BERT-based yes, T5-based no)
        let has_token_type_ids = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            has_token_type_ids,
        })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        let seq_len = ids.len();

        let input_ids: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
        let attention: Vec<i64> = attention_mask.iter().map(|&x| x as i64).collect();

        let input_ids = Array2::from_shape_vec((1, seq_len), input_ids)?;
        let attention_mask = Array2::from_shape_vec((1, seq_len), attention)?;

        let input_ids = Value::from_array(input_ids)?;
        let attention_mask = Value::from_array(attention_mask)?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("Session lock poisoned"))?;

        let outputs = if self.has_token_type_ids {
            let token_type_ids: Vec<i64> = vec![0i64; seq_len];
            let token_type_ids = Array2::from_shape_vec((1, seq_len), token_type_ids)?;
            let token_type_ids = Value::from_array(token_type_ids)?;
            session.run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "token_type_ids" => token_type_ids,
            ])?
        } else {
            session.run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
            ])?
        };

        let output = outputs
            .get("last_hidden_state")
            .ok_or_else(|| anyhow::anyhow!("No last_hidden_state in output"))?;

        let view = output.try_extract_array::<f32>()?;
        let view = view
            .into_dimensionality::<Ix3>()
            .map_err(|_| anyhow::anyhow!("Unexpected tensor shape for last_hidden_state"))?;

        let pooled = view
            .mean_axis(Axis(1))
            .ok_or_else(|| anyhow::anyhow!("Failed to mean-pool embeddings"))?;

        let embedding: Vec<f32> = pooled.index_axis(Axis(0), 0).to_vec();

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = if norm > 0.0 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            embedding
        };

        Ok(normalized)
    }

    pub fn count_tokens(&self, text: &str) -> usize {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().len())
            .unwrap_or(0)
    }
}
