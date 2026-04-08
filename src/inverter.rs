use anyhow::{bail, Result};
use ndarray::{Array2, Array3, Axis};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// Vec2text inverter: embedding → text via T5 ONNX pipeline
///
/// Pipeline:
///   1. embedding → projection ONNX → (1, N, 768) encoder input
///   2. T5 encoder ONNX → encoder hidden states
///   3. T5 decoder ONNX → greedy autoregressive decoding → token IDs → text
pub struct Inverter {
    projection: Mutex<Session>,
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    tokenizer: Tokenizer,
    /// Expected embedding dimension (detected from projection model)
    embedding_dim: usize,
    /// Maximum tokens to generate
    max_length: usize,
    /// T5 decoder start token (pad_token_id = 0 for T5)
    decoder_start_token_id: i64,
    /// EOS token ID
    eos_token_id: i64,
}

impl Inverter {
    fn build_session(model_path: &str) -> Result<Session> {
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(2)?;

        #[cfg(feature = "cuda")]
        {
            use ort::ep::CUDA;
            builder = builder.with_execution_providers([CUDA::default().build()])?;
        }

        Ok(builder.commit_from_file(model_path)?)
    }

    /// Load all ONNX models and tokenizer
    ///
    /// Expected paths:
    ///   - projection_path: projection.onnx (Nd → 16x768)
    ///   - encoder_path: encoder.onnx (inputs_embeds interface)
    ///   - decoder_path: decoder.onnx
    ///   - tokenizer_path: tokenizer.json (T5 tokenizer)
    pub fn new(
        projection_path: &str,
        encoder_path: &str,
        decoder_path: &str,
        tokenizer_path: &str,
    ) -> Result<Self> {
        let projection = Self::build_session(projection_path)?;

        // Detect embedding dimension from the projection model's input shape
        let embedding_dim = projection
            .inputs()
            .first()
            .and_then(|i| i.dtype().tensor_shape())
            .and_then(|s| s.get(1).copied())
            .filter(|&d| d > 0)
            .map(|d| d as usize)
            .unwrap_or(768); // default to 768 (gtr-base)

        let encoder = Self::build_session(encoder_path)?;
        let decoder = Self::build_session(decoder_path)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load T5 tokenizer: {}", e))?;

        println!("Vec2text inverter: {}d embeddings", embedding_dim);

        Ok(Self {
            projection: Mutex::new(projection),
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            tokenizer,
            embedding_dim,
            max_length: 64,
            decoder_start_token_id: 0, // T5 pad token
            eos_token_id: 1,           // T5 EOS token
        })
    }

    /// Invert an embedding back to text
    pub fn invert(&self, embedding: &[f32], max_length: usize) -> Result<String> {
        if embedding.len() != self.embedding_dim {
            bail!(
                "Expected {}d embedding for inversion, got {}d",
                self.embedding_dim,
                embedding.len()
            );
        }

        // Step 1: Project embedding → (1, N, 768) via projection.onnx
        let projected = self.project(embedding)?;

        // Step 2: Encode projected input via T5 encoder
        let encoder_output = self.encode(&projected)?;

        // Step 3: Greedy decode from encoder output
        let token_ids = self.greedy_decode(&encoder_output, max_length)?;

        // Step 4: Decode token IDs to text
        let text = self
            .tokenizer
            .decode(&token_ids, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer decode failed: {}", e))?;

        Ok(text.trim().to_string())
    }

    /// Run projection ONNX: (N,) → (1, seq, 768)
    fn project(&self, embedding: &[f32]) -> Result<Array3<f32>> {
        let input = Array2::from_shape_vec((1, self.embedding_dim), embedding.to_vec())?;
        let input_value = Value::from_array(input)?;

        let mut session = self
            .projection
            .lock()
            .map_err(|_| anyhow::anyhow!("Projection lock poisoned"))?;

        let outputs = session.run(ort::inputs!["input" => input_value])?;

        let output = outputs
            .iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No output from projection model"))?;

        let view = output.1.try_extract_array::<f32>()?;
        let result = view
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| anyhow::anyhow!("Projection output shape error: {}", e))?;

        Ok(result.to_owned())
    }

    /// Run T5 encoder: (1, 16, 768) → encoder hidden states
    fn encode(&self, projected: &Array3<f32>) -> Result<Array3<f32>> {
        // T5 encoder expects input_ids, but we're feeding projected embeddings
        // The encoder model should accept inputs_embeds
        let attention_mask = Array2::from_elem((1, projected.shape()[1]), 1i64);

        let projected_value = Value::from_array(projected.clone())?;
        let attention_value = Value::from_array(attention_mask)?;

        let mut session = self
            .encoder
            .lock()
            .map_err(|_| anyhow::anyhow!("Encoder lock poisoned"))?;

        let outputs = session.run(ort::inputs![
            "inputs_embeds" => projected_value,
            "attention_mask" => attention_value,
        ])?;

        let output = outputs
            .iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No output from encoder"))?;

        let view = output.1.try_extract_array::<f32>()?;
        let result = view
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| anyhow::anyhow!("Encoder output shape error: {}", e))?;

        Ok(result.to_owned())
    }

    /// Greedy autoregressive decoding
    fn greedy_decode(&self, encoder_output: &Array3<f32>, max_length: usize) -> Result<Vec<u32>> {
        let seq_len = encoder_output.shape()[1];
        let encoder_attention_mask = Array2::from_elem((1, seq_len), 1i64);

        let mut generated = vec![self.decoder_start_token_id];

        for _ in 0..max_length {
            let decoder_input =
                Array2::from_shape_vec((1, generated.len()), generated.clone())?;

            let encoder_value = Value::from_array(encoder_output.clone())?;
            let encoder_mask_value = Value::from_array(encoder_attention_mask.clone())?;
            let decoder_value = Value::from_array(decoder_input)?;

            let mut session = self
                .decoder
                .lock()
                .map_err(|_| anyhow::anyhow!("Decoder lock poisoned"))?;

            let outputs = session.run(ort::inputs![
                "input_ids" => decoder_value,
                "encoder_hidden_states" => encoder_value,
                "encoder_attention_mask" => encoder_mask_value,
            ])?;

            let logits_output = outputs
                .iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("No output from decoder"))?;

            let logits = logits_output.1.try_extract_array::<f32>()?;
            let logits = logits
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| anyhow::anyhow!("Decoder logits shape error: {}", e))?;

            // Get logits for the last position
            let last_logits = logits.index_axis(Axis(0), 0);
            let last_step = last_logits.index_axis(Axis(0), last_logits.shape()[0] - 1);

            // Argmax
            let next_token = last_step
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i64)
                .unwrap_or(self.eos_token_id);

            if next_token == self.eos_token_id {
                break;
            }

            generated.push(next_token);
        }

        // Skip the start token
        Ok(generated[1..].iter().map(|&x| x as u32).collect())
    }
}
