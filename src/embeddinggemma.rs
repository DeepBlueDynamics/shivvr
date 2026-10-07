//! EmbeddingGemma 2 text path: text -> 768d unit vector.
//!
//! `scripts/export_embeddinggemma2.py` exports the text backbone with mean
//! pooling (masked) and L2 normalization folded into the graph, so the ONNX
//! model maps `input_ids` + `attention_mask` straight to `embedding` [B, 768].
//! Task prefixes and Matryoshka truncation are applied by the caller (see
//! `embed_api`).

use anyhow::{Context, Result};
use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;
use std::sync::Mutex;
use tokenizers::{Tokenizer, TruncationParams};

use crate::embed_api::EMBEDDINGGEMMA2_DIM;

/// The model's context window, shared across modalities.
pub const EMBEDDINGGEMMA2_MAX_TOKENS: usize = 8192;
/// Default input cap. Cost grows faster than linearly with length (WP1: ~5 s
/// at 2,048 tokens vs 70-90 s and ~8 GB RSS at 8,192 on CPU), so the full
/// window is opt-in via `EMBEDDINGGEMMA2_MAX_TOKENS`.
pub const EMBEDDINGGEMMA2_DEFAULT_MAX_TOKENS: usize = 2048;

pub struct EmbeddingGemma2Embedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl EmbeddingGemma2Embedder {
    /// Load the ONNX text model and the tokenizer JSON written by the export
    /// script (`models/embeddinggemma2-text.onnx`,
    /// `models/embeddinggemma2-tokenizer.json`). Uses CUDA when built with the
    /// `cuda` feature and the GPU passes a probe, CPU otherwise. Inputs are
    /// truncated to `max_tokens` (clamped to the model's 8,192-token window).
    pub fn new(model_path: &str, tokenizer_path: &str, max_tokens: usize) -> Result<Self> {
        let session = crate::embedder::build_session(model_path)
            .with_context(|| format!("Failed to load EmbeddingGemma 2 model from {model_path}"))?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            anyhow::anyhow!("Failed to load EmbeddingGemma 2 tokenizer from {tokenizer_path}: {e}")
        })?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: max_tokens.clamp(1, EMBEDDINGGEMMA2_MAX_TOKENS),
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("Failed to configure EmbeddingGemma 2 truncation: {e}"))?;
        tokenizer.with_padding(None);

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
        })
    }

    pub fn embedding_dim(&self) -> usize {
        EMBEDDINGGEMMA2_DIM
    }

    /// Embed `prefix` + `text`, returns an L2-normalized 768d vector.
    pub fn embed_text(&self, prefix: &str, text: &str) -> Result<Vec<f32>> {
        let input = format!("{prefix}{text}");
        let encoding = self
            .tokenizer
            .encode(input, true)
            .map_err(|e| anyhow::anyhow!("EmbeddingGemma 2 tokenization failed: {e}"))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let len = ids.len();
        let input_ids = Array2::from_shape_vec((1, len), ids).context("input_ids shape")?;
        let attention_mask = Array2::from_elem((1, len), 1i64);

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("Session lock poisoned"))?;
        let outputs = session.run(ort::inputs![
            "input_ids" => Value::from_array(input_ids)?,
            "attention_mask" => Value::from_array(attention_mask)?,
        ])?;
        let (_, slice) = outputs["embedding"].try_extract_tensor::<f32>()?;

        let mut vec = slice.to_vec();
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in vec.iter_mut() {
            *x /= norm;
        }
        Ok(vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixture schema for `tests/fixtures/embeddinggemma2_golden.json`:
    /// ```json
    /// {
    ///   "texts": ["text 1", "text 2"],
    ///   "task": "query",
    ///   "vectors": [[0.0123, -0.0456, ...], [0.0789, 0.0321, ...]]
    /// }
    /// ```
    #[derive(serde::Deserialize)]
    struct GoldenFixture {
        texts: Vec<String>,
        #[serde(default)]
        task: Option<String>,
        vectors: Vec<Vec<f32>>,
    }

    #[test]
    #[ignore = "requires models/embeddinggemma2-text.onnx and tests/fixtures/embeddinggemma2_golden.json"]
    fn test_embeddinggemma2_golden() {
        let fixture_path = "tests/fixtures/embeddinggemma2_golden.json";
        let model_path = "models/embeddinggemma2-text.onnx";
        let tokenizer_path = "models/embeddinggemma2-tokenizer.json";

        let fixture_str = std::fs::read_to_string(fixture_path)
            .unwrap_or_else(|e| panic!("Failed to read golden fixture {fixture_path}: {e}"));
        let fixture: GoldenFixture = serde_json::from_str(&fixture_str)
            .unwrap_or_else(|e| panic!("Failed to parse golden fixture {fixture_path}: {e}"));

        assert_eq!(
            fixture.texts.len(),
            fixture.vectors.len(),
            "fixture texts ({}) and vectors ({}) length mismatch",
            fixture.texts.len(),
            fixture.vectors.len()
        );

        let embedder = EmbeddingGemma2Embedder::new(model_path, tokenizer_path, EMBEDDINGGEMMA2_MAX_TOKENS)
            .expect("Failed to create EmbeddingGemma2Embedder");

        let prefix = crate::embed_api::embeddinggemma2_prefix(fixture.task.as_deref())
            .expect("Invalid task in golden fixture");

        for (i, (text, expected)) in fixture.texts.iter().zip(fixture.vectors.iter()).enumerate() {
            assert_eq!(
                expected.len(),
                EMBEDDINGGEMMA2_DIM,
                "expected vector dim mismatch at index {i}"
            );
            let actual = embedder
                .embed_text(prefix, text)
                .unwrap_or_else(|e| panic!("embed_text failed for text[{i}]: {e}"));
            assert_eq!(
                actual.len(),
                EMBEDDINGGEMMA2_DIM,
                "actual vector dim mismatch at index {i}"
            );

            let dot: f32 = actual.iter().zip(expected.iter()).map(|(a, b)| a * b).sum();
            let norm_a: f32 = actual.iter().map(|a| a * a).sum::<f32>().sqrt();
            let norm_b: f32 = expected.iter().map(|b| b * b).sum::<f32>().sqrt();
            let cosine = dot / (norm_a * norm_b);

            assert!(
                cosine >= 0.9999,
                "text[{i}] cosine similarity {cosine:.6} < 0.9999 for {text:?}"
            );
        }
    }
}
