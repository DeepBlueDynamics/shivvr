use anyhow::{Context, Result};
use image::imageops::FilterType;
use ndarray::{Array2, Array4};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use std::sync::Mutex;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

/// Vision embedder using SigLIP ONNX vision model (224x224 RGB -> 768d unit vector)
pub struct VisionEmbedder {
    session: Mutex<Session>,
    embedding_dim: usize,
}

impl VisionEmbedder {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        // Output dimension is 768 for SigLIP-base
        let embedding_dim = 768;

        Ok(Self {
            session: Mutex::new(session),
            embedding_dim,
        })
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Preprocess an image buffer into [1, 3, 224, 224] normalized tensor (CHW)
    pub fn preprocess_image(image_bytes: &[u8]) -> Result<Array4<f32>> {
        let img = image::load_from_memory(image_bytes)
            .context("Failed to decode image bytes (must be PNG, JPEG, or WebP)")?;

        let resized = img.resize_exact(224, 224, FilterType::Triangle).to_rgb8();

        let mut tensor = Array4::<f32>::zeros((1, 3, 224, 224));

        // SigLIP normalization: mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]
        // (val / 255.0 - 0.5) / 0.5 = (val / 127.5) - 1.0
        for y in 0..224 {
            for x in 0..224 {
                let pixel = resized.get_pixel(x, y);
                tensor[[0, 0, y as usize, x as usize]] = (pixel[0] as f32 / 127.5) - 1.0;
                tensor[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 / 127.5) - 1.0;
                tensor[[0, 2, y as usize, x as usize]] = (pixel[2] as f32 / 127.5) - 1.0;
            }
        }

        Ok(tensor)
    }

    /// Embed an image from raw bytes, returns L2-normalized vector
    pub fn embed_image(&self, image_bytes: &[u8]) -> Result<Vec<f32>> {
        let tensor = Self::preprocess_image(image_bytes)?;
        let value = Value::from_array(tensor)?;

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs!["pixel_values" => value])?;

        let (_, slice) = outputs["embedding"]
            .try_extract_tensor::<f32>()?;

        let mut vec = slice.to_vec();
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in vec.iter_mut() {
            *x /= norm;
        }

        Ok(vec)
    }
}

/// SigLIP was trained with inputs padded to exactly this many tokens and no
/// attention mask, so the exported text model expects that shape.
pub const SIGLIP_TEXT_MAX_TOKENS: usize = 64;

/// SigLIP text tower: text -> 768d unit vector in the same space as
/// [`VisionEmbedder`], so text and images can be compared directly.
pub struct SiglipTextEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    embedding_dim: usize,
}

impl SiglipTextEmbedder {
    /// Load the ONNX text tower and its fast-tokenizer JSON
    /// (`scripts/export_siglip.py` writes `models/siglip-tokenizer.json`).
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to load SigLIP text model from {model_path}"))?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            anyhow::anyhow!("Failed to load SigLIP tokenizer from {tokenizer_path}: {e}")
        })?;
        // SigLIP uses "</s>" as both EOS and PAD (id 1 in the released vocab).
        let pad_id = tokenizer.token_to_id("</s>").unwrap_or(1);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: SIGLIP_TEXT_MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("Failed to configure SigLIP truncation: {e}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(SIGLIP_TEXT_MAX_TOKENS),
            pad_id,
            pad_token: "</s>".to_string(),
            ..Default::default()
        }));

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            embedding_dim: 768,
        })
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Token ids for `text`, truncated and padded to the fixed window.
    pub fn token_ids(&self, text: &str) -> Result<Vec<i64>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("SigLIP tokenization failed: {e}"))?;
        Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
    }

    /// Embed text, returns an L2-normalized 768d vector.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let ids = self.token_ids(text)?;
        let len = ids.len();
        let tensor = Array2::from_shape_vec((1, len), ids).context("SigLIP input tensor shape")?;
        let value = Value::from_array(tensor)?;

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs!["input_ids" => value])?;
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

    #[test]
    fn test_preprocess_synthetic_image() {
        // Create 10x10 synthetic PNG
        let mut imgbuf = image::RgbImage::new(10, 10);
        for pixel in imgbuf.pixels_mut() {
            *pixel = image::Rgb([255, 128, 0]);
        }
        let mut bytes: Vec<u8> = Vec::new();
        imgbuf.write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png).unwrap();

        let tensor = VisionEmbedder::preprocess_image(&bytes).unwrap();
        assert_eq!(tensor.shape(), &[1, 3, 224, 224]);
        // Red = 255 -> (255 / 127.5) - 1.0 = 1.0
        assert!((tensor[[0, 0, 0, 0]] - 1.0).abs() < 1e-3);
    }
}
