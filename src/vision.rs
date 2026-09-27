use anyhow::{Context, Result};
use image::imageops::FilterType;
use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use std::sync::Mutex;

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
