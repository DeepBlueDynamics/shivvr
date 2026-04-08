use anyhow::Result;
use serde::{Deserialize, Serialize};

pub struct OpenAIEmbedder {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl OpenAIEmbedder {
    pub fn new(api_key: String) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;
        let model = std::env::var("OPENAI_EMBEDDING_MODEL")
            .unwrap_or_else(|_| "text-embedding-ada-002".to_string());

        Ok(Self {
            client,
            api_key,
            model,
        })
    }

    /// Embed a single text, returns L2-normalized 1536d vector
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut results = self.embed_batch(&[text.to_string()]).await?;
        results
            .pop()
            .ok_or_else(|| anyhow::anyhow!("Empty response from OpenAI"))
    }

    /// Embed a batch of texts, returns L2-normalized 1536d vectors
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let request = EmbeddingRequest {
            model: self.model.clone(),
            input: texts.to_vec(),
        };

        let response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "OpenAI API error {}: {}",
                status,
                body
            ));
        }

        let resp: EmbeddingResponse = response.json().await?;

        // L2-normalize each embedding
        let normalized: Vec<Vec<f32>> = resp
            .data
            .into_iter()
            .map(|d| l2_normalize(d.embedding))
            .collect();

        Ok(normalized)
    }
}

fn l2_normalize(v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.into_iter().map(|x| x / norm).collect()
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_normalize_unit_vector() {
        let v = vec![3.0, 4.0];
        let n = l2_normalize(v);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "should be unit length");
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_already_unit() {
        let v = vec![1.0, 0.0, 0.0];
        let n = l2_normalize(v);
        assert!((n[0] - 1.0).abs() < 1e-6);
        assert!(n[1].abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let n = l2_normalize(v);
        assert_eq!(n, vec![0.0, 0.0, 0.0], "zero vector should stay zero");
    }

    #[test]
    fn l2_normalize_high_dim() {
        let v: Vec<f32> = (0..384).map(|i| (i as f32) * 0.01).collect();
        let n = l2_normalize(v);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "384d vector should normalize to unit: {}",
            norm
        );
    }

    #[test]
    fn embedder_construction() {
        // Just verify we can construct one (no actual API calls)
        let embedder = OpenAIEmbedder::new("sk-test-fake-key".to_string());
        assert!(embedder.is_ok());
    }
}
