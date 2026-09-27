use anyhow::{bail, Context, Result};
use reqwest::multipart::{Form, Part};
use serde::Deserialize;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct AudioClient {
    client: reqwest::Client,
    base_url: String,
}

#[derive(Deserialize)]
struct TranscribeResponse {
    success: Option<bool>,
    job_id: Option<String>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct JobStatusResponse {
    status: String,
    transcript_preview: Option<String>,
    error: Option<String>,
}

impl AudioClient {
    pub fn new(base_url: Option<String>) -> Self {
        let base_url = base_url.unwrap_or_else(|| {
            std::env::var("TRANSCRIPTION_URL")
                .unwrap_or_else(|_| "http://localhost:8765".to_string())
        });

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self { client, base_url }
    }

    /// Check if transcription service is available
    pub async fn health(&self) -> bool {
        let url = format!("{}/health", self.base_url);
        match self.client.get(&url).timeout(Duration::from_secs(3)).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    /// Transcribe raw audio bytes (WAV/MP3/etc.) via hyperia-transcription service
    pub async fn transcribe(&self, audio_bytes: Vec<u8>, filename: Option<&str>) -> Result<String> {
        let fname = filename.unwrap_or("audio.wav").to_string();
        let part = Part::bytes(audio_bytes)
            .file_name(fname)
            .mime_str("audio/wav")?;

        let form = Form::new().part("file", part);

        let upload_url = format!("{}/transcribe", self.base_url);
        let resp = self
            .client
            .post(&upload_url)
            .multipart(form)
            .send()
            .await
            .context("Failed to connect to transcription service")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("Transcription upload failed ({}): {}", status, body);
        }

        let upload_result: TranscribeResponse = resp.json().await?;
        let job_id = upload_result
            .job_id
            .ok_or_else(|| anyhow::anyhow!("No job_id returned by transcription service"))?;

        // Poll for completion (up to 60s)
        let status_url = format!("{}/status/{}", self.base_url, job_id);
        let mut attempts = 0;
        let max_attempts = 120; // 120 * 500ms = 60s

        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;
            attempts += 1;

            let status_resp = self
                .client
                .get(&status_url)
                .send()
                .await
                .context("Failed to poll transcription status")?;

            if !status_resp.status().is_success() {
                continue;
            }

            let job_status: JobStatusResponse = status_resp.json().await?;

            match job_status.status.as_str() {
                "completed" => {
                    // Fetch full transcript from download endpoint
                    let download_url = format!("{}/download/{}", self.base_url, job_id);
                    let dl_resp = self.client.get(&download_url).send().await?;
                    if dl_resp.status().is_success() {
                        let full_text = dl_resp.text().await?;
                        return Ok(clean_transcript(&full_text));
                    }

                    // Fallback to preview if download endpoint fails
                    if let Some(preview) = job_status.transcript_preview {
                        return Ok(preview);
                    }
                    bail!("Transcription completed but transcript could not be retrieved");
                }
                "failed" => {
                    let err = job_status.error.unwrap_or_else(|| "Unknown error".to_string());
                    bail!("Transcription job failed: {}", err);
                }
                _ => {
                    if attempts >= max_attempts {
                        bail!("Transcription timed out after 60 seconds");
                    }
                }
            }
        }
    }
}

/// Clean transmission status report headers/footers to extract raw transcript text
fn clean_transcript(raw: &str) -> String {
    if let Some(idx) = raw.find("TELEGRAPH COPY FOLLOWS") {
        let after = &raw[idx + "TELEGRAPH COPY FOLLOWS".len()..];
        let end_idx = after.find("END OF TRANSMISSION").unwrap_or(after.len());
        let body = &after[..end_idx];

        // Format is typically: "0001 [0000.00s - 0002.50s] Text..."
        let mut lines = Vec::new();
        for line in body.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(bracket_end) = line.find(']') {
                let text = line[bracket_end + 1..].trim();
                if !text.is_empty() {
                    lines.push(text);
                }
            } else {
                lines.push(line);
            }
        }
        if !lines.is_empty() {
            return lines.join(" ");
        }
    }
    raw.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_transcript() {
        let formatted = r#"========== TRANSMISSION STATUS REPORT ==========
STATUS: TRANSCRIPTION COMPLETE
FILE: sample.wav
MODEL: medium
LANGUAGE: en
------------------------------------------------------------
TELEGRAPH COPY FOLLOWS
0001 [0000.00s - 0002.50s] Hello world from hyperia audio.
END OF TRANSMISSION STOP
"#;
        let cleaned = clean_transcript(formatted);
        assert_eq!(cleaned, "Hello world from hyperia audio.");
    }
}
