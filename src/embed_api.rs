//! Plain text embedding (`POST /embed`): request/response types and validation.
//!
//! Kept free of the `ml` feature so validation is unit-testable without ONNX.
//! The handler itself lives in `api.rs` and uses the GTR-T5 `organize`
//! embedder — the same vector the ingest paths store in `Chunk::embedding`.

use serde::{Deserialize, Serialize};

/// The only text model `/embed` serves today.
pub const EMBED_MODEL: &str = "gtr-t5-base";
/// Output dimension of [`EMBED_MODEL`].
pub const EMBED_DIM: usize = 768;
/// Maximum number of texts per request.
pub const MAX_TEXTS: usize = 256;
/// Maximum size of one text, in bytes (UTF-8).
pub const MAX_TEXT_BYTES: usize = 32 * 1024;

#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    pub texts: Vec<String>,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EmbedResponse {
    pub model: String,
    pub dim: usize,
    pub vectors: Vec<Vec<f32>>,
}

/// Validate an `/embed` request. Returns a human-readable error on rejection.
pub fn validate(req: &EmbedRequest) -> Result<(), String> {
    if let Some(model) = req.model.as_deref() {
        if model != EMBED_MODEL {
            return Err(format!(
                "unsupported model {model:?}; this endpoint serves {EMBED_MODEL:?}"
            ));
        }
    }
    if req.texts.is_empty() {
        return Err("texts must contain at least one string".to_string());
    }
    if req.texts.len() > MAX_TEXTS {
        return Err(format!(
            "too many texts: {} (max {MAX_TEXTS})",
            req.texts.len()
        ));
    }
    for (i, text) in req.texts.iter().enumerate() {
        if text.trim().is_empty() {
            return Err(format!("texts[{i}] is empty"));
        }
        if text.len() > MAX_TEXT_BYTES {
            return Err(format!(
                "texts[{i}] is {} bytes (max {MAX_TEXT_BYTES})",
                text.len()
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(texts: Vec<&str>, model: Option<&str>) -> EmbedRequest {
        EmbedRequest {
            texts: texts.into_iter().map(String::from).collect(),
            model: model.map(String::from),
        }
    }

    #[test]
    fn accepts_default_and_explicit_model() {
        assert!(validate(&req(vec!["hello"], None)).is_ok());
        assert!(validate(&req(vec!["a", "b"], Some("gtr-t5-base"))).is_ok());
    }

    #[test]
    fn rejects_unknown_model() {
        let err = validate(&req(vec!["hello"], Some("siglip-base-patch16-224"))).unwrap_err();
        assert!(err.contains("unsupported model"));
    }

    #[test]
    fn rejects_empty_list_and_empty_text() {
        assert!(validate(&req(vec![], None)).is_err());
        assert!(validate(&req(vec!["ok", "  "], None))
            .unwrap_err()
            .contains("texts[1]"));
    }

    #[test]
    fn enforces_count_and_size_limits() {
        let many = vec!["x"; MAX_TEXTS];
        assert!(validate(&req(many, None)).is_ok());
        let too_many = vec!["x"; MAX_TEXTS + 1];
        assert!(validate(&req(too_many, None)).is_err());

        let max = "a".repeat(MAX_TEXT_BYTES);
        assert!(validate(&EmbedRequest { texts: vec![max], model: None }).is_ok());
        let over = "a".repeat(MAX_TEXT_BYTES + 1);
        assert!(validate(&EmbedRequest { texts: vec![over], model: None })
            .unwrap_err()
            .contains("bytes"));
    }

    #[test]
    fn deserializes_wire_shape() {
        let r: EmbedRequest = serde_json::from_str(r#"{"texts":["a"]}"#).unwrap();
        assert_eq!(r.texts, vec!["a"]);
        assert!(r.model.is_none());
        let body = serde_json::to_value(EmbedResponse {
            model: EMBED_MODEL.into(),
            dim: 2,
            vectors: vec![vec![1.0, 0.0]],
        })
        .unwrap();
        assert_eq!(body["model"], "gtr-t5-base");
        assert_eq!(body["dim"], 2);
        assert_eq!(body["vectors"][0][0], 1.0);
    }
}
