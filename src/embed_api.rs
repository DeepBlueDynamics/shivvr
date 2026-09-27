//! Plain text embedding (`POST /embed`): request/response types and validation.
//!
//! Kept free of the `ml` feature so validation is unit-testable without ONNX.
//! The handler itself lives in `api.rs`. Two models are served:
//!
//! - [`EMBED_MODEL`] (GTR-T5-base, the default): the same vector the ingest
//!   paths store in `Chunk::embedding`.
//! - [`SIGLIP_TEXT_MODEL`]: the SigLIP text tower, which lives in the same
//!   space as `POST /image/embed` so text and images can be compared.

use serde::{Deserialize, Serialize};

/// Default text model: GTR-T5-base.
pub const EMBED_MODEL: &str = "gtr-t5-base";
/// Output dimension of [`EMBED_MODEL`].
pub const EMBED_DIM: usize = 768;
/// SigLIP text tower, same space as `/image/embed`.
pub const SIGLIP_TEXT_MODEL: &str = "siglip-base-patch16-224";
/// Output dimension of [`SIGLIP_TEXT_MODEL`].
pub const SIGLIP_TEXT_DIM: usize = 768;
/// Maximum number of texts per request.
pub const MAX_TEXTS: usize = 256;
/// Maximum size of one text, in bytes (UTF-8).
pub const MAX_TEXT_BYTES: usize = 32 * 1024;

/// Which model a `/embed` request resolved to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedModel {
    GtrT5Base,
    SiglipText,
}

impl EmbedModel {
    /// Resolve the optional `model` field. `None`/empty means the default.
    pub fn resolve(name: Option<&str>) -> Result<Self, String> {
        match name.map(str::trim) {
            None | Some("") | Some(EMBED_MODEL) => Ok(Self::GtrT5Base),
            Some(SIGLIP_TEXT_MODEL) => Ok(Self::SiglipText),
            Some(other) => Err(format!(
                "unsupported model {other:?}; this endpoint serves {EMBED_MODEL:?} and {SIGLIP_TEXT_MODEL:?}"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::GtrT5Base => EMBED_MODEL,
            Self::SiglipText => SIGLIP_TEXT_MODEL,
        }
    }

    pub fn dim(self) -> usize {
        match self {
            Self::GtrT5Base => EMBED_DIM,
            Self::SiglipText => SIGLIP_TEXT_DIM,
        }
    }
}

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

/// Validate an `/embed` request and resolve its model. Returns a
/// human-readable error on rejection.
pub fn validate(req: &EmbedRequest) -> Result<EmbedModel, String> {
    let model = EmbedModel::resolve(req.model.as_deref())?;
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
    Ok(model)
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
        assert_eq!(
            validate(&req(vec!["hello"], None)).unwrap(),
            EmbedModel::GtrT5Base
        );
        assert_eq!(
            validate(&req(vec!["a", "b"], Some("gtr-t5-base"))).unwrap(),
            EmbedModel::GtrT5Base
        );
        assert_eq!(
            validate(&req(vec!["x"], Some(""))).unwrap(),
            EmbedModel::GtrT5Base
        );
    }

    #[test]
    fn accepts_siglip_text_model() {
        let model = validate(&req(
            vec!["a photo of a cat"],
            Some("siglip-base-patch16-224"),
        ))
        .unwrap();
        assert_eq!(model, EmbedModel::SiglipText);
        assert_eq!(model.name(), SIGLIP_TEXT_MODEL);
        assert_eq!(model.dim(), 768);
        assert_eq!(EmbedModel::GtrT5Base.dim(), 768);
    }

    #[test]
    fn rejects_unknown_model() {
        let err = validate(&req(vec!["hello"], Some("bert-base-uncased"))).unwrap_err();
        assert!(err.contains("unsupported model"));
        assert!(err.contains("siglip-base-patch16-224"));
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
        assert!(validate(&EmbedRequest {
            texts: vec![max],
            model: None
        })
        .is_ok());
        let over = "a".repeat(MAX_TEXT_BYTES + 1);
        assert!(validate(&EmbedRequest {
            texts: vec![over],
            model: None
        })
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
