//! Plain text embedding (`POST /embed`): request/response types and validation.
//!
//! Kept free of the `ml` feature so validation is unit-testable without ONNX.
//! The handler itself lives in `api.rs`. Three models are served:
//!
//! - [`EMBED_MODEL`] (GTR-T5-base, the default): the same vector the ingest
//!   paths store in `Chunk::embedding`.
//! - [`SIGLIP_TEXT_MODEL`]: the SigLIP text tower, which lives in the same
//!   space as `POST /image/embed` so text and images can be compared.
//! - [`EMBEDDINGGEMMA2_MODEL`]: EmbeddingGemma 2's text path. Takes an
//!   optional `task` (the prefix the model was trained with) and `dimensions`
//!   (Matryoshka truncation to 128/256/512/768).

use serde::{Deserialize, Serialize};

/// Default text model: GTR-T5-base.
pub const EMBED_MODEL: &str = "gtr-t5-base";
/// Output dimension of [`EMBED_MODEL`].
pub const EMBED_DIM: usize = 768;
/// SigLIP text tower, same space as `/image/embed`.
pub const SIGLIP_TEXT_MODEL: &str = "siglip-base-patch16-224";
/// Output dimension of [`SIGLIP_TEXT_MODEL`].
pub const SIGLIP_TEXT_DIM: usize = 768;
/// EmbeddingGemma 2 text path (the model is multimodal; text is served here).
pub const EMBEDDINGGEMMA2_MODEL: &str = "embeddinggemma-2";
/// Native output dimension of [`EMBEDDINGGEMMA2_MODEL`].
pub const EMBEDDINGGEMMA2_DIM: usize = 768;
/// Matryoshka dimensions EmbeddingGemma 2 was trained to truncate to.
pub const EMBEDDINGGEMMA2_DIMS: [usize; 4] = [128, 256, 512, 768];
/// Maximum number of texts per request.
pub const MAX_TEXTS: usize = 256;
/// Maximum size of one text, in bytes (UTF-8).
pub const MAX_TEXT_BYTES: usize = 32 * 1024;

/// EmbeddingGemma 2 task prefixes, keyed by the prompt names in the model's
/// `config_sentence_transformers.json`. Matched case-insensitively.
const EMBEDDINGGEMMA2_PROMPTS: &[(&str, &str)] = &[
    ("query", "task: search result | query: "),
    ("document", "title: none | text: "),
    ("Retrieval-query", "task: search result | query: "),
    ("Retrieval-document", "title: none | text: "),
    ("Retrieval", "task: search result | query: "),
    ("SearchQuery", "task: search result | query: "),
    ("Reranking", "task: search result | query: "),
    ("BitextMining", "task: search result | query: "),
    ("QuestionAnswering", "task: question answering | query: "),
    ("FactChecking", "task: fact checking | query: "),
    ("Classification", "task: classification | query: "),
    ("MultilabelClassification", "task: classification | query: "),
    ("Clustering", "task: clustering | query: "),
    ("STS", "task: sentence similarity | query: "),
    ("SentenceSimilarity", "task: sentence similarity | query: "),
    ("PairClassification", "task: sentence similarity | query: "),
    ("Summarization", "task: sentence similarity | query: "),
    ("CodeRetrieval", "task: code retrieval | query: "),
    ("InstructionRetrieval", "task: code retrieval | query: "),
];

/// Prefix for an EmbeddingGemma 2 `task`. `None` means no prefix, which is
/// what `SentenceTransformer.encode()` does without a prompt.
pub fn embeddinggemma2_prefix(task: Option<&str>) -> Result<&'static str, String> {
    let Some(task) = task.map(str::trim).filter(|t| !t.is_empty()) else {
        return Ok("");
    };
    EMBEDDINGGEMMA2_PROMPTS
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(task))
        .map(|(_, prefix)| *prefix)
        .ok_or_else(|| {
            let names: Vec<&str> = EMBEDDINGGEMMA2_PROMPTS.iter().map(|(n, _)| *n).collect();
            format!(
                "unsupported task {task:?}; {EMBEDDINGGEMMA2_MODEL} accepts {}",
                names.join(", ")
            )
        })
}

/// Truncate a unit vector to its first `dim` components and re-normalize
/// (Matryoshka). A no-op when `dim` is at least the full length.
pub fn truncate_normalize(mut v: Vec<f32>, dim: usize) -> Vec<f32> {
    if dim >= v.len() {
        return v;
    }
    v.truncate(dim);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= norm;
    }
    v
}

/// Which model a `/embed` request resolved to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedModel {
    GtrT5Base,
    SiglipText,
    EmbeddingGemma2,
}

impl EmbedModel {
    /// Resolve the optional `model` field. `None`/empty means the default.
    pub fn resolve(name: Option<&str>) -> Result<Self, String> {
        match name.map(str::trim) {
            None | Some("") | Some(EMBED_MODEL) => Ok(Self::GtrT5Base),
            Some(SIGLIP_TEXT_MODEL) => Ok(Self::SiglipText),
            Some(EMBEDDINGGEMMA2_MODEL) => Ok(Self::EmbeddingGemma2),
            Some(other) => Err(format!(
                "unsupported model {other:?}; this endpoint serves {EMBED_MODEL:?}, {SIGLIP_TEXT_MODEL:?} and {EMBEDDINGGEMMA2_MODEL:?}"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::GtrT5Base => EMBED_MODEL,
            Self::SiglipText => SIGLIP_TEXT_MODEL,
            Self::EmbeddingGemma2 => EMBEDDINGGEMMA2_MODEL,
        }
    }

    /// Native output dimension.
    pub fn dim(self) -> usize {
        match self {
            Self::GtrT5Base => EMBED_DIM,
            Self::SiglipText => SIGLIP_TEXT_DIM,
            Self::EmbeddingGemma2 => EMBEDDINGGEMMA2_DIM,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    pub texts: Vec<String>,
    #[serde(default)]
    pub model: Option<String>,
    /// EmbeddingGemma 2 only: prompt name selecting the task prefix
    /// (`query`, `document`, `Clustering`, ...).
    #[serde(default)]
    pub task: Option<String>,
    /// EmbeddingGemma 2 only: Matryoshka output size (128, 256, 512 or 768).
    #[serde(default)]
    pub dimensions: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct EmbedResponse {
    pub model: String,
    pub dim: usize,
    pub vectors: Vec<Vec<f32>>,
}

/// A validated `/embed` request: the model, the prefix to prepend to every
/// text, and the output dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbedPlan {
    pub model: EmbedModel,
    pub prefix: &'static str,
    pub dim: usize,
}

/// Validate an `/embed` request and resolve its model, prefix and output
/// dimension. Returns a human-readable error on rejection.
pub fn validate(req: &EmbedRequest) -> Result<EmbedPlan, String> {
    let model = EmbedModel::resolve(req.model.as_deref())?;
    let (prefix, dim) = match model {
        EmbedModel::EmbeddingGemma2 => {
            let prefix = embeddinggemma2_prefix(req.task.as_deref())?;
            let dim = match req.dimensions {
                None => EMBEDDINGGEMMA2_DIM,
                Some(d) if EMBEDDINGGEMMA2_DIMS.contains(&d) => d,
                Some(d) => {
                    return Err(format!(
                        "unsupported dimensions {d}; {EMBEDDINGGEMMA2_MODEL} supports {EMBEDDINGGEMMA2_DIMS:?}"
                    ))
                }
            };
            (prefix, dim)
        }
        EmbedModel::GtrT5Base | EmbedModel::SiglipText => {
            if req.task.as_deref().is_some_and(|t| !t.trim().is_empty()) {
                return Err(format!("task is only supported by {EMBEDDINGGEMMA2_MODEL:?}"));
            }
            if req.dimensions.is_some_and(|d| d != model.dim()) {
                return Err(format!(
                    "dimensions is only adjustable for {EMBEDDINGGEMMA2_MODEL:?}"
                ));
            }
            ("", model.dim())
        }
    };
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
    Ok(EmbedPlan { model, prefix, dim })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(texts: Vec<&str>, model: Option<&str>) -> EmbedRequest {
        EmbedRequest {
            texts: texts.into_iter().map(String::from).collect(),
            model: model.map(String::from),
            task: None,
            dimensions: None,
        }
    }

    fn gemma(task: Option<&str>, dimensions: Option<usize>) -> EmbedRequest {
        EmbedRequest {
            task: task.map(String::from),
            dimensions,
            ..req(vec!["hello"], Some(EMBEDDINGGEMMA2_MODEL))
        }
    }

    #[test]
    fn accepts_default_and_explicit_model() {
        assert_eq!(
            validate(&req(vec!["hello"], None)).unwrap().model,
            EmbedModel::GtrT5Base
        );
        assert_eq!(
            validate(&req(vec!["a", "b"], Some("gtr-t5-base"))).unwrap().model,
            EmbedModel::GtrT5Base
        );
        assert_eq!(
            validate(&req(vec!["x"], Some(""))).unwrap().model,
            EmbedModel::GtrT5Base
        );
    }

    #[test]
    fn accepts_siglip_text_model() {
        let plan = validate(&req(
            vec!["a photo of a cat"],
            Some("siglip-base-patch16-224"),
        ))
        .unwrap();
        assert_eq!(plan.model, EmbedModel::SiglipText);
        assert_eq!(plan.model.name(), SIGLIP_TEXT_MODEL);
        assert_eq!(plan.dim, 768);
        assert_eq!(plan.prefix, "");
        assert_eq!(EmbedModel::GtrT5Base.dim(), 768);
    }

    #[test]
    fn embeddinggemma2_defaults_to_no_prefix_and_full_dim() {
        let plan = validate(&gemma(None, None)).unwrap();
        assert_eq!(plan.model, EmbedModel::EmbeddingGemma2);
        assert_eq!(plan.model.name(), "embeddinggemma-2");
        assert_eq!(plan.prefix, "");
        assert_eq!(plan.dim, 768);
    }

    #[test]
    fn embeddinggemma2_resolves_tasks_case_insensitively() {
        let q = validate(&gemma(Some("query"), None)).unwrap();
        assert_eq!(q.prefix, "task: search result | query: ");
        let d = validate(&gemma(Some("Document"), None)).unwrap();
        assert_eq!(d.prefix, "title: none | text: ");
        let c = validate(&gemma(Some("coderetrieval"), None)).unwrap();
        assert_eq!(c.prefix, "task: code retrieval | query: ");
        let err = validate(&gemma(Some("poetry"), None)).unwrap_err();
        assert!(err.contains("unsupported task"));
    }

    #[test]
    fn embeddinggemma2_accepts_only_matryoshka_dims() {
        for d in EMBEDDINGGEMMA2_DIMS {
            assert_eq!(validate(&gemma(None, Some(d))).unwrap().dim, d);
        }
        assert!(validate(&gemma(None, Some(300))).unwrap_err().contains("dimensions"));
    }

    #[test]
    fn task_and_dimensions_rejected_for_other_models() {
        let mut r = req(vec!["hello"], None);
        r.task = Some("query".into());
        assert!(validate(&r).unwrap_err().contains("task"));
        let mut r = req(vec!["hello"], Some(SIGLIP_TEXT_MODEL));
        r.dimensions = Some(256);
        assert!(validate(&r).unwrap_err().contains("dimensions"));
        // Asking for the native size is harmless.
        r.dimensions = Some(768);
        assert!(validate(&r).is_ok());
    }

    #[test]
    fn truncate_normalize_renormalizes_prefix() {
        let v = vec![0.6, 0.0, 0.8, 0.0];
        let t = truncate_normalize(v.clone(), 2);
        assert_eq!(t.len(), 2);
        assert!((t[0] - 1.0).abs() < 1e-6);
        assert_eq!(truncate_normalize(v.clone(), 4), v);
    }

    #[test]
    fn rejects_unknown_model() {
        let err = validate(&req(vec!["hello"], Some("bert-base-uncased"))).unwrap_err();
        assert!(err.contains("unsupported model"));
        assert!(err.contains("siglip-base-patch16-224"));
        assert!(err.contains("embeddinggemma-2"));
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
        assert!(validate(&req(vec![&max], None)).is_ok());
        let over = "a".repeat(MAX_TEXT_BYTES + 1);
        assert!(validate(&req(vec![&over], None))
            .unwrap_err()
            .contains("bytes"));
    }

    #[test]
    fn deserializes_wire_shape() {
        let r: EmbedRequest = serde_json::from_str(r#"{"texts":["a"]}"#).unwrap();
        assert_eq!(r.texts, vec!["a"]);
        assert!(r.model.is_none());
        assert!(r.task.is_none());
        assert!(r.dimensions.is_none());
        let r: EmbedRequest = serde_json::from_str(
            r#"{"texts":["a"],"model":"embeddinggemma-2","task":"query","dimensions":256}"#,
        )
        .unwrap();
        assert_eq!(r.task.as_deref(), Some("query"));
        assert_eq!(r.dimensions, Some(256));
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
