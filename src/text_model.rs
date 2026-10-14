//! Text models for sessions, temp stores, search, MCP and the agent.
//!
//! EmbeddingGemma 2 is the default when it is loaded; GTR-T5-base is always
//! loaded and can be selected by name (it is the only model `/invert` can
//! reconstruct). EmbeddingGemma 2 gets its `document` prefix for stored text
//! and its `query` prefix for search queries; GTR takes no prefix.

use std::sync::Arc;

use crate::embed_api::{embeddinggemma2_prefix, EMBEDDINGGEMMA2_MODEL, EMBED_MODEL};
use crate::embedder::Embedder;
use crate::embeddinggemma::EmbeddingGemma2Embedder;

/// What a text is embedded for, which picks the EmbeddingGemma 2 prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    /// Text being stored (ingest).
    Document,
    /// A search query.
    Query,
}

/// Why a requested model can't be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    /// Not a text model this service knows (400).
    Unknown(String),
    /// A known model whose files are not loaded (503).
    Unavailable(String),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown(m) => write!(
                f,
                "unsupported model {m:?}; sessions and temp stores accept {EMBEDDINGGEMMA2_MODEL:?} (default) and {EMBED_MODEL:?}"
            ),
            Self::Unavailable(m) => write!(f, "model {m:?} is not loaded on this server (see /health)"),
        }
    }
}

pub struct TextEmbedders {
    pub gtr: Arc<Embedder>,
    pub gemma: Option<Arc<EmbeddingGemma2Embedder>>,
}

impl TextEmbedders {
    pub fn new(gtr: Arc<Embedder>, gemma: Option<Arc<EmbeddingGemma2Embedder>>) -> Self {
        Self { gtr, gemma }
    }

    /// The model used when a request names none: EmbeddingGemma 2 if loaded,
    /// GTR-T5-base otherwise.
    pub fn default_model(&self) -> &'static str {
        if self.gemma.is_some() {
            EMBEDDINGGEMMA2_MODEL
        } else {
            EMBED_MODEL
        }
    }

    /// Resolve a requested model name (`None`/blank = the default).
    pub fn resolve(&self, requested: Option<&str>) -> Result<&'static str, ResolveError> {
        match requested.map(str::trim).filter(|m| !m.is_empty()) {
            None => Ok(self.default_model()),
            Some(EMBED_MODEL) => Ok(EMBED_MODEL),
            Some(EMBEDDINGGEMMA2_MODEL) if self.gemma.is_some() => Ok(EMBEDDINGGEMMA2_MODEL),
            Some(EMBEDDINGGEMMA2_MODEL) => Err(ResolveError::Unavailable(EMBEDDINGGEMMA2_MODEL.to_string())),
            Some(other) => Err(ResolveError::Unknown(other.to_string())),
        }
    }

    /// Embed `text` with `model` (a name returned by [`Self::resolve`] or
    /// stored on a session). Returns an L2-normalized 768d vector.
    pub fn embed(&self, model: &str, purpose: Purpose, text: &str) -> anyhow::Result<Vec<f32>> {
        match model {
            EMBEDDINGGEMMA2_MODEL => {
                let gemma = self
                    .gemma
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("{}", ResolveError::Unavailable(model.to_string())))?;
                let task = match purpose {
                    Purpose::Document => "document",
                    Purpose::Query => "query",
                };
                let prefix = embeddinggemma2_prefix(Some(task)).map_err(|e| anyhow::anyhow!(e))?;
                gemma.embed_text(prefix, text)
            }
            EMBED_MODEL => self.gtr.embed(text),
            other => anyhow::bail!("{}", ResolveError::Unknown(other.to_string())),
        }
    }
}
