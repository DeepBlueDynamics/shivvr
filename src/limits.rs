//! Request budgets shared by the router and documented in README.md.
//!
//! Kept free of the `ml` feature so the pure helpers are unit-testable
//! without ONNX Runtime.

/// Routes that carry base64 media (`image_base64`, `audio_base64`):
/// session/temp ingest, `/image/embed`, `/audio/*`.
pub const MEDIA_BODY_LIMIT_BYTES: usize = 32 * 1024 * 1024;
/// `POST /embed`: up to 256 texts of 32 KiB each, plus JSON overhead.
pub const EMBED_BODY_LIMIT_BYTES: usize = 8 * 1024 * 1024;
/// Everything else keeps axum's default.
pub const DEFAULT_BODY_LIMIT_BYTES: usize = 2 * 1024 * 1024;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_budget_covers_a_rendered_page() {
        // A 300 DPI letter page as PNG is ~5-10 MB; base64 adds a third.
        assert!(MEDIA_BODY_LIMIT_BYTES >= 16 * 1024 * 1024);
        assert!(EMBED_BODY_LIMIT_BYTES >= 256 * 32 * 1024);
        assert!(DEFAULT_BODY_LIMIT_BYTES < EMBED_BODY_LIMIT_BYTES);
    }
}
