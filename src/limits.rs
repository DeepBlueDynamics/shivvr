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

/// Default budget, in seconds, for routes that run inference: `/embed`,
/// `/image/embed`, `/audio/*`, session/temp ingest and `/invert`.
/// Override with `SHIVVR_REQUEST_TIMEOUT_SECS`.
pub const DEFAULT_HEAVY_TIMEOUT_SECS: u64 = 120;
/// Budget for everything else (search, session admin, health). Never exceeds
/// the heavy budget.
pub const DEFAULT_LIGHT_TIMEOUT_SECS: u64 = 30;

/// `(heavy, light)` budgets in seconds from the raw `SHIVVR_REQUEST_TIMEOUT_SECS`
/// value. Unparseable or zero values fall back to the defaults.
pub fn timeout_budgets(raw: Option<&str>) -> (u64, u64) {
    let heavy = raw
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_HEAVY_TIMEOUT_SECS);
    (heavy, DEFAULT_LIGHT_TIMEOUT_SECS.min(heavy))
}

/// Budgets from the process environment.
pub fn request_timeouts() -> (std::time::Duration, std::time::Duration) {
    let (heavy, light) = timeout_budgets(std::env::var("SHIVVR_REQUEST_TIMEOUT_SECS").ok().as_deref());
    (
        std::time::Duration::from_secs(heavy),
        std::time::Duration::from_secs(light),
    )
}

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

    #[test]
    fn timeout_defaults_and_overrides() {
        assert_eq!(timeout_budgets(None), (120, 30));
        assert_eq!(timeout_budgets(Some("300")), (300, 30));
        // A short override caps the light budget too.
        assert_eq!(timeout_budgets(Some("10")), (10, 10));
        // Garbage and zero fall back to the defaults.
        assert_eq!(timeout_budgets(Some("soon")), (120, 30));
        assert_eq!(timeout_budgets(Some("0")), (120, 30));
        assert_eq!(timeout_budgets(Some(" 45 ")), (45, 30));
    }
}
