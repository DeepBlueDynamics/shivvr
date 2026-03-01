/// Cosine similarity (SIMD accelerated when available)
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "ml")]
    {
        use simsimd::SpatialSimilarity;
        // simsimd returns cosine DISTANCE (1 - similarity), convert to similarity
        1.0 - f32::cosine(a, b).unwrap_or(1.0) as f32
    }

    #[cfg(not(feature = "ml"))]
    {
        // Pure Rust fallback
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }
}
