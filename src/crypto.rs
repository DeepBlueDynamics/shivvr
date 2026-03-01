use anyhow::{bail, Result};
use ndarray::Array2;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Per-agent orthogonal encryption keys
pub struct AgentKeys {
    pub agent_id: String,
    /// Orthogonal matrix for organize embeddings (768×768)
    organize: Array2<f32>,
    /// Transpose of organize matrix (cached for decryption)
    organize_t: Array2<f32>,
    /// Optional orthogonal matrix for retrieve embeddings (1536×1536)
    retrieve: Option<Array2<f32>>,
    /// Transpose of retrieve matrix (cached)
    retrieve_t: Option<Array2<f32>>,
}

impl AgentKeys {
    /// Create from flattened row-major key data
    pub fn new(
        agent_id: &str,
        organize_data: &[f32],
        organize_dim: usize,
        retrieve_data: Option<&[f32]>,
        retrieve_dim: Option<usize>,
    ) -> Result<Self> {
        if organize_data.len() != organize_dim * organize_dim {
            bail!(
                "organize key length {} != {}^2",
                organize_data.len(),
                organize_dim
            );
        }

        let organize =
            Array2::from_shape_vec((organize_dim, organize_dim), organize_data.to_vec())?;
        let organize_t = organize.t().to_owned();

        let (retrieve, retrieve_t) = if let (Some(data), Some(dim)) = (retrieve_data, retrieve_dim)
        {
            if data.len() != dim * dim {
                bail!("retrieve key length {} != {}^2", data.len(), dim);
            }
            let mat = Array2::from_shape_vec((dim, dim), data.to_vec())?;
            let mat_t = mat.t().to_owned();
            (Some(mat), Some(mat_t))
        } else {
            (None, None)
        };

        Ok(Self {
            agent_id: agent_id.to_string(),
            organize,
            organize_t,
            retrieve,
            retrieve_t,
        })
    }

    /// Encrypt embedding: v @ Q (preserves cosine similarity)
    pub fn encrypt(&self, embedding: &[f32], role: &str) -> Vec<f32> {
        let mat = if role == "retrieve" {
            self.retrieve.as_ref().unwrap_or(&self.organize)
        } else {
            &self.organize
        };

        let dim = mat.nrows();
        if embedding.len() != dim {
            // Dimension mismatch — return unmodified
            return embedding.to_vec();
        }

        let v = ndarray::ArrayView1::from(embedding);
        let result = v.dot(mat);
        result.to_vec()
    }

    /// Decrypt embedding: v @ Q^T
    pub fn decrypt(&self, embedding: &[f32], role: &str) -> Vec<f32> {
        let mat_t = if role == "retrieve" {
            self.retrieve_t
                .as_ref()
                .unwrap_or(&self.organize_t)
        } else {
            &self.organize_t
        };

        let dim = mat_t.nrows();
        if embedding.len() != dim {
            return embedding.to_vec();
        }

        let v = ndarray::ArrayView1::from(embedding);
        let result = v.dot(mat_t);
        result.to_vec()
    }
}

/// Manages per-agent encryption keys with in-memory cache backed by sled
pub struct CryptoManager {
    db: sled::Db,
    cache: RwLock<HashMap<String, Arc<AgentKeys>>>,
}

impl CryptoManager {
    pub fn new(db: sled::Db) -> Self {
        Self {
            db,
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Register key matrices for an agent
    /// Keys are stored as raw bytes: [dim_u32_le][f32_le × dim²]
    pub fn register_keys(
        &self,
        agent_id: &str,
        organize_data: &[f32],
        organize_dim: usize,
        retrieve_data: Option<&[f32]>,
        retrieve_dim: Option<usize>,
    ) -> Result<()> {
        // Validate by constructing AgentKeys
        let keys = AgentKeys::new(
            agent_id,
            organize_data,
            organize_dim,
            retrieve_data,
            retrieve_dim,
        )?;

        // Store organize key
        let organize_key = format!("keys:{}:organize", agent_id);
        let organize_bytes = encode_key_bytes(organize_dim, organize_data);
        self.db.insert(organize_key.as_bytes(), organize_bytes)?;

        // Store retrieve key if provided
        if let (Some(data), Some(dim)) = (retrieve_data, retrieve_dim) {
            let retrieve_key = format!("keys:{}:retrieve", agent_id);
            let retrieve_bytes = encode_key_bytes(dim, data);
            self.db.insert(retrieve_key.as_bytes(), retrieve_bytes)?;
        }

        self.db.flush()?;

        // Update cache
        let mut cache = self.cache.write().unwrap();
        cache.insert(agent_id.to_string(), Arc::new(keys));

        Ok(())
    }

    /// Get keys for an agent (from cache or sled)
    pub fn get_keys(&self, agent_id: &str) -> Option<Arc<AgentKeys>> {
        // Check cache first
        {
            let cache = self.cache.read().unwrap();
            if let Some(keys) = cache.get(agent_id) {
                return Some(keys.clone());
            }
        }

        // Try loading from sled
        self.load_keys_from_db(agent_id)
    }

    fn load_keys_from_db(&self, agent_id: &str) -> Option<Arc<AgentKeys>> {
        let organize_key = format!("keys:{}:organize", agent_id);
        let organize_bytes = self.db.get(organize_key.as_bytes()).ok()??;
        let (organize_dim, organize_data) = decode_key_bytes(&organize_bytes)?;

        let retrieve_key = format!("keys:{}:retrieve", agent_id);
        let (retrieve_data, retrieve_dim) =
            if let Some(bytes) = self.db.get(retrieve_key.as_bytes()).ok()? {
                let (dim, data) = decode_key_bytes(&bytes)?;
                (Some(data), Some(dim))
            } else {
                (None, None)
            };

        let keys = AgentKeys::new(
            agent_id,
            &organize_data,
            organize_dim,
            retrieve_data.as_deref(),
            retrieve_dim,
        )
        .ok()?;

        let keys = Arc::new(keys);
        let mut cache = self.cache.write().unwrap();
        cache.insert(agent_id.to_string(), keys.clone());

        Some(keys)
    }
}

/// Encode key as [dim_u32_le][f32_le × dim²]
fn encode_key_bytes(dim: usize, data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(4 + data.len() * 4);
    bytes.extend_from_slice(&(dim as u32).to_le_bytes());
    for &val in data {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Decode key from [dim_u32_le][f32_le × dim²]
fn decode_key_bytes(bytes: &[u8]) -> Option<(usize, Vec<f32>)> {
    if bytes.len() < 4 {
        return None;
    }
    let dim = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let expected = 4 + dim * dim * 4;
    if bytes.len() != expected {
        return None;
    }

    let data: Vec<f32> = bytes[4..]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    Some((dim, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small orthogonal matrix (rotation) for testing.
    /// For dim=4, uses a simple rotation in the (0,1) plane.
    fn make_rotation_matrix(dim: usize, angle: f32) -> Vec<f32> {
        let mut data = vec![0.0f32; dim * dim];
        // Start with identity
        for i in 0..dim {
            data[i * dim + i] = 1.0;
        }
        // Rotate in the (0,1) plane
        let c = angle.cos();
        let s = angle.sin();
        data[0 * dim + 0] = c;
        data[0 * dim + 1] = s;
        data[1 * dim + 0] = -s;
        data[1 * dim + 1] = c;
        data
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn norm(v: &[f32]) -> f32 {
        dot(v, v).sqrt()
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let na = norm(a);
        let nb = norm(b);
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        dot(a, b) / (na * nb)
    }

    // ===== AgentKeys unit tests =====

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let dim = 4;
        let key_data = make_rotation_matrix(dim, 0.7);
        let keys = AgentKeys::new("test", &key_data, dim, None, None).unwrap();

        let original = vec![1.0, 2.0, 3.0, 4.0];
        let encrypted = keys.encrypt(&original, "organize");
        let decrypted = keys.decrypt(&encrypted, "organize");

        // Should recover original within floating point tolerance
        for (a, b) in original.iter().zip(decrypted.iter()) {
            assert!((a - b).abs() < 1e-5, "roundtrip mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn encryption_changes_values() {
        let dim = 4;
        let key_data = make_rotation_matrix(dim, 0.7);
        let keys = AgentKeys::new("test", &key_data, dim, None, None).unwrap();

        let original = vec![1.0, 2.0, 3.0, 4.0];
        let encrypted = keys.encrypt(&original, "organize");

        // Encrypted should differ from original
        let differs = original
            .iter()
            .zip(encrypted.iter())
            .any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(differs, "encryption should change values");
    }

    #[test]
    fn cosine_similarity_preserved() {
        let dim = 8;
        let key_data = make_rotation_matrix(dim, 1.2);
        let keys = AgentKeys::new("test", &key_data, dim, None, None).unwrap();

        let a = vec![1.0, 0.5, -0.3, 0.8, 0.2, -0.1, 0.6, 0.4];
        let b = vec![0.3, 0.9, 0.1, -0.5, 0.7, 0.2, -0.4, 0.8];

        let sim_plain = cosine_sim(&a, &b);

        let a_enc = keys.encrypt(&a, "organize");
        let b_enc = keys.encrypt(&b, "organize");

        let sim_enc = cosine_sim(&a_enc, &b_enc);

        assert!(
            (sim_plain - sim_enc).abs() < 1e-5,
            "cosine similarity not preserved: {} vs {}",
            sim_plain,
            sim_enc
        );
    }

    #[test]
    fn norm_preserved_under_encryption() {
        let dim = 4;
        let key_data = make_rotation_matrix(dim, 2.1);
        let keys = AgentKeys::new("test", &key_data, dim, None, None).unwrap();

        let v = vec![0.5, -0.3, 0.8, 0.1];
        let v_enc = keys.encrypt(&v, "organize");

        assert!(
            (norm(&v) - norm(&v_enc)).abs() < 1e-5,
            "norm not preserved: {} vs {}",
            norm(&v),
            norm(&v_enc)
        );
    }

    #[test]
    fn dimension_mismatch_returns_original() {
        let dim = 4;
        let key_data = make_rotation_matrix(dim, 0.5);
        let keys = AgentKeys::new("test", &key_data, dim, None, None).unwrap();

        let wrong_dim = vec![1.0, 2.0, 3.0]; // 3d, not 4d
        let result = keys.encrypt(&wrong_dim, "organize");
        assert_eq!(result, wrong_dim, "should return original on dim mismatch");

        let result = keys.decrypt(&wrong_dim, "organize");
        assert_eq!(result, wrong_dim, "should return original on dim mismatch");
    }

    #[test]
    fn retrieve_role_falls_back_to_organize() {
        let dim = 4;
        let key_data = make_rotation_matrix(dim, 0.5);
        let keys = AgentKeys::new("test", &key_data, dim, None, None).unwrap();

        let v = vec![1.0, 2.0, 3.0, 4.0];
        let enc_organize = keys.encrypt(&v, "organize");
        let enc_retrieve = keys.encrypt(&v, "retrieve");

        // Without retrieve key, retrieve should fall back to organize
        assert_eq!(enc_organize, enc_retrieve);
    }

    #[test]
    fn separate_retrieve_key() {
        let dim = 4;
        let organize_key = make_rotation_matrix(dim, 0.5);
        let retrieve_key = make_rotation_matrix(dim, 1.5); // different angle
        let keys = AgentKeys::new(
            "test",
            &organize_key,
            dim,
            Some(&retrieve_key),
            Some(dim),
        )
        .unwrap();

        let v = vec![1.0, 2.0, 3.0, 4.0];
        let enc_organize = keys.encrypt(&v, "organize");
        let enc_retrieve = keys.encrypt(&v, "retrieve");

        // Different keys should produce different encryptions
        let differs = enc_organize
            .iter()
            .zip(enc_retrieve.iter())
            .any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(differs, "different keys should encrypt differently");

        // Both should roundtrip
        let dec_organize = keys.decrypt(&enc_organize, "organize");
        let dec_retrieve = keys.decrypt(&enc_retrieve, "retrieve");

        for (a, b) in v.iter().zip(dec_organize.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
        for (a, b) in v.iter().zip(dec_retrieve.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn bad_key_length_rejected() {
        let result = AgentKeys::new("test", &[1.0, 2.0, 3.0], 4, None, None);
        assert!(result.is_err(), "should reject wrong key length");
    }

    // ===== Key encoding/decoding tests =====

    #[test]
    fn encode_decode_roundtrip() {
        let dim = 3;
        let data: Vec<f32> = (0..9).map(|i| i as f32 * 0.1).collect();

        let bytes = encode_key_bytes(dim, &data);
        let (dec_dim, dec_data) = decode_key_bytes(&bytes).unwrap();

        assert_eq!(dec_dim, dim);
        assert_eq!(dec_data.len(), data.len());
        for (a, b) in data.iter().zip(dec_data.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn decode_too_short() {
        assert!(decode_key_bytes(&[0, 1, 2]).is_none());
    }

    #[test]
    fn decode_wrong_length() {
        // dim=2 means we need 4 + 4*4 = 20 bytes, give 24
        let mut bytes = vec![0u8; 24];
        bytes[0..4].copy_from_slice(&2u32.to_le_bytes());
        assert!(decode_key_bytes(&bytes).is_none());
    }

    // ===== CryptoManager tests =====

    fn temp_db() -> sled::Db {
        let path = std::env::temp_dir().join(format!("shivvr_test_{}", uuid::Uuid::new_v4()));
        sled::open(path).unwrap()
    }

    #[test]
    fn crypto_manager_register_and_get() {
        let db = temp_db();
        let mgr = CryptoManager::new(db);

        let dim = 4;
        let key_data = make_rotation_matrix(dim, 0.8);
        mgr.register_keys("acala", &key_data, dim, None, None)
            .unwrap();

        let keys = mgr.get_keys("acala");
        assert!(keys.is_some(), "should find registered keys");

        let keys = keys.unwrap();
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let enc = keys.encrypt(&v, "organize");
        let dec = keys.decrypt(&enc, "organize");
        for (a, b) in v.iter().zip(dec.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn crypto_manager_unknown_agent() {
        let db = temp_db();
        let mgr = CryptoManager::new(db);
        assert!(mgr.get_keys("nobody").is_none());
    }

    #[test]
    fn crypto_manager_sled_persistence() {
        let path = std::env::temp_dir().join(format!("shivvr_persist_{}", uuid::Uuid::new_v4()));

        let dim = 4;
        let key_data = make_rotation_matrix(dim, 1.1);

        // Register with one manager instance
        {
            let db = sled::open(&path).unwrap();
            let mgr = CryptoManager::new(db);
            mgr.register_keys("yamantaka", &key_data, dim, None, None)
                .unwrap();
        }

        // Open fresh manager (empty cache), should load from sled
        {
            let db = sled::open(&path).unwrap();
            let mgr = CryptoManager::new(db);
            let keys = mgr.get_keys("yamantaka");
            assert!(keys.is_some(), "should load keys from sled after restart");

            let keys = keys.unwrap();
            let v = vec![1.0, 0.0, 0.0, 0.0];
            let enc = keys.encrypt(&v, "organize");
            let dec = keys.decrypt(&enc, "organize");
            for (a, b) in v.iter().zip(dec.iter()) {
                assert!((a - b).abs() < 1e-5, "roundtrip after sled reload failed");
            }
        }
    }

    #[test]
    fn crypto_manager_with_retrieve_key() {
        let db = temp_db();
        let mgr = CryptoManager::new(db);

        let dim = 4;
        let organize = make_rotation_matrix(dim, 0.3);
        let retrieve = make_rotation_matrix(dim, 2.0);

        mgr.register_keys("trailokya", &organize, dim, Some(&retrieve), Some(dim))
            .unwrap();

        let keys = mgr.get_keys("trailokya").unwrap();
        let v = vec![0.5, -0.3, 0.8, 0.1];

        // Organize roundtrip
        let dec = keys.decrypt(&keys.encrypt(&v, "organize"), "organize");
        for (a, b) in v.iter().zip(dec.iter()) {
            assert!((a - b).abs() < 1e-5);
        }

        // Retrieve roundtrip
        let dec = keys.decrypt(&keys.encrypt(&v, "retrieve"), "retrieve");
        for (a, b) in v.iter().zip(dec.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
