//! Integration tests for shivvr.
//!
//! These tests cover cross-module interactions using synthetic embeddings
//! (no ONNX model required). They run as part of `cargo test`.
//!
//! HTTP-layer tests (ingest/search endpoints) are not included here because
//! they require a live ONNX model. Start the server and use the shell scripts
//! in scripts/ to validate those paths end-to-end.

use chrono::Utc;
use shivvr::crypto::CryptoManager;
use shivvr::similarity::cosine_similarity;
use shivvr::store::{Chunk, Store};
use shivvr::temp_store::TempStore;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

fn make_chunk(id: &str, embedding: Vec<f32>) -> Chunk {
    Chunk {
        id: id.to_string(),
        text: format!("text for {}", id),
        embedding,
        embedding_retrieve: None,
        token_count: 10,
        source: Some("integration-test".to_string()),
        metadata: serde_json::json!({}),
        created_at: Utc::now(),
        emotion_primary: None,
        emotion_secondary: None,
        encrypted: false,
        agent_id: None,
    }
}

fn make_rotation_matrix_4d(angle: f32) -> Vec<f32> {
    let mut data = vec![0.0f32; 16];
    for i in 0..4 {
        data[i * 4 + i] = 1.0;
    }
    let c = angle.cos();
    let s = angle.sin();
    data[0] = c;
    data[1] = s;
    data[4] = -s;
    data[5] = c;
    data
}

// ---------------------------------------------------------------------------
// Store pipeline: ingest → search → validate
// ---------------------------------------------------------------------------

#[test]
fn store_ingest_search_top_result() {
    let store = Store::new();

    // Four chunks in 4D space, each pointing along a different axis
    let chunks = vec![
        make_chunk("north", l2_normalize(&[1.0, 0.0, 0.0, 0.0])),
        make_chunk("east", l2_normalize(&[0.0, 1.0, 0.0, 0.0])),
        make_chunk("south", l2_normalize(&[0.0, 0.0, 1.0, 0.0])),
        make_chunk("west", l2_normalize(&[0.0, 0.0, 0.0, 1.0])),
    ];
    store.add_chunks("sess", chunks, None).unwrap();

    // Query aligned with "east"
    let query = l2_normalize(&[0.0, 1.0, 0.0, 0.0]);
    let results = store.search("sess", &query, 4, None, 168.0, "organize").unwrap();

    assert_eq!(results.len(), 4, "should return all 4 chunks");
    assert_eq!(results[0].0.id, "east", "top result should be 'east'");
    assert!(
        results[0].1 > 0.999,
        "top score should be ~1.0, got {}",
        results[0].1
    );
    assert!(
        results[3].1.abs() < 0.01,
        "worst result (orthogonal) should be ~0.0, got {}",
        results[3].1
    );
}

#[test]
fn store_ingest_returns_n_results() {
    let store = Store::new();

    let chunks: Vec<Chunk> = (0..10)
        .map(|i| {
            let mut v = vec![0.0f32; 10];
            v[i] = 1.0;
            make_chunk(&format!("c{}", i), v)
        })
        .collect();
    store.add_chunks("large", chunks, None).unwrap();

    let query = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let results = store.search("large", &query, 3, None, 168.0, "organize").unwrap();

    assert_eq!(results.len(), 3, "n=3 should return exactly 3 results");
    assert_eq!(results[0].0.id, "c0");
}

#[test]
fn store_session_isolation() {
    let store = Store::new();

    store
        .add_chunks(
            "alpha",
            vec![make_chunk("a1", l2_normalize(&[1.0, 0.0, 0.0, 0.0]))],
            None,
        )
        .unwrap();
    store
        .add_chunks(
            "beta",
            vec![make_chunk("b1", l2_normalize(&[0.0, 1.0, 0.0, 0.0]))],
            None,
        )
        .unwrap();

    // alpha session should not see beta's chunk
    let alpha_chunks = store.get_chunks("alpha").unwrap();
    assert_eq!(alpha_chunks.len(), 1);
    assert_eq!(alpha_chunks[0].id, "a1");

    // searching alpha with beta's embedding should still only return alpha chunks
    let query = l2_normalize(&[0.0, 1.0, 0.0, 0.0]);
    let results = store.search("alpha", &query, 5, None, 168.0, "organize").unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0.id, "a1");
}

#[test]
fn store_delete_session_clears_state() {
    let store = Store::new();
    store
        .add_chunks(
            "s",
            vec![
                make_chunk("c1", l2_normalize(&[1.0, 0.0])),
                make_chunk("c2", l2_normalize(&[0.0, 1.0])),
            ],
            None,
        )
        .unwrap();

    assert_eq!(store.total_chunks().unwrap(), 2);

    let deleted = store.delete_session("s").unwrap();
    assert_eq!(deleted, 2);
    assert_eq!(store.total_chunks().unwrap(), 0);

    let query = vec![1.0, 0.0];
    let results = store.search("s", &query, 5, None, 168.0, "organize").unwrap();
    assert!(results.is_empty(), "deleted session should return no results");
}

// ---------------------------------------------------------------------------
// Retrieve-role fallback
// ---------------------------------------------------------------------------

#[test]
fn store_retrieve_role_uses_separate_embedding() {
    let store = Store::new();

    let mut chunk = make_chunk("dual", l2_normalize(&[1.0, 0.0, 0.0, 0.0]));
    chunk.embedding_retrieve = Some(l2_normalize(&[0.0, 1.0, 0.0, 0.0]));
    store.add_chunks("s", vec![chunk], None).unwrap();

    let query_organize = l2_normalize(&[1.0, 0.0, 0.0, 0.0]);
    let query_retrieve = l2_normalize(&[0.0, 1.0, 0.0, 0.0]);

    // organize role: hits on organize embedding
    let r = store.search("s", &query_organize, 1, None, 168.0, "organize").unwrap();
    assert!(r[0].1 > 0.99, "organize search should score ~1.0");

    // retrieve role: hits on retrieve embedding
    let r = store.search("s", &query_retrieve, 1, None, 168.0, "retrieve").unwrap();
    assert!(r[0].1 > 0.99, "retrieve search should score ~1.0");
}

// ---------------------------------------------------------------------------
// TempStore: add → search → sweep
// ---------------------------------------------------------------------------

#[test]
fn temp_store_add_search() {
    let ts = TempStore::new();

    let chunks = vec![
        make_chunk("t1", l2_normalize(&[1.0, 0.0, 0.0, 0.0])),
        make_chunk("t2", l2_normalize(&[0.0, 1.0, 0.0, 0.0])),
    ];
    ts.add_chunks("my-store", chunks).unwrap();

    let query = l2_normalize(&[1.0, 0.0, 0.0, 0.0]);
    let results = ts.search("my-store", &query, 2, None, 168.0, "organize").unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0.id, "t1");
}

#[test]
fn temp_store_delete_removes_chunks() {
    let ts = TempStore::new();

    ts.add_chunks("x", vec![make_chunk("c1", vec![1.0, 0.0])]).unwrap();
    assert_eq!(ts.get_chunks("x").unwrap().len(), 1);

    let deleted = ts.delete_store("x").unwrap();
    assert_eq!(deleted, 1);
    assert!(ts.get_chunks("x").unwrap().is_empty());
}

#[test]
fn temp_store_sweep_does_not_remove_fresh_stores() {
    let ts = TempStore::new();

    ts.add_chunks("fresh", vec![make_chunk("c1", vec![1.0, 0.0])]).unwrap();

    // Fresh store should not be swept (expires 2 hours from now)
    let removed = ts.sweep_expired();
    assert_eq!(removed, 0, "fresh store should not be swept");
    assert_eq!(ts.get_chunks("fresh").unwrap().len(), 1);
}

// ---------------------------------------------------------------------------
// CryptoManager + Store: encrypted search preserves ranking
// ---------------------------------------------------------------------------

#[test]
fn crypto_encrypted_chunks_preserve_search_ranking() {
    let crypto = CryptoManager::new();
    let store = Store::new();

    let dim = 4;
    let key_data = make_rotation_matrix_4d(0.9);
    crypto
        .register_keys("agent-x", &key_data, dim, None, None)
        .unwrap();
    let keys = crypto.get_keys("agent-x").unwrap();

    // Two plaintext embeddings
    let plain_best = l2_normalize(&[1.0, 0.0, 0.0, 0.0]);
    let plain_other = l2_normalize(&[0.7, 0.7, 0.0, 0.0]);

    // Encrypt before storing
    let enc_best = keys.encrypt(&plain_best, "organize");
    let enc_other = keys.encrypt(&plain_other, "organize");

    let mut c1 = make_chunk("best", enc_best.clone());
    c1.encrypted = true;
    c1.agent_id = Some("agent-x".to_string());

    let mut c2 = make_chunk("other", enc_other.clone());
    c2.encrypted = true;
    c2.agent_id = Some("agent-x".to_string());

    store.add_chunks("enc-session", vec![c1, c2], None).unwrap();

    // Encrypt the query too
    let plain_query = l2_normalize(&[1.0, 0.0, 0.0, 0.0]);
    let enc_query = keys.encrypt(&plain_query, "organize");

    let results = store
        .search("enc-session", &enc_query, 2, None, 168.0, "organize")
        .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0].0.id, "best",
        "encrypted search should still rank 'best' first"
    );
    assert!(
        results[0].1 > results[1].1,
        "scores should be correctly ordered"
    );
}

#[test]
fn crypto_cosine_similarity_preserved_under_rotation() {
    let crypto = CryptoManager::new();
    let dim = 4;
    let key_data = make_rotation_matrix_4d(1.3);
    crypto
        .register_keys("r-agent", &key_data, dim, None, None)
        .unwrap();
    let keys = crypto.get_keys("r-agent").unwrap();

    let a = l2_normalize(&[1.0, 2.0, 3.0, 4.0]);
    let b = l2_normalize(&[4.0, 3.0, 2.0, 1.0]);

    let sim_plain = cosine_similarity(&a, &b);

    let a_enc = keys.encrypt(&a, "organize");
    let b_enc = keys.encrypt(&b, "organize");
    let sim_enc = cosine_similarity(&a_enc, &b_enc);

    assert!(
        (sim_plain - sim_enc).abs() < 1e-5,
        "cosine similarity should be invariant under orthogonal rotation: {} vs {}",
        sim_plain,
        sim_enc
    );
}

// ---------------------------------------------------------------------------
// Multi-session: aggregate stats
// ---------------------------------------------------------------------------

#[test]
fn store_total_chunks_across_sessions() {
    let store = Store::new();

    store
        .add_chunks(
            "s1",
            vec![
                make_chunk("a", vec![1.0, 0.0]),
                make_chunk("b", vec![0.0, 1.0]),
            ],
            None,
        )
        .unwrap();
    store
        .add_chunks("s2", vec![make_chunk("c", vec![0.5, 0.5])], None)
        .unwrap();

    assert_eq!(store.total_chunks().unwrap(), 3);

    let sessions = store.list_sessions(None).unwrap();
    assert_eq!(sessions.len(), 2);
}
