#!/usr/bin/env python3
"""
Functional test suite for shivvr.
Tests the full embed → invert → search pipeline and cleans up after itself.

Usage:
    python tests/functional.py [base_url]

Default base_url: https://shivvr-949870462453.us-central1.run.app
"""

import sys
import json
import math
import urllib.request
import urllib.error
import urllib.parse

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://shivvr-949870462453.us-central1.run.app"
SESSION = "functional-test-session"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

results = []


def req(method, path, body=None, expected_status=200):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r) as resp:
            status = resp.status
            raw = resp.read()
            body = json.loads(raw) if raw else {}
            return status, body
    except urllib.error.HTTPError as e:
        raw = e.read()
        body = json.loads(raw) if raw else {}
        return e.code, body


def check(name, cond, detail=""):
    tag = PASS if cond else FAIL
    results.append(cond)
    suffix = f"  {detail}" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    return cond


def section(title):
    print(f"\n-- {title}")


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ── 1. Health ────────────────────────────────────────────────────────────────
section("Health check")
status, health = req("GET", "/health")
check("returns 200", status == 200)
check("status ok", health.get("status") == "ok")
check("gpu active", health.get("gpu") is True, f"gpu={health.get('gpu')}")
check("encryption available", health.get("encryption_available") is True)
inversion_up = health.get("inversion_available") is True
check("inversion available", inversion_up, f"inversion_available={health.get('inversion_available')}")
embeddinggemma2_up = any(m.get("name") == "embeddinggemma-2" for m in health.get("models", []))

# ── 2. Session lifecycle ──────────────────────────────────────────────────────
section("Session lifecycle")

# Ensure clean slate
req("DELETE", f"/sessions/{SESSION}")
status, sessions = req("GET", "/sessions")
check("session list returns 200", status == 200)
check("session not present before test", SESSION not in sessions.get("sessions", []))

# ── 3. Ingest ────────────────────────────────────────────────────────────────
section("Ingest")
SENTENCES = [
    "The harbor at dawn glows amber beneath a copper sky.",
    "Machine learning models compress semantic meaning into dense vectors.",
    "A lone lighthouse keeper watches storm clouds gather on the horizon.",
]
embeddings = {}

for sentence in SENTENCES:
    status, resp = req("POST", f"/sessions/{SESSION}/ingest", {"text": sentence})
    ok = status == 200 and resp.get("chunks_created", 0) == 1
    check(f"ingest: '{sentence[:40]}...' " if len(sentence) > 40 else f"ingest: '{sentence}'", ok,
          f"chunks={resp.get('chunks_created')}")
    if ok:
        chunk = resp["chunks"][0]
        emb = chunk["embedding"]
        check(f"  embedding is 768d", len(emb) == 768, f"got {len(emb)}d")
        embeddings[sentence] = emb

status, info = req("GET", f"/sessions/{SESSION}")
check("session info: 3 chunks", info.get("chunks") == 3, f"chunks={info.get('chunks')}")

# ── 4. Search by text ────────────────────────────────────────────────────────
section("Text search")
status, resp = req("GET", f"/sessions/{SESSION}/search?q=harbor+dawn+amber")
check("search returns 200", status == 200)
top = resp["results"][0] if resp.get("results") else {}
check("top result is harbor sentence",
      "harbor" in top.get("text", "").lower(),
      f"got: '{top.get('text','')[:60]}'")
check("score > 0.5", top.get("score", 0) > 0.5, f"score={top.get('score'):.4f}")

status, resp = req("GET", f"/sessions/{SESSION}/search?q=machine+learning+vectors")
top2 = resp["results"][0] if resp.get("results") else {}
check("ML query -> ML sentence", "machine" in top2.get("text", "").lower(),
      f"got: '{top2.get('text','')[:60]}'")

# ── 5. Inversion pipeline ────────────────────────────────────────────────────
section("Inversion pipeline")

if not inversion_up:
    print(f"  [{SKIP}] Inverter not available — skipping inversion tests")
else:
    for sentence, emb in embeddings.items():
        status, inv = req("POST", "/invert", {"embedding": emb, "max_length": 64})
        ok_status = check(f"invert returns 200 for '{sentence[:35]}...' " if len(sentence) > 35 else
                          f"invert returns 200 for '{sentence}'",
                          status == 200, f"status={status}")
        if not ok_status:
            print(f"    error: {inv.get('error')}")
            continue

        inverted_text = inv.get("text", "")
        similarity = inv.get("similarity", 0.0)
        check(f"  inverted text non-empty", bool(inverted_text.strip()))
        check(f"  similarity > 0.5", similarity > 0.5, f"similarity={similarity:.4f}")
        print(f"    original:  {sentence}")
        print(f"    inverted:  {inverted_text}")
        print(f"    similarity: {similarity:.4f}")

        # Re-embed the inverted text and search — should still find original chunk
        status, search_resp = req("GET",
            f"/sessions/{SESSION}/search?q={urllib.parse.quote(inverted_text)}")
        if status == 200 and search_resp.get("results"):
            top_inv = search_resp["results"][0]
            # Check that the highest-scoring result contains key words from the original
            original_words = set(sentence.lower().split()) - {"the", "a", "an", "at", "on", "in", "of", "to", "and", "is"}
            result_words = set(top_inv.get("text", "").lower().split())
            overlap = len(original_words & result_words) / len(original_words) if original_words else 0
            check(f"  inverted text retrieves original (word overlap)",
                  overlap > 0.3 or top_inv.get("score", 0) > 0.4,
                  f"overlap={overlap:.2f}, score={top_inv.get('score',0):.4f}")

# -- 6. Embedding consistency ─────────────────────────────────────────────────
section("Embedding consistency")
# Ingest same text twice, embeddings should be identical (deterministic model)
status1, r1 = req("POST", f"/sessions/{SESSION}/ingest", {"text": "consistency check"})
status2, r2 = req("POST", f"/sessions/{SESSION}/ingest", {"text": "consistency check"})
if status1 == 200 and status2 == 200:
    e1 = r1["chunks"][0]["embedding"]
    e2 = r2["chunks"][0]["embedding"]
    sim = cosine(e1, e2)
    check("same text -> same embedding (cosine >= 0.9999)", sim >= 0.9999, f"cosine={sim:.6f}")

# ── 7. EmbeddingGemma 2 (/embed) ──────────────────────────────────────────────
section("EmbeddingGemma 2 (/embed)")

# Request validation: 400s for invalid task, invalid dimensions, and task on gtr-t5-base
status, r = req("POST", "/embed", {
    "model": "embeddinggemma-2",
    "texts": ["test"],
    "task": "invalid_task_name"
})
check("bad task returns 400", status == 400, f"status={status}")

status, r = req("POST", "/embed", {
    "model": "embeddinggemma-2",
    "texts": ["test"],
    "dimensions": 999
})
check("bad dimensions (999) returns 400", status == 400, f"status={status}")

status, r = req("POST", "/embed", {
    "model": "gtr-t5-base",
    "texts": ["test"],
    "task": "query"
})
check("task on gtr-t5-base returns 400", status == 400, f"status={status}")

if not embeddinggemma2_up:
    print(f"  [{SKIP}] embeddinggemma-2 not listed in /health — skipping model inference tests")
else:
    sample_text = "The harbor at dawn glows amber beneath a copper sky."

    # Default 768 dims and unit norm
    status, resp = req("POST", "/embed", {
        "model": "embeddinggemma-2",
        "texts": [sample_text]
    })
    ok = check("default returns 200", status == 200, f"status={status}")
    if ok:
        v = resp["vectors"][0]
        check("default is 768d", len(v) == 768, f"got {len(v)}d")
        v_norm = math.sqrt(sum(x * x for x in v))
        check("default has unit norm", math.isclose(v_norm, 1.0, abs_tol=1e-3), f"norm={v_norm:.4f}")

    # dimensions=128/256/512 return that length with unit norm
    for d in [128, 256, 512]:
        status, resp = req("POST", "/embed", {
            "model": "embeddinggemma-2",
            "texts": [sample_text],
            "dimensions": d
        })
        ok = check(f"dimensions={d} returns 200", status == 200, f"status={status}")
        if ok:
            v = resp["vectors"][0]
            check(f"dimensions={d} returns {d}d", len(v) == d, f"got {len(v)}d")
            v_norm = math.sqrt(sum(x * x for x in v))
            check(f"dimensions={d} has unit norm", math.isclose(v_norm, 1.0, abs_tol=1e-3), f"norm={v_norm:.4f}")

    # task=query vs task=document give different vectors
    status_q, resp_q = req("POST", "/embed", {
        "model": "embeddinggemma-2",
        "texts": [sample_text],
        "task": "query"
    })
    status_d, resp_d = req("POST", "/embed", {
        "model": "embeddinggemma-2",
        "texts": [sample_text],
        "task": "document"
    })
    ok_qd = check("task=query and task=document return 200", status_q == 200 and status_d == 200)
    if ok_qd:
        vq = resp_q["vectors"][0]
        vd = resp_d["vectors"][0]
        sim_qd = cosine(vq, vd)
        check("task=query vs task=document give different vectors", sim_qd < 0.999, f"cosine={sim_qd:.4f}")

    # Ranking sanity check: query ranks its matching passage above unrelated ones
    query_text = "harbor at dawn"
    match_passage = "The harbor at dawn glows amber beneath a copper sky."
    unrelated_passage = "Machine learning models compress semantic meaning into dense vectors."
    status_q, resp_q = req("POST", "/embed", {
        "model": "embeddinggemma-2",
        "texts": [query_text],
        "task": "query"
    })
    status_doc, resp_doc = req("POST", "/embed", {
        "model": "embeddinggemma-2",
        "texts": [match_passage, unrelated_passage],
        "task": "document"
    })
    ok_rank = check("ranking embed requests return 200", status_q == 200 and status_doc == 200)
    if ok_rank:
        vq = resp_q["vectors"][0]
        v_match = resp_doc["vectors"][0]
        v_unrel = resp_doc["vectors"][1]
        sim_match = cosine(vq, v_match)
        sim_unrel = cosine(vq, v_unrel)
        check("query ranks matching passage above unrelated",
              sim_match > sim_unrel,
              f"match={sim_match:.4f} > unrelated={sim_unrel:.4f}")

# ── 8. Cleanup ───────────────────────────────────────────────────────────────
section("Cleanup")
status, resp = req("DELETE", f"/sessions/{SESSION}")
check("delete returns 200", status == 200)
check("deleted chunks > 0", resp.get("deleted_chunks", 0) > 0,
      f"deleted={resp.get('deleted_chunks')}")

status, sessions = req("GET", "/sessions")
check("session gone after delete", SESSION not in sessions.get("sessions", []))

# ── Summary ──────────────────────────────────────────────────────────────────
total = len(results)
passed = sum(results)
failed = total - passed
print(f"\n{'='*50}")
print(f"  {passed}/{total} passed", end="")
if failed:
    print(f"  ({failed} FAILED)")
else:
    print("  — all good")
print(f"{'='*50}")
sys.exit(0 if failed == 0 else 1)
