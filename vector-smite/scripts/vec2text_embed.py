import array
import hashlib
import json
import os
import platform
import sqlite3
import sys


def _install_resource_stub():
    if platform.system() != "Windows":
        return
    if "resource" in sys.modules:
        return

    class _RUsage:
        ru_utime = 0.0
        ru_stime = 0.0
        ru_maxrss = 0
        ru_ixrss = 0
        ru_idrss = 0
        ru_isrss = 0
        ru_minflt = 0
        ru_majflt = 0
        ru_nswap = 0
        ru_inblock = 0
        ru_oublock = 0
        ru_msgsnd = 0
        ru_msgrcv = 0
        ru_nsignals = 0
        ru_nvcsw = 0
        ru_nivcsw = 0

    class _ResourceStub:
        RLIMIT_AS = 9
        RLIMIT_DATA = 2
        RLIMIT_STACK = 3
        RLIM_INFINITY = -1
        RUSAGE_SELF = 0

        @staticmethod
        def getrusage(_who):
            return _RUsage()

        @staticmethod
        def getrlimit(_resource):
            return (_ResourceStub.RLIM_INFINITY, _ResourceStub.RLIM_INFINITY)

        @staticmethod
        def setrlimit(_resource, _limits):
            return None

    sys.modules["resource"] = _ResourceStub()


def _emit(event):
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def _log(message):
    _emit({"event": "log", "message": message})


def _cache_path():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cache_dir = os.path.join(root, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "embeddings.sqlite")


def _hash_text(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _open_cache():
    path = _cache_path()
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings (hash TEXT PRIMARY KEY, dim INTEGER, data BLOB)"
    )
    return conn


def _cache_get(conn, key):
    row = conn.execute(
        "SELECT dim, data FROM embeddings WHERE hash = ?", (key,)
    ).fetchone()
    if not row:
        return None
    _, blob = row
    arr = array.array("f")
    arr.frombytes(blob)
    return list(arr)


def _cache_put(conn, key, embedding):
    arr = array.array("f", embedding)
    blob = arr.tobytes()
    conn.execute(
        "INSERT OR REPLACE INTO embeddings (hash, dim, data) VALUES (?, ?, ?)",
        (key, len(embedding), sqlite3.Binary(blob)),
    )


def main():
    _install_resource_stub()
    payload = json.load(sys.stdin)
    texts = payload.get("texts", [])
    model_name = payload.get("model", "gtr-base")
    max_length = int(payload.get("max_length", 32))
    batch_size = int(payload.get("batch_size", 16))

    if not texts:
        _emit({"event": "done"})
        return

    try:
        import torch
        import vec2text
    except Exception as exc:
        _emit({"event": "error", "message": str(exc)})
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = "cpu"
    if device == "cuda":
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = "cuda"

    conn = _open_cache()
    cache_hits = 0
    cache_misses = 0
    cache_keys = []
    for text in texts:
        cache_keys.append(_hash_text(text))

    cached = [None] * len(texts)
    for idx, key in enumerate(cache_keys):
        hit = _cache_get(conn, key)
        if hit is not None:
            cache_hits += 1
            cached[idx] = hit
        else:
            cache_misses += 1

    _log(f"received {len(texts)} texts | batch {batch_size} | max_len {max_length}")
    _log(f"device {device_name}")
    _log(f"cache hits {cache_hits} | misses {cache_misses}")

    try:
        _log("loading vec2text corrector")
        corrector = vec2text.load_pretrained_corrector(model_name)
        if hasattr(corrector, "to"):
            corrector.to(device)
        if hasattr(corrector, "model") and hasattr(corrector.model, "eval"):
            corrector.model.eval()
        if hasattr(corrector, "inversion_trainer") and hasattr(corrector.inversion_trainer, "model"):
            if hasattr(corrector.inversion_trainer.model, "eval"):
                corrector.inversion_trainer.model.eval()
        tokenizer = corrector.embedder_tokenizer
        _log("model ready, embedding batches")

        with torch.inference_mode():
            for idx, embedding in enumerate(cached):
                if embedding is not None:
                    _emit(
                        {
                            "event": "embedding",
                            "index": idx,
                            "embedding": embedding,
                        }
                    )

            missing_indices = [i for i, emb in enumerate(cached) if emb is None]
            for start in range(0, len(missing_indices), batch_size):
                batch_indices = missing_indices[start : start + batch_size]
                batch_texts = [texts[i] for i in batch_indices]
                _log(f"batch {batch_indices[0] + 1}-{batch_indices[-1] + 1}")
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(device)
                embeddings = corrector.inversion_trainer.call_embedding_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                )
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                for offset, vector in enumerate(embeddings):
                    idx = batch_indices[offset]
                    embedding = vector.detach().cpu().tolist()
                    _cache_put(conn, cache_keys[idx], embedding)
                    _emit(
                        {
                            "event": "embedding",
                            "index": idx,
                            "embedding": embedding,
                        }
                    )
            conn.commit()
        _log("embedding complete")
        _emit({"event": "done"})
    except Exception as exc:
        _emit({"event": "error", "message": str(exc)})
    finally:
        conn.close()


if __name__ == "__main__":
    main()
