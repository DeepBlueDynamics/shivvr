import array
import hashlib
import json
import os
import platform
import random
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


def _cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def _token_offsets(text, tokenizer):
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = encoded.get("offset_mapping")
    if not offsets:
        return []
    return offsets


def _char_span_to_token_span(start_char, end_char, offsets):
    start_token = None
    end_token = None
    for idx, (start, end) in enumerate(offsets):
        if end <= start_char:
            continue
        if start_token is None and end > start_char:
            start_token = idx
        if start < end_char:
            end_token = idx
        if start >= end_char:
            break
    if start_token is None:
        start_token = 0
    if end_token is None:
        end_token = max(start_token, len(offsets) - 1)
    return start_token, end_token + 1


def _token_span_to_text(text, offsets, start, end):
    start_char = offsets[start][0]
    end_char = offsets[end - 1][1]
    return text[start_char:end_char], start_char, end_char


def _embed_texts(texts, tokenizer, model, device, cache, local_cache, max_length):
    import torch

    embeddings = [None] * len(texts)
    missing_texts = []
    missing_indices = []
    missing_keys = []
    for idx, text in enumerate(texts):
        key = _hash_text(text)
        if key in local_cache:
            embeddings[idx] = local_cache[key]
            continue
        cached = _cache_get(cache, key)
        if cached is not None:
            local_cache[key] = cached
            embeddings[idx] = cached
            continue
        missing_texts.append(text)
        missing_indices.append(idx)
        missing_keys.append(key)

    if missing_texts:
        with torch.inference_mode():
            inputs = tokenizer(
                missing_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            vectors = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            vectors = vectors.detach().cpu().tolist()
        for idx, vec, key in zip(missing_indices, vectors, missing_keys):
            embeddings[idx] = vec
            local_cache[key] = vec
            _cache_put(cache, key, vec)
        cache.commit()

    return embeddings


def main():
    _install_resource_stub()
    payload = json.load(sys.stdin)
    text = payload.get("text", "")
    chunks = payload.get("chunks", [])
    max_tokens = int(payload.get("max_tokens", 32))
    min_tokens = int(payload.get("min_tokens", 8))
    max_extra_ratio = float(payload.get("max_extra_ratio", 0.75))
    attempts = int(payload.get("attempts", 6))
    model_name = payload.get("model", "gtr-base")
    num_steps = int(payload.get("num_steps", 20))
    max_length = int(payload.get("max_length", 32))

    if not text.strip() or not chunks:
        _emit({"event": "done"})
        return

    import torch
    import vec2text
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    corrector = vec2text.load_pretrained_corrector(model_name)
    if hasattr(corrector, "to"):
        corrector.to(device)
    if hasattr(corrector, "model") and hasattr(corrector.model, "eval"):
        corrector.model.eval()
    if hasattr(corrector, "inversion_trainer") and hasattr(corrector.inversion_trainer, "model"):
        if hasattr(corrector.inversion_trainer.model, "eval"):
            corrector.inversion_trainer.model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/gtr-t5-base", use_fast=True
    )

    offsets = _token_offsets(text, tokenizer)
    if not offsets:
        _emit({"event": "error", "message": "tokenizer did not return offsets"})
        return

    token_count = len(offsets)
    spans = []
    for chunk in chunks:
        start = int(chunk.get("start", 0))
        end = int(chunk.get("end", 0))
        spans.append(_char_span_to_token_span(start, end, offsets))

    cache = _open_cache()
    local_cache = {}
    final_embeds = [None] * len(spans)
    carry_forward = 0

    _log("process: optimizing boundaries while maintaining coverage")

    # Track final boundaries to ensure contiguous coverage
    final_spans = list(spans)  # Will be updated as we process

    for idx, (base_start, base_end) in enumerate(spans):
        # Start from where previous chunk ended (maintain contiguity)
        if idx > 0:
            base_start = final_spans[idx - 1][1]  # Start where prev ended
        base_len = max(min_tokens, base_end - base_start)
        if carry_forward > 0:
            base_end = min(token_count, base_end + carry_forward)
            base_len = max(min_tokens, base_end - base_start)
            carry_forward = 0

        prev_embed = None
        next_embed = None
        if idx > 0:
            prev_embed = final_embeds[idx - 1]
        if idx + 1 < len(spans):
            next_text, _, _ = _token_span_to_text(text, offsets, spans[idx + 1][0], spans[idx + 1][1])
            next_embed = _embed_texts(
                [next_text],
                tokenizer,
                corrector.inversion_trainer.call_embedding_model,
                device,
                cache,
                local_cache,
                max_length,
            )[0]

        base_text, base_char_start, base_char_end = _token_span_to_text(text, offsets, base_start, base_end)
        base_embed = _embed_texts(
            [base_text],
            tokenizer,
            corrector.inversion_trainer.call_embedding_model,
            device,
            cache,
            local_cache,
            max_length,
        )[0]

        # Combined score: average similarity to both neighbors
        def combined_score(embed):
            scores = []
            if prev_embed:
                scores.append(_cosine_similarity(embed, prev_embed))
            if next_embed:
                scores.append(_cosine_similarity(embed, next_embed))
            return sum(scores) / len(scores) if scores else 0.0

        best_start = base_start
        best_end = base_end
        best_embed = base_embed
        best_sim = combined_score(base_embed)

        # Try sliding start boundary backward (give tokens back to previous chunk)
        slide_back = 0
        if idx > 0:
            max_shift = min(int(max_extra_ratio * base_len), attempts)
            for step in range(1, max_shift + 1):
                cand_start = base_start - step
                cand_end = base_end
                if cand_start < 0 or cand_end - cand_start > max_tokens * 2:
                    break
                # Don't overlap with previous chunk's final position
                if idx > 0 and cand_start < spans[idx - 1][0]:
                    break
                cand_text, cand_char_start, cand_char_end = _token_span_to_text(
                    text, offsets, cand_start, cand_end
                )
                cand_embed = _embed_texts(
                    [cand_text],
                    tokenizer,
                    corrector.inversion_trainer.call_embedding_model,
                    device,
                    cache,
                    local_cache,
                    max_length,
                )[0]
                cand_sim = combined_score(cand_embed)
                if cand_sim >= best_sim + 0.001:
                    best_sim = cand_sim
                    best_start = cand_start
                    best_end = cand_end
                    best_embed = cand_embed
                    slide_back = step
                    _emit(
                        {
                            "event": "update",
                            "index": idx,
                            "start": cand_char_start,
                            "end": cand_char_end,
                            "score": cand_sim,
                        }
                    )
                else:
                    break

        # Try sliding start boundary forward (take tokens from previous chunk)
        slide_forward_start = 0
        max_shift = min(int(max_extra_ratio * base_len), attempts)
        for step in range(1, max_shift + 1):
            cand_start = best_start + step
            cand_end = best_end
            if cand_end - cand_start < min_tokens:
                break
            cand_text, cand_char_start, cand_char_end = _token_span_to_text(
                text, offsets, cand_start, cand_end
            )
            cand_embed = _embed_texts(
                [cand_text],
                tokenizer,
                corrector.inversion_trainer.call_embedding_model,
                device,
                cache,
                local_cache,
                max_length,
            )[0]
            cand_sim = combined_score(cand_embed)
            if cand_sim >= best_sim + 0.001:
                best_sim = cand_sim
                best_start = cand_start
                best_end = cand_end
                best_embed = cand_embed
                slide_forward_start = step
                _emit(
                    {
                        "event": "update",
                        "index": idx,
                        "start": cand_char_start,
                        "end": cand_char_end,
                        "score": cand_sim,
                    }
                )
            else:
                break

        # Try sliding end boundary forward (take tokens from next chunk)
        slide_forward_end = 0
        if idx + 1 < len(spans):
            max_shift = min(int(max_extra_ratio * base_len), attempts)
            for step in range(1, max_shift + 1):
                cand_start = best_start
                cand_end = best_end + step
                if cand_end > token_count or cand_end - cand_start > max_tokens * 2:
                    break
                cand_text, cand_char_start, cand_char_end = _token_span_to_text(
                    text, offsets, cand_start, cand_end
                )
                cand_embed = _embed_texts(
                    [cand_text],
                    tokenizer,
                    corrector.inversion_trainer.call_embedding_model,
                    device,
                    cache,
                    local_cache,
                    max_length,
                )[0]
                cand_sim = combined_score(cand_embed)
                if cand_sim >= best_sim + 0.001:
                    best_sim = cand_sim
                    best_start = cand_start
                    best_end = cand_end
                    best_embed = cand_embed
                    slide_forward_end = step
                    _emit(
                        {
                            "event": "update",
                            "index": idx,
                            "start": cand_char_start,
                            "end": cand_char_end,
                            "score": cand_sim,
                        }
                    )
                else:
                    break

        # Try shrinking end boundary (give tokens to next chunk)
        shrink_end = 0
        for step in range(1, attempts + 1):
            cand_start = best_start
            cand_end = best_end - step
            if cand_end - cand_start < min_tokens:
                break
            cand_text, cand_char_start, cand_char_end = _token_span_to_text(
                text, offsets, cand_start, cand_end
            )
            cand_embed = _embed_texts(
                [cand_text],
                tokenizer,
                corrector.inversion_trainer.call_embedding_model,
                device,
                cache,
                local_cache,
                max_length,
            )[0]
            cand_sim = combined_score(cand_embed)
            if cand_sim >= best_sim - 0.01:
                best_sim = cand_sim
                best_start = cand_start
                best_end = cand_end
                best_embed = cand_embed
                shrink_end = step
                _emit(
                    {
                        "event": "update",
                        "index": idx,
                        "start": cand_char_start,
                        "end": cand_char_end,
                        "score": cand_sim,
                    }
                )
            else:
                break

        chosen_len = best_end - best_start
        carry_forward = max(0, base_len - chosen_len)
        final_embeds[idx] = best_embed
        final_spans[idx] = (best_start, best_end)  # Update for next chunk to use

        final_text, final_start_char, final_end_char = _token_span_to_text(
            text, offsets, best_start, best_end
        )
        _emit(
            {
                "event": "update",
                "index": idx,
                "start": final_start_char,
                "end": final_end_char,
                "score": best_sim,
            }
        )

        _log(
            f"chunk {idx + 1}: len {chosen_len} back {slide_back} fwd_s {slide_forward_start} fwd_e {slide_forward_end} shrink {shrink_end} sim {best_sim:.3f}"
        )

    _emit({"event": "done"})
    cache.close()


if __name__ == "__main__":
    main()
