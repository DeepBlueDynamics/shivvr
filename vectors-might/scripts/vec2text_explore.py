"""
Monte Carlo exploration of chunk boundaries using vec2text round-trip scoring.

For each sample:
1. Pick random start position and length
2. Embed the text span
3. Invert embedding back to text via vec2text
4. Re-embed the inverted text
5. Score = cosine similarity between original and reconstructed embeddings

High scores = coherent semantic units that survive the round-trip.
"""
import json
import platform
import random
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


def main():
    _install_resource_stub()
    payload = json.load(sys.stdin)
    text = payload.get("text", "")
    num_samples = int(payload.get("num_samples", 100))
    min_tokens = int(payload.get("min_tokens", 8))
    max_tokens = int(payload.get("max_tokens", 64))
    model_name = payload.get("model", "gtr-base")
    num_steps = int(payload.get("num_steps", 20))
    max_length = int(payload.get("max_length", 32))
    seed = payload.get("seed", None)

    if seed is not None:
        random.seed(seed)

    if not text.strip():
        _emit({"event": "done"})
        return

    import torch
    import vec2text
    from transformers import AutoTokenizer

    _log("loading vec2text model...")

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

    # Get token offsets for the full text
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = encoded.get("offset_mapping", [])
    if not offsets:
        _emit({"event": "error", "message": "tokenizer returned no offsets"})
        return

    token_count = len(offsets)
    _log(f"text has {token_count} tokens, sampling {num_samples} random spans...")

    embedder = corrector.inversion_trainer.call_embedding_model

    results = []

    for i in range(num_samples):
        # Pick random start and length
        length = random.randint(min_tokens, min(max_tokens, token_count))
        start_token = random.randint(0, max(0, token_count - length))
        end_token = min(start_token + length, token_count)

        # Get character positions
        start_char = offsets[start_token][0]
        end_char = offsets[end_token - 1][1]
        span_text = text[start_char:end_char]

        if not span_text.strip():
            continue

        with torch.inference_mode():
            # Embed original
            inputs = tokenizer(
                [span_text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            orig_embed = embedder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )
            orig_embed = torch.nn.functional.normalize(orig_embed, p=2, dim=1)

            # Invert to text
            inverted = vec2text.invert_embeddings(
                embeddings=orig_embed,
                corrector=corrector,
                num_steps=num_steps,
            )[0]

            # Re-embed inverted text
            recon_inputs = tokenizer(
                [inverted],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            recon_embed = embedder(
                input_ids=recon_inputs.input_ids,
                attention_mask=recon_inputs.attention_mask,
            )
            recon_embed = torch.nn.functional.normalize(recon_embed, p=2, dim=1)

            # Score = cosine similarity
            score = torch.nn.functional.cosine_similarity(
                orig_embed, recon_embed, dim=1
            ).item()

        result = {
            "start": start_char,
            "end": end_char,
            "start_token": start_token,
            "end_token": end_token,
            "length": end_token - start_token,
            "score": score,
            "preview": span_text[:50].replace("\n", " "),
            "inverted": inverted[:50].replace("\n", " "),
        }
        results.append(result)

        _emit({
            "event": "sample",
            "index": i,
            **result,
        })

        if (i + 1) % 10 == 0:
            _log(f"sampled {i + 1}/{num_samples}")

    # Sort by score to find best spans
    results.sort(key=lambda x: x["score"], reverse=True)

    _log(f"done. best score: {results[0]['score']:.3f}, worst: {results[-1]['score']:.3f}")

    # Emit top results
    for rank, r in enumerate(results[:10]):
        _emit({
            "event": "top",
            "rank": rank + 1,
            "start": r["start"],
            "end": r["end"],
            "score": r["score"],
            "preview": r["preview"],
        })

    _emit({"event": "done"})


if __name__ == "__main__":
    main()
