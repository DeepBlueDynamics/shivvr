"""
Single-chunk negotiation using embedding similarity.

Algorithm:
1. Split chunk into halves to determine DIRECTION (which boundary to probe)
2. If left half is more similar to left neighbor -> probe left edge
3. Find smallest piece at that edge that belongs with neighbor
4. Give up only that piece, not the whole half

The half comparison is just a signal for which direction to look.
Then we probe incrementally to find the actual boundary.
"""
import json
import platform
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
    chunks = payload.get("chunks", [])
    focus_chunk = int(payload.get("focus_chunk", 0))
    min_tokens = int(payload.get("min_tokens", 4))
    model_name = payload.get("model", "gtr-base")
    max_length = int(payload.get("max_length", 32))
    max_give = int(payload.get("max_give", 6))  # max words to give up per side

    if not text.strip() or len(chunks) < 1:
        _emit({"event": "done"})
        return

    if focus_chunk < 0 or focus_chunk >= len(chunks):
        _emit({"event": "error", "message": f"invalid focus_chunk: {focus_chunk}"})
        return

    import torch
    import vec2text
    from transformers import AutoTokenizer

    _log("warming engines...")

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
    embedder = corrector.inversion_trainer.call_embedding_model

    _log("engines ready")

    def get_embedding(chunk_text):
        """Get normalized embedding for text."""
        if not chunk_text.strip():
            return None
        with torch.inference_mode():
            inputs = tokenizer(
                [chunk_text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            emb = embedder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb

    def cosine_sim(emb1, emb2):
        """Cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        return torch.nn.functional.cosine_similarity(emb1, emb2, dim=1).item()

    # Convert chunks to working boundaries
    boundaries = []
    for chunk in chunks:
        boundaries.append({
            "start": chunk["start"],
            "end": chunk["end"],
        })

    focus = boundaries[focus_chunk]
    original_start = focus["start"]
    original_end = focus["end"]
    current_text = text[original_start:original_end]
    words = current_text.split()

    _log(f"chunk {focus_chunk + 1}: \"{current_text[:40]}...\" ({len(words)} words)")

    if len(words) < min_tokens * 2:
        _log("too short to negotiate")
        _emit({"event": "done"})
        return

    # Get neighbor texts and embeddings
    left_neighbor_emb = None
    right_neighbor_emb = None
    left_text = ""
    right_text = ""

    if focus_chunk > 0:
        left_b = boundaries[focus_chunk - 1]
        left_text = text[left_b["start"]:left_b["end"]]
        left_neighbor_emb = get_embedding(left_text)
        _log(f"left neighbor: \"{left_text[:30]}...\"")

    if focus_chunk < len(boundaries) - 1:
        right_b = boundaries[focus_chunk + 1]
        right_text = text[right_b["start"]:right_b["end"]]
        right_neighbor_emb = get_embedding(right_text)
        _log(f"right neighbor: \"{right_text[:30]}...\"")

    # Get embedding of full chunk and its halves to determine direction
    mid = len(words) // 2
    left_half = " ".join(words[:mid])
    right_half = " ".join(words[mid:])

    left_half_emb = get_embedding(left_half)
    right_half_emb = get_embedding(right_half)

    current_start = original_start
    current_end = original_end

    # Check LEFT boundary: is left half pulled toward left neighbor?
    if left_neighbor_emb is not None and left_half_emb is not None and right_half_emb is not None:
        sim_to_neighbor = cosine_sim(left_half_emb, left_neighbor_emb)
        sim_to_self = cosine_sim(left_half_emb, right_half_emb)

        _log(f"left half -> neighbor: {sim_to_neighbor:.3f}, -> self: {sim_to_self:.3f}")

        if sim_to_neighbor > sim_to_self:
            _log("left edge is pulled toward neighbor - probing...")

            # Probe incrementally: 1 word, 2 words, etc.
            best_give = 0
            for n_words in range(1, min(max_give + 1, len(words) - min_tokens + 1)):
                prefix = " ".join(words[:n_words])
                remainder = " ".join(words[n_words:])

                if len(remainder.split()) < min_tokens:
                    break

                prefix_emb = get_embedding(prefix)
                remainder_emb = get_embedding(remainder)

                if prefix_emb is None or remainder_emb is None:
                    continue

                # Does this prefix belong more with neighbor than with remainder?
                sim_prefix_neighbor = cosine_sim(prefix_emb, left_neighbor_emb)
                sim_prefix_remainder = cosine_sim(prefix_emb, remainder_emb)

                _log(f"  {n_words} words \"{prefix[:20]}...\" -> neighbor: {sim_prefix_neighbor:.3f}, -> remainder: {sim_prefix_remainder:.3f}")

                if sim_prefix_neighbor > sim_prefix_remainder:
                    best_give = n_words
                else:
                    # Stop probing once we find words that belong with us
                    break

            if best_give > 0:
                # Find character position after best_give words
                give_text = " ".join(words[:best_give])
                # Find where this text ends in original
                pos = current_text.find(words[best_give]) if best_give < len(words) else len(current_text)
                new_start = original_start + pos

                _log(f"GIVING {best_give} words to left: \"{give_text}\"")
                _emit({
                    "event": "give",
                    "direction": "left",
                    "words": best_give,
                    "text": give_text[:50],
                    "to_chunk": focus_chunk - 1,
                })
                current_start = new_start
                boundaries[focus_chunk - 1]["end"] = new_start

    # Update current text after possible left trim
    current_text = text[current_start:current_end]
    words = current_text.split()

    # Check RIGHT boundary: is right half pulled toward right neighbor?
    if right_neighbor_emb is not None and len(words) >= min_tokens * 2:
        mid = len(words) // 2
        left_half = " ".join(words[:mid])
        right_half = " ".join(words[mid:])

        left_half_emb = get_embedding(left_half)
        right_half_emb = get_embedding(right_half)

        if right_half_emb is not None and left_half_emb is not None:
            sim_to_neighbor = cosine_sim(right_half_emb, right_neighbor_emb)
            sim_to_self = cosine_sim(right_half_emb, left_half_emb)

            _log(f"right half -> neighbor: {sim_to_neighbor:.3f}, -> self: {sim_to_self:.3f}")

            if sim_to_neighbor > sim_to_self:
                _log("right edge is pulled toward neighbor - probing...")

                # Probe incrementally from the end
                best_give = 0
                for n_words in range(1, min(max_give + 1, len(words) - min_tokens + 1)):
                    suffix = " ".join(words[-n_words:])
                    remainder = " ".join(words[:-n_words])

                    if len(remainder.split()) < min_tokens:
                        break

                    suffix_emb = get_embedding(suffix)
                    remainder_emb = get_embedding(remainder)

                    if suffix_emb is None or remainder_emb is None:
                        continue

                    # Does this suffix belong more with neighbor than with remainder?
                    sim_suffix_neighbor = cosine_sim(suffix_emb, right_neighbor_emb)
                    sim_suffix_remainder = cosine_sim(suffix_emb, remainder_emb)

                    _log(f"  {n_words} words \"...{suffix[-20:]}\" -> neighbor: {sim_suffix_neighbor:.3f}, -> remainder: {sim_suffix_remainder:.3f}")

                    if sim_suffix_neighbor > sim_suffix_remainder:
                        best_give = n_words
                    else:
                        break

                if best_give > 0:
                    # Find character position before last best_give words
                    give_text = " ".join(words[-best_give:])
                    remainder_text = " ".join(words[:-best_give])
                    # Find where remainder ends
                    new_end = current_start + len(remainder_text.rstrip())
                    # Adjust for whitespace
                    while new_end < current_end and text[new_end] in ' \t\n':
                        new_end += 1

                    _log(f"GIVING {best_give} words to right: \"{give_text}\"")
                    _emit({
                        "event": "give",
                        "direction": "right",
                        "words": best_give,
                        "text": give_text[:50],
                        "to_chunk": focus_chunk + 1,
                    })
                    current_end = new_end
                    boundaries[focus_chunk + 1]["start"] = new_end

    # Update focus chunk
    boundaries[focus_chunk]["start"] = current_start
    boundaries[focus_chunk]["end"] = current_end

    # Report
    total_given = (current_start - original_start) + (original_end - current_end)
    if total_given > 0:
        _log(f"chunk {focus_chunk + 1} adjusted boundaries")
    else:
        _log(f"chunk {focus_chunk + 1}: boundaries are good")

    # Emit final boundaries
    for idx, b in enumerate(boundaries):
        _emit({
            "event": "final",
            "index": idx,
            "start": b["start"],
            "end": b["end"],
            "changed": idx == focus_chunk or idx == focus_chunk - 1 or idx == focus_chunk + 1,
        })

    _emit({"event": "done"})


if __name__ == "__main__":
    main()
