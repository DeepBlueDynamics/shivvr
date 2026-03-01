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


def _score_texts(texts, corrector, tokenizer, num_steps, max_length, device):
    import torch
    import vec2text

    if not texts:
        return []

    with torch.inference_mode():
        inputs = tokenizer(
            texts,
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
        recon_texts = vec2text.invert_embeddings(
            embeddings=embeddings,
            corrector=corrector,
            num_steps=num_steps,
        )
        recon_inputs = tokenizer(
            recon_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        recon_embeddings = corrector.inversion_trainer.call_embedding_model(
            input_ids=recon_inputs.input_ids,
            attention_mask=recon_inputs.attention_mask,
        )
        recon_embeddings = torch.nn.functional.normalize(recon_embeddings, p=2, dim=1)
        scores = torch.nn.functional.cosine_similarity(
            embeddings, recon_embeddings, dim=1
        ).cpu().tolist()
    return scores


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


def main():
    _install_resource_stub()
    payload = json.load(sys.stdin)
    text = payload.get("text", "")
    max_tokens = int(payload.get("max_tokens", 32))
    min_tokens = int(payload.get("min_tokens", 12))
    stride = int(payload.get("stride", max_tokens))
    shift = int(payload.get("shift", 4))
    lengths = payload.get("lengths") or [max_tokens]
    model_name = payload.get("model", "gtr-base")
    num_steps = int(payload.get("num_steps", 20))
    max_length = int(payload.get("max_length", 32))

    if not text.strip():
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
    cursor = 0
    while cursor < token_count:
        end = min(cursor + max_tokens, token_count)
        if end - cursor < min_tokens:
            break
        spans.append((cursor, end))
        if end >= token_count:
            break
        cursor += stride

    init_texts = []
    init_spans = []
    for start, end in spans:
        start_char = offsets[start][0]
        end_char = offsets[end - 1][1]
        init_spans.append((start_char, end_char))
        init_texts.append(text[start_char:end_char])

    init_scores = _score_texts(
        init_texts, corrector, tokenizer, num_steps, max_length, device
    )
    init_chunks = []
    for (start_char, end_char), score in zip(init_spans, init_scores):
        init_chunks.append({"start": start_char, "end": end_char, "score": score})
    _emit({"event": "init", "chunks": init_chunks})

    total = len(spans)
    for index, (start, end) in enumerate(spans):
        _log(f"segmenter chunk {index + 1}/{total}: scoring")
        candidates = []
        for delta in range(-shift, shift + 1):
            for length in lengths:
                if length < min_tokens:
                    continue
                cand_start = max(0, min(token_count - 1, start + delta))
                cand_end = min(token_count, cand_start + length)
                if cand_end - cand_start < min_tokens:
                    continue
                candidates.append((cand_start, cand_end))

        dedup = []
        seen = set()
        for cand in candidates:
            if cand not in seen:
                seen.add(cand)
                dedup.append(cand)
        candidates = dedup

        texts = []
        spans_char = []
        for cand_start, cand_end in candidates:
            start_char = offsets[cand_start][0]
            end_char = offsets[cand_end - 1][1]
            spans_char.append((start_char, end_char))
            texts.append(text[start_char:end_char])

        scores = _score_texts(
            texts, corrector, tokenizer, num_steps, max_length, device
        )
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_start, best_end = spans_char[best_idx]
        best_score = scores[best_idx]
        best_len = candidates[best_idx][1] - candidates[best_idx][0]
        _emit(
            {
                "event": "update",
                "index": index,
                "start": best_start,
                "end": best_end,
                "score": best_score,
            }
        )
        _log(
            f"segmenter chunk {index + 1}/{total}: best len {best_len} score {best_score:.3f}"
        )

    _emit({"event": "done"})


if __name__ == "__main__":
    main()
