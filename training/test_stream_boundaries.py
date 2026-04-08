"""
MC-ablation stream chunker: slide a max-token window, ablate until fidelity pops.
"""

import argparse
import importlib
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch


VEC2TEXT = None


def _install_resource_stub() -> None:
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


def _import_vec2text():
    script_dir = Path(__file__).resolve().parent
    local_pkg = script_dir / "vec2text"
    if local_pkg.exists():
        sys.path = [p for p in sys.path if Path(p).resolve() != script_dir]
        sys.path = [p for p in sys.path if p not in ("", ".")]
    return importlib.import_module("vec2text")


def _get_vec2text():
    global VEC2TEXT
    if VEC2TEXT is None:
        _install_resource_stub()
        VEC2TEXT = _import_vec2text()
    return VEC2TEXT


def _get_device(corrector) -> torch.device:
    if hasattr(corrector, "device"):
        return corrector.device
    if hasattr(corrector, "parameters"):
        try:
            return next(corrector.parameters()).device
        except StopIteration:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_corrector(model_name: str):
    vec2text_mod = _get_vec2text()
    fn = getattr(vec2text_mod, "load_pretrained_corrector", None)
    if fn is None:
        api_mod = importlib.import_module("vec2text.api")
        fn = getattr(api_mod, "load_pretrained_corrector", None)
    if fn is None:
        raise RuntimeError("vec2text.load_pretrained_corrector not found.")

    return fn(model_name)


def _embed_text(text: str, corrector, max_length: int) -> torch.Tensor:
    device = _get_device(corrector)
    inputs = corrector.embedder_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    with torch.inference_mode():
        embeddings = corrector.inversion_trainer.call_embedding_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def _reconstruction_quality(text: str, corrector, num_steps: int, max_length: int) -> float:
    vec2text_mod = _get_vec2text()
    embedding = _embed_text(text, corrector, max_length)
    reconstructed = vec2text_mod.invert_embeddings(
        embeddings=embedding,
        corrector=corrector,
        num_steps=num_steps,
    )[0]
    recon_embedding = _embed_text(reconstructed, corrector, max_length)
    return torch.nn.functional.cosine_similarity(embedding, recon_embedding).item()


def _sample_mask(tokens: List[str], rng: random.Random, drop_rate: float, keep_k: Optional[int]):
    n = len(tokens)
    if keep_k is not None:
        keep_k = max(1, min(keep_k, n))
        kept = set(rng.sample(range(n), keep_k))
        return [i in kept for i in range(n)]

    mask = [rng.random() >= drop_rate for _ in range(n)]
    if not any(mask):
        mask[rng.randrange(n)] = True
    return mask


def _mc_token_deltas(
    tokens: List[str],
    corrector,
    num_steps: int,
    max_length: int,
    trials: int,
    drop_rate: float,
    keep_k: Optional[int],
    seed: int,
) -> List[float]:
    present_sum = [0.0] * len(tokens)
    present_count = [0] * len(tokens)
    absent_sum = [0.0] * len(tokens)
    absent_count = [0] * len(tokens)

    rng = random.Random(seed)
    for _ in range(trials):
        mask = _sample_mask(tokens, rng, drop_rate, keep_k)
        kept_tokens = [tok for tok, keep in zip(tokens, mask) if keep]
        text = " ".join(kept_tokens)
        quality = _reconstruction_quality(text, corrector, num_steps, max_length)

        for idx, keep in enumerate(mask):
            if keep:
                present_sum[idx] += quality
                present_count[idx] += 1
            else:
                absent_sum[idx] += quality
                absent_count[idx] += 1

    deltas = []
    for idx in range(len(tokens)):
        p_count = max(1, present_count[idx])
        a_count = max(1, absent_count[idx])
        present_mean = present_sum[idx] / p_count
        absent_mean = absent_sum[idx] / a_count
        deltas.append(present_mean - absent_mean)

    return deltas


def _best_span(deltas: List[float], min_len: int, max_len: int) -> Tuple[int, int, float]:
    n = len(deltas)
    if n == 0:
        return 0, 0, 0.0
    max_len = max(min_len, min(max_len, n))
    prefix = [0.0]
    for value in deltas:
        prefix.append(prefix[-1] + value)

    best = (0, min_len, float("-inf"))
    for start in range(n):
        max_end = min(n, start + max_len)
        for end in range(start + min_len, max_end + 1):
            total = prefix[end] - prefix[start]
            mean = total / (end - start)
            if mean > best[2]:
                best = (start, end, mean)

    return best


def _clean_text(text: str) -> str:
    text = text.replace("|", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _snip(tokens: List[str], limit: int) -> str:
    text = " ".join(tokens)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path", type=str, default="hn_frontpage.md")
    parser.add_argument("--model", type=str, default="gtr-base")
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--stride_tokens", type=int, default=16)
    parser.add_argument("--target_quality", type=float, default=0.9)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--mc_trials", type=int, default=20)
    parser.add_argument("--mc_drop_rate", type=float, default=0.3)
    parser.add_argument("--mc_keep_k", type=int, default=0)
    parser.add_argument("--min_tokens", type=int, default=8)
    parser.add_argument("--max_iter", type=int, default=3)
    parser.add_argument("--drop_batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--sample_limit", type=int, default=8)
    parser.add_argument("--sample_chars", type=int, default=140)
    parser.add_argument("--max_chunks", type=int, default=0)
    args = parser.parse_args()

    text = Path(args.text_path).read_text(encoding="utf-8")
    text = _clean_text(text)
    tokens = text.split()
    if not tokens:
        raise RuntimeError("No tokens found in input.")

    vec2text_mod = _get_vec2text()
    print(f"Using vec2text from: {getattr(vec2text_mod, '__file__', 'unknown')}")
    print(f"Loading corrector for {args.model}...")
    corrector = _load_corrector(args.model)

    device = _get_device(corrector)
    if hasattr(corrector, "to"):
        corrector.to(device)
    if hasattr(corrector, "model") and hasattr(corrector.model, "eval"):
        corrector.model.eval()
    if hasattr(corrector, "inversion_trainer") and hasattr(corrector.inversion_trainer, "model"):
        if hasattr(corrector.inversion_trainer.model, "eval"):
            corrector.inversion_trainer.model.eval()

    print(f"Tokens: {len(tokens)} | max_tokens={args.max_tokens} stride={args.stride_tokens}")

    keep_k = args.mc_keep_k if args.mc_keep_k > 0 else None
    cursor = 0
    chunk_idx = 0
    start = time.monotonic()
    samples_left = args.sample_limit

    while cursor < len(tokens):
        window_tokens = tokens[cursor : cursor + args.max_tokens]
        if not window_tokens:
            break

        baseline_text = " ".join(window_tokens)
        baseline_quality = _reconstruction_quality(
            baseline_text, corrector, args.num_steps, args.max_length
        )

        chosen_start = cursor
        chosen_end = cursor + len(window_tokens)
        chosen_quality = baseline_quality
        chosen_reason = "baseline"

        if args.mc_trials > 0 and len(window_tokens) >= args.min_tokens:
            current_tokens = list(window_tokens)
            current_start = cursor
            for iteration in range(args.max_iter):
                deltas = _mc_token_deltas(
                    current_tokens,
                    corrector,
                    args.num_steps,
                    args.max_length,
                    args.mc_trials,
                    args.mc_drop_rate,
                    keep_k,
                    args.seed + cursor + iteration,
                )
                span_start, span_end, _span_score = _best_span(
                    deltas, args.min_tokens, min(args.max_tokens, len(current_tokens))
                )
                span_tokens = current_tokens[span_start:span_end]
                span_quality = _reconstruction_quality(
                    " ".join(span_tokens), corrector, args.num_steps, args.max_length
                )

                if span_quality >= chosen_quality or span_quality >= args.target_quality:
                    chosen_quality = span_quality
                    chosen_start = current_start + span_start
                    chosen_end = current_start + span_end
                    chosen_reason = f"mc_iter_{iteration + 1}"

                if span_quality >= args.target_quality or len(span_tokens) <= args.min_tokens:
                    break

                drop = min(args.drop_batch, max(1, len(span_tokens) - args.min_tokens))
                impact = list(enumerate(deltas[span_start:span_end], start=span_start))
                impact.sort(key=lambda x: x[1])
                drop_indices = {idx for idx, _ in impact[:drop]}
                current_tokens = [
                    tok for idx, tok in enumerate(current_tokens) if idx not in drop_indices
                ]
                current_start = current_start

        chunk_tokens = tokens[chosen_start:chosen_end]
        snippet = _snip(chunk_tokens, args.sample_chars)

        if samples_left != 0:
            print(
                f"Chunk {chunk_idx + 1}: q={chosen_quality:.4f} base={baseline_quality:.4f} "
                f"{chosen_reason} | tokens {chosen_start}-{chosen_end} | {snippet}"
            )
            if samples_left > 0:
                samples_left -= 1

        chunk_idx += 1
        if args.max_chunks and chunk_idx >= args.max_chunks:
            break

        advance = args.stride_tokens
        if chosen_reason != "baseline":
            advance = max(1, chosen_end - cursor)
        cursor = max(cursor + 1, cursor + advance)

        if chunk_idx % max(1, args.log_every) == 0:
            elapsed = time.monotonic() - start
            print(f"Progress: chunks={chunk_idx} cursor={cursor}/{len(tokens)} elapsed={elapsed:.0f}s")

    print(f"Done. chunks={chunk_idx}")


if __name__ == "__main__":
    main()
