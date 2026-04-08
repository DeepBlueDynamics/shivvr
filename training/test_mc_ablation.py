"""
Monte Carlo ablation: estimate per-token impact on reconstruction quality.
"""

import argparse
import importlib
import platform
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

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


def _load_corrector(embedder: str):
    vec2text_mod = _get_vec2text()
    fn = getattr(vec2text_mod, "load_pretrained_corrector", None)
    if fn is None:
        api_mod = importlib.import_module("vec2text.api")
        fn = getattr(api_mod, "load_pretrained_corrector", None)
    if fn is None:
        raise RuntimeError("vec2text.load_pretrained_corrector not found.")

    return fn(embedder)


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


def _sample_mask(tokens: List[str], rng: random.Random, drop_rate: float, keep_k: int):
    n = len(tokens)
    if keep_k is not None:
        keep_k = max(1, min(keep_k, n))
        kept = set(rng.sample(range(n), keep_k))
        return [i in kept for i in range(n)]

    mask = [rng.random() >= drop_rate for _ in range(n)]
    if not any(mask):
        mask[rng.randrange(n)] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None, help="Input text")
    parser.add_argument("--model", type=str, default="gtr-base")
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--drop_rate", type=float, default=0.3)
    parser.add_argument("--keep_k", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--show_examples", type=int, default=3)
    args = parser.parse_args()

    text = args.text or (
        "DNA stores genetic information. Stock prices fell sharply. "
        "The Roman army was well organized."
    )

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

    tokens = text.split()
    if not tokens:
        raise RuntimeError("Input text is empty after tokenization.")

    print(f"Tokens: {len(tokens)} | trials={args.trials} | drop_rate={args.drop_rate}")
    if args.keep_k is not None:
        print(f"Using keep_k={args.keep_k}")

    baseline = _reconstruction_quality(text, corrector, args.num_steps, args.max_length)
    print(f"Baseline quality (full text): {baseline:.4f}")

    present_sum = [0.0] * len(tokens)
    present_count = [0] * len(tokens)
    absent_sum = [0.0] * len(tokens)
    absent_count = [0] * len(tokens)

    samples: List[Tuple[str, float]] = []
    rng = random.Random(args.seed)
    start = time.monotonic()

    for i in range(args.trials):
        mask = _sample_mask(tokens, rng, args.drop_rate, args.keep_k)
        kept_tokens = [tok for tok, keep in zip(tokens, mask) if keep]
        ablated_text = " ".join(kept_tokens)
        quality = _reconstruction_quality(ablated_text, corrector, args.num_steps, args.max_length)

        for idx, keep in enumerate(mask):
            if keep:
                present_sum[idx] += quality
                present_count[idx] += 1
            else:
                absent_sum[idx] += quality
                absent_count[idx] += 1

        if i < args.show_examples:
            samples.append((ablated_text, quality))

        if (i + 1) % max(1, args.trials // 5) == 0 or i == args.trials - 1:
            elapsed = time.monotonic() - start
            print(f"Progress: {i + 1}/{args.trials} | elapsed={elapsed:.0f}s")

    impacts = []
    for idx, tok in enumerate(tokens):
        p_count = max(1, present_count[idx])
        a_count = max(1, absent_count[idx])
        present_mean = present_sum[idx] / p_count
        absent_mean = absent_sum[idx] / a_count
        delta = present_mean - absent_mean
        impacts.append((tok, delta, present_mean, absent_mean, p_count, a_count))

    impacts.sort(key=lambda x: x[1], reverse=True)

    print("\nExamples:")
    for ablated, quality in samples:
        snippet = ablated[:120] + ("..." if len(ablated) > 120 else "")
        print(f"  q={quality:.4f} | {snippet}")

    print("\nTop helpful tokens:")
    for tok, delta, p_mean, a_mean, p_count, a_count in impacts[: args.top_k]:
        print(f"  {tok:>12}  delta={delta:+.4f}  present={p_mean:.4f} absent={a_mean:.4f}")

    print("\nTop harmful tokens:")
    for tok, delta, p_mean, a_mean, p_count, a_count in impacts[-args.top_k:][::-1]:
        print(f"  {tok:>12}  delta={delta:+.4f}  present={p_mean:.4f} absent={a_mean:.4f}")


if __name__ == "__main__":
    main()
