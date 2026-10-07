#!/usr/bin/env python3
"""
Export google/embeddinggemma-2 (text path) to ONNX.
Produces:
  models/embeddinggemma2-text.onnx      — input_ids + attention_mask [1, seq] -> embedding [1, 768]
                                          (masked mean pooling + L2 norm folded into the graph)
  models/embeddinggemma2-tokenizer.json — fast-tokenizer JSON loaded by the Rust service

The vision and audio encoders are not exported (see plan/EMBEDDINGGEMMA2_PLAN.md).

Needs a newer stack than the GTR/SigLIP exports (transformers >= 5.18,
sentence-transformers >= 6.1), so Dockerfile.models runs it in its own stage.
Run in float32: the model returns NaN in float16.

Both outputs are checked before the script exits: the tokenizer JSON must
reproduce SentenceTransformer's token ids, and the ONNX model must match
SentenceTransformer.encode() (cosine >= --min_cosine) across short, prefixed
and >512-token inputs, which exercise the sliding-window layers.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MODEL_ID = "google/embeddinggemma-2"

# Text-only load: drops the 170M vision and 300M audio encoders.
TEXT_ONLY = {"vision_config": None, "audio_config": None}

SAMPLE_TEXTS = [
    "hello",
    "Rust is a systems programming language.",
    "The Quick, Brown Fox; jumps over the lazy dog?",
    "def add(a, b):\n    return a + b",
    "Diagram of a UDP connection checker (2026-09-27) — ünïcödé ✓ 日本語",
    # ~900 tokens: longer than the 512-token sliding window.
    " ".join(f"sentence {i} talks about embeddings and retrieval." for i in range(110)),
]
SAMPLE_PROMPTS = [None, "query", "document"]


def load_st(model_id: str):
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(
        model_id,
        device="cpu",
        model_kwargs={"dtype": torch.float32},
        config_kwargs=TEXT_ONLY,
    )
    st.eval()
    return st


def hf_model(st) -> nn.Module:
    """The underlying transformers model inside the ST Transformer module."""
    module = st[0]
    for attr in ("model", "auto_model"):
        m = getattr(module, attr, None)
        if isinstance(m, nn.Module):
            return m
    raise RuntimeError(f"cannot find the transformers model inside {type(module).__name__}")


class TextWrapper(nn.Module):
    """Token states -> masked mean pool -> L2 norm, matching ST's Pooling(mean,
    include_prompt=True) + Normalize modules."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        hs = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled / torch.norm(pooled, p=2, dim=-1, keepdim=True).clamp(min=1e-12)


def st_prompt(st, name):
    return "" if name is None else st.prompts[name]


def export_tokenizer(st, model_id: str, out: Path):
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError("expected a fast tokenizer for embeddinggemma-2")
    tok.backend_tokenizer.save(str(out))

    # What the Rust service will do: Tokenizer::from_file + encode(text, true).
    rust_like = Tokenizer.from_file(str(out))
    rust_like.no_padding()
    rust_like.no_truncation()
    mismatches = []
    for name in SAMPLE_PROMPTS:
        for text in SAMPLE_TEXTS:
            full = st_prompt(st, name) + text
            expected = st.tokenize([full])["input_ids"][0].tolist()
            got = rust_like.encode(full, add_special_tokens=True).ids
            if expected != got:
                mismatches.append((full[:60], expected[:12], got[:12]))
    if mismatches:
        for text, expected, got in mismatches:
            print(f"  MISMATCH {text!r}: st={expected}... json={got}...")
        raise RuntimeError(f"tokenizer JSON disagrees with SentenceTransformer on {len(mismatches)} input(s)")
    print(f"==> Tokenizer saved to {out} and verified on {len(SAMPLE_PROMPTS) * len(SAMPLE_TEXTS)} inputs")
    return rust_like


def export_onnx(wrapper: nn.Module, out: Path, opset: int):
    ids = torch.tensor([[2, 1000, 2000, 3000, 1]], dtype=torch.long)
    mask = torch.ones_like(ids)
    kwargs = dict(
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "embedding": {0: "batch"},
        },
        opset_version=opset,
    )
    # TorchScript export first (what the other exports use); the dynamo
    # exporter handles newer transformers attention code that TorchScript
    # tracing cannot.
    try:
        torch.onnx.export(wrapper, (ids, mask), str(out), dynamo=False, **kwargs)
        print(f"==> Exported with the TorchScript exporter (opset {opset})")
    except Exception as e:  # noqa: BLE001
        print(f"==> TorchScript export failed ({type(e).__name__}: {e}); retrying with dynamo")
        torch.onnx.export(
            wrapper, (ids, mask), str(out), dynamo=True, external_data=False, **kwargs
        )
        print(f"==> Exported with the dynamo exporter (opset {opset})")
    print(f"==> {out} ({out.stat().st_size / 1e6:.1f} MB)")


def verify_onnx(st, tokenizer, onnx_path: Path, min_cosine: float):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst = 1.0
    for name in SAMPLE_PROMPTS:
        ref = st.encode(SAMPLE_TEXTS, prompt_name=name, normalize_embeddings=True, convert_to_numpy=True)
        for text, r in zip(SAMPLE_TEXTS, ref):
            ids = np.array([tokenizer.encode(st_prompt(st, name) + text, add_special_tokens=True).ids], dtype=np.int64)
            (emb,) = sess.run(["embedding"], {"input_ids": ids, "attention_mask": np.ones_like(ids)})
            cos = float(np.dot(emb[0], r) / (np.linalg.norm(emb[0]) * np.linalg.norm(r)))
            worst = min(worst, cos)
            if cos < min_cosine:
                raise RuntimeError(
                    f"ONNX vs SentenceTransformer cosine {cos:.6f} < {min_cosine} "
                    f"(prompt={name}, {ids.shape[1]} tokens, text={text[:40]!r})"
                )
    print(f"==> ONNX matches SentenceTransformer.encode(): worst cosine {worst:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("models"))
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--min_cosine", type=float, default=0.9999)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"==> Loading {args.model_id} (text only, float32)...")
    st = load_st(args.model_id)

    tokenizer = export_tokenizer(st, args.model_id, args.output_dir / "embeddinggemma2-tokenizer.json")

    wrapper = TextWrapper(hf_model(st)).eval()
    onnx_path = args.output_dir / "embeddinggemma2-text.onnx"
    with torch.no_grad():
        # The wrapper must agree with ST before it is worth exporting.
        probe = st.encode(["hello"], normalize_embeddings=True, convert_to_tensor=True)
        ids = st.tokenize(["hello"])
        got = wrapper(ids["input_ids"], ids["attention_mask"])
        if got.shape[-1] != 768:
            raise RuntimeError(f"wrapper returns {got.shape[-1]} dims; expected 768 (projection not in last_hidden_state?)")
        cos = torch.nn.functional.cosine_similarity(got, probe.to(got.dtype)).item()
        if cos < args.min_cosine:
            raise RuntimeError(f"wrapper vs SentenceTransformer cosine {cos:.6f}; pooling does not match")
        export_onnx(wrapper, onnx_path, args.opset)

    verify_onnx(st, tokenizer, onnx_path, args.min_cosine)


if __name__ == "__main__":
    main()
