#!/usr/bin/env python3
"""
Build the int8 model set for CPU / Raspberry Pi deployments.

Reads the fp32 exports from --input_dir and writes to --output_dir, keeping the
file names the service expects, so the output directory can be mounted at
/models as-is:

  gtr-t5-base.onnx                 int8 (dynamic quantization, weights only)
  tokenizer.json                   copied
  embeddinggemma2-text.onnx        int8
  embeddinggemma2-tokenizer.json   copied
  siglip-text.onnx, siglip-vision.onnx, siglip-tokenizer.json   copied as fp32 with --with_siglip

The vec2text inverter is deliberately left out (about 1 GB, and nothing on a Pi needs it).

int8, never fp16: EmbeddingGemma 2 returns NaN in float16, and ONNX Runtime on
ARM CPUs gains nothing from fp16 anyway.

Each quantized model is checked against its fp32 original on sample texts and
the script fails if the worst cosine drops below --min_cosine. int8 is lossy, so
the threshold is looser than the export checks (0.9999); the cosines are printed
so the loss is on record.
"""

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from tokenizers import Tokenizer

SAMPLE_TEXTS = [
    "hello",
    "Rust is a systems programming language.",
    "The anchor windlass draws 1,200 W while hauling in 40 m of chain.",
    "Who protects the Supreme Raven?",
    "def add(a, b):\n    return a + b",
    "Wind 15 knots from the northwest, seas 1.5 m, visibility good.",
    " ".join(f"sentence {i} talks about embeddings and retrieval." for i in range(40)),
]

GEMMA_QUERY_PREFIX = "task: search result | query: "

# MatMul nodes of EmbeddingGemma 2 kept in fp32: the last layer and the final
# 512->768 projection. Quantizing everything gave worst cosine 0.977 vs fp32;
# keeping these (+6 MB) gave 0.988 on SAMPLE_TEXTS (2026-10-08).
GEMMA_FP32_NODE_PATTERNS = ("/layers.23/", "/embedding_projection/")


def quantize(src: Path, dst: Path, keep_fp32_patterns=()):
    print(f"==> Quantizing {src.name} ({src.stat().st_size / 1e6:.0f} MB)")
    # quantize_dynamic writes a shape-inferred copy next to its input, so work
    # from a scratch copy: the input directory may be read-only.
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch) / src.name
        shutil.copy2(src, work)
        graph = onnx.load(str(work), load_external_data=False).graph
        exclude = [
            n.name
            for n in graph.node
            if n.op_type == "MatMul" and any(p in n.name for p in keep_fp32_patterns)
        ]
        if keep_fp32_patterns:
            if not exclude:
                raise RuntimeError(f"no MatMul nodes match {keep_fp32_patterns}; did the export change?")
            print(f"    keeping {len(exclude)} MatMul nodes in fp32")
        quantize_dynamic(
            model_input=str(work),
            model_output=str(dst),
            weight_type=QuantType.QInt8,
            per_channel=True,
            nodes_to_exclude=exclude,
        )
    print(f"    -> {dst} ({dst.stat().st_size / 1e6:.0f} MB)")


def session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def gtr_embed(sess, tok, text):
    """Mirror src/embedder.rs: mean over last_hidden_state, then L2 norm."""
    ids = np.array([tok.encode(text).ids], dtype=np.int64)
    (hidden,) = sess.run(["last_hidden_state"], {"input_ids": ids, "attention_mask": np.ones_like(ids)})
    v = hidden[0].mean(axis=0)
    return v / max(np.linalg.norm(v), 1e-12)


def gemma_embed(sess, tok, text):
    """Mirror src/embeddinggemma.rs: pooled, normalized `embedding` output."""
    ids = np.array([tok.encode(GEMMA_QUERY_PREFIX + text).ids], dtype=np.int64)
    (emb,) = sess.run(["embedding"], {"input_ids": ids, "attention_mask": np.ones_like(ids)})
    v = emb[0]
    return v / max(np.linalg.norm(v), 1e-12)


def compare(name, embed, fp32_path, int8_path, tok, min_cosine):
    fp32, int8 = session(fp32_path), session(int8_path)
    cosines = []
    for text in SAMPLE_TEXTS:
        a, b = embed(fp32, tok, text), embed(int8, tok, text)
        cosines.append(float(np.dot(a, b)))
    worst, mean = min(cosines), sum(cosines) / len(cosines)
    print(f"==> {name}: int8 vs fp32 cosine worst {worst:.4f}, mean {mean:.4f}")
    if worst < min_cosine:
        raise RuntimeError(f"{name}: int8 worst cosine {worst:.4f} < {min_cosine}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("models"))
    parser.add_argument("--output_dir", type=Path, default=Path("models-int8"))
    parser.add_argument("--min_cosine", type=float, default=0.98)
    parser.add_argument("--with_siglip", action="store_true", help="also copy the SigLIP files (fp32)")
    args = parser.parse_args()

    src, dst = args.input_dir, args.output_dir
    dst.mkdir(parents=True, exist_ok=True)

    # GTR-T5-base: required by the service.
    quantize(src / "gtr-t5-base.onnx", dst / "gtr-t5-base.onnx")
    shutil.copy2(src / "tokenizer.json", dst / "tokenizer.json")
    gtr_tok = Tokenizer.from_file(str(src / "tokenizer.json"))
    compare("gtr-t5-base", gtr_embed, src / "gtr-t5-base.onnx", dst / "gtr-t5-base.onnx", gtr_tok, args.min_cosine)

    # EmbeddingGemma 2 text path.
    quantize(src / "embeddinggemma2-text.onnx", dst / "embeddinggemma2-text.onnx", GEMMA_FP32_NODE_PATTERNS)
    shutil.copy2(src / "embeddinggemma2-tokenizer.json", dst / "embeddinggemma2-tokenizer.json")
    gemma_tok = Tokenizer.from_file(str(src / "embeddinggemma2-tokenizer.json"))
    gemma_tok.no_truncation()
    compare(
        "embeddinggemma-2",
        gemma_embed,
        src / "embeddinggemma2-text.onnx",
        dst / "embeddinggemma2-text.onnx",
        gemma_tok,
        args.min_cosine,
    )

    if args.with_siglip:
        for name in ("siglip-text.onnx", "siglip-vision.onnx", "siglip-tokenizer.json"):
            shutil.copy2(src / name, dst / name)
        print("==> SigLIP copied (fp32)")

    # Content stamp: the models image's installer compares it with the copy in the
    # volume and only re-copies when the model set changed.
    digest = hashlib.sha256()
    for p in sorted(dst.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            digest.update(p.name.encode())
            with p.open("rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(block)
    (dst / ".models-version").write_text(digest.hexdigest()[:16] + "\n")

    total = sum(p.stat().st_size for p in dst.iterdir() if p.is_file())
    print(f"==> {dst}: {total / 1e6:.0f} MB total")


if __name__ == "__main__":
    main()
