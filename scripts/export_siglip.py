#!/usr/bin/env python3
"""
Export google/siglip-base-patch16-224 (vision + text) to ONNX.
Produces:
  models/siglip-vision.onnx  — vision encoder (image tensor [1, 3, 224, 224] -> 768d normalized vector)
  models/siglip-text.onnx    — text encoder (input_ids [1, seq] -> 768d normalized vector)
  models/siglip-tokenizer/   — SigLIP tokenizer config/vocab (slow, sentencepiece)
  models/siglip-tokenizer.json — fast-tokenizer JSON loaded by the Rust service
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer

def export_siglip(output_dir: Path, model_id: str = "google/siglip-base-patch16-224"):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"==> Loading {model_id}...")
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    # --- 1. Export Vision Model ---
    vision_model = model.vision_model
    class VisionWrapper(nn.Module):
        def __init__(self, vm):
            super().__init__()
            self.vm = vm

        def forward(self, pixel_values):
            # SigLIP vision model output pooler
            out = self.vm(pixel_values=pixel_values)
            # pooler_output is already pooled (e.g. 768d)
            pooled = out.pooler_output
            norm = torch.norm(pooled, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            return pooled / norm

    v_wrapper = VisionWrapper(vision_model)
    v_wrapper.eval()

    dummy_pixel_values = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    vision_path = output_dir / "siglip-vision.onnx"
    print(f"==> Exporting vision model to {vision_path}...")
    torch.onnx.export(
        v_wrapper,
        dummy_pixel_values,
        str(vision_path),
        opset_version=17,
        dynamo=False,
        input_names=["pixel_values"],
        output_names=["embedding"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "embedding": {0: "batch"}
        }
    )
    print(f"==> Vision exported ({vision_path.stat().st_size / 1e6:.1f} MB)")

    # --- 2. Export Text Model ---
    text_model = model.text_model
    class TextWrapper(nn.Module):
        def __init__(self, tm):
            super().__init__()
            self.tm = tm

        def forward(self, input_ids):
            out = self.tm(input_ids=input_ids)
            pooled = out.pooler_output
            norm = torch.norm(pooled, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            return pooled / norm

    t_wrapper = TextWrapper(text_model)
    t_wrapper.eval()

    dummy_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    text_path = output_dir / "siglip-text.onnx"
    print(f"==> Exporting text model to {text_path}...")
    torch.onnx.export(
        t_wrapper,
        dummy_ids,
        str(text_path),
        opset_version=17,
        dynamo=False,
        input_names=["input_ids"],
        output_names=["embedding"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "embedding": {0: "batch"}
        }
    )
    print(f"==> Text exported ({text_path.stat().st_size / 1e6:.1f} MB)")

    # Save tokenizer
    tokenizer_dir = output_dir / "siglip-tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(str(tokenizer_dir))
    print("==> Tokenizer saved")

    # The Rust `tokenizers` crate can only load the fast-tokenizer JSON. transformers
    # has no slow->fast converter for SiglipTokenizer, so build it directly from
    # spiece.model and verify it reproduces the slow tokenizer before writing it.
    fast_path = output_dir / "siglip-tokenizer.json"
    fast = build_fast_siglip_tokenizer(tokenizer_dir / "spiece.model")
    verify_fast_tokenizer(fast, tokenizer)
    fast.save(str(fast_path))
    print(f"==> Fast tokenizer saved to {fast_path}")


SAMPLE_TEXTS = [
    "A photo of a cat.",
    "Two dogs playing in the snow!",
    "The Quick, Brown Fox; jumps over the lazy dog?",
    "hello   world",
    "Diagram of a UDP connection checker (2026-09-27)",
    "",
]


def build_fast_siglip_tokenizer(spm_path: Path):
    """Fast tokenizer equivalent to transformers' SiglipTokenizer.

    SiglipTokenizer canonicalizes text before SentencePiece: it strips ASCII
    punctuation (string.punctuation), collapses whitespace, strips, lowercases,
    then encodes with the unigram model and appends </s> (id 1). Padding is
    handled by the service (pads with </s> to 64 tokens).
    """
    import string
    from tokenizers import Regex, normalizers, processors
    from tokenizers.implementations import SentencePieceUnigramTokenizer
    import sys
    if "sentencepiece_model_pb2" not in sys.modules:
        # from_spm() imports a top-level sentencepiece_model_pb2; the pip package ships it as a submodule.
        # Use the copy transformers already registered with protobuf; importing the
        # sentencepiece package's copy as well raises a duplicate-proto error.
        _spm_pb2 = None
        try:
            from transformers.convert_slow_tokenizer import import_protobuf  # transformers 4.4x
            _spm_pb2 = import_protobuf()
        except Exception:
            from sentencepiece import sentencepiece_model_pb2 as _spm_pb2  # standalone fallback
        sys.modules["sentencepiece_model_pb2"] = _spm_pb2

    fast = SentencePieceUnigramTokenizer.from_spm(str(spm_path))
    punct_class = "[" + "".join("\\" + c for c in string.punctuation) + "]"
    canonicalize = [
        normalizers.Replace(Regex(punct_class), ""),
        normalizers.Replace(Regex("\\s+"), " "),
        normalizers.Strip(),
        normalizers.Lowercase(),
    ]
    existing = fast.normalizer
    fast.normalizer = normalizers.Sequence(canonicalize + ([existing] if existing is not None else []))
    eos_id = fast.token_to_id("</s>")
    if eos_id is None:
        raise RuntimeError("spiece.model has no </s> token")
    fast.post_processor = processors.TemplateProcessing(
        single="$A </s>",
        pair="$A </s> $B </s>",
        special_tokens=[("</s>", eos_id)],
    )
    return fast


def verify_fast_tokenizer(fast, slow) -> None:
    """Fail loudly if the fast tokenizer disagrees with the slow one on samples."""
    mismatches = []
    for text in SAMPLE_TEXTS:
        expected = slow(text, add_special_tokens=True, padding=False, truncation=False)["input_ids"]
        got = fast.encode(text, add_special_tokens=True).ids
        if list(expected) != list(got):
            mismatches.append((text, expected, got))
    if mismatches:
        for text, expected, got in mismatches:
            print(f"  MISMATCH {text!r}: slow={expected} fast={got}")
        raise RuntimeError(f"fast SigLIP tokenizer disagrees with the slow one on {len(mismatches)} sample(s)")
    print(f"==> Fast tokenizer verified against SiglipTokenizer on {len(SAMPLE_TEXTS)} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("models"))
    parser.add_argument("--model_id", type=str, default="google/siglip-base-patch16-224")
    args = parser.parse_args()
    export_siglip(args.output_dir, args.model_id)
