#!/usr/bin/env python3
"""
Custom ONNX export for ferricula/chonk inverter.

The Rust inverter expects:
  - projection.onnx: input "embedding" (batch, 384) → "encoder_inputs" (batch, 16, 768)
  - encoder.onnx: input "inputs_embeds" (batch, seq, 768) + "attention_mask" (batch, seq)
                   → output hidden states (batch, seq, 768)
  - decoder.onnx: input "input_ids" (batch, dec_seq)
                   + "encoder_hidden_states" (batch, enc_seq, 768)
                   + "encoder_attention_mask" (batch, enc_seq)
                   → output logits (batch, dec_seq, vocab_size)
  - tokenizer.json: T5 tokenizer

Standard Optimum exports the encoder with input_ids, not inputs_embeds.
This script does a custom export that matches what the Rust code expects.
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
import onnx

sys.path.insert(0, str(Path(__file__).parent.parent))
from vec2text.models.bge_inversion import BGEInversionModel


class EncoderWrapper(nn.Module):
    """Wraps T5 encoder to accept inputs_embeds instead of input_ids."""

    def __init__(self, t5_encoder):
        super().__init__()
        self.encoder = t5_encoder

    def forward(self, inputs_embeds, attention_mask):
        outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state


class DecoderWrapper(nn.Module):
    """Wraps T5 for decode step: takes decoder input_ids + encoder outputs."""

    def __init__(self, t5_model):
        super().__init__()
        self.model = t5_model

    def forward(self, input_ids, encoder_hidden_states, encoder_attention_mask):
        outputs = self.model(
            decoder_input_ids=input_ids,
            encoder_outputs=(encoder_hidden_states,),
            attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        return outputs.logits


def export_all(model_path: str, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading hypothesis model from {model_path}...")
    model = BGEInversionModel.from_pretrained(model_path)
    model.eval()

    device = torch.device("cpu")
    model = model.to(device)

    # --- 1. Projection ---
    print("Exporting projection.onnx...")

    class ProjectionWrapper(nn.Module):
        def __init__(self, proj, num_repeat):
            super().__init__()
            self.proj = proj
            self.num_repeat = num_repeat

        def forward(self, embedding):
            projected = self.proj(embedding)
            return projected.unsqueeze(1).repeat(1, self.num_repeat, 1)

    proj_wrapper = ProjectionWrapper(model.embedding_projection, model.num_repeat_tokens)
    proj_wrapper.eval()

    dummy_emb = torch.randn(1, 384)
    torch.onnx.export(
        proj_wrapper, dummy_emb, str(out / "projection.onnx"),
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    # --- 2. Encoder (inputs_embeds) ---
    print("Exporting encoder.onnx...")
    enc_wrapper = EncoderWrapper(model.t5.encoder)
    enc_wrapper.eval()

    dummy_embeds = torch.randn(1, 16, 768)
    dummy_mask = torch.ones(1, 16, dtype=torch.long)

    torch.onnx.export(
        enc_wrapper, (dummy_embeds, dummy_mask), str(out / "encoder.onnx"),
        opset_version=14,
        input_names=["inputs_embeds", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
    )

    # --- 3. Decoder ---
    print("Exporting decoder.onnx...")
    dec_wrapper = DecoderWrapper(model.t5)
    dec_wrapper.eval()

    dummy_dec_ids = torch.tensor([[0]], dtype=torch.long)  # decoder start token
    dummy_enc_hidden = torch.randn(1, 16, 768)
    dummy_enc_mask = torch.ones(1, 16, dtype=torch.long)

    torch.onnx.export(
        dec_wrapper,
        (dummy_dec_ids, dummy_enc_hidden, dummy_enc_mask),
        str(out / "decoder.onnx"),
        opset_version=14,
        input_names=["input_ids", "encoder_hidden_states", "encoder_attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "decoder_seq"},
            "encoder_hidden_states": {0: "batch", 1: "encoder_seq"},
            "encoder_attention_mask": {0: "batch", 1: "encoder_seq"},
            "logits": {0: "batch", 1: "decoder_seq"},
        },
    )

    # --- 4. Tokenizer ---
    print("Saving tokenizer.json...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("t5-base")
    tok.save_pretrained(str(out))

    print(f"\nAll exports complete in {out}/")
    for f in sorted(out.glob("*.onnx")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")
    if (out / "tokenizer.json").exists():
        print("  tokenizer.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default="./models/inverter")
    args = parser.parse_args()
    export_all(args.model_path, args.output_dir)
