#!/usr/bin/env python3
"""
Export gtr-t5-base embedder + pre-trained vec2text inverter to ONNX.

Produces everything chonk needs:
  models/gtr-t5-base.onnx        — embedder (768d)
  models/tokenizer.json           — embedder tokenizer
  models/inverter/projection.onnx — vec2text projection
  models/inverter/encoder.onnx    — T5 encoder (inputs_embeds)
  models/inverter/decoder.onnx    — T5 decoder
  models/inverter/tokenizer.json  — T5 tokenizer

Requirements:
  pip install torch transformers sentence-transformers vec2text onnx

No training needed — uses pre-trained vec2text gtr-base models from HuggingFace.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


def export_embedder(output_dir: Path):
    """Export sentence-transformers/gtr-t5-base to ONNX."""
    from transformers import AutoModel, AutoTokenizer

    print("==> Loading gtr-t5-base...")
    model_name = "sentence-transformers/gtr-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # gtr-t5-base is a T5 model — we only need the encoder
    encoder = model.encoder

    class EncoderForExport(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, input_ids, attention_mask):
            return self.enc(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state

    wrapper = EncoderForExport(encoder)
    wrapper.eval()

    dummy_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    dummy_mask = torch.ones_like(dummy_ids)

    onnx_path = output_dir / "gtr-t5-base.onnx"
    print(f"==> Exporting embedder to {onnx_path}...")
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        str(onnx_path),
        opset_version=14,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
    )

    # Save tokenizer
    tokenizer.save_pretrained(str(output_dir))
    print(f"==> Embedder exported ({onnx_path.stat().st_size / 1e6:.1f} MB)")


def export_inverter(output_dir: Path):
    """Download pre-trained vec2text gtr-base and export to ONNX."""
    import vec2text

    inverter_dir = output_dir / "inverter"
    inverter_dir.mkdir(parents=True, exist_ok=True)

    print("==> Loading pre-trained vec2text gtr-base corrector...")
    corrector = vec2text.load_pretrained_corrector("gtr-base")

    # The corrector contains an inversion_trainer with the hypothesis model
    inversion_model = corrector.inversion_trainer.model
    inversion_model.eval()
    inversion_model.cpu()

    # --- Extract the T5 + projection from the inversion model ---
    # The vec2text InversionModel has:
    #   - call_embedding_model: the GTR embedder
    #   - encoder_decoder: the T5 backbone
    #   - embedding_transform: projection from embedding space to T5 hidden
    #   - sequence_weights: learned repeat weights

    t5_model = inversion_model.encoder_decoder  # T5ForConditionalGeneration
    t5_encoder = t5_model.encoder
    t5_tokenizer_name = "t5-base"

    # Figure out the projection architecture
    # vec2text uses a bottleneck MLP: embed_dim -> hidden -> T5_hidden * num_repeat
    print(f"  T5 hidden size: {t5_model.config.d_model}")

    # --- 1. Projection ---
    print("==> Exporting projection.onnx...")

    class ProjectionWrapper(nn.Module):
        def __init__(self, inv_model):
            super().__init__()
            self.inv_model = inv_model

        def forward(self, embedding):
            # Use the model's own embedding transform
            # This handles the projection + repeat logic
            batch_size = embedding.shape[0]
            transformed = self.inv_model.embedding_transform(embedding)
            # Reshape to (batch, num_repeat, hidden)
            num_repeat = self.inv_model.num_repeat_tokens
            hidden = self.inv_model.encoder_decoder.config.d_model
            return transformed.view(batch_size, num_repeat, hidden)

    proj_wrapper = ProjectionWrapper(inversion_model)
    proj_wrapper.eval()

    # GTR-base produces 768d embeddings
    dummy_emb = torch.randn(1, 768)
    torch.onnx.export(
        proj_wrapper, dummy_emb, str(inverter_dir / "projection.onnx"),
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    # --- 2. Encoder (inputs_embeds interface) ---
    print("==> Exporting encoder.onnx...")

    class EncoderWrapper(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.encoder = enc

        def forward(self, inputs_embeds, attention_mask):
            return self.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state

    enc_wrapper = EncoderWrapper(t5_encoder)
    enc_wrapper.eval()

    num_repeat = inversion_model.num_repeat_tokens
    dummy_embeds = torch.randn(1, num_repeat, t5_model.config.d_model)
    dummy_mask = torch.ones(1, num_repeat, dtype=torch.long)

    torch.onnx.export(
        enc_wrapper, (dummy_embeds, dummy_mask),
        str(inverter_dir / "encoder.onnx"),
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
    print("==> Exporting decoder.onnx...")

    class DecoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, encoder_hidden_states, encoder_attention_mask):
            return self.model(
                decoder_input_ids=input_ids,
                encoder_outputs=(encoder_hidden_states,),
                attention_mask=encoder_attention_mask,
                return_dict=True,
            ).logits

    dec_wrapper = DecoderWrapper(t5_model)
    dec_wrapper.eval()

    dummy_dec_ids = torch.tensor([[0]], dtype=torch.long)
    dummy_enc_hidden = torch.randn(1, num_repeat, t5_model.config.d_model)
    dummy_enc_mask = torch.ones(1, num_repeat, dtype=torch.long)

    torch.onnx.export(
        dec_wrapper,
        (dummy_dec_ids, dummy_enc_hidden, dummy_enc_mask),
        str(inverter_dir / "decoder.onnx"),
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
    print("==> Saving T5 tokenizer...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(t5_tokenizer_name)
    tok.save_pretrained(str(inverter_dir))

    # Summary
    print("\n==> Inverter export complete:")
    for f in sorted(inverter_dir.glob("*.onnx")):
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
    if (inverter_dir / "tokenizer.json").exists():
        print("  tokenizer.json")


def verify(output_dir: Path):
    """Quick sanity check — embed text, invert, check shape."""
    import onnxruntime as ort

    print("\n==> Verifying...")

    # Load embedder
    emb_session = ort.InferenceSession(str(output_dir / "gtr-t5-base.onnx"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(output_dir))

    text = "The quick brown fox jumps over the lazy dog."
    inputs = tok(text, return_tensors="np", padding=True, truncation=True, max_length=128)
    result = emb_session.run(None, {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    })
    hidden = result[0]  # (1, seq, 768)
    # Mean pool
    mask = inputs["attention_mask"][..., np.newaxis].astype(np.float32)
    pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
    norm = np.linalg.norm(pooled, axis=1, keepdims=True)
    embedding = (pooled / norm).flatten()

    print(f"  Embedding shape: {embedding.shape} (expect 768)")
    print(f"  Embedding norm: {np.linalg.norm(embedding):.4f} (expect 1.0)")

    # Load inverter
    inv_dir = output_dir / "inverter"
    proj = ort.InferenceSession(str(inv_dir / "projection.onnx"))
    enc = ort.InferenceSession(str(inv_dir / "encoder.onnx"))
    dec = ort.InferenceSession(str(inv_dir / "decoder.onnx"))

    # Project
    projected = proj.run(None, {"input": embedding.reshape(1, -1).astype(np.float32)})[0]
    print(f"  Projected shape: {projected.shape}")

    # Encode
    seq_len = projected.shape[1]
    attn = np.ones((1, seq_len), dtype=np.int64)
    encoded = enc.run(None, {
        "inputs_embeds": projected.astype(np.float32),
        "attention_mask": attn,
    })[0]
    print(f"  Encoded shape: {encoded.shape}")

    # Decode first token
    dec_input = np.array([[0]], dtype=np.int64)  # decoder start token
    logits = dec.run(None, {
        "input_ids": dec_input,
        "encoder_hidden_states": encoded.astype(np.float32),
        "encoder_attention_mask": attn,
    })[0]
    first_token = int(np.argmax(logits[0, -1, :]))

    t5_tok = AutoTokenizer.from_pretrained(str(inv_dir))
    print(f"  First decoded token: {first_token} = '{t5_tok.decode([first_token])}'")
    print("\n==> Verification passed!")


def main():
    parser = argparse.ArgumentParser(description="Export gtr-t5-base + vec2text to ONNX")
    parser.add_argument("--output_dir", default="./models",
                        help="Output directory (default: ./models)")
    parser.add_argument("--skip-embedder", action="store_true",
                        help="Skip embedder export (inverter only)")
    parser.add_argument("--skip-inverter", action="store_true",
                        help="Skip inverter export (embedder only)")
    parser.add_argument("--verify", action="store_true",
                        help="Run verification after export")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_embedder:
        export_embedder(output_dir)

    if not args.skip_inverter:
        export_inverter(output_dir)

    if args.verify:
        verify(output_dir)

    print("\nDone! Copy models/ into chonk's Docker context and rebuild.")


if __name__ == "__main__":
    main()
