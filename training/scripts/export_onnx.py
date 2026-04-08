#!/usr/bin/env python3
"""Export trained models to ONNX format for Rust inference."""

import torch
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from vec2text.models.bge_inversion import BGEInversionModel
from vec2text.models.bge_corrector import BGECorrectorModel


def export_hypothesis_model(model_path: str, output_path: str):
    """Export hypothesis model to ONNX."""
    print(f"Loading hypothesis model from {model_path}...")
    model = BGEInversionModel.from_pretrained(model_path)
    model.eval()
    
    # For ONNX export of encoder-decoder models, we need to export
    # encoder and decoder separately or use a wrapper
    
    # Simple approach: export just the embedding projection + encoder part
    # The full generate() requires special handling
    
    print("Note: Full T5 generate() export to ONNX is complex.")
    print("Exporting encoder-only for now. Full inference will need HuggingFace Optimum.")
    
    # Save PyTorch model in a format easy to convert later
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export just the projection layer as ONNX
    class ProjectionWrapper(torch.nn.Module):
        def __init__(self, projection, num_repeat):
            super().__init__()
            self.projection = projection
            self.num_repeat = num_repeat
        
        def forward(self, embedding):
            projected = self.projection(embedding)
            repeated = projected.unsqueeze(1).repeat(1, self.num_repeat, 1)
            return repeated
    
    wrapper = ProjectionWrapper(model.embedding_projection, model.num_repeat_tokens)
    wrapper.eval()
    
    dummy_input = torch.randn(1, 384)
    
    onnx_path = output_path.with_suffix(".projection.onnx")
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        opset_version=14,
        input_names=["embedding"],
        output_names=["encoder_inputs"],
        dynamic_axes={
            "embedding": {0: "batch"},
            "encoder_inputs": {0: "batch"},
        },
    )
    print(f"Exported projection to {onnx_path}")
    
    # Also save the full PyTorch model for later conversion
    pt_path = output_path.with_suffix(".pt")
    model.save_pretrained(pt_path)
    print(f"Saved PyTorch model to {pt_path}")
    
    print("\nFor full ONNX export, use Optimum:")
    print("  pip install optimum[exporters]")
    print("  optimum-cli export onnx --model <model_path> --task text2text-generation <output_dir>")


def export_corrector_model(model_path: str, output_path: str):
    """Export corrector model to ONNX."""
    print(f"Loading corrector model from {model_path}...")
    model = BGECorrectorModel.from_pretrained(model_path)
    model.eval()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export prefix creation as ONNX
    class PrefixWrapper(torch.nn.Module):
        def __init__(self, target_proj, hyp_proj, prefix_mlp, num_prefix, hidden):
            super().__init__()
            self.target_proj = target_proj
            self.hyp_proj = hyp_proj
            self.prefix_mlp = prefix_mlp
            self.num_prefix = num_prefix
            self.hidden = hidden
        
        def forward(self, target_emb, hyp_emb):
            target = self.target_proj(target_emb)
            hyp = self.hyp_proj(hyp_emb)
            combined = torch.cat([target, hyp], dim=-1)
            prefix = self.prefix_mlp(combined)
            return prefix.view(-1, self.num_prefix, self.hidden)
    
    wrapper = PrefixWrapper(
        model.target_projection,
        model.hypothesis_projection,
        model.prefix_mlp,
        model.num_prefix_tokens,
        model.hidden_size,
    )
    wrapper.eval()
    
    dummy_target = torch.randn(1, 384)
    dummy_hyp = torch.randn(1, 384)
    
    onnx_path = output_path.with_suffix(".prefix.onnx")
    torch.onnx.export(
        wrapper,
        (dummy_target, dummy_hyp),
        str(onnx_path),
        opset_version=14,
        input_names=["target_embedding", "hypothesis_embedding"],
        output_names=["prefix"],
        dynamic_axes={
            "target_embedding": {0: "batch"},
            "hypothesis_embedding": {0: "batch"},
            "prefix": {0: "batch"},
        },
    )
    print(f"Exported prefix model to {onnx_path}")
    
    # Save full PyTorch model
    pt_path = output_path.with_suffix(".pt")
    model.save_pretrained(pt_path)
    print(f"Saved PyTorch model to {pt_path}")


def main(args):
    if args.model_type == "hypothesis":
        export_hypothesis_model(args.model_path, args.output_path)
    elif args.model_type == "corrector":
        export_corrector_model(args.model_path, args.output_path)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["hypothesis", "corrector"])
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pt file")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for ONNX model")
    
    args = parser.parse_args()
    main(args)
