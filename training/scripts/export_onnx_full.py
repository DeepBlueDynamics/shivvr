#!/usr/bin/env python3
"""Export full ONNX artifacts for the GTR-T5-base inversion stack."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from vec2text.models.bge_inversion import BGEInversionModel
from vec2text.models.bge_corrector import BGECorrectorModel


def _run_optimum_export(hf_dir: Path, onnx_dir: Path, task: str) -> None:
    if not shutil.which("optimum-cli"):
        raise SystemExit(
            "optimum-cli not found. Install with: pip install optimum[exporters] onnxruntime"
        )

    cmd = [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        str(hf_dir),
        "--task",
        task,
        str(onnx_dir),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _export_hypothesis_projection(model: BGEInversionModel, out_path: Path) -> None:
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

    dummy_input = torch.randn(1, model.embedding_dim)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(out_path),
        opset_version=14,
        input_names=["embedding"],
        output_names=["encoder_inputs"],
        dynamic_axes={
            "embedding": {0: "batch"},
            "encoder_inputs": {0: "batch"},
        },
    )


def _export_corrector_prefix(model: BGECorrectorModel, out_path: Path) -> None:
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

    dummy_target = torch.randn(1, model.embedding_dim)
    dummy_hyp = torch.randn(1, model.embedding_dim)

    torch.onnx.export(
        wrapper,
        (dummy_target, dummy_hyp),
        str(out_path),
        opset_version=14,
        input_names=["target_embedding", "hypothesis_embedding"],
        output_names=["prefix"],
        dynamic_axes={
            "target_embedding": {0: "batch"},
            "hypothesis_embedding": {0: "batch"},
            "prefix": {0: "batch"},
        },
    )


def _write_meta(out_dir: Path, meta: dict) -> None:
    meta_path = out_dir / "export_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def export_hypothesis(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = BGEInversionModel.from_pretrained(args.model_path)

    hf_dir = out_dir / "hf-t5"
    model.save_hf_compatible(hf_dir)

    projection_path = out_dir / "projection.onnx"
    _export_hypothesis_projection(model, projection_path)
    print(f"Exported projection to {projection_path}")

    t5_onnx_dir = out_dir / "t5-onnx"
    if not args.skip_optimum:
        _run_optimum_export(hf_dir, t5_onnx_dir, args.task)
    else:
        print("Skipping Optimum export.")
        print(
            f"Run manually: optimum-cli export onnx --model {hf_dir} --task {args.task} {t5_onnx_dir}"
        )

    _write_meta(
        out_dir,
        {
            "model_type": "hypothesis",
            "embedding_dim": model.embedding_dim,
            "hidden_size": model.hidden_size,
            "num_repeat_tokens": model.num_repeat_tokens,
            "projection_onnx": projection_path.name,
            "t5_hf_dir": hf_dir.name,
            "t5_onnx_dir": t5_onnx_dir.name,
            "optimum_task": args.task,
        },
    )


def export_corrector(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = BGECorrectorModel.from_pretrained(args.model_path)

    hf_dir = out_dir / "hf-t5"
    model.save_hf_compatible(hf_dir)

    prefix_path = out_dir / "prefix.onnx"
    _export_corrector_prefix(model, prefix_path)
    print(f"Exported prefix to {prefix_path}")

    t5_onnx_dir = out_dir / "t5-onnx"
    if not args.skip_optimum:
        _run_optimum_export(hf_dir, t5_onnx_dir, args.task)
    else:
        print("Skipping Optimum export.")
        print(
            f"Run manually: optimum-cli export onnx --model {hf_dir} --task {args.task} {t5_onnx_dir}"
        )

    _write_meta(
        out_dir,
        {
            "model_type": "corrector",
            "embedding_dim": model.embedding_dim,
            "hidden_size": model.hidden_size,
            "num_prefix_tokens": model.num_prefix_tokens,
            "prefix_onnx": prefix_path.name,
            "t5_hf_dir": hf_dir.name,
            "t5_onnx_dir": t5_onnx_dir.name,
            "optimum_task": args.task,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["hypothesis", "corrector"],
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--task",
        type=str,
        default="text2text-generation-with-past",
        help="Optimum export task",
    )
    parser.add_argument(
        "--skip_optimum",
        action="store_true",
        help="Skip Optimum export (projection/prefix only)",
    )

    args = parser.parse_args()

    if args.model_type == "hypothesis":
        export_hypothesis(args)
    else:
        export_corrector(args)


if __name__ == "__main__":
    main()
