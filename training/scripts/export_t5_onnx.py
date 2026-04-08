#!/usr/bin/env python3
"""Export T5 backbone from trained hypothesis model to ONNX via Optimum."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vec2text.models.bge_inversion import BGEInversionModel
from optimum.exporters.onnx import main_export


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to best_model.pt")
    parser.add_argument("--output_dir", default="./models/hypothesis",
                        help="Directory for ONNX outputs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    hf_dir = out_dir / "hf-t5"
    onnx_dir = out_dir / "t5-onnx"

    # Load trained model and save T5 backbone in HF format
    print(f"Loading model from {args.model_path}...")
    model = BGEInversionModel.from_pretrained(args.model_path)

    print(f"Saving T5 backbone to {hf_dir}...")
    model.save_hf_compatible(str(hf_dir))

    # Export via Optimum
    print(f"Exporting T5 to ONNX at {onnx_dir}...")
    main_export(
        model_name_or_path=str(hf_dir),
        output=onnx_dir,
        task="text2text-generation",
        opset=14,
    )

    # Copy tokenizer.json to output dir for convenience
    tok_src = onnx_dir / "tokenizer.json"
    tok_dst = out_dir / "tokenizer.json"
    if tok_src.exists() and not tok_dst.exists():
        import shutil
        shutil.copy2(tok_src, tok_dst)
        print(f"Copied tokenizer.json to {tok_dst}")

    print("\nExport complete! Files:")
    for f in sorted(out_dir.rglob("*.onnx")):
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  {f.relative_to(out_dir)} ({size_mb:.1f} MB)")
    if tok_dst.exists():
        print(f"  tokenizer.json")


if __name__ == "__main__":
    main()
