#!/bin/bash
# ============================================================
# BGE-Small Inversion Training — Dual A100 40GB
# ============================================================
# Trains ONLY the hypothesis model (projection + T5 fine-tune).
# Chonk's Rust inverter doesn't use a corrector, so skip it.
#
# Total time: ~12-18 hours on dual A100 40GB
#
# Outputs:
#   models/inverter/projection.onnx   (384 -> 16x768)
#   models/inverter/encoder.onnx      (inputs_embeds interface)
#   models/inverter/decoder.onnx      (greedy decode)
#   models/inverter/tokenizer.json    (T5 tokenizer)
# ============================================================

set -e

EMBEDDINGS_DIR="./embeddings/bge-small-msmarco-full"
SAVES_DIR="./saves"
MODELS_DIR="./models/inverter"

mkdir -p "$EMBEDDINGS_DIR" "$SAVES_DIR" "$MODELS_DIR"

# ── Step 1: Generate BGE embeddings from MSMARCO ────────────
# ~1-2 hours, GPU inference only
echo ""
echo "========================================"
echo "Step 1/3: Generating BGE embeddings"
echo "========================================"
if [ ! -f "$EMBEDDINGS_DIR/embeddings.npy" ]; then
    python scripts/generate_bge_embeddings.py \
        --output_dir "$EMBEDDINGS_DIR" \
        --batch_size 512 \
        --max_length 256
    # Full MSMARCO corpus (~8.8M docs). Remove --max_samples to use all.
    # For a quick test run, add: --max_samples 50000
else
    echo "Skipping — embeddings already exist at $EMBEDDINGS_DIR"
    echo "  Delete $EMBEDDINGS_DIR/embeddings.npy to regenerate"
fi

# ── Step 2: Train hypothesis model ─────────────────────────
# ~8-14 hours depending on dataset size
echo ""
echo "========================================"
echo "Step 2/3: Training hypothesis model"
echo "========================================"
if [ ! -f "$SAVES_DIR/bge-small-hypothesis/best_model.pt" ]; then
    python scripts/train_hypothesis.py \
        --embeddings_path "$EMBEDDINGS_DIR/embeddings.npy" \
        --texts_path "$EMBEDDINGS_DIR/texts.txt" \
        --output_dir "$SAVES_DIR/bge-small-hypothesis" \
        --batch_size 128 \
        --epochs 50 \
        --lr 1e-4 \
        --max_length 128 \
        --num_repeat_tokens 16 \
        --num_workers 8 \
        --save_every 10
else
    echo "Skipping — model already exists"
    echo "  Delete $SAVES_DIR/bge-small-hypothesis/best_model.pt to retrain"
fi

# ── Step 3: Export to ONNX ──────────────────────────────────
# ~2 minutes, CPU only
echo ""
echo "========================================"
echo "Step 3/3: Exporting to ONNX"
echo "========================================"
python scripts/export_custom_onnx.py \
    --model_path "$SAVES_DIR/bge-small-hypothesis/best_model.pt" \
    --output_dir "$MODELS_DIR"

echo ""
echo "========================================"
echo "DONE"
echo "========================================"
echo ""
echo "Models ready at: $MODELS_DIR/"
ls -lh "$MODELS_DIR/"
echo ""
echo "To install into chonk:"
echo "  cp $MODELS_DIR/* /path/to/gnosis-chunk/models/inverter/"
echo "  # Then rebuild chonk Docker image or volume-mount"
echo ""
echo "To test locally (needs torch):"
echo "  python scripts/test_inversion.py \\"
echo "    --hypothesis_model $SAVES_DIR/bge-small-hypothesis/best_model.pt"
