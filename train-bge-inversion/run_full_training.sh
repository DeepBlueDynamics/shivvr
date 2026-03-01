#!/bin/bash
# Full training pipeline for BGE-small inversion models
# Expected runtime: 3-5 days on single A100

set -e  # Exit on error

echo "========================================"
echo "BGE-Small Inversion Model Training"
echo "========================================"
echo ""

# Configuration
EMBEDDINGS_DIR="./embeddings/bge-small-msmarco"
HYPOTHESES_DIR="./embeddings/bge-small-msmarco-hypotheses"
SAVES_DIR="./saves"
MODELS_DIR="./models"

# Create directories
mkdir -p "$EMBEDDINGS_DIR" "$HYPOTHESES_DIR" "$SAVES_DIR" "$MODELS_DIR"

# Step 1: Generate BGE embeddings for MSMARCO
echo ""
echo "========================================"
echo "Step 1/6: Generating BGE embeddings"
echo "========================================"
echo "This will embed ~8.8M documents from MSMARCO"
echo "Expected time: 2-4 hours on GPU"
echo ""

if [ -f "$EMBEDDINGS_DIR/embeddings.npy" ]; then
    echo "Embeddings already exist, skipping..."
else
    python scripts/generate_bge_embeddings.py \
        --output_dir "$EMBEDDINGS_DIR" \
        --batch_size 256 \
        --max_length 256
fi

# Step 2: Train hypothesis model
echo ""
echo "========================================"
echo "Step 2/6: Training hypothesis model"
echo "========================================"
echo "Expected time: 1-2 days on GPU"
echo ""

if [ -f "$SAVES_DIR/bge-small-hypothesis/best_model.pt" ]; then
    echo "Hypothesis model already exists, skipping..."
else
    python scripts/train_hypothesis.py \
        --embeddings_path "$EMBEDDINGS_DIR/embeddings.npy" \
        --texts_path "$EMBEDDINGS_DIR/texts.txt" \
        --output_dir "$SAVES_DIR/bge-small-hypothesis" \
        --batch_size 64 \
        --epochs 100 \
        --lr 1e-4 \
        --num_repeat_tokens 16 \
        --save_every 10
fi

# Step 3: Generate hypotheses for corrector training
echo ""
echo "========================================"
echo "Step 3/6: Generating hypotheses"
echo "========================================"
echo "Expected time: 4-8 hours on GPU"
echo ""

if [ -f "$HYPOTHESES_DIR/hypotheses.txt" ]; then
    echo "Hypotheses already exist, skipping..."
else
    python scripts/generate_hypotheses.py \
        --hypothesis_model_path "$SAVES_DIR/bge-small-hypothesis/best_model.pt" \
        --embeddings_path "$EMBEDDINGS_DIR/embeddings.npy" \
        --output_dir "$HYPOTHESES_DIR" \
        --batch_size 32
fi

# Step 4: Train corrector model
echo ""
echo "========================================"
echo "Step 4/6: Training corrector model"
echo "========================================"
echo "Expected time: 1-2 days on GPU"
echo ""

if [ -f "$SAVES_DIR/bge-small-corrector/best_model.pt" ]; then
    echo "Corrector model already exists, skipping..."
else
    python scripts/train_corrector.py \
        --embeddings_path "$EMBEDDINGS_DIR/embeddings.npy" \
        --texts_path "$EMBEDDINGS_DIR/texts.txt" \
        --hypotheses_path "$HYPOTHESES_DIR/hypotheses.txt" \
        --hypothesis_embeddings_path "$HYPOTHESES_DIR/hypothesis_embeddings.npy" \
        --output_dir "$SAVES_DIR/bge-small-corrector" \
        --batch_size 32 \
        --epochs 100 \
        --lr 1e-4 \
        --num_prefix_tokens 4 \
        --save_every 10
fi

# Step 5: Export to ONNX
echo ""
echo "========================================"
echo "Step 5/6: Exporting to ONNX"
echo "========================================"
echo ""

python scripts/export_onnx_full.py \
    --model_type hypothesis \
    --model_path "$SAVES_DIR/bge-small-hypothesis/best_model.pt" \
    --output_dir "$MODELS_DIR/hypothesis"

python scripts/export_onnx_full.py \
    --model_type corrector \
    --model_path "$SAVES_DIR/bge-small-corrector/best_model.pt" \
    --output_dir "$MODELS_DIR/corrector"

# Step 6: Test
echo ""
echo "========================================"
echo "Step 6/6: Testing inversion"
echo "========================================"
echo ""

python scripts/test_inversion.py \
    --hypothesis_model "$SAVES_DIR/bge-small-hypothesis/best_model.pt" \
    --corrector_model "$SAVES_DIR/bge-small-corrector/best_model.pt" \
    --num_corrections 10

echo ""
echo "========================================"
echo "TRAINING COMPLETE!"
echo "========================================"
echo ""
echo "Models saved to:"
echo "  - $SAVES_DIR/bge-small-hypothesis/best_model.pt"
echo "  - $SAVES_DIR/bge-small-corrector/best_model.pt"
echo ""
echo "ONNX exports saved to:"
echo "  - $MODELS_DIR/hypothesis"
echo "  - $MODELS_DIR/corrector"
echo ""
echo "Next steps:"
echo "  1. Upload to HuggingFace:"
echo "     huggingface-cli upload shivvr/bge-small-hypothesis $MODELS_DIR/hypothesis"
echo "     huggingface-cli upload shivvr/bge-small-corrector $MODELS_DIR/corrector"
echo ""
echo "  2. If Optimum export was skipped, install and re-run:"
echo "     pip install optimum[exporters] onnxruntime"
echo "     python scripts/export_onnx_full.py --model_type hypothesis --model_path $SAVES_DIR/bge-small-hypothesis/best_model.pt --output_dir $MODELS_DIR/hypothesis"
echo ""
