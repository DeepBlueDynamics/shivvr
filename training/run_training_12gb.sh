#!/bin/bash
# Training pipeline optimized for 12GB GPU (RTX 3080/4080)
# Uses smaller batches + gradient accumulation + mixed precision

set -e

echo "========================================"
echo "BGE-Small Inversion Training (12GB GPU)"
echo "========================================"

EMBEDDINGS_DIR="./embeddings/bge-small-msmarco"
HYPOTHESES_DIR="./embeddings/bge-small-msmarco-hypotheses"
SAVES_DIR="./saves"
MODELS_DIR="./models"

mkdir -p "$EMBEDDINGS_DIR" "$HYPOTHESES_DIR" "$SAVES_DIR" "$MODELS_DIR"

# Step 1: Generate embeddings (this is fine, inference only)
echo ""
echo "Step 1/6: Generating BGE embeddings (~2-4 hours)"
if [ ! -f "$EMBEDDINGS_DIR/embeddings.npy" ]; then
    python scripts/generate_bge_embeddings.py \
        --output_dir "$EMBEDDINGS_DIR" \
        --batch_size 128 \
        --max_length 256
else
    echo "Skipping, already exists"
fi

# Step 2: Train hypothesis model
# Original: batch_size=64
# 12GB GPU: batch_size=8, gradient_accumulation=8 (effective batch=64)
echo ""
echo "Step 2/6: Training hypothesis model (~2-3 days)"
if [ ! -f "$SAVES_DIR/bge-small-hypothesis/best_model.pt" ]; then
    python scripts/train_hypothesis_12gb.py \
        --embeddings_path "$EMBEDDINGS_DIR/embeddings.npy" \
        --texts_path "$EMBEDDINGS_DIR/texts.txt" \
        --output_dir "$SAVES_DIR/bge-small-hypothesis" \
        --batch_size 8 \
        --gradient_accumulation_steps 8 \
        --epochs 50 \
        --lr 1e-4 \
        --fp16
else
    echo "Skipping, already exists"
fi

# Step 3: Generate hypotheses
echo ""
echo "Step 3/6: Generating hypotheses (~4-6 hours)"
if [ ! -f "$HYPOTHESES_DIR/hypotheses.txt" ]; then
    python scripts/generate_hypotheses.py \
        --hypothesis_model_path "$SAVES_DIR/bge-small-hypothesis/best_model.pt" \
        --embeddings_path "$EMBEDDINGS_DIR/embeddings.npy" \
        --output_dir "$HYPOTHESES_DIR" \
        --batch_size 16
else
    echo "Skipping, already exists"
fi

# Step 4: Train corrector
echo ""
echo "Step 4/6: Training corrector model (~2-3 days)"
if [ ! -f "$SAVES_DIR/bge-small-corrector/best_model.pt" ]; then
    python scripts/train_corrector_12gb.py \
        --embeddings_path "$EMBEDDINGS_DIR/embeddings.npy" \
        --texts_path "$EMBEDDINGS_DIR/texts.txt" \
        --hypotheses_path "$HYPOTHESES_DIR/hypotheses.txt" \
        --hypothesis_embeddings_path "$HYPOTHESES_DIR/hypothesis_embeddings.npy" \
        --output_dir "$SAVES_DIR/bge-small-corrector" \
        --batch_size 4 \
        --gradient_accumulation_steps 16 \
        --epochs 50 \
        --lr 1e-4 \
        --fp16
else
    echo "Skipping, already exists"
fi

# Step 5: Export
echo ""
echo "Step 5/6: Exporting models"
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
echo "Step 6/6: Testing"
python scripts/test_inversion.py \
    --hypothesis_model "$SAVES_DIR/bge-small-hypothesis/best_model.pt" \
    --corrector_model "$SAVES_DIR/bge-small-corrector/best_model.pt" \
    --num_corrections 10

echo ""
echo "Done! Total time: ~5-7 days on 12GB GPU"
