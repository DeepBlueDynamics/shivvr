#!/bin/bash
# Quick test to verify training works before committing to full run
# Uses only 10K samples, 3 epochs - should complete in ~10 minutes

set -e

echo "========================================"
echo "Quick Test (10K samples, 3 epochs)"
echo "========================================"

EMBEDDINGS_DIR="./embeddings/bge-small-test"
HYPOTHESES_DIR="./embeddings/bge-small-test-hypotheses"
SAVES_DIR="./saves-test"

mkdir -p "$EMBEDDINGS_DIR" "$HYPOTHESES_DIR" "$SAVES_DIR"

# Generate small test embeddings
echo ""
echo "Step 1: Generate test embeddings (10K samples)..."
python scripts/generate_bge_embeddings.py \
    --output_dir "$EMBEDDINGS_DIR" \
    --batch_size 128 \
    --max_samples 10000

# Train hypothesis model briefly
echo ""
echo "Step 2: Quick hypothesis training (3 epochs)..."
python scripts/train_hypothesis_12gb.py \
    --embeddings_path "$EMBEDDINGS_DIR/embeddings.npy" \
    --texts_path "$EMBEDDINGS_DIR/texts.txt" \
    --output_dir "$SAVES_DIR/bge-small-hypothesis" \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --epochs 3 \
    --max_samples 10000 \
    --fp16

# Generate hypotheses
echo ""
echo "Step 3: Generate test hypotheses..."
python scripts/generate_hypotheses.py \
    --hypothesis_model_path "$SAVES_DIR/bge-small-hypothesis/best_model.pt" \
    --embeddings_path "$EMBEDDINGS_DIR/embeddings.npy" \
    --output_dir "$HYPOTHESES_DIR" \
    --batch_size 16 \
    --max_samples 10000

# Train corrector briefly
echo ""
echo "Step 4: Quick corrector training (3 epochs)..."
python scripts/train_corrector_12gb.py \
    --embeddings_path "$EMBEDDINGS_DIR/embeddings.npy" \
    --texts_path "$EMBEDDINGS_DIR/texts.txt" \
    --hypotheses_path "$HYPOTHESES_DIR/hypotheses.txt" \
    --hypothesis_embeddings_path "$HYPOTHESES_DIR/hypothesis_embeddings.npy" \
    --output_dir "$SAVES_DIR/bge-small-corrector" \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --epochs 3 \
    --max_samples 10000 \
    --fp16

# Test
echo ""
echo "Step 5: Test inversion..."
python scripts/test_inversion.py \
    --hypothesis_model "$SAVES_DIR/bge-small-hypothesis/best_model.pt" \
    --corrector_model "$SAVES_DIR/bge-small-corrector/best_model.pt" \
    --num_corrections 5

echo ""
echo "========================================"
echo "Quick test complete!"
echo "========================================"
echo ""
echo "If this worked, run the full training:"
echo "  ./run_training_12gb.sh"
echo ""
echo "Note: Results will be poor with only 3 epochs."
echo "Full training needs 50-100 epochs to converge."
