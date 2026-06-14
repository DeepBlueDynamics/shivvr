#!/usr/bin/env bash
# Fetch / build ONNX model artifacts for shivvr.
#
# Models are NOT committed to git (they are gitignored). Run this script once
# before building the Docker image. Output goes to models/ in the repo root.
#
# What it produces:
#   models/gtr-t5-base.onnx          — embedder (768d, ~230 MB)
#   models/gtr-t5-base.onnx.data     — external weights (PyTorch export artefact)
#   models/tokenizer.json             — embedder tokenizer
#   models/inverter/projection.onnx  — vec2text projection
#   models/inverter/encoder.onnx     — T5 encoder
#   models/inverter/decoder.onnx     — T5 decoder
#   models/inverter/tokenizer.json   — T5 tokenizer
#
# Requirements (Python):
#   pip install torch transformers sentence-transformers vec2text onnx onnxruntime
#
# Estimated time: 5-15 min (downloads ~4 GB of HuggingFace weights first time)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.9+ and retry."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PYTHON_VERSION"

# Check for required Python packages
MISSING_PKGS=()
for pkg in torch transformers sentence_transformers vec2text onnx onnxruntime; do
    if ! python3 -c "import $pkg" &>/dev/null 2>&1; then
        MISSING_PKGS+=("$pkg")
    fi
done

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
    echo ""
    echo "Missing Python packages: ${MISSING_PKGS[*]}"
    echo ""
    echo "Install with:"
    echo "  pip install torch transformers sentence-transformers vec2text onnx onnxruntime"
    echo ""
    read -r -p "Attempt automatic install now? [y/N] " REPLY
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        pip install torch transformers sentence-transformers vec2text onnx onnxruntime
    else
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Determine what needs to be built
# ---------------------------------------------------------------------------

NEED_EMBEDDER=false
NEED_INVERTER=false

if [[ ! -f "$MODELS_DIR/gtr-t5-base.onnx" || ! -f "$MODELS_DIR/tokenizer.json" ]]; then
    NEED_EMBEDDER=true
fi

if [[ ! -f "$MODELS_DIR/inverter/projection.onnx" || \
      ! -f "$MODELS_DIR/inverter/encoder.onnx"    || \
      ! -f "$MODELS_DIR/inverter/decoder.onnx"    || \
      ! -f "$MODELS_DIR/inverter/tokenizer.json" ]]; then
    NEED_INVERTER=true
fi

if [[ "$NEED_EMBEDDER" == "false" && "$NEED_INVERTER" == "false" ]]; then
    echo "All model files already present in $MODELS_DIR"
    echo "Use --force to re-export anyway."
    if [[ "${1:-}" != "--force" ]]; then
        exit 0
    fi
    NEED_EMBEDDER=true
    NEED_INVERTER=true
fi

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

EXPORT_ARGS="--output_dir $MODELS_DIR"
if [[ "$NEED_EMBEDDER" == "false" && "$NEED_INVERTER" == "true" ]]; then
    EXPORT_ARGS="$EXPORT_ARGS --skip-embedder"
elif [[ "$NEED_EMBEDDER" == "true" && "$NEED_INVERTER" == "false" ]]; then
    EXPORT_ARGS="$EXPORT_ARGS --skip-inverter"
fi

echo ""
echo "==> Exporting models to $MODELS_DIR ..."
echo "    (First run downloads ~4 GB from HuggingFace Hub)"
echo ""

# shellcheck disable=SC2086
python3 "$SCRIPT_DIR/export_gtr_models.py" $EXPORT_ARGS --verify

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "==> Done. Model files:"
find "$MODELS_DIR" -name "*.onnx" -o -name "tokenizer.json" | sort | while read -r f; do
    SIZE=$(du -sh "$f" 2>/dev/null | cut -f1)
    echo "    [$SIZE]  ${f#$REPO_ROOT/}"
done

echo ""
echo "You can now build the Docker image:"
echo "  docker build -t shivvr ."
