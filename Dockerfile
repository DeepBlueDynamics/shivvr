FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS builder

# Install build deps + Rust
RUN apt-get update && apt-get install -y \
    pkg-config libssl-dev curl g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.88.0

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
COPY Cargo.toml ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release --features cuda || true
RUN rm -rf src

COPY src ./src
RUN touch src/main.rs
RUN cargo build --release --features cuda

# Embedder + inverter ONNX models (pre-exported via scripts/export_gtr_models.py)
# Run: python scripts/export_gtr_models.py --output_dir models/
# This produces gtr-t5-base.onnx, tokenizer.json, and inverter/ directory
COPY models/gtr-t5-base.onnx /models/gtr-t5-base.onnx
COPY models/tokenizer.json /models/tokenizer.json
COPY models/inverter/projection.onnx /models/inverter/projection.onnx
COPY models/inverter/encoder.onnx /models/inverter/encoder.onnx
COPY models/inverter/decoder.onnx /models/inverter/decoder.onnx
COPY models/inverter/tokenizer.json /models/inverter/tokenizer.json

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/shivvr /shivvr
COPY --from=builder /models /models

ENV PORT=8080
ENV MODEL_PATH=/models/gtr-t5-base.onnx
ENV TOKENIZER_PATH=/models/tokenizer.json
ENV DATA_PATH=/data/shivvr
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Phase 1: OpenAI — set at runtime for ada-002 retrieve embeddings
# ENV OPENAI_API_KEY=

# Phase 3: Vec2text inverter paths
ENV INVERTER_PROJECTION_PATH=/models/inverter/projection.onnx
ENV INVERTER_ENCODER_PATH=/models/inverter/encoder.onnx
ENV INVERTER_DECODER_PATH=/models/inverter/decoder.onnx
ENV INVERTER_TOKENIZER_PATH=/models/inverter/tokenizer.json

EXPOSE 8080
VOLUME ["/data"]
ENTRYPOINT ["/shivvr"]
