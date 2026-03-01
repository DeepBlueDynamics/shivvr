FROM debian:trixie-slim AS builder

# Install build deps + Rust (need glibc 2.38+ for ort/ONNX Runtime)
RUN apt-get update && apt-get install -y \
    pkg-config libssl-dev curl g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.88.0

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
COPY Cargo.toml ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release || true
RUN rm -rf src

COPY src ./src
RUN touch src/main.rs
RUN cargo build --release

# Download BGE-small model and tokenizer
RUN mkdir -p /models && \
    curl -L -o /models/bge-small-en-v1.5.onnx \
    "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx" && \
    curl -L -o /models/tokenizer.json \
    "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json"

# Vec2text inverter models are COPY'd from pre-exported local directory
# (requires running scripts/export_onnx_full.py first)
# Uncomment when ONNX exports are available:
# COPY onnx-export/hypothesis/projection.onnx /models/inverter/projection.onnx
# COPY onnx-export/hypothesis/t5-onnx/encoder.onnx /models/inverter/encoder.onnx
# COPY onnx-export/hypothesis/t5-onnx/decoder.onnx /models/inverter/decoder.onnx
# COPY onnx-export/hypothesis/t5-onnx/tokenizer.json /models/inverter/tokenizer.json

FROM debian:trixie-slim

RUN apt-get update && apt-get install -y \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/shivvr /shivvr
COPY --from=builder /models /models

ENV PORT=8080
ENV MODEL_PATH=/models/bge-small-en-v1.5.onnx
ENV TOKENIZER_PATH=/models/tokenizer.json
ENV DATA_PATH=/data/shivvr

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
