# Stage 1: Export GTR-T5-base to ONNX (downloads from HuggingFace at build time)
FROM python:3.11-slim AS model-exporter

RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.5.1+cpu" \
    "transformers==4.44.2" \
    "sentence-transformers==3.2.1" \
    "vec2text" \
    "onnx==1.16.2" \
    "onnxruntime==1.19.2"

COPY scripts/export_gtr_models.py /export_gtr_models.py

RUN python /export_gtr_models.py --output_dir /models

# Stage 2: Rust build (CUDA)
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y \
    pkg-config libssl-dev curl g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf --retry 5 --retry-delay 10 https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.88.0

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
COPY Cargo.toml ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release --features cuda || true
RUN rm -rf src

COPY src ./src
RUN touch src/main.rs
RUN cargo build --release --features cuda

RUN mkdir -p /ort-libs && \
    find /root/.cache/ort.pyke.io/dfbin -name "libonnxruntime*.so*" -exec cp {} /ort-libs/ \;

# Stage 3: Runtime (CUDA L4)
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    ca-certificates curl cuda-compat-12-6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/shivvr /shivvr
COPY --from=builder /ort-libs /usr/lib/onnxruntime/
COPY --from=model-exporter /models /models

ENV LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:/usr/lib/onnxruntime
ENV PORT=8080
ENV MODEL_PATH=/models/gtr-t5-base.onnx
ENV TOKENIZER_PATH=/models/tokenizer.json
ENV INVERTER_PROJECTION_PATH=/models/inverter/projection.onnx
ENV INVERTER_ENCODER_PATH=/models/inverter/encoder.onnx
ENV INVERTER_DECODER_PATH=/models/inverter/decoder.onnx
ENV INVERTER_TOKENIZER_PATH=/models/inverter/tokenizer.json
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8080
ENTRYPOINT ["/shivvr"]
