# Stage 2: Rust build (CUDA)
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y \
    pkg-config libssl-dev curl g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf --retry 5 --retry-delay 10 https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.88.0

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Vendor sub-crate (lume-hybrid) comes in with the rest of the source tree —
# Cargo's `path = "vendor/lume-hybrid"` resolves inside /app/vendor/.
COPY Cargo.toml ./
COPY vendor ./vendor
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
COPY models/ /models/

ENV LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:/usr/lib/onnxruntime
ENV PORT=8080
ENV MODEL_PATH=/models/gtr-t5-base.onnx
ENV TOKENIZER_PATH=/models/tokenizer.json
ENV VISION_MODEL_PATH=/models/siglip-vision.onnx
ENV SIGLIP_TEXT_MODEL_PATH=/models/siglip-text.onnx
ENV SIGLIP_TOKENIZER_PATH=/models/siglip-tokenizer.json
ENV TRANSCRIPTION_URL=http://hyperia-transcription:8765
ENV INVERTER_PROJECTION_PATH=/models/inverter/projection.onnx
ENV INVERTER_ENCODER_PATH=/models/inverter/encoder.onnx
ENV INVERTER_DECODER_PATH=/models/inverter/decoder.onnx
ENV INVERTER_TOKENIZER_PATH=/models/inverter/tokenizer.json
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8080
ENTRYPOINT ["/shivvr"]
