# Shivvr Review Update

Date: 2026-04-08

## Runtime overview

- The service is a Rust `axum` HTTP API with a single process entrypoint in `src/main.rs`.
- The required local embedding path is ONNX-based `gtr-t5-base` loaded through ONNX Runtime.
- Optional retrieve embeddings come from OpenAI via `/v1/embeddings`.
- Persistence is local `sled`, keyed by session and chunk id.
- Optional per-agent vector encryption is applied before persistence if agent keys have been registered.
- Optional vec2text inversion is available if the inverter ONNX models are present.

## Search path

1. `POST /memory/:session_id/ingest`
2. Text is chunked by sentence splitting plus boundary sampling.
3. Every chunk gets an organize embedding from the local ONNX model.
4. If `OPENAI_API_KEY` is set, every chunk also gets a retrieve embedding from OpenAI.
5. If `agent_id` is provided and keys exist, stored embeddings are encrypted before being written.
6. Chunks are written into `sled` as JSON blobs.

1. `GET /memory/:session_id/search`
2. Query text is embedded with:
   - local `gtr-t5-base` for `role=organize`
   - OpenAI embedding model for `role=retrieve` when available
   - local `gtr-t5-base` as fallback when OpenAI is unavailable
3. If `agent_id` is supplied and keys exist, the query vector is encrypted before comparison.
4. The service loads all chunks for the session from `sled`.
5. It computes cosine similarity in-process over the selected embedding field.
6. Results are sorted in memory and truncated to top `n`.

## What gets stored

- All chunk texts are stored.
- All organize vectors are stored.
- Retrieve vectors are also stored when OpenAI embedding generation succeeds at ingest time.
- Metadata, source, token count, timestamps, emotion fields, encryption flag, and `agent_id` are stored.
- Search does not use an external vector database or ANN index. It scans the full session in memory each time.

## Important operational reality

- Local Docker has persistent storage through the named volume `shivvr-data`.
- Cloud Run does not give persistent local disk across instance shutdowns. `DATA_PATH=/data/shivvr` only persists for the life of an instance.
- Because the deployment sets `--min-instances 0`, Cloud Run can scale to zero and lose the local `sled` state.
- Because the deployment also sets `--max-instances 1`, there is no multi-instance consistency strategy anyway.

## Highest-priority update work

1. Decide whether Cloud Run is allowed to be ephemeral. If not, move persistence off local `sled` or mount a supported external storage layer.
2. Replace `text-embedding-ada-002` with a current embedding model and make the model configurable.
3. Fix the retrieve fallback path so a failed or missing OpenAI embedding does not compare a 768d query against 1536d stored vectors.
4. Bring docs, comments, health output, and training assets into alignment around the actual production model stack.
5. Add a reproducible model build or artifact fetch path because the repo currently excludes `models/` from version control but the Docker build requires those files.
6. Reduce Cloud Run concurrency for GPU-backed inference unless benchmarking proves `80` is safe with the ONNX session locking model.

## Notes on repo drift

- Production code is centered on `gtr-t5-base` and optional OpenAI retrieve embeddings.
- Older specs and training docs still describe `bge-small` heavily.
- The current training directory appears to target the older BGE-based inversion pipeline, while the Docker image expects pre-exported GTR-based inverter assets.
