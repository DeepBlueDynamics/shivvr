# EmbeddingGemma 2 in shivvr, and training our own embedder

Status as of 2026-10-06. Owner: Kord. Work packages (WP) are sized for one worker each.

## 1. The model

`google/embeddinggemma-2` (Apache 2.0, not gated).

| | |
|---|---|
| Parameters | 740M total: 270M text (130M transformer + 140M token embeddings), 170M vision, 300M audio |
| Output | 768d, L2-normalized; Matryoshka truncation to 512/256/128 (take the leading components, re-normalize) |
| Context | 8,192 tokens, shared across modalities. Text layers: 24, sliding window 512 with a full-attention layer every 6th |
| Pooling | Mean over all tokens, prompt included (`1_Pooling`: `include_prompt: true`), then Normalize |
| Prompts | Text only. `query` → `task: search result \| query: `, `document` → `title: none \| text: `, plus Clustering / Classification / STS / CodeRetrieval / QA / FactChecking (`config_sentence_transformers.json`) |
| Precision | float32 or bfloat16. **float16 produces NaN or silently degraded vectors** |
| Stack | `transformers` 5.18 (checkpoint written by 5.18.0.dev0), `sentence-transformers` >= 6.1 |
| Images | `Gemma4ImageProcessor`: patch 16, pool 3, 280 soft tokens by default (70–1120), rescale 1/255, no mean/std normalization |
| Audio | 16 kHz mono, 128-bin log-mel (fft 512, hop 160), 40 ms per token (25 tokens/s) |
| Video | 1 fps frames through the vision encoder, 140 soft tokens per frame, max 32 frames |

Multimodal inputs go through one sequence: placeholders `<|image|>`, `<|audio|>` and `<|video|>` in
the text are filled with soft tokens from the encoders, and the whole sequence is mean-pooled. An
image alone is therefore embedded as a short token sequence run through the text model, not as a
separate tower output like SigLIP's.

## 2. Done in this change

- `POST /embed` accepts `"model": "embeddinggemma-2"` with optional `task` (prompt name, so the
  caller doesn't hand-write prefixes) and `dimensions` (128/256/512/768). Validation and prefix table
  are in `src/embed_api.rs`, with unit tests (`cargo test --lib --no-default-features embed_api`).
  `task`/`dimensions` are rejected for GTR and SigLIP.
- `src/embeddinggemma.rs`: `EmbeddingGemma2Embedder`. Loads the ONNX model and tokenizer JSON,
  truncates input to 8,192 tokens, one text per run, CUDA when available
  (`embedder::build_session`, extracted from the GTR embedder so both share the CUDA probe).
- Optional at startup like SigLIP: missing files log a line and `/embed` answers 503 for that model.
  `/health` lists it with role `text`.
- `scripts/export_embeddinggemma2.py`: text-only load, wrapper with masked mean pool + L2 norm,
  TorchScript export with a dynamo fallback, and two checks that fail the export: the tokenizer JSON
  must reproduce SentenceTransformer's token ids, and the ONNX output must reach cosine >= 0.9999
  against `SentenceTransformer.encode()` (short, prefixed and >512-token inputs).
- `Dockerfile.models`: a separate first stage on Python 3.12 with the new stack; only the two output
  files are copied into the existing stage. Cloud Build timeout raised to 3600s.
- Env vars `EMBEDDINGGEMMA2_MODEL_PATH` and `EMBEDDINGGEMMA2_TOKENIZER_PATH` in the Dockerfile,
  docker-compose, README and OPERATIONS.
- Docs: OPERATIONS.md has the models-image contents, `/health` model list and a troubleshooting entry;
  the backend homepage and landing page (`src/api.rs`, `src/landing.html`) list the `/embed` models and
  the `task`/`dimensions` fields. The MCP tools do not expose `/embed`, and there is no
  OpenAI-compatible `/v1/embeddings` route, so neither needed changes.
**Verified by WP1 (2026-10-07):** export passes both checks, Rust `/embed` matches Python (min cosine
0.99999988). Docker stage and `deploy.sh --rebuild-models` still not run.
produced a real vector yet. WP1 is the gate for everything else.

## 3. Work packages

### WP1: Run the text export and check it (blocker) — PASS (2026-10-07)

Environment: fresh `/workspace/embeddinggemma2-wp1-venv`, Python 3.11.2, Torch 2.14.1+cpu,
Transformers 5.19.0, Sentence Transformers 6.1.0, ONNX 1.23.2, ONNX Script 0.7.2, and
ONNX Runtime 1.30.0. Transformers 5.19.0 was available from PyPI, so no git pin was needed.
Sentence Transformers' processor also imports the image stack during a text-only load; Pillow 12.3.0
and torchvision 0.29.1+cpu were therefore required and were added to `Dockerfile.models`.

Export command: `python scripts/export_embeddinggemma2.py --output_dir models/`.

- Fast-tokenizer JSON matched Sentence Transformers on all 18 strict test inputs.
- The legacy TorchScript exporter succeeded at opset 18; the dynamo fallback and eager attention
  workaround were not needed.
- The wrapper returned 768 dimensions.
- ONNX versus `SentenceTransformer.encode()` passed with worst cosine **1.000000** (required
  minimum remains 0.9999).
- `embeddinggemma2-text.onnx` is **1,085,716,408 bytes (1,085.7 MB / 1,035.4 MiB)**.
- CPU ONNX Runtime session load time was **8.885 s**.

The Windows Rust debug service loaded the exported files and served `POST /embed`. For two query
texts, the minimum cosine against `SentenceTransformer.encode(prompt_name="query")` was
**0.99999988** at 768 dimensions. The 256-dimensional Matryoshka response had the expected
`[2, 256]` shape and minimum cosine **1.00000012** against the truncated, re-normalized Python
vectors.

Latency is warm inference for one text. CPU numbers are direct ONNX Runtime with exactly four
intra-op threads and exact synthetic sequence lengths. RTX 3060 numbers are end-to-end Rust
`POST /embed` wall times with exact tokenizer lengths, so they include tokenization, JSON, and
local HTTP overhead. Each row had one unreported warm-up run.

| Tokens | CPU, 4 threads | RTX 3060 12 GB (CUDA) |
|---:|---:|---:|
| 16 | 42.51 ms (mean of 3) | 51.81 ms (mean of 5) |
| 128 | 135.43 ms (mean of 3) | 158.85 ms (mean of 5) |
| 512 | 596.68 ms (mean of 3) | 696.41 ms (mean of 5) |
| 2,048 | 5,025.27 ms (mean of 2) | 5,303.78 ms (mean of 3) |
| 8,192 | 91,721.23 ms (1 run) | 69,343.32 ms (1 run) |

The RTX service logged that its CUDA session was created and probe-verified. No L4 was available in
this environment, so L4 latency was not measured. The 8,192-token CPU run used about 7.9 GB RSS.
These results make a lower serving cap worth evaluating in WP2.

**Follow-ups from these numbers (2026-10-07):**
- Default input cap lowered to 2,048 tokens (`EMBEDDINGGEMMA2_MAX_TOKENS`, max 8,192), so one request
  can no longer pin a CPU for 90 s and ~8 GB RSS.
- The GPU is no faster than CPU up to 2,048 tokens, so the CUDA path is not effective as exported.
  Suspects: ops falling back to the CPU provider (check ORT verbose logs for node placement),
  per-run host/device copies, and attention exported as dense masked matmuls, so the 512-token
  sliding window saves nothing (that would also explain the super-linear growth). Try: ORT
  `optimum` / transformer optimizer fusion, the dynamo export with SDPA, and IO binding. Measure on
  the L4 before deciding.
- The super-linear growth matters most on the Pi (section 5): expect a cap of 512 tokens or less there.

Build notes:

- Windows `cargo build --release --features cuda --bin shivvr` succeeded in 2m15s.
- The Linux container's release link failed because its downloaded ONNX Runtime referenced newer
  glibc `__isoc23_*` symbols.
- The repository Docker build reached Rust compilation but failed because the application
  `Dockerfile` pins Rust 1.88 while the current lockfile's uuid 1.27.0 requires Rust 1.89. That
  Dockerfile was outside this worker's allowed edit set.
- `bash deploy.sh --rebuild-models` was not run; no deployment was requested for this work package.

### WP2: Tests and throughput
- Golden fixture: export script writes `tests/fixtures/embeddinggemma2_golden.json` (texts, task,
  expected 768d vectors); an `#[ignore]`d Rust test loads the model and checks cosine >= 0.9999.
- Add `/embed` model=embeddinggemma-2 cases to `tests/functional.py`: dims, task prefixes, a
  query-vs-document ranking sanity check, and a 503 when the model is missing.
- Batching: the export already masks padding in the pooling, so padded batches are safe for this
  model (unlike GTR). Batch per length bucket inside the blocking task. Measure before keeping it.
- Long inputs: done. Capped at 2,048 tokens by default (see WP1 follow-ups); revisit after the GPU fix.
- Fix the Windows link error before running tests locally (MSVC `LNK2005` between `msvcprt` and
  `libcpmt`; it exists on `main` already, not caused by this change). Unit tests run with
  `--no-default-features` in the meantime.

### WP3: Images
- Export: a graph taking `pixel_values` (+ the patch position inputs the Gemma4 vision encoder needs)
  and the token sequence `<bos> <|image> [280 × <|image|>] <image|> <eos>` (check the exact
  template in the processor), returning the pooled, normalized vector. Verify against
  `st.encode([{"image": ...}])`.
- Rust: port `Gemma4ImageProcessor` (aspect-preserving resize to a patch grid that yields 280 soft
  tokens after 3×3 pooling, bicubic, rescale 1/255, no normalization). This is the hard part; write
  it against golden tensors from Python, not by reading the spec alone.
- API: `POST /image/embed` gets `"model": "embeddinggemma-2"` (SigLIP stays the default). Optional
  `soft_tokens` (70–1120) later.
- The vision encoder can be a separate ONNX file so text-only deployments skip it.

### WP4: Audio
- Today `/audio/embed` transcribes and embeds the transcript with GTR. EmbeddingGemma 2 can embed
  audio directly, which captures sound and speaker, not just words. Offer it as
  `"model": "embeddinggemma-2"` on `/audio/embed`, transcription path unchanged as the default.
- Export the audio encoder (12 layers, chunked attention) plus the text path. Port the log-mel
  feature extractor to Rust, or fold it into the ONNX graph (STFT is available in opset 17).
- Input: 16 kHz mono; resample in Rust. ~327 s max per input at 25 tokens/s.

### WP5: Using it for sessions and temp stores (decision needed)
Ingest stores GTR vectors and every session assumes one 768d space. Options:
1. Leave ingest on GTR; EmbeddingGemma 2 is `/embed` only. Zero risk. (Current state.)
2. Per-session embedding model chosen at creation (`model` on create), stored with the session;
   search, encryption keys (dim² floats) and `/invert` (GTR only) check it.
3. Store both vectors per chunk. Doubles memory; only worth it if search quality proves it.

Recommendation: option 2 after WP1/WP2, gated on an in-domain retrieval comparison (section 4.2)
showing EmbeddingGemma 2 beats GTR on our data.

### WP6: Inversion for EmbeddingGemma 2 vectors (optional, expensive)
`/invert` is tied to GTR's space. To invert EmbeddingGemma 2 vectors, retrain the vec2text pair on its
embeddings (see 4.4). Not needed unless sessions move off GTR.

## 4. Training our own embedding model

### 4.1 What exists to build on
None of deckhand, sailfish or `shivvr/training` trains an embedder. What we can reuse:

| From | Piece | Use |
|---|---|---|
| sailfish | `train/byo_gcloud/train_on_gcloud.sh.tmpl`, `vm_startup.sh` | Spot A100 40GB (`a2-highgpu-1g`) VM that deletes itself on exit, 3 h max runtime, results pushed to HF |
| sailfish | `app/train.py` | Bundles corpus + trainer + requirements into the launch template |
| sailfish | `train/hfjobs_launch.py` | HF Jobs runs (A10G / A100) without managing a VM |
| sailfish | `app/curate.py` | LLM curation with a cost cap; adapt to generate and filter query/passage pairs |
| sailfish | `scrape/scrape_toolcalls.py` | Transcript scraping and ANSI scrubbing; a source of in-domain text |
| sailfish | `drafter/ngram_tool_drafter.mjs` | Hash-based train/held-out split |
| sailfish | Process lessons (`VSD_BURN_RESULTS.md`, `ARCHITECTURE.md`) | Train and evaluate in the exact precision/format you serve; checkpoint every N steps and keep the best; quality peaked early then diverged |
| shivvr | `training/scripts/generate_bge_embeddings.py` | Streams teacher embeddings to a memmap; reusable for distillation |
| shivvr | `training/` hypothesis + corrector trainers | Retrain the inverter for a new space (WP6). Note: `train_hypothesis.py` / `train_corrector.py` hard-code 384d and `run_training_a100.sh` still references bge-small; the `_12gb` variants are the 768d ones |
| deckhand | `deploy-largish-model.ps1` | Cloud Run GPU (L4) serving with a GCS model cache; only relevant if a large teacher must be served |

Hardware already in use: spot A100 40GB on GCP, HF Jobs A10G/A100, a local RTX 3060 12GB (Ampere,
so bf16 works).

### 4.2 First: an evaluation set (needed whatever we train)
We can't tell whether training helped without in-domain numbers. Build this before training anything:
- Corpus: what shivvr actually stores. Sample ingested chunks (with consent), plus nutnews items,
  docs from our repos, and Claude Code transcripts (sailfish scraper).
- Queries: 1–3 per passage, generated by a frontier model through a `curate.py`-style cost-capped
  step, then filtered (the query must retrieve its passage with a strong model, and must not just copy it).
- Hold out by hash. ~2–5k queries is enough to separate models.
- Metrics: nDCG@10, recall@10/100, MRR. Also run an MTEB English retrieval subset
  (SciFact, NFCorpus, FiQA, ArguAna) to catch regressions on general text.
- Baselines: GTR-T5-base, EmbeddingGemma 2 at 768/256/128, SigLIP text (expected weak), ada-002.
- Script: `training/embed_eval/` that takes any model through the same `/embed` API, so the deployed
  path is what gets measured (the sailfish lesson).

### 4.3 Training options, cheapest first

**A. Fine-tune EmbeddingGemma 2's text path on our data (recommended first).**
- sentence-transformers 6.x `SentenceTransformerTrainer`, loss
  `MatryoshkaLoss(CachedMultipleNegativesRankingLoss)` over dims [768, 512, 256, 128], so truncation
  keeps working.
- Data: (query, positive) pairs from 4.2's generator on a training split, plus 1–3 hard negatives per
  pair mined with the base model (rank 10–50, dropping near-duplicates of the positive to avoid false
  negatives). Use the same `query` / `document` prompts as at serving time.
- Size: 50k–500k pairs. Text-only load (270M). bf16, lr 2e-5, warmup 5%, 1–3 epochs, large in-batch
  negatives via the cached loss (effective batch 512+ on one A100; 64–128 on the 3060).
- Cost: hours on one spot A100 via the sailfish launcher; a 3060 works for small runs.
- Guard against forgetting: mix in 20–30% general retrieval pairs (MS MARCO, NQ) and keep the MTEB
  subset as a gate.
- Ships through the same `export_embeddinggemma2.py` (point `--model_id` at the fine-tuned checkpoint),
  served as a new model name, e.g. `embeddinggemma-2-nuts`.

**B. Distill a small, fast student.**
- When: if WP1 shows EmbeddingGemma 2 is too slow on CPU for ingest volume.
- Teacher: EmbeddingGemma 2 (or the A fine-tune). Student: a 20–110M encoder (MiniLM-L12, bge-small,
  or GTR-T5-base itself).
- Loss: MSE to teacher vectors (plus a projection if dims differ), optionally combined with a
  contrastive loss on 4.2 pairs. `generate_bge_embeddings.py` already streams teacher vectors to a memmap.
- Data: millions of unlabelled in-domain sentences; no labels needed.
- Cost: under a day on an A100; serves on CPU like GTR does today.

**C. Train into GTR's space (keep `/invert` working).**
- Same as B, but the student targets GTR-T5-base vectors while training on a contrastive objective, so
  the existing inverter still decodes its output. It limits how far quality can move from GTR; worth it
  only if inversion is a product requirement for the new model.

**D. Pretrain from scratch.** Not justified. EmbeddingGemma 2 was trained on web-scale multimodal
data; we have neither the data nor the budget to beat it, and A/B get the in-domain gains.

### 4.4 Inverter retraining (if sessions move to a new space)
`training/` trains the vec2text hypothesis + corrector on MS MARCO embeddings: 3–5 days on one
A100/3090, 5–7 days on 12 GB. For a new space: regenerate embeddings with the new model, fix the
384d hard-coding in `train_hypothesis.py` / `train_corrector.py`, run the `_12gb` or A100 scripts, and
evaluate with `test_reconstruction_quality.py` (embed → invert → re-embed cosine).

### 4.5 Proposed order
1. WP1 (export + numbers). Decides CPU vs GPU and whether B is needed.
2. 4.2 eval set and harness. Run baselines.
3. If EmbeddingGemma 2 wins in-domain: WP2, then WP5 option 2.
4. If the in-domain gap to a fine-tune looks worth it: option A on a spot A100, gated on the eval set
   and the MTEB subset.
5. WP3/WP4 (images, native audio) in parallel once WP1 lands; they don't depend on training.

## 5. Running on a Raspberry Pi (HaLOS, via lume)

Source: the lume session, 2026-10-06. Lume calls shivvr over HTTP (`SHIVVR_BASE_URL`) and runs no ML
locally, so nobody has run shivvr or ONNX Runtime on the Pi yet.

**Target as tested by lume:** a plain Raspberry Pi 5, 8 GB, no HAT and no NPU, running the HaLOS Marine
image (Debian 13 trixie, aarch64), with a 28 GB SD card (~12 GB free). A real HALPI2 (CM5) is untested.
The Pi already runs Signal K, InfluxDB, Grafana, QuestDB, OpenCPN, Traefik and Authelia in Docker,
leaving **~3.4 GB of RAM free**. Without an Active Cooler it reaches 74–86 °C under load and throttles
near 85 °C.

**Memory budget (fp32):**

| Model | Size |
|---|---|
| GTR-T5-base | ~0.23 GB |
| SigLIP vision + text | ~0.8 GB |
| vec2text inverter | ~1 GB |
| EmbeddingGemma 2 text | ~1.1 GB |

Plus ORT arenas. Loading everything doesn't fit in 3.4 GB alongside the existing stack.

**Plan for a Pi build (WP7):**
1. **Choose models per deployment.** Each model is already optional at startup, so a Pi profile can just
   leave files out (e.g. GTR + EmbeddingGemma 2 text only, no inverter or SigLIP). Add a
   `SHIVVR_MODELS` allowlist so an unwanted model isn't loaded just because its file exists.
2. **Quantization:** for EmbeddingGemma 2, int8 dynamic quantization
   (`onnxruntime.quantization.quantize_dynamic`, weights only). **Not fp16**: the model produces NaN in
   float16 (model card), and ORT on the Pi's CPU gains nothing from fp16 anyway. Re-run the export's
   parity check with a looser, explicit threshold for the int8 file (e.g. cosine >= 0.99), and run the
   4.2 eval set to measure the retrieval loss. Same treatment for GTR.
3. **Build:** an aarch64 Linux image via `docker buildx --platform linux/arm64` on a dev machine, not
   on-device. Lume's on-device Rust builds take ~50 min and OOM with fat LTO; if building on the Pi, use
   lume's overrides: `CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
   CARGO_BUILD_JOBS=2`. CPU only, so no `cuda` feature. Check that ort's download-binaries ships
   linux-aarch64 and note its glibc floor.
4. **Packaging:** a HaLOS container app, not a process inside the Signal K container. That avoids
   Signal K's glibc 2.39 ceiling. Layout: `apps/shivvr/{docker-compose.yml, metadata.json
   (package_name shivvr-container, architecture arm64, web_ui), config.yml}`, built into a .deb for the
   marine container store; see halos-org/halos-marine-containers `docs/DESIGN.md`. Models come from a
   volume rather than baked into the image, to keep image pulls on the SD card small.
5. **Measure:** latency per text at 16/128/512/2048 tokens, peak RSS and temperature, with and without
   an Active Cooler (recommended for sustained inference). Cap the token count for EmbeddingGemma 2 on
   the Pi based on the numbers.
6. **Fallback:** if on-Pi inference is too slow, the Pi keeps calling a remote shivvr (today's setup),
   or a distilled student (4.3 B) runs on it instead.

Lume references: `plan/SETUP.md` §3 and `plan/STATUS.md` ("Pi" sections) in the lume repo.

## 6. Open decisions
- Default `task` when the caller gives none: currently no prefix (matches `encode()` without a
  prompt). Alternative: default `document`. No prefix is the least surprising default.
- Whether EmbeddingGemma 2 replaces SigLIP for images once WP3 lands, or both stay.
- Which data may be used for training (ingested session text is user data; needs a consent decision).
- Model naming for fine-tunes (`embeddinggemma-2-<suffix>`), and whether `/embed` should list served
  models (`GET /embed/models`).
