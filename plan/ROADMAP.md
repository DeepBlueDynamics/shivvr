# shivvr roadmap

Single source of truth for direction, status and sequencing. Technical detail (model facts,
benchmarks, export notes, training options, Pi measurements) lives in
[EMBEDDINGGEMMA2_PLAN.md](EMBEDDINGGEMMA2_PLAN.md); this file links to it rather than repeating it.

Owner: Kord. Updated 2026-10-14. Released: v0.5.2. In review: v0.6.0 (PR #13).

## Where we're going

1. **One text model by default, GTR kept for inversion.** EmbeddingGemma 2 is the default for
   everything. GTR-T5-base keeps running alongside so every chunk stays invertible, and can be chosen
   per store.
2. **One multimodal model, eventually.** EmbeddingGemma 2 puts text, images, audio and video in one
   space. Once its vision and audio paths are served, it replaces SigLIP (images) and
   transcribe-then-embed (audio). **SigLIP stays until then**: it is the only image embedder shivvr
   has today.
3. **Four images, one release command.** A local GPU image, a local CPU/Pi image that runs with one
   `docker run`, a cloud frontend (gateway) and a cloud backend. One tag builds, publishes and deploys
   all of them.
4. **The GPU must pay for itself.** The cloud backend runs on an L4 only if a benchmark shows it beats
   CPU enough to justify the cost. Today it doesn't, because the export doesn't use the GPU well.
5. **Search on current Lume.** Consume lume directly (0.16 now), not a vendored snapshot.

## Milestones

Status: ✅ done · 🟡 in progress / in review · ⬜ not started · ⛔ blocked

### M1. v0.6.0: Gemma default, GTR alongside, lume 0.16, docs. 🟡 PR #13

| Item | Status |
|---|---|
| Repo root cleanup, README in lume's format with a Docker Hub quick start, `docs/` | ✅ |
| EmbeddingGemma 2 default for `/embed`, ingest, search, `/audio/embed`, MCP, agent; per-store model; prefixes server-side | ✅ |
| GTR vector kept per chunk; `return_vectors: ["gtr-t5-base"]` → `embeddings` map (shape agreed with lume) | ✅ |
| lume v0.16.0 replaces `vendor/lume-hybrid` (BM25 now stems; rankings shift) | ✅ |
| Unit + integration tests; CPU image builds on Rust 1.88; CPU smoke test | ✅ |
| `tests/functional.py` against a live server; CUDA image build | ⬜ runs during release |

**Gates before merge/release:**
- ⛔ **Lume's path-2 change**: its scratch-ingest → `/invert` call must move to `POST /embed`
  `{"model":"gtr-t5-base"}` (~15 lines in lume `src/hybrid.rs`). Needs Kord's OK; it works on 0.5.x
  too, so it can ship first.
- Kord merges PR #13 (merge or rebase, not squash, so `release: v0.6.0` stays its own commit).

**Release steps:**
1. Tag `v0.6.0`. CI publishes the gateway and the amd64/arm64 CPU images to Docker Hub.
2. Run `deploy.sh` (CUDA backend to Cloud Run), then copy the image to `deepbluedynamics/shivvr`.
3. Deploy the gateway to `shivvr-landing`.
4. Run `tests/functional.py` against production.

### M2. One release command for four images. ⬜ after M1

Today two images publish automatically on a tag (gateway, CPU), and two are manual (`deploy.sh` for
the CUDA backend, plus a hand copy to Docker Hub).

| Item | Notes |
|---|---|
| Self-contained CPU image | `latest-cpu` gets the int8 models built in (~440 MB), so the README quick start is one `docker run`. `shivvr-models:int8` stays for HaLOS and custom model sets. |
| CUDA image published on tag | Either a GitHub Actions job (needs a large runner and the models image reachable from CI) or Cloud Build triggered by the tag. Decide (D2). |
| Copy to Docker Hub inside the pipeline | Removes the hand `imagetools create` step. |
| Gateway to GCR on tag | Or deploy `shivvr-landing` from the Docker Hub image directly. |
| Deploy the cloud pair on tag | Optional; keep manual until M3 settles the backend shape. |

### M3. Cloud cost: does the backend keep the L4? ⬜ after M1

| Step | Notes |
|---|---|
| Fix the GPU export | WP1 showed the RTX 3060 no faster than CPU up to 2,048 tokens. Suspects: dense masked attention (sliding window wasted), ops falling back to CPU, host/device copies. Try ORT transformer fusion, dynamo + SDPA export, IO binding. |
| Benchmark L4 vs CPU per route | `/embed` (Gemma, GTR), ingest, search, **`/invert`** (the autoregressive T5 decoder is where a GPU usually helps most), cold start. |
| Decide | GPU stays, or the cloud backend becomes a CPU image (models built in, no CUDA). If inversion alone needs the GPU, consider a separate GPU inversion service that wakes on demand. |

### M4. EmbeddingGemma 2 images and audio, then retire SigLIP. ⬜ independent of M2/M3

| Step | Notes |
|---|---|
| WP3: vision export + Rust `Gemma4ImageProcessor` port | Port the preprocessor against golden tensors from Python. `/image/embed` gets `model`; SigLIP stays the default until the switch. |
| Image chunks in session space | Today an `image_base64` ingest stores a **SigLIP** vector in a Gemma/GTR session, so its search score is meaningless (known bug). With Gemma vision, image chunks land in the session's own space. |
| WP4: native audio | Export the audio encoder; log-mel features in Rust or in the graph. `/audio/embed` gets native audio embedding; the transcription path stays available. |
| Retire SigLIP | One release with Gemma as the image default and SigLIP still selectable (deprecation note in `/health` and docs), then remove SigLIP from the images. Shrinks the CUDA image and the model set. |

### M5. Inverter for EmbeddingGemma 2's space (research). ⬜ after M1

| Step | Gate |
|---|---|
| Fix `training/` (384d hard-coding; corrector re-embeds with the target model) | — |
| Text pilot: ~100k (text, Gemma `document` vector) pairs, hypothesis model only, local RTX 3060 | Measure re-embed cosine and BLEU. Go/no-go. |
| Zero-shot modality check: invert image and audio vectors with the pilot | Sizes the modality gap. Needs Python-side image/audio vectors (M4 not required). |
| Full run: text + caption + transcript pairs, then the corrector, spot A100 (~3–5 days, ~$100–200 estimate) | **Kord's OK for the spend.** |
| Serve as a second inverter; `/invert` picks by the vector's model | — |

Data: MS MARCO, public marine text (lume `docs/cruiser_library.csv`). Never boat logbooks or telemetry.
Lume has no current use for it; don't size it around lume.

### M6. Raspberry Pi on real hardware. 🟡 built, not measured

CPU image, int8 models and the HaLOS app exist (WP7). Left: latency/RSS/temperature on a Pi 5 (with and
without an Active Cooler), the token cap for the Pi, `icon.png`, submit `deploy/halos/shivvr` to
halos-marine-containers.

### M7. Retrieval evaluation, then fine-tuning. ⬜

An in-domain eval set (EMBEDDINGGEMMA2_PLAN §4.2) before any training. Quick check now: `lume eval`
on lume's `docs/monte_cristo` (Hit@k/MRR/nDCG) comparing Gemma vs GTR through shivvr. Fine-tuning
(§4.3 A) only if the eval shows a gap worth closing.

## Sequencing

```
M1 v0.6.0 ──┬─> M2 release pipeline ──> (deploy cloud pair on tag)
            ├─> M3 GPU export fix + L4 benchmark ──> backend: GPU or CPU
            ├─> M5 inverter pilot ──> (Kord OK) A100 run ──> serve
            └─> M7 eval
M4 Gemma vision/audio ──> SigLIP deprecated ──> SigLIP removed      (can start any time)
M6 Pi measurements                                                    (needs hardware time)
```

Suggested worker split after M1: one worker on M2, one on M3 (needs an L4 or Cloud Run access), one on
M4 (WP3 first), and the M5 pilot on the local 3060.

## Known issues (not scheduled yet)

- `image_base64` session ingest mixes SigLIP vectors into text sessions (fixed by M4).
- `GET /sessions` handler exists but isn't routed; the landing page links to it (404).
- MCP and agent-chat routes sit outside the auth gate and skip session ownership checks.
- Rust toolchains differ: CUDA/CPU Dockerfiles pin 1.88, the gateway uses 1.98.
- The chunker's boundary threshold (0.5) is tuned for GTR. Segmentation still uses GTR, so it's fine
  today; revisit if chunking ever moves to Gemma.
- `docker-compose.yml` publishes `127.0.0.1:8085`; anything on the LAN that relied on `:8085` (e.g. a
  Pi calling a desktop shivvr) must change.

## Open decisions

| # | Decision | Default if nobody decides |
|---|---|---|
| D1 | Lume path-2 change approved? (gates M1) | — |
| D2 | CUDA image build on tag: GitHub Actions large runner vs Cloud Build trigger | Cloud Build trigger (models image already in GCR) |
| D3 | Image tag names for the four images (`latest`/`latest-cpu` vs `gpu`/`cpu`) | Keep `latest` (CUDA) and `latest-cpu` |
| D4 | Training data consent: may ingested session text ever be used for eval/training? | No |
| D5 | M5 A100 spend after the pilot | Ask with pilot numbers |
| D6 | Default `task` on `/embed` when none is given | No prefix |
