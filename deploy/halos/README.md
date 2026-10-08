# shivvr on HaLOS (Raspberry Pi)

`shivvr/` is a HaLOS container app definition in the format of
[halos-marine-containers](https://github.com/halos-org/halos-marine-containers/blob/main/docs/DESIGN.md)
(`metadata.json`, `docker-compose.yml`, `config.yml`). Copy the directory into that repo's `apps/`
to package it as `shivvr-container`. An `icon.png` (64×64 or larger) still has to be added there.

## What it runs

| Container | Image | Job |
|---|---|---|
| `shivvr-models` | `deepbluedynamics/shivvr-models:int8` | One-shot: copies the int8 model set into the `shivvr-models` volume when its content stamp changed, then exits |
| `shivvr` | `deepbluedynamics/shivvr:latest-cpu` | The service, CPU only, linux/arm64 + linux/amd64 |

Model set (`int8`, 440 MB): GTR-T5-base (required; the session/ingest space) and the EmbeddingGemma 2
text path. SigLIP and the vec2text inverter are left out to fit the ~3.4 GB a HaLOS Pi has free next
to Signal K, InfluxDB, Grafana and QuestDB.

| Model | int8 size | Worst / mean cosine vs fp32 |
|---|---|---|
| GTR-T5-base | 110 MB | 0.993 / 0.997 |
| EmbeddingGemma 2 | 295 MB | 0.988 / 0.993 (last layer and output projection kept fp32) |

## Defaults

- Listens on `127.0.0.1:8285`. Signal K plugins on the same Pi (lume) use
  `SHIVVR_BASE_URL=http://127.0.0.1:8285`. `BIND_HOST=0.0.0.0` exposes it to the boat network.
- `EMBEDDINGGEMMA2_MAX_TOKENS=512`: longer inputs are truncated (cost grows faster than linearly).
- `MEMORY_LIMIT=1500m`. Measured on amd64 with both models loaded: ~820 MiB.
- No authentication unless `NUTS_AUTH_JWKS_URL` is set; acceptable while bound to loopback.

## Not yet verified on a Pi

Tested on amd64 (Docker Desktop): first boot installs models, later boots skip the copy, `/health`
lists both models, vectors match the fp32 service at the cosines above. Latency, peak memory and
temperature on a Raspberry Pi 5 are still to be measured (plan/EMBEDDINGGEMMA2_PLAN.md, WP7).

## Rebuilding the model image

```bash
python scripts/quantize_models.py --input_dir models --output_dir models-int8
docker buildx build --platform linux/amd64,linux/arm64 -f Dockerfile.models-int8 \
  -t deepbluedynamics/shivvr-models:int8 --push models-int8
```

`quantize_models.py` fails if either int8 model drops below cosine 0.98 against fp32 on its sample texts.
