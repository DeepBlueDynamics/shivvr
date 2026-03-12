# Plan: Physics Arena — Faraday, Assis, Weber, Tesla

## Overview

New arena in `ferricula/arena/` alongside the existing Monte Cristo arena. Four scientists debate electrodynamics, reading from PDFs in `../weber/reference/` and Wikipedia bios.

## Architecture

```
grub-crawl (:6792)     → Wikipedia bios → text
gnosis-ocr (ocr.nuts.services) → PDF → markdown with LaTeX equations
shivvr (shivvr.nuts.services)  → embeddings (768d gtr-t5-base)
ferricula (4 containers)       → memory engines per scientist
Claude Haiku                   → roleplay dialogue
```

## Cast (4 scientists, 4 ferricula containers)

| Name | Port | Focus | Voice |
|------|------|-------|-------|
| **Faraday** | 8773 | experimental observation, lines of force, induction, dielectrics | Self-taught experimentalist. Thinks in physical pictures not math. Distrusts action-at-a-distance. |
| **Weber** | 8774 | force law between charges, velocity-dependent forces, relational mechanics | German precision. His force law unifies Coulomb + Ampère + induction. Everything is relative motion between charges. |
| **Assis** | 8775 | relational mechanics, Mach's principle, Weber applied to gravity, modern interpretation | Brazilian physicist championing Weber's forgotten work. Bridges 19th century to modern physics. Encyclopedic. |
| **Tesla** | 8776 | AC systems, rotating fields, wireless power, practical engineering from theory | Serbian inventor. Sees theory as raw material for machines. Impatient with pure math, obsessed with resonance. |

## Phase 0: Wikipedia Training

Before reading any PDFs, each scientist gets their Wikipedia bio + key related articles ingested via grub-crawl.

1. Crawl Wikipedia pages via `POST http://localhost:6792/api/markdown`:
   - `https://en.wikipedia.org/wiki/Michael_Faraday`
   - `https://en.wikipedia.org/wiki/Wilhelm_Eduard_Weber`
   - `https://en.wikipedia.org/wiki/André_Koch_Torres_Assis`
   - `https://en.wikipedia.org/wiki/Nikola_Tesla`
   - Shared: `https://en.wikipedia.org/wiki/Weber_electrodynamics`, `https://en.wikipedia.org/wiki/Ampère%27s_force_law`
2. Each scientist reads ALL bios (channel=hearing, alpha=0.01)
3. Dream 3 cycles after wiki ingestion

## Phase 1: PDF Processing

Upload PDFs to gnosis-ocr, get back markdown with equations.

**Core reading list** (all 4 read):
- `The-Electric-Force-of-a-Current.pdf` (Assis & Hernandes)
- `Preface-Webers-Electrodynamics.pdf`

**Per-scientist deep reads**:
- Faraday: `mf_ere_vol_1.pdf`, `mf_ere_vol_2.pdf`, `mf_ere_vol_3.pdf`
- Weber: `Weber-in-English-Vol-1.pdf`, `Weber-in-English-Vol-2.pdf`
- Assis: `Relational-Mechanics-1999.pdf`, `Relational-Mechanics-Mach-Weber-2014.pdf`
- Tesla: `tesla_patents/*.pdf`

**OCR Flow** (per PDF):
1. `POST ocr.nuts.services/storage/upload` with PDF file → `session_id`
2. Poll `GET ocr.nuts.services/api/jobs/{session_id}/status` until done
3. Fetch markdown from storage endpoint
4. Split markdown into sections (by `##` headings or page breaks)
5. Send sections to shivvr for chunking + embedding
6. Ingest chunks into the scientist's ferricula

**Skim mode**: Scientists don't read entire books. Process table of contents first, then use fuzzy vector recall to decide which sections to deep-read. A scientist "skims" by:
1. Ingest TOC/intro
2. For each discussion topic, recall from TOC
3. If a section seems relevant, OCR + ingest that section only

## Phase 2: Section-by-Section Reading + Discussion

Unlike Monte Cristo's chapter-by-chapter flow, physics texts are read section-by-section:

```
for each section in core_reading_list:
    1. OCR the section (if not already processed)
    2. Chunk + embed via shivvr
    3. Ingest into ALL 4 scientists
    4. Summarize section (LLM)
    5. Scene: each scientist reacts in character
       - Recall from their own memories
       - Respond based on their focus/voice
       - Others hear the response
    6. Dream 3 cycles
    7. Save log
```

## Phase 3: Discussion Topics

After reading, structured debates on:
- Weber's force law vs Coulomb + Lorentz
- Action at a distance vs field theory
- The role of the medium (Faraday's dielectric vs vacuum)
- Velocity-dependent forces and their experimental evidence
- Relational mechanics and Mach's principle
- Practical applications: can Weber's law build better machines?
- The missing magnetic force in circuit analysis
- Gravitational analogues of electromagnetic phenomena

## New Files

### `ferricula/arena/physics_arena.py`
Main orchestrator. Similar structure to `arena.py` but with:
- `OcrClient` class for gnosis-ocr
- `CrawlerClient` class for grub-crawl
- `CAST` dict with 4 scientists on ports 8773-8776
- Wikipedia training phase
- PDF processing + skim logic
- Section-based reading loop
- Discussion round loop

### `ferricula/arena/physics_chunker.py`
- Split OCR markdown by sections (`##` headings, page breaks)
- Keystone patterns for physics (force law, induction, relational, etc.)
- Character detection (which scientist is mentioned/quoted)
- Emotion: experimental vs theoretical vs critical

### `ferricula/arena/ocr_client.py`
- Upload PDF to gnosis-ocr
- Poll for completion
- Fetch markdown result
- Cache results locally (don't re-OCR same PDF)

### `ferricula/arena/crawler_client.py`
- Crawl URL via grub-crawl
- Extract markdown
- Cache results

### `ferricula/arena/docker-compose.physics.yml`
- 4 ferricula containers (ports 8773-8776)
- Named data volumes per scientist

## Dependencies

- shivvr running at `shivvr.nuts.services` (Cloud Run, already deployed)
- gnosis-ocr running at `ocr.nuts.services`
- grub-crawl running locally at `localhost:6792` (already in Docker)
- ferricula binary built
- `AGENT_KEY` env var for Claude Haiku

## Steps

1. Create `ocr_client.py` and `crawler_client.py`
2. Create `physics_chunker.py` (section splitting, physics keystones)
3. Create `docker-compose.physics.yml` (4 ferricula containers)
4. Create `physics_arena.py` (main orchestrator)
5. Test: start containers, run wiki training, process one PDF, run one discussion
