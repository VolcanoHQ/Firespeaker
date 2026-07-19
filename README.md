# Firespeaker

*The production engine of [Volcano Studios](https://github.com/VolcanoHQ).*

Manuscript in, audiobook out. Firespeaker turns raw text (.txt/.epub) into finished
audiobooks across three production tiers — single narrator, full character cast, and
Graphic-Audio-style productions with scene-scored music and layered sound design —
using free-tier and open-source models end to end. Deterministic-first analysis with
AI gates, schema-validated and text-grounded AI passes, durable human overrides, a
review console with a per-scene mix timeline, and zero-shot voice cloning via
`XTTS-v2`.

---

## 🔒 Security & Licensing Notes

- **Local tool, no auth by default.** The studio server binds all interfaces with
  authentication OFF for a friction-free local workflow. Never port-forward or
  deploy it bare — enable sessions with `FIRESPEAKER_AUTH=on` (magic-link auth,
  see `src/user_db.py`) before any non-local exposure.
- **Engine code is MIT; generation models are not.** XTTS-v2 ships under the Coqui
  Public Model License (non-commercial) and MusicGen's weights under CC-BY-NC.
  Using this engine's *output* commercially requires reviewing those licenses or
  swapping models — the synthesis/generation layers are deliberately swappable
  behind `src/voice_synthesizer.py` and `src/audio_generation.py`.
- API keys live only in a gitignored `.env` (see setup below); nothing in this
  repository or its history contains credentials.

---

## 🛠️ Environment Setup & Installation

Follow these steps to set up your environment and install all necessary dependencies.

### 1. Initialize and Activate Virtual Environment

It is highly recommended to use a clean **Conda** or **venv** environment running Python 3.10 or 3.11.

#### Using Conda (Recommended):
```bash
# Create a dedicated Conda environment
conda create -n firespeaker python=3.10 -y

# Activate the environment
conda activate firespeaker
```

#### Using venv:
```bash
# Create a standard virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate     # On Windows
```

---

### 2. Install Dependencies

Install all core, synthesis, testing, and cloud integration dependencies using the newly created `requirements.txt` file:

```bash
pip install -r requirements.txt
```

> [!TIP]
> **GPU / CUDA Acceleration (Recommended):**
> If you have an NVIDIA GPU, make sure you install a CUDA-enabled version of PyTorch to enable fast inference:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

---

### 3. Download the NLP Language Model

The Firespeaker pipeline uses spaCy's large English model for typographic normalization and fallback character attributions. Download it after installing requirements:

```bash
python -m spacy download en_core_web_lg
```

---

### 4. (Optional) Configure LLM Enrichment API Keys

Tier 1 ingestion runs fully offline by default. If you want to opt into free-tier LLM-assisted speaker attribution (see [Tier 1 LLM Enrichment](#-tier-1-llm-enrichment-free-tier-cloud-llms) below), create a `.env` file at the repo root:

```bash
# .env (gitignored - never commit this file)
GEMINI_API_KEY=your_google_ai_studio_key   # https://aistudio.google.com/apikey (NOT a Vertex AI key)
GROQ_API_KEY=your_groq_key                  # https://console.groq.com/keys

# Optional kill switch - set to "off" to force pure zero-cost Tier 1 even with keys present
# FIRESPEAKER_LLM_ENRICHMENT=off
```

Without a `.env`, `--enable-llm-enrichment` still works if you have a local [Ollama](https://ollama.com) server running (`ollama serve`), since Ollama is the final fallback in the provider chain. Without any of the three (Gemini key, Groq key, or local Ollama), enrichment silently no-ops and Tier 1's default output is unaffected.

---

## 🚀 Running the Studio Server

Once the dependencies are installed and the spaCy model is downloaded, boot the studio server (threaded — slow synthesis requests don't block other pages):

```bash
python -m src.gui_server
```

Three surfaces, one server:

| URL | Surface |
|---|---|
| **[http://localhost:8082/](http://localhost:8082)** | Legacy studio dashboard (upload, analyze, cast, hierarchy editing) |
| **[http://localhost:8082/console](http://localhost:8082/console)** | **Review Console** — read-only Part→Chapter→Scene navigator with attribution provenance chips (which AI answered, reviewer corrections, confidence), the Tier 3 production lane per scene (music events, layered sound design, dramatized inserts), cached-line audition, and a live pipeline-progress chip |
| **[http://localhost:8082/voicestudio](http://localhost:8082/voicestudio)** | **Voice Cloning Studio** — 7-step wizard: room-tone QC → guided recording session with instant measured verdicts → questionnaire → build → hear-your-clone preview → character personas + vocal SFX bank → consent-gated marketplace publish |

The **voice marketplace** is exposed as REST under `/api/marketplace/*` (listings, semantic search, onboard, purchase-license, cast-to-character) — see [`src/voice_marketplace.py`](src/voice_marketplace.py) and the dataset methodology in [`docs/Firespeaker Voice Dataset Methodology.md`](docs/Firespeaker%20Voice%20Dataset%20Methodology.md) (`python -m src.voice_dataset` for the CLI lane).

---

## 🧩 Production Tiers & Current Status

Firespeaker's tiers describe **output ambition** (what kind of audiobook you get), each built on the previous one. The definitive per-AI accounting — how many AI passes each tier needs, their exact system prompts, provider policies, and validation rules — lives in [`docs/Firespeaker AI Roster & System Prompts.md`](docs/Firespeaker%20AI%20Roster%20&%20System%20Prompts.md).

| Tier | Output | Text-analysis AIs | Status |
|---|---|---|---|
| **Tier 1 — Single Narrator** | One narrator voice reads everything. Deterministic slicing (parts → chapters → scenes → lines) with AI *gates* (G1 Part Verifier, G2 Chapter Verifier, G4 Director's Scene Segmenter) that fire only when the deterministic pass signals failure. Zero cost, fully offline on clean manuscripts. | **0 always-on** + up to 4 gate AIs | ✅ Live end-to-end. Text fidelity vs human Tier 1 golds: **Peter Rabbit 99.5%, A Case of Identity 100%, Alice 99.8%** (≤0.31% contamination). Single-narrator master generation: `python -m src.production_mixer --tier1 --manifest <manifest.json> --output master.wav` (ACX-mastered) |
| **Tier 2 — Narrator + Character Voices** | Dialogue attributed to real characters, each with a distinct voice; emotion + vocalization tags. Opt-in via `--enable-llm-enrichment`. Large director-segmented scenes are attributed in ~30-line windows so every call fits every provider's request cap. | **4** (Attribution, Alias Resolver, Clean-Check, Continuity Reviewer) | ✅ Live; **100%** on Peter Rabbit and **99.2%** (124/125) on A Case of Identity vs. human Tier 2 golds; validated on the 12-story Sherlock corpus, Alice, and an 8-book overnight campaign |
| **Tier 3 — Full Production ("Graphic Audio"-style)** | Everything above + scene-scored music with stingers and stop/resume state events, layered/composite SFX, emotionally-directed creature sounds, generated per-scene ambience, grounded dramatization, delivery direction, ducked multi-track mix. | **+8 crew** (Spotter, Music Director, Sound Designer, Dialogue Director, Dramatist, Book Analyst, Character Designer, QC Critic) | ✅ Full masters produced for Peter Rabbit and A Case of Identity (45.6 min); scene stills + seed-locked character sheets via sd-turbo, identity-locked variants via SD1.5+IP-Adapter |

Free-tier LLM budget: a 7-scene book needs **0** LLM calls for Tier 1, **~12** for Tier 2, **~40** for Tier 3 — comfortably inside Groq's free daily limits, with Gemini reserved for the quality-critical tasks (alias merging, clean-check, whole-book analysis).

> Note: two older modules (`looped_analyzer.py`, `hybrid_nlp_pipeline.py`) predate this
> architecture and remain unwired reference implementations; the `xCoRe` coreference upgrade
> for character discovery remains future work (today's roster is a validated heuristic).

---

## 🤖 Tier 1 LLM Enrichment (Free-Tier Cloud LLMs)

An opt-in enrichment pass over Tier 1's output that upgrades speaker attribution beyond the flat "Narrator" default, using a free-tier-first provider fallback chain: **Gemini Flash (Google AI Studio) → Groq (Llama-3.1-8B-Instant) → local Ollama → off**. Each provider is tried in order per call; a rate limit or missing key on one falls through to the next automatically. See [`src/llm_client.py`](src/llm_client.py) for the client and [Section 4 above](#4-optional-configure-llm-enrichment-api-keys) for key setup.

**Enable it:**
```bash
# Standalone (fastest iteration loop, bypasses TTS/audio):
python -m src.tier_1_parser --input data/corpus/TheTaleofPeterRabbit.txt --output scratch/out.json --enable-llm-enrichment

# Full pipeline:
python src/main.py --input data/corpus/TheTaleofPeterRabbit.txt --output scratch/out.wav --enable-llm-enrichment
```

When enabled, Tier 1 writes two extra artifacts per book to `data/corpus/pipeline/{book}/tier1/` (namespaced separately from `src/looped_analyzer.py`'s own artifacts under `.../looped_analyzer/`, since the two pipelines number their loops differently and previously collided in the same directory):
- `loop4_lines_enriched.json` — the enriched line set (vs. the always-written `loop4_lines.json`, the un-enriched baseline)
- `loopE_llm_cleancheck.json` — advisory Gutenberg-boilerplate/formatting-noise flags (non-destructive; nothing is auto-deleted)

Every LLM call is logged to `data/llm_call_audit.jsonl` (provider, model, latency, success/failure) and daily/per-minute quota usage is tracked in `data/llm_usage_state.json`, both gitignored local state.

**Measuring quality — the eval harness:**
```bash
python eval_tier1_llm_enrichment.py
```
Runs enrichment against `TheTaleofPeterRabbit.txt` and diffs the attributed speakers against the hand-authored gold-standard references in `data/corpus/HumanProcessed/Tier 1/` and `Tier 2/`. Tier 2 gold (which has real per-character speaker splits, unlike Tier 1 gold) is the meaningful accuracy bar. Report written to `scratch/eval_tier1_llm_enrichment_report.json`.

**Latest measured result** (small sample, directional not statistical — 5 gold dialogue lines in one short book): **100% dialogue-speaker accuracy (5/5)** on The Tale of Peter Rabbit, up from an initial 20%. The three fixes that got it there: (1) including the full scene text as prompt context so attribution tags in surrounding narrative ("Peter sneezed—") are visible, (2) a book-level character roster so characters named early (Mrs. Rabbit in scene 1) remain candidates in scenes that reference them only by pronoun/epithet ("his mother"), and (3) deterministic sampling (`temperature=0.1`) on Gemini calls. Notably the perfect score was achieved by the Groq 8B fallback while Gemini was rate-limited — the prompt/roster fixes, not raw model strength, carry the result. The fallback chain is verified live (Gemini 429 → 65s cooldown → Groq). The boilerplate clean-check pass is gated to Gemini-only, since the Groq 8B model false-flags real story text as noise.

**On the harder benchmark** — the 125-dialogue-line *A Case of Identity* vs. its human Tier 2 gold — accuracy is **99.2% (124/125)**. The last two fixes that got it from 84.4%: windowed attribution (director-segmented scenes can exceed a provider's per-request token cap; a 136-line scene silently fell back to narrator defaults until attribution ran in ~30-line windows), and the AI-12 Continuity Reviewer, a second pass over the attributed conversation flow that catches exchange drift and addressee inversions, capped at correcting 30% of a scene (an over-eager reviewer pass is discarded whole).

---

## 📂 Project Architecture

*   `src/`: Primary codebase
    *   **Text analysis (Tiers 1-2)**
        *   `tier_1_parser.py`: Tier 1 deterministic ingestion (parts → chapters → scenes → lines) + opt-in Tier 2 enrichment (attribution AI, alias resolver, clean-check, batch `--input-dir` mode)
        *   `llm_client.py`: Free-tier LLM provider chain (Groq volume → Gemini quality reserve → Ollama offline) with quota tracking, cooldowns, per-task provider gates, and audit logging (`data/llm_call_audit.jsonl`)
        *   `models.py`: Centralized Pydantic schemas (`ScriptLine` incl. `utterance_type`, manifests)
    *   **Production (Tier 3)**
        *   `scene_director.py`: The production crew — Spotting Artist, Music Director, Sound Designer, Dialogue Director (see the AI Roster doc) + generation-prompt builder + MemPalace sync
        *   `audio_generation.py`: Local generation — MusicGen (music beds/stingers) + AudioLDM (SFX layers/ambience), all prompt-cached under `data/generated_audio/`
        *   `production_mixer.py`: Deterministic Chain-D assembly — line-anchored timeline, layered SFX composites, sidechain-ducked music, ACX mastering; `line_overrides.json` for human production edits
    *   **Synthesis & audio**
        *   `voice_synthesizer.py`: XTTS-v2 engine (real CPU/GPU synthesis; pinned per-character speakers via MemPalace; `[pause:X]` pacing markup; edge-tts/mock as fallbacks only)
        *   `audio_mixer.py`: Multi-track mixer, ducking, ACX mastering/verification
        *   `spatial_memory.py`: MemPalace — the production knowledge base (scenes/characters/lines/merges)
    *   **Other**
        *   `main.py`: End-to-end CLI entrypoint (`--enable-llm-enrichment` for Tier 2)
        *   `gui_server.py`: Interactive 4-tab dashboard backend
        *   `api_gateway.py`: FastAPI ingestion/state-management gateway
        *   `hierarchical_parser.py`, `nlp_analyzer.py`: legacy higher-tier parsing/coreference paths
        *   `looped_analyzer.py`, `hybrid_nlp_pipeline.py`: unwired reference implementations (predate the current architecture)
*   `eval_tier1_llm_enrichment.py`: Accuracy harness comparing enriched output against hand-authored gold-standard corpus references
*   `data/corpus/HumanProcessed/`: Hand-authored gold-standard Tier 1/2/3 reference scripts, used for evaluation
*   `data/corpus/pipeline/{book}/`: Per-book artifacts — `tier1/` (loops + enrichment sidecars) and `tier3/` (spotting, production script, sound design, generation prompts, line overrides)
*   `docs/`: Architecture docs — **start with [`Firespeaker AI Roster & System Prompts.md`](docs/Firespeaker%20AI%20Roster%20&%20System%20Prompts.md)** (every AI, counted and specified) and [`Firespeaker Production Knowledge & Media Generation Roadmap.md`](docs/Firespeaker%20Production%20Knowledge%20&%20Media%20Generation%20Roadmap.md) (layer model, storage mapping, execution chains); `Firespeaker Tier 1 Cascading Loops Design.md` for the Tier 1 loop spec
*   `voice_synthesis_testing/`: Audio benchmarking, evaluations, and QA metrics
*   `nlp-testing/`: Older (pre-Tier-system) API integration and experimentation notebooks; superseded by `src/llm_client.py`, kept for reference
