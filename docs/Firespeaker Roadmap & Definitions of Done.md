# Firespeaker Roadmap & Definitions of Done

*Authored 2026-07-16. The review ledger: every roadmap item, its Definition of Done
(DoD), its status, and the evidence. An item is marked **✅ DONE** only when every DoD
clause is satisfied with measured evidence and the work is pushed to `origin/main`.
Anything less is ⬜ OPEN or 🔶 PARTIAL, and says so.*

**Universal DoD clauses (apply to every item, never repeated below):**
1. Zero-LLM stress test passes 3/3 after the change.
2. New/changed modules import cleanly under the `firespeaker` conda env.
3. Test residue (fake data, temp listings, test drawers) removed.
4. Committed with a descriptive message and pushed to `origin/main`.

---

## Foundation — the platform (all DONE)

### F-1 · Deterministic Tier 1 structure + gate AIs — ✅ DONE (`cd0ff13`)
**DoD:** Loops 1-4 (parts → chapters → scenes → lines) run with zero LLM calls on clean
manuscripts; gate AIs (G1 Part Verifier, G2 Chapter Verifier, G4 Director's Scene
Segmenter) engage ONLY on deterministic-failure signals; deterministic output always
stands when a gate fails.
**Evidence:** Stress test 3/3 standing gate; Les Misérables 50 parts / 498 chapters /
723 scenes deterministic; G1 fired 5× in the 8-book campaign (4 monolithic books +
Franklin's 26-part runaway); G4 measured re-segmenting Case of Identity 10 → 4 scenes.

### F-2 · Free-tier LLM provider chain — ✅ DONE (`cd0ff13`)
**DoD:** Gemini → Groq → Ollama fallback per call; 429/quota exhaustion cascades
automatically; per-task provider gates; single-provider tasks wait out cooldowns
(bounded retry); every call audit-logged; daily/minute quota tracked.
**Evidence:** Live 429 → 65s cooldown → Groq cascades throughout campaign logs;
`data/llm_call_audit.jsonl` and `llm_usage_state.json` populated; Gemini-only gates
(clean-check, alias merge, book bible) enforced at call sites.

### F-3 · Tier 2 speaker attribution (AI-1/2/3/12) — ✅ DONE (`cd0ff13`)
**DoD:** ≥95% dialogue-speaker accuracy vs both human Tier 2 golds; alias merge never
invents identities; every output schema-validated and grounded.
**Evidence:** Peter Rabbit **5/5 (100%)**; A Case of Identity **124/125 (99.2%)** —
via windowed attribution (~30 lines/call) + AI-12 Continuity Reviewer (30% correction
cap, over-eager passes discarded: 31 applied / 78 rejected across the campaign).

### F-4 · Manuscript scrubbing (ClutterScrubber) — ✅ DONE (`cd0ff13`)
**DoD:** Zero Project Gutenberg license/credit text in any corpus narration stream;
every book's true opening present; fuzzy detectors can never discard story prose.
**Evidence:** 8-book scan: 0 PG mentions, 0 credits; Wuthering Heights' "1801—I have
just returned…" and Frankenstein's Letters 1-2 restored; heading guard + bounded LLM
boundary + works-list exclusions all measured on the books that broke them.

### F-5 · Tier 1 text fidelity vs golds — ✅ DONE (`cd0ff13`)
**DoD:** ≥99% gold-text coverage and ≤1% contamination vs every HumanProcessed Tier 1
gold (prose; Hamlet excluded pending play parser).
**Evidence:** Peter Rabbit 99.5% / 0.31%; A Case of Identity 100.0% / 0.21%;
Alice 99.8% / 0.25%.

### F-6 · Tier 3 production crew — ✅ DONE (`8381fa2`)
**DoD:** Per-scene direction (music with state events, environment, delivery, layered
sound design, grounded dramatization) written as validated + text-grounded artifacts;
advisory QC pass; whole-book bible (never from local 3B).
**Evidence:** Full artifact sets for Peter Rabbit and Case of Identity; dramatization
copy-guard/denylist/declared-cast-only all regression-hardened; QC critic documented
advisory-only.

### F-7 · Media generation chains (C/D/E) — ✅ DONE (`8381fa2`)
**DoD:** Music beds/stingers (MusicGen, loudness-normed), layered SFX + ambience
(AudioLDM), full timeline assembly with ducking + ACX mastering; scene stills +
seed-locked character sheets; identity-locked stills.
**Evidence:** Tier 3 masters: Peter Rabbit (v10 reference) and Case of Identity
(45.6 min, 20 music state events); 68/68 AudioLDM generations; 8/8 identity-locked
stills with sheet conditioning visible in logs.

### F-8 · Tier 1 audiobook lane — ✅ DONE (`8381fa2`)
**DoD:** `production_mixer --tier1` produces a single-narrator, ACX-compliant master
from any manifest with zero LLM calls.
**Evidence:** Peter Rabbit master: 5.8 min, RMS −21.5 dBFS (ACX −23…−18 range).

### F-9 · Review Console (Phase 1) — ✅ DONE (`bfceb18`)
**DoD:** Book inventory, Part→Chapter→Scene navigator with provenance/coverage badges,
transcript with provider chips + confidence + line audition, Tier 3 production lane
view, live progress chip; all read-only; audio serving path-traversal-guarded.
**Evidence:** All endpoints curl-tested including traversal probe (404) and a 15MB
master streamed; Secret Garden's 30-chapter/101-scene tree loads without lines in the
tree payload.

### F-10 · Voice marketplace REST — ✅ DONE (`bfceb18`)
**DoD:** Browse, semantic search, consent-gated onboard, license-ledger purchase,
description-based cast-to-character (MemPalace drawer bind) all over HTTP.
**Evidence:** Live tests: listings returned; "warm male narrator" → scored hits;
"gravelly old wizard" cast correctly bound the Gruff English Gardener (test binding
removed).

### F-11 · Voice Cloning Studio wizard — ✅ DONE (`bfceb18`)
**DoD:** 7-step browser wizard (identity → room-tone QC → first take → questionnaire →
guided session with instant measured verdicts → build + hear-your-clone preview →
personas & vocal SFX → consent-gated publish); every endpoint exercised.
**Evidence:** Synthetic end-to-end test: roomtone −62 dBFS, 6 prompts PASS at SNR ~40,
questionnaire drafted listing copy, build produced 18s reference + emotion ref, XTTS
preview rendered a valid 2.2s clone WAV, publish created the listing with personas +
SFX in the description (all residue then deleted).

### F-12 · Voice dataset methodology + CLI — ✅ DONE (`bfceb18`, doc `816e72a`)
**DoD:** One ~15-min session serves zero-shot reference, per-emotion bank, and
fine-tune corpus; mechanical intake QC (PASS/FLAG/REJECT with measured numbers);
LJSpeech-style manifest; SHA-256 consent provenance; marketplace hand-off.
**Evidence:** QC correctly rejected clipped/short/hot and low-SNR clips and passed
speech-shaped audio; reference assembly prefers narration-register clips;
`data/voice_datasets/tim/recording_script.md` generated for the first real donor.

### F-13 · Documentation set — ✅ DONE (`816e72a`)
**DoD:** AI Roster (per-AI prompts/validation/accuracy), Roadmap (layers, chains,
marketplace lane, cross-book identity), Voice Dataset Methodology, backlog, README
reflecting measured reality.
**Evidence:** Eight docs in `docs/`; README tier table carries the measured numbers.

---

## Hardening — the red items (all DONE)

### R-1 · Voice-fingerprint cache keys — ✅ DONE (`7f874d6`)
**DoD:** Recasting a character invalidates exactly that character's cached line WAVs;
unchanged voices stay cache-hits; legacy caches adopted without wholesale resynthesis.
**Evidence:** 4-assertion test: fingerprinted naming ✓, cache hit ✓, recast → new
fingerprint + fresh synthesis ✓, legacy rename-adoption ✓.

### R-2 · Resume-incomplete enrichment — ✅ DONE (`1c84805`)
**DoD:** `--resume-enrichment` spends LLM calls only on scenes still on Tier 1
defaults; reuse is text-keyed (structure drift falls through safely); previous G4
boundaries reused when still valid; clean-check/SFX carried forward.
**Evidence:** Case of Identity: full artifacts → 5/5 reused, 0 enrichment calls, ~4s
(was ~15 min); one starved scene → exactly 1 fresh, dialogue restored to Holmes/Watson.

### R-3 · Durable human speaker corrections — ✅ DONE (`fef664f`)
**DoD:** Click-to-correct in the console; corrections keyed by content-hash line_id
survive every re-run; clearing a correction restores attribution truth; overrides
never bake into the enriched artifact (it stays the attribution layer's pure output).
**Evidence:** Full cycle tested: set → served as human_override → survived pipeline
re-run into manifest while artifact stayed pure → cleared → truth restored. The
baked-in variant was caught by test and rejected as a layering violation.

### R-4 · Tier preview (trailer scene) — ✅ DONE (`19291a0`)
**DoD:** Deterministic zero-LLM scorer picks the most tier-audible scene; one endpoint
renders it at Tier 1/2/3; results cached; unenriched books get guidance, never a
silent downgrade.
**Evidence:** Scorer picked Peter Rabbit's McGregor chase and Case's interview; all
three tiers rendered distinct audio (tier 3 music bed measurable in voice gaps);
replay cached: true; Les Mis tier-2 request → 409 with guidance.

---

## Tier 1 — MVP closeout (IN PROGRESS)

*The MVP sentence: EPUB in → review/correct in the console → tier preview → one
button → chaptered audiobook out.*

### M-1 · One-button render job — ✅ DONE
**DoD:**
- [x] `POST /api/console/render {book, tier}` starts a detached worker
      (start_new_session; survives server restarts; dead workers reaped to
      `failed` on next listing); job record carries `owner` (default `"local"`),
      status, timestamps, outputs, error — one file per job in `data/render_jobs/`.
- [x] Job runs ingest (resume-enrichment for tier ≥ 2) → tier's mixer →
      chaptered export (M-2), reporting through the existing progress tracker.
- [x] Tier 3 without direction artifacts fails fast with guidance; unknown book
      and duplicate-render also refused with readable errors (409).
- [x] Console Generate card: tier select → job status polling → M4B/WAV download
      links on completion; `GET /api/console/renders?book=` lists job state.
- [x] Full Tier 1 render of Peter Rabbit completed through the API path in 542s:
      wav + m4b + line-timing manifest all produced and non-empty.
**Evidence:** `src/render_job.py`; job record verified with owner field; all
three guard paths curl-tested.

### M-2 · Chaptered export + line-timing manifest — ✅ DONE
**DoD:**
- [x] Voice-track mixers (Tier 1/2) write `{output}.line_timings.json`:
      per line `{line_id, character, chapter_id, start_s, end_s}` — offsets MEASURED
      from each line wav's real duration during assembly.
- [x] M4B export (`export_m4b`, AAC 96k, ipod container) with real chapter markers
      from the same offsets; chapter titles from the manifest.
- [x] Round-trip verified: 3-chapter test manifest → M4B chapter times matched the
      timing manifest to the ms (166.688s / 286.677s); manifest duration matched the
      wav ±0.1s; offsets monotonic; a Mr. McGregor dialogue line's start offset
      contained speech energy (−21 dB, not silence).
- [x] Tier 2 voice-track assembly (`mix_tier2` / `--tier2`): attributed cast voices,
      auto-registered drawers for unseen speakers; 4-speaker timing manifest produced.
**Evidence:** shared `mix_voice_track` engine; console audio serving extended to
`.m4b`/`.mp3` for downloads.

### M-3 · EPUB ingestion — ⬜ OPEN
**DoD:**
- [ ] `--input book.epub` works end-to-end: spine order → per-item text extraction →
      chapters derived from spine (headings preferred, positional fallback) →
      existing loops 3-4 unchanged downstream.
- [ ] Non-content spine items (cover, nav/toc, copyright) excluded; scrubber still
      runs (Gutenberg EPUBs carry the license too).
- [ ] Verified against a known book: EPUB-ingested chapter count and narration
      stream match the .txt-ingested equivalents (±0 chapters, fidelity scorer ≥99%
      agreement).

---

## Tier 2 — post-MVP (OPEN, in build order)

### T2-1 · User management foundation — ⬜ OPEN
**DoD:** Lightweight auth (magic-link or OAuth); session on every studio surface; the
server refuses unauthenticated API access when auth is enabled; existing `owner`
fields map to real user ids with a migration for `"local"`.
*Note: launch gate for any non-local marketplace exposure — jumps the queue if the
marketplace goes public first. Until then the server must remain local-only.*

### T2-2 · User-owned projects — ⬜ OPEN
**DoD:** `project_db` project records own books/renders/exports; console scopes to the
signed-in user's projects; tier + plan chosen at project creation.

### T2-3 · Usage metering per user/project — ⬜ OPEN
**DoD:** Every LLM call and generation job carries owner + project in the audit log;
a per-project usage summary endpoint answers "what did this book cost?"

### T2-4 · Plan → provider entitlement (paid fast lane) — ⬜ OPEN
**DoD:** A project's plan selects its provider chain (free = current chain + resume
slow lane; paid = premium keys, no quota waits); same validation contract regardless
of provider.

### T2-5 · Marketplace identities + payments — ⬜ OPEN
**DoD:** Sellers/buyers are authenticated users; consent + license ledger reference
user ids; payment processing on purchase; payout bookkeeping for sellers.

### T2-6 · GPU synthesis path / render queue — ⬜ OPEN
**DoD:** XTTS on GPU with per-chapter incremental rendering; a novel renders in
hours, not overnight; queue survives restarts.

### Deferred indefinitely (revisit on demand)
Hamlet play parser (Y-1) · marketplace storefront pages · Chain F video ·
cross-book series identity table · full README rewrite.
