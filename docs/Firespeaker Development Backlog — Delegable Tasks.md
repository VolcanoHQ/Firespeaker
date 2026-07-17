# Firespeaker Development Backlog — Delegable Tasks

*Authored 2026-07-07. Tasks graded by the AI capability needed to execute them safely.
GREEN = well-specified, mechanical, low blast radius — suitable for small/cheap coding
models. YELLOW = mid-tier models (needs some judgment, pattern exists to follow).
RED = keep on frontier models (architecture, prompt design, forensic debugging).*

**Universal rules for any agent working these tasks:**
1. After ANY change: `python -m src.tier_1_parser --stress-test` (run with the
   conda env python: `/home/xbyooki/anaconda3/envs/firespeaker/bin/python`) must
   print `3/3 PASSED`. This is the regression gate.
2. Syntax-check every edited file: `python -c "import ast; ast.parse(open(F).read())"`.
3. Never edit prompts, provider gates, or validation/grounding rules unless the task
   explicitly says so — those encode measured failures.
4. `data/` and `scratch/` are gitignored local state; never commit them.
5. All LLM calls go through `src/llm_client.py` `query_llm_json()` with a distinct
   `task_name` — never call a provider SDK directly.

---

## GREEN — small-model safe

### G-1. Finish the composed-name deterministic pre-merge  *(spec complete, half-landed)*
- **File**: `src/tier_1_parser.py`
- **Context**: The alias merger (`merge_speaker_aliases`) asks an LLM to group speaker
  aliases. Measured: Gemini correctly declines ambiguous pairs like "Miss Mary" +
  "Miss Sutherland" — but the book roster contains the composed full name
  "Miss Mary Sutherland", which settles it deterministically.
- **Spec**: Add `_HONORIFIC_TOKENS = {"mr","mrs","ms","miss","dr","sir","lady","lord","monsieur","madame","mademoiselle","mme","mlle"}`
  and `_composed_name_merges(speakers: Dict[str, dict], roster: List[str]) -> List[dict]`:
  tokenize names lowercased minus honorifics/punctuation; for each speaker pair (a,b)
  with different non-empty token sets, if some roster name with >=2 tokens has token
  set exactly `tokens(a) | tokens(b)`, emit
  `{"canonical": <the pair member with higher line count>, "aliases": [other], "resolved_by": "deterministic:composed-name (<full>)"}`;
  each speaker joins at most one group. Change `merge_speaker_aliases` signature to
  accept `roster: Optional[List[str]] = None`; run the deterministic pass FIRST, seed
  `claimed`/`mapping` with its results, then let LLM groups add to (never override)
  them; deterministic groups still pass the `_conversation_switches` blocker. Caller in
  `ingest_manuscript_tier_1` passes `roster=global_roster`.
- **Validate**: script loading `scratch/case_identity.json` via `ManuscriptManifest`,
  calling the merger with the roster from `_extract_global_roster(clean_front_matter(open("data/corpus/SherlockHolmes/A Case of Identity.txt").read()))`
  → must produce a group merging "Miss Mary" into "Miss Sutherland". Then stress test.

### G-2. Unit-test suite for the pure guard functions
- **New file**: `tests/test_guards.py` (create `tests/`), runnable via the env python.
- **Cover**: `production_mixer._is_foley_only` (cases: "Thud, thud, thud!"→True,
  "Achoo! Kertyschoo!"→False, "Huff, huff, huff!"→False, "Squeak... rustle..."→True,
  "Chirp! Chirp! Don't give up!"→False); `scene_director._looks_like_stage_direction`
  ("Closing door"→True, "Gulps, heavy sigh."→True, "Mmm! Nom, nom, crunch!"→False);
  `tier_1_parser._extract_global_roster` (mid-sentence filter kills "However";
  honorifics catch "Mrs. Rabbit", "Monsieur Madeleine"; title pattern catches
  "King of Bohemia"); `tier_1_parser.identify_chapters` (trailing-period headings
  "CHAPTER I."; bare-roman sequence I/II/III; TOC guard drops <200-char blocks);
  `scene_director.DramatistSchema` string-coercion of `new_characters`.
- **Why safe**: pure functions, no LLM, no audio.

### G-3. Parameterize the eval harness
- **File**: `eval_tier1_llm_enrichment.py` — currently hardcodes Peter Rabbit paths.
- **Spec**: argparse: `--book <path>`, `--gold-tier1 <path>`, `--gold-tier2 <path>`
  (each gold optional), `--alias-map <json path>` (defaults preserving current built-in
  map), `--skip-ingest` (score existing `loop4_lines_enriched.json` instead of
  re-running enrichment — needed so scoring is free). Support BOTH gold formats found
  in `data/corpus/HumanProcessed/`: the block format (`Speaker:` line then quoted
  lines) and the inline format (`[Speaker] text` per line, as in
  `HP_Tier2_SherlockHolmes-A Case of Identity.txt`). Keep defaults exactly reproducing
  today's Peter Rabbit behavior.
- **Validate**: default run unchanged; `--book ".../A Case of Identity.txt" --skip-ingest`
  reproduces ~84% (with the Miss Mary alias in the map).

### G-4. Local audio library resolver (CC0 packs before generation)
- **File**: `src/audio_generation.py`
- **Spec**: `resolve_library_asset(prompt: str, kind: str) -> Optional[str]` scanning
  `data/audio_library/{sfx,music,ambience}/**/*.wav`: score each file by count of
  prompt keywords (lowercased, len>3) appearing in its filename; return best match
  with >=2 hits (>=1 if the filename contains a UCS-style tag matching a prompt word),
  else None. Call it FIRST in `generate_sfx`, `generate_music_bed`, `generate_stinger`
  (still apply the existing post-processing: loudnorm/loop/fades to the library file
  copied to out_path). Create the three dirs with a README.md documenting expected
  licensing (CC0/royalty-free only; keep source URL in a sidecar `.txt` per file).
- **Validate**: drop any wav named e.g. `door_creak_wood.wav` into
  `data/audio_library/sfx/`, call `generate_sfx("wooden door creaking open", 3, out)`,
  confirm the library file is used (no AudioLDM load in log).

### G-5. Batch mode for director + mixer  *(copy an existing pattern)*
- **Files**: `src/scene_director.py`, `src/production_mixer.py`
- **Spec**: add `--manifest-dir <dir>` mirroring `tier_1_parser.py --input-dir`
  (lines ~1030-1080 there show the pattern): glob `*.json` manifests, per-book
  failure isolation, summary printout. No new logic — reuse each module's existing
  single-manifest entry function.
- **Validate**: run director batch over a dir containing the two existing manifests.

### G-6. requirements.txt + docs sync
- **Spec**: add `diffusers>=0.39`, note `coqui-tts[codec]` extra and
  `transformers>=4.47,<5` constraint (coqui incompatibility with 5.x is measured),
  `qdrant-client`; README audio section: add the marketplace CLI
  (`python -m src.voice_marketplace onboard|search|cast`) and the progress viewer
  (`python -m src.progress --watch`); AI Roster doc: append AI-9 (Book Analyst,
  Gemini→Groq only, NEVER Ollama — measured garbage bible) and the music-events
  extension to AI-5's schema. Copy wording from module docstrings; invent nothing.

### G-7. Cosmetic: dramatization `new_characters` listing bug
- **File**: `src/scene_director.py`, `dramatize_scene` return statement.
- **Spec**: the returned `new_characters` uses `if c.name in allowed_new or level == "full"`,
  which re-admits denylisted/undeclared names into the *listing* (inserts are already
  filtered). Change to `if c.name in allowed_new`. Nothing else.

### G-8. Progress tracker polish
- **File**: `src/progress.py`
- **Spec**: add per-stage ETA (`(total-current) * elapsed/current`, shown as `~Nm`),
  add `finish()` calls at the end of each hooked loop in `tier_1_parser.py`,
  `scene_director.py`, `production_mixer.py` (hooks exist; just add the completion
  call after each loop), and a `--json` flag printing the raw state for tooling.

---

## YELLOW — mid-tier models

### Y-1. Play parser (Hamlet)  *(deterministic, but new parsing mode)*
- Trigger from the existing play-format detector in `ingest_manuscript_tier_1`.
  Parse `ACT [IVX]+` → parts, `SCENE ...` → chapters, `^[A-Z][A-Z ]{2,24}\.$` speaker
  prefixes → dialogue lines attributed deterministically (confidence 1.0, method
  "Play Format"); bracketed stage directions → narrative lines. Plays skip the
  attribution AI entirely. Output must remain a valid `ManuscriptManifest`.
- Validate against Hamlet: expect 5 acts, 20 scenes, speakers HAMLET/HORATIO/... with
  sane line counts (Hamlet ~1500 lines total dialogue).

### Y-2. Auto-casting loop (marketplace ↔ production)
- New `auto_cast` command in `src/voice_marketplace.py`: load a manifest + its
  `tier3/book_bible.json`; for each character with >=3 dialogue lines, one LLM call
  (task_name `tier3_casting`, default provider chain) producing
  `{character: casting_description}` (era/age/gender/register, grounded in the bible +
  2 sample lines each); then `cast_character()` each. Respect an
  `--exclude Narrator` default.

### Y-3. Real embeddings for marketplace search
- Replace `generate_clap_embedding` with `sentence-transformers` MiniLM (384-dim)
  behind try/except (keyword fallback stays); new collection
  `voice_marketplace_v2`, migrate existing payloads, reseed. Risk: dependency install
  into the conda env — verify `transformers` compatibility before and stress-test after.

### Y-4. Sentence-level SFX anchoring
- In `production_mixer.assemble_scene`: narration lines >25 words get sentence-split
  (reuse `nltk.sent_tokenize` if available, else regex on `[.!?] `) into separate
  synthesis units sharing the parent line's metadata, so events can anchor between
  sentences. Anchor model stays line-index based externally; the split is internal.
  Careful: cache filenames must include a sentence index.

### Y-5. Les Misérables overnight runner + structure refinement
- Batch script chaining enrichment (chapter ranges via `--chapters`) with progress
  reporting; investigate 92-part over-split (Volume/Book double-nesting — likely needs
  part pattern to treat LIVRE under TOME as sub-parts rather than siblings).

### Y-6. Wire delivery notes into synthesis
- Map dialogue-director notes to concrete controls: pause markup insertion
  (pre-line `[pause:0.6]` for "hesitant"), speed/pitch modifier nudges per note
  keyword table, and `From afar`-type spatial notes → post-synthesis reverb/volume
  ffmpeg filters in the mixer. Table-driven; no LLM.

---

## RED — keep on frontier models

- **TODO (2026-07-17, from user listening review of the Case of Identity Tier 3
  trailer preview): narrated-action foley coverage.** When the narration walks
  through a physical transition — entering the manor's front door, crossing a
  threshold, footsteps changing surface — the mix should carry the implied
  sounds (door opens/closes under and synced with the narration), both as
  one-shot foley events and as background-state changes (interior acoustics
  after the door shuts). Today AI-7's sound design leans on explicit/atmospheric
  sound and the loopE cues require verbatim onomatopoeia, so mundane
  action-implied foley (doors, latches, footsteps, chairs) is under-covered.
  Investigate: (a) whether AI-7's prompt asks for action-implied foley at all;
  (b) a deterministic "action foley spotter" pre-pass over narration verbs
  (opened/closed/knocked/entered/climbed/sat) that proposes anchored candidate
  events for AI-7 to direct — grounding stays enforceable because the verb
  anchor is verbatim text; (c) whether environment state (indoors/outdoors)
  should switch the ambience bed at the same anchor.

- Beat-directed dramatization (per-beat targeted prompting; creative-prompt design
  against measured beat-coverage gaps).
- Any new crew role or change to existing role prompts/validation semantics.
- Forensic debugging of emergent model behavior (the alias-merge/Dramatist style
  investigation loops).
- G2 chapter-verifier gate design; cross-book identity architecture.
- Chain E (images) / Chain F (video) design; GUI architecture (consumes
  `data/analysis_progress.json` + MemPalace — data feeds are ready).
- Gold-eval interpretation and accuracy-target decisions.

---

## Current state notes for whoever picks this up
- Case of Identity: 84.4% attribution vs Tier 2 gold (122 lines); merge no-op root
  causes fixed (bounded wait-retry landed in `llm_client.py`; direct-adjacency
  blocker landed in `tier_1_parser.py`); G-1 completes the fix chain, then rescore
  via G-3's `--skip-ingest` after re-running the merge.
- Peter Rabbit: `scratch/peter_rabbit_tier3_master_v10.wav` is the current reference
  master (foley routing + smoothed ambience + anchor remap).
- Roadmap: `docs/Firespeaker Production Knowledge & Media Generation Roadmap.md`
  still needs a marketplace lane paragraph (part of G-6).
