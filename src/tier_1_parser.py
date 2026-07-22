#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Caldera Engine Tier 1 Ingestion Pipeline
Deterministic, zero-cost text-slicing engine using Pydantic validation.
"""

import io
import os
import re
import sys
import json
import hashlib
import logging
import time
import zipfile
from typing import List, Dict, Any, Literal, Optional, Tuple
from pydantic import BaseModel, Field
from xml.etree import ElementTree as ET

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Tier1Parser")

# Import centralized schemas
from src.models import (
    PerformanceMetrics,
    ScriptLine,
    ScenePayload,
    ChapterPayload,
    PartPayload,
    ManuscriptManifest
)
from src.book_structure import materialize_book_structure_from_tier1

_DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _docx_to_text(raw_bytes: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as docx:
            xml_content = docx.read("word/document.xml")
    except KeyError as exc:
        raise ValueError("DOCX is missing word/document.xml") from exc
    except zipfile.BadZipFile as exc:
        raise ValueError("DOCX is not a valid ZIP package") from exc

    root = ET.fromstring(xml_content)
    paragraphs = []
    for para in root.findall(".//w:p", _DOCX_NS):
        text_runs = para.findall(".//w:t", _DOCX_NS)
        para_text = "".join(node.text for node in text_runs if node.text)
        if para_text:
            paragraphs.append(para_text)
    return "\n\n".join(paragraphs).strip()


# ====================================================
# Opt-in LLM Enrichment Schemas (Tier 1 stays deterministic/zero-cost by
# default; these are only touched when enable_llm_enrichment=True)
# ====================================================

class TierOneAttributionLine(BaseModel):
    index: int
    speaker: str
    emotion: str = Field(default="Neutral")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    utterance_type: Literal["speech", "vocalization"] = Field(default="speech")


class TierOneSfxCue(BaseModel):
    sound_text: str = Field(..., description="The onomatopoeic text as it appears in the scene narration")
    description: str = Field(default="", description="What environmental sound this represents")


class TierOneAttributionSchema(BaseModel):
    lines: List[TierOneAttributionLine]
    sfx_cues: List[TierOneSfxCue] = Field(default_factory=list)


class AttributionCorrection(BaseModel):
    index: int
    current_speaker: str
    corrected_speaker: str
    reason: str = Field(default="")


class AttributionReviewSchema(BaseModel):
    corrections: List[AttributionCorrection] = Field(default_factory=list)


class SceneBoundaryProposal(BaseModel):
    start_snippet: str = Field(..., description="The EXACT first 8-15 words of the paragraph that begins a NEW scene, copied verbatim")
    reason: str = Field(..., description="One short phrase naming the continuity break, e.g. 'location shift: garden -> tool-shed'")


class DirectorSceneSchema(BaseModel):
    boundaries: List[SceneBoundaryProposal] = Field(default_factory=list)


class AliasMergeGroup(BaseModel):
    canonical: str = Field(..., description="The most complete/specific name for this person, chosen from the listed speakers")
    aliases: List[str] = Field(..., description="Other listed speaker names referring to the same person")


class AliasMergeSchema(BaseModel):
    groups: List[AliasMergeGroup]


_ROSTER_STOPWORDS = {
    "the", "and", "but", "now", "once", "it", "he", "she", "they", "i", "you",
    "then", "there", "here", "this", "that", "when", "after", "before", "so",
    "if", "yes", "no", "oh", "well", "still", "just", "chapter", "scene",
    "narrator", "mr", "mrs", "dr", "ms",
    # Possessive/object pronouns -- capitalized only by sentence-initial position,
    # not because they're names (e.g. "His mother was..." at a sentence start).
    "his", "her", "hers", "him", "them", "their", "theirs", "my", "mine",
    "your", "yours", "our", "ours", "its",
}


def _extract_global_roster(full_text: str) -> List[str]:
    """Book-level candidate roster: names that recur across the whole manuscript.

    Fixes the per-scene blindness where a character named early (e.g. "Mrs. Rabbit"
    in scene 1) is only referenced by pronoun or epithet ("his mother") in a later
    scene -- without a book-level roster, the LLM has no valid name to attribute
    that scene's dialogue to. Honorific names (Mr./Mrs. X) are matched explicitly
    since the period breaks the generic capitalized-token pattern.
    """
    counts: Dict[str, int] = {}
    midsentence_counts: Dict[str, int] = {}
    honorific_pattern = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Miss|Monsieur|Madame|Mademoiselle|Mme|Mlle)\.?\s+[A-Z][A-Za-z'\-]+")
    # Titles with lowercase connectors ("King of Bohemia", "Duke von Kramm") -- the
    # generic pattern can't cross the lowercase word, leaving only the fragment
    # ("Bohemia") in the roster, which then absorbs the character's lines.
    title_pattern = re.compile(r"\b(?:King|Queen|Prince|Princess|Duke|Duchess|Count|Countess|Baron|Baroness|Lord|Lady|Emperor|Empress)\s+(?:of\s+|von\s+|van\s+|de\s+)?[A-Z][A-Za-z'\-]+")
    honorific_names = set()
    for pattern in (honorific_pattern, title_pattern):
        for match in pattern.finditer(full_text):
            name = re.sub(r"\s+", " ", match.group(0).strip())
            counts[name] = counts.get(name, 0) + 1
            honorific_names.add(name)
    # Second char must be lowercase so ALL-CAPS title-page words (BEATRIX POTTER)
    # don't qualify, while internal caps (McGregor) and hyphens (Cotton-tail) do.
    name_pattern = re.compile(r"\b[A-Z][a-z][A-Za-z'\-]*(?:\s+[A-Z][a-z][A-Za-z'\-]*){0,2}\b")
    sentence_boundary_chars = set('.!?"“”‘’\'\n—-')
    for match in name_pattern.finditer(full_text):
        name = match.group(0).strip()
        if len(name) < 3 or name.split()[0].lower() in _ROSTER_STOPWORDS:
            continue
        counts[name] = counts.get(name, 0) + 1
        # Track whether this token ever appears capitalized MID-sentence: genuine
        # proper nouns do ("made in Bohemia"), while sentence-starter junk ("What",
        # "However") is only ever capitalized right after a boundary/quote.
        prefix = full_text[:match.start()].rstrip()
        if prefix and prefix[-1] not in sentence_boundary_chars:
            midsentence_counts[name] = midsentence_counts.get(name, 0) + 1
    recurring = [
        n for n, c in counts.items()
        if c >= 2 and (n in honorific_names or midsentence_counts.get(n, 0) >= 1)
    ]
    recurring.sort(key=lambda n: (-counts[n], n))
    return recurring[:30]


def _extract_candidate_roster(lines: List[ScriptLine]) -> List[str]:
    """Heuristic character-name candidate extraction from narrative segments only.

    Not ground truth -- just candidates the LLM can confirm or relabel during
    attribution, so a noisy heuristic (capitalized multi-word sequences minus a
    common-word stoplist) is good enough here. Requires >=2 mentions within the
    scene to filter one-off false positives (sentence-initial capitalized words,
    onomatopoeia like "Kertyschoo!") that real recurring character names don't
    share -- a real speaker in an active scene is almost always mentioned more
    than once.
    """
    counts: Dict[str, int] = {}
    name_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b')
    for line in lines:
        if line.segment_type != "narrative":
            continue
        for match in name_pattern.finditer(line.text):
            name = match.group(0).strip()
            if len(name) < 3:
                continue
            if name.split()[0].lower() in _ROSTER_STOPWORDS:
                continue
            counts[name] = counts.get(name, 0) + 1
    return sorted(name for name, count in counts.items() if count >= 2)


def _locate_fragment(scene_text: str, fragment: str) -> int:
    """Whitespace-tolerant position of a line's text inside the scene, -1 if absent.
    Scene text wraps quotes across newlines, so an exact find() misses; matching the
    first few words with \\s+ between them survives the re-wrapping."""
    words = re.sub(r'[“”"‘’\']', "", fragment.strip()).split()[:8]
    if not words:
        return -1
    m = re.search(r"\s+".join(re.escape(w) for w in words), scene_text)
    return m.start() if m else -1


def enrich_scene_lines_with_llm(scene_text: str, lines: List[ScriptLine], global_roster: Optional[List[str]] = None) -> Tuple[List[ScriptLine], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Opt-in LLM enrichment pass for one scene's already-segmented Tier 1 lines.

    Two independent passes, each degrading separately on failure:
    - Clean-check (Loop5-equivalent, prompt reused from looped_analyzer.py): advisory
      only in v1, returns flagged issues rather than mutating scene_text.
    - Attribution (Loop6-inspired but index-preserving): relabels character/speaker_id/
      attribution_method/confidence/emotion/speaker_locked on existing dialogue lines
      without resegmenting -- Tier 1's own regex-derived boundaries stay authoritative.

    Never raises; returns the original `lines` unchanged (plus empty issues) if the
    LLM chain is disabled, unreachable, or returns an unvalidatable response.
    """
    from src.llm_client import query_llm_json
    from src.looped_analyzer import Loop5ResponseSchema

    clean_issues: List[Dict[str, Any]] = []
    try:
        clean_prompt = f"""You are a professional manuscript editor and ingestion auditor for an audiobook production engine.
Your objective is to run a "Manuscript Clean Check" on the provided text block.
Identify any non-narrative elements that should be removed or cleaned before sending to the Speech Synthesis (TTS) engine.

Identify:
1. Project Gutenberg license headers/footers, metadata, translator/author lists, or ebook credits.
2. Illustration tags or description brackets (e.g. "[Illustration: ...]" or "[Illustration]").
3. Page numbers, headers, footers, or transcriber notes.
4. Extraneous formatting noise (e.g. raw underscores representing formatting that shouldn't be read as words).

OUTPUT FORMAT:
You must output a valid JSON object matching this schema:
{{
  "is_clean": [Boolean: true if no issues are found, false if issues are found],
  "issues": [
    {{
      "issue_type": "[e.g. Gutenberg header, illustration tag, page number, formatting noise]",
      "raw_text": "[The exact raw text snippet containing the issue]",
      "description": "[Why this text should be cleaned or omitted for narration]",
      "suggested_action": "['remove' or 'replace']",
      "suggested_text": "[Replacement text if action is 'replace', otherwise empty string]"
    }}
  ]
}}

TEXT BLOCK TO AUDIT:
{scene_text[:4000]}
"""
        # Gemini-only: measured on Peter Rabbit, the Groq 8B fallback false-flags
        # real story text as boilerplate. Advisory pass, so skipping when Gemini
        # is unavailable beats producing garbage flags.
        clean_res, _ = query_llm_json(clean_prompt, schema=Loop5ResponseSchema, task_name="tier1_cleancheck", allowed_providers=("gemini",))
        if clean_res:
            validated_clean = Loop5ResponseSchema.model_validate(clean_res)
            if not validated_clean.is_clean:
                clean_issues = [issue.model_dump() for issue in validated_clean.issues]
    except Exception as e:
        logger.warning(f"Tier 1 clean-check enrichment failed, skipping for this scene: {e}")

    # Tiny quoted fragments (single letters/symbols, e.g. a deciphered watermark's
    # "E" "g" "P") are quoted material being read, not attributable dialogue --
    # leave them on Tier 1 defaults rather than asking the LLM to attribute them.
    dialogue_indices = [
        i for i, l in enumerate(lines)
        if l.segment_type == "dialogue" and len(re.sub(r"[^A-Za-z0-9]", "", l.text)) >= 3
    ]
    if not dialogue_indices:
        return lines, clean_issues, []

    roster = sorted(set(_extract_candidate_roster(lines)) | set(global_roster or []))
    roster_str = ", ".join(roster) if roster else "(no obvious named characters detected)"

    # G4 director scenes run far larger than the old regex scenes (measured: a
    # 136-dialogue-line Case of Identity scene overflowed Groq's 6K-token request
    # cap with a 413 and timed out the 3B Ollama fallback, leaving the whole scene
    # on Tier 1 defaults). Window the attribution so every call fits every
    # provider, each window carrying a scene-text slice local to its own lines.
    _WINDOW = 30
    chunks = [dialogue_indices[i:i + _WINDOW] for i in range(0, len(dialogue_indices), _WINDOW)]

    def _chunk_context(chunk_no: int, chunk: List[int]) -> str:
        if len(chunks) == 1:
            return scene_text[:4000]
        start = _locate_fragment(scene_text, lines[chunk[0]].text)
        if start < 0:
            seg = max(1, len(scene_text) // len(chunks))
            start = chunk_no * seg
        end = _locate_fragment(scene_text[start:], lines[chunk[-1]].text)
        end = (start + end + len(lines[chunk[-1]].text)) if end >= 0 else (start + 3000)
        return scene_text[max(0, start - 500):end + 500][:4000]

    def _build_attribution_prompt(context: str, indexed_lines_str: str) -> str:
        return f"""You are an expert literary analysis AI attributing dialogue lines to speakers for an audiobook engine.
Below is the full text of one scene (for context), a candidate character roster (heuristically
extracted from the WHOLE BOOK, so it may include characters named in earlier scenes but only
referenced by pronoun or epithet here), and a list of dialogue lines from this scene, each tagged
with its index number. Use the surrounding narrative to infer who is speaking -- e.g. an action or
attribution tag ("Peter sneezed", "she gave a dose of it to Peter") immediately before or after a
dialogue line is strong evidence of who that line belongs to, even if the character isn't named
directly inside the quoted line itself. If the scene refers to a speaker only indirectly (e.g.
"his mother" or "the old man"), map that reference to the matching named character from the
roster (e.g. the mother character's actual name) rather than answering "Narrator".

CRITICAL ATTRIBUTION RULES:
- A name, title, or honorific appearing INSIDE a quoted line usually identifies the person being
  ADDRESSED (the listener), NOT the speaker. "I fail to follow your Majesty" is spoken TO the
  royal character BY someone else. "You'll do fine, Watson" is spoken TO Watson.
- In a conversation between two characters, consecutive dialogue lines usually ALTERNATE between
  them. A character almost never replies to their own line. Track who is talking to whom.
- Some roster candidates may be places, objects, or other non-persons picked up by a heuristic.
  NEVER attribute dialogue to a place or object -- pick the actual person, or "Narrator".

FULL SCENE TEXT (for context only -- do not attribute lines from outside the DIALOGUE LINES list):
{context}

CANDIDATE CHARACTER ROSTER: [{roster_str}]

DIALOGUE LINES (attribute each by its index number; do NOT invent new text, only classify):
{indexed_lines_str}

Return a JSON object with two keys:

1. "lines": an array of objects, one per index listed above, each with:
- "index" (int, matching the number given above)
- "speaker" (string, a name from the candidate roster if it plausibly spoke the line, or "Narrator"
  if the true speaker is genuinely ambiguous or not in the roster, even after considering context)
- "emotion" (string, e.g. Joy, Sadness, Fear, Anger, Neutral)
- "confidence" (float between 0.0 and 1.0, your certainty in this attribution)
- "utterance_type" (string): "speech" for normal spoken words, or "vocalization" for a non-lexical
  sound a character produces (a sneeze like "Kertyschoo!", a gasp, a sob, a scream, an animal noise).
  Vocalizations still get attributed to the character who made them, but a speech engine must
  perform them as sounds rather than read them as words.

2. "sfx_cues": an array (possibly empty) of environmental sound effects represented as onomatopoeia
   inside the NARRATIVE text of the scene (not inside dialogue quotes) -- e.g. a tool going
   "scr-r-ritch, scratch, scratch", a door going "bang", hooves clattering. Each entry:
- "sound_text" (string, the onomatopoeic text exactly as it appears in the scene)
- "description" (string, what real-world sound it represents, e.g. "garden hoe scraping soil")
Do NOT include character vocalizations (sneezes, cries) here -- those belong in "lines" as
utterance_type "vocalization". Do NOT include quoted dialogue, shouts, or any words spoken by a
character -- spoken words are never SFX, even when loud. Only non-speech environmental sounds
written as onomatopoeia in the narration qualify. Every "sound_text" MUST be copied verbatim
from the FULL SCENE TEXT above -- never invent one and never copy an example from these
instructions; if the scene contains no such sounds, return an empty "sfx_cues" array.
"""
    enriched_lines = list(lines)
    raw_sfx_cues: List[Any] = []
    enriched_any = False
    for chunk_no, chunk in enumerate(chunks):
        indexed_lines_str = "\n".join(f'{i}: "{lines[i].text}"' for i in chunk)
        attribution_prompt = _build_attribution_prompt(_chunk_context(chunk_no, chunk), indexed_lines_str)
        try:
            attr_res, provider = query_llm_json(attribution_prompt, schema=TierOneAttributionSchema, task_name="tier1_attribution")
            if not attr_res:
                continue
            validated_attr = TierOneAttributionSchema.model_validate(attr_res)
        except Exception as e:
            logger.warning(f"Tier 1 attribution enrichment failed for window {chunk_no + 1}/{len(chunks)}, keeping Tier 1 defaults there: {e}")
            continue

        chunk_set = set(chunk)
        raw_sfx_cues.extend(validated_attr.sfx_cues)
        for attr_line in validated_attr.lines:
            idx = attr_line.index
            if idx not in chunk_set or idx < 0 or idx >= len(enriched_lines):
                continue
            original = enriched_lines[idx]
            if original.segment_type != "dialogue":
                continue
            speaker = attr_line.speaker.strip()
            if not speaker or speaker.lower() in ("narrator", "[narration]", "narration"):
                continue  # keep Tier 1 Default values for narrator-attributed dialogue

            matched_speaker = speaker
            for candidate in roster:
                if candidate.lower() in speaker.lower() or speaker.lower() in candidate.lower():
                    matched_speaker = candidate
                    break

            confidence = max(0.0, min(1.0, attr_line.confidence))
            enriched_lines[idx] = original.model_copy(update={
                "character": matched_speaker,
                "speaker_id": f"char_{matched_speaker.lower().replace(' ', '_')}",
                "attribution_method": provider or "LLM Enrichment",
                "confidence": confidence,
                "emotion": attr_line.emotion.title() if attr_line.emotion else "Neutral",
                "speaker_locked": confidence >= 0.6,
                "utterance_type": attr_line.utterance_type,
            })
            enriched_any = True
    if not enriched_any and len(chunks) > 1:
        logger.warning("Attribution enrichment produced no speaker updates across all windows; scene stays on Tier 1 defaults.")

    # Programmatic grounding check: models (especially the 8B fallbacks) sometimes
    # copy the prompt's few-shot example ("scr-r-ritch...") instead of quoting the
    # scene, despite explicit verbatim-only instructions. Instructions alone don't
    # stop this; dropping any cue whose sound_text isn't literally in the scene does.
    def _norm(t: str) -> str:
        t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        return re.sub(r"\s+", " ", t.strip().lower())

    scene_norm = _norm(scene_text)
    sfx_cues = []
    seen_sounds = set()
    for cue in raw_sfx_cues:
        snippet = _norm(cue.sound_text)
        if len(snippet) < 3 or snippet in seen_sounds:
            continue
        if snippet not in scene_norm:
            logger.warning(f"Dropping ungrounded SFX cue (not in scene text): {cue.sound_text[:50]!r}")
            continue
        seen_sounds.add(snippet)
        sfx_cues.append(cue.model_dump())

    # AI-12: second-pass conversation-flow review over the attributed lines
    # (catches the measured residual classes: exchange drift, addressee inversion,
    # minor characters absorbing lines). No-ops on small/one-speaker scenes.
    try:
        enriched_lines = review_scene_attribution(scene_text, enriched_lines, roster)
    except Exception as e:
        logger.warning(f"Continuity review skipped: {e}")

    return enriched_lines, clean_issues, sfx_cues


class ChapterHeadingProposal(BaseModel):
    heading_line: str = Field(..., description="The EXACT heading line as it appears in the text, copied verbatim")


class ChapterVerifierSchema(BaseModel):
    headings: List[ChapterHeadingProposal] = Field(default_factory=list)


def verify_parts_with_llm(full_text: str, existing_count: int) -> Optional[List[Dict[str, Any]]]:
    """G1: Part Verifier gate (task tier1_part_verification).

    Engaged when top-level division detection looks implausible: regex runaway
    (measured: Les Miserables yields 92 "parts" because VOLUME and BOOK headings
    both match as siblings), or a single monolithic part in a very large book.
    The LLM identifies the TOP-level division scheme and lists those heading lines
    VERBATIM; boundaries are grounded by exact search. Deterministic parts stand
    on any failure.
    """
    from src.llm_client import query_llm_json

    sample = full_text[:15000]
    if len(full_text) > 60000:
        mid = len(full_text) // 2
        sample += "\n\n[...]\n\n" + full_text[mid:mid + 8000]

    prompt = f"""You are a manuscript structure analyst. The automatic detector found
{existing_count} top-level division(s) in this book, which looks wrong. Books often nest
divisions (VOLUME > BOOK > CHAPTER, or TOME > LIVRE > CHAPITRE): identify the single
TOP-MOST division level actually used, and list ONLY those heading lines.

TEXT SAMPLE:
{sample}

Return JSON: {{"headings": [{{"heading_line": "..."}}]}}
- heading_line: copied EXACTLY as it appears, TOP level only (e.g. "VOLUME I" but NOT
  "BOOK FIRST" if books nest inside volumes).
- If the book has no top-level divisions above chapters, return an empty list."""
    try:
        res, provider = query_llm_json(prompt, schema=ChapterVerifierSchema, task_name="tier1_part_verification")
        if not res:
            return None
        validated = ChapterVerifierSchema.model_validate(res)
    except Exception as e:
        logger.warning(f"G1 verifier failed: {e}")
        return None
    if len(validated.headings) < 2:
        return None

    # Derive the heading PATTERN from grounded examples: find all occurrences of
    # each verbatim heading, then extend to same-prefix siblings (VOLUME I grounds
    # -> all "VOLUME <numeral>" lines split the book).
    import re as _re
    prefixes = set()
    for h in validated.headings:
        m = _re.match(r"([A-Za-z]+)\s+[IVXLCDM0-9]", h.heading_line.strip())
        if m and full_text.find(h.heading_line.strip()) != -1:
            prefixes.add(m.group(1).upper())
    if not prefixes:
        return None
    prefix_pattern = _re.compile(
        r"^\s*(?:" + "|".join(_re.escape(p) for p in prefixes) + r")\s+(?:[IVXLCDM]+|[0-9]+|FIRST|SECOND|THIRD|FOURTH|FIFTH)\b.*$",
        _re.MULTILINE | _re.IGNORECASE,
    )
    matches = [(m.start(), m.group(0).strip()) for m in prefix_pattern.finditer(full_text)]
    filtered = []
    last = -10**9
    for pos, heading in matches:
        if pos - last < 2000:
            continue
        filtered.append((pos, heading))
        last = pos
    if len(filtered) < 2:
        return None

    parts = []
    if filtered[0][0] > 500:
        parts.append({"part_id": "part_p0", "title": "Preface/Front Matter", "text_block": full_text[:filtered[0][0]].strip()})
    for i, (pos, heading) in enumerate(filtered):
        end = filtered[i + 1][0] if i + 1 < len(filtered) else len(full_text)
        block = full_text[pos + len(heading):end].strip()
        if len(block) < 500:
            continue
        parts.append({
            "part_id": f"part_p{len(parts) + 1}",
            "title": heading[:80],
            "text_block": block,
        })
    return parts if 2 <= len(parts) <= existing_count else None


def verify_chapters_with_llm(part_text: str, part_id: str, existing_count: int) -> Optional[List[Dict[str, Any]]]:
    """G2: Chapter Verifier gate (task tier1_chapter_verification).

    Engaged only when deterministic chapter detection looks implausible (a huge
    single-chapter block, or wildly nonuniform chapter sizes). Asks the LLM to list
    the book's actual heading lines VERBATIM; every proposal is grounded by exact
    search before any re-split. Deterministic chapters stand on any failure.
    """
    from src.llm_client import query_llm_json

    # Sample the text: heading formats are visible in the first ~15K chars plus a
    # mid-book window (headings repeat consistently).
    sample = part_text[:15000]
    if len(part_text) > 40000:
        mid = len(part_text) // 2
        sample += "\n\n[...]\n\n" + part_text[mid:mid + 8000]

    prompt = f"""You are a manuscript structure analyst. The automatic chapter detector found
{existing_count} chapter(s) in this text, which looks wrong. Examine the sample below and
identify the format of the ACTUAL chapter/section headings, then list heading lines.

TEXT SAMPLE:
{sample}

Return JSON: {{"headings": [{{"heading_line": "..."}}]}}
- Each heading_line must be copied EXACTLY as it appears (a line that begins a chapter
  or section, e.g. "CHAPTER I.", "II. THE RED HEADED LEAGUE", "Letter 4").
- List every heading visible in the sample, in order. If the text genuinely has no
  chapter structure, return an empty list."""
    try:
        res, provider = query_llm_json(prompt, schema=ChapterVerifierSchema, task_name="tier1_chapter_verification")
        if not res:
            return None
        validated = ChapterVerifierSchema.model_validate(res)
    except Exception as e:
        logger.warning(f"G2 verifier failed for {part_id}: {e}")
        return None
    if len(validated.headings) < 2:
        return None

    # Grounding: derive a positional split from verbatim heading matches. Use the
    # SHAPE of validated headings to find all occurrences book-wide: exact matches
    # first; if >=2 exact headings ground, build boundaries from every exact match.
    positions = []
    seen_pos = set()
    for h in validated.headings:
        needle = h.heading_line.strip()
        if len(needle) < 3:
            continue
        start = 0
        while True:
            pos = part_text.find(needle, start)
            if pos == -1:
                break
            line_start = part_text.rfind("\n", 0, pos) + 1
            if abs(pos - line_start) <= 2 and pos not in seen_pos:
                positions.append((pos, needle))
                seen_pos.add(pos)
            start = pos + 1
    positions.sort()
    # De-duplicate near positions and require sane chapter sizes
    filtered = []
    last = -10**9
    for pos, needle in positions:
        if pos - last < 500:
            continue
        filtered.append((pos, needle))
        last = pos
    if len(filtered) < 2:
        return None

    chapters = []
    if filtered[0][0] > 200:
        chapters.append({"chapter_id": f"{part_id}_c0", "title": "Prologue", "text_block": part_text[:filtered[0][0]].strip()})
    for i, (pos, needle) in enumerate(filtered):
        end = filtered[i + 1][0] if i + 1 < len(filtered) else len(part_text)
        block = part_text[pos + len(needle):end].strip()
        if len(block) < 200:
            continue
        chapters.append({
            "chapter_id": f"{part_id}_c{len(chapters) + 1}",
            "title": needle[:80],
            "text_block": block,
        })
    return chapters if len(chapters) >= 2 else None


def director_segment_chapter(chapter_text: str, chapter_id: str) -> Optional[List[Dict[str, Any]]]:
    """G4: Director's Scene Segmenter (gate AI, task tier1_scene_segmentation).

    Semantic scene segmentation using the director's definition: a scene breaks
    when LOCATION, TIME, or the set of PRESENT CHARACTERS changes. Engaged only
    when the deterministic pass found no explicit typographic markers (measured:
    that is 100% of the current corpus).

    Chapters are processed in ~9K-char windows split at paragraph boundaries.
    Every proposed boundary must quote its opening words verbatim (grounding);
    boundaries too close together (<400 chars) are dropped. Returns None if no
    valid boundaries survive -- the deterministic scenes stand.
    """
    from src.llm_client import query_llm_json

    def _norm_g4(t: str) -> str:
        t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        return re.sub(r"\s+", " ", t.lower())

    # Window the chapter at paragraph boundaries
    paragraphs = [p for p in re.split(r'\n\n+', chapter_text) if p.strip()]
    windows: List[str] = []
    current: List[str] = []
    size = 0
    for p in paragraphs:
        current.append(p)
        size += len(p)
        if size >= 9000:
            windows.append("\n\n".join(current))
            current, size = [], 0
    if current:
        windows.append("\n\n".join(current))

    proposals: List[SceneBoundaryProposal] = []
    for w_idx, window in enumerate(windows, 1):
        prompt = f"""You are the SCENE SUPERVISOR for an audiobook production engine, segmenting a chapter
into scenes the way a film director would. THE DIRECTOR'S DEFINITION: a new scene begins when
there is a break in continuity of LOCATION (characters move somewhere else), TIME (a jump
forward or backward), or PRESENT CHARACTERS (the active group changes substantially).
Do NOT break scenes for mere paragraph changes, dialogue turns, or emotional beats within
one continuous moment -- a scene is one continuous dramatic unit in one place and time.

CHAPTER TEXT (window {w_idx} of {len(windows)}):
{window}

Return JSON: {{"boundaries": [{{"start_snippet": "...", "reason": "..."}}]}}
- "start_snippet": the EXACT first 8-15 words of the paragraph where a NEW scene begins,
  copied verbatim from the text above. Do not include the very first paragraph of the window
  unless it genuinely starts a new scene relative to what precedes it.
- "reason": one short phrase naming the shift, e.g. "location shift: garden -> tool-shed",
  "time jump: next morning", "cast change: McGregor enters".
Return 0 boundaries if this window is one continuous scene. Quality over quantity."""
        try:
            res, provider = query_llm_json(prompt, schema=DirectorSceneSchema, task_name="tier1_scene_segmentation")
            if res:
                validated = DirectorSceneSchema.model_validate(res)
                proposals.extend(validated.boundaries)
        except Exception as e:
            logger.warning(f"G4 window {w_idx} failed for {chapter_id}: {e}")

    if not proposals:
        return None

    # Ground every proposal: locate its snippet in the full chapter text
    chapter_norm = _norm_g4(chapter_text)
    positions: List[Tuple[int, str]] = []
    for prop in proposals:
        snippet = _norm_g4(prop.start_snippet)[:80]
        if len(snippet) < 15:
            continue
        pos = chapter_norm.find(snippet)
        if pos <= 200:  # not found, or trivially at chapter start
            if pos == -1:
                logger.warning(f"G4 dropping ungrounded boundary: {prop.start_snippet[:40]!r}")
            continue
        positions.append((pos, prop.reason))

    # Order, dedupe, and enforce minimum scene length
    positions.sort()
    filtered: List[Tuple[int, str]] = []
    last = 0
    for pos, reason in positions:
        if pos - last < 400:
            continue
        filtered.append((pos, reason))
        last = pos
    if not filtered:
        return None

    # Map normalized positions back to raw-text offsets (normalization collapses
    # whitespace, so walk the raw text and normalized text in lockstep)
    raw_offsets: List[int] = []
    norm_to_raw: List[int] = []
    norm_chars = []
    prev_space = False
    for i, ch in enumerate(chapter_text):
        c = {"’": "'", "‘": "'", "“": '"', "”": '"'}.get(ch, ch)
        if c.isspace():
            if prev_space:
                continue
            c = " "
            prev_space = True
        else:
            prev_space = False
        norm_chars.append(c.lower())
        norm_to_raw.append(i)
    for pos, _ in filtered:
        raw_offsets.append(norm_to_raw[min(pos, len(norm_to_raw) - 1)])

    # Slice scenes at raw offsets (snapping each to the nearest preceding paragraph break)
    snapped = []
    for off in raw_offsets:
        para_start = chapter_text.rfind("\n\n", 0, off)
        snapped.append(off if para_start == -1 else para_start + 2)
    boundaries = sorted(set(snapped))

    scenes = []
    starts = [0] + boundaries
    ends = boundaries + [len(chapter_text)]
    reasons = ["chapter opening"] + [r for _, r in filtered][:len(boundaries)]
    for i, (s, e) in enumerate(zip(starts, ends), 1):
        block = chapter_text[s:e].strip()
        if not block:
            continue
        scenes.append({
            "scene_id": f"{chapter_id}_s{i}",
            "text_block": block,
            "boundary_source": "director_ai",
            "boundary_reason": reasons[i - 1] if i - 1 < len(reasons) else "",
        })
    return scenes if len(scenes) >= 2 else None


_HONORIFIC_TOKENS = {"mr", "mrs", "ms", "miss", "dr", "sir", "lady", "lord", "monsieur", "madame", "mademoiselle", "mme", "mlle"}


def _composed_name_merges(speakers: Dict[str, Dict[str, Any]], roster: List[str]) -> List[Dict[str, Any]]:
    """Deterministic pre-merge: if two speakers' distinctive name tokens together
    compose one longer roster name, they are the same person -- no LLM judgment
    needed. Measured case: Gemini correctly declines to merge "Miss Mary" +
    "Miss Sutherland" from names alone, but the book roster contains
    "Miss Mary Sutherland", which settles it deterministically."""
    def _tokens(name: str) -> set:
        return {t for t in re.sub(r"[^\w\s]", "", name.lower()).split() if t and t not in _HONORIFIC_TOKENS}

    full_names = [r for r in roster if len(_tokens(r)) >= 2]
    names = list(speakers.keys())
    groups = []
    used = set()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a in used or b in used:
                continue
            ta, tb = _tokens(a), _tokens(b)
            if not ta or not tb or ta == tb:
                continue
            for full in full_names:
                if (ta | tb) == _tokens(full) and ta != _tokens(full) and tb != _tokens(full):
                    canonical, alias = (a, b) if speakers[a]["count"] >= speakers[b]["count"] else (b, a)
                    groups.append({"canonical": canonical, "aliases": [alias], "resolved_by": f"deterministic:composed-name ({full})"})
                    used.update((a, b))
                    break
    return groups


def review_scene_attribution(scene_text: str, lines: List[ScriptLine], roster: List[str]) -> List[ScriptLine]:
    """AI-12: Continuity Reviewer (task tier1_attribution_review).

    Second-pass review of the ATTRIBUTED conversation flow -- catches the error
    classes per-line attribution can't see (measured residuals on the Case of
    Identity benchmark): rapid two-party exchange drift, addressee-inversion
    survivors, and minor characters absorbing a protagonist's lines.

    Enforcement over trust: corrections only to speakers already present in the
    scene or roster; capped at 30% of dialogue lines (a reviewer rewriting
    everything is itself wrong); failures leave attribution untouched.
    """
    from src.llm_client import query_llm_json

    dialogue_idx = [i for i, l in enumerate(lines) if l.segment_type == "dialogue"]
    speakers_in_scene = {lines[i].character for i in dialogue_idx}
    if len(dialogue_idx) < 6 or len(speakers_in_scene - {"Narrator"}) < 2:
        return lines

    allowed = sorted((speakers_in_scene | set(roster)) - {"Narrator"})
    allowed_set = set(allowed) | {"Narrator"}
    reviewed = list(lines)

    # Same provider-size constraint as attribution: review large G4 scenes in
    # windows so the transcript fits Groq's per-request cap. Conversation-flow
    # errors are local, so windowed review loses nothing.
    _RWINDOW = 40
    for wstart in range(0, len(dialogue_idx), _RWINDOW):
        window = dialogue_idx[wstart:wstart + _RWINDOW]
        if wstart > 0 and len(window) < 6:
            break  # tail too short to judge flow in isolation

        transcript = "\n".join(
            f'{i}: [{reviewed[i].character}] "{reviewed[i].text[:100]}"' for i in window
        )
        ctx_start = _locate_fragment(scene_text, reviewed[window[0]].text) if wstart > 0 else 0
        ctx_start = max(0, ctx_start - 300) if ctx_start >= 0 else 0
        prompt = f"""You are the CONTINUITY REVIEWER for an audiobook attribution engine. Below is one
scene's dialogue AS ATTRIBUTED by a first pass. Most attributions are correct. Your job is
to catch the few that break conversational logic:
- In a two-party exchange, consecutive lines usually ALTERNATE speakers; the same speaker
  rarely replies to their own line.
- A line addressing someone BY NAME ("...you see, Watson") is spoken TO that person, not
  BY them.
- An answer belongs to the person who was asked, not to the questioner.
- A line describing a character's own actions in third person is probably not theirs.

ATTRIBUTED TRANSCRIPT (index: [speaker] "text"):
{transcript}

SCENE NARRATIVE CONTEXT:
{scene_text[ctx_start:ctx_start + 2500]}

ALLOWED SPEAKERS: [{", ".join(allowed)}]

Return JSON: {{"corrections": [{{"index": int, "current_speaker": str,
"corrected_speaker": str (MUST be from ALLOWED SPEAKERS or "Narrator"),
"reason": short phrase}}]}} -- ONLY lines you are confident are wrong; an empty list is
a good answer for a clean scene."""
        try:
            res, provider = query_llm_json(prompt, schema=AttributionReviewSchema, task_name="tier1_attribution_review")
            if not res:
                continue
            validated = AttributionReviewSchema.model_validate(res)
        except Exception as e:
            logger.warning(f"Attribution review failed, keeping first-pass attribution: {e}")
            continue

        max_corrections = max(1, int(0.3 * len(window)))
        if len(validated.corrections) > max_corrections:
            logger.warning(f"Reviewer proposed {len(validated.corrections)} corrections (cap {max_corrections}); rejecting the pass as over-eager.")
            continue

        window_set = set(window)
        applied = 0
        for corr in validated.corrections:
            i = corr.index
            if i not in window_set or reviewed[i].segment_type != "dialogue":
                continue
            if reviewed[i].character != corr.current_speaker:
                continue  # stale reference: reviewer is talking about a different state
            if corr.corrected_speaker not in allowed_set or corr.corrected_speaker == corr.current_speaker:
                continue
            reviewed[i] = reviewed[i].model_copy(update={
                "character": corr.corrected_speaker,
                "speaker_id": f"char_{corr.corrected_speaker.lower().replace(' ', '_')}",
                "attribution_method": (reviewed[i].attribution_method or "") + "+reviewed",
                "confidence": 0.75,
            })
            applied += 1
        if applied:
            logger.info(f"Continuity review ({provider}): applied {applied} correction(s).")
    return reviewed


def merge_speaker_aliases(part_payloads: List["PartPayload"], roster: Optional[List[str]] = None) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Book-level alias resolution: one LLM call grouping attributed speakers that
    refer to the same person (titles, ranks, disguises, partial names -- e.g.
    "Count Von Kramm" / "The Count" / "Majesty" / "King" are all the King of
    Bohemia). Returns (alias -> canonical mapping, validated groups).

    A deterministic composed-name pre-pass (see _composed_name_merges) runs first;
    the LLM adds to, never overrides, its results. Validated so the model can only
    group names that actually exist as speakers, never invent a new canonical, and
    never touch "Narrator". Returns empty on any failure -- attribution stands
    unmerged rather than wrongly merged.
    """
    from src.llm_client import query_llm_json

    speakers: Dict[str, Dict[str, Any]] = {}
    for part in part_payloads:
        for chapter in part.chapters:
            for scene in chapter.scenes:
                for line in scene.lines:
                    if line.segment_type != "dialogue" or line.character == "Narrator":
                        continue
                    rec = speakers.setdefault(line.character, {"count": 0, "samples": []})
                    rec["count"] += 1
                    if len(rec["samples"]) < 2:
                        rec["samples"].append(line.text[:70])

    if len(speakers) < 3:
        return {}, []

    deterministic_groups = _composed_name_merges(speakers, roster or [])

    speaker_listing = "\n".join(
        f'- "{name}" ({rec["count"]} lines; e.g. "{rec["samples"][0]}")'
        for name, rec in sorted(speakers.items(), key=lambda kv: -kv[1]["count"])
    )
    prompt = f"""You are resolving character identities for an audiobook voice-casting engine.
Below are the speaker names an attribution pass assigned to dialogue lines in one book, with
line counts and a sample line each. Some names refer to the SAME person under different labels:
titles vs personal names ("Majesty" / "King" / "Count Von Kramm" may all be one royal character
in disguise), partial names ("McGregor" / "Mr. McGregor"), epithets, or misspellings.

SPEAKERS:
{speaker_listing}

Group ONLY names that clearly refer to the same single person. Rules:
- "canonical" MUST be one of the listed speaker names -- pick the most complete/specific personal
  name in the group. Never invent a name that is not listed.
- Do not merge two distinct people just because their names are similar.
- Omit speakers that have no aliases (singleton groups are not needed).
- Never include "Narrator" in any group.

Return JSON: {{"groups": [{{"canonical": "...", "aliases": ["...", "..."]}}]}}
If no names refer to the same person, return {{"groups": []}}.
"""
    try:
        # Gemini-only: identity merging is quality-critical and a wrong merge is far
        # worse than no merge (measured: the Groq 8B merged "King" into "Holmes").
        # One call per book, so Gemini's low RPM is not a constraint here.
        res, provider = query_llm_json(prompt, schema=AliasMergeSchema, task_name="tier1_alias_merge", allowed_providers=("gemini",))
        if not res:
            return {}, []
        validated = AliasMergeSchema.model_validate(res)
    except Exception as e:
        logger.warning(f"Alias-merge pass failed, keeping speakers unmerged: {e}")
        return {}, []

    def _conversation_switches(a: str, b: str) -> int:
        """Counts DIRECTLY ADJACENT dialogue alternations between a and b (no other
        speaker's dialogue between them). Real conversation partners adjoin
        constantly; two labels for the SAME person almost never do -- their lines
        interleave with a third party (measured false-positive: "Miss Mary" /
        "Miss Sutherland" flapping within the Holmes interview would count as
        conversation under a filtered-sequence definition and block the merge)."""
        switches = 0
        for part in part_payloads:
            for chapter in part.chapters:
                for scene in chapter.scenes:
                    prev_dialogue_char = None
                    for line in scene.lines:
                        if line.segment_type != "dialogue":
                            continue
                        if prev_dialogue_char in (a, b) and line.character in (a, b) and line.character != prev_dialogue_char:
                            switches += 1
                        prev_dialogue_char = line.character
        return switches

    mapping: Dict[str, str] = {}
    valid_groups: List[Dict[str, Any]] = []
    claimed: set = set()
    # Deterministic composed-name groups first (still blocker-checked); the LLM
    # adds to, never overrides, these.
    for group in deterministic_groups:
        alias = group["aliases"][0]
        canonical = group["canonical"]
        if _conversation_switches(alias, canonical) >= 2:
            logger.warning(f"Blocking deterministic merge '{alias}' -> '{canonical}': conversation alternation.")
            continue
        mapping[alias] = canonical
        claimed.update((alias, canonical))
        valid_groups.append(group)
    for group in validated.groups:
        canonical = group.canonical.strip()
        if canonical not in speakers or canonical.lower() == "narrator":
            continue
        aliases = []
        for a in group.aliases:
            a = a.strip()
            if a not in speakers or a == canonical or a.lower() == "narrator" or a in claimed:
                continue
            # A single alternation can be a legitimate identity-reveal boundary
            # (Count Von Kramm -> King mid-scene); repeated alternation means the
            # two names are conversing with each other and must be distinct people.
            switches = _conversation_switches(a, canonical)
            if switches >= 2:
                logger.warning(f"Blocking alias merge '{a}' -> '{canonical}': they alternate in conversation ({switches} switches).")
                continue
            aliases.append(a)
        if not aliases:
            continue
        for alias in aliases:
            mapping[alias] = canonical
            claimed.add(alias)
        claimed.add(canonical)
        valid_groups.append({"canonical": canonical, "aliases": aliases, "resolved_by": provider})

    if mapping:
        logger.info(f"Alias merge resolved {len(mapping)} alias(es) into {len(valid_groups)} identity group(s).")
    return mapping, valid_groups


# Ingestion Logic

def clean_front_matter(text: str) -> str:
    """Strips Project Gutenberg headers and Gutenberg-specific front matter using ClutterScrubber."""
    from nlp_engine.stage_1_ingestion import ClutterScrubber
    scrubber = ClutterScrubber()
    return scrubber.remove_front_matter(text)

def identify_parts(raw_text: str) -> List[Dict[str, Any]]:
    """
    Loop 1: Part Identification (Macro-Router)
    Slices raw text by major book divisions (Volume, Book, Part).
    Gate 1 Fallback: Wraps the text in a single Part if no divisions exist.
    """
    cleaned_text = clean_front_matter(raw_text)
    
    # Pattern to match VOLUME, BOOK, PART on a line by itself (with optional trailing title)
    part_pattern = re.compile(
        r'^\s*(?:VOLUME|BOOK|PART|LIVRE|TOME)\s+(?:[IVXLCDM]+|[0-9]+|FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ONE|TWO|THREE|FOUR|FIVE)\b.*$',
        re.IGNORECASE | re.MULTILINE
    )
    
    headings = [m.strip() for m in part_pattern.findall(cleaned_text)]
    splits = part_pattern.split(cleaned_text)
    
    if len(splits) <= 1:
        # Gate 1 Fallback
        logger.info("Gate 1 Fallback Triggered: Wrapping entire manuscript in a single Part.")
        return [{
            "part_id": "part_p1",
            "title": "Main Narrative",
            "text_block": cleaned_text
        }]
    
    parts = []
    preface = splits[0].strip()
    if preface and len(preface) > 100:
        parts.append({
            "part_id": "part_p0",
            "title": "Preface/Front Matter",
            "text_block": preface
        })
        
    for idx, heading in enumerate(headings):
        text_chunk = splits[idx + 1].strip() if idx + 1 < len(splits) else ""
        if not text_chunk:
            continue
        parts.append({
            "part_id": f"part_p{len(parts) + 1}",
            "title": heading,
            "text_block": text_chunk
        })
        
    return parts

def identify_chapters(part_text: str, part_id: str) -> List[Dict[str, Any]]:
    """
    Loop 2: Chapter Identification (Meso-Router)
    Slices a Part block into distinct Chapters.
    Gate 2 Fallback: Wraps the text in a single Chapter if no chapters exist.
    """
    # Matches standard CHAPTER/ACT/LETTER headings and roman numeral chapter
    # formats like "I--DOWN THE RABBIT-HOLE".
    chapter_pattern = re.compile(
        r'^\s*(?:(?:CHAPTER|CHAPITRE|ACT|LETTER)\s+(?:[IVXLCDM]+|[0-9]+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\b.*|(?:[IVXLCDM]+)(?:--|\s*[-.]\s*).*)$',
        re.IGNORECASE | re.MULTILINE
    )
    
    headings = [m.strip() for m in chapter_pattern.findall(part_text)]
    splits = chapter_pattern.split(part_text)

    if len(splits) <= 1:
        # Bare-roman section headers ("I", "II", "III" alone on a line -- e.g. The
        # Turn of the Screw). Regex alone can't do this safely ("I" is also the
        # pronoun), so require an ascending sequence of >=3 bare-roman lines.
        bare_pattern = re.compile(r'^\s*([IVXLCDM]{1,6})\s*\.?\s*$', re.MULTILINE)
        roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

        def _roman_to_int(s: str) -> int:
            total, prev = 0, 0
            for ch in reversed(s):
                v = roman_values.get(ch, 0)
                total = total - v if v < prev else total + v
                prev = max(prev, v)
            return total

        candidates = [(m.start(), m.group(1)) for m in bare_pattern.finditer(part_text)]
        seq = []
        expected = 1
        for pos, numeral in candidates:
            if _roman_to_int(numeral) == expected:
                seq.append((pos, numeral))
                expected += 1
        if len(seq) >= 3:
            chapters = []
            preface = part_text[:seq[0][0]].strip()
            if preface and len(preface) > 100:
                chapters.append({"chapter_id": f"{part_id}_c0", "title": "Prologue", "text_block": preface})
            for i, (pos, numeral) in enumerate(seq):
                end = seq[i + 1][0] if i + 1 < len(seq) else len(part_text)
                block = part_text[pos:end]
                block = re.sub(r'^\s*[IVXLCDM]{1,6}\s*\.?\s*', '', block, count=1).strip()
                if block:
                    chapters.append({
                        "chapter_id": f"{part_id}_c{len(chapters) + 1}",
                        "title": f"Section {numeral}",
                        "text_block": block,
                    })
            if len(chapters) >= 3:
                logger.info(f"Bare-roman section sequence detected: {len(chapters)} chapters.")
                return chapters

        # Gate 2 Fallback
        return [{
            "chapter_id": f"{part_id}_c1",
            "title": "Chapter 1",
            "text_block": part_text
        }]
        
    chapters = []
    preface = splits[0].strip()
    if preface and len(preface) > 100:
        chapters.append({
            "chapter_id": f"{part_id}_c0",
            "title": "Prologue",
            "text_block": preface
        })
        
    for idx, heading in enumerate(headings):
        text_chunk = splits[idx + 1].strip() if idx + 1 < len(splits) else ""
        # Table-of-contents guard: a TOC entry matching the chapter pattern yields
        # a near-empty block (just the run-up to the next TOC line). Real chapters
        # are never this small.
        if not text_chunk or len(text_chunk) < 200:
            continue
        chapters.append({
            "chapter_id": f"{part_id}_c{len(chapters) + 1}",
            "title": heading,
            "text_block": text_chunk
        })
        
    return chapters

def identify_scenes(chapter_text: str, chapter_id: str, max_chars: int = 15000, is_single_chapter_book: bool = False) -> List[Dict[str, Any]]:
    """
    Loop 3: Scene Identification (Micro-Router / Density Heuristic)
    Looks for explicit breaks and pre-annotated scene markers.
    If absent, applies semantic-proxy transition keywords or the 12-paragraph fallback.
    """
    # Explicit breaks (including optional carriage returns and human-annotated [Scene ...] tags)
    scene_separator = r'(?:\r?\n)\s*(?:\*\s*\*|\#|-{3,}|_{3,}|\[Scene\s+\d+.*?\])\s*(?:\r?\n)'
    splits = [s.strip() for s in re.split(scene_separator, chapter_text) if s.strip()]
    
    if len(splits) > 1:
        scenes = []
        scene_counter = 1
        for block in splits:
            paragraphs = [p.strip() for p in re.split(r'\r?\n\s*\r?\n', block) if p.strip()]
            if len(paragraphs) > 15:
                # Segment long block into logical chunks of 12 paragraphs
                chunk_size = 12
                for i in range(0, len(paragraphs), chunk_size):
                    chunk = paragraphs[i:i + chunk_size]
                    scenes.append({
                        "scene_id": f"{chapter_id}_s{scene_counter}",
                        "text_block": "\n\n".join(chunk),
                        "boundary_source": "marker_then_size_chunk"
                    })
                    scene_counter += 1
            else:
                scenes.append({
                    "scene_id": f"{chapter_id}_s{scene_counter}",
                    "text_block": block,
                    "boundary_source": "explicit_marker"
                })
                scene_counter += 1
        return scenes
        
    # Fallback: Group paragraphs by semantic transition keywords, or fall back to 12-paragraph chunks
    paragraphs = [p.strip() for p in re.split(r'\r?\n\s*\r?\n', chapter_text) if p.strip()]
    
    # If the book has multiple chapters and this chapter block is small enough,
    # keep it as a single scene matching human chapter-centric parsing style.
    if not is_single_chapter_book and len(chapter_text) <= max_chars:
        return [{
            "scene_id": f"{chapter_id}_s1",
            "text_block": chapter_text.strip(),
            "boundary_source": "whole_chapter"
        }]
        
    if len(paragraphs) <= 15:
        return [{
            "scene_id": f"{chapter_id}_s1",
            "text_block": chapter_text.strip(),
            "boundary_source": "whole_chapter"
        }]
        
    logger.info(f"Analyzing narrative transitions in {len(paragraphs)} paragraphs...")
    
    # Heuristic transition word detector: GENERIC temporal/spatial phrases only.
    # (Book-specific phrases -- Peter Rabbit lines like "flopsy, mopsy" -- were
    # removed 2026-07-04: they were overfitting to one gold file and made the
    # measured "scene accuracy" on that book meaningless. Semantic scene
    # segmentation is the Director's Scene Segmenter gate's job, not regex's.)
    transition_pattern = re.compile(
        r'^(?:later|meanwhile|suddenly|the\s+next\s+(?:morning|day|evening)|that\s+(?:evening|night|morning)|one\s+(?:morning|evening|day|night)|during\s+the\s+(?:evening|night)|when\s+(?:he|she|they)\s+(?:awoke|returned)|hours\s+later|some\s+time\s+(?:later|afterwards))',
        re.IGNORECASE
    )
    scenes = []
    current_chunk = []
    scene_counter = 1

    for p in paragraphs:
        is_transition = False
        p_clean = re.sub(r"^['\"`\(\s]+", "", p).strip().lower()
        if transition_pattern.match(p_clean):
            is_transition = True

        # Trigger break if a strong transition keyword matched, or if the chunk has
        # grown beyond a comfortable paragraph count for single-chapter books.
        if (is_transition and len(current_chunk) >= 2) or len(current_chunk) >= 15:
            scenes.append({
                "scene_id": f"{chapter_id}_s{scene_counter}",
                "text_block": "\n\n".join(current_chunk),
                "boundary_source": "transition_heuristic"
            })
            scene_counter += 1
            current_chunk = [p]
        else:
            current_chunk.append(p)
            
    if current_chunk:
        scenes.append({
            "scene_id": f"{chapter_id}_s{scene_counter}",
            "text_block": "\n\n".join(current_chunk),
            "boundary_source": "transition_heuristic"
        })
        
    return scenes

def extract_quotes_safely(paragraph: str) -> List[Dict[str, Any]]:
    """
    Loop 4 (Apostrophe Trap Parser):
    Extracts narrative and dialogue blocks from a single paragraph.
    Uses double quote (") if present, otherwise checks for single quote (') speech tags.
    """
    if not paragraph.strip():
        return []
        
    # Check if straight double quote exists (normalized in clean_front_matter)
    has_double = '"' in paragraph
    
    # If no double quotes are present, check if single quotes are used as speech delimiters.
    # Count single quotes that are NOT contractions (not surrounded by letters).
    non_contraction_single_quotes = 0
    for idx, char in enumerate(paragraph):
        if char == "'":
            prev_char = paragraph[idx - 1] if idx > 0 else ' '
            next_char = paragraph[idx + 1] if idx + 1 < len(paragraph) else ' '
            if not (prev_char.isalpha() and next_char.isalpha()):
                non_contraction_single_quotes += 1
                
    # Single quotes represent dialogue only if we have an even, balanced count
    has_single_speech = (not has_double) and (non_contraction_single_quotes > 0) and (non_contraction_single_quotes % 2 == 0)
    
    if not (has_double or has_single_speech):
        # Pure narrative paragraph
        return [{"type": "narrative", "text": paragraph}]
        
    quote_char = '"' if has_double else "'"
    
    segments = []
    current_buffer = []
    in_quote = False
    
    for idx, char in enumerate(paragraph):
        is_quote = False
        if char == quote_char:
            if quote_char == "'":
                prev_char = paragraph[idx - 1] if idx > 0 else ' '
                next_char = paragraph[idx + 1] if idx + 1 < len(paragraph) else ' '
                if prev_char.isalpha() and next_char.isalpha():
                    is_quote = False  # Contraction
                else:
                    is_quote = True   # Speech boundary
            else:
                is_quote = True       # Double quote boundary
                
        if is_quote:
            text_chunk = "".join(current_buffer).strip()
            if text_chunk:
                segments.append({
                    "type": "dialogue" if in_quote else "narrative",
                    "text": text_chunk
                })
            current_buffer = []
            in_quote = not in_quote
        else:
            current_buffer.append(char)
            
    text_chunk = "".join(current_buffer).strip()
    if text_chunk:
        segments.append({
            "type": "dialogue" if in_quote else "narrative",
            "text": text_chunk
        })
        
    # Clean up whitespace and hyphens
    cleaned_segments = []
    for seg in segments:
        text = re.sub(r'^[\s\-]+', '', seg["text"])
        text = re.sub(r'[\s\-]+$', '', text)
        if text:
            cleaned_segments.append({
                "type": seg["type"],
                "text": text
            })
            
    return cleaned_segments

def parse_tier_1_lines(scene_text: str, part_num: int, chapter_num: int, scene_num: int) -> List[ScriptLine]:
    """Loop 4: Line Parsing & Narrator Attribution (The Handoff)"""
    # Split paragraphs by double newlines to preserve paragraph integrity, then strip
    paragraphs = [p.strip() for p in re.split(r'\n\n+', scene_text) if p.strip()]
    script_lines = []
    line_counter = 1
    
    # Load Narrator drawer config from MemPalace
    pitch_mod = 1.0
    speed_mod = 1.0
    try:
        from src.spatial_memory import MemPalace
        palace = MemPalace()
        drawer = palace.get_character_drawer("Narrator")
        palace.close()
        if drawer:
            pitch_semitones = float(drawer["modulation_config"].get("pitch", 0.0))
            pitch_mod = 2.0 ** (pitch_semitones / 12.0)
            speed_mod = float(drawer["modulation_config"].get("speed", 1.0))
    except Exception:
        pass
        
    performance_config = PerformanceMetrics(
        pitch_modifier=pitch_mod,
        speed_modifier=speed_mod,
        delivery_style="neutral_narrative"
    )
    
    for p in paragraphs:
        # Normalize internal single newlines inside paragraph to space to avoid mid-sentence line wrap splits
        p_normalized = re.sub(r'\s*\n\s*', ' ', p)
        segments = extract_quotes_safely(p_normalized)
        for seg_idx, segment in enumerate(segments):
            # Deterministic line_id based on text content
            raw_id = f"p{part_num}_c{chapter_num}_s{scene_num}_l{line_counter}_{segment['text']}"
            line_id = hashlib.sha256(raw_id.encode('utf-8')).hexdigest()[:16]
            
            # Calculate post-padding based on segment position and punctuation to prevent odd pauses
            is_last_seg = (seg_idx == len(segments) - 1)
            if is_last_seg:
                padding = 600
            else:
                last_char = segment['text'].strip()[-1] if segment['text'].strip() else ''
                if last_char in {'.', '?', '!'}:
                    padding = 200
                else:
                    padding = 50
                    
            line = ScriptLine(
                line_id=line_id,
                chapter=chapter_num,
                scene=scene_num,
                line_number=line_counter,
                character="Narrator",
                speaker_id="char_narrator",
                segment_type=segment["type"],
                text=segment["text"],
                performance=performance_config,
                post_padding_ms=padding
            )
            script_lines.append(line)
            line_counter += 1
            
    return script_lines


def ingest_manuscript_tier_1(file_path: str, chapters: str = None, enable_llm_enrichment: bool = False,
                             resume_enrichment: bool = False) -> ManuscriptManifest:
    """Ingests a raw book text file and outputs a validated ManuscriptManifest.

    enable_llm_enrichment (default False) opts into a post-Loop-4 LLM enrichment pass
    (see enrich_scene_lines_with_llm) that upgrades speaker attribution/emotion beyond
    the flat "Narrator" default. When False, this function makes zero network calls and
    never imports src.llm_client -- Tier 1's zero-cost/offline guarantee is unaffected.

    resume_enrichment (with enable_llm_enrichment): free-tier daily quotas can starve
    the back half of a long book, leaving whole scenes on "Tier 1 Default". Resume
    reuses every previously-enriched scene from the existing artifacts (matching by
    scene text, so structure drift safely falls through to fresh work) and spends LLM
    calls ONLY on scenes that never got attribution -- a multi-day book finishes
    itself across quota windows instead of re-paying for everything.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    from src import progress as _progress
    _book = os.path.splitext(os.path.basename(file_path))[0]
    _progress.clear(_book)
    logger.info(f"Starting Tier 1 Ingestion for: {file_path}")
    print(f"\n--- Starting Ingestion Pipeline for: {os.path.basename(file_path)} ---")
    if file_path.lower().endswith(".epub"):
        # EPUB spine -> CHAPTER-marked plain text; the existing loops (and the
        # scrubber -- Gutenberg EPUBs carry the license as spine items) take
        # over from there unchanged.
        from nlp_engine.epub_ingestion import epub_to_text
        raw_text = epub_to_text(file_path)
        print(f"  [EPUB] Converted spine to {raw_text.count(chr(10)+chr(10)+chr(10)) + 1} chapter block(s).")
    elif file_path.lower().endswith(".docx"):
        with open(file_path, "rb") as f:
            raw_text = _docx_to_text(f.read())
        print("  [DOCX] Extracted UTF-8 text from word/document.xml.")
    else:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
        # Binary-input guard (measured failure: a mis-uploaded EPUB saved as .txt
        # was ingested as ZIP garbage and became "scenes" of binary noise). A
        # manuscript is overwhelmingly printable text; anything else is refused
        # loudly instead of narrated.
        sample = raw_text[:20000]
        if sample:
            printable = sum(1 for ch in sample if ch.isprintable() or ch in "\n\r\t")
            if printable / len(sample) < 0.9:
                raise ValueError(
                    f"'{os.path.basename(file_path)}' does not look like a text manuscript "
                    f"({100 * printable / len(sample):.0f}% printable). If this is an EPUB, "
                    "upload it with its .epub extension so the spine-aware ingestion handles it.")
        
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # Namespaced under tier1/ -- src/looped_analyzer.py writes its own (differently
    # loop-numbered) artifacts to the same {book}/ directory under looped_analyzer/,
    # keeping the two independent pipelines' outputs from colliding or being confused.
    pipeline_dir = os.path.join("data/corpus/pipeline", base_name, "tier1")
    os.makedirs(pipeline_dir, exist_ok=True)

    selected_chapters = None
    if chapters:
        try:
            selected_chapters = set()
            for part in chapters.split(','):
                if '-' in part:
                    start, end = part.split('-')
                    selected_chapters.update(range(int(start), int(end) + 1))
                else:
                    selected_chapters.add(int(part))
            logger.info(f"Ingestion filter active: targeting chapters {selected_chapters}")
        except Exception as e:
            logger.error(f"Failed to parse chapters filter '{chapters}': {e}")

    # Play-format detector: dramas (ACT/SCENE structure, SPEAKER.-prefixed lines,
    # unquoted dialogue) break every prose assumption in loops 3-4 -- quote-based
    # dialogue extraction finds nothing and everything becomes "Narrator". Warn
    # loudly rather than silently produce garbage; a dedicated play parser is a
    # scoped work item (plays actually make attribution DETERMINISTIC).
    speaker_prefix_lines = len(re.findall(r'^[A-Z][A-Z ]{2,24}\.\s*$', raw_text[:60000], re.MULTILINE))
    has_acts = bool(re.search(r'^ACT [IVX]+', raw_text[:60000], re.MULTILINE))
    if has_acts and speaker_prefix_lines >= 10:
        logger.warning(
            f"PLAY FORMAT DETECTED ({speaker_prefix_lines} speaker-prefix lines + ACT structure): "
            "prose loops will misparse this text (dialogue is unquoted). "
            "Proceeding, but output quality will be poor until the play parser exists."
        )
        print("  !! WARNING: play/drama format detected -- prose pipeline will misparse this text.")

    print("Loop 1: Extracting Illustrations & Normalizing Front Matter...")
    # Extract [Illustration] tags (User Request)
    illustration_pattern = re.compile(r'\[Illustration.*?\]', re.IGNORECASE | re.DOTALL)
    illustrations = []
    for match in illustration_pattern.finditer(raw_text):
        illustrations.append({
            "matched_text": match.group(0),
            "start_index": match.start(),
            "end_index": match.end()
        })

    # Clean raw_text of [Illustration] tags by replacing them with double newlines
    cleaned_raw_text = illustration_pattern.sub('\n\n', raw_text)

    parts = identify_parts(cleaned_raw_text)
    print(f"  -> Found {len(parts)} parts in Loop 1.")

    # G1 gate: implausible top-level structure gets LLM verification (verbatim-
    # grounded). Triggers: regex runaway (>20 parts -- nested VOLUME/BOOK headings
    # matching as siblings) or one monolithic part in a very large book.
    if enable_llm_enrichment and (len(parts) > 20 or (len(parts) == 1 and len(cleaned_raw_text) > 300000)):
        logger.info(f"G1 trigger: {len(parts)} parts looks implausible.")
        verified_parts = verify_parts_with_llm(parts[0]["text_block"] if len(parts) == 1 else cleaned_raw_text, len(parts))
        if verified_parts:
            print(f"  -> [G1] Part Verifier re-split: {len(parts)} -> {len(verified_parts)} parts.")
            parts = verified_parts

    global_roster: List[str] = []
    if enable_llm_enrichment:
        # Book-level roster built from front-matter-cleaned part text, so characters
        # named early remain attribution candidates in scenes that reference them
        # only by pronoun/epithet.
        global_roster = _extract_global_roster("\n\n".join(p["text_block"] for p in parts))
        print(f"  -> Global character roster candidates: {global_roster}")
    
    # Save Loop 1 artifacts
    with open(os.path.join(pipeline_dir, "loop1_parts.json"), "w", encoding="utf-8") as f:
        json.dump(parts, f, indent=4)
    with open(os.path.join(pipeline_dir, "loop1_illustrations.json"), "w", encoding="utf-8") as f:
        json.dump(illustrations, f, indent=4)

    print("Loop 2: Identifying Chapters (Meso-Router)...")
    # Pre-calculate total chapters across all parts to determine if this is a single-chapter book
    total_chapters_detected = 0
    chapters_by_part = []
    for part in parts:
        chaps = identify_chapters(part["text_block"], part["part_id"])
        chapters_by_part.append(chaps)
        total_chapters_detected += len(chaps)
        
    is_single_chapter_book = (total_chapters_detected <= 1)
    print(f"  -> Found {total_chapters_detected} chapters in Loop 2.")

    # G2 gate: chapter detection that looks implausible gets an LLM verification
    # pass (verbatim-grounded). Triggers: a single huge chapter, or extreme size
    # nonuniformity (one "chapter" 8x the median suggests missed boundaries).
    if enable_llm_enrichment:
        for p_i, chaps in enumerate(chapters_by_part):
            part_block = parts[p_i]["text_block"]
            sizes = sorted(len(ch["text_block"]) for ch in chaps)
            median = sizes[len(sizes) // 2] if sizes else 0
            implausible = (
                (len(chaps) == 1 and len(part_block) > 80000)
                or (len(chaps) >= 2 and median > 0 and sizes[-1] > 8 * median)
            )
            if implausible:
                logger.info(f"G2 trigger: part {parts[p_i]['part_id']} chapter structure implausible ({len(chaps)} chapters).")
                verified = verify_chapters_with_llm(part_block, parts[p_i]["part_id"], len(chaps))
                if verified:
                    print(f"     [G2] Chapter Verifier re-split part {p_i + 1}: {len(chaps)} -> {len(verified)} chapters.")
                    chapters_by_part[p_i] = verified
        total_chapters_detected = sum(len(c) for c in chapters_by_part)
        is_single_chapter_book = (total_chapters_detected <= 1)

    print("Loop 3 & Loop 4: Segmenting Scenes and Parsing Script Lines...")
    part_payloads = []
    total_chapters = 0
    total_scenes = 0
    
    all_loop2_chapters = []
    all_loop3_scenes = []
    all_loop4_lines = []
    all_loop4_lines_enriched = []
    all_llm_cleancheck_issues = []
    all_llm_sfx_cues = []

    # Resume: index the previous run's artifacts. Reuse is text-keyed, never
    # id-keyed alone -- if the deterministic structure drifted (e.g. a scrubber
    # fix changed scene boundaries), stale entries simply won't match and those
    # scenes re-enrich fresh.
    prev_enriched_by_id: Dict[str, Dict[str, Any]] = {}
    prev_scenes_by_chapter: Dict[str, List[Dict[str, Any]]] = {}
    prev_sfx_by_id: Dict[str, Any] = {}
    prev_clean_by_id: Dict[str, Any] = {}
    resumed_scenes = 0
    if enable_llm_enrichment and resume_enrichment:
        def _load_prev(name):
            p = os.path.join(pipeline_dir, name)
            if os.path.exists(p):
                try:
                    with open(p, encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Resume: unreadable {name} ({e}); ignoring.")
            return []
        for entry in _load_prev("loop4_lines_enriched.json"):
            prev_enriched_by_id[entry.get("scene_id", "")] = entry
        for sc in _load_prev("loop3_scenes.json"):
            chap_key = sc.get("scene_id", "").rsplit("_s", 1)[0]
            prev_scenes_by_chapter.setdefault(chap_key, []).append(sc)
        for entry in _load_prev("loopE_llm_sfx_cues.json"):
            prev_sfx_by_id[entry.get("scene_id", "")] = entry
        for entry in _load_prev("loopE_llm_cleancheck.json"):
            prev_clean_by_id[entry.get("scene_id", "")] = entry
        logger.info(f"Resume: {len(prev_enriched_by_id)} previously-enriched scene(s) indexed.")

    def _reusable_scene(scene_id: str, current_lines: List[ScriptLine]) -> Optional[List[ScriptLine]]:
        """Previous enrichment is reusable when the scene's line TEXTS match
        exactly and its dialogue actually got attributed (or it has none)."""
        prev = prev_enriched_by_id.get(scene_id)
        if not prev or len(prev.get("lines", [])) != len(current_lines):
            return None
        if [l.get("text") for l in prev["lines"]] != [l.text for l in current_lines]:
            return None
        dialogue = [l for l in prev["lines"] if l.get("segment_type") == "dialogue"]
        enriched = [l for l in dialogue if str(l.get("attribution_method")) != "Tier 1 Default"]
        if dialogue and not enriched:
            return None  # this is exactly the quota-starved case resume exists to redo
        try:
            return [ScriptLine.model_validate(l) for l in prev["lines"]]
        except Exception as e:
            logger.warning(f"Resume: previous lines for {scene_id} failed validation ({e}); re-enriching.")
            return None

    # Scene omits (console structure control): a human can exclude any scene
    # from the audiobook. Artifacts stay complete -- the omit applies only to
    # the manifest (and skips enrichment spend on the omitted scene).
    scene_omits: Dict[str, Any] = {}
    omit_path = os.path.join(pipeline_dir, "scene_overrides.json")
    if os.path.exists(omit_path):
        try:
            with open(omit_path, encoding="utf-8") as f:
                scene_omits = {k: v for k, v in json.load(f).items() if v.get("omit")}
            if scene_omits:
                logger.info(f"Scene overrides: {len(scene_omits)} scene(s) omitted from the manifest.")
        except Exception as e:
            logger.warning(f"Scene overrides unreadable ({e}); proceeding without.")

    for p_idx, part in enumerate(parts):
        part_id = part["part_id"]
        part_title = part["title"]
        part_block = part["text_block"]
        
        chapters = chapters_by_part[p_idx]
        all_loop2_chapters.extend(chapters)
        chapter_payloads = []
        
        for c_idx, chap in enumerate(chapters):
            total_chapters += 1
            if selected_chapters and total_chapters not in selected_chapters:
                continue
            chap_id = chap["chapter_id"]
            chap_title = chap["title"]
            chap_block = chap["text_block"]
            
            print(f"  -> Chapter {total_chapters}/{total_chapters_detected}: '{chap_title}'")
            _progress.report(_book, "tier1_structure", total_chapters, total_chapters_detected, chap_title[:40])
            scenes = identify_scenes(chap_block, chap_id, is_single_chapter_book=is_single_chapter_book)
            print(f"     [Loop 3] Found {len(scenes)} scenes.")

            # G4 gate: when no explicit typographic markers produced these scene
            # boundaries (measured: true for 100% of the current corpus), the
            # Director's Scene Segmenter re-segments semantically. Deterministic
            # scenes stand if the gate AI fails or finds nothing better.
            if enable_llm_enrichment and all(
                s.get("boundary_source") in ("transition_heuristic", "whole_chapter", "marker_then_size_chunk")
                for s in scenes
            ):
                # Resume: reuse the previous run's boundaries for this chapter when
                # every previous scene text still exists verbatim in the chapter --
                # re-running the director could draw DIFFERENT boundaries, which
                # would orphan the enriched scenes we're trying to keep.
                prev_chapter_scenes = prev_scenes_by_chapter.get(chap_id, [])
                if prev_chapter_scenes and all(
                    sc.get("text_block") and sc["text_block"] in chap_block
                    for sc in prev_chapter_scenes
                ):
                    scenes = prev_chapter_scenes
                    print(f"     [Resume] Reusing previous run's {len(scenes)} scene boundaries for {chap_id}.")
                else:
                    _progress.report(_book, "g4_scene_segmentation", total_chapters, total_chapters_detected, chap_title[:40])
                    try:
                        ai_scenes = director_segment_chapter(chap_block, chap_id)
                        if ai_scenes:
                            print(f"     [G4] Director's Scene Segmenter re-segmented: {len(scenes)} -> {len(ai_scenes)} scenes.")
                            scenes = ai_scenes
                    except Exception as e:
                        logger.warning(f"G4 gate failed for {chap_id}, keeping deterministic scenes: {e}")
            all_loop3_scenes.extend(scenes)
            scene_payloads = []
            
            for s_idx, scene in enumerate(scenes):
                total_scenes += 1
                scene_id = scene["scene_id"]
                scene_block = scene["text_block"]
                
                lines = parse_tier_1_lines(
                    scene_text=scene_block,
                    part_num=p_idx + 1,
                    chapter_num=total_chapters,
                    scene_num=total_scenes
                )
                print(f"     [Loop 4] Scene {total_scenes} -> Extracted {len(lines)} speech/narrative lines.")

                all_loop4_lines.append({
                    "scene_id": scene_id,
                    "lines": [line.model_dump() for line in lines]
                })

                if scene_id in scene_omits:
                    print(f"     [Omit] Scene {scene_id} excluded from manifest by console override.")
                    if enable_llm_enrichment:
                        # keep the enriched artifact complete (un-enriched lines)
                        all_loop4_lines_enriched.append({
                            "scene_id": scene_id,
                            "lines": [line.model_dump() for line in lines]})
                    continue
                if enable_llm_enrichment:
                    reused = _reusable_scene(scene_id, lines) if resume_enrichment else None
                    if reused is not None:
                        lines = reused
                        resumed_scenes += 1
                        if scene_id in prev_clean_by_id:
                            all_llm_cleancheck_issues.append(prev_clean_by_id[scene_id])
                        if scene_id in prev_sfx_by_id:
                            all_llm_sfx_cues.append(prev_sfx_by_id[scene_id])
                    else:
                        _progress.report(_book, "tier2_enrichment", total_scenes, max(total_scenes, len(scenes) * total_chapters_detected), scene_id)
                        try:
                            enriched_lines, clean_issues, sfx_cues = enrich_scene_lines_with_llm(scene_block, lines, global_roster=global_roster)
                            if clean_issues:
                                all_llm_cleancheck_issues.append({
                                    "scene_id": scene_id,
                                    "issues": clean_issues
                                })
                            if sfx_cues:
                                all_llm_sfx_cues.append({
                                    "scene_id": scene_id,
                                    "sfx_cues": sfx_cues
                                })
                            lines = enriched_lines
                        except Exception as e:
                            logger.warning(f"LLM enrichment failed for scene {scene_id}, keeping Tier 1 defaults: {e}")
                    all_loop4_lines_enriched.append({
                        "scene_id": scene_id,
                        "lines": [line.model_dump() for line in lines]
                    })

                scene_payloads.append(ScenePayload(
                    scene_id=scene_id,
                    lines=lines
                ))
                
            chapter_payloads.append(ChapterPayload(
                chapter_id=chap_id,
                title=chap_title,
                scenes=scene_payloads
            ))
            
        if chapter_payloads:
            part_payloads.append(PartPayload(
                part_id=part_id,
                title=part_title,
                chapters=chapter_payloads
            ))
        
    if enable_llm_enrichment and resume_enrichment:
        fresh = total_scenes - resumed_scenes
        print(f"  [Resume] Reused {resumed_scenes} enriched scene(s); freshly enriched {fresh}.")
        logger.info(f"Resume summary: reused {resumed_scenes}, fresh {fresh} of {total_scenes} scenes.")

    # Book-level alias merge: collapse title/disguise/partial-name duplicates
    # ("Count Von Kramm" / "Majesty" / "King" -> one canonical identity) so a
    # single character doesn't get multiple voices at synthesis time.
    alias_groups: List[Dict[str, Any]] = []
    if enable_llm_enrichment:
        alias_mapping, alias_groups = merge_speaker_aliases(part_payloads, roster=global_roster)
        if alias_mapping:
            for part in part_payloads:
                for chapter in part.chapters:
                    for scene in chapter.scenes:
                        for i, line in enumerate(scene.lines):
                            canonical = alias_mapping.get(line.character)
                            if canonical:
                                scene.lines[i] = line.model_copy(update={
                                    "character": canonical,
                                    "speaker_id": f"char_{canonical.lower().replace(' ', '_')}",
                                })
            # Regenerate the enriched artifact post-merge so it reflects final identities
            all_loop4_lines_enriched = [
                {"scene_id": scene.scene_id, "lines": [line.model_dump() for line in scene.lines]}
                for part in part_payloads
                for chapter in part.chapters
                for scene in chapter.scenes
            ]

    # Human speaker overrides (console Phase 2): applied AFTER attribution,
    # review, and alias merge -- the human's word is final and survives re-runs
    # (line_id is a content hash, so an override orphans harmlessly if the
    # line's text changes). Same durable-veto pattern as confirmed_merges.
    #
    # LAYERING RULE (found by test): overrides are applied to the MANIFEST only,
    # never baked into loop4_lines_enriched.json. The artifact stays the
    # attribution layer's pure output -- it is resume's reuse source and the
    # console overlays overrides at serve time; baking them in made clearing an
    # override impossible (the resumed artifact kept the stale correction).
    overrides_path = os.path.join(pipeline_dir, "speaker_overrides.json")
    if enable_llm_enrichment and os.path.exists(overrides_path):
        try:
            with open(overrides_path, encoding="utf-8") as f:
                speaker_overrides = json.load(f)
            applied = 0
            for part in part_payloads:
                for chapter in part.chapters:
                    for scene in chapter.scenes:
                        for i, line in enumerate(scene.lines):
                            ov = speaker_overrides.get(line.line_id)
                            if ov and ov.get("character") and ov["character"] != line.character:
                                scene.lines[i] = line.model_copy(update={
                                    "character": ov["character"],
                                    "speaker_id": f"char_{ov['character'].lower().replace(' ', '_')}",
                                    "attribution_method": "human_override",
                                    "confidence": 1.0,
                                    "speaker_locked": True,
                                })
                                applied += 1
            if applied:
                print(f"  [Overrides] Applied {applied} human speaker correction(s) to the manifest.")
                logger.info(f"Applied {applied} human speaker override(s) from {overrides_path}.")
        except Exception as e:
            logger.warning(f"Speaker overrides unreadable ({e}); proceeding without.")

    # Save Loop 2-4 artifacts
    with open(os.path.join(pipeline_dir, "loop2_chapters.json"), "w", encoding="utf-8") as f:
        json.dump(all_loop2_chapters, f, indent=4)
        
    with open(os.path.join(pipeline_dir, "loop3_scenes.json"), "w", encoding="utf-8") as f:
        json.dump(all_loop3_scenes, f, indent=4)
        
    with open(os.path.join(pipeline_dir, "loop4_lines.json"), "w", encoding="utf-8") as f:
        json.dump(all_loop4_lines, f, indent=4)

    if enable_llm_enrichment:
        with open(os.path.join(pipeline_dir, "loop4_lines_enriched.json"), "w", encoding="utf-8") as f:
            json.dump(all_loop4_lines_enriched, f, indent=4)
        with open(os.path.join(pipeline_dir, "loopE_llm_cleancheck.json"), "w", encoding="utf-8") as f:
            json.dump(all_llm_cleancheck_issues, f, indent=4)
        with open(os.path.join(pipeline_dir, "loopE_llm_sfx_cues.json"), "w", encoding="utf-8") as f:
            json.dump(all_llm_sfx_cues, f, indent=4)
        with open(os.path.join(pipeline_dir, "loopE_llm_alias_merges.json"), "w", encoding="utf-8") as f:
            json.dump(alias_groups, f, indent=4)

    materialize_book_structure_from_tier1(
        pipeline_dir,
        source_file=file_path,
        source_format=os.path.splitext(file_path)[1].lstrip(".").lower() or "txt",
        book_id=base_name,
    )

    manifest = ManuscriptManifest(
        source_file=os.path.basename(file_path),
        total_parts=len(part_payloads),
        total_chapters=len(all_loop2_chapters) if not selected_chapters else len(selected_chapters),
        total_scenes=total_scenes,
        parts=part_payloads
    )
    
    return manifest


def main():
    """CLI and Stress Testing tool."""
    import argparse
    parser = argparse.ArgumentParser(description="Caldera Engine Tier 1 Ingestion Pipeline")
    parser.add_argument("--input", type=str, help="Path to input raw manuscript text file")
    parser.add_argument("--input-dir", type=str, help="Directory of .txt manuscripts to batch-process (empty files skipped)")
    parser.add_argument("--output", type=str, help="Path to save validated JSON manifest (or output directory in --input-dir mode; default scratch/corpus_analysis)")
    parser.add_argument("--stress-test", action="store_true", help="Run the Public Domain Stress Test on the three corpus books")
    parser.add_argument("--enable-llm-enrichment", action="store_true", help="Opt-in: enrich Tier 1 output with cloud/local LLM speaker attribution (Gemini free tier -> Groq -> Ollama -> off). Not applied during --stress-test, which stays zero-LLM as a stable regression gate.")
    parser.add_argument("--resume-enrichment", action="store_true", help="With --enable-llm-enrichment: reuse previously-enriched scenes from the existing pipeline artifacts and spend LLM calls only on scenes still on Tier 1 defaults (quota-starved runs finish across days)")

    args = parser.parse_args()

    # T2-3: attribute direct CLI runs in the usage meter too (owner "local").
    if args.input and args.enable_llm_enrichment:
        try:
            from src.llm_client import set_usage_context
            set_usage_context(book=os.path.splitext(os.path.basename(args.input))[0], owner="local")
        except Exception:
            pass

    if args.stress_test:
        print("\n=== RUNNING TIER 1 PUBLIC DOMAIN STRESS TEST (DoD) ===")
        books = [
            "data/corpus/TheTaleofPeterRabbit.txt",
            "data/corpus/WutheringHeights.txt",
            "data/corpus/LesMiserables.txt"
        ]
        
        success_count = 0
        for book in books:
            start_time = time.time()
            print(f"\nIngesting: {book}...")
            try:
                manifest = ingest_manuscript_tier_1(book)
                elapsed = time.time() - start_time
                print(f"--> Success! Ingestion time: {elapsed:.2f} seconds")
                print(f"    Parts: {manifest.total_parts} | Chapters: {manifest.total_chapters} | Scenes: {manifest.total_scenes}")
                
                # Sample lines verification
                total_lines = sum(
                    len(scene.lines)
                    for part in manifest.parts
                    for chapter in part.chapters
                    for scene in chapter.scenes
                )
                print(f"    Total script lines: {total_lines}")
                
                # Check for schema validation correctness by dumping to dict
                manifest_dict = manifest.model_dump()
                
                # Save first scene sample check
                sample_file = f"scratch/stress_test_sample_{os.path.basename(book)}.json"
                os.makedirs("scratch", exist_ok=True)
                with open(sample_file, "w", encoding="utf-8") as fs:
                    # Write first 50 lines of the first scene as validation
                    json.dump(manifest_dict["parts"][0]["chapters"][0]["scenes"][0]["lines"][:10], fs, indent=4)
                print(f"    Sample lines written to: {sample_file}")
                
                success_count += 1
            except Exception as e:
                print(f"--> FAILED: {book} | Error: {e}")
                logger.error(f"Failed ingesting {book}", exc_info=True)
                
        print(f"\n=== STRESS TEST COMPLETED: {success_count}/{len(books)} PASSED ===")
        sys.exit(0 if success_count == len(books) else 1)
        
    if args.input_dir:
        import glob as _glob
        out_dir = args.output or "scratch/corpus_analysis"
        os.makedirs(out_dir, exist_ok=True)
        txt_files = sorted(_glob.glob(os.path.join(args.input_dir, "*.txt")))
        summary = []
        for path in txt_files:
            base = os.path.splitext(os.path.basename(path))[0]
            if os.path.getsize(path) == 0:
                print(f"SKIP (empty file): {base}")
                summary.append({"book": base, "status": "skipped_empty"})
                continue
            print(f"\n{'='*60}\nBATCH: {base}\n{'='*60}")
            start_time = time.time()
            try:
                manifest = ingest_manuscript_tier_1(path, enable_llm_enrichment=args.enable_llm_enrichment, resume_enrichment=args.resume_enrichment)
                out_path = os.path.join(out_dir, f"{base}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(manifest.model_dump_json(indent=4))
                speakers: Dict[str, int] = {}
                total_lines = 0
                for part in manifest.parts:
                    for chapter in part.chapters:
                        for scene in chapter.scenes:
                            for line in scene.lines:
                                total_lines += 1
                                if line.segment_type == "dialogue":
                                    speakers[line.character] = speakers.get(line.character, 0) + 1
                summary.append({
                    "book": base,
                    "status": "ok",
                    "elapsed_sec": round(time.time() - start_time, 1),
                    "chapters": manifest.total_chapters,
                    "scenes": manifest.total_scenes,
                    "total_lines": total_lines,
                    "dialogue_lines": sum(speakers.values()),
                    "speakers": dict(sorted(speakers.items(), key=lambda kv: -kv[1])),
                    "manifest": out_path,
                })
            except Exception as e:
                logger.error(f"Batch ingestion failed for {base}: {e}", exc_info=True)
                summary.append({"book": base, "status": "failed", "error": str(e)})

        summary_path = os.path.join(out_dir, "batch_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        print(f"\n{'='*60}\nBATCH SUMMARY ({summary_path})\n{'='*60}")
        for row in summary:
            if row["status"] == "ok":
                top = ", ".join(f"{k}({v})" for k, v in list(row["speakers"].items())[:5])
                print(f"  OK   {row['book']}: {row['chapters']} ch / {row['scenes']} sc / {row['dialogue_lines']} dlg lines in {row['elapsed_sec']}s | top: {top}")
            else:
                print(f"  {row['status'].upper():4s} {row['book']}: {row.get('error', '')}")
        failed = sum(1 for r in summary if r["status"] == "failed")
        sys.exit(1 if failed else 0)

    if not args.input or not args.output:
        parser.print_help()
        sys.exit(1)
        
    try:
        manifest = ingest_manuscript_tier_1(args.input, enable_llm_enrichment=args.enable_llm_enrichment, resume_enrichment=args.resume_enrichment)
        # Write validated JSON manifest
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=4))
        logger.info(f"Manifest successfully written to: {args.output}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline parsing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
