#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Caldera Engine Scene Director (Layer 3: Production Direction)

Consumes a Tier 1 enriched ManuscriptManifest (Chain A output) plus the per-book
tier1 pipeline artifacts, and produces per-scene production direction:
music (mood/style/stingers), scene environment, and delivery notes -- then builds
generation-ready prompts (MusicGen/AudioGen/image models) and syncs the whole
production knowledge record into MemPalace.

Outputs (data/corpus/pipeline/{book}/tier3/):
  production_script.json   machine-readable full direction
  production_script.txt    human-readable, same format as HumanProcessed Tier 3 gold
  generation_prompts.json  per-scene prompts for downstream media generation (Chain C/E)

See docs/Caldera Engine Production Knowledge & Media Generation Roadmap.md.
"""

import os
import re
import sys
import json
import logging
import argparse
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import ManuscriptManifest
from src.llm_client import query_llm_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SceneDirector")


# ====================================================
# Direction Schemas (validated LLM output)
# ====================================================

class StingerCue(BaseModel):
    after_line_index: int = Field(..., description="Index of the line after which the stinger plays")
    description: str = Field(..., description="Musical stinger, e.g. 'a brief, dark cello note'")
    trigger_text: str = Field(..., description="Verbatim snippet from the scene that motivates the stinger")


class MusicEvent(BaseModel):
    after_line_index: int = Field(..., description="Line index after which the event fires")
    action: str = Field(..., description="'stop' (music cuts to silence), 'resume' (bed returns), or 'change' (bed transforms to a new style)")
    new_style: str = Field(default="", description="For 'change'/'resume': the style the bed becomes")
    trigger_text: str = Field(..., description="Verbatim snippet motivating the event")


class MusicDirection(BaseModel):
    base_mood: str = Field(..., description="e.g. gentle, tense, mischievous, mournful")
    style: str = Field(..., description="e.g. whimsical orchestral with flutes and acoustic guitar")
    stingers: List[StingerCue] = Field(default_factory=list)
    events: List[MusicEvent] = Field(default_factory=list)


class SceneEnvironment(BaseModel):
    location: str
    time_of_day: str = Field(default="unknown")
    weather: str = Field(default="unknown")
    physical_confines: str = Field(default="open")
    ambient_noise_level: str = Field(default="quiet")


class DeliveryNote(BaseModel):
    index: int
    note: str = Field(..., description="Short parenthetical acting direction, e.g. 'Warm but stern'")


class SceneDirectionSchema(BaseModel):
    music: MusicDirection
    environment: SceneEnvironment
    delivery_notes: List[DeliveryNote] = Field(default_factory=list)


def _norm(t: str) -> str:
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", t.strip().lower())


# ====================================================
# Sound Design Schemas (layered decomposition)
# ====================================================

class SoundLayer(BaseModel):
    component: str = Field(..., description="One concrete generatable sound, e.g. 'ceramic flower pot scraping on wooden shelf'")
    timing: str = Field(default="start", description="'start' | 'overlap' | 'tail' -- position within the composite")
    level: str = Field(default="medium", description="'prominent' | 'medium' | 'subtle'")


class CompositeSoundEvent(BaseModel):
    name: str = Field(..., description="Short name, e.g. 'McGregor overturning flower-pots'")
    anchor_line_index: int = Field(..., description="Line index this sound accompanies")
    source_text: str = Field(..., description="Verbatim scene snippet motivating this sound")
    category: str = Field(..., description="'action' | 'creature' | 'environment'")
    emotional_intent: str = Field(default="", description="For creature sounds: the emotion conveyed, e.g. 'urgent, imploring'")
    layers: List[SoundLayer] = Field(..., description="1-4 component layers mixed into one composite sound")


class SceneSoundDesign(BaseModel):
    continuous_ambience: List[str] = Field(default_factory=list, description="Ongoing background sound components for the whole scene")
    events: List[CompositeSoundEvent] = Field(default_factory=list)


# Narrated-action foley lexicon (from the user's Tier 3 trailer listening review:
# "when they enter the front door of the manor, we should hear the door" -- the
# Sound Designer caught explicit sound-words but missed action-implied foley).
# Each entry: (regex over NARRATIVE text, suggested foley idea handed to AI-7).
_ACTION_FOLEY_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(?:open(?:ed|ing)?|closed?|closing|shut|slam\w*)\b[^.]{0,40}\bdoors?\b", "door opening/closing (latch, hinge, frame)"),
    (r"\bdoors?\b[^.]{0,40}\b(?:open(?:ed|ing)?|closed?|shut|slam\w*)\b", "door opening/closing (latch, hinge, frame)"),
    (r"\b(?:knock\w*|tap\w*|rapp?\w*)\b[^.]{0,30}\bdoor\b", "knuckles rapping on a wooden door"),
    (r"\brang\b[^.]{0,30}\bbell\b|\bbell\b[^.]{0,30}\brang\b", "a hand bell or doorbell ringing"),
    (r"\benter(?:ed|ing)\b|\bcame? in(?:to)?\b|\bushered\b|\bshown (?:in|into)\b", "door + footsteps entering a room"),
    (r"\b(?:left|leaving|departed|withdrew|hurried) (?:the )?(?:room|house|office|chamber)\b", "footsteps leaving + door"),
    (r"\bfootsteps?\b|\bpaced?\b|\bpacing\b|\bcrossed the room\b|\bwalk(?:ed|ing) (?:across|over|to)\b", "footsteps on the room's floor surface"),
    (r"\b(?:sat|seated) (?:down|himself|herself)\b|\bsank into\b[^.]{0,25}\b(?:chair|seat|sofa)\b", "chair creak and clothing rustle, sitting"),
    (r"\brose\b[^.]{0,20}\b(?:from|to his feet|to her feet)\b|\bstood up\b|\bsprang (?:up|to)\b", "chair shift and fabric movement, standing"),
    (r"\b(?:climb|ascend|descend)\w*\b[^.]{0,25}\bstair", "footsteps on stairs"),
    (r"\bstruck a match\b|\blit (?:a|his|her|the) (?:pipe|candle|lamp|cigarette)\b", "match strike and flare"),
    (r"\bpour(?:ed|ing)\b[^.]{0,30}\b(?:tea|wine|water|brandy|whisky|coffee)\b", "liquid pouring into a cup or glass"),
    (r"\b(?:unfolded|folded|tore|crumpled|opened)\b[^.]{0,30}\b(?:letter|paper|note|envelope|telegram)\b", "paper being handled"),
    (r"\b(?:carriage|cab|hansom|wheels)\b[^.]{0,40}\b(?:drew up|rattl\w*|arriv\w*|stopp\w*|rolled)\b", "carriage wheels and hooves on cobblestones"),
    (r"\bwindow\b[^.]{0,35}\b(?:open\w*|closed?|shut|threw|raised)\b", "sash window sliding open or closed"),
    (r"\bfire\b[^.]{0,25}\b(?:crackl\w*|blaz\w*|burn\w*)\b|\bpoker\w*\b[^.]{0,20}\bfire\b", "fireplace crackle and a stirred grate"),
]


def _spot_action_foley(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministic pre-pass over NARRATIVE lines: physical actions the narration
    walks through are implied foley even without an onomatopoeia. Zero LLM cost;
    grounding is free because the anchor sentence IS verbatim line text. AI-7
    directs (or prunes) the candidates -- the spotter proposes, never decides."""
    candidates = []
    seen = set()
    for i, l in enumerate(lines):
        if l.get("segment_type") == "dialogue":
            continue  # dialogue mentions ("I banged the door") are recollections, not scene foley
        text = l.get("text", "")
        for pattern, idea in _ACTION_FOLEY_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if not m:
                continue
            sent_start = text.rfind(".", 0, m.start()) + 1
            sent_end = text.find(".", m.end())
            sentence = text[sent_start:sent_end + 1 if sent_end != -1 else len(text)].strip()
            key = (i, idea)
            if key in seen or not sentence:
                continue
            seen.add(key)
            candidates.append({"anchor_line_index": i, "source_text": sentence[:160], "suggestion": idea})
    return candidates[:8]


def design_scene_sound(scene_id: str, scene_text: str, lines: List[Dict[str, Any]], spotting: Optional[Dict[str, Any]] = None, bible: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Layered sound decomposition for one scene: continuous ambience components plus
    composite (multi-layer) sound events, including emotionally-directed creature
    sounds. Grounded: every event's source_text must exist verbatim in the scene."""
    line_listing = "\n".join(f'{i}: [{l["character"]}] "{l["text"][:80]}"' for i, l in enumerate(lines))
    action_candidates = _spot_action_foley(lines)
    action_listing = "\n".join(
        f'- line {c["anchor_line_index"]}: "{c["source_text"]}" -> consider: {c["suggestion"]}'
        for c in action_candidates) or "(none detected)"

    spotting = spotting or {}
    prompt = f"""You are the SOUND DESIGNER for a Graphic-Audio-style full-cast audiobook production.
Your craft: foley and effects. The spotting artist has marked the sound moments -- design against
those marks first.
{_bible_block(bible)} Think like a foley artist and field recordist: most real sound moments are
LAYERED composites of 2-4 component sounds.

SCENE TEXT:
{scene_text[:4000]}

LINES (index: [speaker] "text"):
{line_listing}

SPOTTED SOUND MOMENTS (from the spotting session):
{_moments_listing(spotting.get("sound_moments", []))}

NARRATED PHYSICAL ACTIONS (detected deterministically -- when the narration walks a
character through a physical action, the audience should HEAR it even though the text
names no sound: a door being entered means a door we hear. Direct the ones that merit
sound as full layered events; silently drop any that don't serve the scene):
{action_listing}

Return JSON:
1. "continuous_ambience": list of 1-3 ongoing background sound components for this scene's
   setting (e.g. "soft birdsong in distant trees", "wind through garden leaves").
2. "events": list of 0-8 composite sound events, each:
   - "name": short descriptive name
   - "anchor_line_index": which line the sound accompanies (int from the list above)
   - "source_text": the VERBATIM scene text motivating it (copied exactly, for validation)
   - "category": "action" (physical events: overturning pots, squeezing under a gate),
                 "creature" (non-verbal animal/character sounds: birds chirping TO someone,
                 a dog barking), or "environment" (weather, distant activity)
   - "emotional_intent": for creature sounds, what the sound is expressing (e.g. sparrows
     imploring Peter: "urgent, encouraging, don't-give-up"), else ""
   - "layers": 1-4 component layers, each:
       - "component": ONE concrete, generatable sound description
         (e.g. "ceramic pot scraping on wood", "small clay pots clinking together",
          "muffled thud on dirt floor")
       - "timing": "start" (t=0), "overlap" (slightly after), or "tail" (at the end)
       - "level": "prominent", "medium", or "subtle"

Rules: never include spoken dialogue as sound. Character vocalizations that are already
dialogue lines (a sneeze in quotes) are handled elsewhere -- do not duplicate them.
Only describe sounds motivated by this scene's actual text.
"""
    try:
        res, provider = query_llm_json(prompt, schema=SceneSoundDesign, task_name="tier3_sound_design")
        if not res:
            return None
        validated = SceneSoundDesign.model_validate(res)
    except (ValidationError, Exception) as e:
        logger.warning(f"Sound design failed for {scene_id}: {e}")
        return None

    scene_norm = _norm(scene_text)
    events = []
    for ev in validated.events:
        if not (0 <= ev.anchor_line_index < len(lines)):
            logger.warning(f"Dropping sound event with bad anchor {ev.anchor_line_index} in {scene_id}")
            continue
        if _norm(ev.source_text)[:60] not in scene_norm:
            logger.warning(f"Dropping ungrounded sound event (source not in scene): {ev.name!r}")
            continue
        if any(l["segment_type"] == "dialogue" and _norm(ev.source_text) in _norm(l["text"]) for l in lines):
            logger.warning(f"Dropping sound event duplicating dialogue: {ev.name!r}")
            continue
        events.append(ev.model_dump())

    return {
        "scene_id": scene_id,
        "continuous_ambience": validated.continuous_ambience[:3],
        "events": events,
        "designed_by": provider,
    }


# ====================================================
# AI-8: Dramatist / Adaptation Writer (grounded dramatization)
# ====================================================

class DramatizedInsert(BaseModel):
    anchor_line_index: int = Field(..., description="Line index this insert follows")
    source_text: str = Field(..., description="VERBATIM narrated event being dramatized")
    insert_type: str = Field(..., description="'dialogue' (invented spoken line) or 'performance_vocal' (non-lexical performance: groans, munching, wailing)")
    character: str = Field(..., description="Who performs it -- an existing character, or a new minor character")
    # Default "" (not required): partial model compliance should degrade to
    # per-insert drops, not nuke the whole scene's dramatization in validation.
    text: str = Field(default="", description="The performed text (for vocals: onomatopoeic performance, e.g. 'Ohhh... my tummy... Uuurp!')")
    delivery: str = Field(default="", description="Acting direction, e.g. 'Wailing loudly', 'Muffled through the giant pea', 'From afar'")
    meaning_note: str = Field(default="", description="For unintelligible speech: what it means, e.g. '(I don't know!)'")


class NewMinorCharacter(BaseModel):
    name: str = Field(..., description="e.g. 'Sparrow 1', 'Old Mouse'")
    description: str = Field(default="", description="One line: who they are in the scene")


class DramatistSchema(BaseModel):
    inserts: List[DramatizedInsert] = Field(default_factory=list)
    new_characters: List[NewMinorCharacter] = Field(default_factory=list)

    @field_validator("new_characters", mode="before")
    @classmethod
    def _coerce_string_characters(cls, v):
        # Models frequently return plain names (["Old Mouse"]) instead of
        # objects; one type error here must not nuke a scene's valid inserts.
        if isinstance(v, list):
            return [{"name": item} if isinstance(item, str) else item for item in v]
        return v


def _looks_like_stage_direction(text: str) -> bool:
    """Rejects performance_vocal 'text' that is a description rather than a
    performable sound ('Closing door', 'Panting, heavy footsteps', 'Gulps,
    heavy sigh'). Performable text carries interjection punctuation ('Achoo!',
    'Ohhh... my tummy') or onomatopoeic repetition ('creak, creak', 'nom, nom');
    descriptive phrases carry neither."""
    t = text.strip()
    if any(ch in t for ch in "!?…") or "..." in t or "-" in t:
        return False
    words = [w.strip(".,").lower() for w in t.split() if len(w.strip(".,")) > 2]
    if len(words) != len(set(words)):  # repeated token = onomatopoeic rhythm
        return False
    return True


def dramatize_scene(scene_id: str, scene_text: str, lines: List[Dict[str, Any]], level: str = "full", spotting: Optional[Dict[str, Any]] = None, bible: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """ROLE: Dramatist. The one AI allowed to invent text, under the grounded-
    dramatization rule: every insert must cite the verbatim narrated event it
    dramatizes. Inserts are ADDITIVE and flagged is_dramatized -- original lines
    are never altered, and faithful-mode output drops inserts entirely.

    Gemini-first (the most creatively demanding crew job; waits out rate-limit
    cooldowns), Groq fallback. Consumes the Spotter's sound/delivery moments."""
    if level == "faithful":
        return None
    spotting = spotting or {}
    existing_cast = sorted({l["character"] for l in lines if l["character"] != "Narrator"})
    mode_rules = (
        "You may invent short spoken lines for characters AND mint new minor characters\n"
        "(e.g. 'Sparrow 1', 'Old Mouse') when the narration reports creatures acting or speaking."
        if level == "full" else
        "You may ONLY add performance vocalizations for EXISTING characters -- no invented\n"
        "dialogue sentences, no new characters."
    )
    prompt = f"""You are the DRAMATIST (adaptation writer) for a Graphic-Audio-style full-cast audiobook.
{_bible_block(bible)}
Great audio drama converts REPORTED action into PERFORMED action: when the narration says a
character sobbed, the audience hears the sobbing; when it says birds implored him, the birds
get voices. Your inserts play ON TOP of the original narration -- the narrator still reads
the original text; your additions dramatize it.

THE GROUNDING RULE: every insert must dramatize a specific narrated event, and you must copy
that event's text VERBATIM into "source_text". Never dramatize something the text doesn't say.

{mode_rules}

SCENE TEXT:
{scene_text[:4000]}

LINES (index: [speaker] "text"):
{_line_listing(lines)}

EXISTING CAST: [{", ".join(existing_cast) or "Narrator only"}]

SPOTTED MOMENTS (from the spotting session -- dramatize these first):
{_moments_listing(spotting.get("sound_moments", []) + spotting.get("delivery_moments", []))}

PRIORITY BEAT TYPES (dramatize whenever the narration reports them): crying or sobbing;
eating or drinking; physical reactions (sickness, exhaustion, fright, relief); creatures
interacting with a character (birds, mice -- give them voices); shouts heard from a distance.

Return JSON:
- "inserts": 0-6 entries, each:
  - "anchor_line_index": the line this insert plays after
  - "source_text": verbatim narrated event being dramatized
  - "insert_type": "dialogue" (an invented spoken line) or "performance_vocal"
    (non-lexical performance: munching, groaning, wailing, gagging). A performance_vocal
    must be performable BY A HUMAN MOUTH: cries, breaths, hums, interjections. Object and
    impact sounds (thud, creak, splash, rustle) are NOT vocal performances -- they belong
    to the sound designer, never to a voice actor.
  - "character": who performs it -- MUST be a living being (a person or creature).
    Sounds made by objects (doors, pots, wind) are the sound designer's job, not yours.
  - "text": REQUIRED -- the words/sounds the voice actor actually performs. For dialogue,
    the invented line ("Come on Peter, you have to escape!"). For performance_vocal,
    performable onomatopoeia ("Mmm! Nom, nom, crunch! Ooh, a radish!" or
    "Ohhh... my tummy... Uuurp!"). Never leave this empty. NEVER put a description
    here: "Closing door" or "Scurrying footsteps" are stage directions, not performances --
    a voice actor must be able to SAY the text aloud.
  - "delivery": short acting/spatial direction ("Wailing loudly", "From afar",
    "Muffled through the giant pea")
  - "meaning_note": only for unintelligible speech, what it means
- "new_characters": minor characters you minted (empty unless genuinely needed)

EXAMPLE INSERT (format reference only -- never copy its content):
{{"anchor_line_index": 2, "source_text": "his sobs were overheard by some friendly sparrows",
  "insert_type": "dialogue", "character": "Sparrow 1", "text": "Come on, you have to escape!",
  "delivery": "Urgent, encouraging chirp", "meaning_note": ""}}

Quality over quantity: dramatize the emotionally-load-bearing beats, not every sentence.
"""
    try:
        # Gemini first (waits out its rate-limit cooldown when it's the sole
        # allowed provider), then the normal chain (Groq) if Gemini can't serve.
        res, provider = query_llm_json(prompt, schema=DramatistSchema, task_name="tier3_dramatization", allowed_providers=("gemini",))
        if not res:
            res, provider = query_llm_json(prompt, schema=DramatistSchema, task_name="tier3_dramatization")
        if not res:
            return None
        validated = DramatistSchema.model_validate(res)
    except (ValidationError, Exception) as e:
        logger.warning(f"Dramatization failed for {scene_id}: {e}")
        return None

    scene_norm = _norm(scene_text)
    # Living-beings enforcement: the model adapted to declared-only minting by
    # declaring the furniture ("Bed", "Teapot"). Object sounds belong to the
    # sound designer; a voice actor can't be cast as a kitchen.
    _INANIMATE = {
        "bed", "cup", "teapot", "kitchen", "door", "gate", "wall", "window",
        "pot", "pots", "hoe", "sieve", "can", "wheelbarrow", "scarecrow",
        "basket", "sand", "wood", "wind", "tree", "fire", "table", "house",
        "shed", "net", "jacket", "shoes", "watering-can", "pond",
    }
    allowed_new = {
        c.name for c in validated.new_characters
        if c.name.lower() not in _INANIMATE and c.name.split()[-1].lower() not in _INANIMATE
    } if level == "full" else set()
    seen_inserts = set()
    inserts = []
    for ins in validated.inserts[:6]:
        if not ins.text.strip():
            logger.warning(f"Dramatist: dropping insert with empty performed text ({ins.character})")
            continue
        if ins.insert_type == "performance_vocal" and _looks_like_stage_direction(ins.text):
            logger.warning(f"Dramatist: dropping stage-direction-as-text insert ({ins.character}: {ins.text[:30]!r})")
            continue
        # Copy-guard: an insert whose text already exists in the scene is narration
        # or an existing line duplicated, not invention (measured: "His mother put
        # him to bed..." returned verbatim as a Mrs. Rabbit 'performance'). Short
        # interjections are exempt to avoid false positives on "Oh!"-class vocals.
        if len(_norm(ins.text)) >= 15 and _norm(ins.text)[:60] in scene_norm:
            logger.warning(f"Dramatist: dropping copied-text insert ({ins.character}: {ins.text[:40]!r})")
            continue
        if not (0 <= ins.anchor_line_index < len(lines)):
            continue
        if _norm(ins.source_text)[:50] not in scene_norm:
            logger.warning(f"Dramatist: dropping ungrounded insert ({ins.character}: {ins.text[:30]!r})")
            continue
        if level == "enhanced" and (ins.insert_type != "performance_vocal" or ins.character not in existing_cast):
            continue
        if ins.character not in existing_cast and ins.character not in allowed_new and ins.character != "Narrator":
            # Declared-only minting: a new performer must appear in new_characters.
            # (Implicit minting let object-"characters" -- Hoe, Gate, Teapot --
            # slip past the living-beings rule.)
            logger.warning(f"Dramatist: dropping insert by undeclared character {ins.character!r}")
            continue
        dedupe_key = (ins.character.lower(), _norm(ins.text))
        if dedupe_key in seen_inserts:
            continue
        seen_inserts.add(dedupe_key)
        inserts.append({**ins.model_dump(), "is_dramatized": True})

    if not inserts:
        return None
    return {
        "scene_id": scene_id,
        "level": level,
        "inserts": inserts,
        "new_characters": [c.model_dump() for c in validated.new_characters if c.name in allowed_new or level == "full"],
        "dramatized_by": provider,
    }


def _merge_dramatizations(rounds: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Union of multiple Dramatist passes over one scene. Measured: single passes
    cover a different random subset of the dramatizable beats each time (~2/9 vs
    a ~6/9 union), so accumulation-with-dedupe beats prompt tuning for coverage.
    Dedupe: same character + same anchor + same insert_type = same beat; first
    (typically highest-quality provider) wins."""
    valid = [r for r in rounds if r]
    if not valid:
        return None
    merged = dict(valid[0])
    merged["inserts"] = []
    merged["new_characters"] = []
    seen_beats = set()
    seen_chars = set()
    for r in valid:
        for ins in r["inserts"]:
            beat_key = (ins["character"].lower(), ins["anchor_line_index"], ins["insert_type"])
            text_key = (ins["character"].lower(), re.sub(r"\W+", "", ins["text"].lower()))
            if beat_key in seen_beats or text_key in seen_beats:
                continue
            seen_beats.add(beat_key)
            seen_beats.add(text_key)
            merged["inserts"].append(ins)
        for c in r.get("new_characters", []):
            if c["name"].lower() not in seen_chars:
                seen_chars.add(c["name"].lower())
                merged["new_characters"].append(c)
    merged["rounds_merged"] = len(valid)
    return merged if merged["inserts"] else None


def _dramatize_beat(scene_id: str, scene_text: str, lines, moment, existing_cast, bible=None):
    """Beat-directed dramatization: one targeted call for ONE spotted moment the
    broad passes left uncovered. Measured rationale: broad passes stage a different
    random ~2/9 subset of the dramatizable beats each time; precision follow-up
    shots on the specific misses beat more shotgun volleys."""
    idx = moment["line_index"]
    lo, hi = max(0, idx - 2), min(len(lines), idx + 3)
    context_lines = "\n".join(f'{i}: [{lines[i]["character"]}] "{lines[i]["text"][:90]}"' for i in range(lo, hi))
    prompt = f"""You are the DRAMATIST for a Graphic-Audio-style audiobook. ONE specific narrated moment
needs a performed insert. Stage THIS moment only.
{_bible_block(bible)}
THE MOMENT (line {idx}): "{moment['source_text'][:120]}"
Opportunity: {moment['opportunity']}

SURROUNDING LINES:
{context_lines}

EXISTING CAST: [{", ".join(existing_cast) or "Narrator only"}]

Return JSON {{"inserts": [ONE insert or empty if truly undramatizable], "new_characters": [...]}}.
Insert fields: "anchor_line_index" (use {idx} unless clearly better adjacent),
"source_text" (copy the moment text verbatim), "insert_type" ("dialogue" or
"performance_vocal" -- a performance_vocal must be performable BY A HUMAN MOUTH;
object/impact sounds like thud/creak are NOT yours), "character" (a living being;
declare new minor characters in new_characters), "text" (REQUIRED: the exact words/
sounds the voice actor performs), "delivery", "meaning_note".
"""
    try:
        res, provider = query_llm_json(prompt, schema=DramatistSchema, task_name="tier3_beat_dramatization")
        if not res:
            return None
        validated = DramatistSchema.model_validate(res)
    except (ValidationError, Exception):
        return None

    scene_norm = _norm(scene_text)
    _inanimate = {"bed","cup","teapot","kitchen","door","gate","wall","window","pot",
                  "hoe","sieve","can","wheelbarrow","scarecrow","basket","sand","wood",
                  "wind","tree","fire","table","house","shed","net","jacket","shoes"}
    allowed_new = {c.name for c in validated.new_characters if c.name.lower() not in _inanimate}
    for ins in validated.inserts[:1]:
        if not ins.text.strip() or (ins.insert_type == "performance_vocal" and _looks_like_stage_direction(ins.text)):
            continue
        if len(_norm(ins.text)) >= 15 and _norm(ins.text)[:60] in scene_norm:
            continue  # copy-guard: scene text returned as "performance"
        if _norm(ins.source_text)[:50] not in scene_norm:
            continue
        if ins.character not in existing_cast and ins.character not in allowed_new and ins.character != "Narrator":
            continue
        if not (0 <= ins.anchor_line_index < len(lines)):
            continue
        return {"insert": {**ins.model_dump(), "is_dramatized": True, "beat_directed": True},
                "new_characters": [{"name": n} for n in allowed_new]}
    return None


def run_dramatization(manifest_path: str, level: str = "full", rounds: int = 1) -> str:
    """Standalone runner for the Dramatist pass only -> tier3/dramatization.json."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = ManuscriptManifest.model_validate_json(f.read())
    book_stem = os.path.splitext(manifest.source_file)[0]
    tier1_dir = os.path.join("data/corpus/pipeline", book_stem, "tier1")
    tier3_dir = os.path.join("data/corpus/pipeline", book_stem, "tier3")
    os.makedirs(tier3_dir, exist_ok=True)
    with open(os.path.join(tier1_dir, "loop3_scenes.json"), "r", encoding="utf-8") as f:
        scene_texts = {s["scene_id"]: s["text_block"] for s in json.load(f)}

    spotting_by_scene: Dict[str, Dict[str, Any]] = {}
    spotting_path = os.path.join(tier3_dir, "spotting.json")
    if os.path.exists(spotting_path):
        with open(spotting_path, "r", encoding="utf-8") as f:
            spotting_by_scene = {s["scene_id"]: s for s in json.load(f)}

    bible = analyze_book(book_stem, "\n\n".join(scene_texts.values()))

    from src import progress as _progress
    _total_scenes = sum(len(ch.scenes) for p in manifest.parts for ch in p.chapters)
    _done = 0
    results = []
    for part in manifest.parts:
        for chapter in part.chapters:
            for scene in chapter.scenes:
                lines = [l.model_dump() for l in scene.lines]
                scene_text = scene_texts.get(scene.scene_id, " ".join(l["text"] for l in lines))
                _done += 1
                _progress.report(book_stem, "dramatization", _done, _total_scenes, scene.scene_id)
                logger.info(f"Dramatizing scene {scene.scene_id} (level={level}, rounds={rounds})...")
                spotting = spotting_by_scene.get(scene.scene_id) or spot_scene(scene.scene_id, scene_text, lines)
                passes = [
                    dramatize_scene(scene.scene_id, scene_text, lines, level=level, spotting=spotting, bible=bible)
                    for _ in range(max(1, rounds))
                ]
                d = _merge_dramatizations(passes)

                # Beat-directed fill: precision shots at spotted moments the broad
                # passes left uncovered (no insert anchored within +/-1 line).
                if level != "faithful" and spotting:
                    covered = {ins["anchor_line_index"] for ins in (d["inserts"] if d else [])}
                    existing_cast = sorted({l["character"] for l in lines if l["character"] != "Narrator"}
                                           | {c["name"] for c in (d.get("new_characters", []) if d else [])})
                    uncovered = [
                        m for m in (spotting.get("sound_moments", []) + spotting.get("delivery_moments", []))
                        if not any(abs(m["line_index"] - a) <= 1 for a in covered)
                    ][:4]
                    beat_inserts, beat_chars = [], []
                    for m in uncovered:
                        hit = _dramatize_beat(scene.scene_id, scene_text, lines, m, existing_cast, bible=bible)
                        if hit:
                            beat_inserts.append(hit["insert"])
                            beat_chars.extend(hit["new_characters"])
                            covered.add(hit["insert"]["anchor_line_index"])
                    if beat_inserts:
                        if d is None:
                            d = {"scene_id": scene.scene_id, "level": level, "inserts": [],
                                 "new_characters": [], "dramatized_by": "beat_directed"}
                        d["inserts"].extend(beat_inserts)
                        seen = {c["name"].lower() for c in d["new_characters"]}
                        d["new_characters"].extend(c for c in beat_chars if c["name"].lower() not in seen)
                        logger.info(f"Beat-directed fill added {len(beat_inserts)} insert(s) to {scene.scene_id}")
                if d:
                    results.append(d)

    out_path = os.path.join(tier3_dir, "dramatization.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    total = sum(len(d["inserts"]) for d in results)
    new_chars = sorted({c["name"] for d in results for c in d.get("new_characters", [])})
    logger.info(f"Dramatization complete: {total} insert(s) across {len(results)} scene(s); new minor cast: {new_chars} -> {out_path}")
    return out_path


def run_sound_design(manifest_path: str) -> str:
    """Standalone runner: sound-design pass only, leaves existing direction artifacts alone."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = ManuscriptManifest.model_validate_json(f.read())
    book_stem = os.path.splitext(manifest.source_file)[0]
    tier1_dir = os.path.join("data/corpus/pipeline", book_stem, "tier1")
    tier3_dir = os.path.join("data/corpus/pipeline", book_stem, "tier3")
    os.makedirs(tier3_dir, exist_ok=True)
    with open(os.path.join(tier1_dir, "loop3_scenes.json"), "r", encoding="utf-8") as f:
        scene_texts = {s["scene_id"]: s["text_block"] for s in json.load(f)}

    spotting_by_scene: Dict[str, Dict[str, Any]] = {}
    spotting_path = os.path.join(tier3_dir, "spotting.json")
    if os.path.exists(spotting_path):
        with open(spotting_path, "r", encoding="utf-8") as f:
            spotting_by_scene = {s["scene_id"]: s for s in json.load(f)}

    bible = analyze_book(book_stem, "\n\n".join(scene_texts.values()))

    from src import progress as _progress
    _total_scenes = sum(len(ch.scenes) for p in manifest.parts for ch in p.chapters)
    _done = 0
    designs = []
    for part in manifest.parts:
        for chapter in part.chapters:
            for scene in chapter.scenes:
                lines = [l.model_dump() for l in scene.lines]
                scene_text = scene_texts.get(scene.scene_id, " ".join(l["text"] for l in lines))
                _done += 1
                _progress.report(book_stem, "sound_design", _done, _total_scenes, scene.scene_id)
                logger.info(f"Sound-designing scene {scene.scene_id}...")
                spotting = spotting_by_scene.get(scene.scene_id) or spot_scene(scene.scene_id, scene_text, lines)
                design = design_scene_sound(scene.scene_id, scene_text, lines, spotting=spotting, bible=bible)
                if design:
                    designs.append(design)

    out_path = os.path.join(tier3_dir, "sound_design.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(designs, f, indent=4)
    total_events = sum(len(d["events"]) for d in designs)
    total_layers = sum(len(ev["layers"]) for d in designs for ev in d["events"])
    logger.info(f"Sound design complete: {len(designs)} scenes, {total_events} composite events, {total_layers} layers -> {out_path}")
    return out_path


# ====================================================
# AI-9: Book Analyst (whole-book production bible, big-context Gemini)
# ====================================================

class BookBible(BaseModel):
    title: str = Field(default="")
    author: str = Field(default="")
    era_setting: str = Field(default="", description="e.g. 'Victorian England, early 1890s'")
    location_setting: str = Field(default="", description="e.g. 'London: indoor parlors, gaslit streets'")
    genre: str = Field(default="")
    book_type: str = Field(default="", description="novel | short story | fable | play | memoir")
    tone: str = Field(default="")
    target_audience: str = Field(default="")
    music_style_notes: str = Field(default="", description="Global musical palette guidance for the whole production")
    sound_palette_notes: str = Field(default="", description="Period/setting-appropriate sounds; anachronisms to avoid")
    dramatization_register: str = Field(default="", description="How restrained/exuberant invented performance should be")


def analyze_book(book_stem: str, full_text: str) -> Optional[Dict[str, Any]]:
    """One whole-book call (Gemini-first for the large context window) inferring
    setting, era, genre, and tone -- the production bible every crew role inherits
    so scene-level choices stay period- and register-consistent. Cached to
    tier3/book_bible.json; inference (not verbatim grounding) is the point here."""
    tier3_dir = os.path.join("data/corpus/pipeline", book_stem, "tier3")
    os.makedirs(tier3_dir, exist_ok=True)
    bible_path = os.path.join(tier3_dir, "book_bible.json")
    if os.path.exists(bible_path):
        with open(bible_path, "r", encoding="utf-8") as f:
            return json.load(f)

    prompt = f"""You are the HEAD OF PRODUCTION for a Graphic-Audio-style audiobook studio.
Read the manuscript below and produce the PRODUCTION BIBLE: the global facts and creative
register every department (music, sound design, dramatization, casting) must stay consistent
with. Infer what is not explicitly stated (era from technology/idiom, audience from register).

MANUSCRIPT (may be truncated):
{full_text[:150000]}

Return JSON with: "title", "author" (empty if unstated), "era_setting" (period + rough years),
"location_setting", "genre", "book_type" (novel/short story/fable/play/memoir),
"tone", "target_audience", "music_style_notes" (global palette: instruments, idiom, what would
feel wrong), "sound_palette_notes" (period-true sound world; name anachronisms to AVOID, e.g.
'no motor vehicles, no electric hum'), "dramatization_register" (how restrained or exuberant
invented performance should be for this text)."""
    try:
        res, provider = query_llm_json(prompt, schema=BookBible, task_name="tier3_book_analysis", allowed_providers=("gemini",))
        if not res:
            # Groq as fallback, but NEVER local Ollama: measured failure -- the 3B
            # model invented the wrong author and nonsense palette notes, and a
            # wrong bible poisons every downstream crew prompt. No bible beats a
            # confidently wrong one.
            res, provider = query_llm_json(prompt, schema=BookBible, task_name="tier3_book_analysis", allowed_providers=("gemini", "groq"))
        if not res:
            return None
        bible = BookBible.model_validate(res).model_dump()
        bible["analyzed_by"] = provider
    except (ValidationError, Exception) as e:
        logger.warning(f"Book analysis failed for {book_stem}: {e}")
        return None
    with open(bible_path, "w", encoding="utf-8") as f:
        json.dump(bible, f, indent=4)
    logger.info(f"Book bible written: {bible.get('era_setting')!r} / {bible.get('genre')!r} -> {bible_path}")
    return bible


def _bible_block(bible: Optional[Dict[str, Any]]) -> str:
    if not bible:
        return ""
    return f"""
PRODUCTION BIBLE (global constraints -- stay consistent with these):
- Setting/era: {bible.get('era_setting', '?')} | {bible.get('location_setting', '?')}
- Genre/type: {bible.get('genre', '?')} ({bible.get('book_type', '?')}) | Tone: {bible.get('tone', '?')} | Audience: {bible.get('target_audience', '?')}
- Music palette: {bible.get('music_style_notes', '')}
- Sound palette: {bible.get('sound_palette_notes', '')}
- Dramatization register: {bible.get('dramatization_register', '')}
"""


# ====================================================
# AI-10: Character Designer (visual profiles for Chain E consistency)
# ====================================================

class CharacterProfile(BaseModel):
    name: str
    visual_description: str = Field(..., description="Concise paintable description: species/build, age, attire, distinguishing features")
    evidence_snippets: List[str] = Field(default_factory=list, description="Verbatim text snippets supporting the description")
    inferred: bool = Field(default=False, description="True when the text gives no explicit description and this is period/genre-consistent invention")


class CharacterDesignSchema(BaseModel):
    profiles: List[CharacterProfile] = Field(default_factory=list)


def design_characters(manifest: "ManuscriptManifest", scene_texts: Dict[str, str], bible: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """ROLE: Character designer. One call producing a visual profile per main
    character, grounded where the text describes them (Peter's 'blue jacket with
    brass buttons') and explicitly flagged `inferred` where it doesn't. Profiles
    feed every Chain E image prompt so characters look consistent across scenes,
    and are stored in each character's MemPalace drawer (visual_profile key)."""
    mentions: Dict[str, List[str]] = {}
    counts: Dict[str, int] = {}
    full_text = "\n".join(scene_texts.values())
    all_speakers = set()
    for part in manifest.parts:
        for ch in part.chapters:
            for sc in ch.scenes:
                for l in sc.lines:
                    if l.segment_type == "dialogue" and l.character != "Narrator":
                        counts[l.character] = counts.get(l.character, 0) + 1
                        all_speakers.add(l.character)
    # Importance = dialogue + narrative presence: a protagonist can speak once
    # (measured: Peter's only line is "Kertyschoo!") yet dominate the narration.
    mention_counts = {
        name: len(re.findall(r"\b" + re.escape(name.split()[-1]) + r"\b", full_text))
        for name in all_speakers
    }
    importance = {n: counts.get(n, 0) * 3 + mention_counts.get(n, 0) for n in all_speakers}
    main_cast = [n for n, s in sorted(importance.items(), key=lambda kv: -kv[1])
                 if counts.get(n, 0) >= 2 or mention_counts.get(n, 0) >= 3][:10]
    if not main_cast:
        return []
    # STRONG descriptors (attire/color/build) rank above WEAK ones (size/age):
    # measured failure #2 -- "little" marked nearly every early mention descriptive,
    # filling the window cap before the search reached "a blue jacket with brass
    # buttons". Strong-hit windows always win the prompt slots.
    _STRONG_DESC = {"jacket", "coat", "dress", "apron", "hat", "shoes", "wearing", "dressed",
                    "clad", "beard", "whiskers", "blue", "red", "brown", "white", "black",
                    "green", "golden", "button", "buttons", "cap", "gown", "frock",
                    "umbrella", "basket", "stout", "plump", "grey", "gray", "fur"}
    _WEAK_DESC = {"tall", "short", "little", "big", "old", "young", "thin", "fat", "hair"}
    for name in main_cast:
        strong, weak, generic = [], [], []
        for m in re.finditer(re.escape(name.split()[-1]), full_text):
            lo = max(0, m.start() - 160)
            window = re.sub(r"\s+", " ", full_text[lo:m.end() + 200])
            wl = window.lower()
            if any(w in wl for w in _STRONG_DESC):
                strong.append(window)
            elif any(w in wl for w in _WEAK_DESC):
                weak.append(window)
            else:
                generic.append(window)
            if len(strong) >= 4:
                break
        mentions[name] = (strong + weak + generic)[:4]

    mention_block = "\n".join(
        f'- {name}: ' + " | ".join(f'"{s[:130]}"' for s in mentions[name][:3])
        for name in main_cast
    )
    prompt = f"""You are the CHARACTER DESIGNER for an illustrated audiobook production.
Produce one concise, paintable visual profile per character below, so an image model
draws them CONSISTENTLY in every scene.
{_bible_block(bible)}
CHARACTERS (with text mentions):
{mention_block}

Return JSON: {{"profiles": [{{"name", "visual_description" (species/build, age, attire,
distinguishing features -- one dense sentence), "evidence_snippets" (VERBATIM text
fragments that support details, empty if none), "inferred" (true when the text never
describes them and you invented a period/genre-consistent look)}}]}}
Ground what you can; invent the rest deliberately and mark it inferred."""
    try:
        res, provider = query_llm_json(prompt, schema=CharacterDesignSchema, task_name="tier3_character_design", allowed_providers=("gemini", "groq"))
        if not res:
            return []
        validated = CharacterDesignSchema.model_validate(res)
    except (ValidationError, Exception) as e:
        logger.warning(f"Character design failed: {e}")
        return []

    full_norm = _norm(full_text)
    _VISUAL_WORDS = {"jacket", "coat", "dress", "apron", "hat", "shoes", "wearing", "dressed",
                     "clad", "fur", "hair", "beard", "whiskers", "tall", "short", "little",
                     "big", "old", "young", "blue", "red", "brown", "white", "black", "green",
                     "golden", "button", "buttons", "cap", "gown", "frock", "umbrella", "basket"}
    profiles = []
    for p in validated.profiles:
        if p.name not in main_cast:
            continue
        grounded = [s for s in p.evidence_snippets if _norm(s)[:40] in full_norm]
        # A mention isn't a description: inferred=False requires evidence that
        # actually carries visual information (attire, color, build).
        visual_evidence = [s for s in grounded if any(w in s.lower() for w in _VISUAL_WORDS)]
        profiles.append({
            "name": p.name,
            "visual_description": p.visual_description[:300],
            "evidence_snippets": visual_evidence or grounded,
            "inferred": p.inferred or not visual_evidence,
            "designed_by": provider,
        })
    return profiles


def run_character_design(manifest_path: str, sync_mempalace: bool = True) -> str:
    from src.models import ManuscriptManifest as _MM
    manifest = _MM.model_validate_json(open(manifest_path, encoding="utf-8").read())
    book_stem = os.path.splitext(manifest.source_file)[0]
    tier1_dir = os.path.join("data/corpus/pipeline", book_stem, "tier1")
    tier3_dir = os.path.join("data/corpus/pipeline", book_stem, "tier3")
    os.makedirs(tier3_dir, exist_ok=True)
    with open(os.path.join(tier1_dir, "loop3_scenes.json"), encoding="utf-8") as f:
        scene_texts = {s["scene_id"]: s["text_block"] for s in json.load(f)}
    bible = analyze_book(book_stem, "\n\n".join(scene_texts.values()))
    profiles = design_characters(manifest, scene_texts, bible=bible)
    out = os.path.join(tier3_dir, "character_profiles.json")
    json.dump(profiles, open(out, "w", encoding="utf-8"), indent=4)
    logger.info(f"Character design: {len(profiles)} profile(s) -> {out}")
    if sync_mempalace and profiles:
        from src.spatial_memory import MemPalace
        palace = MemPalace(use_chroma=False)
        try:
            for p in profiles:
                drawer = palace.get_character_drawer(p["name"])
                if drawer:
                    config = drawer["modulation_config"]
                    config["visual_profile"] = p["visual_description"]
                    palace.conn.execute("UPDATE drawers SET modulation_config_json = ? WHERE character_name = ?",
                                        (json.dumps(config), p["name"]))
            palace.conn.commit()
        finally:
            palace.close()
    return out


# ====================================================
# AI-11: Production QC Critic (advisory review of the crew's output)
# ====================================================

class QCIssue(BaseModel):
    artifact: str = Field(..., description="'music' | 'sound_design' | 'dramatization' | 'delivery'")
    scene_id: str = Field(default="")
    offending_text: str = Field(..., description="VERBATIM text copied from the artifact being flagged")
    issue: str = Field(..., description="What is wrong: anachronism, register violation, bible inconsistency, repetition")
    severity: str = Field(default="minor", description="'minor' | 'major'")
    suggestion: str = Field(default="")


class QCReportSchema(BaseModel):
    issues: List[QCIssue] = Field(default_factory=list)


def run_qc_review(manifest_path: str) -> str:
    """ROLE: QC critic. Reads the bible + the crew's creative output and flags
    inconsistencies: anachronisms in sound/music prompts, register violations in
    dramatized inserts, bible contradictions. Advisory only; every flag must quote
    the offending artifact text verbatim (grounded against the artifacts)."""
    from src.models import ManuscriptManifest as _MM
    manifest = _MM.model_validate_json(open(manifest_path, encoding="utf-8").read())
    book_stem = os.path.splitext(manifest.source_file)[0]
    tier3_dir = os.path.join("data/corpus/pipeline", book_stem, "tier3")

    def _load(name):
        p = os.path.join(tier3_dir, name)
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

    bible = _load("book_bible.json") or {}
    directions = _load("production_script.json") or []
    designs = _load("sound_design.json") or []
    drams = _load("dramatization.json") or []

    corpus_parts = []
    for d in directions:
        corpus_parts.append(f"[music {d['scene_id']}] style: {d['music'].get('style','')} | stingers: " +
                            "; ".join(s["description"] for s in d["music"].get("stingers", [])))
    for s in designs:
        for ev in s.get("events", []):
            corpus_parts.append(f"[sound_design {s['scene_id']}] {ev['name']}: " + "; ".join(l["component"] for l in ev["layers"]))
        if s.get("continuous_ambience"):
            corpus_parts.append(f"[sound_design {s['scene_id']}] ambience: " + "; ".join(s["continuous_ambience"]))
    for s in drams:
        for ins in s.get("inserts", []):
            corpus_parts.append(f"[dramatization {s['scene_id']}] {ins['character']}: \"{ins['text']}\" ({ins.get('delivery','')})")
    artifact_corpus = "\n".join(corpus_parts)[:14000]

    prompt = f"""You are the QC SUPERVISOR reviewing a Graphic-Audio production's creative output
before it ships. Flag genuine problems only -- anachronisms vs the production bible,
register/tone violations, contradictions, and grating repetition. Do not flag style
preferences.
{_bible_block(bible)}
PRODUCTION ARTIFACTS:
{artifact_corpus}

Return JSON: {{"issues": [{{"artifact" ('music'|'sound_design'|'dramatization'|'delivery'),
"scene_id", "offending_text" (copied VERBATIM from the artifacts above), "issue",
"severity" ('minor'|'major'), "suggestion"}}]}} -- empty list if the production is clean."""
    issues = []
    try:
        res, provider = query_llm_json(prompt, schema=QCReportSchema, task_name="tier3_qc_review", allowed_providers=("gemini", "groq"))
        if res:
            validated = QCReportSchema.model_validate(res)
            corpus_norm = _norm(artifact_corpus)
            for iss in validated.issues:
                if _norm(iss.offending_text)[:40] in corpus_norm:
                    issues.append({**iss.model_dump(), "reviewed_by": provider})
                else:
                    logger.warning(f"QC: dropping ungrounded flag: {iss.offending_text[:40]!r}")
    except (ValidationError, Exception) as e:
        logger.warning(f"QC review failed: {e}")

    out = os.path.join(tier3_dir, "qc_report.json")
    json.dump(issues, open(out, "w", encoding="utf-8"), indent=4)
    majors = sum(1 for i in issues if i["severity"] == "major")
    logger.info(f"QC review: {len(issues)} issue(s) ({majors} major) -> {out}")
    return out


# ====================================================
# Production Crew (role-focused specialist passes)
#
# Mirrors a real Graphic-Audio production team: a SPOTTER breaks the scene down
# first (where are the opportunities?), then each specialist -- MUSIC DIRECTOR,
# SOUND DESIGNER, DIALOGUE DIRECTOR -- works only their own craft against the
# spotted moments. Focused prompts beat one generalist prompt: each role sees
# only its concerns, and each output is schema-validated and text-grounded.
# ====================================================

class SpottedMoment(BaseModel):
    line_index: int = Field(..., description="Line the moment anchors to")
    source_text: str = Field(..., description="Verbatim scene snippet")
    opportunity: str = Field(..., description="What could happen here, in this role's terms")


class SceneSpottingSchema(BaseModel):
    music_moments: List[SpottedMoment] = Field(default_factory=list)
    sound_moments: List[SpottedMoment] = Field(default_factory=list)
    delivery_moments: List[SpottedMoment] = Field(default_factory=list)


def _line_listing(lines: List[Dict[str, Any]], with_emotion: bool = False) -> str:
    return "\n".join(
        f'{i}: [{l["character"]}{" | " + l["emotion"] if with_emotion and l.get("emotion") else ""}] "{l["text"][:90]}"'
        for i, l in enumerate(lines)
    )


def spot_scene(scene_id: str, scene_text: str, lines: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """ROLE: Spotting/breakdown artist. Finds WHERE the opportunities are -- makes no
    creative decisions about WHAT the music/sound/delivery should be (that's the
    specialists' job). Returns grounded moment lists; empty lists on failure."""
    prompt = f"""You are the SPOTTING ARTIST for a Graphic-Audio-style full-cast audiobook production.
Your only job is scene breakdown: mark WHERE production opportunities exist. You do NOT decide
what the music or sounds should be -- the music director and sound designer do that from your marks.

SCENE TEXT:
{scene_text[:4000]}

LINES (index: [speaker] "text"):
{_line_listing(lines)}

Return JSON with three lists (each entry: "line_index" int, "source_text" verbatim snippet,
"opportunity" one short phrase):
1. "music_moments": dramatic beats where music could shift or accent -- reveals, threats,
   mood turns, comic beats, scene-opening tone. 0-4 entries.
2. "sound_moments": physical actions, environmental sounds, and non-verbal creature/character
   sounds implied by the text (things falling, doors, footsteps, animals reacting TO someone).
   Never spoken dialogue. 0-6 entries.
3. "delivery_moments": dialogue lines whose emotional delivery is non-obvious and would benefit
   from acting direction. 0-5 entries.
"""
    empty = {"music_moments": [], "sound_moments": [], "delivery_moments": []}
    try:
        res, provider = query_llm_json(prompt, schema=SceneSpottingSchema, task_name="tier3_spotting")
        if not res:
            return empty
        validated = SceneSpottingSchema.model_validate(res)
    except (ValidationError, Exception) as e:
        logger.warning(f"Spotting failed for {scene_id}: {e}")
        return empty

    scene_norm = _norm(scene_text)
    out = {}
    for key in ("music_moments", "sound_moments", "delivery_moments"):
        moments = []
        for m in getattr(validated, key):
            if 0 <= m.line_index < len(lines) and _norm(m.source_text)[:50] in scene_norm:
                moments.append(m.model_dump())
        out[key] = moments
    return out


def _moments_listing(moments: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f'- line {m["line_index"]}: "{m["source_text"][:70]}" ({m["opportunity"]})'
        for m in moments
    ) or "(the spotter found none; use your judgment sparingly)"


def direct_scene(scene_id: str, scene_text: str, lines: List[Dict[str, Any]], sfx_cues: List[Dict[str, Any]], spotting: Optional[Dict[str, Any]] = None, bible: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """ROLE: Music director (+ set/environment). Consumes the spotter's music_moments.
    Returns validated scene direction dict, or None (caller keeps scene undirected)."""
    spotting = spotting or {}
    prompt = f"""You are the MUSIC DIRECTOR for a Graphic-Audio-style full-cast audiobook production.
Your craft: score direction and the scene's acoustic setting. The spotting artist has already
marked the dramatic beats -- score against those marks. Ground every choice in the actual text.
{_bible_block(bible)}
SCENE TEXT:
{scene_text[:4000]}

LINES (index: [speaker | emotion] "text"):
{_line_listing(lines, with_emotion=True)}

SPOTTED MUSIC MOMENTS (from the spotting session):
{_moments_listing(spotting.get("music_moments", []))}

Return a JSON object:
1. "music": {{
     "base_mood": one or two words for the scene's overall musical mood,
     "style": a concrete musical style description (instruments, feel) fitting the story's tone,
     "stingers": [{{
        "after_line_index": int (line index after which a short musical accent should play),
        "description": what the stinger sounds like, in concrete musical terms,
        "trigger_text": the VERBATIM text snippet from the scene that motivates it
     }}] -- 0 to 3 stingers, prioritizing the spotted moments,
     "events": [{{
        "after_line_index": int, "action": "stop" | "resume" | "change",
        "new_style": for change/resume, what the bed becomes (empty for stop),
        "trigger_text": VERBATIM motivating snippet
     }}] -- 0 to 3 state changes. Music is a living timeline: it can STOP dead for shock
     ("Music stops abruptly. Dead silence."), RESUME when action restarts, or CHANGE
     when the dramatic register shifts mid-scene. Only for genuine turns.
   }}
2. "environment": {{
     "location": brief description, "time_of_day": e.g. morning/night/unknown,
     "weather": e.g. clear/raining/indoors/unknown,
     "physical_confines": e.g. tight space/open field/small room,
     "ambient_noise_level": quiet/moderate/loud
   }}
3. "delivery_notes": [] (leave empty -- the dialogue director handles delivery)
"""
    try:
        res, provider = query_llm_json(prompt, schema=SceneDirectionSchema, task_name="tier3_music_direction")
        if not res:
            return None
        validated = SceneDirectionSchema.model_validate(res)
    except (ValidationError, Exception) as e:
        logger.warning(f"Music direction failed for {scene_id}: {e}")
        return None

    # Grounding/range validation -- enforcement, not trust.
    scene_norm = _norm(scene_text)
    stingers = []
    for s in validated.music.stingers:
        if not (0 <= s.after_line_index < len(lines)):
            logger.warning(f"Dropping stinger with out-of-range line index {s.after_line_index} in {scene_id}")
            continue
        if _norm(s.trigger_text) not in scene_norm:
            logger.warning(f"Dropping ungrounded stinger (trigger not in scene): {s.trigger_text[:40]!r}")
            continue
        stingers.append(s.model_dump())
    music_events = []
    for ev in validated.music.events:
        if not (0 <= ev.after_line_index < len(lines)):
            continue
        if _norm(ev.trigger_text) not in scene_norm:
            logger.warning(f"Dropping ungrounded music event: {ev.trigger_text[:40]!r}")
            continue
        if ev.action not in ("stop", "resume", "change"):
            continue
        music_events.append(ev.model_dump())
    music_events.sort(key=lambda e: e["after_line_index"])
    delivery = [d.model_dump() for d in validated.delivery_notes if 0 <= d.index < len(lines)]

    return {
        "scene_id": scene_id,
        "music": {**validated.music.model_dump(), "stingers": stingers, "events": music_events},
        "environment": validated.environment.model_dump(),
        "delivery_notes": delivery,
        "directed_by": provider,
    }


class DialogueDirectionSchema(BaseModel):
    delivery_notes: List[DeliveryNote] = Field(default_factory=list)


def direct_dialogue(scene_id: str, scene_text: str, lines: List[Dict[str, Any]], spotting: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """ROLE: Dialogue director. Acting direction only, against the spotter's
    delivery_moments. Returns validated delivery notes (empty on failure)."""
    spotting = spotting or {}
    dialogue_indices = [i for i, l in enumerate(lines) if l["segment_type"] == "dialogue"]
    if not dialogue_indices:
        return []
    prompt = f"""You are the DIALOGUE DIRECTOR for a Graphic-Audio-style full-cast audiobook production.
Your craft: acting direction for voice performers. The spotting artist has marked lines whose
delivery is non-obvious -- direct those first; add others only if clearly needed.

SCENE TEXT:
{scene_text[:4000]}

LINES (index: [speaker | emotion] "text"):
{_line_listing(lines, with_emotion=True)}

SPOTTED DELIVERY MOMENTS:
{_moments_listing(spotting.get("delivery_moments", []))}

Return JSON: {{"delivery_notes": [{{"index": int (a DIALOGUE line index), "note": str}}]}}
Notes are short parenthetical acting directions a voice actor performs from -- pacing,
subtext, physical state ("Warm but stern", "Breathless, terrified", "Whispered, conspiratorial",
"Through gritted teeth"). 0 to 6 notes. Do not direct narrative lines.
"""
    try:
        res, provider = query_llm_json(prompt, schema=DialogueDirectionSchema, task_name="tier3_dialogue_direction")
        if not res:
            return []
        validated = DialogueDirectionSchema.model_validate(res)
        return [d.model_dump() for d in validated.delivery_notes if d.index in dialogue_indices]
    except (ValidationError, Exception) as e:
        logger.warning(f"Dialogue direction failed for {scene_id}: {e}")
        return []


# ====================================================
# Prompt builder (deterministic, no LLM)
# ====================================================

def build_generation_prompts(direction: Dict[str, Any], lines: List[Dict[str, Any]], sfx_cues: List[Dict[str, Any]], visual_style: str = "warm storybook watercolor illustration") -> Dict[str, Any]:
    music = direction["music"]
    env = direction["environment"]
    characters_present = sorted({l["character"] for l in lines if l["segment_type"] == "dialogue" and l["character"] != "Narrator"})

    env_bits = [env["location"]]
    if env["time_of_day"] not in ("", "unknown"):
        env_bits.append(env["time_of_day"])
    if env["weather"] not in ("", "unknown", "indoors"):
        env_bits.append(env["weather"])

    return {
        "scene_id": direction["scene_id"],
        "music_prompt": f"{music['style']}, {music['base_mood']} mood, instrumental background bed for an audiobook scene, loopable, no vocals",
        "stinger_prompts": [s["description"] + ", short musical accent, one to three seconds" for s in music["stingers"]],
        "ambience_prompt": f"ambient field recording, {', '.join(env_bits)}, {env['ambient_noise_level']} background level, {env['physical_confines']} acoustics, loopable",
        "sfx_prompts": [f"{c['description']}, isolated sound effect" for c in sfx_cues],
        "image_prompt": (
            f"{visual_style}: {env['location']}"
            + (f", {env['time_of_day']}" if env["time_of_day"] not in ("", "unknown") else "")
            + (f", featuring {', '.join(characters_present)}" if characters_present else "")
            + f", {music['base_mood']} atmosphere"
        ),
        "characters_present": characters_present,
    }


# ====================================================
# Human-readable production script (Tier 3 gold format)
# ====================================================

def render_production_script(scene_directions: List[Dict[str, Any]], scenes_lines: Dict[str, List[Dict[str, Any]]], scenes_sfx: Dict[str, List[Dict[str, Any]]]) -> str:
    out = []
    for n, d in enumerate(scene_directions, 1):
        scene_id = d["scene_id"]
        lines = scenes_lines.get(scene_id, [])
        out.append(f"[Scene {n} - {d['environment']['location']}]")
        out.append("")
        out.append(f"[MUSIC: {d['music']['style']} -- {d['music']['base_mood']} mood.]")
        sfx = scenes_sfx.get(scene_id, [])
        if sfx:
            out.append(f"[SFX: {'; '.join(c['description'] for c in sfx)}.]")
        out.append("")

        notes_by_index = {dn["index"]: dn["note"] for dn in d["delivery_notes"]}
        stingers_by_index: Dict[int, List[str]] = {}
        for s in d["music"]["stingers"]:
            stingers_by_index.setdefault(s["after_line_index"], []).append(s["description"])

        current_speaker = None
        for i, line in enumerate(lines):
            speaker = line["character"] if line["segment_type"] == "dialogue" else "Narrator"
            if speaker != current_speaker:
                out.append(f"{speaker}:")
                current_speaker = speaker
            prefix = f"*({notes_by_index[i]})* " if i in notes_by_index else ""
            vocal = " [vocalization]" if line.get("utterance_type") == "vocalization" else ""
            out.append(f'{prefix}"{line["text"]}"{vocal}')
            for stinger in stingers_by_index.get(i, []):
                out.append(f"[MUSIC: {stinger}]")
            out.append("")
        out.append("")
    return "\n".join(out)


# ====================================================
# MemPalace sync (production knowledge base)
# ====================================================

def sync_to_mempalace(book_filename: str, manifest: ManuscriptManifest, scene_directions: List[Dict[str, Any]], generation_prompts: List[Dict[str, Any]], alias_groups: List[Dict[str, Any]]) -> None:
    from src.spatial_memory import MemPalace
    palace = MemPalace(use_chroma=False)
    try:
        directions_by_scene = {d["scene_id"]: d for d in scene_directions}
        prompts_by_scene = {p["scene_id"]: p for p in generation_prompts}

        chapter_num = 0
        for part in manifest.parts:
            for chapter in part.chapters:
                chapter_num += 1
                for scene in chapter.scenes:
                    d = directions_by_scene.get(scene.scene_id)
                    meta = {
                        "scene_id": scene.scene_id,
                        "book": book_filename,
                        "environment": d["environment"] if d else {},
                        "music": d["music"] if d else {},
                        "delivery_notes": d["delivery_notes"] if d else [],
                        "generation_prompts": prompts_by_scene.get(scene.scene_id, {}),
                    }
                    palace.log_wing(
                        wing_id=scene.scene_id,
                        chapter_number=chapter_num,
                        title=(d["environment"]["location"] if d else scene.scene_id),
                        metadata=meta,
                    )

        for group in alias_groups:
            for alias in group["aliases"]:
                palace.save_confirmed_merge(
                    book_filename=book_filename,
                    original_name=alias,
                    canonical_name=group["canonical"],
                    is_confirmed=True,
                    confidence_score=0.9,
                )
        logger.info(f"MemPalace sync complete: {chapter_num} chapter wing group(s), {len(alias_groups)} alias group(s).")
    finally:
        palace.close()


# ====================================================
# Orchestration
# ====================================================

def direct_manifest(manifest_path: str, sync_mempalace: bool = False) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = ManuscriptManifest.model_validate_json(f.read())

    book_stem = os.path.splitext(manifest.source_file)[0]
    tier1_dir = os.path.join("data/corpus/pipeline", book_stem, "tier1")
    tier3_dir = os.path.join("data/corpus/pipeline", book_stem, "tier3")
    os.makedirs(tier3_dir, exist_ok=True)

    with open(os.path.join(tier1_dir, "loop3_scenes.json"), "r", encoding="utf-8") as f:
        scene_texts = {s["scene_id"]: s["text_block"] for s in json.load(f)}

    bible = analyze_book(book_stem, "\n\n".join(scene_texts.values()))

    sfx_path = os.path.join(tier1_dir, "loopE_llm_sfx_cues.json")
    scenes_sfx: Dict[str, List[Dict[str, Any]]] = {}
    if os.path.exists(sfx_path):
        with open(sfx_path, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                scenes_sfx[entry["scene_id"]] = entry["sfx_cues"]

    alias_path = os.path.join(tier1_dir, "loopE_llm_alias_merges.json")
    alias_groups: List[Dict[str, Any]] = []
    if os.path.exists(alias_path):
        with open(alias_path, "r", encoding="utf-8") as f:
            alias_groups = json.load(f)

    scenes_lines: Dict[str, List[Dict[str, Any]]] = {}
    scene_directions: List[Dict[str, Any]] = []
    generation_prompts: List[Dict[str, Any]] = []
    all_spotting: List[Dict[str, Any]] = []
    from src import progress as _progress
    _total_scenes = sum(len(ch.scenes) for p in manifest.parts for ch in p.chapters)
    _done = 0

    for part in manifest.parts:
        for chapter in part.chapters:
            for scene in chapter.scenes:
                lines = [l.model_dump() for l in scene.lines]
                scenes_lines[scene.scene_id] = lines
                scene_text = scene_texts.get(scene.scene_id, " ".join(l["text"] for l in lines))
                sfx = scenes_sfx.get(scene.scene_id, [])

                _done += 1
                _progress.report(book_stem, "crew_direction", _done, _total_scenes, scene.scene_id)
                logger.info(f"Directing scene {scene.scene_id} ({len(lines)} lines, {len(sfx)} sfx cues)...")
                spotting = spot_scene(scene.scene_id, scene_text, lines)
                all_spotting.append({"scene_id": scene.scene_id, **spotting})
                direction = direct_scene(scene.scene_id, scene_text, lines, sfx, spotting=spotting, bible=bible)
                if direction is None:
                    direction = {
                        "scene_id": scene.scene_id,
                        "music": {"base_mood": "neutral", "style": "soft ambient underscore", "stingers": []},
                        "environment": {"location": f"Scene {scene.scene_id}", "time_of_day": "unknown", "weather": "unknown", "physical_confines": "open", "ambient_noise_level": "quiet"},
                        "delivery_notes": [],
                        "directed_by": None,
                    }
                direction["delivery_notes"] = direct_dialogue(scene.scene_id, scene_text, lines, spotting=spotting) or direction.get("delivery_notes", [])
                scene_directions.append(direction)
                generation_prompts.append(build_generation_prompts(direction, lines, sfx))

    with open(os.path.join(tier3_dir, "spotting.json"), "w", encoding="utf-8") as f:
        json.dump(all_spotting, f, indent=4)
    with open(os.path.join(tier3_dir, "production_script.json"), "w", encoding="utf-8") as f:
        json.dump(scene_directions, f, indent=4)
    with open(os.path.join(tier3_dir, "generation_prompts.json"), "w", encoding="utf-8") as f:
        json.dump(generation_prompts, f, indent=4)
    script_text = render_production_script(scene_directions, scenes_lines, scenes_sfx)
    with open(os.path.join(tier3_dir, "production_script.txt"), "w", encoding="utf-8") as f:
        f.write(script_text)

    if sync_mempalace:
        sync_to_mempalace(manifest.source_file, manifest, scene_directions, generation_prompts, alias_groups)

    directed = sum(1 for d in scene_directions if d.get("directed_by"))
    logger.info(f"Scene direction complete: {directed}/{len(scene_directions)} scenes LLM-directed. Artifacts in {tier3_dir}/")
    return {"tier3_dir": tier3_dir, "scenes": len(scene_directions), "llm_directed": directed}


def main():
    parser = argparse.ArgumentParser(description="Caldera Engine Scene Director (Layer 3: Production Direction)")
    parser.add_argument("--manifest", type=str, required=True, help="Path to a Tier 1 enriched ManuscriptManifest JSON")
    parser.add_argument("--sync-mempalace", action="store_true", help="Sync production records into MemPalace (wings metadata + confirmed alias merges)")
    parser.add_argument("--sound-design-only", action="store_true", help="Run only the layered sound-design pass (writes sound_design.json, leaves direction artifacts alone)")
    parser.add_argument("--dramatize-only", action="store_true", help="Run only the Dramatist pass (writes dramatization.json)")
    parser.add_argument("--dramatization", type=str, default="full", choices=["faithful", "enhanced", "full"], help="Grounded-dramatization fidelity dial (Dramatist pass)")
    parser.add_argument("--dramatize-rounds", type=int, default=1, help="Dramatist passes per scene, merged by union-with-dedupe (coverage scales with rounds)")
    parser.add_argument("--design-characters-only", action="store_true", help="Run only AI-10 Character Designer (writes character_profiles.json + drawer visual profiles)")
    parser.add_argument("--qc-only", action="store_true", help="Run only AI-11 QC review over existing tier3 artifacts (writes qc_report.json)")
    args = parser.parse_args()

    if args.design_characters_only:
        print(f"\nProfiles written to {run_character_design(args.manifest)}")
        sys.exit(0)
    if args.qc_only:
        print(f"\nQC report written to {run_qc_review(args.manifest)}")
        sys.exit(0)

    if args.dramatize_only:
        out = run_dramatization(args.manifest, level=args.dramatization, rounds=args.dramatize_rounds)
        print(f"\nDramatization written to {out}")
        sys.exit(0)

    if args.sound_design_only:
        out = run_sound_design(args.manifest)
        print(f"\nSound design written to {out}")
        sys.exit(0)

    result = direct_manifest(args.manifest, sync_mempalace=args.sync_mempalace)
    print(f"\nDirected {result['llm_directed']}/{result['scenes']} scenes -> {result['tier3_dir']}/")
    sys.exit(0)


if __name__ == "__main__":
    main()
