#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Firespeaker Tier Preview -- "hear your book at every tier" from ONE scene.

The product funnel: tiers are chosen early, and a single-scene preview shows
what each buys before any full-book spend. Two parts:

  pick_trailer_scene  -- deterministic scorer (zero LLM calls) that picks the
                         scene where tier differences are most AUDIBLE:
                         dialogue density, speaker variety, vocalizations,
                         sound opportunities, and a preview-friendly length.
  render_tier_preview -- renders that scene at tier 1 (one narrator), tier 2
                         (attributed cast voices), or tier 3 (full production
                         lane using the crew's existing per-scene artifacts).
                         Results are cached by (book, tier, scene); replays
                         are free.

Preview cost by design: tier 1 = 0 LLM calls; tier 2/3 reuse existing
enrichment/direction artifacts (a book that was never enriched simply can't
preview above tier 1 yet -- the endpoint says so rather than silently
downgrading).
"""

import json
import logging
import os
import re
import subprocess
import threading
from typing import Any, Dict, List, Optional

from src.console_api import (
    _load_json, _safe_book, _tier1_dir, _tier3_dir,
    apply_speaker_overrides, load_speaker_overrides,
)

logger = logging.getLogger("TierPreview")

PREVIEW_DIR = "scratch/tier_previews"
_SYNTH_LOCK = threading.Lock()   # XTTS is not thread-safe

_SOUND_WORDS = re.compile(
    r"\b(bang|crash|scratch|scritch|rattl|thud|whistl|roar|rustl|creak|clatter|"
    r"splash|knock|slam|hiss|howl|thunder|footstep|hoof|bell|gunshot|scream)\w*",
    re.IGNORECASE)


def pick_trailer_scene(book: str) -> Optional[Dict[str, Any]]:
    """Deterministic 'demo reel' pick. A random scene undersells tier 2/3;
    this maximizes what a listener can HEAR differ between tiers."""
    book = _safe_book(book)
    if not book:
        return None
    t1 = _tier1_dir(book)
    payloads = _load_json(os.path.join(t1, "loop4_lines_enriched.json")) \
        or _load_json(os.path.join(t1, "loop4_lines.json")) or []
    best = None
    for p in payloads:
        lines = p.get("lines", [])
        if not lines:
            continue
        dialogue = [l for l in lines if l.get("segment_type") == "dialogue"]
        speakers = {l.get("character") for l in dialogue} - {"Narrator", None}
        vocal = sum(1 for l in lines if l.get("utterance_type") == "vocalization")
        text = " ".join(l.get("text", "") for l in lines)
        sound_hits = len(_SOUND_WORDS.findall(text))
        words = len(text.split())
        # preview sweet spot ~120-600 words (roughly 1-4 minutes read aloud)
        length_fit = 1.0 if 120 <= words <= 600 else (0.5 if words < 120 else max(0.2, 600 / words))
        score = (len(dialogue) * 2 + len(speakers) * 6 + vocal * 4 + sound_hits * 3) * length_fit
        entry = {"scene_id": p.get("scene_id"), "score": round(score, 1),
                 "dialogue": len(dialogue), "speakers": sorted(s for s in speakers if s),
                 "vocalizations": vocal, "sound_words": sound_hits, "words": words}
        if best is None or score > best["score"]:
            best = entry
    return best


def _scene_lines(book: str, scene_id: str) -> Optional[List[Dict[str, Any]]]:
    t1 = _tier1_dir(book)
    payloads = _load_json(os.path.join(t1, "loop4_lines_enriched.json")) \
        or _load_json(os.path.join(t1, "loop4_lines.json")) or []
    p = next((p for p in payloads if p.get("scene_id") == scene_id), None)
    if not p:
        return None
    lines = [dict(l) for l in p.get("lines", [])]
    apply_speaker_overrides(lines, load_speaker_overrides(book))
    return lines


def _run_ffmpeg(args: List[str]) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "quiet", *args], check=True)


def _concat_with_padding(line_wavs, out_path: str) -> None:
    """Voice-only assembly: line wavs in order, each followed by its own
    post_padding silence (same approach as production_mixer.mix_tier1)."""
    import wave as _wave
    with _wave.open(line_wavs[0][0], "rb") as w:
        rate, channels = w.getframerate(), w.getnchannels()
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    silence_cache: Dict[int, str] = {}

    def _silence(ms: int) -> str:
        if ms not in silence_cache:
            p = os.path.join(PREVIEW_DIR, f"_silence_{ms}ms_{rate}.wav")
            _run_ffmpeg(["-f", "lavfi", "-i", f"anullsrc=r={rate}:cl={'mono' if channels == 1 else 'stereo'}",
                         "-t", f"{ms / 1000.0:.3f}", "-c:a", "pcm_s16le", p])
            silence_cache[ms] = p
        return silence_cache[ms]

    listfile = out_path + ".txt"
    with open(listfile, "w") as f:
        for wav, line in line_wavs:
            f.write(f"file '{os.path.abspath(wav)}'\n")
            pad = int(line.get("post_padding_ms") or 0)
            if pad > 0:
                f.write(f"file '{os.path.abspath(_silence(pad))}'\n")
    _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", listfile, "-c:a", "pcm_s16le", "-ar", str(rate), out_path])
    os.remove(listfile)


def render_tier_preview(book: str, tier: int, scene_id: Optional[str] = None,
                        force: bool = False) -> Optional[Dict[str, Any]]:
    """Render the trailer scene at the requested tier. Returns
    {wav, scene_id, tier, cached, [note]} or an {'error': ...} dict when the
    tier's prerequisite artifacts don't exist for this book."""
    book = _safe_book(book)
    if not book or tier not in (1, 2, 3):
        return None
    pick = None
    if not scene_id:
        pick = pick_trailer_scene(book)
        if not pick:
            return {"error": "No line artifacts for this book -- run ingestion first."}
        scene_id = pick["scene_id"]
    if not re.fullmatch(r"[A-Za-z0-9_]+", scene_id or ""):
        return None

    os.makedirs(PREVIEW_DIR, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_\-]", "", book)
    out = os.path.join(PREVIEW_DIR, f"{slug}_tier{tier}_{scene_id}.wav")
    if not force and os.path.exists(out) and os.path.getsize(out) > 0:
        return {"wav": out, "scene_id": scene_id, "tier": tier, "cached": True, "pick": pick}

    lines = _scene_lines(book, scene_id)
    if not lines:
        return {"error": f"Scene {scene_id} not found in line artifacts."}

    # Tier 2/3 need real attribution; a never-enriched book has nothing to cast.
    if tier >= 2:
        has_attr = any(l.get("segment_type") == "dialogue"
                       and str(l.get("attribution_method", "Tier 1 Default")) != "Tier 1 Default"
                       for l in lines)
        if not has_attr:
            return {"error": "This book has no Tier 2 enrichment yet -- run "
                             "--enable-llm-enrichment first (tier 1 preview is available now)."}

    direction = sound_design = None
    sfx_cues: List[Dict[str, Any]] = []
    if tier == 3:
        t3 = _tier3_dir(book)
        directions = _load_json(os.path.join(t3, "production_script.json")) or []
        direction = next((d for d in directions if d.get("scene_id") == scene_id), None)
        if not direction:
            return {"error": "This book has no Tier 3 direction yet -- run the "
                             "scene_director crew first (tier 1/2 previews are available)."}
        designs = _load_json(os.path.join(t3, "sound_design.json")) or []
        sound_design = next((d for d in designs if d.get("scene_id") == scene_id), None)
        for entry in _load_json(os.path.join(_tier1_dir(book), "loopE_llm_sfx_cues.json")) or []:
            if entry.get("scene_id") == scene_id:
                sfx_cues = entry.get("sfx_cues", [])

    if tier == 1:
        for l in lines:
            l["character"] = "Narrator"
            l["speaker_id"] = "char_narrator"

    with _SYNTH_LOCK:
        from src.production_mixer import resolve_line_wavs, assemble_scene
        from src.voice_synthesizer import VoiceSynthesizer
        synth = VoiceSynthesizer()
        # every speaker needs a drawer; unseen ones get a distinct builtin voice
        # via the name-hash pool (same policy as mix_production's minor cast)
        for char in sorted({l["character"] for l in lines}):
            if not synth.palace.get_character_drawer(char):
                logger.info(f"Preview: registering drawer for '{char}'.")
                synth.palace.register_character(
                    character_name=char,
                    voice_ref_path="data/voice_references/narrator_mono.wav",
                    speed=1.0, pitch=0.0)
        line_wavs = resolve_line_wavs(lines, synth)
        if tier == 3:
            from src.mix_timeline import load_mix_overrides
            assemble_scene(scene_id, line_wavs, json.loads(json.dumps(direction)),
                           sfx_cues, out, sound_design=json.loads(json.dumps(sound_design)) if sound_design else None,
                           mix_overrides=load_mix_overrides(book).get(scene_id))
        else:
            _concat_with_padding(line_wavs, out)

    if not (os.path.exists(out) and os.path.getsize(out) > 0):
        return {"error": "Preview render produced no audio."}
    return {"wav": out, "scene_id": scene_id, "tier": tier, "cached": False, "pick": pick}
