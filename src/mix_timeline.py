#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Caldera Engine Mix Timeline -- the scene as a serialized DAW session.

Read side of the advanced mixer: every lane the assembly actually mixes
(voice / music / ambience / sfx), with timestamps computed the same
line-anchored way assemble_scene computes them, plus audition paths into the
last render's persisted workspace assets. Write side is mix_overrides.json
({book}/tier3/), the same durable-override pattern as line text and speaker
corrections: artifacts stay the crew's pure output, human mix decisions live
in their own layer, and every re-render honors them.

Anchoring is by LINE, never by absolute seconds -- when a voice re-synthesizes
longer, every downstream event slides with the narration instead of orphaning.
"""

import json
import logging
import os
import re
import wave
from typing import Any, Dict, List, Optional

from src.console_api import (
    _load_json, _safe_book, _tier1_dir, _tier3_dir,
    apply_speaker_overrides, load_speaker_overrides,
)

logger = logging.getLogger("MixTimeline")

WORKSPACE = "scratch/pipeline_workspace/tier3_mix"   # assemble_scene's scene dirs
LINE_OUTPUTS = "scratch/pipeline_workspace/outputs"  # resolve_line_wavs cache


def _overrides_path(book: str) -> str:
    return os.path.join(_tier3_dir(book), "mix_overrides.json")


def load_mix_overrides(book: str) -> Dict[str, Any]:
    return _load_json(_overrides_path(book)) or {}


def save_mix_override(book: str, scene_id: str, target: str, key: str = "",
                      mute: Optional[bool] = None, gain_db: Optional[float] = None,
                      nudge_s: Optional[float] = None, prompt: Optional[str] = None,
                      variant: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """target: 'event' (key = event:<i> or stinger:<i>) or 'lane'
    (key = voice|music|ambience|sfx). Passing all-None fields clears the entry.

    prompt: author's replacement text for the generated asset (an event's sound,
    the music bed's style, the ambience). variant: 'take number' — appended to
    the generation prompt as a nonce, so bumping it rolls a NEW generation under
    a new cache key while every previous take stays cached (reverting is free)."""
    book = _safe_book(book)
    if not book or target not in ("event", "lane"):
        return None
    if target == "lane" and key not in ("voice", "music", "ambience", "sfx"):
        return None
    if target == "event" and not re.fullmatch(r"(event|stinger|cue):\d+", key or ""):
        return None
    all_ov = load_mix_overrides(book)
    scene_ov = all_ov.setdefault(scene_id, {"events": {}, "lanes": {}})
    bucket = scene_ov["events"] if target == "event" else scene_ov["lanes"]
    entry = {}
    if mute is not None:
        entry["mute"] = bool(mute)
    if gain_db is not None:
        entry["gain_db"] = max(-24.0, min(24.0, float(gain_db)))
    if nudge_s is not None and target == "event":
        entry["nudge_s"] = max(-5.0, min(5.0, float(nudge_s)))
    if prompt is not None and str(prompt).strip():
        entry["prompt"] = str(prompt).strip()[:200]
    if variant is not None and int(variant) > 0:
        entry["variant"] = max(0, min(99, int(variant)))
    if entry:
        bucket[key] = entry
    else:
        bucket.pop(key, None)
    os.makedirs(_tier3_dir(book), exist_ok=True)
    tmp = _overrides_path(book) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(all_ov, f, indent=2)
    os.replace(tmp, _overrides_path(book))
    return {"scene_id": scene_id, "target": target, "key": key, "entry": entry or None}


def _wav_dur(path: str) -> Optional[float]:
    try:
        with wave.open(path) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


def _line_wav(line: Dict[str, Any], fingerprint: str) -> Optional[str]:
    char_slug = re.sub(r"[^a-zA-Z0-9_\-]", "", line.get("character", ""))
    emotion = line.get("emotion", "Neutral")
    p = os.path.join(LINE_OUTPUTS, f"line_{line.get('line_id')}_{char_slug}_tier3_{emotion}_v{fingerprint}.wav")
    if os.path.exists(p):
        return p
    legacy = os.path.join(LINE_OUTPUTS, f"line_{line.get('line_id')}_{char_slug}_tier3_{emotion}.wav")
    return legacy if os.path.exists(legacy) else None


class _PalaceShim:
    """production_mixer._voice_fingerprint only touches synth.palace; loading the
    real synthesizer (torch, XTTS) for a READ api would be absurd."""

    def __init__(self):
        from src.spatial_memory import MemPalace
        self.palace = MemPalace()


def scene_timeline(book: str, scene_id: str) -> Optional[Dict[str, Any]]:
    book = _safe_book(book)
    if not book or not re.fullmatch(r"[A-Za-z0-9_]+", scene_id or ""):
        return None
    t1, t3 = _tier1_dir(book), _tier3_dir(book)
    payload = next((p for p in (_load_json(os.path.join(t1, "loop4_lines_enriched.json")) or
                                _load_json(os.path.join(t1, "loop4_lines.json")) or [])
                    if p.get("scene_id") == scene_id), None)
    if payload is None:
        return None
    lines = [dict(l) for l in payload.get("lines", [])]
    apply_speaker_overrides(lines, load_speaker_overrides(book))

    from src.production_mixer import _voice_fingerprint
    shim = _PalaceShim()
    fp_memo: Dict[str, str] = {}

    # Voice lane: cumulative offsets from cached wav durations; unsynthesized
    # lines get an honest word-rate estimate and say so.
    voice_blocks = []
    t = 0.0
    for i, l in enumerate(lines):
        fp = _voice_fingerprint(shim, l.get("character", ""), fp_memo)
        wav = _line_wav(l, fp)
        dur = _wav_dur(wav) if wav else None
        estimated = dur is None
        if dur is None:
            dur = max(1.0, len(l.get("text", "").split()) * 0.42)
        pad = int(l.get("post_padding_ms", 250)) / 1000.0
        voice_blocks.append({
            "index": i, "line_id": l.get("line_id"), "character": l.get("character"),
            "text": l.get("text", "")[:90], "start_s": round(t, 3),
            "end_s": round(t + dur, 3), "estimated": estimated, "wav": wav,
        })
        t += dur + pad
    total = round(t, 3)

    def _line_end(idx: int) -> float:
        return voice_blocks[idx]["end_s"] if 0 <= idx < len(voice_blocks) else 0.0

    scene_ws = os.path.join(WORKSPACE, scene_id)

    def _ws(name: str) -> Optional[str]:
        p = os.path.join(scene_ws, name)
        return p if os.path.exists(p) else None

    directions = _load_json(os.path.join(t3, "production_script.json")) or []
    direction = next((d for d in directions if d.get("scene_id") == scene_id), None) or {}
    music = direction.get("music", {})
    music_lane = {
        "base_mood": music.get("base_mood"), "style": music.get("style"),
        "bed_wav": _ws("music_bed.wav"),
        "stingers": [{
            "key": f"stinger:{i}", "after_line_index": s.get("after_line_index"),
            "start_s": round(_line_end(s.get("after_line_index", -1)), 3),
            "description": s.get("description") or s.get("mood") or "stinger",
            "wav": _ws(f"stinger_{i}.wav"),
        } for i, s in enumerate(music.get("stingers", []))],
        "state_events": [{
            "after_line_index": ev.get("after_line_index"),
            "start_s": round(_line_end(ev.get("after_line_index", -1)), 3),
            "action": ev.get("action"), "new_style": ev.get("new_style"),
        } for ev in music.get("events", [])],
    }

    designs = _load_json(os.path.join(t3, "sound_design.json")) or []
    design = next((d for d in designs if d.get("scene_id") == scene_id), None) or {}
    sfx_lane = [{
        "key": f"event:{i}", "name": ev.get("name"),
        "category": ev.get("category"), "anchor_line_index": ev.get("anchor_line_index"),
        "start_s": round(_line_end(ev.get("anchor_line_index", -1)), 3),
        "source_text": (ev.get("source_text") or "")[:100],
        "layers": [{"component": l.get("component"), "timing": l.get("timing"),
                    "level": l.get("level")} for l in ev.get("layers", [])],
        "wav": _ws(f"composite_{i}.wav"),
    } for i, ev in enumerate(design.get("events", []))]

    return {
        "book": book, "scene_id": scene_id, "duration_s": total,
        "voice": voice_blocks,
        "music": music_lane,
        "ambience": {"components": design.get("continuous_ambience", []),
                     "wav": _ws("ambience.wav")},
        "sfx": sfx_lane,
        "rendered_scene_wav": _ws(f"{scene_id}_mixed.wav") or _ws("mixed.wav"),
        "overrides": load_mix_overrides(book).get(scene_id, {"events": {}, "lanes": {}}),
    }
