#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Firespeaker Review Console API (Phase 1: read-only).

Pure functions over the pipeline's file artifacts -- no pipeline imports, no
LLM calls, no mutation. The console reads what the loops, the director crew,
and the mixer wrote; corrections/overrides come in Phase 2 and will write only
override/confirmation files, never the artifacts themselves.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ConsoleAPI")

PIPELINE_ROOT = "data/corpus/pipeline"
PROGRESS_FILE = "data/analysis_progress.json"

# Directories audio audition may serve from (path-traversal guard).
AUDIO_WHITELIST = (
    "scratch/pipeline_workspace/outputs",
    "data/generated_audio",
    "data/voice_marketplace",
    "data/voice_references",
    "scratch",
)


def _load_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Console: failed to read {path}: {e}")
        return None


def _tier1_dir(book: str) -> str:
    return os.path.join(PIPELINE_ROOT, book, "tier1")


def _tier3_dir(book: str) -> str:
    return os.path.join(PIPELINE_ROOT, book, "tier3")


def _safe_book(book: str) -> Optional[str]:
    """Book names are directory names under PIPELINE_ROOT -- nothing else."""
    if not book or book in (".", "..") or "/" in book or "\\" in book:
        return None
    if not os.path.isdir(os.path.join(PIPELINE_ROOT, book)):
        return None
    return book


def _load_lines(book: str) -> List[Dict[str, Any]]:
    """Per-scene line payloads; enriched artifact wins over the raw one."""
    t1 = _tier1_dir(book)
    data = _load_json(os.path.join(t1, "loop4_lines_enriched.json"))
    if data is None:
        data = _load_json(os.path.join(t1, "loop4_lines.json")) or []
    return data


def list_books() -> List[Dict[str, Any]]:
    """Every ingested book with an inventory of which artifacts exist."""
    books = []
    if not os.path.isdir(PIPELINE_ROOT):
        return books
    for name in sorted(os.listdir(PIPELINE_ROOT)):
        t1, t3 = _tier1_dir(name), _tier3_dir(name)
        if not os.path.isdir(t1):
            continue
        scenes = _load_json(os.path.join(t1, "loop3_scenes.json")) or []
        entry = {
            "book": name,
            "scenes": len(scenes),
            "artifacts": {
                "enriched": os.path.exists(os.path.join(t1, "loop4_lines_enriched.json")),
                "alias_merges": os.path.exists(os.path.join(t1, "loopE_llm_alias_merges.json")),
                "sfx_cues": os.path.exists(os.path.join(t1, "loopE_llm_sfx_cues.json")),
                "production_script": os.path.exists(os.path.join(t3, "production_script.json")),
                "sound_design": os.path.exists(os.path.join(t3, "sound_design.json")),
                "dramatization": os.path.exists(os.path.join(t3, "dramatization.json")),
                "character_profiles": os.path.exists(os.path.join(t3, "character_profiles.json")),
                "book_bible": os.path.exists(os.path.join(t3, "book_bible.json")),
            },
        }
        books.append(entry)
    return books


def book_tree(book: str) -> Optional[Dict[str, Any]]:
    """Part -> chapter -> scene hierarchy with per-scene review stats.

    Scene ids encode their ancestry (part_p1_c2_s3), so the tree is derived from
    the loop artifacts directly; line text stays out of this payload -- a
    229-scene novel must load fast, scene_detail() carries the heavy data.
    """
    book = _safe_book(book)
    if not book:
        return None
    t1 = _tier1_dir(book)
    parts = _load_json(os.path.join(t1, "loop1_parts.json")) or []
    chapters = _load_json(os.path.join(t1, "loop2_chapters.json")) or []
    scenes = _load_json(os.path.join(t1, "loop3_scenes.json")) or []
    line_payloads = _load_lines(book)

    stats_by_scene: Dict[str, Dict[str, Any]] = {}
    for payload in line_payloads:
        lines = payload.get("lines", [])
        dialogue = [l for l in lines if l.get("segment_type") == "dialogue"]
        enriched = [l for l in dialogue if str(l.get("attribution_method", "")) != "Tier 1 Default"]
        reviewed = [l for l in dialogue if "+reviewed" in str(l.get("attribution_method", ""))]
        speakers = sorted({l.get("character", "?") for l in dialogue} - {"Narrator"})
        stats_by_scene[payload.get("scene_id", "")] = {
            "lines": len(lines),
            "dialogue": len(dialogue),
            "enriched": len(enriched),
            "reviewed": len(reviewed),
            "speakers": speakers,
        }

    scene_nodes: Dict[str, List[Dict[str, Any]]] = {}
    for sc in scenes:
        sid = sc.get("scene_id", "")
        chap_key = sid.rsplit("_s", 1)[0] if "_s" in sid else ""
        text = sc.get("text_block", "")
        node = {
            "scene_id": sid,
            "boundary_source": sc.get("boundary_source", "regex"),
            "chars": len(text),
            "excerpt": " ".join(text[:140].split()),
            **stats_by_scene.get(sid, {"lines": 0, "dialogue": 0, "enriched": 0, "reviewed": 0, "speakers": []}),
        }
        scene_nodes.setdefault(chap_key, []).append(node)

    chapter_nodes: Dict[str, List[Dict[str, Any]]] = {}
    for ch in chapters:
        cid = ch.get("chapter_id", "")
        part_key = cid.rsplit("_c", 1)[0] if "_c" in cid else ""
        chapter_nodes.setdefault(part_key, []).append({
            "chapter_id": cid,
            "title": ch.get("title", cid),
            "scenes": scene_nodes.get(cid, []),
        })

    tree = []
    for p in parts:
        pid = p.get("part_id", "")
        tree.append({
            "part_id": pid,
            "title": p.get("title", pid),
            "chapters": chapter_nodes.get(pid, []),
        })

    merges = _load_json(os.path.join(t1, "loopE_llm_alias_merges.json")) or []
    bible = _load_json(os.path.join(_tier3_dir(book), "book_bible.json"))
    return {"book": book, "parts": tree, "alias_merges": merges, "book_bible": bible}


def _line_wav_path(line: Dict[str, Any]) -> Optional[str]:
    """Mirror of production_mixer.resolve_line_wavs cache naming; only returns
    a path that actually exists (audition is best-effort)."""
    char_slug = re.sub(r"[^a-zA-Z0-9_\-]", "", line.get("character", ""))
    emotion = line.get("emotion", "Neutral")
    p = os.path.join("scratch/pipeline_workspace/outputs",
                     f"line_{line.get('line_id')}_{char_slug}_tier3_{emotion}.wav")
    return p if os.path.exists(p) else None


def scene_detail(book: str, scene_id: str) -> Optional[Dict[str, Any]]:
    """Everything the console shows for one scene: the attributed transcript
    with provenance, plus the tier-3 production lane (direction, sound design,
    dramatized inserts, grounded SFX cues) when the crew has run."""
    book = _safe_book(book)
    if not book or not re.fullmatch(r"[A-Za-z0-9_]+", scene_id or ""):
        return None
    t1, t3 = _tier1_dir(book), _tier3_dir(book)

    payload = next((p for p in _load_lines(book) if p.get("scene_id") == scene_id), None)
    if payload is None:
        return None
    overrides = load_speaker_overrides(book)
    lines = []
    for i, l in enumerate(payload.get("lines", [])):
        l = dict(l)
        apply_speaker_overrides([l], overrides)
        lines.append({
            "index": i,
            "line_id": l.get("line_id"),
            "character": l.get("character"),
            "segment_type": l.get("segment_type"),
            "text": l.get("text"),
            "emotion": l.get("emotion"),
            "confidence": l.get("confidence"),
            "attribution_method": l.get("attribution_method"),
            "utterance_type": l.get("utterance_type", "speech"),
            "wav": _line_wav_path(l),
        })

    scene_text = next((s.get("text_block", "") for s in (_load_json(os.path.join(t1, "loop3_scenes.json")) or [])
                       if s.get("scene_id") == scene_id), "")

    def _for_scene(path: str) -> Optional[Dict[str, Any]]:
        data = _load_json(path)
        if isinstance(data, list):
            return next((d for d in data if d.get("scene_id") == scene_id), None)
        return None

    sfx_entry = _for_scene(os.path.join(t1, "loopE_llm_sfx_cues.json"))
    return {
        "book": book,
        "scene_id": scene_id,
        "scene_text": scene_text,
        "allowed_speakers": book_speakers(book),
        "lines": lines,
        "direction": _for_scene(os.path.join(t3, "production_script.json")),
        "sound_design": _for_scene(os.path.join(t3, "sound_design.json")),
        "dramatization": _for_scene(os.path.join(t3, "dramatization.json")),
        "sfx_cues": (sfx_entry or {}).get("sfx_cues", []),
    }


def load_speaker_overrides(book: str) -> Dict[str, Dict[str, Any]]:
    """Durable human attribution corrections, keyed by line_id (a content hash,
    stable across re-runs while the line's text is unchanged; orphaned entries
    simply stop matching). Same human-veto pattern as confirmed_merges."""
    book = _safe_book(book)
    if not book:
        return {}
    return _load_json(os.path.join(_tier1_dir(book), "speaker_overrides.json")) or {}


def save_speaker_override(book: str, line_id: str, character: str,
                          scene_id: str = "") -> Optional[Dict[str, Any]]:
    """Set (or clear, with character='') one line's human speaker correction."""
    book = _safe_book(book)
    if not book or not re.fullmatch(r"[0-9a-f]{16}", line_id or ""):
        return None
    character = (character or "").strip()[:60]
    path = os.path.join(_tier1_dir(book), "speaker_overrides.json")
    overrides = _load_json(path) or {}
    if character:
        overrides[line_id] = {"character": character, "scene_id": scene_id,
                              "corrected_by": "human", "at": __import__("time").time()}
    else:
        overrides.pop(line_id, None)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2)
    os.replace(tmp, path)
    return {"overrides": len(overrides), "line_id": line_id, "character": character or None}


def apply_speaker_overrides(lines: List[Dict[str, Any]], overrides: Dict[str, Dict[str, Any]]) -> int:
    """In-place application of human corrections to line dicts; returns count.
    The human's word is final: it beats attribution, review, and alias merge."""
    applied = 0
    for l in lines:
        ov = overrides.get(str(l.get("line_id")))
        if ov and ov.get("character"):
            l["character"] = ov["character"]
            l["speaker_id"] = "char_" + re.sub(r"\s+", "_", ov["character"].lower())
            l["attribution_method"] = "human_override"
            l["confidence"] = 1.0
            l["speaker_locked"] = True
            applied += 1
    return applied


def book_speakers(book: str) -> List[str]:
    """Every speaker attributed anywhere in the book (for the correction picker)."""
    speakers = set()
    for payload in _load_lines(book):
        for l in payload.get("lines", []):
            if l.get("segment_type") == "dialogue" and l.get("character"):
                speakers.add(l["character"])
    return sorted(speakers | {"Narrator"})


def progress() -> Dict[str, Any]:
    return _load_json(PROGRESS_FILE) or {}


def resolve_audio(rel_path: str) -> Optional[str]:
    """Whitelisted audition/download: the file must resolve inside one of the
    allowed directories and be an audio deliverable. Returns abs path or None."""
    if not rel_path or not rel_path.endswith((".wav", ".m4b", ".mp3")):
        return None
    abs_path = os.path.abspath(rel_path)
    for allowed in AUDIO_WHITELIST:
        if abs_path.startswith(os.path.abspath(allowed) + os.sep):
            return abs_path if os.path.exists(abs_path) else None
    return None
