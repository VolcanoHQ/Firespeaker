#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Caldera Engine Review Console API (Phase 1: read-only).

Pure functions over the pipeline's file artifacts -- no pipeline imports, no
LLM calls, no mutation. The console reads what the loops, the director crew,
and the mixer wrote; corrections/overrides come in Phase 2 and will write only
override/confirmation files, never the artifacts themselves.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.book_structure import (
    SplitSectionSpec,
    change_content_type,
    remove_section,
    rename_section,
    reorder_section,
    merge_sections,
    save_book_structure,
    split_section,
)
from src.book_structure_adapter import (
    chapter_lookup,
    load_line_payloads as adapter_load_line_payloads,
    load_structure,
    ordered_chapters,
    ordered_parts,
    ordered_scene_ids,
    ordered_scenes,
    resolve_scene,
    scene_lines,
    scene_text as structure_scene_text,
    structure_readiness,
    structure_to_console_tree,
    structure_to_gui_hierarchy,
    structure_to_manifest,
)

logger = logging.getLogger("ConsoleAPI")

PIPELINE_ROOT = "data/corpus/pipeline"
PROGRESS_FILE = "data/analysis_progress.json"
_MARKETPLACE = None

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


def _book_structure_path(book: str) -> str:
    return os.path.join(_tier1_dir(book), "book_structure.json")


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


def _get_marketplace():
    global _MARKETPLACE
    if _MARKETPLACE is None:
        from src.voice_marketplace import VoiceMarketplace
        _MARKETPLACE = VoiceMarketplace()
    return _MARKETPLACE


def _get_drawer_info(character: str) -> Optional[Dict[str, Any]]:
    from src.spatial_memory import MemPalace

    palace = MemPalace(use_chroma=False)
    try:
        return palace.get_character_drawer(character)
    finally:
        palace.close()


def _character_profiles(book: str) -> Dict[str, Dict[str, Any]]:
    profiles = _load_json(os.path.join(_tier3_dir(book), "character_profiles.json")) or []
    return {
        item.get("name", ""): item
        for item in profiles
        if isinstance(item, dict) and item.get("name")
    }


def _character_voice_query(book: str, character: str, explicit_query: str = "") -> str:
    if explicit_query.strip():
        return explicit_query.strip()
    profile = _character_profiles(book).get(character, {})
    profile_description = str(profile.get("visual_description") or profile.get("description") or "").strip()
    if profile_description:
        return f"{character}: {profile_description}"
    return f"{character} audiobook character voice"


def _write_json(path: str, payload: Any) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)
    return path


def _resolve_source_path(book: str, source_hint: str = "") -> Optional[str]:
    from src import render_job

    source = render_job.find_source(book)
    if source:
        return source
    if source_hint and os.path.exists(source_hint):
        return source_hint
    if source_hint:
        hinted = os.path.join("data", "uploads", os.path.basename(source_hint))
        if os.path.exists(hinted):
            return hinted
    return None


def _renumber_lines(lines: List[Dict[str, Any]], *, chapter_num: int, scene_num: int) -> List[Dict[str, Any]]:
    from src.models import ScriptLine

    renumbered = []
    for line_number, line in enumerate(lines, 1):
        line_copy = dict(line)
        line_copy["chapter"] = chapter_num
        line_copy["scene"] = scene_num
        line_copy["line_number"] = line_number
        renumbered.append(ScriptLine.model_validate(line_copy).model_dump())
    return renumbered


def _refresh_line_payloads(book: str, structure) -> List[Dict[str, Any]]:
    from src.tier_1_parser import parse_tier_1_lines

    existing_payloads = adapter_load_line_payloads(book)
    existing_index = {payload.get("scene_id", ""): payload.get("lines", []) for payload in existing_payloads}
    refreshed_payloads: List[Dict[str, Any]] = []
    scene_counter = 0

    for part_index, part in enumerate(ordered_parts(structure), 1):
        for chapter_index, chapter in enumerate(ordered_chapters(structure, part), 1):
            for scene in ordered_scenes(structure, chapter):
                scene_counter += 1
                scene_key = scene.legacy_id or scene.section_id
                scene_body = structure_scene_text(scene, structure)
                existing_lines = existing_index.get(scene_key)
                if existing_lines:
                    lines = _renumber_lines(existing_lines, chapter_num=chapter_index, scene_num=scene_counter)
                else:
                    lines = [
                        line.model_dump()
                        for line in parse_tier_1_lines(scene_body, part_index, chapter_index, scene_counter)
                    ]
                scene.metadata["text_block"] = scene_body
                scene.metadata["source_chars"] = len(scene_body)
                refreshed_payloads.append({"scene_id": scene_key, "lines": lines})

    return refreshed_payloads


def _mark_structure_analysis_refreshed(structure, line_payloads: List[Dict[str, Any]]):
    updated = structure.model_copy(deep=True)
    now = datetime.now(timezone.utc).isoformat()
    payload_scene_ids = {payload.get("scene_id", "") for payload in line_payloads}
    depth_by_id = {None: -1}

    def _depth(section_id: str | None) -> int:
        if section_id in depth_by_id:
            return depth_by_id[section_id]
        parent = next((section.parent_id for section in updated.sections if section.section_id == section_id), None)
        depth_by_id[section_id] = _depth(parent) + 1
        return depth_by_id[section_id]

    for section in updated.sections:
        if section.status != "active":
            continue
        if section.content_type == "scene":
            scene_key = section.legacy_id or section.section_id
            section.artifacts.analysis_valid = scene_key in payload_scene_ids
            section.updated_at = now

    active_sections = [section for section in updated.sections if section.status == "active"]
    for section in sorted(active_sections, key=lambda item: _depth(item.section_id), reverse=True):
        children = [child for child in active_sections if child.parent_id == section.section_id]
        if children:
            section.artifacts.analysis_valid = all(child.artifacts.analysis_valid for child in children)
            section.updated_at = now

    updated.updated_at = now
    return updated


def _invalidate_tier3_artifacts(book: str) -> List[str]:
    tier3_dir = _tier3_dir(book)
    cleared = []
    for filename in (
        "production_script.json",
        "sound_design.json",
        "dramatization.json",
        "character_profiles.json",
        "book_bible.json",
    ):
        path = os.path.join(tier3_dir, filename)
        if os.path.exists(path):
            os.remove(path)
            cleared.append(path)
    return cleared


def get_book_structure(book: str) -> Optional[Dict[str, Any]]:
    book = _safe_book(book)
    if not book:
        return None
    structure = load_structure(book)
    return {
        "book": book,
        "structure": structure.model_dump(),
        "readiness": structure_readiness(structure, require_analysis=True),
    }


def apply_structure_edit(book: str, action: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    book = _safe_book(book)
    if not book:
        return None
    structure = load_structure(book)
    if action == "rename_section":
        updated = rename_section(structure, payload.get("section_id", ""), str(payload.get("title", "")))
    elif action == "reorder_section":
        updated = reorder_section(structure, payload.get("section_id", ""), int(payload.get("order", 0)))
    elif action == "remove_section":
        updated = remove_section(structure, payload.get("section_id", ""))
    elif action == "change_content_type":
        updated = change_content_type(structure, payload.get("section_id", ""), str(payload.get("content_type", "")))
    elif action == "merge_sections":
        updated = merge_sections(
            structure,
            payload.get("section_ids", []),
            title=payload.get("title"),
            content_type=payload.get("content_type"),
        )
    elif action == "split_section":
        specs = [SplitSectionSpec.model_validate(item) for item in payload.get("sections", [])]
        updated = split_section(structure, payload.get("section_id", ""), specs)
    else:
        raise ValueError(f"Unknown structure action: {action}")

    save_book_structure(updated, _book_structure_path(book))
    return {
        "book": book,
        "action": action,
        "structure": updated.model_dump(),
        "readiness": structure_readiness(updated, require_analysis=True),
    }


def refresh_book_structure(book: str) -> Optional[Dict[str, Any]]:
    book = _safe_book(book)
    if not book:
        return None
    structure = load_structure(book)
    source_path = _resolve_source_path(book, structure.source_file)
    if not source_path:
        raise FileNotFoundError(f"Could not locate source manuscript for '{book}'")

    refreshed_payloads = _refresh_line_payloads(book, structure)
    loop4_path = _write_json(os.path.join(_tier1_dir(book), "loop4_lines.json"), refreshed_payloads)
    enriched_path = _write_json(os.path.join(_tier1_dir(book), "loop4_lines_enriched.json"), refreshed_payloads)
    _write_json(os.path.join(_tier1_dir(book), "loopE_llm_cleancheck.json"), [])
    _write_json(os.path.join(_tier1_dir(book), "loopE_llm_sfx_cues.json"), [])
    _write_json(os.path.join(_tier1_dir(book), "loopE_llm_alias_merges.json"), [])

    refreshed_structure = _mark_structure_analysis_refreshed(structure, refreshed_payloads)
    save_book_structure(refreshed_structure, _book_structure_path(book))

    hierarchy_data = structure_to_gui_hierarchy(refreshed_structure, line_payloads=refreshed_payloads)
    slug = re.sub(r"[^a-zA-Z0-9_\-]", "", book)
    cache_dir = os.path.join("data", "processed", slug, "Tier_1")
    hierarchy_cache = _write_json(os.path.join(cache_dir, "hierarchy.json"), hierarchy_data)

    from src.manuscript_profiler import ManuscriptProfiler

    profiler = ManuscriptProfiler(use_gpu=False, production_tier=1)
    profile_data = profiler.profile_book(source_path, hierarchy_data=hierarchy_data)
    profile_cache = os.path.join(cache_dir, "profile.json")
    profiler.save_profile(profile_data, profile_cache)
    cleared_tier3 = _invalidate_tier3_artifacts(book)

    return {
        "book": book,
        "structure": refreshed_structure.model_dump(),
        "readiness": structure_readiness(refreshed_structure, require_analysis=True),
        "hierarchy": hierarchy_data,
        "profile": profile_data,
        "artifacts": {
            "loop4_lines": loop4_path,
            "loop4_lines_enriched": enriched_path,
            "hierarchy_cache": hierarchy_cache,
            "profile_cache": profile_cache,
            "cleared_tier3": cleared_tier3,
        },
    }


def refresh_book_director(
    book: str,
    *,
    refresh_structure: bool = False,
    include_qc: bool = False,
    sync_mempalace: bool = False,
) -> Optional[Dict[str, Any]]:
    book = _safe_book(book)
    if not book:
        return None

    structure_refresh = refresh_book_structure(book) if refresh_structure else None
    structure = load_structure(book)
    readiness = structure_readiness(structure, require_analysis=True)
    if not readiness["ok"]:
        raise ValueError(
            "Canonical structure is stale for scene direction. Refresh structure analysis before "
            "rebuilding Tier 3 artifacts."
        )

    line_payloads = adapter_load_line_payloads(book)
    if not line_payloads:
        raise ValueError(f"No Tier 1 line artifacts are available for '{book}'")

    manifest = structure_to_manifest(structure, line_payloads=line_payloads)
    tier3_dir = _tier3_dir(book)
    os.makedirs(tier3_dir, exist_ok=True)
    manifest_path = _write_json(
        os.path.join(tier3_dir, "canonical_manifest.json"),
        manifest.model_dump(),
    )

    from src import scene_director

    direction_result = scene_director.direct_manifest(manifest_path, sync_mempalace=sync_mempalace)
    sound_design_path = scene_director.run_sound_design(manifest_path)
    dramatization_path = scene_director.run_dramatization(manifest_path)
    character_profiles_path = scene_director.run_character_design(
        manifest_path,
        sync_mempalace=sync_mempalace,
    )
    qc_report_path = scene_director.run_qc_review(manifest_path) if include_qc else None
    if not include_qc:
        stale_qc_path = os.path.join(tier3_dir, "qc_report.json")
        if os.path.exists(stale_qc_path):
            os.remove(stale_qc_path)

    return {
        "book": book,
        "structure": structure.model_dump(),
        "readiness": readiness,
        "structure_refresh": structure_refresh,
        "director": direction_result,
        "artifacts": {
            "manifest": manifest_path,
            "sound_design": sound_design_path,
            "dramatization": dramatization_path,
            "character_profiles": character_profiles_path,
            "qc_report": qc_report_path,
        },
    }


def list_books() -> List[Dict[str, Any]]:
    """Every ingested book with an inventory of which artifacts exist."""
    books = []
    if not os.path.isdir(PIPELINE_ROOT):
        return books
    for name in sorted(os.listdir(PIPELINE_ROOT)):
        t1, t3 = _tier1_dir(name), _tier3_dir(name)
        if not os.path.isdir(t1):
            continue
        try:
            structure = load_structure(name)
            scenes_count = len(ordered_scene_ids(structure))
        except Exception:
            scenes_count = len(_load_json(os.path.join(t1, "loop3_scenes.json")) or [])
        entry = {
            "book": name,
            "scenes": scenes_count,
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

    # New sources: manuscripts on disk that were never ingested. They carry no
    # artifacts yet -- the first render ingests them -- but the console must
    # show them or "drop a file in" is an invisible act.
    ingested = {b["book"] for b in books}
    from src.render_job import CORPUS_ROOTS, SOURCE_EXTS
    import glob as _glob
    for root in CORPUS_ROOTS:
        for ext in SOURCE_EXTS:
            for path in sorted(_glob.glob(os.path.join(root, f"*{ext}"))):
                stem = os.path.splitext(os.path.basename(path))[0]
                if stem in ingested:
                    continue
                ingested.add(stem)
                books.append({"book": stem, "scenes": 0, "new": True,
                              "source_file": path, "artifacts": {}})
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
    structure = load_structure(book)
    line_payloads = adapter_load_line_payloads(book)
    tree = structure_to_console_tree(
        structure,
        line_payloads=line_payloads,
        scene_overrides=load_scene_overrides(book),
    )
    merges = _load_json(os.path.join(t1, "loopE_llm_alias_merges.json")) or []
    bible = _load_json(os.path.join(_tier3_dir(book), "book_bible.json"))
    structure_summary = {
        "path": _book_structure_path(book),
        "version": structure.structure_version,
        "sections": len(structure.sections),
        "updated_at": structure.updated_at,
    }
    return {"book": book, "parts": tree, "alias_merges": merges, "book_bible": bible, "book_structure": structure_summary}


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
    structure = load_structure(book)
    payload_lines = scene_lines(structure, scene_id, line_payloads=adapter_load_line_payloads(book))
    if not payload_lines:
        return None
    overrides = load_speaker_overrides(book)
    lines = []
    for i, l in enumerate(payload_lines):
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
    section = resolve_scene(structure, scene_id)
    if section is None:
        return None
    scene_text = structure_scene_text(section, structure)

    def _for_scene(path: str) -> Optional[Dict[str, Any]]:
        data = _load_json(path)
        if isinstance(data, list):
            return next((d for d in data if d.get("scene_id") == scene_id), None)
        return None

    sfx_entry = _for_scene(os.path.join(t1, "loopE_llm_sfx_cues.json"))
    return {
        "book": book,
        "scene_id": scene_id,
        "omitted": bool(load_scene_overrides(book).get(scene_id, {}).get("omit")),
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


def load_scene_overrides(book: str) -> Dict[str, Dict[str, Any]]:
    """Structure-level human decisions, keyed by scene_id. Today: {"omit": true}
    keeps a scene out of every manifest/render while artifacts stay complete."""
    book = _safe_book(book)
    if not book:
        return {}
    return _load_json(os.path.join(_tier1_dir(book), "scene_overrides.json")) or {}


def save_scene_override(book: str, scene_id: str, omit: Optional[bool] = None) -> Optional[Dict[str, Any]]:
    book = _safe_book(book)
    if not book or not re.fullmatch(r"[A-Za-z0-9_]+", scene_id or ""):
        return None
    overrides = load_scene_overrides(book)
    if omit:
        overrides[scene_id] = {"omit": True, "at": __import__("time").time()}
    else:
        overrides.pop(scene_id, None)
    path = os.path.join(_tier1_dir(book), "scene_overrides.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2)
    os.replace(tmp, path)
    return {"scene_id": scene_id, "omit": bool(omit), "total_omitted": len(overrides)}


def book_speakers(book: str) -> List[str]:
    """Every speaker attributed anywhere in the book (for the correction picker)."""
    speakers = set()
    for payload in _load_lines(book):
        for l in payload.get("lines", []):
            if l.get("segment_type") == "dialogue" and l.get("character"):
                speakers.add(l["character"])
    return sorted(speakers | {"Narrator"})


def director_cast_roster(book: str, *, limit: int = 3) -> Optional[Dict[str, Any]]:
    book = _safe_book(book)
    if not book:
        return None
    marketplace = _get_marketplace()
    profiles = _character_profiles(book)
    roster = []
    for character in [name for name in book_speakers(book) if name != "Narrator"]:
        drawer = _get_drawer_info(character)
        query_text = _character_voice_query(book, character)
        try:
            suggestions = marketplace.search_marketplace(query_text, limit=max(1, limit))
        except Exception:
            suggestions = []
        roster.append({
            "character": character,
            "query": query_text,
            "profile": profiles.get(character),
            "current_voice": drawer.get("voice_ref_path") if drawer else None,
            "suggestions": suggestions,
        })
    return {"book": book, "characters": roster}


def director_search_voices(book: str, character: str, *, query: str = "", limit: int = 5) -> Optional[Dict[str, Any]]:
    book = _safe_book(book)
    if not book or not (character or "").strip():
        return None
    marketplace = _get_marketplace()
    query_text = _character_voice_query(book, character.strip(), query)
    results = marketplace.search_marketplace(query_text, limit=max(1, limit))
    return {
        "book": book,
        "character": character.strip(),
        "query": query_text,
        "results": results,
    }


def director_cast_character(
    book: str,
    character: str,
    *,
    description: str = "",
    buyer: str = "local",
    purpose: str = "audiobook production",
    voice_id: str = "",
) -> Optional[Dict[str, Any]]:
    book = _safe_book(book)
    character = (character or "").strip()
    if not book or not character:
        return None
    marketplace = _get_marketplace()
    if voice_id:
        cast = marketplace.cast_character_with_voice(
            character_name=character,
            voice_id=voice_id,
            buyer=buyer,
            purpose=purpose or f"cast as {character}",
        )
    else:
        query_text = _character_voice_query(book, character, description)
        cast = marketplace.cast_character(
            character_name=character,
            character_description=query_text,
            buyer=buyer,
            purpose=purpose or f"cast as {character}",
        )
    if cast is None:
        raise ValueError(f"No suitable voice found for character '{character}'")
    return {
        "book": book,
        "character": character,
        "cast": cast,
        "current_voice": (_get_drawer_info(character) or {}).get("voice_ref_path"),
    }


def usage_summary(project_id: str = "", book: str = "") -> Dict[str, Any]:
    """The billing meter's read side: aggregate the audit log (and render-job
    wall time) for one project or book. Streams the jsonl -- no index needed
    at current volumes; rotation is a future concern noted in the ledger."""
    audit_path = "data/llm_call_audit.jsonl"
    agg: Dict[str, Any] = {"llm_calls": 0, "successes": 0, "by_provider": {},
                           "by_task": {}, "first_ts": None, "last_ts": None}
    if os.path.exists(audit_path):
        with open(audit_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if project_id and r.get("project_id") != project_id:
                    continue
                if book and r.get("book") != book:
                    continue
                if not project_id and not book:
                    continue
                agg["llm_calls"] += 1
                agg["successes"] += 1 if r.get("success") else 0
                agg["by_provider"][r.get("provider", "?")] = agg["by_provider"].get(r.get("provider", "?"), 0) + 1
                task = r.get("task_name", "?")
                agg["by_task"][task] = agg["by_task"].get(task, 0) + 1
                ts = r.get("timestamp")
                if ts:
                    agg["first_ts"] = min(agg["first_ts"] or ts, ts)
                    agg["last_ts"] = max(agg["last_ts"] or ts, ts)
    agg["success_rate"] = round(agg["successes"] / agg["llm_calls"], 3) if agg["llm_calls"] else None

    render_seconds = 0.0
    render_jobs = 0
    for p in sorted(__import__("glob").glob("data/render_jobs/*.json")):
        j = _load_json(p) or {}
        if project_id and j.get("project_id") != project_id:
            continue
        if book and j.get("book") != book:
            continue
        if j.get("started_at") and j.get("finished_at"):
            render_seconds += j["finished_at"] - j["started_at"]
            render_jobs += 1
    agg["render_jobs_completed"] = render_jobs
    agg["render_minutes"] = round(render_seconds / 60, 1)
    agg["scope"] = {"project_id": project_id or None, "book": book or None}
    return agg


def export_manuscript(book: str, tier: int) -> Optional[Dict[str, str]]:
    """Processed-manuscript text export, honoring human decisions (speaker
    overrides, scene omits). Returns {"filename", "text"} or None.

    tier 1 -> clean narration text (what a single narrator reads)
    tier 2 -> attributed script: [Speaker] per line (the HumanProcessed format)
    tier 3 -> the production crew's full script (tier3/production_script.txt)
    """
    book = _safe_book(book)
    if not book or tier not in (1, 2, 3):
        return None
    if tier == 3:
        path = os.path.join(_tier3_dir(book), "production_script.txt")
        if not os.path.exists(path):
            return {"error": "No Tier 3 production script yet — run the scene_director crew first."}
        with open(path, encoding="utf-8") as f:
            return {"filename": f"{book}_tier3_production_script.txt", "text": f.read()}

    payloads = _load_lines(book)
    if not payloads:
        return {"error": "No line artifacts — ingest the book first."}
    overrides = load_speaker_overrides(book)
    omits = load_scene_overrides(book)
    structure = load_structure(book)
    titles = {key: chapter.title for key, chapter in chapter_lookup(structure).items()}
    payload_by_scene = {payload.get("scene_id", ""): payload for payload in payloads}

    out: List[str] = [f"# {book} — Tier {tier} "
                      f"{'narration text' if tier == 1 else 'attributed script'}", ""]
    seen_chapters = set()
    for sid in ordered_scene_ids(structure):
        p = payload_by_scene.get(sid)
        if p is None:
            continue
        if omits.get(sid, {}).get("omit"):
            continue
        scene_section = resolve_scene(structure, sid)
        chap = scene_section.parent_id if scene_section else sid.rsplit("_s", 1)[0]
        chapter_section = next((section for section in structure.sections if section.section_id == chap), None)
        chapter_key = chapter_section.legacy_id if chapter_section and chapter_section.legacy_id else (chap or "")
        if chap not in seen_chapters:
            seen_chapters.add(chap)
            out += ["", f"## {titles.get(chapter_key) or titles.get(chap) or chap}", ""]
        scene_label = sid.rsplit("_s", 1)[-1] if "_s" in sid else str(scene_section.order if scene_section else sid)
        out.append(f"[Scene {scene_label}]")
        out.append("")
        for l in p.get("lines", []):
            l = dict(l)
            apply_speaker_overrides([l], overrides)
            if tier == 1:
                out.append(l.get("text", ""))
            else:
                speaker = l.get("character", "Narrator") if l.get("segment_type") == "dialogue" else "Narrator"
                out.append(f"[{speaker}] {l.get('text', '')}")
            out.append("")
    return {"filename": f"{book}_tier{tier}_{'narration' if tier == 1 else 'script'}.txt",
            "text": "\n".join(out)}


def delete_book(book: str) -> Optional[Dict[str, Any]]:
    """Two intuitive levels: an INGESTED book's delete removes its analysis,
    renders, jobs and project rows (the source file stays and the book returns
    as 'new'); a NEW source's delete removes the file itself. A full purge is
    therefore two deletes. Sources under data/corpus are never auto-deleted --
    only data/uploads files are removable (corpus is the curated library)."""
    import glob as _glob
    import shutil as _shutil
    if not book or "/" in book or "\\" in book or book in (".", ".."):
        return None
    pipeline_dir = os.path.join(PIPELINE_ROOT, book)
    removed: Dict[str, Any] = {"book": book, "artifacts": False, "renders": 0,
                               "jobs": 0, "projects": 0, "source": None}
    if os.path.isdir(pipeline_dir):
        _shutil.rmtree(pipeline_dir)
        removed["artifacts"] = True
        preview_slug = re.sub(r"[^A-Za-z0-9_\-]", "", book)  # tier_preview's naming
        for p in _glob.glob(f"scratch/renders/{book}_tier*.*") + _glob.glob(f"scratch/tier_previews/{preview_slug}_tier*.*"):
            try:
                os.remove(p)
                removed["renders"] += 1
            except OSError:
                pass
        for p in _glob.glob("data/render_jobs/*.json"):
            j = _load_json(p) or {}
            if j.get("book") == book:
                os.remove(p)
                removed["jobs"] += 1
        try:
            import sqlite3
            c = sqlite3.connect("data/projects.db")
            c.execute("DELETE FROM projects WHERE book_stem = ?", (book,))
            removed["projects"] = c.total_changes
            c.commit(); c.close()
        except Exception as e:
            logger.warning(f"delete_book: project cleanup failed: {e}")
    else:
        # a "new" (never-ingested) source: deleting means the uploaded file
        for ext in (".txt", ".docx", ".epub"):
            p = os.path.join("data/uploads", book + ext)
            if os.path.exists(p):
                os.remove(p)
                removed["source"] = p
    logger.info(f"delete_book({book}): {removed}")
    return removed


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
