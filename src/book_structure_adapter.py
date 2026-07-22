#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from src.book_structure import (
    ArtifactValidity,
    BookSection,
    BookStructure,
    book_structure_path,
    load_book_structure,
    materialize_book_structure_from_tier1,
    sections_by_parent,
)
from src.models import ChapterPayload, ManuscriptManifest, PartPayload, ScenePayload, ScriptLine

PIPELINE_ROOT = os.path.join("data", "corpus", "pipeline")


def pipeline_dir_for_book(book_id: str, *, pipeline_root: str = PIPELINE_ROOT) -> str:
    return os.path.join(pipeline_root, book_id, "tier1")


def load_structure(
    book_id: str,
    *,
    pipeline_root: str = PIPELINE_ROOT,
    source_file: str | None = None,
    source_format: str | None = None,
) -> BookStructure:
    pipeline_dir = pipeline_dir_for_book(book_id, pipeline_root=pipeline_root)
    path = book_structure_path(pipeline_dir)
    if os.path.exists(path):
        structure = load_book_structure(path)
    else:
        structure = materialize_book_structure_from_tier1(
            pipeline_dir,
            source_file=source_file or book_id,
            source_format=source_format,
            book_id=book_id,
        )
    return _hydrate_structure_from_legacy(structure, pipeline_dir)


def load_line_payloads(book_id: str, *, pipeline_root: str = PIPELINE_ROOT) -> list[dict[str, Any]]:
    pipeline_dir = pipeline_dir_for_book(book_id, pipeline_root=pipeline_root)
    enriched = os.path.join(pipeline_dir, "loop4_lines_enriched.json")
    raw = os.path.join(pipeline_dir, "loop4_lines.json")
    if os.path.exists(enriched):
        return _load_json(enriched) or []
    return _load_json(raw) or []


def ordered_parts(structure: BookStructure) -> list[BookSection]:
    return sections_by_parent(structure, None)


def ordered_chapters(structure: BookStructure, part_section: BookSection | str) -> list[BookSection]:
    parent_id = part_section.section_id if isinstance(part_section, BookSection) else part_section
    return sections_by_parent(structure, parent_id)


def ordered_scenes(structure: BookStructure, chapter_section: BookSection | str) -> list[BookSection]:
    parent_id = chapter_section.section_id if isinstance(chapter_section, BookSection) else chapter_section
    return sections_by_parent(structure, parent_id)


def chapter_lookup(structure: BookStructure) -> dict[str, BookSection]:
    lookup: dict[str, BookSection] = {}
    for part in ordered_parts(structure):
        for chapter in ordered_chapters(structure, part):
            lookup[_section_key(chapter)] = chapter
            lookup[chapter.section_id] = chapter
    return lookup


def scene_lookup(structure: BookStructure) -> dict[str, BookSection]:
    lookup: dict[str, BookSection] = {}
    for part in ordered_parts(structure):
        for chapter in ordered_chapters(structure, part):
            for scene in ordered_scenes(structure, chapter):
                lookup[_section_key(scene)] = scene
                lookup[scene.section_id] = scene
    return lookup


def resolve_scene(structure: BookStructure, scene_id: str) -> BookSection | None:
    return scene_lookup(structure).get(scene_id)


def scene_text(section: BookSection, structure: BookStructure, *, legacy_scene_index: dict[str, dict[str, Any]] | None = None) -> str:
    text = str(section.metadata.get("text_block") or "")
    if text:
        return text
    if legacy_scene_index and section.legacy_id and section.legacy_id in legacy_scene_index:
        return str(legacy_scene_index[section.legacy_id].get("text_block") or "")
    source_ids = list(section.text_ref.source_section_ids)
    if source_ids:
        index = scene_lookup(structure)
        blocks = []
        for source_id in source_ids:
            source = index.get(source_id)
            if source:
                source_text = str(source.metadata.get("text_block") or "")
                if source_text:
                    blocks.append(source_text)
        return "\n\n".join(blocks)
    return ""


def structure_to_gui_hierarchy(
    structure: BookStructure,
    *,
    line_payloads: list[dict[str, Any]] | None = None,
    analysis_pipeline: str = "tier1_manifest_v1",
) -> dict[str, Any]:
    line_index = _line_index(line_payloads or [])
    parts = []
    all_characters = set()
    total_scenes = 0

    for part_index, part in enumerate(ordered_parts(structure), 1):
        chapter_nodes = []
        for chapter_index, chapter in enumerate(ordered_chapters(structure, part), 1):
            scenes = []
            for scene_index, scene in enumerate(ordered_scenes(structure, chapter), 1):
                lines = _line_dicts_for_section(structure, scene, line_index)
                for line in lines:
                    if line.get("character") != "Narrator":
                        all_characters.add(line.get("character"))
                dialogue_lines = [line for line in lines if line.get("segment_type") == "dialogue"]
                narration_words = sum(len((line.get("text") or "").split()) for line in lines if line.get("segment_type") == "narrative")
                dialogue_words = sum(len((line.get("text") or "").split()) for line in dialogue_lines)
                scenes.append({
                    "scene_id": _section_key(scene),
                    "scene_number": scene_index,
                    "raw_scene_text": scene_text(scene, structure),
                    "characters_present": sorted({line.get("character") for line in dialogue_lines if line.get("character") != "Narrator"}),
                    "total_dialogue_lines": len(dialogue_lines),
                    "metrics": {
                        "total_words": narration_words + dialogue_words,
                        "narration_words": narration_words,
                        "dialogue_words": dialogue_words,
                    },
                    "lines": lines,
                })
            total_scenes += len(scenes)
            chapter_nodes.append({
                "chapter_id": _section_key(chapter),
                "chapter_number": chapter_index,
                "chapter_title": chapter.title,
                "total_scenes": len(scenes),
                "scenes": scenes,
            })
        parts.append({
            "part_id": _section_key(part),
            "part_title": part.title,
            "total_chapters": len(chapter_nodes),
            "chapters": chapter_nodes,
        })

    return {
        "metadata": {
            "source_file": os.path.basename(structure.source_file),
            "quote_style_detected": "double",
            "total_parts": len(parts),
            "total_chapters": sum(len(part["chapters"]) for part in parts),
            "total_scenes": total_scenes,
            "global_characters": ["Narrator", *sorted(character for character in all_characters if character)],
            "merge_decisions": [],
            "analysis_pipeline": analysis_pipeline,
            "book_structure_version": structure.structure_version,
        },
        "parts": parts,
    }


def structure_to_console_tree(
    structure: BookStructure,
    *,
    line_payloads: list[dict[str, Any]] | None = None,
    scene_overrides: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    scene_overrides = scene_overrides or {}
    line_index = _line_index(line_payloads or [])
    tree: list[dict[str, Any]] = []

    for part in ordered_parts(structure):
        chapter_nodes = []
        for chapter in ordered_chapters(structure, part):
            scene_nodes = []
            for scene in ordered_scenes(structure, chapter):
                sid = _section_key(scene)
                text = scene_text(scene, structure)
                lines = _line_dicts_for_section(structure, scene, line_index)
                dialogue = [line for line in lines if line.get("segment_type") == "dialogue"]
                enriched = [line for line in dialogue if str(line.get("attribution_method", "")) != "Tier 1 Default"]
                reviewed = [line for line in dialogue if "+reviewed" in str(line.get("attribution_method", ""))]
                speakers = sorted({line.get("character", "?") for line in dialogue} - {"Narrator"})
                scene_nodes.append({
                    "scene_id": sid,
                    "omitted": bool(scene_overrides.get(sid, {}).get("omit")),
                    "boundary_source": scene.metadata.get("boundary_source", "unknown"),
                    "chars": len(text),
                    "excerpt": " ".join(text[:140].split()),
                    "lines": len(lines),
                    "dialogue": len(dialogue),
                    "enriched": len(enriched),
                    "reviewed": len(reviewed),
                    "speakers": speakers,
                })
            chapter_nodes.append({
                "chapter_id": _section_key(chapter),
                "title": chapter.title,
                "scenes": scene_nodes,
            })
        tree.append({
            "part_id": _section_key(part),
            "title": part.title,
            "chapters": chapter_nodes,
        })
    return tree


def structure_to_manifest(
    structure: BookStructure,
    *,
    line_payloads: list[dict[str, Any]] | None = None,
) -> ManuscriptManifest:
    line_index = _line_index(line_payloads or [])
    part_payloads: list[PartPayload] = []
    total_scenes = 0

    for part in ordered_parts(structure):
        chapter_payloads: list[ChapterPayload] = []
        for chapter in ordered_chapters(structure, part):
            scene_payloads: list[ScenePayload] = []
            for scene in ordered_scenes(structure, chapter):
                lines = [ScriptLine.model_validate(line) for line in _line_dicts_for_section(structure, scene, line_index)]
                scene_payloads.append(ScenePayload(scene_id=_section_key(scene), lines=lines))
                total_scenes += 1
            chapter_payloads.append(ChapterPayload(
                chapter_id=_section_key(chapter),
                title=chapter.title,
                scenes=scene_payloads,
            ))
        part_payloads.append(PartPayload(
            part_id=_section_key(part),
            title=part.title,
            chapters=chapter_payloads,
        ))

    return ManuscriptManifest(
        source_file=os.path.basename(structure.source_file),
        total_parts=len(part_payloads),
        total_chapters=sum(len(part.chapters) for part in part_payloads),
        total_scenes=total_scenes,
        parts=part_payloads,
    )


def ordered_scene_ids(structure: BookStructure) -> list[str]:
    ids: list[str] = []
    for part in ordered_parts(structure):
        for chapter in ordered_chapters(structure, part):
            ids.extend(_section_key(scene) for scene in ordered_scenes(structure, chapter))
    return ids


def scene_text_map(structure: BookStructure) -> dict[str, str]:
    return {
        _section_key(scene): scene_text(scene, structure)
        for part in ordered_parts(structure)
        for chapter in ordered_chapters(structure, part)
        for scene in ordered_scenes(structure, chapter)
    }


def structure_to_script_data(
    structure: BookStructure,
    *,
    line_payloads: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    hierarchy = structure_to_gui_hierarchy(structure, line_payloads=line_payloads)
    flat_lines: list[dict[str, Any]] = []
    for part in hierarchy.get("parts", []):
        for chapter in part.get("chapters", []):
            for scene in chapter.get("scenes", []):
                flat_lines.extend(scene.get("lines", []))
    return {
        "metadata": {
            "source_file": hierarchy["metadata"].get("source_file"),
            "quote_style_detected": hierarchy["metadata"].get("quote_style_detected", "double"),
            "total_chapters": hierarchy["metadata"].get("total_chapters", 1),
            "total_lines_extracted": len(flat_lines),
            "characters_identified": hierarchy["metadata"].get("global_characters", []),
            "merge_decisions": hierarchy["metadata"].get("merge_decisions", []),
            "book_structure_version": hierarchy["metadata"].get("book_structure_version"),
        },
        "script": flat_lines,
    }


def invalid_sections(
    structure: BookStructure,
    *,
    require_analysis: bool = False,
    require_audio: bool = False,
    require_render: bool = False,
) -> list[BookSection]:
    invalid: list[BookSection] = []
    for section in structure.sections:
        if section.status != "active":
            continue
        if require_analysis and not section.artifacts.analysis_valid:
            invalid.append(section)
            continue
        if require_audio and not section.artifacts.audio_valid:
            invalid.append(section)
            continue
        if require_render and not section.artifacts.render_valid:
            invalid.append(section)
            continue
    return invalid


def structure_readiness(
    structure: BookStructure,
    *,
    require_analysis: bool = False,
    require_audio: bool = False,
    require_render: bool = False,
) -> dict[str, Any]:
    invalid = invalid_sections(
        structure,
        require_analysis=require_analysis,
        require_audio=require_audio,
        require_render=require_render,
    )
    return {
        "ok": not invalid,
        "invalid_sections": [
            {
                "section_id": section.section_id,
                "legacy_id": section.legacy_id,
                "title": section.title,
                "content_type": section.content_type,
                "artifacts": section.artifacts.model_dump(),
            }
            for section in invalid
        ],
        "required": {
            "analysis_valid": require_analysis,
            "audio_valid": require_audio,
            "render_valid": require_render,
        },
        "structure_version": structure.structure_version,
    }


def require_structure_readiness(
    structure: BookStructure,
    *,
    require_analysis: bool = False,
    require_audio: bool = False,
    require_render: bool = False,
    operation: str = "operation",
) -> None:
    readiness = structure_readiness(
        structure,
        require_analysis=require_analysis,
        require_audio=require_audio,
        require_render=require_render,
    )
    if readiness["ok"]:
        return
    affected = ", ".join(
        section["legacy_id"] or section["section_id"]
        for section in readiness["invalid_sections"][:6]
    )
    raise ValueError(
        f"Canonical structure is stale for {operation}. Re-run the required upstream analysis "
        f"before continuing. Affected sections: {affected}"
    )


def scene_lines(structure: BookStructure, scene_id: str, *, line_payloads: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    section = resolve_scene(structure, scene_id)
    if section is None:
        return []
    return _line_dicts_for_section(structure, section, _line_index(line_payloads or []))


def _section_key(section: BookSection) -> str:
    return section.legacy_id or section.section_id


def _line_index(payloads: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {payload.get("scene_id", ""): payload.get("lines", []) for payload in payloads}


def _line_dicts_for_section(
    structure: BookStructure,
    section: BookSection,
    line_index: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    key = _section_key(section)
    if key in line_index:
        return [_line_for_gui(line, key) for line in line_index[key]]
    if section.section_id in line_index:
        return [_line_for_gui(line, key) for line in line_index[section.section_id]]
    if section.text_ref.source_section_ids:
        merged: list[dict[str, Any]] = []
        lookup = scene_lookup(structure)
        for source_id in section.text_ref.source_section_ids:
            source = lookup.get(source_id)
            if source:
                merged.extend(_line_dicts_for_section(structure, source, line_index))
        return merged
    return []


def _line_for_gui(line: dict[str, Any], scene_id: str) -> dict[str, Any]:
    line_copy = dict(line)
    line_copy["dialogue"] = line_copy.get("text")
    line_copy["scene_id"] = scene_id
    return line_copy


def _hydrate_structure_from_legacy(structure: BookStructure, pipeline_dir: str) -> BookStructure:
    parts = _index_legacy(os.path.join(pipeline_dir, "loop1_parts.json"), "part_id")
    chapters = _index_legacy(os.path.join(pipeline_dir, "loop2_chapters.json"), "chapter_id")
    scenes = _index_legacy(os.path.join(pipeline_dir, "loop3_scenes.json"), "scene_id")

    updated = structure.model_copy(deep=True)
    for section in updated.sections:
        if section.metadata.get("text_block"):
            continue
        legacy_id = section.legacy_id or ""
        source = None
        if section.text_ref.artifact == "loop1_parts.json":
            source = parts.get(legacy_id)
        elif section.text_ref.artifact == "loop2_chapters.json":
            source = chapters.get(legacy_id)
        elif section.text_ref.artifact == "loop3_scenes.json":
            source = scenes.get(legacy_id)
        if source:
            if "text_block" in source:
                section.metadata["text_block"] = source.get("text_block", "")
                section.metadata["source_chars"] = len(section.metadata["text_block"])
            if "boundary_source" in source:
                section.metadata.setdefault("boundary_source", source.get("boundary_source"))
    return updated


def _index_legacy(path: str, id_key: str) -> dict[str, dict[str, Any]]:
    return {
        item.get(id_key, ""): item
        for item in (_load_json(path) or [])
        if isinstance(item, dict) and item.get(id_key)
    }


def _load_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
