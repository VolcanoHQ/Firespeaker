#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field

CONTENT_TYPES = (
    "part",
    "chapter",
    "letter",
    "preface",
    "prologue",
    "appendix",
    "epilogue",
    "front_matter",
    "back_matter",
    "scene",
    "unknown",
)
STATUS_TYPES = ("active", "inactive", "deleted", "merged", "split")
BOOK_STRUCTURE_FILENAME = "book_structure.json"
BOOK_STRUCTURE_HISTORY_DIRNAME = "structure_history"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return f"sha256:{hashlib.sha256((text or '').encode('utf-8')).hexdigest()}"


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _slug_book_id(value: str) -> str:
    stem = os.path.splitext(os.path.basename(value or ""))[0].strip().lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem or "book"


class StructureCreatedFrom(BaseModel):
    pipeline: str = "tier1"
    artifacts: list[str] = Field(default_factory=list)


class ArtifactValidity(BaseModel):
    analysis_valid: bool = True
    audio_valid: bool = True
    render_valid: bool = True


class ApprovalMetadata(BaseModel):
    state: Literal["not_required", "pending", "approved", "stale"] = "not_required"
    approved_by: str | None = None
    approved_at: str | None = None
    approved_structure_version: int | None = None


class TextReference(BaseModel):
    kind: Literal["tier1_artifact", "derived", "epub_spine", "text_span"] = "tier1_artifact"
    artifact: str | None = None
    legacy_id: str | None = None
    href: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    source_section_ids: list[str] = Field(default_factory=list)


class BookSection(BaseModel):
    section_id: str
    parent_id: str | None = None
    order: int
    content_type: Literal[
        "part",
        "chapter",
        "letter",
        "preface",
        "prologue",
        "appendix",
        "epilogue",
        "front_matter",
        "back_matter",
        "scene",
        "unknown",
    ] = "unknown"
    title: str
    legacy_id: str | None = None
    text_ref: TextReference = Field(default_factory=TextReference)
    text_hash: str
    status: Literal["active", "inactive", "deleted", "merged", "split"] = "active"
    artifacts: ArtifactValidity = Field(default_factory=ArtifactValidity)
    approval: ApprovalMetadata = Field(default_factory=ApprovalMetadata)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utcnow)
    updated_at: str = Field(default_factory=_utcnow)


class BookStructure(BaseModel):
    book_id: str
    source_file: str
    source_format: str
    structure_version: int = 1
    created_from: StructureCreatedFrom = Field(default_factory=StructureCreatedFrom)
    sections: list[BookSection] = Field(default_factory=list)
    future_approval: dict[str, Any] = Field(default_factory=lambda: {"state": "not_configured"})
    created_at: str = Field(default_factory=_utcnow)
    updated_at: str = Field(default_factory=_utcnow)


class SplitSectionSpec(BaseModel):
    title: str
    content_type: Literal[
        "part",
        "chapter",
        "letter",
        "preface",
        "prologue",
        "appendix",
        "epilogue",
        "front_matter",
        "back_matter",
        "scene",
        "unknown",
    ] = "unknown"
    text_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def book_structure_path(pipeline_dir: str) -> str:
    return os.path.join(pipeline_dir, BOOK_STRUCTURE_FILENAME)


def book_structure_history_dir(pipeline_dir: str) -> str:
    return os.path.join(pipeline_dir, BOOK_STRUCTURE_HISTORY_DIRNAME)


def load_book_structure(path: str) -> BookStructure:
    return BookStructure.model_validate(_load_json(path))


def save_book_structure(structure: BookStructure, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        existing = load_book_structure(path)
        history_dir = book_structure_history_dir(os.path.dirname(path))
        os.makedirs(history_dir, exist_ok=True)
        history_path = os.path.join(history_dir, f"book_structure.v{existing.structure_version}.json")
        if not os.path.exists(history_path):
            with open(history_path, "w", encoding="utf-8") as handle:
                json.dump(existing.model_dump(), handle, indent=2)

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(structure.model_dump(), handle, indent=2)
    os.replace(tmp_path, path)
    return path


def sections_by_parent(structure: BookStructure, parent_id: str | None, *, active_only: bool = True) -> list[BookSection]:
    sections = [
        section for section in structure.sections
        if section.parent_id == parent_id and (section.status == "active" or not active_only)
    ]
    return sorted(sections, key=lambda section: (section.order, section.section_id))


def get_section(structure: BookStructure, section_id: str) -> BookSection:
    for section in structure.sections:
        if section.section_id == section_id:
            return section
    raise KeyError(f"Unknown section_id: {section_id}")


def descendants_of(structure: BookStructure, section_id: str, *, active_only: bool = False) -> list[BookSection]:
    descendants = []
    queue = [section_id]
    while queue:
        parent = queue.pop(0)
        children = sections_by_parent(structure, parent, active_only=active_only)
        descendants.extend(children)
        queue.extend(child.section_id for child in children)
    return descendants


def migrate_tier1_artifacts(
    pipeline_dir: str,
    *,
    source_file: str | None = None,
    source_format: str | None = None,
    book_id: str | None = None,
) -> BookStructure:
    parts_path = os.path.join(pipeline_dir, "loop1_parts.json")
    chapters_path = os.path.join(pipeline_dir, "loop2_chapters.json")
    scenes_path = os.path.join(pipeline_dir, "loop3_scenes.json")

    parts = _load_json(parts_path) if os.path.exists(parts_path) else []
    chapters = _load_json(chapters_path) if os.path.exists(chapters_path) else []
    scenes = _load_json(scenes_path) if os.path.exists(scenes_path) else []

    now = _utcnow()
    inferred_book_id = book_id or _slug_book_id(source_file or os.path.basename(os.path.dirname(pipeline_dir)))
    inferred_source_file = source_file or inferred_book_id
    inferred_source_format = source_format or os.path.splitext(inferred_source_file)[1].lstrip(".").lower() or "txt"

    sections: list[BookSection] = []

    for index, part in enumerate(parts, 1):
        legacy_id = part.get("part_id") or f"part_p{index}"
        text_block = part.get("text_block", "")
        sections.append(BookSection(
            section_id=_legacy_section_id(legacy_id),
            parent_id=None,
            order=index,
            content_type=_infer_content_type(part.get("title", ""), default="part", level="part"),
            title=part.get("title", legacy_id),
            legacy_id=legacy_id,
            text_ref=TextReference(kind="tier1_artifact", artifact="loop1_parts.json", legacy_id=legacy_id),
            text_hash=_sha256_text(text_block),
            metadata={"source_chars": len(text_block), "text_block": text_block},
            created_at=now,
            updated_at=now,
        ))

    chapter_groups: dict[str, list[dict[str, Any]]] = {}
    for chapter in chapters:
        legacy_id = chapter.get("chapter_id", "")
        parent_legacy = legacy_id.rsplit("_c", 1)[0] if "_c" in legacy_id else None
        chapter_groups.setdefault(parent_legacy, []).append(chapter)

    for parent_legacy, group in chapter_groups.items():
        for index, chapter in enumerate(group, 1):
            legacy_id = chapter.get("chapter_id") or f"{parent_legacy}_c{index}"
            text_block = chapter.get("text_block", "")
            sections.append(BookSection(
                section_id=_legacy_section_id(legacy_id),
                parent_id=_legacy_section_id(parent_legacy) if parent_legacy else None,
                order=index,
                content_type=_infer_content_type(chapter.get("title", ""), default="chapter", level="chapter"),
                title=chapter.get("title", legacy_id),
                legacy_id=legacy_id,
                text_ref=TextReference(kind="tier1_artifact", artifact="loop2_chapters.json", legacy_id=legacy_id),
                text_hash=_sha256_text(text_block),
                metadata={"source_chars": len(text_block), "text_block": text_block},
                created_at=now,
                updated_at=now,
            ))

    scene_groups: dict[str, list[dict[str, Any]]] = {}
    for scene in scenes:
        legacy_id = scene.get("scene_id", "")
        parent_legacy = legacy_id.rsplit("_s", 1)[0] if "_s" in legacy_id else None
        scene_groups.setdefault(parent_legacy, []).append(scene)

    for parent_legacy, group in scene_groups.items():
        for index, scene in enumerate(group, 1):
            legacy_id = scene.get("scene_id") or f"{parent_legacy}_s{index}"
            text_block = scene.get("text_block", "")
            metadata = {
                "source_chars": len(text_block),
                "boundary_source": scene.get("boundary_source", "unknown"),
                "text_block": text_block,
            }
            sections.append(BookSection(
                section_id=_legacy_section_id(legacy_id),
                parent_id=_legacy_section_id(parent_legacy) if parent_legacy else None,
                order=index,
                content_type="scene",
                title=f"Scene {index}",
                legacy_id=legacy_id,
                text_ref=TextReference(kind="tier1_artifact", artifact="loop3_scenes.json", legacy_id=legacy_id),
                text_hash=_sha256_text(text_block),
                metadata=metadata,
                created_at=now,
                updated_at=now,
            ))

    return BookStructure(
        book_id=inferred_book_id,
        source_file=source_file or inferred_source_file,
        source_format=inferred_source_format,
        structure_version=1,
        created_from=StructureCreatedFrom(
            pipeline="tier1",
            artifacts=["loop1_parts.json", "loop2_chapters.json", "loop3_scenes.json"],
        ),
        sections=sections,
        created_at=now,
        updated_at=now,
    )


def materialize_book_structure_from_tier1(
    pipeline_dir: str,
    *,
    source_file: str,
    source_format: str | None = None,
    book_id: str | None = None,
    overwrite_existing: bool = False,
) -> BookStructure:
    path = book_structure_path(pipeline_dir)
    if os.path.exists(path) and not overwrite_existing:
        return load_book_structure(path)
    structure = migrate_tier1_artifacts(
        pipeline_dir,
        source_file=source_file,
        source_format=source_format,
        book_id=book_id,
    )
    save_book_structure(structure, path)
    return structure


def rename_section(structure: BookStructure, section_id: str, new_title: str) -> BookStructure:
    updated = _clone_new_version(structure)
    section = get_section(updated, section_id)
    section.title = new_title
    section.artifacts.render_valid = False
    _mark_approval_stale(section)
    section.updated_at = updated.updated_at
    return updated


def change_content_type(structure: BookStructure, section_id: str, new_content_type: str) -> BookStructure:
    if new_content_type not in CONTENT_TYPES:
        raise ValueError(f"Unsupported content type: {new_content_type}")
    updated = _clone_new_version(structure)
    section = get_section(updated, section_id)
    section.content_type = new_content_type
    _mark_approval_stale(section)
    section.updated_at = updated.updated_at
    return updated


def reorder_section(structure: BookStructure, section_id: str, new_order: int) -> BookStructure:
    updated = _clone_new_version(structure)
    target = get_section(updated, section_id)
    siblings = sections_by_parent(updated, target.parent_id)
    if target not in siblings:
        raise ValueError(f"Cannot reorder inactive section: {section_id}")
    bounded_order = max(1, min(new_order, len(siblings)))
    siblings.remove(target)
    siblings.insert(bounded_order - 1, target)
    _reassign_active_orders(siblings, timestamp=updated.updated_at)
    for sibling in siblings:
        sibling.artifacts.render_valid = False
        _mark_approval_stale(sibling)
    return updated


def remove_section(structure: BookStructure, section_id: str) -> BookStructure:
    updated = _clone_new_version(structure)
    target = get_section(updated, section_id)
    impacted = [target, *descendants_of(updated, section_id, active_only=False)]
    for section in impacted:
        section.status = "deleted"
        section.artifacts = ArtifactValidity(analysis_valid=False, audio_valid=False, render_valid=False)
        _mark_approval_stale(section)
        section.updated_at = updated.updated_at
    _normalize_sibling_orders(updated, target.parent_id, timestamp=updated.updated_at)
    return updated


def merge_sections(
    structure: BookStructure,
    section_ids: Sequence[str],
    *,
    title: str | None = None,
    content_type: str | None = None,
) -> BookStructure:
    if len(section_ids) < 2:
        raise ValueError("merge_sections requires at least two section IDs")
    updated = _clone_new_version(structure)
    sources = [get_section(updated, section_id) for section_id in section_ids]
    _assert_same_parent(sources)

    parent_id = sources[0].parent_id
    active_siblings = sections_by_parent(updated, parent_id)
    active_ids = [section.section_id for section in active_siblings]
    positions = [active_ids.index(section.section_id) for section in sources]
    if positions != list(range(min(positions), max(positions) + 1)):
        raise ValueError("Merged sections must be contiguous active siblings")

    merged_section_id = _generated_section_id("merge", updated.structure_version, section_ids)
    merged_title = title or " / ".join(section.title for section in sources)
    merged_content_type = content_type or sources[0].content_type
    if merged_content_type not in CONTENT_TYPES:
        raise ValueError(f"Unsupported content type: {merged_content_type}")
    merged_hash = _sha256_text("|".join(section.text_hash for section in sources))
    merged_order = min(section.order for section in sources)

    for section in sources:
        section.status = "merged"
        section.artifacts = ArtifactValidity(analysis_valid=False, audio_valid=False, render_valid=False)
        _mark_approval_stale(section)
        section.updated_at = updated.updated_at
        for descendant in descendants_of(updated, section.section_id, active_only=False):
            descendant.artifacts = ArtifactValidity(analysis_valid=False, audio_valid=False, render_valid=False)
            _mark_approval_stale(descendant)
            descendant.updated_at = updated.updated_at

    merged_section = BookSection(
        section_id=merged_section_id,
        parent_id=parent_id,
        order=merged_order,
        content_type=merged_content_type,
        title=merged_title,
        text_ref=TextReference(
            kind="derived",
            source_section_ids=list(section_ids),
        ),
        text_hash=merged_hash,
        metadata={"merged_from": list(section_ids)},
        created_at=updated.updated_at,
        updated_at=updated.updated_at,
        artifacts=ArtifactValidity(analysis_valid=False, audio_valid=False, render_valid=False),
        approval=ApprovalMetadata(state="stale"),
    )
    updated.sections.append(merged_section)

    direct_children: list[BookSection] = []
    for section in sources:
        direct_children.extend(sections_by_parent(updated, section.section_id, active_only=False))
    for index, child in enumerate(sorted(direct_children, key=lambda item: (item.order, item.section_id)), 1):
        child.parent_id = merged_section_id
        child.order = index
        child.artifacts = ArtifactValidity(analysis_valid=False, audio_valid=False, render_valid=False)
        _mark_approval_stale(child)
        child.updated_at = updated.updated_at

    _normalize_sibling_orders(updated, parent_id, timestamp=updated.updated_at)
    return updated


def split_section(structure: BookStructure, section_id: str, split_specs: Sequence[SplitSectionSpec]) -> BookStructure:
    if len(split_specs) < 2:
        raise ValueError("split_section requires at least two split specs")
    updated = _clone_new_version(structure)
    source = get_section(updated, section_id)
    if sections_by_parent(updated, section_id):
        raise ValueError("split_section currently supports leaf sections only")

    source.status = "split"
    source.artifacts = ArtifactValidity(analysis_valid=False, audio_valid=False, render_valid=False)
    _mark_approval_stale(source)
    source.updated_at = updated.updated_at

    new_sections: list[BookSection] = []
    for index, spec in enumerate(split_specs, source.order):
        section_id_value = _generated_section_id("split", updated.structure_version, [section_id, str(index), spec.title])
        new_sections.append(BookSection(
            section_id=section_id_value,
            parent_id=source.parent_id,
            order=index,
            content_type=spec.content_type,
            title=spec.title,
            text_ref=TextReference(kind="derived", source_section_ids=[section_id]),
            text_hash=spec.text_hash or _sha256_text(f"{source.text_hash}:{spec.title}:{index}"),
            metadata={"split_from": section_id, **spec.metadata},
            created_at=updated.updated_at,
            updated_at=updated.updated_at,
            artifacts=ArtifactValidity(analysis_valid=False, audio_valid=False, render_valid=False),
            approval=ApprovalMetadata(state="stale"),
        ))
    updated.sections.extend(new_sections)
    _normalize_sibling_orders(updated, source.parent_id, timestamp=updated.updated_at)
    return updated


def structure_to_outline(structure: BookStructure) -> dict[str, Any]:
    def _serialize(parent_id: str | None) -> list[dict[str, Any]]:
        nodes = []
        for section in sections_by_parent(structure, parent_id):
            nodes.append({
                "section_id": section.section_id,
                "title": section.title,
                "content_type": section.content_type,
                "order": section.order,
                "status": section.status,
                "children": _serialize(section.section_id),
            })
        return nodes

    return {
        "book_id": structure.book_id,
        "source_file": structure.source_file,
        "source_format": structure.source_format,
        "structure_version": structure.structure_version,
        "sections": _serialize(None),
    }


def _legacy_section_id(legacy_id: str | None) -> str | None:
    if legacy_id is None:
        return None
    return f"sec_{legacy_id}"


def _infer_content_type(title: str, *, default: str, level: str) -> str:
    probe = (title or "").strip().lower()
    if "appendix" in probe:
        return "appendix"
    if "epilogue" in probe:
        return "epilogue"
    if "prologue" in probe:
        return "prologue"
    if "preface" in probe:
        return "preface"
    if "front matter" in probe:
        return "front_matter"
    if "back matter" in probe:
        return "back_matter"
    if probe.startswith("letter "):
        return "letter"
    if level == "part":
        return "part" if default == "part" else default
    if probe.startswith("chapter ") or default == "chapter":
        return "chapter"
    return default if default in CONTENT_TYPES else "unknown"


def _clone_new_version(structure: BookStructure) -> BookStructure:
    updated = structure.model_copy(deep=True)
    updated.structure_version += 1
    updated.updated_at = _utcnow()
    return updated


def _reassign_active_orders(sections: Sequence[BookSection], *, timestamp: str) -> None:
    for index, section in enumerate(sections, 1):
        section.order = index
        section.updated_at = timestamp


def _normalize_sibling_orders(structure: BookStructure, parent_id: str | None, *, timestamp: str) -> None:
    active_siblings = sections_by_parent(structure, parent_id)
    _reassign_active_orders(active_siblings, timestamp=timestamp)


def _generated_section_id(kind: str, version: int, parts: Iterable[str]) -> str:
    payload = "|".join(parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"sec_{kind}_v{version}_{digest}"


def _assert_same_parent(sections: Sequence[BookSection]) -> None:
    parent_ids = {section.parent_id for section in sections}
    if len(parent_ids) != 1:
        raise ValueError("Sections must share the same parent")


def _mark_approval_stale(section: BookSection) -> None:
    section.approval.state = "stale"
    section.approval.approved_by = None
    section.approval.approved_at = None
    section.approval.approved_structure_version = None
