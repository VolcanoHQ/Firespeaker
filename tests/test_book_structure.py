#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

from src.book_structure import (
    ArtifactValidity,
    SplitSectionSpec,
    book_structure_history_dir,
    book_structure_path,
    change_content_type,
    get_section,
    load_book_structure,
    materialize_book_structure_from_tier1,
    merge_sections,
    migrate_tier1_artifacts,
    remove_section,
    rename_section,
    reorder_section,
    save_book_structure,
    sections_by_parent,
    split_section,
)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _fixture_pipeline(tmp_path):
    pipeline_dir = tmp_path / "data" / "corpus" / "pipeline" / "frankenstein" / "tier1"
    _write_json(str(pipeline_dir / "loop1_parts.json"), [
        {
            "part_id": "part_p1",
            "title": "Main Narrative",
            "text_block": "LETTER I text\nLETTER II text\nCHAPTER I text",
        }
    ])
    _write_json(str(pipeline_dir / "loop2_chapters.json"), [
        {
            "chapter_id": "part_p1_c1",
            "title": "LETTER I",
            "text_block": "You will rejoice to hear that no disaster has accompanied the commencement.",
        },
        {
            "chapter_id": "part_p1_c2",
            "title": "LETTER II",
            "text_block": "I am already far north of London and the wind of promise follows me.",
        },
        {
            "chapter_id": "part_p1_c3",
            "title": "CHAPTER I",
            "text_block": "Chapter one opens after the letters.",
        },
    ])
    _write_json(str(pipeline_dir / "loop3_scenes.json"), [
        {
            "scene_id": "part_p1_c1_s1",
            "text_block": "Letter one scene text.",
            "boundary_source": "whole_chapter",
        },
        {
            "scene_id": "part_p1_c1_s2",
            "text_block": "Letter one follow-up scene text.",
            "boundary_source": "transition_heuristic",
        },
        {
            "scene_id": "part_p1_c2_s1",
            "text_block": "Letter two scene text.",
            "boundary_source": "whole_chapter",
        },
        {
            "scene_id": "part_p1_c3_s1",
            "text_block": "Chapter one scene text.",
            "boundary_source": "whole_chapter",
        },
    ])
    return pipeline_dir


def test_migrates_tier1_artifacts_to_book_structure(tmp_path):
    pipeline_dir = _fixture_pipeline(tmp_path)
    structure = migrate_tier1_artifacts(str(pipeline_dir), source_file="data/uploads/frankenstein.epub")

    assert structure.book_id == "frankenstein"
    assert structure.source_format == "epub"
    assert structure.structure_version == 1
    assert structure.created_from.artifacts == ["loop1_parts.json", "loop2_chapters.json", "loop3_scenes.json"]
    assert any(section.legacy_id == "part_p1_c1" and section.content_type == "letter" for section in structure.sections)
    assert any(section.legacy_id == "part_p1_c3_s1" and section.content_type == "scene" for section in structure.sections)


def test_structure_has_stable_ordered_sections(tmp_path):
    structure = migrate_tier1_artifacts(str(_fixture_pipeline(tmp_path)), source_file="data/uploads/frankenstein.epub")
    chapters = sections_by_parent(structure, "sec_part_p1")
    assert [section.section_id for section in chapters] == ["sec_part_p1_c1", "sec_part_p1_c2", "sec_part_p1_c3"]
    assert [section.order for section in chapters] == [1, 2, 3]


def test_frankenstein_letters_remain_before_chapter_one(tmp_path):
    structure = migrate_tier1_artifacts(str(_fixture_pipeline(tmp_path)), source_file="data/uploads/frankenstein.epub")
    chapters = sections_by_parent(structure, "sec_part_p1")
    assert [section.title for section in chapters] == ["LETTER I", "LETTER II", "CHAPTER I"]
    assert [section.content_type for section in chapters] == ["letter", "letter", "chapter"]


def test_remove_section_creates_new_version(tmp_path):
    structure = migrate_tier1_artifacts(str(_fixture_pipeline(tmp_path)), source_file="data/uploads/frankenstein.epub")
    updated = remove_section(structure, "sec_part_p1_c2")

    assert structure.structure_version == 1
    assert updated.structure_version == 2
    removed = get_section(updated, "sec_part_p1_c2")
    assert removed.status == "deleted"
    assert removed.artifacts == ArtifactValidity(analysis_valid=False, audio_valid=False, render_valid=False)


def test_reorder_section_creates_new_version(tmp_path):
    structure = migrate_tier1_artifacts(str(_fixture_pipeline(tmp_path)), source_file="data/uploads/frankenstein.epub")
    updated = reorder_section(structure, "sec_part_p1_c3", 1)
    chapters = sections_by_parent(updated, "sec_part_p1")

    assert updated.structure_version == 2
    assert [section.section_id for section in chapters] == ["sec_part_p1_c3", "sec_part_p1_c1", "sec_part_p1_c2"]


def test_rename_section_preserves_text_hash(tmp_path):
    structure = migrate_tier1_artifacts(str(_fixture_pipeline(tmp_path)), source_file="data/uploads/frankenstein.epub")
    before = get_section(structure, "sec_part_p1_c1").text_hash
    updated = rename_section(structure, "sec_part_p1_c1", "Letter One")
    after = get_section(updated, "sec_part_p1_c1")

    assert after.title == "Letter One"
    assert after.text_hash == before
    assert after.artifacts.analysis_valid is True
    assert after.artifacts.audio_valid is True
    assert after.artifacts.render_valid is False


def test_change_content_type_updates_section(tmp_path):
    structure = migrate_tier1_artifacts(str(_fixture_pipeline(tmp_path)), source_file="data/uploads/frankenstein.epub")
    updated = change_content_type(structure, "sec_part_p1_c3", "prologue")
    changed = get_section(updated, "sec_part_p1_c3")

    assert changed.content_type == "prologue"
    assert changed.approval.state == "stale"
    assert changed.artifacts.analysis_valid is True


def test_merge_sections_creates_new_section_and_marks_sources_inactive(tmp_path):
    structure = migrate_tier1_artifacts(str(_fixture_pipeline(tmp_path)), source_file="data/uploads/frankenstein.epub")
    updated = merge_sections(
        structure,
        ["sec_part_p1_c1_s1", "sec_part_p1_c1_s2"],
        title="Merged Letter I Scene",
        content_type="scene",
    )

    source_one = get_section(updated, "sec_part_p1_c1_s1")
    source_two = get_section(updated, "sec_part_p1_c1_s2")
    merged = next(section for section in updated.sections if section.metadata.get("merged_from") == ["sec_part_p1_c1_s1", "sec_part_p1_c1_s2"])

    assert source_one.status == "merged"
    assert source_two.status == "merged"
    assert merged.title == "Merged Letter I Scene"
    assert merged.parent_id == "sec_part_p1_c1"
    assert merged.artifacts.render_valid is False


def test_split_section_creates_two_sections(tmp_path):
    structure = migrate_tier1_artifacts(str(_fixture_pipeline(tmp_path)), source_file="data/uploads/frankenstein.epub")
    updated = split_section(
        structure,
        "sec_part_p1_c2_s1",
        [
            SplitSectionSpec(title="Letter II Scene A", content_type="scene"),
            SplitSectionSpec(title="Letter II Scene B", content_type="scene"),
        ],
    )

    original = get_section(updated, "sec_part_p1_c2_s1")
    new_sections = [section for section in sections_by_parent(updated, "sec_part_p1_c2") if section.metadata.get("split_from") == "sec_part_p1_c2_s1"]
    assert original.status == "split"
    assert len(new_sections) == 2
    assert [section.title for section in new_sections] == ["Letter II Scene A", "Letter II Scene B"]


def test_edit_marks_expected_artifacts_stale(tmp_path):
    structure = migrate_tier1_artifacts(str(_fixture_pipeline(tmp_path)), source_file="data/uploads/frankenstein.epub")
    updated = merge_sections(
        structure,
        ["sec_part_p1_c1_s1", "sec_part_p1_c1_s2"],
        title="Merged Letter I Scene",
        content_type="scene",
    )

    source = get_section(updated, "sec_part_p1_c1_s1")
    merged = next(section for section in updated.sections if section.metadata.get("merged_from"))
    assert source.artifacts == ArtifactValidity(analysis_valid=False, audio_valid=False, render_valid=False)
    assert merged.artifacts == ArtifactValidity(analysis_valid=False, audio_valid=False, render_valid=False)


def test_legacy_artifacts_are_not_mutated(tmp_path):
    pipeline_dir = _fixture_pipeline(tmp_path)
    original_chapters = json.loads((pipeline_dir / "loop2_chapters.json").read_text(encoding="utf-8"))
    structure = migrate_tier1_artifacts(str(pipeline_dir), source_file="data/uploads/frankenstein.epub")
    updated = rename_section(structure, "sec_part_p1_c1", "Letter One")
    save_book_structure(updated, book_structure_path(str(pipeline_dir)))

    current_chapters = json.loads((pipeline_dir / "loop2_chapters.json").read_text(encoding="utf-8"))
    assert current_chapters == original_chapters


def test_book_structure_round_trips_json(tmp_path):
    pipeline_dir = _fixture_pipeline(tmp_path)
    structure = materialize_book_structure_from_tier1(
        str(pipeline_dir),
        source_file="data/uploads/frankenstein.epub",
        source_format="epub",
        book_id="frankenstein",
    )
    loaded = load_book_structure(book_structure_path(str(pipeline_dir)))
    assert loaded.model_dump() == structure.model_dump()


def test_structure_save_preserves_history_snapshots(tmp_path):
    pipeline_dir = _fixture_pipeline(tmp_path)
    path = book_structure_path(str(pipeline_dir))
    structure = materialize_book_structure_from_tier1(
        str(pipeline_dir),
        source_file="data/uploads/frankenstein.epub",
        source_format="epub",
        book_id="frankenstein",
    )
    renamed = rename_section(structure, "sec_part_p1_c1", "Letter One")
    save_book_structure(renamed, path)

    history_path = os.path.join(book_structure_history_dir(str(pipeline_dir)), "book_structure.v1.json")
    assert os.path.exists(history_path)
    snapshot = load_book_structure(history_path)
    assert snapshot.structure_version == 1
