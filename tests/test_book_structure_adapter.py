#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

from src.book_structure import (
    book_structure_path,
    load_book_structure,
    rename_section,
    reorder_section,
    save_book_structure,
)
from src.book_structure_adapter import (
    load_line_payloads,
    load_structure,
    ordered_scene_ids,
    structure_to_console_tree,
    structure_to_gui_hierarchy,
    structure_to_manifest,
)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _fixture_pipeline(tmp_path):
    pipeline_root = tmp_path / "data" / "corpus" / "pipeline"
    pipeline_dir = pipeline_root / "frankenstein" / "tier1"
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
    _write_json(str(pipeline_dir / "loop4_lines.json"), [
        {
            "scene_id": "part_p1_c1_s1",
            "lines": [
                {
                    "line_id": "l1",
                    "chapter": 1,
                    "scene": 1,
                    "line_number": 1,
                    "character": "Narrator",
                    "speaker_id": "char_narrator",
                    "segment_type": "narrative",
                    "text": "Letter one scene text.",
                }
            ],
        },
        {
            "scene_id": "part_p1_c2_s1",
            "lines": [
                {
                    "line_id": "l2",
                    "chapter": 2,
                    "scene": 2,
                    "line_number": 1,
                    "character": "Walton",
                    "speaker_id": "char_walton",
                    "segment_type": "dialogue",
                    "text": "Letter two scene text.",
                    "attribution_method": "LLM+reviewed",
                }
            ],
        },
        {
            "scene_id": "part_p1_c3_s1",
            "lines": [
                {
                    "line_id": "l3",
                    "chapter": 3,
                    "scene": 3,
                    "line_number": 1,
                    "character": "Narrator",
                    "speaker_id": "char_narrator",
                    "segment_type": "narrative",
                    "text": "Chapter one scene text.",
                }
            ],
        },
    ])
    return pipeline_root, pipeline_dir


def test_loads_existing_book_structure(tmp_path):
    pipeline_root, pipeline_dir = _fixture_pipeline(tmp_path)
    structure = load_structure("frankenstein", pipeline_root=str(pipeline_root), source_file="data/uploads/frankenstein.epub")
    reloaded = load_structure("frankenstein", pipeline_root=str(pipeline_root), source_file="data/uploads/frankenstein.epub")

    assert structure.structure_version == 1
    assert reloaded.structure_version == 1
    assert os.path.exists(book_structure_path(str(pipeline_dir)))


def test_legacy_artifacts_auto_migrate(tmp_path):
    pipeline_root, pipeline_dir = _fixture_pipeline(tmp_path)
    assert not os.path.exists(book_structure_path(str(pipeline_dir)))

    structure = load_structure("frankenstein", pipeline_root=str(pipeline_root), source_file="data/uploads/frankenstein.epub")

    assert structure.book_id == "frankenstein"
    assert os.path.exists(book_structure_path(str(pipeline_dir)))


def test_gui_hierarchy_generated_from_canonical_structure(tmp_path):
    pipeline_root, pipeline_dir = _fixture_pipeline(tmp_path)
    structure = load_structure("frankenstein", pipeline_root=str(pipeline_root), source_file="data/uploads/frankenstein.epub")
    edited = rename_section(structure, "sec_part_p1_c1", "Letter One")
    save_book_structure(edited, book_structure_path(str(pipeline_dir)))

    loaded = load_structure("frankenstein", pipeline_root=str(pipeline_root), source_file="data/uploads/frankenstein.epub")
    hierarchy = structure_to_gui_hierarchy(loaded, line_payloads=load_line_payloads("frankenstein", pipeline_root=str(pipeline_root)))

    assert hierarchy["parts"][0]["chapters"][0]["chapter_title"] == "Letter One"
    assert hierarchy["parts"][0]["chapters"][1]["chapter_title"] == "LETTER II"


def test_console_tree_generated_from_canonical_structure(tmp_path):
    pipeline_root, pipeline_dir = _fixture_pipeline(tmp_path)
    structure = load_structure("frankenstein", pipeline_root=str(pipeline_root), source_file="data/uploads/frankenstein.epub")
    edited = reorder_section(structure, "sec_part_p1_c3", 1)
    save_book_structure(edited, book_structure_path(str(pipeline_dir)))

    loaded = load_structure("frankenstein", pipeline_root=str(pipeline_root), source_file="data/uploads/frankenstein.epub")
    tree = structure_to_console_tree(loaded, line_payloads=load_line_payloads("frankenstein", pipeline_root=str(pipeline_root)))

    assert [chapter["title"] for chapter in tree[0]["chapters"]] == ["CHAPTER I", "LETTER I", "LETTER II"]


def test_structure_order_matches_original_artifacts(tmp_path):
    pipeline_root, _ = _fixture_pipeline(tmp_path)
    structure = load_structure("frankenstein", pipeline_root=str(pipeline_root), source_file="data/uploads/frankenstein.epub")

    assert ordered_scene_ids(structure) == ["part_p1_c1_s1", "part_p1_c2_s1", "part_p1_c3_s1"]


def test_structure_adapter_round_trip(tmp_path):
    pipeline_root, _ = _fixture_pipeline(tmp_path)
    structure = load_structure("frankenstein", pipeline_root=str(pipeline_root), source_file="data/uploads/frankenstein.epub")
    hierarchy = structure_to_gui_hierarchy(structure, line_payloads=load_line_payloads("frankenstein", pipeline_root=str(pipeline_root)))
    manifest = structure_to_manifest(structure, line_payloads=load_line_payloads("frankenstein", pipeline_root=str(pipeline_root)))

    assert manifest.total_chapters == hierarchy["metadata"]["total_chapters"]
    assert manifest.total_scenes == hierarchy["metadata"]["total_scenes"]


def test_missing_structure_falls_back_to_migration(tmp_path):
    pipeline_root, pipeline_dir = _fixture_pipeline(tmp_path)
    structure_path = book_structure_path(str(pipeline_dir))
    if os.path.exists(structure_path):
        os.remove(structure_path)

    structure = load_structure("frankenstein", pipeline_root=str(pipeline_root), source_file="data/uploads/frankenstein.epub")

    assert structure.source_file == "data/uploads/frankenstein.epub"
    assert os.path.exists(structure_path)


def test_manifest_adapter_preserves_chapter_order(tmp_path):
    pipeline_root, _ = _fixture_pipeline(tmp_path)
    structure = load_structure("frankenstein", pipeline_root=str(pipeline_root), source_file="data/uploads/frankenstein.epub")
    manifest = structure_to_manifest(structure, line_payloads=load_line_payloads("frankenstein", pipeline_root=str(pipeline_root)))

    assert [chapter.title for chapter in manifest.parts[0].chapters] == ["LETTER I", "LETTER II", "CHAPTER I"]


def test_frankenstein_letter_order_preserved(tmp_path):
    pipeline_root, _ = _fixture_pipeline(tmp_path)
    structure = load_structure("frankenstein", pipeline_root=str(pipeline_root), source_file="data/uploads/frankenstein.epub")
    tree = structure_to_console_tree(structure, line_payloads=load_line_payloads("frankenstein", pipeline_root=str(pipeline_root)))

    assert [chapter["title"] for chapter in tree[0]["chapters"][:2]] == ["LETTER I", "LETTER II"]


def test_edited_book_prefers_canonical_without_mutating_legacy(tmp_path):
    pipeline_root, pipeline_dir = _fixture_pipeline(tmp_path)
    legacy_chapters = json.loads((pipeline_dir / "loop2_chapters.json").read_text(encoding="utf-8"))
    structure = load_structure("frankenstein", pipeline_root=str(pipeline_root), source_file="data/uploads/frankenstein.epub")
    edited = rename_section(reorder_section(structure, "sec_part_p1_c3", 1), "sec_part_p1_c3", "Prologue")
    save_book_structure(edited, book_structure_path(str(pipeline_dir)))

    loaded = load_structure("frankenstein", pipeline_root=str(pipeline_root), source_file="data/uploads/frankenstein.epub")
    hierarchy = structure_to_gui_hierarchy(loaded, line_payloads=load_line_payloads("frankenstein", pipeline_root=str(pipeline_root)))
    tree = structure_to_console_tree(loaded, line_payloads=load_line_payloads("frankenstein", pipeline_root=str(pipeline_root)))
    current_legacy = json.loads((pipeline_dir / "loop2_chapters.json").read_text(encoding="utf-8"))

    assert hierarchy["parts"][0]["chapters"][0]["chapter_title"] == "Prologue"
    assert tree[0]["chapters"][0]["title"] == "Prologue"
    assert current_legacy == legacy_chapters
    assert load_book_structure(book_structure_path(str(pipeline_dir))).structure_version == 3
