#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys
import types

from src import console_api, render_job, scene_director
from src.book_structure import load_book_structure, rename_section, reorder_section, save_book_structure
from src.book_structure_adapter import load_line_payloads, load_structure, scene_text_map, structure_to_manifest


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _fixture_pipeline(tmp_path):
    pipeline_root = tmp_path / "data" / "corpus" / "pipeline"
    pipeline_dir = pipeline_root / "frankenstein" / "tier1"
    uploads_dir = tmp_path / "data" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    (uploads_dir / "frankenstein.txt").write_text("placeholder", encoding="utf-8")
    _write_json(str(pipeline_dir / "loop1_parts.json"), [
        {"part_id": "part_p1", "title": "Main Narrative", "text_block": "LETTER I\nLETTER II\nCHAPTER I"}
    ])
    _write_json(str(pipeline_dir / "loop2_chapters.json"), [
        {"chapter_id": "part_p1_c1", "title": "LETTER I", "text_block": "Letter one chapter text."},
        {"chapter_id": "part_p1_c2", "title": "LETTER II", "text_block": "Letter two chapter text."},
        {"chapter_id": "part_p1_c3", "title": "CHAPTER I", "text_block": "Chapter one text."},
    ])
    _write_json(str(pipeline_dir / "loop3_scenes.json"), [
        {"scene_id": "part_p1_c1_s1", "text_block": "Legacy scene one.", "boundary_source": "whole_chapter"},
        {"scene_id": "part_p1_c2_s1", "text_block": "Legacy scene two.", "boundary_source": "whole_chapter"},
        {"scene_id": "part_p1_c3_s1", "text_block": "Legacy scene three.", "boundary_source": "whole_chapter"},
    ])
    _write_json(str(pipeline_dir / "loop4_lines.json"), [
        {"scene_id": "part_p1_c1_s1", "lines": [
            {"line_id": "l1", "chapter": 1, "scene": 1, "line_number": 1, "character": "Narrator", "speaker_id": "char_narrator", "segment_type": "narrative", "text": "Legacy scene one."}
        ]},
        {"scene_id": "part_p1_c2_s1", "lines": [
            {"line_id": "l2", "chapter": 2, "scene": 2, "line_number": 1, "character": "Walton", "speaker_id": "char_walton", "segment_type": "dialogue", "text": "Legacy scene two.", "attribution_method": "LLM+reviewed"}
        ]},
        {"scene_id": "part_p1_c3_s1", "lines": [
            {"line_id": "l3", "chapter": 3, "scene": 3, "line_number": 1, "character": "Narrator", "speaker_id": "char_narrator", "segment_type": "narrative", "text": "Legacy scene three."}
        ]},
    ])
    return pipeline_root, pipeline_dir


def test_console_api_structure_helpers_round_trip(tmp_path, monkeypatch):
    pipeline_root, pipeline_dir = _fixture_pipeline(tmp_path)
    monkeypatch.chdir(tmp_path)

    payload = console_api.get_book_structure("frankenstein")
    assert payload["structure"]["book_id"] == "frankenstein"

    result = console_api.apply_structure_edit(
        "frankenstein",
        "rename_section",
        {"section_id": "sec_part_p1_c1", "title": "Letter One"},
    )
    assert result["structure"]["structure_version"] == 2
    assert load_book_structure(str(pipeline_dir / "book_structure.json")).structure_version == 2
    legacy = json.loads((pipeline_dir / "loop2_chapters.json").read_text(encoding="utf-8"))
    assert legacy[0]["title"] == "LETTER I"


def test_render_start_fails_on_stale_structure(tmp_path, monkeypatch):
    pipeline_root, pipeline_dir = _fixture_pipeline(tmp_path)
    monkeypatch.chdir(tmp_path)
    structure = load_structure("frankenstein", source_file="data/uploads/frankenstein.txt")
    stale = console_api.apply_structure_edit(
        "frankenstein",
        "split_section",
        {
            "section_id": "sec_part_p1_c1_s1",
            "sections": [
                {"title": "Letter one A", "content_type": "scene"},
                {"title": "Letter one B", "content_type": "scene"},
            ],
        },
    )

    result = render_job.start_render("frankenstein", 1)
    assert result["status"] == "failed"
    assert "Canonical structure is stale for render" in result["error"]


def test_structure_refresh_restores_render_readiness_and_rebuilds_cache(tmp_path, monkeypatch):
    pipeline_root, pipeline_dir = _fixture_pipeline(tmp_path)
    monkeypatch.chdir(tmp_path)
    load_structure("frankenstein", source_file="data/uploads/frankenstein.txt")
    console_api.apply_structure_edit(
        "frankenstein",
        "merge_sections",
        {
            "section_ids": ["sec_part_p1_c1", "sec_part_p1_c2"],
            "title": "Letters",
            "content_type": "letter",
        },
    )

    tier3_dir = pipeline_root / "frankenstein" / "tier3"
    _write_json(str(tier3_dir / "production_script.json"), [{"scene_id": "part_p1_c1_s1"}])

    refreshed = console_api.refresh_book_structure("frankenstein")
    assert refreshed["readiness"]["ok"] is True
    assert refreshed["hierarchy"]["metadata"]["book_structure_version"] == refreshed["structure"]["structure_version"]
    assert os.path.exists(tmp_path / "data" / "processed" / "frankenstein" / "Tier_1" / "hierarchy.json")
    assert os.path.exists(tmp_path / "data" / "processed" / "frankenstein" / "Tier_1" / "profile.json")
    assert refreshed["artifacts"]["cleared_tier3"]
    assert not (tier3_dir / "production_script.json").exists()

    class _Proc:
        pid = 4242

    monkeypatch.setattr(render_job.subprocess, "Popen", lambda *args, **kwargs: _Proc())
    queued = render_job.start_render("frankenstein", 1)
    assert queued["status"] == "queued"


def test_render_job_uses_canonical_manifest_order(tmp_path, monkeypatch):
    pipeline_root, pipeline_dir = _fixture_pipeline(tmp_path)
    monkeypatch.chdir(tmp_path)
    structure = load_structure("frankenstein", source_file="data/uploads/frankenstein.txt")
    edited = reorder_section(structure, "sec_part_p1_c3", 1)
    save_book_structure(edited, str(pipeline_dir / "book_structure.json"))

    captured = {}

    def fake_mix_voice_track(manifest_path, out_wav, single_narrator=False):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        captured["chapter_titles"] = [chapter["title"] for part in manifest["parts"] for chapter in part["chapters"]]
        return {"output": out_wav, "m4b": None, "timings": None}

    monkeypatch.setitem(sys.modules, "src.llm_client", types.SimpleNamespace(set_usage_context=lambda **kwargs: None))
    monkeypatch.setitem(
        sys.modules,
        "src.production_mixer",
        types.SimpleNamespace(mix_voice_track=fake_mix_voice_track, mix_production=fake_mix_voice_track),
    )
    os.makedirs("scratch/renders", exist_ok=True)

    job = {
        "job_id": "job123",
        "book": "frankenstein",
        "source_file": "data/uploads/frankenstein.txt",
        "tier": 1,
        "owner": "local",
        "project_id": None,
        "status": "queued",
        "created_at": 0,
        "started_at": None,
        "finished_at": None,
        "pid": None,
        "output_wav": None,
        "output_m4b": None,
        "timings": None,
        "error": None,
    }
    os.makedirs("data/render_jobs", exist_ok=True)
    with open("data/render_jobs/job123.json", "w", encoding="utf-8") as handle:
        json.dump(job, handle)

    exit_code = render_job.run_job("job123")

    assert exit_code == 0
    assert captured["chapter_titles"] == ["CHAPTER I", "LETTER I", "LETTER II"]


def test_scene_director_prefers_canonical_scene_texts(tmp_path, monkeypatch):
    pipeline_root, pipeline_dir = _fixture_pipeline(tmp_path)
    monkeypatch.chdir(tmp_path)
    structure = load_structure("frankenstein", source_file="data/uploads/frankenstein.txt")
    scene = next(section for section in structure.sections if section.legacy_id == "part_p1_c1_s1")
    scene.metadata["text_block"] = "Canonical scene one."
    save_book_structure(structure, str(pipeline_dir / "book_structure.json"))

    manifest = structure_to_manifest(structure, line_payloads=load_line_payloads("frankenstein"))
    scene_texts = scene_director._canonical_scene_texts("frankenstein", manifest)

    assert scene_texts["part_p1_c1_s1"] == "Canonical scene one."


def test_canonical_edits_drive_console_tree_and_leave_legacy_unchanged(tmp_path, monkeypatch):
    pipeline_root, pipeline_dir = _fixture_pipeline(tmp_path)
    monkeypatch.chdir(tmp_path)
    original = json.loads((pipeline_dir / "loop2_chapters.json").read_text(encoding="utf-8"))

    console_api.apply_structure_edit(
        "frankenstein",
        "rename_section",
        {"section_id": "sec_part_p1_c3", "title": "Prologue"},
    )

    tree = console_api.book_tree("frankenstein")
    current = json.loads((pipeline_dir / "loop2_chapters.json").read_text(encoding="utf-8"))

    assert tree["parts"][0]["chapters"][2]["title"] == "Prologue"
    assert current == original


def test_structure_refresh_updates_gui_hierarchy_from_canonical_titles(tmp_path, monkeypatch):
    pipeline_root, pipeline_dir = _fixture_pipeline(tmp_path)
    monkeypatch.chdir(tmp_path)

    console_api.apply_structure_edit(
        "frankenstein",
        "rename_section",
        {"section_id": "sec_part_p1_c3", "title": "Opening Chapter"},
    )
    refreshed = console_api.refresh_book_structure("frankenstein")

    chapter_titles = [chapter["chapter_title"] for chapter in refreshed["hierarchy"]["parts"][0]["chapters"]]
    legacy = json.loads((pipeline_dir / "loop2_chapters.json").read_text(encoding="utf-8"))

    assert "Opening Chapter" in chapter_titles
    assert legacy[2]["title"] == "CHAPTER I"


def test_director_refresh_rebuilds_tier3_from_canonical_manifest(tmp_path, monkeypatch):
    pipeline_root, pipeline_dir = _fixture_pipeline(tmp_path)
    monkeypatch.chdir(tmp_path)
    structure = load_structure("frankenstein", source_file="data/uploads/frankenstein.txt")
    edited = reorder_section(structure, "sec_part_p1_c3", 1)
    save_book_structure(edited, str(pipeline_dir / "book_structure.json"))
    console_api.refresh_book_structure("frankenstein")

    captured = {}

    def fake_direct_manifest(manifest_path, sync_mempalace=False):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        captured["chapter_titles"] = [chapter["title"] for part in manifest["parts"] for chapter in part["chapters"]]
        tier3_dir = tmp_path / "data" / "corpus" / "pipeline" / "frankenstein" / "tier3"
        _write_json(str(tier3_dir / "production_script.json"), [{"scene_id": "part_p1_c3_s1"}])
        return {"tier3_dir": str(tier3_dir), "scenes": 3, "llm_directed": 0}

    def fake_write_artifact(filename):
        def _writer(manifest_path, *args, **kwargs):
            tier3_dir = tmp_path / "data" / "corpus" / "pipeline" / "frankenstein" / "tier3"
            _write_json(str(tier3_dir / filename), [{"ok": True}])
            return str(tier3_dir / filename)
        return _writer

    monkeypatch.setattr(scene_director, "direct_manifest", fake_direct_manifest)
    monkeypatch.setattr(scene_director, "run_sound_design", fake_write_artifact("sound_design.json"))
    monkeypatch.setattr(scene_director, "run_dramatization", fake_write_artifact("dramatization.json"))
    monkeypatch.setattr(scene_director, "run_character_design", fake_write_artifact("character_profiles.json"))

    refreshed = console_api.refresh_book_director("frankenstein")

    assert refreshed["readiness"]["ok"] is True
    assert captured["chapter_titles"] == ["CHAPTER I", "LETTER I", "LETTER II"]
    assert os.path.exists(tmp_path / "data" / "corpus" / "pipeline" / "frankenstein" / "tier3" / "canonical_manifest.json")


def test_director_refresh_can_chain_structure_refresh_for_stale_structure(tmp_path, monkeypatch):
    pipeline_root, pipeline_dir = _fixture_pipeline(tmp_path)
    monkeypatch.chdir(tmp_path)
    load_structure("frankenstein", source_file="data/uploads/frankenstein.txt")
    console_api.apply_structure_edit(
        "frankenstein",
        "split_section",
        {
            "section_id": "sec_part_p1_c1_s1",
            "sections": [
                {"title": "Letter one A", "content_type": "scene"},
                {"title": "Letter one B", "content_type": "scene"},
            ],
        },
    )

    monkeypatch.setattr(scene_director, "direct_manifest", lambda manifest_path, sync_mempalace=False: {"tier3_dir": "tier3", "scenes": 4, "llm_directed": 0})
    monkeypatch.setattr(scene_director, "run_sound_design", lambda manifest_path: "sound_design.json")
    monkeypatch.setattr(scene_director, "run_dramatization", lambda manifest_path, *args, **kwargs: "dramatization.json")
    monkeypatch.setattr(scene_director, "run_character_design", lambda manifest_path, sync_mempalace=False: "character_profiles.json")

    refreshed = console_api.refresh_book_director("frankenstein", refresh_structure=True)

    assert refreshed["structure_refresh"] is not None
    assert refreshed["readiness"]["ok"] is True


def test_director_cast_roster_and_search_use_marketplace_bridge(tmp_path, monkeypatch):
    _fixture_pipeline(tmp_path)
    monkeypatch.chdir(tmp_path)

    class FakeMarketplace:
        def search_marketplace(self, query, limit=5):
            return [{"voice_id": "voice_1", "voice_name": "Warm Narrator", "voice_ref_path": "data/voice_marketplace/v1/reference_mono.wav", "description": query, "score": 0.92}]

    monkeypatch.setattr(console_api, "_get_marketplace", lambda: FakeMarketplace())
    monkeypatch.setattr(console_api, "_get_drawer_info", lambda character: {"voice_ref_path": f"data/voice_refs/{character}.wav"})

    roster = console_api.director_cast_roster("frankenstein", limit=2)
    search = console_api.director_search_voices("frankenstein", "Walton", query="", limit=2)

    assert roster["book"] == "frankenstein"
    assert roster["characters"][0]["character"] == "Walton"
    assert roster["characters"][0]["suggestions"][0]["voice_id"] == "voice_1"
    assert "Walton" in search["query"]
    assert search["results"][0]["voice_name"] == "Warm Narrator"


def test_director_cast_character_supports_specific_listing(tmp_path, monkeypatch):
    _fixture_pipeline(tmp_path)
    monkeypatch.chdir(tmp_path)

    class FakeMarketplace:
        def cast_character_with_voice(self, character_name, voice_id, buyer="local", purpose=""):
            return {"character": character_name, "voice": {"voice_id": voice_id, "voice_ref_path": "data/voice_marketplace/v2/reference_mono.wav"}, "license": {"buyer": buyer, "purpose": purpose}}

        def cast_character(self, character_name, character_description, buyer="local", purpose=""):
            return {"character": character_name, "voice": {"voice_id": "voice_auto", "voice_ref_path": "data/voice_marketplace/auto/reference_mono.wav"}, "license": {"buyer": buyer, "purpose": purpose}}

    monkeypatch.setattr(console_api, "_get_marketplace", lambda: FakeMarketplace())
    monkeypatch.setattr(console_api, "_get_drawer_info", lambda character: {"voice_ref_path": f"data/mempalace/{character}.wav"})

    explicit = console_api.director_cast_character(
        "frankenstein",
        "Walton",
        voice_id="voice_2",
        buyer="director",
        purpose="cast for trailer",
    )
    auto = console_api.director_cast_character("frankenstein", "Walton", description="calm british male")

    assert explicit["cast"]["voice"]["voice_id"] == "voice_2"
    assert explicit["cast"]["license"]["buyer"] == "director"
    assert auto["cast"]["voice"]["voice_id"] == "voice_auto"
