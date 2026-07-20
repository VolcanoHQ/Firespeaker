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

    scene_overrides = load_scene_overrides(book)
    scene_nodes: Dict[str, List[Dict[str, Any]]] = {}
    for sc in scenes:
        sid = sc.get("scene_id", "")
        chap_key = sid.rsplit("_s", 1)[0] if "_s" in sid else ""
        text = sc.get("text_block", "")
        node = {
            "scene_id": sid,
            "omitted": bool(scene_overrides.get(sid, {}).get("omit")),
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
    chapters = _load_json(os.path.join(_tier1_dir(book), "loop2_chapters.json")) or []
    titles = {c.get("chapter_id"): c.get("title", "") for c in chapters}

    out: List[str] = [f"# {book} — Tier {tier} "
                      f"{'narration text' if tier == 1 else 'attributed script'}", ""]
    seen_chapters = set()
    for p in payloads:
        sid = p.get("scene_id", "")
        if omits.get(sid, {}).get("omit"):
            continue
        chap = sid.rsplit("_s", 1)[0]
        if chap not in seen_chapters:
            seen_chapters.add(chap)
            out += ["", f"## {titles.get(chap) or chap}", ""]
        out.append(f"[Scene {sid.rsplit('_s', 1)[-1]}]")
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
        for ext in (".txt", ".epub"):
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
