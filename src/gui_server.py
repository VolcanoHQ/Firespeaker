#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Volcano Studios - Local GUI Server
A zero-dependency standard Python HTTP API server hosting the workspace.
"""

import os
import re
import sys
import json
import time
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.parse

# Ensure the root project directory is in the sys.path for absolute modular imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.hierarchical_parser import HierarchicalParser
from src.manuscript_profiler import ManuscriptProfiler
from src.book_structure_adapter import load_line_payloads, load_structure, structure_to_gui_hierarchy
from src.upload_contract import (
    UploadContractError,
    error_response,
    http_status_for_error,
    process_upload,
    request_from_json,
    success_response,
)

import collections

# Setup logging
LOG_BUFFER_SIZE = 500
log_buffer = collections.deque(maxlen=LOG_BUFFER_SIZE)
log_lock = threading.Lock()

class DequeHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        with log_lock:
            log_buffer.append(log_entry)

# Ensure data directory exists for log file
os.makedirs("data", exist_ok=True)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Stream Handler (Stdout)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)

# File Handler (development.log)
file_handler = logging.FileHandler('development.log', encoding='utf-8')
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Deque Handler (API Polling)
deque_handler = DequeHandler()
deque_handler.setFormatter(formatter)
root_logger.addHandler(deque_handler)

logger = logging.getLogger("StudioServer")

TIER1_GUI_PIPELINE = "tier1_manifest_v1"


def _manifest_to_gui_hierarchy(manifest):
    """Adapt the Tier 1 manifest to the hierarchy schema used by the review GUI."""
    parts = []
    all_characters = set()
    book_stem = os.path.splitext(os.path.basename(manifest.source_file))[0]
    structure_path = os.path.join("data/corpus/pipeline", book_stem, "tier1", "book_structure.json")
    structure_version = None
    if os.path.exists(structure_path):
        try:
            with open(structure_path, "r", encoding="utf-8") as handle:
                structure_version = json.load(handle).get("structure_version")
        except Exception as e:
            logger.warning(f"Could not read canonical book structure for {book_stem}: {e}")

    for part_index, part in enumerate(manifest.parts, 1):
        chapters = []
        for chapter_index, chapter in enumerate(part.chapters, 1):
            scenes = []
            for scene_index, scene in enumerate(chapter.scenes, 1):
                lines = []
                for line in scene.lines:
                    line_data = line.model_dump()
                    line_data["dialogue"] = line_data["text"]
                    line_data["scene_id"] = scene.scene_id
                    lines.append(line_data)
                    if line.character != "Narrator":
                        all_characters.add(line.character)

                dialogue_lines = [line for line in lines if line["segment_type"] == "dialogue"]
                narration_words = sum(
                    len(line["text"].split()) for line in lines if line["segment_type"] == "narrative"
                )
                dialogue_words = sum(len(line["text"].split()) for line in dialogue_lines)
                scenes.append({
                    "scene_id": scene.scene_id,
                    "scene_number": scene_index,
                    "raw_scene_text": "\n\n".join(line["text"] for line in lines),
                    "characters_present": sorted({line["character"] for line in dialogue_lines if line["character"] != "Narrator"}),
                    "total_dialogue_lines": len(dialogue_lines),
                    "metrics": {
                        "total_words": narration_words + dialogue_words,
                        "narration_words": narration_words,
                        "dialogue_words": dialogue_words,
                    },
                    "lines": lines,
                })
            chapters.append({
                "chapter_id": chapter.chapter_id,
                "chapter_number": chapter_index,
                "chapter_title": chapter.title,
                "total_scenes": len(scenes),
                "scenes": scenes,
            })
        parts.append({
            "part_id": part.part_id,
            "part_title": part.title,
            "total_chapters": len(chapters),
            "chapters": chapters,
        })

    return {
        "metadata": {
            "source_file": manifest.source_file,
            "quote_style_detected": "double",
            "total_parts": len(parts),
            "total_chapters": manifest.total_chapters,
            "total_scenes": sum(len(chapter["scenes"]) for part in parts for chapter in part["chapters"]),
            "global_characters": ["Narrator", *sorted(all_characters)],
            "merge_decisions": [],
            "analysis_pipeline": TIER1_GUI_PIPELINE,
            "book_structure_version": structure_version,
        },
        "parts": parts,
    }


def build_gui_hierarchy(filepath: str, tier: int):
    """Build hierarchy data using canonical book structure for Tier 1 reads."""
    if int(tier) == 1 or filepath.lower().endswith((".docx", ".epub")):
        book = os.path.splitext(os.path.basename(filepath))[0]
        pipeline_dir = os.path.join("data", "corpus", "pipeline", book, "tier1")
        structure_path = os.path.join(pipeline_dir, "book_structure.json")
        if os.path.exists(structure_path) or os.path.isdir(pipeline_dir):
            try:
                structure = load_structure(
                    book,
                    source_file=filepath,
                    source_format=os.path.splitext(filepath)[1].lstrip(".").lower() or "txt",
                )
                line_payloads = load_line_payloads(book)
                if line_payloads:
                    return structure_to_gui_hierarchy(structure, line_payloads=line_payloads, analysis_pipeline=TIER1_GUI_PIPELINE)
            except Exception as e:
                logger.warning(f"Canonical structure load failed for {book}; falling back to Tier 1 manifest: {e}")
        from src.tier_1_parser import ingest_manuscript_tier_1
        manifest = ingest_manuscript_tier_1(filepath)
        try:
            structure = load_structure(
                book,
                source_file=filepath,
                source_format=os.path.splitext(filepath)[1].lstrip(".").lower() or "txt",
            )
            line_payloads = load_line_payloads(book)
            if line_payloads:
                return structure_to_gui_hierarchy(structure, line_payloads=line_payloads, analysis_pipeline=TIER1_GUI_PIPELINE)
        except Exception:
            pass
        return _manifest_to_gui_hierarchy(manifest)

    parser = HierarchicalParser(use_gpu=False, production_tier=int(tier))
    return parser.parse_hierarchy(filepath)


def resolve_manuscript_path(filename: str):
    if not filename or "/" in filename or "\\" in filename:
        return None
    from src import render_job

    for root in render_job.CORPUS_ROOTS:
        candidate = os.path.join(root, filename)
        if os.path.exists(candidate):
            return candidate

    stem, ext = os.path.splitext(filename)
    if ext:
        return None
    return render_job.find_source(stem)


# Global non-blocking pipeline compilation state
PIPELINE_STATUS = {
    "status": "idle",       # "idle", "running", "completed", "failed"
    "progress": 0,          # 0 to 100
    "step": "",             # active step description
    "error": None,
    "qc_report": None
}

def bg_run_pipeline(filename: str, tier: int = 1, user_tier: str = "free"):
    """
    Executes the CalderaPipeline full run on a background thread.
    Updates the global PIPELINE_STATUS object for real-time progress polling.
    """
    global PIPELINE_STATUS
    try:
        from src.main import CalderaPipeline
        filepath = resolve_manuscript_path(filename)
        if not filepath:
            raise FileNotFoundError(f"Manuscript not found: {filename}")
        
        logger.info(f"[BG Compiler] Initializing pipeline run for {filename} (Tier {tier}, User Tier {user_tier})...")
        PIPELINE_STATUS["status"] = "running"
        PIPELINE_STATUS["step"] = "Preparing workspace & loading character drawers..."
        PIPELINE_STATUS["progress"] = 15
        
        pipeline = CalderaPipeline(production_tier=tier)
        
        logger.info("[BG Compiler] Ingesting and parsing script lines...")
        PIPELINE_STATUS["step"] = "Running manuscript text parsing & LLM dialogue attribution..."
        PIPELINE_STATUS["progress"] = 40
        
        logger.info("[BG Compiler] Initiating speech synthesis and track mixing...")
        PIPELINE_STATUS["step"] = "Synthesizing voice lines and compiling ambient sidechain filters..."
        PIPELINE_STATUS["progress"] = 75
        
        output_master = "scratch/pipeline_workspace/output_master.wav"
        success = pipeline.run_full_pipeline(filepath, output_master, user_tier=user_tier)
        
        if success:
            logger.info("[BG Compiler] Performing mathematical ACX compliant QC check...")
            PIPELINE_STATUS["step"] = "Analyzing master peak and RMS loudness metrics..."
            PIPELINE_STATUS["progress"] = 95
            
            qc_report = {}
            qc_report_path = "scratch/master_qc_report.json"
            if os.path.exists(qc_report_path):
                with open(qc_report_path, "r", encoding="utf-8") as f:
                    qc_report = json.load(f)
                    
            PIPELINE_STATUS["status"] = "completed"
            PIPELINE_STATUS["progress"] = 100
            PIPELINE_STATUS["step"] = "Audiobook compiled successfully! ACX QC check passed."
            PIPELINE_STATUS["qc_report"] = qc_report
            logger.info("[BG Compiler] Pipeline successfully finished!")
        else:
            logger.error("[BG Compiler] physical mastering QC check failed.")
            PIPELINE_STATUS["status"] = "failed"
            PIPELINE_STATUS["progress"] = 100
            PIPELINE_STATUS["step"] = "Compilation failed: mastering QC limit checks failed."
            PIPELINE_STATUS["error"] = "Audio mastering QC checks failed. Physical waves exceed standard ACX limits."
    except Exception as e:
        logger.error(f"[BG Compiler] Pipeline crashed: {e}", exc_info=True)
        PIPELINE_STATUS["status"] = "failed"
        PIPELINE_STATUS["progress"] = 100
        PIPELINE_STATUS["step"] = f"Compilation failed: {e}"
        PIPELINE_STATUS["error"] = str(e)


class StudioRequestHandler(BaseHTTPRequestHandler):
    """Handles REST API requests and serves the studio Single-Page Application."""

    # ------------------------------------------------------------------
    # Auth (T2-1): OFF by default -- identical legacy behavior, owner "local".
    # CALDERA_AUTH=on refuses unauthenticated API access on every surface.
    # ------------------------------------------------------------------
    _AUTH_EXEMPT = ("/login", "/api/auth/")

    def _current_user(self):
        from src import user_db
        cookies = self.headers.get("Cookie") or ""
        for part in cookies.split(";"):
            if "=" in part:
                k, v = part.strip().split("=", 1)
                if k == user_db.COOKIE_NAME:
                    return user_db.session_user(v)
        return None

    def _owner(self):
        user = self._current_user()
        return user["user_id"] if user else "local"

    def _auth_gate(self, path) -> bool:
        """True = allowed to proceed. When auth is on, everything except the
        login surface requires a valid session (401 for APIs, 302 for pages)."""
        from src import user_db
        if not user_db.auth_enabled():
            return True
        if any(path == e or path.startswith(e) for e in self._AUTH_EXEMPT):
            return True
        if self._current_user():
            return True
        if path.startswith("/api/"):
            self.send_json_error(401, "Authentication required")
        else:
            self.send_response(302)
            self.send_header("Location", "/login")
            self.end_headers()
        return False

    def handle_auth(self, path, query_string, body=None):
        from src import user_db
        params = urllib.parse.parse_qs(query_string or "")
        q = lambda k: (params.get(k) or [""])[0]
        body = body or {}
        if path == "/api/auth/request_link":
            ok = user_db.request_login(body.get("email", ""))
            if not ok:
                self.send_json_error(400, "Valid email required")
                return
            payload, cookie = {"sent": True, "note": f"Dev outbox: {user_db.OUTBOX_DIR}/"}, None
        elif path == "/api/auth/redeem":
            session = user_db.redeem_code(q("code") or body.get("code", ""))
            if not session:
                self.send_json_error(400, "Invalid or expired code")
                return
            cookie = (f"{user_db.COOKIE_NAME}={session['token']}; Path=/; HttpOnly; "
                      f"SameSite=Lax; Max-Age={user_db.SESSION_TTL_S}")
            if self.command == "GET":
                self.send_response(302)
                self.send_header("Set-Cookie", cookie)
                self.send_header("Location", "/console")
                self.end_headers()
                return
            payload = {"email": session["email"], "user_id": session["user_id"]}
        elif path == "/api/auth/profile":
            user = self._current_user()
            uid = user["user_id"] if user else "local"
            if self.command == "POST":
                payload = user_db.update_profile(
                    uid, display_name=body.get("display_name"),
                    roles=body.get("roles"), bio=body.get("bio"))
            else:
                payload = user_db.get_profile(uid)
            if payload is None:
                self.send_json_error(404, "No such user")
                return
            cookie = None
        elif path == "/api/auth/me":
            user = self._current_user()
            if not user:
                self.send_json_error(401, "Not signed in")
                return
            payload, cookie = user, None
        elif path == "/api/auth/logout":
            cookies = self.headers.get("Cookie") or ""
            for part in cookies.split(";"):
                if part.strip().startswith(user_db.COOKIE_NAME + "="):
                    user_db.logout(part.strip().split("=", 1)[1])
            payload = {"signed_out": True}
            cookie = f"{user_db.COOKIE_NAME}=; Path=/; Max-Age=0"
        else:
            self.send_json_error(404, "Unknown auth endpoint")
            return
        resp = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(resp)

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path.startswith("/api/auth/"):
            self.handle_auth(path, parsed_url.query)
            return
        if path == "/login":
            self.serve_static_file("src/static/login.html", "text/html")
            return
        if not self._auth_gate(path):
            return
        if path == "/api/account/overview":
            from src import user_db
            from src.project_db import ProjectDB
            import glob as _glob
            try:
                user = self._current_user()
                uid = user["user_id"] if user else "local"
                profile = user_db.get_profile(uid)
                projects = ProjectDB().list_product_projects(owner=uid)
                listings = [l for l in self._get_marketplace().list_all()
                            if (l.get("seller_id") or "local") == uid]
                datasets = []
                for sp in _glob.glob("data/voice_datasets/*/studio_session.json"):
                    try:
                        with open(sp, encoding="utf-8") as f:
                            st = json.load(f)
                        if (st.get("owner") or "local") == uid:
                            datasets.append({"session": st.get("session"),
                                             "speaker": st.get("speaker"),
                                             "built": st.get("built"),
                                             "published": st.get("published")})
                    except Exception:
                        continue
                jobs = []
                for jp in sorted(_glob.glob("data/render_jobs/*.json"), reverse=True):
                    try:
                        with open(jp, encoding="utf-8") as f:
                            j = json.load(f)
                        if (j.get("owner") or "local") == uid:
                            jobs.append({"job_id": j["job_id"], "book": j["book"],
                                         "tier": j["tier"], "status": j["status"]})
                    except Exception:
                        continue
                payload = {"profile": profile, "projects": projects,
                           "voice_listings": listings, "datasets": datasets,
                           "renders": jobs[:10]}
                resp = json.dumps(payload, default=str).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(resp)
            except Exception as e:
                logger.error(f"account overview error: {e}")
                self.send_json_error(500, f"Overview error: {e}")
            return
        if path == "/" or path == "/index.html":
            self.serve_static_file("src/static/index.html", "text/html")
        elif path == "/console":
            self.serve_static_file("src/static/console.html", "text/html")
        elif path == "/voicestudio":
            self.serve_static_file("src/static/voicestudio.html", "text/html")
        elif path.startswith("/api/voicestudio/"):
            self.handle_voicestudio(path, parsed_url.query, body=None)
        elif path.startswith("/api/console/"):
            self.handle_get_console(path, parsed_url.query)
        elif path.startswith("/api/marketplace/"):
            self.handle_marketplace(path, parsed_url.query, body=None)
        elif path == "/api/books":
            self.handle_get_books()
        elif path == "/api/cast":
            self.handle_get_cast()
        elif path == "/api/pipeline_status":
            self.handle_get_pipeline_status()
        elif path == "/api/logs":
            self.handle_get_logs()
        elif path == "/api/download":
            self.handle_get_download()
        elif path == "/api/preview_voice":
            self.handle_get_preview_voice(parsed_url.query)
        else:
            self.send_error(404, "File not found")

    _marketplace = None

    @classmethod
    def _get_marketplace(cls):
        """Lazy singleton: the local Qdrant store allows one client per process,
        so the server holds a single instance for all marketplace requests."""
        if cls._marketplace is None:
            from src.voice_marketplace import VoiceMarketplace
            cls._marketplace = VoiceMarketplace()
        return cls._marketplace

    def handle_marketplace(self, path, query_string, body=None):
        """Voice Marketplace REST surface over src/voice_marketplace.py.
        GET  /api/marketplace/listings                    -- browse all
        GET  /api/marketplace/search?q=...&limit=5        -- semantic search
        POST /api/marketplace/upload_sample {filename, data: dataURL} -> {path}
        POST /api/marketplace/onboard {seller, name, samples[], description, price, consent}
        POST /api/marketplace/purchase {voice_id, buyer, purpose}
        POST /api/marketplace/cast {character, description, buyer}
        """
        try:
            params = urllib.parse.parse_qs(query_string or "")
            q = lambda k, d="": (params.get(k) or [d])[0]
            mp = self._get_marketplace()
            if path == "/api/marketplace/listings":
                payload = {"listings": mp.list_all()}
            elif path == "/api/marketplace/search":
                if not q("q"):
                    self.send_json_error(400, "Missing query parameter q")
                    return
                payload = {"results": mp.search_marketplace(q("q"), limit=int(q("limit", "5")))}
            elif path == "/api/marketplace/upload_sample" and body is not None:
                filename = os.path.basename(body.get("filename") or "")
                data_url = body.get("data") or ""
                if not filename or "," not in data_url:
                    self.send_json_error(400, "Need filename and data (dataURL)")
                    return
                import base64
                raw = base64.b64decode(data_url.split(",", 1)[1])
                updir = "data/voice_marketplace/uploads"
                os.makedirs(updir, exist_ok=True)
                dest = os.path.join(updir, filename)
                with open(dest, "wb") as f:
                    f.write(raw)
                payload = {"path": dest, "bytes": len(raw)}
            elif path == "/api/marketplace/onboard" and body is not None:
                listing = mp.onboard_voice(
                    seller_name=body.get("seller", ""),
                    voice_name=body.get("name", ""),
                    sample_wav_paths=body.get("samples", []),
                    description=body.get("description", ""),
                    price_usd=float(body.get("price", 0.0)),
                    consent_confirmed=bool(body.get("consent", False)),
                )
                payload = {"listing": listing}
            elif path == "/api/marketplace/purchase" and body is not None:
                payload = {"license": mp.purchase_voice(
                    voice_id=body.get("voice_id", ""),
                    buyer=body.get("buyer", "local"),
                    purpose=body.get("purpose", ""),
                )}
            elif path == "/api/marketplace/cast" and body is not None:
                result = mp.cast_character(
                    character_name=body.get("character", ""),
                    character_description=body.get("description", ""),
                    buyer=body.get("buyer", "local"),
                    purpose=body.get("purpose", "audiobook production"),
                )
                if result is None:
                    self.send_json_error(404, "No suitable voice found for that description")
                    return
                payload = {"cast": result}
            else:
                self.send_json_error(404, "Unknown marketplace endpoint")
                return
            resp = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(resp)
        except ValueError as e:
            self.send_json_error(400, str(e))
        except Exception as e:
            logger.error(f"Marketplace endpoint error ({path}): {e}")
            self.send_json_error(500, f"Marketplace error: {e}")

    def handle_voicestudio(self, path, query_string, body=None):
        """Voice Cloning Studio wizard REST surface over src/voice_studio.py.
        POST /api/voicestudio/start     {name, speaker}
        POST /api/voicestudio/record    {session, prompt_id, kind, data: dataURL}
        POST /api/voicestudio/questionnaire {session, answers}
        POST /api/voicestudio/build     {session}
        POST /api/voicestudio/preview   {session, text, pitch, speed}   (slow: XTTS)
        POST /api/voicestudio/persona   {session, label, description, pitch, speed}
        POST /api/voicestudio/publish   {session, seller, description, price, consent}
        GET  /api/voicestudio/session?name=...
        """
        from src import voice_studio
        try:
            body = body or {}
            if path == "/api/voicestudio/session":
                params = urllib.parse.parse_qs(query_string or "")
                name = (params.get("name") or [""])[0]
                payload = voice_studio.start_session(name, name)
            elif path == "/api/voicestudio/start":
                payload = voice_studio.start_session(body.get("name", ""), body.get("speaker", ""),
                                                     owner=self._owner())
            elif path == "/api/voicestudio/record":
                payload = voice_studio.save_recording(
                    body.get("session", ""), body.get("prompt_id", ""),
                    body.get("data", ""), kind=body.get("kind", "prompt"))
            elif path == "/api/voicestudio/questionnaire":
                payload = voice_studio.save_questionnaire(body.get("session", ""), body.get("answers", {}))
            elif path == "/api/voicestudio/build":
                payload = voice_studio.build(body.get("session", ""))
            elif path == "/api/voicestudio/preview":
                payload = voice_studio.preview(
                    body.get("session", ""), body.get("text", ""),
                    pitch=float(body.get("pitch", 0.0)), speed=float(body.get("speed", 1.0)))
            elif path == "/api/voicestudio/persona":
                payload = voice_studio.save_persona(
                    body.get("session", ""), body.get("label", ""), body.get("description", ""),
                    pitch=float(body.get("pitch", 0.0)), speed=float(body.get("speed", 1.0)))
            elif path == "/api/voicestudio/publish":
                payload = voice_studio.publish(
                    body.get("session", ""), body.get("seller", ""), body.get("description", ""),
                    float(body.get("price", 0.0)), bool(body.get("consent", False)),
                    self._get_marketplace())
            else:
                self.send_json_error(404, "Unknown voicestudio endpoint")
                return
            if payload is None:
                self.send_json_error(400, "Invalid session or payload")
                return
            resp = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(resp)
        except ValueError as e:
            self.send_json_error(400, str(e))
        except Exception as e:
            logger.error(f"VoiceStudio endpoint error ({path}): {e}")
            self.send_json_error(500, f"VoiceStudio error: {e}")

    def handle_get_console(self, path, query_string):
        """Review Console (Phase 1, read-only): thin dispatch over console_api."""
        from src import console_api
        try:
            params = urllib.parse.parse_qs(query_string or "")
            q = lambda k: (params.get(k) or [""])[0]
            if path == "/api/console/books":
                payload = console_api.list_books()
            elif path == "/api/console/book":
                payload = console_api.book_tree(q("name"))
            elif path == "/api/console/scene":
                payload = console_api.scene_detail(q("book"), q("scene"))
            elif path == "/api/console/structure":
                payload = console_api.get_book_structure(q("book"))
                if payload is None:
                    self.send_json_error(400, "Invalid or unknown book")
                    return
            elif path == "/api/console/director_cast":
                payload = console_api.director_cast_roster(q("book"), limit=int(q("limit") or "3"))
                if payload is None:
                    self.send_json_error(400, "Invalid or unknown book")
                    return
            elif path == "/api/console/director_voice_search":
                payload = console_api.director_search_voices(
                    q("book"),
                    q("character"),
                    query=q("q"),
                    limit=int(q("limit") or "5"),
                )
                if payload is None:
                    self.send_json_error(400, "Invalid book or character")
                    return
            elif path == "/api/console/progress":
                payload = console_api.progress()
            elif path == "/api/console/trailer_scene":
                from src import tier_preview
                payload = tier_preview.pick_trailer_scene(q("book"))
            elif path == "/api/console/renders":
                from src import render_job
                payload = {"jobs": render_job.list_jobs(q("book") or None)}
            elif path == "/api/console/projects":
                from src.project_db import ProjectDB
                from src import user_db
                owner = self._owner() if user_db.auth_enabled() else None
                projects = ProjectDB().list_product_projects(owner=owner)
                if q("book"):
                    projects = [p for p in projects if p.get("book_stem") == q("book")]
                payload = {"projects": projects}
            elif path == "/api/console/usage":
                if not q("project") and not q("book"):
                    self.send_json_error(400, "Provide project= or book=")
                    return
                payload = console_api.usage_summary(project_id=q("project"), book=q("book"))
            elif path == "/api/console/mix_timeline":
                from src import mix_timeline
                payload = mix_timeline.scene_timeline(q("book"), q("scene"))
            elif path == "/api/console/export":
                result = console_api.export_manuscript(q("book"), int(q("tier") or 0))
                if result is None:
                    self.send_json_error(400, "Invalid book or tier")
                    return
                if "error" in result:
                    self.send_json_error(409, result["error"])
                    return
                body_bytes = result["text"].encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{result["filename"]}"')
                self.send_header("Content-Length", str(len(body_bytes)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body_bytes)
                return
            elif path == "/api/console/health":
                from src import boot_check
                import json as _json
                try:
                    with open(boot_check.REPORT_PATH, encoding="utf-8") as f:
                        payload = _json.load(f)
                    if time.time() - payload.get("at", 0) > 300:
                        payload = boot_check.run_boot_checks(fast=True)
                except Exception:
                    payload = boot_check.run_boot_checks(fast=True)
            elif path == "/api/console/audio":
                wav = console_api.resolve_audio(q("file"))
                if not wav:
                    self.send_json_error(404, "Audio not found or not allowed")
                    return
                with open(wav, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(content)
                return
            else:
                self.send_json_error(404, "Unknown console endpoint")
                return
            if payload is None:
                self.send_json_error(404, "Not found")
                return
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            logger.error(f"Console endpoint error ({path}): {e}")
            self.send_json_error(500, f"Console error: {e}")

    def handle_get_download(self):
        try:
            file_path = "scratch/pipeline_workspace/output_master.wav"
            if not os.path.exists(file_path):
                os.makedirs("scratch/pipeline_workspace", exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x02\x00\x44\xac\x00\x00\x10\xb1\x02\x00\x04\x00\x10\x00data\x00\x00\x00\x00')
            
            with open(file_path, "rb") as f:
                content = f.read()
                
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Disposition", "attachment; filename=audiobook_mastered.wav")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            logger.error(f"Error serving audio download: {e}")
            self.send_error(500, f"Server error: {e}")

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path.startswith("/api/auth/"):
            try:
                content_length = int(self.headers.get('Content-Length') or 0)
                body = json.loads(self.rfile.read(content_length).decode('utf-8')) if content_length else {}
            except Exception:
                body = {}
            self.handle_auth(path, parsed_url.query, body=body)
            return
        if not self._auth_gate(path):
            return
        if path.startswith("/api/marketplace/") or path.startswith("/api/voicestudio/") or path in ("/api/console/correct_speaker", "/api/console/preview_tier", "/api/console/render", "/api/console/projects", "/api/console/project_update", "/api/console/mix_override", "/api/console/omit_scene", "/api/console/upload_source", "/api/console/delete_book", "/api/console/structure_edit", "/api/console/structure_refresh", "/api/console/director_refresh", "/api/console/director_cast_character"):
            try:
                content_length = int(self.headers.get('Content-Length') or 0)
                body = json.loads(self.rfile.read(content_length).decode('utf-8')) if content_length else {}
            except Exception:
                self.send_json_error(400, "Invalid JSON body")
                return
            if path.startswith("/api/marketplace/"):
                self.handle_marketplace(path, parsed_url.query, body=body)
            elif path in ("/api/console/projects", "/api/console/project_update"):
                from src.project_db import ProjectDB
                from src import console_api, render_job
                try:
                    db = ProjectDB()
                    if path == "/api/console/projects":
                        book = console_api._safe_book(body.get("book", ""))
                        if not book:
                            self.send_json_error(400, "Unknown book")
                            return
                        project = db.create_product_project(
                            book, render_job.find_source(book) or "",
                            owner=self._owner(),
                            tier=int(body.get("tier", 1)), plan=body.get("plan", "free"))
                    else:
                        project = db.get_product_project(body.get("project_id", ""))
                        if not project:
                            self.send_json_error(404, "No such project")
                            return
                        if project["owner"] != self._owner():
                            self.send_json_error(403, "Not your project")
                            return
                        project = db.update_product_project(
                            body["project_id"],
                            tier=int(body["tier"]) if body.get("tier") is not None else None,
                            plan=body.get("plan"))
                    resp = json.dumps({"project": project}, default=str).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(resp)
                except Exception as e:
                    logger.error(f"project endpoint error: {e}")
                    self.send_json_error(500, f"Project error: {e}")
            elif path == "/api/console/render":
                from src import render_job
                try:
                    # adopt-on-first-render: every render belongs to a project;
                    # books without one get a project at the requested tier
                    from src.project_db import ProjectDB
                    db = ProjectDB()
                    owner = self._owner()
                    book = body.get("book", "")
                    project = db.get_project_for_book(book, owner=owner) or db.get_project_for_book(book)
                    if not project and render_job.find_source(book):
                        project = db.create_product_project(
                            book, render_job.find_source(book) or "",
                            owner=owner, tier=int(body.get("tier", 1)))
                    job = render_job.start_render(
                        book, int(body.get("tier", 1)),
                        owner=owner,
                        project_id=(project or {}).get("id"))
                    code = 200 if job.get("status") != "failed" else 409
                    resp = json.dumps(job, default=str).encode("utf-8")
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(resp)
                except Exception as e:
                    logger.error(f"render start error: {e}")
                    self.send_json_error(500, f"Render error: {e}")
            elif path == "/api/console/delete_book":
                from src import console_api as _ca
                try:
                    result = _ca.delete_book(body.get("book", ""))
                    if result is None:
                        self.send_json_error(400, "Invalid book")
                        return
                    resp = json.dumps(result).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(resp)
                except Exception as e:
                    logger.error(f"delete_book error: {e}")
                    self.send_json_error(500, f"Delete error: {e}")
            elif path == "/api/console/upload_source":
                try:
                    result = process_upload(request_from_json(body, surface="console"))
                    logger.info(f"Source uploaded: {result.source_file} ({result.bytes} bytes)")
                    resp = json.dumps(success_response(result)).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(resp)
                except UploadContractError as e:
                    resp = json.dumps(error_response(e.error)).encode("utf-8")
                    self.send_response(http_status_for_error(e.error))
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(resp)
                except Exception as e:
                    logger.error(f"upload_source error: {e}")
                    self.send_json_error(500, f"Upload error: {e}")
            elif path == "/api/console/omit_scene":
                from src import console_api as _ca
                try:
                    result = _ca.save_scene_override(
                        body.get("book", ""), body.get("scene_id", ""),
                        omit=bool(body.get("omit")))
                    if result is None:
                        self.send_json_error(400, "Invalid book or scene_id")
                        return
                    resp = json.dumps(result).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(resp)
                except Exception as e:
                    logger.error(f"omit_scene error: {e}")
                    self.send_json_error(500, f"Omit error: {e}")
            elif path == "/api/console/mix_override":
                from src import mix_timeline
                try:
                    result = mix_timeline.save_mix_override(
                        body.get("book", ""), body.get("scene_id", ""),
                        body.get("target", ""), body.get("key", ""),
                        mute=body.get("mute"), gain_db=body.get("gain_db"),
                        nudge_s=body.get("nudge_s"), prompt=body.get("prompt"),
                        variant=body.get("variant"))
                    if result is None:
                        self.send_json_error(400, "Invalid override target/key")
                        return
                    resp = json.dumps(result, default=str).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(resp)
                except Exception as e:
                    logger.error(f"mix_override error: {e}")
                    self.send_json_error(500, f"Mix override error: {e}")
            elif path == "/api/console/preview_tier":
                from src import tier_preview
                try:
                    result = tier_preview.render_tier_preview(
                        body.get("book", ""), int(body.get("tier", 0)), body.get("scene_id") or None,
                        force=bool(body.get("force")))
                    if result is None:
                        self.send_json_error(400, "Invalid book or tier")
                        return
                    code = 200 if "wav" in result else 409
                    resp = json.dumps(result, default=str).encode("utf-8")
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(resp)
                except Exception as e:
                    logger.error(f"preview_tier error: {e}")
                    self.send_json_error(500, f"Preview error: {e}")
            elif path == "/api/console/correct_speaker":
                from src import console_api
                try:
                    result = console_api.save_speaker_override(
                        body.get("book", ""), body.get("line_id", ""),
                        body.get("character", ""), body.get("scene_id", ""))
                    if result is None:
                        self.send_json_error(400, "Invalid book or line_id")
                        return
                    resp = json.dumps(result).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(resp)
                except Exception as e:
                    logger.error(f"correct_speaker error: {e}")
                    self.send_json_error(500, f"Override error: {e}")
            elif path == "/api/console/structure_edit":
                from src import console_api
                try:
                    result = console_api.apply_structure_edit(body.get("book", ""), body.get("action", ""), body)
                    if result is None:
                        self.send_json_error(400, "Invalid or unknown book")
                        return
                    resp = json.dumps(result).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(resp)
                except Exception as e:
                    logger.error(f"structure_edit error: {e}")
                    self.send_json_error(409, f"Structure edit error: {e}")
            elif path == "/api/console/structure_refresh":
                from src import console_api
                try:
                    result = console_api.refresh_book_structure(body.get("book", ""))
                    if result is None:
                        self.send_json_error(400, "Invalid or unknown book")
                        return
                    resp = json.dumps(result).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(resp)
                except Exception as e:
                    logger.error(f"structure_refresh error: {e}")
                    self.send_json_error(409, f"Structure refresh error: {e}")
            elif path == "/api/console/director_refresh":
                from src import console_api
                try:
                    result = console_api.refresh_book_director(
                        body.get("book", ""),
                        refresh_structure=bool(body.get("refresh_structure")),
                        include_qc=bool(body.get("include_qc")),
                        sync_mempalace=bool(body.get("sync_mempalace")),
                    )
                    if result is None:
                        self.send_json_error(400, "Invalid or unknown book")
                        return
                    resp = json.dumps(result).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(resp)
                except Exception as e:
                    logger.error(f"director_refresh error: {e}")
                    self.send_json_error(409, f"Director refresh error: {e}")
            elif path == "/api/console/director_cast_character":
                from src import console_api
                try:
                    result = console_api.director_cast_character(
                        body.get("book", ""),
                        body.get("character", ""),
                        description=body.get("description", ""),
                        buyer=body.get("buyer", self._owner()),
                        purpose=body.get("purpose", "audiobook production"),
                        voice_id=body.get("voice_id", ""),
                    )
                    if result is None:
                        self.send_json_error(400, "Invalid book or character")
                        return
                    resp = json.dumps(result).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(resp)
                except Exception as e:
                    logger.error(f"director_cast_character error: {e}")
                    self.send_json_error(409, f"Director cast error: {e}")
            else:
                self.handle_voicestudio(path, parsed_url.query, body=body)
        elif path == "/api/analyze":
            self.handle_post_analyze()
        elif path == "/api/upload":
            self.handle_post_upload()
        elif path == "/api/confirm_merge":
            self.handle_post_confirm_merge()
        elif path == "/api/update_character":
            self.handle_post_update_character()
        elif path == "/api/process_manuscript":
            self.handle_post_process_manuscript()
        elif path == "/api/process_scenes_async":
            self.handle_post_process_scenes_async()
        elif path == "/api/override_line_speaker":
            self.handle_post_override_line_speaker()
        elif path == "/api/verify_line":
            self.handle_post_verify_line()
        elif path == "/api/verify_aspect":
            self.handle_post_verify_aspect()
        elif path == "/api/save_hierarchy":
            self.handle_post_save_hierarchy()
        elif path == "/api/telemetry/correction":
            self.handle_post_telemetry_correction()
        else:
            self.send_error(404, "Endpoint not found")

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests gracefully."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def serve_static_file(self, filepath, content_type):
        try:
            if not os.path.exists(filepath):
                self.send_error(404, f"File {filepath} not found")
                return
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            logger.error(f"Error serving static file: {e}")
            self.send_error(500, f"Server error: {e}")

    def handle_get_books(self):
        try:
            from src import render_job

            files = set()
            for root in render_job.CORPUS_ROOTS:
                os.makedirs(root, exist_ok=True)
                for name in os.listdir(root):
                    if name.endswith(render_job.SOURCE_EXTS):
                        files.add(name)
            files = sorted(files)
            
            response = json.dumps({"books": files}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response)
        except Exception as e:
            logger.error(f"Error listing books: {e}")
            self.send_json_error(500, str(e))

    def handle_get_preview_voice(self, query_string):
        try:
            params = urllib.parse.parse_qs(query_string)
            voice_val = params.get("voice", [""])[0]
            text = params.get("text", [""])[0]
            
            if not text:
                text = "Volcano Works is preparing your story preview."
                
            from src.voice_synthesizer import VoiceSynthesizer
            from src.spatial_memory import MemPalace
            
            # Ensure directories exist
            os.makedirs("data/mempalace", exist_ok=True)
            os.makedirs("scratch", exist_ok=True)
            
            palace = MemPalace(db_dir="data/mempalace")
            
            # Register characters if they don't exist
            if not palace.get_character_drawer("Arthur"):
                palace.register_character("Arthur", "data/voice_references/narrator_mono.wav", speed=1.0, pitch=-0.9)
            if not palace.get_character_drawer("Emily"):
                palace.register_character("Emily", "data/voice_references/narrator_mono.wav", speed=1.05, pitch=3.86)
            if not palace.get_character_drawer("Michael"):
                palace.register_character("Michael", "data/voice_references/narrator_mono.wav", speed=0.9, pitch=-3.86)
            if not palace.get_character_drawer("Narrator"):
                palace.register_character("Narrator", "data/voice_references/narrator_mono.wav", speed=1.0, pitch=0.0)
            if "Neural" in voice_val and not palace.get_character_drawer(voice_val):
                palace.register_character(voice_val, voice_val, speed=1.0, pitch=0.0)
                
            # Map selected voice to character
            char_name = "Narrator"
            if voice_val == "preset_narrator_1":
                char_name = "Arthur"
            elif voice_val == "preset_narrator_2":
                char_name = "Emily"
            elif voice_val == "preset_narrator_3":
                char_name = "Michael"
            elif voice_val == "cloned_voice":
                char_name = "Narrator"
            elif "Neural" in voice_val:
                char_name = voice_val
                
            drawer = palace.get_character_drawer(char_name)
            palace.close()
            
            # Prioritize query parameters if present (e.g. from Cast Manager preview sliders)
            speed_mod = 1.0
            pitch_mod = 1.0
            
            if "speed" in params:
                try:
                    speed_mod = float(params.get("speed")[0])
                except ValueError:
                    speed_mod = 1.0
            elif drawer:
                speed_mod = drawer["modulation_config"].get("speed", 1.0)
                
            if "pitch" in params:
                try:
                    pitch_semitones = float(params.get("pitch")[0])
                    pitch_mod = 2.0 ** (pitch_semitones / 12.0)
                except ValueError:
                    pitch_mod = 1.0
            elif drawer:
                pitch_semitones = drawer["modulation_config"].get("pitch", 0.0)
                pitch_mod = 2.0 ** (pitch_semitones / 12.0)
            
            synth = VoiceSynthesizer(mempalace_path="data/mempalace")
            
            import uuid
            preview_filename = f"scratch/preview_{uuid.uuid4().hex}.wav"
            
            synth.synthesize_line(
                character_name=char_name,
                dialogue_text=text,
                target_emotion="Neutral",
                output_wav_path=preview_filename,
                pitch_modifier=pitch_mod,
                speed_modifier=speed_mod
            )
            
            if os.path.exists(preview_filename):
                with open(preview_filename, "rb") as f:
                    content = f.read()
                
                try:
                    os.remove(preview_filename)
                except Exception:
                    pass
                    
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_json_error(500, "Failed to generate preview audio file.")
        except Exception as e:
            logger.error(f"Error serving voice preview: {e}", exc_info=True)
            self.send_json_error(500, str(e))

    def handle_get_logs(self):
        try:
            with log_lock:
                logs_copy = list(log_buffer)
            response = json.dumps({"logs": logs_copy}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response)
        except Exception as e:
            self.send_json_error(500, str(e))

    def handle_post_analyze(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            filename = params.get("filename")
            tier = params.get("tier", 1)
            
            if not filename:
                self.send_json_error(400, "Missing filename parameter")
                return
                
            filepath = resolve_manuscript_path(filename)
            if not filepath or not os.path.exists(filepath):
                self.send_json_error(404, f"Book not found: {filename}")
                return
                
            # Sanitize book name slug
            pass  # re is imported at module level
            base_name = os.path.splitext(filename)[0]
            slug = re.sub(r'[^a-zA-Z0-9_\-]', '', base_name)
            
            # Tier-specific cache paths
            cache_dir = f"data/processed/{slug}/Tier_{tier}"
            hierarchy_cache = os.path.join(cache_dir, "hierarchy.json")
            profile_cache = os.path.join(cache_dir, "profile.json")
            
            # Tier 1 caches created by the legacy hierarchy parser include raw
            # Gutenberg front matter. Regenerate those once through the canonical
            # manifest pipeline; current-format caches retain reviewer edits.
            cache_is_current = False
            if os.path.exists(hierarchy_cache) and os.path.exists(profile_cache):
                with open(hierarchy_cache, "r", encoding="utf-8") as f:
                    cached_hierarchy = json.load(f)
                cache_is_current = int(tier) != 1 or (
                    cached_hierarchy.get("metadata", {}).get("analysis_pipeline") == TIER1_GUI_PIPELINE
                )
                if cache_is_current and (int(tier) == 1 or filepath.lower().endswith((".docx", ".epub"))):
                    try:
                        structure = load_structure(
                            base_name,
                            source_file=filepath,
                            source_format=os.path.splitext(filepath)[1].lstrip(".").lower() or "txt",
                        )
                        cache_is_current = (
                            cached_hierarchy.get("metadata", {}).get("book_structure_version")
                            == structure.structure_version
                        )
                    except Exception:
                        cache_is_current = False

            if cache_is_current:
                logger.info(f"Retrieving cached profiling metrics for: {filename} (Tier {tier})")
                hierarchy_data = cached_hierarchy
                with open(profile_cache, "r", encoding="utf-8") as f:
                    profile_data = json.load(f)
            else:
                logger.info(f"Analyzing and profiling {filename} from scratch (Tier {tier})...")
                profiler = ManuscriptProfiler(use_gpu=False, production_tier=int(tier))
                
                hierarchy_data = build_gui_hierarchy(filepath, int(tier))
                profile_data = profiler.profile_book(filepath, hierarchy_data=hierarchy_data)
                
                # Cache results
                os.makedirs(cache_dir, exist_ok=True)
                with open(hierarchy_cache, "w", encoding="utf-8") as f:
                    json.dump(hierarchy_data, f, indent=4)
                profiler.save_profile(profile_data, profile_cache)
                
            response_data = {
                "profile": profile_data,
                "hierarchy": hierarchy_data
            }
            
            response = json.dumps(response_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response)
            
        except Exception as e:
            logger.error(f"Error during manuscript profiling: {e}", exc_info=True)
            self.send_json_error(500, f"Analysis failed: {str(e)}")

    def handle_post_upload(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            tier = params.get("tier", 1)
            upload_request = request_from_json(params, surface="studio")
            result = process_upload(upload_request)
            filepath = result.source_file

            logger.info(f"Custom book saved successfully to: {filepath}")

            # Analyze immediately through the canonical upload contract/Tier 1 path.
            profiler = ManuscriptProfiler(use_gpu=False, production_tier=int(tier))
            hierarchy_data = build_gui_hierarchy(filepath, 1)
            profile_data = profiler.profile_book(filepath, hierarchy_data=hierarchy_data)
            
            # Save cache
            pass  # re is imported at module level
            filename = result.filename
            base_name = os.path.splitext(filename)[0]
            slug = re.sub(r'[^a-zA-Z0-9_\-]', '', base_name)
            
            cache_dir = f"data/processed/{slug}/Tier_{tier}"
            hierarchy_cache = os.path.join(cache_dir, "hierarchy.json")
            profile_cache = os.path.join(cache_dir, "profile.json")
            os.makedirs(cache_dir, exist_ok=True)
            with open(hierarchy_cache, "w", encoding="utf-8") as f:
                json.dump(hierarchy_data, f, indent=4)
            profiler.save_profile(profile_data, profile_cache)
            
            response_data = success_response(result)
            response_data["profile"] = profile_data
            response_data["hierarchy"] = hierarchy_data
            
            response = json.dumps(response_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response)
        except UploadContractError as e:
            payload = json.dumps(error_response(e.error)).encode("utf-8")
            self.send_response(http_status_for_error(e.error))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            logger.error(f"Error handling file upload: {e}", exc_info=True)
            self.send_json_error(500, f"Upload and analysis failed: {str(e)}")

    def handle_post_confirm_merge(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            filename = params.get("filename")
            original_name = params.get("original_name")
            canonical_name = params.get("canonical_name")
            is_confirmed = params.get("is_confirmed")
            
            if not filename or not original_name or not canonical_name or is_confirmed is None:
                self.send_json_error(400, "Missing required parameters: filename, original_name, canonical_name, is_confirmed")
                return
                
            filepath = resolve_manuscript_path(filename)
            if not filepath or not os.path.exists(filepath):
                self.send_json_error(404, f"Book not found: {filename}")
                return
                
            # 1. Connect to MemPalace database and save decision
            from src.spatial_memory import MemPalace
            palace = MemPalace()
            # Default confidence score is 1.0 if confirmed, or 0.0 if split/rejected
            confidence_score = 1.0 if is_confirmed else 0.0
            palace.save_confirmed_merge(filename, original_name, canonical_name, is_confirmed, confidence_score)
            palace.close()
            
            # 2. Modify existing hierarchy cache dynamically to preserve structural scene splits
            pass  # re is imported at module level
            base_name = os.path.splitext(filename)[0]
            slug = re.sub(r'[^a-zA-Z0-9_\-]', '', base_name)
            tier = params.get("tier", 1)
            
            cache_dir = f"data/processed/{slug}/Tier_{tier}"
            hierarchy_cache = os.path.join(cache_dir, "hierarchy.json")
            profile_cache = os.path.join(cache_dir, "profile.json")
            
            # Load existing cache hierarchy
            if os.path.exists(hierarchy_cache):
                with open(hierarchy_cache, "r", encoding="utf-8") as f:
                    hierarchy_data = json.load(f)
            else:
                hierarchy_data = build_gui_hierarchy(filepath, int(tier))
                
            # If confirmed, merge original_name into canonical_name in the hierarchy cache
            if is_confirmed:
                for part in hierarchy_data.get("parts", []):
                    for chapter in part.get("chapters", []):
                        for scene in chapter.get("scenes", []):
                            for line in scene.get("lines", []):
                                if line.get("character") == original_name:
                                    line["character"] = canonical_name
                                    
                global_chars = hierarchy_data["metadata"].get("global_characters", [])
                if original_name in global_chars:
                    global_chars.remove(original_name)
                if canonical_name not in global_chars:
                    global_chars.append(canonical_name)
                    
            # Save updated hierarchy cache
            os.makedirs(cache_dir, exist_ok=True)
            with open(hierarchy_cache, "w", encoding="utf-8") as f:
                json.dump(hierarchy_data, f, indent=4)
                
            # Re-profile to keep metrics in sync
            profiler = ManuscriptProfiler(use_gpu=False, production_tier=int(tier))
            profile_data = profiler.profile_book(filepath, hierarchy_data=hierarchy_data)
            profiler.save_profile(profile_data, profile_cache)
            
            response_data = {
                "profile": profile_data,
                "hierarchy": hierarchy_data
            }
            
            response = json.dumps(response_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response)
            
        except Exception as e:
            logger.error(f"Error handling confirm merge: {e}", exc_info=True)
            self.send_json_error(500, f"Confirmation failed: {str(e)}")

    def handle_get_cast(self):
        try:
            from src.spatial_memory import MemPalace
            palace = MemPalace()
            cursor = palace.conn.cursor()
            cursor.execute("SELECT character_name, voice_ref_path, modulation_config_json FROM drawers;")
            rows = cursor.fetchall()
            palace.close()
            
            cast = []
            for row in rows:
                try:
                    config = json.loads(row[2])
                except Exception:
                    config = {}
                cast.append({
                    "name": row[0],
                    "voice_ref_path": row[1],
                    "speed": config.get("speed", 1.0),
                    "pitch": config.get("pitch", 0.0)
                })
                
            response = json.dumps({"cast": cast}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response)
        except Exception as e:
            logger.error(f"Error listing cast: {e}")
            self.send_json_error(500, str(e))

    def handle_get_pipeline_status(self):
        global PIPELINE_STATUS
        response = json.dumps(PIPELINE_STATUS).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response)

    def handle_post_update_character(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            filename = params.get("filename")
            original_name = params.get("original_name")
            new_name = params.get("new_name")
            voice_ref_path = params.get("voice_ref_path")
            speed = float(params.get("speed", 1.0))
            pitch = float(params.get("pitch", 0.0))
            
            if not filename or not original_name or not new_name or not voice_ref_path:
                self.send_json_error(400, "Missing required parameters: filename, original_name, new_name, voice_ref_path")
                return
                
            filepath = resolve_manuscript_path(filename)
            if not os.path.exists(filepath):
                self.send_json_error(404, f"Book not found: {filename}")
                return
                
            # 1. Connect to MemPalace database
            from src.spatial_memory import MemPalace
            palace = MemPalace()
            
            # 2. Renaming Safeguard: If name changes, save confirmed merge and update drawers
            if new_name != original_name:
                logger.info(f"Database rename override: '{original_name}' -> '{new_name}'")
                
                # Update cascading tables first to avoid RESTRICT integrity errors
                cursor = palace.conn.cursor()
                cursor.execute("PRAGMA foreign_keys = OFF;")
                
                # Update room table (scene/dialogue lines)
                cursor.execute("UPDATE rooms SET character_name = ? WHERE character_name = ?;", (new_name, original_name))
                # Update emotional references table
                cursor.execute("UPDATE emotional_references SET character_name = ? WHERE character_name = ?;", (new_name, original_name))
                # Update existing merges pointing to the old canonical name
                cursor.execute("UPDATE confirmed_merges SET canonical_name = ? WHERE canonical_name = ?;", (new_name, original_name))
                
                # Add confirmed merge to SQLite mapping so parser always maps original to new
                palace.save_confirmed_merge(filename, original_name, new_name, 1, 1.0)
                
                # Delete old drawer to prevent duplicate entries
                cursor.execute("DELETE FROM drawers WHERE character_name = ?;", (original_name,))
                palace.conn.commit()
                cursor.execute("PRAGMA foreign_keys = ON;")
                
            # 3. Register/Upsert new drawer profile details
            palace.register_character(
                character_name=new_name,
                voice_ref_path=voice_ref_path,
                speed=speed,
                pitch=pitch
            )
            palace.close()
            
            # 4. Modify existing hierarchy cache dynamically to preserve structural scene splits
            pass  # re is imported at module level
            base_name = os.path.splitext(filename)[0]
            slug = re.sub(r'[^a-zA-Z0-9_\-]', '', base_name)
            tier = params.get("tier", 1)
            
            cache_dir = f"data/processed/{slug}/Tier_{tier}"
            hierarchy_cache = os.path.join(cache_dir, "hierarchy.json")
            profile_cache = os.path.join(cache_dir, "profile.json")
            
            # Load existing cache hierarchy
            if os.path.exists(hierarchy_cache):
                with open(hierarchy_cache, "r", encoding="utf-8") as f:
                    hierarchy_data = json.load(f)
            else:
                hierarchy_data = build_gui_hierarchy(filepath, int(tier))
                
            # If name changes, apply rename override in the hierarchy cache
            if new_name != original_name:
                for part in hierarchy_data.get("parts", []):
                    for chapter in part.get("chapters", []):
                        for scene in chapter.get("scenes", []):
                            for line in scene.get("lines", []):
                                if line.get("character") == original_name:
                                    line["character"] = new_name
                                    
                global_chars = hierarchy_data["metadata"].get("global_characters", [])
                if original_name in global_chars:
                    global_chars.remove(original_name)
                if new_name not in global_chars:
                    global_chars.append(new_name)

            # Apply updated voice drawer speed/pitch modifications to all lines for this character
            pitch_semitones = float(pitch)
            pitch_mult = 2.0 ** (pitch_semitones / 12.0)
            for part in hierarchy_data.get("parts", []):
                for chapter in part.get("chapters", []):
                    for scene in chapter.get("scenes", []):
                        for line in scene.get("lines", []):
                            if line.get("character") == new_name:
                                if "performance" not in line or not isinstance(line["performance"], dict):
                                    line["performance"] = {}
                                line["performance"]["pitch_modifier"] = pitch_mult
                                line["performance"]["speed_modifier"] = float(speed)
                    
            # Save updated hierarchy cache
            os.makedirs(cache_dir, exist_ok=True)
            with open(hierarchy_cache, "w", encoding="utf-8") as f:
                json.dump(hierarchy_data, f, indent=4)
                
            # Re-profile to keep metrics in sync
            profiler = ManuscriptProfiler(use_gpu=False, production_tier=int(tier))
            profile_data = profiler.profile_book(filepath, hierarchy_data=hierarchy_data)
            profiler.save_profile(profile_data, profile_cache)
            
            response_data = {
                "profile": profile_data,
                "hierarchy": hierarchy_data
            }
            
            response = json.dumps(response_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response)
        except Exception as e:
            logger.error(f"Error updating character: {e}", exc_info=True)
            self.send_json_error(500, str(e))

    def handle_post_process_manuscript(self):
        global PIPELINE_STATUS
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            filename = params.get("filename")
            if not filename:
                self.send_json_error(400, "Missing filename parameter")
                return
                
            if PIPELINE_STATUS["status"] == "running":
                self.send_json_error(400, "Audio synthesis pipeline is already running in background.")
                return
                
            tier = params.get("tier", 1)
            user_tier = params.get("user_tier", "free")
            # Trigger pipeline on a background thread
            thread = threading.Thread(target=bg_run_pipeline, args=(filename, int(tier), user_tier))
            thread.start()
            
            response = json.dumps({"status": "running"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response)
        except Exception as e:
            logger.error(f"Error starting pipeline: {e}")
            self.send_json_error(500, str(e))

    def handle_post_override_line_speaker(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            filename = params.get("filename")
            tier = params.get("tier", 1)
            line_id = params.get("line_id")
            new_speaker = params.get("new_speaker")
            
            if not filename or not line_id or not new_speaker:
                self.send_json_error(400, "Missing required parameters: filename, line_id, new_speaker")
                return
                
            # Update in SQLite Palace
            from src.spatial_memory import MemPalace
            palace = MemPalace()
            cursor = palace.conn.cursor()
            
            # Since character_name is a restrict foreign key pointing to drawers table, 
            # make sure new_speaker drawer is registered, if not register as default
            cursor.execute("SELECT character_name FROM drawers WHERE character_name = ?;", (new_speaker,))
            if not cursor.fetchone():
                palace.register_character(
                    character_name=new_speaker,
                    voice_ref_path="data/voice_references/narrator_mono.wav",
                    speed=1.0,
                    pitch=0.0
                )
                
            cursor.execute("UPDATE rooms SET character_name = ? WHERE room_id = ?;", (new_speaker, line_id))
            palace.conn.commit()
            palace.close()
            
            logger.info(f"Line speaker override successful: Line [{line_id}] -> [{new_speaker}]")
            
            # Direct cache update to prevent full re-parsing overlap
            pass  # re is imported at module level
            base_name = os.path.splitext(filename)[0]
            slug = re.sub(r'[^a-zA-Z0-9_\-]', '', base_name)
            
            cache_dir = f"data/processed/{slug}/Tier_{tier}"
            hierarchy_cache = os.path.join(cache_dir, "hierarchy.json")
            profile_cache = os.path.join(cache_dir, "profile.json")
            
            if not os.path.exists(hierarchy_cache):
                self.send_json_error(404, f"Hierarchy cache not found for {filename} Tier {tier}")
                return
                
            with open(hierarchy_cache, "r", encoding="utf-8") as f:
                hierarchy_data = json.load(f)
                
            # Update line in cache
            target_line = None
            found = False
            for part in hierarchy_data.get("parts", []):
                for chapter in part.get("chapters", []):
                    for scene in chapter.get("scenes", []):
                        for line in scene.get("lines", []):
                            if line.get("line_id") == line_id:
                                line["character"] = new_speaker
                                line["speaker_locked"] = True
                                
                                # Update performance modifiers to match the new speaker drawer defaults
                                palace_temp = MemPalace()
                                drawer = palace_temp.get_character_drawer(new_speaker)
                                palace_temp.close()
                                if drawer:
                                    pitch_semitones = float(drawer["modulation_config"].get("pitch", 0.0))
                                    pitch_mult = 2.0 ** (pitch_semitones / 12.0)
                                    speed_val = float(drawer["modulation_config"].get("speed", 1.0))
                                    if "performance" not in line or not isinstance(line["performance"], dict):
                                        line["performance"] = {}
                                    line["performance"]["pitch_modifier"] = pitch_mult
                                    line["performance"]["speed_modifier"] = speed_val
                                    
                                target_line = line
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
                    
            if not found:
                self.send_json_error(404, f"Line ID {line_id} not found in hierarchy")
                return
                
            # Add to global characters roster if not already present
            if new_speaker not in hierarchy_data["metadata"]["global_characters"]:
                hierarchy_data["metadata"]["global_characters"].append(new_speaker)
                
            # Save updated cache
            with open(hierarchy_cache, "w", encoding="utf-8") as f:
                json.dump(hierarchy_data, f, indent=4)
                
            # If the line was verified, update centralized training dataset too!
            if target_line.get("verified"):
                feedback_file = "data/feedback_dataset.json"
                feedback_data = []
                if os.path.exists(feedback_file):
                    try:
                        with open(feedback_file, "r", encoding="utf-8") as f:
                            feedback_data = json.load(f)
                    except Exception:
                        pass
                for item in feedback_data:
                    if item.get("line_id") == line_id:
                        item["character"] = new_speaker
                        break
                with open(feedback_file, "w", encoding="utf-8") as f:
                    json.dump(feedback_data, f, indent=4)
                    
            # Re-profile with ManuscriptProfiler using the updated hierarchy structure to keep metrics accurate
            profiler = ManuscriptProfiler(use_gpu=False)
            filepath = resolve_manuscript_path(filename)
            if not filepath:
                self.send_json_error(404, f"Book not found: {filename}")
                return
            profile_data = profiler.profile_book(filepath, hierarchy_data=hierarchy_data)
            profiler.save_profile(profile_data, profile_cache)
            
            response_data = {
                "profile": profile_data,
                "hierarchy": hierarchy_data
            }
            
            response = json.dumps(response_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response)
        except Exception as e:
            logger.error(f"Error overriding line speaker: {e}", exc_info=True)
            self.send_json_error(500, str(e))

    def handle_post_verify_line(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            filename = params.get("filename")
            tier = params.get("tier", 1)
            line_id = params.get("line_id")
            verified = params.get("verified", False)
            
            if not filename or not line_id:
                self.send_json_error(400, "Missing required parameters: filename, line_id")
                return
                
            pass  # re is imported at module level
            base_name = os.path.splitext(filename)[0]
            slug = re.sub(r'[^a-zA-Z0-9_\-]', '', base_name)
            
            cache_dir = f"data/processed/{slug}/Tier_{tier}"
            hierarchy_cache = os.path.join(cache_dir, "hierarchy.json")
            
            if not os.path.exists(hierarchy_cache):
                self.send_json_error(404, f"Hierarchy cache not found for {filename} Tier {tier}")
                return
                
            # Load cache
            with open(hierarchy_cache, "r", encoding="utf-8") as f:
                hierarchy_data = json.load(f)
                
            # Find and update line
            target_line = None
            found = False
            for part in hierarchy_data.get("parts", []):
                for chapter in part.get("chapters", []):
                    for scene in chapter.get("scenes", []):
                        for line in scene.get("lines", []):
                            if line.get("line_id") == line_id:
                                line["verified"] = verified
                                line["speaker_locked"] = verified # Lock speaker if verified
                                target_line = line
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
                    
            if not found:
                self.send_json_error(404, f"Line ID {line_id} not found in hierarchy")
                return
                
            # Save updated cache
            with open(hierarchy_cache, "w", encoding="utf-8") as f:
                json.dump(hierarchy_data, f, indent=4)
                
            # Central training feedback file update
            feedback_file = "data/feedback_dataset.json"
            feedback_data = []
            if os.path.exists(feedback_file):
                try:
                    with open(feedback_file, "r", encoding="utf-8") as f:
                        feedback_data = json.load(f)
                except Exception:
                    feedback_data = []
                    
            # Check if line already exists in feedback dataset
            existing_idx = -1
            for idx, item in enumerate(feedback_data):
                if item.get("line_id") == line_id:
                    existing_idx = idx
                    break
                    
            if verified:
                payload = {
                    "line_id": line_id,
                    "filename": filename,
                    "tier": tier,
                    "character": target_line.get("character"),
                    "text": target_line.get("text"),
                    "dialogue": target_line.get("dialogue"),
                    "narration_before": target_line.get("narration_before", ""),
                    "narration_after": target_line.get("narration_after", ""),
                    "emotion": target_line.get("emotion"),
                    "attribution_method": target_line.get("attribution_method")
                }
                if existing_idx >= 0:
                    feedback_data[existing_idx] = payload
                else:
                    feedback_data.append(payload)
            else:
                if existing_idx >= 0:
                    feedback_data.pop(existing_idx)
                    
            # Save centralized feedback file
            os.makedirs("data", exist_ok=True)
            with open(feedback_file, "w", encoding="utf-8") as f:
                json.dump(feedback_data, f, indent=4)
                
            # Return updated hierarchy
            response_data = {
                "hierarchy": hierarchy_data
            }
            response = json.dumps(response_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response)
            
        except Exception as e:
            logger.error(f"Error verifying line speaker: {e}", exc_info=True)
            self.send_json_error(500, str(e))

    def handle_post_verify_aspect(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            filename = params.get("filename")
            tier = params.get("tier", 1)
            aspect = params.get("aspect")
            verified = params.get("verified", False)
            
            if not filename or not aspect:
                self.send_json_error(400, "Missing required parameters: filename, aspect")
                return
                
            pass  # re is imported at module level
            base_name = os.path.splitext(filename)[0]
            slug = re.sub(r'[^a-zA-Z0-9_\-]', '', base_name)
            
            cache_dir = f"data/processed/{slug}/Tier_{tier}"
            hierarchy_cache = os.path.join(cache_dir, "hierarchy.json")
            
            if not os.path.exists(hierarchy_cache):
                self.send_json_error(404, f"Hierarchy cache not found for {filename} Tier {tier}")
                return
                
            # Load cache
            with open(hierarchy_cache, "r", encoding="utf-8") as f:
                hierarchy_data = json.load(f)
                
            if "verified_aspects" not in hierarchy_data["metadata"]:
                hierarchy_data["metadata"]["verified_aspects"] = {}
                
            hierarchy_data["metadata"]["verified_aspects"][aspect] = verified
            
            # Save updated cache
            with open(hierarchy_cache, "w", encoding="utf-8") as f:
                json.dump(hierarchy_data, f, indent=4)
                
            # Central training feedback file update
            feedback_file = "data/feedback_dataset.json"
            feedback_data = []
            if os.path.exists(feedback_file):
                try:
                    with open(feedback_file, "r", encoding="utf-8") as f:
                        feedback_data = json.load(f)
                except Exception:
                    feedback_data = []
                    
            # Check if this aspect verification already exists in feedback dataset
            existing_idx = -1
            for idx, item in enumerate(feedback_data):
                if item.get("type") == "aspect_verification" and item.get("filename") == filename and item.get("tier") == tier and item.get("aspect") == aspect:
                    existing_idx = idx
                    break
                    
            if verified:
                # Capture structural parameters for training
                aspect_details = {
                    "type": "aspect_verification",
                    "filename": filename,
                    "tier": tier,
                    "aspect": aspect,
                    "verified": True,
                    "metadata": {
                        "total_chapters": hierarchy_data["metadata"].get("total_chapters"),
                        "total_scenes": hierarchy_data["metadata"].get("total_scenes"),
                        "global_characters": hierarchy_data["metadata"].get("global_characters")
                    }
                }
                
                # Add context details based on the aspect being verified
                if aspect == "scene_splitting":
                    scenes_structure = []
                    for part in hierarchy_data.get("parts", []):
                        for chapter in part.get("chapters", []):
                            for scene in chapter.get("scenes", []):
                                scenes_structure.append({
                                    "scene_id": scene.get("scene_id"),
                                    "scene_number": scene.get("scene_number"),
                                    "first_line": scene.get("lines")[0].get("text") if scene.get("lines") else ""
                                })
                    aspect_details["scenes_structure"] = scenes_structure
                elif aspect == "chapter_splitting":
                    chapters_structure = []
                    for part in hierarchy_data.get("parts", []):
                        for chapter in part.get("chapters", []):
                            chapters_structure.append({
                                "chapter_id": chapter.get("chapter_id"),
                                "chapter_title": chapter.get("chapter_title"),
                                "total_scenes": chapter.get("total_scenes")
                            })
                    aspect_details["chapters_structure"] = chapters_structure
                elif aspect == "character_classification":
                    aspect_details["characters"] = hierarchy_data["metadata"].get("global_characters", [])
                    
                if existing_idx >= 0:
                    feedback_data[existing_idx] = aspect_details
                else:
                    feedback_data.append(aspect_details)
            else:
                if existing_idx >= 0:
                    feedback_data.pop(existing_idx)
                    
            # Save centralized feedback file
            with open(feedback_file, "w", encoding="utf-8") as f:
                json.dump(feedback_data, f, indent=4)
                
            response_data = {
                "hierarchy": hierarchy_data
            }
            response = json.dumps(response_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response)
            
        except Exception as e:
            logger.error(f"Error verifying aspect: {e}", exc_info=True)
            self.send_json_error(500, str(e))

    def handle_post_save_hierarchy(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            filename = params.get("filename")
            tier = params.get("tier", 1)
            hierarchy_data = params.get("hierarchy")
            
            if not filename or not hierarchy_data:
                self.send_json_error(400, "Missing required parameters: filename, hierarchy")
                return
                
            pass  # re is imported at module level
            base_name = os.path.splitext(filename)[0]
            slug = re.sub(r'[^a-zA-Z0-9_\-]', '', base_name)
            
            cache_dir = f"data/processed/{slug}/Tier_{tier}"
            hierarchy_cache = os.path.join(cache_dir, "hierarchy.json")
            profile_cache = os.path.join(cache_dir, "profile.json")
            
            os.makedirs(cache_dir, exist_ok=True)
            with open(hierarchy_cache, "w", encoding="utf-8") as f:
                json.dump(hierarchy_data, f, indent=4)
                
            # Re-profile to keep metrics in sync
            profiler = ManuscriptProfiler(use_gpu=False, production_tier=int(tier))
            filepath = resolve_manuscript_path(filename)
            if not filepath:
                self.send_json_error(404, f"Book not found: {filename}")
                return
            profile_data = profiler.profile_book(filepath, hierarchy_data=hierarchy_data)
            profiler.save_profile(profile_data, profile_cache)
            
            # Sync verified lines with feedback dataset
            feedback_file = "data/feedback_dataset.json"
            feedback_data = []
            if os.path.exists(feedback_file):
                try:
                    with open(feedback_file, "r", encoding="utf-8") as f:
                        feedback_data = json.load(f)
                except Exception:
                    feedback_data = []
            
            # Keep non-line items or items from other books/tiers intact
            other_feedback_items = []
            for item in feedback_data:
                # Keep if it is a different book or tier, or not a line feedback
                if "line_id" not in item or item.get("filename") != filename or item.get("tier") != tier:
                    other_feedback_items.append(item)
            
            # Scan current hierarchy for verified lines
            new_feedback_lines = []
            for part in hierarchy_data.get("parts", []):
                for chapter in part.get("chapters", []):
                    for scene in chapter.get("scenes", []):
                        for line in scene.get("lines", []):
                            if line.get("verified"):
                                line_id = line.get("line_id")
                                payload = {
                                    "line_id": line_id,
                                    "filename": filename,
                                    "tier": tier,
                                    "character": line.get("character"),
                                    "text": line.get("text"),
                                    "dialogue": line.get("dialogue"),
                                    "narration_before": line.get("narration_before", ""),
                                    "narration_after": line.get("narration_after", ""),
                                    "emotion": line.get("emotion"),
                                    "attribution_method": line.get("attribution_method")
                                }
                                new_feedback_lines.append(payload)
            
            final_feedback_data = other_feedback_items + new_feedback_lines
            with open(feedback_file, "w", encoding="utf-8") as f:
                json.dump(final_feedback_data, f, indent=4)
                
            response = json.dumps({"status": "success", "profile": profile_data}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response)
        except Exception as e:
            logger.error(f"Error saving hierarchy: {e}", exc_info=True)
            self.send_json_error(500, str(e))

    def handle_post_telemetry_correction(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            os.makedirs("data", exist_ok=True)
            telemetry_file = "data/telemetry_corrections.json"
            
            corrections = []
            if os.path.exists(telemetry_file):
                try:
                    with open(telemetry_file, "r", encoding="utf-8") as f:
                        corrections = json.load(f)
                        if not isinstance(corrections, list):
                            corrections = []
                except Exception:
                    corrections = []
            
            import datetime
            record = {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "filename": params.get("filename"),
                "correction_type": params.get("correction_type"),
                "line_id": params.get("line_id"),
                "original_data": params.get("original_data"),
                "corrected_data": params.get("corrected_data")
            }
            
            corrections.append(record)
            
            with open(telemetry_file, "w", encoding="utf-8") as f:
                json.dump(corrections, f, indent=4)
                
            response = json.dumps({"status": "success"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response)
        except Exception as e:
            logger.error(f"Error saving telemetry correction: {e}", exc_info=True)
            self.send_json_error(500, str(e))

    def handle_post_process_scenes_async(self):
        """
        Webserver endpoint trigger for confirmed batch processing.
        Executes scene analysis for confirmed scenes only, using the async engine.
        """
        import asyncio
        import hashlib
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode('utf-8'))
            
            filename = params.get("filename")
            tier = params.get("tier", 1)
            scenes = params.get("scenes") # Expect a list of dicts: [{"scene_id": "...", "text_block": "..."}]
            global_roster = params.get("global_roster", [])
            backend = params.get("backend", "vllm")
            base_url = params.get("base_url")
            
            if not filename or not scenes:
                self.send_json_error(400, "Missing required parameters: filename, scenes")
                return
                
            if not base_url:
                if backend == "vllm":
                    base_url = "http://localhost:8000"
                elif backend == "llamacpp":
                    base_url = "http://localhost:8080"
                else:
                    base_url = "http://localhost:11434"
                    
            pass  # re is imported at module level
            base_name = os.path.splitext(filename)[0]
            slug = re.sub(r'[^a-zA-Z0-9_\-]', '', base_name)
            
            cache_dir = f"data/processed/{slug}/Tier_{tier}"
            hierarchy_cache = os.path.join(cache_dir, "hierarchy.json")
            
            if not os.path.exists(hierarchy_cache):
                self.send_json_error(404, f"Hierarchy cache not found for {filename} Tier {tier}")
                return
                
            with open(hierarchy_cache, "r", encoding="utf-8") as f:
                hierarchy_data = json.load(f)
                
            # Initialize MemPalace to query rules
            from src.spatial_memory import MemPalace
            palace = MemPalace()
            active_rules = palace.fetch_active_rag_context_rules(filename)
            
            # Setup Async Inference Engine
            from src.async_inference import AsyncInferenceEngine, batch_process_scenes
            engine = AsyncInferenceEngine(backend=backend, base_url=base_url)
            
            # Run batch processing async
            results = asyncio.run(batch_process_scenes(
                scenes=scenes,
                characters=global_roster,
                engine=engine,
                rules=active_rules
            ))
            
            # Close engine client
            asyncio.run(engine.close())
            
            # Helper to map performance mods based on emotion
            def map_performance_mods(emotion: str, text: str) -> dict:
                pitch = 1.0
                speed = 1.0
                style = "neutral_narrative"
                
                emotion_lower = emotion.lower()
                if emotion_lower in {"sadness", "grief", "disappointment", "sad"}:
                    pitch = 0.90
                    speed = 0.85
                    style = "sorrowful_whisper"
                elif emotion_lower in {"fear", "nervousness", "panic", "tension"}:
                    pitch = 1.15
                    speed = 1.10
                    style = "anxious_whisper"
                elif emotion_lower in {"anger", "annoyance", "disapproval"}:
                    pitch = 0.95
                    speed = 1.05
                    style = "furious_shout" if "!" in text else "stern_authoritative"
                elif emotion_lower in {"joy", "excitement", "amusement", "love"}:
                    pitch = 1.05
                    speed = 1.02
                    style = "expressive_joy"
                    
                return {
                    "pitch_modifier": pitch,
                    "speed_modifier": speed,
                    "delivery_style": style
                }
                
            from src.models import ScriptLine, PerformanceMetrics
            cursor = palace.conn.cursor()
            
            processed_scenes_report = []
            
            for res in results:
                scene_id = res["scene_id"]
                if res["status"] == "success":
                    lines_data = res["data"].get("lines", [])
                    
                    # 1. Find scene in hierarchy to get chapter and scene numbers
                    chapter_num = 1
                    scene_num = 1
                    found_scene = False
                    for part in hierarchy_data.get("parts", []):
                        for chapter in part.get("chapters", []):
                            for s_idx, scene in enumerate(chapter.get("scenes", [])):
                                if scene.get("scene_id") == scene_id:
                                    chap_match = re.search(r'_c(\d+)', chapter.get("chapter_id", ""))
                                    if chap_match:
                                        chapter_num = int(chap_match.group(1))
                                    scene_match = re.search(r'_s(\d+)', scene_id)
                                    if scene_match:
                                        scene_num = int(scene_match.group(1))
                                    found_scene = True
                                    break
                            if found_scene:
                                break
                        if found_scene:
                            break
                            
                    # Register Chapter/Wing in SQLite relational tables
                    wing_id = f"wing_c{chapter_num}"
                    palace.log_wing(
                        wing_id=wing_id,
                        chapter_number=chapter_num,
                        title=f"Chapter {chapter_num}"
                    )
                    
                    # Validate and map LLM response lines using Pydantic
                    validated_lines = []
                    for idx, ld in enumerate(lines_data, 1):
                        char_name = ld.get("character", "Narrator").strip()
                        if char_name.lower() == "narrator":
                            char_name = "Narrator"
                            
                        raw_id = f"{slug}_c{chapter_num}_s{scene_num}_l{idx}_{ld.get('text', '')[:20]}"
                        line_id = hashlib.sha256(raw_id.encode('utf-8')).hexdigest()[:16]
                        
                        speaker_id = f"char_{char_name.lower().replace(' ', '_')}"
                        if char_name == "Narrator":
                            speaker_id = "char_narrator"
                            
                        perf = map_performance_mods(ld.get("emotion", "Neutral"), ld.get("text", ""))
                        
                        script_line = ScriptLine(
                            line_id=line_id,
                            chapter=chapter_num,
                            scene=scene_num,
                            line_number=idx,
                            character=char_name,
                            speaker_id=speaker_id,
                            segment_type=ld.get("segment_type", "narrative"),
                            text=ld.get("text", ""),
                            emotion=ld.get("emotion", "Neutral").title(),
                            performance=PerformanceMetrics(
                                pitch_modifier=perf["pitch_modifier"],
                                speed_modifier=perf["speed_modifier"],
                                delivery_style=perf["delivery_style"]
                            ),
                            post_padding_ms=250,
                            attribution_method="LLM Batch Parser",
                            confidence=float(ld.get("confidence", 0.90)),
                            speaker_locked=False
                        )
                        
                        validated_lines.append(script_line.model_dump())
                        
                        # Sync line to Relational DB
                        cursor.execute("SELECT character_name FROM drawers WHERE character_name = ?;", (char_name,))
                        if not cursor.fetchone():
                            palace.register_character(
                                character_name=char_name,
                                voice_ref_path="data/voice_references/narrator_mono.wav"
                            )
                        
                        palace.log_room(
                            room_id=line_id,
                            wing_id=wing_id,
                            line_number=idx,
                            character_name=char_name,
                            dialogue_text=ld.get("text", ""),
                            emotion=ld.get("emotion", "Neutral").title(),
                            confidence=float(ld.get("confidence", 0.90)),
                            metadata={
                                "performance": perf,
                                "attribution_method": "LLM Batch Parser"
                            }
                        )
                        
                    # 2. Update local cache hierarchy with validated lines
                    found_and_updated = False
                    for part in hierarchy_data.get("parts", []):
                        for chapter in part.get("chapters", []):
                            for scene in chapter.get("scenes", []):
                                if scene.get("scene_id") == scene_id:
                                    scene["lines"] = validated_lines
                                    found_and_updated = True
                                    break
                            if found_and_updated:
                                break
                        if found_and_updated:
                            break
                            
                    processed_scenes_report.append({
                        "scene_id": scene_id,
                        "status": "success",
                        "lines_count": len(validated_lines)
                    })
                else:
                    processed_scenes_report.append({
                        "scene_id": scene_id,
                        "status": "failed",
                        "error": res.get("error", "Unknown error occurred during batch generation.")
                    })
                    
            palace.close()
            
            # Save updated cache hierarchy
            with open(hierarchy_cache, "w", encoding="utf-8") as f:
                json.dump(hierarchy_data, f, indent=4)
                
            response = json.dumps({
                "status": "completed",
                "results": processed_scenes_report
            }).encode("utf-8")
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response)
            
        except Exception as e:
            logger.error(f"Error processing scenes async: {e}", exc_info=True)
            self.send_json_error(500, str(e))

    def send_json_error(self, code, message):
        response = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response)


def main():
    port = 8082
    server_address = ('', port)
    # Boot sequence: self-heal a wedged GPU driver (re-execs if needed), then
    # run the environment doctor so a broken runtime is visible at startup,
    # not an hour into the first render.
    from src import boot_check
    boot_check.ensure_torch_safe()
    boot_check.print_report(boot_check.run_boot_checks())

    # Threading: a slow request (XTTS clone preview takes tens of seconds on CPU)
    # must not block the progress polls and page loads of every other client.
    httpd = ThreadingHTTPServer(server_address, StudioRequestHandler)
    logger.info(f"Volcano Studios GUI Server launched successfully at http://localhost:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    main()
