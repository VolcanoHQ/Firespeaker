#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Firespeaker Studio - Local GUI Server
A zero-dependency standard Python HTTP API server hosting the workspace.
"""

import os
import sys
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse

# Ensure the root project directory is in the sys.path for absolute modular imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.hierarchical_parser import HierarchicalParser
from src.manuscript_profiler import ManuscriptProfiler

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


# Global non-blocking pipeline compilation state
PIPELINE_STATUS = {
    "status": "idle",       # "idle", "running", "completed", "failed"
    "progress": 0,          # 0 to 100
    "step": "",             # active step description
    "error": None,
    "qc_report": None
}

def bg_run_pipeline(filename: str):
    """
    Executes the FirespeakerPipeline full run on a background thread.
    Updates the global PIPELINE_STATUS object for real-time progress polling.
    """
    global PIPELINE_STATUS
    try:
        from src.main import FirespeakerPipeline
        filepath = os.path.join("data/corpus", filename)
        
        logger.info(f"[BG Compiler] Initializing pipeline run for {filename}...")
        PIPELINE_STATUS["status"] = "running"
        PIPELINE_STATUS["step"] = "Preparing workspace & loading character drawers..."
        PIPELINE_STATUS["progress"] = 15
        
        pipeline = FirespeakerPipeline()
        
        logger.info("[BG Compiler] Ingesting and parsing script lines...")
        PIPELINE_STATUS["step"] = "Running manuscript text parsing & LLM dialogue attribution..."
        PIPELINE_STATUS["progress"] = 40
        
        logger.info("[BG Compiler] Initiating speech synthesis and track mixing...")
        PIPELINE_STATUS["step"] = "Synthesizing voice lines and compiling ambient sidechain filters..."
        PIPELINE_STATUS["progress"] = 75
        
        output_master = "scratch/pipeline_workspace/output_master.wav"
        success = pipeline.run_full_pipeline(filepath, output_master)
        
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

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        if path == "/" or path == "/index.html":
            self.serve_static_file("src/static/index.html", "text/html")
        elif path == "/api/books":
            self.handle_get_books()
        elif path == "/api/cast":
            self.handle_get_cast()
        elif path == "/api/pipeline_status":
            self.handle_get_pipeline_status()
        elif path == "/api/logs":
            self.handle_get_logs()
        else:
            self.send_error(404, "File not found")
            
    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        if path == "/api/analyze":
            self.handle_post_analyze()
        elif path == "/api/upload":
            self.handle_post_upload()
        elif path == "/api/confirm_merge":
            self.handle_post_confirm_merge()
        elif path == "/api/update_character":
            self.handle_post_update_character()
        elif path == "/api/process_manuscript":
            self.handle_post_process_manuscript()
        elif path == "/api/override_line_speaker":
            self.handle_post_override_line_speaker()
        elif path == "/api/verify_line":
            self.handle_post_verify_line()
        elif path == "/api/verify_aspect":
            self.handle_post_verify_aspect()
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
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            logger.error(f"Error serving static file: {e}")
            self.send_error(500, f"Server error: {e}")

    def handle_get_books(self):
        try:
            corpus_dir = "data/corpus"
            if not os.path.exists(corpus_dir):
                os.makedirs(corpus_dir, exist_ok=True)
            files = [f for f in os.listdir(corpus_dir) if f.endswith(".txt")]
            files.sort()
            
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
                
            filepath = os.path.join("data/corpus", filename)
            if not os.path.exists(filepath):
                self.send_json_error(404, f"Book not found in corpus: {filename}")
                return
                
            # Sanitize book name slug
            import re
            base_name = os.path.splitext(filename)[0]
            slug = re.sub(r'[^a-zA-Z0-9_\-]', '', base_name)
            
            # Tier-specific cache paths
            cache_dir = f"data/processed/{slug}/Tier_{tier}"
            hierarchy_cache = os.path.join(cache_dir, "hierarchy.json")
            profile_cache = os.path.join(cache_dir, "profile.json")
            
            # Check cached files
            if os.path.exists(hierarchy_cache) and os.path.exists(profile_cache):
                logger.info(f"Retrieving cached profiling metrics for: {filename} (Tier {tier})")
                with open(hierarchy_cache, "r", encoding="utf-8") as f:
                    hierarchy_data = json.load(f)
                with open(profile_cache, "r", encoding="utf-8") as f:
                    profile_data = json.load(f)
            else:
                logger.info(f"Analyzing and profiling {filename} from scratch (Tier {tier})...")
                parser = HierarchicalParser(use_gpu=False, production_tier=int(tier))
                profiler = ManuscriptProfiler(use_gpu=False)
                
                hierarchy_data = parser.parse_hierarchy(filepath)
                profile_data = profiler.profile_book(filepath, hierarchy_data=hierarchy_data)
                
                # Cache results
                os.makedirs(cache_dir, exist_ok=True)
                parser.save_index(hierarchy_data, hierarchy_cache)
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
            
            filename = params.get("filename")
            text = params.get("text")
            tier = params.get("tier", 1)
            
            if not filename or not text:
                self.send_json_error(400, "Missing filename or text parameter")
                return
                
            filename = os.path.basename(filename)
            if not filename.endswith(".txt"):
                filename += ".txt"
                
            filepath = os.path.join("data/corpus", filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
                
            logger.info(f"Custom book saved successfully to: {filepath}")
            
            # Analyze immediately
            parser = HierarchicalParser(use_gpu=False, production_tier=int(tier))
            profiler = ManuscriptProfiler(use_gpu=False)
            
            hierarchy_data = parser.parse_hierarchy(filepath)
            profile_data = profiler.profile_book(filepath, hierarchy_data=hierarchy_data)
            
            # Save cache
            import re
            base_name = os.path.splitext(filename)[0]
            slug = re.sub(r'[^a-zA-Z0-9_\-]', '', base_name)
            
            cache_dir = f"data/processed/{slug}/Tier_{tier}"
            hierarchy_cache = os.path.join(cache_dir, "hierarchy.json")
            profile_cache = os.path.join(cache_dir, "profile.json")
            os.makedirs(cache_dir, exist_ok=True)
            parser.save_index(hierarchy_data, hierarchy_cache)
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
                
            filepath = os.path.join("data/corpus", filename)
            if not os.path.exists(filepath):
                self.send_json_error(404, f"Book not found in corpus: {filename}")
                return
                
            # 1. Connect to MemPalace database and save decision
            from src.spatial_memory import MemPalace
            palace = MemPalace()
            # Default confidence score is 1.0 if confirmed, or 0.0 if split/rejected
            confidence_score = 1.0 if is_confirmed else 0.0
            palace.save_confirmed_merge(filename, original_name, canonical_name, is_confirmed, confidence_score)
            palace.close()
            
            # 2. Invalidate cache files in scratch/
            cache_base = os.path.splitext(filename)[0]
            hierarchy_cache = os.path.join("scratch", f"{cache_base}_hierarchy.json")
            profile_cache = os.path.join("scratch", f"{cache_base}_profile.json")
            
            if os.path.exists(hierarchy_cache):
                os.remove(hierarchy_cache)
            if os.path.exists(profile_cache):
                os.remove(profile_cache)
                
            logger.info(f"Cache cleared for {filename}. Re-profiling manuscript...")
            
            # 3. Re-run parsing and profiling under new database rules
            parser = HierarchicalParser(use_gpu=False)
            profiler = ManuscriptProfiler(use_gpu=False)
            
            hierarchy_data = parser.parse_hierarchy(filepath)
            profile_data = profiler.profile_book(filepath)
            
            # 4. Save new cache
            os.makedirs("scratch", exist_ok=True)
            parser.save_index(hierarchy_data, hierarchy_cache)
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
                
            filepath = os.path.join("data/corpus", filename)
            if not os.path.exists(filepath):
                self.send_json_error(404, f"Book not found in corpus: {filename}")
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
            
            # 4. Invalidate parser caches to trigger complete re-profiling pass
            cache_base = os.path.splitext(filename)[0]
            hierarchy_cache = os.path.join("scratch", f"{cache_base}_hierarchy.json")
            profile_cache = os.path.join("scratch", f"{cache_base}_profile.json")
            
            if os.path.exists(hierarchy_cache):
                os.remove(hierarchy_cache)
            if os.path.exists(profile_cache):
                os.remove(profile_cache)
                
            # 5. Re-run parsing under new drawers/merges
            parser = HierarchicalParser(use_gpu=False)
            profiler = ManuscriptProfiler(use_gpu=False)
            
            hierarchy_data = parser.parse_hierarchy(filepath)
            profile_data = profiler.profile_book(filepath)
            
            # 6. Save cache
            os.makedirs("scratch", exist_ok=True)
            parser.save_index(hierarchy_data, hierarchy_cache)
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
                
            # Trigger pipeline on a background thread
            thread = threading.Thread(target=bg_run_pipeline, args=(filename,))
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
            import re
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
            filepath = os.path.join("data/corpus", filename)
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
                
            import re
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
                
            import re
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
    httpd = HTTPServer(server_address, StudioRequestHandler)
    logger.info(f"Firespeaker Studio GUI Server launched successfully at http://localhost:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    main()
