#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Caldera Engine API Gateway
FastAPI server hosting the Studio REST endpoints, WebSocket Lookahead status alerts,
and routing final GUI override payloads to the TTS compilation pipeline.
"""

import os
import re
import uuid
import json
import logging
import asyncio
import threading
import traceback
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Ensure root is in path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.project_db import ProjectDB
from src.hierarchical_parser import HierarchicalParser
from src.nlp_analyzer import TextToScriptPipeline
from src.tts_compiler import compile_modified_json
from src.voice_marketplace import VoiceMarketplace
from src.vllm_client import VLLMClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("APIGateway")

app = FastAPI(
    title="Volcano Studios Backend Orchestration API",
    description="Tier 1 State Management & Asynchronous Lookahead Queue Gateway",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
db = ProjectDB()
marketplace = VoiceMarketplace()
vllm = VLLMClient()

# Store reference to main loop for thread-safe WebSocket broadcasts
main_loop = None

@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()
    logger.info("Caldera Engine API Gateway startup complete.")

# ----------------------------------------------------
# WebSocket Connection Manager
# ----------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, project_id: str, websocket: WebSocket):
        await websocket.accept()
        if project_id not in self.active_connections:
            self.active_connections[project_id] = []
        self.active_connections[project_id].append(websocket)
        logger.info(f"WebSocket client connected to project: {project_id}")

    def disconnect(self, project_id: str, websocket: WebSocket):
        if project_id in self.active_connections:
            if websocket in self.active_connections[project_id]:
                self.active_connections[project_id].remove(websocket)
            if not self.active_connections[project_id]:
                del self.active_connections[project_id]
        logger.info(f"WebSocket client disconnected from project: {project_id}")

    async def broadcast_status(self, project_id: str, status_payload: Dict[str, Any]):
        if project_id in self.active_connections:
            logger.info(f"Broadcasting status update to {len(self.active_connections[project_id])} clients for project: {project_id}")
            for connection in list(self.active_connections[project_id]):
                try:
                    await connection.send_json(status_payload)
                except Exception as e:
                    logger.warning(f"Failed to send WS status message, disconnecting client: {e}")
                    self.disconnect(project_id, connection)

manager = ConnectionManager()

def broadcast_from_thread(project_id: str, status_payload: Dict[str, Any]):
    """Thread-safe WebSocket broadcaster for background thread workers."""
    if main_loop:
        asyncio.run_coroutine_threadsafe(
            manager.broadcast_status(project_id, status_payload),
            main_loop
        )

# Helper: Fetch characters from database drawers to keep character rosters compliant
def get_global_characters() -> List[str]:
    try:
        from src.spatial_memory import MemPalace
        palace = MemPalace()
        cursor = palace.conn.cursor()
        cursor.execute("SELECT character_name FROM drawers;")
        chars = [row[0] for row in cursor.fetchall()]
        palace.close()
        if "Narrator" not in chars:
            chars.append("Narrator")
        return chars
    except Exception as e:
        logger.warning(f"Could not read characters from MemPalace drawers: {e}")
        return ["Narrator"]

# ----------------------------------------------------
# Background Lookahead Processing Workers
# ----------------------------------------------------
def process_lookahead_queue(project_id: str, chapters_to_process: List[Dict[str, Any]], filename: str):
    """
    Background worker that runs scene slicing (Loop 3) on Chapters 2, 3, etc.,
    sequentially, saving intermediate JSON results to the DB.
    """
    logger.info(f"[Lookahead Queue] Starting background processing for project: {project_id}")
    try:
        parser = HierarchicalParser(use_gpu=False, production_tier=1)
        global_characters = get_global_characters()
        
        for chapter in chapters_to_process:
            chapter_id = chapter["chapter_id"]
            text_block = chapter["text_block"]
            order_idx = chapter["order_idx"]
            
            logger.info(f"[Lookahead Queue] Processing {chapter_id} ({chapter['chapter_title']})...")
            
            # Update chapter state to processing
            db.update_chapter_status(project_id, chapter_id, "processing")
            
            # Send status update via WS
            status_payload = db.get_project_lookahead_status(project_id)
            broadcast_from_thread(project_id, status_payload)
            
            try:
                # Run Loop 3 (Scene Slicing) and Loop 4 (Line Parsing)
                scenes = parser._split_into_scenes(text_block)
                scene_payloads = []
                for s_idx, scene_text in enumerate(scenes, 1):
                    scene_id = f"{chapter_id}_s{s_idx}"
                    lines = parser.analyzer.parse_manuscript_for_segment(
                        segment_text=scene_text,
                        file_name=filename,
                        chapter_num=order_idx,
                        scene_num=s_idx,
                        characters_list=global_characters,
                        merge_map={},
                        production_tier=1
                    )
                    scene_payloads.append({
                        "scene_id": scene_id,
                        "lines": lines
                    })
                
                # Save parsed scenes and set chapter status to completed
                db.save_chapter_scenes(project_id, chapter_id, scene_payloads)
                logger.info(f"[Lookahead Queue] Completed {chapter_id}.")
                
            except Exception as e:
                logger.error(f"[Lookahead Queue] Failed to process chapter {chapter_id}: {e}\n{traceback.format_exc()}")
                db.update_chapter_status(project_id, chapter_id, "failed")
            
            # Broadcast update
            status_payload = db.get_project_lookahead_status(project_id)
            broadcast_from_thread(project_id, status_payload)
            
        # Check if all chapters are completed and update project status
        all_completed = True
        for chap in db.get_chapters(project_id):
            if chap["status"] != "completed":
                all_completed = False
                break
                
        if all_completed:
            db.update_project_status(project_id, "completed")
        else:
            db.update_project_status(project_id, "failed")
            
        status_payload = db.get_project_lookahead_status(project_id)
        broadcast_from_thread(project_id, status_payload)
        logger.info(f"[Lookahead Queue] Finished background processing for project: {project_id}")
        
    except Exception as e:
        logger.error(f"[Lookahead Queue] Fatal error in background lookahead thread: {e}\n{traceback.format_exc()}")
        db.update_project_status(project_id, "failed")
        status_payload = db.get_project_lookahead_status(project_id)
        broadcast_from_thread(project_id, status_payload)

# ----------------------------------------------------
# Pydantic Schemas
# ----------------------------------------------------
class ProjectUploadJSON(BaseModel):
    filename: str
    text: str

class PatchSceneOverride(BaseModel):
    chapter_id: str
    action: Optional[str] = None  # None (direct scenes replace), 'merge', 'split'
    scenes: Optional[List[Dict[str, Any]]] = None # For direct replace
    scene_ids: Optional[List[str]] = None # For merge action
    scene_id: Optional[str] = None # For split action
    line_id: Optional[str] = None # For split action (split at/after this line_id)

class CompileRequest(BaseModel):
    profile: str = Field(default="standard")
    script_data: Optional[Dict[str, Any]] = None # Optional final GUI JSON override
    user_tier: str = Field(default="free")

# ----------------------------------------------------
# API Routing
# ----------------------------------------------------

@app.get("/")
@app.get("/index.html")
def get_gui():
    """Serves the central Volcano Studios editor dashboard."""
    gui_file = "src/static/index.html"
    if os.path.exists(gui_file):
        return FileResponse(gui_file)
    return JSONResponse(status_code=404, content={"error": f"GUI file not found at {gui_file}"})

from fastapi import Request

@app.post("/api/v1/projects/upload")
async def upload_manuscript(
    request: Request,
    file: Optional[UploadFile] = File(None)
):
    """
    Accepts raw manuscript upload.
    Runs Loops 1 & 2 (Parts & Chapters meso-structures).
    Stores project state and returns the Macro-Structure JSON.
    """
    try:
        # Resolve inputs
        upload_filename = ""
        manuscript_text = ""
        
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            upload_filename = body.get("filename")
            manuscript_text = body.get("text")
        elif file is not None:
            upload_filename = file.filename
            content_bytes = await file.read()
            manuscript_text = content_bytes.decode("utf-8")
            
        if not upload_filename or not manuscript_text:
            raise HTTPException(status_code=400, detail="Missing manuscript parameters. Provide file upload or filename/text body.")

        if not upload_filename.endswith(".txt"):
            upload_filename += ".txt"
            
        # Save manuscript file in corpus directory
        corpus_path = os.path.join("data/corpus", upload_filename)
        os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write(manuscript_text)
            
        logger.info(f"Manuscript saved to: {corpus_path}")

        # Run Loops 1 & 2 (Meso decomposition via HierarchicalParser)
        parser = HierarchicalParser(use_gpu=False, production_tier=1)
        tp = TextToScriptPipeline()
        normalized_text = tp.normalize_typography(manuscript_text)
        
        parts_data = parser._split_into_parts(normalized_text)
        
        project_id = str(uuid.uuid4())[:8]
        db.create_project(project_id, upload_filename, "awaiting_macro_approval")
        
        macro_structure = []
        chapter_counter = 0
        
        for p_idx, (part_title, part_content) in enumerate(parts_data, 1):
            part_id = f"part_p{p_idx}"
            chapters_data = parser._split_into_chapters(part_content, p_idx)
            
            chap_list = []
            for c_idx, (chap_title, chap_content) in enumerate(chapters_data, 1):
                chapter_counter += 1
                chap_id = f"{part_id}_c{c_idx}"
                
                # Insert chapter record in pending state
                db.insert_chapter(
                    project_id=project_id,
                    chapter_id=chap_id,
                    part_id=part_id,
                    part_title=part_title,
                    chapter_title=chap_title,
                    text_block=chap_content,
                    status="pending",
                    order_idx=chapter_counter
                )
                
                chap_list.append({
                    "chapter_id": chap_id,
                    "title": chap_title,
                    "text_block_preview": chap_content[:200] + "..." if len(chap_content) > 200 else chap_content
                })
                
            macro_structure.append({
                "part_id": part_id,
                "title": part_title,
                "chapters": chap_list
            })
            
        return {
            "project_id": project_id,
            "filename": upload_filename,
            "status": "awaiting_macro_approval",
            "macro_structure": macro_structure
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")

@app.post("/api/v1/projects/{project_id}/approve_macro")
def approve_macro(project_id: str, background_tasks: BackgroundTasks):
    """
    Triggers Loop 3 (Scene Slicing) on Chapter 1 and returns it instantly.
    Dispatches lookahead threads to parse background chapters (Chapter 2, 3, etc.) asynchronously.
    """
    # 1. Fetch project details
    filename = db.get_project_filename(project_id)
    if not filename:
        raise HTTPException(status_code=404, detail="Project not found.")
        
    chapters = db.get_chapters(project_id)
    if not chapters:
        raise HTTPException(status_code=500, detail="Chapters not found for project.")
        
    # Sort chapters by order_idx
    chapters.sort(key=lambda x: x["order_idx"])
    
    # 2. Slice Chapter 1 synchronously
    chap_one = chapters[0]
    parser = HierarchicalParser(use_gpu=False, production_tier=1)
    global_characters = get_global_characters()
    
    logger.info(f"Approving macro-structure for {project_id}. Slicing Chapter 1 ({chap_one['chapter_id']}) synchronously...")
    
    try:
        db.update_chapter_status(project_id, chap_one["chapter_id"], "processing")
        
        scenes = parser._split_into_scenes(chap_one["text_block"])
        scene_payloads = []
        for s_idx, scene_text in enumerate(scenes, 1):
            scene_id = f"{chap_one['chapter_id']}_s{s_idx}"
            lines = parser.analyzer.parse_manuscript_for_segment(
                segment_text=scene_text,
                file_name=filename,
                chapter_num=chap_one["order_idx"],
                scene_num=s_idx,
                characters_list=global_characters,
                merge_map={},
                production_tier=1
            )
            scene_payloads.append({
                "scene_id": scene_id,
                "lines": lines
            })
            
        db.save_chapter_scenes(project_id, chap_one["chapter_id"], scene_payloads)
        
    except Exception as e:
        db.update_chapter_status(project_id, chap_one["chapter_id"], "failed")
        logger.error(f"Sync scene slice on Chapter 1 failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scene slicing Chapter 1 failed: {str(e)}")
        
    # Update project status to indicate active lookahead queues
    db.update_project_status(project_id, "processing_lookahead")
    
    # 3. Dispatch remaining chapters (Chapter 2, 3, etc.) to background lookahead thread
    if len(chapters) > 1:
        remaining_chapters = chapters[1:]
        # Use background_tasks or a daemon Thread
        threading.Thread(
            target=process_lookahead_queue,
            args=(project_id, remaining_chapters, filename),
            daemon=True
        ).start()
        logger.info(f"Dispatched background lookahead processor for {len(remaining_chapters)} chapters.")
    else:
        # Only one chapter existed, project completes instantly
        db.update_project_status(project_id, "completed")
        
    return {
        "project_id": project_id,
        "chapter_id": chap_one["chapter_id"],
        "title": chap_one["chapter_title"],
        "scenes": scene_payloads
    }

@app.get("/api/v1/projects/{project_id}/status")
def get_lookahead_status(project_id: str):
    """Status-polling endpoint showing lookahead queue progression."""
    status = db.get_project_lookahead_status(project_id)
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    return status

@app.get("/api/v1/projects/{project_id}/chapters/{chapter_id}")
def get_chapter_data(project_id: str, chapter_id: str):
    """Serves sliced scenes/dialogues for a specific chapter."""
    chapter = db.get_chapter(project_id, chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found.")
        
    if chapter["status"] != "completed":
        raise HTTPException(status_code=202, detail={"status": chapter["status"], "msg": "Chapter scenes are still being parsed by lookahead queue."})
        
    return {
        "project_id": project_id,
        "chapter_id": chapter["chapter_id"],
        "title": chapter["chapter_title"],
        "scenes": chapter["scenes"]
    }

@app.patch("/api/v1/projects/{project_id}/scenes")
def patch_scenes(project_id: str, payload: PatchSceneOverride):
    """
    Accepts manual merge/split overrides from the user's GUI session.
    Also handles direct complete scene overrides from GUI edits.
    """
    chapter = db.get_chapter(project_id, payload.chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found.")
        
    # Option 1: Direct full replacement of scenes list (from frontend drag-n-drop or edits)
    if payload.scenes is not None:
        db.save_chapter_scenes(project_id, payload.chapter_id, payload.scenes)
        return {"status": "success", "msg": "Chapter scenes updated directly.", "scenes": payload.scenes}
        
    # Option 2: Action operations
    action = payload.action
    if not action:
        raise HTTPException(status_code=400, detail="Missing instruction. Supply 'scenes' directly, or define 'action'.")
        
    current_scenes = chapter.get("scenes") or []
    
    if action == "merge":
        if not payload.scene_ids or len(payload.scene_ids) < 2:
            raise HTTPException(status_code=400, detail="Merge action requires 'scene_ids' list containing at least 2 IDs.")
            
        # Verify all target scenes exist
        scenes_map = {s["scene_id"]: s for s in current_scenes}
        for s_id in payload.scene_ids:
            if s_id not in scenes_map:
                raise HTTPException(status_code=404, detail=f"Scene ID {s_id} not found in chapter.")
                
        # Merge all lines into the first scene
        target_scene_id = payload.scene_ids[0]
        merged_lines = []
        for s_id in payload.scene_ids:
            merged_lines.extend(scenes_map[s_id]["lines"])
            
        # Re-sort line numbers in merged list
        for i, line in enumerate(merged_lines, 1):
            line["line_number"] = i
            
        # Construct updated scenes list
        updated_scenes = []
        for scene in current_scenes:
            if scene["scene_id"] == target_scene_id:
                scene["lines"] = merged_lines
                updated_scenes.append(scene)
            elif scene["scene_id"] in payload.scene_ids:
                # Drop merged duplicates
                continue
            else:
                updated_scenes.append(scene)
                
        db.save_chapter_scenes(project_id, payload.chapter_id, updated_scenes)
        return {"status": "success", "msg": "Scenes merged successfully.", "scenes": updated_scenes}
        
    elif action == "split":
        if not payload.scene_id or not payload.line_id:
            raise HTTPException(status_code=400, detail="Split action requires both 'scene_id' and 'line_id'.")
            
        # Find scene to split
        target_scene_idx = -1
        for idx, scene in enumerate(current_scenes):
            if scene["scene_id"] == payload.scene_id:
                target_scene_idx = idx
                break
                
        if target_scene_idx == -1:
            raise HTTPException(status_code=404, detail=f"Scene ID {payload.scene_id} not found.")
            
        target_scene = current_scenes[target_scene_idx]
        lines = target_scene["lines"]
        
        # Find splitting line index
        split_idx = -1
        for idx, line in enumerate(lines):
            if line["line_id"] == payload.line_id:
                split_idx = idx
                break
                
        if split_idx == -1:
            raise HTTPException(status_code=404, detail=f"Line ID {payload.line_id} not found in scene.")
            
        # Split lines
        part_a_lines = lines[:split_idx]
        part_b_lines = lines[split_idx:]
        
        if not part_a_lines or not part_b_lines:
            raise HTTPException(status_code=400, detail="Cannot split scene at boundaries. Both split halves must contain lines.")
            
        # Re-index line numbers for both parts
        for i, l in enumerate(part_a_lines, 1):
            l["line_number"] = i
            
        for i, l in enumerate(part_b_lines, 1):
            l["line_number"] = i
            
        # Create new scene IDs
        new_scene_id = f"{payload.scene_id}_split_{uuid.uuid4().hex[:4]}"
        
        scene_a = {"scene_id": payload.scene_id, "lines": part_a_lines}
        scene_b = {"scene_id": new_scene_id, "lines": part_b_lines}
        
        # Insert new scene
        current_scenes[target_scene_idx] = scene_a
        current_scenes.insert(target_scene_idx + 1, scene_b)
        
        db.save_chapter_scenes(project_id, payload.chapter_id, current_scenes)
        return {"status": "success", "msg": "Scene split successfully.", "scenes": current_scenes}
        
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported action: {action}")

@app.post("/api/v1/projects/{project_id}/compile")
def compile_project(project_id: str, payload: CompileRequest):
    """
    Compiles the final, modified JSON from the database/GUI into the TTS compiler.
    Saves the master compiled ACX WAV file and returns the compliance QC report.
    """
    # 1. Fetch project chapters and verify status
    chapters = db.get_chapters(project_id)
    if not chapters:
        raise HTTPException(status_code=404, detail="Project or chapters not found.")
        
    # Check if any chapter is still processing
    for chap in chapters:
        if chap["status"] != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot compile project: Chapter {chap['chapter_title']} has status '{chap['status']}'."
            )
            
    # 2. Assemble script_data
    # Use optional script_data override if provided directly by client, otherwise assemble from DB
    if payload.script_data is not None:
        logger.info("Using client-provided script_data JSON override for compilation.")
        script_data = payload.script_data
    else:
        logger.info("Assembling script_data JSON from SQLite database state...")
        chapters.sort(key=lambda x: x["order_idx"])
        all_lines = []
        global_line_counter = 1
        
        for chap in chapters:
            scenes = chap.get("scenes") or []
            for scene in scenes:
                for line in scene.get("lines") or []:
                    # Copy and update line_number to be continuous
                    line_copy = dict(line)
                    line_copy["line_number"] = global_line_counter
                    all_lines.append(line_copy)
                    global_line_counter += 1
                    
        script_data = {
            "metadata": {
                "total_parts": len(set(c["part_id"] for c in chapters)),
                "total_chapters": len(chapters),
                "total_lines_extracted": len(all_lines)
            },
            "script": all_lines
        }
        
    # 3. Call TTS pipeline
    output_master_wav = f"scratch/pipeline_workspace/project_{project_id}_master.wav"
    os.makedirs(os.path.dirname(output_master_wav), exist_ok=True)
    
    try:
        qc_report = compile_modified_json(
            script_data=script_data,
            output_master_wav=output_master_wav,
            profile_name=payload.profile,
            user_tier=payload.user_tier
        )
        return {
            "project_id": project_id,
            "master_audio_path": output_master_wav,
            "qc_report": qc_report
        }
    except Exception as e:
        logger.error(f"Compilation failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Audiobook TTS compilation crashed: {str(e)}")

# ----------------------------------------------------
# WebSocket Endpoint
# ----------------------------------------------------
@app.websocket("/api/v1/projects/{project_id}/ws")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    """WebSocket connection allowing client to listen for real-time lookahead queue status updates."""
    await manager.connect(project_id, websocket)
    try:
        # Immediately send current status upon joining
        status_payload = db.get_project_lookahead_status(project_id)
        if "error" not in status_payload:
            await websocket.send_json(status_payload)
            
        while True:
            # Keep connection alive; discard any client-to-server messages for now
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(project_id, websocket)
    except Exception as e:
        logger.warning(f"WebSocket connection error: {e}")
        manager.disconnect(project_id, websocket)


# ----------------------------------------------------
# Voice Marketplace Endpoints (Qdrant Semantic Timbre Marketplace)
# ----------------------------------------------------
class RegisterVoiceRequest(BaseModel):
    voice_name: str
    voice_ref_path: str
    description: str

@app.post("/api/v1/marketplace/voices")
def register_marketplace_voice(payload: RegisterVoiceRequest):
    """Registers a cloned voice in the marketplace vector index."""
    try:
        voice_id = marketplace.register_voice(
            voice_name=payload.voice_name,
            voice_ref_path=payload.voice_ref_path,
            description=payload.description
        )
        return {"status": "success", "voice_id": voice_id}
    except Exception as e:
        logger.error(f"Failed to register voice in marketplace: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/marketplace/search")
def search_marketplace_voices(query: str, limit: int = 5):
    """Semantically searches the voice marketplace using acoustic vector similarities."""
    try:
        results = marketplace.search_marketplace(query=query, limit=limit)
        return {"query": query, "results": results}
    except Exception as e:
        logger.error(f"Marketplace query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------
# NLP High-Throughput Batch Inference Endpoints (vLLM)
# ----------------------------------------------------
class BatchAttributionRequest(BaseModel):
    prompts: List[str]
    system_prompt: str = Field(default="You are a helpful audiobook parsing assistant.")

@app.post("/api/v1/nlp/batch_attribution")
async def batch_attribution(payload: BatchAttributionRequest):
    """Executes high-throughput batch attribution queries concurrently via vLLM client."""
    try:
        results = await vllm.query_batch(
            prompts=payload.prompts,
            system_prompt=payload.system_prompt
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Batch attribution query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
