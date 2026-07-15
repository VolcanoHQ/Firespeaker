#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Firespeaker Hierarchical Looped Analysis Engine
Implements the 6-loop Hierarchical Looped Analysis Algorithm ruleset:
- Loop 1: Metadata Extraction
- Loop 2: Macro-Structure (Part/Volume/Act)
- Loop 3: Meso-Structure (Chapter)
- Loop 4: Cognitive Scene Extraction & MemPalace Integration
- Loop 5: Entity Resolution & State Tracking
- Loop 6: Line-by-Line Attribution (Dialogue/Narration)
"""

import os
import re
import json
import logging
import urllib.request
import urllib.error
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Callable
from pydantic import BaseModel, Field, ValidationError, TypeAdapter
from src.spatial_memory import MemPalace
from src.llm_client import (
    OLLAMA_MODEL_PREFERENCE_PATTERNS,
    _list_local_ollama_models,
    _select_preferred_ollama_model,
    query_llm_json,
)

logger = logging.getLogger("LoopedAnalyzer")

# ====================================================
# Pydantic Schemas for JSON Validation (Loops 4, 5, 6)
# ====================================================

class Loop4Environment(BaseModel):
    location: str = Field(..., description="Brief description")
    time_of_day: str = Field(..., description="e.g., night, morning, unknown")
    weather: str = Field(..., description="e.g., raining, clear, indoors")
    physical_confines: str = Field(..., description="e.g., tight space, open field, echoing hall")
    ambient_noise_level: str = Field(..., description="quiet, moderate, loud")

class Loop4Scene(BaseModel):
    scene_number: int = Field(..., description="Scene number index")
    start_sentence: str = Field(..., description="Exact first sentence of the scene")
    end_sentence: str = Field(..., description="Exact last sentence of the scene")
    environment: Loop4Environment

class Loop5CleanCheckIssue(BaseModel):
    issue_type: str = Field(..., description="e.g. Gutenberg header, illustration tag, publisher metadata, page number, formatting noise")
    raw_text: str = Field(..., description="The exact raw text snippet that contains the issue")
    description: str = Field(..., description="Why this should be cleaned or omitted for narration")
    suggested_action: str = Field(..., description="'remove' to delete the text, or 'replace' to substitute it")
    suggested_text: str = Field(default="", description="The replacement text if action is 'replace'")

class Loop5ResponseSchema(BaseModel):
    is_clean: bool = Field(..., description="True if the text is clean and ready for speech, False otherwise")
    issues: List[Loop5CleanCheckIssue]

class Loop6Line(BaseModel):
    segment_type: str = Field(..., description="'dialogue' or 'narrative'")
    speaker: str = Field(..., description="Speaker name matching active cast list or '[NARRATION]'")
    text: str = Field(..., description="Spoken dialogue or narrative text block")
    emotion: str = Field(..., description="Emotion label, e.g. Joy, Sadness, Tension, Neutral")
    delivery_style: str = Field(..., description="Delivery style descriptive tag")

class Loop6ResponseSchema(BaseModel):
    lines: List[Loop6Line]

class HierarchicalLoopedAnalyzer:
    """Analyzes book manuscripts through the 6-loop Hierarchical Looped Analysis Algorithm."""

    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate", model_name: Optional[str] = None):
        self.ollama_url = ollama_url
        available_models = _list_local_ollama_models()
        self.model_name = _select_preferred_ollama_model(available_models, model_name)
        if self.model_name:
            logger.info(f"Looped analyzer selected Ollama model: {self.model_name}")
        else:
            logger.warning("No local Ollama model detected; Looped analyzer will fall back when queries fail.")

    def _query_ollama_json(self, prompt: str, timeout: float = 180.0, retries: int = 2, task_name: str = "generic") -> Optional[Dict[str, Any]]:
        """Queries the LLM fallback chain (Gemini -> Groq -> local Ollama) in JSON mode.

        Name/signature kept stable for the 4 existing call sites (loop1/4/5/6); only the
        body changed to delegate to src.llm_client, which tries free-tier cloud providers
        before falling back to this instance's local Ollama model.
        """
        result, provider = query_llm_json(prompt, timeout=timeout, task_name=task_name)
        if provider:
            logger.info(f"Looped analyzer query ({task_name}) served by {provider}")
        return result

    # ====================================================
    # Loop 1: Metadata Extraction
    # ====================================================
    def loop1_extract_metadata(self, raw_text: str) -> Dict[str, Any]:
        """Extracts standard metadata points and start/end markers of the narrative."""
        metadata = {
            "title": "Unknown Title",
            "author": "Unknown Author",
            "translator_illustrator": "None",
            "publisher": "Unknown Publisher",
            "publication_date": "Unknown Date",
            "source_format": "Project Gutenberg eBook"
        }

        # Scan first 5% of the text for metadata
        scan_len = int(len(raw_text) * 0.05)
        intro_text = raw_text[:max(scan_len, 4000)]

        # Find start and end markers
        start_match = re.search(r'\*\*\*\s*START OF (?:THE\s+)?PROJECT GUTENBERG EBOOK.*?\*\*\*', raw_text, re.IGNORECASE)
        start_pos = start_match.end() if start_match else 0
        
        end_match = re.search(r'\*\*\*\s*END OF (?:THE\s+)?PROJECT GUTENBERG EBOOK.*?\*\*\*', raw_text, re.IGNORECASE)
        end_pos = end_match.start() if end_match else len(raw_text)

        # Use LLM to extract metadata parameters cleanly
        prompt = f"""Read the book header introduction snippet below.
Extract and return a JSON object with the following keys:
- "title" (string, the title of the book)
- "author" (string, the author of the book)
- "translator_illustrator" (string, the illustrator or translator if mentioned, otherwise "None")
- "publisher" (string, publisher name if mentioned, otherwise "Unknown")
- "publication_date" (string, original release or update date)

HEADER SNIPPET:
{intro_text[:1500]}
"""
        llm_data = self._query_ollama_json(prompt, task_name="loop1_metadata")
        if llm_data:
            metadata.update(llm_data)
        else:
            # Fallback regex metadata parsing
            title_m = re.search(r'Title:\s*(.+)', intro_text, re.IGNORECASE)
            if title_m:
                metadata["title"] = title_m.group(1).strip()
            author_m = re.search(r'Author:\s*(.+)', intro_text, re.IGNORECASE)
            if author_m:
                metadata["author"] = author_m.group(1).strip()
            pub_m = re.search(r'Release date:\s*(.+)', intro_text, re.IGNORECASE)
            if pub_m:
                metadata["publication_date"] = pub_m.group(1).strip()

        # Extract [Illustration] tags (User Request)
        illustration_pattern = re.compile(r'\[Illustration.*?\]', re.IGNORECASE | re.DOTALL)
        illustrations = []
        for match in illustration_pattern.finditer(raw_text):
            illustrations.append({
                "matched_text": match.group(0),
                "start_index": match.start(),
                "end_index": match.end()
            })
        metadata["illustrations"] = illustrations

        metadata["narrative_start_marker_found"] = start_match is not None
        metadata["narrative_start_index"] = start_pos
        metadata["narrative_end_index"] = end_pos

        return metadata

    # ====================================================
    # Loop 2: Macro-Structure (Part/Volume/Act)
    # ====================================================
    def loop2_extract_parts(self, cleaned_text: str) -> Tuple[List[Dict[str, Any]], bool]:
        """Identifies Part/Volume divisions. Returns parts list and a bypass flag."""
        part_pattern = re.compile(
            r'^\s*(?:VOLUME|BOOK|PART|LIVRE|TOME|ACT)\s+(?:[IVXLCDM]+|[0-9]+|FIRST|SECOND|THIRD|FOURTH|FIFTH|ONE|TWO|THREE|FOUR|FIVE)\b.*$',
            re.IGNORECASE | re.MULTILINE
        )
        
        headings = [m.strip() for m in part_pattern.findall(cleaned_text)]
        splits = part_pattern.split(cleaned_text)

        if len(splits) <= 1:
            return [], True  # Parts Detected: None, Bypass to Loop 3

        parts = []
        preface = splits[0].strip()
        if preface and len(preface) > 100:
            parts.append({
                "part_id": "part_p0",
                "title": "Preface/Front Matter",
                "text_block": preface
            })

        for idx, heading in enumerate(headings):
            text_chunk = splits[idx + 1].strip() if idx + 1 < len(splits) else ""
            if not text_chunk:
                continue
            parts.append({
                "part_id": f"part_p{len(parts) + 1}",
                "title": heading,
                "text_block": text_chunk
            })

        return parts, False

    # ====================================================
    # Loop 3: Meso-Structure (Chapter)
    # ====================================================
    def loop3_extract_chapters(self, text_block: str, part_id: str = "main") -> Tuple[List[Dict[str, Any]], bool]:
        """Slices part text block into sequential chapters. Returns chapters list and bypass flag."""
        chapter_pattern = re.compile(
            r'^\s*(?:CHAPTER|CHAPITRE|SCENE|LETTER)\s+(?:[IVXLCDM]+|[0-9]+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|FIRST|SECOND|THIRD)\b.*$',
            re.IGNORECASE | re.MULTILINE
        )
        
        headings = [m.strip() for m in chapter_pattern.findall(text_block)]
        splits = chapter_pattern.split(text_block)

        if len(splits) <= 1:
            return [], True  # Chapters Detected: None, Bypass to Loop 4

        chapters = []
        preface = splits[0].strip()
        if preface and len(preface) > 100:
            chapters.append({
                "chapter_id": f"{part_id}_c0",
                "title": "Prologue",
                "text_block": preface
            })

        for idx, heading in enumerate(headings):
            text_chunk = splits[idx + 1].strip() if idx + 1 < len(splits) else ""
            if not text_chunk:
                continue
            chapters.append({
                "chapter_id": f"{part_id}_c{len(chapters) + 1}",
                "title": heading,
                "text_block": text_chunk
            })

        return chapters, False

    # ====================================================
    # Loop 4: Cognitive Scene Extraction & MemPalace Integration
    # ====================================================
    def loop4_extract_scenes(self, chapter_text: str, chapter_id: str, chapter_num: int, palace: MemPalace) -> List[Dict[str, Any]]:
        """
        Semantically divides chapters into discrete cinematic scenes,
        extracts environmental variables, and commits them to MemPalace.
        """
        logger.info(f"Loop 4: Extracting scenes for Chapter {chapter_id}...")
        
        # Check for explicit typographical boundaries first (e.g. ***, ---, [Illustration])
        scene_markers = r'(?:\n)\s*(?:\*\s*\*|\#|-{3,}|_{3,})\s*(?:\n)'
        typographical_splits = [s.strip() for s in re.split(scene_markers, chapter_text) if s.strip()]
        
        # Divide chapter text into non-overlapping blocks of 4,000 characters (split on paragraph breaks) to avoid timeouts
        chunks = []
        start = 0
        while start < len(chapter_text):
            if len(chapter_text) - start <= 4500:
                chunks.append(chapter_text[start:])
                break
            else:
                end = start + 4000
                # Find nearest paragraph break to split cleanly
                split_idx = chapter_text.find("\n\n", end - 500, end + 500)
                if split_idx != -1:
                    end = split_idx
                chunks.append(chapter_text[start:end].strip())
                start = end

        scenes_data = []
        for c_idx, chunk in enumerate(chunks, 1):
            prompt = f"""You are a cinematic script supervisor and acoustic environment analyst for an audiobook engine. Your objective is to divide a chapter into discrete scenes and extract environmental metadata.

INSTRUCTIONS:
1. Read the provided chapter text and trigger a hard scene break ONLY upon:
   - Spatial Shifts: A physical change in location (e.g., indoors to outdoors).
   - Temporal Shifts: A jump in time (e.g., "The next morning...").
   - POV Shifts: A distinct change in narrative perspective.
2. For each identified scene, deduce the environmental variables to inform the audio engine's reverb and ambient noise profiles.

OUTPUT FORMAT:
You must output a valid JSON array of objects. Each object must strictly adhere to this schema:
{{
  "scene_number": [Integer],
  "start_sentence": "[Exact first sentence of the scene]",
  "end_sentence": "[Exact last sentence of the scene]",
  "environment": {{
    "location": "[Brief description]",
    "time_of_day": "[e.g., night, morning, unknown]",
    "weather": "[e.g., raining, clear, indoors]",
    "physical_confines": "[e.g., tight space, open field, echoing hall]",
    "ambient_noise_level": "[quiet, moderate, loud]"
  }}
}}

CHAPTER TEXT SEGMENT:
{chunk}
"""
            llm_res = self._query_ollama_json(prompt, task_name="loop4_scenes")
            chunk_scenes_data = []
            schema_validated = False
            if llm_res:
                try:
                    candidate_scenes = llm_res
                    if isinstance(llm_res, dict):
                        if isinstance(llm_res.get("scenes"), list):
                            candidate_scenes = llm_res["scenes"]
                        else:
                            candidate_scenes = [llm_res]

                    # Validate list of objects using TypeAdapter
                    adapter = TypeAdapter(List[Loop4Scene])
                    validated = adapter.validate_python(candidate_scenes)
                    chunk_scenes_data = [s.model_dump() for s in validated]
                    schema_validated = True
                    logger.info(f"Loop 4 Chunk {c_idx}/{len(chunks)}: Successfully validated {len(chunk_scenes_data)} scenes.")
                except ValidationError as ve:
                    logger.error(f"Loop 4 Chunk {c_idx}/{len(chunks)} JSON Schema Validation failed: {ve}.")
            
            if schema_validated:
                for s in chunk_scenes_data:
                    s["source_chunk_idx"] = c_idx
                scenes_data.extend(chunk_scenes_data)
            else:
                # Fallback heuristic for this chunk: Treat the chunk as a single scene
                scenes_data.append({
                    "scene_number": c_idx,
                    "start_sentence": chunk[:50],
                    "end_sentence": chunk[-50:],
                    "source_chunk_idx": c_idx,
                    "environment": {
                        "location": f"Main Narrative Scene {c_idx}",
                        "time_of_day": "unknown",
                        "weather": "unknown",
                        "physical_confines": "open",
                        "ambient_noise_level": "quiet"
                    }
                })

        scenes = []
        for s_idx, s_data in enumerate(scenes_data, 1):
            scene_id = f"{chapter_id}_s{s_idx}"
            
            # Find the actual text chunk in the source chunk using the start and end markers
            start_marker = s_data.get("start_sentence", "")
            end_marker = s_data.get("end_sentence", "")
            
            chunk_idx = s_data.get("source_chunk_idx")
            source_text = chapter_text
            if chunk_idx is not None and chunk_idx <= len(chunks):
                source_text = chunks[chunk_idx - 1]
                
            scene_text = self._extract_scene_text_chunk(source_text, start_marker, end_marker)
            if not scene_text:
                # Fallback to typographical split or source chunk text
                chunk_idx = s_data.get("source_chunk_idx")
                if len(typographical_splits) >= s_idx:
                    scene_text = typographical_splits[s_idx - 1]
                elif chunk_idx is not None and chunk_idx <= len(chunks):
                    scene_text = chunks[chunk_idx - 1]
                elif len(chunks) >= s_idx:
                    scene_text = chunks[s_idx - 1]
                else:
                    scene_text = chapter_text
            
            # Pack metadata matching Loop 4 output specs
            env_data = s_data.get("environment", {})
            location_text = env_data.get("location", "Narrative Block")
            
            environment_profile = {
                "location": location_text,
                "time_of_day": env_data.get("time_of_day", "unknown"),
                "weather": env_data.get("weather", "unknown"),
                "physical_confines": env_data.get("physical_confines", "open"),
                "ambient_noise_level": env_data.get("ambient_noise_level", "quiet")
            }
            
            summary_text = f"Scene at {location_text}."
            
            # MemPalace Synchronization: Write to SQLite wings table
            palace.log_wing(
                wing_id=scene_id,
                chapter_number=chapter_num,
                title=location_text,
                metadata={
                    "scene_id": scene_id,
                    "location": location_text,
                    "summary": summary_text,
                    "environment": environment_profile,
                    "start_marker": start_marker,
                    "end_marker": end_marker
                }
            )
            
            scenes.append({
                "scene_id": scene_id,
                "location": location_text,
                "summary": summary_text,
                "environment": environment_profile,
                "text_block": scene_text
            })
            
        return scenes

    def _extract_scene_text_chunk(self, chapter_text: str, start_marker: str, end_marker: str) -> Optional[str]:
        """Utility to slice chapter text using flexible word markers."""
        if not start_marker or not end_marker:
            return None
        # Clean markers of surrounding quotes
        start_marker = start_marker.strip('"\' ')
        end_marker = end_marker.strip('"\' ')
        
        # Word search
        start_words = start_marker.split()[:3]
        end_words = end_marker.split()[-3:]
        
        if not start_words or not end_words:
            return None
            
        start_pat = r'\s+'.join(re.escape(w) for w in start_words)
        end_pat = r'\s+'.join(re.escape(w) for w in end_words)
        
        start_match = re.search(start_pat, chapter_text)
        end_match = re.search(end_pat, chapter_text)
        
        if start_match and end_match and start_match.start() <= end_match.end():
            return chapter_text[start_match.start():end_match.end()]
            
        return None

    # ====================================================
    # Loop 5: Entity Resolution & State Tracking
    # ====================================================
    def loop5_manuscript_clean_check(self, scene_id: str, scene_text: str, palace: MemPalace) -> Dict[str, Any]:
        """
        Runs a manuscript clean check (Loop 5) on a raw text block to detect Gutenberg headers,
        license clutter, page numbers, illustrations, or formatting errors.
        """
        logger.info(f"Loop 5: Running manuscript clean check for {scene_id}...")
        
        prompt = f"""You are a professional manuscript editor and ingestion auditor for an audiobook production engine.
Your objective is to run a "Manuscript Clean Check" (Loop 5) on the provided text block.
Identify any non-narrative elements that should be removed or cleaned before sending to the Speech Synthesis (TTS) engine.

Identify:
1. Project Gutenberg license headers/footers, metadata, translator/author lists, or ebook credits.
2. Illustration tags or description brackets (e.g. "[Illustration: ...]" or "[Illustration]").
3. Page numbers, headers, footers, or transcriber notes.
4. Extraneous formatting noise (e.g. raw underscores representing formatting that shouldn't be read as words).

OUTPUT FORMAT:
You must output a valid JSON object matching this schema:
{{
  "is_clean": [Boolean: true if no issues are found, false if issues are found],
  "issues": [
    {{
      "issue_type": "[e.g. Gutenberg header, illustration tag, page number, formatting noise]",
      "raw_text": "[The exact raw text snippet containing the issue]",
      "description": "[Why this text should be cleaned or omitted for narration]",
      "suggested_action": "['remove' or 'replace']",
      "suggested_text": "[Replacement text if action is 'replace', otherwise empty string]"
    }}
  ]
}}

TEXT BLOCK TO AUDIT:
{scene_text[:4000]}
"""
        llm_res = self._query_ollama_json(prompt, task_name="loop5_clean")
        if llm_res:
            try:
                validated = Loop5ResponseSchema.model_validate(llm_res)
                res_dict = validated.model_dump()
                logger.info(f"Loop 5: Successfully validated clean check response. Found {len(res_dict['issues'])} issues.")
                return res_dict
            except ValidationError as ve:
                logger.error(f"Loop 5 Clean Check Schema validation failed: {ve}")
                
        # Return default clean state if LLM fails
        return {"is_clean": True, "issues": []}

    # ====================================================
    # Loop 6: Line-by-Line Attribution (Dialogue/Narration)
    # ====================================================
    def loop6_attribute_lines(self, scene_id: str, scene_text: str, characters_present: List[Any], palace: MemPalace) -> List[Dict[str, Any]]:
        """
        Segments dialogue and narration, resolves attributions to the active cast,
        packages environmental/emotional parameters, and logs script rooms to MemPalace.
        """
        logger.info(f"Loop 6: Attributing lines for Scene {scene_id}...")
        
        # Read the environmental settings and cast from MemPalace scene record
        cursor = palace.conn.cursor()
        cursor.execute("SELECT metadata_json FROM wings WHERE wing_id = ?;", (scene_id,))
        row = cursor.fetchone()
        wing_metadata = json.loads(row[0]) if row else {}
        environment = wing_metadata.get("environment", {})
        
        # Database-Driven Attribution: Analyze active characters logged in MemPalace scene record
        db_characters_present = wing_metadata.get("characters_present", [])
        if db_characters_present:
            characters_present = db_characters_present

        normalized_characters_present = []
        for character_entry in characters_present:
            if isinstance(character_entry, str):
                normalized_characters_present.append({"name": character_entry})
                continue

            if isinstance(character_entry, dict):
                character_name = (
                    character_entry.get("name")
                    or character_entry.get("character_name")
                    or character_entry.get("character")
                )
                if character_name:
                    normalized_characters_present.append({
                        **character_entry,
                        "name": character_name
                    })

        characters_present = normalized_characters_present
            
        # Available cast list names for attribution locking
        active_cast = [c.get("name") for c in characters_present]
        active_cast_str = ", ".join(active_cast)

        # Parse text into paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n+', scene_text) if p.strip()]
        
        # Process in batches of 4 paragraphs to prevent Ollama CPU timeouts
        batch_size = 4
        raw_lines = []
        
        for b_idx in range(0, len(paragraphs), batch_size):
            batch_paras = paragraphs[b_idx:b_idx + batch_size]
            batch_text = "\n\n".join(batch_paras)
            
            # Build prompt to extract sequential segments with schema mapping speaker, emotion, and style
            prompt = f"""Segment the text block into consecutive narrative and dialogue blocks in exact chronological order.
- For dialogue (quoted text), attribute to a speaker from the Active Cast List.
- For narration (non-spoken text), set speaker to "[NARRATION]" and segment_type to "narrative".

Here is a reference example of how to parse text:
---
EXAMPLE SCENE TEXT:
"The boy stood up. 'I must go now,' he whispered."

EXAMPLE ACTIVE CAST LIST: [Peter Rabbit]

EXAMPLE OUTPUT JSON:
{{
  "lines": [
    {{
      "segment_type": "narrative",
      "text": "The boy stood up.",
      "speaker": "[NARRATION]",
      "emotion": "Neutral",
      "delivery_style": "descriptive"
    }},
    {{
      "segment_type": "dialogue",
      "speaker": "Peter Rabbit",
      "text": "I must go now,",
      "emotion": "Whispering",
      "delivery_style": "quiet"
    }}
  ]
}}
---

Active Cast List (relational character data queried from MemPalace): [{active_cast_str}]

SCENE TEXT:
{batch_text}

Now, segment the SCENE TEXT above.
Do NOT copy the example text ("The boy stood up.", "I must go now,", etc.). Every 'text' value in the JSON MUST be verbatim from the SCENE TEXT.

Return a JSON object containing a "lines" key with an array of objects matching the schema structure.
"""
            llm_res = self._query_ollama_json(prompt, task_name="loop6_attribution")
            batch_validated = False
            if llm_res:
                try:
                    validated = Loop6ResponseSchema.model_validate(llm_res)
                    raw_lines.extend([line.model_dump() for line in validated.lines])
                    batch_validated = True
                    logger.info(f"Loop 6 Batch {b_idx//batch_size + 1}: Successfully validated {len(validated.lines)} lines against Loop6ResponseSchema.")
                except ValidationError as ve:
                    logger.error(f"Loop 6 Batch {b_idx//batch_size + 1} JSON Schema Validation failed: {ve}.")

            if not batch_validated:
                # Fallback heuristic for this batch
                for p_idx, para in enumerate(batch_paras):
                    is_spoken = '"' in para or "'" in para or '“' in para or '”' in para
                    clean_text = para.strip('“"\'” ')
                    speaker = "[NARRATION]"
                    segment_type = "narrative"
                    if is_spoken:
                        segment_type = "dialogue"
                        speaker = active_cast[0] if active_cast else "[NARRATION]"
                    raw_lines.append({
                        "segment_type": segment_type,
                        "text": clean_text,
                        "speaker": speaker,
                        "emotion": "Neutral",
                        "delivery_style": "descriptive"
                    })

        attributed_blocks = []
        registered_speakers = set()
        for idx, line_data in enumerate(raw_lines, 1):
            text = line_data.get("text", "").strip()
            if not text:
                continue
                
            speaker = line_data.get("speaker", "[NARRATION]").strip()
            if speaker.lower() in ("narrator", "[narration]", "narration"):
                speaker = "[NARRATION]"
                
            # Secure speaker is locked inside active roster
            if speaker != "[NARRATION]" and active_cast and speaker not in active_cast:
                # Lock to closest active character matching
                best_match = active_cast[0]
                for c in active_cast:
                    if c.lower() in speaker.lower() or speaker.lower() in c.lower():
                        best_match = c
                        break
                speaker = best_match

            if speaker not in registered_speakers and not palace.get_character_drawer(speaker):
                logger.info(f"Loop 6: Auto-registering default drawer for speaker '{speaker}'")
                palace.register_character(
                    character_name=speaker,
                    voice_ref_path="data/voice_references/narrator_mono.wav",
                    speed=1.0,
                    pitch=0.0
                )
            registered_speakers.add(speaker)

            # Generate unique room_id
            slug = re.sub(r'[^a-zA-Z0-9]', '', scene_id)
            raw_id = f"{slug}_l{idx}_{text[:20]}"
            room_id = hashlib.sha256(raw_id.encode('utf-8')).hexdigest()[:16]
            
            emotion = line_data.get("emotion", "Neutral").title()
            delivery = line_data.get("delivery_style", "descriptive")
            
            # Resolve performance speed and pitch modifiers based on emotion
            pitch = 1.0
            speed = 1.0
            if emotion.lower() in {"sadness", "grief", "sad"}:
                pitch = 0.90
                speed = 0.85
            elif emotion.lower() in {"fear", "nervousness", "panic", "tension"}:
                pitch = 1.15
                speed = 1.10
            elif emotion.lower() in {"anger", "annoyance"}:
                pitch = 0.95
                speed = 1.05
            elif emotion.lower() in {"joy", "excitement"}:
                pitch = 1.05
                speed = 1.02

            performance_payload = {
                "pitch_modifier": pitch,
                "speed_modifier": speed,
                "delivery_style": delivery,
                "scene_environment": environment
            }

            # Sync to Relational Database (Rooms Table)
            palace.log_room(
                room_id=room_id,
                wing_id=scene_id,
                line_number=idx,
                character_name=speaker,
                dialogue_text=text,
                emotion=emotion,
                metadata=performance_payload,
                confidence=float(line_data.get("confidence", 0.90))
            )
            
            # Calculate post-padding based on segment position and punctuation to prevent odd pauses
            is_last_seg = (idx == len(raw_lines))
            if is_last_seg:
                padding = 600
            else:
                last_char = text.strip()[-1] if text.strip() else ''
                if last_char in {'.', '?', '!'}:
                    padding = 200
                else:
                    padding = 50

            attributed_blocks.append({
                "line_id": room_id,
                "line_number": idx,
                "segment_type": line_data.get("segment_type", "narrative"),
                "speaker_id": "char_narrator" if speaker == "[NARRATION]" else f"char_{speaker.lower().replace(' ', '_')}",
                "character": speaker,
                "text": text,
                "emotion": emotion,
                "performance": performance_payload,
                "post_padding_ms": padding
            })
            
        return attributed_blocks

    def analyze_book(self, filepath: str, global_roster: List[str] = None, review_callback: Optional[Callable[[Dict[str, Any], str], bool]] = None, chapters: str = None) -> Dict[str, Any]:
        """Runs the complete 6-loop analysis on the target book manuscript, locking state in MemPalace."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Manuscript file not found: {filepath}")
            
        with open(filepath, "r", encoding="utf-8") as f:
            raw_content = f.read()

        logger.info(f"Starting 6-loop hierarchical analysis on {os.path.basename(filepath)}")

        # Initialize MemPalace connection
        palace = MemPalace(use_chroma=False)

        # Pre-register default narration speakers to satisfy database foreign keys
        cursor = palace.conn.cursor()
        for default_sp in ["Narrator", "[NARRATION]"]:
            cursor.execute("SELECT character_name FROM drawers WHERE character_name = ?;", (default_sp,))
            if not cursor.fetchone():
                palace.register_character(
                    character_name=default_sp,
                    voice_ref_path="data/voice_references/narrator_mono.wav"
                )

        selected_chapters = None
        if chapters:
            try:
                selected_chapters = set()
                for part in chapters.split(','):
                    if '-' in part:
                        start, end = part.split('-')
                        selected_chapters.update(range(int(start), int(end) + 1))
                    else:
                        selected_chapters.add(int(part))
                logger.info(f"Analysis filter active: targeting chapters {selected_chapters}")
            except Exception as e:
                logger.error(f"Failed to parse chapters filter '{chapters}': {e}")

        # Loop 1: Metadata Extraction
        metadata = self.loop1_extract_metadata(raw_content)
        
        # Chop text to narrative boundaries
        narrative_text = raw_content[metadata["narrative_start_index"]:metadata["narrative_end_index"]].strip()

        # Clean [Illustration] tags from narrative_text (User Request)
        illustration_pattern = re.compile(r'\[Illustration.*?\]', re.IGNORECASE | re.DOTALL)
        narrative_text = illustration_pattern.sub('\n\n', narrative_text).strip()

        # Write extracted loop1 text to folder (User Request 8)
        os.makedirs("data/corpus/extracted", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        loop1_file = os.path.join("data/corpus/extracted", f"{base_name}_loop1.txt")
        with open(loop1_file, "w", encoding="utf-8") as f:
            f.write(narrative_text)
        logger.info(f"Loop 1: Extracted clean narrative saved to {loop1_file}")

        # Create manuscript pipeline folder for all loop artifacts (User Request)
        # Namespaced under looped_analyzer/ -- src/tier_1_parser.py writes its own
        # (differently loop-numbered) artifacts to the same {book}/ directory under
        # tier1/, keeping the two independent pipelines' outputs from colliding.
        pipeline_dir = os.path.join("data/corpus/pipeline", base_name, "looped_analyzer")
        os.makedirs(pipeline_dir, exist_ok=True)

        # Save Loop 1 artifacts
        with open(os.path.join(pipeline_dir, "loop1_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        with open(os.path.join(pipeline_dir, "loop1_narrative.txt"), "w", encoding="utf-8") as f:
            f.write(narrative_text)
        with open(os.path.join(pipeline_dir, "loop1_illustrations.json"), "w", encoding="utf-8") as f:
            json.dump(metadata.get("illustrations", []), f, indent=4)

        # Sanity Checks / Final Review Gate
        if not narrative_text:
            logger.error("Loop 1 Validation Failed: Narrative text is empty!")
            raise ValueError("Loop 1 narrative text cannot be empty.")
            
        if len(narrative_text) < 100:
            logger.error("Loop 1 Validation Failed: Narrative text is extremely short (< 100 characters)!")
            raise ValueError("Loop 1 narrative text is too short. Boundary extraction likely failed.")

        if metadata["narrative_start_index"] >= metadata["narrative_end_index"]:
            logger.error(f"Loop 1 Validation Failed: Start index ({metadata['narrative_start_index']}) is >= End index ({metadata['narrative_end_index']})")
            raise ValueError("Invalid narrative boundary indices detected in Loop 1.")

        # Interactive/External Review Gate Callback
        if review_callback:
            logger.info("Loop 1 Review Gate: Invoking review callback...")
            approved = review_callback(metadata, narrative_text)
            if not approved:
                logger.warning("Loop 1 Review Gate: Analysis rejected by reviewer.")
                raise PermissionError("Analysis aborted at Loop 1 review stage.")
        
        # Loop 2: Macro-Structure
        parts, bypass_parts = self.loop2_extract_parts(narrative_text)

        # Save Loop 2 artifact
        loop2_artifact = {
            "parts": parts,
            "bypass_parts": bypass_parts
        }
        with open(os.path.join(pipeline_dir, "loop2_parts.json"), "w", encoding="utf-8") as f:
            json.dump(loop2_artifact, f, indent=4)
        
        final_hierarchy = []
        total_chapters = 0
        total_scenes = 0
        
        # Initialize collectors for loop 3-6 artifacts
        all_loop3_chapters = []
        all_loop4_scenes = []
        all_loop5_clean_checks = []
        all_loop6_lines = []
        
        if bypass_parts:
            # Bypassed Parts: Parse Chapters directly
            chapters_list, bypass_chapters = self.loop3_extract_chapters(narrative_text, "part1")
            all_loop3_chapters.extend(chapters_list)
            chapter_payloads = []
            
            if bypass_chapters:
                # Bypassed Chapters: Slice Scenes directly
                total_chapters += 1
                if not (selected_chapters and total_chapters not in selected_chapters):
                    scenes = self.loop4_extract_scenes(narrative_text, "part1_c1", 1, palace)
                    all_loop4_scenes.extend(scenes)
                    for scene in scenes:
                        total_scenes += 1
                        scene_id = scene["scene_id"]
                        
                        # Loop 5: Manuscript Clean Check
                        clean_check = self.loop5_manuscript_clean_check(scene_id, scene["text_block"], palace)
                        all_loop5_clean_checks.append({
                            "scene_id": scene_id,
                            "clean_check": clean_check
                        })
                        # Heuristically resolve characters present for Loop 6
                        present_chars = []
                        for char in (global_roster or ["Narrator"]):
                            if char.lower() != "narrator" and re.search(r'\b' + re.escape(char.lower()) + r'\b', scene["text_block"].lower()):
                                present_chars.append(char)
                        if not present_chars:
                            present_chars = ["Narrator"]
                        # Loop 6: Line-by-Line Attribution
                        lines = self.loop6_attribute_lines(scene_id, scene["text_block"], present_chars, palace)
                        all_loop6_lines.append({
                            "scene_id": scene_id,
                            "lines": lines
                        })
                        scene["characters_present"] = present_chars
                        scene["lines"] = lines
                        
                    chapter_payloads.append({
                        "chapter_id": "part1_c1",
                        "title": "Main Narrative Block",
                        "scenes": scenes
                    })
            else:
                for c_idx, chap in enumerate(chapters_list, 1):
                    total_chapters += 1
                    if selected_chapters and total_chapters not in selected_chapters:
                        continue
                    scenes = self.loop4_extract_scenes(chap["text_block"], chap["chapter_id"], total_chapters, palace)
                    all_loop4_scenes.extend(scenes)
                    for scene in scenes:
                        total_scenes += 1
                        scene_id = scene["scene_id"]
                        
                        # Loop 5: Manuscript Clean Check
                        clean_check = self.loop5_manuscript_clean_check(scene_id, scene["text_block"], palace)
                        all_loop5_clean_checks.append({
                            "scene_id": scene_id,
                            "clean_check": clean_check
                        })
                        # Heuristically resolve characters present for Loop 6
                        present_chars = []
                        for char in (global_roster or ["Narrator"]):
                            if char.lower() != "narrator" and re.search(r'\b' + re.escape(char.lower()) + r'\b', scene["text_block"].lower()):
                                present_chars.append(char)
                        if not present_chars:
                            present_chars = ["Narrator"]
                        lines = self.loop6_attribute_lines(scene_id, scene["text_block"], present_chars, palace)
                        all_loop6_lines.append({
                            "scene_id": scene_id,
                            "lines": lines
                        })
                        scene["characters_present"] = present_chars
                        scene["lines"] = lines
                    chap["scenes"] = scenes
                    chapter_payloads.append(chap)
                    
            if chapter_payloads:
                final_hierarchy.append({
                    "part_id": "part1",
                    "title": "Main Part",
                    "chapters": chapter_payloads
                })
        else:
            for p_idx, part in enumerate(parts):
                chapters_list, bypass_chapters = self.loop3_extract_chapters(part["text_block"], part["part_id"])
                all_loop3_chapters.extend(chapters_list)
                chapter_payloads = []
                
                if bypass_chapters:
                    total_chapters += 1
                    if selected_chapters and total_chapters not in selected_chapters:
                        continue
                    scenes = self.loop4_extract_scenes(part["text_block"], f"{part['part_id']}_c1", total_chapters, palace)
                    all_loop4_scenes.extend(scenes)
                    for scene in scenes:
                        total_scenes += 1
                        scene_id = scene["scene_id"]
                        
                        # Loop 5: Manuscript Clean Check
                        clean_check = self.loop5_manuscript_clean_check(scene_id, scene["text_block"], palace)
                        all_loop5_clean_checks.append({
                            "scene_id": scene_id,
                            "clean_check": clean_check
                        })
                        # Heuristically resolve characters present for Loop 6
                        present_chars = []
                        for char in (global_roster or ["Narrator"]):
                            if char.lower() != "narrator" and re.search(r'\b' + re.escape(char.lower()) + r'\b', scene["text_block"].lower()):
                                present_chars.append(char)
                        if not present_chars:
                            present_chars = ["Narrator"]
                        lines = self.loop6_attribute_lines(scene_id, scene["text_block"], present_chars, palace)
                        all_loop6_lines.append({
                            "scene_id": scene_id,
                            "lines": lines
                        })
                        scene["characters_present"] = present_chars
                        scene["lines"] = lines
                    chapter_payloads.append({
                        "chapter_id": f"{part['part_id']}_c1",
                        "title": "Main Narrative Block",
                        "scenes": scenes
                    })
                else:
                    for chap in chapters_list:
                        total_chapters += 1
                        if selected_chapters and total_chapters not in selected_chapters:
                            continue
                        scenes = self.loop4_extract_scenes(chap["text_block"], chap["chapter_id"], total_chapters, palace)
                        all_loop4_scenes.extend(scenes)
                        for scene in scenes:
                            total_scenes += 1
                            scene_id = scene["scene_id"]
                            
                            # Loop 5: Manuscript Clean Check
                            clean_check = self.loop5_manuscript_clean_check(scene_id, scene["text_block"], palace)
                            all_loop5_clean_checks.append({
                                "scene_id": scene_id,
                                "clean_check": clean_check
                            })
                            # Heuristically resolve characters present for Loop 6
                            present_chars = []
                            for char in (global_roster or ["Narrator"]):
                                if char.lower() != "narrator" and re.search(r'\b' + re.escape(char.lower()) + r'\b', scene["text_block"].lower()):
                                    present_chars.append(char)
                            if not present_chars:
                                present_chars = ["Narrator"]
                            lines = self.loop6_attribute_lines(scene_id, scene["text_block"], present_chars, palace)
                            all_loop6_lines.append({
                                "scene_id": scene_id,
                                "lines": lines
                            })
                            scene["characters_present"] = present_chars
                            scene["lines"] = lines
                        chap["scenes"] = scenes
                        chapter_payloads.append(chap)
                if chapter_payloads:
                    part["chapters"] = chapter_payloads
                    final_hierarchy.append(part)

        # Write Loop 3-6 artifacts
        loop3_artifact = {
            "chapters": all_loop3_chapters
        }
        with open(os.path.join(pipeline_dir, "loop3_chapters.json"), "w", encoding="utf-8") as f:
            json.dump(loop3_artifact, f, indent=4)

        with open(os.path.join(pipeline_dir, "loop4_scenes.json"), "w", encoding="utf-8") as f:
            json.dump(all_loop4_scenes, f, indent=4)

        with open(os.path.join(pipeline_dir, "loop5_clean_check.json"), "w", encoding="utf-8") as f:
            json.dump(all_loop5_clean_checks, f, indent=4)

        with open(os.path.join(pipeline_dir, "loop6_lines.json"), "w", encoding="utf-8") as f:
            json.dump(all_loop6_lines, f, indent=4)

        palace.close()
        return {
            "metadata": metadata,
            "stats": {
                "parts_detected": "None" if bypass_parts else len(parts),
                "total_chapters": len(all_loop3_chapters) if not selected_chapters else len(selected_chapters),
                "total_scenes": total_scenes
            },
            "parts": final_hierarchy
        }
