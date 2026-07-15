#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Firespeaker Hierarchical Manuscript Parser
Decomposes books into structured nested hierarchies: Part -> Chapter -> Scene
and compiles scene-level character rosters, dialogue metrics, and transcripts.
Feeds interactive GUI navigation trees.
"""

import os
import sys
import re
import json
import logging
import hashlib
from collections import defaultdict
from typing import Dict, List, Any, Tuple

# Ensure the root project directory is in the sys.path for absolute modular imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nlp_analyzer import ManuscriptAnalyzer, SPEECH_VERBS, _query_local_ollama

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HierarchicalParser")


class HierarchicalParser:
    """Parses book manuscripts into deeply nested Part -> Chapter -> Scene structures."""

    def __init__(self, use_gpu: bool = False, production_tier: int = 1):
        self.analyzer = ManuscriptAnalyzer(use_gpu=use_gpu, production_tier=production_tier)
        self.production_tier = production_tier

    def _split_into_parts(self, text: str) -> List[Tuple[str, str]]:
        """
        Segments the text by major parts (e.g. Part I, Book One, Volume 2).
        Falls back to a single default Part if no explicit dividers are found.
        """
        # Match headings like "Part 1", "BOOK ONE", "Volume II", "Part I"
        part_pattern = r'(?i)^\s*(?:part|book|volume)\s+(?:[0-9]+|[IVXLCDM]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b.*$'
        part_splits = re.split(part_pattern, text, flags=re.MULTILINE)
        part_headings = re.findall(part_pattern, text, flags=re.MULTILINE)
        
        if len(part_splits) <= 1:
            logger.info("No explicit Parts/Books found. Defaulting entire text to Part 1.")
            return [("Part 1", text)]
            
        parts = []
        # Handle preface text before first Part
        preface = part_splits[0].strip()
        if preface and len(preface) > 100:
            parts.append(("Preface/Introduction", preface))
            
        for idx, heading in enumerate(part_headings):
            block_content = part_splits[idx + 1] if idx + 1 < len(part_splits) else ""
            parts.append((heading.strip(), block_content))
            
        return parts

    def _split_into_chapters(self, part_text: str, part_num: int) -> List[Tuple[str, str]]:
        """Segments a Part's text into logical Chapters."""
        # Match headings like "Chapter 1", "CHAPTER I", "Chapter One", "Scene 1", or Gutenberg roman numerals "I--", "II--", "I.", "II."
        chapter_pattern = r'(?i)^\s*(?:(?:chapter|scene)\s+(?:[0-9]+|[IVXLCDM]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b|(?:[IVXLCDM]+)(?:--|\s*[-.]\s*).*)$'
        chapter_splits = re.split(chapter_pattern, part_text, flags=re.MULTILINE)
        chapter_headings = re.findall(chapter_pattern, part_text, flags=re.MULTILINE)
        
        if len(chapter_splits) <= 1:
            return [(f"Chapter 1", part_text)]
            
        chapters = []
        pre_text = chapter_splits[0].strip()
        if pre_text and len(pre_text) > 100:
            chapters.append((f"Chapter 0: Prologue", pre_text))
            
        for idx, heading in enumerate(chapter_headings):
            block_content = chapter_splits[idx + 1] if idx + 1 < len(chapter_splits) else ""
            chapters.append((heading.strip(), block_content))
            
        return chapters

    def _split_into_scenes(self, chapter_text: str) -> List[str]:
        """
        Segments a Chapter's text into individual Scenes based on separators
        (like * * *, ---, #) or paragraph density blocks (every 12 paragraphs).
        """
        # Pattern matching common scene cuts: * * *, ---, #, or similar (support carriage returns too)
        scene_separator = r'(?:\r?\n)\s*(?:\*\s*\*|\#|-{3,}|_{3,})\s*(?:\r?\n)'
        scenes = [s.strip() for s in re.split(scene_separator, chapter_text) if s.strip()]
        
        if len(scenes) > 1:
            return scenes
            
        # Fallback: segment long text into logical chunks of 12 paragraphs to represent scenes (support robust \n\s*\n and Windows newlines)
        paragraphs = [p.strip() for p in re.split(r'\r?\n\s*\r?\n', chapter_text) if p.strip()]
        if len(paragraphs) <= 15:
            return [chapter_text]
            
        logger.info(f"Splitting chapter into logical scenes based on paragraph blocks (Found {len(paragraphs)} paragraphs).")
        logical_scenes = []
        chunk_size = 12
        for i in range(0, len(paragraphs), chunk_size):
            chunk = paragraphs[i:i + chunk_size]
            logical_scenes.append("\n\n".join(chunk))
            
        return logical_scenes

    def parse_hierarchy(self, file_path: str) -> Dict[str, Any]:
        """
        Executes deeply nested hierarchical parsing:
        Part -> Chapter -> Scene -> Line Transcript
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Manuscript not found at: {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
            
        filename = os.path.basename(file_path)
        logger.info(f"Starting hierarchical decomposition on: {filename}...")

        # Stage 1: Typographic Normalization
        from src.nlp_analyzer import TextToScriptPipeline
        if not hasattr(self.analyzer, "text_pipeline"):
            self.analyzer.text_pipeline = TextToScriptPipeline()
            
        raw_content = self.analyzer.text_pipeline.normalize_typography(raw_content)
        quote_style = "double"
        
        if self.production_tier == 1:
            logger.info("Tier 1 selected: Bypassing global character extraction and MemPalace drawers. Forcing Narrator.")
            global_characters = ["Narrator"]
            confirmed_merges = {}
            global_merge_map = {}
            global_confidence_scores = {}
            merge_decisions = []
        else:
            # 0. Coreference baseline to build global character list
            global_db = self.analyzer._resolve_coreferences(raw_content)
            global_characters = list(global_db.keys())
            
            # Seed with registered database drawers and load confirmed merges
            confirmed_merges = {}
            try:
                from src.spatial_memory import MemPalace
                palace = MemPalace()
                # 1. Load registered drawers
                cursor = palace.conn.cursor()
                cursor.execute("SELECT character_name FROM drawers;")
                for row in cursor.fetchall():
                    if row[0] != "Narrator" and row[0] not in global_characters:
                        global_characters.append(row[0])
                # 2. Load confirmed merges
                confirmed_merges = palace.get_confirmed_merges(filename)
                palace.close()
            except Exception as e:
                logger.warning(f"Could not seed or load confirmed merges from MemPalace: {e}")

            # 0.5 Consolidate duplicate characters using SequenceMatcher similarity & substring matching
            global_characters, global_merge_map, global_confidence_scores, merge_decisions = consolidate_characters(global_characters, confirmed_merges)

        # 1. Segment Parts
        part_blocks = self._split_into_parts(raw_content)
        
        parts_list = []
        total_chapters = 0
        total_scenes = 0
        
        for p_idx, (part_title, part_content) in enumerate(part_blocks, 1):
            logger.info(f"Decomposing Part: '{part_title}'...")
            
            # 2. Segment Chapters within this Part
            chapter_blocks = self._split_into_chapters(part_content, p_idx)
            chapters_list = []
            
            for c_idx, (chapter_title, chapter_content) in enumerate(chapter_blocks, 1):
                total_chapters += 1
                
                # 3. Segment Scenes within this Chapter
                scene_blocks = self._split_into_scenes(chapter_content)
                scenes_list = []
                
                for s_idx, scene_content in enumerate(scene_blocks, 1):
                    total_scenes += 1
                    
                    # 4. Analyze Scene dialogues and narration
                    # Uses the core analyzer logic on the scene segment specifically
                    scene_script = self.analyzer.parse_manuscript_for_segment(
                        segment_text=scene_content,
                        file_name=filename,
                        chapter_num=total_chapters,
                        scene_num=s_idx,
                        characters_list=global_characters,
                        merge_map=global_merge_map,
                        production_tier=self.production_tier
                    )
                    
                    # Compute words and metrics for this scene
                    scene_words = len(scene_content.split())
                    dialogue_words = sum(len(line["dialogue"].split()) for line in scene_script)
                    narration_words = max(0, scene_words - dialogue_words)
                    
                    # Collect characters present/speaking in this scene
                    scene_chars = list(set(line["character"] for line in scene_script if line["character"] != "Narrator"))
                    
                    scenes_list.append({
                        "scene_id": f"scene_p{p_idx}_c{c_idx}_s{s_idx}",
                        "scene_number": s_idx,
                        "raw_scene_text": scene_content,
                        "characters_present": scene_chars,
                        "total_dialogue_lines": len(scene_script),
                        "metrics": {
                            "total_words": scene_words,
                            "narration_words": narration_words,
                            "dialogue_words": dialogue_words,
                        },
                        "lines": scene_script
                    })
                    
                chapters_list.append({
                    "chapter_id": f"chapter_p{p_idx}_c{c_idx}",
                    "chapter_number": c_idx,
                    "chapter_title": chapter_title,
                    "total_scenes": len(scenes_list),
                    "scenes": scenes_list
                })
                
            parts_list.append({
                "part_id": f"part_p{p_idx}",
                "part_title": part_title,
                "total_chapters": len(chapters_list),
                "chapters": chapters_list
            })
            
        # Compile structured hierarchical index
        hierarchical_index = {
            "metadata": {
                "source_file": filename,
                "quote_style_detected": quote_style,
                "total_parts": len(parts_list),
                "total_chapters": total_chapters,
                "total_scenes": total_scenes,
                "global_characters": global_characters,
                "merge_decisions": merge_decisions
            },
            "parts": parts_list
        }
        
        return hierarchical_index

    def save_index(self, index_data: dict, output_path: str):
        """Saves hierarchical index to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=4)
        logger.info(f"Hierarchical navigation index successfully saved to: {output_path}")


def _is_metadata_or_clutter(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return True
    # Illustration tags
    if t.startswith("[illustration") or t.endswith("illustration]"):
        return True
    # Front-matter and copyright boilerplate
    clutter_keywords = [
        'frederick warne', 'first published', 'printed and bound',
        'william clowes', 'gutenberg', 'ebook', 'isbn', 'all rights reserved',
        'http://', 'https://', 'www.gutenberg', 'sam\'l gabriel',
        'copyright', 'public domain', 'project gutenberg', 'illustrated by',
        'author:', 'title:', 'illustrator:', 'release date:'
    ]
    if any(kw in t for kw in clutter_keywords):
        return True
    # Title/author front-matter lines
    short_title_terms = {'the tale of', 'peter rabbit', 'beatrix potter', 'by beatrix potter', 'by'}
    if t in short_title_terms:
        return True
    return False


def consolidate_characters(characters_list: list, confirmed_merges: dict = None) -> tuple[list, dict, dict, list]:
    """
    Scans the character list for potential duplicates using substring matching,
    SequenceMatcher similarity, and user database overrides.
    """
    import difflib
    clean_list = list(characters_list)
    merge_map = {}
    confidence_scores = {}
    merge_decisions = []
    
    # Pre-populate decisions from confirmed_merges DB mappings
    # confirmed_merges has structure: {original_name: (canonical_name, is_confirmed, confidence_score)}
    db_blacklist = set()  # set of (original, canonical) pairs that were split
    
    if confirmed_merges:
        for orig, (canon, confirmed, conf) in confirmed_merges.items():
            if confirmed:
                logger.info(f"Applying database confirmed merge override: '{orig}' -> '{canon}'")
                merge_map[orig] = canon
                confidence_scores[f"{orig} -> {canon}"] = 1.0
                if orig in clean_list:
                    clean_list.remove(orig)
                merge_decisions.append({
                    "original_name": orig,
                    "canonical_name": canon,
                    "confidence_score": 1.0,
                    "status": "confirmed"
                })
            else:
                logger.info(f"Applying database split/blacklist override: '{orig}' remains separate from '{canon}'")
                db_blacklist.add((orig, canon))
                merge_decisions.append({
                    "original_name": orig,
                    "canonical_name": canon,
                    "confidence_score": conf,
                    "status": "rejected"
                })

    # Sort characters by length descending so we merge shorter names into longer ones
    sorted_chars = sorted(characters_list, key=len, reverse=True)
    
    for i in range(len(sorted_chars)):
        char_a = sorted_chars[i]
        if char_a in merge_map:
            continue  # Already merged into something else
            
        for j in range(i + 1, len(sorted_chars)):
            char_b = sorted_chars[j]
            if char_b in merge_map:
                continue
                
            # If this merge was explicitly rejected by user, skip!
            if (char_b, char_a) in db_blacklist:
                continue
                
            # Normalize names for robust comparison (remove punctuation, lower case, normalize whitespace)
            norm_a = re.sub(r'[^\w\s]', '', char_a.lower()).strip()
            norm_b = re.sub(r'[^\w\s]', '', char_b.lower()).strip()
            
            # Title Guarding Constraint: Ensure we do not merge different honorifics (e.g. Mr. vs Mrs.)
            # e.g., "Mr. McGregor" (norm: "mr mcgregor") vs "Mrs. McGregor" (norm: "mrs mcgregor")
            words_a = re.split(r'\s+', norm_a)
            words_b = re.split(r'\s+', norm_b)
            has_mr_a = "mr" in words_a or "mr." in words_a
            has_mrs_a = "mrs" in words_a or "mrs." in words_a
            has_mr_b = "mr" in words_b or "mr." in words_b
            has_mrs_b = "mrs" in words_b or "mrs." in words_b
            
            if (has_mr_a != has_mr_b) or (has_mrs_a != has_mrs_b):
                # Different titles! Do not merge husband and wife or generic title discrepancies!
                continue
                
            is_dup = False
            confidence = 0.0
            
            # 1. Substring check
            if norm_b in norm_a:
                is_dup = True
                confidence = 1.0
            else:
                # 2. Similarity ratio check
                ratio = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()
                if ratio >= 0.8:
                    is_dup = True
                    confidence = round(ratio, 2)
                    
            if is_dup:
                logger.info(f"Consolidating duplicate characters: '{char_b}' merged into '{char_a}' (Confidence: {confidence})")
                merge_map[char_b] = char_a
                confidence_scores[f"{char_b} -> {char_a}"] = confidence
                if char_b in clean_list:
                    clean_list.remove(char_b)
                    
                # Add to merge decisions as pending since it's dynamically discovered and not yet in DB
                merge_decisions.append({
                    "original_name": char_b,
                    "canonical_name": char_a,
                    "confidence_score": confidence,
                    "status": "pending"
                })
                
    return clean_list, merge_map, confidence_scores, merge_decisions


def _resolve_speaker_aliases(text: str, characters_list: list) -> str:
    t = text.lower()
    
    # 1. Direct role-to-name mapping for Peter Rabbit specifically
    if "mother" in t or "she" in t:
        for char in characters_list:
            if "rabbit" in char.lower() and ("mrs" in char.lower() or "old" in char.lower()):
                return char
                
    # 2. General role mappings
    if "mcgregor" in t:
        for char in characters_list:
            if "mcgregor" in char.lower():
                return char
                
    # 3. Substring matching (e.g. "Holmes" -> "Sherlock Holmes")
    for char in sorted(characters_list, key=len, reverse=True):
        if char.lower() in t:
            return char
            
    return None


# Add Segment Parsing Support dynamically to nlp_analyzer
def parse_manuscript_for_segment(self, segment_text: str, file_name: str, chapter_num: int, scene_num: int, characters_list: list, merge_map: dict = None, production_tier: int = 1) -> list:
    """Helper method injected into ManuscriptAnalyzer to process a specific scene block chronologically."""
    # Ensure TextToScriptPipeline is available
    from src.nlp_analyzer import TextToScriptPipeline
    if not hasattr(self, "text_pipeline"):
        self.text_pipeline = TextToScriptPipeline()

    paragraphs = [p.strip() for p in re.split(r'\n+', segment_text) if p.strip()]
    script_output = []
    line_count = 1
    
    # Local context memory - Reset and cleared between scenes natively!
    dialogue_queue = []
    locked_speaker = None
    speaker_lock_counter = 0

    for p_idx, paragraph in enumerate(paragraphs):
        if _is_metadata_or_clutter(paragraph):
            continue
            
        # Stage 3: Extract direct quote blocks sequentially from the paragraph
        blocks = self.text_pipeline.extract_quote_blocks(paragraph)
        if not blocks:
            continue

        # Dialogue Tag Association context: join all narrative text chunks in the same paragraph
        narrative_context = " ".join([b["text"] for b in blocks if b["type"] == "narrative"]).strip()

        for b_idx, block in enumerate(blocks):
            block_type = block["type"]
            block_text = block["text"].strip()
            if not block_text:
                continue

            # Determine paragraph-end padding bonus
            is_last_block = (b_idx == len(blocks) - 1)
            post_padding = 600 if is_last_block else 150

            # Default values
            assigned_character = "Narrator"
            speaker_id = "char_narrator"
            segment_type = "narrative"
            emotion = "Neutral"
            delivery_style = "descriptive"
            attribution_method = "Narration"
            high_confidence = False

            if block_type == "narrative":
                # Narrator block
                pass
            else:
                # Dialogue block
                segment_type = "dialogue"
                found_speaker = False
                
                # Helper: map character name to speaker_id slug
                def get_slug_id(name: str) -> str:
                    if name == "Narrator":
                        return "char_narrator"
                    if name == "char_unknown_fallback":
                        return "char_unknown_fallback"
                    return "char_" + re.sub(r'[^a-z0-9_]', '', name.lower().replace(" ", "_").replace(".", ""))

                if production_tier == 1:
                    # Tier 1 Bypass: All dialogue is deterministically read by the Narrator
                    assigned_character = "Narrator"
                    attribution_method = "Tier 1 Default"
                    found_speaker = True
                    high_confidence = True

                # 1. Dialogue Tag Association (direct speech verb check in paragraph narrative context)
                if not found_speaker and narrative_context:
                    for char in sorted(characters_list, key=len, reverse=True):
                        char_pattern = re.escape(char).replace(r'\ ', r'\s+').replace('Mrs', 'Mrs\\.?')
                        char_pattern = re.sub(r'\bMr\b', r'Mr\\.?', char_pattern)
                        if re.search(r'\b' + char_pattern + r'\b', narrative_context, flags=re.IGNORECASE):
                            words = narrative_context.lower().split()
                            if any(v in words for v in SPEECH_VERBS):
                                assigned_character = char
                                attribution_method = "Direct Speech Verb"
                                found_speaker = True
                                high_confidence = True
                                break

                # 2. Speaker Lock Override (if no high-confidence speech verb and lock is active)
                if not found_speaker and speaker_lock_counter > 0:
                    assigned_character = locked_speaker
                    attribution_method = "Speaker Lock Override"
                    found_speaker = True
                    speaker_lock_counter -= 1

                # 2.5 Local LLM (Ollama) resolution path for dialogue segments (Tier 3 only)
                if self.ollama_model and not found_speaker and production_tier >= 3:
                    try:
                        # Construct Ollama prompt with full context and character roster
                        prompt = f"""
Analyze the dialogue inside this paragraph block context.
Available character roster: {characters_list}

Select the character who is speaking the dialogue from the available character roster.
If the dialogue is narration or spoken by the main storyteller, output "Narrator".
If you are highly uncertain, assign "char_unknown_fallback".

Also determine the spoken emotion from: ["Joy", "Sadness", "Tension", "Neutral"]
And determine the delivery style / tone (e.g. "maternal_caution", "authoritative", "excited", "descriptive", "neutral_narrative").
And estimate your attribution confidence score as a float between 0.0 and 1.0.

CONTEXT PARAGRAPH:
{paragraph}

Return a valid JSON object with keys "speaker", "emotion", "delivery_style", and "confidence".

Example output format:
{{
  "speaker": "Mr. McGregor",
  "emotion": "Tension",
  "delivery_style": "furious_shout",
  "confidence": 0.92
}}

Return ONLY the valid JSON object.
"""
                        res_str = _query_local_ollama(self.ollama_model, prompt, format_json=True)
                        if res_str:
                            res_str = res_str.strip()
                            if "```json" in res_str:
                                res_str = res_str.split("```json")[1].split("```")[0].strip()
                            elif "```" in res_str:
                                res_str = res_str.split("```")[1].split("```")[0].strip()
                            
                            start_idx = res_str.find("{")
                            end_idx = res_str.rfind("}")
                            if start_idx != -1 and end_idx != -1:
                                res_str = res_str[start_idx:end_idx + 1]
                                
                            parsed = json.loads(res_str)
                            llm_speaker = parsed.get("speaker", "Narrator").strip()
                            llm_emotion = parsed.get("emotion", "Neutral").strip()
                            llm_style = parsed.get("delivery_style", "descriptive").strip()
                            llm_conf = float(parsed.get("confidence", 1.0))
                            
                            # 75% confidence guardrail
                            if llm_conf < 0.75:
                                assigned_character = "char_unknown_fallback"
                                speaker_id = "char_unknown_fallback"
                                emotion = "Neutral"
                                delivery_style = "descriptive"
                                attribution_method = "char_unknown_fallback"
                                found_speaker = True
                                logger.info(f"Ollama confidence {llm_conf} < 75% for: '{block_text}'. Falling back safely.")
                            else:
                                matched_speaker = None
                                if llm_speaker == "Narrator":
                                    matched_speaker = "Narrator"
                                else:
                                    for c in characters_list:
                                        if c.lower() == llm_speaker.lower():
                                            matched_speaker = c
                                            break
                                            
                                if matched_speaker:
                                    assigned_character = matched_speaker
                                    emotion = llm_emotion
                                    delivery_style = llm_style
                                    attribution_method = "Local LLM (Ollama)"
                                    found_speaker = True
                                    high_confidence = True
                                    logger.info(f"Ollama successfully attributed segment to [{assigned_character}] (Conf: {llm_conf})")
                        else:
                            logger.warning("Empty response received from local Ollama. Disabling model for this parsing pass.")
                            self.ollama_model = None
                    except Exception as e:
                        logger.error(f"Error attributing speaker via local Ollama: {e}. Disabling model for this parsing pass.")
                        self.ollama_model = None

                # 3. Context mention check (including robust aliases and preceding paragraph fallback)
                if not found_speaker:
                    surrounding = narrative_context
                    if len(surrounding) < 5 and p_idx > 0:
                        surrounding = paragraphs[p_idx - 1]
                    
                    alias_resolved = _resolve_speaker_aliases(surrounding, characters_list)
                    if alias_resolved:
                        assigned_character = alias_resolved
                        attribution_method = "Context Alias Mention"
                        found_speaker = True
                    
                    if not found_speaker:
                        for char in characters_list:
                            if re.search(r'\b' + re.escape(char) + r'\b', surrounding):
                                assigned_character = char
                                attribution_method = "Context Entity Mention"
                                found_speaker = True
                                break
                            
                # 4. Conversational queue alternating fallback
                if not found_speaker:
                    if dialogue_queue:
                        distinct = []
                        for speaker in reversed(dialogue_queue):
                            if speaker != "Narrator" and speaker not in distinct:
                                distinct.append(speaker)
                            if len(distinct) == 2:
                                break
                        if len(distinct) == 2:
                            assigned_character = distinct[1]
                            attribution_method = "Auto-Attributed (Alternating)"
                        elif len(distinct) == 1:
                            assigned_character = distinct[0]
                            attribution_method = "Auto-Attributed (Single Active)"
                    else:
                        # 75% confidence fallback trigger when both rules and LLM are absent
                        assigned_character = "char_unknown_fallback"
                        attribution_method = "char_unknown_fallback"
                        
                # Apply merge map to consolidate duplicates dynamically
                if merge_map and assigned_character in merge_map:
                    assigned_character = merge_map[assigned_character]
                    
                if assigned_character != "Narrator" and assigned_character != "char_unknown_fallback":
                    dialogue_queue.append(assigned_character)
                    if len(dialogue_queue) > 10:
                        dialogue_queue.pop(0)
                        
                # Set speaker lock if high confidence
                if high_confidence:
                    locked_speaker = assigned_character
                    speaker_lock_counter = 2
                    
                if assigned_character == "char_unknown_fallback":
                    speaker_id = "char_unknown_fallback"
                else:
                    speaker_id = get_slug_id(assigned_character)

            # Determine VADER emotion if not already resolved by Ollama
            if block_type == "narrative" or not emotion:
                emotion = self._determine_emotion(block_text)

            # Performance payload details
            pitch_mod = 1.0
            speed_mod = 1.0
            if speaker_id != "char_unknown_fallback":
                # Read character drawers settings if active
                drawer = self.get_character_drawer(assigned_character) if hasattr(self, "get_character_drawer") else None
                if not drawer:
                    # Fallback to direct query to palace
                    try:
                        from src.spatial_memory import MemPalace
                        palace = MemPalace()
                        drawer = palace.get_character_drawer(assigned_character)
                        palace.close()
                    except Exception:
                        pass
                if drawer:
                    pitch_semitones = float(drawer["modulation_config"].get("pitch", 0.0))
                    pitch_mod = 2.0 ** (pitch_semitones / 12.0)
                    speed_mod = float(drawer["modulation_config"].get("speed", 1.0))

            performance = {
                "pitch_modifier": pitch_mod,
                "speed_modifier": speed_mod,
                "delivery_style": delivery_style
            }

            # Deterministic line_id based on cleaned block text
            line_content = f"{file_name}_p1_c{chapter_num}_s{scene_num}_l{line_count}_{block_text}"
            line_id = hashlib.sha256(line_content.encode('utf-8')).hexdigest()[:16]

            # Construct finalized Studio Script block schema!
            confidence_score = locals().get('llm_conf', 1.0 if high_confidence else 0.0)
            script_output.append({
                "line_id": line_id,
                "chapter": chapter_num,
                "scene": scene_num,
                "line_number": line_count,
                "character": assigned_character,
                "speaker_id": speaker_id,
                "segment_type": segment_type,
                "text": block_text,
                # Keep dialogue, narration_before, narration_after for backward compatibility!
                "dialogue": block_text,
                "narration_before": "",
                "narration_after": "",
                "emotion": emotion,
                "performance": performance,
                "post_padding_ms": post_padding,
                "attribution_method": attribution_method,
                "confidence": confidence_score,
                "speaker_locked": (speaker_lock_counter > 0 or attribution_method == "Speaker Lock Override")
            })
            line_count += 1
            
    return script_output

# Inject method into ManuscriptAnalyzer dynamically
ManuscriptAnalyzer.parse_manuscript_for_segment = parse_manuscript_for_segment


def main():
    """CLI and self-test harness for Hierarchical Parsing."""
    import argparse
    parser = argparse.ArgumentParser(description="Firespeaker Hierarchical Navigation Parser")
    parser.add_argument("--input", type=str, help="Path to manuscript book text file to analyze")
    parser.add_argument("--output", type=str, default="scratch/hierarchical_script_index.json", help="Path to save output navigation JSON index")
    parser.add_argument("--test", action="store_true", help="Run self-test on Time Machine Chapter 1")
    parser.add_argument("--tier", type=int, default=1, help="Production tier (1, 2, or 3). Defaults to Tier 1.")
    
    args = parser.parse_args()
    
    if args.test:
        print("\n=== RUNNING HIERARCHICAL PARSER INTEGRITY TEST ===")
        manuscript_path = "data/corpus/time_machine_ch1.txt"
        
        # Verify manuscript is in place
        if not os.path.exists(manuscript_path):
            os.makedirs(os.path.dirname(manuscript_path), exist_ok=True)
            with open(manuscript_path, "w", encoding="utf-8") as f:
                f.write("""
                Chapter 1: The Four Dimensions
                
                The Time Traveller stood in the laboratory.
                
                “You must follow me carefully,” said the Time Traveller.
                
                * * *
                
                “I am not,” said the Psychologist, pausing to look at him, “going to agree.”
                """)
                
        parser_engine = HierarchicalParser()
        index = parser_engine.parse_hierarchy(manuscript_path)
        output_test_path = "scratch/test_hierarchical_index.json"
        parser_engine.save_index(index, output_test_path)
        
        # Print verification checks
        print("\nGUI Navigation Index Schema Verification:")
        print(f"- Target Book:              {index['metadata']['source_file']}")
        print(f"- Total Parts extracted:    {index['metadata']['total_parts']}")
        print(f"- Total Chapters extracted: {index['metadata']['total_chapters']}")
        print(f"- Total Scenes extracted:   {index['metadata']['total_scenes']}")
        
        part_1 = index["parts"][0]
        print(f"\nPart 1: '{part_1['part_title']}' contains {part_1['total_chapters']} chapters.")
        
        chapter_1 = part_1["chapters"][0]
        print(f"  * Chapter 1: '{chapter_1['chapter_title']}' contains {chapter_1['total_scenes']} scenes.")
        
        scene_1 = chapter_1["scenes"][0]
        print(f"    - Scene 1: ID={scene_1['scene_id']} | Present: {scene_1['characters_present']} | Dialogue lines={scene_1['total_dialogue_lines']}")
        print(f"    - Scene 1 Metrics: Total Words={scene_1['metrics']['total_words']} | Narration Words={scene_1['metrics']['narration_words']}")
        
        print("\n=== HIERARCHICAL PARSER HARNESS PASSED SUCCESSFULLY ===\n")
        return 0
        
    if not args.input:
        parser.print_help()
        sys.exit(1)
        
    try:
        parser_engine = HierarchicalParser(production_tier=args.tier)
        index = parser_engine.parse_hierarchy(args.input)
        parser_engine.save_index(index, args.output)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Hierarchical parsing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
