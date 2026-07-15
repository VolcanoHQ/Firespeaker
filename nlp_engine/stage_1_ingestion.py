#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Firespeaker Ingestion Stage 1: Clutter Scrubber
Heuristically and semantically removes Gutenberg front-matter, title pages,
publisher info, and credits using Narrative Density and Local LLM fallback.
"""

import re
import json
import logging
import urllib.request
import urllib.error
from typing import Optional, List

logger = logging.getLogger("ClutterScrubber")

class ClutterScrubber:
    """Removes non-narrative front-matter, credits, and metadata from manuscripts."""

    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate", model_name: str = "llama3:8b-instruct-q8_0"):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.quote_normalization_map = {
            "“": '"', "”": '"',
            "‘": "'", "’": "'",
            "‹": "'", "›": "'"
        }

    def normalize_typography(self, text: str) -> str:
        """Standardizes typography to straight quotes and normalized line endings."""
        if not text:
            return ""
        cleaned = text
        for smart, straight in self.quote_normalization_map.items():
            cleaned = cleaned.replace(smart, straight)
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        return cleaned

    def remove_front_matter(self, raw_text: str) -> str:
        """
        Standardizes text and removes publisher credits, Gutenberg tags,
        and intro pages using the Density Heuristic and Local LLM fallback.
        """
        if not raw_text:
            return ""

        # 1. Normalize typography first
        normalized = self.normalize_typography(raw_text)

        # 2a. Chop trailing Gutenberg license/back matter. Both marker phrasings are
        # unmistakable, so no positional guard is needed: the modern
        # "*** END OF THE/THIS PROJECT GUTENBERG EBOOK ... ***" and the older bare
        # "End of the Project Gutenberg EBook of ..." line. A single narrator would
        # otherwise read the entire multi-page license aloud.
        end_match = re.search(
            r'^\s*(?:\*\*\*\s*)?END OF (?:THE\s+|THIS\s+)?PROJECT GUTENBERG.*$',
            normalized, re.IGNORECASE | re.MULTILINE)
        if end_match:
            normalized = normalized[:end_match.start()].rstrip()

        # 2b. Chop initial legal header at the START marker ("THE" or "THIS" both occur)
        start_match = re.search(r'\*\*\*\s*START OF (?:THE\s+|THIS\s+)?PROJECT GUTENBERG (?:EBOOK|E-BOOK)?.*?\*\*\*', normalized, re.IGNORECASE)
        if start_match:
            text_to_scrub = normalized[start_match.end():].lstrip()
        else:
            text_to_scrub = normalized.lstrip()

        # 2c. Old-format transcriber credit sits AFTER the START marker ("Produced by
        # ...", "E-text prepared by ..."). Drop that leading block through its blank line.
        credit_match = re.match(
            r'(?:produced by|e-text prepared|transcribed from|updated by)[^\n]*(?:\n[^\n]+)*',
            text_to_scrub, re.IGNORECASE)
        if credit_match:
            text_to_scrub = text_to_scrub[credit_match.end():].lstrip()

        # Split text into lines for parsing
        lines = text_to_scrub.splitlines()
        first_50_lines = "\n".join(lines[:50])

        # 3. Heading guard: if a structural heading sits within the first few
        # non-empty lines, the text after the marker is already clean story --
        # cut at the heading and DO NOT run the fuzzy detectors below (measured:
        # they ate Wuthering Heights' opening paragraphs and Frankenstein's
        # Letters 1-2 when the local LLM "found" a sentence from deep inside).
        heading_re = re.compile(
            r'^\s*(?:CHAPTER|CHAPITRE|LETTER|PART|BOOK|VOLUME|ACT|PROLOGUE)\b[\s.:]*(?:[IVXLCDM]+|[0-9]+|ONE|TWO|THREE|FIRST|SECOND|THIRD)?\.?\s*$'
            r'|^\s*[IVXLCDM]+\.?--.+$',
            re.IGNORECASE)
        seen_nonempty = 0
        for idx, line in enumerate(lines):
            if not line.strip():
                continue
            seen_nonempty += 1
            if heading_re.match(line):
                logger.info(f"Heading guard: structural heading {line.strip()!r} within front lines; cutting there deterministically.")
                return "\n".join(lines[idx:]).strip()
            if seen_nonempty >= 15:
                break

        # 4. Heuristic Fallback: Density Heuristic (deterministic before LLM)
        heuristic_idx = self._run_density_heuristic(lines[:120])
        if heuristic_idx is not None:
            logger.info(f"Density Heuristic triggered. Splicing text starting at line {heuristic_idx}.")
            return "\n".join(lines[heuristic_idx:]).strip()

        # 5. Last resort: Local LLM Boundary Detector. Bounded: the boundary may
        # only discard a short leading block -- a match deep in the text means the
        # model quoted story prose, not the story's first sentence.
        llm_first_sentence = self._query_llm_for_start(first_50_lines)
        if llm_first_sentence:
            clean_sentence = llm_first_sentence.strip('"\' ')
            matched_pos = self._find_sentence_position(text_to_scrub, clean_sentence)
            if matched_pos is not None and matched_pos <= 2500:
                logger.info(f"LLM successfully matched start boundary sentence: '{clean_sentence[:40]}...'")
                return text_to_scrub[matched_pos:].strip()
            if matched_pos is not None:
                logger.warning(f"LLM boundary at char {matched_pos} discards too much text; ignoring it.")

        # 6. Default Fallback if all fail: Return text_to_scrub untouched
        logger.warning("Heading guard, Density Heuristic, and LLM Detector all failed. Returning standard Gutenberg-scrubbed text.")
        return text_to_scrub

    def _run_density_heuristic(self, lines: List[str]) -> Optional[int]:
        """
        Density Heuristic: Identifies where metadata ends and narrative text begins.
        Scans lines sequentially for a narrative density spike.
        """
        metadata_keywords = {
            "title:", "author:", "illustrator:", "release date:", "translator:",
            "credits:", "copyright", "isbn", "new york", "published by", 
            "online distributed", "proofreading", "illustration", "ebook",
            "printed", "bound in", "gutenberg", "edition", "all rights reserved",
            "published", "publisher", "london", "sam'l gabriel", "credits",
            # transcriber-credit fragments: "online distributed" alone misses when the
            # credit wraps mid-phrase across lines (measured: Franklin)
            "produced by", "pgdp", "www.", "http", "proofread"
        }
        
        for idx, line in enumerate(lines):
            clean_line = line.strip()
            if not clean_line:
                continue
            
            # If line is too short, skip it
            if len(clean_line) < 50:
                continue
                
            # If line contains metadata keywords, skip it
            if any(k in clean_line.lower() for k in metadata_keywords):
                continue
                
            # A line ending mid-clause on a comma is list/title-page matter (an
            # "Author of ..." works list), never a story's opening sentence; and a
            # line quoting 3+ separate titles is that same list wrapped mid-phrase.
            if clean_line[-1] == "," or clean_line[-2:] in {',"', ",'"}:
                continue
            if clean_line.count('"') >= 3:
                continue

            # Check if it starts with capital and ends with punctuation or letter
            starts_capital = clean_line[0].isupper() or clean_line[0] in {"'", '"'}
            ends_valid = clean_line[-1] in {".", "?", "!", '"', "'", "—", ";"} or clean_line[-1].isalnum()
            
            if starts_capital and ends_valid:
                # Common narrative starting words
                common_starters = {"once", "in", "the", "it", "a", "he", "she", "they", "on", "when", "there", "long", "yesterday", "you"}
                first_word = clean_line.split()[0].lower().strip("“'\"")
                
                is_story_start = False
                if first_word in common_starters:
                    is_story_start = True
                else:
                    # Look at next non-empty line length or case
                    next_idx = idx + 1
                    while next_idx < len(lines) and not lines[next_idx].strip():
                        next_idx += 1
                    if next_idx < len(lines):
                        next_line = lines[next_idx].strip()
                        if len(next_line) > 50 or (next_line and next_line[0].islower()):
                            is_story_start = True
                
                if is_story_start:
                    # Check if previous non-empty line is a chapter heading
                    prev_idx = idx - 1
                    while prev_idx >= 0 and not lines[prev_idx].strip():
                        prev_idx -= 1
                    if prev_idx >= 0:
                        prev_line = lines[prev_idx].strip()
                        if len(prev_line) < 50 and (prev_line.isupper() or re.match(r'^(?:CHAPTER|CHAPITRE|PART|BOOK|VOLUME|LETTER|ACT|SCENE)\b', prev_line, re.IGNORECASE)):
                            return prev_idx
                    return idx
                    
        return None

    def _get_available_ollama_models(self) -> List[str]:
        """Queries local Ollama to find installed models."""
        try:
            req = urllib.request.urlopen(f"http://localhost:11434/api/tags", timeout=2.0)
            if req.status == 200:
                data = json.loads(req.read().decode("utf-8"))
                models = [m["name"] for m in data.get("models", [])]
                return models
        except Exception:
            pass
        return []

    def _query_llm_for_start(self, text_snippet: str) -> Optional[str]:
        """Queries the local Ollama model to find the exact first sentence of the story."""
        prompt = f"""Read this book introduction. Reply ONLY with the exact first sentence of the actual story. Do not include introductory text, explanations, or quotes around it.

BOOK SNIPPET:
{text_snippet}
"""
        models_to_try = [self.model_name]
        
        # Intercept installed local models dynamically
        available = self._get_available_ollama_models()
        for m in available:
            if m not in models_to_try:
                models_to_try.append(m)
                
        # Static standard fallback list
        for m in ["llama3:8b-instruct-q8_0", "llama3", "phi3", "gemma2", "qwen2.5-coder:3b"]:
            if m not in models_to_try:
                models_to_try.append(m)
                
        for model in models_to_try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 100
                }
            }
            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    self.ollama_url,
                    data=data,
                    headers={"Content-Type": "application/json"}
                )
                res = urllib.request.urlopen(req, timeout=5.0)
                if res.status == 200:
                    response_obj = json.loads(res.read().decode("utf-8"))
                    sentence = response_obj.get("response", "").strip()
                    if sentence:
                        sentence = re.sub(r'^["\'`\s]+|["\'`\s]+$', '', sentence).strip()
                        if len(sentence) > 10:
                            return sentence
            except Exception as e:
                logger.debug(f"LLM boundary query failed for model {model}: {e}")
                
        return None

    def _find_sentence_position(self, text: str, sentence: str) -> Optional[int]:
        """Locates the position of a sentence in the text using flexible match rules."""
        norm_text = re.sub(r'\s+', ' ', text)
        norm_sentence = re.sub(r'\s+', ' ', sentence)

        pos = norm_text.find(norm_sentence)
        if pos != -1:
            words = norm_sentence.split()[:4]
            if not words:
                return None
            search_pat = r'\s+'.join(re.escape(w) for w in words)
            match = re.search(search_pat, text)
            if match:
                return match.start()

        # Let's try matching the first 30 characters
        short_sentence = norm_sentence[:35]
        words = short_sentence.split()
        if len(words) >= 3:
            search_pat = r'\s+'.join(re.escape(w) for w in words)
            match = re.search(search_pat, text)
            if match:
                return match.start()

        return None
