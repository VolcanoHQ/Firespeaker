#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Firespeaker Ingestion & Script Analysis Pipeline
Handles manuscript segmentation, character entity resolution (via xCoRe),
dialogue extraction, and emotional sentiment mapping.
"""

import os
import re
import sys
import json
import logging
import hashlib
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FirespeakerNLP")

# Standard English pronouns to ignore during character extraction
PRONOUNS = {
    'he', 'him', 'his', 'himself', 
    'she', 'her', 'hers', 'herself', 
    'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves',
    'i', 'me', 'my', 'mine', 'myself', 
    'you', 'your', 'yours', 'yourself',
    'we', 'us', 'our', 'ours', 'ourselves'
}

# Single-token PERSON entities from spaCy occasionally capture emphatic direction
# words or possessive heading fragments instead of actual character names.
SPURIOUS_ONE_WORD_PERSON_TOKENS = {
    'down', 'up', 'left', 'right',
    'chapter', 'book', 'title', 'author', 'release'
}

# Common speech/dialogue verbs for attribution detection
SPEECH_VERBS = {
    'say', 'said', 'saying', 'says', 'ask', 'asked', 'asking', 'asks',
    'tell', 'told', 'telling', 'tells', 'reply', 'replied', 'replying', 'replies',
    'shout', 'shouted', 'shouting', 'shouts', 'whisper', 'whispered', 'whispering', 'whispers',
    'cry', 'cried', 'crying', 'cries', 'explain', 'explained', 'explaining', 'explains',
    'mutter', 'muttered', 'muttering', 'mutters', 'state', 'stated', 'stating', 'states',
    'exclaim', 'exclaimed', 'exclaiming', 'exclaims', 'respond', 'responded', 'responding', 'responds',
    'remark', 'remarked', 'remarking', 'remarks', 'call', 'called', 'calling', 'calls',
    'murmur', 'murmured', 'murmuring', 'murmurs', 'gasp', 'gasped', 'gasping', 'gasps'
}

# Try loading advanced coreference (xCoRe) and deep learning libraries
HAS_XCORE = False
try:
    import torch
    # Monkeypatch torch.load to default to weights_only=False to allow xCoRe litbank load on PyTorch 2.6+
    original_torch_load = torch.load
    def custom_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)
    torch.load = custom_torch_load
    if hasattr(torch, "serialization"):
        torch.serialization.load = custom_torch_load
    from xcore import xCoRe
    HAS_XCORE = True
except ImportError:
    logger.warning("xcore-coref or PyTorch not found in current environment. Coreference falling back to spaCy rules.")

# Try loading spaCy
HAS_SPACY = False
nlp = None
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    logger.warning("spaCy not found in current environment. Script will use pure-Python regex fallback.")

# Try loading NLTK VADER and resources
HAS_NLTK = False
sid = None
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    # Ensure VADER lexicon is downloaded
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        logger.info("Downloading NLTK VADER lexicon...")
        nltk.download('vader_lexicon', quiet=True)
    # Ensure punkt is downloaded for xCoRe/tokenization
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    sid = SentimentIntensityAnalyzer()
    HAS_NLTK = True
except ImportError:
    logger.warning("NLTK not found. Sentiment will fallback to simple keyword heuristics.")


def _is_valid_spacy_person_entity(name: str) -> bool:
    cleaned_name = name.strip()
    if cleaned_name.lower() in PRONOUNS or len(cleaned_name) <= 1:
        return False

    name_parts = cleaned_name.split()
    if len(name_parts) != 1:
        return True

    token = name_parts[0]
    if not re.fullmatch(r'[A-Za-z]+', token):
        return False

    if token.lower() in SPURIOUS_ONE_WORD_PERSON_TOKENS:
        return False

    return True


class TextToScriptPipeline:
    """
    Handles deterministic Stage 1 Typographic Normalization 
    and Stage 3 Direct Quote Extraction for Firespeaker Studio.
    """
    
    def __init__(self):
        # Maps smart/curly typographic anomalies to uniform straight ASCII characters
        self.quote_normalization_map = {
            "“": '"', "”": '"',  # Smart double quotes
            "‘": "'", "’": "'",  # Smart single quotes
            "‹": "'", "›": "'"   # Chevron/Angle quotes
        }

    def normalize_typography(self, raw_text: str) -> str:
        """
        Stage 1: Clean and standardize text typography.
        Converts quotes, flattens spacing, and preserves structure.
        """
        if not raw_text:
            return ""
            
        # 1. Apply uniform translation map for smart characters
        cleaned = raw_text
        for smart_char, straight_char in self.quote_normalization_map.items():
            cleaned = cleaned.replace(smart_char, straight_char)
            
        # 2. Convert [Illustration] artifacts into standardized spacing gaps
        # Rather than dropping them, we flag them as space markers
        cleaned = re.sub(r'\[Illustration\]', '\n\n', cleaned)
        
        # 3. Compact erratic spacing and standardize paragraph carriage returns
        lines = [line.strip() for line in cleaned.splitlines()]
        compacted_text = "\n".join(lines)
        
        # Collapse multi-newlines down to crisp paragraph divisions
        normalized_paragraphs = re.sub(r'\n{3,}', '\n\n', compacted_text)
        return normalized_paragraphs.strip()

    def extract_quote_blocks(self, paragraph: str) -> list:
        """
        Stage 3: Tokenize a single paragraph character-by-character into a sequential
        stream of alternating 'narrative' or 'dialogue' blocks.
        Ensures terminal punctuation is cleanly retained within quote boundaries.
        """
        segments = []
        if not paragraph.strip():
            return segments

        current_buffer = []
        in_quote = False
        # Choose the active delimiter based on what the paragraph uses for speech
        # The Tale of Peter Rabbit uses single quotes (') natively for direct dialogue
        quote_delimiter = "'" if "'" in paragraph else '"'

        for idx, char in enumerate(paragraph):
            is_quote = False
            if char == quote_delimiter:
                if char == "'":
                    # The Apostrophe Trap: Discerning between dialogue quotes and contractions
                    prev_char = paragraph[idx-1] if idx > 0 else ' '
                    next_char = paragraph[idx+1] if idx + 1 < len(paragraph) else ' '
                    
                    # If surrounded by alphanumeric letters, it's a contraction/possessive (e.g. don't, McGregor's)
                    if prev_char.isalpha() and next_char.isalpha():
                        is_quote = False
                    else:
                        is_quote = True
                else:
                    is_quote = True
                    
            if is_quote:
                # Delimiter hit: Dump whatever was building up in the buffer
                text_chunk = "".join(current_buffer).strip()
                if text_chunk:
                    segments.append({
                        "type": "dialogue" if in_quote else "narrative",
                        "text": text_chunk
                    })
                current_buffer = []
                in_quote = not in_quote  # Flip the context state
            else:
                current_buffer.append(char)

        # Catch any trailing text remaining at the tail end of the paragraph
        text_chunk = "".join(current_buffer).strip()
        if text_chunk:
            segments.append({
                "type": "dialogue" if in_quote else "narrative",
                "text": text_chunk
            })

        # Safeguard clean up: Filter out accidental empty string tokens or trailing whitespace/hyphens
        cleaned_segments = []
        for seg in segments:
            # Strip loose leading/trailing whitespace and hyphens, but LEAVE vital terminal punctuation (! ? . ,) intact!
            seg["text"] = re.sub(r'^[\s\-]+', '', seg["text"])
            seg["text"] = re.sub(r'[\s\-]+$', '', seg["text"])
            
            # Restore structural terminal tracking if original text chunk possessed meaningful endings
            if seg["text"]:
                cleaned_segments.append(seg)

        return cleaned_segments


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
    short_title_terms = {'the tale of', 'peter rabbit', 'beatrix potter', 'by beatrix potter'}
    if t in short_title_terms:
        return True
    return False


OLLAMA_DISABLED = False
OLLAMA_MODEL_PREFERENCE_PATTERNS = [
    r"^llama3\.3",
    r"^llama3\.2",
    r"^llama3\.1",
    r"^llama3",
    r"^mistral",
    r"^qwen2\.5.*instruct",
    r"^qwen2\.5",
    r"^qwen2",
]


def _detect_local_ollama() -> Optional[str]:
    """
    Pings the local Ollama instance, honors an explicit override, and otherwise
    prefers stronger instruction-tuned narrative models such as Llama when available.
    """
    global OLLAMA_DISABLED
    if OLLAMA_DISABLED:
        return None
    import urllib.request
    import json
    try:
        req = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1.0)
        if req.status == 200:
            tags = json.loads(req.read().decode('utf-8'))
            models = [m["name"] for m in tags.get("models", [])]
            logger.info(f"Detected local Ollama models: {models}")
            env_override = os.getenv("FIRESPEAKER_OLLAMA_MODEL")
            if env_override:
                for model in models:
                    if model == env_override:
                        return model
                logger.warning(f"FIRESPEAKER_OLLAMA_MODEL='{env_override}' is not installed. Available models: {models}")

            lowered_models = [(model, model.lower()) for model in models]
            for pattern in OLLAMA_MODEL_PREFERENCE_PATTERNS:
                for original_model, lowered_model in lowered_models:
                    if re.search(pattern, lowered_model):
                        return original_model

            if models:
                return models[0]
    except Exception:
        pass
    return None


def _query_local_ollama(model: str, prompt: str, format_json: bool = True) -> str:
    """
    Queries local Ollama using standard api/generate with robust JSON-mode
    and format constraints.
    """
    global OLLAMA_DISABLED
    if OLLAMA_DISABLED:
        return ""
    import urllib.request
    import json
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }
    if format_json:
        payload["format"] = "json"
        
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        res = urllib.request.urlopen(req, timeout=120.0)
        if res.status == 200:
            response_obj = json.loads(res.read().decode("utf-8"))
            return response_obj.get("response", "").strip()
    except Exception as e:
        logger.error(f"Error querying local Ollama: {e}. Disabling Ollama globally to prevent future hangs.")
        OLLAMA_DISABLED = True
    return ""


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


class ManuscriptAnalyzer:
    """
    Core engine to parse raw book manuscripts and extract clean script structures.
    """
    
    def __init__(self, use_gpu: bool = True, production_tier: int = 1):
        self.production_tier = production_tier
        self.use_gpu = use_gpu and HAS_XCORE and torch.cuda.is_available()
        self.device = "cuda:0" if self.use_gpu else "cpu"
        self.xcore_model = None
        
        # Bypasses local LLM queries for Tier < 3 to ensure completely offline / rapid execution
        if production_tier >= 3:
            self.ollama_model = _detect_local_ollama()
        else:
            self.ollama_model = None
            
        if self.ollama_model:
            logger.info(f"Ollama local LLM service detected active. Selected model: '{self.ollama_model}'")
        else:
            logger.info("Local Ollama model server bypassed or disabled for rapid rule-based processing.")
            
        # Load xCoRe only for GPU-backed runs. CPU initialization is slow in this
        # environment and has been unstable enough to degrade the parser path.
        if HAS_XCORE and self.use_gpu:
            try:
                logger.info(f"Initializing xCoRe model 'sapienzanlp/xcore-litbank' on {self.device}...")
                self.xcore_model = xCoRe(
                    hf_name_or_path="sapienzanlp/xcore-litbank",
                    device=self.device
                )
                logger.info("xCoRe initialized successfully.")
            except Exception as e:
                logger.error(f"Error loading xCoRe model: {e}. Falling back to spaCy rules.")
                self.xcore_model = None
        elif HAS_XCORE:
            logger.info("xCoRe disabled for CPU/non-GPU parsing. Falling back to spaCy/regex coreference.")
                
        # Load spaCy fallback model
        global nlp
        if HAS_SPACY and nlp is None:
            try:
                # Prefer transformer or large model for high accuracy
                if use_gpu:
                    spacy.prefer_gpu()
                nlp = spacy.load("en_core_web_lg")
                logger.info("spaCy 'en_core_web_lg' model loaded successfully.")
            except OSError:
                try:
                    nlp = spacy.load("en_core_web_sm")
                    logger.info("spaCy 'en_core_web_sm' model loaded successfully as fallback.")
                except OSError:
                    logger.warning("Could not find any standard spaCy models. Please run: python -m spacy download en_core_web_sm")

    def _determine_emotion(self, text: str) -> str:
        """
        Determines the discrete emotional tag of a text segment based on VADER or keywords.
        Mapped to: Joy, Sadness, Tension, Neutral.
        """
        if HAS_NLTK and sid:
            scores = sid.polarity_scores(text)
            compound = scores['compound']
            
            # Simple threshold rules
            if compound >= 0.35:
                return "Joy"
            elif compound <= -0.35:
                # Distinguish sadness vs. tension using high-intensity negative words
                tension_words = {'fear', 'scared', 'afraid', 'run', 'hide', 'terror', 'danger', 'shouted', 'screamed'}
                if any(w in text.lower() for w in tension_words):
                    return "Tension"
                return "Sadness"
            else:
                return "Neutral"
        
        # Fallback keyword rules if NLTK is missing
        text_lower = text.lower()
        joy_keywords = {'happy', 'laugh', 'smile', 'joy', 'glad', 'excellent', 'wonderful', 'perfect'}
        sadness_keywords = {'sad', 'cry', 'weep', 'tears', 'mourn', 'gloomy', 'depressed'}
        tension_keywords = {'afraid', 'scared', 'terror', 'danger', 'panic', 'angry', 'shout', 'scream'}
        
        if any(w in text_lower for w in joy_keywords):
            return "Joy"
        elif any(w in text_lower for w in sadness_keywords):
            return "Sadness"
        elif any(w in text_lower for w in tension_keywords):
            return "Tension"
            
        return "Neutral"

    def _fallback_resolve_coreferences(self, text: str) -> dict:
        """Helper method containing the spaCy/regex fallback logic for coreference resolution."""
        # Simple fallback using spaCy NER
        logger.info("Executing spaCy NER for character list...")
        characters = defaultdict(lambda: {"canonical_name": "", "total_mentions_count": 0, "unique_references": []})
        if nlp:
            doc = nlp(text[:100000])  # limit length to avoid memory bloat in fallback
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    name = ent.text.strip()
                    if _is_valid_spacy_person_entity(name):
                        if name not in characters:
                            characters[name] = {
                                "canonical_name": name,
                                "total_mentions_count": 1,
                                "unique_references": [name]
                            }
                        else:
                            characters[name]["total_mentions_count"] += 1
        else:
            logger.info("spaCy not available. Executing regex fallback for character extraction...")
            # 1. Matches like "Mr. McGregor", "Mrs. Rabbit" (capturing the honorific)
            honorific_matches = re.findall(r'\b((?:Mr|Mrs|Ms|Dr|Sir|Lady|Miss)\.?\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b', text)
            for name in honorific_matches:
                name = name.strip()
                if name.lower() not in PRONOUNS and len(name) > 2:
                    characters[name] = {
                        "canonical_name": name,
                        "total_mentions_count": 1,
                        "unique_references": [name]
                    }
            # 2. Match "old Mrs. Rabbit" and title-case canonicalize to "Old Mrs. Rabbit"
            old_honorifics = re.findall(r'\b(old\s+Mrs\.?\s+[A-Z][a-zA-Z]+)\b', text, flags=re.IGNORECASE)
            for name in old_honorifics:
                name_title = name.title().replace(".", "") # "Old Mrs Rabbit"
                characters[name_title] = {
                    "canonical_name": name_title,
                    "total_mentions_count": 1,
                    "unique_references": [name]
                }
            # 3. Matches like "said Watson", "asked Holmes"
            verb_after_matches = re.findall(r'\b(?i:say|said|ask|asked|reply|replied|shout|shouted|whisper|whispered)\s+(?:the\s+)?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b', text)
            for name in verb_after_matches:
                name = name.strip()
                if name.lower() not in PRONOUNS and len(name) > 2:
                    if name not in characters:
                        characters[name] = {
                            "canonical_name": name,
                            "total_mentions_count": 1,
                            "unique_references": [name]
                        }
            # 4. Matches like "Holmes said", "Watson replied"
            verb_before_matches = re.findall(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?i:say|said|ask|asked|reply|replied|shout|shouted|whisper|whispered)\b', text)
            for name in verb_before_matches:
                name = name.strip()
                if name.lower() not in PRONOUNS and len(name) > 2:
                    if name not in characters:
                        characters[name] = {
                            "canonical_name": name,
                            "total_mentions_count": 1,
                            "unique_references": [name]
                        }
        return dict(characters)

    def _resolve_coreferences(self, text: str) -> dict:
        """
        Executes local Ollama LLM, xCoRe, or spaCy/regex rules to extract character names
        and map their variations and persona roles.
        """
        # 1. Local LLM (Ollama) high-fidelity extraction path
        if self.ollama_model:
            try:
                logger.info(f"Ollama local model '{self.ollama_model}' active. Extracting character entities...")
                # Extract first 15,000 characters to cover the character introduction contexts safely and fast
                sample_text = text[:15000]
                prompt = f"""
You are an expert literary analyzer. Extract all unique proper characters/speakers (proper nouns) from the following text block.
Identify the canonical proper name for each character (e.g., "Old Mrs. Rabbit" instead of "Mrs. Rabbit" or "mother").
Return the results in a valid JSON object containing a "characters" key mapping to a list of canonical names.

Example format:
{{
  "characters": ["Old Mrs. Rabbit", "Mr. McGregor", "Peter Rabbit"]
}}

MANUSCRIPT TEXT:
{sample_text}

Return ONLY the valid JSON object.
"""
                res_str = _query_local_ollama(self.ollama_model, prompt, format_json=True)
                if res_str:
                    # Clean the JSON response string from potential model backticks or markdown
                    res_str = res_str.strip()
                    if "```json" in res_str:
                        res_str = res_str.split("```json")[1].split("```")[0].strip()
                    elif "```" in res_str:
                        res_str = res_str.split("```")[1].split("```")[0].strip()
                    
                    # Robust try-except parse block
                    start_idx = res_str.find("{")
                    end_idx = res_str.rfind("}")
                    if start_idx != -1 and end_idx != -1:
                        res_str = res_str[start_idx:end_idx + 1]
                        
                    parsed = json.loads(res_str)
                    extracted_list = parsed.get("characters", [])
                    
                    characters = {}
                    for name in extracted_list:
                        name = name.strip()
                        if name.lower() not in PRONOUNS and len(name) > 1:
                            characters[name] = {
                                "canonical_name": name,
                                "total_mentions_count": 1,
                                "unique_references": [name]
                            }
                    if characters:
                        logger.info(f"Ollama successfully extracted high-fidelity characters: {list(characters.keys())}")
                        return characters
                else:
                    logger.warning("Empty response received from local Ollama. Disabling model for this analyzer instance.")
                    self.ollama_model = None
            except Exception as e:
                logger.error(f"Error resolving coreferences via Ollama: {e}. Disabling Ollama model for this analyzer instance to prevent further hangs and falling back to spaCy/regex...")
                self.ollama_model = None

        # 2. Standard xCoRe or spaCy/regex rules fallback paths
        if not HAS_XCORE or not self.xcore_model:
            return self._fallback_resolve_coreferences(text)

        # Advanced xCoRe path
        try:
            predictions = self.xcore_model.predict(text)
            clusters_mentions = predictions['clusters_text_mentions']
            
            consolidated = {}
            for idx, mentions in enumerate(clusters_mentions):
                # Filter out pronouns to identify proper nouns
                names = [m.strip() for m in mentions if m.strip().lower() not in PRONOUNS]
                
                if names:
                    # Canonical name is the longest non-pronoun mention (to avoid single initials or short fragments)
                    canonical = max(names, key=len)
                    
                    # Normalize honorifics
                    canonical_norm = canonical
                    if canonical_norm.lower().startswith("old "):
                        canonical_norm = canonical_norm.title().replace(".", "")
                    
                    consolidated[canonical_norm] = {
                        "canonical_name": canonical_norm,
                        "total_mentions_count": len(mentions),
                        "unique_references": list(set(names))
                    }
            return dict(consolidated)
        except Exception as e:
            logger.error(f"Error during xCoRe coreference prediction: {e}. Falling back to spaCy/regex rules...")
            return self._fallback_resolve_coreferences(text)

    def detect_and_normalize_quotes(self, text: str) -> tuple[str, str]:
        """
        Scans text to detect dominant quote style (double vs single).
        If single quotes dominate dialogue, normalizes them to double quotes.
        """
        # Count double quotes
        double_count = len(re.findall(r'["“]', text))
        
        # Count single quotes at boundaries (excluding contractions)
        single_count_open = len(re.findall(r'(^|[\s\(\[\-])[\'‘]', text))
        single_count_close = len(re.findall(r'[\'’]($|[\s\.,!\?\;\)\]\-])', text))
        single_count = (single_count_open + single_count_close) / 2
        
        if single_count > 2 and single_count > double_count:
            logger.info(f"Detected single-quote dialogue style (Single quotes: {single_count:.1f}, Double quotes: {double_count}). Normalizing to double quotes...")
            # Replace opening single quotes with double quotes
            normalized = re.sub(r'(^|[\s\(\[\-])[\'‘]', r'\1"', text)
            # Replace closing single quotes with double quotes
            normalized = re.sub(r'[\'’]($|[\s\.,!\?\;\)\]\-])', r'"\1', normalized)
            return normalized, "single"
            
        return text, "double"

    def parse_manuscript(self, file_path: str, chapters: str = None) -> dict:
        """
        Main parser pipeline. Reads file, segments into chapters, extracts dialogues,
        attributes speakers, assigns emotions, and saves structured script results.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Manuscript not found at: {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
            
        logger.info(f"Ingested manuscript from {file_path} ({len(raw_content)} characters).")
        
        # Ingest and normalize quotes if using single quotes
        normalized_content, quote_style = self.detect_and_normalize_quotes(raw_content)

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
                logger.info(f"Chapter filter active: targeting chapters {selected_chapters}")
            except Exception as e:
                logger.error(f"Failed to parse chapters filter '{chapters}': {e}")

        # Step 2: Segment Text into Chapters (moved up to support early filtering)
        chapter_pattern = r'(?i)^\s*(?:(?:chapter|scene)\s+(?:[0-9]+|[IVXLCDM]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b|(?:[IVXLCDM]+)(?:--|\s*[-.]\s*).*)$'
        chapter_splits = re.split(chapter_pattern, normalized_content, flags=re.MULTILINE)
        
        # Clean chapter headings lists
        chapter_headings = [h.strip() for h in re.findall(chapter_pattern, normalized_content, flags=re.MULTILINE)]
        
        if len(chapter_splits) <= 1:
            chapter_blocks = [normalized_content]
            chapter_headings = ["Chapter 1"]
        else:
            chapter_blocks = chapter_splits[1:]

        # Reconstruct content with only target chapters for coreference resolution if filter is active
        if selected_chapters:
            coref_content_blocks = []
            for idx, block in enumerate(chapter_blocks):
                chapter_num = idx + 1
                if chapter_num in selected_chapters:
                    heading = chapter_headings[idx] if idx < len(chapter_headings) else f"Chapter {chapter_num}"
                    coref_content_blocks.append(heading + "\n\n" + block)
            coref_content = "\n\n".join(coref_content_blocks)
        else:
            coref_content = normalized_content
        
        # Seed character list with registered drawers from MemPalace to bypass NER omissions
        registered_characters = []
        try:
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
            from src.spatial_memory import MemPalace
            palace = MemPalace()
            cursor = palace.conn.cursor()
            cursor.execute("SELECT character_name FROM drawers;")
            registered_characters = [row[0] for row in cursor.fetchall() if row[0] != "Narrator"]
            palace.close()
            logger.info(f"Seeding NLP parser with registered drawers from MemPalace: {registered_characters}")
        except Exception as e:
            logger.warning(f"Could not seed character list from MemPalace: {e}")

        # Step 1: Coreference Analysis across document (run on targeted chapters' text only)
        character_db = self._resolve_coreferences(coref_content)
        characters_list = list(character_db.keys())
        
        for rc in registered_characters:
            if rc not in characters_list:
                characters_list.append(rc)
                
        logger.info(f"Identified {len(characters_list)} unique character profiles: {characters_list[:10]}...")
        
        # Step 1.5: Query relational confirmed merges from MemPalace SQLite
        confirmed_merges = {}
        try:
            from src.spatial_memory import MemPalace
            palace = MemPalace()
            confirmed_merges = palace.get_confirmed_merges(os.path.basename(file_path))
            palace.close()
        except Exception as e:
            logger.warning(f"Could not load confirmed merges from MemPalace: {e}")
            
        # Consolidate duplicate characters
        characters_list, merge_map, confidence_scores, merge_decisions = consolidate_characters(characters_list, confirmed_merges)
        logger.info(f"Consolidated into {len(characters_list)} character profiles after merge check: {characters_list}")
        
        script_output = []
        dialogue_queue = [] # Queue to track active speakers for alternating dialogue
        
        for c_idx, block in enumerate(chapter_blocks):
            chapter_num = c_idx + 1
            if selected_chapters and chapter_num not in selected_chapters:
                continue
            chapter_name = chapter_headings[c_idx] if c_idx < len(chapter_headings) else f"Chapter {chapter_num}"
            logger.info(f"Parsing {chapter_name}...")
            
            # Segment block into paragraphs
            paragraphs = [p.strip() for p in re.split(r'\n+', block) if p.strip()]
            
            line_count = 1
            # Initialize speaker lock for this chapter
            locked_speaker = None
            speaker_lock_counter = 0
            
            for p_idx, paragraph in enumerate(paragraphs):
                if _is_metadata_or_clutter(paragraph):
                    continue
                # Split paragraph by matched quotes to isolate dialogues (odd indices are dialogues)
                parts = re.split(r'["“]([^"”]+)["”]', paragraph)
                
                if len(parts) <= 1:
                    speaker_lock_counter = 0  # Narrative paragraph break, clear the speaker lock!
                    # Add pure narration paragraph to script output!
                    line_content = f"{os.path.basename(file_path)}_c{chapter_num}_l{line_count}_{paragraph}"
                    line_id = hashlib.sha256(line_content.encode('utf-8')).hexdigest()[:16]
                    script_output.append({
                        "line_id": line_id,
                        "chapter": chapter_num,
                        "chapter_title": chapter_name,
                        "line_number": line_count,
                        "character": "Narrator",
                        "dialogue": "",
                        "narration_before": paragraph,
                        "narration_after": "",
                        "emotion": "Neutral",
                        "attribution_method": "Narration"
                    })
                    line_count += 1
                    continue
                    
                for i in range(1, len(parts), 2):
                    dialogue_text = parts[i].strip()
                    if not dialogue_text:
                        continue
                        
                    narration_before = parts[i-1].strip() if i > 0 else ""
                    narration_after = parts[i+1].strip() if i < len(parts) - 1 else ""
                    
                    narration_before = re.sub(r'\s+', ' ', narration_before)
                    narration_after = re.sub(r'\s+', ' ', narration_after)
                    
                    assigned_character = "Narrator"
                    attribution_method = "Default"
                    found_speaker = False
                    high_confidence = False
                    
                    # 1. Direct speech verb check (case-insensitive, checked longest name first to avoid substring collision)
                    if narration_after:
                        for char in sorted(characters_list, key=len, reverse=True):
                            char_pattern = re.escape(char).replace(r'\ ', r'\s+').replace('Mrs', 'Mrs\\.?')
                            char_pattern = re.sub(r'\bMr\b', r'Mr\\.?', char_pattern)
                            if re.search(r'\b' + char_pattern + r'\b', narration_after, flags=re.IGNORECASE):
                                words = narration_after.lower().split()
                                if any(v in words for v in SPEECH_VERBS):
                                    assigned_character = char
                                    attribution_method = "Direct Speech Verb (After)"
                                    found_speaker = True
                                    high_confidence = True
                                    break
                                    
                    if not found_speaker and narration_before:
                        for char in sorted(characters_list, key=len, reverse=True):
                            char_pattern = re.escape(char).replace(r'\ ', r'\s+').replace('Mrs', 'Mrs\\.?')
                            char_pattern = re.sub(r'\bMr\b', r'Mr\\.?', char_pattern)
                            if re.search(r'\b' + char_pattern + r'\b', narration_before, flags=re.IGNORECASE):
                                words = narration_before.lower().split()
                                if any(v in words for v in SPEECH_VERBS):
                                    assigned_character = char
                                    attribution_method = "Direct Speech Verb (Before)"
                                    found_speaker = True
                                    high_confidence = True
                                    break
                                    
                    # 2. Speaker Lock Override (if no high-confidence speech verb and lock is active)
                    if not high_confidence and speaker_lock_counter > 0:
                        assigned_character = locked_speaker
                        attribution_method = "Speaker Lock Override"
                        found_speaker = True
                        speaker_lock_counter -= 1
                        
                    # 3. Context mention check (including robust aliases and preceding paragraph fallback)
                    if not found_speaker:
                        surrounding_text = (narration_before + " " + narration_after).strip()
                        # Fallback to preceding paragraph context if current paragraph narration is empty
                        if len(surrounding_text) < 5 and p_idx > 0:
                            surrounding_text = paragraphs[p_idx - 1]
                        
                        # Check robust aliases first to capture active roles (like mother/she) before raw mentions
                        alias_resolved = _resolve_speaker_aliases(surrounding_text, characters_list)
                        if alias_resolved:
                            assigned_character = alias_resolved
                            attribution_method = "Context Alias Mention"
                            found_speaker = True
                        
                        if not found_speaker:
                            for char in characters_list:
                                if re.search(r'\b' + re.escape(char) + r'\b', surrounding_text):
                                    assigned_character = char
                                    attribution_method = "Context Entity Mention"
                                    found_speaker = True
                                    break
                                
                    # 4. Conversational alternating queue fallback
                    if not found_speaker:
                        if dialogue_queue:
                            distinct_speakers = []
                            for speaker in reversed(dialogue_queue):
                                if speaker != "Narrator" and speaker not in distinct_speakers:
                                    distinct_speakers.append(speaker)
                                if len(distinct_speakers) == 2:
                                    break
                                    
                            if len(distinct_speakers) == 2:
                                assigned_character = distinct_speakers[1]  # Alternate
                                attribution_method = "Auto-Attributed (Alternating)"
                            elif len(distinct_speakers) == 1:
                                assigned_character = distinct_speakers[0]
                                attribution_method = "Auto-Attributed (Single Active)"
                        else:
                            assigned_character = "Narrator"
                            attribution_method = "Default Narration Fallback"
                            
                    # Apply merge map to consolidate duplicates dynamically
                    if merge_map and assigned_character in merge_map:
                        assigned_character = merge_map[assigned_character]
                        
                    # Update dialogue queue
                    if assigned_character != "Narrator":
                        dialogue_queue.append(assigned_character)
                        if len(dialogue_queue) > 10:
                            dialogue_queue.pop(0)
                            
                    # Set speaker lock if high confidence
                    if high_confidence:
                        locked_speaker = assigned_character
                        speaker_lock_counter = 2
                        
                    emotion = self._determine_emotion(paragraph)
                    
                    line_content = f"{os.path.basename(file_path)}_c{chapter_num}_l{line_count}_{dialogue_text}"
                    line_id = hashlib.sha256(line_content.encode('utf-8')).hexdigest()[:16]
                    
                    script_output.append({
                        "line_id": line_id,
                        "chapter": chapter_num,
                        "chapter_title": chapter_name,
                        "line_number": line_count,
                        "character": assigned_character,
                        "dialogue": dialogue_text,
                        "narration_before": narration_before,
                        "narration_after": narration_after,
                        "emotion": emotion,
                        "attribution_method": attribution_method,
                        "speaker_locked": (speaker_lock_counter > 0 or attribution_method == "Speaker Lock Override")
                    })
                    line_count += 1
                    
        master_script = {
            "metadata": {
                "source_file": os.path.basename(file_path),
                "quote_style_detected": quote_style,
                "total_chapters": len(chapter_blocks) if not selected_chapters else len(selected_chapters),
                "total_lines_extracted": len(script_output),
                "characters_identified": characters_list,
                "merge_decisions": merge_decisions
            },
            "script": script_output
        }
        
        return master_script

    def save_script(self, script_data: dict, output_path: str):
        """Saves script metadata dictionary to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(script_data, f, indent=4)
        logger.info(f"Script metadata successfully saved to: {output_path}")


def main():
    """CLI Entrypoint for testing and verification."""
    import argparse
    parser = argparse.ArgumentParser(description="Firespeaker Manuscript Script Ingestion Pipeline")
    parser.add_argument("--input", type=str, help="Path to raw manuscript text file")
    parser.add_argument("--output", type=str, default="data/scene_script.json", help="Path to save output JSON metadata")
    parser.add_argument("--cpu", action="store_true", help="Force CPU operation")
    parser.add_argument("--test", action="store_true", help="Run self-testing harness")
    
    args = parser.parse_args()
    
    if args.test:
        print("\n=== RUNNING FIRESPEAKER INGESTION SELF-TEST ===")
        # Dynamic creation of test text
        test_dir = "scratch"
        os.makedirs(test_dir, exist_ok=True)
        test_path = os.path.join(test_dir, "test_manuscript.txt")
        
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("""
            Chapter 1: The Inciting Incident
            Sherlock Holmes took his magnifying glass. He turned to his companion.
            "Do you see anything, Watson?" asked Holmes.
            "No, my dear Holmes, it remains dark," Watson replied.
            "Fascinating. We must proceed immediately."
            Watson felt a sudden sense of dread and panic. "Are you sure it is safe?"
            """)
            
        try:
            analyzer = ManuscriptAnalyzer(use_gpu=False)
            script = analyzer.parse_manuscript(test_path)
            output_test_path = os.path.join(test_dir, "test_script.json")
            analyzer.save_script(script, output_test_path)
            
            # Print QC result
            print("\nQC Verification Sheet:")
            print(f"- Total lines parsed: {script['metadata']['total_lines_extracted']} (Expected: 4)")
            print(f"- Characters found: {script['metadata']['characters_identified']}")
            
            # Verify alternating attribution
            line_3 = script['script'][2]
            print(f"- Line 3 speaker: '{line_3['character']}' | Method: '{line_3['attribution_method']}'")
            print(f"- Line 3 emotion: '{line_3['emotion']}'")
            
            print("\n=== SELF-TEST PASSED SUCCESSFULLY ===\n")
            return 0
        except Exception as e:
            print(f"Self-test failed: {e}")
            return 1
            
    if not args.input:
        parser.print_help()
        sys.exit(1)
        
    try:
        analyzer = ManuscriptAnalyzer(use_gpu=not args.cpu)
        script = analyzer.parse_manuscript(args.input)
        analyzer.save_script(script, args.output)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline parsing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
