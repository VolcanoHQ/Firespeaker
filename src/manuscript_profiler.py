#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Firespeaker Manuscript Profiling Engine
Analyzes manuscripts to extract high-level overview statistics,
character voice distributions, emotional profiles, and attribution confidence
to feed interactive user-facing GUI dashboards.
"""

import os
import sys
import json
import logging
from collections import defaultdict
from typing import Dict, Any

# Ensure the root project directory is in the sys.path for absolute modular imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nlp_analyzer import ManuscriptAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ManuscriptProfiler")


class ManuscriptProfiler:
    """Extracts aggregated metadata and statistics designed for interactive GUI dashboards."""

    def __init__(self, use_gpu: bool = False):
        self.analyzer = ManuscriptAnalyzer(use_gpu=use_gpu)

    def profile_book(self, manuscript_path: str, hierarchy_data: dict = None) -> Dict[str, Any]:
        """
        Runs full parser (or uses provided hierarchy index) and aggregates script indexes into a clean dashboard schema.
        """
        if not os.path.exists(manuscript_path):
            raise FileNotFoundError(f"Manuscript not found at: {manuscript_path}")
            
        logger.info(f"Profiling manuscript: {manuscript_path}...")
        
        # 1. Read raw file metrics
        with open(manuscript_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
            
        file_size_bytes = os.path.getsize(manuscript_path)
        char_count = len(raw_text)
        words_list = raw_text.split()
        word_count = len(words_list)
        
        # Estimate processing tier
        if file_size_bytes < 50 * 1024:
            tier = "Micro-Tier"
        elif file_size_bytes < 500 * 1024:
            tier = "Standard Tier"
        elif file_size_bytes < 1500 * 1024:
            tier = "Epic Tier"
        else:
            tier = "Behemoth Tier"
            
        # Estimate spoken duration (150 words per minute average)
        spoken_min = word_count / 150.0
        
        # 2. Extract paragraph information (support Windows \r\n and whitespace-spaced double newlines)
        import re
        paragraphs = [p.strip() for p in re.split(r'\r?\n\s*\r?\n', raw_text) if p.strip()]
        paragraph_count = len(paragraphs)
        
        # 3. Parse manuscript into structured dialogue script using nlp_analyzer, or extract from hierarchy
        script_lines = []
        if hierarchy_data:
            for part in hierarchy_data.get("parts", []):
                for chap in part.get("chapters", []):
                    for scene in chap.get("scenes", []):
                        script_lines.extend(scene.get("lines", []))
        else:
            script_output = self.analyzer.parse_manuscript(manuscript_path)
            script_lines = script_output["script"]
            
        # We only count lines that actually have dialogue for the character dialogue breakdown
        total_dialogue_lines = sum(1 for line in script_lines if line.get("segment_type", "") == "dialogue")
        
        # 4. Calculate Character Spoken Breakdown
        # Tracks spoken dialogue words vs. narration
        character_lines = defaultdict(int)
        character_words = defaultdict(int)
        
        dialogue_word_count = 0
        for line in script_lines:
            dialogue_text = line.get("dialogue", "")
            if not dialogue_text:
                continue
                
            char_name = line["character"]
            line_words = len(dialogue_text.split())
            character_lines[char_name] += 1
            character_words[char_name] += line_words
            
            if line.get("segment_type", "") == "dialogue":
                dialogue_word_count += line_words
            
        # Narrator text is estimated as total words minus spoken words
        narrator_word_count = max(0, word_count - dialogue_word_count)
        
        # 5. Calculate Emotional Mood Distributions
        emotion_counts = defaultdict(int)
        for line in script_lines:
            emotion_counts[line["emotion"]] += 1
            
        total_emotions = sum(emotion_counts.values())
        emotional_breakdown = {}
        for emotion, count in emotion_counts.items():
            emotional_breakdown[emotion] = round((count / total_emotions) * 100.0, 1) if total_emotions > 0 else 0.0
            
        # Ensure Neutral baseline is populated
        for emotion in ["Neutral", "Joy", "Sadness", "Tension"]:
            if emotion not in emotional_breakdown:
                emotional_breakdown[emotion] = 0.0
                
        # 6. Calculate Dialogue Attribution Quality Metrics
        attribution_counts = defaultdict(int)
        for line in script_lines:
            attribution_counts[line["attribution_method"]] += 1
            
        # 7. Auto-detect Narration POV (First-Person vs Third-Person)
        first_person_count = 0
        third_person_count = 0
        first_person_pronouns = {'i', 'me', 'my', 'myself', 'we', 'us', 'our', 'ourselves'}
        third_person_pronouns = {'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'they', 'them', 'their', 'themselves'}
        
        import re
        for para in paragraphs:
            words = para.lower().split()
            for w in words:
                w_clean = re.sub(r'[^a-z]', '', w)
                if w_clean in first_person_pronouns:
                    first_person_count += 1
                elif w_clean in third_person_pronouns:
                    third_person_count += 1
                    
        if first_person_count > 15 and first_person_count > (third_person_count * 0.25):
            pov_style = "First-Person"
        else:
            pov_style = "Third-Person"
            
        # Construct finalized GUI Dashboard JSON schema
        dashboard_profile = {
            "file_metadata": {
                "filename": os.path.basename(manuscript_path),
                "file_size_bytes": file_size_bytes,
                "word_count": word_count,
                "char_count": char_count,
                "estimated_spoken_duration_min": round(spoken_min, 1),
                "processing_tier": tier,
                "narration_pov": pov_style
            },
            "structure_metadata": {
                "total_chapters": hierarchy_data["metadata"]["total_chapters"] if hierarchy_data else script_output["metadata"]["total_chapters"],
                "total_paragraphs": paragraph_count,
                "total_dialogue_lines_extracted": total_dialogue_lines,
                "narration_word_ratio_percent": round((narrator_word_count / word_count) * 100.0, 1) if word_count > 0 else 100.0,
                "dialogue_word_ratio_percent": round((dialogue_word_count / word_count) * 100.0, 1) if word_count > 0 else 0.0
            },
            "character_dialogue_breakdown": {},
            "emotional_breakdown": emotional_breakdown,
            "attribution_summary": dict(attribution_counts)
        }
        
        # Populate character breakdown
        # Base Narrator entry
        narrator_dialogue_lines = character_lines.get("Narrator", 0)
        narrator_total_words = character_words.get("Narrator", 0)
        
        dashboard_profile["character_dialogue_breakdown"]["Narrator"] = {
            "dialogue_line_count": narrator_dialogue_lines,
            "spoken_words": narrator_total_words,
            "dialogue_share_percent": round((narrator_dialogue_lines / total_dialogue_lines) * 100.0, 1) if total_dialogue_lines > 0 else 0.0,
            "word_share_percent": round((narrator_total_words / word_count) * 100.0, 1) if word_count > 0 else 0.0
        }
        
        for char, lines in character_lines.items():
            if char == "Narrator":
                continue
            words = character_words[char]
            dashboard_profile["character_dialogue_breakdown"][char] = {
                "dialogue_line_count": lines,
                "spoken_words": words,
                "dialogue_share_percent": round((lines / total_dialogue_lines) * 100.0, 1) if total_dialogue_lines > 0 else 0.0,
                "word_share_percent": round((words / word_count) * 100.0, 1) if word_count > 0 else 0.0
            }
            
        return dashboard_profile

    def save_profile(self, profile_data: dict, output_path: str):
        """Saves aggregated dashboard statistics to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=4)
        logger.info(f"Dashboard overview statistics successfully saved to: {output_path}")


def main():
    """CLI and self-test harness for the profiling suite."""
    import argparse
    parser = argparse.ArgumentParser(description="Firespeaker Manuscript Dashboard Profiler")
    parser.add_argument("--input", type=str, help="Path to manuscript book text file to analyze")
    parser.add_argument("--output", type=str, default="scratch/manuscript_profile_summary.json", help="Path to save output dashboard JSON metadata")
    parser.add_argument("--test", action="store_true", help="Run self-test on the smallest Micro-Tier book (Peter Rabbit)")
    
    args = parser.parse_args()
    
    if args.test:
        print("\n=== RUNNING MANUSCRIPT PROFILER SELF-TEST (PETER RABBIT) ===")
        peter_rabbit_path = "data/corpus/TheTaleofPeterRabbit.txt"
        
        # Verify the file is in place (or create a quick segment if missing)
        if not os.path.exists(peter_rabbit_path):
            os.makedirs(os.path.dirname(peter_rabbit_path), exist_ok=True)
            with open(peter_rabbit_path, "w", encoding="utf-8") as f:
                f.write("""
                Once upon a time there were four little Rabbits, and their names were— Flopsy, Mopsy, Cottontail, and Peter.
                
                “Now, my dears,” said old Mrs. Rabbit one morning, “you may go into the fields or down the lane, but don’t go into Mr. McGregor’s garden.”
                
                Peter ran straight away to Mr. McGregor’s garden, and squeezed under the gate!
                
                First he ate some lettuces and some French beans; and then he ate some radishes.
                
                “Stop thief!” called Mr. McGregor.
                """)
                
        profiler = ManuscriptProfiler()
        profile = profiler.profile_book(peter_rabbit_path)
        output_test_path = "scratch/test_manuscript_profile.json"
        profiler.save_profile(profile, output_test_path)
        
        # Print verification checks
        print("\nGUI Dashboard Schema Verification:")
        print(f"- Target Book:              {profile['file_metadata']['filename']}")
        print(f"- Total Words counted:       {profile['file_metadata']['word_count']}")
        print(f"- Estimated spoken minutes:  {profile['file_metadata']['estimated_spoken_duration_min']} minutes")
        print(f"- Estimated Spoken Tier:     {profile['file_metadata']['processing_tier']}")
        print(f"- Dialogue ratio percent:    {profile['structure_metadata']['dialogue_word_ratio_percent']}%")
        print(f"- Narration ratio percent:   {profile['structure_metadata']['narration_word_ratio_percent']}%")
        
        print("\nCharacter Roster Breakdowns:")
        for char, data in profile["character_dialogue_breakdown"].items():
            print(f"  * {char}: Dialogue lines={data['dialogue_line_count']} | Spoken words={data['spoken_words']} | Share={data['word_share_percent']}%")
            
        print("\nEmotional Distribution:")
        print(f"- Mood breakdowns: {profile['emotional_breakdown']}")
        
        print("\n=== MANUSCRIPT PROFILER HARNESS PASSED SUCCESSFULLY ===\n")
        return 0
        
    if not args.input:
        parser.print_help()
        sys.exit(1)
        
    try:
        profiler = ManuscriptProfiler()
        profile = profiler.profile_book(args.input)
        profiler.save_profile(profile, args.output)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Profiling failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
