#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Firespeaker Studio Pipeline Orchestrator
Ties together Ingestion & NLP Analysis, Spatial Memory Management,
Expressive TTS Synthesis, and ACX Mixing & Mastering into a unified,
production-grade command line interface.
"""

import os
import sys
import json
import argparse
import logging
from typing import List

# Ensure the root project directory is in the sys.path for absolute modular imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nlp_analyzer import ManuscriptAnalyzer
from src.spatial_memory import MemPalace
from src.voice_synthesizer import VoiceSynthesizer
from src.audio_mixer import AudioMixer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FirespeakerStudio")


class FirespeakerPipeline:
    """Master orchestrator managing cross-component relational and file flows."""

    def __init__(self, workspace_dir: str = "scratch/pipeline_workspace", production_tier: int = 1):
        self.workspace_dir = workspace_dir
        self.mempalace_dir = "data/mempalace"
        self.outputs_dir = os.path.join(workspace_dir, "outputs")
        
        os.makedirs(self.workspace_dir, exist_ok=True)
        os.makedirs(self.mempalace_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)
        
        # Initialize sub-engines
        self.analyzer = ManuscriptAnalyzer(use_gpu=False, production_tier=production_tier)
        self.palace = MemPalace(db_dir=self.mempalace_dir)
        self.synth = VoiceSynthesizer(mempalace_path=self.mempalace_dir, force_cpu=True)
        self.mixer = AudioMixer()

    def register_voice_cast(self):
        """Pre-registers default voice profiles (timbre drawers) to guarantee compliance."""
        logger.info("Pre-registering default voice cast drawers in MemPalace...")
        # 1. Register Narrator
        if not self.palace.get_character_drawer("Narrator"):
            self.palace.register_character(
                character_name="Narrator",
                voice_ref_path="data/voice_references/narrator_mono.wav",
                speed=1.0,
                pitch=0.0
            )
        # 2. Register Sherlock Holmes
        if not self.palace.get_character_drawer("Holmes"):
            self.palace.register_character(
                character_name="Holmes",
                voice_ref_path="data/voice_references/holmes_mono.wav",
                speed=1.05,
                pitch=1.5
            )
        # 3. Register John Watson
        if not self.palace.get_character_drawer("Watson"):
            self.palace.register_character(
                character_name="Watson",
                voice_ref_path="data/voice_references/watson_mono.wav",
                speed=0.95,
                pitch=-1.0
            )
        # 4. Register Arthur (preset)
        if not self.palace.get_character_drawer("Arthur"):
            self.palace.register_character(
                character_name="Arthur",
                voice_ref_path="data/voice_references/narrator_mono.wav",
                speed=1.0,
                pitch=-0.9
            )
        # 5. Register Emily (preset)
        if not self.palace.get_character_drawer("Emily"):
            self.palace.register_character(
                character_name="Emily",
                voice_ref_path="data/voice_references/narrator_mono.wav",
                speed=1.05,
                pitch=3.86
            )
        # 6. Register Michael (preset)
        if not self.palace.get_character_drawer("Michael"):
            self.palace.register_character(
                character_name="Michael",
                voice_ref_path="data/voice_references/narrator_mono.wav",
                speed=0.9,
                pitch=-3.86
            )
        
        # Pre-register emotional timbre references for Holmes if they do not exist
        cursor = self.palace.conn.cursor()
        cursor.execute("SELECT 1 FROM emotional_references WHERE character_name = 'Holmes' AND emotion = 'Tension';")
        if not cursor.fetchone():
            self.palace.register_emotional_reference(
                character_name="Holmes",
                emotion="Tension",
                voice_ref_path="data/voice_references/holmes_anxious.wav",
                embedding=[0.9, 0.3, -0.05, 0.4]
            )
        
        cursor.execute("SELECT 1 FROM emotional_references WHERE character_name = 'Holmes' AND emotion = 'Joy';")
        if not cursor.fetchone():
            self.palace.register_emotional_reference(
                character_name="Holmes",
                emotion="Joy",
                voice_ref_path="data/voice_references/holmes_excited.wav",
                embedding=[0.5, 0.8, 0.2, -0.1]
            )

    def run_full_pipeline(self, manuscript_path: str, output_master_wav: str, profile_name: str = "standard", user_tier: str = "free", chapters: str = None, enable_llm_enrichment: bool = False) -> bool:
        print("\n=======================================================")
        print("=== FIRESPEAKER STUDIO: END-TO-END PIPELINE RUN ===")
        print("=======================================================\n")
        
        # 0. Set up default voices in database (if not registered already)
        self.register_voice_cast()
        
        # Check if there is a cached hierarchy file to preserve human-in-the-loop edits.
        import re
        base_name = os.path.splitext(os.path.basename(manuscript_path))[0]
        slug = re.sub(r'[^a-zA-Z0-9_\-]', '', base_name)
        tier = self.analyzer.production_tier
        cached_hierarchy_path = f"data/processed/{slug}/Tier_{tier}/hierarchy.json"
        
        if os.path.exists(cached_hierarchy_path):
            print(f"Loading script data from cached hierarchy: {cached_hierarchy_path}")
            with open(cached_hierarchy_path, "r", encoding="utf-8") as f:
                hierarchy_data = json.load(f)
            
            # Flatten lines from parts -> chapters -> scenes -> lines
            flat_lines = []
            for part in hierarchy_data.get("parts", []):
                for chapter in part.get("chapters", []):
                    for scene in chapter.get("scenes", []):
                        for line in scene.get("lines", []):
                            flat_lines.append(line)
            
            script_data = {
                "metadata": {
                    "source_file": hierarchy_data["metadata"].get("source_file"),
                    "quote_style_detected": hierarchy_data["metadata"].get("quote_style_detected", "double"),
                    "total_chapters": hierarchy_data["metadata"].get("total_chapters", 1),
                    "total_lines_extracted": len(flat_lines),
                    "characters_identified": hierarchy_data["metadata"].get("global_characters", []),
                    "merge_decisions": hierarchy_data["metadata"].get("merge_decisions", [])
                },
                "script": flat_lines
            }
        elif self.analyzer.production_tier == 1:
            print("Step 1: Running Tier 1 Deterministic Ingestion Pipeline...")
            from src.tier_1_parser import ingest_manuscript_tier_1
            manifest = ingest_manuscript_tier_1(manuscript_path, chapters=chapters, enable_llm_enrichment=enable_llm_enrichment)

            # Convert Manifest Payload to the dictionary script format expected by main.py
            flat_lines = []
            for part in manifest.parts:
                for chapter in part.chapters:
                    for scene in chapter.scenes:
                        for line in scene.lines:
                            flat_lines.append(line.model_dump())

            script_data = {
                "metadata": {
                    "source_file": manifest.source_file,
                    "quote_style_detected": "double",
                    "total_chapters": manifest.total_chapters,
                    "total_lines_extracted": len(flat_lines),
                    "characters_identified": sorted(set(line["character"] for line in flat_lines)),
                    "merge_decisions": []
                },
                "script": flat_lines
            }
        else:
            print("Step 1: Parsing manuscript and running coreference/emotion extraction...")
            script_data = self.analyzer.parse_manuscript(manuscript_path, chapters=chapters)
            
        # Filter by chapters if requested
        if chapters:
            try:
                selected_chapters = set()
                for part in chapters.split(','):
                    if '-' in part:
                        start, end = part.split('-')
                        selected_chapters.update(range(int(start), int(end) + 1))
                    else:
                        selected_chapters.add(int(part))
                
                filtered_lines = [line for line in script_data["script"] if line.get("chapter") in selected_chapters]
                print(f"Filtering script to chapters {chapters}. Lines before: {len(script_data['script'])}, after: {len(filtered_lines)}")
                
                script_data["script"] = filtered_lines
                script_data["metadata"]["total_lines_extracted"] = len(filtered_lines)
                
                unique_chapters = set(line.get("chapter") for line in filtered_lines if line.get("chapter") is not None)
                script_data["metadata"]["total_chapters"] = len(unique_chapters)
            except Exception as e:
                logger.error(f"Failed to filter chapters '{chapters}': {e}")

        script_output_path = os.path.join(self.workspace_dir, "script_output.json")
        self.analyzer.save_script(script_data, script_output_path)
        print(f"  --> Ingested successfully. Total lines: {script_data['metadata']['total_lines_extracted']}\n")
        
        # Write human-readable attribution summary check-sheet for editors
        attribution_log_path = "scratch/attribution_summary.txt"
        os.makedirs(os.path.dirname(attribution_log_path), exist_ok=True)
        with open(attribution_log_path, "w", encoding="utf-8") as f_log:
            f_log.write("=========================================================\n")
            f_log.write("FIRESPEAKER ATTRIBUTION LOG: AUTO-ATTRIBUTED LINES\n")
            f_log.write("=========================================================\n\n")
            
            auto_count = 0
            for line in script_data["script"]:
                if "Auto-Attributed" in line.get("attribution_method", ""):
                    f_log.write(f"Line {line['line_number']} [{line['character']}]: \"{line['dialogue']}\"\n")
                    f_log.write(f"  --> Method: {line['attribution_method']}\n\n")
                    auto_count += 1
            
            if auto_count == 0:
                f_log.write("No auto-attributed dialogue lines found in this run.\n")
        logger.info(f"Human-readable attribution summary compiled at: {attribution_log_path}")

        # Log Chapters (Wings) in MemPalace database
        logged_chapters = set()
        for line in script_data["script"]:
            ch = line.get("chapter")
            if ch is not None and ch not in logged_chapters:
                self.palace.log_wing(f"wing_c{ch}", ch, f"Chapter {ch}")
                logged_chapters.add(ch)
            
        # 1.8. Ensure all characters in the manuscript have a registered voice drawer
        logger.info("Validating character drawer assignments to prevent identity blocks...")
        unique_characters = set(line["character"] for line in script_data["script"])
        for char in unique_characters:
            if char != "Narrator":
                drawer = self.palace.get_character_drawer(char)
                if not drawer:
                    logger.info(f"Dynamically registering default voice drawer for character: '{char}'")
                    self.palace.register_character(
                        character_name=char,
                        voice_ref_path="data/voice_references/narrator_mono.wav",
                        speed=1.0,
                        pitch=0.0
                    )
            
        # Import compiler dynamically to prevent circular dependencies
        from src.tts_compiler import compile_modified_json
        
        print("Step 2-5: Executing TTS Synthesis, Modifier Application, and Mastering Mix...")
        qc_report = compile_modified_json(
            script_data=script_data,
            output_master_wav=output_master_wav,
            profile_name=profile_name,
            user_tier=user_tier
        )
        
        print("\n=======================================================")
        print("=== PIPELINE COMPLETION REPORT ===")
        print("=======================================================")
        print(f"- Master File:    {qc_report['metadata']['file_analyzed']}")
        print(f"- Peak Amplitude: {qc_report['metrics']['physical_peak_amplitude_dbfs']} dBFS (Standards: <= -3.0)")
        print(f"- RMS Loudness:   {qc_report['metrics']['root_mean_square_rms_dbfs']} dBFS (Standards: -23.0 to -18.0)")
        print(f"- Peak QC status: {qc_report['validation']['peak_check']}")
        print(f"- RMS QC status:  {qc_report['validation']['rms_check']}")
        print(f"- ACX Compliance: {qc_report['validation']['overall_acx_compliance']}")
        print("=======================================================\n")
        
        return qc_report["validation"]["overall_acx_compliance"] == "PASSED"



def main():
    """Main CLI orchestrator entrance."""
    parser = argparse.ArgumentParser(description="Firespeaker Studio Pipeline CLI")
    
    # Complete end-to-end run commands
    parser.add_argument("--run-all", action="store_true", help="Run the complete end-to-end audiobook synthesis pipeline")
    parser.add_argument("--manuscript", type=str, default=None, help="Path to input text manuscript")
    parser.add_argument("--input", type=str, default=None, help="Fallback path to input text manuscript")
    parser.add_argument("--output", type=str, default="scratch/pipeline_workspace/output_master.wav", help="Path to output mastered ACX audio")
    parser.add_argument("--profile", type=str, default="standard", help="Target mixing profile (standard, dramatic)")
    parser.add_argument("--chapters", type=str, default=None, help="Chapter(s) to process, e.g. '1', '1,2,3', or '1-3'")
    parser.add_argument("--enable-llm-enrichment", action="store_true", help="Opt-in: enrich Tier 1 output with cloud/local LLM speaker attribution (Gemini free tier -> Groq -> Ollama -> off). No effect outside production_tier == 1.")
    
    # Character Voice Drawer registration commands
    parser.add_argument("--register", type=str, default=None, help="Register a character Drawer name in MemPalace")
    parser.add_argument("--wav", type=str, default=None, help="Path to reference mono WAV file for register")
    parser.add_argument("--speed", type=float, default=1.0, help="XTTS speed pacing multiplier")
    parser.add_argument("--pitch", type=float, default=0.0, help="Voice pitch shift offset")
    parser.add_argument("--energy_bias", type=float, default=0.0, help="Voice energy multiplier bias")
    
    args = parser.parse_args()
    
    # Handle character drawer registration
    if args.register:
        if not args.wav:
            print("Error: Registering a character Drawer requires a target reference WAV path (--wav).")
            sys.exit(1)
        palace = MemPalace()
        success = palace.register_character(
            character_name=args.register,
            voice_ref_path=args.wav,
            speed=args.speed,
            pitch=args.pitch,
            energy_bias=args.energy_bias
        )
        palace.close()
        sys.exit(0 if success else 1)
        
    # Handle end-to-end pipeline executions
    if args.run_all:
        input_path = args.manuscript if args.manuscript else args.input
        if not input_path:
            input_path = "scratch/audit_manuscript.txt"
            
        pipeline = FirespeakerPipeline()
        # Verify input manuscript exists, or create a mock one
        if not os.path.exists(input_path):
            logger.info(f"Creating default audit manuscript at: {input_path}")
            os.makedirs(os.path.dirname(input_path), exist_ok=True)
            with open(input_path, "w", encoding="utf-8") as f:
                f.write("""
                Chapter 1: The Inciting Incident
                Sherlock Holmes stood in the center of the room.
                "Watson, come here quickly! [laughs] Look at this creaking door," said Holmes.
                Watson replied, "No, it is too dangerous."
                """)
                
        success = pipeline.run_full_pipeline(input_path, args.output, args.profile, chapters=args.chapters, enable_llm_enrichment=args.enable_llm_enrichment)
        sys.exit(0 if success else 1)
        
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
