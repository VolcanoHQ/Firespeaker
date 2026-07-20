#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Caldera Engine Ingestion Test Suite
Runs bulk Stage 1 Ingestion (Clutter Scrubber) across the entire corpus
to verify start boundaries and generate a verification report.
"""

import os
import json
from nlp_engine.stage_1_ingestion import ClutterScrubber

# Search for the corpus in standard locations
CORPUS_DIR = "data/corpus" if os.path.exists("data/corpus") else "./corpus"
REPORT_FILE = "ingestion_report.json"

def run_bulk_ingestion_test():
    report = {}
    scrubber = ClutterScrubber()

    print(f"Starting bulk Stage 1 ingestion test on {CORPUS_DIR}...\n")
    
    if not os.path.exists(CORPUS_DIR):
        print(f"❌ Error: Corpus directory '{CORPUS_DIR}' does not exist.")
        return

    for filename in os.listdir(CORPUS_DIR):
        if filename.endswith(".txt") or filename.endswith(".docx"):
            filepath = os.path.join(CORPUS_DIR, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    raw_text = file.read()
                
                # Run the Stage 1 scrubber
                clean_text = scrubber.remove_front_matter(raw_text)
                
                # Extract just the first 200 characters of the CLEANED text
                # so a human can easily verify the starting point.
                starting_snippet = clean_text[:200].replace('\n', ' ')
                
                report[filename] = {
                    "status": "Processed",
                    "first_detected_sentence": starting_snippet
                }
                print(f"✅ {filename} -> Starts with: '{starting_snippet[:50]}...'")
            except Exception as e:
                print(f"❌ {filename} -> Error: {e}")
                report[filename] = {
                    "status": "Failed",
                    "error": str(e)
                }

    # Save to a JSON report for review
    with open(REPORT_FILE, 'w', encoding='utf-8') as out:
        json.dump(report, out, indent=4)
        
    print(f"\nTest complete. Review {REPORT_FILE} to verify start boundaries.")

if __name__ == "__main__":
    run_bulk_ingestion_test()
