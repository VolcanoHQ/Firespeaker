#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility script to parse a single text block (paragraph) through Firespeaker's hierarchical parser.
It leverages the existing ManuscriptAnalyzer.parse_manuscript_for_segment method to produce
the detailed Studio Script block, including the confidence score.

Usage:
    python -m src.single_block_parser "<text>"

Example:
    python -m src.single_block_parser "\"You must follow me carefully,\" said the Time Traveller."
"""

import sys
from src.hierarchical_parser import HierarchicalParser

def parse_single_block(text: str):
    parser = HierarchicalParser()
    # Use a dummy file name and default chapter/scene numbers for isolated processing
    result = parser.analyzer.parse_manuscript_for_segment(
        segment_text=text,
        file_name="single_block.txt",
        chapter_num=1,
        scene_num=1,
        characters_list=[],
        merge_map=None,
    )
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a text block to parse as a command‑line argument.")
        sys.exit(1)
    block = sys.argv[1]
    output = parse_single_block(block)
    import json
    print(json.dumps(output, indent=2, ensure_ascii=False))
