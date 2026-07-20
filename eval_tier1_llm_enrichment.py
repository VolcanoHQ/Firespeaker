#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Caldera Engine Tier 1 LLM Enrichment Eval Harness

Runs Tier 1 ingestion with enable_llm_enrichment=True against a corpus book,
then diffs the LLM-attributed speaker/segment-type values against the
hand-authored gold-standard reference files in data/corpus/HumanProcessed/.

Kept separate from test_corpus_ingestion.py, which only exercises the
unrelated ClutterScrubber front-matter layer. This is a metrics report, not
a CI gate -- it always exits 0.
"""

import os
import re
import sys
import json
import time
import difflib
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tier_1_parser import ingest_manuscript_tier_1

BOOK_PATH = "data/corpus/TheTaleofPeterRabbit.txt"
GOLD_TIER1_PATH = "data/corpus/HumanProcessed/Tier 1/HP_Tier1_TheTaleofPeterRabbit.txt"
GOLD_TIER2_PATH = "data/corpus/HumanProcessed/Tier 2/HP_Tier2_TheTaleofPeterRabbit.txt"
REPORT_PATH = "scratch/eval_tier1_llm_enrichment_report.json"
AUDIT_LOG_PATH = "data/llm_call_audit.jsonl"

# Peter Rabbit-specific alias map: the roster heuristic and small local models
# often surface a plausible-but-different name for the same gold character
# (e.g. "Mother" for "Mrs. Rabbit"). Hand-curated, not meant to generalize.
SPEAKER_ALIASES = {
    "mother": "mrs. rabbit",
    "old mrs. rabbit": "mrs. rabbit",
    "mcgregor": "mr. mcgregor",
}

SCENE_HEADER_RE = re.compile(r'^\[Scene\s+(\d+).*\]$')
SPEAKER_HEADER_RE = re.compile(r'^([A-Z][A-Za-z.]+(?:\s[A-Z][A-Za-z.]+)*):\s*$')
QUOTE_LINE_RE = re.compile(r'^["“](.*)["”]$')


def normalize_speaker(name: str) -> str:
    key = name.strip().lower()
    return SPEAKER_ALIASES.get(key, key)


def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip().lower())


def parse_human_processed_gold(path: str) -> List[Dict[str, Any]]:
    """State-machine parser for the observed HumanProcessed format:
    '[Scene N - title]' starts a scene and resets speaker to Narrator,
    'Name:' on its own line sets the active speaker until the next such
    marker, and subsequent '"..."'-quoted lines become gold entries.
    """
    entries: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return entries

    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [l.rstrip("\n") for l in f]

    current_scene = 0
    current_speaker = "Narrator"

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue

        scene_match = SCENE_HEADER_RE.match(stripped)
        if scene_match:
            current_scene = int(scene_match.group(1))
            current_speaker = "Narrator"
            continue

        speaker_match = SPEAKER_HEADER_RE.match(stripped)
        if speaker_match:
            current_speaker = speaker_match.group(1).strip()
            continue

        quote_match = QUOTE_LINE_RE.match(stripped)
        if quote_match:
            entries.append({
                "scene": current_scene,
                "speaker": current_speaker,
                "text": quote_match.group(1).strip(),
            })

    return entries


def load_predicted_lines(book_path: str) -> Tuple[List[Dict[str, Any]], int]:
    manifest = ingest_manuscript_tier_1(book_path, enable_llm_enrichment=True)
    predicted = []
    for part in manifest.parts:
        for chapter in part.chapters:
            for scene in chapter.scenes:
                for line in scene.lines:
                    d = line.model_dump()
                    predicted.append(d)
    total_scenes = manifest.total_scenes
    return predicted, total_scenes


def count_gold_scenes(entries: List[Dict[str, Any]]) -> int:
    return len({e["scene"] for e in entries}) if entries else 0


def align_and_score(gold_entries: List[Dict[str, Any]], predicted: List[Dict[str, Any]], scene_counts_match: bool, threshold: float = 0.5) -> Dict[str, Any]:
    matched = 0
    unmatched_gold: List[Dict[str, Any]] = []
    speaker_correct = 0
    speaker_total = 0
    segment_type_agree = 0

    # Predicted lines already carry a 1-based "scene" field per ScriptLine.
    by_scene: Dict[int, List[Dict[str, Any]]] = {}
    for p in predicted:
        by_scene.setdefault(p["scene"], []).append(p)

    for gold in gold_entries:
        gold_norm = normalize_text(gold["text"])
        candidates = by_scene.get(gold["scene"], []) if scene_counts_match else predicted

        best_ratio = 0.0
        best_pred = None
        for pred in candidates:
            ratio = difflib.SequenceMatcher(None, gold_norm, normalize_text(pred["text"])).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pred = pred

        if best_ratio < threshold or best_pred is None:
            unmatched_gold.append(gold)
            continue

        matched += 1
        gold_is_dialogue = normalize_speaker(gold["speaker"]) != "narrator"
        pred_is_dialogue = best_pred["segment_type"] == "dialogue"
        if gold_is_dialogue == pred_is_dialogue:
            segment_type_agree += 1

        if gold_is_dialogue:
            speaker_total += 1
            if normalize_speaker(gold["speaker"]) == normalize_speaker(best_pred["character"]):
                speaker_correct += 1

    return {
        "gold_entries": len(gold_entries),
        "matched": matched,
        "unmatched_gold_count": len(unmatched_gold),
        "unmatched_gold_samples": unmatched_gold[:5],
        "segment_type_agreement_pct": round(100 * segment_type_agree / matched, 1) if matched else None,
        "dialogue_speaker_accuracy_pct": round(100 * speaker_correct / speaker_total, 1) if speaker_total else None,
        "dialogue_lines_in_gold": speaker_total,
        "dialogue_lines_correctly_attributed": speaker_correct,
    }


def check_boilerplate_cleancheck(book_base_name: str) -> Dict[str, Any]:
    path = f"data/corpus/pipeline/{book_base_name}/tier1/loopE_llm_cleancheck.json"
    if not os.path.exists(path):
        return {"status": "no cleancheck sidecar found"}
    with open(path, "r", encoding="utf-8") as f:
        issues = json.load(f)
    total_issues = sum(len(scene["issues"]) for scene in issues)
    if total_issues == 0:
        return {
            "status": "no boilerplate flagged (Peter Rabbit's own text may simply not contain any -- "
                      "the known leak examples were Franklin/Alice, not confirmed present in this book)",
            "scenes_flagged": 0,
            "total_issues": 0,
        }
    return {
        "status": f"{total_issues} issue(s) flagged across {len(issues)} scene(s)",
        "scenes_flagged": len(issues),
        "total_issues": total_issues,
        "sample_issues": issues[:3],
    }


def summarize_provider_usage(since_timestamp: float) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not os.path.exists(AUDIT_LOG_PATH):
        return counts
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("timestamp", 0) < since_timestamp:
                continue
            key = f"{record['provider']}:{'ok' if record['success'] else 'fail'}"
            counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> None:
    run_start = time.time()

    print(f"=== Tier 1 LLM Enrichment Eval: {BOOK_PATH} ===")
    print("Running ingest_manuscript_tier_1(enable_llm_enrichment=True)...")
    predicted, total_scenes = load_predicted_lines(BOOK_PATH)
    print(f"Predicted lines: {len(predicted)} across {total_scenes} scenes")

    gold_tier1 = parse_human_processed_gold(GOLD_TIER1_PATH)
    gold_tier2 = parse_human_processed_gold(GOLD_TIER2_PATH)
    print(f"Gold Tier 1 entries: {len(gold_tier1)} | Gold Tier 2 entries: {len(gold_tier2)}")

    tier1_scene_count = count_gold_scenes(gold_tier1)
    tier2_scene_count = count_gold_scenes(gold_tier2)

    tier1_scores = align_and_score(gold_tier1, predicted, scene_counts_match=(tier1_scene_count == total_scenes))
    tier2_scores = align_and_score(gold_tier2, predicted, scene_counts_match=(tier2_scene_count == total_scenes))

    boilerplate = check_boilerplate_cleancheck("TheTaleofPeterRabbit")
    provider_usage = summarize_provider_usage(run_start)

    report = {
        "book": BOOK_PATH,
        "predicted_total_lines": len(predicted),
        "predicted_total_scenes": total_scenes,
        "vs_gold_tier1": {**tier1_scores, "gold_scene_count": tier1_scene_count,
                           "note": "Tier 1 gold labels nearly everything Narrator -- accuracy here "
                                   "understates real improvement, see vs_gold_tier2 for the meaningful bar."},
        "vs_gold_tier2": {**tier2_scores, "gold_scene_count": tier2_scene_count,
                           "note": "Tier 2 gold has real per-character speaker splits; this is the headline metric."},
        "boilerplate_cleancheck": boilerplate,
        "provider_usage_this_run": provider_usage,
    }

    os.makedirs("scratch", exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== RESULTS ===")
    print(f"vs. Tier 1 gold: {tier1_scores['dialogue_speaker_accuracy_pct']}% dialogue speaker accuracy "
          f"({tier1_scores['dialogue_lines_correctly_attributed']}/{tier1_scores['dialogue_lines_in_gold']} lines), "
          f"segment-type agreement {tier1_scores['segment_type_agreement_pct']}%")
    print(f"vs. Tier 2 gold (headline): {tier2_scores['dialogue_speaker_accuracy_pct']}% dialogue speaker accuracy "
          f"({tier2_scores['dialogue_lines_correctly_attributed']}/{tier2_scores['dialogue_lines_in_gold']} lines), "
          f"segment-type agreement {tier2_scores['segment_type_agreement_pct']}%")
    print(f"Boilerplate clean-check: {boilerplate['status']}")
    print(f"Provider usage this run: {provider_usage}")
    print(f"\nFull report written to: {REPORT_PATH}")

    sys.exit(0)


if __name__ == "__main__":
    main()
