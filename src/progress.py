#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Firespeaker Analysis Progress Tracker

Every pipeline stage reports its position (book, stage, current/total) here;
state is written atomically to data/analysis_progress.json so progress is
checkable in real time from anywhere -- CLI, another terminal, or (later) the
GUI, which reads the same file.

Check progress:
    python -m src.progress            # one-shot status table
    python -m src.progress --watch    # live-updating view

Reporting never raises: a progress failure must not break the pipeline.
"""

import os
import sys
import json
import time
import argparse
from typing import Any, Dict, Optional

PROGRESS_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "data", "analysis_progress.json",
)

# Stage weights for the whole-pipeline percentage (rough share of wall-clock)
STAGE_ORDER = [
    "tier1_structure",      # loops 1-3 (parts/chapters/scenes) -- fast
    "g4_scene_segmentation",
    "tier2_enrichment",     # attribution/clean-check per scene
    "crew_direction",       # spotting + music + dialogue per scene
    "sound_design",
    "dramatization",
    "mixing",               # synthesis + assembly per scene
]


def _load() -> Dict[str, Any]:
    try:
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(state: Dict[str, Any]) -> None:
    tmp = PROGRESS_PATH + ".tmp"
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, PROGRESS_PATH)


def report(book: str, stage: str, current: int, total: int, detail: str = "") -> None:
    """Record progress for one stage of one book. Never raises."""
    try:
        state = _load()
        entry = state.setdefault(book, {"stages": {}, "started_at": time.time()})
        pct = round(100.0 * current / total, 1) if total else 0.0
        entry["stages"][stage] = {
            "current": current, "total": total, "pct": pct,
            "detail": detail[:120], "updated_at": time.time(),
        }
        entry["active_stage"] = stage
        entry["updated_at"] = time.time()
        _save(state)
    except Exception:
        pass


def finish(book: str, stage: str) -> None:
    try:
        state = _load()
        st = state.get(book, {}).get("stages", {}).get(stage)
        if st:
            st["current"] = st["total"] = max(st["total"], st["current"], 1)
            st["pct"] = 100.0
            st["updated_at"] = time.time()
            _save(state)
    except Exception:
        pass


def clear(book: str) -> None:
    try:
        state = _load()
        state.pop(book, None)
        _save(state)
    except Exception:
        pass


def _render(state: Dict[str, Any]) -> str:
    if not state:
        return "(no analysis in progress)"
    out = []
    now = time.time()
    for book, entry in sorted(state.items(), key=lambda kv: -kv[1].get("updated_at", 0)):
        age = now - entry.get("updated_at", now)
        live = "RUNNING" if age < 120 else f"idle {int(age//60)}m"
        out.append(f"\n{book}  [{live}]")
        stages = entry.get("stages", {})
        ordered = [s for s in STAGE_ORDER if s in stages] + [s for s in stages if s not in STAGE_ORDER]
        for stage in ordered:
            st = stages[stage]
            bar_len = 28
            filled = int(bar_len * st["pct"] / 100)
            bar = "#" * filled + "-" * (bar_len - filled)
            marker = " <-- " + st.get("detail", "") if stage == entry.get("active_stage") and age < 120 else ""
            out.append(f"  {stage:22s} [{bar}] {st['pct']:5.1f}%  ({st['current']}/{st['total']}){marker}")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Firespeaker analysis progress viewer")
    parser.add_argument("--watch", action="store_true", help="Live-updating view (refreshes every 3s)")
    args = parser.parse_args()

    if not args.watch:
        print(_render(_load()))
        return
    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            print(f"Firespeaker Analysis Progress  ({time.strftime('%H:%M:%S')})")
            print(_render(_load()))
            time.sleep(3)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
