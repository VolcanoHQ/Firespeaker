#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Caldera Engine Render Jobs -- the one-button path from manuscript to audiobook.

start_render() spawns a DETACHED worker process (survives server restarts) that
runs: ingest (with --resume-enrichment semantics for tier >= 2, so quota
interruptions never waste prior work) -> the tier's mixer -> chaptered export
(wav + m4b + line-timing manifest, from src/production_mixer). Job records live
one-file-per-job under data/render_jobs/ and carry an `owner` field (default
"local") so tier-2 user management maps onto existing data instead of a
migration.

Tier 3 renders only when the scene_director crew's artifacts exist -- a book
without direction fails fast with guidance (same policy as tier preview),
never a silent downgrade.
"""

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RenderJob")

JOBS_DIR = "data/render_jobs"
RENDERS_DIR = "scratch/renders"
CORPUS_ROOTS = ("data/corpus", "data/uploads")
SOURCE_EXTS = (".txt", ".epub")


def _job_path(job_id: str) -> str:
    return os.path.join(JOBS_DIR, f"{job_id}.json")


def _write_job(job: Dict[str, Any]) -> None:
    os.makedirs(JOBS_DIR, exist_ok=True)
    tmp = _job_path(job["job_id"]) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)
    os.replace(tmp, _job_path(job["job_id"]))


def _read_job(job_id: str) -> Optional[Dict[str, Any]]:
    try:
        with open(_job_path(job_id), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def find_source(book: str) -> Optional[str]:
    """Locate the manuscript whose stem matches the pipeline book name."""
    for root in CORPUS_ROOTS:
        for ext in SOURCE_EXTS:
            hits = glob.glob(os.path.join(root, "**", f"{book}{ext}"), recursive=True)
            if hits:
                return hits[0]
    return None


def list_jobs(book: Optional[str] = None) -> List[Dict[str, Any]]:
    jobs = []
    if os.path.isdir(JOBS_DIR):
        for p in sorted(glob.glob(os.path.join(JOBS_DIR, "*.json")), reverse=True):
            try:
                with open(p, encoding="utf-8") as f:
                    j = json.load(f)
            except Exception:
                continue
            # a queued/running job whose worker died is failed, not stuck forever
            # (queued covers a worker killed before it wrote its running state)
            if j.get("status") in ("queued", "running") and j.get("pid") and not _pid_alive(j.get("pid")):
                j["status"] = "failed"
                j["error"] = j.get("error") or "Worker process died unexpectedly."
                _write_job(j)
            if book is None or j.get("book") == book:
                jobs.append(j)
    return jobs


def start_render(book: str, tier: int, owner: str = "local",
                 project_id: Optional[str] = None) -> Dict[str, Any]:
    """Create the job record and spawn the detached worker. Returns the record
    (status 'failed' immediately for input problems -- no zombie jobs)."""
    if tier not in (1, 2, 3):
        return {"status": "failed", "error": f"Unknown tier {tier}"}
    source = find_source(book)
    if not source:
        return {"status": "failed", "error": f"No manuscript named '{book}' under {CORPUS_ROOTS}"}
    running = [j for j in list_jobs(book) if j.get("status") in ("queued", "running")]
    if running:
        return {"status": "failed", "error": f"A render for '{book}' is already {running[0]['status']} "
                                             f"(job {running[0]['job_id']}).", "job": running[0]}
    if tier == 3 and not os.path.exists(
            os.path.join("data/corpus/pipeline", book, "tier3", "production_script.json")):
        return {"status": "failed",
                "error": "Tier 3 needs the scene_director crew's artifacts -- run the crew first "
                         "(tier 1/2 render available now)."}

    job = {
        "job_id": uuid.uuid4().hex[:12],
        "book": book, "source_file": source, "tier": tier, "owner": owner or "local",
        "project_id": project_id,
        "status": "queued", "created_at": time.time(),
        "started_at": None, "finished_at": None, "pid": None,
        "output_wav": None, "output_m4b": None, "timings": None, "error": None,
    }
    _write_job(job)
    os.makedirs(RENDERS_DIR, exist_ok=True)
    log_path = os.path.join(RENDERS_DIR, f"{job['job_id']}.log")
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "src.render_job", "--run", job["job_id"]],
            stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, cwd=os.getcwd(), env=os.environ.copy())
    job["pid"] = proc.pid
    _write_job(job)
    logger.info(f"Render job {job['job_id']} started for '{book}' tier {tier} (pid {proc.pid}).")
    return job


def run_job(job_id: str) -> int:
    """Worker entry point (runs in the detached process)."""
    job = _read_job(job_id)
    if not job:
        logger.error(f"No such job: {job_id}")
        return 1
    job.update(status="running", started_at=time.time(), pid=os.getpid())
    _write_job(job)
    try:
        from src.llm_client import set_usage_context
        plan = None
        if job.get("project_id"):
            try:
                from src.project_db import ProjectDB
                project = ProjectDB().get_product_project(job["project_id"])
                plan = (project or {}).get("plan")
            except Exception:
                pass
        set_usage_context(book=job["book"], owner=job.get("owner"),
                          project_id=job.get("project_id"), plan=plan)
        from src.tier_1_parser import ingest_manuscript_tier_1
        tier = job["tier"]
        manifest = ingest_manuscript_tier_1(
            job["source_file"],
            enable_llm_enrichment=(tier >= 2),
            resume_enrichment=(tier >= 2),
        )
        manifest_path = os.path.join(RENDERS_DIR, f"{job['book']}_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))

        out_wav = os.path.join(RENDERS_DIR, f"{job['book']}_tier{tier}.wav")
        from src import production_mixer as pm
        if tier == 3:
            result = pm.mix_production(manifest_path, out_wav)
        else:
            result = pm.mix_voice_track(manifest_path, out_wav, single_narrator=(tier == 1))
        job.update(status="done", finished_at=time.time(),
                   output_wav=result.get("output"), output_m4b=result.get("m4b"),
                   timings=result.get("timings"))
        _write_job(job)
        logger.info(f"Render job {job_id} DONE: {result.get('output')}")
        return 0
    except Exception as e:
        logger.exception(f"Render job {job_id} failed")
        job.update(status="failed", finished_at=time.time(), error=str(e)[:500])
        _write_job(job)
        return 1


def main():
    p = argparse.ArgumentParser(description="Caldera Engine render job runner")
    p.add_argument("--run", metavar="JOB_ID", help="Execute a queued job (worker mode)")
    p.add_argument("--start", metavar="BOOK", help="Queue + spawn a render for a book")
    p.add_argument("--tier", type=int, default=1)
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.run:
        sys.exit(run_job(a.run))
    elif a.start:
        print(json.dumps(start_render(a.start, a.tier), indent=2))
    elif a.list:
        print(json.dumps(list_jobs(), indent=2))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
