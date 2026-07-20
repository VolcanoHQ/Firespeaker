#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Caldera Engine Boot Sequence -- environment doctor + self-healing startup.

Run at server start (and as `python -m src.boot_check`): verifies the runtime
the pipeline depends on -- binaries, models, GPU driver state, provider keys,
data stores, disk -- and reports ok/warn/fail per check instead of letting the
first render discover a broken environment an hour in.

The GPU check exists because of a measured failure: WSL's libcuda/libnvidia-ml
shims can wedge mid-session, after which merely IMPORTING torch segfaults the
process. The probe therefore runs in a SUBPROCESS (a segfault there is a
readable exit code, not a dead server), and when the driver is wedged the boot
sequence SELF-HEALS: it compiles tiny stub libraries that shadow the broken
shims (CPU-only mode) and re-execs the server with them preloaded. A WSL
restart restores the GPU; the stubs live in the repo's data dir and are only
used while the driver is actually broken.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger("BootCheck")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

STUB_DIR = os.path.abspath("data/runtime_stubs")
REPORT_PATH = "data/boot_report.json"
_REEXEC_FLAG = "CALDERA_BOOT_REEXEC"

_CUDA_STUB_C = "void cuInit(void){}\n"
_NVML_STUB_C = """int nvmlInit_v2(void){return 1;}
int nvmlInit(void){return 1;}
int nvmlShutdown(void){return 1;}
int nvmlDeviceGetCount_v2(void *c){return 1;}
int nvmlDeviceGetHandleByIndex_v2(unsigned int i, void *h){return 1;}
"""


def _check(name: str, status: str, detail: str) -> Dict[str, str]:
    return {"check": name, "status": status, "detail": detail}


def _probe_torch(extra_env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Import torch in a subprocess: a wedged WSL driver segfaults the probe,
    never the caller. Returns {'ok', 'cuda', 'wedged', 'detail'}."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')"],
            capture_output=True, text=True, timeout=120, env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "cuda": False, "wedged": False, "detail": "torch import timed out"}
    if r.returncode == 0:
        cuda = "CUDA" in r.stdout
        return {"ok": True, "cuda": cuda, "wedged": False,
                "detail": "GPU available" if cuda else "CPU mode"}
    wedged = r.returncode in (-11, 139) or "Segmentation" in (r.stderr or "")
    return {"ok": False, "cuda": False, "wedged": wedged,
            "detail": f"exit {r.returncode}" + (" (segfault -- GPU driver shim wedged)" if wedged else "")}


def _build_stubs() -> Optional[str]:
    """Compile the shadow libraries. Returns the stub dir, or None without a compiler."""
    cc = shutil.which("gcc") or shutil.which("cc")
    if not cc:
        return None
    os.makedirs(STUB_DIR, exist_ok=True)
    for name, src in (("libcuda.so.1", _CUDA_STUB_C), ("libnvidia-ml.so.1", _NVML_STUB_C)):
        out = os.path.join(STUB_DIR, name)
        if os.path.exists(out):
            continue
        src_path = os.path.join(STUB_DIR, name + ".c")
        with open(src_path, "w") as f:
            f.write(src)
        subprocess.run([cc, "-shared", "-fPIC", "-o", out, src_path], check=True, capture_output=True)
    return STUB_DIR


def ensure_torch_safe() -> Dict[str, Any]:
    """Call FIRST in a long-lived entry point, before anything imports torch.
    If the GPU driver is wedged, builds the stubs and RE-EXECS the current
    process with them preloaded (LD_LIBRARY_PATH is read once at process
    start, so an in-process fix is impossible -- re-exec is the honest one)."""
    probe = _probe_torch()
    if probe["ok"]:
        return probe
    if not probe["wedged"] or os.environ.get(_REEXEC_FLAG):
        return probe  # not the wedge, or we already re-execed once -- don't loop
    stub = _build_stubs()
    if not stub:
        probe["detail"] += "; no compiler to build stubs -- torch will crash this process"
        return probe
    healed = _probe_torch({"LD_LIBRARY_PATH": stub + ":" + os.environ.get("LD_LIBRARY_PATH", "")})
    if not healed["ok"]:
        probe["detail"] += "; stub shadowing did not recover torch"
        return probe
    logger.warning("GPU driver shim is wedged (WSL). Re-execing in CPU-only stub mode; "
                   "restart WSL to restore the GPU.")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = stub + ":" + env.get("LD_LIBRARY_PATH", "")
    env[_REEXEC_FLAG] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def run_boot_checks(fast: bool = False) -> Dict[str, Any]:
    checks: List[Dict[str, str]] = []

    # 1. Interpreter / env
    checks.append(_check("python", "ok", f"{sys.version.split()[0]} @ {sys.executable}"))

    # 2. Media binaries
    for binary in ("ffmpeg", "ffprobe"):
        path = shutil.which(binary)
        checks.append(_check(binary, "ok" if path else "fail", path or "not on PATH -- audio assembly cannot run"))

    # 3. GPU / torch (subprocess probe; skipped in fast mode -- it costs ~10s)
    if not fast:
        probe = _probe_torch()
        if probe["ok"]:
            status = "ok" if probe["cuda"] else "warn"
            detail = probe["detail"] + ("" if probe["cuda"] else
                                        " (synthesis/generation run slower; restart WSL if the GPU should be present)")
        else:
            status = "fail"
            detail = probe["detail"]
        if os.environ.get(_REEXEC_FLAG):
            status, detail = "warn", "running in CPU-only stub mode (GPU driver was wedged at boot; restart WSL to restore)"
        policy = os.getenv("CALDERA_TTS_DEVICE", "auto")
        detail += f" | TTS device policy: {policy}"
        checks.append(_check("torch/gpu", status, detail))

    # 4. Voice / model assets
    tts_cache = os.path.expanduser("~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2")
    checks.append(_check("xtts_model", "ok" if os.path.isdir(tts_cache) else "warn",
                         tts_cache if os.path.isdir(tts_cache) else "not downloaded yet -- first synthesis will fetch ~2GB"))
    narrator = "data/voice_references/narrator_mono.wav"
    checks.append(_check("narrator_reference", "ok" if os.path.exists(narrator) else "fail",
                         narrator if os.path.exists(narrator) else "missing default narrator reference"))

    # 5. LLM providers (Tier 1 needs none of these -- warn, never fail)
    gem, groq = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")), bool(os.getenv("GROQ_API_KEY"))
    checks.append(_check("gemini_key", "ok" if gem else "warn", "present" if gem else "absent (Tier 2 enrichment degrades to Groq/Ollama)"))
    checks.append(_check("groq_key", "ok" if groq else "warn", "present" if groq else "absent"))
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1.0)
        checks.append(_check("ollama", "ok", "reachable"))
    except Exception:
        checks.append(_check("ollama", "warn", "not reachable (offline fallback unavailable)"))

    # 6. Data stores
    for name, path in (("mempalace_db", "data/mempalace/palace_relational.db"),
                       ("projects_db", "data/projects.db"),
                       ("users_db", "data/users.db")):
        checks.append(_check(name, "ok" if os.path.exists(path) else "warn",
                             path if os.path.exists(path) else "will be created on first use"))
    try:
        os.makedirs("data", exist_ok=True)
        probe_file = "data/.boot_write_probe"
        with open(probe_file, "w") as f:
            f.write("ok")
        os.remove(probe_file)
        checks.append(_check("data_writable", "ok", "data/ writable"))
    except Exception as e:
        checks.append(_check("data_writable", "fail", f"data/ not writable: {e}"))

    # 7. Disk
    du = shutil.disk_usage(".")
    free_gb = du.free / 1e9
    checks.append(_check("disk_free", "ok" if free_gb > 10 else ("warn" if free_gb > 2 else "fail"),
                         f"{free_gb:.1f} GB free"))

    # 8. Housekeeping: reap render jobs whose workers died
    try:
        from src.render_job import list_jobs
        jobs = list_jobs()
        active = sum(1 for j in jobs if j.get("status") in ("queued", "running"))
        checks.append(_check("render_jobs", "ok", f"{len(jobs)} job(s) on record, {active} active (dead workers reaped)"))
    except Exception as e:
        checks.append(_check("render_jobs", "warn", f"job sweep failed: {e}"))

    worst = "ok"
    for c in checks:
        if c["status"] == "fail":
            worst = "fail"
            break
        if c["status"] == "warn":
            worst = "warn"
    report = {"status": worst, "at": time.time(), "checks": checks}
    try:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    except Exception:
        pass
    return report


def print_report(report: Dict[str, Any]) -> None:
    icons = {"ok": "✓", "warn": "!", "fail": "✗"}
    print(f"\n=== CALDERA ENGINE BOOT CHECK: {report['status'].upper()} ===")
    for c in report["checks"]:
        print(f"  [{icons[c['status']]}] {c['check']:20} {c['detail']}")
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    print_report(run_boot_checks())
