#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Firespeaker Voice Cloning Studio -- backend for the guided wizard.

The dataset builder (src/voice_dataset.py) is the engine; this module adds the
interactive session layer the wizard drives:

    start      -> create/resume a session (dataset dir + prompt plan)
    record     -> accept a browser recording for one prompt, convert, QC, store
    questionnaire -> subjective self-assessment; drafts the listing description
    build      -> assemble reference sets from accepted clips
    preview    -> synthesize a test line with the caller's own clone (XTTS)
    persona    -> named character versions of the voice (pitch/speed/style)
    sfx        -> vocal sound-effect bank (growls, whooshes, creature calls)
    publish    -> consent-gated hand-off to the voice marketplace

Recordings are subjective; QC is not. Every clip gets the same measured
PASS/FLAG/REJECT verdict the CLI intake applies -- the wizard just makes
re-recording a one-click loop instead of a filesystem chore.
"""

import base64
import json
import logging
import os
import re
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

from src.voice_dataset import (
    DATASET_ROOT, _prompts, _qc_clip, _read_wav_mono, cmd_build, cmd_init,
)

logger = logging.getLogger("VoiceStudio")

PREVIEW_DIR = "scratch/voice_studio"
_SYNTH_LOCK = threading.Lock()   # XTTS is not thread-safe; one preview at a time
_SFX_PROMPTS = [
    ("growl", "A low creature growl, menacing and sustained"),
    ("wind", "A wind whoosh made with your breath"),
    ("impact", "A percussive impact or thud sound with your mouth"),
    ("bird", "A birdcall or chirp imitation"),
    ("hiss", "A snake-like or steam-like hiss"),
    ("roar", "A big creature roar, as loud as your room allows"),
]


def _slug(name: str) -> Optional[str]:
    s = re.sub(r"[^a-z0-9_\-]", "", (name or "").lower().replace(" ", "_"))
    return s or None


def _dir(session: str) -> str:
    return os.path.join(DATASET_ROOT, session)


def _state_path(session: str) -> str:
    return os.path.join(_dir(session), "studio_session.json")


def _load_state(session: str) -> Optional[Dict[str, Any]]:
    p = _state_path(session)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _save_state(session: str, state: Dict[str, Any]) -> None:
    state["updated_at"] = time.time()
    with open(_state_path(session), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def start_session(name: str, speaker: str) -> Optional[Dict[str, Any]]:
    session = _slug(name)
    if not session:
        return None
    state = _load_state(session)
    if state is None:
        cmd_init(session, speaker or session)
        state = {
            "session": session, "speaker": speaker or session,
            "clips": {},          # prompt_id -> latest QC record
            "questionnaire": {},
            "personas": [],
            "sfx": {},            # tag -> {path, duration_s}
            "built": False, "published": False,
            "created_at": time.time(),
        }
        _save_state(session, state)
    plan = [{"prompt_id": pid, "emotion": emo, "text": text} for pid, emo, text in _prompts()]
    return {"state": state, "prompts": plan, "sfx_prompts": [{"tag": t, "text": d} for t, d in _SFX_PROMPTS]}


# ---------------------------------------------------------------------------
# Recording intake (browser blob -> wav -> QC -> raw/)
# ---------------------------------------------------------------------------

def _decode_data_url(data_url: str) -> bytes:
    return base64.b64decode(data_url.split(",", 1)[1])


def save_recording(session: str, prompt_id: str, data_url: str, kind: str = "prompt") -> Optional[Dict[str, Any]]:
    """kind: 'prompt' (scripted clip), 'roomtone', or 'sfx' (prompt_id = tag)."""
    session = _slug(session)
    state = _load_state(session or "")
    if state is None or "," not in (data_url or ""):
        return None
    d = _dir(session)
    os.makedirs(os.path.join(d, "raw"), exist_ok=True)

    blob = os.path.join(d, "raw", f"_upload_{prompt_id}.bin")
    with open(blob, "wb") as f:
        f.write(_decode_data_url(data_url))

    # Browsers deliver webm/opus (or wav); normalize container to 48k mono wav.
    if kind == "sfx":
        os.makedirs(os.path.join(d, "sfx"), exist_ok=True)
        wav = os.path.join(d, "sfx", f"{_slug(prompt_id)}.wav")
    elif kind == "roomtone":
        wav = os.path.join(d, "raw", "roomtone.wav")
    else:
        known = {pid for pid, _, _ in _prompts()}
        if prompt_id not in known:
            os.remove(blob)
            return None
        wav = os.path.join(d, "raw", f"{prompt_id}.wav")
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-i", blob,
                        "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", wav], check=True)
    finally:
        os.remove(blob)

    audio, rate = _read_wav_mono(wav)
    if kind == "sfx":
        dur = len(audio) / max(rate, 1)
        if dur < 0.3:
            os.remove(wav)
            return {"verdict": "REJECT", "problems": ["too short"], "duration_s": round(dur, 2)}
        state["sfx"][_slug(prompt_id)] = {"path": wav, "duration_s": round(dur, 2)}
        _save_state(session, state)
        return {"verdict": "PASS", "duration_s": round(dur, 2), "path": wav}
    if kind == "roomtone":
        import numpy as np
        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
        floor_db = 20 * np.log10(rms) if rms > 0 else None
        state["roomtone_dbfs"] = round(floor_db, 1) if floor_db is not None else None
        _save_state(session, state)
        return {"verdict": "PASS" if (floor_db is None or floor_db < -45) else "FLAG",
                "noise_floor_dbfs": state["roomtone_dbfs"],
                "problems": [] if (floor_db is None or floor_db < -45) else
                            [f"room is loud ({floor_db:.0f} dBFS) -- find somewhere quieter if you can"]}

    qc = _qc_clip(audio, rate, state.get("roomtone_dbfs"))
    prompts = {pid: (emo, text) for pid, emo, text in _prompts()}
    qc["prompt_id"], qc["emotion"], qc["transcript"] = prompt_id, prompts[prompt_id][0], prompts[prompt_id][1]
    if qc["verdict"] == "REJECT":
        os.remove(wav)   # keep raw/ clean; the wizard offers instant re-record
    state["clips"][prompt_id] = qc
    state["built"] = False
    _save_state(session, state)

    # keep the CLI-compatible qc_report.json in sync so `build` sees wizard clips
    accepted = {pid: c for pid, c in state["clips"].items() if c["verdict"] != "REJECT"}
    with open(os.path.join(d, "qc_report.json"), "w", encoding="utf-8") as f:
        json.dump({"noise_floor_dbfs": state.get("roomtone_dbfs"), "clips": accepted,
                   "intake_at": time.time()}, f, indent=2)
    return qc


# ---------------------------------------------------------------------------
# Questionnaire -> listing description draft
# ---------------------------------------------------------------------------

def save_questionnaire(session: str, answers: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    session = _slug(session)
    state = _load_state(session or "")
    if state is None:
        return None
    state["questionnaire"] = answers
    parts = []
    if answers.get("age_range"): parts.append(f"{answers['age_range']}")
    if answers.get("gender"): parts.append(answers["gender"])
    if answers.get("accent"): parts.append(f"{answers['accent']} accent")
    if answers.get("tone_words"): parts.append(str(answers["tone_words"]))
    if answers.get("register"): parts.append(f"suits {answers['register']}")
    draft = ("A " + ", ".join(p for p in parts if p) + " voice.") if parts else ""
    state["listing_description_draft"] = draft
    _save_state(session, state)
    return {"description_draft": draft}


# ---------------------------------------------------------------------------
# Build + clone preview
# ---------------------------------------------------------------------------

def build(session: str) -> Optional[Dict[str, Any]]:
    session = _slug(session)
    state = _load_state(session or "")
    if state is None:
        return None
    if not os.path.exists(os.path.join(_dir(session), "qc_report.json")):
        return {"built": False, "error": "No accepted recordings yet."}
    cmd_build(session)
    ref = os.path.join(_dir(session), "refs", "reference_mono.wav")
    ok = os.path.exists(ref)
    state["built"] = ok
    _save_state(session, state)
    emo = [f for f in os.listdir(os.path.join(_dir(session), "refs")) if f.startswith("emotion_")] if ok else []
    return {"built": ok, "reference": ref if ok else None, "emotion_refs": emo,
            "accepted_clips": sum(1 for c in state["clips"].values() if c["verdict"] != "REJECT")}


def _drawer_name(session: str) -> str:
    return f"VoiceStudio {session}"


def preview(session: str, text: str, pitch: float = 0.0, speed: float = 1.0) -> Optional[Dict[str, Any]]:
    """Synthesize `text` with the caller's own cloned voice. Serialized: XTTS
    on CPU takes tens of seconds and is not thread-safe."""
    session = _slug(session)
    state = _load_state(session or "")
    ref = os.path.join(_dir(session or ""), "refs", "reference_mono.wav")
    if state is None or not os.path.exists(ref):
        return None
    text = (text or "").strip()[:300] or "This is my cloned voice, speaking inside Volcano Studios."

    os.makedirs(PREVIEW_DIR, exist_ok=True)
    out = os.path.join(PREVIEW_DIR, f"{session}_preview_{int(time.time())}.wav")
    with _SYNTH_LOCK:
        from src.voice_synthesizer import VoiceSynthesizer
        synth = VoiceSynthesizer(force_cpu=True)
        drawer = _drawer_name(session)
        if not synth.palace.get_character_drawer(drawer):
            synth.palace.register_character(character_name=drawer, voice_ref_path=ref,
                                            speed=1.0, pitch=0.0)
        synth.synthesize_line(character_name=drawer, dialogue_text=text,
                              target_emotion="Neutral", output_wav_path=out)
    if not (os.path.exists(out) and os.path.getsize(out) > 0):
        return None
    if pitch or speed != 1.0:
        from src.tts_compiler import get_ffmpeg_filters
        import wave as _wave
        with _wave.open(out, "rb") as w:
            rate = w.getframerate()
        shifted = out.replace(".wav", "_mod.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-i", out,
                        "-af", get_ffmpeg_filters(speed, pitch, rate), shifted], check=True)
        out = shifted
    return {"wav": out, "text": text}


# ---------------------------------------------------------------------------
# Personas: character versions of the same voice
# ---------------------------------------------------------------------------

def save_persona(session: str, label: str, description: str,
                 pitch: float = 0.0, speed: float = 1.0) -> Optional[Dict[str, Any]]:
    session = _slug(session)
    state = _load_state(session or "")
    if state is None or not (label or "").strip():
        return None
    persona = {"label": label.strip()[:60], "description": (description or "").strip()[:300],
               "pitch": max(-6.0, min(6.0, float(pitch))), "speed": max(0.6, min(1.6, float(speed))),
               "created_at": time.time()}
    state["personas"] = [p for p in state["personas"] if p["label"] != persona["label"]] + [persona]
    _save_state(session, state)
    return {"personas": state["personas"]}


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------

def publish(session: str, seller: str, description: str, price: float, consent: bool,
            marketplace) -> Optional[Dict[str, Any]]:
    session = _slug(session)
    state = _load_state(session or "")
    ref = os.path.join(_dir(session or ""), "refs", "reference_mono.wav")
    if state is None or not os.path.exists(ref):
        return None
    if not consent:
        raise ValueError("Publishing requires explicit consent.")
    desc = (description or state.get("listing_description_draft") or "").strip()
    if state["personas"]:
        desc += " Personas: " + ", ".join(p["label"] for p in state["personas"]) + "."
    if state["sfx"]:
        desc += " Vocal SFX bank: " + ", ".join(sorted(state["sfx"])) + "."
    listing = marketplace.onboard_voice(
        seller_name=seller or state["speaker"], voice_name=state["speaker"],
        sample_wav_paths=[ref], description=desc, price_usd=float(price or 0.0),
        consent_confirmed=True)
    consent_path = os.path.join(_dir(session), "consent.json")
    if os.path.exists(consent_path):
        with open(consent_path, encoding="utf-8") as f:
            record = json.load(f)
        record["statement"] = f"{seller or state['speaker']} consents to license this voice via Volcano Studios marketplace."
        record["signed_at"] = time.time()
        with open(consent_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
    state["published"] = True
    state["listing"] = {k: listing.get(k) for k in ("voice_id", "voice_name", "voice_ref_path") if isinstance(listing, dict)}
    _save_state(session, state)
    return {"listing": listing, "description": desc}
