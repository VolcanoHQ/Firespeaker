#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Firespeaker Voice Dataset Builder.

The guided lane from "a person with a microphone" to a production voice asset:
    init    -> personalized recording script (prompt sheet)
    intake  -> mechanical QC of recorded clips (PASS/FLAG/REJECT with numbers)
    build   -> normalized dataset + manifest + zero-shot/emotion reference sets
    onboard -> hand the reference set to the voice marketplace

Methodology: docs/Firespeaker Voice Dataset Methodology.md. Scripted recording
means transcripts are known -- no ASR pass required.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import wave
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("VoiceDataset")

DATASET_ROOT = "data/voice_datasets"
TARGET_RATE = 24000

# ---------------------------------------------------------------------------
# Session script: Block A phonetic coverage, Block B domain match, Block C range
# ---------------------------------------------------------------------------

RAINBOW = [
    "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.",
    "The rainbow is a division of white light into many beautiful colors.",
    "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon.",
    "There is, according to legend, a boiling pot of gold at one end. People look, but no one ever finds it.",
]
HARVARD = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "The box was thrown beside the parked truck.",
]
DOMAIN = [
    "Once upon a time there were four little Rabbits, and their names were Flopsy, Mopsy, Cotton-tail, and Peter. They lived with their Mother in a sand-bank, underneath the root of a very big fir-tree.",
    "Peter was most dreadfully frightened; he rushed all over the garden, for he had forgotten the way back to the gate. He lost one of his shoes among the cabbages, and the other shoe amongst the potatoes.",
    "To Sherlock Holmes she is always the woman. I have seldom heard him mention her under any other name. In his eyes she eclipses and predominates the whole of her sex.",
    "The story had held us, round the fire, sufficiently breathless, but except the obvious remark that it was gruesome, as, on Christmas Eve in an old house, a strange tale should essentially be, I remember no comment uttered.",
    "It was a bright cold day in the garden, and the robin, who was watching, hopped closer along the wall, tilting his small head as if he understood every word that was said.",
    "He waited a long time, listening. At last, far away, he heard the slow scrape of a door, and footsteps on the gravel path, coming nearer and nearer through the dark.",
]
CARRIER = "I never expected to find it here, of all places, after everything that happened."
EMOTIONS = ["Neutral", "Joy", "Sadness", "Anger", "Fear", "Surprise", "Whisper", "Projected"]
CHARACTER_PROMPTS = [
    ("C09_character1", "A gruff old gardener, shouting across a field: \"Stop! Thief! Come back here with that, you little rascal!\""),
    ("C10_character2", "A gentle storyteller, drawing listeners close: \"Now, settle in, my dears, for this is where the tale takes a turn nobody saw coming.\""),
]


def _prompts() -> List[Tuple[str, str, str]]:
    """[(prompt_id, emotion_label, text)] for the whole session."""
    out = []
    for i, t in enumerate(RAINBOW + HARVARD, 1):
        out.append((f"A{i:02d}", "Neutral", t))
    for i, t in enumerate(DOMAIN, 1):
        out.append((f"B{i:02d}", "Neutral", t))
    for i, emo in enumerate(EMOTIONS, 1):
        out.append((f"C{i:02d}_{emo.lower()}", emo, CARRIER))
    for pid, t in CHARACTER_PROMPTS:
        out.append((pid, "Character", t))
    return out


def _dataset_dir(name: str) -> str:
    return os.path.join(DATASET_ROOT, name)


def cmd_init(name: str, speaker: str) -> None:
    d = _dataset_dir(name)
    for sub in ("raw", "clean", "refs"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    lines = [
        f"# Recording script for voice dataset '{name}' ({speaker})",
        "",
        "Read docs/Firespeaker Voice Dataset Methodology.md section 2 first (room, mic, format).",
        "Record ONE WAV PER PROMPT, named exactly by its id (e.g. A01.wav), into any folder.",
        f"Then run:  python -m src.voice_dataset intake --name {name} --input <that folder>",
        "",
        "FIRST: record 10 seconds of silence as roomtone.wav (used for noise-floor QC).",
        "",
    ]
    section = ""
    for pid, emo, text in _prompts():
        blk = pid[0]
        if blk != section:
            section = blk
            titles = {"A": "Block A — phonetic coverage (neutral, natural pace)",
                      "B": "Block B — audiobook narration (unhurried, expressive)",
                      "C": "Block C — emotional & character range"}
            lines += ["", f"## {titles[blk]}", ""]
        tag = f" **[{emo}]**" if emo not in ("Neutral",) else ""
        lines.append(f"- `{pid}`{tag}: {text}")
    path = os.path.join(d, "recording_script.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Dataset initialized. Recording script: {path}")
    print(f"\nYour prompt sheet is ready: {path}\n{len(_prompts())} prompts, ~15 minutes of speech.")


# ---------------------------------------------------------------------------
# Intake QC
# ---------------------------------------------------------------------------

def _read_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    """Decode any audio ffmpeg can read to float32 mono at native rate."""
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    rate = int(probe.stdout.strip() or 0)
    raw = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", path, "-f", "f32le", "-ac", "1", "-"],
        capture_output=True)
    audio = np.frombuffer(raw.stdout, dtype=np.float32)
    return audio, rate


def _qc_clip(audio: np.ndarray, rate: int, noise_floor_db: Optional[float]) -> Dict[str, Any]:
    dur = len(audio) / max(rate, 1)
    clip_frac = float(np.mean(np.abs(audio) >= 0.99)) if len(audio) else 1.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    rms_db = 20 * np.log10(rms) if rms > 0 else -120.0

    # frame-energy SNR estimate: 10th percentile (noise) vs median (speech)
    frame = max(1, int(0.03 * rate))
    n = len(audio) // frame
    snr_db = None
    if n > 10:
        energies = np.array([np.sqrt(np.mean(audio[i*frame:(i+1)*frame] ** 2)) for i in range(n)])
        energies = energies[energies > 0]
        if len(energies) > 10:
            noise = np.percentile(energies, 10)
            speech = np.percentile(energies, 60)
            if noise > 0:
                snr_db = float(20 * np.log10(speech / noise))

    problems, verdict = [], "PASS"
    if rate < 22050: problems.append(f"sample rate {rate} < 22050")
    if dur < 2.0: problems.append(f"too short ({dur:.1f}s)")
    if dur > 25.0: problems.append(f"too long ({dur:.1f}s)")
    if clip_frac > 0.001: problems.append(f"clipping ({clip_frac*100:.2f}% of samples)")
    if rms_db < -45: problems.append(f"too quiet ({rms_db:.1f} dBFS)")
    if rms_db > -6: problems.append(f"too hot ({rms_db:.1f} dBFS)")
    if snr_db is not None and snr_db < 12: problems.append(f"SNR {snr_db:.0f} dB < 12")
    if problems:
        verdict = "REJECT"
    elif (snr_db is not None and snr_db < 20) or rms_db < -35:
        verdict = "FLAG"
    return {"duration_s": round(dur, 2), "sample_rate": rate, "clip_frac": round(clip_frac, 5),
            "rms_dbfs": round(rms_db, 1), "snr_db": round(snr_db, 1) if snr_db is not None else None,
            "verdict": verdict, "problems": problems}


def cmd_intake(name: str, input_dir: str) -> None:
    d = _dataset_dir(name)
    if not os.path.isdir(d):
        sys.exit(f"Dataset '{name}' not initialized -- run init first.")
    prompts = {pid: (emo, text) for pid, emo, text in _prompts()}

    noise_floor_db = None
    roomtone = os.path.join(input_dir, "roomtone.wav")
    if os.path.exists(roomtone):
        audio, rate = _read_wav_mono(roomtone)
        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0
        noise_floor_db = 20 * np.log10(rms) if rms > 0 else None
        logger.info(f"Room tone noise floor: {noise_floor_db:.1f} dBFS" if noise_floor_db else "Room tone unreadable.")

    report, copied = {}, 0
    for fname in sorted(os.listdir(input_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in (".wav", ".flac", ".m4a", ".mp3") or stem == "roomtone":
            continue
        pid = stem if stem in prompts else None
        if pid is None:
            logger.warning(f"'{fname}' doesn't match any prompt id -- skipped (name files A01.wav, B02.wav...).")
            continue
        audio, rate = _read_wav_mono(os.path.join(input_dir, fname))
        qc = _qc_clip(audio, rate, noise_floor_db)
        qc["prompt_id"], qc["emotion"], qc["transcript"] = pid, prompts[pid][0], prompts[pid][1]
        report[pid] = qc
        if qc["verdict"] != "REJECT":
            shutil.copy2(os.path.join(input_dir, fname), os.path.join(d, "raw", f"{pid}{ext.lower()}"))
            copied += 1
        logger.info(f"{pid}: {qc['verdict']}" + (f" -- {'; '.join(qc['problems'])}" if qc["problems"] else ""))

    with open(os.path.join(d, "qc_report.json"), "w", encoding="utf-8") as f:
        json.dump({"noise_floor_dbfs": noise_floor_db, "clips": report, "intake_at": time.time()}, f, indent=2)
    missing = sorted(set(prompts) - set(report))
    print(f"\nIntake: {copied} clip(s) accepted, {sum(1 for r in report.values() if r['verdict']=='REJECT')} rejected.")
    print(f"QC report: {os.path.join(d, 'qc_report.json')}")
    if missing:
        print(f"Still missing {len(missing)} prompt(s): {', '.join(missing[:12])}{'...' if len(missing) > 12 else ''}")


# ---------------------------------------------------------------------------
# Build: normalize, manifest, reference sets
# ---------------------------------------------------------------------------

def cmd_build(name: str, denoise: bool = False) -> None:
    d = _dataset_dir(name)
    qc_path = os.path.join(d, "qc_report.json")
    if not os.path.exists(qc_path):
        sys.exit("No qc_report.json -- run intake first.")
    with open(qc_path, encoding="utf-8") as f:
        qc = json.load(f)["clips"]

    rows = []
    for pid, meta in sorted(qc.items()):
        if meta["verdict"] == "REJECT":
            continue
        src = next((os.path.join(d, "raw", pid + e) for e in (".wav", ".flac", ".m4a", ".mp3")
                    if os.path.exists(os.path.join(d, "raw", pid + e))), None)
        if not src:
            continue
        clean = os.path.join(d, "clean", f"{pid}.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "quiet", "-i", src,
             "-af", "loudnorm=I=-23:TP=-2:LRA=7", "-ar", str(TARGET_RATE), "-ac", "1",
             "-c:a", "pcm_s16le", clean], check=True)
        if denoise and meta["verdict"] == "FLAG":
            try:
                from src.voice_synthesizer import denoise_audio_file
                denoise_audio_file(clean, clean)
            except Exception as e:
                logger.warning(f"Denoise unavailable ({e}); keeping normalized clip.")
        rows.append((pid, meta["transcript"], meta["emotion"], meta["duration_s"],
                     meta.get("snr_db"), meta["verdict"]))

    with open(os.path.join(d, "manifest.csv"), "w", encoding="utf-8") as f:
        f.write("id|transcript|emotion|duration_s|snr_db|verdict\n")
        for r in rows:
            f.write("|".join(str(x) for x in r) + "\n")

    # Zero-shot reference: best PASS clips by SNR, narration register first (B block),
    # 8-20s total -- XTTS conditions well on that range.
    scored = sorted((r for r in rows if r[5] == "PASS"),
                    key=lambda r: ((0 if r[0].startswith("B") else 1), -(r[4] or 0)))
    picked, total = [], 0.0
    for r in scored:
        if total >= 18.0:
            break
        picked.append(r)
        total += float(r[3])
    if picked:
        listfile = os.path.join(d, "refs", "_concat.txt")
        with open(listfile, "w") as f:
            for r in picked:
                f.write(f"file '{os.path.abspath(os.path.join(d, 'clean', r[0] + '.wav'))}'\n")
        ref = os.path.join(d, "refs", "reference_mono.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-f", "concat", "-safe", "0",
                        "-i", listfile, "-c", "copy", ref], check=True)
        os.remove(listfile)
        logger.info(f"Zero-shot reference: {ref} ({total:.1f}s from {[r[0] for r in picked]})")

    for r in rows:
        if r[0].startswith("C") and r[2] not in ("Neutral", "Character"):
            shutil.copy2(os.path.join(d, "clean", r[0] + ".wav"),
                         os.path.join(d, "refs", f"emotion_{r[2].lower()}.wav"))

    # Consent/provenance: hash every raw file
    hashes = {}
    for fn in sorted(os.listdir(os.path.join(d, "raw"))):
        with open(os.path.join(d, "raw", fn), "rb") as f:
            hashes[fn] = hashlib.sha256(f.read()).hexdigest()
    with open(os.path.join(d, "consent.json"), "w", encoding="utf-8") as f:
        json.dump({"statement": "PENDING -- signed at onboard time", "created_at": time.time(),
                   "raw_sha256": hashes}, f, indent=2)

    emo_refs = [f for f in os.listdir(os.path.join(d, "refs")) if f.startswith("emotion_")]
    print(f"\nBuild complete: {len(rows)} clips in manifest, reference set "
          f"{'ready' if picked else 'MISSING (no PASS clips)'}, {len(emo_refs)} emotion reference(s).")


def cmd_onboard(name: str, seller: str, description: str, price: float, consent: bool) -> None:
    d = _dataset_dir(name)
    ref = os.path.join(d, "refs", "reference_mono.wav")
    if not os.path.exists(ref):
        sys.exit("No reference_mono.wav -- run build first.")
    if not consent:
        sys.exit("Onboarding requires --consent (the seller's explicit statement).")
    consent_path = os.path.join(d, "consent.json")
    if os.path.exists(consent_path):
        with open(consent_path, encoding="utf-8") as f:
            record = json.load(f)
        record["statement"] = f"{seller} consents to license this voice via Volcano Studios marketplace."
        record["signed_at"] = time.time()
        with open(consent_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
    from src.voice_marketplace import VoiceMarketplace
    mp = VoiceMarketplace()
    listing = mp.onboard_voice(
        seller_name=seller, voice_name=name, sample_wav_paths=[ref],
        description=description, price_usd=price, consent_confirmed=True)
    # attach the dataset path so the future fine-tuning tier knows where the corpus lives
    print(f"\nListed: {json.dumps(listing, indent=2, default=str)[:500]}")
    print(f"Dataset (fine-tune corpus grows here): {d}")


def main():
    p = argparse.ArgumentParser(description="Firespeaker Voice Dataset Builder")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init"); s.add_argument("--name", required=True); s.add_argument("--speaker", default="")
    s = sub.add_parser("intake"); s.add_argument("--name", required=True); s.add_argument("--input", required=True)
    s = sub.add_parser("build"); s.add_argument("--name", required=True); s.add_argument("--denoise", action="store_true")
    s = sub.add_parser("onboard"); s.add_argument("--name", required=True); s.add_argument("--seller", required=True)
    s.add_argument("--description", required=True); s.add_argument("--price", type=float, default=0.0)
    s.add_argument("--consent", action="store_true")
    a = p.parse_args()
    if a.cmd == "init": cmd_init(a.name, a.speaker or a.name)
    elif a.cmd == "intake": cmd_intake(a.name, a.input)
    elif a.cmd == "build": cmd_build(a.name, a.denoise)
    elif a.cmd == "onboard": cmd_onboard(a.name, a.seller, a.description, a.price, a.consent)


if __name__ == "__main__":
    main()
