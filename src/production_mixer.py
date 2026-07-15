#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Firespeaker Production Mixer (Chain D: Graphic-Audio assembly)

Assembles a Tier 3 full-production audiobook from:
  - per-line character voice WAVs (XTTS, cached or synthesized on demand)
  - the scene_director production script (music direction, stingers, SFX anchors)
  - music/ambience/SFX assets -- PLACEHOLDER-FIRST: ffmpeg-generated tones and the
    two existing library assets stand in until Chain C (MusicGen/AudioGen) supplies
    real ones. The timeline mechanics (line-anchored events -> timestamps ->
    ducked multi-track mix) are identical either way; only asset resolution swaps.

Timeline model: every stinger/SFX references a line index within its scene; line WAV
durations (+ post-padding) are accumulated into offsets, so "after line 4" becomes an
exact timestamp mechanically -- no LLM anywhere in this stage.

Usage:
  python -m src.production_mixer --manifest scratch/book.json --output scratch/book_tier3.wav
"""

import os
import re
import sys
import json
import wave
import logging
import argparse
import subprocess
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import ManuscriptManifest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ProductionMixer")

WORKSPACE = "scratch/pipeline_workspace/tier3_mix"
AMBIENT_ASSET = "data/ambient/room_tone_ambience_ucs_AMB.wav"


_FOLEY_TOKENS = {
    "thud", "thuds", "creak", "creaks", "splash", "splaash", "gurgle", "rustle",
    "flap", "scratch", "scritch", "scr-r-ritch", "squeak", "thump", "bang",
    "whoosh", "click", "clink", "clank", "scrape", "crash", "crunch", "sizzle",
    "crackle", "swish", "patter", "plink", "plop", "knock", "rattle", "scramble",
    "scurry", "scurrying", "scrambling", "tap", "shuffle",
}
_VOCAL_TOKENS = {
    "oh", "ohh", "ohhh", "ah", "ahh", "mmm", "mm", "mmph", "mmmph", "ooh", "boo",
    "boo-hoo", "huff", "puff", "achoo", "kertyschoo", "eep", "whoa", "hmph", "hmm",
    "uuurp", "urp", "gulp", "sigh", "yawn", "sob", "sniff", "sniffle", "wail",
    "gasp", "groan", "zzz", "hhhmm", "nom", "chomp", "smack", "ugh", "argh", "hey",
}


def _is_foley_only(text: str) -> bool:
    """True when a 'performance_vocal' is really object/impact onomatopoeia
    ('Thud, thud, thud!') that must be GENERATED as sound, not spoken by a voice
    actor -- the measured failure mode: XTTS reading 'thud' aloud in Peter's voice.
    Any mouth-performable interjection token keeps it on the voice track."""
    words = [re.sub(r"[^a-z\-]", "", w.lower()) for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return False
    if any(w in _VOCAL_TOKENS or w.rstrip("h") in _VOCAL_TOKENS for w in words):
        return False
    foley_hits = sum(1 for w in words if w in _FOLEY_TOKENS)
    return foley_hits >= max(1, len(words) // 2)


def _run_ffmpeg(cmd: List[str]) -> None:
    subprocess.run(["ffmpeg", "-y", *cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def _wav_duration(path: str) -> float:
    with wave.open(path) as w:
        return w.getnframes() / float(w.getframerate())


# ----------------------------------------------------
# Placeholder asset factory (Chain C replaces this)
# ----------------------------------------------------

def make_placeholder_music_bed(duration_sec: float, mood: str, out_path: str) -> str:
    """Soft filtered-noise pad as a stand-in music bed. Mood only nudges the filter
    frequency so different moods are at least audibly distinct in review."""
    mood_l = mood.lower()
    if any(w in mood_l for w in ("tense", "alarm", "urgent", "dark", "fear", "cautionary")):
        cutoff = 400
    elif any(w in mood_l for w in ("joy", "playful", "whimsical", "mischievous", "sweet")):
        cutoff = 1200
    else:
        cutoff = 800
    _run_ffmpeg([
        "-f", "lavfi",
        "-i", f"anoisesrc=color=brown:duration={duration_sec:.2f}:sample_rate=22050",
        "-af", f"lowpass=f={cutoff},volume=0.55,afade=t=in:d=1.5,afade=t=out:st={max(duration_sec - 1.5, 0):.2f}:d=1.5",
        "-ac", "1", out_path,
    ])
    return out_path


def make_placeholder_stinger(out_path: str) -> str:
    """1.2s low sine swell -- stands in for e.g. 'a brief, dark cello note'."""
    _run_ffmpeg([
        "-f", "lavfi", "-i", "sine=frequency=196:duration=1.2:sample_rate=22050",
        "-af", "volume=0.85,afade=t=in:d=0.15,afade=t=out:st=0.6:d=0.6",
        "-ac", "1", out_path,
    ])
    return out_path


def make_placeholder_sfx(description: str, out_path: str) -> str:
    """0.6s noise burst placeholder; reuses the laughter asset when apt."""
    if "laugh" in description.lower() and os.path.exists("data/sfx/laughter_generic_ucs_LAUGH.wav"):
        _run_ffmpeg(["-i", "data/sfx/laughter_generic_ucs_LAUGH.wav", "-t", "2.0", "-af", "volume=0.8", "-ac", "1", "-ar", "22050", out_path])
        return out_path
    _run_ffmpeg([
        "-f", "lavfi", "-i", "anoisesrc=color=pink:duration=0.6:sample_rate=22050",
        "-af", "volume=0.8,afade=t=in:d=0.05,afade=t=out:st=0.3:d=0.3",
        "-ac", "1", out_path,
    ])
    return out_path


def make_ambience_loop(duration_sec: float, out_path: str) -> Optional[str]:
    if not os.path.exists(AMBIENT_ASSET):
        return None
    _run_ffmpeg([
        "-stream_loop", "-1", "-i", AMBIENT_ASSET,
        "-t", f"{duration_sec:.2f}",
        "-af", "volume=0.25", "-ac", "1", "-ar", "22050", out_path,
    ])
    return out_path


# ----------------------------------------------------
# Voice line resolution (cache-or-synthesize)
# ----------------------------------------------------

def resolve_line_wavs(lines: List[Dict[str, Any]], synth, overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Tuple[str, Dict[str, Any]]]:
    """Returns [(wav_path, line_dict)] for every line, synthesizing on cache miss.
    Uses the same cache-name convention as tts_compiler so prior runs are reused.

    `overrides` (from {book}/tier3/line_overrides.json, keyed by line_id) supports
    human-in-the-loop production edits: {"text": ...} replaces the synthesized text
    (may contain [pause:X] markup the synthesizer honors). Overridden lines use a
    distinct cache name so edits actually take effect."""
    outputs_dir = "scratch/pipeline_workspace/outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    overrides = overrides or {}
    resolved = []
    for line in lines:
        char = line["character"]
        char_slug = re.sub(r'[^a-zA-Z0-9_\-]', '', char)
        emotion = line.get("emotion", "Neutral")
        override = overrides.get(line["line_id"])
        text = override["text"] if override and override.get("text") else line["text"]
        if override:
            text_tag = "ov" + __import__("hashlib").sha1(text.encode()).hexdigest()[:8]
            wav = os.path.join(outputs_dir, f"line_{line['line_id']}_{char_slug}_tier3_{text_tag}_{emotion}.wav")
        else:
            wav = os.path.join(outputs_dir, f"line_{line['line_id']}_{char_slug}_tier3_{emotion}.wav")
        if not (os.path.exists(wav) and os.path.getsize(wav) > 0):
            synth.synthesize_line(
                character_name=char,
                dialogue_text=text,
                target_emotion=emotion,
                output_wav_path=wav,
            )
        resolved.append((wav, line))
    return resolved


# ----------------------------------------------------
# Scene assembly
# ----------------------------------------------------

def assemble_scene(scene_id: str, line_wavs: List[Tuple[str, Dict[str, Any]]], direction: Dict[str, Any], sfx_cues: List[Dict[str, Any]], out_path: str, sound_design: Optional[Dict[str, Any]] = None) -> str:
    """Concatenates voice lines, computes line-anchored event timestamps, and mixes
    voice + music bed (ducked) + ambience + stingers/SFX into one scene WAV."""
    scene_dir = os.path.join(WORKSPACE, scene_id)
    os.makedirs(scene_dir, exist_ok=True)

    # 1. Voice track: concat lines with their post-padding as silence
    concat_parts = []
    offsets: List[float] = []  # start time of each line
    t = 0.0
    for wav, line in line_wavs:
        offsets.append(t)
        dur = _wav_duration(wav)
        pad_ms = int(line.get("post_padding_ms", 250))
        padded = os.path.join(scene_dir, f"pad_{os.path.basename(wav)}")
        _run_ffmpeg(["-i", wav, "-af", f"apad=pad_dur={pad_ms/1000:.3f}", "-ac", "1", "-ar", "22050", padded])
        concat_parts.append(padded)
        t += dur + pad_ms / 1000.0

    concat_list = os.path.join(scene_dir, "concat.txt")
    with open(concat_list, "w") as f:
        for p in concat_parts:
            f.write(f"file '{os.path.abspath(p)}'\n")
    voice_track = os.path.join(scene_dir, "voice.wav")
    _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", voice_track])
    total_dur = _wav_duration(voice_track)

    # 2. Event timeline: stingers fire after their anchor line ends; SFX cues are
    # placed after the line whose text contains (or is nearest to) their sound_text
    from src.audio_generation import generate_stinger, generate_music_bed
    events: List[Tuple[float, str]] = []  # (timestamp, asset_path)
    for i, s in enumerate(direction.get("music", {}).get("stingers", [])):
        idx = s["after_line_index"]
        if 0 <= idx < len(line_wavs):
            end_of_line = offsets[idx] + _wav_duration(line_wavs[idx][0])
            sting_path = os.path.join(scene_dir, f"stinger_{i}.wav")
            if generate_stinger(s.get("description", "dramatic musical accent"), sting_path) is None:
                sting_path = make_placeholder_stinger(sting_path)
            events.append((end_of_line, sting_path))
    if sound_design and sound_design.get("events"):
        # Layered sound design: each event is a composite of generated component
        # layers (foley-style), placed at its anchor line. Creature sounds carry
        # emotional intent into the generation prompt ("sparrows chirping rapidly,
        # urgent encouraging tone").
        from src.audio_generation import compose_layered_sfx
        for i, ev in enumerate(sound_design["events"]):
            idx = ev["anchor_line_index"]
            if not (0 <= idx < len(line_wavs)):
                continue
            anchor = offsets[idx] + _wav_duration(line_wavs[idx][0])
            layers = ev["layers"]
            if ev.get("category") == "creature" and ev.get("emotional_intent"):
                layers = [
                    {**l, "component": f"{l['component']}, {ev['emotional_intent']} tone"}
                    for l in layers
                ]
            composite_path = os.path.join(scene_dir, f"composite_{i}.wav")
            if compose_layered_sfx(layers, composite_path) is not None:
                events.append((anchor, composite_path))
                logger.info(f"Composite event '{ev['name']}' ({len(layers)} layers) anchored at {anchor:.1f}s")
            else:
                events.append((anchor, make_placeholder_sfx(ev["name"], composite_path)))
    else:
        for i, cue in enumerate(sfx_cues):
            cue_norm = re.sub(r"\s+", " ", cue["sound_text"].lower())
            # Guard: a "cue" whose text IS a dialogue line is speech misflagged as SFX
            # (e.g. "Stop thief!" flagged as "man shouting") -- overlaying noise on the
            # actor's own delivery is exactly wrong, so skip it entirely.
            if any(
                line["segment_type"] == "dialogue"
                and cue_norm in re.sub(r"\s+", " ", line["text"].lower())
                for _, line in line_wavs
            ):
                logger.info(f"Skipping SFX cue that duplicates spoken dialogue: {cue['sound_text'][:40]!r}")
                continue
            anchor = 0.0
            for j, (wav, line) in enumerate(line_wavs):
                if cue_norm[:30] in re.sub(r"\s+", " ", line["text"].lower()):
                    anchor = offsets[j] + _wav_duration(wav)
                    break
            sfx_path = make_placeholder_sfx(cue["description"], os.path.join(scene_dir, f"sfx_{i}.wav"))
            events.append((anchor, sfx_path))

    # 3. Beds: real MusicGen from the director's style prompt, placeholder fallback.
    # Music state machine: 'stop'/'resume'/'change' events segment the bed timeline
    # (the gold's "Music stops abruptly. Dead silence." / "resumes instantly").
    music = direction.get("music", {})
    base_style = music.get("style", "soft ambient underscore")
    base_mood = music.get("base_mood", "neutral")

    def _bed_prompt(style: str) -> str:
        return f"{style}, {base_mood} mood, instrumental, no vocals"

    music_events = music.get("events", [])
    music_bed = os.path.join(scene_dir, "music_bed.wav")
    if not music_events:
        if generate_music_bed(_bed_prompt(base_style), total_dur, music_bed) is None:
            music_bed = make_placeholder_music_bed(total_dur, base_mood, music_bed)
    else:
        # Build segments: [(start_ts, end_ts, style_or_None)]; None = silence
        seg_bounds = [0.0]
        seg_styles: List[Optional[str]] = [base_style]
        for ev in music_events:
            idx = ev["after_line_index"]
            if not (0 <= idx < len(line_wavs)):
                continue
            ts = offsets[idx] + _wav_duration(line_wavs[idx][0])
            action = ev["action"]
            new = None if action == "stop" else (ev.get("new_style") or base_style)
            if ts <= seg_bounds[-1] + 1.0:
                seg_styles[-1] = new
                continue
            seg_bounds.append(ts)
            seg_styles.append(new)
        seg_bounds.append(total_dur)

        seg_files = []
        for i in range(len(seg_styles)):
            seg_dur = seg_bounds[i + 1] - seg_bounds[i]
            if seg_dur < 0.3:
                continue
            seg_path = os.path.join(scene_dir, f"bed_seg_{i}.wav")
            if seg_styles[i] is None:
                _run_ffmpeg(["-f", "lavfi", "-i", f"anullsrc=r=22050:cl=mono", "-t", f"{seg_dur:.2f}", seg_path])
            else:
                if generate_music_bed(_bed_prompt(seg_styles[i]), seg_dur, seg_path) is None:
                    make_placeholder_music_bed(seg_dur, base_mood, seg_path)
            seg_files.append(seg_path)
        seg_list = os.path.join(scene_dir, "bed_segs.txt")
        with open(seg_list, "w") as f:
            for p in seg_files:
                f.write(f"file '{os.path.abspath(p)}'\n")
        _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", seg_list, "-ar", "22050", "-ac", "1", music_bed])
        logger.info(f"Scene {scene_id}: segmented music bed ({len(seg_files)} segment(s), {sum(1 for s in seg_styles if s is None)} silence window(s)).")

    # Scene-specific generated ambience (from the sound design's continuous layers)
    # beats the generic room-tone loop when available.
    ambience = None
    if sound_design and sound_design.get("continuous_ambience"):
        from src.audio_generation import generate_sfx
        amb_prompt = ", ".join(sound_design["continuous_ambience"]) + ", soft continuous background ambience, gentle field recording, smooth"
        amb_clip = generate_sfx(amb_prompt, 8.0, os.path.join(scene_dir, "ambience_clip.wav"), steps=50)
        if amb_clip:
            # Tame diffusion harshness (lowpass) and de-click the loop point
            # (fade the clip's own edges) before looping to scene length.
            smooth_clip = os.path.join(scene_dir, "ambience_clip_smooth.wav")
            clip_dur = _wav_duration(amb_clip)
            _run_ffmpeg([
                "-i", amb_clip,
                "-af", f"lowpass=f=6000,highpass=f=80,afade=t=in:d=0.25,afade=t=out:st={max(clip_dur-0.25,0):.2f}:d=0.25",
                "-ac", "1", "-ar", "22050", smooth_clip,
            ])
            ambience = os.path.join(scene_dir, "ambience.wav")
            _run_ffmpeg([
                "-stream_loop", "-1", "-i", smooth_clip, "-t", f"{total_dur:.2f}",
                "-af", f"loudnorm=I=-36:TP=-8,afade=t=in:d=1.5,afade=t=out:st={max(total_dur-1.5,0):.2f}:d=1.5",
                "-ac", "1", "-ar", "22050", ambience,
            ])
    if ambience is None:
        ambience = make_ambience_loop(total_dur, os.path.join(scene_dir, "ambience.wav"))

    # 4. Mix: music ducked under voice (sidechain), ambience constant-low,
    # events overlaid at their timestamps via adelay.
    inputs = ["-i", voice_track, "-i", music_bed]
    filters = ["[1:a][0:a]sidechaincompress=threshold=0.15:ratio=2:attack=100:release=300[ducked]"]
    mix_labels = ["[0:a]", "[ducked]"]
    idx = 2
    if ambience:
        inputs += ["-i", ambience]
        mix_labels.append(f"[{idx}:a]")
        idx += 1
    for ts, asset in events:
        inputs += ["-i", asset]
        delay_ms = int(ts * 1000)
        filters.append(f"[{idx}:a]adelay={delay_ms}|{delay_ms}[ev{idx}]")
        mix_labels.append(f"[ev{idx}]")
        idx += 1
    filters.append(f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=first:normalize=0[out]")
    _run_ffmpeg([*inputs, "-filter_complex", ";".join(filters), "-map", "[out]", "-ac", "1", "-ar", "22050", out_path])
    logger.info(f"Scene {scene_id}: {total_dur:.1f}s, {len(events)} timeline event(s) placed.")
    return out_path


# ----------------------------------------------------
# Orchestration
# ----------------------------------------------------

def mix_tier1(manifest_path: str, output_path: str) -> Dict[str, Any]:
    """Tier 1 assembly: the whole manuscript read by ONE narrator voice.

    No music, no SFX, no per-character casting -- every line (dialogue included)
    is synthesized as the Narrator drawer, concatenated in manuscript order with
    each line's own post_padding, an extra beat between chapters, then the same
    ACX mastering chain as Tier 3. Line-level emotion is kept: it modulates the
    narrator's pitch/speed slightly (a single voice with cadence, not a cast).
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = ManuscriptManifest.model_validate_json(f.read())
    book_stem = os.path.splitext(manifest.source_file)[0]

    from src.voice_synthesizer import VoiceSynthesizer
    synth = VoiceSynthesizer(force_cpu=True)
    if not synth.palace.get_character_drawer("Narrator"):
        synth.palace.register_character(
            character_name="Narrator",
            voice_ref_path="data/voice_references/narrator_mono.wav",
            speed=1.0, pitch=0.0,
        )

    os.makedirs(WORKSPACE, exist_ok=True)
    from src import progress as _progress
    total_scenes = sum(len(ch.scenes) for p in manifest.parts for ch in p.chapters)
    done = 0

    CHAPTER_GAP_MS = 1500
    segments: List[Tuple[str, int]] = []  # (wav_path, trailing_silence_ms)
    for part in manifest.parts:
        for chapter in part.chapters:
            for scene in chapter.scenes:
                lines = []
                for l in scene.lines:
                    d = l.model_dump()
                    d["character"] = "Narrator"
                    d["speaker_id"] = "char_narrator"
                    lines.append(d)
                done += 1
                _progress.report(book_stem, "tier1_narration", done, total_scenes, scene.scene_id)
                for wav, line in resolve_line_wavs(lines, synth):
                    segments.append((wav, int(line.get("post_padding_ms") or 0)))
            if segments:
                segments[-1] = (segments[-1][0], segments[-1][1] + CHAPTER_GAP_MS)
    _progress.finish(book_stem, "tier1_narration")

    if not segments:
        raise RuntimeError("Manifest produced no narration lines.")

    # Concat with per-line trailing silence. Silence chunks are generated once per
    # distinct duration at the voice track's own sample format so stream params match.
    import wave as _wave
    with _wave.open(segments[0][0], "rb") as w:
        rate, channels = w.getframerate(), w.getnchannels()
    silence_cache: Dict[int, str] = {}

    def _silence(ms: int) -> str:
        if ms not in silence_cache:
            p = os.path.join(WORKSPACE, f"tier1_silence_{ms}ms_{rate}.wav")
            _run_ffmpeg(["-f", "lavfi", "-i", f"anullsrc=r={rate}:cl={'mono' if channels == 1 else 'stereo'}",
                         "-t", f"{ms / 1000.0:.3f}", "-c:a", "pcm_s16le", p])
            silence_cache[ms] = p
        return silence_cache[ms]

    concat_list = os.path.join(WORKSPACE, "tier1_segments.txt")
    with open(concat_list, "w") as f:
        for wav, pad_ms in segments:
            f.write(f"file '{os.path.abspath(wav)}'\n")
            if pad_ms > 0:
                f.write(f"file '{os.path.abspath(_silence(pad_ms))}'\n")
    _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", concat_list, "-c:a", "pcm_s16le", "-ar", str(rate), "-ac", str(channels), output_path])

    from src.audio_mixer import AudioMixer
    mixer = AudioMixer()
    mixer.apply_post_mastering(output_path, profile_name="standard")
    compliance = mixer.verify_acx_compliance(output_path)
    logger.info(f"Tier 1 narration master: {output_path} ({_wav_duration(output_path)/60:.1f} min, {len(segments)} lines)")
    return {"output": output_path, "lines": len(segments), "acx": compliance}


def mix_production(manifest_path: str, output_path: str) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = ManuscriptManifest.model_validate_json(f.read())

    book_stem = os.path.splitext(manifest.source_file)[0]
    tier3_dir = os.path.join("data/corpus/pipeline", book_stem, "tier3")
    tier1_dir = os.path.join("data/corpus/pipeline", book_stem, "tier1")

    with open(os.path.join(tier3_dir, "production_script.json"), "r", encoding="utf-8") as f:
        directions = {d["scene_id"]: d for d in json.load(f)}
    sound_designs: Dict[str, Dict[str, Any]] = {}
    sd_path = os.path.join(tier3_dir, "sound_design.json")
    if os.path.exists(sd_path):
        with open(sd_path, "r", encoding="utf-8") as f:
            sound_designs = {d["scene_id"]: d for d in json.load(f)}
        logger.info(f"Loaded layered sound design for {len(sound_designs)} scene(s).")

    dramatizations: Dict[str, Dict[str, Any]] = {}
    dram_path = os.path.join(tier3_dir, "dramatization.json")
    if os.path.exists(dram_path):
        with open(dram_path, "r", encoding="utf-8") as f:
            dramatizations = {d["scene_id"]: d for d in json.load(f)}
        n_inserts = sum(len(d["inserts"]) for d in dramatizations.values())
        logger.info(f"Loaded {n_inserts} dramatized insert(s) across {len(dramatizations)} scene(s).")
    scenes_sfx: Dict[str, List[Dict[str, Any]]] = {}
    sfx_path = os.path.join(tier1_dir, "loopE_llm_sfx_cues.json")
    if os.path.exists(sfx_path):
        with open(sfx_path, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                scenes_sfx[entry["scene_id"]] = entry["sfx_cues"]

    os.makedirs(WORKSPACE, exist_ok=True)

    overrides: Dict[str, Dict[str, Any]] = {}
    overrides_path = os.path.join(tier3_dir, "line_overrides.json")
    if os.path.exists(overrides_path):
        with open(overrides_path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        logger.info(f"Loaded {len(overrides)} production line override(s).")

    from src.voice_synthesizer import VoiceSynthesizer
    synth = VoiceSynthesizer(force_cpu=True)

    # Auto-register dramatized minor-cast drawers (Sparrow 1, Old Mouse, ...) so
    # synthesis doesn't hit MissingDrawerError; each gets a distinct builtin voice
    # via the name-hash pool since their ref is the shared narrator sample.
    dram_characters = {
        ins["character"]
        for d in dramatizations.values()
        for ins in d["inserts"]
    }
    for char in sorted(dram_characters):
        if not synth.palace.get_character_drawer(char):
            logger.info(f"Registering dramatized minor character drawer: '{char}'")
            synth.palace.register_character(
                character_name=char,
                voice_ref_path="data/voice_references/narrator_mono.wav",
                speed=1.0, pitch=0.0,
            )

    import hashlib as _hashlib
    from src import progress as _progress
    _total_scenes = sum(len(ch.scenes) for p in manifest.parts for ch in p.chapters)
    _done = 0

    scene_wavs = []
    for part in manifest.parts:
        for chapter in part.chapters:
            for scene in chapter.scenes:
                lines = [l.model_dump() for l in scene.lines]

                # Splice dramatized inserts (additive, flagged) after their anchors.
                # Foley-only "vocals" (Thud, Creak) are rerouted to generated SFX
                # instead of being read aloud by a voice. Splicing shifts line
                # indices, so all index-anchored direction (stingers, music events,
                # sound events) is remapped through orig->spliced positions.
                d = dramatizations.get(scene.scene_id)
                index_map = list(range(len(lines)))
                extra_sfx_events: List[Dict[str, Any]] = []
                if d:
                    by_anchor: Dict[int, List[Dict[str, Any]]] = {}
                    for ins in d["inserts"]:
                        by_anchor.setdefault(ins["anchor_line_index"], []).append(ins)
                    spliced = []
                    index_map = []
                    for i, l in enumerate(lines):
                        index_map.append(len(spliced))
                        spliced.append(l)
                        for ins in by_anchor.get(i, []):
                            if ins["insert_type"] == "performance_vocal" and _is_foley_only(ins["text"]):
                                extra_sfx_events.append({
                                    "name": f"dramatized foley: {ins['text'][:30]}",
                                    "anchor_line_index": i,
                                    "category": "action",
                                    "emotional_intent": "",
                                    "layers": [{"component": ins.get("delivery") or ins["text"], "timing": "start", "level": "medium"}],
                                })
                                logger.info(f"Rerouting foley-only vocal to SFX: {ins['character']}: {ins['text'][:40]!r}")
                                continue
                            spliced.append({
                                "line_id": "dram_" + _hashlib.sha1((scene.scene_id + ins["character"] + ins["text"]).encode()).hexdigest()[:12],
                                "character": ins["character"],
                                "speaker_id": f"char_{ins['character'].lower().replace(' ', '_')}",
                                "text": ins["text"],
                                "segment_type": "dialogue",
                                "emotion": "Dramatized",
                                "post_padding_ms": 200,
                                "utterance_type": "vocalization" if ins["insert_type"] == "performance_vocal" else "speech",
                                "is_dramatized": True,
                            })
                    lines = spliced

                def _remap(idx: int) -> int:
                    return index_map[idx] if 0 <= idx < len(index_map) else idx

                direction = json.loads(json.dumps(directions.get(scene.scene_id, {"music": {"base_mood": "neutral", "stingers": []}})))
                for s in direction.get("music", {}).get("stingers", []):
                    s["after_line_index"] = _remap(s["after_line_index"])
                for ev in direction.get("music", {}).get("events", []):
                    ev["after_line_index"] = _remap(ev["after_line_index"])
                sound_design = sound_designs.get(scene.scene_id)
                if sound_design or extra_sfx_events:
                    sound_design = json.loads(json.dumps(sound_design)) if sound_design else {"continuous_ambience": [], "events": []}
                    for ev in sound_design.get("events", []):
                        ev["anchor_line_index"] = _remap(ev["anchor_line_index"])
                    for ev in extra_sfx_events:
                        ev["anchor_line_index"] = _remap(ev["anchor_line_index"])
                    sound_design["events"] = sound_design.get("events", []) + extra_sfx_events

                _done += 1
                _progress.report(book_stem, "mixing", _done, _total_scenes, scene.scene_id)
                line_wavs = resolve_line_wavs(lines, synth, overrides=overrides)
                scene_wav = assemble_scene(
                    scene.scene_id, line_wavs, direction,
                    scenes_sfx.get(scene.scene_id, []),
                    os.path.join(WORKSPACE, f"{scene.scene_id}_mixed.wav"),
                    sound_design=sound_design,
                )
                scene_wavs.append(scene_wav)

    concat_list = os.path.join(WORKSPACE, "scenes.txt")
    with open(concat_list, "w") as f:
        for p in scene_wavs:
            f.write(f"file '{os.path.abspath(p)}'\n")
    _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", output_path])

    # Master + ACX verification via the existing mixer
    from src.audio_mixer import AudioMixer
    mixer = AudioMixer()
    mixer.apply_post_mastering(output_path, profile_name="standard")
    compliance = mixer.verify_acx_compliance(output_path)

    logger.info(f"Tier 3 production master: {output_path} ({_wav_duration(output_path)/60:.1f} min)")
    return {"output": output_path, "scenes": len(scene_wavs), "acx": compliance}


def main():
    parser = argparse.ArgumentParser(description="Firespeaker Production Mixer (Chain D: Tier 3 assembly / Tier 1 narration)")
    parser.add_argument("--manifest", type=str, required=True, help="ManuscriptManifest JSON (Tier 3 additionally needs scene_director artifacts)")
    parser.add_argument("--output", type=str, required=True, help="Output master WAV path")
    parser.add_argument("--tier1", action="store_true", help="Single-narrator audiobook: one voice, no music/SFX/casting")
    args = parser.parse_args()
    if args.tier1:
        result = mix_tier1(args.manifest, args.output)
        print(f"\nTier 1 narration master: {result['output']} | lines: {result['lines']}")
    else:
        result = mix_production(args.manifest, args.output)
        print(f"\nTier 3 master: {result['output']} | scenes: {result['scenes']}")
    print(f"ACX: {json.dumps(result['acx'], indent=2, default=str)[:400]}")
    sys.exit(0)


if __name__ == "__main__":
    main()
