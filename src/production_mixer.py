#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Caldera Engine Production Mixer (Chain D: Graphic-Audio assembly)

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
import hashlib
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

def _voice_fingerprint(synth, character: str, memo: Dict[str, str]) -> str:
    """Short hash of everything that determines a character's rendered VOICE:
    reference wav, pinned builtin speaker, and pitch/speed modulation. Recasting
    a character changes the fingerprint, which invalidates its cached line wavs
    -- without this, a recast silently reuses audio in the OLD voice."""
    if character in memo:
        return memo[character]
    fp = "novoice"
    try:
        drawer = synth.palace.get_character_drawer(character)
        if drawer:
            mod = drawer.get("modulation_config") or {}
            if isinstance(mod, str):
                try:
                    mod = json.loads(mod)
                except Exception:
                    mod = {}
            key = "|".join(str(x) for x in (
                drawer.get("voice_ref_path"), mod.get("xtts_speaker"),
                mod.get("speed"), mod.get("pitch")))
            fp = hashlib.sha1(key.encode()).hexdigest()[:8]
    except Exception as e:
        logger.warning(f"Voice fingerprint unavailable for '{character}' ({e}); using shared key.")
    memo[character] = fp
    return fp


def resolve_line_wavs(lines: List[Dict[str, Any]], synth, overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Tuple[str, Dict[str, Any]]]:
    """Returns [(wav_path, line_dict)] for every line, synthesizing on cache miss.

    Cache names carry a voice fingerprint (see _voice_fingerprint) so recasting a
    character invalidates exactly that character's cached audio. Legacy cache files
    from the pre-fingerprint convention are adopted (renamed) on first touch --
    a one-time migration that avoids resynthesizing whole books.

    `overrides` (from {book}/tier3/line_overrides.json, keyed by line_id) supports
    human-in-the-loop production edits: {"text": ...} replaces the synthesized text
    (may contain [pause:X] markup the synthesizer honors). Overridden lines use a
    distinct cache name so edits actually take effect."""
    outputs_dir = "scratch/pipeline_workspace/outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    overrides = overrides or {}
    fp_memo: Dict[str, str] = {}
    resolved = []
    for line in lines:
        char = line["character"]
        char_slug = re.sub(r'[^a-zA-Z0-9_\-]', '', char)
        emotion = line.get("emotion", "Neutral")
        fp = _voice_fingerprint(synth, char, fp_memo)
        override = overrides.get(line["line_id"])
        text = override["text"] if override and override.get("text") else line["text"]
        if override:
            text_tag = "ov" + hashlib.sha1(text.encode()).hexdigest()[:8]
            stem = f"line_{line['line_id']}_{char_slug}_tier3_{text_tag}_{emotion}"
        else:
            stem = f"line_{line['line_id']}_{char_slug}_tier3_{emotion}"
        wav = os.path.join(outputs_dir, f"{stem}_v{fp}.wav")
        legacy = os.path.join(outputs_dir, f"{stem}.wav")
        if not (os.path.exists(wav) and os.path.getsize(wav) > 0):
            if os.path.exists(legacy) and os.path.getsize(legacy) > 0:
                # pre-fingerprint cache: adopt under the CURRENT voice once
                os.replace(legacy, wav)
            else:
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

def assemble_scene(scene_id: str, line_wavs: List[Tuple[str, Dict[str, Any]]], direction: Dict[str, Any], sfx_cues: List[Dict[str, Any]], out_path: str, sound_design: Optional[Dict[str, Any]] = None,
                   mix_overrides: Optional[Dict[str, Any]] = None) -> str:
    """Concatenates voice lines, computes line-anchored event timestamps, and mixes
    voice + music bed (ducked) + ambience + stingers/SFX into one scene WAV.

    mix_overrides (the console mixer's write layer, {book}/tier3/mix_overrides.json):
    {"events": {"event:<i>"|"stinger:<i>"|"cue:<i>": {mute, gain_db, nudge_s}},
     "lanes": {"voice"|"music"|"ambience"|"sfx": {mute, gain_db}}}.
    Human mix decisions apply at assembly time only -- artifacts stay pure."""
    scene_dir = os.path.join(WORKSPACE, scene_id)
    os.makedirs(scene_dir, exist_ok=True)

    ov_events = (mix_overrides or {}).get("events", {})
    ov_lanes = (mix_overrides or {}).get("lanes", {})

    def _ev_params(key: str, base_ts: float) -> Optional[Tuple[float, float]]:
        """(timestamp, linear_gain) for an event after overrides; None = muted."""
        ov = ov_events.get(key, {})
        lane = ov_lanes.get("sfx", {}) if key.startswith(("event:", "cue:")) else ov_lanes.get("music", {})
        if ov.get("mute") or lane.get("mute"):
            logger.info(f"Mix override: '{key}' muted.")
            return None
        gain_db = float(ov.get("gain_db", 0.0)) + float(lane.get("gain_db", 0.0))
        ts = max(0.0, base_ts + float(ov.get("nudge_s", 0.0)))
        return ts, 10 ** (gain_db / 20.0)

    def _take(text: str, ov: Dict[str, Any]) -> str:
        """Author regeneration: a prompt override replaces the crew's text; a
        variant number is a nonce appended to the generation prompt, rolling a
        NEW cached generation while every earlier take stays retrievable."""
        out = ov.get("prompt") or text
        v = int(ov.get("variant", 0) or 0)
        return f"{out}, take {v}" if v else out

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
    events: List[Tuple[float, str, float]] = []  # (timestamp, asset_path, linear_gain)
    for i, s in enumerate(direction.get("music", {}).get("stingers", [])):
        idx = s["after_line_index"]
        if 0 <= idx < len(line_wavs):
            params = _ev_params(f"stinger:{i}", offsets[idx] + _wav_duration(line_wavs[idx][0]))
            if params is None:
                continue
            sting_path = os.path.join(scene_dir, f"stinger_{i}.wav")
            if generate_stinger(_take(s.get("description", "dramatic musical accent"), ov_events.get(f"stinger:{i}", {})), sting_path) is None:
                sting_path = make_placeholder_stinger(sting_path)
            events.append((*params[:1], sting_path, params[1]))
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
            params = _ev_params(f"event:{i}", offsets[idx] + _wav_duration(line_wavs[idx][0]))
            if params is None:
                continue
            anchor, ev_gain = params
            ev_ov = ov_events.get(f"event:{i}", {})
            if ev_ov.get("prompt"):
                # author's text wins over the crew's decomposition: one clean layer
                layers = [{"component": _take(ev_ov["prompt"], {"variant": ev_ov.get("variant")}),
                           "timing": "start", "level": "prominent"}]
            elif ev_ov.get("variant"):
                layers = [{**l, "component": _take(l["component"], {"variant": ev_ov.get("variant")})}
                          for l in ev["layers"]]
            else:
                layers = ev["layers"]
            if ev.get("category") == "creature" and ev.get("emotional_intent"):
                layers = [
                    {**l, "component": f"{l['component']}, {ev['emotional_intent']} tone"}
                    for l in layers
                ]
            composite_path = os.path.join(scene_dir, f"composite_{i}.wav")
            if compose_layered_sfx(layers, composite_path) is not None:
                events.append((anchor, composite_path, ev_gain))
                logger.info(f"Composite event '{ev['name']}' ({len(layers)} layers) anchored at {anchor:.1f}s")
            else:
                events.append((anchor, make_placeholder_sfx(ev["name"], composite_path), ev_gain))
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
            params = _ev_params(f"cue:{i}", anchor)
            if params is None:
                continue
            sfx_path = make_placeholder_sfx(cue["description"], os.path.join(scene_dir, f"sfx_{i}.wav"))
            events.append((params[0], sfx_path, params[1]))

    # 3. Beds: real MusicGen from the director's style prompt, placeholder fallback.
    # Music state machine: 'stop'/'resume'/'change' events segment the bed timeline
    # (the gold's "Music stops abruptly. Dead silence." / "resumes instantly").
    music = direction.get("music", {})
    music_lane_ov = ov_lanes.get("music", {})
    base_style = _take(music_lane_ov.get("prompt") or music.get("style", "soft ambient underscore"),
                       {"variant": music_lane_ov.get("variant")})
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
    amb_lane_ov = ov_lanes.get("ambience", {})
    if (sound_design and sound_design.get("continuous_ambience")) or amb_lane_ov.get("prompt"):
        from src.audio_generation import generate_sfx
        amb_base = amb_lane_ov.get("prompt") or ", ".join((sound_design or {}).get("continuous_ambience", []))
        amb_prompt = _take(amb_base, {"variant": amb_lane_ov.get("variant")}) + ", soft continuous background ambience, gentle field recording, smooth"
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
    def _lane_gain(lane: str) -> float:
        return 10 ** (float(ov_lanes.get(lane, {}).get("gain_db", 0.0)) / 20.0)

    music_muted = bool(ov_lanes.get("music", {}).get("mute"))
    ambience_muted = bool(ov_lanes.get("ambience", {}).get("mute"))
    inputs = ["-i", voice_track]
    filters = [f"[0:a]volume={_lane_gain('voice'):.4f}[vox]"]
    mix_labels = ["[vox]"]
    idx = 1
    if not music_muted:
        inputs += ["-i", music_bed]
        filters.append(f"[{idx}:a][0:a]sidechaincompress=threshold=0.15:ratio=2:attack=100:release=300,volume={_lane_gain('music'):.4f}[ducked]")
        mix_labels.append("[ducked]")
        idx += 1
    else:
        logger.info("Mix override: music lane muted.")
    if ambience and not ambience_muted:
        inputs += ["-i", ambience]
        filters.append(f"[{idx}:a]volume={_lane_gain('ambience'):.4f}[amb]")
        mix_labels.append("[amb]")
        idx += 1
    elif ambience_muted:
        logger.info("Mix override: ambience lane muted.")
    for ts, asset, gain in events:
        inputs += ["-i", asset]
        delay_ms = int(ts * 1000)
        filters.append(f"[{idx}:a]adelay={delay_ms}|{delay_ms},volume={gain:.4f}[ev{idx}]")
        mix_labels.append(f"[ev{idx}]")
        idx += 1
    filters.append(f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=first:normalize=0[out]")
    _run_ffmpeg([*inputs, "-filter_complex", ";".join(filters), "-map", "[out]", "-ac", "1", "-ar", "22050", out_path])
    logger.info(f"Scene {scene_id}: {total_dur:.1f}s, {len(events)} timeline event(s) placed.")
    return out_path


# ----------------------------------------------------
# Orchestration
# ----------------------------------------------------

def mix_voice_track(manifest_path: str, output_path: str, single_narrator: bool) -> Dict[str, Any]:
    """Voice-track assembly shared by Tier 1 (one narrator) and Tier 2 (cast).

    Lines are synthesized (or cache-hit), concatenated in manuscript order with
    each line's own post_padding and a beat between chapters, then ACX-mastered.
    While assembling, every line's real start/end offset is MEASURED from its wav
    duration and written to {output}.line_timings.json -- the data a synced
    read-along transcript, chapter markers, and Chain F cut lists all need.
    An .m4b with real chapter markers is exported next to the wav.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = ManuscriptManifest.model_validate_json(f.read())
    book_stem = os.path.splitext(manifest.source_file)[0]

    from src.voice_synthesizer import VoiceSynthesizer
    synth = VoiceSynthesizer()
    needed = {"Narrator"} if single_narrator else {
        l.character for p in manifest.parts for c in p.chapters for s in c.scenes for l in s.lines}
    for char in sorted(needed):
        if not synth.palace.get_character_drawer(char):
            logger.info(f"Registering drawer for '{char}' (builtin voice via name-hash pool).")
            synth.palace.register_character(
                character_name=char,
                voice_ref_path="data/voice_references/narrator_mono.wav",
                speed=1.0, pitch=0.0,
            )

    os.makedirs(WORKSPACE, exist_ok=True)
    from src import progress as _progress
    total_scenes = sum(len(ch.scenes) for p in manifest.parts for ch in p.chapters)
    done = 0
    stage = "tier1_narration" if single_narrator else "tier2_narration"

    CHAPTER_GAP_MS = 1500
    segments: List[Tuple[str, int]] = []          # (wav_path, trailing_silence_ms)
    timing_lines: List[Dict[str, Any]] = []       # measured offsets, filled below
    chapter_marks: List[Dict[str, Any]] = []      # (chapter_id, title, first segment index)
    for part in manifest.parts:
        for chapter in part.chapters:
            chapter_marks.append({"chapter_id": chapter.chapter_id, "title": chapter.title,
                                  "segment_index": len(segments)})
            for scene in chapter.scenes:
                lines = []
                for l in scene.lines:
                    d = l.model_dump()
                    if single_narrator:
                        d["character"] = "Narrator"
                        d["speaker_id"] = "char_narrator"
                    lines.append(d)
                done += 1
                _progress.report(book_stem, stage, done, total_scenes, scene.scene_id)
                for wav, line in resolve_line_wavs(lines, synth):
                    segments.append((wav, int(line.get("post_padding_ms") or 0)))
                    timing_lines.append({"line_id": line["line_id"], "character": line["character"],
                                         "chapter_id": chapter.chapter_id})
            if segments:
                segments[-1] = (segments[-1][0], segments[-1][1] + CHAPTER_GAP_MS)
    _progress.finish(book_stem, stage)

    if not segments:
        raise RuntimeError("Manifest produced no narration lines.")

    # Measure real offsets segment by segment (wav duration + its padding).
    cursor = 0.0
    chapter_starts: Dict[int, float] = {}
    for i, (wav, pad_ms) in enumerate(segments):
        chapter_starts.setdefault(i, cursor)
        dur = _wav_duration(wav)
        timing_lines[i]["start_s"] = round(cursor, 3)
        timing_lines[i]["end_s"] = round(cursor + dur, 3)
        cursor += dur + pad_ms / 1000.0
    for cm in chapter_marks:
        cm["start_s"] = round(chapter_starts.get(cm.pop("segment_index"), 0.0), 3)

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

    total = _wav_duration(output_path)
    timings_path = os.path.splitext(output_path)[0] + ".line_timings.json"
    with open(timings_path, "w", encoding="utf-8") as f:
        json.dump({"book": book_stem, "tier": 1 if single_narrator else 2,
                   "duration_s": round(total, 3), "chapters": chapter_marks,
                   "lines": timing_lines}, f, indent=2)
    m4b_path = export_m4b(output_path, chapter_marks, total, title=book_stem)

    logger.info(f"{'Tier 1' if single_narrator else 'Tier 2'} voice master: {output_path} "
                f"({total/60:.1f} min, {len(segments)} lines, {len(chapter_marks)} chapters)")
    return {"output": output_path, "lines": len(segments), "acx": compliance,
            "timings": timings_path, "m4b": m4b_path, "chapters": len(chapter_marks)}


def mix_tier1(manifest_path: str, output_path: str) -> Dict[str, Any]:
    """Tier 1: the whole manuscript read by ONE narrator voice (no music/SFX/cast)."""
    return mix_voice_track(manifest_path, output_path, single_narrator=True)


def mix_tier2(manifest_path: str, output_path: str) -> Dict[str, Any]:
    """Tier 2: narrator + attributed character voices (no music/SFX). Speakers
    without a MemPalace drawer get a distinct builtin voice via the name-hash pool."""
    return mix_voice_track(manifest_path, output_path, single_narrator=False)


def export_m4b(master_wav: str, chapters: List[Dict[str, Any]], total_s: float,
               title: str = "") -> Optional[str]:
    """Audiobook container: AAC .m4b with REAL chapter markers (ffmetadata),
    chapter titles from the manifest. Returns the m4b path, or None if the
    encode fails (the wav master always remains the source of truth)."""
    meta = os.path.splitext(master_wav)[0] + ".ffmeta"
    with open(meta, "w", encoding="utf-8") as f:
        f.write(";FFMETADATA1\n")
        if title:
            f.write(f"title={title}\nalbum={title}\nartist=Volcano Studios\n")
        for i, ch in enumerate(chapters):
            start_ms = int(ch["start_s"] * 1000)
            end_ms = int((chapters[i + 1]["start_s"] if i + 1 < len(chapters) else total_s) * 1000)
            f.write(f"\n[CHAPTER]\nTIMEBASE=1/1000\nSTART={start_ms}\nEND={end_ms}\n"
                    f"title={ch.get('title') or ch.get('chapter_id')}\n")
    out = os.path.splitext(master_wav)[0] + ".m4b"
    try:
        _run_ffmpeg(["-i", master_wav, "-i", meta, "-map", "0:a", "-map_metadata", "1",
                     "-c:a", "aac", "-b:a", "96k", "-f", "ipod", out])
    except Exception as e:
        logger.warning(f"M4B export failed ({e}); the wav master stands alone.")
        return None
    finally:
        if os.path.exists(meta):
            os.remove(meta)
    return out if os.path.exists(out) and os.path.getsize(out) > 0 else None


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

    mix_overrides_all: Dict[str, Any] = {}
    mix_ov_path = os.path.join(tier3_dir, "mix_overrides.json")
    if os.path.exists(mix_ov_path):
        with open(mix_ov_path, "r", encoding="utf-8") as f:
            mix_overrides_all = json.load(f)
        logger.info(f"Loaded console mix overrides for {len(mix_overrides_all)} scene(s).")

    overrides: Dict[str, Dict[str, Any]] = {}
    overrides_path = os.path.join(tier3_dir, "line_overrides.json")
    if os.path.exists(overrides_path):
        with open(overrides_path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        logger.info(f"Loaded {len(overrides)} production line override(s).")

    from src.voice_synthesizer import VoiceSynthesizer
    synth = VoiceSynthesizer()

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
                    mix_overrides=mix_overrides_all.get(scene.scene_id),
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
    parser = argparse.ArgumentParser(description="Caldera Engine Production Mixer (Chain D: Tier 3 assembly / Tier 1 narration)")
    parser.add_argument("--manifest", type=str, required=True, help="ManuscriptManifest JSON (Tier 3 additionally needs scene_director artifacts)")
    parser.add_argument("--output", type=str, required=True, help="Output master WAV path")
    parser.add_argument("--tier1", action="store_true", help="Single-narrator audiobook: one voice, no music/SFX/casting")
    parser.add_argument("--tier2", action="store_true", help="Narrator + attributed character voices, no music/SFX")
    args = parser.parse_args()
    if args.tier1 or args.tier2:
        result = mix_voice_track(args.manifest, args.output, single_narrator=args.tier1)
        print(f"\nTier {'1' if args.tier1 else '2'} voice master: {result['output']} | lines: {result['lines']} | chapters: {result['chapters']}")
        print(f"Timings: {result['timings']} | M4B: {result['m4b']}")
    else:
        result = mix_production(args.manifest, args.output)
        print(f"\nTier 3 master: {result['output']} | scenes: {result['scenes']}")
    print(f"ACX: {json.dumps(result['acx'], indent=2, default=str)[:400]}")
    sys.exit(0)


if __name__ == "__main__":
    main()
