#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Caldera Engine TTS Compiler Wrapper
Executes speech synthesis, applies speed and pitch modifiers, appends post-padding silence,
concatenates tracks, mixes background sounds, and generates the mastered QC compliance report.
"""

import os
import json
import logging
import subprocess
import wave
import numpy as np
from typing import Dict, Any, List
from src.main import CalderaPipeline

logger = logging.getLogger("TTSCompiler")


def load_cast_manifest(workspace_dir: str) -> Dict[str, Any]:
    """
    Locates and loads the project Cast Manifest.
    Checks workspace directory, data directory, and project database fallbacks.
    """
    paths_to_check = [
        os.path.join(workspace_dir, "cast_manifest.json"),
        "data/cast_manifest.json",
        "cast_manifest.json"
    ]
    for path in paths_to_check:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                    logger.info(f"Loaded Cast Manifest from {path}")
                    return manifest
            except Exception as e:
                logger.warning(f"Failed to parse Cast Manifest at {path}: {e}")

    # Fallback default Cast Manifest mapping char_narrator to Narrator drawer
    default_manifest = {
        "char_narrator": {
            "character_name": "Narrator",
            "voice_ref_path": "data/voice_references/narrator_mono.wav",
            "speed": 1.0,
            "pitch": 0.0
        }
    }
    
    # Save the default manifest to data/cast_manifest.json for persistence and auditability
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/cast_manifest.json", "w", encoding="utf-8") as f:
            json.dump(default_manifest, f, indent=4)
        logger.info("Created default Cast Manifest at data/cast_manifest.json")
    except Exception as e:
        logger.warning(f"Could not save default Cast Manifest: {e}")

    return default_manifest


def get_ffmpeg_filters(speed_modifier: float, pitch_modifier: float, sample_rate: int) -> str:
    """Constructs FFmpeg filter chain for speed and pitch shifts, chaining atempo if needed."""
    filters = []
    if pitch_modifier != 1.0:
        target_sr = int(sample_rate * pitch_modifier)
        filters.append(f"asetrate={target_sr}")
        # Compensate tempo shift introduced by asetrate to reach target speed
        tempo_comp = speed_modifier / pitch_modifier
        while tempo_comp > 2.0:
            filters.append("atempo=2.0")
            tempo_comp /= 2.0
        while tempo_comp < 0.5:
            filters.append("atempo=0.5")
            tempo_comp /= 0.5
        if abs(tempo_comp - 1.0) > 1e-4:
            filters.append(f"atempo={tempo_comp}")
    elif speed_modifier != 1.0:
        tempo_comp = speed_modifier
        while tempo_comp > 2.0:
            filters.append("atempo=2.0")
            tempo_comp /= 2.0
        while tempo_comp < 0.5:
            filters.append("atempo=0.5")
            tempo_comp /= 0.5
        if abs(tempo_comp - 1.0) > 1e-4:
            filters.append(f"atempo={tempo_comp}")
            
    return ",".join(filters) if filters else ""


def apply_audio_modifiers(input_wav_path: str, output_wav_path: str, speed_modifier: float, pitch_modifier: float):
    """
    Applies pitch and speed modifications.
    Uses FFmpeg as primary processor and NumPy interpolation as fallback.
    """
    if abs(speed_modifier - 1.0) < 1e-4 and abs(pitch_modifier - 1.0) < 1e-4:
        import shutil
        shutil.copyfile(input_wav_path, output_wav_path)
        return

    # Try reading sample rate to set up FFmpeg filtering properly
    sample_rate = 22050
    try:
        with wave.open(input_wav_path, "rb") as w:
            sample_rate = w.getframerate()
    except Exception as e:
        logger.warning(f"Could not detect WAV sample rate: {e}. Defaulting to 22050.")

    filters = get_ffmpeg_filters(speed_modifier, pitch_modifier, sample_rate)
    if filters:
        filters += f",aresample={sample_rate}"
        cmd = ["ffmpeg", "-y", "-i", input_wav_path, "-af", filters, output_wav_path]
        try:
            logger.info(f"Applying FFmpeg modifiers: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.warning(f"FFmpeg process unavailable or failed: {e}. Falling back to NumPy time-stretch.")

    # NumPy Fallback (resampling-based speed stretch)
    logger.info("Applying NumPy speed modifier fallback...")
    try:
        with wave.open(input_wav_path, "rb") as w:
            params = w.getparams()
            frames = w.readframes(params.nframes)
            sample_rate = params.framerate
            sample_width = params.sampwidth
            channels = params.nchannels

        if sample_width == 2:
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        elif sample_width == 1:
            samples = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0
        else:
            samples = np.frombuffer(frames, dtype=np.float32)

        # Naive linear interpolation to stretch length
        if speed_modifier != 1.0 and len(samples) > 0:
            old_indices = np.arange(len(samples))
            new_indices = np.arange(0, len(samples), speed_modifier)
            samples = np.interp(new_indices, old_indices, samples)

        if sample_width == 2:
            out_frames = np.clip(samples, -32768, 32767).astype(np.int16).tobytes()
        elif sample_width == 1:
            out_frames = np.clip(samples + 128.0, 0, 255).astype(np.uint8).tobytes()
        else:
            out_frames = samples.tobytes()

        with wave.open(output_wav_path, "wb") as w_out:
            w_out.setparams(params)
            w_out.writeframes(out_frames)
    except Exception as ex:
        logger.error(f"NumPy fallback failed: {ex}. Copying original file directly.")
        import shutil
        shutil.copyfile(input_wav_path, output_wav_path)


def append_silence_to_wav(input_wav_path: str, output_wav_path: str, padding_ms: int):
    """Appends exact silent samples to the end of the WAV block."""
    if padding_ms <= 0:
        import shutil
        shutil.copyfile(input_wav_path, output_wav_path)
        return

    try:
        with wave.open(input_wav_path, "rb") as w_in:
            params = w_in.getparams()
            frames = w_in.readframes(w_in.getnframes())
            sample_rate = params.framerate
            sample_width = params.sampwidth
            channels = params.nchannels

        silence_duration_sec = padding_ms / 1000.0
        num_silent_samples = int(sample_rate * silence_duration_sec)
        silence_bytes = bytes(num_silent_samples * sample_width * channels)

        with wave.open(output_wav_path, "wb") as w_out:
            w_out.setparams(params)
            w_out.writeframes(frames)
            w_out.writeframes(silence_bytes)
        logger.info(f"Appended {padding_ms}ms of silence to: {output_wav_path}")
    except Exception as e:
        logger.error(f"Failed to append silence to WAV: {e}")
        import shutil
        shutil.copyfile(input_wav_path, output_wav_path)


def compile_modified_json(script_data: Dict[str, Any], output_master_wav: str, profile_name: str = "standard", user_tier: str = "free") -> Dict[str, Any]:
    """
    Executes end-to-end voice synthesis and mixing starting directly from a structured script JSON.
    Dynamically applies performance modifiers (speed/pitch) and custom post-padding silence.
    """
    logger.info("Initializing Caldera Engine pipeline compiler...")
    pipeline = CalderaPipeline()
    
    # Retrieve user_tier from script metadata if present
    tier = script_data.get("metadata", {}).get("user_tier", user_tier)
    
    # 0. Set up default voices in database
    pipeline.register_voice_cast()
    
    # Load Cast Manifest
    cast_manifest = load_cast_manifest(pipeline.workspace_dir)
    
    # Save script output to the workspace directory for compliance / audit checks
    script_output_path = os.path.join(pipeline.workspace_dir, "script_output.json")
    os.makedirs(os.path.dirname(script_output_path), exist_ok=True)
    with open(script_output_path, "w", encoding="utf-8") as f:
        json.dump(script_data, f, indent=4)
        
    # Log Chapters (Wings) in MemPalace database
    logged_chapters = set()
    for line in script_data["script"]:
        ch = line.get("chapter")
        if ch is not None and ch not in logged_chapters:
            pipeline.palace.log_wing(f"wing_c{ch}", ch, f"Chapter {ch}")
            logged_chapters.add(ch)
        
    # Register character drawers based on Cast Manifest / script
    unique_speakers = set(line.get("speaker_id", "char_narrator") for line in script_data["script"])
    for speaker_id in unique_speakers:
        cast_config = cast_manifest.get(speaker_id, cast_manifest.get("char_narrator"))
        char_name = cast_config.get("character_name", "Narrator")
        voice_ref = cast_config.get("voice_ref_path", "data/voice_references/narrator_mono.wav")
        
        drawer = pipeline.palace.get_character_drawer(char_name)
        if not drawer:
            logger.info(f"Registering Cast Manifest character '{char_name}' inside MemPalace...")
            pipeline.palace.register_character(
                character_name=char_name,
                voice_ref_path=voice_ref,
                speed=cast_config.get("speed", 1.0),
                pitch=cast_config.get("pitch", 0.0)
            )
            
    # 2. Synthesize Character Dialogue/Narrative Lines
    logger.info(f"Synthesizing {len(script_data['script'])} script lines...")
    voice_files: List[str] = []
    
    for idx, line in enumerate(script_data["script"]):
        # Cross-reference speaker_id with Cast Manifest to get character name.
        # Fall back to the line's own attributed character (from Tier 1 LLM
        # enrichment or higher tiers) when the manifest doesn't know this
        # speaker_id -- collapsing unknown speakers to Narrator silently
        # discards upstream attribution.
        speaker_id = line.get("speaker_id", "char_narrator")
        cast_config = cast_manifest.get(speaker_id)
        if cast_config:
            char_name = cast_config.get("character_name", "Narrator")
        else:
            char_name = line.get("character") or "Narrator"

        # Determine voice ref and emotion to build a cache-safe filename
        drawer = pipeline.palace.get_character_drawer(char_name)
        voice_ref = drawer.get("voice_ref_path", "default") if drawer else "default"
        import re
        voice_slug = re.sub(r'[^a-zA-Z0-9_\-]', '', voice_ref)
        char_slug = re.sub(r'[^a-zA-Z0-9_\-]', '', char_name)
        emotion = line.get("emotion", "Neutral")

        line_wav = os.path.join(pipeline.outputs_dir, f"line_{line['line_id']}_{char_slug}_{voice_slug}_{emotion}.wav")
        text_to_synth = line.get("dialogue") or line.get("text") or line.get("narration_before") or "..."
        
        # Check if the base synthesized WAV already exists on disk
        if os.path.exists(line_wav) and os.path.getsize(line_wav) > 0:
            logger.info(f"Cache hit: Reusing existing voice chunk for Line {line.get('line_number', idx+1)} [{char_name}] at: {line_wav}")
            res = {
                "status": "SUCCESS",
                "character": char_name,
                "output_path": line_wav,
                "reference_used": voice_ref,
                "modulation_applied": drawer.get("modulation_config", {}) if drawer else {}
            }
        else:
            logger.info(f"Cache miss: Synthesizing Line {line.get('line_number', idx+1)} [{char_name} - Mood: {emotion}]: '{text_to_synth[:40]}...'")
            # Generates base audio block with user_tier routing (local XTTSv2 vs commercial APIs)
            res = pipeline.synth.synthesize_line(
                character_name=char_name,
                dialogue_text=text_to_synth,
                target_emotion=emotion,
                output_wav_path=line_wav,
                user_tier=tier
            )
        
        # Parse performance overrides
        performance = line.get("performance", {})
        if hasattr(performance, "get") is False:
            performance = {}
        speed_modifier = float(performance.get("speed_modifier", 1.0))
        pitch_modifier = float(performance.get("pitch_modifier", 1.0))
        post_padding_ms = int(line.get("post_padding_ms", 250))
        
        # Build cache-safe filename for the final padded WAV of this line
        padded_wav = os.path.join(pipeline.outputs_dir, f"line_{line['line_id']}_{char_slug}_{voice_slug}_{emotion}_sp_{speed_modifier}_pi_{pitch_modifier}_pad_{post_padding_ms}.wav")
        
        if os.path.exists(padded_wav) and os.path.getsize(padded_wav) > 0:
            logger.info(f"Cache hit: Reusing existing fully processed padded WAV for Line {line.get('line_number', idx+1)} [{char_name}] at: {padded_wav}")
        else:
            # Check if the base synthesized WAV already exists on disk
            if os.path.exists(line_wav) and os.path.getsize(line_wav) > 0:
                logger.info(f"Cache hit: Reusing existing voice chunk for Line {line.get('line_number', idx+1)} [{char_name}] at: {line_wav}")
                res = {
                    "status": "SUCCESS",
                    "character": char_name,
                    "output_path": line_wav,
                    "reference_used": voice_ref,
                    "modulation_applied": drawer.get("modulation_config", {}) if drawer else {}
                }
            else:
                logger.info(f"Cache miss: Synthesizing Line {line.get('line_number', idx+1)} [{char_name} - Mood: {emotion}]: '{text_to_synth[:40]}...'")
                # Generates base audio block with user_tier routing (local XTTSv2 vs commercial APIs)
                res = pipeline.synth.synthesize_line(
                    character_name=char_name,
                    dialogue_text=text_to_synth,
                    target_emotion=emotion,
                    output_wav_path=line_wav,
                    user_tier=tier
                )
            
            # Apply modifiers
            modified_wav = os.path.join(pipeline.outputs_dir, f"line_{line['line_id']}_{char_slug}_{voice_slug}_{emotion}_sp_{speed_modifier}_pi_{pitch_modifier}_modified.wav")
            if not (os.path.exists(modified_wav) and os.path.getsize(modified_wav) > 0):
                apply_audio_modifiers(
                    input_wav_path=res["output_path"],
                    output_wav_path=modified_wav,
                    speed_modifier=speed_modifier,
                    pitch_modifier=pitch_modifier
                )
            
            # Append padding silence
            append_silence_to_wav(
                input_wav_path=modified_wav,
                output_wav_path=padded_wav,
                padding_ms=post_padding_ms
            )
        
        voice_files.append(padded_wav)
        
    logger.info("Speech synthesis and modifiers complete. Beginning multi-track stitching and sidechain mixing...")
    
    # 3. Compile voice segments timeline (no extra gap since padding is already in voice_files)
    compiled_voice = os.path.join(pipeline.workspace_dir, "compiled_voice.wav")
    stitch_success = pipeline.mixer.concatenate_voice_segments(
        voice_files=voice_files,
        output_path=compiled_voice,
        silence_gap_sec=0.0
    )
    
    if not stitch_success:
        logger.error("Failed to compile voice timeline. Falling back to a synthetic tone for master mix.")
        compiled_voice = os.path.join(pipeline.outputs_dir, "fallback_test_voice.wav")
        pipeline.mixer._generate_sine_wav(compiled_voice, 440.0, 3.0)
        
    # 4. Map background ambient layer and SFX
    first_emotion = "Neutral"
    if script_data["script"]:
        first_emotion = script_data["script"][0].get("emotion", "Neutral")
    ambient_music_asset = pipeline.mixer.get_ambient_music_for_mood(first_emotion)
    
    sfx_asset_path = None
    for line in script_data["script"]:
        text = line.get("dialogue") or line.get("text") or ""
        sfx_match = pipeline.mixer.map_dialogue_sfx(text, line.get("emotion", "Neutral"))
        if sfx_match:
            logger.info(f"Dynamic Sound Design Event: {sfx_match['description']} (UCS: {sfx_match['ucs_code']})")
            sfx_asset_path = sfx_match["asset_path"]
            break
            
    # 5. Execute mastering mix
    pipeline.mixer.mix_tracks(
        voice_path=compiled_voice,
        music_path=ambient_music_asset,
        sfx_path=sfx_asset_path,
        output_path=output_master_wav,
        profile_name=profile_name
    )
    
    # 6. Run Quality Control Compliance
    qc_report = pipeline.mixer.verify_acx_compliance(output_master_wav, profile_name)
    
    # Clean up resources
    pipeline.palace.close()
    pipeline.synth.unload_models()
    
    return qc_report
