#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Caldera Engine Audio Generation (Chain C: real music/stinger assets)

Generates music beds and stingers locally with Meta's MusicGen (open weights) via
the transformers library -- deliberately NOT the audiocraft package, whose old
torch pins would break this environment's TTS stack.

All clips are cached by prompt+duration hash under data/generated_audio/, so
re-mixing a book never regenerates audio. Set CALDERA_MUSICGEN=off to force
the production mixer back onto placeholder assets.
"""

import os
import re
import hashlib
import logging
import subprocess
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AudioGeneration")

CACHE_DIR = "data/generated_audio"
MODEL_ID = "facebook/musicgen-small"
FRAME_RATE = 50  # MusicGen audio-token frame rate (tokens per second)

_model = None
_processor = None
_device = None


def _is_enabled() -> bool:
    return os.getenv("CALDERA_MUSICGEN", "on").strip().lower() not in ("off", "0", "false")


def _load_model():
    global _model, _processor, _device
    if _model is not None:
        return True
    try:
        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)
        _device = "cpu"
        if torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info(0)
                if free_bytes > 2.5 * 1024**3:
                    _model = _model.to("cuda:0")
                    _device = "cuda:0"
            except Exception:
                pass
        logger.info(f"MusicGen loaded on {_device}.")
        return True
    except Exception as e:
        logger.warning(f"MusicGen unavailable ({e}); production mixer will use placeholders.")
        _model = None
        return False


def _run_ffmpeg(cmd) -> None:
    subprocess.run(["ffmpeg", "-y", *cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def generate_clip(prompt: str, duration_sec: float, out_path: str) -> Optional[str]:
    """Generates a music clip at 22050Hz mono. Returns out_path or None on failure.
    Clips are capped at ~28s (MusicGen's practical single-pass limit); callers loop
    longer beds to length."""
    if not _is_enabled():
        return None

    duration_sec = min(duration_sec, 28.0)
    cache_key = hashlib.sha1(f"{prompt}|{duration_sec:.0f}".encode()).hexdigest()[:16]
    os.makedirs(CACHE_DIR, exist_ok=True)
    cached = os.path.join(CACHE_DIR, f"musicgen_{cache_key}.wav")

    if not (os.path.exists(cached) and os.path.getsize(cached) > 0):
        # Cache check happens BEFORE model load: a remix whose prompts are all
        # cached never loads MusicGen at all, leaving GPU memory for the SFX model.
        if not _load_model():
            return None
        try:
            import torch
            import soundfile as sf
            inputs = _processor(text=[prompt], padding=True, return_tensors="pt").to(_device)
            max_new_tokens = min(int(duration_sec * FRAME_RATE), 1500)
            with torch.no_grad():
                audio = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, guidance_scale=3.0)
            sr = _model.config.audio_encoder.sampling_rate
            raw = cached + ".raw.wav"
            sf.write(raw, audio[0, 0].cpu().float().numpy(), sr)
            _run_ffmpeg(["-i", raw, "-ac", "1", "-ar", "22050", cached])
            os.remove(raw)
            logger.info(f"MusicGen generated {duration_sec:.0f}s clip for: {prompt[:60]!r}")
        except Exception as e:
            logger.warning(f"MusicGen generation failed for {prompt[:50]!r}: {e}")
            return None

    if os.path.abspath(cached) != os.path.abspath(out_path):
        _run_ffmpeg(["-i", cached, "-c", "copy", out_path])
    return out_path


def generate_music_bed(prompt: str, scene_duration_sec: float, out_path: str, volume: float = 0.5) -> Optional[str]:
    """Scene-length music bed: generates up to 28s and loops it to the full scene
    duration with fade in/out, at background level. Cached by prompt."""
    clip = generate_clip(prompt, min(scene_duration_sec, 28.0), out_path + ".clip.wav")
    if clip is None:
        return None
    fade_out_start = max(scene_duration_sec - 2.0, 0)
    _run_ffmpeg([
        "-stream_loop", "-1", "-i", clip,
        "-t", f"{scene_duration_sec:.2f}",
        "-af", f"loudnorm=I=-23:TP=-2.5,afade=t=in:d=2.0,afade=t=out:st={fade_out_start:.2f}:d=2.0",
        "-ac", "1", "-ar", "22050", out_path,
    ])
    try:
        os.remove(clip)
    except OSError:
        pass
    return out_path


# ----------------------------------------------------
# SFX generation (AudioLDM2 via diffusers) + layered composition
# ----------------------------------------------------

_sfx_pipe = None
_sfx_device = None
SFX_MODEL_ID = "cvssp/audioldm-s-full-v2"


def _load_sfx_model():
    global _sfx_pipe, _sfx_device
    if _sfx_pipe is not None:
        return True
    if os.getenv("CALDERA_SFXGEN", "on").strip().lower() in ("off", "0", "false"):
        return False
    try:
        import torch
        from diffusers import AudioLDMPipeline
        _sfx_device = "cpu"
        dtype = torch.float32
        if torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info(0)
                if free_bytes > 1.8 * 1024**3:
                    _sfx_device = "cuda:0"
                    dtype = torch.float16
            except Exception:
                pass
        # AudioLDM v1, not v2: the v2 pipeline in diffusers 0.39 calls a private
        # transformers API removed in 4.5x, and downgrading transformers would
        # re-break coqui-tts. v1 uses a simpler CLAP text-encoder path that works.
        _sfx_pipe = AudioLDMPipeline.from_pretrained(SFX_MODEL_ID, torch_dtype=dtype).to(_sfx_device)
        logger.info(f"AudioLDM SFX model loaded on {_sfx_device}.")
        return True
    except Exception as e:
        logger.warning(f"AudioLDM2 unavailable ({e}); SFX layers fall back to placeholders.")
        _sfx_pipe = None
        return False


def generate_sfx(prompt: str, duration_sec: float, out_path: str, steps: int = 25) -> Optional[str]:
    """Generates a single SFX layer at 22050Hz mono via AudioLDM2. Cached by prompt."""
    duration_sec = max(1.0, min(duration_sec, 8.0))
    cache_key = hashlib.sha1(f"sfx|{prompt}|{duration_sec:.0f}|{steps}".encode()).hexdigest()[:16]
    os.makedirs(CACHE_DIR, exist_ok=True)
    cached = os.path.join(CACHE_DIR, f"audioldm2_{cache_key}.wav")

    if not (os.path.exists(cached) and os.path.getsize(cached) > 0):
        if not _load_sfx_model():
            return None
        try:
            import soundfile as sf
            result = _sfx_pipe(
                prompt,
                negative_prompt="music, melody, speech, talking, human voice, harsh noise, static, hiss, distortion, low quality",
                num_inference_steps=steps,
                audio_length_in_s=duration_sec,
            )
            raw = cached + ".raw.wav"
            sf.write(raw, result.audios[0], 16000)
            _run_ffmpeg(["-i", raw, "-ac", "1", "-ar", "22050", cached])
            os.remove(raw)
            logger.info(f"AudioLDM2 generated {duration_sec:.0f}s SFX for: {prompt[:60]!r}")
        except Exception as e:
            logger.warning(f"AudioLDM2 generation failed for {prompt[:50]!r}: {e}")
            return None

    if os.path.abspath(cached) != os.path.abspath(out_path):
        _run_ffmpeg(["-i", cached, "-c", "copy", out_path])
    return out_path


_LAYER_TIMING_OFFSETS = {"start": 0.0, "overlap": 0.25, "tail": 0.6}
_LAYER_LEVELS = {"prominent": 1.0, "medium": 0.6, "subtle": 0.35}


def compose_layered_sfx(layers, out_path: str, base_duration: float = 3.0) -> Optional[str]:
    """Mixes generated component layers into ONE composite sound event, ffmpeg-side:
    per-layer timing offsets (start/overlap/tail) and levels (prominent/medium/subtle),
    then loudness-normalized. E.g. 'McGregor overturning flower-pots' = ceramic scrape
    (start, prominent) + pot clink (overlap, medium) + wooden thud (tail, medium)."""
    layer_files = []
    for i, layer in enumerate(layers):
        path = out_path + f".layer{i}.wav"
        if generate_sfx(layer["component"], base_duration, path) is not None:
            layer_files.append((layer, path))
    if not layer_files:
        return None

    inputs = []
    filters = []
    mix_labels = []
    for i, (layer, path) in enumerate(layer_files):
        inputs += ["-i", path]
        delay_ms = int(_LAYER_TIMING_OFFSETS.get(layer.get("timing", "start"), 0.0) * 1000)
        level = _LAYER_LEVELS.get(layer.get("level", "medium"), 0.6)
        filters.append(f"[{i}:a]volume={level},adelay={delay_ms}|{delay_ms}[l{i}]")
        mix_labels.append(f"[l{i}]")
    filters.append(f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=longest:normalize=0,loudnorm=I=-16:TP=-2.0[out]")
    _run_ffmpeg([*inputs, "-filter_complex", ";".join(filters), "-map", "[out]", "-ac", "1", "-ar", "22050", out_path])
    for _, path in layer_files:
        try:
            os.remove(path)
        except OSError:
            pass
    return out_path


def generate_stinger(prompt: str, out_path: str, volume: float = 0.85) -> Optional[str]:
    """Short (~3s) musical accent."""
    clip = generate_clip(f"{prompt}, short musical sting, dramatic accent", 3.0, out_path + ".clip.wav")
    if clip is None:
        return None
    _run_ffmpeg(["-i", clip, "-af", "loudnorm=I=-16:TP=-2.0,afade=t=out:st=2.2:d=0.8", "-ac", "1", "-ar", "22050", out_path])
    try:
        os.remove(clip)
    except OSError:
        pass
    return out_path
