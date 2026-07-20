#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Caldera Engine Image Generation (Chain E: scene stills & character sheets)

Local Stable Diffusion (sd-turbo: 1-4 inference steps, fits the 6GB GPU) consuming
the prompts the production pipeline already builds: per-scene image_prompt from the
Music Director's environment + the Character Designer's (AI-10) visual profiles.

Cached by prompt hash under data/generated_images/. Set CALDERA_IMAGEGEN=off
to disable. Same contract as audio generation: try local model, degrade to None.

Usage:
  python -m src.image_generation --prompt "..." --out scratch/test.png
  python -m src.image_generation --scene-stills --manifest scratch/book.json
"""

import os
import sys
import json
import hashlib
import logging
import argparse
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ImageGeneration")

CACHE_DIR = "data/generated_images"
MODEL_ID = "stabilityai/sd-turbo"

_pipe = None
_device = None


def _load_model() -> bool:
    global _pipe, _device
    if _pipe is not None:
        return True
    if os.getenv("CALDERA_IMAGEGEN", "on").strip().lower() in ("off", "0", "false"):
        return False
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
        _device = "cpu"
        dtype = torch.float32
        if torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info(0)
                if free_bytes > 2.5 * 1024**3:
                    _device = "cuda:0"
                    dtype = torch.float16
            except Exception:
                pass
        _pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=dtype).to(_device)
        logger.info(f"sd-turbo loaded on {_device}.")
        return True
    except Exception as e:
        logger.warning(f"Image model unavailable ({e}); Chain E disabled.")
        _pipe = None
        return False


def stable_seed(key: str) -> int:
    """Deterministic per-entity seed (character name, scene id) so re-generation is
    reproducible and a character's look is anchored across renders. Python's hash()
    is process-salted; sha1 is not."""
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % (2**31)


def generate_image(prompt: str, out_path: str, negative: str = "text, watermark, low quality, deformed", seed: Optional[int] = None) -> Optional[str]:
    """Generates one 512x512 still. Cached by prompt+seed hash; returns out_path or None."""
    cache_key = hashlib.sha1(f"img|{prompt}|{seed}".encode()).hexdigest()[:16]
    os.makedirs(CACHE_DIR, exist_ok=True)
    cached = os.path.join(CACHE_DIR, f"sdturbo_{cache_key}.png")

    if not (os.path.exists(cached) and os.path.getsize(cached) > 0):
        if not _load_model():
            return None
        try:
            import torch
            generator = torch.Generator(device=_device.split(":")[0] if _device != "cpu" else "cpu")
            if seed is not None:
                generator = generator.manual_seed(seed)
            # sd-turbo: guidance must be 0.0, 1-4 steps
            image = _pipe(prompt[:300], num_inference_steps=2, guidance_scale=0.0, generator=generator).images[0]
            image.save(cached)
            logger.info(f"Generated image (seed={seed}) for: {prompt[:70]!r}")
        except Exception as e:
            logger.warning(f"Image generation failed for {prompt[:50]!r}: {e}")
            return None

    if os.path.abspath(cached) != os.path.abspath(out_path):
        import shutil
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        shutil.copy(cached, out_path)
    return out_path


# ----------------------------------------------------
# Identity lock: SD1.5 + IP-Adapter conditioned on the character sheet
# ----------------------------------------------------

_ip_pipe = None
_ip_device = None
IP_BASE_MODEL = "runwayml/stable-diffusion-v1-5"


def _load_ip_model() -> bool:
    global _ip_pipe, _ip_device
    if _ip_pipe is not None:
        return True
    if os.getenv("CALDERA_IMAGEGEN", "on").strip().lower() in ("off", "0", "false"):
        return False
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
        _ip_device = "cpu"
        dtype = torch.float32
        if torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info(0)
                if free_bytes > 3.0 * 1024**3:
                    _ip_device = "cuda:0"
                    dtype = torch.float16
            except Exception:
                pass
        _ip_pipe = AutoPipelineForText2Image.from_pretrained(IP_BASE_MODEL, torch_dtype=dtype, safety_checker=None).to(_ip_device)
        _ip_pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        _ip_pipe.set_ip_adapter_scale(0.55)
        logger.info(f"SD1.5 + IP-Adapter loaded on {_ip_device}.")
        return True
    except Exception as e:
        logger.warning(f"IP-Adapter pipeline unavailable ({e}); identity lock disabled.")
        _ip_pipe = None
        return False


def generate_image_identity_locked(prompt: str, reference_image_path: str, out_path: str, seed: Optional[int] = None) -> Optional[str]:
    """Scene render conditioned on a character reference sheet via IP-Adapter --
    real identity anchoring, not just seed/prompt anchoring. Cache key includes
    the reference image's content hash."""
    ref_hash = hashlib.sha1(open(reference_image_path, "rb").read()).hexdigest()[:10]
    cache_key = hashlib.sha1(f"ipimg|{prompt}|{seed}|{ref_hash}".encode()).hexdigest()[:16]
    os.makedirs(CACHE_DIR, exist_ok=True)
    cached = os.path.join(CACHE_DIR, f"sd15ip_{cache_key}.png")

    if not (os.path.exists(cached) and os.path.getsize(cached) > 0):
        if not _load_ip_model():
            return None
        try:
            import torch
            from PIL import Image
            ref = Image.open(reference_image_path).convert("RGB")
            generator = torch.Generator(device=_ip_device.split(":")[0] if _ip_device != "cpu" else "cpu")
            if seed is not None:
                generator = generator.manual_seed(seed)
            image = _ip_pipe(
                prompt[:300],
                ip_adapter_image=ref,
                negative_prompt="text, watermark, low quality, deformed",
                num_inference_steps=25,
                guidance_scale=7.0,
                generator=generator,
            ).images[0]
            image.save(cached)
            logger.info(f"Identity-locked image (ref={os.path.basename(reference_image_path)}, seed={seed}): {prompt[:60]!r}")
        except Exception as e:
            logger.warning(f"Identity-locked generation failed: {e}")
            return None

    if os.path.abspath(cached) != os.path.abspath(out_path):
        import shutil
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        shutil.copy(cached, out_path)
    return out_path


def generate_scene_stills(manifest_path: str, identity_lock: bool = False) -> str:
    """Chain E over a directed book: one still per scene from generation_prompts'
    image_prompt, enriched with AI-10 character visual profiles where available.

    identity_lock=True routes through SD1.5 + IP-Adapter conditioned on the FIRST
    present profiled character's reference sheet (v1 limitation, documented:
    single-character conditioning -- multi-character identity needs per-region
    adapters or per-character LoRAs)."""
    from src.models import ManuscriptManifest
    manifest = ManuscriptManifest.model_validate_json(open(manifest_path, encoding="utf-8").read())
    book_stem = os.path.splitext(manifest.source_file)[0]
    tier3_dir = os.path.join("data/corpus/pipeline", book_stem, "tier3")
    out_dir = os.path.join(tier3_dir, "stills_locked" if identity_lock else "stills")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(tier3_dir, "generation_prompts.json"), encoding="utf-8") as f:
        prompts = json.load(f)
    profiles = {}
    prof_path = os.path.join(tier3_dir, "character_profiles.json")
    if os.path.exists(prof_path):
        with open(prof_path, encoding="utf-8") as f:
            profiles = {p["name"]: p for p in json.load(f)}

    def _sheet_path(char: str) -> Optional[str]:
        slug = "".join(ch for ch in char if ch.isalnum() or ch in " -_").replace(" ", "_")
        p = os.path.join(tier3_dir, "character_sheets", f"{slug}.png")
        return p if os.path.exists(p) else None

    made = 0
    for p in prompts:
        prompt = p["image_prompt"]
        # Character consistency: append the visual profile of each present character
        for char in p.get("characters_present", []):
            prof = profiles.get(char)
            if prof and prof.get("visual_description"):
                prompt += f"; {char}: {prof['visual_description'][:100]}"
        out = os.path.join(out_dir, f"{p['scene_id']}.png")
        seed = stable_seed(p["scene_id"])

        result = None
        if identity_lock:
            anchor = next((c for c in p.get("characters_present", []) if _sheet_path(c)), None)
            if anchor:
                result = generate_image_identity_locked(prompt, _sheet_path(anchor), out, seed=seed)
        if result is None:
            result = generate_image(prompt, out, seed=seed)
        if result:
            made += 1
    logger.info(f"Chain E{' (identity-locked)' if identity_lock else ''}: {made}/{len(prompts)} scene stills -> {out_dir}/")
    return out_dir


def generate_character_sheets(manifest_path: str) -> str:
    """One full-body reference sheet per profiled character, seed-locked to the
    character's name so the look is stable across regenerations. Honest limitation
    (documented): sd-turbo + seed gives style/palette anchoring, not true identity
    consistency across different scene compositions -- that requires IP-Adapter or
    a per-character LoRA (the specified next step for Chain E)."""
    from src.models import ManuscriptManifest
    manifest = ManuscriptManifest.model_validate_json(open(manifest_path, encoding="utf-8").read())
    book_stem = os.path.splitext(manifest.source_file)[0]
    tier3_dir = os.path.join("data/corpus/pipeline", book_stem, "tier3")
    out_dir = os.path.join(tier3_dir, "character_sheets")
    os.makedirs(out_dir, exist_ok=True)

    prof_path = os.path.join(tier3_dir, "character_profiles.json")
    if not os.path.exists(prof_path):
        logger.warning("No character_profiles.json -- run scene_director --design-characters-only first.")
        return out_dir
    with open(prof_path, encoding="utf-8") as f:
        profiles = json.load(f)

    style = "warm storybook watercolor illustration"
    bible_path = os.path.join(tier3_dir, "book_bible.json")
    if os.path.exists(bible_path):
        bible = json.load(open(bible_path, encoding="utf-8"))
        if bible.get("genre"):
            style = f"illustration in the style fitting {bible['genre']}, {bible.get('era_setting', '')}"

    made = 0
    for p in profiles:
        prompt = f"character reference sheet, full body, plain white background, {style}: {p['name']}, {p['visual_description']}"
        slug = "".join(ch for ch in p["name"] if ch.isalnum() or ch in " -_").replace(" ", "_")
        out = os.path.join(out_dir, f"{slug}.png")
        if generate_image(prompt, out, seed=stable_seed(p["name"])):
            made += 1
    logger.info(f"Character sheets: {made}/{len(profiles)} -> {out_dir}/")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Caldera Engine Image Generation (Chain E)")
    parser.add_argument("--prompt", type=str, help="Single test image prompt")
    parser.add_argument("--out", type=str, default="scratch/test_image.png")
    parser.add_argument("--scene-stills", action="store_true", help="Generate one still per scene for a directed book")
    parser.add_argument("--character-sheets", action="store_true", help="Generate seed-locked character reference sheets from AI-10 profiles")
    parser.add_argument("--identity-lock", action="store_true", help="Scene stills via SD1.5 + IP-Adapter conditioned on character sheets (real identity anchoring)")
    parser.add_argument("--manifest", type=str, help="Manifest for --scene-stills / --character-sheets")
    args = parser.parse_args()

    if args.character_sheets:
        if not args.manifest:
            parser.error("--character-sheets requires --manifest")
        print(generate_character_sheets(args.manifest))
        return
    if args.scene_stills:
        if not args.manifest:
            parser.error("--scene-stills requires --manifest")
        print(generate_scene_stills(args.manifest, identity_lock=args.identity_lock))
    elif args.prompt:
        result = generate_image(args.prompt, args.out)
        print(result or "generation unavailable")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
