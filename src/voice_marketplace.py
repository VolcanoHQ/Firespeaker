#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Firespeaker Voice Marketplace
Uses Qdrant vector database to store and semantically retrieve voice marketplace audio references
by projecting user description prompts into an acoustic embedding space (cloned CLAP vectors).
"""

import os
import re
import uuid
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = logging.getLogger("VoiceMarketplace")

class VoiceMarketplace:
    """Manages the Voice Marketplace vector database using Qdrant."""

    def __init__(self, db_path: str = "data/qdrant_db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize Qdrant Client (running locally in storage directory)
        self.client = QdrantClient(path=self.db_path)
        self._init_collection()

    def _init_collection(self):
        """Creates the voice_marketplace collection if it doesn't already exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if "voice_marketplace" not in collection_names:
                logger.info("Creating 'voice_marketplace' collection in Qdrant...")
                self.client.create_collection(
                    collection_name="voice_marketplace",
                    vectors_config=VectorParams(size=128, distance=Distance.COSINE)
                )
                self._seed_default_marketplace()
            else:
                logger.info("Qdrant collection 'voice_marketplace' already exists.")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")

    def generate_clap_embedding(self, text: str) -> List[float]:
        """
        Simulates an acoustic-text joint embedding space (CLAP) of 128 dimensions.
        Projects semantic keywords describing tone, age, accent, and role to orthogonal coordinates,
        ensuring reliable and deterministic cosine similarity queries.
        """
        vector = np.zeros(128, dtype=np.float32)
        
        mappings = {
            r"\b(british|england|english|uk)\b": 0,
            r"\b(elderly|old|aged|senior)\b": 1,
            r"\b(sea|captain|sailor|nautical|maritime|pirate)\b": 2,
            r"\b(raspy|hoarse|gravelly|rough|throat)\b": 3,
            r"\b(female|woman|lady|girl|maternal)\b": 4,
            r"\b(male|man|gentleman|boy|masculine)\b": 5,
            r"\b(young|youthful|child|kid|teen)\b": 6,
            r"\b(energetic|excited|happy|loud|cheerful)\b": 7,
            r"\b(calm|smooth|soothing|soft|quiet)\b": 8,
            r"\b(narrator|neutral|standard|flat)\b": 9,
            r"\b(watson)\b": 10,
            r"\b(holmes)\b": 11,
            r"\b(irish|dublin)\b": 12,
            r"\b(scottish|scotland|highland)\b": 13,
            r"\b(american|us|usa|yankee)\b": 14,
            r"\b(deep|low|baritone|bass)\b": 15,
            r"\b(high|squeaky|treble)\b": 16,
            r"\b(authoritative|stern|commanding|bossy)\b": 17,
        }
        
        lowered = text.lower()
        found_any = False
        for pattern, idx in mappings.items():
            if re.search(pattern, lowered):
                vector[idx] = 1.0
                found_any = True
                
        # If no keywords are found, use hash-based deterministic pseudo-random embedding
        if not found_any:
            import hashlib
            h = hashlib.sha256(text.encode("utf-8")).digest()
            for idx in range(128):
                vector[idx] = float(h[idx % len(h)]) / 255.0
                
        # Normalize to unit length for cosine similarity
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector.tolist()

    def register_voice(self, voice_name: str, voice_ref_path: str, description: str) -> str:
        """Saves a cloned voice in Qdrant with its CLAP embedding and metadata."""
        voice_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, voice_name))
        vector = self.generate_clap_embedding(description)
        
        point = PointStruct(
            id=voice_id,
            vector=vector,
            payload={
                "voice_name": voice_name,
                "voice_ref_path": voice_ref_path,
                "description": description
            }
        )
        
        self.client.upsert(
            collection_name="voice_marketplace",
            points=[point]
        )
        logger.info(f"Registered voice '{voice_name}' in Qdrant marketplace.")
        return voice_id

    def list_all(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Every listing in the marketplace (browse view, no query)."""
        points, _ = self.client.scroll(
            collection_name="voice_marketplace",
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [{
            "voice_id": p.id,
            "voice_name": p.payload.get("voice_name"),
            "voice_ref_path": p.payload.get("voice_ref_path"),
            "description": p.payload.get("description"),
            "seller": p.payload.get("seller"),
            "price_usd": p.payload.get("price_usd"),
            "consent_confirmed": p.payload.get("consent_confirmed"),
        } for p in points]

    def search_marketplace(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Queries Qdrant for semantic similarity match against CLAP embeddings."""
        query_vector = self.generate_clap_embedding(query)
        
        results = self.client.query_points(
            collection_name="voice_marketplace",
            query=query_vector,
            limit=limit
        )
        
        hits = []
        for hit in results.points:
            hits.append({
                "voice_id": hit.id,
                "voice_name": hit.payload["voice_name"],
                "voice_ref_path": hit.payload["voice_ref_path"],
                "description": hit.payload["description"],
                "score": hit.score
            })
            
        return hits

    # ----------------------------------------------------
    # Seller onboarding: raw voice acting -> validated, licensed listing
    # ----------------------------------------------------

    def onboard_voice(self, seller_name: str, voice_name: str, sample_wav_paths: List[str],
                      description: str, price_usd: float = 0.0, consent_confirmed: bool = False) -> Dict[str, Any]:
        """Full seller pipeline: validate samples -> denoise -> assemble the cloning
        reference -> register a listing with provenance and consent metadata.

        Returns the listing record. Raises ValueError on unusable input -- a bad
        reference silently degrading every book it's cast in is the worst outcome.
        """
        import wave as _wave
        import shutil
        import subprocess

        if not consent_confirmed:
            raise ValueError("Seller consent must be explicitly confirmed before listing a voice.")
        if not sample_wav_paths:
            raise ValueError("At least one sample WAV is required.")

        total_dur = 0.0
        for p in sample_wav_paths:
            if not os.path.exists(p):
                raise ValueError(f"Sample not found: {p}")
            with _wave.open(p) as w:
                total_dur += w.getnframes() / float(w.getframerate())
        if total_dur < 6.0:
            raise ValueError(f"Samples total {total_dur:.1f}s; at least 6s of clean speech is required for reliable cloning (30-60s recommended).")

        voice_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{seller_name}:{voice_name}"))
        voice_dir = os.path.join("data", "voice_marketplace", voice_id)
        os.makedirs(voice_dir, exist_ok=True)

        # Denoise each sample (DeepFilterNet if installed, SciPy gate fallback),
        # then concatenate into the single mono cloning reference.
        from src.voice_synthesizer import denoise_audio_file
        cleaned = []
        for i, p in enumerate(sample_wav_paths):
            out = os.path.join(voice_dir, f"sample_{i}_clean.wav")
            try:
                if not denoise_audio_file(p, out):
                    shutil.copy(p, out)
            except Exception:
                shutil.copy(p, out)
            cleaned.append(out)
        concat_list = os.path.join(voice_dir, "concat.txt")
        with open(concat_list, "w") as f:
            for p in cleaned:
                f.write(f"file '{os.path.abspath(p)}'\n")
        reference_wav = os.path.join(voice_dir, "reference_mono.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
             "-ac", "1", "-ar", "24000", reference_wav],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )

        # Single listing under the seller-scoped id (register_voice would create a
        # second point under a name-only id -- duplicate listings)
        vector = self.generate_clap_embedding(description)
        self.client.upsert(collection_name="voice_marketplace", points=[PointStruct(
            id=voice_id, vector=vector, payload={
                "voice_name": voice_name,
                "voice_ref_path": reference_wav,
                "description": description,
                "seller": seller_name,
                "price_usd": price_usd,
                "consent_confirmed": True,
                "sample_seconds": round(total_dur, 1),
                "status": "listed",
            },
        )])
        logger.info(f"Onboarded voice '{voice_name}' by {seller_name} ({total_dur:.0f}s of samples) -> {reference_wav}")
        return {"voice_id": voice_id, "reference_wav": reference_wav, "sample_seconds": total_dur}

    # ----------------------------------------------------
    # Purchase / licensing ledger + casting integration
    # ----------------------------------------------------

    LICENSE_LEDGER = os.path.join("data", "voice_marketplace", "licenses.json")

    def purchase_voice(self, voice_id: str, buyer: str, purpose: str) -> Dict[str, Any]:
        """Records a license grant (append-only ledger). Payment processing is out
        of scope here -- this records WHAT was licensed, to WHOM, and FOR WHAT
        PURPOSE, which is the part the production pipeline must respect."""
        import json as _json
        import time as _time
        points = self.client.retrieve(collection_name="voice_marketplace", ids=[voice_id])
        if not points:
            raise ValueError(f"No such voice listing: {voice_id}")
        payload = points[0].payload
        license_record = {
            "license_id": str(uuid.uuid4()),
            "voice_id": voice_id,
            "voice_name": payload.get("voice_name"),
            "voice_ref_path": payload.get("voice_ref_path"),
            "seller": payload.get("seller", "(seed)"),
            "buyer": buyer,
            "purpose": purpose,
            "price_usd": payload.get("price_usd", 0.0),
            "granted_at": _time.time(),
        }
        os.makedirs(os.path.dirname(self.LICENSE_LEDGER), exist_ok=True)
        ledger = []
        if os.path.exists(self.LICENSE_LEDGER):
            with open(self.LICENSE_LEDGER) as f:
                ledger = _json.load(f)
        ledger.append(license_record)
        with open(self.LICENSE_LEDGER, "w") as f:
            _json.dump(ledger, f, indent=2)
        logger.info(f"License granted: '{payload.get('voice_name')}' -> {buyer} for {purpose!r}")
        return license_record

    def cast_character(self, character_name: str, character_description: str,
                       buyer: str = "local", purpose: str = "") -> Optional[Dict[str, Any]]:
        """The marketplace<->production bridge: search listings by a character
        description (era/age/accent/register -- typically derived from the book
        bible + the character's role), license the best match, and bind it to the
        character's MemPalace drawer so every synthesis of that character uses the
        purchased voice. Zero-shot XTTS conditioning makes the 'training' instant."""
        hits = self.search_marketplace(character_description, limit=3)
        hits = [h for h in hits if os.path.exists(h["voice_ref_path"])]
        if not hits:
            logger.warning(f"No castable marketplace voice found for '{character_name}' ({character_description[:50]!r})")
            return None
        best = hits[0]
        license_record = self.purchase_voice(best["voice_id"], buyer=buyer, purpose=purpose or f"cast as {character_name}")

        from src.spatial_memory import MemPalace
        palace = MemPalace(use_chroma=False)
        try:
            import json as _json
            if not palace.get_character_drawer(character_name):
                palace.register_character(character_name=character_name, voice_ref_path=best["voice_ref_path"], speed=1.0, pitch=0.0)
            else:
                drawer = palace.get_character_drawer(character_name)
                config = drawer["modulation_config"]
                config.pop("xtts_speaker", None)  # purchased reference beats builtin pin
                palace.conn.execute(
                    "UPDATE drawers SET voice_ref_path = ?, modulation_config_json = ? WHERE character_name = ?",
                    (best["voice_ref_path"], _json.dumps(config), character_name),
                )
                palace.conn.commit()
        finally:
            palace.close()
        logger.info(f"Cast '{character_name}' with marketplace voice '{best['voice_name']}' (score {best['score']:.2f})")
        return {"character": character_name, "voice": best, "license": license_record}

    def _seed_default_marketplace(self):
        """Seeds the Voice Marketplace with standard acoustic profiles."""
        logger.info("Seeding default voice profiles in Qdrant marketplace...")
        default_voices = [
            (
                "Old Salt Captain",
                "data/voice_references/captain_mono.wav",
                "A raspy, elderly British sea captain voice with deep nautical gravel and stern tone"
            ),
            (
                "Sherlock Holmes",
                "data/voice_references/holmes_mono.wav",
                "An energetic, commanding, authoritative British male voice with high intelligence tone"
            ),
            (
                "Dr. John Watson",
                "data/voice_references/watson_mono.wav",
                "A calm, soothing, deep British male narrator voice with mature stability"
            ),
            (
                "Maternal Storyteller",
                "data/voice_references/narrator_mono.wav",
                "A soft, soothing, calm female storyteller voice with warm maternal caring tone"
            ),
            (
                "Dublin Scholar",
                "data/voice_references/irish_scholar.wav",
                "A young, energetic Irish male voice with academic accent"
            ),
        ]
        for name, ref, desc in default_voices:
            self.register_voice(name, ref, desc)


def main():
    import argparse
    import json as _json
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Firespeaker Voice Marketplace")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("onboard", help="Seller: list a voice from sample WAVs")
    p.add_argument("--seller", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--samples", nargs="+", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--price", type=float, default=0.0)
    p.add_argument("--consent", action="store_true", help="Seller confirms consent to license this voice")

    p = sub.add_parser("search", help="Browse listings by description")
    p.add_argument("query")

    p = sub.add_parser("cast", help="License the best-matching voice and bind it to a character drawer")
    p.add_argument("--character", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--buyer", default="local")

    args = parser.parse_args()
    m = VoiceMarketplace()
    if args.cmd == "onboard":
        print(_json.dumps(m.onboard_voice(args.seller, args.name, args.samples, args.description, args.price, args.consent), indent=2))
    elif args.cmd == "search":
        for h in m.search_marketplace(args.query):
            exists = "" if os.path.exists(h["voice_ref_path"]) else "  [reference missing]"
            print(f"{h['score']:.2f}  {h['voice_name']:24s} {h['description'][:60]}{exists}")
    elif args.cmd == "cast":
        result = m.cast_character(args.character, args.description, buyer=args.buyer)
        print(_json.dumps(result, indent=2, default=str) if result else "No castable voice found.")


if __name__ == "__main__":
    main()
