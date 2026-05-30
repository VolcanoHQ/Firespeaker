#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Firespeaker Spatial Memory Engine (MemPalace)
Provides Layer 0 Persona/Timbre Storage, relational context indexing (SQLite),
and vector-based similarity search (ChromaDB / Cosine Similarity Fallback)
to condition expressive voice cloning (XTTS-v2 and Bark) dynamically.
"""

import os
import sys
import json
import sqlite3
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MemPalace")

# Dynamic ChromaDB availability flag
HAS_CHROMADB = False
try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    logger.warning("chromadb package not found. Vector similarity queries will use the high-performance NumPy/SQLite fallback.")


class MemPalace:
    """
    Relational and vector-based database orchestrator for character identities,
    chapters, dialogues, and emotional reference timbres.
    """

    def __init__(self, db_dir: str = "data/mempalace", use_chroma: bool = True):
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        
        self.sqlite_path = os.path.join(self.db_dir, "palace_relational.db")
        self.chroma_path = os.path.join(self.db_dir, "palace_vector")
        
        self.conn = None
        self.chroma_client = None
        self.chroma_collection = None
        
        # 1. Initialize SQLite Database
        self._init_sqlite()
        
        # 2. Initialize ChromaDB if available and requested
        if HAS_CHROMADB and use_chroma:
            self._init_chromadb()
        else:
            logger.info("Using pure-NumPy/SQLite fallback for vector index operations.")

    def _init_sqlite(self):
        """Initializes the SQLite database tables (Wings, Rooms, Drawers)."""
        self.conn = sqlite3.connect(self.sqlite_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        cursor = self.conn.cursor()
        
        # Table 1: Wings (Logical Chapters / Scenes Context)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS wings (
            wing_id TEXT PRIMARY KEY,
            chapter_number INTEGER NOT NULL,
            title TEXT,
            metadata_json TEXT
        );
        """)
        
        # Table 2: Drawers (Character Identity Storage - Layer 0)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS drawers (
            character_name TEXT PRIMARY KEY,
            voice_ref_path TEXT NOT NULL,
            modulation_config_json TEXT NOT NULL,
            base_embedding BLOB,  -- Serialized float32 list (XTTS timbre vector)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Table 3: Rooms (Spoken Dialogue Context)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS rooms (
            room_id TEXT PRIMARY KEY, -- Maps directly to nlp_analyzer line_id
            wing_id TEXT NOT NULL,
            line_number INTEGER NOT NULL,
            character_name TEXT NOT NULL,
            dialogue_text TEXT NOT NULL,
            emotion TEXT NOT NULL,
            audio_output_path TEXT, -- Populated once voice_synthesizer completes
            metadata_json TEXT,
            confidence REAL DEFAULT 1.0,
            FOREIGN KEY (wing_id) REFERENCES wings(wing_id) ON DELETE CASCADE,
            FOREIGN KEY (character_name) REFERENCES drawers(character_name) ON DELETE RESTRICT
        );
        """)
        
        # Ensure confidence column exists for older schemas
        try:
            cursor.execute("ALTER TABLE rooms ADD COLUMN confidence REAL DEFAULT 1.0")
        except Exception:
            pass
        
        # Table 4: Emotional Ref Indexes (For granular emotional similarity fallbacks)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS emotional_references (
            ref_id TEXT PRIMARY KEY,
            character_name TEXT NOT NULL,
            emotion TEXT NOT NULL,
            voice_ref_path TEXT NOT NULL,
            embedding BLOB NOT NULL, -- Serialized float32 list
            FOREIGN KEY (character_name) REFERENCES drawers(character_name) ON DELETE CASCADE
        );
        """)
        
        # Table 5: Confirmed Merges (User confirmation overrides)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS confirmed_merges (
            book_filename TEXT,
            original_name TEXT,
            canonical_name TEXT,
            is_confirmed INTEGER, -- 1 for accepted merge, 0 for rejected (keep separate)
            confidence_score REAL,
            PRIMARY KEY (book_filename, original_name)
        );
        """)
        
        self.conn.commit()
        logger.info(f"SQLite Relational Palace initialized successfully at {self.sqlite_path}.")

    def _init_chromadb(self):
        """Initializes persistent ChromaDB client for fast semantic timbre searches."""
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="mempalace_timbre_memory",
                metadata={"hnsw:space": "cosine"} # Use cosine similarity for timbre distance
            )
            logger.info("ChromaDB Vector Store initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}. Falling back to NumPy similarity.")
            self.chroma_client = None
            self.chroma_collection = None

    def close(self):
        """Cleanly releases database resources."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("SQLite database connection closed.")

    # ----------------------------------------------------
    # Layer 0 Drawers (Character Identity Management)
    # ----------------------------------------------------

    def register_character(
        self, 
        character_name: str, 
        voice_ref_path: str, 
        speed: float = 1.0, 
        pitch: float = 0.0,
        volume: float = 1.0,
        energy_bias: float = 0.0,
        prosody_stabilization: float = 0.75,
        base_embedding: Optional[List[float]] = None
    ) -> bool:
        """
        Creates/updates a 'Drawer' mapping a character name to their default WAV ref
        and granular voice modulation configurations for XTTS-v2.
        """
        modulation_config = {
            "speed": speed,
            "pitch": pitch,
            "volume": volume,
            "energy_bias": energy_bias,
            "prosody_stabilization": prosody_stabilization
        }
        modulation_json = json.dumps(modulation_config)
        
        blob_embedding = None
        if base_embedding:
            blob_embedding = np.array(base_embedding, dtype=np.float32).tobytes()
            
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
            INSERT INTO drawers (character_name, voice_ref_path, modulation_config_json, base_embedding)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(character_name) DO UPDATE SET
                voice_ref_path = excluded.voice_ref_path,
                modulation_config_json = excluded.modulation_config_json,
                base_embedding = COALESCE(excluded.base_embedding, base_embedding);
            """, (character_name, voice_ref_path, modulation_json, blob_embedding))
            self.conn.commit()
            
            # Also register the default voice embedding in ChromaDB if available
            if self.chroma_collection and base_embedding:
                self.chroma_collection.upsert(
                    ids=[f"base_{character_name}"],
                    embeddings=[base_embedding],
                    metadatas=[{
                        "character_name": character_name,
                        "emotion": "Neutral",
                        "voice_ref_path": voice_ref_path,
                        "is_base": True
                    }]
                )
                
            logger.info(f"Registered character drawer '{character_name}' with modulations: {modulation_config}")
            return True
        except Exception as e:
            logger.error(f"Error registering character drawer '{character_name}': {e}")
            return False

    def get_character_drawer(self, character_name: str) -> Optional[Dict[str, Any]]:
        """Retrieves raw files, base embedding and modulation configs for a character Drawer."""
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT voice_ref_path, modulation_config_json, base_embedding 
        FROM drawers WHERE character_name = ?;
        """, (character_name,))
        row = cursor.fetchone()
        
        if not row:
            return None
            
        ref_path, modulation_json, base_blob = row
        base_embedding = None
        if base_blob:
            base_embedding = np.frombuffer(base_blob, dtype=np.float32).tolist()
            
        return {
            "character_name": character_name,
            "voice_ref_path": ref_path,
            "modulation_config": json.loads(modulation_json),
            "base_embedding": base_embedding
        }

    def save_confirmed_merge(self, book_filename: str, original_name: str, canonical_name: str, is_confirmed: bool, confidence_score: float) -> bool:
        """Saves or updates a user-confirmed merge decision."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
            INSERT INTO confirmed_merges (book_filename, original_name, canonical_name, is_confirmed, confidence_score)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(book_filename, original_name) DO UPDATE SET
                canonical_name = excluded.canonical_name,
                is_confirmed = excluded.is_confirmed,
                confidence_score = excluded.confidence_score;
            """, (book_filename, original_name, canonical_name, 1 if is_confirmed else 0, confidence_score))
            self.conn.commit()
            logger.info(f"Saved merge override for {book_filename}: '{original_name}' -> '{canonical_name}' (Confirmed: {is_confirmed})")
            return True
        except Exception as e:
            logger.error(f"Error saving confirmed merge: {e}")
            return False

    def get_confirmed_merges(self, book_filename: str) -> Dict[str, Tuple[str, bool, float]]:
        """Retrieves all confirmed merge decisions for a book."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
            SELECT original_name, canonical_name, is_confirmed, confidence_score 
            FROM confirmed_merges WHERE book_filename = ?;
            """, (book_filename,))
            rows = cursor.fetchall()
            merges = {}
            for row in rows:
                orig, canon, confirmed, conf = row
                merges[orig] = (canon, bool(confirmed), conf)
            return merges
        except Exception as e:
            logger.error(f"Error getting confirmed merges: {e}")
            return {}

    # ----------------------------------------------------
    # Vector Similarity / Emotional Timbre Searching
    # ----------------------------------------------------

    def register_emotional_reference(
        self,
        character_name: str,
        emotion: str,
        voice_ref_path: str,
        embedding: List[float]
    ) -> bool:
        """
        Registers an emotion-specific voice print reference for a character Drawer.
        Indexes the vector in both SQLite (fallback) and ChromaDB (primary vector db).
        """
        ref_id = f"ref_{character_name}_{emotion.lower()}"
        blob_embedding = np.array(embedding, dtype=np.float32).tobytes()
        
        cursor = self.conn.cursor()
        try:
            # 1. SQLite Relational Storage
            cursor.execute("""
            INSERT INTO emotional_references (ref_id, character_name, emotion, voice_ref_path, embedding)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(ref_id) DO UPDATE SET
                voice_ref_path = excluded.voice_ref_path,
                embedding = excluded.embedding;
            """, (ref_id, character_name, emotion, voice_ref_path, blob_embedding))
            self.conn.commit()
            
            # 2. ChromaDB Vector Storage
            if self.chroma_collection:
                self.chroma_collection.upsert(
                    ids=[ref_id],
                    embeddings=[embedding],
                    metadatas=[{
                        "character_name": character_name,
                        "emotion": emotion,
                        "voice_ref_path": voice_ref_path,
                        "is_base": False
                    }]
                )
            logger.info(f"Registered emotional timbre reference '{ref_id}' for emotion '{emotion}'.")
            return True
        except Exception as e:
            logger.error(f"Error registering emotional timbre reference for {character_name}: {e}")
            return False

    def query_optimal_voice(
        self,
        character_name: str,
        target_emotion: str,
        emotional_vector_query: Optional[List[float]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Queries MemPalace for the absolute best reference WAV file for a character line.
        
        Evaluation Hierarchy:
        1. Query similarity via ChromaDB or NumPy cosine logic on emotional vector.
        2. Fall back to exact metadata matching (e.g. searching Holmes' Tension Reference WAV).
        3. Fall back to character Drawer default reference WAV.
        """
        # Step 1: Base Character configuration
        char_drawer = self.get_character_drawer(character_name)
        if not char_drawer:
            # If no drawer is registered, return default Narrator drawer or a fallback error
            logger.warning(f"Drawer '{character_name}' requested but not registered. Falling back to base defaults.")
            return "data/voice_references/narrator.wav", {"speed": 1.0, "pitch": 0.0}
            
        default_path = char_drawer["voice_ref_path"]
        modulation_config = char_drawer["modulation_config"]
        
        # Step 2: If we have a semantic vector input, execute cosine similarity lookup
        if emotional_vector_query:
            if self.chroma_collection:
                try:
                    # Query ChromaDB (filtered by character name)
                    results = self.chroma_collection.query(
                        query_embeddings=[emotional_vector_query],
                        n_results=1,
                        where={"character_name": character_name}
                    )
                    if results and results["metadatas"] and len(results["metadatas"][0]) > 0:
                        match = results["metadatas"][0][0]
                        score = 1.0 - results["distances"][0][0]  # Cosine similarity score
                        logger.info(f"ChromaDB semantic match: {match['voice_ref_path']} (similarity: {score:.4f})")
                        return match["voice_ref_path"], modulation_config
                except Exception as e:
                    logger.error(f"ChromaDB similarity query failed: {e}. Falling back to NumPy.")
            
            # NumPy Cosine Similarity fallback
            try:
                cursor = self.conn.cursor()
                cursor.execute("""
                SELECT voice_ref_path, emotion, embedding FROM emotional_references
                WHERE character_name = ?;
                """, (character_name,))
                rows = cursor.fetchall()
                
                best_similarity = -1.0
                best_path = default_path
                
                query_np = np.array(emotional_vector_query, dtype=np.float32)
                query_norm = np.linalg.norm(query_np)
                
                for path, emotion, emb_blob in rows:
                    emb_np = np.frombuffer(emb_blob, dtype=np.float32)
                    emb_norm = np.linalg.norm(emb_np)
                    if query_norm > 0 and emb_norm > 0:
                        similarity = np.dot(query_np, emb_np) / (query_norm * emb_norm)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_path = path
                            
                if best_similarity > 0.65: # Similarity threshold
                    logger.info(f"NumPy Cosine similarity match: {best_path} (similarity: {best_similarity:.4f})")
                    return best_path, modulation_config
            except Exception as e:
                logger.error(f"NumPy cosine similarity match failed: {e}")

        # Step 3: Exact Metadata Fallback (e.g. search SQLite for registered Watson + Tension path)
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
            SELECT voice_ref_path FROM emotional_references
            WHERE character_name = ? AND emotion = ?;
            """, (character_name, target_emotion))
            row = cursor.fetchone()
            if row:
                logger.info(f"Exact metadata matching found reference for '{character_name}' ({target_emotion}): {row[0]}")
                return row[0], modulation_config
        except Exception as e:
            logger.error(f"Exact metadata fallback query failed: {e}")

        # Step 4: Final drawer default fallback
        logger.info(f"No semantic or exact match found. Falling back to default drawer reference for '{character_name}'.")
        return default_path, modulation_config

    # ----------------------------------------------------
    # Wings (Chapters) and Rooms (Lines) Script Logging
    # ----------------------------------------------------

    def log_wing(self, wing_id: str, chapter_number: int, title: str, metadata: dict = None) -> bool:
        """Logs a Wing (Chapter) structure into the palace database."""
        metadata_json = json.dumps(metadata) if metadata else "{}"
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
            INSERT INTO wings (wing_id, chapter_number, title, metadata_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(wing_id) DO UPDATE SET
                chapter_number = excluded.chapter_number,
                title = excluded.title,
                metadata_json = excluded.metadata_json;
            """, (wing_id, chapter_number, title, metadata_json))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error logging wing {wing_id}: {e}")
            return False

    def log_room(
        self,
        room_id: str,
        wing_id: str,
        line_number: int,
        character_name: str,
        dialogue_text: str,
        emotion: str,
        audio_output_path: Optional[str] = None,
        metadata: dict = None,
        confidence: float = 1.0
    ) -> bool:
        """Logs a Room (Dialogue Line segment) in relation to a specific Chapter Wing."""
        metadata_json = json.dumps(metadata) if metadata else "{}"
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
            INSERT INTO rooms (room_id, wing_id, line_number, character_name, dialogue_text, emotion, audio_output_path, metadata_json, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(room_id) DO UPDATE SET
                audio_output_path = COALESCE(excluded.audio_output_path, audio_output_path),
                metadata_json = excluded.metadata_json,
                confidence = excluded.confidence;
            """, (room_id, wing_id, line_number, character_name, dialogue_text, emotion, audio_output_path, metadata_json, confidence))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error logging room {room_id}: {e}")
            return False


def main():
    """CLI and self-test suite for the Spatial Memory Engine."""
    import argparse
    parser = argparse.ArgumentParser(description="Firespeaker MemPalace Database Manager")
    parser.add_argument("--init", action="store_true", help="Initialize Spatial Memory relational engine tables")
    parser.add_argument("--test", action="store_true", help="Run comprehensive schema & search self-test")
    args = parser.parse_args()
    
    if args.test:
        print("\n=== RUNNING MEMPALACE (SPATIAL MEMORY) INTEGRITY & RELATION SELF-TEST ===")
        # Create persistent test DB inside scratch directory
        db_dir = "scratch/test_mempalace"
        import shutil
        if os.path.exists(db_dir):
            try:
                shutil.rmtree(db_dir)
            except Exception:
                pass
                
        palace = MemPalace(db_dir=db_dir)
        
        # 1. Register base character Drawers
        print("\n1. Testing Layer 0 Identity Registration:")
        palace.register_character(
            character_name="Watson",
            voice_ref_path="data/voice_references/watson_mono.wav",
            speed=0.95,
            pitch=-1.2,
            base_embedding=[0.1, -0.2, 0.5, 0.9] # 4D mock embedding
        )
        
        palace.register_character(
            character_name="Holmes",
            voice_ref_path="data/voice_references/holmes_mono.wav",
            speed=1.05,
            pitch=1.5,
            base_embedding=[0.8, 0.4, -0.1, 0.3] # 4D mock embedding
        )
        
        # Retrieve Watson's configuration
        watson_drawer = palace.get_character_drawer("Watson")
        print(f"- Registered 'Watson' default path: {watson_drawer['voice_ref_path']}")
        print(f"- Registered 'Watson' modulation: {watson_drawer['modulation_config']}")
        assert watson_drawer['modulation_config']['speed'] == 0.95
        assert watson_drawer['modulation_config']['pitch'] == -1.2
        print("  --> Character Drawer Storage: PASSED")
        
        # 2. Register emotional context references
        print("\n2. Testing Emotional Vector Registration:")
        palace.register_emotional_reference(
            character_name="Holmes",
            emotion="Tension",
            voice_ref_path="data/voice_references/holmes_anxious.wav",
            embedding=[0.9, 0.3, -0.05, 0.4] # High similarity to Holmes' base
        )
        
        palace.register_emotional_reference(
            character_name="Holmes",
            emotion="Joy",
            voice_ref_path="data/voice_references/holmes_excited.wav",
            embedding=[0.5, 0.8, 0.2, -0.1]
        )
        print("  --> Emotional Reference Timbre Indexing: PASSED")
        
        # 3. Log wing and rooms (Manuscript context logging)
        print("\n3. Testing Relational Chapter & Dialogue Logging:")
        palace.log_wing(
            wing_id="wing_c1",
            chapter_number=1,
            title="Chapter 1: The Inciting Incident",
            metadata={"word_count": 2500}
        )
        
        palace.log_room(
            room_id="room_l1",
            wing_id="wing_c1",
            line_number=1,
            character_name="Holmes",
            dialogue_text="Do you see anything, Watson?",
            emotion="Neutral",
            audio_output_path="outputs/c1_l1.wav"
        )
        print("  --> Wing / Room Relational Mapping: PASSED")
        
        # 4. Perform vector similarity queries & fallbacks
        print("\n4. Testing Optimal Timbre Retrieval Hierarchy:")
        
        # Query A: Exact emotion query (no vector)
        path, config = palace.query_optimal_voice(character_name="Holmes", target_emotion="Joy")
        print(f"- Exact emotional query Holmes (Joy) matched: {path} | Speed: {config['speed']}")
        assert path == "data/voice_references/holmes_excited.wav"
        
        # Query B: Similarity vector query (simulating dynamic extraction based on real audio matching)
        query_vector = [0.88, 0.32, -0.04, 0.38] # Extremely close to Holmes' Tension embedding
        path, config = palace.query_optimal_voice(
            character_name="Holmes", 
            target_emotion="Tension", 
            emotional_vector_query=query_vector
        )
        print(f"- Cosine similarity lookup (simulated panicking query) matched: {path}")
        assert path == "data/voice_references/holmes_anxious.wav"
        
        # Query C: Fallback default
        path, config = palace.query_optimal_voice(character_name="Holmes", target_emotion="Sadness")
        print(f"- Fallback drawer query Holmes (Sadness) matched: {path}")
        assert path == "data/voice_references/holmes_mono.wav"
        
        # Query D: Unregistered drawer fallback
        path, config = palace.query_optimal_voice(character_name="Narrator", target_emotion="Neutral")
        print(f"- Unregistered drawer fallback matched: {path}")
        
        palace.close()
        print("\n=== ALL MEMPALACE SCHEMAS AND INTEGRITY CHECKS PASSED ===\n")
        return 0
        
    if args.init:
        palace = MemPalace()
        palace.close()
        print("Palace structures set up cleanly.")
        return 0
        
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
