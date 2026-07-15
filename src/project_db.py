#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Firespeaker Project State & Orchestration Database
Manages relational states for projects, chapters, scene slicing, and lookahead queues.
"""

import os
import json
import sqlite3
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("ProjectDB")

class ProjectDB:
    """Manages SQLite storage for the Backend Orchestration state."""

    def __init__(self, db_path: str = "data/projects.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self):
        """Initializes database tables for projects and chapters."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # Projects Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                status TEXT NOT NULL, -- awaiting_macro_approval, processing_lookahead, completed, failed
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Chapters Table (to track Meso-structure and background processing state)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS chapters (
                project_id TEXT NOT NULL,
                chapter_id TEXT NOT NULL,
                part_id TEXT NOT NULL,
                part_title TEXT NOT NULL,
                chapter_title TEXT NOT NULL,
                text_block TEXT NOT NULL,
                status TEXT NOT NULL, -- pending, processing, completed, failed
                scene_data TEXT, -- JSON-serialized list of scenes with lines
                order_idx INTEGER NOT NULL,
                PRIMARY KEY (project_id, chapter_id),
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );
            """)
            conn.commit()
            logger.info("Project database initialized successfully.")

    def create_project(self, project_id: str, filename: str, status: str) -> bool:
        """Inserts a new project record."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO projects (id, filename, status) VALUES (?, ?, ?);",
                    (project_id, filename, status)
                )
                conn.commit()
                return True
            except sqlite3.Error as e:
                logger.error(f"Failed to create project: {e}")
                return False

    def update_project_status(self, project_id: str, status: str) -> bool:
        """Updates the status of a project."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "UPDATE projects SET status = ? WHERE id = ?;",
                    (status, project_id)
                )
                conn.commit()
                return True
            except sqlite3.Error as e:
                logger.error(f"Failed to update project status: {e}")
                return False

    def get_project_status(self, project_id: str) -> Optional[str]:
        """Gets the status of a project."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM projects WHERE id = ?;", (project_id,))
            row = cursor.fetchone()
            return row[0] if row else None

    def get_project_filename(self, project_id: str) -> Optional[str]:
        """Gets the manuscript filename for a project."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filename FROM projects WHERE id = ?;", (project_id,))
            row = cursor.fetchone()
            return row[0] if row else None

    def insert_chapter(
        self,
        project_id: str,
        chapter_id: str,
        part_id: str,
        part_title: str,
        chapter_title: str,
        text_block: str,
        status: str,
        order_idx: int
    ) -> bool:
        """Inserts a new chapter record."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                INSERT INTO chapters (
                    project_id, chapter_id, part_id, part_title, chapter_title, text_block, status, order_idx
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, (project_id, chapter_id, part_id, part_title, chapter_title, text_block, status, order_idx))
                conn.commit()
                return True
            except sqlite3.Error as e:
                logger.error(f"Failed to insert chapter: {e}")
                return False

    def update_chapter_status(self, project_id: str, chapter_id: str, status: str) -> bool:
        """Updates a chapter's background queue processing status."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "UPDATE chapters SET status = ? WHERE project_id = ? AND chapter_id = ?;",
                    (status, project_id, chapter_id)
                )
                conn.commit()
                return True
            except sqlite3.Error as e:
                logger.error(f"Failed to update chapter status: {e}")
                return False

    def save_chapter_scenes(self, project_id: str, chapter_id: str, scenes: List[Dict[str, Any]]) -> bool:
        """Saves parsed scene/line JSON to a chapter record."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            try:
                scenes_json = json.dumps(scenes)
                cursor.execute("""
                UPDATE chapters 
                SET scene_data = ?, status = 'completed'
                WHERE project_id = ? AND chapter_id = ?;
                """, (scenes_json, project_id, chapter_id))
                conn.commit()
                return True
            except sqlite3.Error as e:
                logger.error(f"Failed to save chapter scenes: {e}")
                return False

    def get_chapters(self, project_id: str) -> List[Dict[str, Any]]:
        """Retrieves all chapters for a project, sorted by order_idx."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT chapter_id, part_id, part_title, chapter_title, text_block, status, scene_data, order_idx
            FROM chapters
            WHERE project_id = ?
            ORDER BY order_idx ASC;
            """, (project_id,))
            rows = cursor.fetchall()
            
            chapters = []
            for row in rows:
                scenes = None
                if row[6]:
                    try:
                        scenes = json.loads(row[6])
                    except Exception:
                        pass
                chapters.append({
                    "chapter_id": row[0],
                    "part_id": row[1],
                    "part_title": row[2],
                    "chapter_title": row[3],
                    "text_block": row[4],
                    "status": row[5],
                    "scenes": scenes,
                    "order_idx": row[7]
                })
            return chapters

    def get_chapter(self, project_id: str, chapter_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single chapter's data."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT chapter_id, part_id, part_title, chapter_title, text_block, status, scene_data, order_idx
            FROM chapters
            WHERE project_id = ? AND chapter_id = ?;
            """, (project_id, chapter_id))
            row = cursor.fetchone()
            if not row:
                return None
                
            scenes = None
            if row[6]:
                try:
                    scenes = json.loads(row[6])
                except Exception:
                    pass
            return {
                "chapter_id": row[0],
                "part_id": row[1],
                "part_title": row[2],
                "chapter_title": row[3],
                "text_block": row[4],
                "status": row[5],
                "scenes": scenes,
                "order_idx": row[7]
            }

    def get_project_lookahead_status(self, project_id: str) -> Dict[str, Any]:
        """Compiles the processing state for all chapters in a project."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM projects WHERE id = ?;", (project_id,))
            proj_row = cursor.fetchone()
            if not proj_row:
                return {"error": "Project not found"}
                
            cursor.execute("""
            SELECT chapter_id, chapter_title, status, order_idx
            FROM chapters
            WHERE project_id = ?
            ORDER BY order_idx ASC;
            """, (project_id,))
            rows = cursor.fetchall()
            
            chapters = []
            for row in rows:
                chapters.append({
                    "chapter_id": row[0],
                    "title": row[1],
                    "status": row[2],
                    "order_idx": row[3]
                })
                
            return {
                "project_id": project_id,
                "project_status": proj_row[0],
                "chapters": chapters
            }
