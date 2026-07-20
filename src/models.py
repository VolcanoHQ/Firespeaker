#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Volcano Studios - Pydantic Data Models
Centralizes standard validation schemas for manuscript manifests, chapters, scenes, and lines.
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any, Optional

class PerformanceMetrics(BaseModel):
    pitch_modifier: float = Field(default=1.0)
    speed_modifier: float = Field(default=1.0)
    delivery_style: str = Field(default="neutral_narrative")

class ScriptLine(BaseModel):
    line_id: str
    chapter: int
    scene: int
    line_number: int
    character: str
    speaker_id: str
    segment_type: Literal["dialogue", "narrative"]
    text: str
    
    # Auto-populated defaults if missing
    emotion: str = Field(default="Neutral")
    performance: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    post_padding_ms: int = Field(default=250)
    attribution_method: str = Field(default="Tier 1 Default")
    confidence: float = Field(default=1.0)
    speaker_locked: bool = Field(default=True)
    # "speech" = normal spoken words; "vocalization" = non-lexical utterance
    # (sneeze like "Kertyschoo!", gasp, cry) that TTS should not read as words
    utterance_type: Literal["speech", "vocalization"] = Field(default="speech")

class ScenePayload(BaseModel):
    scene_id: str
    lines: List[ScriptLine]

class ChapterPayload(BaseModel):
    chapter_id: str
    title: str
    scenes: List[ScenePayload]

class PartPayload(BaseModel):
    part_id: str
    title: str
    chapters: List[ChapterPayload]

class ManuscriptManifest(BaseModel):
    source_file: str
    total_parts: int
    total_chapters: int
    total_scenes: int
    parts: List[PartPayload]
