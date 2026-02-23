"""
V2 Perception Module

- Object tracking with persistent IDs (ByteTrack)
- Rich state encoding for tracked objects
- HUD OCR for reading HP/Shield from game UI
"""

from .tracker import ObjectTracker, TrackedObject
from .state_encoder import StateEncoderV2
from .hud_ocr import HUDReader, HUDConfig, HUDValues, TargetHPReader, create_hud_reader

__all__ = [
    'ObjectTracker', 'TrackedObject', 'StateEncoderV2',
    'HUDReader', 'HUDConfig', 'HUDValues', 'TargetHPReader', 'create_hud_reader'
]
