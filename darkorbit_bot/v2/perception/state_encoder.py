"""
V2 State Encoder

Converts tracked objects into neural network features.
Much richer representation than V1's 128-dim state vector.

V2 Enhancement: Now includes visual/pixel features alongside YOLO coordinates.
This provides "Game Sense" - understanding status effects, environmental hazards,
and fine-grained object recognition (e.g., Boss vs Normal enemy).

Visual Feature Integration:
- Strategist: Global context (224x224 -> 512-dim) for environmental awareness
- Tactician: RoI features (per-object -> 128-dim) for object state/identity
- Executor: Local precision (64x64 patch -> 64-dim) for precise aiming
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

from .tracker import TrackedObject
from ..config import (
    ENEMY_CLASSES, LOOT_CLASSES,
    NUM_OBJECTS_IN_FLAT_STATE, ASSUMED_FPS, POSITION_HISTORY_LEN, OBJECT_FEATURE_DIM,
    FULL_STATE_DIM
)

# Conditional import for vision encoder
try:
    from .vision_encoder import VisionEncoder, LightweightColorEncoder, VisionConfig
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    VisionEncoder = None
    LightweightColorEncoder = None
    VisionConfig = None

logger = logging.getLogger(__name__)


@dataclass
class PlayerState:
    """Current player state."""
    x: float = 0.5                  # Normalized position (on screen)
    y: float = 0.5
    vx: float = 0.0                 # Velocity
    vy: float = 0.0
    hp: float = 1.0                 # Health (0-1)
    hp_delta: float = 0.0           # Recent HP change
    shield: float = 1.0             # Shield (0-1)
    shield_delta: float = 0.0
    is_attacking: bool = False
    attack_duration: float = 0.0    # Seconds in current attack
    idle_time: float = 0.0
    last_damage_time: float = 0.0
    # Map position awareness (from minimap)
    map_x: float = 0.5              # Position on actual game map (0-1)
    map_y: float = 0.5
    near_boundary: bool = False     # Whether near map edge
    boundary_distance: float = 1.0  # Distance to nearest boundary (0-1)


class StateEncoderV2:
    """
    Encodes game state into neural network features.

    Provides:
    - Player state features (16 dim)
    - Object features (20 dim per object, padded to max_objects)
    - Global context features (16 dim)
    - Goal-ready format for hierarchical model

    V2 Enhancement - Visual Features:
    - Global visual context (512 dim) for Strategist
    - Per-object visual features (128 dim) for Tactician
    - Local precision features (64 dim) for Executor
    """

    def __init__(self,
                 max_objects: int = 16,
                 object_dim: int = OBJECT_FEATURE_DIM,  # From config.py
                 player_dim: int = 16,
                 context_dim: int = 16,
                 use_visual_features: bool = True,
                 visual_config: Optional[dict] = None):
        """
        Args:
            max_objects: Maximum objects to encode
            object_dim: Features per object (from TrackedObject.to_feature_vector)
            player_dim: Player state features
            context_dim: Global context features
            use_visual_features: Whether to extract visual features
            visual_config: Optional VisionConfig dict
        """
        self.max_objects = max_objects
        self.object_dim = object_dim
        self.player_dim = player_dim
        self.context_dim = context_dim
        self.use_visual_features = use_visual_features and VISION_AVAILABLE

        # Visual feature dimensions
        self.global_visual_dim = 512
        self.roi_visual_dim = 128
        self.local_visual_dim = 64

        # Initialize vision encoder if enabled
        self.vision_encoder = None
        self.lightweight_encoder = None

        if self.use_visual_features:
            try:
                if visual_config and visual_config.get('use_lightweight', False):
                    self.lightweight_encoder = LightweightColorEncoder(feature_dim=32)
                    logger.info("Using LightweightColorEncoder for visual features")
                else:
                    config = VisionConfig() if VisionConfig else None
                    if config and visual_config:
                        # Apply any custom config
                        for k, v in visual_config.items():
                            if hasattr(config, k):
                                setattr(config, k, v)
                    self.vision_encoder = VisionEncoder(config)
                    logger.info("Using VisionEncoder (CNN) for visual features")
            except Exception as e:
                logger.warning(f"Failed to initialize vision encoder: {e}")
                self.use_visual_features = False

        # History for velocity estimation
        self.player_history: List[Tuple[float, float]] = []
        self.hp_history: List[float] = []

        # State tracking
        self.player_state = PlayerState()
        self.last_detections_count = 0

        # Cache for visual features
        self._last_frame = None
        self._global_visual_cache = None
        self._roi_visual_cache = None

    def encode(self,
               tracked_objects: List[TrackedObject],
               player_x: float = 0.5,
               player_y: float = 0.5,
               hp: float = 1.0,
               shield: float = 1.0,
               is_attacking: bool = False,
               idle_time: float = 0.0,
               frame: Optional[np.ndarray] = None,
               current_time_ms: int = 0) -> Dict[str, np.ndarray]:
        """
        Encode current state into features for neural network.

        Args:
            tracked_objects: List of TrackedObject from tracker
            player_x, player_y: Player position (normalized 0-1)
            hp: Player health (0-1)
            shield: Player shield (0-1)
            is_attacking: Whether player is attacking
            idle_time: Time without targets
            frame: Optional BGR frame for visual feature extraction
            current_time_ms: Current timestamp for caching

        Returns:
            Dict with:
                'player': [player_dim] player features
                'objects': [max_objects, object_dim] object features (YOLO only)
                'objects_visual': [max_objects, object_dim + visual_dim] (with visual)
                'object_mask': [max_objects] which objects are valid
                'context': [context_dim] global context
                'full_state': [total_dim] concatenated for simple models
                'global_visual': [global_visual_dim] for Strategist
                'roi_visual': [max_objects, roi_visual_dim] for Tactician
        """
        # Update player state
        self._update_player_state(player_x, player_y, hp, shield, is_attacking, idle_time)

        # Encode player
        player_features = self._encode_player()

        # Encode objects (YOLO-based)
        object_features, object_mask = self._encode_objects(tracked_objects, player_x, player_y)

        # Encode context
        context_features = self._encode_context(tracked_objects, player_x, player_y)

        # Create full state for simple models (flatten objects)
        flat_objects = object_features[:NUM_OBJECTS_IN_FLAT_STATE].flatten()

        full_state = np.concatenate([
            player_features,
            flat_objects,
            context_features
        ])

        result = {
            'player': player_features,
            'objects': object_features,
            'object_mask': object_mask,
            'context': context_features,
            'full_state': full_state,
            'num_objects': min(len(tracked_objects), self.max_objects)
        }

        # Extract visual features if enabled and frame provided
        if self.use_visual_features and frame is not None:
            visual_result = self._encode_visual_features(
                frame, tracked_objects, player_x, player_y, current_time_ms
            )
            result.update(visual_result)

            # Update tracked objects with visual features
            self._update_object_visual_features(tracked_objects, visual_result)

            # Create combined object features (YOLO + visual)
            objects_visual = self._encode_objects_with_visual(
                tracked_objects, player_x, player_y
            )
            result['objects_visual'] = objects_visual

        return result

    def _encode_visual_features(self, frame: np.ndarray,
                                 tracked_objects: List[TrackedObject],
                                 player_x: float, player_y: float,
                                 current_time_ms: int) -> Dict[str, np.ndarray]:
        """Extract visual features from frame."""
        result = {}

        # Get bounding boxes for RoI extraction
        bboxes = []
        for obj in tracked_objects[:self.max_objects]:
            bboxes.append((obj.x, obj.y, obj.width, obj.height))

        if self.vision_encoder is not None:
            # Full CNN-based encoding
            result['global_visual'] = self.vision_encoder.encode_global(
                frame, current_time_ms
            )
            result['roi_visual'] = self.vision_encoder.encode_rois(frame, bboxes)
            result['color_features'] = self.vision_encoder.encode_colors(frame, bboxes)

        elif self.lightweight_encoder is not None:
            # Lightweight color-only encoding
            result['global_visual'] = self.lightweight_encoder.encode_global(frame)

            # Pad to expected dimension
            if len(result['global_visual']) < self.global_visual_dim:
                padded = np.zeros(self.global_visual_dim, dtype=np.float32)
                padded[:len(result['global_visual'])] = result['global_visual']
                result['global_visual'] = padded

            # Per-object color features
            roi_features = np.zeros((self.max_objects, self.roi_visual_dim), dtype=np.float32)
            for i, (cx, cy, w, h) in enumerate(bboxes):
                color_feat = self.lightweight_encoder.encode_object(frame, cx, cy, w, h)
                # Pad to roi_visual_dim
                roi_features[i, :len(color_feat)] = color_feat

            result['roi_visual'] = roi_features
            result['color_features'] = roi_features[:, :32]  # First 32 dims are color

        else:
            # Fallback: zeros
            result['global_visual'] = np.zeros(self.global_visual_dim, dtype=np.float32)
            result['roi_visual'] = np.zeros((self.max_objects, self.roi_visual_dim), dtype=np.float32)
            result['color_features'] = np.zeros((self.max_objects, 32), dtype=np.float32)

        return result

    def _update_object_visual_features(self, tracked_objects: List[TrackedObject],
                                        visual_result: Dict[str, np.ndarray]):
        """Update TrackedObject instances with visual features."""
        roi_visual = visual_result.get('roi_visual')
        color_features = visual_result.get('color_features')

        for i, obj in enumerate(tracked_objects[:self.max_objects]):
            if roi_visual is not None and i < len(roi_visual):
                obj.update_visual_features(
                    visual_features=roi_visual[i],
                    color_features=color_features[i] if color_features is not None else None
                )

    def _encode_objects_with_visual(self, objects: List[TrackedObject],
                                     player_x: float, player_y: float) -> np.ndarray:
        """
        Encode objects with visual features concatenated.

        Returns:
            features: [max_objects, object_dim + roi_visual_dim] combined features
        """
        total_dim = self.object_dim + self.roi_visual_dim
        features = np.zeros((self.max_objects, total_dim), dtype=np.float32)

        # Sort objects by distance to player (closest first)
        sorted_objects = sorted(
            objects,
            key=lambda o: (o.x - player_x)**2 + (o.y - player_y)**2
        )

        for i, obj in enumerate(sorted_objects[:self.max_objects]):
            features[i] = obj.to_feature_vector_with_visual(
                player_x, player_y, self.roi_visual_dim
            )

        return features

    def encode_for_executor(self, frame: np.ndarray,
                            target_x: float, target_y: float) -> np.ndarray:
        """
        Encode local visual features for Executor (precision aiming).

        Args:
            frame: BGR frame
            target_x, target_y: Normalized target position

        Returns:
            Local visual features [local_visual_dim]
        """
        if self.vision_encoder is not None:
            return self.vision_encoder.encode_local(frame, target_x, target_y)
        elif self.lightweight_encoder is not None:
            # Use object encoder around target point
            return self.lightweight_encoder.encode_object(
                frame, target_x, target_y, 0.15, 0.15
            )
        else:
            return np.zeros(self.local_visual_dim, dtype=np.float32)

    def update_map_position(self, map_x: float, map_y: float):
        """
        Update player's position on the game map (from minimap detection).

        Args:
            map_x: Normalized map X (0-1)
            map_y: Normalized map Y (0-1)
        """
        self.player_state.map_x = map_x
        self.player_state.map_y = map_y

        # Calculate boundary distance (distance to nearest edge)
        # Map boundaries are at 0.0 and 1.0
        dist_to_edges = [
            map_x,          # Distance to left edge
            1.0 - map_x,    # Distance to right edge
            map_y,          # Distance to top edge
            1.0 - map_y     # Distance to bottom edge
        ]
        self.player_state.boundary_distance = min(dist_to_edges)
        self.player_state.near_boundary = self.player_state.boundary_distance < 0.15

    def _update_player_state(self, x: float, y: float, hp: float, shield: float,
                              is_attacking: bool, idle_time: float):
        """Update player state with new frame data."""
        # Velocity from position history
        if self.player_history:
            last_x, last_y = self.player_history[-1]
            self.player_state.vx = (x - last_x) * ASSUMED_FPS
            self.player_state.vy = (y - last_y) * ASSUMED_FPS

        self.player_history.append((x, y))
        if len(self.player_history) > POSITION_HISTORY_LEN:
            self.player_history.pop(0)

        # HP delta
        if self.hp_history:
            self.player_state.hp_delta = hp - self.hp_history[-1]
        self.hp_history.append(hp)
        if len(self.hp_history) > POSITION_HISTORY_LEN:
            self.hp_history.pop(0)

        # Shield delta (approximate)
        shield_delta = 0.0
        if hasattr(self, '_last_shield'):
            shield_delta = shield - self._last_shield
        self._last_shield = shield
        self.player_state.shield_delta = shield_delta

        # Update state
        self.player_state.x = x
        self.player_state.y = y
        self.player_state.hp = hp
        self.player_state.shield = shield
        self.player_state.is_attacking = is_attacking
        self.player_state.idle_time = idle_time

        # Track damage time
        if self.player_state.hp_delta < -0.01:
            self.player_state.last_damage_time = 0.0
        else:
            self.player_state.last_damage_time += 1/30  # Approximate

        # Track attack duration
        if is_attacking:
            self.player_state.attack_duration += 1/30
        else:
            self.player_state.attack_duration = 0.0

    def _encode_player(self) -> np.ndarray:
        """Encode player state into feature vector."""
        s = self.player_state

        return np.array([
            # Position (4)
            s.x, s.y, s.vx / 10, s.vy / 10,
            # Health (4)
            s.hp, s.hp_delta * 10, s.shield, s.shield_delta * 10,
            # Combat (4)
            1.0 if s.is_attacking else 0.0,
            min(s.attack_duration / 10, 1.0),
            min(s.last_damage_time / 5, 1.0),
            0.0,  # Reserved
            # Context (4) - NOW INCLUDES MAP AWARENESS
            min(s.idle_time / 5, 1.0),
            s.map_x,  # Map position X
            s.map_y,  # Map position Y
            s.boundary_distance  # Distance to map boundary
        ], dtype=np.float32)

    def _encode_objects(self, objects: List[TrackedObject],
                        player_x: float, player_y: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode tracked objects into fixed-size feature matrix.

        Returns:
            features: [max_objects, object_dim] object features
            mask: [max_objects] boolean mask for valid objects
        """
        features = np.zeros((self.max_objects, self.object_dim), dtype=np.float32)
        mask = np.zeros(self.max_objects, dtype=np.float32)

        # Sort objects by distance to player (closest first)
        sorted_objects = sorted(
            objects,
            key=lambda o: (o.x - player_x)**2 + (o.y - player_y)**2
        )

        for i, obj in enumerate(sorted_objects[:self.max_objects]):
            features[i] = obj.to_feature_vector(player_x, player_y)
            mask[i] = 1.0

        return features, mask

    def _encode_context(self, objects: List[TrackedObject],
                        player_x: float, player_y: float) -> np.ndarray:
        """Encode global context features."""
        # Count objects by type
        enemies = [o for o in objects if o.class_name in ENEMY_CLASSES]
        loot = [o for o in objects if o.class_name in LOOT_CLASSES]

        num_enemies = len(enemies)
        num_loot = len(loot)

        # Nearest distances
        nearest_enemy_dist = 1.0
        nearest_loot_dist = 1.0

        if enemies:
            nearest_enemy_dist = min(
                np.sqrt((e.x - player_x)**2 + (e.y - player_y)**2)
                for e in enemies
            )

        if loot:
            nearest_loot_dist = min(
                np.sqrt((l.x - player_x)**2 + (l.y - player_y)**2)
                for l in loot
            )

        # Threat assessment
        total_threat = sum(1.0 / max(0.1, np.sqrt((e.x - player_x)**2 + (e.y - player_y)**2))
                          for e in enemies) if enemies else 0.0
        total_threat = min(total_threat / 10, 1.0)  # Normalize

        # Loot value (distance-weighted)
        loot_value = sum(1.0 / max(0.1, np.sqrt((l.x - player_x)**2 + (l.y - player_y)**2))
                        for l in loot) if loot else 0.0
        loot_value = min(loot_value / 5, 1.0)  # Normalize

        # Screen position (are we near edges?)
        near_left = 1.0 if player_x < 0.15 else 0.0
        near_right = 1.0 if player_x > 0.85 else 0.0
        near_top = 1.0 if player_y < 0.15 else 0.0
        near_bottom = 1.0 if player_y > 0.85 else 0.0

        return np.array([
            # Object counts (4)
            min(num_enemies / 5, 1.0),
            min(num_loot / 5, 1.0),
            1.0 if num_enemies > 0 else 0.0,
            1.0 if num_loot > 0 else 0.0,
            # Distances (4)
            nearest_enemy_dist,
            nearest_loot_dist,
            total_threat,
            loot_value,
            # Position (4)
            near_left, near_right, near_top, near_bottom,
            # Reserved (4)
            0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)

    def get_state_dim(self) -> int:
        """Get total state dimension for full_state."""
        return self.player_dim + (NUM_OBJECTS_IN_FLAT_STATE * self.object_dim) + self.context_dim

    def reset(self):
        """Reset encoder state."""
        self.player_history.clear()
        self.hp_history.clear()
        self.player_state = PlayerState()


class StateSummarizer:
    """
    Creates temporal summaries for the Strategist.

    Maintains a buffer of recent states and creates downsampled
    summaries for long-term decision making.
    """

    def __init__(self, history_seconds: int = 60, sample_rate_hz: int = 1):
        """
        Args:
            history_seconds: How many seconds of history to maintain
            sample_rate_hz: How many samples per second
        """
        self.history_seconds = history_seconds
        self.sample_rate_hz = sample_rate_hz
        self.max_samples = history_seconds * sample_rate_hz

        self.state_buffer: List[np.ndarray] = []
        self.last_sample_time = 0.0

    def add_state(self, state: np.ndarray, current_time: float):
        """Add a new state, downsampling as needed."""
        sample_interval = 1.0 / self.sample_rate_hz

        if current_time - self.last_sample_time >= sample_interval:
            self.state_buffer.append(state.copy())
            self.last_sample_time = current_time

            # Trim to max size
            if len(self.state_buffer) > self.max_samples:
                self.state_buffer.pop(0)

    def get_history(self, pad_to_length: Optional[int] = None) -> np.ndarray:
        """
        Get state history as array.

        Args:
            pad_to_length: Pad/truncate to this length (default: max_samples)

        Returns:
            [T, state_dim] array of historical states
        """
        target_len = pad_to_length or self.max_samples

        if len(self.state_buffer) == 0:
            # Return zeros if no history
            state_dim = FULL_STATE_DIM
            return np.zeros((target_len, state_dim), dtype=np.float32)

        state_dim = self.state_buffer[0].shape[0]

        # Pad or truncate
        if len(self.state_buffer) >= target_len:
            # Take most recent
            history = np.array(self.state_buffer[-target_len:])
        else:
            # Pad with first state (or zeros)
            padding_size = target_len - len(self.state_buffer)
            if self.state_buffer:
                padding = np.tile(self.state_buffer[0], (padding_size, 1))
            else:
                padding = np.zeros((padding_size, state_dim))
            history = np.vstack([padding, np.array(self.state_buffer)])

        return history.astype(np.float32)

    def reset(self):
        """Reset history buffer."""
        self.state_buffer.clear()
        self.last_sample_time = 0.0
