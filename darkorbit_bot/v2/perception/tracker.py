"""
ByteTrack-style Object Tracker

Assigns persistent IDs to detected objects across frames.
This is CRITICAL for target lock - without it, the bot can't track "enemy #1".

Based on ByteTrack algorithm:
1. High-confidence detections match first
2. Low-confidence detections match remaining tracks
3. Unmatched detections create new tracks
4. Lost tracks are kept for a buffer period
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from collections import deque
import time

from ..config import ENEMY_CLASSES, LOOT_CLASSES, PLAYER_CLASSES, VELOCITY_SCALE, OBJECT_FEATURE_DIM

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """A tracked object with persistent identity."""
    # Identity
    track_id: int                       # Persistent ID
    class_name: str                     # Object class (enemy type, box, etc.)

    # Current state
    x: float                            # Normalized x position (0-1)
    y: float                            # Normalized y position (0-1)
    width: float                        # Bounding box width (normalized)
    height: float                       # Bounding box height (normalized)
    confidence: float                   # Detection confidence

    # Velocity estimate (pixels per frame, normalized)
    vx: float = 0.0
    vy: float = 0.0

    # Tracking metadata
    age: int = 0                        # Total frames since track created
    hits: int = 1                       # Total successful matches
    time_since_update: int = 0          # Frames since last detection match
    state: str = "tracked"              # "tracked", "lost", "removed"

    # History for smoothing
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))

    # Timestamps
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # Visual features (from VisionEncoder)
    visual_features: Optional[np.ndarray] = None    # [roi_dim] from CNN encoder
    color_features: Optional[np.ndarray] = None     # [color_dim] from color stats
    is_glowing: bool = False                        # Detected glow effect
    dominant_color: str = "unknown"                 # "red", "blue", "green", "gold", etc.
    visual_confidence: float = 0.0                  # How confident are visual features

    def __post_init__(self):
        if not self.position_history:
            self.position_history = deque(maxlen=10)
        self.position_history.append((self.x, self.y))

    def predict(self) -> Tuple[float, float]:
        """Predict next position using velocity."""
        pred_x = np.clip(self.x + self.vx, 0, 1)
        pred_y = np.clip(self.y + self.vy, 0, 1)
        return pred_x, pred_y

    def update(self, x: float, y: float, width: float, height: float,
               confidence: float, class_name: str):
        """Update track with new detection."""
        # Update velocity estimate using exponential smoothing
        alpha = 0.3  # Smoothing factor
        new_vx = x - self.x
        new_vy = y - self.y

        # NOISE THRESHOLD: Ignore tiny movements (detection jitter)
        # ~0.01 normalized = ~20 pixels on 1920px screen (increased threshold)
        noise_threshold = 0.01
        displacement = np.sqrt(new_vx**2 + new_vy**2)

        if displacement > noise_threshold:
            # Real movement - update velocity
            self.vx = alpha * new_vx + (1 - alpha) * self.vx
            self.vy = alpha * new_vy + (1 - alpha) * self.vy
        else:
            # Below noise floor - FAST decay (0.3 = 70% reduction per frame)
            self.vx *= 0.3
            self.vy *= 0.3
            # Zero out threshold (more aggressive)
            if abs(self.vx) < 0.003:
                self.vx = 0.0
            if abs(self.vy) < 0.003:
                self.vy = 0.0

        # Update position
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence
        self.class_name = class_name

        # Update tracking stats
        self.hits += 1
        self.time_since_update = 0
        self.state = "tracked"
        self.last_seen = time.time()

        # Add to history
        self.position_history.append((x, y))

    def mark_missed(self):
        """Mark track as missed this frame (no matching detection)."""
        self.time_since_update += 1
        self.age += 1

        # Predict position for next frame
        pred_x, pred_y = self.predict()
        self.x = pred_x
        self.y = pred_y

        if self.time_since_update > 1:
            self.state = "lost"

    def get_bbox(self) -> Tuple[float, float, float, float]:
        """Get bounding box as (x1, y1, x2, y2)."""
        half_w = self.width / 2
        half_h = self.height / 2
        return (
            self.x - half_w,
            self.y - half_h,
            self.x + half_w,
            self.y + half_h
        )

    def get_speed(self) -> float:
        """Get current speed magnitude."""
        return np.sqrt(self.vx**2 + self.vy**2)

    def get_heading(self) -> float:
        """Get movement direction in radians."""
        return np.arctan2(self.vy, self.vx)

    def update_visual_features(self, visual_features: Optional[np.ndarray] = None,
                                color_features: Optional[np.ndarray] = None):
        """
        Update visual features from VisionEncoder.

        Args:
            visual_features: [roi_dim] CNN-extracted features
            color_features: [color_dim] Color statistics
        """
        if visual_features is not None:
            self.visual_features = visual_features.copy()
            self.visual_confidence = 1.0

        if color_features is not None:
            self.color_features = color_features.copy()

            # Detect glow (bright pixels ratio > threshold)
            if len(color_features) >= 20:
                self.is_glowing = color_features[19] > 0.3  # Glow index

            # Determine dominant color from color ratios
            if len(color_features) >= 19:
                red_ratio = color_features[16]
                blue_ratio = color_features[17]
                green_ratio = color_features[18]

                max_ratio = max(red_ratio, blue_ratio, green_ratio)
                if max_ratio > 0.4:  # Clear dominance
                    if red_ratio == max_ratio:
                        self.dominant_color = "red"
                    elif blue_ratio == max_ratio:
                        self.dominant_color = "blue"
                    else:
                        self.dominant_color = "green"
                else:
                    self.dominant_color = "neutral"

    def to_feature_vector(self, player_x: float = 0.5, player_y: float = 0.5) -> np.ndarray:
        """
        Convert to feature vector for neural network input.

        Returns OBJECT_FEATURE_DIM vector (default 20):
        [0-3]   Position: x, y, distance_to_player, angle_to_player
        [4-7]   Velocity: vx, vy, speed, heading
        [8-11]  Bbox: width, height, confidence, age_normalized
        [12-15] Tracking: hits, time_since_update, is_tracked, is_lost
        [16-19] Class: one-hot or embedding (simplified to 4 values)

        To add more features:
        1. Add them to the array below
        2. Update OBJECT_FEATURE_DIM in config.py to match
        """
        # Distance and angle to player
        dx = self.x - player_x
        dy = self.y - player_y
        distance = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx) / np.pi  # Normalize to [-1, 1]

        # Class encoding (using centralized constants)
        is_enemy = 1.0 if self.class_name in ENEMY_CLASSES else 0.0
        is_loot = 1.0 if self.class_name in LOOT_CLASSES else 0.0
        is_player = 1.0 if self.class_name in PLAYER_CLASSES else 0.0
        is_other = 1.0 - is_enemy - is_loot - is_player

        features = np.array([
            # Position (4)
            self.x, self.y, distance, angle,
            # Velocity (4)
            self.vx * VELOCITY_SCALE, self.vy * VELOCITY_SCALE,
            self.get_speed() * VELOCITY_SCALE, self.get_heading() / np.pi,
            # Bbox (4)
            self.width, self.height, self.confidence,
            min(self.age / 100, 1.0),  # Normalize age
            # Tracking (4)
            min(self.hits / 50, 1.0),  # Normalize hits
            min(self.time_since_update / 10, 1.0),  # Normalize
            1.0 if self.state == "tracked" else 0.0,
            1.0 if self.state == "lost" else 0.0,
            # Class (4)
            is_enemy, is_loot, is_player, is_other
        ], dtype=np.float32)

        # Validate dimension matches config (catches mismatch early)
        assert len(features) == OBJECT_FEATURE_DIM, \
            f"Feature vector has {len(features)} dims but OBJECT_FEATURE_DIM={OBJECT_FEATURE_DIM}. Update config.py!"

        return features

    def to_feature_vector_with_visual(self, player_x: float = 0.5, player_y: float = 0.5,
                                       visual_dim: int = 128) -> np.ndarray:
        """
        Convert to feature vector including visual features for Tactician.

        Returns (20 + visual_dim)-dim vector:
        [0-19]  Base features from to_feature_vector()
        [20+]   Visual features (padded if not available)
        """
        base_features = self.to_feature_vector(player_x, player_y)

        if self.visual_features is not None and len(self.visual_features) == visual_dim:
            visual = self.visual_features
        else:
            # Pad with zeros if visual features not available
            visual = np.zeros(visual_dim, dtype=np.float32)

        return np.concatenate([base_features, visual])


def iou(box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union between two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


class ObjectTracker:
    """
    ByteTrack-style multi-object tracker.

    Assigns persistent IDs to detected objects across frames.
    """

    def __init__(self,
                 high_thresh: float = 0.6,
                 low_thresh: float = 0.1,
                 match_thresh: float = 0.8,
                 track_buffer: int = 30):
        """
        Args:
            high_thresh: Confidence threshold for primary matching
            low_thresh: Confidence threshold for secondary matching
            match_thresh: IoU threshold for matching
            track_buffer: Frames to keep lost tracks
        """
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer

        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id = 1
        self.frame_count = 0

    def update(self, detections: List) -> List[TrackedObject]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections, each with:
                - x_center, y_center (normalized 0-1)
                - width, height (normalized)
                - confidence
                - class_name

        Returns:
            List of TrackedObject with persistent IDs
        """
        self.frame_count += 1

        # Convert detections to standard format
        det_boxes = []
        det_confs = []
        det_classes = []

        for d in detections:
            # Handle both object and dict formats
            if hasattr(d, 'x_center'):
                x, y = d.x_center, d.y_center
                w = getattr(d, 'width', 0.05)
                h = getattr(d, 'height', 0.05)
                conf = getattr(d, 'confidence', 0.5)
                cls = d.class_name
            else:
                x = d.get('x_center', 0.5)
                y = d.get('y_center', 0.5)
                w = d.get('width', 0.05)
                h = d.get('height', 0.05)
                conf = d.get('confidence', 0.5)
                cls = d.get('class_name', 'unknown')

            det_boxes.append((x, y, w, h))
            det_confs.append(conf)
            det_classes.append(cls)

        # Split detections by confidence
        high_det_idx = [i for i, c in enumerate(det_confs) if c >= self.high_thresh]
        low_det_idx = [i for i, c in enumerate(det_confs)
                       if self.low_thresh <= c < self.high_thresh]

        # Get current tracks
        track_ids = list(self.tracks.keys())
        tracked_ids = [tid for tid in track_ids if self.tracks[tid].state == "tracked"]
        lost_ids = [tid for tid in track_ids if self.tracks[tid].state == "lost"]

        # === FIRST ASSOCIATION: High-confidence detections with tracked objects ===
        matched_track_ids = set()
        matched_det_idx = set()

        if tracked_ids and high_det_idx:
            # Build cost matrix (IoU-based)
            cost_matrix = np.zeros((len(tracked_ids), len(high_det_idx)))

            for i, tid in enumerate(tracked_ids):
                track = self.tracks[tid]
                track_box = track.get_bbox()

                for j, det_i in enumerate(high_det_idx):
                    x, y, w, h = det_boxes[det_i]
                    det_box = (x - w/2, y - h/2, x + w/2, y + h/2)
                    cost_matrix[i, j] = 1 - iou(track_box, det_box)

            # Simple greedy matching (could use Hungarian for optimal)
            for _ in range(min(len(tracked_ids), len(high_det_idx))):
                if cost_matrix.size == 0:
                    break

                # Find minimum cost
                min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                min_cost = cost_matrix[min_idx]

                if min_cost > (1 - self.match_thresh):
                    break

                track_i, det_j = min_idx
                tid = tracked_ids[track_i]
                det_i = high_det_idx[det_j]

                # Update track
                x, y, w, h = det_boxes[det_i]
                self.tracks[tid].update(x, y, w, h, det_confs[det_i], det_classes[det_i])

                matched_track_ids.add(tid)
                matched_det_idx.add(det_i)

                # Remove matched row/col from cost matrix
                cost_matrix[track_i, :] = np.inf
                cost_matrix[:, det_j] = np.inf

        # === SECOND ASSOCIATION: Low-confidence detections with remaining tracks ===
        unmatched_tracks = [tid for tid in tracked_ids if tid not in matched_track_ids]
        unmatched_high_det = [i for i in high_det_idx if i not in matched_det_idx]

        if unmatched_tracks and low_det_idx:
            cost_matrix = np.zeros((len(unmatched_tracks), len(low_det_idx)))

            for i, tid in enumerate(unmatched_tracks):
                track = self.tracks[tid]
                track_box = track.get_bbox()

                for j, det_i in enumerate(low_det_idx):
                    x, y, w, h = det_boxes[det_i]
                    det_box = (x - w/2, y - h/2, x + w/2, y + h/2)
                    cost_matrix[i, j] = 1 - iou(track_box, det_box)

            # Greedy matching
            for _ in range(min(len(unmatched_tracks), len(low_det_idx))):
                if cost_matrix.size == 0:
                    break

                min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                min_cost = cost_matrix[min_idx]

                if min_cost > (1 - self.match_thresh * 0.7):  # Lower threshold for low-conf
                    break

                track_i, det_j = min_idx
                tid = unmatched_tracks[track_i]
                det_i = low_det_idx[det_j]

                x, y, w, h = det_boxes[det_i]
                self.tracks[tid].update(x, y, w, h, det_confs[det_i], det_classes[det_i])

                matched_track_ids.add(tid)
                matched_det_idx.add(det_i)

                cost_matrix[track_i, :] = np.inf
                cost_matrix[:, det_j] = np.inf

        # === THIRD ASSOCIATION: Lost tracks with remaining high-conf detections ===
        unmatched_high_det = [i for i in high_det_idx if i not in matched_det_idx]

        if lost_ids and unmatched_high_det:
            cost_matrix = np.zeros((len(lost_ids), len(unmatched_high_det)))

            for i, tid in enumerate(lost_ids):
                track = self.tracks[tid]
                pred_x, pred_y = track.predict()
                pred_box = (pred_x - track.width/2, pred_y - track.height/2,
                           pred_x + track.width/2, pred_y + track.height/2)

                for j, det_i in enumerate(unmatched_high_det):
                    x, y, w, h = det_boxes[det_i]
                    det_box = (x - w/2, y - h/2, x + w/2, y + h/2)
                    cost_matrix[i, j] = 1 - iou(pred_box, det_box)

            # Greedy matching
            for _ in range(min(len(lost_ids), len(unmatched_high_det))):
                if cost_matrix.size == 0:
                    break

                min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                min_cost = cost_matrix[min_idx]

                if min_cost > (1 - self.match_thresh * 0.6):
                    break

                track_i, det_j = min_idx
                tid = lost_ids[track_i]
                det_i = unmatched_high_det[det_j]

                x, y, w, h = det_boxes[det_i]
                self.tracks[tid].update(x, y, w, h, det_confs[det_i], det_classes[det_i])

                matched_track_ids.add(tid)
                matched_det_idx.add(det_i)

                cost_matrix[track_i, :] = np.inf
                cost_matrix[:, det_j] = np.inf

        # === Mark unmatched tracks as missed ===
        for tid in track_ids:
            if tid not in matched_track_ids:
                self.tracks[tid].mark_missed()

        # === Create new tracks for unmatched high-confidence detections ===
        final_unmatched = [i for i in high_det_idx if i not in matched_det_idx]

        for det_i in final_unmatched:
            x, y, w, h = det_boxes[det_i]
            new_track = TrackedObject(
                track_id=self.next_id,
                class_name=det_classes[det_i],
                x=x, y=y,
                width=w, height=h,
                confidence=det_confs[det_i]
            )
            self.tracks[self.next_id] = new_track
            self.next_id += 1

        # === Remove stale tracks ===
        to_remove = []
        for tid, track in self.tracks.items():
            if track.time_since_update > self.track_buffer:
                track.state = "removed"
                to_remove.append(tid)

        for tid in to_remove:
            del self.tracks[tid]

        # === Return active tracks ===
        return [t for t in self.tracks.values() if t.state != "removed"]

    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        """Get a specific track by ID."""
        return self.tracks.get(track_id)

    def get_enemies(self) -> List[TrackedObject]:
        """Get all tracked enemies."""
        return [t for t in self.tracks.values()
                if t.state != "removed" and t.class_name in ENEMY_CLASSES]

    def get_loot(self) -> List[TrackedObject]:
        """Get all tracked loot boxes."""
        return [t for t in self.tracks.values()
                if t.state != "removed" and t.class_name in LOOT_CLASSES]

    def get_nearest_enemy(self, x: float, y: float) -> Optional[TrackedObject]:
        """Get nearest enemy to position."""
        enemies = self.get_enemies()
        if not enemies:
            return None

        return min(enemies, key=lambda e: (e.x - x)**2 + (e.y - y)**2)

    def get_nearest_loot(self, x: float, y: float) -> Optional[TrackedObject]:
        """Get nearest loot to position."""
        loot = self.get_loot()
        if not loot:
            return None

        return min(loot, key=lambda l: (l.x - x)**2 + (l.y - y)**2)

    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0

    def get_stats(self) -> Dict:
        """Get tracking statistics."""
        tracked = len([t for t in self.tracks.values() if t.state == "tracked"])
        lost = len([t for t in self.tracks.values() if t.state == "lost"])

        return {
            'total_tracks': len(self.tracks),
            'tracked': tracked,
            'lost': lost,
            'frame_count': self.frame_count,
            'next_id': self.next_id
        }
