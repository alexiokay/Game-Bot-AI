"""
DarkOrbit Bot - State Builder

Converts raw YOLO detections into fixed-size state vectors
that the Bi-LSTM can process.

The state vector contains:
- Object information (position, class, confidence)
- Player state estimates (health, ammo, velocity)
- Context flags (mode, combat status)
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PlayerState:
    """Estimated player state"""
    health: float = 1.0  # 0.0 - 1.0
    mouse_x: float = 0.5  # Normalized screen position
    mouse_y: float = 0.5
    velocity_x: float = 0.0  # Mouse velocity
    velocity_y: float = 0.0
    mode: str = "PASSIVE"  # Current context


class MovementPatternAnalyzer:
    """
    Analyzes movement patterns to detect tactics like orbiting, kiting, etc.

    This helps V1 understand WHAT kind of movement pattern is happening,
    not just WHERE the mouse is moving.
    """

    def __init__(self, history_size: int = 30):
        self.history_size = history_size
        self.position_history = []  # List of (x, y, time) tuples
        self.target_history = []    # List of (x, y) - enemy positions

    def add_position(self, x: float, y: float, target_x: float = None, target_y: float = None):
        """Add current position and optionally the target (enemy) position"""
        import time
        self.position_history.append((x, y, time.time()))
        if target_x is not None and target_y is not None:
            self.target_history.append((target_x, target_y))

        # Keep only recent history
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)
        if len(self.target_history) > self.history_size:
            self.target_history.pop(0)

    def get_pattern_features(self, target_x: float = None, target_y: float = None) -> np.ndarray:
        """
        Compute movement pattern features relative to a target.

        Returns array of [
            angular_velocity,    # Positive = orbiting clockwise around target
            orbit_intensity,     # How circular the movement is (0-1)
            distance_trend,      # Negative = approaching, positive = retreating
            distance_variance,   # Low = stable orbit, high = erratic
            movement_speed,      # Current movement speed (normalized)
            direction_changes,   # How often direction changes (jitter detection)
        ]
        """
        if len(self.position_history) < 5:
            return np.zeros(6, dtype=np.float32)

        positions = np.array([(p[0], p[1]) for p in self.position_history])

        # Use provided target or average of target history or center
        if target_x is not None and target_y is not None:
            target = np.array([target_x, target_y])
        elif self.target_history:
            target = np.mean(self.target_history, axis=0)
        else:
            target = np.array([0.5, 0.5])  # Default to center

        # Calculate angles from target to each position
        deltas = positions - target
        angles = np.arctan2(deltas[:, 1], deltas[:, 0])

        # Angular velocity (change in angle over time)
        angle_diffs = np.diff(angles)
        # Handle wraparound (-pi to pi)
        angle_diffs = np.where(angle_diffs > np.pi, angle_diffs - 2*np.pi, angle_diffs)
        angle_diffs = np.where(angle_diffs < -np.pi, angle_diffs + 2*np.pi, angle_diffs)
        angular_velocity = np.mean(angle_diffs) if len(angle_diffs) > 0 else 0

        # Orbit intensity: consistent angular change = high, random = low
        orbit_intensity = 1.0 - min(1.0, np.std(angle_diffs) * 5) if len(angle_diffs) > 0 else 0

        # Distance trend (approaching or retreating)
        distances = np.linalg.norm(deltas, axis=1)
        distance_diffs = np.diff(distances)
        distance_trend = np.mean(distance_diffs) if len(distance_diffs) > 0 else 0

        # Distance variance (stable orbit vs erratic movement)
        distance_variance = np.std(distances) if len(distances) > 0 else 0

        # Movement speed
        position_diffs = np.diff(positions, axis=0)
        speeds = np.linalg.norm(position_diffs, axis=1)
        movement_speed = np.mean(speeds) if len(speeds) > 0 else 0

        # Direction changes (count significant angle changes in movement direction)
        if len(position_diffs) > 1:
            movement_angles = np.arctan2(position_diffs[:, 1], position_diffs[:, 0])
            movement_angle_diffs = np.abs(np.diff(movement_angles))
            direction_changes = np.sum(movement_angle_diffs > 0.5) / len(movement_angle_diffs)
        else:
            direction_changes = 0

        return np.array([
            angular_velocity * 10,  # Scale for neural network
            orbit_intensity,
            distance_trend * 10,    # Scale
            min(1.0, distance_variance * 5),
            min(1.0, movement_speed * 10),
            direction_changes
        ], dtype=np.float32)

    def classify_tactic(self) -> str:
        """Classify current movement as a known tactic"""
        features = self.get_pattern_features()
        angular_vel = features[0]
        orbit_intensity = features[1]
        distance_trend = features[2]
        direction_changes = features[5]

        # Orbiting: consistent angular movement, stable distance
        if abs(angular_vel) > 0.1 and orbit_intensity > 0.5:
            return "orbiting"

        # Kiting: moving away while fighting
        if distance_trend > 0.05:
            return "kiting"

        # Face-tanking: moving toward or stationary
        if distance_trend < -0.05:
            return "approaching"

        # Dodging: lots of direction changes
        if direction_changes > 0.5:
            return "dodging"

        return "none"

    def clear(self):
        """Clear history"""
        self.position_history.clear()
        self.target_history.clear()


class StateBuilder:
    """
    Builds fixed-size state vectors for the Bi-LSTM from detections.
    
    Output format:
    - Object slots: max_objects * features_per_object
    - Player state: player_features
    - Total: fixed size regardless of actual detections
    """
    
    # Class name to ID mapping (from your YOLO model)
    CLASS_MAP = {
        'BonusBox': 0,
        'Devo': 1,
        'Lordakia': 2,
        'Mordon': 3,
        'Player': 4,
        'Saimon': 5,
        'Sibelon': 6,
        'Struener': 7
    }
    
    # Which classes are enemies vs collectibles
    ENEMY_IDS = [1, 2, 3, 5, 6, 7]  # Devo, Lordakia, Mordon, Saimon, Sibelon, Struener
    COLLECTIBLE_IDS = [0]  # BonusBox
    
    def __init__(self,
                 max_objects: int = 20,
                 features_per_object: int = 6,
                 player_features: int = None,  # Auto-set based on include_movement_patterns
                 include_movement_patterns: bool = True):
        """
        Args:
            max_objects: Maximum objects to track (padded with zeros if fewer)
            features_per_object: Features per detection [x, y, w, h, class_id, conf]
            player_features: Player state features (auto: 14 with patterns, 8 without)
            include_movement_patterns: Whether to include orbiting/kiting pattern features
        """
        self.max_objects = max_objects
        self.features_per_object = features_per_object
        self.include_movement_patterns = include_movement_patterns

        # Auto-set player_features based on include_movement_patterns
        if player_features is None:
            self.player_features = 14 if include_movement_patterns else 8
        else:
            self.player_features = player_features

        self.state_size = max_objects * features_per_object + self.player_features

        # Movement pattern analyzer for detecting orbiting, kiting, etc.
        self.pattern_analyzer = MovementPatternAnalyzer() if include_movement_patterns else None
        
    def build(self, 
             detections: List[Any], 
             player_state: PlayerState,
             screen_width: int = 1920,
             screen_height: int = 1080) -> np.ndarray:
        """
        Build state vector from detections.
        
        Args:
            detections: List of Detection objects from YOLO
            player_state: Current player state estimate
            screen_width, screen_height: For normalization
            
        Returns:
            np.ndarray of shape (state_size,) - fixed size vector
        """
        # Initialize object slots with zeros
        object_features = np.zeros((self.max_objects, self.features_per_object), dtype=np.float32)
        
        # Sort detections by priority (enemies first, then by confidence)
        sorted_detections = self._prioritize_detections(detections)
        
        # Fill in object slots
        for i, det in enumerate(sorted_detections[:self.max_objects]):
            object_features[i] = self._detection_to_features(det, screen_width, screen_height)
            
        # Build player state vector
        player_vector = self._player_to_features(player_state, detections)
        
        # Concatenate and flatten
        state = np.concatenate([
            object_features.flatten(),
            player_vector
        ])
        
        return state
    
    def _detection_to_features(self, det, screen_width: int, screen_height: int) -> np.ndarray:
        """Convert a single detection to features"""
        # Get class ID
        if hasattr(det, 'class_name'):
            class_id = self.CLASS_MAP.get(det.class_name, -1)
        else:
            class_id = getattr(det, 'class_id', -1)
            
        # Normalize coordinates to 0-1
        x_center = getattr(det, 'x_center', 0.5)
        y_center = getattr(det, 'y_center', 0.5)
        width = getattr(det, 'width', 0.0)
        height = getattr(det, 'height', 0.0)
        confidence = getattr(det, 'confidence', 0.0)
        
        # One-hot encode whether it's an enemy or collectible
        is_enemy = 1.0 if class_id in self.ENEMY_IDS else 0.0
        
        return np.array([
            x_center,
            y_center,
            width,
            height,
            class_id / len(self.CLASS_MAP),  # Normalize class ID
            confidence
        ], dtype=np.float32)
    
    def _player_to_features(self, player_state: PlayerState, detections: List) -> np.ndarray:
        """Build player state features including movement pattern analysis"""
        # Calculate additional metrics from detections
        enemies = [d for d in detections
                  if hasattr(d, 'class_name') and self.CLASS_MAP.get(d.class_name, -1) in self.ENEMY_IDS]
        collectibles = [d for d in detections
                       if hasattr(d, 'class_name') and self.CLASS_MAP.get(d.class_name, -1) in self.COLLECTIBLE_IDS]

        # Find nearest enemy for pattern analysis
        nearest_enemy_dist = 1.0
        nearest_enemy_x, nearest_enemy_y = None, None
        if enemies:
            distances = []
            for d in enemies:
                dist = np.sqrt((d.x_center - player_state.mouse_x)**2 +
                              (d.y_center - player_state.mouse_y)**2)
                distances.append((dist, d.x_center, d.y_center))
            if distances:
                nearest = min(distances, key=lambda x: x[0])
                nearest_enemy_dist = nearest[0]
                nearest_enemy_x, nearest_enemy_y = nearest[1], nearest[2]

        # Distance to nearest collectible
        nearest_box_dist = 1.0
        if collectibles:
            distances = [np.sqrt((d.x_center - player_state.mouse_x)**2 +
                                (d.y_center - player_state.mouse_y)**2) for d in collectibles]
            nearest_box_dist = min(distances) if distances else 1.0

        # Mode encoding
        mode_aggressive = 1.0 if player_state.mode == "AGGRESSIVE" else 0.0

        # Base features (8)
        base_features = np.array([
            player_state.health,
            player_state.mouse_x,
            player_state.mouse_y,
            player_state.velocity_x / 1000.0,  # Normalize velocity
            player_state.velocity_y / 1000.0,
            nearest_enemy_dist,
            nearest_box_dist,
            mode_aggressive
        ], dtype=np.float32)

        # Movement pattern features (6) - helps V1 learn orbiting, kiting, etc.
        # Only include if include_movement_patterns=True (for new 134-feature models)
        if self.include_movement_patterns and self.pattern_analyzer is not None:
            # Update pattern analyzer with current position and nearest enemy
            self.pattern_analyzer.add_position(
                player_state.mouse_x,
                player_state.mouse_y,
                nearest_enemy_x,
                nearest_enemy_y
            )

            # Get pattern features relative to nearest enemy
            pattern_features = self.pattern_analyzer.get_pattern_features(
                nearest_enemy_x, nearest_enemy_y
            )
            return np.concatenate([base_features, pattern_features])
        else:
            # Old 128-feature models: just return base features (8)
            return base_features
    
    def _prioritize_detections(self, detections: List) -> List:
        """
        Sort detections by importance:
        1. Enemies (by proximity and threat)
        2. Collectibles
        3. Other
        """
        def priority_key(det):
            class_name = getattr(det, 'class_name', '')
            class_id = self.CLASS_MAP.get(class_name, -1)
            confidence = getattr(det, 'confidence', 0.0)
            
            # Enemies get highest priority
            if class_id in self.ENEMY_IDS:
                return (0, -confidence)  # 0 = highest priority, sort by confidence
            elif class_id in self.COLLECTIBLE_IDS:
                return (1, -confidence)
            else:
                return (2, -confidence)
                
        return sorted(detections, key=priority_key)
    
    def get_state_size(self) -> int:
        """Return the total state vector size"""
        return self.state_size
    
    def describe_state(self, state: np.ndarray) -> Dict:
        """
        Convert state vector back to human-readable dict (for debugging).
        """
        object_end = self.max_objects * self.features_per_object
        objects_flat = state[:object_end]
        player = state[object_end:]

        objects = objects_flat.reshape(self.max_objects, self.features_per_object)
        active_objects = [o for o in objects if o[5] > 0]  # confidence > 0

        result = {
            'num_objects': len(active_objects),
            'player_health': player[0],
            'player_pos': (player[1], player[2]),
            'player_velocity': (player[3], player[4]),
            'nearest_enemy': player[5],
            'nearest_box': player[6],
            'mode_aggressive': player[7] > 0.5
        }

        # Add movement pattern features if available (indices 8-13)
        if len(player) >= 14:
            result['movement_pattern'] = {
                'angular_velocity': player[8],   # Orbiting indicator
                'orbit_intensity': player[9],    # How circular
                'distance_trend': player[10],    # Approaching/retreating
                'distance_variance': player[11], # Stable vs erratic
                'movement_speed': player[12],    # Current speed
                'direction_changes': player[13], # Jitter/dodging
            }

            # Classify tactic from features
            if player[9] > 0.5 and abs(player[8]) > 0.1:
                result['detected_tactic'] = 'orbiting'
            elif player[10] > 0.05:
                result['detected_tactic'] = 'kiting'
            elif player[10] < -0.05:
                result['detected_tactic'] = 'approaching'
            elif player[13] > 0.5:
                result['detected_tactic'] = 'dodging'
            else:
                result['detected_tactic'] = 'none'

        return result


class StateSequenceBuilder:
    """
    Builds sequences of states for the Bi-LSTM.
    Maintains a sliding window of recent states.
    """
    
    def __init__(self, 
                 sequence_length: int = 50,
                 state_builder: StateBuilder = None):
        """
        Args:
            sequence_length: Number of frames in each sequence (20-50 recommended)
            state_builder: StateBuilder instance (created if None)
        """
        self.sequence_length = sequence_length
        self.state_builder = state_builder or StateBuilder()
        self.state_history: List[np.ndarray] = []
        
    def add_frame(self, 
                 detections: List, 
                 player_state: PlayerState) -> Optional[np.ndarray]:
        """
        Add a frame and return sequence if we have enough history.
        
        Returns:
            Sequence array of shape (sequence_length, state_size) or None if not enough history
        """
        state = self.state_builder.build(detections, player_state)
        self.state_history.append(state)
        
        # Keep only what we need
        if len(self.state_history) > self.sequence_length:
            self.state_history.pop(0)
            
        # Return sequence if we have enough
        if len(self.state_history) >= self.sequence_length:
            return np.array(self.state_history[-self.sequence_length:])
        return None
    
    def get_current_sequence(self) -> Optional[np.ndarray]:
        """Get current sequence without adding new frame"""
        if len(self.state_history) >= self.sequence_length:
            return np.array(self.state_history[-self.sequence_length:])
        return None
    
    def clear(self):
        """Clear history (new game/respawn)"""
        self.state_history.clear()
