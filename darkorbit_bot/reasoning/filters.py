"""
DarkOrbit Bot - Data Filters for Filtered Behavioral Cloning (FBC)

These filters ensure the bot only learns from "good" gameplay moments:
- KillFilter: Saves data before kills, deletes data before deaths
- AdvantageCritic: Scores actions based on health/progress trends
- GaussianSmoother: Removes shaky micro-movements from mouse paths
- RollingBuffer: Keeps last N seconds of data for filtering
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time


@dataclass
class BufferFrame:
    """Single frame of recorded data"""
    timestamp: float
    frame: np.ndarray  # Screenshot
    detections: List[Any]  # Detection objects
    mouse_pos: tuple
    mouse_velocity: float
    keys_pressed: List[str]
    clicks: List[Dict]
    mode: str  # "PASSIVE" or "AGGRESSIVE"
    health_estimate: float  # 0.0 - 1.0
    

class RollingBuffer:
    """
    Keeps last N seconds of gameplay data.
    Allows saving "good" segments and discarding "bad" ones.
    """
    
    def __init__(self, max_seconds: float = 30.0, fps: float = 30.0):
        self.max_seconds = max_seconds
        self.max_frames = int(max_seconds * fps)
        self.buffer: deque = deque(maxlen=self.max_frames)
        self.start_time = time.time()
        
    def add(self, frame_data: BufferFrame):
        """Add a frame to the buffer"""
        self.buffer.append(frame_data)
        
    def get_last_seconds(self, seconds: float) -> List[BufferFrame]:
        """Get frames from the last N seconds"""
        if not self.buffer:
            return []
            
        cutoff_time = time.time() - seconds
        return [f for f in self.buffer if f.timestamp >= cutoff_time]
    
    def clear(self):
        """Clear the buffer (called on death)"""
        self.buffer.clear()
        
    def get_all(self) -> List[BufferFrame]:
        """Get all frames in buffer"""
        return list(self.buffer)
    
    def __len__(self):
        return len(self.buffer)


class KillFilter:
    """
    Detects kills and deaths to filter training data.
    
    - On KILL: Save last 10 seconds (successful combat)
    - On DEATH: Delete last 30 seconds (failed play)
    
    Uses stable counting to avoid false triggers from YOLO flickering.
    """
    
    # Enemy classes from the YOLO model
    ENEMY_CLASSES = ['Devo', 'Lordakia', 'Mordon', 'Saimon', 'Sibelon', 'Struener']
    
    def __init__(self, save_before_kill: float = 10.0, delete_before_death: float = 30.0):
        self.save_before_kill = save_before_kill
        self.delete_before_death = delete_before_death
        
        # Stable counting - track over multiple frames
        self.enemy_history: deque = deque(maxlen=15)  # Last 15 frames (~0.5s at 30fps)
        self.stable_enemy_count = 0
        
        # Cooldown to prevent spam
        self.last_kill_time = 0.0
        self.kill_cooldown = 3.0  # Minimum seconds between kills
        
        # Track enemy positions for movement prediction
        self.enemy_positions: Dict[int, deque] = {}  # enemy_id -> position history
        
    def update(self, detections: List[Any]) -> Dict[str, bool]:
        """
        Process new detections and return events.
        
        Returns:
            {"kill": True/False, "death": False, "enemy_velocities": {...}}
        """
        current_enemies = []
        for det in detections:
            if hasattr(det, 'class_name') and det.class_name in self.ENEMY_CLASSES:
                current_enemies.append(det)
        
        curr_count = len(current_enemies)
        self.enemy_history.append(curr_count)
        
        # Track enemy positions for velocity estimation
        enemy_velocities = self._update_enemy_positions(current_enemies)
        
        # Calculate stable count (median of recent counts to filter flicker)
        if len(self.enemy_history) >= 10:
            sorted_counts = sorted(self.enemy_history)
            new_stable = sorted_counts[len(sorted_counts) // 2]  # Median
            
            # Kill detected only if:
            # 1. Stable count dropped significantly (not just flicker)
            # 2. Previous stable was > 0
            # 3. Cooldown passed
            now = time.time()
            kill_detected = (
                new_stable < self.stable_enemy_count - 0  # At least 1 enemy gone
                and self.stable_enemy_count >= 2  # Had at least 2 enemies
                and new_stable <= self.stable_enemy_count - 1  # Lost at least 1
                and (now - self.last_kill_time) >= self.kill_cooldown
            )
            
            if kill_detected:
                self.last_kill_time = now
                
            self.stable_enemy_count = new_stable
        else:
            kill_detected = False
            
        return {
            "kill": kill_detected,
            "death": False,  # Will be detected via health
            "enemy_velocities": enemy_velocities
        }
    
    def _update_enemy_positions(self, enemies: List[Any]) -> Dict[str, tuple]:
        """
        Track enemy positions and estimate velocities.
        
        Returns:
            {class_name: (vx, vy)} - velocity per enemy type
        """
        velocities = {}
        
        for i, enemy in enumerate(enemies):
            # Create ID from class + approximate position
            enemy_id = hash((enemy.class_name, int(enemy.x_center * 10), int(enemy.y_center * 10)))
            
            pos = (enemy.x_center, enemy.y_center)
            
            if enemy_id not in self.enemy_positions:
                self.enemy_positions[enemy_id] = deque(maxlen=10)
            
            self.enemy_positions[enemy_id].append((time.time(), pos))
            
            # Calculate velocity if we have history
            history = self.enemy_positions[enemy_id]
            if len(history) >= 2:
                dt = history[-1][0] - history[0][0]
                if dt > 0:
                    dx = history[-1][1][0] - history[0][1][0]
                    dy = history[-1][1][1] - history[0][1][1]
                    vx = dx / dt
                    vy = dy / dt
                    velocities[enemy.class_name] = (vx, vy)
        
        # Clean old entries
        now = time.time()
        self.enemy_positions = {
            eid: hist for eid, hist in self.enemy_positions.items()
            if hist and (now - hist[-1][0]) < 1.0  # Keep only recent
        }
        
        return velocities
    
    def detect_death(self, health_ratio: float, prev_health: float) -> bool:
        """
        Detect player death.
        
        Args:
            health_ratio: Current health (0.0 - 1.0)
            prev_health: Previous health value
            
        Returns:
            True if player died
        """
        # Death = health dropped to 0 (or very low)
        return health_ratio <= 0.05 and prev_health > 0.05


class AdvantageCritic:
    """
    Scores gameplay actions to determine quality.
    
    High score = keep the data
    Low score = discard the data
    
    Scoring factors:
    - Health trend (stable/increasing = good)
    - Enemy damage (dealing damage = good)
    - Distance to objectives (getting closer = good)
    - Survival time (staying alive = good)
    """
    
    def __init__(self, 
                 health_weight: float = 10.0,
                 damage_weight: float = 5.0,
                 distance_weight: float = -0.1,
                 threshold: float = 0.0):
        self.health_weight = health_weight
        self.damage_weight = damage_weight
        self.distance_weight = distance_weight
        self.threshold = threshold  # Scores above this are "good"
        
    def score_frame(self, 
                   prev_state: Dict, 
                   curr_state: Dict,
                   action: Dict) -> float:
        """
        Score a single frame transition.
        
        Args:
            prev_state: Previous game state
            curr_state: Current game state
            action: Action taken
            
        Returns:
            Score (positive = good, negative = bad)
        """
        score = 0.0
        
        # Health factor
        health_delta = curr_state.get('health', 1.0) - prev_state.get('health', 1.0)
        score += health_delta * self.health_weight
        
        # Enemy count factor (fewer enemies = we killed some = good)
        enemy_delta = prev_state.get('enemy_count', 0) - curr_state.get('enemy_count', 0)
        score += enemy_delta * self.damage_weight
        
        # Distance to nearest enemy (aggressive mode) or box (passive mode)
        if curr_state.get('mode') == 'AGGRESSIVE':
            # Getting closer to enemies is good in combat
            dist_delta = prev_state.get('enemy_distance', 0) - curr_state.get('enemy_distance', 0)
            score += dist_delta * abs(self.distance_weight)
        else:
            # Getting closer to loot boxes is good in passive
            dist_delta = prev_state.get('box_distance', 0) - curr_state.get('box_distance', 0)
            score += dist_delta * abs(self.distance_weight)
            
        return score
    
    def is_good_sequence(self, frames: List[BufferFrame]) -> bool:
        """
        Determine if a sequence of frames is worth training on.
        """
        if len(frames) < 2:
            return False
            
        total_score = 0.0
        
        for i in range(1, len(frames)):
            prev = self._frame_to_state(frames[i-1])
            curr = self._frame_to_state(frames[i])
            action = self._frame_to_action(frames[i])
            total_score += self.score_frame(prev, curr, action)
            
        avg_score = total_score / len(frames)
        return avg_score >= self.threshold
    
    def _frame_to_state(self, frame: BufferFrame) -> Dict:
        """Convert BufferFrame to state dict"""
        enemy_count = sum(1 for d in frame.detections 
                        if hasattr(d, 'class_name') and d.class_name in KillFilter.ENEMY_CLASSES)
        return {
            'health': frame.health_estimate,
            'enemy_count': enemy_count,
            'mode': frame.mode
        }
    
    def _frame_to_action(self, frame: BufferFrame) -> Dict:
        """Convert BufferFrame to action dict"""
        return {
            'mouse_pos': frame.mouse_pos,
            'keys': frame.keys_pressed,
            'clicks': frame.clicks
        }


class GaussianSmoother:
    """
    Smooths mouse movement paths to remove human "jitter".
    
    This makes the bot's movements look like your movements
    on a "very good day" - same path, but butter smooth.
    """
    
    def __init__(self, sigma: float = 3.0):
        """
        Args:
            sigma: Smoothing strength. Higher = smoother but less precise.
                   Recommended: 2-5 for natural feel.
        """
        self.sigma = sigma
        
    def smooth_path(self, mouse_positions: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian smoothing to a mouse path.
        
        Args:
            mouse_positions: Array of (x, y) coordinates, shape (N, 2)
            
        Returns:
            Smoothed positions, same shape
        """
        try:
            from scipy.ndimage import gaussian_filter1d
        except ImportError:
            print("Warning: scipy not available, skipping smoothing")
            return mouse_positions
            
        if len(mouse_positions) < 3:
            return mouse_positions
            
        smoothed = np.zeros_like(mouse_positions, dtype=float)
        smoothed[:, 0] = gaussian_filter1d(mouse_positions[:, 0].astype(float), sigma=self.sigma)
        smoothed[:, 1] = gaussian_filter1d(mouse_positions[:, 1].astype(float), sigma=self.sigma)
        
        return smoothed.astype(int)
    
    def smooth_buffer(self, frames: List[BufferFrame]) -> List[BufferFrame]:
        """
        Smooth mouse movements across a sequence of frames.
        """
        if len(frames) < 3:
            return frames
            
        # Extract mouse positions
        positions = np.array([f.mouse_pos for f in frames])
        
        # Smooth
        smoothed = self.smooth_path(positions)
        
        # Update frames (create copies to avoid mutation)
        result = []
        for i, frame in enumerate(frames):
            new_frame = BufferFrame(
                timestamp=frame.timestamp,
                frame=frame.frame,
                detections=frame.detections,
                mouse_pos=tuple(smoothed[i]),
                mouse_velocity=frame.mouse_velocity,
                keys_pressed=frame.keys_pressed,
                clicks=frame.clicks,
                mode=frame.mode,
                health_estimate=frame.health_estimate
            )
            result.append(new_frame)
            
        return result


class HealthEstimator:
    """
    Estimates player health from game state.
    
    Since we don't have direct health bar detection yet,
    we estimate based on:
    - Proximity to enemies over time
    - Combat duration
    - Number of hits taken (inferred from enemy attacks)
    """
    
    def __init__(self, decay_rate: float = 0.01, regen_rate: float = 0.005):
        self.current_health = 1.0
        self.decay_rate = decay_rate  # Health loss per second near enemies
        self.regen_rate = regen_rate  # Health regen per second when safe
        self.last_update = time.time()
        
    def update(self, detections: List[Any], delta_time: float = None) -> float:
        """
        Update health estimate based on current game state.
        
        Args:
            detections: Current detections from YOLO
            delta_time: Time since last update (auto-calculated if None)
            
        Returns:
            Estimated health ratio (0.0 - 1.0)
        """
        now = time.time()
        if delta_time is None:
            delta_time = now - self.last_update
        self.last_update = now
        
        # Count nearby enemies
        enemies_nearby = sum(1 for d in detections 
                           if hasattr(d, 'class_name') and d.class_name in KillFilter.ENEMY_CLASSES)
        
        if enemies_nearby > 0:
            # Lose health based on number of enemies
            damage = self.decay_rate * enemies_nearby * delta_time
            self.current_health = max(0.0, self.current_health - damage)
        else:
            # Regenerate when safe
            self.current_health = min(1.0, self.current_health + self.regen_rate * delta_time)
            
        return self.current_health
    
    def reset(self):
        """Reset health to full (new life/respawn)"""
        self.current_health = 1.0
        self.last_update = time.time()


# Convenience function to create all filters
def create_filters(config: Dict = None) -> Dict:
    """
    Create a complete filter pipeline.
    
    Returns:
        Dict with: buffer, kill_filter, critic, smoother, health
    """
    config = config or {}
    
    return {
        'buffer': RollingBuffer(
            max_seconds=config.get('buffer_seconds', 30.0),
            fps=config.get('fps', 30.0)
        ),
        'kill_filter': KillFilter(
            save_before_kill=config.get('save_before_kill', 10.0),
            delete_before_death=config.get('delete_before_death', 30.0)
        ),
        'critic': AdvantageCritic(
            threshold=config.get('quality_threshold', 0.0)
        ),
        'smoother': GaussianSmoother(
            sigma=config.get('smooth_sigma', 3.0)
        ),
        'health': HealthEstimator()
    }
