"""
DarkOrbit Bot - Context Detector

Automatically detects when you're playing:
- PASSIVE: Looting, navigation, exploring (slow movement, no combat)
- AGGRESSIVE: Combat, targeting, fighting (fast mouse, shooting)

The Bi-LSTM uses different "heads" for each mode.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class ContextState:
    """Current detected context"""
    mode: str  # "PASSIVE" or "AGGRESSIVE"
    confidence: float  # 0.0 - 1.0
    mouse_velocity: float
    enemies_visible: int
    recent_clicks: int
    combat_duration: float  # seconds in current combat


class ContextDetector:
    """
    Detects gameplay context (Passive vs Aggressive) in real-time.
    
    Uses multiple signals:
    - Mouse movement velocity
    - Number of enemies on screen
    - Click frequency
    - Combat engagement duration
    """
    
    # Thresholds for mode detection
    VELOCITY_THRESHOLD = 500  # pixels/second - above = aggressive
    ENEMY_THRESHOLD = 1  # enemies visible to trigger aggressive
    CLICK_RATE_THRESHOLD = 2  # clicks per second

    # Hysteresis to prevent rapid switching
    MODE_SWITCH_DELAY = 1.5  # seconds before switching modes (prevent rapid oscillation)

    # Known enemy classes from the YOLO model (must match bot_controller.py)
    ENEMY_CLASSES = ['Devo', 'Lordakia', 'Mordon', 'Saimon', 'Sibelon', 'Struener',
                     'npc', 'enemy']
    # Known non-enemy classes (won't trigger AGGRESSIVE mode)
    NON_ENEMY_CLASSES = ['BonusBox', 'Player', 'player_ship', 'box', 'bonus_box', 'portal']
    
    def __init__(self):
        self.current_mode = "PASSIVE"
        self.mode_confidence = 1.0
        self.last_mode_switch = time.time()
        
        # Mouse tracking
        self.mouse_positions: deque = deque(maxlen=30)  # Last 30 positions
        self.mouse_times: deque = deque(maxlen=30)
        
        # Click tracking
        self.recent_clicks: deque = deque(maxlen=100)  # Click timestamps
        
        # Combat tracking
        self.combat_start: Optional[float] = None
        self.last_enemy_seen = 0.0
        
    def update_mouse(self, x: int, y: int) -> float:
        """
        Update mouse position and calculate velocity.
        
        Returns:
            Current mouse velocity in pixels/second
        """
        now = time.time()
        self.mouse_positions.append((x, y))
        self.mouse_times.append(now)
        
        if len(self.mouse_positions) < 2:
            return 0.0
            
        # Calculate velocity over recent movement
        positions = list(self.mouse_positions)
        times = list(self.mouse_times)
        
        total_distance = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)
            
        time_span = times[-1] - times[0]
        if time_span > 0:
            return total_distance / time_span
        return 0.0
    
    def record_click(self):
        """Record a mouse click"""
        self.recent_clicks.append(time.time())
        
    def get_click_rate(self) -> float:
        """Get clicks per second over last 3 seconds"""
        now = time.time()
        cutoff = now - 3.0
        recent = [t for t in self.recent_clicks if t >= cutoff]
        return len(recent) / 3.0
    
    def detect(self, 
               mouse_x: int, 
               mouse_y: int, 
               detections: List,
               clicked: bool = False) -> ContextState:
        """
        Detect current gameplay context.
        
        Args:
            mouse_x, mouse_y: Current mouse position
            detections: List of YOLO detections
            clicked: Whether a click just happened
            
        Returns:
            ContextState with mode and metrics
        """
        now = time.time()
        
        # Update tracking
        velocity = self.update_mouse(mouse_x, mouse_y)
        if clicked:
            self.record_click()
        click_rate = self.get_click_rate()
        
        # Count enemies using the same logic as bot_controller
        # Treat anything that's not explicitly non-enemy as potential target
        enemies_visible = 0
        for d in detections:
            class_name = d.class_name if hasattr(d, 'class_name') else d.get('class_name', '')
            if class_name in self.ENEMY_CLASSES or class_name not in self.NON_ENEMY_CLASSES:
                enemies_visible += 1
        
        # Track combat duration
        if enemies_visible > 0:
            self.last_enemy_seen = now
            if self.combat_start is None:
                self.combat_start = now
        else:
            # No enemies for 3 seconds = combat ended
            if now - self.last_enemy_seen > 3.0:
                self.combat_start = None
                
        combat_duration = 0.0
        if self.combat_start is not None:
            combat_duration = now - self.combat_start
        
        # Calculate mode score
        aggressive_score = 0.0

        # ENEMY FACTOR - Most important!
        # If enemies are visible, we MUST be aggressive
        if enemies_visible >= self.ENEMY_THRESHOLD:
            aggressive_score += 0.6  # Enough by itself to trigger AGGRESSIVE

        # Velocity factor
        if velocity > self.VELOCITY_THRESHOLD:
            aggressive_score += 0.25
        elif velocity > self.VELOCITY_THRESHOLD / 2:
            aggressive_score += 0.1

        # Click rate factor
        if click_rate >= self.CLICK_RATE_THRESHOLD:
            aggressive_score += 0.15

        # Combat duration factor (longer combat = more aggressive)
        if combat_duration > 5.0:
            aggressive_score += 0.1

        # Determine mode with asymmetric hysteresis
        # - Switch TO AGGRESSIVE quickly when enemies appear (0.5s delay)
        # - Switch BACK TO PASSIVE slowly (3s delay after enemies gone)
        new_mode = "AGGRESSIVE" if aggressive_score >= 0.5 else "PASSIVE"

        # Apply asymmetric hysteresis
        if new_mode != self.current_mode:
            if new_mode == "AGGRESSIVE":
                # Quick switch to AGGRESSIVE when enemies appear
                if now - self.last_mode_switch >= 0.5:
                    self.current_mode = new_mode
                    self.last_mode_switch = now
                    print(f"🔄 Context switched to: {new_mode} (enemies detected)")
            else:
                # Slow switch back to PASSIVE - wait until combat is truly over
                # Combat must have ended (no enemies for 3s) AND hysteresis delay passed
                if self.combat_start is None and now - self.last_mode_switch >= self.MODE_SWITCH_DELAY:
                    self.current_mode = new_mode
                    self.last_mode_switch = now
                    print(f"🔄 Context switched to: {new_mode} (combat ended)")
                
        # Calculate confidence
        if self.current_mode == "AGGRESSIVE":
            confidence = min(1.0, aggressive_score * 2)
        else:
            confidence = min(1.0, (1.0 - aggressive_score) * 2)
            
        self.mode_confidence = confidence
        
        return ContextState(
            mode=self.current_mode,
            confidence=confidence,
            mouse_velocity=velocity,
            enemies_visible=enemies_visible,
            recent_clicks=int(click_rate * 3),
            combat_duration=combat_duration
        )
    
    def get_mode(self) -> Tuple[str, float]:
        """
        Get current mode and confidence.
        
        Returns:
            (mode, confidence) tuple
        """
        return self.current_mode, self.mode_confidence
    
    def reset(self):
        """Reset context (new game/respawn)"""
        self.current_mode = "PASSIVE"
        self.mode_confidence = 1.0
        self.combat_start = None
        self.last_enemy_seen = 0.0
        self.mouse_positions.clear()
        self.mouse_times.clear()
        self.recent_clicks.clear()


class ContextAwareTracker:
    """
    High-level tracker that combines context detection with data tagging.
    Used by the recorder to label data with the correct mode.
    """
    
    def __init__(self):
        self.detector = ContextDetector()
        self.mode_history: deque = deque(maxlen=1000)  # Track mode over time
        
    def update(self, mouse_x: int, mouse_y: int, detections: List, clicked: bool = False) -> str:
        """
        Update and return current mode.
        """
        state = self.detector.detect(mouse_x, mouse_y, detections, clicked)
        self.mode_history.append((time.time(), state.mode))
        return state.mode
    
    def get_mode_distribution(self, last_seconds: float = 60.0) -> dict:
        """
        Get distribution of modes over recent time.
        
        Returns:
            {"PASSIVE": 0.8, "AGGRESSIVE": 0.2} (example)
        """
        now = time.time()
        cutoff = now - last_seconds
        
        recent = [m for t, m in self.mode_history if t >= cutoff]
        if not recent:
            return {"PASSIVE": 1.0, "AGGRESSIVE": 0.0}
            
        passive_count = sum(1 for m in recent if m == "PASSIVE")
        aggressive_count = len(recent) - passive_count
        
        total = len(recent)
        return {
            "PASSIVE": passive_count / total,
            "AGGRESSIVE": aggressive_count / total
        }
