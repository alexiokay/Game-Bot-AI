"""
DarkOrbit Bot - Movement Generator
Generates mouse movements that match YOUR personal style.

This uses your movement profile to create paths that:
- Move at YOUR speed (with YOUR variance)
- Curve like YOU do
- Decelerate like YOU do before clicking
- Have YOUR timing patterns
"""

import json
import os
import time
import random
import math
import threading
from queue import Queue, Empty
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class MovementProfile:
    """Your personal movement style"""
    # Movement
    speed_mean: float = 0.5
    speed_std: float = 0.3
    speed_min: float = 0.1
    speed_max: float = 1.5
    click_interval_mean: float = 1.0
    click_interval_std: float = 0.5
    curve_factor_mean: float = 1.2
    curve_factor_std: float = 0.3
    deceleration_ratio: float = 1.3

    # Click timing (learned from YOUR recordings)
    click_hold_mean: float = 0.12      # How long you hold the button
    click_hold_std: float = 0.04
    click_hold_min: float = 0.06
    click_hold_max: float = 0.25
    pre_click_pause_mean: float = 0.03  # Pause before clicking
    pre_click_pause_std: float = 0.02
    post_click_pause_mean: float = 0.05  # Pause after releasing
    post_click_pause_std: float = 0.03
    double_click_rate: float = 0.05     # How often you double-click
    double_click_interval_mean: float = 0.15

    # Keyboard timing (learned from YOUR recordings)
    ctrl_hold_mean: float = 0.15       # Ctrl (attack) hold duration
    ctrl_hold_std: float = 0.05
    ctrl_hold_min: float = 0.10
    ctrl_hold_max: float = 0.25
    space_hold_mean: float = 0.12      # Space (rocket) hold duration
    space_hold_std: float = 0.04
    space_hold_min: float = 0.08
    space_hold_max: float = 0.20
    shift_hold_mean: float = 0.15      # Shift (special) hold duration
    shift_hold_std: float = 0.05
    shift_hold_min: float = 0.10
    shift_hold_max: float = 0.25

    @classmethod
    def load(cls, filepath: str) -> 'MovementProfile':
        """Load profile from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class MovementGenerator:
    """Generates human-like mouse movements based on YOUR profile"""
    
    def __init__(self, profile: MovementProfile):
        self.profile = profile
    
    def _bezier_point(self, t: float, p0: Tuple[float, float], p1: Tuple[float, float], 
                      p2: Tuple[float, float], p3: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate point on cubic bezier curve at parameter t (0-1)"""
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        
        x = mt3 * p0[0] + 3 * mt2 * t * p1[0] + 3 * mt * t2 * p2[0] + t3 * p3[0]
        y = mt3 * p0[1] + 3 * mt2 * t * p1[1] + 3 * mt * t2 * p2[1] + t3 * p3[1]
        
        return (x, y)
    
    def _generate_control_points(self, start: Tuple[float, float], end: Tuple[float, float]) -> Tuple:
        """Generate bezier control points that match YOUR curve style"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Your curve intensity (from profile)
        curve_amount = random.gauss(
            self.profile.curve_factor_mean - 1,  # Convert to offset
            self.profile.curve_factor_std * 0.5
        ) * 0.3  # Scale down for control point offset
        
        # Perpendicular direction for curve
        if distance > 0:
            perp_x = -dy / distance
            perp_y = dx / distance
        else:
            perp_x, perp_y = 0, 0
        
        # Random curve direction (left or right)
        curve_dir = random.choice([-1, 1])
        
        # Control points at 1/3 and 2/3 of the path
        offset1 = curve_amount * distance * curve_dir * random.uniform(0.5, 1.5)
        offset2 = curve_amount * distance * curve_dir * random.uniform(0.3, 1.0)
        
        p1 = (
            start[0] + dx * 0.33 + perp_x * offset1,
            start[1] + dy * 0.33 + perp_y * offset1
        )
        p2 = (
            start[0] + dx * 0.66 + perp_x * offset2,
            start[1] + dy * 0.66 + perp_y * offset2
        )
        
        return (start, p1, p2, end)
    
    def _apply_speed_profile(self, t: float) -> float:
        """Apply YOUR acceleration/deceleration pattern"""
        # Your deceleration ratio means you slow down at the end
        decel = self.profile.deceleration_ratio
        
        if decel > 1:
            # You decelerate: fast start, slow end
            # Use ease-out curve
            return 1 - (1 - t) ** (1 + (decel - 1) * 0.5)
        else:
            # You accelerate: slow start, fast end
            return t ** (1 + (1 - decel) * 0.5)
    
    def generate_path(self, start: Tuple[float, float], end: Tuple[float, float],
                      target_duration: Optional[float] = None) -> List[dict]:
        """
        Generate a path from start to end that moves like YOU.
        
        Args:
            start: Starting position (normalized 0-1 or pixels)
            end: Ending position (normalized 0-1 or pixels)
            target_duration: Optional duration in seconds (auto-calculated if None)
        
        Returns:
            List of movement points with timestamps
        """
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 0.001:
            return [{"x": end[0], "y": end[1], "time": 0}]
        
        # Calculate duration based on YOUR speed
        if target_duration is None:
            speed = max(0.1, random.gauss(self.profile.speed_mean, self.profile.speed_std))
            speed = max(self.profile.speed_min, min(self.profile.speed_max, speed))
            target_duration = distance / speed
        
        # Minimum duration for very short movements
        target_duration = max(0.05, target_duration)
        
        # Generate bezier control points (YOUR curve style)
        p0, p1, p2, p3 = self._generate_control_points(start, end)
        
        # Generate points along the path
        # More points = smoother movement
        num_points = max(50, int(target_duration * 500))  # ~500 points per second for ultra-smooth
        
        path = []
        for i in range(num_points + 1):
            # Linear progress
            t_linear = i / num_points
            
            # Apply YOUR speed profile (acceleration/deceleration)
            t_curved = self._apply_speed_profile(t_linear)
            
            # Get position on bezier curve
            x, y = self._bezier_point(t_curved, p0, p1, p2, p3)
            
            # Jitter disabled - was causing shakiness
            # if 0 < t_linear < 1:
            #     jitter = 0.001 * random.gauss(0, 1)
            #     x += jitter
            #     y += jitter
            
            # Timestamp
            timestamp = t_linear * target_duration
            
            path.append({
                "x": x,
                "y": y,
                "time": timestamp
            })
        
        return path
    
    def add_overshoot(self, path: List[dict], probability: float = 0.15) -> List[dict]:
        """Occasionally overshoot the target and correct (very human!)"""
        if random.random() > probability or len(path) < 5:
            return path
        
        # Get the target (last point)
        target = path[-1]
        
        # Overshoot by 2-8%
        overshoot_amount = random.uniform(0.02, 0.08)
        
        # Direction of movement
        if len(path) >= 2:
            dx = path[-1]["x"] - path[-2]["x"]
            dy = path[-1]["y"] - path[-2]["y"]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 0:
                dx, dy = dx/dist, dy/dist
            else:
                dx, dy = random.uniform(-1, 1), random.uniform(-1, 1)
        else:
            dx, dy = random.uniform(-1, 1), random.uniform(-1, 1)
        
        # Overshoot point
        overshoot = {
            "x": target["x"] + dx * overshoot_amount,
            "y": target["y"] + dy * overshoot_amount,
            "time": target["time"] + 0.05
        }
        
        # Correction back to target
        correction = {
            "x": target["x"],
            "y": target["y"],
            "time": target["time"] + 0.12
        }
        
        # Update last point time and add overshoot
        path[-1]["time"] = target["time"]
        path.append(overshoot)
        path.append(correction)
        
        return path
    
    def generate_click_timing(self, is_double_click: bool = False) -> dict:
        """Generate click timing that matches YOUR patterns (learned from recordings)"""
        # Hold duration from YOUR profile (gaussian distribution)
        hold_duration = random.gauss(
            self.profile.click_hold_mean,
            self.profile.click_hold_std
        )
        # Clamp to your observed range
        hold_duration = max(
            self.profile.click_hold_min,
            min(self.profile.click_hold_max, hold_duration)
        )

        # Pre-click pause from YOUR profile
        pre_delay = random.gauss(
            self.profile.pre_click_pause_mean,
            self.profile.pre_click_pause_std
        )
        pre_delay = max(0.005, pre_delay)  # At least 5ms

        # Post-click pause from YOUR profile
        post_delay = random.gauss(
            self.profile.post_click_pause_mean,
            self.profile.post_click_pause_std
        )
        post_delay = max(0.005, post_delay)

        # Should this be a double-click? (based on YOUR rate)
        should_double = random.random() < self.profile.double_click_rate
        double_click_interval = None
        if should_double or is_double_click:
            double_click_interval = random.gauss(
                self.profile.double_click_interval_mean,
                0.03  # Small variance for double-clicks
            )
            double_click_interval = max(0.08, min(0.3, double_click_interval))

        return {
            "pre_delay": pre_delay,
            "hold_duration": hold_duration,
            "post_delay": post_delay,
            "double_click": double_click_interval
        }
    
    def generate_idle_movement(self, center: Tuple[float, float], 
                               duration: float = 1.0) -> List[dict]:
        """Generate small idle movements (micro-jitter while waiting)"""
        movements = []
        current_time = 0
        current_pos = center
        
        while current_time < duration:
            # Small random movement
            offset_x = random.gauss(0, 0.005)
            offset_y = random.gauss(0, 0.005)
            
            new_pos = (
                center[0] + offset_x,
                center[1] + offset_y
            )
            
            # Time until next micro-movement
            wait = random.uniform(0.05, 0.2)
            current_time += wait
            
            movements.append({
                "x": new_pos[0],
                "y": new_pos[1],
                "time": current_time
            })
        
        return movements


class MouseController:
    """Controls the actual mouse movement - SENDS HARDWARE EVENTS (DirectInput) for Game Compatibility"""

    # Extended pynput to use DirectInput for clicks
    def __init__(self, screen_width: int, screen_height: int, game_rect: dict = None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.game_rect = game_rect or {
            "left": 0, "top": 0,
            "width": screen_width, "height": screen_height
        }

        # Init DirectInput structures
        import ctypes
        from ctypes import wintypes
        self.ctypes = ctypes

        # MOUSEINPUT structure for SendInput
        class MOUSEINPUT(ctypes.Structure):
                _fields_ = [
                    ("dx", wintypes.LONG),
                    ("dy", wintypes.LONG),
                    ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
                ]

        class INPUT(ctypes.Structure):
            class _INPUT(ctypes.Union):
                _fields_ = [("mi", MOUSEINPUT)]
            _fields_ = [("type", wintypes.DWORD), ("union", _INPUT)]

        self.MOUSEINPUT = MOUSEINPUT
        self.INPUT = INPUT
        self.SendInput = ctypes.windll.user32.SendInput
        
        # Input constants
        self.INPUT_MOUSE = 0
        self.MOUSEEVENTF_CES = 0x8000  # MOUSEEVENTF_ABSOLUTE if needed
        self.MOUSEEVENTF_LEFTDOWN = 0x0002
        self.MOUSEEVENTF_LEFTUP = 0x0004
        self.MOUSEEVENTF_RIGHTDOWN = 0x0008
        self.MOUSEEVENTF_RIGHTUP = 0x0010

        # Import pynput for MOVEMENT (it's smoother and less buggy for position than SendInput absolute)
        # But clicks MUST be SendInput for games.
        try:
            from pynput.mouse import Controller, Button
            self.mouse = Controller()
            self.Button = Button
            self.available = True
        except ImportError:
            self.available = False
            print("Warning: pynput not available, mouse control disabled")

        # Async movement: background thread executes paths without blocking
        self._move_queue = Queue()
        self._move_thread = None
        self._stop_thread = threading.Event()
        self._current_target = None  # Track where we're heading
        self._start_movement_thread()

    def _start_movement_thread(self):
        """Start the background movement worker thread"""
        if self._move_thread is not None and self._move_thread.is_alive():
            return
        self._stop_thread.clear()
        self._move_thread = threading.Thread(target=self._movement_worker, daemon=True)
        self._move_thread.start()

    def _movement_worker(self):
        """Background thread that executes movement paths"""
        while not self._stop_thread.is_set():
            try:
                # Get next path from queue (short timeout to check stop flag)
                path, speed_mult = self._move_queue.get(timeout=0.05)
            except Empty:
                continue

            if not path or len(path) < 2:
                continue

            # Execute the path (this is the blocking part, but in background)
            self._execute_path_blocking(path, speed_mult)

    def _execute_path_blocking(self, path: list, speed_multiplier: float):
        """Actually execute the path (called from background thread)"""
        if not self.available:
            return

        total_duration = path[-1]["time"] / speed_multiplier

        # FAST PATH: Very short movements - just move directly
        if total_duration < 0.03:
            self.move_to(path[-1]["x"], path[-1]["y"])
            return

        # Determine step count based on duration
        if total_duration < 0.10:
            num_steps = max(int(total_duration * 150), 5)  # 150 Hz
        else:
            num_steps = max(int(total_duration * 200), 15)  # 200 Hz

        start_time = time.perf_counter()

        for step in range(num_steps + 1):
            # Check if we should abort (new path queued)
            if not self._move_queue.empty():
                break  # Abort current path, new one waiting

            progress = step / num_steps
            target_time = progress * path[-1]["time"]

            # Interpolate position
            x, y = self._interpolate_path(path, target_time)
            self.move_to(x, y)

            # Timing: sleep for most, spin-wait for precision
            target_clock = start_time + (progress * total_duration)
            remaining = target_clock - time.perf_counter()

            if remaining > 0.003:
                time.sleep(remaining - 0.001)
            while time.perf_counter() < target_clock:
                pass

        # Final position
        self.move_to(path[-1]["x"], path[-1]["y"])

    def stop(self):
        """Stop the movement thread (call on shutdown)"""
        self._stop_thread.set()
        if self._move_thread and self._move_thread.is_alive():
            self._move_thread.join(timeout=0.5)

    def normalized_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert normalized (0-1) coordinates to screen pixels"""
        screen_x = int(self.game_rect["left"] + x * self.game_rect["width"])
        screen_y = int(self.game_rect["top"] + y * self.game_rect["height"])
        return (screen_x, screen_y)
    
    def move_to(self, x: float, y: float):
        """Move mouse to normalized position"""
        if not self.available:
            return
        screen_x, screen_y = self.normalized_to_screen(x, y)
        self.mouse.position = (screen_x, screen_y)
    
    def _send_click(self, down: bool, button: str):
        """Send hardware-level click event using SendInput"""
        extra = self.ctypes.c_ulong(0)
        flags = 0
        
        if button == "left":
            flags = self.MOUSEEVENTF_LEFTDOWN if down else self.MOUSEEVENTF_LEFTUP
        else:
            flags = self.MOUSEEVENTF_RIGHTDOWN if down else self.MOUSEEVENTF_RIGHTUP

        if down:
            print(f"[MOUSE] Hardware {button} click sent")

        mi = self.MOUSEINPUT(
            dx=0, dy=0, mouseData=0, dwFlags=flags, time=0,
            dwExtraInfo=self.ctypes.pointer(extra)
        )
        inp = self.INPUT(type=self.INPUT_MOUSE, union=self.INPUT._INPUT(mi=mi))
        self.SendInput(1, self.ctypes.pointer(inp), self.ctypes.sizeof(self.INPUT))

    def click(self, button: str = "left", hold_duration: float = 0.1):
        """Click with specified hold duration - NON-BLOCKING"""
        if not self.available:
            return

        # Press (Hardware event)
        self._send_click(True, button)

        # Release after hold_duration in background thread
        def release_after():
            time.sleep(hold_duration)
            self._send_click(False, button)

        threading.Thread(target=release_after, daemon=True).start()
    
    def execute_path(self, path: List[dict], speed_multiplier: float = 1.0):
        """Execute a movement path with SMOOTH interpolation"""
        if not self.available or not path:
            return
        
        # Use high-precision timer
        start_time = time.perf_counter()
        total_duration = path[-1]["time"] / speed_multiplier
        
        # Interpolate at high frequency (120+ Hz) for smooth movement
        update_interval = 0.004  # ~250 Hz update rate
        
        last_pos = None
        path_idx = 0
        
        while True:
            current_time = time.perf_counter() - start_time
            
            if current_time >= total_duration:
                # Ensure we reach the final position
                self.move_to(path[-1]["x"], path[-1]["y"])
                break
            
            # Find the two path points we're between
            target_time = current_time * speed_multiplier
            
            # Advance path index
            while path_idx < len(path) - 1 and path[path_idx + 1]["time"] <= target_time:
                path_idx += 1
            
            # Interpolate between points
            if path_idx < len(path) - 1:
                p1 = path[path_idx]
                p2 = path[path_idx + 1]
                
                # Calculate interpolation factor
                if p2["time"] != p1["time"]:
                    t = (target_time - p1["time"]) / (p2["time"] - p1["time"])
                    t = max(0, min(1, t))  # Clamp
                else:
                    t = 1
                
                # Smooth interpolation (ease function for extra smoothness)
                t_smooth = t * t * (3 - 2 * t)  # Smoothstep
                
                x = p1["x"] + (p2["x"] - p1["x"]) * t_smooth
                y = p1["y"] + (p2["y"] - p1["y"]) * t_smooth
            else:
                x = path[-1]["x"]
                y = path[-1]["y"]
            
            # Only update if position changed (avoid redundant updates)
            new_pos = self.normalized_to_screen(x, y)
            if new_pos != last_pos:
                self.mouse.position = new_pos
                last_pos = new_pos
            
            # High-precision sleep (spin-lock for timing accuracy)
            next_update = start_time + current_time + update_interval
            while time.perf_counter() < next_update:
                pass  # Spin-wait for precise timing
    
    def execute_path_smooth(self, path: List[dict], speed_multiplier: float = 1.0):
        """
        Execute path with smooth interpolation - NON-BLOCKING.

        Queues the path for execution by background thread.
        Returns immediately so control loop can continue.
        """
        if not self.available or not path or len(path) < 2:
            return

        total_duration = path[-1]["time"] / speed_multiplier

        # ULTRA-FAST: Tiny movements executed directly (no thread overhead)
        if total_duration < 0.02:
            self.move_to(path[-1]["x"], path[-1]["y"])
            return

        # Clear any pending paths (we want latest target, not queue buildup)
        while not self._move_queue.empty():
            try:
                self._move_queue.get_nowait()
            except Empty:
                break

        # Queue for async execution - returns immediately!
        self._current_target = (path[-1]["x"], path[-1]["y"])
        self._move_queue.put((path, speed_multiplier))
    
    def _interpolate_path(self, path: List[dict], target_time: float) -> Tuple[float, float]:
        """Find position at given time by interpolating path"""
        # Binary search for efficiency
        lo, hi = 0, len(path) - 1
        
        while lo < hi:
            mid = (lo + hi) // 2
            if path[mid]["time"] < target_time:
                lo = mid + 1
            else:
                hi = mid
        
        if lo == 0:
            return path[0]["x"], path[0]["y"]
        
        p1 = path[lo - 1]
        p2 = path[lo]
        
        if p2["time"] == p1["time"]:
            return p2["x"], p2["y"]
        
        t = (target_time - p1["time"]) / (p2["time"] - p1["time"])
        t = max(0, min(1, t))
        
        # Smoothstep for extra smoothness
        t = t * t * (3 - 2 * t)
        
        x = p1["x"] + (p2["x"] - p1["x"]) * t
        y = p1["y"] + (p2["y"] - p1["y"]) * t
        
        return x, y


class KeyboardController:
    """
    Controls keyboard input for game actions using DirectInput scan codes.

    Uses ctypes SendInput with proper structure for hardware-level key injection
    that works with games.
    """

    # DirectInput scan codes (hardware level - works with games)
    SCAN_CODES = {
        'ctrl': 0x1D,      # Left Control
        'space': 0x39,     # Spacebar
        'shift': 0x2A,     # Left Shift
        'alt': 0x38,       # Left Alt
        'tab': 0x0F,
        'esc': 0x01,
        'enter': 0x1C,
        # Number row
        '1': 0x02, '2': 0x03, '3': 0x04, '4': 0x05, '5': 0x06,
        '6': 0x07, '7': 0x08, '8': 0x09, '9': 0x0A, '0': 0x0B,
        # Letters
        'a': 0x1E, 'b': 0x30, 'c': 0x2E, 'd': 0x20, 'e': 0x12,
        'f': 0x21, 'g': 0x22, 'h': 0x23, 'i': 0x17, 'j': 0x24,
        'k': 0x25, 'l': 0x26, 'm': 0x32, 'n': 0x31, 'o': 0x18,
        'p': 0x19, 'q': 0x10, 'r': 0x13, 's': 0x1F, 't': 0x14,
        'u': 0x16, 'v': 0x2F, 'w': 0x11, 'x': 0x2D, 'y': 0x15,
        'z': 0x2C,
    }

    # Game action to key mapping
    KEY_MAP = {
        'attack': 'ctrl',      # Start/stop attack
        'rocket': 'space',     # Fire rocket
        'ability_1': '1',
        'ability_2': '2',
        'ability_3': '3',
        'ability_4': '4',
        'ability_5': '5',
        'ability_6': '6',
        'ability_7': '7',
        'ability_8': '8',
        'ability_9': '9',
        'ability_0': '0',
        'jump': 'j',           # Jump gate
        'config_1': 'c',       # Config switch
        'config_2': 'v',
    }

    # SendInput constants
    INPUT_KEYBOARD = 1
    KEYEVENTF_SCANCODE = 0x0008
    KEYEVENTF_KEYUP = 0x0002

    def __init__(self, profile: MovementProfile = None):
        import ctypes
        from ctypes import wintypes

        self.ctypes = ctypes
        self.available = True
        self.profile = profile  # For humanized timing from YOUR recordings

        # Define proper INPUT union structure for SendInput
        # This must match the Windows API exactly

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
            ]

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
            ]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [
                ("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD)
            ]

        class INPUT_UNION(ctypes.Union):
            _fields_ = [
                ("ki", KEYBDINPUT),
                ("mi", MOUSEINPUT),
                ("hi", HARDWAREINPUT)
            ]

        class INPUT(ctypes.Structure):
            _fields_ = [
                ("type", wintypes.DWORD),
                ("union", INPUT_UNION)
            ]

        self.KEYBDINPUT = KEYBDINPUT
        self.INPUT = INPUT
        self.INPUT_UNION = INPUT_UNION

        # Get SendInput function - don't set argtypes, let ctypes handle it
        self.SendInput = ctypes.windll.user32.SendInput

        print("  Keyboard: DirectInput scan codes (game-compatible)")

    def _send_key(self, scan_code: int, key_up: bool = False):
        """Send a key event using DirectInput scan codes"""
        flags = self.KEYEVENTF_SCANCODE
        if key_up:
            flags |= self.KEYEVENTF_KEYUP

        extra = self.ctypes.c_ulong(0)

        ki = self.KEYBDINPUT(
            wVk=0,  # 0 = use scan code
            wScan=scan_code,
            dwFlags=flags,
            time=0,
            dwExtraInfo=self.ctypes.pointer(extra)
        )

        # Create INPUT with union
        inp = self.INPUT()
        inp.type = self.INPUT_KEYBOARD
        inp.union.ki = ki

        # Call SendInput - create array of 1 INPUT
        inp_array = (self.INPUT * 1)(inp)
        result = self.SendInput(1, inp_array, self.ctypes.sizeof(self.INPUT))
        if result != 1:
            print(f"  Warning: SendInput returned {result}")

    def press_key(self, key_name: str, hold_duration: float = 0.05):
        """Press a key with optional hold duration"""
        if not self.available:
            return

        # Get actual key from map or use directly
        actual_key = self.KEY_MAP.get(key_name, key_name).lower()

        # Get scan code
        scan_code = self.SCAN_CODES.get(actual_key)
        if scan_code is None:
            print(f"  Warning: Unknown key '{actual_key}'")
            return

        # Press key
        self._send_key(scan_code, key_up=False)
        time.sleep(hold_duration)
        # Release key
        self._send_key(scan_code, key_up=True)

    def toggle_attack(self):
        """
        Toggle attack mode (Ctrl) - tap to toggle.
        Uses YOUR timing from recordings for human-like behavior.
        """
        if self.profile:
            # Use YOUR timing with variance
            hold_duration = random.gauss(
                self.profile.ctrl_hold_mean,
                self.profile.ctrl_hold_std
            )
            # Clamp to your observed range
            hold_duration = max(self.profile.ctrl_hold_min,
                               min(self.profile.ctrl_hold_max, hold_duration))
        else:
            # Default: 120-220ms with variance
            hold_duration = random.uniform(0.12, 0.22)

        self.press_key('attack', hold_duration=hold_duration)

    def hold_attack(self, hold: bool = True):
        """
        Hold or release Ctrl for continuous attacking.
        In DarkOrbit, Ctrl must be HELD DOWN (not toggled) to attack.
        """
        if not self.available:
            return

        scan_code = self.SCAN_CODES.get('ctrl')
        if scan_code is None:
            return

        if hold:
            # Press and keep Ctrl held
            self._send_key(scan_code, key_up=False)
        else:
            # Release Ctrl
            self._send_key(scan_code, key_up=True)

    def hold_key(self, key_name: str):
        """Hold a key down (for modifiers like shift, alt)."""
        if not self.available:
            return

        actual_key = self.KEY_MAP.get(key_name, key_name).lower()
        scan_code = self.SCAN_CODES.get(actual_key)
        if scan_code is None:
            return

        self._send_key(scan_code, key_up=False)

    def release_key(self, key_name: str):
        """Release a held key."""
        if not self.available:
            return

        actual_key = self.KEY_MAP.get(key_name, key_name).lower()
        scan_code = self.SCAN_CODES.get(actual_key)
        if scan_code is None:
            return

        self._send_key(scan_code, key_up=True)

    def fire_rocket(self):
        """Fire a rocket (Space) using YOUR timing from recordings"""
        if self.profile:
            hold_duration = random.gauss(
                self.profile.space_hold_mean,
                self.profile.space_hold_std
            )
            hold_duration = max(self.profile.space_hold_min,
                               min(self.profile.space_hold_max, hold_duration))
        else:
            hold_duration = random.uniform(0.08, 0.15)

        self.press_key('rocket', hold_duration=hold_duration)

    def press_shift(self):
        """Press Shift (special ability) using YOUR timing"""
        if self.profile:
            hold_duration = random.gauss(
                self.profile.shift_hold_mean,
                self.profile.shift_hold_std
            )
            hold_duration = max(self.profile.shift_hold_min,
                               min(self.profile.shift_hold_max, hold_duration))
        else:
            hold_duration = random.uniform(0.10, 0.20)

        self.press_key('shift', hold_duration=hold_duration)

    def use_ability(self, slot: int):
        """Use ability in slot 1-9 or 0"""
        if 0 <= slot <= 9:
            # Use similar timing to other keys
            hold_duration = random.uniform(0.08, 0.15)
            self.press_key(str(slot), hold_duration=hold_duration)


def demo():
    """Demo the movement generator"""
    print("\n" + "="*60)
    print("  MOVEMENT GENERATOR DEMO")
    print("="*60)
    
    # Load YOUR profile
    profile_path = os.path.join(os.path.dirname(__file__), "..", "data", "my_movement_profile.json")
    
    if not os.path.exists(profile_path):
        print(f"\nProfile not found at: {profile_path}")
        print("Run analyze_patterns.py first!")
        return
    
    profile = MovementProfile.load(profile_path)
    print(f"\n✅ Loaded YOUR movement profile:")
    print(f"   Speed: {profile.speed_mean:.2f} ± {profile.speed_std:.2f}")
    print(f"   Curve factor: {profile.curve_factor_mean:.2f}")
    print(f"   Deceleration: {profile.deceleration_ratio:.2f}")
    
    generator = MovementGenerator(profile)
    
    # Generate a sample path
    print("\n📍 Generating sample path...")
    start = (0.2, 0.3)
    end = (0.7, 0.6)
    
    path = generator.generate_path(start, end)
    path = generator.add_overshoot(path, probability=0.5)  # Higher prob for demo
    
    print(f"   From: ({start[0]:.2f}, {start[1]:.2f})")
    print(f"   To:   ({end[0]:.2f}, {end[1]:.2f})")
    print(f"   Points: {len(path)}")
    print(f"   Duration: {path[-1]['time']:.3f}s")
    
    # Show some points
    print("\n📊 Sample points:")
    for i in [0, len(path)//4, len(path)//2, 3*len(path)//4, -1]:
        p = path[i]
        print(f"   t={p['time']:.3f}s: ({p['x']:.4f}, {p['y']:.4f})")
    
    # Ask to demo actual movement
    print("\n" + "-"*60)
    demo_input = input("🖱️  Demo ACTUAL mouse movement? [y/N]: ").strip().lower()
    
    if demo_input == 'y':
        import ctypes
        user32 = ctypes.windll.user32
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
        
        controller = MouseController(screen_w, screen_h)
        
        print("\n⚠️  Mouse will move in 3 seconds...")
        print("   Press Ctrl+C to cancel")
        time.sleep(3)
        
        # Move to start
        controller.move_to(start[0], start[1])
        time.sleep(0.3)
        
        # Execute path
        print("   Moving...")
        controller.execute_path(path)
        
        # Click
        click_timing = generator.generate_click_timing()
        time.sleep(click_timing["pre_delay"])
        controller.click(hold_duration=click_timing["hold_duration"])
        
        print("   ✅ Done!")
    
    print("\n🎉 Movement generator is ready!")
    print("   Use this in the bot to move like YOU!")


if __name__ == "__main__":
    demo()
