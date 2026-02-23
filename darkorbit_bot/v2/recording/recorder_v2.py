"""
V2 Recorder - Records gameplay with full V2-compatible state encoding.

Records:
- YOLO detections with ByteTrack tracking (20-dim per object)
- Full V2 state encoding (player, objects, context)
- Mouse/keyboard actions
- Mode tagging (AGGRESSIVE/PASSIVE)

Output format matches what V2 training expects:
- objects: [max_objects, 20] - tracked object features
- object_mask: [max_objects] - which slots are valid
- player: [16] - player state features
- context: [16] - global context features
- actions: [3 + NUM_KEYBOARD_KEYS] - [mouse_x, mouse_y, click, ...keyboard_keys...]

Hotkeys:
- F5 = Start/Stop recording
- F6 = Mark as "good" (manual save)
- F7 = Discard buffer

Usage:
    python -m darkorbit_bot.v2.recording.recorder_v2 --model best.pt --monitor 1
"""

import time
import json
import sys
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# pynput for input capture
try:
    from pynput import mouse, keyboard
    from pynput.mouse import Button
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("pynput not installed. Run: pip install pynput")

# Local imports
from ..config import (
    V2Config, TacticianConfig, StrategistConfig,
    ENEMY_CLASSES, LOOT_CLASSES,
    KEYBOARD_KEYS, NUM_KEYBOARD_KEYS,
    NUM_OBJECTS_IN_FLAT_STATE
)
from ..perception.tracker import ObjectTracker, TrackedObject
from ..perception.state_encoder import StateEncoderV2, PlayerState

# Detection imports
try:
    from detection.detector import GameDetector, ScreenCapture
except ImportError:
    try:
        from ...detection.detector import GameDetector, ScreenCapture
    except ImportError:
        GameDetector = None
        ScreenCapture = None


@dataclass
class V2Frame:
    """Single recorded frame with V2 state encoding."""
    timestamp: float

    # V2 State Components
    player_features: np.ndarray      # [16] player state
    object_features: np.ndarray      # [max_objects, 20] tracked objects
    object_mask: np.ndarray          # [max_objects] valid mask
    context_features: np.ndarray     # [16] global context

    # Raw data for debugging
    num_detections: int
    num_tracked: int

    # Input
    mouse_pos: Tuple[float, float]   # Normalized (0-1)
    mouse_velocity: float
    keys_pressed: List[str]
    clicks: List[Dict]

    # Context
    mode: str                        # AGGRESSIVE or PASSIVE
    health_estimate: float


class V2Recorder:
    """
    Records gameplay with full V2-compatible state encoding.

    Uses:
    - YOLO for object detection
    - ByteTrack for object tracking
    - StateEncoderV2 for feature encoding
    """

    def __init__(self,
                 model_path: str,
                 monitor: int = 1,
                 buffer_seconds: float = 30.0,
                 save_before_kill: float = 10.0,
                 fps_target: float = 30.0,
                 config: V2Config = None):

        self.config = config or V2Config()

        # Vision components
        if GameDetector is None:
            raise ImportError("GameDetector not found. Make sure detection module is available.")
        self.detector = GameDetector(model_path, confidence_threshold=0.4)
        self.screen = ScreenCapture(monitor_index=monitor)

        # V2 components
        self.tracker = ObjectTracker(
            high_thresh=self.config.perception.track_high_thresh,
            low_thresh=self.config.perception.track_low_thresh,
            match_thresh=self.config.perception.match_thresh,
            track_buffer=self.config.perception.track_buffer
        )

        self.state_encoder = StateEncoderV2(
            max_objects=self.config.tactician.max_objects,
            object_dim=self.config.tactician.object_dim,
            player_dim=16,
            context_dim=16
        )

        # Recording state
        self.is_recording = False
        self.buffer: List[V2Frame] = []
        self.max_buffer_frames = int(buffer_seconds * fps_target)
        self.fps_target = fps_target
        self.save_before_kill_frames = int(save_before_kill * fps_target)

        # Current frame data
        self.current_mouse = (0, 0)
        self.prev_mouse = (0, 0)
        self.current_keys: set = set()
        self.pending_clicks: List[Dict] = []
        self.scroll_delta = 0

        # Screen dimensions (for normalization) - get actual screen size
        try:
            import ctypes
            user32 = ctypes.windll.user32
            # Get virtual screen size (handles multi-monitor)
            self.screen_width = user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
            self.screen_height = user32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN
            if self.screen_width == 0 or self.screen_height == 0:
                # Fallback to primary monitor
                self.screen_width = user32.GetSystemMetrics(0)
                self.screen_height = user32.GetSystemMetrics(1)
            print(f"[V2 Recorder] Screen size: {self.screen_width}x{self.screen_height}")
        except Exception as e:
            print(f"[V2 Recorder] Could not detect screen size, using 1920x1080: {e}")
            self.screen_width = 1920
            self.screen_height = 1080

        # Output (same folder as shadow training and training scripts for consistency)
        self.output_dir = Path("darkorbit_bot/data/recordings_v2")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Don't create session subfolder - save directly to recordings_v2 for easier training
        self.session_dir = self.output_dir

        # Stats
        self.total_kills = 0
        self.total_deaths = 0
        self.sequences_saved = 0

        # Health tracking for kill/death detection
        self.prev_health = 1.0
        self.enemy_count_history: List[int] = []

        # Threading
        self.record_thread: Optional[threading.Thread] = None
        self.running = False

        # Input listeners
        self.mouse_listener = None
        self.keyboard_listener = None

        # Mouse cursor confinement (prevent leaving screen during recording)
        import ctypes
        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long),
                       ("top", ctypes.c_long),
                       ("right", ctypes.c_long),
                       ("bottom", ctypes.c_long)]
        self._rect = RECT()
        self._cursor_confined = False

    def _setup_listeners(self):
        """Setup input listeners."""
        if not PYNPUT_AVAILABLE:
            return

        def on_move(x, y):
            self.current_mouse = (x, y)

        def on_click(x, y, button, pressed):
            if pressed:
                self.pending_clicks.append({
                    'x': x, 'y': y,
                    'button': 'left' if button == Button.left else 'right',
                    'time': time.time()
                })

        def on_scroll(x, y, dx, dy):
            self.scroll_delta += dy

        def on_key_press(key):
            try:
                key_name = key.char if hasattr(key, 'char') and key.char else key.name
                self.current_keys.add(key_name)
            except:
                pass

        def on_key_release(key):
            try:
                key_name = key.char if hasattr(key, 'char') and key.char else key.name
                self.current_keys.discard(key_name)
            except:
                pass

            # Hotkeys
            if hasattr(key, 'name'):
                if key.name == 'f5':
                    self._toggle_recording()
                elif key.name == 'f6':
                    self._manual_save()
                elif key.name == 'f7':
                    self._manual_discard()

        self.mouse_listener = mouse.Listener(
            on_move=on_move,
            on_click=on_click,
            on_scroll=on_scroll
        )
        self.keyboard_listener = keyboard.Listener(
            on_press=on_key_press,
            on_release=on_key_release
        )

    def confine_cursor(self):
        """Confine mouse cursor to screen bounds (prevents leaving game window)."""
        if self._cursor_confined:
            print("[MouseLock] Cursor already confined")
            return

        # Get the monitor bounds from screen capture object
        monitor = self.screen.monitor
        screen_left = monitor.get('left', 0)
        screen_top = monitor.get('top', 0)
        screen_width = monitor['width']
        screen_height = monitor['height']

        # Set up RECT for the game monitor bounds
        self._rect.left = screen_left
        self._rect.top = screen_top
        self._rect.right = screen_left + screen_width
        self._rect.bottom = screen_top + screen_height

        print(f"[MouseLock] Attempting to confine cursor to RECT:")
        print(f"            left={self._rect.left}, top={self._rect.top}")
        print(f"            right={self._rect.right}, bottom={self._rect.bottom}")
        print(f"            (size: {screen_width}x{screen_height})")

        # Call ClipCursor to confine the cursor
        import ctypes
        result = ctypes.windll.user32.ClipCursor(ctypes.byref(self._rect))
        if result:
            self._cursor_confined = True
            print(f"[MouseLock] ✓ Cursor confined successfully!")
            print(f"[MouseLock] You should NOT be able to move mouse outside the game screen")
        else:
            error_code = ctypes.get_last_error()
            print(f"[MouseLock] ✗ Failed to confine cursor (error code: {error_code})")

    def release_cursor(self):
        """Release mouse cursor confinement."""
        if not self._cursor_confined:
            return
        import ctypes
        result = ctypes.windll.user32.ClipCursor(None)
        if result:
            self._cursor_confined = False
            print("[MouseLock] Cursor released")

    def _toggle_recording(self):
        """Toggle recording on/off. When stopping, exits the app."""
        if self.is_recording:
            self.stop(exit_app=True)  # Exit when F5 stops recording
        else:
            self.start()

    def _manual_save(self):
        """Manually save current buffer as successful."""
        if self.buffer:
            print("\nF6 - Manual save triggered!")
            self._save_sequence("MANUAL_SUCCESS", len(self.buffer))

    def _manual_discard(self):
        """Manually discard buffer."""
        print("\nF7 - Buffer discarded!")
        self.buffer.clear()
        self.tracker.reset()
        self.state_encoder.reset()

    def start(self):
        """Start recording."""
        if self.is_recording:
            return

        print("\n" + "="*60)
        print("  V2 RECORDER - Starting")
        print("="*60)

        # Create session directory
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Setup listeners
        self._setup_listeners()
        if self.mouse_listener:
            self.mouse_listener.start()
        if self.keyboard_listener:
            self.keyboard_listener.start()

        # Start recording thread
        self.running = True
        self.is_recording = True
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.record_thread.start()

        # Lock mouse cursor to prevent leaving screen (prevents bad training data)
        self.confine_cursor()

        print(f"\nSession: {self.session_dir}")
        print("\nHotkeys:")
        print("   F5 = Stop recording")
        print("   F6 = Manual save (mark as good)")
        print("   F7 = Discard buffer")
        print("\nAuto-triggers:")
        print("   Kill detected -> Save last 10s")
        print("   Death detected -> Discard last 30s")
        print("-"*60)

    def stop(self, exit_app: bool = False, force: bool = False):
        """Stop recording.

        Args:
            exit_app: If True, exit the entire application after stopping
            force: If True, skip saving and exit immediately
        """
        if force:
            print("\nForce stopping recorder (no save)...")
            self.running = False
            self.is_recording = False
            self.release_cursor()
            import sys
            sys.exit(0)

        print("\nStopping recorder...")

        self.running = False
        self.is_recording = False

        # Release mouse cursor confinement
        self.release_cursor()

        if self.record_thread:
            self.record_thread.join(timeout=2.0)

        # Stop listeners forcefully
        try:
            if self.mouse_listener:
                self.mouse_listener.stop()
                self.mouse_listener = None
        except Exception:
            pass  # Ignore errors on shutdown

        try:
            if self.keyboard_listener:
                self.keyboard_listener.stop()
                self.keyboard_listener = None
        except Exception:
            pass  # Ignore errors on shutdown

        # Save remaining buffer if anything good
        if self.buffer and len(self.buffer) > 30:
            self._save_sequence("SESSION_END", len(self.buffer))

        self._print_stats()

        # Force exit if requested (when running standalone)
        if exit_app:
            print("\nExiting recorder...")
            import sys
            sys.exit(0)

    def _detect_mode(self, tracked_objects: List[TrackedObject],
                     keys_pressed: List[str], clicks: List[Dict]) -> str:
        """Detect current mode from game state."""
        # Check for enemies nearby
        enemies = [t for t in tracked_objects if t.class_name in ENEMY_CLASSES]
        boxes = [t for t in tracked_objects if t.class_name in LOOT_CLASSES]

        # Check for attack keys
        attack_keys = {'ctrl', 'ctrl_l', 'ctrl_r', 'space'}
        is_attacking = bool(attack_keys & set(keys_pressed)) or len(clicks) > 0

        # Simple heuristic:
        # - If attacking or enemies close -> AGGRESSIVE
        # - Otherwise -> PASSIVE
        if is_attacking and enemies:
            return "AGGRESSIVE"
        elif enemies and any(e.confidence > 0.5 for e in enemies):
            # Enemies detected but not attacking yet
            return "AGGRESSIVE"
        else:
            return "PASSIVE"

    def _estimate_health(self, detections) -> float:
        """Estimate player health (placeholder - needs HUD reading)."""
        # TODO: Implement HUD reading for actual health
        # For now, return 1.0 (full health)
        return 1.0

    def _detect_kill(self, tracked_objects: List[TrackedObject], is_attacking: bool = False) -> bool:
        """
        Detect if a kill just happened.
        
        Requirements for a valid kill detection:
        1. Cooldown: At least 5 seconds since last kill
        2. Drop threshold: Enemy count dropped by 2+ OR dropped to 0
        3. Sustained: The drop persisted for at least 5 frames (not just YOLO flicker)
        4. Context: Player was recently attacking (clicking/ctrl)
        
        Returns:
            True if a kill was detected, False otherwise
        """
        current_enemies = len([t for t in tracked_objects if t.class_name in ENEMY_CLASSES])
        now = time.time()
        
        # Update history
        self.enemy_count_history.append(current_enemies)
        if len(self.enemy_count_history) > 90:  # 3 seconds at 30fps
            self.enemy_count_history.pop(0)
        
        # Check cooldown (5 seconds between kills)
        if not hasattr(self, '_last_kill_time'):
            self._last_kill_time = 0
        if now - self._last_kill_time < 5.0:
            return False
        
        # Need enough history
        if len(self.enemy_count_history) < 30:
            return False
        
        # Calculate stats
        # Look at enemies 1-2 seconds ago (frames 30-60 ago)
        old_count = max(self.enemy_count_history[-60:-30]) if len(self.enemy_count_history) >= 60 else max(self.enemy_count_history[:-15])
        # Look at enemies in last 0.5 seconds (sustained low count)
        recent_counts = self.enemy_count_history[-15:]
        recent_max = max(recent_counts)
        recent_min = min(recent_counts)
        
        # Kill detection criteria:
        # 1. There WERE enemies before (old_count >= 1)
        # 2. The count dropped significantly (by 2+ OR to 0)
        # 3. The drop is sustained (recent_max == recent_min, no flickering)
        # 4. Current count is lower than before
        drop = old_count - current_enemies
        
        if old_count >= 2 and drop >= 2 and recent_max == recent_min:
            # Significant sustained drop
            self._last_kill_time = now
            return True
        
        if old_count >= 1 and current_enemies == 0 and recent_max == 0:
            # All enemies cleared (sustained)
            self._last_kill_time = now
            return True
        
        return False

    def _detect_death(self, health: float) -> bool:
        """Detect if player died."""
        # Health dropped to 0 or very low
        if self.prev_health > 0.3 and health < 0.1:
            return True
        return False

    def _record_loop(self):
        """Main recording loop."""
        frame_interval = 1.0 / self.fps_target

        while self.running:
            start = time.time()

            try:
                # 1. Capture frame
                frame = self.screen.capture()

                # 2. Run YOLO detection
                detections = self.detector.detect_frame(frame)

                # 3. Update tracker with detections
                tracked_objects = self.tracker.update(detections)

                # 4. Get current input state
                mouse_x, mouse_y = self.current_mouse
                mouse_dx = mouse_x - self.prev_mouse[0]
                mouse_dy = mouse_y - self.prev_mouse[1]
                self.prev_mouse = self.current_mouse

                # Normalize mouse position
                mouse_x_norm = mouse_x / self.screen_width
                mouse_y_norm = mouse_y / self.screen_height

                # Check if mouse button is HELD (not just click events)
                # This captures sustained clicks, not just the moment of press
                import ctypes
                lmb_held = ctypes.windll.user32.GetAsyncKeyState(0x01) < 0
                rmb_held = ctypes.windll.user32.GetAsyncKeyState(0x02) < 0

                # Build clicks list from held state (more reliable than events)
                clicks = []
                if lmb_held:
                    clicks.append({'x': mouse_x, 'y': mouse_y, 'button': 'left', 'held': True})
                if rmb_held:
                    clicks.append({'x': mouse_x, 'y': mouse_y, 'button': 'right', 'held': True})

                # Clear pending click events (we're using held state instead)
                self.pending_clicks.clear()

                keys = list(self.current_keys)

                # 5. Detect mode and health
                mode = self._detect_mode(tracked_objects, keys, clicks)
                health = self._estimate_health(detections)

                # 6. Build V2 state using StateEncoderV2
                is_attacking = len(clicks) > 0 or any(k in ['ctrl', 'ctrl_l', 'ctrl_r'] for k in keys)

                encoded_state = self.state_encoder.encode(
                    tracked_objects=tracked_objects,
                    player_x=mouse_x_norm,
                    player_y=mouse_y_norm,
                    hp=health,
                    shield=1.0,
                    is_attacking=is_attacking,
                    idle_time=0.0
                )

                # 7. Create V2 frame
                v2_frame = V2Frame(
                    timestamp=time.time(),
                    player_features=encoded_state['player'],
                    object_features=encoded_state['objects'],
                    object_mask=encoded_state['object_mask'],
                    context_features=encoded_state['context'],
                    num_detections=len(detections),
                    num_tracked=len(tracked_objects),
                    mouse_pos=(mouse_x_norm, mouse_y_norm),
                    mouse_velocity=np.sqrt(mouse_dx**2 + mouse_dy**2),
                    keys_pressed=keys,
                    clicks=clicks,
                    mode=mode,
                    health_estimate=health
                )

                # 8. Add to buffer
                self.buffer.append(v2_frame)
                if len(self.buffer) > self.max_buffer_frames:
                    self.buffer.pop(0)

                # 9. Check for kill
                if self._detect_kill(tracked_objects):
                    print("\nKILL DETECTED! Saving last 10 seconds...")
                    self.total_kills += 1
                    self._save_sequence("KILL", self.save_before_kill_frames)

                # 10. Check for death
                if self._detect_death(health):
                    print("\nDEATH DETECTED! Discarding last 30 seconds...")
                    self.total_deaths += 1
                    self.buffer.clear()
                    self.tracker.reset()
                    self.state_encoder.reset()

                self.prev_health = health

                # 11. Print status
                self._print_status(mode, health, tracked_objects)

            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()

            # Maintain FPS
            elapsed = time.time() - start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _save_sequence(self, label: str, num_frames: int):
        """Save a sequence to disk in V2 format."""
        if not self.buffer:
            return

        # Get last N frames
        frames_to_save = self.buffer[-num_frames:] if num_frames < len(self.buffer) else self.buffer.copy()

        if len(frames_to_save) < 10:
            print("   (Too short, skipping)")
            return

        # Build V2-compatible data structure
        # For Tactician training:
        objects_list = []
        masks_list = []

        # For Strategist training:
        states_list = []

        # Actions
        actions_list = []

        for frame in frames_to_save:
            # Objects for Tactician (shape: [max_objects, 20])
            objects_list.append(frame.object_features.tolist())
            masks_list.append(frame.object_mask.tolist())

            # Full state for Strategist (concatenate player + flat objects + context)
            flat_objects = frame.object_features[:NUM_OBJECTS_IN_FLAT_STATE].flatten()
            full_state = np.concatenate([
                frame.player_features,
                flat_objects,
                frame.context_features
            ])
            states_list.append(full_state.tolist())

            # Actions: [mouse_x, mouse_y, click, ...keyboard_keys...]
            # Total: 3 + NUM_KEYBOARD_KEYS = 31 dimensions (matching shadow trainer format)
            clicked = 3.0 if frame.clicks else -3.0  # Use logit format like shadow trainer

            # Build full action array
            action = [
                frame.mouse_pos[0],
                frame.mouse_pos[1],
                clicked,
            ]

            # Add all keyboard keys in KEYBOARD_KEYS order
            # Normalize key names from pynput format
            keys_lower = {k.lower().replace('_l', '').replace('_r', '') for k in frame.keys_pressed}
            for key_name in KEYBOARD_KEYS:
                key_pressed = key_name in keys_lower
                action.append(3.0 if key_pressed else -3.0)  # Logit format

            actions_list.append(action)

        # Determine mode (majority vote)
        modes = [f.mode for f in frames_to_save]
        mode = max(set(modes), key=modes.count)

        # Save
        seq_id = self.sequences_saved
        filename = f"sequence_{seq_id:04d}_{label}_{mode}.json"
        filepath = self.session_dir / filename

        data = {
            'id': seq_id,
            'label': 'SUCCESS' if 'KILL' in label or 'SUCCESS' in label else label,
            'mode': mode,
            'num_frames': len(frames_to_save),
            'timestamp': datetime.now().isoformat(),
            'format_version': 'v2',

            # V2 state data
            'states': states_list,           # For Strategist (128-dim per frame)
            'actions': actions_list,          # [mouse_x, mouse_y, click, ctrl, space]

            # V2 object data (for Tactician)
            'objects': objects_list,          # [max_objects, 20] per frame
            'object_masks': masks_list,       # [max_objects] per frame

            # Metadata
            'config': {
                'max_objects': self.config.tactician.max_objects,
                'object_dim': self.config.tactician.object_dim,
                'state_dim': len(states_list[0]) if states_list else 0,
                'action_dim': 3 + NUM_KEYBOARD_KEYS,  # mouse_x, mouse_y, click + keyboard keys
                'keyboard_keys': list(KEYBOARD_KEYS)  # Key order for reference
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f)

        self.sequences_saved += 1
        print(f"\n   Saved: {filename} ({len(frames_to_save)} frames)")

    def _print_status(self, mode: str, health: float, tracked_objects: List[TrackedObject]):
        """Print current status."""
        enemies = len([t for t in tracked_objects if t.class_name in ENEMY_CLASSES])
        boxes = len([t for t in tracked_objects if t.class_name in LOOT_CLASSES])
        tracked = len(tracked_objects)

        status = f"\rREC | Buffer: {len(self.buffer):4d} | "
        status += f"Mode: {mode:10} | "
        status += f"Tracked: {tracked:2d} | "
        status += f"Enemies: {enemies} Boxes: {boxes} | "
        status += f"Saved: {self.sequences_saved}    "

        print(status, end="")

    def _print_stats(self):
        """Print session stats."""
        print("\n" + "="*60)
        print("  SESSION STATS")
        print("="*60)
        print(f"   Kills detected: {self.total_kills}")
        print(f"   Deaths detected: {self.total_deaths}")
        print(f"   Sequences saved: {self.sequences_saved}")
        print(f"   Output: {self.session_dir}")
        print("="*60)


def main():
    import argparse
    import signal
    import sys

    parser = argparse.ArgumentParser(description='V2 Recorder')
    parser.add_argument('--model', type=str, default='F:/dev/bot/best.pt',
                       help='Path to YOLO model')
    parser.add_argument('--monitor', type=int, default=1,
                       help='Monitor to capture')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Recording FPS')
    parser.add_argument('--buffer', type=float, default=30.0,
                       help='Buffer size in seconds')

    args = parser.parse_args()

    config = V2Config()

    recorder = V2Recorder(
        model_path=args.model,
        monitor=args.monitor,
        fps_target=args.fps,
        buffer_seconds=args.buffer,
        config=config
    )

    # Signal handler for clean shutdown (when launcher stops the process)
    # Use force=True to exit immediately without saving (launcher wants fast stop)
    def signal_handler(signum, _frame):
        print(f"\nReceived signal {signum}, force stopping...")
        recorder.stop(force=True)  # Fast exit, no save

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if hasattr(signal, 'SIGBREAK'):  # Windows-specific
        signal.signal(signal.SIGBREAK, signal_handler)

    print("\nV2 Recorder Ready")
    print("   Press F5 to start recording")
    print("   Press Ctrl+C to exit")

    # Setup listeners but don't start recording yet - wait for F5
    recorder._setup_listeners()
    if recorder.keyboard_listener:
        recorder.keyboard_listener.start()
    if recorder.mouse_listener:
        recorder.mouse_listener.start()

    # Set running flag so the main loop continues
    recorder.running = True

    try:
        while recorder.running:
            time.sleep(0.5)  # Check more frequently for stop signal
    except (KeyboardInterrupt, SystemExit):
        pass  # Signal handler will handle cleanup
    finally:
        # Ensure cleanup happens even if main loop exits unexpectedly
        if recorder.running or recorder.is_recording:
            recorder.stop(exit_app=False)  # Don't call sys.exit in finally
        print("\nRecorder terminated.")


if __name__ == "__main__":
    main()
