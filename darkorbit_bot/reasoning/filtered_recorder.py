"""
DarkOrbit Bot - Filtered Recorder

Records gameplay with intelligent filtering:
- Only saves data from "successful" plays (kills, survival)
- Auto-deletes data before deaths
- Tags data as PASSIVE or AGGRESSIVE
- Applies Gaussian smoothing to mouse movements

Hotkeys:
- F5 = Start/Stop recording
- F6 = Mark as "good" (manual save trigger)
- F7 = Mark as "bad" (discard buffer)

Usage:
    python filtered_recorder.py --model path/to/best.pt --monitor 1
"""

import time
import json
import sys
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# pynput for input capture
try:
    from pynput import mouse, keyboard
    from pynput.mouse import Button
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("⚠️ pynput not installed. Run: pip install pynput")

# Local imports
try:
    from ..detection.detector import GameDetector, ScreenCapture
    from .filters import create_filters, BufferFrame, KillFilter
    from .context_detector import ContextDetector
    from .state_builder import StateBuilder, PlayerState
except ImportError:
    from detection.detector import GameDetector, ScreenCapture
    from reasoning.filters import create_filters, BufferFrame, KillFilter
    from reasoning.context_detector import ContextDetector
    from reasoning.state_builder import StateBuilder, PlayerState


@dataclass
class RecordedAction:
    """Single recorded action"""
    timestamp: float
    mouse_x: int
    mouse_y: int
    mouse_dx: float  # Velocity
    mouse_dy: float
    left_click: bool
    right_click: bool
    keys_pressed: List[str]
    scroll_delta: int


class FilteredRecorder:
    """
    Intelligent gameplay recorder with filtering.
    
    Features:
    - Rolling buffer (keeps last 30s)
    - Auto-saves on kill detected
    - Auto-discards on death
    - Mode tagging (PASSIVE/AGGRESSIVE)
    - Gaussian smoothing of mouse paths
    """
    
    def __init__(self, 
                 model_path: str,
                 monitor: int = 1,
                 buffer_seconds: float = 30.0,
                 save_before_kill: float = 10.0,
                 fps_target: float = 30.0):
        
        # Vision
        self.detector = GameDetector(model_path, confidence_threshold=0.4)
        self.screen = ScreenCapture(monitor_index=monitor)
        
        # Filters
        self.filters = create_filters({
            'buffer_seconds': buffer_seconds,
            'save_before_kill': save_before_kill,
            'smooth_sigma': 3.0
        })
        
        # Context
        self.context = ContextDetector()
        # Default to old format (128 features) for backwards compatibility
        # New recordings can be converted to 134-feature format during training
        self.state_builder = StateBuilder(include_movement_patterns=False)
        
        # Recording state
        self.is_recording = False
        self.buffer: List[BufferFrame] = []
        self.max_buffer_frames = int(buffer_seconds * fps_target)
        self.fps_target = fps_target
        
        # Current frame data
        self.current_mouse = (0, 0)
        self.prev_mouse = (0, 0)
        self.current_keys: set = set()
        self.pending_clicks: List[Dict] = []
        self.scroll_delta = 0
        
        # Output
        self.output_dir = Path("data/recordings")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"session_{self.session_id}"

        # Screenshot saving for VLM annotation
        self.screenshot_interval = 2.0  # Save screenshot every 2 seconds
        self.last_screenshot_time = 0
        self.screenshot_buffer: List[Dict] = []  # Recent screenshots with context
        self.max_screenshot_buffer = 15  # Keep last 15 screenshots (30s at 2s interval)
        
        # Stats
        self.total_kills = 0
        self.total_deaths = 0
        self.sequences_saved = 0
        
        # Threading
        self.record_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Input listeners
        self.mouse_listener = None

        # Screen dimensions (for normalization) - get actual screen size
        try:
            import ctypes
            user32 = ctypes.windll.user32
            self.screen_width = user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
            self.screen_height = user32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN
            if self.screen_width == 0 or self.screen_height == 0:
                self.screen_width = user32.GetSystemMetrics(0)
                self.screen_height = user32.GetSystemMetrics(1)
            print(f"[Recorder] Screen size: {self.screen_width}x{self.screen_height}")
        except Exception:
            self.screen_width = 1920
            self.screen_height = 1080
        self.keyboard_listener = None
        
    def _setup_listeners(self):
        """Setup input listeners"""
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
                key_name = key.char if hasattr(key, 'char') else key.name
                self.current_keys.add(key_name)
            except:
                pass
                
        def on_key_release(key):
            try:
                key_name = key.char if hasattr(key, 'char') else key.name
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
                elif key.name == 'f8':
                    self._manual_snapshot()
                elif key.name == 'f9':
                    self._toggle_continuous_collection()
                    
        self.mouse_listener = mouse.Listener(
            on_move=on_move,
            on_click=on_click,
            on_scroll=on_scroll
        )
        self.keyboard_listener = keyboard.Listener(
            on_press=on_key_press,
            on_release=on_key_release
        )

        # Collection state
        self.collect_mode = False
        self.collect_label = "unknown"
        self.collect_dir = Path("data/collect")
        self.collect_dir.mkdir(parents=True, exist_ok=True)
        self.last_collect_time = 0

    def _get_label_from_user(self):
        """Popup dialog to get label from user"""
        try:
            import tkinter as tk
            from tkinter import simpledialog
            
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            
            label = simpledialog.askstring("Data Collection", "Enter Label:", parent=root)
            root.destroy()
            return label
        except Exception as e:
            print(f"Error showing dialog: {e}")
            return "unknown"

    def _manual_snapshot(self):
        """F8: Snapshot current frame with label"""
        label = self._get_label_from_user()
        if not label:
            return

        self.collect_label = label # Remember for F9
        
        # Capture now
        frame = self.screen.capture()
        if frame is None: 
            return

        timestamp = int(time.time() * 1000)
        filename = f"{label}_{timestamp}.jpg"
        path = self.collect_dir / filename
        
        import cv2
        cv2.imwrite(str(path), frame)
        print(f"\n📸 SNAPSHOT SAVED: {filename}")

    def _toggle_continuous_collection(self):
        """F9: Toggle continuous collection"""
        self.collect_mode = not self.collect_mode
        
        if self.collect_mode:
            if self.collect_label == "unknown":
                label = self._get_label_from_user()
                if label:
                    self.collect_label = label
            
            print(f"\n🎥 CONTINUOUS COLLECT STARTED (Label: {self.collect_label})")
        else:
            print(f"\n⏹️ CONTINUOUS COLLECT STOPPED")
        
    def _toggle_recording(self):
        """Toggle recording on/off"""
        if self.is_recording:
            self.stop()
        else:
            self.start()
            
    def _manual_save(self):
        """Manually save current buffer as successful"""
        if self.buffer:
            print("\n💾 F6 - Manual save triggered!")
            self._save_sequence("MANUAL_SUCCESS", len(self.buffer))
            
    def _manual_discard(self):
        """Manually discard buffer"""
        print("\n🗑️ F7 - Buffer discarded!")
        self.buffer.clear()
        
    def start(self):
        """Start recording"""
        if self.is_recording:
            return
            
        print("\n" + "="*60)
        print("  🔴 FILTERED RECORDER - Starting")
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
        
        print(f"\n📁 Session: {self.session_dir}")
        print("\n🎮 Hotkeys:")
        print("   F5 = Stop recording")
        print("   F6 = Manual save (mark as good)")
        print("   F7 = Discard buffer (mark as bad)")
        print("   F8 = Snapshot & Label (New Object)")
        print("   F9 = Continuous Collect (Orbit/Train)")
        print("\n🎯 Auto-triggers:")
        print("   Kill detected → Save last 10s")
        print("   Death detected → Discard last 30s")
        print("-"*60)
        
    def stop(self):
        """Stop recording"""
        print("\n⏹️ Stopping recorder...")
        
        self.running = False
        self.is_recording = False
        
        if self.record_thread:
            self.record_thread.join(timeout=2.0)
            
        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            
        # Save remaining buffer if anything good
        if self.buffer and len(self.buffer) > 30:
            self._save_sequence("SESSION_END", len(self.buffer))
            
        self._print_stats()
        
    def _record_loop(self):
        """Main recording loop"""
        frame_interval = 1.0 / self.fps_target
        prev_health = 1.0
        
        while self.running:
            start = time.time()
            
            try:
                # 1. Capture frame
                frame = self.screen.capture()
                
                # 1b. Handle Continuous Collection (F9)
                if self.collect_mode:
                    now_collect = time.time()
                    if now_collect - self.last_collect_time >= 0.2: # 5 FPS
                        self.last_collect_time = now_collect
                        
                        # Save frame
                        ts = int(now_collect * 1000)
                        label = self.collect_label
                        # Create label dir
                        label_dir = self.collect_dir / label
                        label_dir.mkdir(exist_ok=True)
                        
                        fname = f"{ts}.jpg"
                        import cv2
                        cv2.imwrite(str(label_dir / fname), frame)
                        print(f"\r📸 Collecting [{label}]: {fname}", end="")

                # 2. Run detection
                detections = self.detector.detect_frame(frame)
                
                # 3. Get current input state
                mouse_x, mouse_y = self.current_mouse
                mouse_dx = mouse_x - self.prev_mouse[0]
                mouse_dy = mouse_y - self.prev_mouse[1]
                self.prev_mouse = self.current_mouse
                
                clicks = self.pending_clicks.copy()
                self.pending_clicks.clear()
                
                scroll = self.scroll_delta
                self.scroll_delta = 0
                
                keys = list(self.current_keys)
                
                # 4. Detect context
                has_click = len(clicks) > 0
                context_state = self.context.detect(mouse_x, mouse_y, detections, has_click)
                
                # 5. Update health estimate
                health = self.filters['health'].update(detections)
                
                # 6. Create buffer frame
                buffer_frame = BufferFrame(
                    timestamp=time.time(),
                    frame=None,  # Don't store full frames (memory)
                    detections=detections,
                    mouse_pos=(mouse_x, mouse_y),
                    mouse_velocity=np.sqrt(mouse_dx**2 + mouse_dy**2),
                    keys_pressed=keys,
                    clicks=clicks,
                    mode=context_state.mode,
                    health_estimate=health
                )
                
                # 7. Add to buffer
                self.buffer.append(buffer_frame)
                if len(self.buffer) > self.max_buffer_frames:
                    self.buffer.pop(0)

                # 7b. Periodic screenshot capture for VLM
                current_time = time.time()
                if current_time - self.last_screenshot_time >= self.screenshot_interval:
                    self._capture_screenshot_with_context(frame, detections, context_state, mouse_x, mouse_y)
                    self.last_screenshot_time = current_time
                    
                # 8. Check for kill
                events = self.filters['kill_filter'].update(detections)
                if events['kill']:
                    print("\n🎯 KILL DETECTED! Saving last 10 seconds...")
                    self.total_kills += 1
                    self._save_sequence("KILL", int(self.fps_target * 10))
                    
                # 9. Check for death
                if self.filters['kill_filter'].detect_death(health, prev_health):
                    print("\n💀 DEATH DETECTED! Discarding last 30 seconds...")
                    self.total_deaths += 1
                    self.buffer.clear()
                    self.filters['health'].reset()
                    
                prev_health = health
                
                # 10. Print status
                self._print_status(context_state, health, detections)
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
                
            # Maintain FPS
            elapsed = time.time() - start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def _save_sequence(self, label: str, num_frames: int):
        """Save a sequence to disk"""
        if not self.buffer:
            return
            
        # Get last N frames
        frames_to_save = self.buffer[-num_frames:] if num_frames < len(self.buffer) else self.buffer.copy()
        
        if len(frames_to_save) < 10:
            print("   (Too short, skipping)")
            return
            
        # Apply smoothing
        frames_to_save = self.filters['smoother'].smooth_buffer(frames_to_save)
        
        # Build state sequences
        states = []
        actions = []
        
        for frame in frames_to_save:
            # Build state vector
            mouse_x_norm = frame.mouse_pos[0] / self.screen_width
            mouse_y_norm = frame.mouse_pos[1] / self.screen_height

            player = PlayerState(
                health=frame.health_estimate,
                mouse_x=mouse_x_norm,
                mouse_y=mouse_y_norm,
                velocity_x=frame.mouse_velocity,
                velocity_y=0,
                mode=frame.mode
            )
            state = self.state_builder.build(frame.detections, player)
            states.append(state.tolist())

            # Find what the player is targeting (nearest object to cursor)
            target_info = self._find_cursor_target(frame.detections, mouse_x_norm, mouse_y_norm)
            clicked = 1.0 if frame.clicks else 0.0

            # Extract keyboard actions (attack, rockets, special)
            keys = frame.keys_pressed
            ctrl_pressed = 1.0 if any(k in ['ctrl', 'ctrl_l', 'ctrl_r'] for k in keys) else 0.0
            space_pressed = 1.0 if 'space' in keys else 0.0
            shift_pressed = 1.0 if any(k in ['shift', 'shift_l', 'shift_r'] for k in keys) else 0.0

            # Build SMART action vector that understands targets AND keyboard actions
            # If clicking AND near a target, record the TARGET position, not just cursor
            if clicked > 0 and target_info:
                # Player clicked on something - record the target's position
                action = [
                    target_info['x'],  # Target X (what they aimed at)
                    target_info['y'],  # Target Y
                    clicked,
                    target_info['is_enemy'],  # 1.0 if enemy, 0.0 if box
                    target_info['distance'],  # How close cursor was to target
                    ctrl_pressed,             # Attack key (Ctrl)
                    space_pressed,            # Rocket key (Space)
                    shift_pressed             # Special key (Shift)
                ]
            else:
                # No click or no target - record cursor position
                action = [
                    mouse_x_norm,
                    mouse_y_norm,
                    clicked,
                    0.0,  # No target type
                    1.0,  # Max distance (no target)
                    ctrl_pressed,
                    space_pressed,
                    shift_pressed
                ]
            actions.append(action)
            
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
            'states': states,
            'actions': actions
        }

        # Save screenshots for VLM annotation (if any in buffer)
        screenshots_saved = 0
        if self.screenshot_buffer:
            screenshots_dir = self.session_dir / "screenshots"
            screenshots_dir.mkdir(exist_ok=True)

            screenshot_refs = []
            for i, ss_data in enumerate(self.screenshot_buffer):
                ss_filename = f"seq{seq_id:04d}_frame{i:02d}.jpg"
                ss_path = screenshots_dir / ss_filename
                ss_data['image'].save(ss_path, quality=85)

                # Save context JSON alongside
                context_filename = f"seq{seq_id:04d}_frame{i:02d}_context.json"
                context_path = screenshots_dir / context_filename
                with open(context_path, 'w') as cf:
                    json.dump(ss_data['context'], cf)

                screenshot_refs.append({
                    'image': ss_filename,
                    'context': context_filename,
                    'timestamp': ss_data['timestamp']
                })
                screenshots_saved += 1

            data['screenshots'] = screenshot_refs
            self.screenshot_buffer.clear()  # Clear after saving

        with open(filepath, 'w') as f:
            json.dump(data, f)

        self.sequences_saved += 1
        print(f"   ✅ Saved: {filename} ({len(frames_to_save)} frames, {screenshots_saved} screenshots)")

    def _resize_maintain_aspect(self, img, target_width: int, target_height: int):
        """
        Resize image to fit within target dimensions while maintaining aspect ratio.
        Adds black letterboxing/pillarboxing if needed to fill the exact target size.

        This prevents distortion of game screenshots (e.g., 16:9 squeezed to 4:3).
        """
        from PIL import Image as PILImage

        orig_width, orig_height = img.size
        orig_aspect = orig_width / orig_height
        target_aspect = target_width / target_height

        if orig_aspect > target_aspect:
            # Image is wider - fit to width, letterbox top/bottom
            new_width = target_width
            new_height = int(target_width / orig_aspect)
        else:
            # Image is taller - fit to height, pillarbox left/right
            new_height = target_height
            new_width = int(target_height * orig_aspect)

        # Resize maintaining aspect ratio
        img_resized = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)

        # Create black background at exact target size
        result = PILImage.new('RGB', (target_width, target_height), (0, 0, 0))

        # Center the resized image
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        result.paste(img_resized, (paste_x, paste_y))

        return result

    def _capture_screenshot_with_context(self, frame, detections, context_state, mouse_x, mouse_y):
        """
        Capture a screenshot with rich context for VLM annotation.
        This captures SHORT-TERM context (recent actions) to include with the image.
        """
        try:
            from PIL import Image

            # Convert numpy frame to PIL and resize for storage efficiency
            # Maintain 16:9 aspect ratio to avoid distortion
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            img = Image.fromarray(frame)
            # Resize maintaining aspect ratio (target 640x360 for 16:9)
            img = self._resize_maintain_aspect(img, 640, 360)

            # Build context from recent buffer frames (short-term memory)
            recent_actions = []
            recent_frames = self.buffer[-30:] if len(self.buffer) >= 30 else self.buffer  # Last ~1 second
            for bf in recent_frames[-10:]:  # Last 10 frames context
                recent_actions.append({
                    'mouse': bf.mouse_pos,
                    'clicked': len(bf.clicks) > 0,
                    'mode': bf.mode
                })

            # Count objects in scene
            enemies = [d for d in detections if d.class_name in ['Devo', 'Lordakia', 'Mordon', 'Saimon', 'Sibelon', 'Struener']]
            boxes = [d for d in detections if d.class_name == 'BonusBox']

            screenshot_data = {
                'timestamp': time.time(),
                'image': img,
                'context': {
                    'mode': context_state.mode,
                    'mouse_pos': (mouse_x, mouse_y),
                    'num_enemies': len(enemies),
                    'num_boxes': len(boxes),
                    'enemy_positions': [(d.x_center, d.y_center) for d in enemies[:5]],  # Top 5
                    'box_positions': [(d.x_center, d.y_center) for d in boxes[:5]],
                    'recent_clicks': sum(1 for a in recent_actions if a['clicked']),
                    'recent_actions': recent_actions[-5:]  # Last 5 actions
                }
            }

            self.screenshot_buffer.append(screenshot_data)
            if len(self.screenshot_buffer) > self.max_screenshot_buffer:
                self.screenshot_buffer.pop(0)

        except Exception as e:
            pass  # Don't crash recording for screenshot errors

    def _find_cursor_target(self, detections, mouse_x: float, mouse_y: float, threshold: float = 0.08):
        """
        Find what object the cursor is near/on.
        This helps the AI understand WHAT you're clicking on, not just WHERE.

        Args:
            detections: List of detected objects
            mouse_x, mouse_y: Normalized cursor position (0-1)
            threshold: Max distance to consider "targeting" (8% of screen)

        Returns:
            dict with target info or None if no target nearby
        """
        best_target = None
        best_distance = threshold

        for det in detections:
            # Calculate distance from cursor to detection center
            dist = np.sqrt((det.x_center - mouse_x)**2 + (det.y_center - mouse_y)**2)

            if dist < best_distance:
                is_enemy = det.class_name in ['Devo', 'Lordakia', 'Mordon', 'Saimon', 'Sibelon', 'Struener']
                is_box = det.class_name == 'BonusBox'

                if is_enemy or is_box:
                    best_distance = dist
                    best_target = {
                        'x': det.x_center,
                        'y': det.y_center,
                        'class': det.class_name,
                        'is_enemy': 1.0 if is_enemy else 0.0,
                        'distance': dist
                    }

        return best_target

    def _print_status(self, context, health, detections):
        """Print current status"""
        enemies = sum(1 for d in detections if d.class_name in KillFilter.ENEMY_CLASSES)
        boxes = sum(1 for d in detections if d.class_name == 'BonusBox')
        
        status = f"\r🔴 REC | Buffer: {len(self.buffer):4d} | "
        status += f"Mode: {context.mode:10} | "
        status += f"Health: {health:3.0%} | "
        status += f"Enemies: {enemies} | "
        status += f"Saved: {self.sequences_saved}    "
        
        print(status, end="")
        
    def _print_stats(self):
        """Print session stats"""
        print("\n" + "="*60)
        print("  📊 SESSION STATS")
        print("="*60)
        print(f"   Kills detected: {self.total_kills}")
        print(f"   Deaths detected: {self.total_deaths}")
        print(f"   Sequences saved: {self.sequences_saved}")
        print(f"   Output: {self.session_dir}")
        print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Filtered Recorder')
    parser.add_argument('--model', type=str, default='F:/dev/bot/best.pt',
                       help='Path to YOLO model')
    parser.add_argument('--monitor', type=int, default=1,
                       help='Monitor to capture')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Recording FPS')
    
    args = parser.parse_args()
    
    recorder = FilteredRecorder(
        model_path=args.model,
        monitor=args.monitor,
        fps_target=args.fps
    )
    
    print("\n🎮 Filtered Recorder Ready")
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
            time.sleep(1)
    except KeyboardInterrupt:
        if recorder.is_recording:
            recorder.stop()
        else:
            print("\n👋 Exiting...")
            if recorder.keyboard_listener:
                recorder.keyboard_listener.stop()
            if recorder.mouse_listener:
                recorder.mouse_listener.stop()


if __name__ == "__main__":
    main()
