"""
DarkOrbit Bot - Synced Recorder
Records screen AND input simultaneously with perfect sync.

Usage:
    python synced_recorder.py

Controls:
    - Press ESC to stop recording
    - All data saves to data/recordings/session_TIMESTAMP/
"""

import json
import time
import ctypes
import os
import threading
from datetime import datetime
from queue import Queue
from pynput import mouse, keyboard
import mss
import mss.tools


class SyncedRecorder:
    def __init__(self, screenshot_interval=0.5, capture_on_click=True):
        """
        Args:
            screenshot_interval: Seconds between automatic screenshots (0 = disabled)
            capture_on_click: Take screenshot on every click
        """
        # Get screen resolution
        user32 = ctypes.windll.user32
        self.screen_width = user32.GetSystemMetrics(0)
        self.screen_height = user32.GetSystemMetrics(1)
        
        # Settings
        self.screenshot_interval = screenshot_interval
        self.capture_on_click = capture_on_click
        
        # State
        self.input_data = []
        self.screenshot_queue = Queue()
        self.start_time = None
        self.is_running = False
        self.session_dir = None
        self.screenshot_count = 0
        
        # Metadata
        self.metadata = {
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "screenshot_interval": screenshot_interval,
            "capture_on_click": capture_on_click,
            "recorded_at": None,
            "duration_seconds": 0,
            "total_input_events": 0,
            "total_screenshots": 0
        }
        
        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
    
    def _get_timestamp(self):
        """Get time elapsed since recording started"""
        return time.time() - self.start_time
    
    def _create_session_dir(self):
        """Create directory for this recording session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "recordings")
        self.session_dir = os.path.join(base_dir, f"session_{timestamp}")
        
        # Create subdirectories
        os.makedirs(os.path.join(self.session_dir, "screenshots"), exist_ok=True)
        
        print(f"Session directory: {self.session_dir}")
    
    # ==================== INPUT HANDLERS ====================
    
    def _on_move(self, x, y):
        """Record mouse movement"""
        if not self.is_running:
            return
        
        self.input_data.append({
            "type": "move",
            "time": self._get_timestamp(),
            "x_rel": x / self.screen_width,
            "y_rel": y / self.screen_height,
            "x_abs": x,
            "y_abs": y
        })
    
    def _on_click(self, x, y, button, pressed):
        """Record mouse click and optionally capture screenshot"""
        if not self.is_running:
            return
        
        timestamp = self._get_timestamp()
        
        self.input_data.append({
            "type": "click",
            "time": timestamp,
            "x_rel": x / self.screen_width,
            "y_rel": y / self.screen_height,
            "x_abs": x,
            "y_abs": y,
            "button": str(button).replace("Button.", ""),
            "pressed": pressed
        })
        
        # Capture screenshot on click
        if self.capture_on_click and pressed:
            self.screenshot_queue.put(("click", timestamp, x, y))
    
    def _on_scroll(self, x, y, dx, dy):
        """Record scroll events"""
        if not self.is_running:
            return
        
        self.input_data.append({
            "type": "scroll",
            "time": self._get_timestamp(),
            "x_rel": x / self.screen_width,
            "y_rel": y / self.screen_height,
            "dx": dx,
            "dy": dy
        })
    
    def _on_key_press(self, key):
        """Record key press"""
        if not self.is_running:
            return
        
        # Stop on ESC
        if key == keyboard.Key.esc:
            self.is_running = False
            return False
        
        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key).replace("Key.", "")
        
        self.input_data.append({
            "type": "key",
            "time": self._get_timestamp(),
            "key": key_name,
            "pressed": True
        })
    
    def _on_key_release(self, key):
        """Record key release"""
        if not self.is_running:
            return
        
        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key).replace("Key.", "")
        
        self.input_data.append({
            "type": "key",
            "time": self._get_timestamp(),
            "key": key_name,
            "pressed": False
        })
    
    # ==================== SCREENSHOT CAPTURE ====================
    
    def _screenshot_worker(self):
        """Background thread that captures screenshots"""
        screenshots_dir = os.path.join(self.session_dir, "screenshots")
        screenshot_index = []
        
        with mss.mss() as sct:
            # Capture full screen
            monitor = sct.monitors[0]  # Full screen (all monitors)
            
            while self.is_running:
                try:
                    # Check for queued screenshots (from clicks)
                    while not self.screenshot_queue.empty():
                        trigger, timestamp, x, y = self.screenshot_queue.get_nowait()
                        
                        # Capture
                        img = sct.grab(monitor)
                        
                        # Save
                        filename = f"frame_{self.screenshot_count:06d}.png"
                        filepath = os.path.join(screenshots_dir, filename)
                        mss.tools.to_png(img.rgb, img.size, output=filepath)
                        
                        # Index
                        screenshot_index.append({
                            "filename": filename,
                            "time": timestamp,
                            "trigger": trigger,
                            "click_x": x,
                            "click_y": y
                        })
                        
                        self.screenshot_count += 1
                    
                    # Periodic screenshot
                    if self.screenshot_interval > 0:
                        timestamp = self._get_timestamp()
                        
                        img = sct.grab(monitor)
                        
                        filename = f"frame_{self.screenshot_count:06d}.png"
                        filepath = os.path.join(screenshots_dir, filename)
                        mss.tools.to_png(img.rgb, img.size, output=filepath)
                        
                        screenshot_index.append({
                            "filename": filename,
                            "time": timestamp,
                            "trigger": "interval"
                        })
                        
                        self.screenshot_count += 1
                        
                        time.sleep(self.screenshot_interval)
                    else:
                        time.sleep(0.01)  # Small sleep if no interval capture
                        
                except Exception as e:
                    print(f"Screenshot error: {e}")
                    time.sleep(0.1)
        
        # Save screenshot index
        index_path = os.path.join(self.session_dir, "screenshot_index.json")
        with open(index_path, 'w') as f:
            json.dump(screenshot_index, f, indent=2)
        
        print(f"Captured {self.screenshot_count} screenshots")
    
    # ==================== MAIN RECORDING ====================
    
    def start_recording(self):
        """Start synchronized recording"""
        self._create_session_dir()
        
        self.input_data = []
        self.start_time = time.time()
        self.is_running = True
        self.screenshot_count = 0
        self.metadata["recorded_at"] = datetime.now().isoformat()
        
        print("\n" + "="*60)
        print("  🔴 SYNCED RECORDING STARTED")
        print("="*60)
        print(f"  📁 Saving to: {self.session_dir}")
        print(f"  📸 Screenshots every {self.screenshot_interval}s + on clicks")
        print(f"  ⏹️  Press ESC to stop")
        print("="*60 + "\n")
        
        # Start screenshot thread
        screenshot_thread = threading.Thread(target=self._screenshot_worker, daemon=True)
        screenshot_thread.start()
        
        # Start input listeners
        mouse_listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll
        )
        keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        
        mouse_listener.start()
        keyboard_listener.start()
        
        # Wait for ESC
        keyboard_listener.join()
        
        # Cleanup
        mouse_listener.stop()
        self.is_running = False
        screenshot_thread.join(timeout=2)
        
        # Update metadata
        self.metadata["duration_seconds"] = round(time.time() - self.start_time, 2)
        self.metadata["total_input_events"] = len(self.input_data)
        self.metadata["total_screenshots"] = self.screenshot_count
        
        print("\n" + "="*60)
        print("  ⏹️  RECORDING STOPPED")
        print("="*60)
        print(f"  ⏱️  Duration: {self.metadata['duration_seconds']:.1f} seconds")
        print(f"  🖱️  Input events: {self.metadata['total_input_events']}")
        print(f"  📸 Screenshots: {self.metadata['total_screenshots']}")
        print("="*60)
    
    def save(self):
        """Save all recording data"""
        # Save input data
        input_path = os.path.join(self.session_dir, "input_data.json")
        output = {
            "metadata": self.metadata,
            "events": self.input_data
        }
        with open(input_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Save metadata separately for quick access
        meta_path = os.path.join(self.session_dir, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"\n✅ Session saved to: {self.session_dir}")
        print(f"   - input_data.json ({len(self.input_data)} events)")
        print(f"   - screenshot_index.json")
        print(f"   - screenshots/ ({self.screenshot_count} images)")
        
        return self.session_dir


def main():
    print("\n" + "="*60)
    print("  DARKORBIT BOT - SYNCED RECORDER")
    print("="*60)
    print("""
This records BOTH your screen AND inputs perfectly synced!

What gets captured:
  🖱️  Mouse movements (position, timing)
  🖱️  Mouse clicks (with screenshot on each click)
  ⌨️  Keyboard inputs
  📸 Screenshots every 0.5 seconds

Tips:
  - Record 10-30 minutes of normal gameplay
  - Play naturally - fly around, collect boxes, fight NPCs
  - Screenshots are for YOLO training later
""")
    
    # Ask for screenshot interval
    print("Screenshot interval options:")
    print("  1. Every 0.5 seconds (recommended, ~120 screenshots/min)")
    print("  2. Every 1.0 second (~60 screenshots/min)")
    print("  3. Only on clicks (fewer screenshots)")
    
    choice = input("\nChoice [1/2/3, default=1]: ").strip()
    
    if choice == "2":
        interval = 1.0
        on_click = True
    elif choice == "3":
        interval = 0
        on_click = True
    else:
        interval = 0.5
        on_click = True
    
    input("\nPress ENTER to start recording...")
    
    recorder = SyncedRecorder(
        screenshot_interval=interval,
        capture_on_click=on_click
    )
    recorder.start_recording()
    recorder.save()
    
    print("\n🎉 Recording complete!")
    print("Next step: Run 'python analysis/analyze_patterns.py' to analyze your movement style")


if __name__ == "__main__":
    main()
