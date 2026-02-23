"""
DarkOrbit Bot - Game Window Recorder
Records screen AND input synchronized, focused on the GAME WINDOW.

Features:
- Detects game window automatically
- Coordinates relative to game window (works if you move/resize it)
- Captures all actions: clicks, keys, HUD interactions, map clicks

Usage:
    python game_recorder.py

Controls:
    - Press ESC to stop recording
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

# Windows API for window detection
try:
    import win32gui
    import win32process
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False
    print("Note: pywin32 not installed - will use full screen capture")


class GameWindowRecorder:
    def __init__(self, game_window_title=None, screenshot_interval=0.5, capture_on_click=True):
        """
        Args:
            game_window_title: Part of window title to find (e.g., "DarkOrbit", "Space")
            screenshot_interval: Seconds between automatic screenshots
            capture_on_click: Take screenshot on every click
        """
        self.game_window_title = game_window_title
        self.screenshot_interval = screenshot_interval
        self.capture_on_click = capture_on_click
        
        # Screen info
        user32 = ctypes.windll.user32
        self.screen_width = user32.GetSystemMetrics(0)
        self.screen_height = user32.GetSystemMetrics(1)
        
        # Game window info (will be detected)
        self.game_window = None
        self.game_rect = None  # (left, top, width, height)
        
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
            "game_window_title": None,
            "game_rect": None,
            "screenshot_interval": screenshot_interval,
            "recorded_at": None,
            "duration_seconds": 0,
            "total_input_events": 0,
            "total_screenshots": 0
        }
    
    def find_game_window(self, title_contains=None):
        """Find the game window by title"""
        if not HAS_WIN32:
            print("⚠️  pywin32 not available - using full screen")
            return None
        
        found_windows = []
        
        def enum_callback(hwnd, results):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    results.append((hwnd, title))
        
        win32gui.EnumWindows(enum_callback, found_windows)
        
        if title_contains:
            # Find window matching the search
            for hwnd, title in found_windows:
                if title_contains.lower() in title.lower():
                    return hwnd, title
            return None, None
        else:
            # Show all windows for user to choose
            print("\nDetected windows:")
            game_candidates = []
            for i, (hwnd, title) in enumerate(found_windows):
                # Filter out system windows
                if len(title) > 3 and not title.startswith("_"):
                    game_candidates.append((hwnd, title))
                    print(f"  {len(game_candidates)}. {title[:60]}")
            
            return game_candidates
    
    def select_game_window(self):
        """Let user select the game window"""
        if not HAS_WIN32:
            print("\n⚠️  Using FULL SCREEN capture")
            print("   Install pywin32 for window-specific capture: uv pip install pywin32")
            self.game_rect = {
                "left": 0,
                "top": 0, 
                "width": self.screen_width,
                "height": self.screen_height
            }
            return True
        
        # Try to find by known game names
        known_games = ["Space Aces | Game", "Space Aces", "DarkOrbit", "darkorbit", "Bigpoint"]
        
        for game_name in known_games:
            hwnd, title = self.find_game_window(game_name)
            if hwnd:
                print(f"\n✅ Found game window: {title}")
                use_it = input("Use this window? [Y/n]: ").strip().lower()
                if use_it != 'n':
                    self.game_window = hwnd
                    self._update_game_rect()
                    return True
        
        # Manual selection
        print("\n🔍 Searching for game window...")
        candidates = self.find_game_window()
        
        if not candidates:
            print("No windows found!")
            return False
        
        choice = input("\nEnter window number (or press ENTER for full screen): ").strip()
        
        if choice == "":
            self.game_rect = {
                "left": 0,
                "top": 0,
                "width": self.screen_width,
                "height": self.screen_height
            }
            print("Using full screen capture")
        else:
            try:
                idx = int(choice) - 1
                self.game_window, title = candidates[idx]
                self._update_game_rect()
                print(f"Selected: {title}")
            except (ValueError, IndexError):
                print("Invalid choice, using full screen")
                self.game_rect = {
                    "left": 0,
                    "top": 0,
                    "width": self.screen_width,
                    "height": self.screen_height
                }
        
        return True
    
    def _update_game_rect(self):
        """Update game window position/size"""
        if self.game_window and HAS_WIN32:
            try:
                rect = win32gui.GetWindowRect(self.game_window)
                self.game_rect = {
                    "left": rect[0],
                    "top": rect[1],
                    "width": rect[2] - rect[0],
                    "height": rect[3] - rect[1]
                }
                self.metadata["game_rect"] = self.game_rect
                self.metadata["game_window_title"] = win32gui.GetWindowText(self.game_window)
            except:
                pass
    
    def _is_in_game_window(self, x, y):
        """Check if coordinates are inside game window"""
        if not self.game_rect:
            return True
        
        return (self.game_rect["left"] <= x <= self.game_rect["left"] + self.game_rect["width"] and
                self.game_rect["top"] <= y <= self.game_rect["top"] + self.game_rect["height"])
    
    def _normalize_to_game(self, x, y):
        """Convert screen coordinates to game window coordinates (0-1)"""
        if not self.game_rect:
            return x / self.screen_width, y / self.screen_height
        
        game_x = (x - self.game_rect["left"]) / self.game_rect["width"]
        game_y = (y - self.game_rect["top"]) / self.game_rect["height"]
        
        # Clamp to 0-1
        game_x = max(0, min(1, game_x))
        game_y = max(0, min(1, game_y))
        
        return game_x, game_y
    
    def _get_timestamp(self):
        return time.time() - self.start_time
    
    def _create_session_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "recordings")
        self.session_dir = os.path.join(base_dir, f"session_{timestamp}")
        os.makedirs(os.path.join(self.session_dir, "screenshots"), exist_ok=True)
    
    # ==================== INPUT HANDLERS ====================
    
    def _on_move(self, x, y):
        if not self.is_running:
            return
        
        # Update game window position periodically
        if len(self.input_data) % 1000 == 0:
            self._update_game_rect()
        
        game_x, game_y = self._normalize_to_game(x, y)
        in_game = self._is_in_game_window(x, y)
        
        self.input_data.append({
            "type": "move",
            "time": self._get_timestamp(),
            "x_game": game_x,  # Relative to game window (0-1)
            "y_game": game_y,
            "x_screen": x,     # Absolute screen position
            "y_screen": y,
            "in_game": in_game
        })
    
    def _on_click(self, x, y, button, pressed):
        if not self.is_running:
            return
        
        self._update_game_rect()
        
        timestamp = self._get_timestamp()
        game_x, game_y = self._normalize_to_game(x, y)
        in_game = self._is_in_game_window(x, y)
        
        self.input_data.append({
            "type": "click",
            "time": timestamp,
            "x_game": game_x,
            "y_game": game_y,
            "x_screen": x,
            "y_screen": y,
            "button": str(button).replace("Button.", ""),
            "pressed": pressed,
            "in_game": in_game
        })
        
        # Screenshot on click (if in game window)
        if self.capture_on_click and pressed and in_game:
            self.screenshot_queue.put(("click", timestamp, game_x, game_y))
    
    def _on_scroll(self, x, y, dx, dy):
        if not self.is_running:
            return
        
        game_x, game_y = self._normalize_to_game(x, y)
        
        self.input_data.append({
            "type": "scroll",
            "time": self._get_timestamp(),
            "x_game": game_x,
            "y_game": game_y,
            "dx": dx,
            "dy": dy
        })
    
    def _on_key_press(self, key):
        if not self.is_running:
            return
        
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
        screenshots_dir = os.path.join(self.session_dir, "screenshots")
        screenshot_index = []
        last_interval_capture = 0
        
        with mss.mss() as sct:
            while self.is_running:
                try:
                    # Update game rect for capture
                    self._update_game_rect()
                    
                    # Define capture region
                    if self.game_rect:
                        monitor = {
                            "left": self.game_rect["left"],
                            "top": self.game_rect["top"],
                            "width": self.game_rect["width"],
                            "height": self.game_rect["height"]
                        }
                    else:
                        monitor = sct.monitors[0]
                    
                    # Handle click-triggered screenshots
                    while not self.screenshot_queue.empty():
                        trigger, timestamp, gx, gy = self.screenshot_queue.get_nowait()
                        
                        img = sct.grab(monitor)
                        filename = f"frame_{self.screenshot_count:06d}.png"
                        filepath = os.path.join(screenshots_dir, filename)
                        mss.tools.to_png(img.rgb, img.size, output=filepath)
                        
                        screenshot_index.append({
                            "filename": filename,
                            "time": timestamp,
                            "trigger": trigger,
                            "click_x": gx,
                            "click_y": gy,
                            "game_rect": self.game_rect.copy() if self.game_rect else None
                        })
                        self.screenshot_count += 1
                    
                    # Periodic screenshots
                    current_time = self._get_timestamp()
                    if self.screenshot_interval > 0 and current_time - last_interval_capture >= self.screenshot_interval:
                        img = sct.grab(monitor)
                        filename = f"frame_{self.screenshot_count:06d}.png"
                        filepath = os.path.join(screenshots_dir, filename)
                        mss.tools.to_png(img.rgb, img.size, output=filepath)
                        
                        screenshot_index.append({
                            "filename": filename,
                            "time": current_time,
                            "trigger": "interval",
                            "game_rect": self.game_rect.copy() if self.game_rect else None
                        })
                        self.screenshot_count += 1
                        last_interval_capture = current_time
                    
                    time.sleep(0.05)
                    
                except Exception as e:
                    time.sleep(0.1)
        
        # Save index
        index_path = os.path.join(self.session_dir, "screenshot_index.json")
        with open(index_path, 'w') as f:
            json.dump(screenshot_index, f, indent=2)
    
    # ==================== MAIN ====================
    
    def start_recording(self):
        self._create_session_dir()
        
        self.input_data = []
        self.start_time = time.time()
        self.is_running = True
        self.screenshot_count = 0
        self.metadata["recorded_at"] = datetime.now().isoformat()
        
        print("\n" + "="*60)
        print("  🔴 RECORDING STARTED")
        print("="*60)
        
        if self.game_rect:
            print(f"  🎮 Game window: {self.game_rect['width']}x{self.game_rect['height']}")
            print(f"     Position: ({self.game_rect['left']}, {self.game_rect['top']})")
        
        print(f"  📁 Saving to: {os.path.basename(self.session_dir)}")
        print(f"  ⏹️  Press ESC to stop")
        print("="*60 + "\n")
        print("▶️  Recording... Play the game now!\n")
        
        # Start threads
        screenshot_thread = threading.Thread(target=self._screenshot_worker, daemon=True)
        screenshot_thread.start()
        
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
        keyboard_listener.join()
        
        mouse_listener.stop()
        self.is_running = False
        screenshot_thread.join(timeout=2)
        
        self.metadata["duration_seconds"] = round(time.time() - self.start_time, 2)
        self.metadata["total_input_events"] = len(self.input_data)
        self.metadata["total_screenshots"] = self.screenshot_count
        
        # Count in-game events
        in_game_events = sum(1 for e in self.input_data if e.get("in_game", True))
        
        print("\n" + "="*60)
        print("  ⏹️  RECORDING STOPPED")
        print("="*60)
        print(f"  ⏱️  Duration: {self.metadata['duration_seconds']:.1f}s")
        print(f"  🖱️  Total events: {self.metadata['total_input_events']}")
        print(f"  🎮 In-game events: {in_game_events}")
        print(f"  📸 Screenshots: {self.screenshot_count}")
        print("="*60)
    
    def save(self):
        input_path = os.path.join(self.session_dir, "input_data.json")
        with open(input_path, 'w') as f:
            json.dump({"metadata": self.metadata, "events": self.input_data}, f, indent=2)
        
        meta_path = os.path.join(self.session_dir, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"\n✅ Saved to: {self.session_dir}")
        return self.session_dir


def main():
    print("\n" + "="*60)
    print("  DARKORBIT BOT - GAME WINDOW RECORDER")
    print("="*60)
    print("""
This records your gameplay with GAME WINDOW detection!

What gets captured:
  🖱️  Mouse movements (relative to game window)
  🖱️  Clicks on HUD, map, anywhere in game
  ⌨️  Keyboard (C for config, ammo changes, etc.)
  📸 Screenshots of the game window only

The bot will learn:
  - How you move the mouse
  - How you click on things
  - Your keyboard shortcuts
  - Everything synced perfectly!
""")
    
    recorder = GameWindowRecorder(screenshot_interval=0.5)
    
    if not recorder.select_game_window():
        print("Failed to set up recording")
        return
    
    print("\nScreenshot options:")
    print("  1. Every 0.5 seconds (recommended)")
    print("  2. Every 1.0 second")  
    print("  3. Only on clicks")
    
    choice = input("\nChoice [1/2/3, default=1]: ").strip()
    if choice == "2":
        recorder.screenshot_interval = 1.0
    elif choice == "3":
        recorder.screenshot_interval = 0
    
    input("\n🎮 Open the game, then press ENTER to start recording...")
    
    recorder.start_recording()
    recorder.save()
    
    print("\n🎉 Done! Next: python analysis/analyze_patterns.py")


if __name__ == "__main__":
    main()
