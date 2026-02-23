"""
DarkOrbit Bot - Input Logger
Records YOUR mouse movements and keyboard inputs with normalized coordinates.

Usage:
    python input_logger.py

Controls:
    - Press ESC to stop recording
    - Data saves automatically to recordings/ folder
"""

import json
import time
import ctypes
import os
from datetime import datetime
from pynput import mouse, keyboard


class InputLogger:
    def __init__(self):
        # Get screen resolution for normalization
        user32 = ctypes.windll.user32
        self.screen_width = user32.GetSystemMetrics(0)
        self.screen_height = user32.GetSystemMetrics(1)
        
        self.data = []
        self.start_time = None
        self.is_running = False
        
        # Metadata
        self.metadata = {
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "recorded_at": None,
            "duration_seconds": 0,
            "total_events": 0
        }
        
        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
    
    def _get_timestamp(self):
        """Get time elapsed since recording started"""
        return time.time() - self.start_time
    
    def _on_move(self, x, y):
        """Record mouse movement with normalized coordinates"""
        if not self.is_running:
            return
            
        self.data.append({
            "type": "move",
            "time": self._get_timestamp(),
            "x_rel": x / self.screen_width,  # 0.0 to 1.0
            "y_rel": y / self.screen_height, # 0.0 to 1.0
            "x_abs": x,
            "y_abs": y
        })
    
    def _on_click(self, x, y, button, pressed):
        """Record mouse clicks"""
        if not self.is_running:
            return
            
        self.data.append({
            "type": "click",
            "time": self._get_timestamp(),
            "x_rel": x / self.screen_width,
            "y_rel": y / self.screen_height,
            "x_abs": x,
            "y_abs": y,
            "button": str(button).replace("Button.", ""),
            "pressed": pressed  # True = down, False = up
        })
    
    def _on_scroll(self, x, y, dx, dy):
        """Record scroll events"""
        if not self.is_running:
            return
            
        self.data.append({
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
        
        # Stop recording on ESC
        if key == keyboard.Key.esc:
            self.is_running = False
            return False  # Stop listener
        
        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key).replace("Key.", "")
        
        self.data.append({
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
        
        self.data.append({
            "type": "key",
            "time": self._get_timestamp(),
            "key": key_name,
            "pressed": False
        })
    
    def start_recording(self):
        """Start recording inputs"""
        self.data = []
        self.start_time = time.time()
        self.is_running = True
        self.metadata["recorded_at"] = datetime.now().isoformat()
        
        print("\n" + "="*50)
        print("  RECORDING STARTED")
        print("  Press ESC to stop recording")
        print("="*50 + "\n")
        
        # Start listeners
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
        
        # Stop mouse listener
        mouse_listener.stop()
        
        # Update metadata
        self.metadata["duration_seconds"] = round(time.time() - self.start_time, 2)
        self.metadata["total_events"] = len(self.data)
        
        print("\n" + "="*50)
        print(f"  RECORDING STOPPED")
        print(f"  Duration: {self.metadata['duration_seconds']:.1f} seconds")
        print(f"  Events captured: {self.metadata['total_events']}")
        print("="*50)
    
    def save(self, filename=None):
        """Save recording to JSON file"""
        # Create recordings directory
        recordings_dir = os.path.join(os.path.dirname(__file__), "..", "data", "recordings")
        os.makedirs(recordings_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.json"
        
        filepath = os.path.join(recordings_dir, filename)
        
        # Save data
        output = {
            "metadata": self.metadata,
            "events": self.data
        }
        
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\nSaved to: {filepath}")
        print(f"File size: {os.path.getsize(filepath) / 1024:.1f} KB")
        
        return filepath


def main():
    print("\n" + "="*50)
    print("  DARKORBIT BOT - INPUT LOGGER")
    print("="*50)
    print("\nThis will record your mouse movements and keyboard inputs.")
    print("Use this while playing the game to capture YOUR movement style.")
    print("\nTips:")
    print("  - Record 10-30 minutes of normal gameplay")
    print("  - Play naturally, don't think about it")
    print("  - Include: flying, collecting boxes, combat")
    print("")
    
    input("Press ENTER to start recording...")
    
    logger = InputLogger()
    logger.start_recording()
    logger.save()
    
    print("\nRecording complete! Run this again to record more sessions.")


if __name__ == "__main__":
    main()
