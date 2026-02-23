"""
Socket IPC Debug Viewer - Realtime Passive Visualizer

Visualizes the internal state of the DarkOrbit V2 bot by reading the
JSON state stream broadcasted by the main bot process via TCP Socket.

Features:
- REALTIME: Zero disk I/O, uses local TCP loopback
- NON-BLOCKING: Efficient stream reading
- HEATMAP: Overlays bot's vision attention
"""

import cv2
import numpy as np
import time
import json
import os
import sys
import socket
import struct
import threading
from typing import Optional, Dict, List
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .config import ENEMY_CLASSES, LOOT_CLASSES

class SocketDebugViewer:
    """
    Realtime passive debug viewer using TCP Socket.
    """

    def __init__(self, host='127.0.0.1', port=9999, capture_bg=False, monitor=1):
        self.host = host
        self.port = port
        self.capture_bg = capture_bg
        self.monitor = monitor

        self.screen = None
        if capture_bg:
            try:
                from detection.detector import ScreenCapture
                self.screen = ScreenCapture(monitor_index=monitor)
                print(f"[Viewer] Background capture enabled on Monitor {monitor}")
            except Exception as e:
                print(f"[Viewer] Warning: Could not init screen capture: {e}")

        # Visualization settings
        self.viz_window_name = "Bot Debug Viewer (Socket IPC)"
        self.viz_display_scale = 0.6
        self.viz_show_help = True
        self.viz_show_heatmap = True
        
        # Caching
        self.last_heatmap = None
        self.latest_state = None
        
        # Logic
        self.running = True
        self.connected = False
        self.sock = None
        
        # Metrics
        self.fps = 0.0
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # Colors (BGR)
        self.viz_colors = {
            'enemy': (0, 0, 255),      # red
            'loot': (0, 255, 0),       # green
            'ship': (255, 0, 0),       # blue
            'other': (128, 128, 128),  # gray
            'target': (0, 255, 255),   # yellow
            'mouse': (255, 255, 0),    # cyan
            'text': (255, 255, 255),   # white
            'bg': (0, 0, 0),           # black
            'click': (0, 0, 255),      # red
        }

        cv2.namedWindow(self.viz_window_name, cv2.WINDOW_NORMAL)
        
        # Start receiver thread
        self.thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self.thread.start()

    def _receiver_loop(self):
        """Background thread to maintain connection and receive state."""
        print(f"[Viewer] Receiver thread started, connecting to {self.host}:{self.port}...")
        connect_attempts = 0

        while self.running:
            try:
                if not self.connected:
                    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    # TCP_NODELAY disables Nagle's algorithm - critical for low latency!
                    self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self.sock.settimeout(2.0)
                    try:
                        self.sock.connect((self.host, self.port))
                        self.connected = True
                        connect_attempts = 0
                        print("[Viewer] Connected to Bot! (TCP_NODELAY enabled)")
                    except ConnectionRefusedError:
                        connect_attempts += 1
                        if connect_attempts % 5 == 1:  # Print every 5 attempts
                            print(f"[Viewer] Connection refused - is the bot running? (attempt {connect_attempts})")
                        time.sleep(1.0)
                        continue
                    except Exception as e:
                        connect_attempts += 1
                        if connect_attempts % 5 == 1:
                            print(f"[Viewer] Connection error: {e} (attempt {connect_attempts})")
                        time.sleep(1.0)
                        continue

                # Read Header (4 bytes length)
                # We need blocking read here
                try:
                    header = self._recv_exact(4)
                    if not header:
                         raise ConnectionResetError("Closed")
                    
                    length = struct.unpack('>I', header)[0]
                    
                    # Read Data
                    data = self._recv_exact(length)
                    if not data:
                        raise ConnectionResetError("Closed")
                        
                    # Decode
                    json_str = data.decode('utf-8')
                    state = json.loads(json_str)
                    
                    # Update (Atomic prompt swap)
                    self.latest_state = state
                    
                except socket.timeout:
                    continue
                except (ConnectionResetError, BrokenPipeError):
                    print("[Viewer] Disconnected, retrying...")
                    self.connected = False
                    if self.sock: self.sock.close()
                    continue
                except Exception as e:
                    print(f"[Viewer] Receive error: {e}")
                    self.connected = False
                    time.sleep(1.0)
                    
            except Exception as e:
                print(f"[Viewer] Loop error: {e}")
                time.sleep(1.0)

    def _recv_exact(self, n):
        """Helper to receive exactly n bytes."""
        data = b''
        while len(data) < n:
            packet = self.sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def _safe_imshow(self, frame: np.ndarray) -> bool:
        try:
            if cv2.getWindowProperty(self.viz_window_name, cv2.WND_PROP_VISIBLE) < 1:
                self.running = False
                return False
            cv2.imshow(self.viz_window_name, frame)
            return True
        except:
            self.running = False
            return False

    def _visualize_frame(self, frame: np.ndarray, bot_state: Dict) -> None:
        if bot_state is None: return
        
        viz_frame = frame.copy()
        h, w = viz_frame.shape[:2]

        # Extract basic data
        detections = bot_state.get('detections', [])
        tracked_objects = bot_state.get('tracked_objects', [])
        mode = bot_state.get('mode', 'UNKNOWN')
        target_idx = bot_state.get('target_idx', -1)
        action = bot_state.get('action', [0.5, 0.5, 0.0])
        
        # 1. HEATMAP (Cached) - Now using base64 for speed
        if self.viz_show_heatmap:
            import base64
            heatmap_b64 = bot_state.get('heatmap_b64')
            shape = bot_state.get('heatmap_shape')
            if heatmap_b64 and shape and shape[0] > 0:
                self.last_heatmap = (heatmap_b64, shape)

            if self.last_heatmap:
                try:
                    b64_data, sh = self.last_heatmap
                    # Decode base64 to bytes, then to numpy
                    raw_bytes = base64.b64decode(b64_data)
                    hmap = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(sh)
                    hmap_rz = cv2.resize(hmap, (w, h), interpolation=cv2.INTER_LINEAR)
                    hmap_em = cv2.applyColorMap(hmap_rz, cv2.COLORMAP_JET)
                    viz_frame = cv2.addWeighted(hmap_em, 0.4, viz_frame, 0.6, 0)
                except: pass

        # 2. DETECTIONS
        for det in detections:
            x1, y1 = int(det['x_min']*w), int(det['y_min']*h)
            x2, y2 = int(det['x_max']*w), int(det['y_max']*h)
            cls = det.get('class_name', '?')
            color = self.viz_colors['other']
            if cls in ENEMY_CLASSES: color = self.viz_colors['enemy']
            elif cls in LOOT_CLASSES: color = self.viz_colors['loot']
            elif cls.lower() in ['ship', 'player', 'player_ship']: color = self.viz_colors['ship']
            cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, 1)

        # 3. TRACKED OBJECTS with VELOCITY VECTORS
        for i, obj in enumerate(tracked_objects):
            x, y = int(obj['x']*w), int(obj['y']*h)
            cls = obj.get('class_name', '?')
            tid = obj.get('track_id', -1)
            vx = obj.get('velocity_x', 0.0)
            vy = obj.get('velocity_y', 0.0)

            color = self.viz_colors['other']
            if cls in ENEMY_CLASSES: color = self.viz_colors['enemy']
            elif cls in LOOT_CLASSES: color = self.viz_colors['loot']

            cv2.circle(viz_frame, (x, y), 5, color, -1)
            cv2.putText(viz_frame, f"#{tid}", (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Draw VELOCITY VECTOR (predicted movement)
            # Scale velocity for visibility (velocity is in normalized units/frame)
            vel_scale = 50.0  # pixels per unit velocity
            if abs(vx) > 0.001 or abs(vy) > 0.001:
                vx_px = int(vx * w * vel_scale)
                vy_px = int(vy * h * vel_scale)
                end_x = x + vx_px
                end_y = y + vy_px
                # Draw arrow line
                cv2.arrowedLine(viz_frame, (x, y), (end_x, end_y), (255, 0, 255), 2, tipLength=0.3)

            if i == target_idx:
                 cv2.circle(viz_frame, (x, y), 40, self.viz_colors['target'], 2)
                 cv2.putText(viz_frame, "TARGET", (x+45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.viz_colors['target'], 2)

        # 4. MOUSE & KEYBOARD
        keys = bot_state.get('keys', {})
        if keys:
             # Draw active keys at bottom center
             key_str = "KEYS: " + " ".join([k.upper() for k, v in keys.items() if v])
             cv2.putText(viz_frame, key_str, (w//2 - 100, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if action and len(action) >= 2:
            mx, my = int(action[0]*w), int(action[1]*h)
            click = len(action) > 2 and action[2] > 0.5
            
            # Crosshair
            cv2.line(viz_frame, (mx-20,my), (mx+20,my), self.viz_colors['mouse'], 2)
            cv2.line(viz_frame, (mx,my-20), (mx,my+20), self.viz_colors['mouse'], 2)
            
            if click:
                cv2.circle(viz_frame, (mx, my), 35, self.viz_colors['click'], 3)
                cv2.putText(viz_frame, "CLICK", (mx+40, my), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.viz_colors['click'], 2)
            
            # Draw target line if locking
            if target_idx >= 0 and target_idx < len(tracked_objects):
                t = tracked_objects[target_idx]
                tx, ty = int(t['x']*w), int(t['y']*h)
                cv2.line(viz_frame, (mx, my), (tx, ty), (255, 255, 0), 1)

        # 5. OVERLAY TEXT (Left Panel)
        timestamp = bot_state.get('timestamp', 0)
        delay = (time.time() - timestamp) * 1000
        
        def txt(s, y_off, col=(255,255,255), size=0.6, bold=False):
            cv2.putText(viz_frame, s, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, size, col, 2 if bold else 1)
            
        overlay = viz_frame.copy()
        cv2.rectangle(overlay, (0,0), (320, h), (0,0,0), -1)
        viz_frame = cv2.addWeighted(overlay, 0.7, viz_frame, 0.3, 0)
        
        y = 35
        txt("SOCKET DEBUG VIEWER", y, self.viz_colors['target'], 0.7, True)
        y += 25
        col = (0, 255, 0) if delay < 100 else (0, 0, 255)
        txt(f"Latency: {delay:.0f}ms", y, col)
        y += 25
        txt(f"FPS: {int(self.fps)}", y)
        y += 35
        
        txt("=== STRATEGY ===", y, (255, 255, 0), 0.6, True)
        y += 25
        txt(f"Mode: {mode}", y)
        y += 25
        conf = bot_state.get('confidence', 0.5) * 100
        txt(f"Confidence: {conf:.0f}%", y)
        y += 35

        txt("=== TACTICS ===", y, (255, 255, 0), 0.6, True)
        y += 25
        if target_idx >= 0 and target_idx < len(tracked_objects):
            t = tracked_objects[target_idx]
            txt(f"Target: #{t['track_id']}", y, (0, 255, 255))
            y += 20
            txt(f"Class: {t['class_name']}", y, (200, 200, 200), 0.5)
        else:
            txt("Target: None", y, (150, 150, 150))
        y += 35

        txt("=== STATE ===", y, (255, 255, 0), 0.6, True)
        y += 25
        hp = bot_state.get('health', 0)*100
        sh = bot_state.get('shield', 0)*100
        txt(f"HP: {hp:.0f}%", y, (0, 255, 0) if hp > 30 else (0, 0, 255))
        y += 20
        txt(f"Shield: {sh:.0f}%", y, (0, 100, 255))
        y += 20
        atk = bot_state.get('is_attacking', False)
        txt(f"Attacking: {'YES' if atk else 'No'}", y, (255, 0, 0) if atk else (255, 255, 255))

        y = h - 100
        txt("Controls:", y, (150, 150, 150), 0.5)
        y += 20
        txt("V: Toggle Heatmap", y, (150, 150, 150), 0.5)
        y += 20
        txt("Q: Quit", y, (150, 150, 150), 0.5)
        
        # Resize
        display_h = int(h * self.viz_display_scale)
        display_w = int(w * self.viz_display_scale)
        final = cv2.resize(viz_frame, (display_w, display_h))
        self._safe_imshow(final)


    def start(self):
        print(f"[Viewer] Starting... connecting to {self.host}:{self.port}")
        
        blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        while self.running:
            start_t = time.time()
            
            # Use background capture or blank
            frame = blank.copy()
            if self.screen:
                cap = self.screen.capture()
                if cap is not None: frame = cap

            # Overlay state
            if self.latest_state:
                self._visualize_frame(frame, self.latest_state)
            else:
                # Waiting screen
                cv2.putText(frame, f"Waiting for connection...", (400, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                self._safe_imshow(frame)
            
            # FPS loop (60hz for lower latency display)
            elapsed = time.time() - start_t
            if elapsed < 0.016:
                time.sleep(0.016 - elapsed)

            self.frame_count += 1
            if self.frame_count % 60 == 0:
                self.fps = 60 / (time.time() - self.last_frame_time)
                self.last_frame_time = time.time()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): self.running = False
            elif key == ord('v'): self.viz_show_heatmap = not self.viz_show_heatmap

        cv2.destroyAllWindows()
        if self.sock:
             self.sock.close()
        print("[Viewer] Stopped.")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', action='store_true', help='Enable background capture')
    parser.add_argument('--monitor', type=int, default=1)
    parser.add_argument('--ipc', action='store_true', help='Legacy flag (ignored, always uses socket)')
    args = parser.parse_args()
    
    viewer = SocketDebugViewer(capture_bg=args.capture, monitor=args.monitor)
    viewer.start()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[CRITICAL] Debug Viewer crashed: {e}")
        input("Press Enter to close window...")
