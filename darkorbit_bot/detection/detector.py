"""
DarkOrbit Bot - Real-time Object Detector
Detects boxes, NPCs, and other game objects using trained YOLO model.

Usage:
    python detector.py          # Test detection on screenshots
    python detector.py --live   # Live detection from screen

In bot code:
    from detection.detector import GameDetector
    detector = GameDetector("models/detector.pt")
    objects = detector.detect_frame(screenshot)
"""

import os
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    """A detected game object"""
    class_name: str      # "box", "npc", etc.
    class_id: int        # Class index
    confidence: float    # 0-1
    x_center: float      # Normalized center x (0-1)
    y_center: float      # Normalized center y (0-1)
    width: float         # Normalized width
    height: float        # Normalized height
    x_min: float         # Bounding box corners
    y_min: float
    x_max: float
    y_max: float
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.x_center, self.y_center)
    
    @property
    def area(self) -> float:
        return self.width * self.height


class GameDetector:
    """Real-time game object detector using YOLOv8"""
    
    # Default class names (matches dataset.yaml)
    DEFAULT_CLASSES = {
        0: "box",
        1: "npc",
        2: "player_ship",
        3: "bonus_box",
        4: "portal"
    }
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.3,
                 iou_threshold: float = 0.3, device: str = "auto"):
        """
        Initialize detector.

        Args:
            model_path: Path to trained .pt model
            confidence_threshold: Minimum confidence to accept detection (default 0.3)
            iou_threshold: NMS IoU threshold for duplicate removal (default 0.3, lower = more aggressive)
            device: "auto", "cpu", "cuda", or "mps"
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model_path = model_path
        self.model = None
        self.class_names = self.DEFAULT_CLASSES.copy()

        # Load model
        self._load_model(model_path, device)
    
    def _load_model(self, model_path: str, device: str):
        """Load YOLO model and initialize on GPU with warmup"""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)

        # Determine device
        self.device = device
        if device == "auto":
            try:
                import torch
                self.device = 0 if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

        # Move model to device and verify
        if self.device != "cpu":
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"   ✅ YOLO using GPU: {gpu_name}")

                    # Enable cuDNN benchmark for consistent input sizes (gaming = same res)
                    torch.backends.cudnn.benchmark = True

                    # Warmup: run a dummy inference to initialize CUDA kernels
                    print(f"   Warming up GPU...")
                    dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)  # 1080p dummy frame
                    _ = self.model.predict(dummy, conf=0.5, iou=0.7, verbose=False,
                                          imgsz=1280, half=True, device=self.device, max_det=300)
                    print(f"   ✅ GPU warmup complete")
                else:
                    print(f"   ⚠️ CUDA not available, falling back to CPU")
                    self.device = "cpu"
            except Exception as e:
                print(f"   ⚠️ GPU init failed ({e}), using CPU")
                self.device = "cpu"
        else:
            print(f"   Running on CPU (slower)")

        # Get class names from model if available
        if hasattr(self.model, 'names') and self.model.names:
            self.class_names = self.model.names

        print(f"Classes: {self.class_names}")
    
    def detect_frame(self, frame: np.ndarray, 
                    filter_classes: List[str] = None) -> List[Detection]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Image as numpy array (BGR or RGB)
            filter_classes: Only return these classes (None = all)
        
        Returns:
            List of Detection objects
        """
        if self.model is None:
            return []
        
        # Run inference (GPU accelerated)
        # Device is set once at init, not per-call (avoids re-initialization)
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,  # Confidence threshold from __init__
            iou=self.iou_threshold,          # NMS IoU threshold from __init__
            verbose=False,
            imgsz=1280,      # FullHD quality
            half=True,       # FP16 for speed
            device=self.device,
            max_det=300,     # Maximum detections per image
            agnostic_nms=False  # Class-specific NMS (don't merge different classes)
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                # Get box data
                box = boxes[i]
                
                # Class
                class_id = int(box.cls[0])
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                
                # Filter by class if specified
                if filter_classes and class_name not in filter_classes:
                    continue
                
                # Confidence
                conf = float(box.conf[0])
                
                # Bounding box (normalized)
                xyxy = box.xyxyn[0].cpu().numpy()  # Normalized coordinates
                x_min, y_min, x_max, y_max = xyxy
                
                # Calculate center and size
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                
                detection = Detection(
                    class_name=class_name,
                    class_id=class_id,
                    confidence=conf,
                    x_center=float(x_center),
                    y_center=float(y_center),
                    width=float(width),
                    height=float(height),
                    x_min=float(x_min),
                    y_min=float(y_min),
                    x_max=float(x_max),
                    y_max=float(y_max)
                )
                
                detections.append(detection)
        
        return detections
    
    def detect_screenshot(self, image_path: str) -> List[Detection]:
        """Detect objects in a screenshot file"""
        import cv2
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.detect_frame(frame)
    
    def find_nearest(self, detections: List[Detection], 
                    target_class: str,
                    from_pos: Tuple[float, float] = (0.5, 0.5)) -> Optional[Detection]:
        """Find the nearest object of a given class"""
        matching = [d for d in detections if d.class_name == target_class]
        
        if not matching:
            return None
        
        def distance(det):
            dx = det.x_center - from_pos[0]
            dy = det.y_center - from_pos[1]
            return dx*dx + dy*dy
        
        return min(matching, key=distance)
    
    def find_highest_confidence(self, detections: List[Detection],
                               target_class: str) -> Optional[Detection]:
        """Find the object with highest confidence"""
        matching = [d for d in detections if d.class_name == target_class]
        
        if not matching:
            return None
        
        return max(matching, key=lambda d: d.confidence)


class ScreenCapture:
    """Capture game screen for detection - uses fast DXCam on Windows"""

    def __init__(self, game_rect: dict = None, monitor_index: int = 0, continuous: bool = True):
        """
        Args:
            game_rect: {"left": x, "top": y, "width": w, "height": h}
                      None = full screen
            monitor_index: 0 = all monitors, 1 = first monitor, 2 = second, etc.
            continuous: True = DXCam continuous mode (fast, for real-time)
                       False = Single-shot mode (for slow processing like bootstrap)
        """
        self.use_dxcam = False
        self.camera = None
        self.monitor = None  # Always store monitor geometry for coordinate conversion
        self.monitor_index = monitor_index
        self.continuous = continuous

        # Get monitor geometry using mss (works for both DXCam and mss capture)
        import mss
        self.sct = mss.mss()
        if game_rect:
            self.monitor = {
                "left": game_rect["left"],
                "top": game_rect["top"],
                "width": game_rect["width"],
                "height": game_rect["height"]
            }
        else:
            # monitors[0] = all, monitors[1] = first, monitors[2] = second, etc.
            if monitor_index < len(self.sct.monitors):
                self.monitor = dict(self.sct.monitors[monitor_index])
            else:
                print(f"Monitor {monitor_index} not found, using primary (1)")
                self.monitor = dict(self.sct.monitors[1] if len(self.sct.monitors) > 1 else self.sct.monitors[0])

        # Only use DXCam continuous mode if requested (fast real-time capture)
        if continuous:
            try:
                import dxcam
                # dxcam uses 0-based indexing, mss uses 1-based
                device_idx = max(0, monitor_index - 1) if monitor_index > 0 else 0
                self.camera = dxcam.create(device_idx=device_idx, output_color="BGR")
                self.camera.start(target_fps=60)
                self.use_dxcam = True
                print(f"   Using DXCam (fast) for screen capture")
            except Exception as e:
                print(f"   DXCam not available ({e}), using mss (slower)")
        else:
            # Single-shot mode - use mss (more reliable for slow processing)
            print(f"   Using mss for screen capture (single-shot mode)")

    def capture(self) -> np.ndarray:
        """Capture current screen"""
        if self.use_dxcam and self.camera is not None:
            frame = self.camera.get_latest_frame()
            if frame is None:
                # Wait for first frame
                import time
                time.sleep(0.01)
                frame = self.camera.get_latest_frame()
            return frame if frame is not None else np.zeros((100, 100, 3), dtype=np.uint8)
        else:
            img = self.sct.grab(self.monitor)
            # Convert to numpy array (BGR format for OpenCV)
            frame = np.array(img)
            # Remove alpha channel if present
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            return frame

    def stop(self):
        """Stop DXCam if running"""
        if self.use_dxcam and self.camera is not None:
            try:
                self.camera.stop()
            except:
                pass
    
    def capture_and_detect(self, detector: GameDetector) -> Tuple[np.ndarray, List[Detection]]:
        """Capture and detect in one call"""
        frame = self.capture()
        detections = detector.detect_frame(frame)
        return frame, detections


def demo_on_screenshots():
    """Demo detection on existing screenshots"""
    print("\n" + "="*60)
    print("  DETECTOR DEMO - Screenshots")
    print("="*60)
    
    # Find model
    script_dir = Path(__file__).parent
    model_path = script_dir.parent / "models" / "detector.pt"
    
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        print("   Train the model first with train_yolo.py")
        return
    
    # Find screenshots
    recordings_dir = script_dir.parent / "data" / "recordings"
    screenshots = []
    
    for session in recordings_dir.glob("session_*"):
        shots_dir = session / "screenshots"
        if shots_dir.exists():
            screenshots.extend(list(shots_dir.glob("*.png"))[:5])  # Take 5 per session
    
    if not screenshots:
        print("\n❌ No screenshots found")
        return
    
    print(f"\nFound {len(screenshots)} screenshots to test")
    
    # Initialize detector
    detector = GameDetector(str(model_path))
    
    # Test on each
    print("\n📊 Detection results:")
    for img_path in screenshots[:10]:  # Test first 10
        detections = detector.detect_screenshot(str(img_path))
        
        if detections:
            print(f"\n  {img_path.name}:")
            for det in detections:
                print(f"    - {det.class_name}: {det.confidence:.2f} at ({det.x_center:.2f}, {det.y_center:.2f})")
        else:
            print(f"  {img_path.name}: No detections")



def draw_detections(frame, detections):
    """Draw bounding boxes on frame"""
    import cv2
    import numpy as np
    
    # Make contiguous copy to ensure OpenCV compatibility
    # mss returns arrays with unusual memory layouts that cv2 doesn't like
    frame = np.ascontiguousarray(frame)
        
    h, w = frame.shape[:2]
    
    for det in detections:
        # Convert to pixels
        x1 = int(det.x_min * w)
        y1 = int(det.y_min * h)
        x2 = int(det.x_max * w)
        y2 = int(det.y_max * h)
        
        # Color based on class hash (consistent random color)
        color_seed = hash(det.class_name) % (256*256*256)
        color = (
            color_seed & 255, 
            (color_seed >> 8) & 255, 
            (color_seed >> 16) & 255
        )
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Background for text
        cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return frame

def demo_live(model_path: str = None, visualize: bool = False, monitor: int = 1):
    """Demo live detection from screen"""
    import cv2
    
    print("\n" + "="*60)
    print("  DETECTOR DEMO - Live")
    print("="*60)
    
    # Find model
    if not model_path:
        script_dir = Path(__file__).parent
        model_path = script_dir.parent / "models" / "detector.pt"
        if not model_path.exists():
             # Try local best.pt if exists or ask user
             pass
    
    print(f"Loading model: {model_path}")
    
    if not Path(model_path).exists():
        print(f"\n❌ Model not found: {model_path}")
        return
    
    # Initialize
    detector = GameDetector(str(model_path), confidence_threshold=0.1)  # Lowered from 0.4 to test overfitting
    # NOTE: ScreenCapture created inside inference_thread only (DXCam doesn't like multiple instances)
    
    print(f"\n🎮 Starting live detection on monitor {monitor}...")
    print("   Press 'q' or Ctrl+C to stop")
    print("-"*60)
    
    # Create resizable window on DIFFERENT monitor to avoid feedback loop
    window_name = "Bot Vision (Press 'q' to quit)"
    if visualize:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Move window to different position to avoid capturing itself
        # This helps prevent the feedback loop where the window sees itself
        try:
            cv2.moveWindow(window_name, -1920, 100)  # Move to left monitor if exists
        except:
            pass  # Ignore if can't move window
    
    # Threading for smooth display
    import threading
    
    latest_frame = None
    latest_detections = []
    data_lock = threading.Lock()
    running = True
    inference_fps = 0
    
    def inference_thread():
        nonlocal latest_frame, latest_detections, running, inference_fps
        # Only ONE screen capture instance (DXCam doesn't support multiple)
        screen = ScreenCapture(monitor_index=monitor)
        while running:
            start = time.time()
            frame = screen.capture()
            dets = detector.detect_frame(frame)
            with data_lock:
                latest_frame = frame
                latest_detections = dets
            elapsed = time.time() - start
            inference_fps = 1 / elapsed if elapsed > 0 else 0
    
    # Start inference in background
    inf_thread = threading.Thread(target=inference_thread, daemon=True)
    inf_thread.start()
    
    # Wait for first frame
    print("   Warming up...")
    while latest_frame is None:
        time.sleep(0.01)
    
    try:
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Get latest frame and detections (thread-safe)
            with data_lock:
                frame = latest_frame
                detections = latest_detections.copy() if latest_detections else []
            
            if frame is None:
                continue
            
            # Draw visual
            if visualize:
                vis_frame = draw_detections(frame, detections)
                cv2.imshow(window_name, vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            frame_count += 1
            
            # Print results every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                display_fps = frame_count / elapsed if elapsed > 0 else 0
                det_summary = {}
                for det in detections:
                    det_summary[det.class_name] = det_summary.get(det.class_name, 0) + 1
                print(f"\rDisplay: {display_fps:.0f} FPS | Inference: {inference_fps:.1f} FPS | {det_summary}    ", end="")
            
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        print("\n\nStopped.")
        if visualize:
            cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='DarkOrbit Object Detector')
    parser.add_argument('--live', action='store_true', help='Run live detection on screen')
    parser.add_argument('--model', type=str, help='Path to .pt model file')
    parser.add_argument('--show', action='store_true', help='Show visualization window')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--monitor', type=int, default=1, help='Monitor to capture (1=primary, 2=secondary, 0=all)')
    
    args = parser.parse_args()
    
    if args.live:
        demo_live(model_path=args.model, visualize=args.show, monitor=args.monitor)
    else:
        # Pass model if provided, else defaults inside
        demo_on_screenshots()


if __name__ == "__main__":
    main()
