"""
FastSAM Real-time Segmentation Visualizer

FastSAM is 50x faster than SAM2 - runs at 25-40 FPS on RTX 3080.
Uses YOLO architecture for speed while maintaining segmentation quality.

Usage:
    python fastsam_visualizer.py              # Default - segment everything
    python fastsam_visualizer.py --monitor 2  # Use second monitor

Controls:
    - Left click: Segment object at point
    - 'a': Toggle auto-segment mode (segment everything)
    - 'c': Clear selection
    - 'q': Quit

Requirements:
    pip install ultralytics  (you already have this!)
"""

import time
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2


class FastSAMSegmenter:
    """FastSAM segmentation - much faster than SAM2."""

    def __init__(self, model_size: str = "s", device: str = "cuda"):
        """
        Initialize FastSAM.

        Args:
            model_size: "s" (small/fast) or "x" (large/accurate)
            device: "cuda" or "cpu"
        """
        self.device = device
        self.model = None

        self._load_model(model_size)

    def _load_model(self, model_size: str):
        """Load FastSAM model."""
        try:
            from ultralytics import FastSAM
        except ImportError:
            print("\n" + "="*60)
            print("  FastSAM not installed!")
            print("="*60)
            print("\nInstall with:")
            print("  pip install ultralytics")
            raise

        # FastSAM models: FastSAM-s (small) or FastSAM-x (large)
        model_name = f"FastSAM-{model_size}.pt"

        print(f"Loading FastSAM ({model_size})...")
        print(f"  Device: {self.device}")

        self.model = FastSAM(model_name)

        print(f"  Model loaded!")

        # Warmup
        print("  Warming up...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self.model(dummy, device=self.device, verbose=False)
        print("  Warmup complete!")

    def segment_everything(self, image: np.ndarray, conf: float = 0.4, iou: float = 0.9) -> List[dict]:
        """
        Segment all objects in image.

        Returns list of masks with bbox info.
        """
        results = self.model(
            image,
            device=self.device,
            retina_masks=True,
            conf=conf,
            iou=iou,
            verbose=False
        )

        masks_data = []
        if results and len(results) > 0 and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes

            for i, mask in enumerate(masks):
                # Get bounding box
                if boxes is not None and i < len(boxes):
                    box = boxes[i].xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = box
                    bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    conf_score = float(boxes[i].conf[0])
                else:
                    # Compute bbox from mask
                    ys, xs = np.where(mask > 0.5)
                    if len(xs) > 0:
                        bbox = [float(xs.min()), float(ys.min()),
                               float(xs.max() - xs.min()), float(ys.max() - ys.min())]
                    else:
                        bbox = [0, 0, 0, 0]
                    conf_score = 0.5

                masks_data.append({
                    'segmentation': mask > 0.5,
                    'bbox': bbox,
                    'area': int(np.sum(mask > 0.5)),
                    'predicted_iou': conf_score
                })

        return masks_data

    def segment_point(self, image: np.ndarray, point: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Segment object at specific point.

        Returns the mask containing that point.
        """
        results = self.model(
            image,
            device=self.device,
            retina_masks=True,
            conf=0.4,
            iou=0.9,
            verbose=False
        )

        if results and len(results) > 0 and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()

            # Find mask containing the point
            x, y = point
            for mask in masks:
                # Resize mask to image size if needed
                if mask.shape != image.shape[:2]:
                    mask = cv2.resize(mask.astype(np.float32),
                                     (image.shape[1], image.shape[0])) > 0.5

                if y < mask.shape[0] and x < mask.shape[1] and mask[y, x]:
                    return mask

        return None


class ScreenCapture:
    """Fast screen capture using DXCam (Windows) or mss (fallback)."""

    def __init__(self, monitor_index: int = 1):
        self.use_dxcam = False
        self.camera = None
        self.monitor = None

        import mss
        sct = mss.mss()
        if monitor_index < len(sct.monitors):
            self.monitor = dict(sct.monitors[monitor_index])
        else:
            self.monitor = dict(sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0])

        try:
            import dxcam
            device_idx = max(0, monitor_index - 1) if monitor_index > 0 else 0
            self.camera = dxcam.create(device_idx=device_idx, output_color="BGR")
            self.camera.start(target_fps=60)
            self.use_dxcam = True
            print(f"  Using DXCam for screen capture")
        except Exception as e:
            print(f"  DXCam not available ({e}), using mss")
            self.sct = sct

    def capture(self) -> np.ndarray:
        if self.use_dxcam:
            frame = self.camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                frame = self.camera.get_latest_frame()
            return frame if frame is not None else np.zeros((100, 100, 3), dtype=np.uint8)
        else:
            img = self.sct.grab(self.monitor)
            frame = np.array(img)
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            return frame


def draw_masks(frame: np.ndarray, masks: List[dict], alpha: float = 0.5) -> np.ndarray:
    """Draw all segmentation masks."""
    overlay = frame.copy()

    # Sort by area (largest first)
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    for i, mask_data in enumerate(sorted_masks):
        mask = mask_data['segmentation']

        # Resize mask if needed
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(mask.astype(np.float32),
                            (frame.shape[1], frame.shape[0])) > 0.5

        # Random color
        np.random.seed(i)
        color = tuple(int(c) for c in np.random.randint(50, 255, 3))

        # Apply mask
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = color

        # Draw bbox
        bbox = mask_data['bbox']
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+int(w), y+int(h)), color, 1)

    result = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
    return result


def draw_single_mask(frame: np.ndarray, mask: np.ndarray,
                    color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.5) -> np.ndarray:
    """Draw a single mask."""
    # Resize mask if needed
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(mask.astype(np.float32),
                         (frame.shape[1], frame.shape[0])) > 0.5

    overlay = frame.copy()
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = color

    result = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)

    # Draw contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)

    return result


def run_visualizer(model_size: str = "s", monitor: int = 1):
    """Run FastSAM real-time visualizer."""
    print("\n" + "="*60)
    print("  FastSAM Real-time Visualizer (FAST!)")
    print("="*60)

    # Initialize FastSAM
    try:
        segmenter = FastSAMSegmenter(model_size=model_size, device="cuda")
    except Exception as e:
        print(f"\nFailed to initialize FastSAM: {e}")
        return

    # Initialize screen capture
    print(f"\nInitializing screen capture on monitor {monitor}...")
    screen = ScreenCapture(monitor_index=monitor)

    # Create window
    window_name = "FastSAM Visualizer (q=quit, a=auto, c=clear)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # State
    selected_point = None
    selected_mask = None
    auto_mode = True  # Start in auto mode to show all masks

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_point, selected_mask, auto_mode
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_point = (x, y)
            selected_mask = None  # Will be computed in main loop
            auto_mode = False
            print(f"  Selected point at ({x}, {y})")

    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\n{'='*60}")
    print(f"  Mode: {'auto' if auto_mode else 'point'}")
    print(f"  Controls:")
    print(f"    Left click: Segment object at point")
    print(f"    'a': Toggle auto-segment mode")
    print(f"    'c': Clear selection")
    print(f"    'q': Quit")
    print(f"{'='*60}\n")

    frame_count = 0
    last_fps_time = time.time()
    fps = 0
    seg_time = 0

    try:
        while True:
            # Capture frame
            frame = screen.capture()
            if frame is None:
                continue

            h, w = frame.shape[:2]

            # Process at slightly reduced resolution for speed
            process_scale = 0.75
            small_w, small_h = int(w * process_scale), int(h * process_scale)
            small_frame = cv2.resize(frame, (small_w, small_h))

            seg_start = time.time()

            if auto_mode:
                # Segment everything
                masks = segmenter.segment_everything(small_frame)
                vis_frame = draw_masks(small_frame.copy(), masks)
                mask_count = len(masks)
            else:
                vis_frame = small_frame.copy()
                mask_count = 0

                if selected_point:
                    # Scale point
                    sp = (int(selected_point[0] * process_scale),
                         int(selected_point[1] * process_scale))

                    # Get or update mask
                    if selected_mask is None:
                        selected_mask = segmenter.segment_point(small_frame, sp)

                    if selected_mask is not None:
                        vis_frame = draw_single_mask(vis_frame, selected_mask)
                        mask_count = 1

                    # Draw point
                    cv2.circle(vis_frame, sp, 8, (0, 255, 0), -1)
                    cv2.circle(vis_frame, sp, 8, (255, 255, 255), 2)

            seg_time = time.time() - seg_start

            # Scale back up
            vis_frame = cv2.resize(vis_frame, (w, h))

            # Draw info
            info_text = [
                f"Mode: {'AUTO' if auto_mode else 'POINT'} | FPS: {fps:.1f}",
                f"Seg: {seg_time*1000:.0f}ms | Masks: {mask_count}",
            ]

            for i, text in enumerate(info_text):
                cv2.putText(vis_frame, text, (10, 30 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(vis_frame, text, (10, 30 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

            cv2.imshow(window_name, vis_frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                auto_mode = not auto_mode
                selected_point = None
                selected_mask = None
                print(f"  Mode: {'auto' if auto_mode else 'point'}")
            elif key == ord('c'):
                selected_point = None
                selected_mask = None
                print("  Cleared selection")

            # FPS calculation
            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_count / (now - last_fps_time)
                frame_count = 0
                last_fps_time = now

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("\nStopped.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='FastSAM Real-time Visualizer')
    parser.add_argument('--model', type=str, default='s',
                       choices=['s', 'x'],
                       help='FastSAM model size (s=fast, x=accurate)')
    parser.add_argument('--monitor', type=int, default=1,
                       help='Monitor to capture (1=primary, 2=secondary)')

    args = parser.parse_args()
    run_visualizer(model_size=args.model, monitor=args.monitor)


if __name__ == "__main__":
    main()
