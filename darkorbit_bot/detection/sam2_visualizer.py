"""
SAM2 Real-time Segmentation Visualizer

Runs SAM2 on your screen in real-time to visualize what it segments.
This helps understand how SAM2 perceives game elements.

Usage:
    python sam2_visualizer.py                    # Auto-segment everything
    python sam2_visualizer.py --points           # Click to add prompt points
    python sam2_visualizer.py --grid             # Use grid of points
    python sam2_visualizer.py --monitor 2        # Use second monitor

Controls (in visualization window):
    - Left click: Add positive point (include this region)
    - Right click: Add negative point (exclude this region)
    - 'c': Clear all points
    - 'a': Toggle auto-segment mode (segment everything)
    - 'g': Toggle grid mode
    - 'q': Quit

Note: SAM2 is slow (~1 FPS). For faster segmentation, use fastsam_visualizer.py instead.

Requirements:
    pip install sam2 torch torchvision

    Download SAM2 checkpoint from:
    https://github.com/facebookresearch/sam2#sam-21-checkpoints
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class SAM2Segmenter:
    """SAM2 segmentation wrapper for real-time use."""

    # Model configs (smallest to largest)
    MODEL_CONFIGS = {
        "tiny": "sam2_hiera_t.yaml",
        "small": "sam2_hiera_s.yaml",
        "base_plus": "sam2_hiera_b+.yaml",
        "large": "sam2_hiera_l.yaml",
    }

    CHECKPOINT_URLS = {
        "tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        "small": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "large": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
    }

    def __init__(self, model_size: str = "tiny", device: str = "cuda"):
        """
        Initialize SAM2.

        Args:
            model_size: "tiny", "small", "base_plus", or "large"
            device: "cuda" or "cpu"
        """
        self.device = device
        self.model_size = model_size
        self.model = None
        self.predictor = None
        self.mask_generator = None

        self._load_model(model_size)

    def _load_model(self, model_size: str):
        """Load SAM2 model."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError:
            print("\n" + "="*60)
            print("  SAM2 not installed!")
            print("="*60)
            print("\nInstall with:")
            print("  pip install sam2")
            print("\nOr from source:")
            print("  git clone https://github.com/facebookresearch/sam2.git")
            print("  cd sam2")
            print("  pip install -e .")
            print("\nThen download a checkpoint:")
            print(f"  {self.CHECKPOINT_URLS.get(model_size, self.CHECKPOINT_URLS['tiny'])}")
            raise

        # Find checkpoint
        checkpoint_path = self._find_checkpoint(model_size)
        config = self.MODEL_CONFIGS.get(model_size, self.MODEL_CONFIGS["tiny"])

        print(f"Loading SAM2 ({model_size})...")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Config: {config}")
        print(f"  Device: {self.device}")

        # Build model
        self.model = build_sam2(config, checkpoint_path, device=self.device)

        # Create predictor for point/box prompts
        self.predictor = SAM2ImagePredictor(self.model)

        # Create automatic mask generator (for segmenting everything)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            self.model,
            points_per_side=16,  # Reduced for speed (default 32)
            points_per_batch=64,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.8,
            crop_n_layers=0,  # No multi-crop for speed
            min_mask_region_area=100,
        )

        print(f"  Model loaded successfully!")

        # Warmup
        print("  Warming up...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.predictor.set_image(dummy)
        _ = self.predictor.predict(
            point_coords=np.array([[320, 240]]),
            point_labels=np.array([1]),
            multimask_output=False
        )
        print("  Warmup complete!")

    def _find_checkpoint(self, model_size: str) -> str:
        """Find SAM2 checkpoint file."""
        # Common checkpoint locations
        search_paths = [
            Path(__file__).parent.parent / "models",
            Path(__file__).parent.parent / "checkpoints",
            Path.home() / ".cache" / "sam2",
            Path.home() / "sam2_checkpoints",
            Path("./checkpoints"),
            Path("./models"),
        ]

        checkpoint_names = [
            f"sam2_hiera_{model_size}.pt",
            f"sam2_{model_size}.pt",
            "sam2_hiera_tiny.pt",  # Fallback to tiny
        ]

        for search_path in search_paths:
            for name in checkpoint_names:
                path = search_path / name
                if path.exists():
                    return str(path)

        # Not found - provide download instructions
        print("\n" + "="*60)
        print("  SAM2 checkpoint not found!")
        print("="*60)
        print(f"\nDownload from:")
        print(f"  {self.CHECKPOINT_URLS.get(model_size, self.CHECKPOINT_URLS['tiny'])}")
        print(f"\nPlace in one of these locations:")
        for p in search_paths[:3]:
            print(f"  {p}")
        raise FileNotFoundError(f"SAM2 checkpoint not found for model size '{model_size}'")

    def segment_with_points(self,
                           image: np.ndarray,
                           points: List[Tuple[int, int]],
                           labels: List[int]) -> Tuple[np.ndarray, float]:
        """
        Segment using point prompts.

        Args:
            image: BGR image
            points: List of (x, y) coordinates
            labels: List of labels (1=positive, 0=negative)

        Returns:
            mask: Binary mask
            score: Confidence score
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Set image
        self.predictor.set_image(image_rgb)

        # Predict
        point_coords = np.array(points)
        point_labels = np.array(labels)

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False
        )

        return masks[0], scores[0]

    def segment_with_box(self,
                        image: np.ndarray,
                        box: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float]:
        """
        Segment using box prompt.

        Args:
            image: BGR image
            box: (x1, y1, x2, y2) bounding box

        Returns:
            mask: Binary mask
            score: Confidence score
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        masks, scores, _ = self.predictor.predict(
            box=np.array(box),
            multimask_output=False
        )

        return masks[0], scores[0]

    def segment_everything(self, image: np.ndarray) -> List[dict]:
        """
        Automatically segment all objects in the image.

        Args:
            image: BGR image

        Returns:
            List of mask dicts with keys: segmentation, area, bbox, predicted_iou, stability_score
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image_rgb)
        return masks

    def segment_grid(self, image: np.ndarray, grid_size: int = 4) -> List[Tuple[np.ndarray, float, Tuple[int, int]]]:
        """
        Segment using a grid of points (fast alternative to segment_everything).

        Args:
            image: BGR image
            grid_size: Number of points per dimension

        Returns:
            List of (mask, score, point) tuples
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        h, w = image.shape[:2]
        results = []

        # Generate grid points
        for i in range(grid_size):
            for j in range(grid_size):
                x = int((j + 0.5) * w / grid_size)
                y = int((i + 0.5) * h / grid_size)

                masks, scores, _ = self.predictor.predict(
                    point_coords=np.array([[x, y]]),
                    point_labels=np.array([1]),
                    multimask_output=False
                )

                results.append((masks[0], scores[0], (x, y)))

        return results


class ScreenCapture:
    """Fast screen capture using DXCam (Windows) or mss (fallback)."""

    def __init__(self, monitor_index: int = 1):
        self.use_dxcam = False
        self.camera = None
        self.monitor = None

        # Get monitor geometry
        import mss
        sct = mss.mss()
        if monitor_index < len(sct.monitors):
            self.monitor = dict(sct.monitors[monitor_index])
        else:
            self.monitor = dict(sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0])

        # Try DXCam (much faster on Windows)
        try:
            import dxcam
            device_idx = max(0, monitor_index - 1) if monitor_index > 0 else 0
            self.camera = dxcam.create(device_idx=device_idx, output_color="BGR")
            self.camera.start(target_fps=30)
            self.use_dxcam = True
            print(f"  Using DXCam for screen capture")
        except Exception as e:
            print(f"  DXCam not available ({e}), using mss")
            self.sct = sct

    def capture(self) -> np.ndarray:
        """Capture current screen."""
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
    """Draw automatic segmentation masks with random colors."""
    overlay = frame.copy()

    # Sort by area (largest first, so smaller objects on top)
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    for i, mask_data in enumerate(sorted_masks):
        mask = mask_data['segmentation']

        # Random color based on index
        np.random.seed(i)
        color = tuple(int(c) for c in np.random.randint(50, 255, 3))

        # Apply color to mask area (convert to boolean for indexing)
        mask_bool = np.asarray(mask, dtype=bool)
        overlay[mask_bool] = color

        # Draw bbox
        bbox = mask_data['bbox']  # x, y, w, h format
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)

        # Draw score
        score = mask_data.get('predicted_iou', 0)
        cv2.putText(frame, f"{score:.2f}", (x, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Blend overlay
    result = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
    return result


def draw_single_mask(frame: np.ndarray, mask: np.ndarray, score: float,
                    color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.5) -> np.ndarray:
    """Draw a single mask."""
    overlay = frame.copy()
    # Convert mask to boolean for indexing
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = color
    result = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)

    # Draw contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)

    return result


def draw_grid_masks(frame: np.ndarray, results: List[Tuple[np.ndarray, float, Tuple[int, int]]],
                   alpha: float = 0.4) -> np.ndarray:
    """Draw grid-based segmentation results."""
    overlay = frame.copy()

    for i, (mask, score, point) in enumerate(results):
        if score < 0.5:  # Skip low confidence
            continue

        # Color based on index
        np.random.seed(i * 42)
        color = tuple(int(c) for c in np.random.randint(50, 255, 3))

        # Apply mask (convert to boolean for indexing)
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = color

        # Draw point
        cv2.circle(frame, point, 5, (255, 255, 255), -1)
        cv2.circle(frame, point, 5, color, 2)

        # Draw score near point
        cv2.putText(frame, f"{score:.2f}", (point[0]+8, point[1]+5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    result = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
    return result


def run_visualizer(model_size: str = "tiny", monitor: int = 1, mode: str = "points"):
    """
    Run SAM2 real-time visualizer.

    Args:
        model_size: SAM2 model size
        monitor: Monitor index (1=primary, 2=secondary)
        mode: "points", "auto", or "grid"
    """
    print("\n" + "="*60)
    print("  SAM2 Real-time Visualizer")
    print("="*60)

    # Initialize SAM2
    try:
        segmenter = SAM2Segmenter(model_size=model_size, device="cuda")
    except Exception as e:
        print(f"\nFailed to initialize SAM2: {e}")
        return

    # Initialize screen capture
    print(f"\nInitializing screen capture on monitor {monitor}...")
    screen = ScreenCapture(monitor_index=monitor)

    # Create window
    window_name = "SAM2 Visualizer (q=quit, c=clear, a=auto, g=grid)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # State
    points = []  # List of (x, y)
    labels = []  # List of 1 (positive) or 0 (negative)
    current_mode = mode
    running = True

    # Mouse callback
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, labels
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            labels.append(1)  # Positive point
            print(f"  Added positive point at ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((x, y))
            labels.append(0)  # Negative point
            print(f"  Added negative point at ({x}, {y})")

    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\n{'='*60}")
    print(f"  Mode: {current_mode}")
    print(f"  Controls:")
    print(f"    Left click: Add positive point")
    print(f"    Right click: Add negative point")
    print(f"    'c': Clear points")
    print(f"    'a': Toggle auto-segment mode")
    print(f"    'g': Toggle grid mode")
    print(f"    'q': Quit")
    print(f"{'='*60}\n")

    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    fps = 0
    seg_time = 0

    try:
        while running:
            # Capture frame
            frame = screen.capture()
            if frame is None:
                continue

            # Scale down if image is too large (max 1280px on longest side)
            h, w = frame.shape[:2]
            max_dim = 1280
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                process_frame = cv2.resize(frame, (new_w, new_h))
                scale_factor = scale
            else:
                process_frame = frame
                scale_factor = 1.0

            seg_start = time.time()

            # Segment based on mode
            if current_mode == "auto":
                # Automatic segmentation
                masks = segmenter.segment_everything(process_frame)
                vis_frame = draw_masks(process_frame.copy(), masks)
                mask_count = len(masks)

            elif current_mode == "grid":
                # Grid-based segmentation
                grid_results = segmenter.segment_grid(process_frame, grid_size=3)
                vis_frame = draw_grid_masks(process_frame.copy(), grid_results)
                mask_count = len([r for r in grid_results if r[1] >= 0.5])

            else:  # points mode
                vis_frame = process_frame.copy()
                mask_count = 0
                if points:
                    # Scale points if we resized the frame
                    scaled_points = [(int(x * scale_factor), int(y * scale_factor)) for x, y in points]
                    mask, score = segmenter.segment_with_points(process_frame, scaled_points, labels)
                    vis_frame = draw_single_mask(vis_frame, mask, score)
                    mask_count = 1

            seg_time = time.time() - seg_start

            # Draw points (always, in points mode)
            if current_mode == "points":
                for (x, y), label in zip(points, labels):
                    sx, sy = int(x * scale_factor), int(y * scale_factor)
                    color = (0, 255, 0) if label == 1 else (0, 0, 255)
                    cv2.circle(vis_frame, (sx, sy), 6, color, -1)
                    cv2.circle(vis_frame, (sx, sy), 6, (255, 255, 255), 2)

            # Scale back up for display if needed
            if scale_factor != 1.0:
                vis_frame = cv2.resize(vis_frame, (w, h))

            # Draw info
            info_text = [
                f"Mode: {current_mode} | FPS: {fps:.1f}",
                f"Seg: {seg_time*1000:.0f}ms | Masks: {mask_count}",
            ]
            if current_mode == "points":
                info_text.append(f"Points: {len(points)}")

            for i, text in enumerate(info_text):
                cv2.putText(vis_frame, text, (10, 30 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(vis_frame, text, (10, 30 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            # Show
            cv2.imshow(window_name, vis_frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                points = []
                labels = []
                print("  Cleared points")
            elif key == ord('a'):
                current_mode = "auto" if current_mode != "auto" else "points"
                print(f"  Mode: {current_mode}")
            elif key == ord('g'):
                current_mode = "grid" if current_mode != "grid" else "points"
                print(f"  Mode: {current_mode}")

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
    parser = argparse.ArgumentParser(description='SAM2 Real-time Visualizer')
    parser.add_argument('--model', type=str, default='tiny',
                       choices=['tiny', 'small', 'base_plus', 'large'],
                       help='SAM2 model size (tiny is fastest)')
    parser.add_argument('--monitor', type=int, default=1,
                       help='Monitor to capture (1=primary, 2=secondary)')
    parser.add_argument('--points', action='store_true',
                       help='Start in point prompt mode (default)')
    parser.add_argument('--auto', action='store_true',
                       help='Start in auto-segment mode')
    parser.add_argument('--grid', action='store_true',
                       help='Start in grid mode')

    args = parser.parse_args()

    # Determine mode
    if args.auto:
        mode = "auto"
    elif args.grid:
        mode = "grid"
    else:
        mode = "points"

    run_visualizer(model_size=args.model, monitor=args.monitor, mode=mode)


if __name__ == "__main__":
    main()
