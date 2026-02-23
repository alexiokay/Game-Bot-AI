"""
HP Bar Labeling Tool

Interactive GUI for labeling enemy HP bars to train the HP regression model.

Usage:
    python tools/hp_labeler.py

Controls:
    - Click on HP bar to mark its right edge → Auto-calculates HP%
    - Number keys 0-9: Quick HP percentage (0%, 10%, 20%, ..., 90%, 100%)
    - S: Skip this RoI
    - Q: Quit and save
    - U: Undo last label

The tool saves labeled data to: data/hp_labels/
Format: {roi_image: np.ndarray, hp_percent: float, enemy_class: str, timestamp: int}
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse


class HPLabeler:
    """Interactive HP bar labeling tool."""

    def __init__(self, output_dir: str = "data/hp_labels"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.labels: List[Dict] = []
        self.current_roi: Optional[np.ndarray] = None
        self.current_bbox: Optional[Tuple[float, float, float, float]] = None
        self.current_enemy_class: str = "unknown"

        # State
        self.click_position: Optional[Tuple[int, int]] = None
        self.hp_percent: Optional[float] = None

        # Display
        self.window_name = "HP Bar Labeler"
        self.display_image: Optional[np.ndarray] = None

        # Stats
        self.labeled_count = 0
        self.skipped_count = 0

        print("[HP-LABELER] Initialized")
        print(f"[HP-LABELER] Output directory: {self.output_dir}")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to mark HP bar edge."""
        if event == cv2.EVENT_LBUTTONDOWN and self.current_roi is not None:
            self.click_position = (x, y)

            # Calculate HP% based on click position
            # Assume HP bar is horizontal near top of RoI
            roi_width = self.current_roi.shape[1]

            # Click X position as fraction of RoI width
            hp_fraction = x / roi_width

            # Clamp to [0, 1]
            self.hp_percent = np.clip(hp_fraction, 0.0, 1.0)

            print(f"[HP-LABELER] Clicked at ({x}, {y}) → HP: {self.hp_percent*100:.1f}%")

            # Redraw with HP bar visualization
            self._redraw_display()

    def _redraw_display(self):
        """Redraw the display with current HP visualization."""
        if self.current_roi is None:
            return

        # Create display image (larger for visibility)
        display_h, display_w = 400, 600
        self.display_image = cv2.resize(self.current_roi, (display_w, display_h))

        # Draw HP bar overlay if HP% is set
        if self.hp_percent is not None:
            # Draw HP bar visualization (green = filled, red = empty)
            bar_height = 20
            bar_y = 10

            # Background (red = empty)
            cv2.rectangle(
                self.display_image,
                (10, bar_y),
                (display_w - 10, bar_y + bar_height),
                (0, 0, 255),  # Red background
                -1
            )

            # Foreground (green = filled HP)
            bar_width = int((display_w - 20) * self.hp_percent)
            cv2.rectangle(
                self.display_image,
                (10, bar_y),
                (10 + bar_width, bar_y + bar_height),
                (0, 255, 0),  # Green foreground
                -1
            )

            # Text overlay
            cv2.putText(
                self.display_image,
                f"HP: {self.hp_percent*100:.1f}%",
                (10, bar_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

        # Instructions
        instructions = [
            "Click on HP bar right edge to mark HP%",
            "Or press 0-9 for quick HP% (0%, 10%, ..., 100%)",
            "S = Skip | Q = Quit | U = Undo",
            f"Labeled: {self.labeled_count} | Skipped: {self.skipped_count}"
        ]

        y_offset = display_h - 120
        for i, text in enumerate(instructions):
            cv2.putText(
                self.display_image,
                text,
                (10, y_offset + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )

        cv2.imshow(self.window_name, self.display_image)

    def label_roi(self, roi: np.ndarray, bbox: Tuple[float, float, float, float], enemy_class: str = "enemy") -> Optional[float]:
        """
        Show RoI and wait for user to label HP%.

        Args:
            roi: Enemy RoI image [H, W, 3]
            bbox: Normalized bbox (x_center, y_center, width, height)
            enemy_class: Enemy class name

        Returns:
            HP percentage (0.0-1.0) or None if skipped
        """
        self.current_roi = roi.copy()
        self.current_bbox = bbox
        self.current_enemy_class = enemy_class
        self.click_position = None
        self.hp_percent = None

        # Initial display
        self._redraw_display()

        # Wait for input
        while True:
            key = cv2.waitKey(1) & 0xFF

            # Number keys: Quick HP percentage
            if ord('0') <= key <= ord('9'):
                digit = key - ord('0')
                self.hp_percent = digit / 10.0
                print(f"[HP-LABELER] Quick label: {self.hp_percent*100:.0f}%")
                self._redraw_display()

            # Enter/Space: Confirm label
            elif key in [13, 32]:  # Enter or Space
                if self.hp_percent is not None:
                    return self.hp_percent
                else:
                    print("[HP-LABELER] No HP% set! Click on HP bar or press 0-9.")

            # S: Skip
            elif key == ord('s') or key == ord('S'):
                self.skipped_count += 1
                print("[HP-LABELER] Skipped")
                return None

            # Q: Quit
            elif key == ord('q') or key == ord('Q'):
                print("[HP-LABELER] Quitting...")
                return None

            # U: Undo (remove last label)
            elif key == ord('u') or key == ord('U'):
                if self.labels:
                    removed = self.labels.pop()
                    self.labeled_count -= 1
                    print(f"[HP-LABELER] Undone: {removed['hp_percent']*100:.1f}%")

            # ESC: Quit
            elif key == 27:
                return None

    def save_label(self, roi: np.ndarray, hp_percent: float, enemy_class: str, bbox: Tuple[float, float, float, float]):
        """Save labeled RoI to disk."""
        timestamp = int(time.time() * 1000)
        filename = f"hp_label_{timestamp}_{self.labeled_count:05d}.npz"
        filepath = self.output_dir / filename

        # Save as compressed numpy archive
        np.savez_compressed(
            filepath,
            roi=roi,
            hp_percent=hp_percent,
            enemy_class=enemy_class,
            bbox=np.array(bbox),
            timestamp=timestamp
        )

        # Also save metadata
        self.labels.append({
            'filename': filename,
            'hp_percent': hp_percent,
            'enemy_class': enemy_class,
            'bbox': list(bbox),
            'timestamp': timestamp
        })

        self.labeled_count += 1
        print(f"[HP-LABELER] Saved: {filename} (HP: {hp_percent*100:.1f}%)")

    def save_metadata(self):
        """Save metadata JSON."""
        metadata_path = self.output_dir / "labels_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'total_labels': self.labeled_count,
                'total_skipped': self.skipped_count,
                'labels': self.labels
            }, f, indent=2)

        print(f"[HP-LABELER] Metadata saved: {metadata_path}")
        print(f"[HP-LABELER] Total labeled: {self.labeled_count}")
        print(f"[HP-LABELER] Total skipped: {self.skipped_count}")

    def run_from_video(self, video_path: str, yolo_model_path: str, sample_interval: float = 1.0):
        """
        Label HP bars from a recorded video.

        Args:
            video_path: Path to recorded gameplay video
            yolo_model_path: Path to YOLO model for enemy detection
            sample_interval: Seconds between samples
        """
        import torch

        # Load YOLO
        print(f"[HP-LABELER] Loading YOLO model: {yolo_model_path}")
        model = torch.hub.load('ultralytics/yolov8', 'custom', path=yolo_model_path, force_reload=False)
        model.conf = 0.3

        # Open video
        print(f"[HP-LABELER] Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[HP-LABELER] ERROR: Could not open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(fps * sample_interval)

        print(f"[HP-LABELER] Video: {total_frames} frames @ {fps:.1f} FPS")
        print(f"[HP-LABELER] Sampling every {sample_interval:.1f}s ({frame_interval} frames)")

        # Setup window
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample at intervals
                if frame_idx % frame_interval != 0:
                    frame_idx += 1
                    continue

                # Run YOLO
                results = model(frame, verbose=False)

                # Extract enemy RoIs
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        class_name = result.names[cls_id]

                        # Only process enemies
                        if 'enemy' not in class_name.lower():
                            continue

                        # Extract bbox
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        h, w = frame.shape[:2]

                        # Crop RoI
                        roi = frame[y1:y2, x1:x2]

                        if roi.size == 0:
                            continue

                        # Convert to normalized bbox
                        cx = (x1 + x2) / 2 / w
                        cy = (y1 + y2) / 2 / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        bbox = (cx, cy, bw, bh)

                        # Label this RoI
                        hp_percent = self.label_roi(roi, bbox, class_name)

                        if hp_percent is not None:
                            self.save_label(roi, hp_percent, class_name, bbox)
                        elif hp_percent is None and cv2.waitKey(1) & 0xFF == ord('q'):
                            # User quit
                            raise KeyboardInterrupt

                frame_idx += 1

                # Progress
                if frame_idx % 100 == 0:
                    progress = frame_idx / total_frames * 100
                    print(f"[HP-LABELER] Progress: {frame_idx}/{total_frames} ({progress:.1f}%)")

        except KeyboardInterrupt:
            print("[HP-LABELER] Interrupted by user")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_metadata()

    def run_from_bot(self, save_interval: float = 1.0, max_labels: int = 5000):
        """
        Label HP bars from live bot gameplay.

        Args:
            save_interval: Seconds between RoI captures
            max_labels: Stop after this many labels
        """
        # TODO: Integrate with bot_controller_v2
        # This would run the bot and periodically pause to label enemy RoIs
        print("[HP-LABELER] Live bot labeling not implemented yet")
        print("[HP-LABELER] Use run_from_video() instead")


def main():
    parser = argparse.ArgumentParser(description="HP Bar Labeling Tool")
    parser.add_argument('--video', type=str, help='Path to gameplay video')
    parser.add_argument('--yolo', type=str, default='yolo/best.pt', help='Path to YOLO model')
    parser.add_argument('--interval', type=float, default=1.0, help='Sampling interval (seconds)')
    parser.add_argument('--output', type=str, default='data/hp_labels', help='Output directory')

    args = parser.parse_args()

    labeler = HPLabeler(output_dir=args.output)

    if args.video:
        labeler.run_from_video(args.video, args.yolo, args.interval)
    else:
        print("Usage: python hp_labeler.py --video <video_path> --yolo <yolo_model_path>")
        print("Example: python hp_labeler.py --video recordings/gameplay.mp4 --yolo yolo/best.pt")


if __name__ == "__main__":
    main()
