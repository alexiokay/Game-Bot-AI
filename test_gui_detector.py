#!/usr/bin/env python3
"""
Test script for YOLO-based GUI detector and minimap position extraction.

Usage:
    python test_gui_detector.py

Press 'q' to quit, 's' to save a frame with GUI regions highlighted.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add bot to path
sys.path.insert(0, str(Path(__file__).parent))

from darkorbit_bot.v2.perception.gui_detector import GUIDetector
from darkorbit_bot.detection.detector import ScreenCapture, GameDetector


def main():
    print("GUI Detector Test (YOLO-based)")
    print("=" * 60)
    print("Starting screen capture and YOLO detection...")
    print("Press 'q' to quit, 's' to save frame with GUI visualization")
    print("=" * 60)

    # Initialize YOLO detector
    model_path = "F:/dev/bot/best.pt"  # Adjust path as needed
    detector = GameDetector(model_path, confidence_threshold=0.4)
    print(f"Loaded YOLO model: {model_path}")
    print(f"Classes: {detector.class_names}")

    # Initialize screen capture (monitor 1)
    screen = ScreenCapture(monitor_index=1)

    # Get screen dimensions
    monitor = screen.monitor
    screen_width = monitor['width']
    screen_height = monitor['height']

    print(f"Screen: {screen_width}x{screen_height}")

    # Initialize GUI detector
    gui_detector = GUIDetector(
        screen_width=screen_width,
        screen_height=screen_height
    )

    print(f"\nGUI classes to mask: {gui_detector.GUI_CLASSES}")
    print("\nDetecting GUI elements...")

    frame_count = 0
    minimap_detected = False
    gui_count = 0

    while True:
        # Capture frame
        frame = screen.capture()
        if frame is None:
            continue

        frame_count += 1

        # Run YOLO detection
        detections = detector.detect_frame(frame)

        # Update GUI detector with YOLO detections
        gui_detector.update_from_detections(detections)

        # Count GUI elements detected
        if len(gui_detector.gui_regions) != gui_count:
            gui_count = len(gui_detector.gui_regions)
            print(f"Detected {gui_count} GUI elements: {[r.name for r in gui_detector.gui_regions]}")

        # Check if minimap was detected
        if gui_detector.minimap_detector.minimap_region is not None:
            if not minimap_detected:
                print("✅ Minimap detected!")
                minimap_detected = True
                region = gui_detector.minimap_detector.minimap_region
                print(f"   Position: ({region.x:.2f}, {region.y:.2f})")
                print(f"   Size: {region.width:.2f}x{region.height:.2f}")

        # Extract map position
        map_x, map_y = gui_detector.get_map_position(frame)

        # Visualize GUI regions
        vis_frame = gui_detector.visualize(frame)

        # Add map position text
        cv2.putText(vis_frame, f"Map Pos: ({map_x:.2f}, {map_y:.2f})",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Add GUI count
        cv2.putText(vis_frame, f"GUI Elements: {len(gui_detector.gui_regions)}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Test some click positions
        test_positions = [
            (0.5, 0.5, "Center"),
            (0.5, 0.05, "Top bar"),
            (0.9, 0.2, "Right panel"),
            (0.05, 0.8, "Left panel"),
        ]

        y_offset = 90
        for tx, ty, name in test_positions:
            allowed = gui_detector.is_click_allowed(tx, ty)
            color = (0, 255, 0) if allowed else (0, 0, 255)
            status = "OK" if allowed else "BLOCKED"
            cv2.putText(vis_frame, f"{name}: {status}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25

            # Draw dot at test position
            px = int(tx * screen_width)
            py = int(ty * screen_height)
            cv2.circle(vis_frame, (px, py), 5, color, -1)

        # Resize for display (if too large)
        display_frame = vis_frame
        if screen_width > 1920:
            scale = 1920 / screen_width
            new_width = int(screen_width * scale)
            new_height = int(screen_height * scale)
            display_frame = cv2.resize(vis_frame, (new_width, new_height))

        # Show frame
        cv2.imshow("GUI Detector Test", display_frame)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            output_path = Path("gui_detection_test.png")
            cv2.imwrite(str(output_path), vis_frame)
            print(f"\n✅ Saved visualization to {output_path}")

    cv2.destroyAllWindows()
    print("\nTest complete!")


if __name__ == "__main__":
    main()
