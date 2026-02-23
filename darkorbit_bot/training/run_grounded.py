#!/usr/bin/env python
"""
Grounded SAM 3 Training Data Generator

Uses text-guided detection (Florence-2 or Grounding DINO) instead of
blind auto-segmentation. Much cleaner results with polygon masks!

Usage:
    # Manual mode - press F8 to capture
    python run_grounded.py --manual

    # Auto mode - capture every N frames
    python run_grounded.py --interval 60

    # Use Grounding DINO instead of Florence-2
    python run_grounded.py --detector dino --manual

    # Skip SAM3 refinement (faster, boxes only)
    python run_grounded.py --no-sam --manual

Comparison with bootstrap_labeler.py:
    - bootstrap_labeler: SAM3 auto-segments EVERYTHING, LLM classifies each
    - run_grounded: Text prompts find SPECIFIC objects, no garbage

The ontology (text prompts -> classes) is defined in grounded_labeler.py
You can customize it for your specific game objects.
"""

import argparse
import sys
import time
import threading
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from grounded_labeler import GroundedLabeler


def main():
    parser = argparse.ArgumentParser(description='Grounded SAM 3 Training Data Generator')
    parser.add_argument('--output', type=str, default='data/grounded',
                       help='Output directory for training data')
    parser.add_argument('--monitor', type=int, default=1,
                       help='Monitor to capture')
    parser.add_argument('--interval', type=int, default=60,
                       help='Capture every N frames in auto mode (default: 60)')
    parser.add_argument('--detector', type=str, default='florence',
                       choices=['florence', 'dino'],
                       help='Detector: florence (Florence-2) or dino (Grounding DINO)')
    parser.add_argument('--no-sam', action='store_true',
                       help='Skip SAM3 mask refinement (faster, boxes only instead of polygons)')
    parser.add_argument('--no-sam2', action='store_true',
                       help='(Deprecated) Same as --no-sam')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Minimum detection confidence (default: 0.3)')
    parser.add_argument('--manual', action='store_true',
                       help='Manual mode: press F8 to capture instead of auto-capture')
    parser.add_argument('--duration', type=int, default=0,
                       help='Run for N seconds (0 = until Ctrl+C)')

    args = parser.parse_args()

    # Initialize screen capture
    try:
        from detection.detector import ScreenCapture
    except ImportError:
        from darkorbit_bot.detection.detector import ScreenCapture

    # Handle both --no-sam and deprecated --no-sam2
    skip_sam = args.no_sam or args.no_sam2

    print(f"\n{'='*60}")
    print(f"  Grounded SAM 3 Training Data Generator")
    print(f"{'='*60}")
    print(f"  Detector: {args.detector.upper()}")
    print(f"  SAM3 refinement: {'disabled (boxes only)' if skip_sam else 'enabled (polygon masks)'}")
    print(f"  Confidence: {args.confidence}")
    print(f"  Output: {args.output}")
    print(f"  Monitor: {args.monitor}")
    print(f"  Mode: {'MANUAL (F8 to capture)' if args.manual else f'auto every {args.interval} frames'}")
    print(f"{'='*60}\n")

    # Use single-shot mode for reliability
    screen = ScreenCapture(monitor_index=args.monitor, continuous=False)

    # Manual capture mode setup
    capture_requested = threading.Event()
    if args.manual:
        try:
            import keyboard
            keyboard.add_hotkey('F8', lambda: capture_requested.set())
            print("Press F8 to capture a screenshot, Ctrl+C to stop\n")
        except ImportError:
            print("ERROR: Manual mode requires 'keyboard' package. Run: pip install keyboard")
            sys.exit(1)

    # Initialize grounded labeler
    labeler = GroundedLabeler(
        output_dir=args.output,
        use_florence=(args.detector == 'florence'),
        confidence_threshold=args.confidence,
        use_sam3_refinement=not skip_sam
    )
    labeler.start()

    if not args.manual:
        print("\nCapturing... Press Ctrl+C to stop\n")

    frame_count = 0
    manual_captures = 0
    start_time = time.time()
    last_status_time = start_time

    try:
        while True:
            # Check duration limit
            if args.duration > 0 and time.time() - start_time > args.duration:
                print(f"\nDuration limit ({args.duration}s) reached")
                break

            # Manual mode: wait for F8
            if args.manual:
                if not capture_requested.is_set():
                    time.sleep(0.1)
                    continue
                capture_requested.clear()
                manual_captures += 1
                print(f"\n[Capture #{manual_captures}] Capturing and processing...")
                frame = screen.capture()
                if frame is None:
                    print("  Failed to capture!")
                    continue
                labeler.queue_frame(frame)
            else:
                # Auto mode
                frame = screen.capture()
                if frame is None:
                    time.sleep(0.01)
                    continue

                frame_count += 1

                # Only process every N frames
                if frame_count % args.interval == 0:
                    labeler.queue_frame(frame)

            # Status update every 10 seconds
            now = time.time()
            if now - last_status_time >= 10:
                stats = labeler.get_stats()
                elapsed = now - start_time
                print(f"[{elapsed:.0f}s] Processed: {stats['frames_processed']} | "
                      f"Detections: {stats['detections_found']} | "
                      f"Labeled: {stats['objects_labeled']} | "
                      f"Classes: {stats['total_classes']}")
                last_status_time = now

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nStopping...")

    labeler.stop()

    print(f"\nDone! Training data saved to: {args.output}")
    print(f"To train YOLO segmentation: yolo segment train data={args.output}/dataset.yaml epochs=100")
    print(f"To train YOLO detection:    yolo detect train data={args.output}/dataset.yaml epochs=100")


if __name__ == "__main__":
    main()
