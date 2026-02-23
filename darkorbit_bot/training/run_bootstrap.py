#!/usr/bin/env python
"""
Bootstrap YOLO Training Data Generator

Run this to build training data from scratch using FastSAM + LLM.

RECOMMENDED TWO-STEP WORKFLOW:

Step 1: CLEAN COLLECTION (no gallery pollution)
    python run_bootstrap.py --phase discovery --llm gemini --no-gallery --manual
    - Press F8 to capture screenshot when something interesting is on screen
    - FastSAM segments everything
    - Gemini classifies each crop WITHOUT seeing previous examples
    - No feedback loop from wrong classifications
    - After collection, manually review/clean data in LabelImg or similar

    Auto mode (without --manual):
    python run_bootstrap.py --phase discovery --llm gemini --no-gallery
    - Captures automatically every N frames

Step 2: GALLERY-ASSISTED (after manual cleanup)
    python run_bootstrap.py --phase discovery --llm gemini
    - Uses cleaned gallery as reference examples
    - Gemini matches new crops against known-good examples
    - Higher accuracy due to clean references

Phase 2: VALIDATION (after training initial YOLO)
    python run_bootstrap.py --phase validation --yolo-model path/to/model.pt
    - Compares SAM vs YOLO detections
    - Adds missed objects to training
    - Tracks YOLO accuracy

Phase 3: Auto-transition to YOLO-only when accuracy >= 85%
"""

import argparse
import sys
import time
import threading
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env file BEFORE any other imports
import os

# Manually load .env - most reliable method
env_path = "F:/dev/bot/.env"
print(f"[Bootstrap] Loading .env from: {env_path}")
try:
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
                print(f"[Bootstrap]   Set {key.strip()}=***")
except FileNotFoundError:
    print(f"[Bootstrap] ❌ .env file not found at {env_path}")
except Exception as e:
    print(f"[Bootstrap] ❌ Error loading .env: {e}")

# Verify
if os.environ.get('GOOGLE_API'):
    print(f"[Bootstrap] ✅ GOOGLE_API ready: {os.environ.get('GOOGLE_API')[:15]}...")
else:
    print("[Bootstrap] ❌ GOOGLE_API not set after loading .env")

from bootstrap_labeler import BootstrapLabeler


def main():
    parser = argparse.ArgumentParser(description='Bootstrap YOLO Training Data')
    parser.add_argument('--phase', type=str, default='discovery',
                       choices=['discovery', 'validation'],
                       help='Phase: discovery (no YOLO) or validation (compare with YOLO)')
    parser.add_argument('--output', type=str, default='data/bootstrap',
                       help='Output directory for training data')
    parser.add_argument('--monitor', type=int, default=1,
                       help='Monitor to capture')
    parser.add_argument('--interval', type=int, default=60,
                       help='Process every N frames (default: 60 = ~2s at 30fps)')
    parser.add_argument('--yolo-model', type=str, default=None,
                       help='YOLO model path (required for validation phase)')
    parser.add_argument('--fastsam-model', type=str, default='FastSAM-s.pt',
                       help='FastSAM model (FastSAM-s.pt or FastSAM-x.pt)')
    parser.add_argument('--segmenter', type=str, default='sam3',
                       choices=['sam3', 'sam2', 'fastsam'],
                       help='Segmenter: sam3 (best, needs sam3.pt), sam2, or fastsam (fastest)')
    parser.add_argument('--llm', type=str, default='local',
                       choices=['local', 'gemini'],
                       help='LLM mode: local (LM Studio, fast) or gemini (API, rate limited)')
    parser.add_argument('--local-url', type=str, default='http://localhost:1234',
                       help='LM Studio server URL')
    parser.add_argument('--duration', type=int, default=0,
                       help='Run for N seconds (0 = until Ctrl+C)')
    parser.add_argument('--no-gallery', action='store_true',
                       help='Disable gallery references (clean collection mode - Step 1)')
    parser.add_argument('--manual', action='store_true',
                       help='Manual mode: press F8 to capture screenshot instead of auto-capture')
    parser.add_argument('--padding', type=float, default=0.5,
                       help='Crop padding as fraction of bbox size (default: 0.5 = 50%%)')
    parser.add_argument('--points', type=int, default=16,
                       help='SAM3 point grid density per side (default: 16 = 256 points, use 24-32 for more masks)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='SAM3 batch size for point processing (default: 32, lower if OOM)')

    args = parser.parse_args()

    # Validation phase requires YOLO model
    if args.phase == 'validation' and not args.yolo_model:
        print("ERROR: --yolo-model required for validation phase")
        sys.exit(1)

    # Initialize screen capture
    try:
        from detection.detector import ScreenCapture
    except ImportError:
        from darkorbit_bot.detection.detector import ScreenCapture

    print(f"\n{'='*60}")
    print(f"  Bootstrap YOLO Training Data Generator")
    print(f"{'='*60}")
    print(f"  Phase: {args.phase.upper()}")
    print(f"  Segmenter: {args.segmenter.upper()}")
    if args.segmenter == 'sam3':
        print(f"    Point grid: {args.points}x{args.points} = {args.points**2} points")
        print(f"    Batch size: {args.batch_size} (lower if OOM)")
    print(f"  LLM: {args.llm.upper()}" + (f" ({args.local_url})" if args.llm == 'local' else ""))
    print(f"  Output: {args.output}")
    print(f"  Monitor: {args.monitor}")
    print(f"  Sample interval: every {args.interval} frames")
    print(f"  Crop padding: {args.padding:.0%}")
    print(f"  Gallery: {'DISABLED (clean mode)' if args.no_gallery else 'enabled'}")
    print(f"  Mode: {'MANUAL (F8 to capture)' if args.manual else 'auto-capture'}")
    print(f"{'='*60}\n")

    # Use single-shot mode (continuous=False) - DXCam crashes if processing takes too long
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

    # Initialize YOLO if validation phase
    yolo_detector = None
    if args.phase == 'validation':
        try:
            from detection.detector import GameDetector
        except ImportError:
            from darkorbit_bot.detection.detector import GameDetector

        yolo_detector = GameDetector(args.yolo_model, confidence_threshold=0.3)
        print(f"[Bootstrap] YOLO loaded: {args.yolo_model}")

    # Initialize bootstrap labeler
    labeler = BootstrapLabeler(
        output_dir=args.output,
        phase=args.phase,
        llm_mode=args.llm,
        local_url=args.local_url,
        fastsam_model=args.fastsam_model,
        segmenter=args.segmenter,
        sample_interval=args.interval,
        use_gallery=not args.no_gallery,
        crop_padding=args.padding,
        sam3_points_per_side=args.points,
        sam3_batch_size=args.batch_size
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

            # Manual mode: wait for F8, then capture
            if args.manual:
                if not capture_requested.is_set():
                    time.sleep(0.1)  # Idle wait - minimal CPU
                    continue
                capture_requested.clear()
                manual_captures += 1
                print(f"\n[Capture #{manual_captures}] Capturing and processing...")
                frame = screen.capture()
                if frame is None:
                    print("  Failed to capture!")
                    continue
            else:
                # Auto mode: capture continuously
                frame = screen.capture()
                if frame is None:
                    time.sleep(0.01)
                    continue

            frame_count += 1

            # Get YOLO detections if in validation phase
            yolo_dets = []
            if yolo_detector:
                raw_dets = yolo_detector.detect_frame(frame)
                for d in raw_dets:
                    yolo_dets.append({
                        'class_name': getattr(d, 'class_name', 'unknown'),
                        'confidence': getattr(d, 'confidence', 0.5),
                        'bbox': [getattr(d, 'x1', 0), getattr(d, 'y1', 0),
                                getattr(d, 'x2', 0) - getattr(d, 'x1', 0),
                                getattr(d, 'y2', 0) - getattr(d, 'y1', 0)]
                    })

            # Queue for processing (force=True in manual mode to bypass interval check)
            if args.manual:
                labeler.queue_frame(frame, yolo_dets)
            else:
                labeler.check_and_queue(frame, yolo_dets)

            # Status update every 5 seconds
            now = time.time()
            if now - last_status_time >= 5:
                stats = labeler.get_stats()
                fps = frame_count / (now - start_time)
                print(f"[{now-start_time:.0f}s] Frames: {frame_count} ({fps:.1f} fps) | "
                      f"Processed: {stats['frames_processed']} | "
                      f"Segments: {stats['segments_found']} | "
                      f"Labels: {stats['objects_labeled']} | "
                      f"Classes: {stats['total_classes']}")

                if args.phase == 'validation':
                    matches = stats.get('yolo_matches', 0)
                    misses = stats.get('yolo_misses', 0)
                    total = matches + misses
                    if total > 0:
                        print(f"      YOLO accuracy: {matches}/{total} ({matches/total:.1%})")

                last_status_time = now

            # Check for phase transition
            if labeler.should_transition_to_yolo():
                print("\n" + "="*60)
                print("  🎉 YOLO is ready! Switch to AutoLabeler (Phase 3)")
                print("="*60)
                break

            # Small delay to not hammer CPU
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nStopping...")

    labeler.stop()

    print(f"\nDone! Training data saved to: {args.output}")
    print(f"To train YOLO: yolo train data={args.output}/dataset.yaml epochs=100")


if __name__ == "__main__":
    main()
