"""
Auto-capture screenshots for YOLO training data.
Takes screenshots every N seconds and saves to dataset folder.

Usage:
    python auto_screenshot.py --output "F:/dev/bot/yolo/training_screenshots" --interval 2
"""

import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import mss
import mss.tools


def capture_screenshots(output_dir: str, interval: float = 2.0, monitor: int = 1):
    """
    Capture screenshots at regular intervals.

    Args:
        output_dir: Directory to save screenshots
        interval: Seconds between screenshots
        monitor: Monitor to capture (1=primary, 2=secondary, 0=all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("  AUTO SCREENSHOT CAPTURE")
    print("="*60)
    print(f"Output: {output_path}")
    print(f"Interval: {interval} seconds")
    print(f"Monitor: {monitor}")
    print("\nPress Ctrl+C to stop")
    print("-"*60)

    with mss.mss() as sct:
        # Get monitor
        if monitor == 0:
            mon = sct.monitors[0]  # All monitors
        elif monitor < len(sct.monitors):
            mon = sct.monitors[monitor]
        else:
            print(f"Monitor {monitor} not found, using primary")
            mon = sct.monitors[1]

        print(f"Capturing: {mon['width']}x{mon['height']} at ({mon['left']}, {mon['top']})")
        print("\nStarting capture in 3 seconds...")
        time.sleep(3)

        screenshot_count = 0
        start_time = time.time()

        try:
            while True:
                # Capture
                screenshot = sct.grab(mon)

                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"capture_{timestamp}.png"
                filepath = output_path / filename

                # Save
                mss.tools.to_png(screenshot.rgb, screenshot.size, output=str(filepath))

                screenshot_count += 1
                elapsed = time.time() - start_time
                rate = screenshot_count / elapsed if elapsed > 0 else 0

                print(f"\r{screenshot_count} screenshots | {rate:.2f}/sec | Latest: {filename}", end="", flush=True)

                # Wait for next interval
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print("STOPPED")
            print("="*60)
            print(f"Total screenshots: {screenshot_count}")
            print(f"Saved to: {output_path}")
            print(f"\nNext steps:")
            print(f"1. Upload images to Roboflow")
            print(f"2. Label the objects")
            print(f"3. Export as YOLOv11 format")
            print(f"4. Train with train_detect.py")


def main():
    parser = argparse.ArgumentParser(description='Auto-capture screenshots for training')
    parser.add_argument('--output', type=str,
                       default='F:/dev/bot/yolo/training_screenshots',
                       help='Output directory for screenshots')
    parser.add_argument('--interval', type=float, default=2.0,
                       help='Seconds between screenshots (default: 2)')
    parser.add_argument('--monitor', type=int, default=1,
                       help='Monitor to capture: 0=all, 1=primary, 2=secondary (default: 1)')

    args = parser.parse_args()

    capture_screenshots(
        output_dir=args.output,
        interval=args.interval,
        monitor=args.monitor
    )


if __name__ == '__main__':
    main()
