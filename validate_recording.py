#!/usr/bin/env python3
"""Quick script to validate shadow recording files."""

import pickle
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python validate_recording.py <path_to_recording.pkl>")
    sys.exit(1)

filepath = sys.argv[1]

try:
    print(f"Loading {filepath}...")
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    print(f"\nValid recording file!")
    print(f"  Demos: {len(data['demos'])}")

    meta = data['metadata']
    print(f"\nMetadata:")
    print(f"  Total updates: {meta.get('total_updates', 'N/A')}")
    print(f"  Timestamp: {meta.get('timestamp', 'N/A')}")
    print(f"  Frame format: {meta.get('frame_format', 'raw numpy')}")

    if 'stats' in meta:
        stats = meta['stats']
        print(f"\nTraining Stats:")
        print(f"  Avg loss: {stats.get('avg_loss', 0):.4f}")
        print(f"  Avg position error: {stats.get('avg_pos_error', 0):.3f}")
        print(f"  Avg click accuracy: {stats.get('avg_click_accuracy', 0):.1%}")

    if len(data['demos']) > 0:
        demo = data['demos'][0]
        print(f"\nSample demo keys: {list(demo.keys())}")
        print(f"  Tracked objects: {len(demo.get('tracked_objects', []))}")
        print(f"  Human target ID: {demo.get('human_target_id', 'N/A')}")
        print(f"  Human mode: {demo.get('human_mode', 'N/A')}")

        # Check frame format
        frame = demo.get('frame')
        if frame is not None:
            if isinstance(frame, bytes):
                print(f"  Frame: JPEG compressed ({len(frame)} bytes)")
            else:
                print(f"  Frame: Raw numpy array {frame.shape if hasattr(frame, 'shape') else 'N/A'}")

    print(f"\nRecording is ready for offline training!")
    print(f"  Run: python -m darkorbit_bot.v2.training.train_full --data data/recordings")

except Exception as e:
    print(f"\nError loading recording: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
