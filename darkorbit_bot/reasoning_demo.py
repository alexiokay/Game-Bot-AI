"""
DarkOrbit Bot - Reasoning Core Demo

Run this to see the full pipeline in action:
1. Screen capture
2. YOLO detection
3. State building
4. Context detection (Passive/Aggressive)
5. Policy network inference

NOTE: The policy network has random weights until you train it!
This demo shows the structure is working.

Usage:
    python reasoning_demo.py --model path/to/best.pt
"""

import time
import sys
import argparse
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

# Local imports
from detection.detector import GameDetector, ScreenCapture
from reasoning.filters import create_filters, BufferFrame
from reasoning.context_detector import ContextDetector
from reasoning.state_builder import StateBuilder, StateSequenceBuilder, PlayerState
from reasoning.policy_network import create_policy, DualHeadPolicy


def run_demo(model_path: str, monitor: int = 1, show_visual: bool = False):
    """
    Run the full reasoning pipeline demo.
    """
    print("\n" + "="*60)
    print("  🧠 REASONING CORE DEMO")
    print("="*60)
    
    # 1. Initialize Vision (YOLO)
    print("\n📷 Loading Vision Module...")
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    detector = GameDetector(model_path, confidence_threshold=0.4)
    screen = ScreenCapture(monitor_index=monitor)
    
    # 2. Initialize Reasoning Core
    print("🧠 Initializing Reasoning Core...")
    
    # Filters
    filters = create_filters({
        'buffer_seconds': 30.0,
        'save_before_kill': 10.0,
        'smooth_sigma': 3.0
    })
    
    # Context detector
    context = ContextDetector()
    
    # State builder (matches our YOLO model's classes)
    state_builder = StateBuilder(max_objects=20)
    sequence_builder = StateSequenceBuilder(sequence_length=50, state_builder=state_builder)
    
    # Policy network (random weights - not trained yet!)
    print("🤖 Creating Policy Network (untrained)...")
    state_size = state_builder.get_state_size()
    policy = create_policy(state_size=state_size, device="cuda")
    
    print(f"\n✅ Pipeline ready!")
    print(f"   State vector size: {state_size}")
    print(f"   Sequence length: 50 frames")
    print(f"   Policy: Dual-head Bi-LSTM")
    
    print("\n" + "-"*60)
    print("🎮 Starting real-time demo... (Ctrl+C to stop)")
    print("-"*60)
    print("\nLegend:")
    print("  Mode: PASSIVE (looting) / AGGRESSIVE (combat)")
    print("  Action: What the bot WOULD do (random until trained)")
    print()
    
    # Track mouse for velocity
    prev_mouse = None
    frame_count = 0
    
    try:
        while True:
            start = time.time()
            
            # 1. Capture frame
            frame = screen.capture()
            
            # 2. Run YOLO detection
            detections = detector.detect_frame(frame)
            
            # 3. Get mouse position (simulated from center for demo)
            # In real use, this comes from pynput
            import ctypes
            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
            pt = POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            mouse_x, mouse_y = pt.x, pt.y
            
            # Calculate velocity
            velocity = 0.0
            if prev_mouse:
                dx = mouse_x - prev_mouse[0]
                dy = mouse_y - prev_mouse[1]
                velocity = np.sqrt(dx**2 + dy**2) * 30  # Approx FPS
            prev_mouse = (mouse_x, mouse_y)
            
            # 4. Detect context (passive vs aggressive)
            context_state = context.detect(mouse_x, mouse_y, detections)
            mode = context_state.mode
            
            # 5. Build player state
            player = PlayerState(
                health=filters['health'].update(detections),
                mouse_x=mouse_x / 1920,
                mouse_y=mouse_y / 1080,
                velocity_x=velocity,
                velocity_y=0,
                mode=mode
            )
            
            # 6. Build state sequence
            state_seq = sequence_builder.add_frame(detections, player)
            
            # 7. Update kill filter
            events = filters['kill_filter'].update(detections)
            
            # 8. If we have enough sequence, run policy
            action = None
            if state_seq is not None:
                action = policy.get_action(state_seq, mode=mode)
            
            # FPS
            elapsed = time.time() - start
            fps = 1 / elapsed if elapsed > 0 else 0
            
            # Print status
            frame_count += 1
            if frame_count % 15 == 0:  # Every ~0.5 seconds
                enemies = sum(1 for d in detections if d.class_name in 
                            ['Devo', 'Lordakia', 'Mordon', 'Saimon', 'Sibelon', 'Struener'])
                boxes = sum(1 for d in detections if d.class_name == 'BonusBox')
                
                status = f"FPS: {fps:.0f} | Mode: {mode:10} | "
                status += f"Enemies: {enemies} | Boxes: {boxes} | "
                status += f"Health: {player.health:.0%} | "
                
                if events['kill']:
                    status += "🎯 KILL DETECTED! "
                    
                if action:
                    if mode == "AGGRESSIVE":
                        status += f"Aim: ({action['aim_x']:.0f},{action['aim_y']:.0f})"
                    else:
                        status += f"Move: ({action['move_x']:.0f},{action['move_y']:.0f})"
                else:
                    status += "(warming up...)"
                    
                print(f"\r{status}    ", end="")
                
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\n✅ Demo stopped.")
        print("\n📝 To train the policy network:")
        print("   1. Record gameplay with the filtered recorder")
        print("   2. Press F9 to start training")
        print("   3. The bot will learn from your successful plays!")


def main():
    parser = argparse.ArgumentParser(description='Reasoning Core Demo')
    parser.add_argument('--model', type=str, default='F:/dev/bot/best.pt',
                       help='Path to YOLO model')
    parser.add_argument('--monitor', type=int, default=1,
                       help='Monitor to capture (1=primary)')
    
    args = parser.parse_args()
    
    run_demo(args.model, args.monitor)


if __name__ == "__main__":
    main()
