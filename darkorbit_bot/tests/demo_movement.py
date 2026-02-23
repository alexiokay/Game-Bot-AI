"""
DarkOrbit Bot - Movement Demo
Test YOUR movement style by watching the mouse move like you!

This demo:
1. Moves to random points on screen using YOUR profile
2. Shows how the bot will move when running
3. No YOLO/detection needed - just movement testing
"""

import os
import sys
import time
import random
import ctypes
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from movement.generator import MovementProfile, MovementGenerator, MouseController


def demo_movement():
    print("\n" + "="*60)
    print("  MOVEMENT DEMO - Watch the bot move like YOU!")
    print("="*60)
    
    # Load YOUR profile
    data_dir = Path(__file__).parent.parent / "data"
    profile_path = data_dir / "my_movement_profile.json"
    
    if not profile_path.exists():
        print(f"\n❌ Profile not found: {profile_path}")
        print("   Run analyze_patterns.py first!")
        return
    
    profile = MovementProfile.load(str(profile_path))
    
    print("\n📊 YOUR movement profile loaded:")
    print(f"   Speed: {profile.speed_mean:.2f} ± {profile.speed_std:.2f}")
    print(f"   Curve factor: {profile.curve_factor_mean:.2f}")
    print(f"   Deceleration: {profile.deceleration_ratio:.2f}x")
    
    # Initialize
    generator = MovementGenerator(profile)
    
    user32 = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
    
    controller = MouseController(screen_w, screen_h)
    
    print("\n" + "="*60)
    print("  🖱️  MOUSE DEMO")
    print("="*60)
    print("""
The mouse will move to several random points using YOUR style.

Controls:
  - Watch how it curves and decelerates
  - Press Ctrl+C to stop anytime
  
""")
    
    input("Press ENTER to start...")
    
    print("\n⏳ Starting in 3 seconds...")
    print("   Move your hands away from the mouse!")
    time.sleep(3)
    
    # Demo movements
    num_moves = 15
    
    # Start from current position (normalized)
    current_pos = (0.5, 0.5)  # Center of screen
    controller.move_to(current_pos[0], current_pos[1])
    time.sleep(0.5)
    
    print(f"\n🎯 Making {num_moves} movements...\n")
    
    try:
        for i in range(num_moves):
            # Random target (stay in safe area)
            target = (
                random.uniform(0.15, 0.85),
                random.uniform(0.15, 0.85)
            )
            
            # Generate path with YOUR style - force longer duration for demo
            path = generator.generate_path(current_pos, target, target_duration=random.uniform(1.5, 3.0))
            
            # Sometimes add overshoot
            if random.random() < 0.3:
                path = generator.add_overshoot(path, probability=1.0)
            
            print(f"  Move {i+1}: ({current_pos[0]:.2f}, {current_pos[1]:.2f}) → ({target[0]:.2f}, {target[1]:.2f})")
            print(f"         Duration: {path[-1]['time']:.2f}s, Points: {len(path)}")
            
            # Execute movement
            controller.execute_path(path)
            
            # Click (optional)
            if random.random() < 0.5:
                click_timing = generator.generate_click_timing()
                time.sleep(click_timing["pre_delay"])
                controller.click(hold_duration=click_timing["hold_duration"])
                print(f"         *click*")
            
            current_pos = target
            
            # Pause between moves
            time.sleep(random.uniform(0.3, 0.8))
        
        print("\n✅ Demo complete!")
        print("   Did the movement look natural?")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    
    print("\n" + "="*60)


def demo_specific_path():
    """Demo a specific path for detailed viewing"""
    print("\n" + "="*60)
    print("  SPECIFIC PATH DEMO")
    print("="*60)
    
    # Load profile
    data_dir = Path(__file__).parent.parent / "data"
    profile_path = data_dir / "my_movement_profile.json"
    
    if not profile_path.exists():
        print("\n❌ Profile not found")
        return
    
    profile = MovementProfile.load(str(profile_path))
    generator = MovementGenerator(profile)
    
    print("\nEnter coordinates (0.0 to 1.0):")
    print("  0.0 = left/top edge")
    print("  0.5 = center")
    print("  1.0 = right/bottom edge")
    
    try:
        start_x = float(input("\nStart X [0.3]: ").strip() or "0.3")
        start_y = float(input("Start Y [0.3]: ").strip() or "0.3")
        end_x = float(input("End X [0.7]: ").strip() or "0.7")
        end_y = float(input("End Y [0.7]: ").strip() or "0.7")
    except ValueError:
        print("Invalid input, using defaults")
        start_x, start_y = 0.3, 0.3
        end_x, end_y = 0.7, 0.7
    
    start = (start_x, start_y)
    end = (end_x, end_y)
    
    # Generate path
    path = generator.generate_path(start, end)
    
    print(f"\n📍 Path generated:")
    print(f"   From: ({start[0]:.2f}, {start[1]:.2f})")
    print(f"   To: ({end[0]:.2f}, {end[1]:.2f})")
    print(f"   Duration: {path[-1]['time']:.3f}s")
    print(f"   Points: {len(path)}")
    
    # Execute?
    execute = input("\n🖱️  Execute this path? [y/N]: ").strip().lower()
    
    if execute == 'y':
        user32 = ctypes.windll.user32
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
        controller = MouseController(screen_w, screen_h)
        
        print("\n⏳ Moving in 2 seconds...")
        time.sleep(2)
        
        # Move to start
        controller.move_to(start[0], start[1])
        time.sleep(0.3)
        
        # Execute path
        controller.execute_path(path)
        
        print("✅ Done!")


def main():
    print("\n" + "="*60)
    print("  DARKORBIT BOT - MOVEMENT TESTER")
    print("="*60)
    print("""
Options:
  1. Random movement demo (15 random points)
  2. Custom path demo (you specify coordinates)
  3. Speed comparison (slow vs fast)
""")
    
    choice = input("Choice [1/2/3, default=1]: ").strip()
    
    if choice == "2":
        demo_specific_path()
    elif choice == "3":
        demo_speed_comparison()
    else:
        demo_movement()


def demo_speed_comparison():
    """Compare slow and fast movements"""
    print("\n" + "="*60)
    print("  SPEED COMPARISON DEMO")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent / "data"
    profile_path = data_dir / "my_movement_profile.json"
    
    if not profile_path.exists():
        print("\n❌ Profile not found")
        return
    
    profile = MovementProfile.load(str(profile_path))
    generator = MovementGenerator(profile)
    
    user32 = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
    controller = MouseController(screen_w, screen_h)
    
    print("\nThis will show 3 movements with different speeds.")
    input("Press ENTER to start...")
    
    print("\n⏳ Starting in 2 seconds...")
    time.sleep(2)
    
    start = (0.2, 0.5)
    end = (0.8, 0.5)
    
    # Normal speed
    print("\n1️⃣ Normal speed (your average)...")
    controller.move_to(start[0], start[1])
    time.sleep(0.3)
    path = generator.generate_path(start, end)
    controller.execute_path(path, speed_multiplier=1.0)
    time.sleep(1)
    
    # Slow
    print("2️⃣ Slow (0.5x)...")
    controller.move_to(start[0], start[1])
    time.sleep(0.3)
    path = generator.generate_path(start, end)
    controller.execute_path(path, speed_multiplier=0.5)
    time.sleep(1)
    
    # Fast
    print("3️⃣ Fast (1.5x)...")
    controller.move_to(start[0], start[1])
    time.sleep(0.3)
    path = generator.generate_path(start, end)
    controller.execute_path(path, speed_multiplier=1.5)
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
