"""
DarkOrbit Bot - Pattern Analyzer
Analyzes your recorded movements to extract YOUR personal movement style.

Usage:
    python analyze_patterns.py

This reads all recordings from data/recordings/ and outputs your movement profile.
Extracts:
- Movement speed, curves, deceleration
- Click hold durations (mouse down -> up timing)
- Pre/post click delays
- Double-click patterns
"""

import json
import os
import numpy as np
from pathlib import Path


class MovementAnalyzer:
    def __init__(self):
        self.all_movements = []  # Movement segments between clicks
        self.all_clicks = []     # Click data
        self.all_speeds = []     # Speed samples

        # Click timing data
        self.click_hold_durations = []  # Time between mouse down and mouse up
        self.pre_click_pauses = []      # Time between last movement and click
        self.post_click_pauses = []     # Time between click release and next movement
        self.double_click_intervals = []  # Time between consecutive clicks

        # Keyboard timing data - for humanizing key presses
        self.key_hold_durations = {}    # key_name -> [hold_duration, ...]
        self.key_intervals = {}         # key_name -> [interval_between_presses, ...]
        self.key_press_counts = {}      # key_name -> count

        self.profile = {}
    
    def load_recording(self, filepath):
        """Load a single recording file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        events = data['events']
        metadata = data.get('metadata', {})

        print(f"Loading: {filepath.name if hasattr(filepath, 'name') else os.path.basename(str(filepath))}")
        duration = metadata.get('duration_seconds', 0)
        total_events = metadata.get('total_input_events', metadata.get('total_events', len(events)))
        print(f"  Duration: {duration:.1f}s, Events: {total_events}")

        # Extract movement segments (movements between clicks)
        current_segment = []

        # Track click timing
        pending_click_down = {}  # button -> {time, last_move_time}
        last_click_release_time = None
        last_move_time = None
        last_click_time = None

        # Track keyboard timing
        pending_key_down = {}   # key_name -> down_time
        last_key_time = {}      # key_name -> last_press_time

        for event in events:
            if event['type'] == 'move':
                # Handle both old format (x_rel) and new format (x_game)
                x = event.get('x_game', event.get('x_rel', 0))
                y = event.get('y_game', event.get('y_rel', 0))

                current_segment.append({
                    'time': event['time'],
                    'x': x,
                    'y': y
                })
                last_move_time = event['time']

                # Track post-click pause (time from click release to next movement)
                if last_click_release_time is not None:
                    pause = event['time'] - last_click_release_time
                    if 0 < pause < 1.0:  # Reasonable pause range
                        self.post_click_pauses.append(pause)
                    last_click_release_time = None

            elif event['type'] == 'click':
                button = event.get('button', 'left')
                pressed = event.get('pressed', True)

                if pressed:  # Mouse DOWN
                    # Track pre-click pause (time from last movement to click)
                    if last_move_time is not None:
                        pause = event['time'] - last_move_time
                        if 0 < pause < 1.0:  # Reasonable pause range
                            self.pre_click_pauses.append(pause)

                    # Track double-click intervals
                    if last_click_time is not None:
                        interval = event['time'] - last_click_time
                        if interval < 0.5:  # Double-click threshold
                            self.double_click_intervals.append(interval)

                    pending_click_down[button] = {
                        'time': event['time'],
                        'last_move': last_move_time
                    }
                    last_click_time = event['time']

                    # Click = end of segment
                    if len(current_segment) > 5:  # Minimum 5 points
                        self.all_movements.append(current_segment)
                    current_segment = []

                    x = event.get('x_game', event.get('x_rel', 0))
                    y = event.get('y_game', event.get('y_rel', 0))

                    self.all_clicks.append({
                        'time': event['time'],
                        'x': x,
                        'y': y,
                        'button': button
                    })

                else:  # Mouse UP
                    if button in pending_click_down:
                        down_info = pending_click_down.pop(button)
                        hold_duration = event['time'] - down_info['time']

                        # Filter reasonable hold durations (10ms to 2 seconds)
                        if 0.01 < hold_duration < 2.0:
                            self.click_hold_durations.append(hold_duration)

                    last_click_release_time = event['time']

            elif event['type'] == 'key':
                # Keyboard event - track timing for humanization
                key_name = event.get('key', '').lower()
                pressed = event.get('pressed', True)

                # Normalize key names (ctrl_l, ctrl_r -> ctrl)
                if key_name in ['ctrl_l', 'ctrl_r', 'control']:
                    key_name = 'ctrl'
                elif key_name in ['shift_l', 'shift_r']:
                    key_name = 'shift'
                elif key_name in ['alt_l', 'alt_r']:
                    key_name = 'alt'

                if pressed:  # Key DOWN
                    pending_key_down[key_name] = event['time']

                    # Track interval between same-key presses
                    if key_name in last_key_time:
                        interval = event['time'] - last_key_time[key_name]
                        if 0.05 < interval < 30.0:  # Reasonable interval
                            if key_name not in self.key_intervals:
                                self.key_intervals[key_name] = []
                            self.key_intervals[key_name].append(interval)

                    last_key_time[key_name] = event['time']

                    # Count key presses
                    self.key_press_counts[key_name] = self.key_press_counts.get(key_name, 0) + 1

                else:  # Key UP
                    if key_name in pending_key_down:
                        down_time = pending_key_down.pop(key_name)
                        hold_duration = event['time'] - down_time

                        # Filter reasonable hold durations (20ms to 10 seconds)
                        # Keys like Ctrl can be held longer than mouse buttons
                        if 0.02 < hold_duration < 10.0:
                            if key_name not in self.key_hold_durations:
                                self.key_hold_durations[key_name] = []
                            self.key_hold_durations[key_name].append(hold_duration)

        # Don't forget last segment
        if len(current_segment) > 5:
            self.all_movements.append(current_segment)
    
    def load_all_recordings(self, directory):
        """Load all recordings from a directory"""
        recordings_path = Path(directory)
        
        if not recordings_path.exists():
            print(f"No recordings found in {directory}")
            return False
        
        # Look for input_data.json files in session folders
        json_files = list(recordings_path.glob("session_*/input_data.json"))
        
        # Also check for old-style recordings directly in folder
        json_files.extend(recordings_path.glob("recording_*.json"))
        
        if not json_files:
            print(f"No recording files found in {directory}")
            return False
        
        print(f"\nFound {len(json_files)} recording(s)\n")
        
        for filepath in json_files:
            self.load_recording(filepath)
        
        return True
    
    def analyze_speeds(self):
        """Analyze movement speeds"""
        speeds = []
        
        for segment in self.all_movements:
            for i in range(1, len(segment)):
                dx = segment[i]['x'] - segment[i-1]['x']
                dy = segment[i]['y'] - segment[i-1]['y']
                dt = segment[i]['time'] - segment[i-1]['time']
                
                if dt > 0.001:  # Avoid division by zero
                    # Speed in "screen widths per second" (normalized)
                    speed = np.sqrt(dx**2 + dy**2) / dt
                    if speed < 10:  # Filter out teleports/glitches
                        speeds.append(speed)
        
        self.all_speeds = speeds
        
        if speeds:
            self.profile['speed_mean'] = float(np.mean(speeds))
            self.profile['speed_std'] = float(np.std(speeds))
            self.profile['speed_min'] = float(np.percentile(speeds, 5))
            self.profile['speed_max'] = float(np.percentile(speeds, 95))
    
    def analyze_click_timing(self):
        """Analyze click patterns including hold durations"""
        if len(self.all_clicks) < 2:
            return

        # Time between clicks
        intervals = []
        for i in range(1, len(self.all_clicks)):
            interval = self.all_clicks[i]['time'] - self.all_clicks[i-1]['time']
            if interval < 10:  # Reasonable interval
                intervals.append(interval)

        if intervals:
            self.profile['click_interval_mean'] = float(np.mean(intervals))
            self.profile['click_interval_std'] = float(np.std(intervals))

        # Click hold duration (mouse down -> up)
        if self.click_hold_durations:
            self.profile['click_hold_mean'] = float(np.mean(self.click_hold_durations))
            self.profile['click_hold_std'] = float(np.std(self.click_hold_durations))
            self.profile['click_hold_min'] = float(np.percentile(self.click_hold_durations, 10))
            self.profile['click_hold_max'] = float(np.percentile(self.click_hold_durations, 90))

        # Pre-click pause (stop moving before clicking)
        if self.pre_click_pauses:
            self.profile['pre_click_pause_mean'] = float(np.mean(self.pre_click_pauses))
            self.profile['pre_click_pause_std'] = float(np.std(self.pre_click_pauses))

        # Post-click pause (pause after releasing before moving again)
        if self.post_click_pauses:
            self.profile['post_click_pause_mean'] = float(np.mean(self.post_click_pauses))
            self.profile['post_click_pause_std'] = float(np.std(self.post_click_pauses))

        # Double-click rate
        if self.double_click_intervals:
            self.profile['double_click_rate'] = len(self.double_click_intervals) / max(len(self.all_clicks), 1)
            self.profile['double_click_interval_mean'] = float(np.mean(self.double_click_intervals))

    def analyze_keyboard_timing(self):
        """Analyze keyboard press timing patterns for humanization"""
        keyboard_profile = {}

        # Analyze each key's timing
        for key_name, durations in self.key_hold_durations.items():
            if len(durations) >= 3:  # Need at least 3 samples
                keyboard_profile[key_name] = {
                    'hold_mean': float(np.mean(durations)),
                    'hold_std': float(np.std(durations)),
                    'hold_min': float(np.percentile(durations, 10)),
                    'hold_max': float(np.percentile(durations, 90)),
                    'sample_count': len(durations)
                }

        # Add interval data
        for key_name, intervals in self.key_intervals.items():
            if len(intervals) >= 3:
                if key_name not in keyboard_profile:
                    keyboard_profile[key_name] = {}
                keyboard_profile[key_name]['interval_mean'] = float(np.mean(intervals))
                keyboard_profile[key_name]['interval_std'] = float(np.std(intervals))

        # Add press counts
        for key_name, count in self.key_press_counts.items():
            if key_name not in keyboard_profile:
                keyboard_profile[key_name] = {}
            keyboard_profile[key_name]['press_count'] = count

        self.profile['keyboard_timing'] = keyboard_profile

        # Also add commonly used keys as top-level for easy access
        # Ctrl (attack toggle)
        if 'ctrl' in keyboard_profile and 'hold_mean' in keyboard_profile['ctrl']:
            self.profile['ctrl_hold_mean'] = keyboard_profile['ctrl']['hold_mean']
            self.profile['ctrl_hold_std'] = keyboard_profile['ctrl']['hold_std']
            self.profile['ctrl_hold_min'] = keyboard_profile['ctrl']['hold_min']
            self.profile['ctrl_hold_max'] = keyboard_profile['ctrl']['hold_max']

        # Space (rocket)
        if 'space' in keyboard_profile and 'hold_mean' in keyboard_profile['space']:
            self.profile['space_hold_mean'] = keyboard_profile['space']['hold_mean']
            self.profile['space_hold_std'] = keyboard_profile['space']['hold_std']
            self.profile['space_hold_min'] = keyboard_profile['space']['hold_min']
            self.profile['space_hold_max'] = keyboard_profile['space']['hold_max']

        # Shift (special)
        if 'shift' in keyboard_profile and 'hold_mean' in keyboard_profile['shift']:
            self.profile['shift_hold_mean'] = keyboard_profile['shift']['hold_mean']
            self.profile['shift_hold_std'] = keyboard_profile['shift']['hold_std']
            self.profile['shift_hold_min'] = keyboard_profile['shift']['hold_min']
            self.profile['shift_hold_max'] = keyboard_profile['shift']['hold_max']
    
    def analyze_path_curvature(self):
        """Analyze how much you curve vs straight lines"""
        curvatures = []
        
        for segment in self.all_movements:
            if len(segment) < 3:
                continue
            
            # Compare actual path length to straight line distance
            start = segment[0]
            end = segment[-1]
            
            straight_dist = np.sqrt(
                (end['x'] - start['x'])**2 + 
                (end['y'] - start['y'])**2
            )
            
            # Actual path length
            path_length = 0
            for i in range(1, len(segment)):
                dx = segment[i]['x'] - segment[i-1]['x']
                dy = segment[i]['y'] - segment[i-1]['y']
                path_length += np.sqrt(dx**2 + dy**2)
            
            if straight_dist > 0.01:  # Avoid tiny movements
                curvature = path_length / straight_dist
                if curvature < 3:  # Filter outliers
                    curvatures.append(curvature)
        
        if curvatures:
            # 1.0 = perfectly straight, higher = more curved
            self.profile['curve_factor_mean'] = float(np.mean(curvatures))
            self.profile['curve_factor_std'] = float(np.std(curvatures))
    
    def analyze_acceleration(self):
        """Analyze acceleration patterns (speed up/slow down)"""
        # For each segment, analyze speed at start vs end
        accel_patterns = []
        
        for segment in self.all_movements:
            if len(segment) < 10:
                continue
            
            # Speed in first third
            first_third = segment[:len(segment)//3]
            speeds_start = []
            for i in range(1, len(first_third)):
                dx = first_third[i]['x'] - first_third[i-1]['x']
                dy = first_third[i]['y'] - first_third[i-1]['y']
                dt = first_third[i]['time'] - first_third[i-1]['time']
                if dt > 0.001:
                    speeds_start.append(np.sqrt(dx**2 + dy**2) / dt)
            
            # Speed in last third
            last_third = segment[-len(segment)//3:]
            speeds_end = []
            for i in range(1, len(last_third)):
                dx = last_third[i]['x'] - last_third[i-1]['x']
                dy = last_third[i]['y'] - last_third[i-1]['y']
                dt = last_third[i]['time'] - last_third[i-1]['time']
                if dt > 0.001:
                    speeds_end.append(np.sqrt(dx**2 + dy**2) / dt)
            
            if speeds_start and speeds_end:
                # Ratio > 1 means decelerating (slower at end)
                ratio = np.mean(speeds_start) / max(np.mean(speeds_end), 0.001)
                if 0.1 < ratio < 10:
                    accel_patterns.append(ratio)
        
        if accel_patterns:
            self.profile['deceleration_ratio'] = float(np.mean(accel_patterns))
    
    def analyze(self):
        """Run all analyses"""
        print("\n" + "="*50)
        print("  ANALYZING YOUR MOVEMENT PATTERNS")
        print("="*50 + "\n")

        self.analyze_speeds()
        self.analyze_click_timing()
        self.analyze_keyboard_timing()  # NEW: Analyze keyboard patterns
        self.analyze_path_curvature()
        self.analyze_acceleration()

        # Add counts
        self.profile['total_movements'] = len(self.all_movements)
        self.profile['total_clicks'] = len(self.all_clicks)
        self.profile['total_speed_samples'] = len(self.all_speeds)
        self.profile['total_click_hold_samples'] = len(self.click_hold_durations)
        self.profile['total_pre_click_pauses'] = len(self.pre_click_pauses)
        self.profile['total_post_click_pauses'] = len(self.post_click_pauses)
        self.profile['total_key_samples'] = sum(len(v) for v in self.key_hold_durations.values())

        return self.profile
    
    def print_profile(self):
        """Print the analyzed profile"""
        print("\n" + "="*50)
        print("  YOUR MOVEMENT PROFILE")
        print("="*50)

        print(f"\nData analyzed:")
        print(f"  Movement segments: {self.profile.get('total_movements', 0)}")
        print(f"  Click events: {self.profile.get('total_clicks', 0)}")
        print(f"  Speed samples: {self.profile.get('total_speed_samples', 0)}")
        print(f"  Click hold samples: {self.profile.get('total_click_hold_samples', 0)}")

        print(f"\nSpeed (screen-widths/second):")
        print(f"  Average: {self.profile.get('speed_mean', 0):.4f}")
        print(f"  Variance: {self.profile.get('speed_std', 0):.4f}")
        print(f"  Range: {self.profile.get('speed_min', 0):.4f} - {self.profile.get('speed_max', 0):.4f}")

        print(f"\nClick intervals (seconds between clicks):")
        print(f"  Average interval: {self.profile.get('click_interval_mean', 0):.2f}s")
        print(f"  Variance: {self.profile.get('click_interval_std', 0):.2f}s")

        print(f"\nClick hold timing (mouse down -> up):")
        hold_mean = self.profile.get('click_hold_mean', 0)
        hold_min = self.profile.get('click_hold_min', 0)
        hold_max = self.profile.get('click_hold_max', 0)
        print(f"  Average hold: {hold_mean*1000:.0f}ms")
        print(f"  Range: {hold_min*1000:.0f}ms - {hold_max*1000:.0f}ms")

        print(f"\nClick pauses:")
        pre_mean = self.profile.get('pre_click_pause_mean', 0)
        post_mean = self.profile.get('post_click_pause_mean', 0)
        print(f"  Pre-click pause: {pre_mean*1000:.0f}ms (stop before click)")
        print(f"  Post-click pause: {post_mean*1000:.0f}ms (pause after release)")

        dbl_rate = self.profile.get('double_click_rate', 0)
        if dbl_rate > 0:
            print(f"  Double-click rate: {dbl_rate*100:.1f}%")

        print(f"\nPath characteristics:")
        curve = self.profile.get('curve_factor_mean', 1)
        print(f"  Curve factor: {curve:.3f} ({'straight' if curve < 1.05 else 'slightly curved' if curve < 1.15 else 'curved'})")

        decel = self.profile.get('deceleration_ratio', 1)
        print(f"  Deceleration: {decel:.2f}x ({'accelerating' if decel < 0.9 else 'constant' if decel < 1.1 else 'decelerating'})")

        # Keyboard timing
        print(f"\nKeyboard timing (for humanized key presses):")
        keyboard_data = self.profile.get('keyboard_timing', {})
        if keyboard_data:
            for key_name, data in sorted(keyboard_data.items()):
                if 'hold_mean' in data:
                    count = data.get('sample_count', data.get('press_count', 0))
                    print(f"  {key_name.upper():>6}: hold={data['hold_mean']*1000:.0f}ms "
                          f"(range: {data.get('hold_min', 0)*1000:.0f}-{data.get('hold_max', 0)*1000:.0f}ms) "
                          f"[{count} samples]")
        else:
            print("  (no keyboard data - record with Ctrl/Space/Shift to capture)")

        # Highlight important keys for combat
        ctrl_hold = self.profile.get('ctrl_hold_mean')
        space_hold = self.profile.get('space_hold_mean')
        if ctrl_hold or space_hold:
            print(f"\nCombat keys (used by bot):")
            if ctrl_hold:
                print(f"  Ctrl (attack): {ctrl_hold*1000:.0f}ms hold")
            if space_hold:
                print(f"  Space (rocket): {space_hold*1000:.0f}ms hold")

        print("\n" + "="*50)
    
    def save_profile(self, filename="my_movement_profile.json"):
        """Save profile to file"""
        output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.profile, f, indent=2)
        
        print(f"\nProfile saved to: {filepath}")
        return filepath


def main():
    analyzer = MovementAnalyzer()
    
    # Find recordings directory
    script_dir = os.path.dirname(__file__)
    recordings_dir = os.path.join(script_dir, "..", "data", "recordings")
    
    if not analyzer.load_all_recordings(recordings_dir):
        print("\nNo recordings found!")
        print(f"Please run input_logger.py first to create recordings.")
        print(f"Expected location: {os.path.abspath(recordings_dir)}")
        return
    
    # Analyze
    analyzer.analyze()
    analyzer.print_profile()
    analyzer.save_profile()
    
    print("\nNext step: Run the movement generator to test your profile!")


if __name__ == "__main__":
    main()
