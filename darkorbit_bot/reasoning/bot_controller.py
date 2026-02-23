"""
DarkOrbit Bot - Autonomous Controller

The main bot that ties everything together:
1. Vision (YOLO detection)
2. Reasoning (Bi-LSTM policy)
3. Humanizer (Bezier curves + delay)
4. Input (Win32 mouse/keyboard)

Hotkeys:
- F1 = Pause/Resume bot
- F2 = BAD STOP - save recent actions as negative training data
- F3 = Emergency stop (kill switch)
- F6 = Toggle passive/aggressive mode override

Usage:
    python bot_controller.py --model path/to/best.pt --policy path/to/policy.pt
"""

import time
import sys
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local imports
try:
    from detection.detector import GameDetector, ScreenCapture
    from reasoning.filters import create_filters
    from reasoning.context_detector import ContextDetector
    from reasoning.state_builder import StateBuilder, StateSequenceBuilder, PlayerState
    from reasoning.policy_network import load_policy, create_policy
    from movement.generator import MovementGenerator, MouseController, MovementProfile, KeyboardController
    from reasoning.vision_context import AsyncVisionAnalyzer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the darkorbit_bot directory")
    sys.exit(1)


@dataclass
class BotConfig:
    """Bot configuration"""
    model_path: str = "F:/dev/bot/best.pt"
    policy_path: Optional[str] = None  # None = untrained (random)
    monitor: int = 1

    # Humanization - uses YOUR movement profile from analyze_patterns.py
    # Movement speed, curves, click timing all come from your recordings
    precision_noise: float = 0.02  # Small noise for human-like imperfection

    # Safety
    max_actions_per_second: int = 60  # Smooth 60fps updates
    emergency_stop_key: str = "f3"

    # Vision AI - DISABLED during bot runtime (too slow for real-time)
    # VLM is used during RECORDING to annotate training data instead
    use_vision_ai: bool = False  # Disabled - VLM is too slow for real-time bot
    vision_model: str = "qwen/qwen3-vl-8b"
    vision_backend: str = "openai"
    vision_url: str = "http://localhost:1234"

    # Self-improvement: VLM watches bot and generates corrections for training
    self_improve: bool = False  # Enable with --self-improve flag

    # Enhanced VLM: Multi-level hierarchical analysis (Strategic/Tactical/Execution)
    # More thorough than basic self-improve but uses more API calls
    enhanced_vlm: bool = False  # Enable with --enhanced-vlm flag

    # Auto-mode: let the network learn WHEN to be aggressive vs passive
    # Instead of hardcoded rules like "if enemy detected: attack"
    # Enable with --auto-mode flag (requires training with mode labels)
    auto_mode: bool = False

    # Meta-learning: analyze VLM performance and suggest improvements
    # Runs at end of session to evaluate VLM corrections and propose prompt changes
    meta_learn: bool = False  # Enable with --meta-learn flag

    # Skip all interactive prompts (for automated/batch runs)
    no_prompt: bool = False  # Enable with --no-prompt flag


class BotController:
    """
    Main bot controller - See → Think → Act loop.
    """
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.running = False
        self.paused = False
        self.mode_override: Optional[str] = None  # Force PASSIVE or AGGRESSIVE
        self.use_auto_mode = config.auto_mode  # Let network decide mode from learned behavior
        
        # Vision
        print("🔧 Initializing Vision...")

        # Check GPU/CUDA availability
        self._check_gpu_status()

        # Optimized detection settings (conf=0.3, iou=0.3 for duplicate removal)
        self.detector = GameDetector(
            config.model_path,
            confidence_threshold=0.3,
            iou_threshold=0.3
        )
        self.screen = ScreenCapture(monitor_index=config.monitor)
        
        # Reasoning
        print("🧠 Initializing Reasoning Core...")
        self.filters = create_filters()
        self.context = ContextDetector()

        # Policy (load or create untrained)
        # Note: StateBuilder size depends on whether model expects movement patterns
        if config.policy_path and Path(config.policy_path).exists():
            self.policy = load_policy(config.policy_path)
            print(f"   ✅ Loaded trained policy: {config.policy_path}")

            # Match StateBuilder to loaded model's input size
            model_input_size = self.policy.lstm.input_size
            if model_input_size == 128:
                # Old model without movement patterns
                self.state_builder = StateBuilder(include_movement_patterns=False)
                print(f"   Using old state format (128 features, no movement patterns)")
            else:
                # New model with movement patterns
                self.state_builder = StateBuilder(include_movement_patterns=True)
                print(f"   Using new state format ({model_input_size} features, with movement patterns)")
        else:
            print("   ⚠️ No trained policy - using random weights (demo mode)")
            self.state_builder = StateBuilder(include_movement_patterns=True)  # New models get patterns
            self.policy = create_policy(state_size=self.state_builder.get_state_size())

        self.sequence_builder = StateSequenceBuilder(sequence_length=50, state_builder=self.state_builder)
            
        # Humanizer - load YOUR profile if available
        print("🎮 Initializing Humanizer...")
        profile_path = Path(__file__).parent.parent / "data" / "my_movement_profile.json"
        if profile_path.exists():
            try:
                self.profile = MovementProfile.load(str(profile_path))
                print(f"   ✅ Loaded YOUR movement profile!")
                print(f"      Click hold: {self.profile.click_hold_mean*1000:.0f}ms")
                print(f"      Pre-click pause: {self.profile.pre_click_pause_mean*1000:.0f}ms")
            except Exception as e:
                print(f"   ⚠️ Error loading profile: {e}, using defaults")
                self.profile = MovementProfile()
        else:
            print("   ⚠️ No personal profile found - using defaults")
            print("      Run: python analysis/analyze_patterns.py to create one")
            self.profile = MovementProfile()
        self.movement_gen = MovementGenerator(self.profile)

        # Get actual screen resolution
        import ctypes
        user32 = ctypes.windll.user32
        self.screen_width = user32.GetSystemMetrics(0)
        self.screen_height = user32.GetSystemMetrics(1)
        print(f"   Screen resolution: {self.screen_width}x{self.screen_height}")

        self.mouse = MouseController(self.screen_width, self.screen_height)
        self.keyboard = KeyboardController(profile=self.profile)  # Pass profile for humanized timing

        # Timing
        self.last_action_time = 0.0
        self.action_cooldown = 1.0 / config.max_actions_per_second

        # Stats
        self.total_actions = 0
        self.total_clicks = 0
        self.session_start = None

        # Debug mode
        self.debug_mode = False

        # Reasoning log - shows bot's "thinking"
        self.reasoning_log_enabled = True  # Always show reasoning by default
        self.last_reasoning = ""
        self.reasoning_cooldown = 0.5  # Only log reasoning every 0.5s to reduce spam
        self.last_reasoning_time = 0

        # Combat state
        self.is_attacking = False  # Tracks if Ctrl (attack) is toggled on
        self.last_attack_toggle = 0.0

        # Movement logging
        self.last_mouse_update = 0.0

        # Exploration state - for when nothing is visible
        self.exploration_target = None  # (x, y) normalized target
        self.last_exploration_time = 0.0
        self.exploration_cooldown = 1.5  # Seconds between exploration moves (faster)
        self.idle_time = 0.0  # How long we've been without targets

        # Vision AI (LM Studio/Ollama) - for contextual understanding
        self.vision_analyzer = None
        self.last_vision_context = None
        if config.use_vision_ai:
            try:
                self.vision_analyzer = AsyncVisionAnalyzer(
                    backend=config.vision_backend,
                    model=config.vision_model,
                    base_url=config.vision_url
                )
                print(f"   Vision AI: {config.vision_model} @ {config.vision_url} ({config.vision_backend})")
            except Exception as e:
                print(f"   ⚠️ Vision AI not available: {e}")

        # Threading
        self.bot_thread: Optional[threading.Thread] = None
        self.kb_listener = None  # Keyboard listener for hotkeys

        # Self-improvement: VLM watches and critiques bot behavior
        self.self_improver = None
        self.enhanced_vlm = None

        if config.enhanced_vlm:
            # Use enhanced multi-level VLM (Strategic/Tactical/Execution)
            try:
                from reasoning.vlm_enhanced import EnhancedVLM
                self.enhanced_vlm = EnhancedVLM(
                    base_url=config.vision_url,
                    model=config.vision_model,
                    fast_model=config.vision_model  # Could use a faster model for execution checks
                )
                print("🔬 Enhanced VLM enabled (Strategic/Tactical/Execution analysis)")
                print(f"   Strategic: every {self.enhanced_vlm.STRATEGIC_INTERVAL}s")
                print(f"   Tactical: every {self.enhanced_vlm.TACTICAL_INTERVAL}s")
                print(f"   Execution: every {self.enhanced_vlm.EXECUTION_INTERVAL}s")
            except Exception as e:
                print(f"   ⚠️ Enhanced VLM not available: {e}")

        elif config.self_improve:
            # Use basic self-improver
            try:
                from reasoning.self_improver import SelfImprover
                self.self_improver = SelfImprover(
                    model=config.vision_model,
                    base_url=config.vision_url,
                    critique_interval=3.0  # Critique every 3 seconds
                )
                print("📊 Self-improvement enabled (VLM will critique actions)")
            except Exception as e:
                print(f"   ⚠️ Self-improver not available: {e}")

        # Kill/event tracking for temporal context (with debouncing to reduce false positives)
        self.last_visible_enemies = set()  # Track enemy IDs to detect kills
        self.last_visible_boxes = set()    # Track box IDs to detect pickups
        self.recent_attack_target = None   # Last target we attacked

        # Debouncing: track how many consecutive frames an entity has been visible/gone
        self.enemy_visible_frames = {}  # enemy_id -> count of frames visible
        self.enemy_gone_frames = {}     # enemy_id -> count of frames gone
        self.box_visible_frames = {}
        self.box_gone_frames = {}
        self.MIN_FRAMES_TO_CONFIRM = 3  # Must be visible for 3+ frames to count
        self.MIN_FRAMES_GONE = 5        # Must be gone for 5+ frames to count as kill/pickup

        # Bad stop buffer - tracks recent actions for negative corrections
        # When user presses F2 (bad stop), these are saved as "what NOT to do"
        self.bad_stop_buffer = []  # List of {state_vector, action, mode, timestamp}
        self.BAD_STOP_BUFFER_SIZE = 50  # Keep last 50 frames (~2-3 seconds at 20fps)
        self.BAD_STOP_SAVE_COUNT = 20   # Save last 20 frames when F2 pressed

        # Frame timing for performance monitoring
        self.frame_times = []  # Recent frame durations
        self.MAX_FRAME_TIMES = 100  # Keep last 100 frames

    def _check_gpu_status(self):
        """Check and report GPU/CUDA status for YOLO and PyTorch"""
        try:
            import torch

            print("\n   GPU Status:")

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"   ✅ CUDA Available: {gpu_name} ({gpu_mem:.1f} GB)")

                # Check if TensorRT is available (for RT cores)
                try:
                    if hasattr(torch.backends, 'cudnn') and torch.backends.cudnn.enabled:
                        print(f"   ✅ cuDNN: Enabled (v{torch.backends.cudnn.version()})")
                except:
                    pass

                # Set optimal settings for gaming
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                print(f"   ✅ cuDNN Benchmark: Enabled (optimized for inference)")

            else:
                print(f"   ⚠️ CUDA NOT AVAILABLE - Running on CPU (SLOW!)")
                print(f"      Install CUDA: https://developer.nvidia.com/cuda-downloads")
                print(f"      Then: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

        except ImportError:
            print(f"   ⚠️ PyTorch not installed")
        except Exception as e:
            print(f"   ⚠️ GPU check failed: {e}")

    def start(self):
        """Start the bot"""
        if self.running:
            return
            
        print("\n" + "="*60)
        print("  🤖 BOT CONTROLLER - Starting")
        print("="*60)
        print("\n🎮 Hotkeys:")
        print("   F1 = Pause/Resume")
        print("   F2 = BAD STOP (save as negative training data)")
        print("   F3 = EMERGENCY STOP")
        print("   F4 = Toggle debug logging")
        print("   F5 = Toggle reasoning log")
        print("   F6 = Toggle mode override (PASSIVE/AGGRESSIVE/AUTO)")
        print("-"*60)
        
        self.running = True
        self.paused = False
        self.session_start = time.time()

        # Start vision analyzer if available
        if self.vision_analyzer:
            self.vision_analyzer.start()
            print("   Vision AI started")

        # Start self-improver if enabled
        if self.self_improver:
            self.self_improver.start_session()
            self.self_improver.start_watching()
            print("   Self-improver watching...")

        # Start enhanced VLM correction saving if enabled
        if self.enhanced_vlm:
            self.enhanced_vlm.enable_corrections(True)
            self.enhanced_vlm.start_async()  # Start background analysis thread
            print("   Enhanced VLM watching (async, will save corrections for training)...")

        # Start control thread
        self.bot_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.bot_thread.start()

        # Setup hotkeys
        self._setup_hotkeys()
        
    def stop(self):
        """Stop the bot"""
        self.running = False
        if self.vision_analyzer:
            self.vision_analyzer.stop()
        if self.self_improver:
            self.self_improver.stop_watching()  # Saves corrections
        if self.enhanced_vlm:
            self.enhanced_vlm.stop_async()  # Stop background analysis thread
            self.enhanced_vlm.stop_and_save()  # Save enhanced VLM corrections
        if self.bot_thread:
            self.bot_thread.join(timeout=2.0)
        if self.kb_listener:
            self.kb_listener.stop()
        # Stop mouse movement thread
        if hasattr(self, 'mouse') and hasattr(self.mouse, 'stop'):
            self.mouse.stop()
        self._print_stats()
        print("\n🛑 Bot stopped.")

        # Handle meta-learning at end of session
        self._handle_end_of_session_meta()

    def _handle_end_of_session_meta(self):
        """
        Handle meta-learning at end of session.
        - If --meta-learn: run analysis automatically
        - If not --meta-learn and not --no-prompt: ask user if they want to run it
        - After analysis, ask if they want to apply (unless --no-prompt)
        """
        run_meta = self.config.meta_learn

        # If meta-learn not enabled, ask user (unless --no-prompt)
        if not run_meta and not self.config.no_prompt:
            # Check if we have VLM corrections to analyze
            has_corrections = self._has_vlm_corrections()
            if has_corrections:
                print("\n" + "-"*60)
                response = input("🧠 Run meta-analysis on VLM corrections? [y/N]: ").strip().lower()
                run_meta = response == 'y'

        if run_meta:
            analysis = self._run_meta_learning()

            # Ask if user wants to apply (unless --no-prompt)
            if analysis and not self.config.no_prompt:
                print("\n" + "-"*60)
                response = input("✨ Apply these suggestions to the VLM prompt? [y/N]: ").strip().lower()
                if response == 'y':
                    self._apply_meta_suggestions(analysis)

    def _has_vlm_corrections(self) -> bool:
        """Check if there are any VLM corrections from this session."""
        from pathlib import Path
        import time

        corrections_dir = Path(__file__).parent.parent / "data" / "vlm_corrections"
        if not corrections_dir.exists():
            return False

        # Check for recent files (last hour)
        cutoff = time.time() - 3600
        for f in corrections_dir.glob("*.json"):
            if f.stat().st_mtime > cutoff:
                return True
        for d in corrections_dir.glob("session_*"):
            if d.stat().st_mtime > cutoff:
                return True
        return False

    def _run_meta_learning(self):
        """
        Run meta-learning analysis at end of session.
        Analyzes VLM corrections and suggests improvements to the system prompt.

        Returns:
            Analysis dict if successful, None otherwise
        """
        try:
            from reasoning.vlm_meta_learner import MetaLearner

            print("\n" + "="*60)
            print("  🧠 RUNNING META-LEARNING ANALYSIS")
            print("="*60)

            learner = MetaLearner(base_url=self.config.vision_url)
            analysis = learner.analyze_session(hours=1, apply_changes=False)

            if analysis:
                print("\n✅ Meta-analysis complete!")
                print(f"   View suggestions: {learner.PROMPT_SUGGESTIONS_FILE}")
                return analysis
            else:
                print("\n⚠️ Meta-analysis skipped (no data or LLM unavailable)")
                return None

        except Exception as e:
            print(f"\n⚠️ Meta-learning failed: {e}")
            return None

    def _apply_meta_suggestions(self, analysis):
        """Apply meta-learning suggestions to the VLM prompt."""
        try:
            from reasoning.vlm_meta_learner import MetaLearner

            learner = MetaLearner(base_url=self.config.vision_url)
            if learner.apply_suggestions(analysis, auto_apply=False):
                print("\n✅ Suggestions applied to VLM prompt!")
            else:
                print("\n⚠️ No changes applied")

        except Exception as e:
            print(f"\n⚠️ Failed to apply suggestions: {e}")
        
    def _setup_hotkeys(self):
        """Setup control hotkeys using pynput (works without admin privileges)"""
        try:
            from pynput import keyboard as pynput_keyboard

            def on_key_release(key):
                if hasattr(key, 'name'):
                    if key.name == 'f1':
                        self._toggle_pause()
                    elif key.name == 'f2':
                        self._bad_stop()  # Save recent actions as negative training data
                    elif key.name == 'f3':
                        self._emergency_stop()
                    elif key.name == 'f4':
                        self._toggle_debug()
                    elif key.name == 'f5':
                        self._toggle_reasoning()
                    elif key.name == 'f6':
                        self._toggle_mode_override()

            self.kb_listener = pynput_keyboard.Listener(on_release=on_key_release)
            self.kb_listener.start()
        except ImportError:
            print("⚠️ 'pynput' module not installed - hotkeys disabled")
            
    def _toggle_pause(self):
        self.paused = not self.paused
        status = "PAUSED" if self.paused else "RUNNING"
        print(f"\n⏯️ Bot {status}")

    def _bad_stop(self):
        """
        BAD STOP - User pressed F2 to indicate bot did something wrong.
        Saves recent state/actions as NEGATIVE training examples.

        The bot learns "don't do this" from these corrections.
        """
        import json
        from pathlib import Path

        # Pause the bot
        self.paused = True

        if not self.bad_stop_buffer:
            print("\n⚠️ Bad stop buffer is empty - nothing to save")
            return

        # Get the most recent frames to save
        frames_to_save = self.bad_stop_buffer[-self.BAD_STOP_SAVE_COUNT:]

        if not frames_to_save:
            print("\n⚠️ No recent actions to save")
            return

        # Create corrections directory
        corrections_dir = Path(__file__).parent.parent / "data" / "vlm_corrections"
        corrections_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = int(time.time())
        filename = f"bad_stop_{timestamp}.json"
        filepath = corrections_dir / filename

        # Build correction entries
        corrections = []
        for frame in frames_to_save:
            state_vector = frame.get('state_vector')
            if state_vector is None:
                continue

            # The action the bot took (WRONG)
            bot_action = frame.get('action', {})
            mode = frame.get('mode', 'PASSIVE')

            # Convert bot_action to JSON-serializable format (handle numpy bools)
            serializable_action = {}
            for k, v in bot_action.items():
                if hasattr(v, 'item'):  # numpy type
                    serializable_action[k] = v.item()
                elif isinstance(v, (bool, int, float, str, type(None))):
                    serializable_action[k] = v
                else:
                    serializable_action[k] = bool(v) if isinstance(v, (np.bool_,)) else float(v)

            corrections.append({
                'timestamp': frame.get('timestamp', time.time()),
                'quality': 'bad',  # User said it was wrong
                'state_vector': state_vector if isinstance(state_vector, list) else state_vector.tolist(),
                'bot_action': serializable_action,
                'correct_action': None,  # We don't know what's correct, just that this was wrong
                'source': 'bad_stop',
                'mode': mode,
            })

        if not corrections:
            print("\n⚠️ No valid frames with state vectors to save")
            return

        # Save to file
        data = {
            'source': 'bad_stop',
            'timestamp': timestamp,
            'reason': 'User pressed F2 to indicate bot behavior was wrong',
            'corrections': corrections
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n❌ BAD STOP! Saved {len(corrections)} frames as negative training data")
        print(f"   File: {filepath.name}")
        print(f"   These will be used to teach the model what NOT to do")
        print(f"   Press F1 to resume or F3 to fully stop")

    def _detect_events(self, detections: list):
        """
        Detect game events (kills, pickups) by comparing current vs previous frame.
        Records events to self-improver for temporal context.

        Uses DEBOUNCING to reduce false positives from YOLO flickering:
        - Entity must be visible for MIN_FRAMES_TO_CONFIRM frames to be "confirmed"
        - Entity must be gone for MIN_FRAMES_GONE frames to count as kill/pickup
        """
        if not self.self_improver:
            return

        # Get current visible enemies and boxes with unique IDs (position-based)
        current_enemies = set()
        current_boxes = set()

        for d in detections:
            # Create a rough ID based on position (round to reduce jitter)
            pos_id = f"{d.class_name}_{d.x_center:.2f}_{d.y_center:.2f}"

            if d.class_name in self.ENEMY_CLASSES or d.class_name not in self.NON_ENEMY_CLASSES:
                current_enemies.add(pos_id)
            elif d.class_name in ['BonusBox', 'bonus_box', 'box']:
                current_boxes.add(pos_id)

        # Update enemy frame counters (debouncing)
        for enemy_id in current_enemies:
            # Reset gone counter, increment visible counter
            self.enemy_gone_frames[enemy_id] = 0
            self.enemy_visible_frames[enemy_id] = self.enemy_visible_frames.get(enemy_id, 0) + 1

        # Check for enemies that disappeared
        for enemy_id in list(self.enemy_visible_frames.keys()):
            if enemy_id not in current_enemies:
                # Enemy not visible this frame
                self.enemy_gone_frames[enemy_id] = self.enemy_gone_frames.get(enemy_id, 0) + 1

                # Only count as kill if:
                # 1. Was visible for enough frames (not a flicker)
                # 2. Has been gone for enough frames (confirmed gone)
                # 3. We were attacking
                if (self.enemy_visible_frames.get(enemy_id, 0) >= self.MIN_FRAMES_TO_CONFIRM and
                    self.enemy_gone_frames.get(enemy_id, 0) == self.MIN_FRAMES_GONE and
                    self.is_attacking):
                    enemy_name = enemy_id.split('_')[0]
                    self.self_improver.record_event(f"Killed {enemy_name}")
                    if self.reasoning_log_enabled:
                        print(f"\n   ⚔️ [EVENT] Killed {enemy_name}!")

                # Clean up old entries after a while
                if self.enemy_gone_frames.get(enemy_id, 0) > self.MIN_FRAMES_GONE * 2:
                    self.enemy_visible_frames.pop(enemy_id, None)
                    self.enemy_gone_frames.pop(enemy_id, None)

        # Update box frame counters (debouncing)
        for box_id in current_boxes:
            self.box_gone_frames[box_id] = 0
            self.box_visible_frames[box_id] = self.box_visible_frames.get(box_id, 0) + 1

        # Check for boxes that disappeared
        for box_id in list(self.box_visible_frames.keys()):
            if box_id not in current_boxes:
                self.box_gone_frames[box_id] = self.box_gone_frames.get(box_id, 0) + 1

                # Only count as pickup if was visible long enough and gone long enough
                if (self.box_visible_frames.get(box_id, 0) >= self.MIN_FRAMES_TO_CONFIRM and
                    self.box_gone_frames.get(box_id, 0) == self.MIN_FRAMES_GONE):
                    self.self_improver.record_event("Picked up BonusBox")
                    if self.reasoning_log_enabled:
                        print(f"\n   📦 [EVENT] Picked up BonusBox!")

                # Clean up
                if self.box_gone_frames.get(box_id, 0) > self.MIN_FRAMES_GONE * 2:
                    self.box_visible_frames.pop(box_id, None)
                    self.box_gone_frames.pop(box_id, None)

        # Update tracking for next frame
        self.last_visible_enemies = current_enemies
        self.last_visible_boxes = current_boxes
        
    def _toggle_mode_override(self):
        if self.mode_override is None:
            self.mode_override = "PASSIVE"
        elif self.mode_override == "PASSIVE":
            self.mode_override = "AGGRESSIVE"
        else:
            self.mode_override = None
        print(f"\n🎯 Mode override: {self.mode_override or 'AUTO'}")
        
    def _emergency_stop(self):
        print("\n🚨 EMERGENCY STOP!")
        # Save corrections before stopping (don't lose self-improvement data!)
        if self.self_improver:
            print("   Saving self-improver corrections...")
            self.self_improver.stop_watching()
        if self.enhanced_vlm:
            print("   Saving enhanced VLM corrections...")
            self.enhanced_vlm.stop_and_save()
        self.running = False

    def _toggle_debug(self):
        self.debug_mode = not self.debug_mode
        status = "ON" if self.debug_mode else "OFF"
        print(f"\n🔍 Debug mode: {status}")

    def _toggle_reasoning(self):
        self.reasoning_log_enabled = not self.reasoning_log_enabled
        status = "ON" if self.reasoning_log_enabled else "OFF"
        print(f"\n💭 Reasoning log: {status}")

    def _log_reasoning(self, reasoning: str, force: bool = False):
        """
        Log the bot's reasoning/thinking process.
        Shows what the bot is "thinking" as it makes decisions.
        """
        now = time.time()

        # Avoid spam - only log if reasoning changed or cooldown passed
        if not force and reasoning == self.last_reasoning:
            return
        if not force and now - self.last_reasoning_time < self.reasoning_cooldown:
            return

        self.last_reasoning = reasoning
        self.last_reasoning_time = now

        if self.reasoning_log_enabled:
            print(f"\n💭 {reasoning}")

    def _build_reasoning(self, detections: list, mode: str, action: dict,
                         health: float, has_targets: bool) -> str:
        """
        Build a reasoning string that explains what the bot is thinking.
        This makes the bot's decision-making transparent.
        """
        # Count objects (flexible class name matching)
        enemies = [d for d in detections if
                   d.class_name in self.ENEMY_CLASSES or
                   d.class_name not in self.NON_ENEMY_CLASSES]
        boxes = [d for d in detections if d.class_name in ['BonusBox', 'bonus_box', 'box']]

        # Start building reasoning
        parts = []

        # What do I see?
        if enemies:
            enemy_names = [e.class_name for e in enemies[:3]]
            parts.append(f"SEE: {len(enemies)} enemies ({', '.join(enemy_names)})")
        if boxes:
            parts.append(f"SEE: {len(boxes)} boxes")
        if not enemies and not boxes:
            parts.append("SEE: Nothing interesting")

        # How am I feeling?
        if health < 0.3:
            parts.append("HEALTH: Critical! Need to retreat")
        elif health < 0.6:
            parts.append("HEALTH: Damaged, be careful")

        # What mode am I in and why?
        if mode == "AGGRESSIVE":
            if enemies:
                parts.append("MODE: AGGRESSIVE (enemies detected)")
            else:
                parts.append("MODE: AGGRESSIVE (override)")
        else:
            if boxes:
                parts.append("MODE: PASSIVE (collecting boxes)")
            else:
                parts.append("MODE: PASSIVE (exploring)")

        # What am I going to do?
        if action:
            # Handle both key formats flexibly (mode may not match action keys)
            if mode == "AGGRESSIVE":
                target_x = action.get('aim_x', action.get('move_x', 0.5))
                target_y = action.get('aim_y', action.get('move_y', 0.5))
                should_fire = action.get('should_fire', action.get('should_click', False))
                if should_fire and enemies:
                    nearest = min(enemies, key=lambda e: (e.x_center - target_x)**2 + (e.y_center - target_y)**2)
                    parts.append(f"DECIDE: Attack {nearest.class_name}!")
                elif should_fire:
                    parts.append("DECIDE: Fire at target")
                else:
                    parts.append("DECIDE: Track target, waiting for shot")
            else:
                target_x = action.get('move_x', action.get('aim_x', 0.5))
                target_y = action.get('move_y', action.get('aim_y', 0.5))
                should_click = action.get('should_click', action.get('should_fire', False))
                if should_click and boxes:
                    parts.append("DECIDE: Click to collect box")
                elif should_click:
                    parts.append("DECIDE: Click to move")
                else:
                    parts.append("DECIDE: Moving toward target")

        # Am I exploring?
        if not has_targets and self.idle_time > 1.0:
            parts.append(f"EXPLORE: Idle for {self.idle_time:.1f}s, searching map...")

        # Am I attacking?
        if self.is_attacking:
            parts.append("COMBAT: Attack mode ON (Ctrl held)")

        return " | ".join(parts)
        
    def _control_loop(self):
        """Main See → Think → Act loop"""
        prev_mouse = (0, 0)

        # Performance profiling (disable after debugging)
        self._profile_times = {'capture': [], 'yolo': [], 'policy': [], 'execute': []}
        self._profile_count = 0

        while self.running:
            try:
                if self.paused:
                    time.sleep(0.1)
                    continue

                loop_start = time.time()

                # ═══════════════════════════════════════
                # 1. SEE - Capture and detect
                # ═══════════════════════════════════════
                t0 = time.perf_counter()
                frame = self.screen.capture()
                t1 = time.perf_counter()
                detections = self.detector.detect_frame(frame)
                t2 = time.perf_counter()

                # Track timing
                self._profile_times['capture'].append(t1 - t0)
                self._profile_times['yolo'].append(t2 - t1)

                # Submit frame to vision AI for context analysis (async, non-blocking)
                if self.vision_analyzer and frame is not None:
                    self.vision_analyzer.submit_image(frame)
                    # Check for new context
                    new_context = self.vision_analyzer.get_context()
                    if new_context and new_context != self.last_vision_context:
                        self.last_vision_context = new_context
                        if self.debug_mode:
                            print(f"\n   [VLM] {new_context.situation} | Threat: {new_context.threat_level} | {new_context.recommended_action[:50]}")
                
                # Get mouse position
                import ctypes
                class POINT(ctypes.Structure):
                    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
                pt = POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
                mouse_x, mouse_y = pt.x, pt.y
                
                # Calculate velocity
                velocity = np.sqrt((mouse_x - prev_mouse[0])**2 + (mouse_y - prev_mouse[1])**2)
                prev_mouse = (mouse_x, mouse_y)
                
                # ═══════════════════════════════════════
                # 2. THINK - Process and decide
                # ═══════════════════════════════════════
                
                # Detect context (used for fallback/display only)
                context_state = self.context.detect(mouse_x, mouse_y, detections)

                # Mode selection: use learned mode or manual override
                # If auto_mode enabled, the network learns WHEN to be aggressive from YOUR gameplay
                if self.mode_override:
                    mode = self.mode_override
                    mode_source = "override"
                elif self.use_auto_mode and hasattr(self.policy, 'predict_mode'):
                    # Will be set below when we have state sequence
                    mode = context_state.mode  # Temporary, updated below
                    mode_source = "auto"
                else:
                    mode = context_state.mode
                    mode_source = "context"

                # Build player state
                health = self.filters['health'].update(detections)
                player = PlayerState(
                    health=health,
                    mouse_x=mouse_x / self.screen_width,
                    mouse_y=mouse_y / self.screen_height,
                    velocity_x=velocity,
                    velocity_y=0,
                    mode=mode
                )
                
                # Build state sequence
                state_seq = self.sequence_builder.add_frame(detections, player)
                
                # Get action from policy
                action = None
                if state_seq is not None:
                    if mode_source == "auto" and hasattr(self.policy, 'get_action_auto'):
                        # Use learned mode selection - network decides when to attack!
                        action = self.policy.get_action_auto(state_seq)
                        mode = action.get('predicted_mode', mode)
                        if self.debug_mode and action.get('mode_confidence'):
                            conf = action['mode_confidence']
                            print(f"   [AUTO-MODE] {mode} (conf: {conf:.2f})")
                    else:
                        action = self.policy.get_action(state_seq, mode=mode)
                else:
                    # No policy output yet (building sequence) - create default action
                    # This ensures smart targeting can still work during warmup
                    if mode == "AGGRESSIVE":
                        action = {'aim_x': 0.5, 'aim_y': 0.5, 'should_fire': False, 'raw_fire_value': 0.0}
                    else:
                        action = {'move_x': 0.5, 'move_y': 0.5, 'should_click': False, 'raw_click_value': 0.0}

                # ═══════════════════════════════════════
                # POLICY NETWORK CONTROLS BEHAVIOR
                # ═══════════════════════════════════════
                # The trained model should decide:
                # - Where to aim/move (learns from your recordings)
                # - When to click (learns from your click patterns)
                # - When to change targets (learns dynamic behavior)
                #
                # We only use smart targeting as FALLBACK when:
                # 1. Model hasn't been trained yet (no policy loaded)
                # 2. Model outputs invalid/no action
                # 3. During warmup (building state sequence)

                has_targets = self._has_visible_targets(detections)

                # ═══════════════════════════════════════
                # ENHANCED VLM - Observe and generate training corrections
                # (NO runtime overrides - just watches and learns)
                # Uses ASYNC processing to avoid blocking the control loop!
                # ═══════════════════════════════════════
                if self.enhanced_vlm and frame is not None:
                    # Convert detections to dict format
                    detection_dicts = [
                        {'class_name': d.class_name, 'x_center': d.x_center, 'y_center': d.y_center}
                        for d in detections
                    ] if detections else []

                    # Normalize bot action for VLM context
                    if action:
                        if mode == "AGGRESSIVE":
                            vlm_bot_action = {
                                'move_x': action.get('aim_x', 0.5),
                                'move_y': action.get('aim_y', 0.5),
                            }
                        else:
                            vlm_bot_action = {
                                'move_x': action.get('move_x', 0.5),
                                'move_y': action.get('move_y', 0.5),
                            }
                    else:
                        vlm_bot_action = {'move_x': 0.5, 'move_y': 0.5}

                    # Build context for correction saving
                    # Include state_vector so corrections can be used for training!
                    vlm_context = {
                        'mode': mode,
                        'health': health,
                        'is_attacking': self.is_attacking,
                        'idle_time': self.idle_time,
                        'state_vector': state_seq[-1].tolist() if state_seq is not None else None,
                    }

                    # Submit frame for ASYNC analysis (non-blocking!)
                    # Background thread handles VLM queries without blocking control loop
                    self.enhanced_vlm.submit_frame(
                        frame, detection_dicts, vlm_bot_action, self.is_attacking,
                        context=vlm_context
                    )

                    # Get latest results (non-blocking, returns cached results)
                    vlm_results = self.enhanced_vlm.get_last_results()

                    # Log VLM observations (for debugging, no runtime changes)
                    if vlm_results.get('strategic'):
                        strat = vlm_results['strategic']
                        if self.reasoning_log_enabled:
                            print(f"\n   [VLM-STRATEGIC] Strategy: {strat.get('recommended_strategy', '?')} | "
                                  f"Threat: {strat.get('threat_assessment', '?')} | "
                                  f"Area: {strat.get('area_type', '?')}")

                    if vlm_results.get('tactical'):
                        tact = vlm_results['tactical']
                        target_info = tact.get('priority_target', {})
                        if self.reasoning_log_enabled:
                            print(f"   [VLM-TACTICAL] Target: {target_info.get('name', 'none')} | "
                                  f"Tactic: {tact.get('recommended_tactic', '?')} | "
                                  f"Conf: {tact.get('confidence', 0):.0%}")

                    if vlm_results.get('execution'):
                        exec_result = vlm_results['execution']
                        if not exec_result.get('action_correct', True):
                            issue = exec_result.get('issue', 'unknown')
                            if self.reasoning_log_enabled:
                                print(f"   [VLM-EXEC] Issue: {issue} (correction saved)")

                # Check if policy gave a valid action
                policy_has_target = False
                if action:
                    # Check if model is aiming at something (not just center)
                    # Handle both key formats flexibly
                    if mode == "AGGRESSIVE":
                        aim_x = action.get('aim_x', action.get('move_x', 0.5))
                        aim_y = action.get('aim_y', action.get('move_y', 0.5))
                    else:
                        aim_x = action.get('move_x', action.get('aim_x', 0.5))
                        aim_y = action.get('move_y', action.get('aim_y', 0.5))
                    policy_has_target = abs(aim_x - 0.5) > 0.05 or abs(aim_y - 0.5) > 0.05

                # FALLBACK: Use smart targeting only if model isn't giving good output
                # This helps untrained/poorly-trained models still function
                if has_targets and not policy_has_target:
                    if self.debug_mode:
                        print(f"\n   [FALLBACK] Policy output near center, using smart targeting")
                    action = self._smart_target_override(action, detections, mode,
                                                         mouse_x / self.screen_width,
                                                         mouse_y / self.screen_height)

                # Track idle time for exploration
                if not has_targets:
                    self.idle_time += (1.0 / 30)  # Approximate frame time
                    # Only explore if policy isn't giving us movement
                    if not policy_has_target:
                        exploration_action = self._get_exploration_action()
                        if exploration_action:
                            action = exploration_action
                else:
                    self.idle_time = 0

                # ═══════════════════════════════════════
                # DETECT EVENTS - For temporal context
                # ═══════════════════════════════════════
                # Track kills and pickups by comparing frames
                self._detect_events(detections)

                # Track what we're currently targeting (for kill detection)
                target = self._find_nearest_target(detections, mode,
                                                   mouse_x / self.screen_width,
                                                   mouse_y / self.screen_height)
                if target:
                    self.recent_attack_target = target[2]  # Store target name

                # ═══════════════════════════════════════
                # LOG REASONING - What is the bot thinking?
                # ═══════════════════════════════════════
                reasoning = self._build_reasoning(detections, mode, action, health, has_targets)
                self._log_reasoning(reasoning)

                # ═══════════════════════════════════════
                # 3. ACT - Execute with humanization
                # ═══════════════════════════════════════
                t3 = time.perf_counter()

                if action and self._can_act():
                    self._execute_action(action, mode, mouse_x, mouse_y, detections)

                t4 = time.perf_counter()
                self._profile_times['policy'].append(t3 - t2)
                self._profile_times['execute'].append(t4 - t3)

                # Print profiling every 50 frames
                self._profile_count += 1
                if self._profile_count % 50 == 0:
                    def avg_ms(lst): return sum(lst[-50:]) / min(len(lst), 50) * 1000
                    print(f"\n   [PROFILE] capture:{avg_ms(self._profile_times['capture']):.1f}ms | "
                          f"yolo:{avg_ms(self._profile_times['yolo']):.1f}ms | "
                          f"policy:{avg_ms(self._profile_times['policy']):.1f}ms | "
                          f"execute:{avg_ms(self._profile_times['execute']):.1f}ms")

                # Record to bad_stop buffer (for F2 negative training data)
                if state_seq is not None:
                    bad_stop_entry = {
                        'timestamp': time.time(),
                        'state_vector': state_seq[-1].tolist(),
                        'action': {
                            'move_x': action.get('aim_x', action.get('move_x', 0.5)),
                            'move_y': action.get('aim_y', action.get('move_y', 0.5)),
                            'clicked': action.get('should_fire', action.get('should_click', False)),
                            'ctrl_attack': action.get('ctrl_attack', False),
                            'space_rocket': action.get('space_rocket', False),
                            'shift_special': action.get('shift_special', False),
                        },
                        'mode': mode,
                    }
                    self.bad_stop_buffer.append(bad_stop_entry)
                    # Keep buffer from growing indefinitely
                    if len(self.bad_stop_buffer) > self.BAD_STOP_BUFFER_SIZE:
                        self.bad_stop_buffer.pop(0)

                # Submit frame to self-improver for VLM critique
                if self.self_improver and frame is not None:
                    # Convert detections to dict format for self_improver
                    detection_dicts = [
                        {'class_name': d.class_name, 'x_center': d.x_center, 'y_center': d.y_center}
                        for d in detections
                    ] if detections else []

                    # Normalize action dict for self-improver (use consistent keys)
                    # Include keyboard actions so VLM can see if bot is attacking
                    if action:
                        if mode == "AGGRESSIVE":
                            normalized_action = {
                                'move_x': action.get('aim_x', 0.5),
                                'move_y': action.get('aim_y', 0.5),
                                'clicked': action.get('should_fire', False),
                                'mode': mode,
                                # Keyboard actions from policy output
                                'ctrl_attack': action.get('ctrl_attack', False),
                                'space_rocket': action.get('space_rocket', False),
                                'shift_special': action.get('shift_special', False),
                                # Also include if bot is currently attacking (from hardcoded logic)
                                'is_attacking': self.is_attacking,
                            }
                        else:
                            normalized_action = {
                                'move_x': action.get('move_x', 0.5),
                                'move_y': action.get('move_y', 0.5),
                                'clicked': action.get('should_click', False),
                                'mode': mode,
                                'ctrl_attack': action.get('ctrl_attack', False),
                                'space_rocket': action.get('space_rocket', False),
                                'shift_special': action.get('shift_special', False),
                                'is_attacking': self.is_attacking,
                            }
                    else:
                        normalized_action = {
                            'move_x': 0.5, 'move_y': 0.5, 'clicked': False, 'mode': mode,
                            'ctrl_attack': False, 'space_rocket': False, 'shift_special': False,
                            'is_attacking': self.is_attacking,
                        }

                    self.self_improver.submit_frame(frame, {
                        'bot_action': normalized_action,
                        'detections': detection_dicts,
                        'mode': mode,
                        'idle_time': self.idle_time,
                        'state_vector': state_seq[-1].tolist() if state_seq is not None else None
                    })

                # Status
                self._print_status(mode, health, detections)
                
                # Maintain reasonable loop speed
                elapsed = time.time() - loop_start
                sleep_time = (1.0 / 30) - elapsed  # Target 30 FPS
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Track frame timing for performance monitoring
                frame_time = time.time() - loop_start
                self.frame_times.append(frame_time)
                if len(self.frame_times) > self.MAX_FRAME_TIMES:
                    self.frame_times.pop(0)

                # Log performance warning if frames are slow
                if frame_time > 0.1 and self.debug_mode:  # >100ms is slow
                    print(f"\n   ⚠️ SLOW FRAME: {frame_time*1000:.0f}ms")

            except Exception as e:
                print(f"\n❌ Error: {e}")
                time.sleep(0.1)
                
    def _can_act(self) -> bool:
        """Check if we can perform an action (cooldown)"""
        now = time.time()
        if now - self.last_action_time >= self.action_cooldown:
            return True
        return False
        
    def _execute_action(self, action: Dict, mode: str, current_x: int, current_y: int,
                        detections: list = None):
        """Execute an action with smooth continuous tracking"""

        # Handle action keys flexibly - mode may have changed but action dict wasn't updated
        # AGGRESSIVE uses aim_x/aim_y, PASSIVE uses move_x/move_y
        if mode == "AGGRESSIVE":
            target_x = action.get('aim_x', action.get('move_x', 0.5))
            target_y = action.get('aim_y', action.get('move_y', 0.5))
            should_click = action.get('should_fire', action.get('should_click', False))
        else:
            target_x = action.get('move_x', action.get('aim_x', 0.5))
            target_y = action.get('move_y', action.get('aim_y', 0.5))
            should_click = action.get('should_click', action.get('should_fire', False))

        # Add small precision noise for human-like imperfection
        noise = self.config.precision_noise
        target_x += np.random.uniform(-noise, noise)
        target_y += np.random.uniform(-noise, noise)
        target_x = max(0.0, min(1.0, target_x))
        target_y = max(0.0, min(1.0, target_y))

        # Current mouse position (normalized)
        current_norm_x = current_x / self.screen_width
        current_norm_y = current_y / self.screen_height

        # Generate humanized path using YOUR movement profile
        start = (current_norm_x, current_norm_y)
        end = (target_x, target_y)
        dist = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

        # Only move if there's meaningful distance
        if dist > 0.005:  # More than 0.5% of screen
            # Use YOUR profile's speed, scaled by distance
            # Profile speed is in "screen widths per second"
            base_speed = self.profile.speed_mean

            # Combat mode: faster movements (cap duration)
            if mode == "AGGRESSIVE":
                # Fast combat: 0.05-0.25s based on distance
                target_duration = max(0.05, min(0.25, dist / max(base_speed * 1.5, 0.5)))
            else:
                # Passive mode: use YOUR natural speed from profile
                target_duration = max(0.08, min(0.4, dist / max(base_speed, 0.3)))

            # Generate bezier path with YOUR curve style
            path = self.movement_gen.generate_path(start, end, target_duration=target_duration)

            # Add slight overshoot (less in combat for precision)
            overshoot_prob = 0.03 if mode == "AGGRESSIVE" else 0.08
            path = self.movement_gen.add_overshoot(path, probability=overshoot_prob)

            # Execute smoothly (200 updates/sec interpolation)
            self.mouse.execute_path_smooth(path)

            # Log occasionally
            now = time.time()
            if self.reasoning_log_enabled and now - self.last_mouse_update > 0.5:
                target_info = ""
                if detections:
                    t = self._find_nearest_target(detections, mode, current_norm_x, current_norm_y)
                    if t:
                        target_info = f" | Target: {t[2]}"
                print(f"\n   [MOVE] → ({target_x:.3f}, {target_y:.3f}) in {target_duration*1000:.0f}ms{target_info}")
                self.last_mouse_update = now

        # Click logic - separate from movement
        # Always click if network says so OR if we have a target nearby
        if should_click:
            self._human_click(mode)
            self.total_clicks += 1
        # Also click if we're close to a valid target (auto-click assist)
        elif detections:
            target = self._find_nearest_target(detections, mode,
                                               target_x, target_y)
            if target and target[3] < 0.03:  # Within 3% of target
                self._human_click(mode)
                self.total_clicks += 1
                if self.debug_mode:
                    print(f"\n   [AUTO] Clicking on {target[2]}")

        # Handle combat keys (Ctrl for attack, Space for rockets)
        # Pass health so bot can flee when critical
        has_enemy = any(d.class_name in self.ENEMY_CLASSES or d.class_name not in self.NON_ENEMY_CLASSES
                        for d in (detections or []))
        # Get current health from filters (approximate)
        current_health = self.filters['health'].update(detections or [])
        self._handle_combat_keys(mode, has_enemy, health=current_health)

        self.total_actions += 1
        self.last_action_time = time.time()

    def _human_click(self, mode: str):
        """Execute a human-like click - NON-BLOCKING"""
        # Get click timing from YOUR profile
        timing = self.movement_gen.generate_click_timing()

        # COMBAT MODE: Much faster clicks for responsive gameplay
        if mode == "AGGRESSIVE":
            # Fast click hold (50-80ms)
            hold_duration = min(timing['hold_duration'], np.random.uniform(0.05, 0.08))
            do_double = False
        else:
            # Passive mode: use profile timing but cap it
            hold_duration = min(timing['hold_duration'], 0.12)  # Max 120ms
            do_double = timing['double_click'] is not None

        # Non-blocking click (mouse.click now returns immediately)
        if do_double:
            self.mouse.click(hold_duration=hold_duration * 0.7)
            # Schedule second click after interval (non-blocking)
            def delayed_click():
                time.sleep(min(timing['double_click'], 0.15))
                self.mouse.click(hold_duration=hold_duration * 0.8)
            import threading
            threading.Thread(target=delayed_click, daemon=True).start()
            if self.debug_mode:
                print(f"   [DEBUG] Double-click!")
        else:
            self.mouse.click(hold_duration=hold_duration)

        if self.debug_mode:
            print(f"   [DEBUG] Click: hold={hold_duration*1000:.0f}ms")

    def _orbit_click_drag(self, target_x: float, target_y: float, duration: float = 0.5):
        """
        Perform click-and-drag orbiting motion (hold click while moving).
        Used for orbiting enemies in DarkOrbit.
        """
        from pynput.mouse import Button

        # Calculate orbit path (arc around target)
        start_x, start_y = self.mouse.mouse.position
        start_norm = (start_x / self.screen_width, start_y / self.screen_height)

        # Generate arc path
        path = self.movement_gen.generate_path(start_norm, (target_x, target_y))

        # Mouse down
        self.mouse.mouse.press(Button.left)

        # Move while holding
        self.mouse.execute_path_smooth(path, speed_multiplier=0.8)

        # Mouse up
        time.sleep(np.random.uniform(0.02, 0.05))
        self.mouse.mouse.release(Button.left)

        if self.debug_mode:
            print(f"   [DEBUG] Orbit drag: {duration*1000:.0f}ms")

    # Known enemy classes from the YOLO model
    ENEMY_CLASSES = ['Devo', 'Lordakia', 'Mordon', 'Saimon', 'Sibelon', 'Struener',
                     'npc', 'enemy']  # Include generic names as fallback
    # Known non-enemy classes (won't be attacked)
    NON_ENEMY_CLASSES = ['BonusBox', 'Player', 'player_ship', 'box', 'bonus_box', 'portal']

    def _find_nearest_target(self, detections: list, mode: str, mouse_x: float, mouse_y: float):
        """
        Find the nearest clickable target based on mode.
        Returns (target_x, target_y, target_type, distance) or None
        """
        if mode == "AGGRESSIVE":
            # Look for enemies - be flexible with class names
            # Treat anything that's not explicitly non-enemy as potential target
            enemies = [d for d in detections if
                       d.class_name in self.ENEMY_CLASSES or
                       d.class_name not in self.NON_ENEMY_CLASSES]
            if enemies:
                nearest = min(enemies, key=lambda d:
                    np.sqrt((d.x_center - mouse_x)**2 + (d.y_center - mouse_y)**2))
                dist = np.sqrt((nearest.x_center - mouse_x)**2 + (nearest.y_center - mouse_y)**2)
                return (nearest.x_center, nearest.y_center, nearest.class_name, dist)
        else:
            # Look for boxes - multiple possible names
            boxes = [d for d in detections if d.class_name in ['BonusBox', 'bonus_box', 'box']]
            if boxes:
                nearest = min(boxes, key=lambda d:
                    np.sqrt((d.x_center - mouse_x)**2 + (d.y_center - mouse_y)**2))
                dist = np.sqrt((nearest.x_center - mouse_x)**2 + (nearest.y_center - mouse_y)**2)
                return (nearest.x_center, nearest.y_center, 'BonusBox', dist)

        return None

    def _smart_target_override(self, action: Dict, detections: list, mode: str,
                                mouse_x: float, mouse_y: float) -> Dict:
        """
        Smart targeting: If there's a valid target visible, aim at it and click.

        AGGRESSIVE BEHAVIOR - network is biased to not click, so we override:
        - ALWAYS aim at nearest target
        - ALWAYS click when target is visible (in AGGRESSIVE mode)
        - Click when close in PASSIVE mode
        - If enemies visible in PASSIVE mode, switch to attack them anyway
        """
        if mode == "AGGRESSIVE":
            pos_key_x, pos_key_y = 'aim_x', 'aim_y'
            click_key = 'should_fire'
        else:
            pos_key_x, pos_key_y = 'move_x', 'move_y'
            click_key = 'should_click'

        # DEBUG: Log what we're working with
        if self.debug_mode:
            enemies = [d for d in detections if d.class_name in
                      ['Devo', 'Lordakia', 'Mordon', 'Saimon', 'Sibelon', 'Struener']]
            boxes = [d for d in detections if d.class_name == 'BonusBox']
            print(f"\n   [SMART] Mode={mode}, Enemies={len(enemies)}, Boxes={len(boxes)}, "
                  f"ActionBefore=({action.get(pos_key_x, 'N/A'):.3f}, {action.get(pos_key_y, 'N/A'):.3f})")

        # Find nearest valid target for current mode
        target = self._find_nearest_target(detections, mode, mouse_x, mouse_y)

        # FALLBACK: If in PASSIVE mode but no boxes, check for enemies
        if target is None and mode == "PASSIVE":
            target = self._find_nearest_target(detections, "AGGRESSIVE", mouse_x, mouse_y)
            if target and self.debug_mode:
                print(f"\n   [FALLBACK] No boxes, found enemy: {target[2]}")

        if target:
            target_x, target_y, target_type, distance = target

            # ALWAYS aim at the target
            action[pos_key_x] = target_x
            action[pos_key_y] = target_y

            # Determine if this is an enemy (always click) or a box (click when close)
            is_enemy = target_type not in ['BonusBox']

            if is_enemy:
                # Always attack enemies when visible (regardless of mode)
                action[click_key] = True
                if self.debug_mode:
                    print(f"\n   [ATTACK] {target_type} @ dist={distance:.3f}")
            else:
                # Click on boxes when reasonably close
                if distance < 0.15:  # Within 15% of screen
                    action[click_key] = True
                    if self.debug_mode:
                        print(f"\n   [COLLECT] {target_type} @ dist={distance:.3f}")
                elif self.debug_mode:
                    print(f"\n   [MOVE] Toward {target_type} @ dist={distance:.3f}")

        # DEBUG: Log final action after override
        if self.debug_mode:
            print(f"   [SMART] ActionAfter=({action.get(pos_key_x, 'N/A'):.3f}, {action.get(pos_key_y, 'N/A'):.3f}), Click={action.get(click_key, 'N/A')}")

        return action

    def _has_visible_targets(self, detections: list) -> bool:
        """Check if there are any actionable targets visible"""
        for d in detections:
            # Check for enemies (flexible matching)
            if d.class_name in self.ENEMY_CLASSES or d.class_name not in self.NON_ENEMY_CLASSES:
                return True
            # Check for boxes
            if d.class_name in ['BonusBox', 'bonus_box', 'box']:
                return True
        return False

    def _get_exploration_action(self) -> Optional[Dict]:
        """
        Generate exploration action when no targets visible.
        Clicks on random map positions to fly around and discover targets.
        """
        now = time.time()

        # Only explore after being idle briefly (0.3s - quick to start moving)
        if self.idle_time < 0.3:
            return None

        # Cooldown between exploration moves (faster exploration)
        if now - self.last_exploration_time < self.exploration_cooldown:
            return None

        # Generate new exploration target (random position on map)
        # Avoid edges, focus on playable area (roughly center 70% of screen)
        self.exploration_target = (
            np.random.uniform(0.15, 0.85),
            np.random.uniform(0.15, 0.85)
        )
        self.last_exploration_time = now

        # Faster exploration cooldown (1-2.5s between moves)
        self.exploration_cooldown = np.random.uniform(1.0, 2.5)

        if self.debug_mode:
            print(f"\n   [EXPLORE] Flying to ({self.exploration_target[0]:.2f}, {self.exploration_target[1]:.2f})")

        return {
            'move_x': self.exploration_target[0],
            'move_y': self.exploration_target[1],
            'should_click': True,  # Click to move ship
            'raw_click_value': 1.0,
            'key_action': 0,
            'wait_time': 0
        }

    def _apply_vision_context(self, action: Dict, mode: str) -> Dict:  # noqa: ARG002
        """
        Use VLM context to modify behavior.
        E.g., flee when health critical, be more aggressive when threat is low.
        Mode param reserved for future mode-specific VLM behavior.
        """
        if not self.last_vision_context:
            return action

        ctx = self.last_vision_context

        # If VLM says we're in critical danger, prioritize fleeing
        if ctx.threat_level == "critical" or ctx.player_status == "critical":
            # Move away from center (edge escape)
            if self.debug_mode:
                print(f"\n   [VLM OVERRIDE] Critical threat - fleeing!")
            # Don't attack, just move to safety
            if 'should_fire' in action:
                action['should_fire'] = False
            if 'should_click' in action:
                action['should_click'] = False

        # If VLM recommends specific action, log it
        if ctx.recommended_action and self.debug_mode:
            if 'flee' in ctx.recommended_action.lower():
                print(f"\n   [VLM] Suggests fleeing")
            elif 'attack' in ctx.recommended_action.lower():
                print(f"\n   [VLM] Suggests attacking")

        return action

    def _handle_combat_keys(self, mode: str, has_enemy: bool, health: float = 1.0):  # noqa: ARG002
        """
        Handle combat-related key presses with human-like timing.
        Ctrl = toggle attack, Space = fire rocket

        Human-like behaviors:
        - Variable delays between key presses
        - Occasional hesitation before pressing
        - Doesn't spam keys robotically
        """
        now = time.time()

        if mode == "AGGRESSIVE" and has_enemy:
            # Toggle attack ON if not already attacking
            # FAST reaction - good players react in 0.1-0.3s
            min_delay = np.random.uniform(0.1, 0.25)
            if not self.is_attacking and now - self.last_attack_toggle > min_delay:
                # Quick hesitation before pressing key (human-like)
                time.sleep(np.random.uniform(0.02, 0.06))
                self.keyboard.toggle_attack()
                self.is_attacking = True
                self.last_attack_toggle = now
                # Always log when toggling attack - important for debugging
                print(f"\n   [KEY] Attack ON (Ctrl pressed for toggle)")

            # Re-press Ctrl periodically to ensure attack stays on
            # Some games reset attack state, so re-toggle every 2-4 seconds
            elif self.is_attacking and now - self.last_attack_toggle > np.random.uniform(2.0, 4.0):
                self.keyboard.toggle_attack()
                self.last_attack_toggle = now
                if self.debug_mode:
                    print(f"\n   [KEY] Attack refresh (Ctrl re-pressed)")

            # Occasionally fire rockets in combat
            # More frequent rocket usage (every ~2-3 seconds on average)
            if np.random.random() < 0.05:
                time.sleep(np.random.uniform(0.03, 0.08))
                self.keyboard.fire_rocket()
                if self.debug_mode:
                    print(f"\n   [KEY] Rocket fired (Space)")

        elif self.is_attacking and not has_enemy:
            # Toggle attack OFF when no enemies
            # Quick to stop attacking (0.5-1.0s after enemies gone)
            off_delay = np.random.uniform(0.5, 1.0)
            if now - self.last_attack_toggle > off_delay:
                time.sleep(np.random.uniform(0.02, 0.05))
                self.keyboard.toggle_attack()
                self.is_attacking = False
                self.last_attack_toggle = now
                if self.debug_mode:
                    print(f"\n   [KEY] Attack OFF (Ctrl)")

    def _print_status(self, mode: str, health: float, detections: list):
        """Print current status"""
        enemies = sum(1 for d in detections if
                      d.class_name in self.ENEMY_CLASSES or d.class_name not in self.NON_ENEMY_CLASSES)
        boxes = sum(1 for d in detections if d.class_name in ['BonusBox', 'bonus_box', 'box'])

        # Show current state
        if self.idle_time > 1.0:
            state = "EXPLORE"
        elif self.is_attacking:
            state = "ATTACK!"
        else:
            state = mode

        status = f"\r🤖 {'⏸️' if self.paused else '▶️'} | "
        status += f"{state:10} | "
        status += f"HP: {health:3.0%} | "
        status += f"E:{enemies} B:{boxes} | "
        status += f"Act:{self.total_actions} Clk:{self.total_clicks}    "

        print(status, end="")
        
    def _print_stats(self):
        """Print session stats including performance metrics"""
        if self.session_start:
            duration = time.time() - self.session_start
            print(f"\n\n📊 Session: {duration:.0f}s | Actions: {self.total_actions} | Clicks: {self.total_clicks}")

            # Performance metrics
            if self.frame_times:
                avg_frame = sum(self.frame_times) / len(self.frame_times)
                max_frame = max(self.frame_times)
                fps = 1.0 / avg_frame if avg_frame > 0 else 0

                print(f"   Performance: {fps:.1f} FPS avg | Frame: {avg_frame*1000:.1f}ms avg, {max_frame*1000:.0f}ms max")

                # Warn if performance is poor
                if avg_frame > 0.05:  # Less than 20 FPS
                    print(f"   ⚠️ Performance is slow! Check GPU usage or reduce VLM frequency.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='DarkOrbit Bot Controller')
    parser.add_argument('--model', type=str, default='F:/dev/bot/best.pt',
                       help='Path to YOLO model')
    parser.add_argument('--policy', type=str, default=None,
                       help='Path to trained policy (optional)')
    parser.add_argument('--monitor', type=int, default=1,
                       help='Monitor to capture')
    parser.add_argument('--noise', type=float, default=0.02,
                       help='Precision noise (0-0.1)')
    parser.add_argument('--self-improve', action='store_true',
                       help='Enable VLM self-improvement (requires LM Studio)')
    parser.add_argument('--enhanced-vlm', action='store_true',
                       help='Enable enhanced multi-level VLM analysis (Strategic/Tactical/Execution)')
    parser.add_argument('--auto-mode', action='store_true',
                       help='Let the network learn when to be aggressive (from your gameplay)')
    parser.add_argument('--meta-learn', action='store_true',
                       help='Run meta-analysis at end of session to improve VLM prompts')
    parser.add_argument('--no-prompt', action='store_true',
                       help='Skip all interactive prompts (for automated runs)')

    args = parser.parse_args()

    config = BotConfig(
        model_path=args.model,
        policy_path=args.policy,
        monitor=args.monitor,
        precision_noise=args.noise,
        self_improve=args.self_improve,
        enhanced_vlm=args.enhanced_vlm,
        auto_mode=args.auto_mode,
        meta_learn=args.meta_learn,
        no_prompt=args.no_prompt
    )
    
    bot = BotController(config)
    
    print("\n🤖 Bot Controller Ready")
    if args.enhanced_vlm:
        print("   🔬 ENHANCED-VLM: Multi-level training data generation")
        print("      Corrections saved to: data/vlm_corrections/enhanced_*.json")
    elif args.self_improve:
        print("   📊 SELF-IMPROVE: Basic VLM critique enabled")
    if args.auto_mode:
        print("   🧠 AUTO-MODE: Network decides when to attack (learned from gameplay)")
    else:
        print("   📋 CONTEXT-MODE: Rule-based mode switching")
    if args.meta_learn:
        print("   🔄 META-LEARN: Will analyze VLM performance at session end")
    print("   Press F1 to start/pause")
    print("   Press F3 for emergency stop")
    print("   Press Ctrl+C to exit")
    
    bot.start()
    
    try:
        while bot.running:
            time.sleep(1)
    except KeyboardInterrupt:
        bot.stop()


if __name__ == "__main__":
    main()
