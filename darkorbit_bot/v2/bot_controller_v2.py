"""
V2 Bot Controller - Hierarchical Temporal Architecture

Uses the V2 hierarchical policy:
- Strategist (1Hz): Goal and mode selection
- Tactician (10Hz): Target selection
- Executor (60Hz): Motor control

Features:
- ByteTrack object tracking for persistent target IDs
- Rich state encoding with temporal context
- Multi-timescale decision making
- Clean separation of perception, decision, and action

Hotkeys:
- F1 = Start/Stop bot
- F2 = BAD STOP (save as negative training data, like V1)
- F3 = Emergency stop
- F4 = Toggle debug logging
- F5 = Toggle reasoning log
- F6 = Cycle mode override (AUTO → FIGHT → LOOT → FLEE → EXPLORE → CAUTIOUS → AUTO)
"""

import time
import sys
import threading
import json
import os
from pathlib import Path
from dataclasses import dataclass
from collections import deque
from typing import Optional, Dict, List
import numpy as np
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# V2 imports
from .perception.tracker import ObjectTracker, TrackedObject
from .perception.state_encoder import StateEncoderV2, StateSummarizer, PlayerState
from .perception.vision_encoder import VisionEncoder, VisionConfig, create_vision_encoder
from .perception.hud_ocr import HUDReader, HUDConfig, create_hud_reader, TargetHPReader
from .perception.gui_detector import GUIDetector, MinimapDetector
from .models.unified import HierarchicalPolicy, create_hierarchical_policy, load_hierarchical_policy

# V2 VLM
from .vlm.vlm_v2 import VLM_V2

# V2 Online Learning
from .training.online_learner import OnlineLearner, OutcomeTracker, VisualOutcomeLearner
from .training.shadow_trainer import ShadowTrainer
from .training.dagger import DAggerTrainer
from .training.training_utils import CheckpointManager, compute_score

# Auto-Labeler (self-improving YOLO)
AutoLabeler = None
_auto_labeler_error = None
try:
    # Try direct import first (works when darkorbit_bot is in path)
    from training.auto_labeler import AutoLabeler as _AutoLabeler
    AutoLabeler = _AutoLabeler
except ImportError as e1:
    try:
        # Try relative to this file's parent (darkorbit_bot/)
        import importlib.util
        _auto_labeler_path = Path(__file__).parent.parent / "training" / "auto_labeler.py"
        if _auto_labeler_path.exists():
            spec = importlib.util.spec_from_file_location("auto_labeler", _auto_labeler_path)
            _auto_labeler_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_auto_labeler_module)
            AutoLabeler = _auto_labeler_module.AutoLabeler
        else:
            _auto_labeler_error = f"File not found: {_auto_labeler_path}"
    except Exception as e2:
        _auto_labeler_error = f"Import failed: {e1}, then {e2}"

# V1 imports (reuse detection and humanization)
try:
    from detection.detector import GameDetector, ScreenCapture
    from reasoning.filters import create_filters
    from movement.generator import MovementGenerator, MouseController, MovementProfile, KeyboardController
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the darkorbit_bot directory")
    sys.exit(1)


def detection_attention_map(detections: list, output_size: tuple = (28, 28)) -> np.ndarray:
    """
    Creates an attention heatmap based on ACTUAL AI detections (YOLO).

    This shows EXACTLY what the AI is "seeing" - the detected objects.
    Each detection creates a Gaussian blob at its location, with intensity
    based on confidence. This is the TRUE AI attention map.

    Args:
        detections: List of detection dicts with x_min, y_min, x_max, y_max, confidence
        output_size: Output heatmap size

    Returns:
        Attention map [H, W] (uint8, 0-255)
    """
    import cv2

    # Create blank heatmap at output resolution
    h, w = output_size
    heatmap = np.zeros((h, w), dtype=np.float32)

    for det in detections:
        # Get normalized center position (0-1)
        x_min = det.get('x_min', 0)
        y_min = det.get('y_min', 0)
        x_max = det.get('x_max', 0)
        y_max = det.get('y_max', 0)

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        conf = det.get('confidence', 0.5)

        # Get object size for blob radius (larger objects = larger blobs)
        obj_w = x_max - x_min
        obj_h = y_max - y_min
        # Radius proportional to object size, scaled to heatmap
        radius = max(obj_w * w, obj_h * h) * 0.6
        radius = max(2, min(radius, 10))  # Clamp between 2-10 pixels

        # Convert to heatmap pixel coordinates
        hx, hy = int(cx * w), int(cy * h)

        # Draw Gaussian blob (intensity = confidence)
        r_int = int(radius * 2) + 1
        for dy in range(-r_int, r_int + 1):
            for dx in range(-r_int, r_int + 1):
                px, py = hx + dx, hy + dy
                if 0 <= px < w and 0 <= py < h:
                    dist_sq = dx * dx + dy * dy
                    sigma_sq = radius * radius
                    intensity = conf * np.exp(-dist_sq / (2 * sigma_sq))
                    heatmap[py, px] = max(heatmap[py, px], intensity)

    # Normalize to 0-255
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)

    return heatmap


class DebugSocketServer:
    """
    Non-blocking TCP Socket Server for Debug Broadcasting.
    Replaces slow file-based IPC.
    """
    def __init__(self, port=9999):
        self.port = port
        self.running = False
        self.server_socket = None
        self.clients = []
        self.latest_state = None
        self.lock = threading.Lock()
        self.thread = None
        self.broadcast_thread = None

    def start(self):
        self.running = True
        try:
            import socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # TCP_NODELAY disables Nagle's algorithm - critical for low latency!
            self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.server_socket.bind(('127.0.0.1', self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)

            self.thread = threading.Thread(target=self._accept_loop, daemon=True)
            self.thread.start()

            self.broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True)
            self.broadcast_thread.start()

            print(f"[DebugServer] Listening on 127.0.0.1:{self.port} (TCP_NODELAY enabled)")
        except Exception as e:
            print(f"[DebugServer] Failed to start: {e}")
            self.running = False

    def stop(self):
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except: 
                pass
        # Close all clients
        with self.lock:
            for client in self.clients:
                try: client.close()
                except: pass
            self.clients.clear()

    def update(self, state: Dict):
        """Update the state to be broadcast (Non-blocking)."""
        # Just swap the pointer, super fast
        self.latest_state = state

    def _accept_loop(self):
        import socket
        while self.running:
            try:
                client, addr = self.server_socket.accept()
                # TCP_NODELAY on client socket too - critical for low latency!
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client.setblocking(False)  # Non-blocking send
                with self.lock:
                    self.clients.append(client)
                print(f"[DebugServer] Client connected: {addr} (TCP_NODELAY)")
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[DebugServer] Accept error: {e}")
                break

    def _broadcast_loop(self):
        """Broadcasts latest state with minimal latency."""
        import struct
        import json

        last_sent_time = 0

        while self.running:
            # Short sleep - check for new state frequently (60Hz)
            # Lower latency than 30Hz, but still CPU-friendly
            time.sleep(0.016)

            state = self.latest_state
            if state is None:
                continue

            # Anti-Stale: Don't resend the exact same state timestamp multiple times
            state_time = state.get('timestamp', 0)
            if state_time <= last_sent_time:
                continue
            last_sent_time = state_time
                
            try:
                # Serialize
                json_str = json.dumps(state)
                data = json_str.encode('utf-8')
                # Header: 4 bytes length (Big Endian)
                header = struct.pack('>I', len(data))
                message = header + data
                
                # Send to all clients
                to_remove = []
                with self.lock:
                    if not self.clients:
                        continue
                        
                    for client in self.clients:
                        try:
                            # Non-blocking send
                            # If buffer is full, it raises BlockingIOError (Windows) or EAGAIN (Linux)
                            # We just catch it and drop the frame for that client (LAG PREVENTION)
                            client.sendall(message) 
                        except (BlockingIOError, OSError) as e:
                            # "WinError 10035" or "EAGAIN" means buffer full.
                            # Drop frame. Do NOT disconnect.
                            continue
                        except Exception as e:
                            # Real error, disconnect
                            to_remove.append(client)
                    
                    for client in to_remove:
                        self.clients.remove(client)
                        try: client.close()
                        except: pass
            except Exception as e:
                print(f"[DebugServer] Broadcast error: {e}") 


@dataclass
class BotConfigV2:
    """V2 Bot configuration"""
    # Detection
    model_path: str = "F:/dev/bot/best.pt"
    monitor: int = 1

    # Policy
    policy_dir: Optional[str] = None  # Directory with strategist.pt, tactician.pt, executor.pt
    device: str = "cuda"

    # Humanization
    reaction_delay_ms: int = 40  # Fast reactions for V2
    precision_noise: float = 0.03  # Lower noise for trained model

    # Safety
    max_actions_per_second: int = 60  # V2 runs faster

    # Tracking
    tracker_high_thresh: float = 0.6
    tracker_low_thresh: float = 0.1
    tracker_buffer: int = 30

    # VLM
    vlm_enabled: bool = False  # Enable EnhancedVLM analysis
    vlm_url: str = "http://localhost:1234"
    vlm_model: str = "local-model"
    vlm_corrections: bool = False  # Save VLM corrections for training

    # Online Learning
    online_learning: bool = False  # Enable real-time learning from hits/misses (disabled by default)
    online_learning_rate: float = 1e-5  # Small LR to avoid destabilizing
    online_update_interval: float = 5.0  # Seconds between weight updates

    # Auto-Labeling (self-improving YOLO with Gemini)
    auto_label: bool = False  # Enable auto-labeling during gameplay
    auto_label_dir: str = "data/auto_labeled"  # Output directory for labeled data
    auto_label_interval: int = 600  # Sample every N frames (higher = less lag)
    gemini_api_key: str = None  # Google API key (or set GOOGLE_API_KEY env var)
    gemini_model: str = "gemini-2.0-flash"  # Fast Gemini model for classification

    # Visual Features (ALWAYS enabled - for policy + debug heatmap)
    visual_features: bool = True  # Enable visual feature extraction (CNN encoder)
    visual_lightweight: bool = False  # Use lightweight color encoder (faster, no GPU)

    # HUD OCR (for real HP/Shield reading)
    hud_ocr_enabled: bool = True  # Enable OCR-based HP/Shield reading
    hud_ocr_backend: str = "color_only"  # "easyocr", "tesseract", "paddleocr", "color_only"

    # Shadow Training (learn from human play)
    shadow_train: bool = False  # Watch human play and learn from their actions
    shadow_train_lr: float = 1e-4  # Higher LR for direct imitation learning
    save_recordings: bool = False  # Save full hierarchical recordings for offline training

    # DAgger (corrective learning during bot play)
    dagger_enabled: bool = False  # Enable DAgger: human corrections during bot play get 3x weight
    dagger_mouse_threshold: float = 0.1  # Mouse distance to detect correction

    # Smart Checkpointing
    checkpoint_keep_top_n: int = 5  # Keep top N checkpoints by score
    checkpoint_keep_latest_n: int = 3  # Also keep N most recent
    checkpoint_save_interval: int = 100  # Save every N training updates

    # Tensorboard Logging
    log_dir: str = "runs"  # Directory for tensorboard logs

    # GUI Masking and Map Awareness
    gui_masking: bool = True  # Prevent clicks on GUI elements
    minimap_tracking: bool = True  # Extract position from minimap
    minimap_position: str = "top-right"  # Minimap location: "top-right", "top-left", "bottom-right", "bottom-left"

    # Debug IPC Broadcasting (for debug viewer)
    broadcast_debug: bool = True  # Always enabled - minimal overhead when no viewer connected
    debug_ipc_path: str = "C:/tmp/darkorbit_bot_debug.json"  # Legacy IPC file path (unused)


class BotControllerV2:
    """
    V2 Bot Controller with Hierarchical Policy.

    Architecture:
    1. PERCEIVE: YOLO → ByteTrack → StateEncoder
    2. DECIDE: Strategist → Tactician → Executor (hierarchical)
    3. ACT: Humanizer → Mouse/Keyboard
    """

    # Mode definitions
    MODES = ['FIGHT', 'LOOT', 'FLEE', 'EXPLORE', 'CAUTIOUS']

    # Object class definitions
    ENEMY_CLASSES = ['Devo', 'Lordakia', 'Mordon', 'Saimon', 'Sibelon', 'Struener', 'npc', 'enemy']
    LOOT_CLASSES = ['BonusBox', 'bonus_box', 'box']
    NON_TARGET_CLASSES = ['Player', 'player_ship', 'portal']

    def __init__(self, config: BotConfigV2):
        self.config = config
        self.running = False
        self.paused = False
        self.mode_override: Optional[str] = None  # Force specific mode

        # ═══════════════════════════════════════════════════════════
        # GPU STATUS CHECK
        # ═══════════════════════════════════════════════════════════
        self._check_gpu_status()

        # ═══════════════════════════════════════════════════════════
        # PERCEPTION LAYER
        # ═══════════════════════════════════════════════════════════
        print("[V2] Initializing Perception...")

        # YOLO detector (from V1) with optimized NMS settings
        self.detector = GameDetector(
            config.model_path,
            confidence_threshold=0.3,  # Optimized: filter weak predictions
            iou_threshold=0.3          # Optimized: aggressive duplicate removal
        )
        self.screen = ScreenCapture(monitor_index=config.monitor) if config.monitor else None
        
        # Socket IPC Server
        self.debug_server = None
        if config.broadcast_debug:
             self.debug_server = DebugSocketServer(port=9999)
             self.debug_server.start()

        # ByteTrack object tracker
        self.tracker = ObjectTracker(
            high_thresh=config.tracker_high_thresh,
            low_thresh=config.tracker_low_thresh,
            track_buffer=config.tracker_buffer,
            match_thresh=0.3  # Allow matches with just 30% overlap (fixes fast movement tracking)
        )

        # State encoder
        self.state_encoder = StateEncoderV2(max_objects=16)

        # Temporal summarizer for strategist (60s at 1Hz)
        self.state_summarizer = StateSummarizer(history_seconds=60, sample_rate_hz=1)

        # Health filter (from V1 - FALLBACK)
        self.filters = create_filters()

        # HUD OCR Reader (for REAL HP/Shield values)
        self.hud_reader: Optional[HUDReader] = None
        self.target_hp_reader: Optional[TargetHPReader] = None
        self._use_hud_ocr = config.hud_ocr_enabled

        if config.hud_ocr_enabled:
            try:
                self.hud_reader = create_hud_reader(ocr_backend=config.hud_ocr_backend)
                self.target_hp_reader = TargetHPReader()
                print(f"   ✅ HUD OCR enabled (backend: {config.hud_ocr_backend})")
            except Exception as e:
                print(f"   ⚠️ HUD OCR init failed: {e}, using fallback estimation")
                self._use_hud_ocr = False

        # GUI Detector config (will be initialized after screen dimensions are known)
        self.gui_detector: Optional[GUIDetector] = None
        self._gui_masking_enabled = config.gui_masking
        self._minimap_tracking_enabled = config.minimap_tracking

        # ═══════════════════════════════════════════════════════════
        # DECISION LAYER
        # ═══════════════════════════════════════════════════════════
        print("[V2] Initializing Hierarchical Policy...")

        if config.policy_dir and Path(config.policy_dir).exists():
            self.policy = load_hierarchical_policy(config.policy_dir, device=config.device)
            print(f"   Loaded trained policy from {config.policy_dir}")
        else:
            self.policy = create_hierarchical_policy(device=config.device)
            print("   Using untrained policy (demo mode)")
            
        # Vision Encoder (Game Sense)
        # Even if policy doesn't need it, we might want it for the Heatmap Debugger
        self.vision_encoder = None
        if config.visual_features:
            print("[V2] Initializing Vision Encoder (MobileNetV3)...")
            try:
                self.vision_encoder = create_vision_encoder(
                    VisionConfig(backbone="mobilenet_v3_small", device=config.device)
                )
                self.vision_encoder.eval()
            except Exception as e:
                print(f"[V2] Vision Init Warning: {e}")

        # ═══════════════════════════════════════════════════════════
        # BACKGROUND HEATMAP THREAD (for low-latency debug viewer)
        # ═══════════════════════════════════════════════════════════
        # Heatmap is generated in background at 60 FPS, never blocking main loop
        self._heatmap_cache = ("", [0, 0])  # (base64_data, shape)
        self._heatmap_lock = threading.Lock()
        self._heatmap_thread = None
        self.latest_frame = None  # Shared with heatmap thread
        self._latest_detections_for_heatmap = []  # YOLO detections for true AI attention map

        # ═══════════════════════════════════════════════════════════
        # ACTION LAYER (Humanization from V1)
        # ═══════════════════════════════════════════════════════════
        print("[V2] Initializing Humanizer...")

        # Load movement profile
        profile_path = Path(__file__).parent.parent / "data" / "my_movement_profile.json"
        if profile_path.exists():
            try:
                self.profile = MovementProfile.load(str(profile_path))
                print(f"   Loaded movement profile")
            except Exception:
                self.profile = MovementProfile()
        else:
            self.profile = MovementProfile()

        self.movement_gen = MovementGenerator(self.profile)

        # Get screen resolution from the ScreenCapture monitor (matches what we're capturing)
        # This ensures mouse coordinates align with the captured game area
        self.screen_width = self.screen.monitor['width']
        self.screen_height = self.screen.monitor['height']
        self.screen_left = self.screen.monitor.get('left', 0)
        self.screen_top = self.screen.monitor.get('top', 0)
        print(f"   Screen: {self.screen_width}x{self.screen_height} at ({self.screen_left}, {self.screen_top})")

        # Initialize GUI detector now that screen dimensions are known
        if config.gui_masking or config.minimap_tracking:
            self.gui_detector = GUIDetector(
                screen_width=self.screen_width,
                screen_height=self.screen_height
            )

            features = []
            if config.gui_masking:
                features.append("YOLO-based click masking")
            if config.minimap_tracking:
                features.append("minimap position tracking")
            print(f"   ✅ GUI detector enabled ({' + '.join(features)})")

        self.mouse = MouseController(self.screen_width, self.screen_height, game_rect={
            "left": self.screen_left,
            "top": self.screen_top,
            "width": self.screen_width,
            "height": self.screen_height
        })
        self.keyboard = KeyboardController()

        # ═══════════════════════════════════════════════════════════
        # STATE TRACKING
        # ═══════════════════════════════════════════════════════════
        self.last_action_time = 0.0
        self.action_cooldown = 1.0 / config.max_actions_per_second

        # Combat state
        self.is_attacking = False
        self.last_attack_toggle = 0.0
        self.current_target_id: Optional[int] = None  # Track ID of current target

        # Player state
        self.player_x = 0.5
        self.player_y = 0.5
        self.health = 1.0
        self.shield = 1.0
        self.idle_time = 0.0

        # Stats
        self.total_actions = 0
        self.total_clicks = 0
        self.session_start = None
        self.frame_count = 0

        # Debug
        self.debug_mode = False
        self.reasoning_log_enabled = True

        # Bad stop buffer (for F2 negative training data like V1)
        self.BAD_STOP_BUFFER_SIZE = 100  # Keep last 100 frames
        self.bad_stop_buffer = deque(maxlen=self.BAD_STOP_BUFFER_SIZE)
        self.BAD_STOP_SAVE_COUNT = 20    # Save last 20 frames when F2 pressed
        self.last_reasoning = ""
        self.last_reasoning_time = 0

        # VLM result tracking (to avoid printing same result multiple times)
        self._last_vlm_strategist_time = 0
        self._last_vlm_tactician_time = 0
        self._last_vlm_executor_time = 0

        # Threading
        self.bot_thread: Optional[threading.Thread] = None
        self.kb_listener = None

        # Performance profiling (capped to prevent unbounded growth)
        self._profile_times = {k: deque(maxlen=100) for k in ['capture', 'yolo', 'tracker', 'policy', 'execute']}
        self._profile_count = 0

        # ═══════════════════════════════════════════════════════════
        # V2 VLM (Optional) - Hierarchical architecture aware
        # ═══════════════════════════════════════════════════════════
        self.vlm: Optional[VLM_V2] = None
        if config.vlm_enabled:
            print("[V2] Initializing V2 VLM (hierarchical)...")
            self.vlm = VLM_V2(
                base_url=config.vlm_url,
                model=config.vlm_model
            )
            if config.vlm_corrections:
                self.vlm.enable_corrections(True)
            self.vlm.start_async()
            print(f"   ✅ VLM-V2 connected to {config.vlm_url}")

        # ═══════════════════════════════════════════════════════════
        # ONLINE LEARNING (Real-time from hits/misses + visual outcomes)
        # ═══════════════════════════════════════════════════════════
        self.online_learner: Optional[VisualOutcomeLearner] = None
        self.outcome_tracker: Optional[OutcomeTracker] = None  # Legacy, kept for compatibility
        self._use_visual_outcomes = config.visual_features  # Use visual detection when enabled
        if config.online_learning:
            print("[V2] Initializing Visual Outcome Learner...")
            # Use VisualOutcomeLearner which combines:
            # 1. Base OnlineLearner for hit/miss, distance rewards
            # 2. VisualOutcomeTracker for AI-detected events (explosions, damage, etc.)
            self.online_learner = VisualOutcomeLearner(
                executor=self.policy.executor,
                device=config.device,
                learning_rate=config.online_learning_rate,
                buffer_size=2000,
                use_visual_outcomes=self._use_visual_outcomes,
                log_dir=config.log_dir
            )
            # Keep legacy OutcomeTracker for fallback/comparison
            self.outcome_tracker = OutcomeTracker(
                reward_decay=0.95,
                history_length=100
            )
            print(f"   ✅ Online learning enabled (LR={config.online_learning_rate})")
            if self._use_visual_outcomes:
                print(f"   ✅ Visual outcome detection enabled (AI-based event recognition)")
                print(f"   ✅ Weak supervision ENABLED - will self-train from HP drops, kills, deaths")
                print(f"   ℹ️  Detection activates after ~100 weak supervision samples")
            else:
                print(f"   ℹ️  Using basic outcome tracking (visual features disabled)")

        # Track target HP for hit detection (basic fallback)
        self._prev_target_hp: Optional[float] = None
        self._current_target_id: Optional[int] = None
        self._prev_player_hp: float = 1.0
        self._tracked_enemy_ids: set = set()  # For legacy kill detection

        # ═══════════════════════════════════════════════════════════
        # SMART CHECKPOINT MANAGER
        # ═══════════════════════════════════════════════════════════
        self.checkpoint_manager: Optional[CheckpointManager] = None
        if config.online_learning or config.shadow_train:
            default_policy_dir = Path(__file__).parent / "policies"
            checkpoint_dir = Path(config.policy_dir) / "checkpoints" if config.policy_dir else default_policy_dir / "checkpoints"
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=str(checkpoint_dir),
                model_name="executor",
                keep_top_n=config.checkpoint_keep_top_n,
                keep_latest_n=config.checkpoint_keep_latest_n
            )
            print(f"   [CHECKPOINT] Smart checkpointing enabled: {checkpoint_dir}")

        # ═══════════════════════════════════════════════════════════
        # SHADOW TRAINING (Learn from human play)
        # ═══════════════════════════════════════════════════════════
        self.shadow_trainer: Optional[ShadowTrainer] = None
        self._shadow_mode = config.shadow_train
        if config.shadow_train:
            print("[V2] Initializing Shadow Trainer...")
            print("[V2] 🔒 Mouse cursor will be locked to screen during recording (prevents bad training data)")
            self.shadow_trainer = ShadowTrainer(
                executor=self.policy.executor,
                device=config.device,
                learning_rate=config.shadow_train_lr,
                buffer_size=5000,
                screen_left=self.screen_left,
                screen_top=self.screen_top,
                screen_width=self.screen_width,
                screen_height=self.screen_height,
                log_dir=config.log_dir,
                save_full_demos=config.save_recordings,
                recording_dir="darkorbit_bot/data/recordings_v2"  # Same folder as recorder_v2 and training scripts
            )
            print(f"   Shadow training enabled (LR={config.shadow_train_lr})")
            print(f"   Bot will WATCH you play - no mouse/keyboard control")
            if config.save_recordings:
                print(f"   Full recordings ENABLED - saving to darkorbit_bot/data/recordings_v2/ for offline training")
                print(f"   This captures mode + target selection + actions for all 3 models!")

        # DAgger corrective learning (active during bot play, not shadow mode)
        self.dagger: Optional[DAggerTrainer] = None
        if config.dagger_enabled and not self._shadow_mode:
            # DAgger needs a shadow_trainer for its demo buffer + training
            if not self.shadow_trainer:
                self.shadow_trainer = ShadowTrainer(
                    executor=self.policy.executor,
                    device=config.device,
                    learning_rate=config.shadow_train_lr,
                    buffer_size=5000,
                    screen_left=self.screen_left,
                    screen_top=self.screen_top,
                    screen_width=self.screen_width,
                    screen_height=self.screen_height,
                    log_dir=config.log_dir,
                )
            self.dagger = DAggerTrainer(
                shadow_trainer=self.shadow_trainer,
                executor=self.policy.executor,
                device=config.device,
                mouse_threshold=config.dagger_mouse_threshold,
            )
            print(f"   [DAgger] Corrective learning enabled (corrections get 3x weight)")

        # ═══════════════════════════════════════════════════════════
        # AUTO-LABELER (Self-improving YOLO with Gemini)
        # ═══════════════════════════════════════════════════════════
        self.auto_labeler: Optional['AutoLabeler'] = None
        if config.auto_label:
            if AutoLabeler is None:
                err_msg = _auto_labeler_error or "unknown import error"
                print(f"[V2] ⚠️ Auto-labeler requested but not available: {err_msg}")
            else:
                print("[V2] Initializing Auto-Labeler...")
                try:
                    self.auto_labeler = AutoLabeler(
                        output_dir=config.auto_label_dir,
                        periodic_interval=config.auto_label_interval,
                        gemini_api_key=config.gemini_api_key,
                        gemini_model=config.gemini_model
                    )
                    # Pass YOLO class map so labels match your model
                    if hasattr(self.detector, 'class_names') and self.detector.class_names:
                        self.auto_labeler.set_class_map(self.detector.class_names)
                    self.auto_labeler.start()
                    print(f"   ✅ Auto-labeler enabled (output: {config.auto_label_dir})")
                except Exception as e:
                    print(f"[V2] ⚠️ Auto-labeler init failed: {e}")

        # ═══════════════════════════════════════════════════════════
        # VISUAL ENCODER (for visual-enabled models)
        # ═══════════════════════════════════════════════════════════
        self.vision_encoder = None
        self._visual_history_max = 60  # 60 samples for 60s at 1Hz
        self._visual_history_buffer = deque(maxlen=self._visual_history_max)

        # Check if policy has visual-enabled models (for info only)
        policy_has_visual = (
            hasattr(self.policy, 'strategist_has_visual') and self.policy.strategist_has_visual
        )

        # ALWAYS create vision encoder when visual_features=True
        # Even if policy doesn't use visual, we want heatmap for debug + future training
        if config.visual_features:
            print(f"[V2] Initializing Visual Encoder...")
            try:
                vision_config = VisionConfig(device=config.device)
                self.vision_encoder = create_vision_encoder(
                    config=vision_config,
                    lightweight=config.visual_lightweight
                )
                print(f"   ✅ Visual encoder enabled ({'lightweight' if config.visual_lightweight else 'CNN'})")
                if not policy_has_visual:
                    print(f"   ℹ️ Policy not visual-enabled (visual features will be zero-padded)")
            except Exception as e:
                print(f"   ⚠️ Visual encoder init failed: {e}")
                self.vision_encoder = None
        else:
            print("[V2] Visual features disabled")

        # Warmup V2 models
        self._warmup_models()

    def _check_gpu_status(self):
        """Check and report GPU/CUDA status."""
        try:
            import torch

            print("\n[V2] GPU Status:")

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"   [OK] CUDA Available: {gpu_name} ({gpu_mem:.1f} GB)")

                if hasattr(torch.backends, 'cudnn') and torch.backends.cudnn.enabled:
                    print(f"   [OK] cuDNN: Enabled")

                torch.backends.cudnn.benchmark = True
                print(f"   [OK] cuDNN Benchmark: Enabled")
            else:
                print(f"   [WARNING] CUDA NOT AVAILABLE - Running on CPU (SLOW!)")

        except ImportError:
            print(f"   [WARNING] PyTorch not installed")
        except Exception as e:
            print(f"   [WARNING] GPU check failed: {e}")

    def _warmup_models(self):
        """Warmup V2 hierarchical policy models on GPU."""
        print("[V2] Warming up policy models...")
        try:
            import torch

            # Create dummy inputs matching actual dimensions
            # state_dim = player(16) + flat_objects(16*20=320) + context(16) = 352
            from .config import FULL_STATE_DIM
            dummy_history = np.zeros((60, FULL_STATE_DIM), dtype=np.float32)  # 60s of history
            dummy_state = np.zeros(FULL_STATE_DIM, dtype=np.float32)
            dummy_objects = np.zeros((16, 20), dtype=np.float32)
            dummy_mask = np.ones(16, dtype=np.float32)  # Use ones to avoid empty mask issues

            # Run a few warmup iterations
            for _ in range(3):
                _ = self.policy.step(
                    state_history=dummy_history,
                    current_state=dummy_state,
                    objects=dummy_objects,
                    object_mask=dummy_mask,
                    force_strategist=True,
                    force_tactician=True
                )

            print("   ✅ Policy warmup complete")
        except Exception as e:
            print(f"   ⚠️ Policy warmup failed: {e}")

    def _heatmap_loop(self):
        """
        Background thread that generates VISION ENCODER attention heatmaps at 30 FPS.
        Shows what MobileNetV3 is "seeing" - textures, colors, patterns.
        """
        import base64

        has_vision = self.vision_encoder is not None
        print(f"[HeatmapThread] Started (30 FPS, {'MobileNetV3' if has_vision else 'detection-based'})")
        heatmap_count = 0
        last_heatmap_sum = 0  # Track changes

        while self.running:
            try:
                frame = self.latest_frame
                if frame is None:
                    time.sleep(0.016)
                    continue

                start_t = time.perf_counter()

                # Use REAL vision encoder if available
                if has_vision:
                    try:
                        # Pass current time to avoid caching (we want fresh heatmaps!)
                        current_ms = int(time.time() * 1000)
                        _, heatmap_u8 = self.vision_encoder.encode_global(frame, current_time_ms=current_ms, return_map=True)
                        heatmap_count += 1

                        # Track if heatmap is actually changing
                        heatmap_sum = int(heatmap_u8.sum())
                        changed = abs(heatmap_sum - last_heatmap_sum) > 100

                        if heatmap_count % 90 == 1:  # Log every 3s at 30fps
                            status = "CHANGED" if changed else "STABLE"
                            print(f"[HeatmapThread] #{heatmap_count} | {status} | sum={heatmap_sum} | range=[{heatmap_u8.min()}-{heatmap_u8.max()}]")

                        last_heatmap_sum = heatmap_sum

                        # Periodic GPU memory cleanup to prevent lag over time
                        if heatmap_count % 300 == 0:  # Every 10s
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    except Exception as e:
                        # Fallback to detection-based
                        if heatmap_count % 300 == 1:
                            print(f"[HeatmapThread] Vision encoder error: {e}, using detection fallback")
                        detections = self._latest_detections_for_heatmap
                        heatmap_u8 = detection_attention_map(detections, output_size=(14, 14)) if detections else np.zeros((14, 14), dtype=np.uint8)
                else:
                    detections = self._latest_detections_for_heatmap
                    heatmap_u8 = detection_attention_map(detections, output_size=(28, 28)) if detections else np.zeros((28, 28), dtype=np.uint8)

                # Base64 encode
                heatmap_b64 = base64.b64encode(heatmap_u8.tobytes()).decode('ascii')
                heatmap_shape = list(heatmap_u8.shape)

                # Update cache
                with self._heatmap_lock:
                    self._heatmap_cache = (heatmap_b64, heatmap_shape)

                # Adaptive sleep - target 30 FPS but don't oversleep
                elapsed = time.perf_counter() - start_t
                sleep_time = max(0.001, 0.033 - elapsed)  # 33ms target, min 1ms
                time.sleep(sleep_time)

            except Exception as e:
                if self.debug_mode:
                    print(f"[HeatmapThread] Error: {e}")
                time.sleep(0.033)

        print("[HeatmapThread] Stopped")

    def start(self):
        """Start the bot."""
        if self.running:
            return

        print("\n" + "="*60)
        print("  V2 BOT CONTROLLER - Hierarchical Temporal Architecture")
        print("="*60)
        print("\nHotkeys:")
        print("   F1 = Pause/Resume")
        print("   F2 = BAD STOP (save negative training data)")
        print("   F3 = EMERGENCY STOP")
        print("   F4 = Toggle debug")
        print("   F5 = Toggle reasoning log")
        print("   F6 = Cycle mode override")
        print("-"*60)

        self.running = True
        self.paused = False
        self.session_start = time.time()
        self.frame_count = 0

        # Reset components
        self.policy.reset()
        self.tracker.reset()
        self.state_encoder.reset()
        self.state_summarizer.reset()
        self._visual_history_buffer.clear()  # Reset visual history
        self._tracked_enemy_ids = set()  # Reset enemy tracking
        self._tracked_loot_ids = set()   # Reset loot tracking
        self._prev_player_hp = 1.0       # Reset HP tracking
        if self.online_learner:
            self.online_learner.reset_episode()
        if self.outcome_tracker:
            self.outcome_tracker.reset()

        # Lock mouse cursor if shadow training (prevents leaving game screen)
        if self.shadow_trainer:
            self.shadow_trainer.input_capture.confine_cursor()

        # Start control thread
        self.bot_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.bot_thread.start()

        # Start heatmap background thread (low-latency debug viewer)
        self._heatmap_thread = threading.Thread(target=self._heatmap_loop, daemon=True)
        self._heatmap_thread.start()

        # Setup hotkeys
        self._setup_hotkeys()

    def stop(self):
        """Stop the bot."""
        print("\n   [STOP] Stopping bot...")
        self.running = False

        # Release mouse cursor if it was locked
        if self.shadow_trainer:
            print("   [STOP] Releasing cursor...")
            self.shadow_trainer.input_capture.release_cursor()

        if self.bot_thread:
            print("   [STOP] Waiting for bot thread to finish...")
            self.bot_thread.join(timeout=2.0)
            if self.bot_thread.is_alive():
                print("   [STOP] WARNING: Bot thread did not finish in time!")
        if self.kb_listener:
            self.kb_listener.stop()
        # Stop mouse movement thread
        if hasattr(self, 'mouse') and hasattr(self.mouse, 'stop'):
            self.mouse.stop()
        # Stop VLM and save corrections
        if self.vlm:
            self.vlm.stop_async()
            self.vlm.stop_and_save()
            summary = self.vlm.get_summary()
            print(f"   [VLM-V2] Analyses: S:{summary['strategist']} T:{summary['tactician']} E:{summary['executor']}")
            print(f"   [VLM-V2] Corrections: {summary['action_corrections']} | Mode overrides: {summary['mode_overrides']}")

            # Run meta-analysis if VLM collected corrections
            if self.config.vlm_corrections and summary.get('action_corrections', 0) > 0:
                self._run_meta_analysis()

        # Save online learner checkpoint with smart checkpointing
        if self.online_learner:
            stats = self.online_learner.get_stats()
            print(f"   [ONLINE] Stats: hits={stats['hits']} misses={stats['misses']} "
                  f"hit_rate={stats['hit_rate']:.0%} updates={stats['updates']}")

            if self.config.policy_dir and stats['updates'] > 0:
                # Use smart checkpoint manager if available
                if self.checkpoint_manager:
                    score = compute_score({
                        'hit_rate': stats['hit_rate'],
                        'position_error': -stats.get('avg_pos_error', 0.1),  # Lower is better
                    })
                    self.checkpoint_manager.save(
                        model=self.policy.executor,
                        optimizer=self.online_learner.base_learner.optimizer,
                        score=score,
                        step=stats['updates'],
                        loss=stats.get('avg_loss', 0),
                        metrics=stats
                    )
                    print(f"   [ONLINE] Smart checkpoint saved (score={score:.4f})")

                # Also save to main executor.pt for backward compatibility
                checkpoint_path = Path(self.config.policy_dir) / "executor.pt"
                self.online_learner.save_checkpoint(str(checkpoint_path))
                print(f"   [ONLINE] Saved to {checkpoint_path}")

            # Close tensorboard logger
            self.online_learner.base_learner.close()

        # Print outcome stats (visual or legacy)
        if self.online_learner and self._use_visual_outcomes:
            # Visual outcome detection stats
            visual_stats = self.online_learner.get_stats()
            if 'visual_tracker' in visual_stats:
                vt = visual_stats['visual_tracker']
                print(f"   [VISUAL] Events detected: {vt.get('event_counts', {})}")
                print(f"   [VISUAL] Total visual reward: {vt.get('total_reward', 0):.1f}")
                print(f"   [VISUAL] Frames processed: {vt.get('frames_processed', 0)}")
            if 'visual' in visual_stats:
                v = visual_stats['visual']
                print(f"   [VISUAL] Events: {v.get('visual_events_detected', 0)} | "
                      f"Reward: {v.get('visual_reward_total', 0):.1f}")
        elif self.outcome_tracker:
            # Legacy heuristic stats
            outcome_stats = self.outcome_tracker.get_stats()
            print(f"   [LEGACY] Stats: kills={outcome_stats['kills']} deaths={outcome_stats['deaths']} "
                  f"loot={outcome_stats['loot_collected']} K/D={outcome_stats['kd_ratio']:.1f} "
                  f"win_rate={outcome_stats['combat_win_rate']:.0%}")

        # Stop auto-labeler and save dataset
        if self.auto_labeler:
            stats = self.auto_labeler.get_stats()
            print(f"   [AUTO-LABEL] Stats: queued={stats['frames_queued']} "
                  f"processed={stats['frames_processed']} labels={stats['labels_generated']}")
            self.auto_labeler.stop()

        # Save shadow trainer checkpoint with smart checkpointing
        if self.shadow_trainer:
            print("   [STOP] Saving shadow trainer...")
            try:
                stats = self.shadow_trainer.get_stats()
                print(f"   [SHADOW] Stats: demos={stats['demos_recorded']} updates={stats['updates']} "
                      f"clicks={stats['human_clicks']} ({stats['human_click_rate']:.1%} click rate)")
                print(f"   [SHADOW] Avg loss={stats['avg_loss']:.4f} | "
                      f"pos_err={stats['avg_pos_error']:.3f} | click_acc={stats['avg_click_accuracy']:.0%}")
            except Exception as e:
                print(f"   [STOP] ERROR getting stats: {e}")
                stats = {'updates': 0}

            if stats['updates'] > 0:
                # Determine save directory - use policy_dir if provided, otherwise default
                save_dir = Path(self.config.policy_dir) if self.config.policy_dir else Path(__file__).parent / "policies"
                save_dir.mkdir(parents=True, exist_ok=True)

                # Use smart checkpoint manager if available
                if self.checkpoint_manager:
                    score = compute_score({
                        'click_accuracy': stats['avg_click_accuracy'],
                        'position_error': -stats['avg_pos_error'],  # Lower is better
                    })
                    self.checkpoint_manager.save(
                        model=self.policy.executor,
                        optimizer=self.shadow_trainer.optimizer,
                        score=score,
                        step=stats['updates'],
                        loss=stats['avg_loss'],
                        metrics=stats
                    )
                    print(f"   [SHADOW] Smart checkpoint saved (score={score:.4f})")

                # Also save to main executor.pt for backward compatibility
                checkpoint_path = save_dir / "executor.pt"
                self.shadow_trainer.save_checkpoint(str(checkpoint_path))
                print(f"   [SHADOW] Saved trained model to {checkpoint_path}")
            else:
                print(f"   [SHADOW] ⚠️ No training updates - nothing to save")

            # Close tensorboard logger
            self.shadow_trainer.close()

        self._print_stats()

        if self.debug_server:
            self.debug_server.stop()
            
        print("\n[V2] Bot stopped.")

    def _setup_hotkeys(self):
        """Setup control hotkeys."""
        try:
            from pynput import keyboard as pynput_keyboard

            # Track which keys are currently pressed to avoid duplicate triggers
            self._hotkey_pressed = set()

            def on_key_press(key):
                """Handle key press - trigger action on press for responsiveness."""
                key_name = getattr(key, 'name', None)
                if key_name and key_name not in self._hotkey_pressed:
                    self._hotkey_pressed.add(key_name)
                    self._handle_hotkey(key_name)

            def on_key_release(key):
                """Handle key release - clear pressed state."""
                key_name = getattr(key, 'name', None)
                if key_name:
                    self._hotkey_pressed.discard(key_name)

            self.kb_listener = pynput_keyboard.Listener(
                on_press=on_key_press,
                on_release=on_key_release
            )
            self.kb_listener.start()
        except ImportError:
            print("[V2] pynput not installed - hotkeys disabled")

    def _handle_hotkey(self, key_name: str):
        """Process hotkey action."""
        if key_name == 'f1':
            # F1 = Pause/Resume
            if not self._shadow_mode:
                self._toggle_pause()
            else:
                print("\n[V2] Pause disabled in shadow training mode - bot stays passive!")
        elif key_name == 'f2':
            self._bad_stop()  # Save recent actions as negative training data
        elif key_name == 'f3':
            self._emergency_stop()
        elif key_name == 'f4':
            self._toggle_debug()
        elif key_name == 'f5':
            self._toggle_reasoning()
        elif key_name == 'f6':
            self._cycle_mode_override()

    def _toggle_pause(self):
        self.paused = not self.paused
        status = "PAUSED" if self.paused else "RUNNING"
        print(f"\n{'='*60}")
        print(f"  BOT {status}")
        print(f"{'='*60}")
        if self.paused:
            print("  VLM corrections will NOT be saved while paused")
            print("  Press F1 to resume")
        else:
            print("  Bot resumed - VLM corrections enabled")

    def _cycle_mode_override(self):
        """Cycle through mode overrides."""
        if self.mode_override is None:
            self.mode_override = "FIGHT"
        elif self.mode_override == "FIGHT":
            self.mode_override = "LOOT"
        elif self.mode_override == "LOOT":
            self.mode_override = "FLEE"
        elif self.mode_override == "FLEE":
            self.mode_override = "EXPLORE"
        elif self.mode_override == "EXPLORE":
            self.mode_override = "CAUTIOUS"
        else:
            self.mode_override = None
        print(f"\n[V2] Mode: {self.mode_override or 'AUTO'}")

    def _emergency_stop(self):
        print("\n[V2] EMERGENCY STOP!")
        self.running = False
        # CRITICAL: Call stop() to save models and recordings!
        # Previously this was missing, causing all training data to be lost on F3
        self.stop()

    def _toggle_debug(self):
        self.debug_mode = not self.debug_mode
        print(f"\n[V2] Debug: {'ON' if self.debug_mode else 'OFF'}")

    def _toggle_reasoning(self):
        self.reasoning_log_enabled = not self.reasoning_log_enabled
        print(f"\n[V2] Reasoning log: {'ON' if self.reasoning_log_enabled else 'OFF'}")

    def _bad_stop(self):
        """
        BAD STOP - User pressed F2 to indicate bot did something wrong.
        Saves recent state/actions as NEGATIVE training examples (like V1).

        The bot learns "don't do this" from these corrections.
        """
        import json
        from pathlib import Path

        # Pause the bot
        self.paused = True

        if not self.bad_stop_buffer:
            print("\n[V2] ⚠️ Bad stop buffer is empty - nothing to save")
            return

        # Get the most recent frames to save
        frames_to_save = list(self.bad_stop_buffer)[-self.BAD_STOP_SAVE_COUNT:]

        if not frames_to_save:
            print("\n[V2] ⚠️ No recent actions to save")
            return

        # Create corrections directory (V2 specific)
        corrections_dir = Path(__file__).parent.parent / "data" / "v2_corrections"
        corrections_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = int(time.time())
        filename = f"v2_bad_stop_{timestamp}.json"
        filepath = corrections_dir / filename

        # Build V2 correction entries
        corrections = []
        for frame in frames_to_save:
            state_dict = frame.get('state_dict', {})
            if not state_dict:
                continue

            # The action/decision the bot took (WRONG)
            policy_output = frame.get('policy_output', {})
            mode = policy_output.get('mode', 'EXPLORE')

            # Convert to JSON-serializable format
            def make_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(v) for v in obj]
                return obj

            corrections.append({
                'timestamp': frame.get('timestamp', time.time()),
                'quality': 'bad',  # User said it was wrong
                'state_dict': make_serializable(state_dict),
                'policy_output': make_serializable(policy_output),
                'tracked_objects': frame.get('tracked_objects_info', []),
                'source': 'bad_stop_v2',
                'mode': mode,
            })

        if not corrections:
            print("\n[V2] ⚠️ No valid frames with state data to save")
            return

        # Save to file
        data = {
            'source': 'bad_stop_v2',
            'timestamp': timestamp,
            'reason': 'User pressed F2 to indicate bot behavior was wrong',
            'corrections': corrections
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n[V2] ❌ BAD STOP! Saved {len(corrections)} frames as negative training data")
        print(f"     File: {filepath.name}")
        print(f"     These will be used to teach the model what NOT to do")
        print(f"     Press F1 to resume or F3 to fully stop")

    def _control_loop(self):
        """
        Main control loop: PERCEIVE → DECIDE → ACT

        Runs at ~60Hz, but different policy components update at different rates.
        """
        # NOTE: ScreenCapture is already initialized in __init__
        # DO NOT re-initialize here - DXCam only allows ONE instance per device
        # Re-creating causes "already created" warning and race conditions between threads

        while self.running:
            try:
                if self.paused:
                    time.sleep(0.1)
                    continue

                loop_start = time.time()
                self.frame_count += 1

                # ═══════════════════════════════════════
                # 1. PERCEIVE
                # ═══════════════════════════════════════
                t0 = time.perf_counter()

                # Capture frame (with DXCam crash recovery)
                try:
                    frame = self.screen.capture()
                except Exception as e:
                    # DXCam can crash on alt-tab, screen changes, etc.
                    # Try to recover by falling back to mss
                    if self.screen.use_dxcam:
                        print(f"[V2] DXCam error: {e} - switching to mss fallback")
                        self.screen.use_dxcam = False
                        try:
                            if self.screen.camera:
                                self.screen.camera.stop()
                        except:
                            pass
                        self.screen.camera = None
                    time.sleep(0.01)
                    continue

                if frame is None:
                    time.sleep(0.01)
                    continue
                
                self.latest_frame = frame

                t1 = time.perf_counter()

                # YOLO detection
                detections = self.detector.detect_frame(frame)

                # Store detections for heatmap thread (convert to normalized coords)
                # This shows EXACTLY what the AI sees in the heatmap overlay
                h, w = frame.shape[:2]
                self._latest_detections_for_heatmap = [
                    {
                        'x_min': getattr(d, 'x1', 0) / w,
                        'y_min': getattr(d, 'y1', 0) / h,
                        'x_max': getattr(d, 'x2', 0) / w,
                        'y_max': getattr(d, 'y2', 0) / h,
                        'confidence': getattr(d, 'confidence', 0.5),
                        'class_name': getattr(d, 'class_name', 'unknown')
                    }
                    for d in detections
                ]

                t2 = time.perf_counter()

                # Update object tracker (ByteTrack)
                tracked_objects = self.tracker.update(detections)

                # Auto-labeler: queue interesting frames for background processing
                if self.auto_labeler:
                    # Convert detections to list format for auto-labeler
                    det_list = []
                    for det in detections:
                        det_list.append({
                            'class_name': getattr(det, 'class_name', 'unknown'),
                            'confidence': getattr(det, 'confidence', 0.5),
                            'bbox': [getattr(det, 'x1', 0), getattr(det, 'y1', 0),
                                    getattr(det, 'x2', 0) - getattr(det, 'x1', 0),
                                    getattr(det, 'y2', 0) - getattr(det, 'y1', 0)]
                        })
                    self.auto_labeler.check_and_queue(frame, det_list)

                t3 = time.perf_counter()

                # Track perception timing
                self._profile_times['capture'].append(t1 - t0)
                self._profile_times['yolo'].append(t2 - t1)
                self._profile_times['tracker'].append(t3 - t2)

                # Get player ship position from YOLO detections (not cursor!)
                # In DarkOrbit the ship is on screen, YOLO detects it as Player/PlayerShip
                player_detected = False
                for det in detections:
                    cls = det.class_name if hasattr(det, 'class_name') else det.get('class_name', '')
                    if cls in ('Player', 'PlayerShip', 'player_ship'):
                        if hasattr(det, 'x_center'):
                            self.player_x = det.x_center
                            self.player_y = det.y_center
                        else:
                            self.player_x = det.get('x_center', 0.5)
                            self.player_y = det.get('y_center', 0.5)
                        player_detected = True
                        break

                if not player_detected:
                    # Fallback: ship is roughly at screen center in DarkOrbit
                    # (map scrolls around the player)
                    self.player_x = 0.5
                    self.player_y = 0.5

                # Update health (prefer HUD OCR over fallback estimation)
                if self._use_hud_ocr and self.hud_reader is not None and frame is not None:
                    try:
                        hud_values = self.hud_reader.read(frame)
                        self.health = hud_values.hp
                        self.shield = hud_values.shield
                    except Exception as e:
                        # Fallback to estimation if OCR fails
                        self.health = self.filters['health'].update(detections)
                else:
                    # Fallback: use V1 estimation (based on enemy proximity)
                    self.health = self.filters['health'].update(detections)

                # Initialize log events (empty if no GUI detector)
                log_events = {'kills': 0, 'damage_taken': 0, 'rewards': 0, 'alerts': 0}

                # GUI Detection and Minimap Position Extraction (YOLO-based)
                if self.gui_detector is not None:
                    # Update GUI regions from YOLO detections (every frame)
                    self.gui_detector.update_from_detections(detections)

                    # Extract player position from minimap (every frame is fast)
                    if self._minimap_tracking_enabled:
                        map_x, map_y = self.gui_detector.get_map_position(frame)
                        # Update state encoder with map position
                        self.state_encoder.update_map_position(map_x, map_y)

                    # Read combat logs (OCR, rate-limited internally to 100ms)
                    log_events = self.gui_detector.get_log_events(frame)

                    # DEBUG: Print raw OCR combat logs when they appear
                    if self.gui_detector.log_reader and self.gui_detector.log_reader.recent_logs:
                        if self._profile_count % 30 == 0:  # Every 30 frames
                            print(f"   [OCR-LOGS] {self.gui_detector.log_reader.recent_logs}")

                # Encode state
                state_dict = self.state_encoder.encode(
                    tracked_objects=tracked_objects,
                    player_x=self.player_x,
                    player_y=self.player_y,
                    hp=self.health,
                    shield=self.shield,
                    is_attacking=self.is_attacking,
                    idle_time=self.idle_time
                )

                # Add to temporal history (for Strategist)
                self.state_summarizer.add_state(state_dict['full_state'], loop_start)

                # Get history for strategist (60s at 1Hz = 60 samples)
                state_history = self.state_summarizer.get_history(pad_to_length=60)

                # ═══════════════════════════════════════
                # 2. DECIDE (Hierarchical Policy)
                # ═══════════════════════════════════════

                # Apply mode override if set
                if self.mode_override:
                    self.policy.set_mode_override(self.mode_override)

                # Extract visual features (if visual encoder enabled)
                visual_history = None
                roi_visual = None
                local_visual = None

                if self.vision_encoder is not None and frame is not None:
                    current_time_ms = int(loop_start * 1000)

                    # Global features for Strategist (extract at 1Hz, cached)
                    global_visual = self.vision_encoder.encode_global(frame, current_time_ms)

                    # Add to visual history buffer (for strategist)
                    self._visual_history_buffer.append(global_visual)

                    # Build visual history array [T, visual_dim]
                    visual_history = np.array(self._visual_history_buffer, dtype=np.float32)
                    # Pad to 60 if needed
                    if len(visual_history) < 60:
                        pad_size = 60 - len(visual_history)
                        visual_history = np.pad(visual_history, ((pad_size, 0), (0, 0)), mode='constant')

                    # RoI features for Tactician (per-object)
                    # Build bboxes from tracked objects
                    bboxes = []
                    for obj in tracked_objects[:16]:
                        # Get normalized bbox (center_x, center_y, width, height)
                        cx = getattr(obj, 'x', 0.5)
                        cy = getattr(obj, 'y', 0.5)
                        w = getattr(obj, 'width', 0.1)
                        h = getattr(obj, 'height', 0.1)
                        bboxes.append((cx, cy, w, h))

                    if bboxes:
                        roi_visual = self.vision_encoder.encode_rois(frame, bboxes)
                    else:
                        roi_visual = None  # Will be zero-padded by policy

                    # Local features for Executor (around target)
                    # Use target position from previous frame or center
                    target_x = getattr(self.policy.state, 'target_x', 0.5)
                    target_y = getattr(self.policy.state, 'target_y', 0.5)
                    local_visual = self.vision_encoder.encode_local(frame, target_x, target_y)

                # Run hierarchical policy
                # Strategist runs at 1Hz, Tactician at 10Hz, Executor at 60Hz
                policy_output = self.policy.step(
                    state_history=state_history,
                    current_state=state_dict['full_state'],
                    objects=state_dict['objects'],
                    object_mask=state_dict['object_mask'],
                    player_x=self.player_x,
                    player_y=self.player_y,
                    visual_history=visual_history,
                    roi_visual=roi_visual,
                    local_visual=local_visual
                )

                # Extract outputs
                action = policy_output['action']
                mode = policy_output['mode']
                target_idx = int(policy_output['target_idx'])  # Ensure int, not numpy
                confidence = float(policy_output.get('confidence', 0.5))

                # DEBUG: Log policy outputs to diagnose bottom-right clicking
                if self._profile_count % 30 == 0:  # Every 30 frames
                    target_pos = "No target"
                    if target_idx >= 0 and target_idx < len(tracked_objects):
                        obj = tracked_objects[target_idx]
                        target_pos = f"({obj.x:.3f}, {obj.y:.3f}) [{obj.class_name}]"

                    mouse_pos = "No action"
                    if action is not None:
                        mouse_pos = f"({action.get('mouse_x', 0.5):.3f}, {action.get('mouse_y', 0.5):.3f})"

                    # Add log info to debug output if available
                    log_info = ""
                    if self.gui_detector is not None and self.gui_detector.log_reader:
                        # Show raw combat log messages
                        if self.gui_detector.log_reader.recent_logs:
                            logs_str = " | ".join(self.gui_detector.log_reader.recent_logs[:3])  # Show up to 3 logs
                            log_info = f" | Logs: {logs_str}"
                        # Or show event counts if no messages but events detected
                        elif 'log_events' in locals() and any(log_events.values()):
                            events_str = ", ".join([f"{k}:{v}" for k, v in log_events.items() if v > 0])
                            log_info = f" | Events: {events_str}"

                    print(f"\n   [DEBUG] Mode: {mode} | Target #{target_idx}: {target_pos} | Mouse: {mouse_pos} | Objects: {len(tracked_objects)}{log_info}")

                # Update idle time
                has_targets = len(tracked_objects) > 0
                if has_targets:
                    self.idle_time = 0.0
                else:
                    self.idle_time += 1.0 / 60

                t4 = time.perf_counter()
                self._profile_times['policy'].append(t4 - t3)

                # ═══════════════════════════════════════
                # 3. ACT (Execute with humanization)
                # ═══════════════════════════════════════

                # In shadow mode: DON'T execute actions, just watch human
                if self._shadow_mode:
                    # Skip action execution - human is playing
                    pass
                elif action is not None and self._can_act():
                    self._execute_action(
                        action=action,
                        mode=mode,
                        tracked_objects=tracked_objects,
                        target_idx=target_idx
                    )

                t5 = time.perf_counter()
                self._profile_times['execute'].append(t5 - t4)

                # ═══════════════════════════════════════
                # 3a. BROADCAST DEBUG STATE (for debug viewer)
                # ═══════════════════════════════════════
                if self.config.broadcast_debug:
                    self._broadcast_debug_state(
                        detections=detections,
                        tracked_objects=tracked_objects,
                        mode=mode,
                        target_idx=target_idx,
                        action=action,
                        policy_output=policy_output
                    )

                # Print profiling every 50 frames
                self._profile_count += 1
                if self._profile_count % 50 == 0:
                    def avg_ms(lst): return sum(lst) / max(len(lst), 1) * 1000
                    print(f"\n   [V2-PROFILE] cap:{avg_ms(self._profile_times['capture']):.1f}ms | "
                          f"yolo:{avg_ms(self._profile_times['yolo']):.1f}ms | "
                          f"track:{avg_ms(self._profile_times['tracker']):.1f}ms | "
                          f"policy:{avg_ms(self._profile_times['policy']):.1f}ms | "
                          f"exec:{avg_ms(self._profile_times['execute']):.1f}ms")

                # Log reasoning
                self._log_reasoning(
                    mode=mode,
                    target_idx=target_idx,
                    tracked_objects=tracked_objects,
                    confidence=confidence,
                    action=action
                )

                # ═══════════════════════════════════════
                # 3b. SHADOW TRAINING (Learn from human play)
                # ═══════════════════════════════════════
                if self.shadow_trainer:
                    # Build target_info for shadow trainer
                    target_info_34 = np.zeros(34, dtype=np.float32)
                    if target_idx >= 0 and target_idx < len(state_dict['objects']):
                        obj = state_dict['objects'][target_idx]
                        target_info_34[0] = obj[0]  # x
                        target_info_34[1] = obj[1]  # y
                        for j in range(min(20, len(obj))):
                            if 12 + j < 32:
                                target_info_34[12 + j] = obj[j]
                    else:
                        target_info_34[0] = 0.5
                        target_info_34[1] = 0.5

                    # Record human action and train (with full hierarchical data if enabled)
                    human_action = self.shadow_trainer.record(
                        state=state_dict['full_state'][:64],
                        goal=policy_output.get('goal', np.zeros(64, dtype=np.float32)),
                        target_info=target_info_34,
                        tracked_objects=tracked_objects,  # For full recording
                        mode=mode,  # Current mode
                        frame=frame,  # Frame for full recording
                        local_visual=local_visual,    # [64] executor precision features
                        roi_visual=roi_visual,         # [16, 128] tactician per-object features
                        global_visual=self._visual_history_buffer[-1] if self._visual_history_buffer else None  # [512] strategist global features
                    )

                    # Compare bot vs human (for display)
                    if action is not None:
                        bot_x, bot_y = action.get('mouse_x', 0.5), action.get('mouse_y', 0.5)
                        human_x, human_y = human_action['mouse_x'], human_action['mouse_y']
                        pos_diff = np.sqrt((bot_x - human_x)**2 + (bot_y - human_y)**2)

                        # Show when bot would click but human doesn't (or vice versa)
                        bot_click = action.get('should_click', False)
                        human_click = human_action['should_click']

                        if human_click:
                            click_str = " 🖱️CLICK" if bot_click else " 🖱️HUMAN-CLICK"
                        else:
                            click_str = " ❌bot-click" if bot_click else ""

                        if self.frame_count % 30 == 0:  # Every ~1 second
                            print(f"\r   [SHADOW] Bot:({bot_x:.2f},{bot_y:.2f}) Human:({human_x:.2f},{human_y:.2f}) diff:{pos_diff:.3f}{click_str}    ", end="")

                    # Periodic training update
                    update_result = self.shadow_trainer.update()
                    if update_result:
                        stats = update_result['stats']
                        print(f"\n   [SHADOW] Update #{update_result['total_updates']}: "
                              f"loss={update_result['loss']:.4f} | "
                              f"pos_err={update_result['pos_error']:.3f} | "
                              f"click_acc={update_result['click_accuracy']:.0%} | "
                              f"demos={stats['demos_recorded']} clicks={stats['human_clicks']}")

                        # PERIODIC CHECKPOINT SAVING (every N updates)
                        total_updates = update_result['total_updates']
                        if total_updates > 0 and total_updates % self.config.checkpoint_save_interval == 0:
                            if self.checkpoint_manager and self.config.policy_dir:
                                from .training.checkpoint_manager import compute_score
                                score = compute_score({
                                    'click_accuracy': stats['avg_click_accuracy'],
                                    'position_error': -stats['avg_pos_error'],
                                })
                                self.checkpoint_manager.save(
                                    model=self.policy.executor,
                                    optimizer=self.shadow_trainer.optimizer,
                                    score=score,
                                    step=total_updates,
                                    loss=stats['avg_loss'],
                                    metrics=stats
                                )
                                print(f"   [SHADOW] 💾 Auto-saved checkpoint #{total_updates} (score={score:.4f})")

                # ═══════════════════════════════════════
                # 3c. DAGGER (Corrective learning during bot play)
                # ═══════════════════════════════════════
                if self.dagger and not self._shadow_mode and action is not None:
                    # Capture human input for correction detection
                    human_action = self.shadow_trainer.input_capture.get_action()
                    if self.dagger.detect_correction(action, human_action):
                        # Build target_info if not already built (shadow mode builds it above)
                        if 'target_info_34' not in dir():
                            target_info_34 = np.zeros(34, dtype=np.float32)
                            if target_idx >= 0 and target_idx < len(state_dict['objects']):
                                obj = state_dict['objects'][target_idx]
                                target_info_34[0] = obj[0]
                                target_info_34[1] = obj[1]
                            else:
                                target_info_34[0] = 0.5
                                target_info_34[1] = 0.5
                        goal = policy_output.get('goal', np.zeros(64, dtype=np.float32))
                        state_compact = state_dict['full_state'][:64]
                        self.dagger.record_correction(
                            state=state_compact,
                            goal=goal,
                            target_info=target_info_34,
                            human_action=human_action,
                            bot_action=action,
                            local_visual=local_visual,
                        )

                    # Periodic DAgger training update (reuses shadow_trainer.update)
                    if self.frame_count % 120 == 0 and len(self.shadow_trainer.buffer) >= 32:
                        update_result = self.shadow_trainer.update()
                        if update_result:
                            dagger_stats = self.dagger.get_stats()
                            print(f"\n   [DAgger] Update: loss={update_result['loss']:.4f} | "
                                  f"corrections={dagger_stats['total_corrections']} "
                                  f"({dagger_stats['correction_rate']:.1%})")

                # ═══════════════════════════════════════
                # 4. VLM-V2 ANALYSIS (Async, non-blocking)
                # ═══════════════════════════════════════
                # DON'T submit new frames when paused to avoid VLM processing old frames
                if self.vlm and frame is not None and not self.paused:
                    # Get recent combat logs with timestamps from OCR (if available)
                    combat_logs = []
                    if self.gui_detector and self.gui_detector.log_reader:
                        # Format logs with relative timestamps (e.g., "2.3s ago: You destroyed X")
                        current_time = time.time()
                        for msg, timestamp in self.gui_detector.log_reader.recent_logs_with_time:
                            age = current_time - timestamp
                            combat_logs.append(f"{age:.1f}s ago: {msg}")

                    # Submit frame for V2 VLM analysis (uses tracked objects with IDs)
                    self.vlm.submit_frame(
                        frame=frame,
                        tracked_objects=tracked_objects,
                        policy_output=policy_output,
                        state_dict=state_dict,
                        combat_logs=combat_logs
                    )

                    # Check for VLM recommendations (non-blocking)
                    # Only print NEW results (not same cached result every frame)
                    vlm_results = self.vlm.get_last_results()

                    if vlm_results.get('strategist'):
                        strat = vlm_results['strategist']
                        strat_time = strat.get('timestamp', 0)
                        if strat_time > self._last_vlm_strategist_time:
                            self._last_vlm_strategist_time = strat_time
                            if not strat.get('current_mode_correct', True):
                                rec_mode = strat.get('recommended_mode', mode)
                                reason = strat.get('reason', '')
                                print(f"\n   [VLM-STRATEGIST] ❌ Mode {mode} incorrect → Suggests: {rec_mode}")
                                if reason:
                                    print(f"      Reason: {reason}")

                    if vlm_results.get('tactician'):
                        tact = vlm_results['tactician']
                        tact_time = tact.get('timestamp', 0)
                        if tact_time > self._last_vlm_tactician_time:
                            self._last_vlm_tactician_time = tact_time
                            if not tact.get('target_correct', True):
                                rec = tact.get('recommended_target', {})
                                reason = tact.get('reason', '')
                                print(f"\n   [VLM-TACTICIAN] ❌ Wrong target → Better: {rec.get('class')}#{rec.get('id')}")
                                if reason:
                                    print(f"      Reason: {reason}")

                    if vlm_results.get('executor'):
                        exe = vlm_results['executor']
                        exe_time = exe.get('timestamp', 0)
                        if exe_time > self._last_vlm_executor_time:
                            self._last_vlm_executor_time = exe_time
                            quality = exe.get('quality', 'unknown')
                            if quality not in ['good', 'unknown']:
                                issue = exe.get('issue', '')
                                print(f"\n   [VLM-EXECUTOR] ⚠️ Movement quality: {quality}")
                                if issue:
                                    print(f"      Issue: {issue}")

                # ═══════════════════════════════════════
                # 5. ONLINE LEARNING (Real-time from hits/misses)
                # ═══════════════════════════════════════
                if self.online_learner and action is not None:
                    # Get target HP for hit detection
                    target_hp = None
                    target_pos = None
                    target_id = None
                    if target_idx >= 0 and target_idx < len(tracked_objects):
                        target_obj = tracked_objects[target_idx]
                        target_pos = (target_obj.x, target_obj.y)
                        target_id = str(getattr(target_obj, 'track_id', -1))

                        # Read target HP from bar above the target (if HUD OCR enabled)
                        if self.target_hp_reader is not None and frame is not None:
                            try:
                                bbox = (target_obj.x, target_obj.y,
                                       getattr(target_obj, 'width', 0.1),
                                       getattr(target_obj, 'height', 0.1))
                                target_hp, _ = self.target_hp_reader.read_target_hp(frame, bbox)
                            except Exception:
                                # Fallback to confidence as proxy
                                target_hp = getattr(target_obj, 'confidence', 0.8)
                        else:
                            # Fallback: estimate from detection confidence
                            target_hp = getattr(target_obj, 'confidence', 0.8)

                    # Build proper 34-dim target_info matching unified.py format
                    target_info_34 = np.zeros(34, dtype=np.float32)
                    if target_idx >= 0 and target_idx < len(state_dict['objects']):
                        obj = state_dict['objects'][target_idx]
                        # Object feature layout from TrackedObject.to_feature_vector (20-dim):
                        # [0-3]:   x, y, distance_to_player, angle_to_player (position)
                        # [4-7]:   vx, vy, speed, heading (velocity)
                        # [8-11]:  width, height, confidence, age_normalized (bbox)
                        # [12-15]: hits_norm, time_since_update_norm, is_tracked, is_lost (tracking)
                        # [16-19]: is_enemy, is_loot, is_player, is_other (class)
                        target_info_34[0] = obj[0]   # x position
                        target_info_34[1] = obj[1]   # y position
                        target_info_34[2] = obj[4] if len(obj) > 4 else 0.0   # velocity x
                        target_info_34[3] = obj[5] if len(obj) > 5 else 0.0   # velocity y
                        target_info_34[4] = obj[6] if len(obj) > 6 else 0.0   # speed
                        target_info_34[5] = obj[8] if len(obj) > 8 else 0.05  # width
                        target_info_34[6] = obj[9] if len(obj) > 9 else 0.05  # height
                        target_info_34[7] = obj[10] if len(obj) > 10 else 1.0 # confidence
                        target_info_34[8] = obj[16] if len(obj) > 16 else 0.0 # is_enemy (FIXED: was obj[14])
                        target_info_34[9] = obj[17] if len(obj) > 17 else 0.0 # is_loot (FIXED: was obj[15])
                        # Copy remaining features as context
                        for j in range(min(20, len(obj))):
                            if 12 + j < 32:
                                target_info_34[12 + j] = obj[j]
                    else:
                        # No valid target - use center
                        target_info_34[0] = 0.5
                        target_info_34[1] = 0.5

                    # Get visual features for outcome detection (if available)
                    global_visual = None
                    local_visual = None
                    roi_visual = None

                    if self.vision_encoder is not None and frame is not None:
                        try:
                            # Global scene features for detecting screen-wide events
                            global_visual = self.vision_encoder.encode_global(
                                frame, int(time.time() * 1000)
                            )

                            # Local features around target for precision detection
                            if target_pos is not None:
                                local_visual = self.vision_encoder.encode_local(
                                    frame, target_pos[0], target_pos[1]
                                )

                            # ROI features for each tracked object
                            if tracked_objects:
                                bboxes = [(obj.x, obj.y, obj.w, obj.h) for obj in tracked_objects[:16]]
                                roi_visual = self.vision_encoder.encode_rois(frame, bboxes)
                        except Exception as e:
                            if self.debug_mode:
                                print(f"[Visual] Feature extraction error: {e}")

                    # Record experience with visual features for AI-based outcome detection
                    self.online_learner.record(
                        state=state_dict['full_state'][:64],
                        goal=policy_output.get('goal', np.zeros(64, dtype=np.float32)),
                        target_info=target_info_34,
                        action=action,
                        target_hp=target_hp,
                        clicked=action.get('should_click', False),
                        target_pos=target_pos,
                        player_hp=self.health,
                        target_idx=target_idx,
                        is_attacking=(mode == 'FIGHT'),
                        global_visual=global_visual,
                        local_visual=local_visual,
                        roi_visual=roi_visual
                    )

                    # ═══════════════════════════════════════
                    # 5b. VISUAL OUTCOME DETECTION (AI-based)
                    # ═══════════════════════════════════════
                    # Check for visually detected events
                    if self._use_visual_outcomes and hasattr(self.online_learner, 'get_detected_events'):
                        detected_events = self.online_learner.get_detected_events()
                        if detected_events:
                            event_names = list(detected_events.keys())
                            if self.debug_mode:
                                print(f"   [Visual] Detected: {event_names}")

                            # Print significant events
                            if 'explosion_large' in detected_events or 'target_killed' in detected_events:
                                print(f"\n   [VISUAL] 🎯 KILL DETECTED! (AI confidence: {detected_events.get('target_killed', {}).get('probability', 0):.2f})")
                            if 'loot_collected' in detected_events:
                                print(f"\n   [VISUAL] 📦 LOOT COLLECTED! (AI confidence: {detected_events['loot_collected'].get('probability', 0):.2f})")
                            if 'death_screen' in detected_events:
                                print(f"\n   [VISUAL] 💀 DEATH DETECTED!")

                    # ═══════════════════════════════════════
                    # 5c. LEGACY OUTCOME TRACKING (Fallback)
                    # ═══════════════════════════════════════
                    # Keep heuristic tracking for comparison/debugging
                    if self.outcome_tracker and not self._use_visual_outcomes:
                        # Build experience for credit assignment
                        experience = {
                            'state': state_dict['full_state'][:64],
                            'goal': policy_output.get('goal', np.zeros(64, dtype=np.float32)),
                            'target_info': target_info_34,
                            'action': action
                        }

                        # Record state for outcome tracker
                        self.outcome_tracker.record_state(
                            experience=experience,
                            target_id=target_id,
                            target_hp=target_hp,
                            player_hp=self.health,
                            player_shield=self.shield
                        )

                        # Track current enemies for kill detection (legacy heuristic)
                        current_enemy_ids = set()
                        for obj in tracked_objects:
                            if obj.class_name in self.ENEMY_CLASSES:
                                current_enemy_ids.add(str(obj.track_id))

                        # Detect kills: enemy was being tracked but disappeared
                        if self._tracked_enemy_ids and self.outcome_tracker.current_target_id:
                            disappeared = self._tracked_enemy_ids - current_enemy_ids
                            if self.outcome_tracker.current_target_id in disappeared:
                                rewarded = self.outcome_tracker.check_kill(
                                    self.outcome_tracker.current_target_id
                                )
                                if rewarded:
                                    for exp in rewarded:
                                        self.online_learner.base_learner.buffer.add(exp)
                                    print(f"\n   [LEGACY] 🎯 KILL! Credited {len(rewarded)} experiences")

                        self._tracked_enemy_ids = current_enemy_ids

                        # Detect loot collection (legacy)
                        current_loot_ids = set()
                        for obj in tracked_objects:
                            if obj.class_name in self.LOOT_CLASSES:
                                current_loot_ids.add(str(obj.track_id))

                        if hasattr(self, '_tracked_loot_ids'):
                            disappeared_loot = self._tracked_loot_ids - current_loot_ids
                            if disappeared_loot and mode == 'LOOT':
                                rewarded = self.outcome_tracker.check_loot()
                                if rewarded:
                                    for exp in rewarded:
                                        self.online_learner.base_learner.buffer.add(exp)
                                    print(f"\n   [LEGACY] 📦 LOOT! Credited {len(rewarded)} experiences")

                        self._tracked_loot_ids = current_loot_ids

                        # Detect death (legacy)
                        if self.health <= 0.01 and self._prev_player_hp > 0.1:
                            penalized = self.outcome_tracker.check_death()
                            if penalized:
                                for exp in penalized:
                                    self.online_learner.base_learner.buffer.add(exp)
                                print(f"\n   [LEGACY] 💀 DEATH! Penalized {len(penalized)} experiences")

                        # Detect HP recovery (legacy)
                        if self.health > self._prev_player_hp + 0.05:
                            rewarded = self.outcome_tracker.check_hp_recovery(
                                self._prev_player_hp, self.health
                            )
                            if rewarded:
                                for exp in rewarded:
                                    self.online_learner.base_learner.buffer.add(exp)

                    self._prev_player_hp = self.health

                    # Periodic weight update
                    update_result = self.online_learner.update()
                    if update_result:
                        stats = update_result['stats']
                        hit_rate = stats['hits'] / max(stats['hits'] + stats['misses'], 1)

                        # Get visual or legacy outcome stats
                        if self._use_visual_outcomes and 'visual_stats' in update_result:
                            v_stats = update_result['visual_stats']
                            events = v_stats.get('events_detected', 0)
                            v_reward = v_stats.get('visual_reward_total', 0)
                            print(f"\n   [ONLINE] Update #{update_result['total_updates']}: "
                                  f"loss={update_result['loss']:.4f} | "
                                  f"hits={stats['hits']} misses={stats['misses']} ({hit_rate:.0%}) | "
                                  f"visual_events:{events} reward:{v_reward:.1f}")
                        else:
                            outcome_stats = self.outcome_tracker.get_stats() if self.outcome_tracker else {}
                            kills = outcome_stats.get('kills', 0)
                            deaths = outcome_stats.get('deaths', 0)
                            loot = outcome_stats.get('loot_collected', 0)
                            print(f"\n   [ONLINE] Update #{update_result['total_updates']}: "
                                  f"loss={update_result['loss']:.4f} | "
                                  f"hits={stats['hits']} misses={stats['misses']} ({hit_rate:.0%}) | "
                                  f"K:{kills} D:{deaths} L:{loot}")

                        # PERIODIC CHECKPOINT SAVING (every N updates)
                        total_updates = update_result['total_updates']
                        if total_updates > 0 and total_updates % self.config.checkpoint_save_interval == 0:
                            if self.checkpoint_manager and self.config.policy_dir:
                                from .training.checkpoint_manager import compute_score
                                score = compute_score({
                                    'hit_rate': hit_rate,
                                    'position_error': -stats.get('avg_pos_error', 0.1),
                                })
                                self.checkpoint_manager.save(
                                    model=self.policy.executor,
                                    optimizer=self.online_learner.base_learner.optimizer,
                                    score=score,
                                    step=total_updates,
                                    loss=stats.get('avg_loss', 0),
                                    metrics=stats
                                )
                                print(f"   [ONLINE] 💾 Auto-saved checkpoint #{total_updates} (score={score:.4f})")

                # ═══════════════════════════════════════
                # 6. RECORD TO BAD STOP BUFFER (for F2)
                # ═══════════════════════════════════════
                # Store recent frames so F2 can save them as negative training data
                tracked_objects_info = []
                for obj in tracked_objects[:10]:
                    tracked_objects_info.append({
                        'track_id': getattr(obj, 'track_id', -1),
                        'class_name': getattr(obj, 'class_name', 'unknown'),
                        'x': float(getattr(obj, 'x', 0.5)),
                        'y': float(getattr(obj, 'y', 0.5)),
                        'confidence': float(getattr(obj, 'confidence', 0.5))
                    })

                self.bad_stop_buffer.append({
                    'timestamp': time.time(),
                    'state_dict': state_dict,
                    'policy_output': policy_output,
                    'tracked_objects_info': tracked_objects_info,
                    'player_x': self.player_x,
                    'player_y': self.player_y,
                    'health': self.health,
                    'is_attacking': self.is_attacking
                })

                # (deque auto-trims to BAD_STOP_BUFFER_SIZE)

                # Print status
                self._print_status(mode, tracked_objects)

                # Maintain loop rate
                elapsed = time.time() - loop_start
                sleep_time = (1.0 / 60) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                print(f"\n[V2] Error: {e}")
                if self.debug_mode:
                    import traceback
                    traceback.print_exc()
                time.sleep(0.1)

    def _can_act(self) -> bool:
        """Check cooldown."""
        now = time.time()
        return now - self.last_action_time >= self.action_cooldown

    def _execute_action(self, action: Dict, mode: str,
                        tracked_objects: List[TrackedObject],
                        target_idx: int = -1):
        """
        Execute action with humanization - NON-BLOCKING.

        Args:
            action: Dict with:
                - mouse_x, mouse_y: target position (0-1)
                - should_click: whether to click
                - keyboard: Dict[str, bool] of all learned key presses
            mode: Current mode (unused, kept for compatibility)
            tracked_objects: Current tracked objects (unused, kept for compatibility)
            target_idx: Index of current target (unused, kept for compatibility)
        """
        # Get action values from model
        # The neural network learns ALL actions from human demonstrations
        target_x = action.get('mouse_x', 0.5)
        target_y = action.get('mouse_y', 0.5)
        should_click = action.get('should_click', False)
        keyboard = action.get('keyboard', {})  # All learned keyboard actions

        # DEBUG: Trace click decision
        if should_click and self.frame_count % 10 == 0:
             print(f"   [CONTROLLER-DEBUG] should_click=True | Target: ({target_x:.3f}, {target_y:.3f})")

        # Add precision noise
        noise = self.config.precision_noise
        target_x += np.random.uniform(-noise, noise)
        target_y += np.random.uniform(-noise, noise)
        target_x = np.clip(target_x, 0, 1)
        target_y = np.clip(target_y, 0, 1)

        # GUI MASKING: Find safe click position (avoid GUI elements)
        if self._gui_masking_enabled and self.gui_detector is not None and should_click:
            if not self.gui_detector.is_click_allowed(target_x, target_y):
                print(f"   [GUI-MASK] BLOCKED click at ({target_x:.3f}, {target_y:.3f})")
                # Target is on GUI element, find nearest safe position
                safe_x, safe_y = self.gui_detector.get_safe_click_position(target_x, target_y)
                if self.debug_mode:
                    logger.debug(f"GUI mask: ({target_x:.2f}, {target_y:.2f}) → ({safe_x:.2f}, {safe_y:.2f})")
                target_x, target_y = safe_x, safe_y

        # Get current mouse position relative to game monitor
        import ctypes
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
        pt = POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        # Convert absolute cursor position to normalized position within game monitor
        current_x = (pt.x - self.screen_left) / self.screen_width
        current_y = (pt.y - self.screen_top) / self.screen_height

        # CLAMP: If mouse is on another screen, clamp to valid range
        # This prevents invalid training data when user moves cursor off-screen
        current_x = np.clip(current_x, 0.0, 1.0)
        current_y = np.clip(current_y, 0.0, 1.0)

        # Calculate distance
        dist = np.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)

        # Move if significant distance
        if dist > 0.01:
            # Movement duration based on mode
            if mode in ['FIGHT', 'FLEE']:
                duration = max(0.06, min(0.25, dist * 0.5))
            else:
                duration = max(0.08, min(0.4, dist * 0.8))

            # Generate humanized path
            path = self.movement_gen.generate_path(
                (current_x, current_y),
                (target_x, target_y),
                target_duration=duration
            )

            # Add minimal overshoot in combat
            overshoot_prob = 0.03 if mode == 'FIGHT' else 0.08
            path = self.movement_gen.add_overshoot(path, probability=overshoot_prob)

            # Execute movement
            self.mouse.execute_path_smooth(path)

        # Click if needed (bot learns when to click)
        if should_click:
            self._human_click(mode)
            self.total_clicks += 1

        # Handle ALL keyboard keys from learned decisions
        # The neural network learns which keys to press based on game state
        self._execute_keyboard(keyboard)

        self.total_actions += 1
        self.last_action_time = time.time()

    def _human_click(self, mode: str):
        """Execute human-like click - NON-BLOCKING."""
        timing = self.movement_gen.generate_click_timing()

        # Use learned profile timing directly - NO artificial clamps
        # Your profile says you hold for ~0.26s, so we should respect that!
        hold = timing['hold_duration']

        # Non-blocking click (mouse.click returns immediately, release in background)
        self.mouse.click(hold_duration=hold)

    def _execute_keyboard(self, keyboard: Dict[str, bool]):
        """Execute learned keyboard actions.

        The neural network outputs which keys to press/release based on:
        - Current game state (enemies, HP, target type)
        - Historical patterns learned from human demonstrations

        No hardcoded logic - all decisions come from the trained model.

        Args:
            keyboard: Dict mapping key names to pressed state (True/False)
        """
        # Store for debug viewer
        self.latest_keys = keyboard
        
        if not keyboard:
            return

        # Handle Ctrl specially for attack toggle (stateful)
        should_ctrl = keyboard.get('ctrl', False)
        now = time.time()

        if should_ctrl and not self.is_attacking:
            if now - self.last_attack_toggle > 0.2:
                self.keyboard.toggle_attack()
                self.is_attacking = True
                self.last_attack_toggle = now
        elif not should_ctrl and self.is_attacking:
            if now - self.last_attack_toggle > 0.3:
                self.keyboard.toggle_attack()
                self.is_attacking = False
                self.last_attack_toggle = now

        # Handle Space for rockets
        if keyboard.get('space', False):
            self.keyboard.fire_rocket()

        # Handle modifier keys (hold/release)
        for key in ['shift', 'alt']:
            if keyboard.get(key, False):
                self.keyboard.hold_key(key)
            else:
                self.keyboard.release_key(key)

        # Handle number keys (1-9, 0) for abilities
        for key in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
            if keyboard.get(key, False):
                self.keyboard.press_key(key)

        # Handle letter keys (commonly used in game)
        for key in ['q', 'w', 'e', 'r', 't', 'a', 's', 'd', 'f', 'g', 'z', 'x', 'c', 'v', 'b', 'j']:
            if keyboard.get(key, False):
                self.keyboard.press_key(key)

        # Handle special keys
        if keyboard.get('tab', False):
            self.keyboard.press_key('tab')
        if keyboard.get('esc', False):
            self.keyboard.press_key('esc')

    def _log_reasoning(self, mode: str, target_idx: int,
                       tracked_objects: List[TrackedObject],
                       confidence: float, action: Dict):
        """Log what the bot is thinking."""
        if not self.reasoning_log_enabled:
            return

        now = time.time()
        if now - self.last_reasoning_time < 0.5:
            return

        # Count object types
        enemies = [o for o in tracked_objects if o.class_name in self.ENEMY_CLASSES]
        loot = [o for o in tracked_objects if o.class_name in self.LOOT_CLASSES]

        # Build reasoning
        parts = []

        # What we see
        if enemies:
            parts.append(f"E:{len(enemies)}")
        if loot:
            parts.append(f"L:{len(loot)}")
        if not enemies and not loot:
            parts.append("CLEAR")

        # Mode and confidence
        parts.append(f"{mode}({confidence:.0%})")

        # Target
        if target_idx >= 0 and target_idx < len(tracked_objects):
            target = tracked_objects[target_idx]
            parts.append(f"→{target.class_name}#{target.track_id}")

        # Health
        if self.health < 0.3:
            parts.append("HP:CRIT")
        elif self.health < 0.6:
            parts.append("HP:LOW")

        # Action summary
        if action is not None:
            if action.get('should_click'):
                parts.append("CLICK")

        reasoning = " | ".join(parts)

        if reasoning != self.last_reasoning:
            print(f"\n   {reasoning}")
            self.last_reasoning = reasoning
            self.last_reasoning_time = now

    def _print_status(self, mode: str, tracked_objects: List[TrackedObject]):
        """Print current status."""
        enemies = len([o for o in tracked_objects if o.class_name in self.ENEMY_CLASSES])
        loot = len([o for o in tracked_objects if o.class_name in self.LOOT_CLASSES])

        state = "PAUSE" if self.paused else mode
        attack = "*" if self.is_attacking else " "

        status = f"\r[V2] {state:8}{attack} | HP:{self.health:3.0%} | "
        status += f"E:{enemies} L:{loot} | "
        status += f"T:{self.tracker.get_stats()['tracked']}/{self.tracker.get_stats()['total_tracks']} | "
        status += f"A:{self.total_actions} C:{self.total_clicks}    "

        print(status, end="")

    def _print_stats(self):
        """Print session stats."""
        if self.session_start:
            duration = time.time() - self.session_start
            fps = self.frame_count / max(duration, 1)
            print(f"\n\n[V2] Session: {duration:.0f}s | Frames: {self.frame_count} ({fps:.1f} FPS)")
            print(f"     Actions: {self.total_actions} | Clicks: {self.total_clicks}")

    def _run_meta_analysis(self):
        """
        Run meta-analysis on VLM corrections to detect systematic errors.

        This analyzes corrections at three levels:
        - Strategist: Mode selection patterns
        - Tactician: Target selection errors
        - Executor: Mouse positioning overfitting and clustering
        """
        try:
            from .vlm.vlm_meta_learner_v2 import MetaLearnerV2

            print("\n" + "="*60)
            print("  Running V2 Meta-Learning Analysis...")
            print("="*60)

            learner = MetaLearnerV2()
            learner.analyze_session(hours=1)  # Analyze last hour only (current session)

        except ImportError as e:
            print(f"[V2-META] Meta-learner not available: {e}")
        except Exception as e:
            print(f"[V2-META] Meta-analysis failed: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()

    def _broadcast_debug_state(self, detections, tracked_objects, mode: str,
                                target_idx: int, action: Dict, policy_output: Dict):
        """
        Broadcast current bot state via IPC for debug viewer.

        Writes state to a JSON file that can be read by a separate debug viewer process.
        Uses atomic write (write to temp, then rename) to prevent corruption.
        """
        try:
            # Build detection list
            # Detection class uses x_min/y_min/x_max/y_max and they're already normalized (0-1)
            det_list = []
            for det in detections:
                det_list.append({
                    'x_min': float(getattr(det, 'x_min', 0)),
                    'y_min': float(getattr(det, 'y_min', 0)),
                    'x_max': float(getattr(det, 'x_max', 0)),
                    'y_max': float(getattr(det, 'y_max', 0)),
                    'class_name': getattr(det, 'class_name', 'unknown'),
                    'confidence': float(getattr(det, 'confidence', 0.5))
                })

            # Build tracked objects list
            obj_list = []
            for obj in tracked_objects:
                obj_list.append({
                    'track_id': int(getattr(obj, 'track_id', -1)),
                    'x': float(getattr(obj, 'x', 0.5)),
                    'y': float(getattr(obj, 'y', 0.5)),
                    'width': float(getattr(obj, 'width', 0.1)),
                    'height': float(getattr(obj, 'height', 0.1)),
                    'class_name': getattr(obj, 'class_name', 'unknown'),
                    'confidence': float(getattr(obj, 'confidence', 0.5)),
                    'velocity_x': float(getattr(obj, 'vx', 0.0)),
                    'velocity_y': float(getattr(obj, 'vy', 0.0))
                })

            # Build action info
            action_info = [0.5, 0.5, 0.0]  # Default: center, no click
            if action is not None:
                action_info = [
                    float(action.get('mouse_x', 0.5)),
                    float(action.get('mouse_y', 0.5)),
                    1.0 if action.get('should_click', False) else 0.0
                ]

            # Get heatmap from background thread cache (INSTANT - never blocks!)
            # The _heatmap_loop thread generates heatmaps at 30 FPS using fast saliency
            with self._heatmap_lock:
                heatmap_b64, heatmap_shape = self._heatmap_cache

            # Build state dict for IPC
            state = {
                'timestamp': time.time(),
                'frame_count': self.frame_count,
                'heatmap_b64': heatmap_b64,  # Base64 encoded for speed
                'heatmap_shape': heatmap_shape,

                # Perception data
                'detections': det_list,
                'tracked_objects': obj_list,

                # Decision data
                'mode': mode,
                'target_idx': int(target_idx),
                'action': action_info,
                'keys': self.latest_keys if hasattr(self, 'latest_keys') else {},
                'confidence': float(policy_output.get('confidence', 0.5)),

                # Bot state
                'player_x': float(self.player_x),
                'player_y': float(self.player_y),
                'health': float(self.health),
                'shield': float(self.shield),
                'is_attacking': bool(self.is_attacking),
                'idle_time': float(self.idle_time),

                # Screen info (for viewer to know dimensions)
                'screen_width': self.screen_width,
                'screen_height': self.screen_height,
                'screen_left': self.screen_left,
                'screen_top': self.screen_top,

                # Tracker stats
                'tracker_stats': self.tracker.get_stats()
            }

            # Send to Socket Server (Non-blocking update)
            # This is extremely fast (just pointer assignment)
            if self.debug_server:
                self.debug_server.update(state)

        except Exception as e:
            if self.debug_mode:
                print(f"[IPC] Broadcast state creation error: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='DarkOrbit V2 Bot Controller')
    parser.add_argument('--model', type=str, default='F:/dev/bot/best.pt',
                       help='Path to YOLO model')
    parser.add_argument('--policy-dir', type=str, default=None,
                       help='Directory with trained hierarchical policy')
    parser.add_argument('--monitor', type=int, default=1,
                       help='Monitor to capture')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for policy (cuda/cpu)')
    parser.add_argument('--delay', type=int, default=40,
                       help='Reaction delay in ms')
    parser.add_argument('--noise', type=float, default=0.03,
                       help='Precision noise (0-0.1)')
    parser.add_argument('--vlm', action='store_true',
                       help='Enable Enhanced VLM analysis')
    parser.add_argument('--vlm-url', type=str, default='http://localhost:1234',
                       help='VLM server URL (LM Studio)')
    parser.add_argument('--vlm-model', type=str, default='local-model',
                       help='VLM model name')
    parser.add_argument('--vlm-corrections', action='store_true',
                       help='Save VLM corrections for training')
    parser.add_argument('--online-learning', action='store_true',
                       help='Enable online learning from hits/misses (updates executor.pt)')
    parser.add_argument('--online-lr', type=float, default=1e-5,
                       help='Online learning rate (default: 1e-5)')
    parser.add_argument('--auto-label', action='store_true',
                       help='Enable auto-labeling for self-improving YOLO (uses Gemini)')
    parser.add_argument('--auto-label-dir', type=str, default='data/auto_labeled',
                       help='Output directory for auto-labeled data')
    parser.add_argument('--gemini-key', type=str, default=None,
                       help='Google Gemini API key (or set GOOGLE_API_KEY env var)')
    parser.add_argument('--gemini-model', type=str, default='gemini-2.0-flash',
                       help='Gemini model for auto-labeling (default: gemini-2.0-flash)')
    parser.add_argument('--no-visual', action='store_true',
                       help='Disable visual features (NOT recommended - disables heatmap + scene understanding)')
    parser.add_argument('--visual-lightweight', action='store_true',
                       help='Use lightweight color encoder instead of CNN (faster, no extra GPU)')
    parser.add_argument('--shadow-train', action='store_true',
                       help='Shadow training mode: watch human play and learn from their actions')
    parser.add_argument('--shadow-lr', type=float, default=1e-4,
                       help='Shadow training learning rate (default: 1e-4, 10x faster than online)')
    parser.add_argument('--save-recordings', action='store_true',
                       help='Save full hierarchical recordings during shadow training for offline training')
    parser.add_argument('--dagger', action='store_true',
                       help='Enable DAgger: human corrections during bot play get 3x training weight')
    parser.add_argument('--no-broadcast-debug', action='store_true',
                       help='Disable debug viewer broadcasting (enabled by default)')

    args = parser.parse_args()

    config = BotConfigV2(
        model_path=args.model,
        policy_dir=args.policy_dir,
        monitor=args.monitor,
        device=args.device,
        reaction_delay_ms=args.delay,
        precision_noise=args.noise,
        vlm_enabled=args.vlm,
        vlm_url=args.vlm_url,
        vlm_model=args.vlm_model,
        vlm_corrections=args.vlm_corrections,
        online_learning=args.online_learning,
        online_learning_rate=args.online_lr,
        auto_label=args.auto_label,
        auto_label_dir=args.auto_label_dir,
        gemini_api_key=args.gemini_key,
        gemini_model=args.gemini_model,
        visual_features=not args.no_visual,
        visual_lightweight=args.visual_lightweight,
        shadow_train=args.shadow_train,
        shadow_train_lr=args.shadow_lr,
        save_recordings=args.save_recordings,
        dagger_enabled=args.dagger,
        broadcast_debug=not args.no_broadcast_debug
    )

    bot = BotControllerV2(config)

    print("\n[V2] Bot Controller Ready")
    if config.broadcast_debug:
        print(f"     📡 Debug broadcasting ENABLED (port 9999)")
        print("     Run debug_viewer.py in another terminal to visualize")
    if args.shadow_train:
        print("     🎮 SHADOW TRAINING MODE - Play the game, bot learns from you!")
        print("     Bot is PASSIVE - just play normally!")
        print("     Press F3 to stop")
    else:
        print("     Press F1 to start/pause bot control")
        print("     Press F3 for emergency stop")
    print("     Press Ctrl+C to exit")

    bot.start()

    try:
        while bot.running:
            time.sleep(1)
    except KeyboardInterrupt:
        bot.stop()


if __name__ == "__main__":
    main()
