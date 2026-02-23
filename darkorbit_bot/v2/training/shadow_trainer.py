"""
Shadow Trainer - Learn from Human Demonstrations

Watches human play and trains the executor to imitate their actions.
This is Behavioral Cloning (BC) - the most direct form of imitation learning.

Usage:
    trainer = ShadowTrainer(executor, device='cuda', learning_rate=1e-4)

    # In main loop while human plays:
    trainer.record(state, goal, target_info, human_action)

    # Periodically (every few seconds):
    trainer.update()  # Train on recorded demonstrations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import time
import logging
import ctypes

from .training_utils import TrainingLogger, WarmupCosineScheduler, gradient_norm
from ..config import KEYBOARD_KEYS, NUM_KEYBOARD_KEYS

logger = logging.getLogger(__name__)


class HumanInputCapture:
    """Captures human mouse position and clicks in real-time."""

    def __init__(self, screen_left: int = 0, screen_top: int = 0,
                 screen_width: int = 1920, screen_height: int = 1080,
                 confine_mouse: bool = True):
        """
        Args:
            screen_left: Left edge of game monitor
            screen_top: Top edge of game monitor
            screen_width: Width of game monitor
            screen_height: Height of game monitor
            confine_mouse: Whether to lock mouse cursor to screen bounds during recording
        """
        self.screen_left = screen_left
        self.screen_top = screen_top
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.confine_mouse = confine_mouse

        # POINT structure for GetCursorPos
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
        self._point = POINT()

        # RECT structure for ClipCursor
        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long),
                       ("top", ctypes.c_long),
                       ("right", ctypes.c_long),
                       ("bottom", ctypes.c_long)]
        self._rect = RECT()
        self._cursor_confined = False

    def get_mouse_position(self) -> Tuple[float, float]:
        """Get normalized mouse position (0-1 range)."""
        ctypes.windll.user32.GetCursorPos(ctypes.byref(self._point))

        # Convert to normalized position within game monitor
        x = (self._point.x - self.screen_left) / self.screen_width
        y = (self._point.y - self.screen_top) / self.screen_height

        # Clamp to valid range
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))

        return x, y

    def get_click_state(self) -> bool:
        """Check if left mouse button is currently HELD down.

        NOTE: Changed from rising-edge detection to HELD detection.
        Rising edge only captured 1 frame per click, causing the model
        to learn "almost never click" from 59 no-click vs 1 click frames.

        Now captures HELD state - same as keyboard keys.
        If user holds mouse for 60 frames, all 60 frames record True.
        """
        # GetAsyncKeyState returns negative if key is currently pressed
        lmb_held = ctypes.windll.user32.GetAsyncKeyState(0x01) < 0
        return lmb_held

    # Virtual key codes for game-relevant keys (matches KEYBOARD_KEYS from config)
    # Only keys in this dict will be captured - hotkeys (F1-F12) are not included
    VK_GAME_KEYS = {
        # Modifiers
        'ctrl': [0xA2, 0xA3],      # Left/Right Ctrl - attack toggle
        'shift': [0xA0, 0xA1],     # Left/Right Shift
        'alt': [0xA4, 0xA5],       # Left/Right Alt
        'space': [0x20],           # Spacebar - rockets

        # Number keys (top row) - abilities, ammo
        '1': [0x31], '2': [0x32], '3': [0x33], '4': [0x34], '5': [0x35],
        '6': [0x36], '7': [0x37], '8': [0x38], '9': [0x39], '0': [0x30],

        # Letter keys commonly used in games
        'q': [0x51], 'w': [0x57], 'e': [0x45], 'r': [0x52], 't': [0x54],
        'a': [0x41], 's': [0x53], 'd': [0x44], 'f': [0x46], 'g': [0x47],
        'z': [0x5A], 'x': [0x58], 'c': [0x43], 'v': [0x56], 'b': [0x42],
        'j': [0x4A],  # Jump gate
        'tab': [0x09],
        'esc': [0x1B],
    }
    # NOTE: F1-F12 and other hotkeys are automatically excluded because
    # they're not in VK_GAME_KEYS. See SCRIPT_HOTKEYS in config.py for
    # which keys are used as hotkeys per script type.

    def get_keyboard_state(self) -> Dict[str, bool]:
        """Get current keyboard state for game-relevant keys.

        Only captures keys in VK_GAME_KEYS (game actions).
        Hotkeys (F1-F12 etc.) are not captured because they're not in VK_GAME_KEYS.

        Returns:
            Dict mapping key names to pressed state (True/False)
        """
        result = {}
        for key_name, vk_list in self.VK_GAME_KEYS.items():
            pressed = any(ctypes.windll.user32.GetAsyncKeyState(vk) < 0 for vk in vk_list)
            result[key_name] = pressed

        return result

    def confine_cursor(self):
        """Confine mouse cursor to screen bounds (prevents leaving game window)."""
        if not self.confine_mouse:
            print("[MouseLock] Mouse confinement disabled")
            return

        if self._cursor_confined:
            print("[MouseLock] Cursor already confined")
            return

        # Set up RECT for the game monitor bounds
        self._rect.left = self.screen_left
        self._rect.top = self.screen_top
        self._rect.right = self.screen_left + self.screen_width
        self._rect.bottom = self.screen_top + self.screen_height

        print(f"[MouseLock] Attempting to confine cursor to RECT:")
        print(f"            left={self._rect.left}, top={self._rect.top}")
        print(f"            right={self._rect.right}, bottom={self._rect.bottom}")
        print(f"            (size: {self.screen_width}x{self.screen_height})")

        # Call ClipCursor to confine the cursor
        result = ctypes.windll.user32.ClipCursor(ctypes.byref(self._rect))
        if result:
            self._cursor_confined = True
            print(f"[MouseLock] ✓ Cursor confined successfully!")
            print(f"[MouseLock] You should NOT be able to move mouse outside the game screen")
        else:
            error_code = ctypes.get_last_error()
            print(f"[MouseLock] ✗ Failed to confine cursor (error code: {error_code})")
            logger.warning(f"[MouseLock] Failed to confine cursor (error: {error_code})")

    def release_cursor(self):
        """Release mouse cursor confinement."""
        if not self._cursor_confined:
            return

        # Passing NULL to ClipCursor releases the cursor
        result = ctypes.windll.user32.ClipCursor(None)
        if result:
            self._cursor_confined = False
            logger.info("[MouseLock] Cursor released")
        else:
            logger.warning("[MouseLock] Failed to release cursor")

    def get_action(self) -> Dict:
        """Get current human action including full keyboard state."""
        x, y = self.get_mouse_position()
        clicked = self.get_click_state()
        keyboard = self.get_keyboard_state()

        # Determine which ability key (1-9, 0) is pressed
        ability = 0
        for i in range(10):
            key = str(i) if i > 0 else '0'
            if keyboard.get(key, False):
                ability = i if i > 0 else 10  # 0 key = ability 10
                break

        action = {
            'mouse_x': x,
            'mouse_y': y,
            'should_click': clicked,
            'ability': ability,
            # Full keyboard state for training
            'keyboard': keyboard,
        }

        # Add commonly used keys at top level for easy access
        action['ctrl'] = keyboard.get('ctrl', False)
        action['space'] = keyboard.get('space', False)
        action['shift'] = keyboard.get('shift', False)

        return action


class DemonstrationBuffer:
    """
    Buffer for storing human demonstrations.

    Stores TWO types of data:
    1. Executor demos: (state, goal, target_info) → action
    2. Full demos: Complete hierarchical data for offline training
    """

    def __init__(self, max_size: int = 5000, save_full_demos: bool = False):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.save_full_demos = save_full_demos

        # Full demo buffer for offline training (strategist/tactician)
        # Capped to prevent memory explosion (each demo stores a full frame ~922KB)
        self.full_buffer = deque(maxlen=500) if save_full_demos else None

    def add(self, demo: Dict):
        """Add a demonstration."""
        self.buffer.append(demo)

        # If full demo data is present, save to full buffer
        if self.save_full_demos and self.full_buffer is not None:
            if 'full_demo_data' in demo:
                self.full_buffer.append(demo['full_demo_data'])

    def sample(self, batch_size: int) -> List[Dict]:
        """Sample random batch."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def sample_balanced(self, batch_size: int, click_ratio: float = 0.5) -> List[Dict]:
        """
        Sample batch with balanced click/non-click examples.

        Args:
            batch_size: Total batch size
            click_ratio: Target ratio of click examples (default 0.5 = 50% clicks)

        This helps the model learn clicking by ensuring each batch has
        enough click examples, even when clicks are rare in the data.
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        # Separate click and non-click indices
        click_indices = []
        non_click_indices = []

        for i, demo in enumerate(self.buffer):
            # Check if this demo has a click
            action = demo.get('action')
            if action is not None:
                # action[2] is click logit: >0 means clicked
                if isinstance(action, np.ndarray):
                    is_click = action[2] > 0
                elif isinstance(action, dict):
                    is_click = action.get('should_click', False)
                else:
                    is_click = False

                if is_click:
                    click_indices.append(i)
                else:
                    non_click_indices.append(i)
            else:
                non_click_indices.append(i)

        # Calculate how many of each to sample
        n_clicks_wanted = int(batch_size * click_ratio)
        n_non_clicks_wanted = batch_size - n_clicks_wanted

        # Clamp to available
        n_clicks = min(n_clicks_wanted, len(click_indices))
        n_non_clicks = min(n_non_clicks_wanted, len(non_click_indices))

        # If not enough clicks, fill with non-clicks (and vice versa)
        if n_clicks < n_clicks_wanted:
            n_non_clicks = min(batch_size - n_clicks, len(non_click_indices))
        if n_non_clicks < n_non_clicks_wanted:
            n_clicks = min(batch_size - n_non_clicks, len(click_indices))

        # Sample indices
        sampled_indices = []
        if n_clicks > 0 and click_indices:
            sampled_indices.extend(np.random.choice(click_indices, n_clicks, replace=len(click_indices) < n_clicks))
        if n_non_clicks > 0 and non_click_indices:
            sampled_indices.extend(np.random.choice(non_click_indices, n_non_clicks, replace=len(non_click_indices) < n_non_clicks))

        # Shuffle and return
        np.random.shuffle(sampled_indices)
        return [self.buffer[i] for i in sampled_indices[:batch_size]]

    def get_recent(self, n: int) -> List[Dict]:
        """Get most recent n demonstrations."""
        return list(self.buffer)[-n:]

    def __len__(self):
        return len(self.buffer)

    def get_full_demos(self) -> List[Dict]:
        """Get all full demonstrations for offline training."""
        if self.full_buffer is None:
            return []
        return list(self.full_buffer)

    def clear(self):
        self.buffer.clear()
        if self.full_buffer is not None:
            self.full_buffer.clear()


class ShadowTrainer:
    """
    Behavioral Cloning trainer that learns from human demonstrations.

    The human plays the game while the bot watches and learns:
    - Records (state, goal, target_info) → human_action pairs
    - Trains executor to minimize prediction error vs human actions
    - Uses higher learning rate than online learning (1e-4 vs 1e-5)
    """

    def __init__(self,
                 executor: nn.Module,
                 device: str = "cuda",
                 learning_rate: float = 1e-4,
                 buffer_size: int = 5000,
                 batch_size: int = 64,
                 update_interval: float = 3.0,
                 min_samples: int = 32,
                 screen_left: int = 0,
                 screen_top: int = 0,
                 screen_width: int = 1920,
                 screen_height: int = 1080,
                 log_dir: str = "runs",
                 use_lr_scheduler: bool = True,
                 warmup_steps: int = 50,
                 total_steps: int = 5000,
                 save_full_demos: bool = False,
                 recording_dir: str = "darkorbit_bot/data/recordings_v2"):
        """
        Args:
            executor: The executor model to train
            device: cuda or cpu
            learning_rate: BC learning rate (higher than online, 1e-4 typical)
            buffer_size: Max demonstrations to store
            batch_size: Batch size for training
            update_interval: Seconds between weight updates
            min_samples: Minimum samples before training starts
            screen_*: Game monitor dimensions for input capture
            log_dir: Directory for tensorboard logs
            use_lr_scheduler: Whether to use warmup + cosine LR schedule
            warmup_steps: Number of warmup steps for LR
            total_steps: Total expected training steps for LR schedule
            save_full_demos: Whether to save full hierarchical demos for offline training
            recording_dir: Directory to save full recordings
        """
        self.executor = executor
        self.device = device
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.min_samples = min_samples
        self.learning_rate = learning_rate

        # Detect model type
        from ..models.executor import ExecutorV2
        self.is_v2 = isinstance(executor, ExecutorV2)

        # Detect visual model
        self.visual_model = False
        self.target_info_dim = 34
        if self.is_v2:
            self.visual_model = executor.visual_dim > 0
            if self.visual_model:
                logger.info("[ShadowTrainer] ExecutorV2 with visual features")
        elif hasattr(executor, 'input_proj') and hasattr(executor.input_proj[0], 'in_features'):
            input_dim = executor.input_proj[0].in_features
            if input_dim > 200:
                self.visual_model = True
                self.target_info_dim = 98
                logger.info("[ShadowTrainer] Detected VISUAL executor")

        # V2 loss function
        self.loss_fn_v2 = None
        if self.is_v2:
            from .train_executor import ExecutorLossV2
            from ..config import ExecutorConfig
            cfg = ExecutorConfig()
            self.loss_fn_v2 = ExecutorLossV2(
                mouse_weight=1.0, click_weight=0.5, keyboard_weight=0.5,
                click_pos_weight=5.0,
                focal_alpha=cfg.focal_alpha, focal_gamma=cfg.focal_gamma,
                use_beta=cfg.use_beta_distribution
            )

        # Human input capture
        self.input_capture = HumanInputCapture(
            screen_left, screen_top, screen_width, screen_height
        )

        # Demonstration buffer
        self.save_full_demos = save_full_demos
        self.recording_dir = recording_dir
        self.buffer = DemonstrationBuffer(buffer_size, save_full_demos=save_full_demos)

        # Optimizer with higher LR for BC
        self.optimizer = torch.optim.AdamW(
            executor.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Learning rate scheduler (warmup + cosine decay)
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler = None
        if use_lr_scheduler:
            self.lr_scheduler = WarmupCosineScheduler(
                self.optimizer,
                base_lr=learning_rate,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr=learning_rate * 0.01
            )

        # Tensorboard logging with timestamp for unique runs
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_logger = TrainingLogger(
            log_dir=log_dir,
            experiment_name=f"shadow_{timestamp}",
            use_tensorboard=True,
            log_to_file=True
        )

        # Training tracking
        self.last_update_time = time.time()
        self.total_updates = 0
        self.total_demos = 0

        # Click tracking for statistics
        self.human_clicks = 0
        self.total_frames = 0

        # Statistics
        self.stats = {
            'demos_recorded': 0,
            'updates': 0,
            'avg_loss': 0.0,
            'avg_pos_error': 0.0,
            'avg_click_accuracy': 0.0,
            'human_clicks': 0,
            'human_click_rate': 0.0,
            'learning_rate': learning_rate,
        }

        logger.info(f"[ShadowTrainer] Initialized with LR={learning_rate}, buffer={buffer_size}")
        print(f"\n   [SHADOW] Shadow training mode ACTIVE")
        print(f"   [SHADOW] Bot is WATCHING - play the game normally!")
        print(f"   [SHADOW] Learning rate: {learning_rate} (with warmup + cosine decay)")
        print(f"   [SHADOW] Tensorboard: tensorboard --logdir={log_dir}")

    def record(self,
               state: np.ndarray,
               goal: np.ndarray,
               target_info: np.ndarray,
               tracked_objects: Optional[List] = None,
               mode: Optional[str] = None,
               frame: Optional[np.ndarray] = None,
               local_visual: Optional[np.ndarray] = None,
               roi_visual: Optional[np.ndarray] = None,
               global_visual: Optional[np.ndarray] = None) -> Dict:
        """
        Record current state and capture human action.

        Args:
            state: Current state (64-dim)
            goal: Goal embedding (64-dim)
            target_info: Target info (34 or 98-dim)
            tracked_objects: Optional list of TrackedObject for full recording
            mode: Optional current mode for full recording
            frame: Optional frame for full recording
            local_visual: Optional [64] local precision features for executor
            roi_visual: Optional [16, 128] per-object features for tactician
            global_visual: Optional [512] global features for strategist

        Returns:
            Human action dict (for display/comparison)
        """
        self.total_frames += 1

        # Capture human action
        human_action = self.input_capture.get_action()

        # Track clicks
        if human_action['should_click']:
            self.human_clicks += 1
            self.stats['human_clicks'] = self.human_clicks

        # Calculate click rate
        if self.total_frames > 0:
            self.stats['human_click_rate'] = self.human_clicks / self.total_frames

        # Validate and fix input shapes
        if len(state) < 64:
            state = np.pad(state, (0, 64 - len(state)), mode='constant')
        elif len(state) > 64:
            state = state[:64]

        if len(goal) < 64:
            goal = np.pad(goal, (0, 64 - len(goal)), mode='constant')
        elif len(goal) > 64:
            goal = goal[:64]

        # Handle target_info dimension
        if len(target_info) < 34:
            target_info = np.pad(target_info, (0, 34 - len(target_info)), mode='constant')
        if self.visual_model and len(target_info) < self.target_info_dim:
            target_info = np.pad(target_info, (0, self.target_info_dim - len(target_info)), mode='constant')
        elif len(target_info) > self.target_info_dim:
            target_info = target_info[:self.target_info_dim]

        # Build action dict for training (flexible format)
        # Stores raw values - training scripts decide how to encode
        action_dict = {
            'mouse_x': float(human_action['mouse_x']),
            'mouse_y': float(human_action['mouse_y']),
            'click': bool(human_action['should_click']),
            'ability': int(human_action.get('ability', 0)),
        }

        # Add full keyboard state
        keyboard = human_action.get('keyboard', {})
        action_dict['keyboard'] = {k: bool(v) for k, v in keyboard.items()}

        # Full action_array with keyboard state: [mouse_x, mouse_y, click, ...keyboard_keys...]
        # Total: 3 + NUM_KEYBOARD_KEYS = 31 dimensions
        action_array = np.zeros(3 + NUM_KEYBOARD_KEYS, dtype=np.float32)
        action_array[0] = human_action['mouse_x']
        action_array[1] = human_action['mouse_y']
        action_array[2] = 3.0 if human_action['should_click'] else -3.0

        # Add keyboard state (each key as logit: +3.0 if pressed, -3.0 if not)
        for i, key_name in enumerate(KEYBOARD_KEYS):
            key_pressed = keyboard.get(key_name, False)
            action_array[3 + i] = 3.0 if key_pressed else -3.0

        # Build base demonstration for executor training
        demo = {
            'state': state.astype(np.float32),
            'goal': goal.astype(np.float32),
            'target_info': target_info.astype(np.float32),
            'action': action_array,  # Legacy format
            'action_dict': action_dict,  # New flexible format with keyboard
            'timestamp': time.time()
        }

        # Store visual features if provided (for training with visual context)
        if local_visual is not None:
            demo['local_visual'] = local_visual.astype(np.float32)
        if roi_visual is not None:
            demo['roi_visual'] = roi_visual.astype(np.float32)
        if global_visual is not None:
            demo['global_visual'] = global_visual.astype(np.float32)

        # If full recording enabled, infer tactician/strategist labels
        if self.save_full_demos and tracked_objects is not None:
            full_demo = self._build_full_demo(
                human_action=human_action,
                tracked_objects=tracked_objects,
                mode=mode,
                frame=frame,
                state=state,
                goal=goal,
                target_info=target_info,
                local_visual=local_visual,
                roi_visual=roi_visual,
                global_visual=global_visual
            )
            demo['full_demo_data'] = full_demo

        # Store demonstration
        self.buffer.add(demo)

        self.total_demos += 1
        self.stats['demos_recorded'] = self.total_demos

        # Show progress every 100 demos (first few, then every 100)
        if self.total_demos in [1, 10, 32, 50] or self.total_demos % 100 == 0:
            full_count = len(self.buffer.full_buffer) if self.buffer.full_buffer else 0
            print(f"\n   [SHADOW] 📊 Demos: {self.total_demos} | Full recordings: {full_count} | Clicks: {self.human_clicks}")

        return human_action

    def _build_full_demo(self, human_action: Dict, tracked_objects: List,
                         mode: Optional[str], frame: Optional[np.ndarray],
                         state: np.ndarray, goal: np.ndarray, target_info: np.ndarray,
                         local_visual: Optional[np.ndarray] = None,
                         roi_visual: Optional[np.ndarray] = None,
                         global_visual: Optional[np.ndarray] = None) -> Dict:
        """
        Build full demonstration data for offline training (strategist/tactician).

        Infers human's target selection and mode preference from their actions.
        """
        mouse_x = human_action['mouse_x']
        mouse_y = human_action['mouse_y']
        clicked = human_action['should_click']

        # Infer which object human is targeting (closest to mouse)
        target_id = -1
        min_dist = float('inf')

        for i, obj in enumerate(tracked_objects):
            dist = np.sqrt((obj.x - mouse_x)**2 + (obj.y - mouse_y)**2)
            if dist < min_dist:
                min_dist = dist
                target_id = i

        # Only consider it a target if mouse is within 0.15 of object center
        if min_dist > 0.15:
            target_id = -1

        # Infer mode from behavior (heuristic)
        inferred_mode = mode if mode else "FIGHT"  # Use bot's mode as baseline

        # Build full demo compatible with offline trainers
        from ..config import ENEMY_CLASSES, LOOT_CLASSES

        enemies = [obj for obj in tracked_objects if obj.class_name in ENEMY_CLASSES]
        loot = [obj for obj in tracked_objects if obj.class_name in LOOT_CLASSES]

        full_demo = {
            'timestamp': time.time(),
            'frame': frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8),
            'state': state,
            'goal': goal,
            'target_info': target_info,
            'action': human_action,

            # Tracked objects data
            'tracked_objects': [{
                'id': obj.track_id,
                'class_name': obj.class_name,
                'x': obj.x,
                'y': obj.y,
                'width': obj.width,
                'height': obj.height,
                'confidence': obj.confidence,
                'vx': obj.vx,
                'vy': obj.vy,
            } for obj in tracked_objects],

            # Human labels (inferred)
            'human_target_id': target_id,  # Which object human is targeting
            'human_mode': inferred_mode,    # Which mode human is in
            'human_clicked': clicked,       # Whether human clicked

            # Contextual info
            'num_enemies': len(enemies),
            'num_loot': len(loot),
            'mouse_target_distance': min_dist,
        }

        # Store visual features for offline training of all components
        if local_visual is not None:
            full_demo['local_visual'] = local_visual.astype(np.float32)
        if roi_visual is not None:
            full_demo['roi_visual'] = roi_visual.astype(np.float32)
        if global_visual is not None:
            full_demo['global_visual'] = global_visual.astype(np.float32)

        return full_demo

    def should_update(self) -> bool:
        """Check if it's time for a training update."""
        if len(self.buffer) < self.min_samples:
            return False
        if time.time() - self.last_update_time < self.update_interval:
            return False
        return True

    def update(self) -> Optional[Dict]:
        """
        Perform behavioral cloning update.

        Minimizes MSE between executor output and human action.

        Returns:
            Dict with training stats, or None if no update
        """
        if not self.should_update():
            return None

        self.executor.train()

        # Sample batch with BALANCED click/non-click examples
        # This is critical for learning clicks - with 7% click rate and random sampling,
        # most batches have 0-2 clicks which isn't enough for the model to learn
        batch = self.buffer.sample_balanced(min(self.batch_size, len(self.buffer)), click_ratio=0.3)

        if len(batch) < 8:
            return None

        # Convert to tensors
        states = torch.tensor(
            np.stack([b['state'] for b in batch]),
            dtype=torch.float32
        ).to(self.device)

        goals = torch.tensor(
            np.stack([b['goal'] for b in batch]),
            dtype=torch.float32
        ).to(self.device)

        target_infos = torch.tensor(
            np.stack([b['target_info'] for b in batch]),
            dtype=torch.float32
        ).to(self.device)

        human_actions = torch.tensor(
            np.stack([b['action'] for b in batch]),
            dtype=torch.float32
        ).to(self.device)

        # Check if batch has visual features (local_visual for executor)
        has_visual = all('local_visual' in b for b in batch)
        local_visuals = None
        if has_visual:
            local_visuals = torch.tensor(
                np.stack([b['local_visual'] for b in batch]),
                dtype=torch.float32
            ).to(self.device)

        # Forward pass
        self.optimizer.zero_grad()

        with torch.enable_grad():
            if self.is_v2 and self.loss_fn_v2 is not None:
                # === ExecutorV2 path: separate heads + Beta/Focal losses ===
                vis = local_visuals if (has_visual and local_visuals is not None and self.executor.visual_dim > 0) else None
                pred_dict = self.executor.forward(states, goals, target_infos, visual_features=vis)
                loss, loss_metrics = self.loss_fn_v2(pred_dict, human_actions)
                pos_loss_val = loss_metrics['mouse_loss']
                click_loss_val = loss_metrics['click_loss']
                keyboard_loss_val = loss_metrics['keyboard_loss']
            else:
                # === Legacy path ===
                if has_visual and local_visuals is not None and hasattr(self.executor, 'forward_with_visual'):
                    predicted_action, _ = self.executor.forward_with_visual(
                        states, goals, target_infos, local_visuals)
                elif has_visual and local_visuals is not None and self.visual_model:
                    combined = torch.cat([target_infos, local_visuals], dim=-1)
                    predicted_action, _ = self.executor.forward(states, goals, combined)
                else:
                    predicted_action, _ = self.executor.forward(states, goals, target_infos)

                pred_mouse = torch.sigmoid(predicted_action[:, :2])
                target_mouse = human_actions[:, :2].clamp(0.0, 1.0)
                pos_loss = F.mse_loss(pred_mouse, target_mouse)

                human_click_labels = (human_actions[:, 2] > 0).float()
                click_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0, device=self.device))
                click_loss = click_criterion(predicted_action[:, 2], human_click_labels)

                keyboard_loss = torch.tensor(0.0, device=self.device)
                if predicted_action.shape[1] > 3 and human_actions.shape[1] > 3:
                    num_keys = min(predicted_action.shape[1] - 3, human_actions.shape[1] - 3, NUM_KEYBOARD_KEYS)
                    if num_keys > 0:
                        human_keyboard_labels = (human_actions[:, 3:3+num_keys] > 0).float()
                        keyboard_loss = F.binary_cross_entropy_with_logits(
                            predicted_action[:, 3:3+num_keys], human_keyboard_labels)

                loss = pos_loss + click_loss * 0.5 + keyboard_loss * 0.5
                pos_loss_val = pos_loss.item()
                click_loss_val = click_loss.item()
                keyboard_loss_val = keyboard_loss.item()

            # Backprop
            loss.backward()
            grad_norm = gradient_norm(self.executor)
            self.training_logger.log_scalar("Gradients/norm", grad_norm, self.total_updates)
            torch.nn.utils.clip_grad_norm_(self.executor.parameters(), 1.0)
            self.optimizer.step()

            if self.lr_scheduler:
                self.lr_scheduler.step()
                current_lr = self.lr_scheduler.get_lr()
                self.stats['learning_rate'] = current_lr

        self.executor.eval()

        # Calculate metrics
        with torch.no_grad():
            if self.is_v2:
                # V2: get legacy flat output for metrics
                flat_action = self.executor._dict_to_flat(pred_dict)
                pred_xy = flat_action[:, :2]  # Already in [0,1] from Beta mean
                # Clamp for safety
                pred_xy = torch.sigmoid(pred_xy) if not self.executor.use_beta_distribution else pred_xy.clamp(0, 1)
            else:
                pred_xy = torch.sigmoid(predicted_action[:, :2])

            pos_error = (pred_xy - human_actions[:, :2].clamp(0, 1)).abs().mean().item()

            if self.is_v2:
                click_logit = pred_dict['click']
                if click_logit.dim() == 3:
                    click_logit = click_logit[:, 0]
                click_logit = click_logit.squeeze(-1)
                pred_click = (click_logit > 0).float()
            else:
                pred_click = (predicted_action[:, 2] > 0).float()

            human_click_labels = (human_actions[:, 2] > 0).float()
            click_acc = (pred_click == human_click_labels).float().mean().item()

            keyboard_acc = 0.0
            if self.is_v2:
                kb_logits = pred_dict['keyboard']
                if kb_logits.dim() == 3:
                    kb_logits = kb_logits[:, 0]
                pred_kb = (kb_logits > 0).float()
                human_kb = (human_actions[:, 3:3+NUM_KEYBOARD_KEYS] > 0).float()
                keyboard_acc = (pred_kb == human_kb).float().mean().item()
            elif predicted_action.shape[1] > 3 and human_actions.shape[1] > 3:
                num_keys = min(predicted_action.shape[1] - 3, human_actions.shape[1] - 3, NUM_KEYBOARD_KEYS)
                if num_keys > 0:
                    pred_keyboard = (predicted_action[:, 3:3+num_keys] > 0).float()
                    human_keyboard = (human_actions[:, 3:3+num_keys] > 0).float()
                    keyboard_acc = (pred_keyboard == human_keyboard).float().mean().item()

        # Update stats
        self.last_update_time = time.time()
        self.total_updates += 1
        self.stats['updates'] = self.total_updates
        self.stats['avg_loss'] = 0.9 * self.stats['avg_loss'] + 0.1 * loss.item()
        self.stats['avg_pos_error'] = 0.9 * self.stats['avg_pos_error'] + 0.1 * pos_error
        self.stats['avg_click_accuracy'] = 0.9 * self.stats['avg_click_accuracy'] + 0.1 * click_acc
        self.stats['avg_keyboard_accuracy'] = 0.9 * self.stats.get('avg_keyboard_accuracy', 0.0) + 0.1 * keyboard_acc

        # Log to tensorboard
        self.training_logger.log_scalar("Loss/total", loss.item(), self.total_updates)
        self.training_logger.log_scalar("Loss/position", pos_loss_val, self.total_updates)
        self.training_logger.log_scalar("Loss/click", click_loss_val, self.total_updates)
        self.training_logger.log_scalar("Loss/keyboard", keyboard_loss_val, self.total_updates)
        self.training_logger.log_scalar("Metrics/position_error", pos_error, self.total_updates)
        self.training_logger.log_scalar("Metrics/click_accuracy", click_acc, self.total_updates)
        self.training_logger.log_scalar("Metrics/keyboard_accuracy", keyboard_acc, self.total_updates)
        self.training_logger.log_scalar("Metrics/buffer_size", len(self.buffer), self.total_updates)
        self.training_logger.log_scalar("LR/learning_rate", self.stats['learning_rate'], self.total_updates)

        result = {
            'loss': loss.item(),
            'pos_loss': pos_loss_val,
            'click_loss': click_loss_val,
            'keyboard_loss': keyboard_loss_val,
            'pos_error': pos_error,
            'click_accuracy': click_acc,
            'keyboard_accuracy': keyboard_acc,
            'batch_size': len(batch),
            'buffer_size': len(self.buffer),
            'total_updates': self.total_updates,
            'learning_rate': self.stats['learning_rate'],
            'stats': self.stats.copy()
        }

        logger.info(f"[ShadowTrainer] Update #{self.total_updates}: loss={loss.item():.4f}, "
                   f"pos_err={pos_error:.3f}, click_acc={click_acc:.0%}, kb_acc={keyboard_acc:.0%}")

        return result

    def get_comparison(self, state: np.ndarray, goal: np.ndarray,
                       target_info: np.ndarray) -> Dict:
        """
        Get bot's prediction for comparison with human action.

        Returns:
            Dict with 'bot_action' and 'human_action'
        """
        # Get human action
        human_action = self.input_capture.get_action()

        # Get bot prediction
        self.executor.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            goal_t = torch.tensor(goal, dtype=torch.float32).unsqueeze(0).to(self.device)
            target_t = torch.tensor(target_info, dtype=torch.float32).unsqueeze(0).to(self.device)

            action, _ = self.executor.forward(state_t, goal_t, target_t)
            action = action.cpu().numpy()[0]

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

        bot_action = {
            'mouse_x': float(sigmoid(action[0])),
            'mouse_y': float(sigmoid(action[1])),
            'should_click': bool(action[2] > 0),
            'raw_click': float(action[2])
        }

        # Calculate error
        pos_error = np.sqrt(
            (bot_action['mouse_x'] - human_action['mouse_x'])**2 +
            (bot_action['mouse_y'] - human_action['mouse_y'])**2
        )

        return {
            'bot': bot_action,
            'human': human_action,
            'pos_error': pos_error,
            'click_match': bot_action['should_click'] == human_action['should_click']
        }

    def get_stats(self) -> Dict:
        """Get training statistics."""
        return {
            **self.stats,
            'buffer_size': len(self.buffer),
            'total_demos': self.total_demos,
            'total_frames': self.total_frames
        }

    def save_checkpoint(self, path: str):
        """Save model and training state."""
        checkpoint = {
            'model_state_dict': self.executor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stats': self.stats,
            'total_updates': self.total_updates,
            'total_demos': self.total_demos
        }
        # Save LR scheduler state if using one
        if self.lr_scheduler:
            checkpoint['lr_scheduler_state'] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"[ShadowTrainer] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model and training state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.executor.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.stats = checkpoint.get('stats', self.stats)
        self.total_updates = checkpoint.get('total_updates', 0)
        self.total_demos = checkpoint.get('total_demos', 0)

        # Restore LR scheduler state if available
        if self.lr_scheduler and 'lr_scheduler_state' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])

        logger.info(f"[ShadowTrainer] Checkpoint loaded from {path}")

    def reset(self):
        """Reset for new session (keeps learned weights)."""
        self.human_clicks = 0
        self.total_frames = 0
        # Don't clear buffer - keep demonstrations across sessions

    def save_full_recordings(self, filename: Optional[str] = None):
        """
        Save full demonstrations to disk for offline training.

        Args:
            filename: Optional filename, auto-generated if not provided
        """
        if not self.save_full_demos or self.buffer.full_buffer is None:
            logger.warning("[ShadowTrainer] Full demos not enabled, cannot save")
            return

        import os
        import json
        import pickle
        import signal
        from datetime import datetime
        from io import BytesIO
        from PIL import Image

        # Create recording directory
        os.makedirs(self.recording_dir, exist_ok=True)

        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shadow_recording_{timestamp}.pkl"

        filepath = os.path.join(self.recording_dir, filename)

        # Get all full demos
        full_demos = self.buffer.get_full_demos()

        if len(full_demos) == 0:
            print(f"\n   [SHADOW] ⚠️ No recordings to save - session too short!")
            print(f"   [SHADOW] Play longer (at least 30 seconds) before stopping")
            logger.warning("[ShadowTrainer] No full demos to save")
            return

        # Block interrupts during save to prevent corruption
        print(f"\n   [SHADOW] Saving {len(full_demos)} demonstrations...")
        print(f"   [SHADOW] Preparing data... (skipping frames to save time)")

        old_handler = None
        try:
            # Temporarily block SIGINT (Ctrl+C) on Unix systems
            if hasattr(signal, 'SIGINT'):
                old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (ValueError, OSError):
            # On Windows or if signal handling fails, just proceed
            pass

        try:
            # SKIP frame compression - it hangs on Windows and training doesn't need raw frames
            # V2 training only needs state vectors and actions, not raw pixels
            compressed_demos = []
            for i, demo in enumerate(full_demos):
                if i % 1000 == 0:
                    print(f"   [SHADOW] Progress: {i}/{len(full_demos)} demos prepared...")

                # Remove frame entirely - training doesn't need it
                compressed_demo = {
                    k: v for k, v in demo.items() if k != 'frame'
                }
                compressed_demos.append(compressed_demo)

            # Prepare data
            data = {
                'demos': compressed_demos,
                'metadata': {
                    'num_demos': len(compressed_demos),
                    'total_updates': self.total_updates,
                    'stats': self.stats,
                    'timestamp': datetime.now().isoformat(),
                    'frame_format': 'none'  # Frames not saved (training doesn't need them)
                }
            }

            print(f"   [SHADOW] Writing to disk... (don't interrupt!)")

            # Convert numpy arrays to lists for JSON serialization
            def serialize_action(action):
                """Convert action dict or array to JSON-serializable format."""
                if isinstance(action, dict):
                    return {
                        'mouse_x': float(action.get('mouse_x', 0.5)),
                        'mouse_y': float(action.get('mouse_y', 0.5)),
                        'should_click': bool(action.get('should_click', False)),
                        'keyboard': action.get('keyboard', {})
                    }
                elif isinstance(action, np.ndarray):
                    return action.tolist()
                return action

            json_data = {
                'demos': [
                    {
                        'state': demo['state'].tolist() if isinstance(demo['state'], np.ndarray) else demo['state'],
                        'goal': demo['goal'].tolist() if isinstance(demo.get('goal'), np.ndarray) else demo.get('goal', []),
                        'target_info': demo['target_info'].tolist() if isinstance(demo.get('target_info'), np.ndarray) else demo.get('target_info', []),
                        'action': serialize_action(demo.get('action')),
                        'human_mode': demo.get('human_mode'),
                        'human_clicked': demo.get('human_clicked', False),
                        'tracked_objects': demo.get('tracked_objects', [])
                    }
                    for demo in compressed_demos
                ],
                'metadata': data['metadata']
            }

            # Save as JSON (more compatible, human-readable, same format as V2 Recorder)
            json_filepath = filepath.replace('.pkl', '.json')
            with open(json_filepath, 'w') as f:
                json.dump(json_data, f)

            # Get file size
            file_size_mb = os.path.getsize(json_filepath) / (1024 * 1024)

            logger.info(f"[ShadowTrainer] Saved {len(full_demos)} full demos to {json_filepath} ({file_size_mb:.1f} MB)")
            print(f"   [SHADOW] ✓ Saved to {json_filepath} ({file_size_mb:.1f} MB)")
            print(f"   [SHADOW] Use for offline training: python -m darkorbit_bot.v2.training.train_full --data {self.recording_dir}")

        except Exception as e:
            logger.error(f"[ShadowTrainer] Failed to save recordings: {e}")
            print(f"   [SHADOW] ✗ Failed to save: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Restore original signal handler
            if old_handler is not None and hasattr(signal, 'SIGINT'):
                try:
                    signal.signal(signal.SIGINT, old_handler)
                except (ValueError, OSError):
                    pass

    def close(self):
        """Clean up resources (call when done training)."""
        # Save full recordings if enabled
        if self.save_full_demos:
            self.save_full_recordings()

        if self.training_logger:
            self.training_logger.close()
