"""
V2 Executor Training - Behavior Cloning with Object State

Trains the Executor (motor control) using behavior cloning from human recordings.
The Executor learns to map (state, goal, target_info) → action (mouse + keyboard).

IMPORTANT: The model learns PREDICTIVE AIMING, not just copying coordinates!

Training data format (V2 with objects):
- state: Current compact state (64-dim)
- goal: Goal embedding (64-dim) - from strategist or synthetic
- target_info: Object features (34-dim):
    [0-1]: Object x, y position (where the target IS)
    [2-3]: Object velocity x, y (which way it's moving)
    [4]: Object speed
    [5-6]: Object size (width, height)
    [7]: Object confidence
    [8-9]: Is enemy, Is loot
    [10-11]: Relative aim offset (mouse - object)
    [12-31]: Additional object features
- action: Human action [mouse_x, mouse_y, click, key1, key2, ..., keyN]
    - mouse_x, mouse_y: continuous (0-1)
    - click: binary logit
    - key1..keyN: binary logits for each key in KEYBOARD_KEYS (from config)
    - Total dim = 3 + NUM_KEYBOARD_KEYS (currently 31)

The model learns: "given object at position X moving at velocity V, aim at position Y"
This enables lead shots, tracking, and smooth pursuit.
The model also learns ALL keyboard actions from human demonstrations - no hardcoded logic.

Usage:
    python train_executor.py --data recordings_v2/ --epochs 100 --batch-size 64
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..models.executor import create_executor, save_executor, Executor, ExecutorV2
from ..config import ExecutorConfig, TrainingConfig, KEYBOARD_KEYS, NUM_KEYBOARD_KEYS, KEYBOARD_KEY_TO_IDX


class ExecutorDataset(Dataset):
    """
    Dataset for Executor training.

    Loads human recordings and extracts (state, goal, target_info, action) tuples.
    """

    def __init__(self, data_dir: str, sequence_length: int = 1):
        """
        Args:
            data_dir: Directory containing recording files
            sequence_length: Number of frames per sequence (1 for single-step)
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.samples: List[Dict] = []

        self._load_recordings()

    def _load_recordings(self):
        """Load all recordings from data directory (recursively)."""
        recording_files = list(self.data_dir.glob("**/*.npz"))
        json_files = list(self.data_dir.glob("**/*.json"))
        pkl_files = list(self.data_dir.glob("**/*.pkl"))

        # Filter to only sequence files (not metadata.json etc)
        json_files = [f for f in json_files if 'sequence_' in f.name or 'recording' in f.name.lower()]
        pkl_files = [f for f in pkl_files if 'shadow_recording' in f.name.lower() or 'recording' in f.name.lower()]

        recording_files.extend(json_files)
        recording_files.extend(pkl_files)

        print(f"[Executor] Found {len(recording_files)} recording files")

        for file_path in recording_files:
            try:
                if file_path.suffix == '.npz':
                    self._load_npz(file_path)
                elif file_path.suffix == '.json':
                    self._load_json(file_path)
                elif file_path.suffix == '.pkl':
                    self._load_pkl(file_path)
            except Exception as e:
                print(f"   Warning: Could not load {file_path}: {e}")

        print(f"[Executor] Loaded {len(self.samples)} training samples")

    def _load_npz(self, path: Path):
        """Load from numpy archive."""
        data = np.load(path, allow_pickle=True)

        # Extract arrays
        states = data.get('states', data.get('state_sequences', None))
        actions = data.get('actions', None)
        goals = data.get('goals', None)
        target_infos = data.get('target_infos', None)

        if states is None or actions is None:
            return

        n_frames = len(states)

        # Generate synthetic goals and target_infos if not provided
        if goals is None:
            # Use random goal embeddings for now (will learn from actual goals later)
            goals = np.random.randn(n_frames, 64).astype(np.float32) * 0.1

        if target_infos is None:
            # Synthetic target embedding (32-dim learned part)
            target_infos = np.random.randn(n_frames, 32).astype(np.float32) * 0.1

        for i in range(n_frames):
            # Get state - handle different formats
            if len(states[i].shape) > 1:
                state = states[i][-1] if states[i].shape[0] > 1 else states[i][0]
            else:
                state = states[i]

            # Compact to 64 dims
            state_compact = state[:64] if len(state) >= 64 else np.pad(state, (0, 64 - len(state)))

            # Parse action
            action = self._parse_action(actions[i])

            # Extract target position from action (human aimed at this position)
            target_x = action[0]  # mouse_x is where human aimed
            target_y = action[1]  # mouse_y is where human aimed

            # Build target_info: [target_x, target_y, ...32-dim embedding...]
            target_embed = target_infos[i][:32] if len(target_infos[i]) >= 32 else np.pad(target_infos[i], (0, 32 - len(target_infos[i])))
            target_info_full = np.concatenate([
                np.array([target_x, target_y], dtype=np.float32),
                target_embed.astype(np.float32)
            ])

            self.samples.append({
                'state': state_compact.astype(np.float32),
                'goal': goals[i][:64] if len(goals[i]) >= 64 else np.pad(goals[i], (0, 64 - len(goals[i]))),
                'target_info': target_info_full,
                'action': action
            })

    def _load_json(self, path: Path):
        """Load from JSON recording."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Check for V2 format with object data
        is_v2 = data.get('format_version') == 'v2' and 'objects' in data

        # Handle V2 sequence format with rich object data
        if is_v2:
            self._load_v2_json(data)
            return

        # Handle SHADOW TRAINING format: {demos: [...], metadata: {...}}
        if 'demos' in data and 'metadata' in data:
            self._load_shadow_json(data)
            return

        # Handle older V2 sequence format: {states: [[...], ...], actions: [[...], ...]}
        if 'states' in data and 'actions' in data:
            states = data['states']
            actions = data['actions']
            n_frames = min(len(states), len(actions))

            for i in range(n_frames):
                state_vec = states[i]
                if len(state_vec) < 64:
                    state_vec = state_vec + [0.0] * (64 - len(state_vec))
                state = np.array(state_vec[:64], dtype=np.float32)

                # Synthetic goal (will be learned)
                goal = np.random.randn(64).astype(np.float32) * 0.1
                # Synthetic target embedding (32-dim learned part)
                target_embed = np.random.randn(32).astype(np.float32) * 0.1

                # Parse action: [mouse_x, mouse_y, click, ctrl, space]
                act = actions[i]
                mouse_x = act[0] if len(act) > 0 else 0.5
                mouse_y = act[1] if len(act) > 1 else 0.5
                clicked = act[2] if len(act) > 2 else 0

                action = np.array([
                    mouse_x,
                    mouse_y,
                    1.0 if clicked else -1.0,  # Logit space
                    0.0  # Ability (not in recordings)
                ], dtype=np.float32)

                # Build target_info: [target_x, target_y, ...32-dim embedding...]
                # WARNING: This is the old "cheating" format - mouse position = target position
                # V2 format with objects is preferred
                target_info = np.concatenate([
                    np.array([mouse_x, mouse_y], dtype=np.float32),
                    target_embed
                ])

                self.samples.append({
                    'state': state,
                    'goal': goal,
                    'target_info': target_info,
                    'action': action
                })
            return

        # Fallback: Handle old frames format
        frames = data.get('frames', data.get('recordings', []))

        for frame in frames:
            # Extract state
            state_vec = frame.get('state_vector', frame.get('state', []))
            if len(state_vec) < 64:
                state_vec = state_vec + [0.0] * (64 - len(state_vec))
            state = np.array(state_vec[:64], dtype=np.float32)

            # Synthetic goal and target embedding
            goal = np.random.randn(64).astype(np.float32) * 0.1
            target_embed = np.random.randn(32).astype(np.float32) * 0.1

            # Parse action from frame
            mouse_x = frame.get('mouse_x', frame.get('aim_x', 0.5))
            mouse_y = frame.get('mouse_y', frame.get('aim_y', 0.5))
            clicked = frame.get('clicked', frame.get('should_click', frame.get('should_fire', False)))

            action = np.array([
                mouse_x,
                mouse_y,
                1.0 if clicked else -1.0,  # Logit space
                0.0  # Ability (not in recordings)
            ], dtype=np.float32)

            # Build target_info: [target_x, target_y, ...32-dim embedding...]
            target_info = np.concatenate([
                np.array([mouse_x, mouse_y], dtype=np.float32),
                target_embed
            ])

            self.samples.append({
                'state': state,
                'goal': goal,
                'target_info': target_info,
                'action': action
            })

    def _load_pkl(self, path: Path):
        """Load from shadow recording pickle file."""
        import pickle

        with open(path, 'rb') as f:
            data = pickle.load(f)

        demos = data.get('demos', [])
        print(f"   Loading {len(demos)} demos from {path.name}")

        for demo in demos:
            # Shadow recordings have: state, goal, target_info, action/action_dict
            state = demo.get('state')
            goal = demo.get('goal')
            target_info = demo.get('target_info')

            # Prefer action_dict (new format with keyboard) over action (legacy array)
            action_data = demo.get('action_dict', demo.get('action'))

            if state is None or goal is None or target_info is None or action_data is None:
                continue

            # Parse action from dict or array format (handles both)
            action = self._parse_action(action_data)

            # Ensure correct dimensions
            state = state[:64] if len(state) >= 64 else np.pad(state, (0, 64 - len(state)))
            goal = goal[:64] if len(goal) >= 64 else np.pad(goal, (0, 64 - len(goal)))

            self.samples.append({
                'state': state.astype(np.float32),
                'goal': goal.astype(np.float32),
                'target_info': target_info.astype(np.float32),
                'action': action
            })

    def _parse_action(self, action) -> np.ndarray:
        """Parse action to generic format with all keyboard keys.

        Output format: [mouse_x, mouse_y, click, key1, key2, ..., keyN]
        - mouse_x, mouse_y: continuous (0-1)
        - click: binary logit (1.0 = clicked, -1.0 = not clicked)
        - key1..keyN: binary logits for each key in KEYBOARD_KEYS

        Total dim = 3 + NUM_KEYBOARD_KEYS
        """
        # Total action dimension
        action_dim = 3 + NUM_KEYBOARD_KEYS

        if isinstance(action, dict):
            mouse_x = action.get('mouse_x', action.get('aim_x', action.get('move_x', 0.5)))
            mouse_y = action.get('mouse_y', action.get('aim_y', action.get('move_y', 0.5)))
            clicked = action.get('click', action.get('should_click', action.get('should_fire', action.get('clicked', False))))

            # Parse keyboard state from action_dict
            # Keyboard state can be in nested 'keyboard' dict or at top level
            keyboard = action.get('keyboard', {})

            # Build action array
            result = np.full(action_dim, -1.0, dtype=np.float32)
            result[0] = mouse_x
            result[1] = mouse_y
            result[2] = 1.0 if clicked else -1.0

            # Fill keyboard keys (generic, uses KEYBOARD_KEYS order)
            for i, key_name in enumerate(KEYBOARD_KEYS):
                # Check nested keyboard dict first, then top level
                pressed = keyboard.get(key_name, action.get(key_name, False))
                result[3 + i] = 1.0 if pressed else -1.0

            return result

        elif isinstance(action, (list, np.ndarray)):
            action = np.array(action, dtype=np.float32)
            # Pad to full action_dim if needed
            if len(action) < action_dim:
                # Pad with -1.0 (not pressed) for missing keyboard keys
                result = np.full(action_dim, -1.0, dtype=np.float32)
                result[:len(action)] = action
                return result
            return action[:action_dim]
        else:
            # Default: mouse at center, nothing pressed
            result = np.full(action_dim, -1.0, dtype=np.float32)
            result[0] = 0.5  # mouse_x
            result[1] = 0.5  # mouse_y
            return result

    def _load_shadow_json(self, data: Dict):
        """
        Load SHADOW TRAINING format: {demos: [...], metadata: {...}}

        Each demo contains:
        - state: 64-dim state vector
        - goal: 64-dim goal vector
        - target_info: 34-dim target info (object position, velocity, etc.)
        - action: {mouse_x, mouse_y, should_click, keyboard: {...}}
        - tracked_objects: list of detected objects
        - human_mode: inferred mode (FIGHT/LOOT/etc)
        """
        demos = data.get('demos', [])
        metadata = data.get('metadata', {})

        print(f"   [Shadow] Loading {len(demos)} demos from shadow recording")
        print(f"   [Shadow] Session stats: {metadata.get('stats', {}).get('human_clicks', 0)} clicks")

        from ..config import KEYBOARD_KEYS, NUM_KEYBOARD_KEYS

        loaded = 0
        skipped = 0

        for demo in demos:
            # Extract state
            state_data = demo.get('state', [])
            if not state_data or len(state_data) < 64:
                skipped += 1
                continue
            state = np.array(state_data[:64], dtype=np.float32)

            # Extract goal
            goal_data = demo.get('goal', [])
            if goal_data and len(goal_data) >= 64:
                goal = np.array(goal_data[:64], dtype=np.float32)
            else:
                goal = np.zeros(64, dtype=np.float32)

            # Extract target_info
            target_data = demo.get('target_info', [])
            if target_data and len(target_data) >= 34:
                target_info = np.array(target_data[:34], dtype=np.float32)
            else:
                # Build target_info from tracked_objects if available
                target_info = np.zeros(34, dtype=np.float32)
                target_info[0] = 0.5  # default center
                target_info[1] = 0.5

                tracked = demo.get('tracked_objects', [])
                if tracked:
                    # Use first object as target
                    obj = tracked[0]
                    target_info[0] = obj.get('x', 0.5)
                    target_info[1] = obj.get('y', 0.5)
                    target_info[2] = obj.get('vx', 0.0)
                    target_info[3] = obj.get('vy', 0.0)
                    target_info[5] = obj.get('width', 0.05)
                    target_info[6] = obj.get('height', 0.05)
                    target_info[7] = obj.get('confidence', 0.5)

            # Extract action
            action_data = demo.get('action')
            if action_data is None:
                skipped += 1
                continue

            # Build action array: [mouse_x, mouse_y, click_logit, key1_logit, ..., keyN_logit]
            action_dim = 3 + NUM_KEYBOARD_KEYS
            action = np.full(action_dim, -3.0, dtype=np.float32)  # Default: nothing pressed

            if isinstance(action_data, dict):
                action[0] = float(action_data.get('mouse_x', 0.5))
                action[1] = float(action_data.get('mouse_y', 0.5))
                action[2] = 3.0 if action_data.get('should_click', False) else -3.0

                # Parse keyboard
                keyboard = action_data.get('keyboard', {})
                for i, key_name in enumerate(KEYBOARD_KEYS):
                    if keyboard.get(key_name, False):
                        action[3 + i] = 3.0
            elif isinstance(action_data, list):
                # Action is already an array
                for i, val in enumerate(action_data[:action_dim]):
                    action[i] = float(val)

            sample = {
                'state': state,
                'goal': goal,
                'target_info': target_info,
                'action': action
            }

            # Store visual features if available (from updated shadow trainer)
            if 'local_visual' in demo and demo['local_visual'] is not None:
                sample['local_visual'] = np.array(demo['local_visual'], dtype=np.float32)

            self.samples.append(sample)
            loaded += 1

        n_visual = sum(1 for s in self.samples[-loaded:] if 'local_visual' in s) if loaded > 0 else 0
        print(f"   [Shadow] Loaded {loaded} samples, skipped {skipped}")
        if n_visual > 0:
            print(f"   [Shadow] {n_visual}/{loaded} samples have visual features")

    def _load_v2_json(self, data: Dict):
        """
        Load V2 format recording with rich object data.
        
        This is the PROPER training format where:
        - target_info contains OBJECT features (position, velocity, etc.)
        - action contains the MOUSE position (where human actually aimed)
        
        The model learns: given object state → where should mouse be?
        This enables predictive aiming, tracking, and lead shots.
        """
        states = data['states']
        actions = data['actions']
        objects_list = data['objects']  # [n_frames, max_objects, 20]
        masks_list = data.get('object_masks', None)  # [n_frames, max_objects]
        
        n_frames = min(len(states), len(actions), len(objects_list))
        samples_added = 0
        
        for i in range(n_frames):
            # Parse state
            state_vec = states[i]
            if len(state_vec) < 64:
                state_vec = state_vec + [0.0] * (64 - len(state_vec))
            state = np.array(state_vec[:64], dtype=np.float32)
            
            # Parse action using generic parser (handles both dict and array formats)
            action = self._parse_action(actions[i])
            mouse_x = action[0]
            mouse_y = action[1]
            
            # Parse objects for this frame
            objects = np.array(objects_list[i], dtype=np.float32)  # [max_objects, 20]
            mask = np.array(masks_list[i], dtype=np.float32) if masks_list else np.ones(len(objects))
            
            # Find the target object (closest to mouse position)
            target_obj, target_idx = self._find_target_object(objects, mask, mouse_x, mouse_y)
            
            if target_obj is not None:
                # Build target_info from OBJECT features (not mouse position!)
                # Object feature layout from TrackedObject.to_feature_vector (20-dim):
                # [0-3]:   x, y, distance_to_player, angle_to_player (position)
                # [4-7]:   vx, vy, speed, heading (velocity)
                # [8-11]:  width, height, confidence, age_normalized (bbox)
                # [12-15]: hits_norm, time_since_update_norm, is_tracked, is_lost (tracking)
                # [16-19]: is_enemy, is_loot, is_player, is_other (class)

                obj_x = target_obj[0]
                obj_y = target_obj[1]
                obj_vx = target_obj[4] if len(target_obj) > 4 else 0.0
                obj_vy = target_obj[5] if len(target_obj) > 5 else 0.0
                obj_speed = target_obj[6] if len(target_obj) > 6 else 0.0
                obj_width = target_obj[8] if len(target_obj) > 8 else 0.05
                obj_height = target_obj[9] if len(target_obj) > 9 else 0.05
                obj_confidence = target_obj[10] if len(target_obj) > 10 else 1.0
                obj_is_enemy = target_obj[16] if len(target_obj) > 16 else 0.0  # FIXED: was [14]
                obj_is_loot = target_obj[17] if len(target_obj) > 17 else 0.0   # FIXED: was [15]
                
                # Build 34-dim target_info:
                # [0-1]: Object x, y (where the target IS)
                # [2-3]: Object velocity x, y (which way it's moving)
                # [4]: Object speed
                # [5-6]: Object size (width, height)
                # [7]: Object confidence
                # [8]: Is enemy
                # [9]: Is loot
                # [10-11]: Distance from player (mouse_x - obj_x, mouse_y - obj_y)
                # [12-31]: Padding/additional features (20 dims)
                # [32-33]: Reserved
                
                target_info = np.zeros(34, dtype=np.float32)
                target_info[0] = obj_x
                target_info[1] = obj_y
                target_info[2] = obj_vx
                target_info[3] = obj_vy
                target_info[4] = obj_speed
                target_info[5] = obj_width
                target_info[6] = obj_height
                target_info[7] = obj_confidence
                target_info[8] = obj_is_enemy
                target_info[9] = obj_is_loot
                target_info[10] = mouse_x - obj_x  # Relative aim offset X
                target_info[11] = mouse_y - obj_y  # Relative aim offset Y
                # Copy remaining object features as context
                for j in range(min(20, len(target_obj))):
                    if 12 + j < 32:
                        target_info[12 + j] = target_obj[j]
                
                # Synthetic goal (will be learned from strategist later)
                goal = np.random.randn(64).astype(np.float32) * 0.1
                
                self.samples.append({
                    'state': state,
                    'goal': goal,
                    'target_info': target_info,
                    'action': action
                })
                samples_added += 1
            else:
                # No valid object found - still train on this frame
                # Use mouse position as a fallback (center of attention)
                target_info = np.zeros(34, dtype=np.float32)
                target_info[0] = mouse_x  # Fallback to mouse position
                target_info[1] = mouse_y
                # Other features stay at 0
                
                goal = np.random.randn(64).astype(np.float32) * 0.1
                
                self.samples.append({
                    'state': state,
                    'goal': goal,
                    'target_info': target_info,
                    'action': action
                })
                samples_added += 1
    
    def _find_target_object(self, objects: np.ndarray, mask: np.ndarray, 
                            mouse_x: float, mouse_y: float, 
                            max_distance: float = 0.15) -> Tuple[Optional[np.ndarray], int]:
        """
        Find the object closest to the mouse position.
        
        This identifies what the human was targeting based on cursor proximity.
        
        Args:
            objects: [max_objects, 20] object features
            mask: [max_objects] valid object mask
            mouse_x, mouse_y: Normalized mouse position
            max_distance: Maximum distance to consider (15% of screen)
            
        Returns:
            (target_object_features, target_index) or (None, -1) if no valid target
        """
        best_obj = None
        best_idx = -1
        best_dist = max_distance
        
        for idx in range(len(objects)):
            if mask[idx] < 0.5:  # Not a valid object
                continue
                
            obj = objects[idx]
            obj_x = obj[0]
            obj_y = obj[1]
            
            # Skip objects with zero position (padding)
            if obj_x == 0.0 and obj_y == 0.0:
                continue
            
            # Calculate distance
            dist = np.sqrt((mouse_x - obj_x) ** 2 + (mouse_y - obj_y) ** 2)
            
            if dist < best_dist:
                best_dist = dist
                best_obj = obj
                best_idx = idx
        
        return best_obj, best_idx


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        sample = self.samples[idx]
        state = torch.from_numpy(sample['state'])
        goal = torch.from_numpy(sample['goal'])
        target_info = torch.from_numpy(sample['target_info'])
        action = torch.from_numpy(sample['action'])

        if 'local_visual' in sample:
            local_visual = torch.from_numpy(sample['local_visual'])
        else:
            local_visual = torch.zeros(64, dtype=torch.float32)

        return state, goal, target_info, action, local_visual

    def has_visual_features(self) -> bool:
        """Check if any samples have visual features."""
        return any('local_visual' in s for s in self.samples)


class ExecutorLoss(nn.Module):
    """
    Custom loss for Executor training.

    Combines:
    - MSE for mouse position (continuous)
    - BCE for click decision (binary)
    - BCE for all keyboard keys (binary, generic)

    Action format: [mouse_x, mouse_y, click, key1, key2, ..., keyN]
    where keys are in KEYBOARD_KEYS order (NUM_KEYBOARD_KEYS total)
    """

    def __init__(self, mouse_weight: float = 1.0, click_weight: float = 0.5,
                 keyboard_weight: float = 0.3):
        super().__init__()
        self.mouse_weight = mouse_weight
        self.click_weight = click_weight
        self.keyboard_weight = keyboard_weight
        self.action_dim = 3 + NUM_KEYBOARD_KEYS

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            pred: [B, action_dim] predicted actions
            target: [B, action_dim] target actions

        Returns:
            total_loss: Combined loss
            metrics: Dict of individual losses
        """
        # Ensure correct dimensions
        if pred.shape[1] < self.action_dim:
            pad_size = self.action_dim - pred.shape[1]
            pred = F.pad(pred, (0, pad_size), value=0.0)
        if target.shape[1] < self.action_dim:
            pad_size = self.action_dim - target.shape[1]
            target = F.pad(target, (0, pad_size), value=-1.0)

        # Mouse position loss (indices 0-1)
        pred_mouse = torch.sigmoid(pred[:, :2])
        target_mouse = target[:, :2]
        mouse_loss = F.mse_loss(pred_mouse, target_mouse)

        # Click loss (index 2)
        pred_click = pred[:, 2]
        target_click = (target[:, 2] > 0).float()
        
        # Weighted BCE for click (clicks are uncommon but not extremely rare)
        click_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0, device=pred.device))
        click_loss = click_criterion(pred_click, target_click)

        # Keyboard losses (indices 3 onwards, one per key in KEYBOARD_KEYS)
        keyboard_start = 3
        pred_keyboard = pred[:, keyboard_start:keyboard_start + NUM_KEYBOARD_KEYS]
        target_keyboard = (target[:, keyboard_start:keyboard_start + NUM_KEYBOARD_KEYS] > 0).float()

        # Average BCE across all keyboard keys
        keyboard_loss = F.binary_cross_entropy_with_logits(
            pred_keyboard, target_keyboard, reduction='mean'
        )

        # Combined loss
        total_loss = (self.mouse_weight * mouse_loss +
                     self.click_weight * click_loss +
                     self.keyboard_weight * keyboard_loss)

        metrics = {
            'mouse_loss': mouse_loss.item(),
            'click_loss': click_loss.item(),
            'keyboard_loss': keyboard_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, metrics


class ExecutorLossV2(nn.Module):
    """
    Loss function for ExecutorV2 with separate heads.

    - Mouse: Beta NLL (proper bounded [0,1] distribution) or MSE fallback
    - Click: BCEWithLogits with pos_weight
    - Keyboard: Focal Loss (handles extreme class imbalance - keys 99% not pressed)
    """

    def __init__(self,
                 mouse_weight: float = 1.0,
                 click_weight: float = 0.5,
                 keyboard_weight: float = 0.3,
                 click_pos_weight: float = 5.0,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 use_beta: bool = True,
                 chunk_discount: float = 0.95):
        super().__init__()
        self.mouse_weight = mouse_weight
        self.click_weight = click_weight
        self.keyboard_weight = keyboard_weight
        self.click_pos_weight = click_pos_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_beta = use_beta
        self.chunk_discount = chunk_discount

    def beta_nll_loss(self, alpha: torch.Tensor, beta_param: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of target under Beta(alpha, beta)."""
        target = target.clamp(0.001, 0.999)
        dist = torch.distributions.Beta(alpha, beta_param)
        return -dist.log_prob(target).mean()

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal Loss: down-weights easy negatives, focuses on rare key presses."""
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        focal_weight = self.focal_alpha * (1 - pt) ** self.focal_gamma
        return (focal_weight * bce).mean()

    def _compute_single_step(self, pred_dict: Dict, target_action: torch.Tensor,
                             device: torch.device) -> Tuple[torch.Tensor, Dict]:
        """Compute loss for a single timestep."""
        target_mouse = target_action[:, :2].clamp(0.001, 0.999)
        target_click = (target_action[:, 2] > 0).float()
        target_keyboard = (target_action[:, 3:3 + NUM_KEYBOARD_KEYS] > 0).float()

        # Mouse loss
        mouse_raw = pred_dict['mouse']
        if self.use_beta:
            alpha_x = F.softplus(mouse_raw[:, 0]) + 1.0
            beta_x = F.softplus(mouse_raw[:, 1]) + 1.0
            alpha_y = F.softplus(mouse_raw[:, 2]) + 1.0
            beta_y = F.softplus(mouse_raw[:, 3]) + 1.0
            mouse_loss = (self.beta_nll_loss(alpha_x, beta_x, target_mouse[:, 0]) +
                          self.beta_nll_loss(alpha_y, beta_y, target_mouse[:, 1])) / 2
        else:
            pred_mouse = torch.sigmoid(mouse_raw[:, :2])
            mouse_loss = F.mse_loss(pred_mouse, target_mouse)

        # Click loss
        click_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(self.click_pos_weight, device=device))
        click_logit = pred_dict['click']
        if click_logit.dim() > 1:
            click_logit = click_logit.squeeze(-1)
        click_loss = click_criterion(click_logit, target_click)

        # Keyboard loss (Focal)
        keyboard_loss = self.focal_loss(pred_dict['keyboard'], target_keyboard)

        total = (self.mouse_weight * mouse_loss +
                 self.click_weight * click_loss +
                 self.keyboard_weight * keyboard_loss)

        return total, {
            'mouse_loss': mouse_loss.item(),
            'click_loss': click_loss.item(),
            'keyboard_loss': keyboard_loss.item(),
            'total_loss': total.item()
        }

    def forward(self, pred_dict: Dict, target_action: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute loss. Handles both single-step and chunked predictions.

        Args:
            pred_dict: {'mouse': [B,C,4] or [B,4], 'click': [B,C,1] or [B,1], 'keyboard': [B,C,28] or [B,28]}
            target_action: [B, 31] for single-step or [B, C, 31] for chunked
        """
        device = target_action.device
        mouse = pred_dict['mouse']

        # Single-step case
        if mouse.dim() == 2:
            return self._compute_single_step(pred_dict, target_action, device)

        # Chunked case: [B, C, ...]
        chunk_size = mouse.shape[1]
        total_loss = torch.tensor(0.0, device=device)
        agg_metrics = {'mouse_loss': 0, 'click_loss': 0, 'keyboard_loss': 0, 'total_loss': 0}
        discount_sum = 0.0

        for t in range(chunk_size):
            step_pred = {
                'mouse': mouse[:, t],
                'click': pred_dict['click'][:, t],
                'keyboard': pred_dict['keyboard'][:, t]
            }

            # Target for this timestep
            if target_action.dim() == 3:
                step_target = target_action[:, t]
            else:
                step_target = target_action  # Single target, replicated

            step_loss, step_metrics = self._compute_single_step(step_pred, step_target, device)
            discount = self.chunk_discount ** t
            total_loss = total_loss + step_loss * discount
            discount_sum += discount

            for k in agg_metrics:
                agg_metrics[k] += step_metrics[k] * discount

        # Normalize by discount sum
        total_loss = total_loss / discount_sum
        for k in agg_metrics:
            agg_metrics[k] /= discount_sum

        return total_loss, agg_metrics


class ExecutorDatasetChunked(Dataset):
    """
    Dataset that provides action chunks (sequences of consecutive frames).
    Extends ExecutorDataset with sequence extraction for action chunking.
    """

    def __init__(self, data_dir: str, chunk_size: int = 8, stride: int = 4,
                 frame_stack_size: int = 3):
        self.chunk_size = chunk_size
        self.stride = stride
        self.frame_stack_size = frame_stack_size

        # Load data using base dataset
        self._base = ExecutorDataset(data_dir)
        self.samples = self._base.samples

        # Build valid sequence indices (consecutive frames within same recording)
        self._build_sequence_indices()

    def _build_sequence_indices(self):
        """Find valid start indices for action chunks."""
        self.valid_starts = []

        if len(self.samples) == 0:
            return

        # Group samples by their source file for contiguity
        # Samples from the same file are assumed consecutive
        current_file = None
        run_start = 0

        for i, s in enumerate(self.samples):
            src = s.get('_source_file', '')
            if src != current_file:
                # End of previous run
                if current_file is not None:
                    self._add_runs(run_start, i)
                current_file = src
                run_start = i

        # Final run
        if current_file is not None:
            self._add_runs(run_start, len(self.samples))

        # If no source file info, treat all as one contiguous run
        if not self.valid_starts and len(self.samples) >= self.chunk_size + self.frame_stack_size - 1:
            for i in range(self.frame_stack_size - 1, len(self.samples) - self.chunk_size + 1, self.stride):
                self.valid_starts.append(i)

    def _add_runs(self, start: int, end: int):
        """Add valid start indices within a contiguous run."""
        run_length = end - start
        min_required = self.chunk_size + self.frame_stack_size - 1
        if run_length >= min_required:
            for i in range(start + self.frame_stack_size - 1,
                           end - self.chunk_size + 1, self.stride):
                self.valid_starts.append(i)

    def __len__(self):
        return max(len(self.valid_starts), 1)

    def has_visual_features(self):
        return self._base.has_visual_features()

    def __getitem__(self, idx):
        if not self.valid_starts:
            # Fallback to single-sample if no valid sequences
            return self._base[min(idx, len(self._base) - 1)]

        start = self.valid_starts[idx % len(self.valid_starts)]
        s = self.samples[start]

        # Frame-stacked state: concatenate last frame_stack_size states
        states = []
        for offset in range(-(self.frame_stack_size - 1), 1):
            si = max(0, start + offset)
            states.append(self.samples[si]['state'][:64] if len(self.samples[si]['state']) > 64
                          else self.samples[si]['state'])
        stacked_state = np.concatenate(states).astype(np.float32)

        goal = s['goal'].astype(np.float32) if isinstance(s['goal'], np.ndarray) else np.array(s['goal'], dtype=np.float32)
        target_info = s['target_info'].astype(np.float32) if isinstance(s['target_info'], np.ndarray) else np.array(s['target_info'], dtype=np.float32)

        # Action chunk: [chunk_size, action_dim]
        action_chunk = np.stack([
            self.samples[start + t]['action'].astype(np.float32)
            if isinstance(self.samples[start + t]['action'], np.ndarray)
            else np.array(self.samples[start + t]['action'], dtype=np.float32)
            for t in range(self.chunk_size)
        ])

        # Visual features
        local_visual = np.zeros(64, dtype=np.float32)
        if 'local_visual' in s and s['local_visual'] is not None:
            local_visual = s['local_visual'].astype(np.float32)

        return (torch.from_numpy(stacked_state),
                torch.from_numpy(goal),
                torch.from_numpy(target_info),
                torch.from_numpy(action_chunk),
                torch.from_numpy(local_visual))


def train_executor(
    data_dir: str,
    output_path: str = "executor.pt",
    config: Optional[ExecutorConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    device: str = "cuda",
    pretrained_path: Optional[str] = None
) -> nn.Module:
    """
    Train the Executor model.

    Args:
        data_dir: Directory containing training data
        output_path: Where to save trained model
        config: Executor model configuration
        training_config: Training hyperparameters
        device: Training device
        pretrained_path: Path to pretrained model for fine-tuning (optional)

    Returns:
        Trained Executor model
    """
    if config is None:
        config = ExecutorConfig()
    if training_config is None:
        training_config = TrainingConfig()

    print("\n" + "="*60)
    print("  V2 EXECUTOR TRAINING")
    print("="*60)

    # Load data
    print("\n[1/4] Loading training data...")
    dataset = ExecutorDataset(data_dir)

    if len(dataset) == 0:
        raise ValueError(f"No training data found in {data_dir}")

    # Count clicks for balanced sampling
    n_clicks = sum(1 for s in dataset.samples if s['action'][2] > 0)
    click_rate = n_clicks / len(dataset) if len(dataset) > 0 else 0
    print(f"   Click rate: {click_rate:.1%} ({n_clicks} clicks / {len(dataset)} samples)")

    # Split train/val
    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create weighted sampler to oversample clicks (critical for learning!)
    # Without this, the model sees ~7% clicks and learns "never click"
    sample_weights = []
    for idx in train_dataset.indices:
        sample = dataset.samples[idx]
        is_click = sample['action'][2] > 0
        # Weight clicks 5x more than non-clicks (adjustable)
        weight = 5.0 if is_click else 1.0
        sample_weights.append(weight)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        sampler=sampler,  # Use weighted sampler instead of shuffle
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")

    # Create model
    print("\n[2/4] Creating Executor model...")
    use_v2 = True  # Default to V2 architecture
    if pretrained_path and Path(pretrained_path).exists():
        print(f"   Loading pretrained weights from: {pretrained_path}")
        from ..models.executor import load_executor
        model = load_executor(pretrained_path, device=device, force_v2=use_v2)
        print("   Fine-tuning from pretrained model")
    else:
        model = create_executor(config, device=device, use_v2=use_v2)
        if pretrained_path:
            print(f"   Pretrained path not found: {pretrained_path}")
            print("   Training from scratch instead")
    model.train()

    is_v2 = isinstance(model, ExecutorV2)
    print(f"   Model: {model.__class__.__name__} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    if is_v2:
        criterion = ExecutorLossV2(
            mouse_weight=1.0,
            click_weight=0.5,
            keyboard_weight=0.3,
            click_pos_weight=5.0,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            use_beta=config.use_beta_distribution,
            chunk_discount=0.95
        )
    else:
        criterion = ExecutorLoss(
            mouse_weight=1.0,
            click_weight=0.5,
            keyboard_weight=0.3
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.executor_epochs,
        eta_min=training_config.learning_rate * 0.01
    )

    # TensorBoard logging
    from .training_utils import TrainingLogger
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TrainingLogger(
        log_dir="runs",
        experiment_name=f"executor_{timestamp}",
        use_tensorboard=True,
        log_to_file=True
    )

    # Training loop
    print("\n[3/4] Training...")
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(training_config.executor_epochs):
        model.train()
        train_metrics = {'mouse_loss': 0, 'click_loss': 0, 'keyboard_loss': 0, 'total_loss': 0}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_config.executor_epochs}")

        for batch_data in pbar:
            state, goal, target_info, action, local_visual = batch_data
            state = state.to(device)
            goal = goal.to(device)
            target_info = target_info.to(device)
            action = action.to(device)
            local_visual = local_visual.to(device)

            optimizer.zero_grad()

            # Forward pass
            if is_v2:
                vis = local_visual if model.visual_dim > 0 else None
                pred = model(state, goal, target_info, visual_features=vis)
                loss, metrics = criterion(pred, action)
            elif hasattr(model, 'forward_with_visual') and dataset.has_visual_features():
                pred, _ = model.forward_with_visual(state, goal, target_info, local_visual)
                loss, metrics = criterion(pred, action)
            else:
                pred, _ = model(state, goal, target_info)
                loss, metrics = criterion(pred, action)

            # Backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)

            optimizer.step()

            # Accumulate metrics
            for k in train_metrics:
                train_metrics[k] += metrics.get(k, 0)
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'mouse': f"{metrics['mouse_loss']:.4f}",
                'kbd': f"{metrics['keyboard_loss']:.4f}"
            })

        scheduler.step()

        # Average train metrics
        for k in train_metrics:
            train_metrics[k] /= max(num_batches, 1)

        # Validation
        model.eval()
        val_metrics = {'mouse_loss': 0, 'click_loss': 0, 'keyboard_loss': 0, 'total_loss': 0}
        num_val_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                state, goal, target_info, action, local_visual = batch_data
                state = state.to(device)
                goal = goal.to(device)
                target_info = target_info.to(device)
                action = action.to(device)
                local_visual = local_visual.to(device)

                if is_v2:
                    vis = local_visual if model.visual_dim > 0 else None
                    pred = model(state, goal, target_info, visual_features=vis)
                elif hasattr(model, 'forward_with_visual') and dataset.has_visual_features():
                    pred, _ = model.forward_with_visual(state, goal, target_info, local_visual)
                else:
                    pred, _ = model(state, goal, target_info)
                loss, metrics = criterion(pred, action)

                for k in val_metrics:
                    val_metrics[k] += metrics.get(k, 0)
                num_val_batches += 1

        for k in val_metrics:
            val_metrics[k] /= max(num_val_batches, 1)

        # Log to TensorBoard
        logger.log_scalar("Loss/train", train_metrics['total_loss'], epoch)
        logger.log_scalar("Loss/val", val_metrics['total_loss'], epoch)
        logger.log_scalar("Loss/mouse_train", train_metrics['mouse_loss'], epoch)
        logger.log_scalar("Loss/mouse_val", val_metrics['mouse_loss'], epoch)
        logger.log_scalar("Loss/click_train", train_metrics['click_loss'], epoch)
        logger.log_scalar("Loss/click_val", val_metrics['click_loss'], epoch)
        logger.log_scalar("Loss/keyboard_train", train_metrics['keyboard_loss'], epoch)
        logger.log_scalar("Loss/keyboard_val", val_metrics['keyboard_loss'], epoch)
        logger.log_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        # Log
        print(f"   Train Loss: {train_metrics['total_loss']:.4f} | "
              f"Val Loss: {val_metrics['total_loss']:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_model_state = model.state_dict().copy()
            print(f"   New best model! Val loss: {best_val_loss:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save
    print(f"\n[4/4] Saving to {output_path}...")
    save_executor(model, output_path)

    # Close logger
    logger.close()

    print("\n" + "="*60)
    print(f"  Training complete! Best val loss: {best_val_loss:.4f}")
    print("="*60)

    return model


def main():
    parser = argparse.ArgumentParser(description='Train V2 Executor')
    parser.add_argument('--data', type=str, default='darkorbit_bot/data/recordings_v2',
                       help='Directory containing training recordings (default: darkorbit_bot/data/recordings_v2)')
    parser.add_argument('--output', type=str, default='models/v2/executor/best_model.pt',
                       help='Output model path (default: models/v2/executor/best_model.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--use-lstm', action='store_true',
                       help='Force LSTM instead of Mamba')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained model for fine-tuning')

    args = parser.parse_args()

    config = ExecutorConfig()
    training_config = TrainingConfig(
        executor_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    train_executor(
        data_dir=args.data,
        output_path=args.output,
        config=config,
        training_config=training_config,
        device=args.device,
        pretrained_path=args.pretrained
    )


if __name__ == "__main__":
    main()
