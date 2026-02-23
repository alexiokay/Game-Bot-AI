"""
V2 Tactician Training - Target Selection

Trains the Tactician to select targets based on goal and visible objects.
Uses cross-attention to determine which object to focus on.

Training data:
- objects: [max_objects, object_dim] detected objects with features
- object_mask: [max_objects] which objects are valid
- goal: [goal_dim] current goal embedding
- target_idx: Index of human's chosen target
- approach: Approach vector (distance, angle, speed to approach)

Usage:
    python train_tactician.py --data recordings/ --epochs 50 --batch-size 32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import pickle
import argparse
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from io import BytesIO
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..models.tactician import create_tactician, save_tactician, Tactician
from ..models.tactician import create_tactician, save_tactician, Tactician
from ..config import TacticianConfig, TrainingConfig, ENEMY_CLASSES, LOOT_CLASSES, PLAYER_CLASSES


class TacticianDataset(Dataset):
    """
    Dataset for Tactician training.

    Extracts target selection from human recordings by analyzing:
    - Where the human clicked/aimed
    - Which objects were visible
    - What mode they were in
    """

    def __init__(self, data_dir: str, max_objects: int = 16, object_dim: int = 20):
        self.data_dir = Path(data_dir)
        self.max_objects = max_objects
        self.object_dim = object_dim
        self.samples: List[Dict] = []

        self._load_recordings()

    def _load_recordings(self):
        """Load all recordings (recursively)."""
        recording_files = list(self.data_dir.glob("**/*.npz"))
        json_files = list(self.data_dir.glob("**/*.json"))
        pkl_files = list(self.data_dir.glob("**/*.pkl"))
        # Filter to only sequence files (not metadata.json etc)
        json_files = [f for f in json_files if 'sequence_' in f.name or 'recording' in f.name.lower()]
        pkl_files = [f for f in pkl_files if 'shadow_recording' in f.name.lower() or 'recording' in f.name.lower()]
        recording_files.extend(json_files)
        recording_files.extend(pkl_files)

        print(f"[Tactician] Found {len(recording_files)} recording files")

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

        print(f"[Tactician] Loaded {len(self.samples)} training samples")

    def _load_npz(self, path: Path):
        """Load from numpy archive."""
        data = np.load(path, allow_pickle=True)

        objects_arr = data.get('objects', None)
        masks_arr = data.get('object_masks', None)
        goals_arr = data.get('goals', None)
        actions = data.get('actions', None)
        detections = data.get('detections', None)

        if actions is None:
            return

        n_frames = len(actions)

        for i in range(n_frames):
            # Get or synthesize objects
            if objects_arr is not None and i < len(objects_arr):
                objects = objects_arr[i]
                mask = masks_arr[i] if masks_arr is not None else np.ones(len(objects))
            elif detections is not None and i < len(detections):
                objects, mask = self._detections_to_features(detections[i])
            else:
                # Skip if no object data
                continue

            # Pad/truncate objects
            objects = self._pad_objects(objects, mask)
            mask = self._pad_mask(mask)

            # Get or synthesize goal
            if goals_arr is not None and i < len(goals_arr):
                goal = goals_arr[i]
            else:
                goal = np.random.randn(64).astype(np.float32) * 0.1

            goal = goal[:64] if len(goal) >= 64 else np.pad(goal, (0, 64 - len(goal)))

            # Determine target from action
            action = actions[i]
            target_idx, approach = self._action_to_target(action, objects, mask)

            if target_idx >= 0:  # Only include if valid target
                self.samples.append({
                    'objects': objects.astype(np.float32),
                    'mask': mask.astype(np.float32),
                    'goal': goal.astype(np.float32),
                    'target_idx': target_idx,
                    'approach': approach.astype(np.float32)
                })

    def _load_json(self, path: Path):
        """
        Load from JSON recording.

        Supports three formats:
        1. V2 native format (from recorder_v2.py): has 'objects' and 'object_masks'
        2. Shadow training format: has 'demos' and 'metadata'
        3. V1 format (from filtered_recorder.py): has 'states' and 'actions' only
        """
        with open(path, 'r') as f:
            data = json.load(f)

        # Check for V2 native format with real object data
        if 'objects' in data and 'object_masks' in data:
            self._load_v2_native(data)
            return

        # Check for shadow training format
        if 'demos' in data and 'metadata' in data:
            self._load_shadow_format(data)
            return

        # Fallback to V1 format (states + actions only)
        if 'states' in data and 'actions' in data:
            self._load_v1_format(data)
            return

    def _load_v2_native(self, data: Dict):
        """Load V2 native format with real tracked objects."""
        objects_list = data['objects']       # [n_frames, max_objects, 20]
        masks_list = data['object_masks']    # [n_frames, max_objects]
        actions = data['actions']            # [n_frames, 5]
        mode = data.get('mode', 'EXPLORE')
        n_frames = min(len(objects_list), len(masks_list), len(actions))

        for i in range(n_frames):
            objects = np.array(objects_list[i], dtype=np.float32)
            mask = np.array(masks_list[i], dtype=np.float32)
            action = actions[i]

            # Skip frames with no valid objects - can't train target selection without targets
            if mask.sum() < 0.5:
                continue

            # Pad/truncate to expected dimensions
            if objects.shape[0] < self.max_objects:
                pad_rows = self.max_objects - objects.shape[0]
                objects = np.vstack([objects, np.zeros((pad_rows, objects.shape[1]))])
                mask = np.concatenate([mask, np.zeros(pad_rows)])
            elif objects.shape[0] > self.max_objects:
                objects = objects[:self.max_objects]
                mask = mask[:self.max_objects]

            if objects.shape[1] < self.object_dim:
                pad_cols = self.object_dim - objects.shape[1]
                objects = np.hstack([objects, np.zeros((objects.shape[0], pad_cols))])
            elif objects.shape[1] > self.object_dim:
                objects = objects[:, :self.object_dim]

            # Find target: object closest to mouse position
            mouse_x = action[0] if len(action) > 0 else 0.5
            mouse_y = action[1] if len(action) > 1 else 0.5

            target_idx = self._find_target_object(objects, mask, mouse_x, mouse_y)

            # Ensure target_idx points to a valid object
            if mask[target_idx] < 0.5:
                # Fallback: use first valid object
                valid_indices = np.where(mask > 0.5)[0]
                if len(valid_indices) == 0:
                    continue  # Skip if no valid objects
                target_idx = int(valid_indices[0])

            # Goal based on mode
            goal = self._mode_to_goal(mode)

            # Approach vector
            dx = mouse_x - 0.5
            dy = mouse_y - 0.5
            dist = np.sqrt(dx**2 + dy**2)
            vx = dx / max(dist, 0.01)
            vy = dy / max(dist, 0.01)
            urgency = min(dist * 2, 1.0)
            aggression = 1.0 if (len(action) > 2 and action[2] > 0) else 0.5

            approach = np.array([vx, vy, urgency, aggression], dtype=np.float32)

            self.samples.append({
                'objects': objects.astype(np.float32),
                'mask': mask.astype(np.float32),
                'goal': goal.astype(np.float32),
                'target_idx': target_idx,
                'approach': approach
            })

    def _load_shadow_format(self, data: Dict):
        """
        Load shadow training format: {demos: [...], metadata: {...}}

        Each demo contains:
        - tracked_objects: list of detected objects
        - action: {mouse_x, mouse_y, should_click, keyboard: {...}}
        - human_mode: inferred mode (FIGHT/LOOT/etc)
        """
        demos = data.get('demos', [])
        metadata = data.get('metadata', {})

        print(f"   [Shadow] Loading {len(demos)} demos for tactician training")
        print(f"   [Shadow] Session stats: {metadata.get('stats', {}).get('human_clicks', 0)} clicks")

        from ..config import ENEMY_CLASSES, LOOT_CLASSES

        loaded = 0
        skipped = 0

        for demo in demos:
            tracked = demo.get('tracked_objects', [])

            # Skip frames with no objects
            if not tracked:
                skipped += 1
                continue

            # Build object tensor from tracked_objects
            objects = np.zeros((self.max_objects, self.object_dim), dtype=np.float32)
            mask = np.zeros(self.max_objects, dtype=np.float32)

            for j, obj in enumerate(tracked[:self.max_objects]):
                # Object features: [x, y, vx, vy, speed, w, h, conf, is_enemy, is_loot, ...]
                objects[j, 0] = obj.get('x', 0.5)
                objects[j, 1] = obj.get('y', 0.5)
                objects[j, 2] = obj.get('vx', 0.0)
                objects[j, 3] = obj.get('vy', 0.0)
                objects[j, 4] = np.sqrt(obj.get('vx', 0.0)**2 + obj.get('vy', 0.0)**2)  # speed
                objects[j, 5] = obj.get('width', 0.05)
                objects[j, 6] = obj.get('height', 0.05)
                objects[j, 7] = obj.get('confidence', 0.5)

                # Determine if enemy or loot from class name
                class_name = obj.get('class_name', '')
                is_enemy = 1.0 if class_name in ENEMY_CLASSES else 0.0
                is_loot = 1.0 if class_name in LOOT_CLASSES else 0.0
                objects[j, 8] = is_enemy
                objects[j, 9] = is_loot

                mask[j] = 1.0

            # Get action data
            action_data = demo.get('action')
            if action_data is None:
                skipped += 1
                continue

            if isinstance(action_data, dict):
                mouse_x = float(action_data.get('mouse_x', 0.5))
                mouse_y = float(action_data.get('mouse_y', 0.5))
                clicked = action_data.get('should_click', False)
            elif isinstance(action_data, list):
                mouse_x = float(action_data[0]) if len(action_data) > 0 else 0.5
                mouse_y = float(action_data[1]) if len(action_data) > 1 else 0.5
                clicked = action_data[2] > 0 if len(action_data) > 2 else False
            else:
                skipped += 1
                continue

            # Find target: object closest to mouse position
            target_idx = self._find_target_object(objects, mask, mouse_x, mouse_y)

            # Ensure target_idx points to a valid object
            if mask[target_idx] < 0.5:
                valid_indices = np.where(mask > 0.5)[0]
                if len(valid_indices) == 0:
                    skipped += 1
                    continue
                target_idx = int(valid_indices[0])

            # Goal based on mode
            mode = demo.get('human_mode', 'EXPLORE')
            goal = self._mode_to_goal(mode)

            # Approach vector
            dx = mouse_x - 0.5
            dy = mouse_y - 0.5
            dist = np.sqrt(dx**2 + dy**2)
            vx = dx / max(dist, 0.01)
            vy = dy / max(dist, 0.01)
            urgency = min(dist * 2, 1.0)
            aggression = 1.0 if clicked else 0.5

            approach = np.array([vx, vy, urgency, aggression], dtype=np.float32)

            sample = {
                'objects': objects,
                'mask': mask,
                'goal': goal.astype(np.float32),
                'target_idx': target_idx,
                'approach': approach
            }

            # Store per-object visual features if available
            if 'roi_visual' in demo and demo['roi_visual'] is not None:
                sample['roi_visual'] = np.array(demo['roi_visual'], dtype=np.float32)

            self.samples.append(sample)
            loaded += 1

        n_visual = sum(1 for s in self.samples[-loaded:] if 'roi_visual' in s) if loaded > 0 else 0
        print(f"   [Shadow] Loaded {loaded} samples, skipped {skipped}")
        if n_visual > 0:
            print(f"   [Shadow] {n_visual}/{loaded} samples have roi_visual features")

    def _load_v1_format(self, data: Dict):
        """Load V1 format - creates synthetic target from mouse position."""
        states = data['states']
        actions = data['actions']
        n_frames = min(len(states), len(actions))
        mode = data.get('mode', 'EXPLORE')

        for i in range(n_frames):
            action = actions[i]

            # Get mouse position as target
            mouse_x = action[0] if len(action) > 0 else 0.5
            mouse_y = action[1] if len(action) > 1 else 0.5

            # Create a synthetic "object" at the mouse position
            objects = np.zeros((self.max_objects, self.object_dim), dtype=np.float32)
            mask = np.zeros(self.max_objects, dtype=np.float32)

            # Object 0: The target position (where human aimed)
            objects[0, 0] = mouse_x  # x
            objects[0, 1] = mouse_y  # y
            objects[0, 10] = 1.0     # confidence
            objects[0, 16] = 1.0     # is_enemy
            mask[0] = 1.0

            # Goal based on mode
            goal = self._mode_to_goal(mode)

            # Approach vector
            dx = mouse_x - 0.5
            dy = mouse_y - 0.5
            dist = np.sqrt(dx**2 + dy**2)
            vx = dx / max(dist, 0.01)
            vy = dy / max(dist, 0.01)
            urgency = min(dist * 2, 1.0)
            aggression = 1.0 if (len(action) > 2 and action[2] > 0) else 0.5

            approach = np.array([vx, vy, urgency, aggression], dtype=np.float32)

            self.samples.append({
                'objects': objects,
                'mask': mask,
                'goal': goal.astype(np.float32),
                'target_idx': 0,
                'approach': approach
            })

    def _load_pkl(self, path: Path):
        """
        Load from shadow recording pickle file.

        Shadow recordings contain full hierarchical demos with:
        - tracked_objects: List of object dicts
        - human_target_id: Which object the human targeted
        - human_mode: The mode the human was in
        - frame: JPEG-compressed frame bytes (or raw numpy array for legacy)
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        demos = data.get('demos', [])
        metadata = data.get('metadata', {})
        frame_format = metadata.get('frame_format', None)

        print(f"   Loading {len(demos)} demos from {path.name}")

        samples_added = 0
        for demo in demos:
            # Extract tracked objects
            tracked_objects = demo.get('tracked_objects', [])
            human_target_id = demo.get('human_target_id', -1)
            human_mode = demo.get('human_mode', 'EXPLORE')

            # Skip if no valid objects
            if not tracked_objects or human_target_id < 0 or human_target_id >= len(tracked_objects):
                continue

            # Convert tracked objects to feature arrays
            objects, mask = self._objects_to_features(tracked_objects)

            # Skip if no valid objects after conversion
            if mask.sum() < 0.5:
                continue

            # Pad/truncate to expected dimensions
            objects = self._pad_objects(objects, mask)
            mask = self._pad_mask(mask)

            # Ensure target_idx is within valid range
            target_idx = min(human_target_id, self.max_objects - 1)

            # Verify target is valid
            if mask[target_idx] < 0.5:
                # Fallback: use first valid object
                valid_indices = np.where(mask > 0.5)[0]
                if len(valid_indices) == 0:
                    continue
                target_idx = int(valid_indices[0])

            # Goal based on mode
            goal = self._mode_to_goal(human_mode)

            # Get target object for approach vector
            target_obj = objects[target_idx]
            target_x = target_obj[0]  # x position
            target_y = target_obj[1]  # y position

            # Approach vector (normalized direction + urgency + aggression)
            dx = target_x - 0.5
            dy = target_y - 0.5
            dist = np.sqrt(dx**2 + dy**2)
            vx = dx / max(dist, 0.01)
            vy = dy / max(dist, 0.01)
            urgency = min(dist * 2, 1.0)
            # Aggression based on mode
            aggression = 1.0 if human_mode.upper() in ['FIGHT', 'AGGRESSIVE'] else 0.5

            approach = np.array([vx, vy, urgency, aggression], dtype=np.float32)

            self.samples.append({
                'objects': objects.astype(np.float32),
                'mask': mask.astype(np.float32),
                'goal': goal.astype(np.float32),
                'target_idx': target_idx,
                'approach': approach
            })
            samples_added += 1

        print(f"   Added {samples_added} samples from {path.name}")

    def _objects_to_features(self, tracked_objects: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert list of tracked object dicts to feature arrays.

        Args:
            tracked_objects: List of dicts with keys: x, y, width, height, class_name, etc.

        Returns:
            objects: [n_objects, object_dim] feature array
            mask: [n_objects] validity mask
        """
        n_objects = len(tracked_objects)
        objects = np.zeros((n_objects, self.object_dim), dtype=np.float32)
        mask = np.ones(n_objects, dtype=np.float32)

        for i, obj in enumerate(tracked_objects):
            # Position
            objects[i, 0] = obj.get('x', 0.5)
            objects[i, 1] = obj.get('y', 0.5)

            # Distance (relative to center)
            objects[i, 2] = obj.get('x', 0.5) - 0.5
            objects[i, 3] = obj.get('y', 0.5) - 0.5

            # Velocity
            objects[i, 4] = obj.get('vx', 0.0)
            objects[i, 5] = obj.get('vy', 0.0)
            speed = np.sqrt(objects[i, 4]**2 + objects[i, 5]**2)
            objects[i, 6] = speed

            # Heading (angle of velocity)
            if speed > 0.001:
                objects[i, 7] = np.arctan2(objects[i, 5], objects[i, 4])

            # Size
            objects[i, 8] = obj.get('width', 0.05)
            objects[i, 9] = obj.get('height', 0.05)

            # Confidence
            objects[i, 10] = obj.get('confidence', 1.0)

            # Class type indicators
            class_name = obj.get('class_name', '')
            
            # Use centralized constants for consistent encoding
            is_enemy = 1.0 if class_name in ENEMY_CLASSES else 0.0
            is_loot = 1.0 if class_name in LOOT_CLASSES else 0.0
            is_player = 1.0 if class_name in PLAYER_CLASSES else 0.0
            
            objects[i, 16] = is_enemy
            objects[i, 17] = is_loot
            objects[i, 18] = is_player
            objects[i, 19] = 1.0 - is_enemy - is_loot - is_player  # is_other

        return objects, mask

    def _pad_objects(self, objects: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Pad or truncate objects to max_objects."""
        if objects.shape[0] < self.max_objects:
            pad_rows = self.max_objects - objects.shape[0]
            objects = np.vstack([objects, np.zeros((pad_rows, objects.shape[1]))])
        elif objects.shape[0] > self.max_objects:
            objects = objects[:self.max_objects]

        if objects.shape[1] < self.object_dim:
            pad_cols = self.object_dim - objects.shape[1]
            objects = np.hstack([objects, np.zeros((objects.shape[0], pad_cols))])
        elif objects.shape[1] > self.object_dim:
            objects = objects[:, :self.object_dim]

        return objects

    def _pad_mask(self, mask: np.ndarray) -> np.ndarray:
        """Pad or truncate mask to max_objects."""
        if len(mask) < self.max_objects:
            mask = np.concatenate([mask, np.zeros(self.max_objects - len(mask))])
        elif len(mask) > self.max_objects:
            mask = mask[:self.max_objects]
        return mask

    def _find_target_object(self, objects: np.ndarray, mask: np.ndarray,
                            mouse_x: float, mouse_y: float) -> int:
        """Find the object closest to mouse position."""
        best_idx = 0
        best_dist = float('inf')

        for i in range(len(mask)):
            if mask[i] < 0.5:
                continue

            # Object position is in indices 0, 1
            obj_x = objects[i, 0]
            obj_y = objects[i, 1]

            dist = np.sqrt((obj_x - mouse_x)**2 + (obj_y - mouse_y)**2)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        return best_idx

    def _mode_to_goal(self, mode: str) -> np.ndarray:
        """Convert mode string to goal embedding."""
        goal = np.zeros(64, dtype=np.float32)

        # Encode mode in first 5 dimensions (one-hot)
        mode_map = {'FIGHT': 0, 'AGGRESSIVE': 0, 'LOOT': 1, 'FLEE': 2, 'EXPLORE': 3, 'PASSIVE': 3, 'CAUTIOUS': 4}
        mode_idx = mode_map.get(mode.upper(), 3)  # Default to EXPLORE
        goal[mode_idx] = 1.0

        # Add some noise to make goal vector richer
        goal[5:] = np.random.randn(59).astype(np.float32) * 0.1

        return goal

    def __len__(self) -> int:
        return len(self.samples)

    def has_visual_features(self) -> bool:
        """Check if any samples have roi_visual features."""
        return any('roi_visual' in s for s in self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        objects = sample['objects']
        mask = sample['mask']
        goal = sample['goal']
        approach = sample['approach']

        # Replace any NaN/Inf with 0 (shouldn't happen but be safe)
        objects = np.nan_to_num(objects, nan=0.0, posinf=1.0, neginf=-1.0)
        goal = np.nan_to_num(goal, nan=0.0, posinf=1.0, neginf=-1.0)
        approach = np.nan_to_num(approach, nan=0.0, posinf=1.0, neginf=-1.0)

        # ROI visual features for per-object visual context
        if 'roi_visual' in sample:
            roi_visual = np.nan_to_num(sample['roi_visual'], nan=0.0, posinf=1.0, neginf=-1.0)
        else:
            roi_visual = np.zeros((self.max_objects, 128), dtype=np.float32)

        return (
            torch.from_numpy(objects),
            torch.from_numpy(mask),
            torch.from_numpy(goal),
            torch.tensor(sample['target_idx'], dtype=torch.long),
            torch.from_numpy(approach),
            torch.from_numpy(roi_visual)
        )


class TacticianLoss(nn.Module):
    """
    Loss for Tactician training.

    Combines:
    - Cross entropy for target selection
    - MSE for approach vector
    - Attention entropy regularization (encourage confident selections)
    """

    def __init__(self, target_weight: float = 1.0, approach_weight: float = 0.3,
                 entropy_weight: float = 0.1):
        super().__init__()
        self.target_weight = target_weight
        self.approach_weight = approach_weight
        self.entropy_weight = entropy_weight

    def forward(self, target_weights: torch.Tensor, pred_approach: torch.Tensor,
                target_idx: torch.Tensor, target_approach: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            target_weights: [B, max_objects] attention weights over objects (already softmax)
            pred_approach: [B, approach_dim] predicted approach vector
            target_idx: [B] ground truth target indices
            target_approach: [B, approach_dim] ground truth approach

        Returns:
            total_loss: Combined loss
            metrics: Dict of individual losses
        """
        # Target selection loss
        # target_weights are already probabilities (from attention softmax)
        # Use NLL loss with log-probabilities
        weights_clamped = target_weights.clamp(min=1e-7, max=1.0)
        log_probs = torch.log(weights_clamped)
        target_loss = F.nll_loss(log_probs, target_idx, reduction='mean')

        # Approach vector loss
        approach_loss = F.mse_loss(pred_approach, target_approach)

        # Entropy regularization (encourage confident selections)
        # Lower entropy = more confident
        entropy = -(weights_clamped * log_probs).sum(dim=-1).mean()
        entropy_loss = entropy  # We want low entropy

        # Total
        total_loss = (self.target_weight * target_loss +
                     self.approach_weight * approach_loss +
                     self.entropy_weight * entropy_loss)

        # Accuracy
        pred_target = target_weights.argmax(dim=-1)
        accuracy = (pred_target == target_idx).float().mean().item()

        metrics = {
            'target_loss': target_loss.item(),
            'approach_loss': approach_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item(),
            'accuracy': accuracy
        }

        return total_loss, metrics


def train_tactician(
    data_dir: str,
    output_path: str = "tactician.pt",
    config: Optional[TacticianConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    device: str = "cuda",
    pretrained_path: Optional[str] = None
) -> Tactician:
    """
    Train the Tactician model.

    Args:
        data_dir: Directory with training recordings
        output_path: Where to save the model
        config: Model configuration
        training_config: Training hyperparameters
        device: cuda or cpu
        pretrained_path: Path to pretrained model for fine-tuning (optional)
    """
    if config is None:
        config = TacticianConfig()
    if training_config is None:
        training_config = TrainingConfig()

    print("\n" + "="*60)
    print("  V2 TACTICIAN TRAINING")
    print("="*60)

    # Load data
    print("\n[1/4] Loading training data...")
    dataset = TacticianDataset(data_dir, max_objects=config.max_objects, object_dim=config.object_dim)

    if len(dataset) == 0:
        raise ValueError(f"No training data found in {data_dir}")

    # Split
    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
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
    print("\n[2/4] Creating Tactician model...")
    if pretrained_path and Path(pretrained_path).exists():
        print(f"   Loading pretrained weights from: {pretrained_path}")
        from ..models.tactician import load_tactician
        model = load_tactician(pretrained_path, device=device)
        print("   ✅ Fine-tuning from pretrained model")
    else:
        model = create_tactician(config, device=device)
        if pretrained_path:
            print(f"   ⚠️ Pretrained path not found: {pretrained_path}")
            print("   Training from scratch instead")
    model.train()

    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = TacticianLoss(
        target_weight=1.0,
        approach_weight=0.3,
        entropy_weight=0.1
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.tactician_epochs,
        eta_min=training_config.learning_rate * 0.01
    )

    # TensorBoard logging
    from .training_utils import TrainingLogger
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TrainingLogger(
        log_dir="runs",
        experiment_name=f"tactician_{timestamp}",
        use_tensorboard=True,
        log_to_file=True
    )

    # Training loop
    print("\n[3/4] Training...")
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(training_config.tactician_epochs):
        model.train()
        train_metrics = {'target_loss': 0, 'accuracy': 0, 'total_loss': 0}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_config.tactician_epochs}")

        for batch_data in pbar:
            objects, mask, goal, target_idx, approach, roi_visual = batch_data
            objects = objects.to(device)
            mask = mask.to(device)
            goal = goal.to(device)
            target_idx = target_idx.to(device)
            approach = approach.to(device)
            roi_visual = roi_visual.to(device)

            optimizer.zero_grad()

            # Forward (use visual features if model supports it)
            if hasattr(model, 'forward_with_visual') and dataset.has_visual_features():
                target_weights, pred_approach, _ = model.forward_with_visual(
                    objects, roi_visual, mask, goal)
            else:
                target_weights, pred_approach, _ = model(objects, mask, goal)

            # Check for NaN in model output (for debugging)
            if torch.isnan(target_weights).any():
                print(f"\n   WARNING: NaN in target_weights!")
                print(f"   mask.sum(): {mask.sum(dim=1).tolist()}")
                print(f"   objects has NaN: {torch.isnan(objects).any()}")
                print(f"   goal has NaN: {torch.isnan(goal).any()}")
                continue  # Skip this batch

            # Loss
            loss, metrics = criterion(target_weights, pred_approach, target_idx, approach)

            # Skip if loss is NaN
            if torch.isnan(loss):
                print(f"\n   WARNING: NaN loss, skipping batch")
                continue

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
            optimizer.step()

            # Accumulate
            for k in train_metrics:
                train_metrics[k] += metrics.get(k, 0)
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'acc': f"{metrics['accuracy']:.2%}"
            })

        scheduler.step()

        # Average
        for k in train_metrics:
            train_metrics[k] /= max(num_batches, 1)

        # Validation
        model.eval()
        val_metrics = {'target_loss': 0, 'accuracy': 0, 'total_loss': 0}
        num_val_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                objects, mask, goal, target_idx, approach, roi_visual = batch_data
                objects = objects.to(device)
                mask = mask.to(device)
                goal = goal.to(device)
                target_idx = target_idx.to(device)
                approach = approach.to(device)
                roi_visual = roi_visual.to(device)

                if hasattr(model, 'forward_with_visual') and dataset.has_visual_features():
                    target_weights, pred_approach, _ = model.forward_with_visual(
                        objects, roi_visual, mask, goal)
                else:
                    target_weights, pred_approach, _ = model(objects, mask, goal)
                loss, metrics = criterion(target_weights, pred_approach, target_idx, approach)

                for k in val_metrics:
                    val_metrics[k] += metrics.get(k, 0)
                num_val_batches += 1

        for k in val_metrics:
            val_metrics[k] /= max(num_val_batches, 1)

        # Log to TensorBoard
        logger.log_scalar("Loss/train", train_metrics['total_loss'], epoch)
        logger.log_scalar("Loss/val", val_metrics['total_loss'], epoch)
        logger.log_scalar("Accuracy/train", train_metrics['accuracy'], epoch)
        logger.log_scalar("Accuracy/val", val_metrics['accuracy'], epoch)
        logger.log_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        print(f"   Train Loss: {train_metrics['total_loss']:.4f} Acc: {train_metrics['accuracy']:.2%} | "
              f"Val Loss: {val_metrics['total_loss']:.4f} Acc: {val_metrics['accuracy']:.2%}")

        # Save best
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_model_state = model.state_dict().copy()
            print(f"   New best model!")

    # Load best
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save
    print(f"\n[4/4] Saving to {output_path}...")
    save_tactician(model, output_path)

    # Close logger
    logger.close()

    print("\n" + "="*60)
    print(f"  Training complete! Best val loss: {best_val_loss:.4f}")
    print("="*60)

    return model


def main():
    parser = argparse.ArgumentParser(description='Train V2 Tactician')
    parser.add_argument('--data', type=str, default='darkorbit_bot/data/recordings_v2',
                       help='Directory containing training recordings (default: darkorbit_bot/data/recordings_v2)')
    parser.add_argument('--output', type=str, default='models/v2/tactician/best_model.pt',
                       help='Output model path (default: models/v2/tactician/best_model.pt)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained model for fine-tuning')

    args = parser.parse_args()

    config = TacticianConfig()
    training_config = TrainingConfig(
        tactician_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    train_tactician(
        data_dir=args.data,
        output_path=args.output,
        config=config,
        training_config=training_config,
        device=args.device,
        pretrained_path=args.pretrained
    )


if __name__ == "__main__":
    main()
