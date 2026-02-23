"""
V2 Fine-tuning with VLM Corrections

Fine-tunes the V2 models using VLM feedback collected during bot operation.
The VLM provides corrections for:
- Strategist: Mode selection errors
- Tactician: Target selection errors
- Executor: Movement quality issues

Usage:
    python finetune_with_vlm.py --corrections data/vlm_corrections_v2/ --epochs 20
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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v2.models.executor import create_executor, save_executor, load_executor, ExecutorV2
from v2.models.strategist import create_strategist, save_strategist, load_strategist
from v2.models.tactician import create_tactician, save_tactician, load_tactician
from v2.config import ExecutorConfig, StrategistConfig, TacticianConfig, TrainingConfig, Mode, KEYBOARD_KEYS, NUM_KEYBOARD_KEYS, FULL_STATE_DIM
from v2.training.train_executor import ExecutorLossV2
from v2.training.train_tactician import TacticianLoss


class VLMCorrectionDataset(Dataset):
    """
    Dataset for fine-tuning from VLM corrections.

    VLM corrections contain:
    - vlm_result: What VLM thinks should happen
    - policy_output: What the model actually did
    - state_vector: The state at that moment

    Supports visual models by zero-padding visual features when not present in corrections.
    """

    def __init__(self, corrections_dir: str, component: str = 'executor',
                 visual_model: bool = False, visual_dim: int = 0):
        """
        Args:
            corrections_dir: Directory with v2_corrections_*.json files
            component: Which component to fine-tune ('executor', 'strategist', 'tactician')
            visual_model: Whether the model being trained has visual features
            visual_dim: Dimension of visual features (64 for executor, 512 for strategist)
        """
        self.corrections_dir = Path(corrections_dir)
        self.component = component
        self.visual_model = visual_model
        self.visual_dim = visual_dim
        self.samples: List[Dict] = []

        self._load_corrections()

    def _load_corrections(self):
        """Load all correction files."""
        correction_files = list(self.corrections_dir.glob("v2_corrections_*.json"))
        print(f"[VLM-Finetune] Found {len(correction_files)} correction files")

        total_corrections = 0
        valid_corrections = 0

        # Track correction types for auto-detection
        correction_types = {'executor': 0, 'strategist': 0, 'tactician': 0}

        for file_path in correction_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                corrections = data.get('corrections', [])
                total_corrections += len(corrections)

                # Analyze correction types
                for corr in corrections:
                    vlm_result = corr.get('vlm_result', {})

                    # Detect type of correction
                    if 'current_mode_correct' in vlm_result or 'recommended_mode' in vlm_result:
                        correction_types['strategist'] += 1
                    if 'quality' in vlm_result or 'correction' in vlm_result:
                        correction_types['executor'] += 1
                    if 'target_correct' in vlm_result or 'recommended_target' in vlm_result:
                        correction_types['tactician'] += 1

                for corr in corrections:
                    sample = self._parse_correction(corr)
                    if sample:
                        self.samples.append(sample)
                        valid_corrections += 1

            except Exception as e:
                print(f"   Warning: Could not load {file_path}: {e}")

        # Show correction type breakdown
        print(f"\n[VLM-Finetune] Correction breakdown:")
        print(f"   Strategist (mode): {correction_types['strategist']} corrections")
        print(f"   Executor (aim): {correction_types['executor']} corrections")
        print(f"   Tactician (target): {correction_types['tactician']} corrections")
        print(f"\n[VLM-Finetune] Loaded {valid_corrections}/{total_corrections} valid corrections for {self.component}")

        # Warn if user selected wrong component
        if self.component == 'executor' and correction_types['executor'] == 0 and correction_types['strategist'] > 0:
            print(f"\n[WARNING] You selected 'executor' but all corrections are for 'strategist'!")
            print(f"[WARNING] Try: --component strategist")
        elif self.component == 'strategist' and correction_types['strategist'] == 0 and correction_types['executor'] > 0:
            print(f"\n[WARNING] You selected 'strategist' but all corrections are for 'executor'!")
            print(f"[WARNING] Try: --component executor")

    def _parse_correction(self, corr: Dict) -> Optional[Dict]:
        """Parse a single correction into a training sample."""
        vlm_result = corr.get('vlm_result', {})
        policy_output = corr.get('policy_output', {})
        state_vector = corr.get('state_vector', [])

        if not state_vector:
            return None

        if self.component == 'executor':
            return self._parse_executor_correction(vlm_result, policy_output, state_vector, corr)
        elif self.component == 'strategist':
            return self._parse_strategist_correction(vlm_result, policy_output, state_vector, corr)
        elif self.component == 'tactician':
            return self._parse_tactician_correction(vlm_result, policy_output, state_vector, corr)

        return None

    def _parse_executor_correction(self, vlm_result: Dict, policy_output: Dict,
                                     state_vector: List, corr: Dict = None) -> Optional[Dict]:
        """
        Parse executor correction - movement quality.

        IMPORTANT: Properly separates object features (target_info) from action (mouse position).
        The model learns: given object at position X → aim at position Y
        """
        # Check if this is an executor-level correction
        quality = vlm_result.get('quality', 'unknown')
        correction = vlm_result.get('correction', {})

        if quality == 'good' or not correction:
            return None  # No correction needed

        # Get the corrected action from VLM (where mouse SHOULD be)
        corrected_x = correction.get('move_x', correction.get('mouse_x', 0.5))
        corrected_y = correction.get('move_y', correction.get('mouse_y', 0.5))
        should_click = correction.get('should_click', False)

        # Build state (compact to 64 dims)
        state = np.array(state_vector[:64], dtype=np.float32)
        if len(state) < 64:
            state = np.pad(state, (0, 64 - len(state)))

        # Build goal from policy output
        goal_data = policy_output.get('goal', [])
        if isinstance(goal_data, list) and len(goal_data) >= 64:
            goal = np.array(goal_data[:64], dtype=np.float32)
        else:
            goal = np.zeros(64, dtype=np.float32)

        # Build target_info from OBJECT features (not mouse position!)
        target_info = np.zeros(34, dtype=np.float32)

        # Try to get object data from correction
        objects = corr.get('objects', []) if corr else []
        object_mask = corr.get('object_mask', []) if corr else []
        target_idx = corr.get('target_idx', policy_output.get('target_idx', -1)) if corr else -1

        if objects and target_idx >= 0 and target_idx < len(objects):
            obj = objects[target_idx]
            if isinstance(obj, list) and len(obj) >= 2:
                # Object feature layout from TrackedObject.to_feature_vector (20-dim):
                # [0-3]:   x, y, distance_to_player, angle_to_player (position)
                # [4-7]:   vx, vy, speed, heading (velocity)
                # [8-11]:  width, height, confidence, age_normalized (bbox)
                # [12-15]: hits_norm, time_since_update_norm, is_tracked, is_lost (tracking)
                # [16-19]: is_enemy, is_loot, is_player, is_other (class)

                target_info[0] = obj[0]   # Object x position
                target_info[1] = obj[1]   # Object y position
                target_info[2] = obj[4] if len(obj) > 4 else 0.0   # velocity x
                target_info[3] = obj[5] if len(obj) > 5 else 0.0   # velocity y
                target_info[4] = obj[6] if len(obj) > 6 else 0.0   # speed
                target_info[5] = obj[8] if len(obj) > 8 else 0.05  # width
                target_info[6] = obj[9] if len(obj) > 9 else 0.05  # height
                target_info[7] = obj[10] if len(obj) > 10 else 1.0 # confidence
                target_info[8] = obj[16] if len(obj) > 16 else 0.0 # is_enemy (FIXED: was obj[14])
                target_info[9] = obj[17] if len(obj) > 17 else 0.0 # is_loot (FIXED: was obj[15])
                # [10-11] aim offset - will be learned
                target_info[10] = corrected_x - obj[0]  # Relative aim offset X
                target_info[11] = corrected_y - obj[1]  # Relative aim offset Y
                # Copy remaining features
                for j in range(min(20, len(obj))):
                    if 12 + j < 32:
                        target_info[12 + j] = obj[j]
        else:
            # Fallback: no object data available (old format corrections)
            # Use corrected position as both - this is less useful but better than nothing
            target_info[0] = corrected_x
            target_info[1] = corrected_y

        # Build corrected action (where mouse SHOULD be)
        # Full 31-dim format: [mouse_x, mouse_y, click, ...keyboard_keys...]
        action = np.zeros(3 + NUM_KEYBOARD_KEYS, dtype=np.float32)
        action[0] = corrected_x
        action[1] = corrected_y
        action[2] = 3.0 if should_click else -3.0  # Logit space
        # Keyboard keys default to -3.0 (not pressed) - VLM doesn't correct keyboard

        # If training visual model, pad target_info with zeros for visual features
        # Non-visual: target_info is 34 dims
        # Visual: target_info is 98 dims (34 + 64 visual)
        if self.visual_model and self.visual_dim > 0:
            visual_padding = np.zeros(self.visual_dim, dtype=np.float32)
            target_info = np.concatenate([target_info, visual_padding])

        return {
            'state': state,
            'goal': goal,
            'target_info': target_info,
            'action': action,
            'weight': 2.0 if quality == 'poor' else 1.5  # Higher weight for worse errors
        }

    def _parse_strategist_correction(self, vlm_result: Dict, policy_output: Dict,
                                     state_vector: List, corr: Dict) -> Optional[Dict]:
        """Parse strategist correction - mode selection."""
        # Check if mode was incorrect
        if vlm_result.get('current_mode_correct', True):
            return None

        recommended_mode = vlm_result.get('recommended_mode', '')
        if not recommended_mode:
            return None

        # Map mode name to index
        mode_map = {'FIGHT': 0, 'LOOT': 1, 'FLEE': 2, 'EXPLORE': 3, 'CAUTIOUS': 4}
        if recommended_mode not in mode_map:
            return None

        target_mode_idx = mode_map[recommended_mode]

        # Build state history (for strategist, we need temporal data)
        # For now, just use the current state repeated
        state = np.array(state_vector, dtype=np.float32)
        if len(state) < FULL_STATE_DIM:
            state = np.pad(state, (0, FULL_STATE_DIM - len(state)))
        state = state[:FULL_STATE_DIM]

        # Create state history (60 timesteps for strategist)
        state_history = np.tile(state, (60, 1))

        # If training visual model, pad state_history with zeros for visual features
        # Non-visual: state_history is [60, FULL_STATE_DIM]
        # Visual: state_history is [60, FULL_STATE_DIM + 512]
        if self.visual_model and self.visual_dim > 0:
            visual_padding = np.zeros((60, self.visual_dim), dtype=np.float32)
            state_history = np.concatenate([state_history, visual_padding], axis=1)

        return {
            'state_history': state_history.astype(np.float32),
            'target_mode': target_mode_idx,
            'weight': 2.0  # VLM corrections are high priority
        }

    def _parse_tactician_correction(self, vlm_result: Dict, policy_output: Dict, state_vector: List, corr: Dict = None) -> Optional[Dict]:
        """Parse tactician correction - target selection."""
        # Check if target was incorrect
        if vlm_result.get('target_correct', True):
            return None

        recommended = vlm_result.get('recommended_target', {})
        if not recommended:
            return None

        # Get the full correction data with objects
        if not corr:
            return None

        objects = corr.get('objects', [])
        object_mask = corr.get('object_mask', [])
        object_track_ids = corr.get('object_track_ids', [])

        if not objects or not object_mask:
            return None

        # Find the recommended target in the objects array
        # The recommended target has a ByteTrack ID
        recommended_id = recommended.get('id', -1)
        if recommended_id < 0:
            return None

        # Match the recommended ID to an object using object_track_ids array
        target_idx = -1
        for idx, track_id in enumerate(object_track_ids):
            if track_id == recommended_id:
                target_idx = idx
                break

        # If we couldn't find the recommended target in current objects,
        # try to reconstruct it from the tracked_objects_snapshot
        if target_idx < 0:
            snapshot = corr.get('tracked_objects_snapshot', [])
            if not snapshot:
                return None  # No snapshot, can't recover

            # Find the recommended object in the snapshot
            recommended_obj = None
            for obj_data in snapshot:
                if obj_data.get('track_id') == recommended_id:
                    recommended_obj = obj_data
                    break

            if not recommended_obj:
                return None  # Not in snapshot either

            # Reconstruct objects array with the recommended target
            # Strategy: Replace the FIRST invalid object (mask=0) with the recommended target
            # Or if all valid, append to the end (will be truncated to 16 anyway)

            # Find first invalid slot or use len(objects)
            insert_idx = len([m for m in object_mask if m > 0])

            # Get feature vector from snapshot
            feature_vec = recommended_obj.get('feature_vector', [0] * 20)
            if len(feature_vec) < 20:
                feature_vec = feature_vec + [0] * (20 - len(feature_vec))
            elif len(feature_vec) > 20:
                feature_vec = feature_vec[:20]

            # Insert the object
            if insert_idx < len(objects):
                # Replace existing invalid entry
                objects[insert_idx] = feature_vec
                object_mask[insert_idx] = 1.0
                object_track_ids[insert_idx] = recommended_id
                target_idx = insert_idx
            else:
                # Append to end
                objects.append(feature_vec)
                object_mask.append(1.0)
                object_track_ids.append(recommended_id)
                target_idx = len(objects) - 1

        # Build goal from policy output (64-dim)
        goal_data = policy_output.get('goal', [])
        if isinstance(goal_data, list) and len(goal_data) >= 64:
            goal = np.array(goal_data[:64], dtype=np.float32)
        else:
            goal = np.zeros(64, dtype=np.float32)

        # Build approach vector from VLM recommendation
        approach_info = vlm_result.get('approach', {})
        approach = np.zeros(4, dtype=np.float32)

        # Parse approach hints from VLM
        # approach_dim is [vx, vy, urgency, aggression]
        # VLM provides: distance (too_close/good/too_far), angle (adjust_left/good/adjust_right), tactic
        distance = approach_info.get('distance', 'good')
        angle = approach_info.get('angle', 'good')
        tactic = approach_info.get('tactic', 'direct')

        # Convert VLM hints to approach vector
        # vx, vy: velocity suggestion
        if angle == 'adjust_left':
            approach[0] = -0.3  # Move left
        elif angle == 'adjust_right':
            approach[0] = 0.3   # Move right
        else:
            approach[0] = 0.0   # Good angle

        if distance == 'too_close':
            approach[1] = -0.3  # Back away
            approach[2] = 0.3   # Low urgency
        elif distance == 'too_far':
            approach[1] = 0.3   # Move closer
            approach[2] = 0.7   # Higher urgency
        else:
            approach[1] = 0.0   # Good distance
            approach[2] = 0.5   # Medium urgency

        # Aggression based on tactic
        if tactic == 'orbit':
            approach[3] = 0.5   # Medium aggression
        elif tactic == 'retreat':
            approach[3] = 0.2   # Low aggression
        else:  # direct/aggressive
            approach[3] = 0.8   # High aggression

        # Pad/truncate objects to max_objects
        max_objects = 16
        padded_objects = []
        for i in range(max_objects):
            if i < len(objects):
                obj = objects[i]
                if isinstance(obj, list):
                    # Pad or truncate to 20 dims
                    obj_arr = np.array(obj[:20], dtype=np.float32)
                    if len(obj_arr) < 20:
                        obj_arr = np.pad(obj_arr, (0, 20 - len(obj_arr)))
                    padded_objects.append(obj_arr)
                else:
                    padded_objects.append(np.zeros(20, dtype=np.float32))
            else:
                padded_objects.append(np.zeros(20, dtype=np.float32))

        objects_array = np.stack(padded_objects, axis=0)  # [max_objects, 20]

        # Pad/truncate mask
        mask_array = np.zeros(max_objects, dtype=np.float32)
        for i in range(min(len(object_mask), max_objects)):
            mask_array[i] = float(object_mask[i])

        return {
            'objects': objects_array.astype(np.float32),
            'mask': mask_array.astype(np.float32),
            'goal': goal.astype(np.float32),
            'target_idx': target_idx,
            'approach': approach.astype(np.float32),
            'weight': 2.0  # VLM corrections are high priority
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_executor(batch: List[Dict]) -> Tuple[torch.Tensor, ...]:
    """Collate function for executor samples."""
    states = torch.stack([torch.from_numpy(s['state']) for s in batch])
    goals = torch.stack([torch.from_numpy(s['goal']) for s in batch])
    target_infos = torch.stack([torch.from_numpy(s['target_info']) for s in batch])
    actions = torch.stack([torch.from_numpy(s['action']) for s in batch])
    weights = torch.tensor([s['weight'] for s in batch], dtype=torch.float32)

    return states, goals, target_infos, actions, weights


def collate_strategist(batch: List[Dict]) -> Tuple[torch.Tensor, ...]:
    """Collate function for strategist samples."""
    histories = torch.stack([torch.from_numpy(s['state_history']) for s in batch])
    targets = torch.tensor([s['target_mode'] for s in batch], dtype=torch.long)
    weights = torch.tensor([s['weight'] for s in batch], dtype=torch.float32)

    return histories, targets, weights


def collate_tactician(batch: List[Dict]) -> Tuple[torch.Tensor, ...]:
    """Collate function for tactician samples."""
    objects = torch.stack([torch.from_numpy(s['objects']) for s in batch])
    masks = torch.stack([torch.from_numpy(s['mask']) for s in batch])
    goals = torch.stack([torch.from_numpy(s['goal']) for s in batch])
    target_indices = torch.tensor([s['target_idx'] for s in batch], dtype=torch.long)
    approaches = torch.stack([torch.from_numpy(s['approach']) for s in batch])
    weights = torch.tensor([s['weight'] for s in batch], dtype=torch.float32)

    return objects, masks, goals, target_indices, approaches, weights


def finetune_executor(
    corrections_dir: str,
    pretrained_path: str,
    output_path: str,
    epochs: int = 20,
    lr: float = 1e-5,  # Lower LR for fine-tuning
    device: str = "cuda"
):
    """Fine-tune executor with VLM corrections."""
    print("\n" + "="*60)
    print("  V2 EXECUTOR FINE-TUNING WITH VLM CORRECTIONS")
    print("="*60)

    # Load pretrained model FIRST to detect if it's visual-enabled
    print("\n[1/4] Loading pretrained executor...")
    visual_model = False
    visual_dim = 0

    if Path(pretrained_path).exists():
        model = load_executor(pretrained_path, device=device)
        print(f"   Loaded from: {pretrained_path}")

        # Detect if this is a visual model by checking input dimension
        if hasattr(model, 'input_proj') and hasattr(model.input_proj[0], 'in_features'):
            input_dim = model.input_proj[0].in_features
            # Non-visual: 64+64+34=162, Visual: 64+64+98=226
            if input_dim > 200:
                visual_model = True
                visual_dim = 64  # Executor visual features are 64-dim
                print(f"   Detected VISUAL model (input_dim={input_dim}, visual_dim={visual_dim})")
            else:
                print(f"   Detected non-visual model (input_dim={input_dim})")
    else:
        print(f"   Warning: No pretrained model at {pretrained_path}")
        print("   Creating new model...")
        model = create_executor(ExecutorConfig(), device=device)

    # Load dataset with visual model info
    print("\n[2/4] Loading VLM corrections...")
    dataset = VLMCorrectionDataset(corrections_dir, component='executor',
                                    visual_model=visual_model, visual_dim=visual_dim)

    if len(dataset) == 0:
        print("   No valid executor corrections found!")
        return None

    if len(dataset) < 2:
        print(f"   Only {len(dataset)} executor correction(s) - need at least 2 for train/val split")
        print("   Skipping executor fine-tuning (collect more corrections first)")
        return None

    # Split train/val (ensure at least 1 sample in each)
    train_size = max(1, int(len(dataset) * 0.9))
    val_size = len(dataset) - train_size
    if val_size == 0:
        val_size = 1
        train_size = len(dataset) - 1
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_executor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_executor)

    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")

    model.train()

    # Detect ExecutorV2 for proper forward/loss path
    is_v2 = isinstance(model, ExecutorV2)
    if is_v2:
        loss_fn = ExecutorLossV2(
            use_beta=getattr(model, 'use_beta_distribution', True)
        )
        print(f"   Using ExecutorV2 loss (Beta NLL + Focal Loss)")
    else:
        print(f"   Using legacy loss (MSE + BCE)")

    # Optimizer with low LR for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    # TensorBoard logging
    from .training_utils import TrainingLogger
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TrainingLogger(
        log_dir="runs",
        experiment_name=f"finetune_executor_{timestamp}",
        use_tensorboard=True,
        log_to_file=True
    )

    # Training loop
    print("\n[3/4] Fine-tuning...")
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for state, goal, target_info, action, weight in pbar:
            state = state.to(device)
            goal = goal.to(device)
            target_info = target_info.to(device)
            action = action.to(device)
            weight = weight.to(device)

            optimizer.zero_grad()

            if is_v2:
                # ExecutorV2: forward returns dict, use ExecutorLossV2
                pred_dict = model.forward(state, goal, target_info)
                loss, metrics = loss_fn(pred_dict, action)
                # Apply sample weights
                loss = loss * weight.mean()
            else:
                # Legacy: forward returns (action_tensor, hidden)
                pred, _ = model(state, goal, target_info)

                # Weighted MSE loss for mouse position
                pred_mouse = torch.sigmoid(pred[:, :2])
                target_mouse = action[:, :2]
                mouse_loss = ((pred_mouse - target_mouse) ** 2).mean(dim=1)
                mouse_loss = (mouse_loss * weight).mean()

                # Weighted BCE loss for click
                pred_click = pred[:, 2]
                target_click = (action[:, 2] > 0).float()
                pos_weight = torch.tensor(20.0, device=device)
                click_loss = F.binary_cross_entropy_with_logits(
                    pred_click, target_click, pos_weight=pos_weight, reduction='none'
                )
                click_loss = (click_loss * weight).mean()

                loss = mouse_loss + 2.0 * click_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        train_loss /= max(num_batches, 1)

        # Validation
        model.eval()
        val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for state, goal, target_info, action, weight in val_loader:
                state = state.to(device)
                goal = goal.to(device)
                target_info = target_info.to(device)
                action = action.to(device)

                if is_v2:
                    pred_dict = model.forward(state, goal, target_info)
                    loss, _ = loss_fn(pred_dict, action)
                else:
                    pred, _ = model(state, goal, target_info)
                    pred_mouse = torch.sigmoid(pred[:, :2])
                    target_mouse = action[:, :2]
                    mouse_loss = ((pred_mouse - target_mouse) ** 2).mean()
                    pred_click = pred[:, 2]
                    target_click = (action[:, 2] > 0).float()
                    click_loss = F.binary_cross_entropy_with_logits(pred_click, target_click)
                    loss = mouse_loss + 0.5 * click_loss

                val_loss += loss.item()
                num_val_batches += 1

        val_loss /= max(num_val_batches, 1)

        # Log to TensorBoard
        logger.log_scalar("Loss/train", train_loss, epoch)
        logger.log_scalar("Loss/val", val_loss, epoch)
        logger.log_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"   [OK] New best model! Val loss: {best_val_loss:.4f}")

    # Load best and save
    if best_model_state:
        model.load_state_dict(best_model_state)

    print(f"\n[4/4] Saving to {output_path}...")
    save_executor(model, output_path)

    # Close logger
    logger.close()

    print("\n" + "="*60)
    print(f"  Fine-tuning complete! Best val loss: {best_val_loss:.4f}")
    print("="*60)

    return model


def finetune_strategist(
    corrections_dir: str,
    pretrained_path: str,
    output_path: str,
    epochs: int = 20,
    lr: float = 1e-5,
    device: str = "cuda"
):
    """Fine-tune strategist with VLM corrections."""
    print("\n" + "="*60)
    print("  V2 STRATEGIST FINE-TUNING WITH VLM CORRECTIONS")
    print("="*60)

    # Load pretrained model FIRST to detect if it's visual-enabled
    print("\n[1/4] Loading pretrained strategist...")
    visual_model = False
    visual_dim = 0

    if Path(pretrained_path).exists():
        model = load_strategist(pretrained_path, device=device)
        print(f"   Loaded from: {pretrained_path}")

        # Detect if this is a visual model by checking input dimension
        if hasattr(model, 'state_encoder') and hasattr(model.state_encoder[0], 'in_features'):
            input_dim = model.state_encoder[0].in_features
            # Non-visual: FULL_STATE_DIM (352), Visual: 352 + 512 = 864
            if input_dim > FULL_STATE_DIM + 100:
                visual_model = True
                visual_dim = 512  # Strategist visual features are 512-dim
                print(f"   Detected VISUAL model (input_dim={input_dim}, visual_dim={visual_dim})")
            else:
                print(f"   Detected non-visual model (input_dim={input_dim})")
    else:
        print(f"   Warning: No pretrained model at {pretrained_path}")
        model = create_strategist(StrategistConfig(), device=device)

    # Load dataset with visual model info
    print("\n[2/4] Loading VLM corrections...")
    dataset = VLMCorrectionDataset(corrections_dir, component='strategist',
                                    visual_model=visual_model, visual_dim=visual_dim)

    if len(dataset) == 0:
        print("   No valid strategist corrections found!")
        return None

    if len(dataset) < 2:
        print(f"   Only {len(dataset)} strategist correction(s) - need at least 2 for train/val split")
        print("   Skipping strategist fine-tuning (collect more corrections first)")
        return None

    # Split train/val (ensure at least 1 sample in each)
    train_size = max(1, int(len(dataset) * 0.9))
    val_size = len(dataset) - train_size
    if val_size == 0:
        val_size = 1
        train_size = len(dataset) - 1
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_strategist)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_strategist)

    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")

    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    # TensorBoard logging
    from .training_utils import TrainingLogger
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TrainingLogger(
        log_dir="runs",
        experiment_name=f"finetune_strategist_{timestamp}",
        use_tensorboard=True,
        log_to_file=True
    )

    print("\n[3/4] Fine-tuning...")
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for history, target_mode, weight in pbar:
            history = history.to(device)
            target_mode = target_mode.to(device)
            weight = weight.to(device)

            optimizer.zero_grad()

            # Forward - strategist outputs (goal, mode_logits, confidence)
            _, mode_logits, _ = model(history)

            # Weighted CE loss for mode (with label smoothing for +3% accuracy)
            loss = F.cross_entropy(mode_logits, target_mode, reduction='none', label_smoothing=0.1)
            loss = (loss * weight).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        train_loss /= max(num_batches, 1)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for history, target_mode, weight in val_loader:
                history = history.to(device)
                target_mode = target_mode.to(device)

                _, mode_logits, _ = model(history)
                loss = F.cross_entropy(mode_logits, target_mode, label_smoothing=0.1)
                val_loss += loss.item()

                pred_mode = mode_logits.argmax(dim=1)
                correct += (pred_mode == target_mode).sum().item()
                total += target_mode.size(0)

        val_loss /= max(len(val_loader), 1)
        accuracy = correct / max(total, 1) * 100

        # Log to TensorBoard
        logger.log_scalar("Loss/train", train_loss, epoch)
        logger.log_scalar("Loss/val", val_loss, epoch)
        logger.log_scalar("Accuracy/val", accuracy, epoch)
        logger.log_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {accuracy:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"   [OK] New best model!")

    if best_model_state:
        model.load_state_dict(best_model_state)

    print(f"\n[4/4] Saving to {output_path}...")
    save_strategist(model, output_path)

    # Close logger
    logger.close()

    print("\n" + "="*60)
    print(f"  Fine-tuning complete!")
    print("="*60)

    return model


def finetune_tactician(
    corrections_dir: str,
    pretrained_path: str,
    output_path: str,
    epochs: int = 20,
    lr: float = 1e-5,
    device: str = "cuda"
):
    """Fine-tune tactician with VLM corrections."""
    print("\n" + "="*60)
    print("  V2 TACTICIAN FINE-TUNING WITH VLM CORRECTIONS")
    print("="*60)

    # Load pretrained model
    print("\n[1/4] Loading pretrained tactician...")
    if Path(pretrained_path).exists():
        model = load_tactician(pretrained_path, device=device)
        print(f"   Loaded from: {pretrained_path}")
    else:
        print(f"   Warning: No pretrained model at {pretrained_path}")
        print("   Creating new model...")
        model = create_tactician(TacticianConfig(), device=device)

    # Load dataset
    print("\n[2/4] Loading VLM corrections...")
    dataset = VLMCorrectionDataset(corrections_dir, component='tactician')

    if len(dataset) == 0:
        print("   No valid tactician corrections found!")
        return None

    if len(dataset) < 2:
        print(f"   Only {len(dataset)} tactician correction(s) - need at least 2 for train/val split")
        print("   Skipping tactician fine-tuning (collect more corrections first)")
        return None

    # Split train/val (ensure at least 1 sample in each)
    train_size = max(1, int(len(dataset) * 0.9))
    val_size = len(dataset) - train_size
    if val_size == 0:
        val_size = 1
        train_size = len(dataset) - 1
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_tactician)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_tactician)

    # Setup training
    print(f"\n[3/4] Training for {epochs} epochs...")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Learning rate: {lr}")

    from .training_utils import TrainingLogger
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TrainingLogger(
        log_dir="runs",
        experiment_name=f"finetune_tactician_{timestamp}",
        use_tensorboard=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = TacticianLoss(target_weight=1.0, approach_weight=0.3, entropy_weight=0.1)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Train
        model.train()
        train_loss = 0
        train_metrics = {'target_loss': 0, 'approach_loss': 0, 'entropy': 0, 'total_loss': 0, 'accuracy': 0}
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training")
        for objects, masks, goals, target_indices, approaches, weights in pbar:
            objects = objects.to(device)
            masks = masks.to(device)
            goals = goals.to(device)
            target_indices = target_indices.to(device)
            approaches = approaches.to(device)
            weights = weights.to(device)

            optimizer.zero_grad()

            # Forward through tactician
            target_weights, pred_approach, _ = model(objects, masks, goals)

            # Compute loss
            loss, metrics = criterion(target_weights, pred_approach, target_indices, approaches)

            # Apply sample weights
            loss = (loss * weights.mean()).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            for k, v in metrics.items():
                train_metrics[k] += v
            num_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{metrics['accuracy']:.2f}"})

        scheduler.step()
        train_loss /= max(num_batches, 1)
        for k in train_metrics:
            train_metrics[k] /= max(num_batches, 1)

        # Validation
        model.eval()
        val_loss = 0
        val_metrics = {'target_loss': 0, 'approach_loss': 0, 'entropy': 0, 'total_loss': 0, 'accuracy': 0}
        num_val_batches = 0

        with torch.no_grad():
            for objects, masks, goals, target_indices, approaches, weights in val_loader:
                objects = objects.to(device)
                masks = masks.to(device)
                goals = goals.to(device)
                target_indices = target_indices.to(device)
                approaches = approaches.to(device)

                target_weights, pred_approach, _ = model(objects, masks, goals)
                loss, metrics = criterion(target_weights, pred_approach, target_indices, approaches)

                val_loss += loss.item()
                for k, v in metrics.items():
                    val_metrics[k] += v
                num_val_batches += 1

        val_loss /= max(num_val_batches, 1)
        for k in val_metrics:
            val_metrics[k] /= max(num_val_batches, 1)

        # Log to TensorBoard
        logger.log_scalar("Loss/train", train_loss, epoch)
        logger.log_scalar("Loss/val", val_loss, epoch)
        logger.log_scalar("Accuracy/train", train_metrics['accuracy'], epoch)
        logger.log_scalar("Accuracy/val", val_metrics['accuracy'], epoch)
        logger.log_scalar("TargetLoss/train", train_metrics['target_loss'], epoch)
        logger.log_scalar("TargetLoss/val", val_metrics['target_loss'], epoch)
        logger.log_scalar("ApproachLoss/train", train_metrics['approach_loss'], epoch)
        logger.log_scalar("ApproachLoss/val", val_metrics['approach_loss'], epoch)
        logger.log_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"   Train Acc: {train_metrics['accuracy']:.1f}% | Val Acc: {val_metrics['accuracy']:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"   [OK] New best model!")

    if best_model_state:
        model.load_state_dict(best_model_state)

    print(f"\n[4/4] Saving to {output_path}...")
    save_tactician(model, output_path)

    # Close logger
    logger.close()

    print("\n" + "="*60)
    print(f"  Fine-tuning complete!")
    print("="*60)

    return model


def preview_corrections(corrections_dir: str) -> dict:
    """
    Preview what types of corrections exist in a directory.
    Returns dict with counts for each component type.
    """
    from pathlib import Path
    import json

    corrections_path = Path(corrections_dir)
    if not corrections_path.exists():
        return {'executor': 0, 'strategist': 0, 'tactician': 0, 'total': 0}

    correction_files = list(corrections_path.glob("v2_corrections_*.json"))
    correction_types = {'executor': 0, 'strategist': 0, 'tactician': 0, 'total': 0}

    for file_path in correction_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            corrections = data.get('corrections', [])
            correction_types['total'] += len(corrections)

            for corr in corrections:
                vlm_result = corr.get('vlm_result', {})

                # Detect type of correction
                if 'current_mode_correct' in vlm_result or 'recommended_mode' in vlm_result:
                    correction_types['strategist'] += 1
                if 'quality' in vlm_result or 'correction' in vlm_result:
                    correction_types['executor'] += 1
                if 'target_correct' in vlm_result or 'recommended_target' in vlm_result:
                    correction_types['tactician'] += 1

        except Exception:
            pass

    return correction_types


def main():
    parser = argparse.ArgumentParser(description='Fine-tune V2 models with VLM corrections')
    parser.add_argument('--corrections', type=str, default='data/vlm_corrections_v2',
                       help='Directory containing VLM correction files')
    parser.add_argument('--component', type=str, default='executor',
                       choices=['executor', 'strategist', 'tactician', 'all'],
                       help='Which component to fine-tune')
    parser.add_argument('--pretrained-dir', type=str, default='v2/checkpoints',
                       help='Directory containing pretrained models')
    parser.add_argument('--output-dir', type=str, default='v2/checkpoints',
                       help='Directory to save fine-tuned models')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--preview', action='store_true',
                       help='Preview correction types and exit')

    args = parser.parse_args()

    # Preview mode
    if args.preview:
        base_dir = Path(__file__).parent.parent.parent.parent  # Go up to repo root
        corrections_dir = base_dir / args.corrections
        counts = preview_corrections(str(corrections_dir))
        print(f"\nVLM Corrections Preview ({corrections_dir}):")
        print(f"  Total: {counts['total']} corrections")
        print(f"  Strategist (mode): {counts['strategist']}")
        print(f"  Executor (aim): {counts['executor']}")
        print(f"  Tactician (target): {counts['tactician']}")
        return

    # Resolve paths
    base_dir = Path(__file__).parent.parent.parent.parent  # Go up to repo root (f:\dev\bot)
    corrections_dir = base_dir / args.corrections
    pretrained_dir = base_dir / args.pretrained_dir
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.component in ['executor', 'all']:
        finetune_executor(
            corrections_dir=str(corrections_dir),
            pretrained_path=str(pretrained_dir / "executor" / "best_model.pt"),
            output_path=str(output_dir / "executor" / "best_model.pt"),
            epochs=args.epochs,
            lr=args.lr,
            device=args.device
        )

    if args.component in ['strategist', 'all']:
        finetune_strategist(
            corrections_dir=str(corrections_dir),
            pretrained_path=str(pretrained_dir / "strategist" / "best_model.pt"),
            output_path=str(output_dir / "strategist" / "best_model.pt"),
            epochs=args.epochs,
            lr=args.lr,
            device=args.device
        )

    if args.component in ['tactician', 'all']:
        finetune_tactician(
            corrections_dir=str(corrections_dir),
            pretrained_path=str(pretrained_dir / "tactician" / "best_model.pt"),
            output_path=str(output_dir / "tactician" / "best_model.pt"),
            epochs=args.epochs,
            lr=args.lr,
            device=args.device
        )

    print("\n[SUCCESS] Fine-tuning complete!")
    print(f"  Models saved directly to: {output_dir}")
    print("  Ready to use - no copying needed!")


if __name__ == "__main__":
    main()
