"""
V2 Strategist Training - Goal and Mode Selection

Trains the Strategist to select goals and modes based on long-term game state.
Uses a Transformer to process 60 seconds of history and output:
- Goal embedding (64-dim continuous vector)
- Mode (FIGHT, LOOT, FLEE, EXPLORE, CAUTIOUS)
- Confidence

Training data:
- state_history: [T, state_dim] past 60 seconds of state
- target_mode: Human's chosen mode
- goal_context: Contextual info about the goal

Usage:
    python train_strategist.py --data recordings/ --epochs 30 --batch-size 16
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

from ..models.strategist import create_strategist, save_strategist, Strategist
from ..config import StrategistConfig, TrainingConfig, FULL_STATE_DIM, CONTEXT_START


class StrategistDataset(Dataset):
    """
    Dataset for Strategist training.

    Processes human recordings to extract:
    - Long-term state history (60s)
    - Mode labels based on human behavior
    - Goal context from actions
    """

    def __init__(self, data_dir: str, history_length: int = 60, state_dim: int = FULL_STATE_DIM):
        """
        Args:
            data_dir: Directory containing recordings
            history_length: Number of timesteps in history (at 1Hz = seconds)
            state_dim: Dimension of state vector
        """
        self.data_dir = Path(data_dir)
        self.history_length = history_length
        self.state_dim = state_dim
        self.samples: List[Dict] = []

        # Mode mapping
        self.mode_map = {'FIGHT': 0, 'LOOT': 1, 'FLEE': 2, 'EXPLORE': 3, 'CAUTIOUS': 4}
        self.reverse_mode_map = {v: k for k, v in self.mode_map.items()}

        self._load_recordings()

    def _load_recordings(self):
        """Load all recordings (recursively) from JSON, PKL, and NPZ formats."""
        # JSON recordings
        json_files = list(self.data_dir.glob("**/*.json"))
        json_files = [f for f in json_files if 'sequence_' in f.name or 'recording' in f.name.lower() or 'shadow' in f.name.lower()]

        # PKL recordings (shadow training)
        pkl_files = list(self.data_dir.glob("**/*.pkl"))
        pkl_files = [f for f in pkl_files if 'shadow' in f.name.lower() or 'recording' in f.name.lower()]

        # NPZ recordings (legacy)
        npz_files = list(self.data_dir.glob("**/*.npz"))

        print(f"[Strategist] Found {len(json_files)} JSON, {len(pkl_files)} PKL, {len(npz_files)} NPZ recording files")

        for file_path in json_files:
            try:
                self._load_json(file_path)
            except Exception as e:
                print(f"   Warning: Could not load {file_path}: {e}")

        for file_path in pkl_files:
            try:
                self._load_pkl(file_path)
            except Exception as e:
                print(f"   Warning: Could not load {file_path}: {e}")

        for file_path in npz_files:
            try:
                self._load_npz(file_path)
            except Exception as e:
                print(f"   Warning: Could not load {file_path}: {e}")

        print(f"[Strategist] Loaded {len(self.samples)} training samples")

    def _load_npz(self, path: Path):
        """Load from numpy archive."""
        data = np.load(path, allow_pickle=True)

        states = data.get('states', data.get('state_sequences', None))
        modes = data.get('modes', None)
        actions = data.get('actions', None)
        detections = data.get('detections', None)

        if states is None:
            return

        n_frames = len(states)

        # Process at 1Hz (assuming original is 30Hz)
        sample_rate = 30  # Frames per second in recording
        downsample = sample_rate  # Take 1 frame per second

        for i in range(self.history_length, n_frames, downsample):
            # Get history window (60 samples at 1Hz = 60 seconds)
            history_indices = list(range(max(0, i - self.history_length * downsample),
                                        i, downsample))
            if len(history_indices) < self.history_length:
                # Pad with first state
                padding_needed = self.history_length - len(history_indices)
                history_indices = [history_indices[0]] * padding_needed + history_indices

            # Extract state history
            history = []
            for idx in history_indices[-self.history_length:]:
                state = states[idx]
                if len(state.shape) > 1:
                    state = state[-1] if state.shape[0] > 1 else state[0]
                state = self._pad_state(state)
                history.append(state)

            history = np.array(history, dtype=np.float32)

            # Auto-label mode from the last state in history window
            mode_idx = self._infer_mode_from_state(history[-1])

            self.samples.append({
                'history': history,
                'mode_idx': mode_idx,
                'mask': np.ones(self.history_length, dtype=np.float32)
            })

    def _load_json(self, path: Path):
        """Load from JSON recording."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Handle Shadow Training JSON format (from shadow_trainer.py)
        if 'demos' in data and 'metadata' in data:
            demos = data['demos']
            print(f"   Loading {len(demos)} demos from {path.name} (Shadow Training format)")

            samples_added = 0
            mode_counts = {}
            for demo in demos:
                state = demo.get('state')

                if state is None:
                    continue

                state = np.array(state, dtype=np.float32)
                state = self._pad_state(state)

                # Auto-label mode from game state context
                # This lets the strategist learn ALL 5 modes (including FLEE/EXPLORE)
                # even if the human only demonstrated FIGHT/LOOT/CAUTIOUS
                mode_idx = self._infer_mode_from_state(state)

                # Build pseudo-history by repeating this state
                history = np.tile(state, (self.history_length, 1))

                sample = {
                    'history': history,
                    'mode_idx': mode_idx,
                    'mask': np.ones(self.history_length, dtype=np.float32)
                }

                # Store global visual if available (tile to match history length)
                if 'global_visual' in demo and demo['global_visual'] is not None:
                    gv = np.array(demo['global_visual'], dtype=np.float32)
                    sample['visual_history'] = np.tile(gv, (self.history_length, 1))

                self.samples.append(sample)
                samples_added += 1

                mode_name = self.reverse_mode_map.get(mode_idx, '?')
                mode_counts[mode_name] = mode_counts.get(mode_name, 0) + 1

            n_visual = sum(1 for s in self.samples[-samples_added:] if 'visual_history' in s) if samples_added > 0 else 0
            print(f"   Added {samples_added} samples from {path.name}")
            print(f"   Auto-labeled modes: {mode_counts}")
            if n_visual > 0:
                print(f"   {n_visual}/{samples_added} samples have global visual features")
            return

        # Handle V2 native format (from recorder_v2.py) and V1 format (from filtered_recorder.py)
        # Both have 'states' and 'actions', V2 native also has 'objects' and 'object_masks'
        if 'states' in data and 'actions' in data:
            states = data['states']
            n_frames = len(states)

            print(f"   Loading {n_frames} frames from {path.name} (V2 Recorder format)")

            # Create multiple samples by sliding window through the sequence
            samples_added = 0
            mode_counts = {}
            stride = max(1, self.history_length // 4)  # Slide by 25% of history length

            for i in range(0, n_frames - self.history_length + 1, stride):
                # Get history window
                history = []
                for j in range(self.history_length):
                    frame_idx = i + j
                    if frame_idx < n_frames:
                        state = np.array(states[frame_idx], dtype=np.float32)
                    else:
                        state = np.array(states[-1], dtype=np.float32)
                    state = self._pad_state(state)
                    history.append(state)

                history = np.array(history, dtype=np.float32)

                # Auto-label mode from the last frame's state in this window
                mode_idx = self._infer_mode_from_state(history[-1])

                self.samples.append({
                    'history': history,
                    'mode_idx': mode_idx,
                    'mask': np.ones(self.history_length, dtype=np.float32)
                })
                samples_added += 1

                mode_name = self.reverse_mode_map.get(mode_idx, '?')
                mode_counts[mode_name] = mode_counts.get(mode_name, 0) + 1

            # If sequence is shorter than history_length, still create one sample with padding
            if samples_added == 0:
                history = []
                for i in range(self.history_length):
                    if i < n_frames:
                        state = np.array(states[i], dtype=np.float32)
                    else:
                        state = np.array(states[-1], dtype=np.float32)
                    state = self._pad_state(state)
                    history.append(state)

                history = np.array(history, dtype=np.float32)
                mode_idx = self._infer_mode_from_state(history[-1])

                self.samples.append({
                    'history': history,
                    'mode_idx': mode_idx,
                    'mask': np.ones(self.history_length, dtype=np.float32)
                })
                samples_added = 1

                mode_name = self.reverse_mode_map.get(mode_idx, '?')
                mode_counts[mode_name] = mode_counts.get(mode_name, 0) + 1

            print(f"   Added {samples_added} samples from {path.name}")
            print(f"   Auto-labeled modes: {mode_counts}")
            return

        # Fallback: Handle old frames format
        frames = data.get('frames', data.get('recordings', []))

        if len(frames) < self.history_length:
            return

        # Downsample to ~1Hz (assuming 30fps recordings)
        downsample = 30

        for i in range(self.history_length * downsample, len(frames), downsample):
            # Build history
            history = []
            for j in range(self.history_length):
                frame_idx = i - (self.history_length - j) * downsample
                frame_idx = max(0, min(frame_idx, len(frames) - 1))

                frame = frames[frame_idx]
                state = frame.get('state_vector', frame.get('state', []))
                state = np.array(state, dtype=np.float32)
                state = self._pad_state(state)
                history.append(state)

            history = np.array(history, dtype=np.float32)

            # Infer mode
            current_frame = frames[min(i, len(frames) - 1)]
            mode_str = current_frame.get('mode', 'EXPLORE')
            if mode_str in self.mode_map:
                mode_idx = self.mode_map[mode_str]
            else:
                mode_idx = self._infer_mode(
                    current_frame.get('detections'),
                    current_frame,
                    None
                )

            self.samples.append({
                'history': history,
                'mode_idx': mode_idx,
                'mask': np.ones(self.history_length, dtype=np.float32)
            })

    def _load_pkl(self, path: Path):
        """
        Load from shadow recording pickle file.

        Shadow recordings contain full hierarchical demos with:
        - state: Current state vector
        - human_mode: The mode the human was in
        - tracked_objects: List of tracked objects
        - frame: JPEG-compressed frame bytes (or raw numpy array for legacy)
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        demos = data.get('demos', [])
        metadata = data.get('metadata', {})
        frame_format = metadata.get('frame_format', None)

        print(f"   Loading {len(demos)} demos from {path.name}")

        # Shadow recordings are individual frames, not sequences
        # We need to build pseudo-histories from these individual frames
        # Strategy: Use the same demo N times to fill the history (simple but works)

        samples_added = 0
        mode_counts = {}
        for demo in demos:
            state = demo.get('state')

            if state is None:
                continue

            state = np.array(state, dtype=np.float32)
            state = self._pad_state(state)

            # Auto-label mode from game state context
            mode_idx = self._infer_mode_from_state(state)

            # Build pseudo-history by repeating this state
            history = np.tile(state, (self.history_length, 1))

            self.samples.append({
                'history': history,
                'mode_idx': mode_idx,
                'mask': np.ones(self.history_length, dtype=np.float32)
            })
            samples_added += 1

            mode_name = self.reverse_mode_map.get(mode_idx, '?')
            mode_counts[mode_name] = mode_counts.get(mode_name, 0) + 1

        print(f"   Added {samples_added} samples from {path.name}")
        print(f"   Auto-labeled modes: {mode_counts}")

    def _pad_state(self, state: np.ndarray) -> np.ndarray:
        """Pad or truncate state to state_dim."""
        if len(state) >= self.state_dim:
            return state[:self.state_dim].astype(np.float32)
        return np.pad(state, (0, self.state_dim - len(state))).astype(np.float32)

    def _infer_mode_from_state(self, state: np.ndarray) -> int:
        """
        Infer the correct mode purely from the state vector.

        This allows the strategist to learn ALL 5 modes without the human
        needing to explicitly demonstrate each one. The state vector contains
        enough information to determine what mode SHOULD be active.

        State vector layout (from state_encoder.py):
          Player [0:16]:
            [0-1] x, y position
            [2-3] vx/10, vy/10 velocity
            [4] hp (0-1)
            [5] hp_delta * 10
            [6] shield (0-1)
            [7] shield_delta * 10
            [8] is_attacking (0 or 1)
            [9] attack_duration / 10
            [10] last_damage_time / 5
            [11] reserved
            [12] idle_time / 5
            [13] map_x
            [14] map_y
            [15] boundary_distance

          Objects [16:CONTEXT_START]: NUM_OBJECTS_IN_FLAT_STATE objects * 20 features each

          Context [CONTEXT_START:CONTEXT_START+16]:
            [+0] num_enemies / 5
            [+1] num_loot / 5
            [+2] has_enemies (0 or 1)
            [+3] has_loot (0 or 1)
            [+4] nearest_enemy_dist
            [+5] nearest_loot_dist
            [+6] total_threat (normalized)
            [+7] loot_value (normalized)
            [+8-11] near_left, near_right, near_top, near_bottom
            [+12-15] reserved
        """
        if not isinstance(state, np.ndarray) or len(state) < FULL_STATE_DIM:
            return self.mode_map['EXPLORE']

        # Extract key features
        hp = state[4]
        hp_delta = state[5] / 10.0  # Undo the *10 scaling
        shield = state[6]
        is_attacking = state[8] > 0.5

        ctx = CONTEXT_START  # Context features start index
        has_enemies = state[ctx + 2] > 0.5 if len(state) > ctx + 2 else False
        has_loot = state[ctx + 3] > 0.5 if len(state) > ctx + 3 else False
        nearest_enemy_dist = state[ctx + 4] if len(state) > ctx + 4 else 1.0
        nearest_loot_dist = state[ctx + 5] if len(state) > ctx + 5 else 1.0

        # FLEE: Low HP or taking heavy damage with enemies around
        if has_enemies and (hp < 0.3 or (hp < 0.5 and hp_delta < -0.02)):
            return self.mode_map['FLEE']

        # FLEE: Very low shield and enemies close
        if has_enemies and shield < 0.15 and hp < 0.5 and nearest_enemy_dist < 0.3:
            return self.mode_map['FLEE']

        # FIGHT: Attacking enemies with decent health
        if has_enemies and is_attacking and hp > 0.3:
            return self.mode_map['FIGHT']

        # LOOT: Loot nearby, no enemies or enemies far away
        if has_loot and (not has_enemies or nearest_enemy_dist > 0.4):
            return self.mode_map['LOOT']

        # LOOT: Loot very close even with distant enemies
        if has_loot and nearest_loot_dist < 0.15:
            return self.mode_map['LOOT']

        # CAUTIOUS: Enemies nearby but not attacking (observing, waiting)
        if has_enemies and not is_attacking:
            return self.mode_map['CAUTIOUS']

        # CAUTIOUS: Enemies far but present, moderate health
        if has_enemies and hp < 0.6 and not is_attacking:
            return self.mode_map['CAUTIOUS']

        # EXPLORE: Nothing around, just moving
        return self.mode_map['EXPLORE']

    def _infer_mode(self, detections, action, state) -> int:
        """
        Legacy fallback: infer mode from detections/action/state.
        Prefer _infer_mode_from_state() when full state vector is available.
        """
        # If we have a full state vector, use the better method
        if state is not None and isinstance(state, np.ndarray) and len(state) >= FULL_STATE_DIM:
            return self._infer_mode_from_state(state)

        has_enemy = False
        has_loot = False
        is_clicking = False
        low_health = False

        if detections:
            for d in detections if isinstance(detections, list) else [detections]:
                cls = d.get('class_name', '') if isinstance(d, dict) else getattr(d, 'class_name', '')
                if cls in ['Devo', 'Lordakia', 'Mordon', 'Saimon', 'Sibelon', 'Struener', 'npc', 'enemy']:
                    has_enemy = True
                if cls in ['BonusBox', 'box', 'bonus_box']:
                    has_loot = True

        if action:
            if isinstance(action, dict):
                is_clicking = action.get('clicked', action.get('should_click', action.get('should_fire', False)))
            elif isinstance(action, (list, np.ndarray)) and len(action) > 2:
                is_clicking = action[2] > 0

        if state is not None:
            if isinstance(state, np.ndarray) and len(state) > 4:
                low_health = state[4] < 0.3

        if low_health and has_enemy:
            return self.mode_map['FLEE']
        elif has_enemy and is_clicking:
            return self.mode_map['FIGHT']
        elif has_loot and is_clicking:
            return self.mode_map['LOOT']
        elif has_enemy and not is_clicking:
            return self.mode_map['CAUTIOUS']
        else:
            return self.mode_map['EXPLORE']

    def __len__(self) -> int:
        return len(self.samples)

    def has_visual_features(self) -> bool:
        """Check if any samples have visual_history features."""
        return any('visual_history' in s for s in self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        if 'visual_history' in sample:
            visual_history = torch.from_numpy(sample['visual_history'])
        else:
            visual_history = torch.zeros(self.history_length, 512, dtype=torch.float32)

        return (
            torch.from_numpy(sample['history']),
            torch.from_numpy(sample['mask']),
            torch.tensor(sample['mode_idx'], dtype=torch.long),
            visual_history
        )


class StrategistLoss(nn.Module):
    """
    Loss for Strategist training.

    Combines:
    - Cross entropy for mode classification
    - Goal consistency loss (encourages smooth goals over time)
    - Confidence calibration
    """

    def __init__(self, mode_weight: float = 1.0, goal_weight: float = 0.3,
                 confidence_weight: float = 0.1):
        super().__init__()
        self.mode_weight = mode_weight
        self.goal_weight = goal_weight
        self.confidence_weight = confidence_weight

    def forward(self, mode_logits: torch.Tensor, goal: torch.Tensor,
                confidence: torch.Tensor, target_mode: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            mode_logits: [B, num_modes] mode classification logits
            goal: [B, goal_dim] goal embedding
            confidence: [B, 1] confidence score
            target_mode: [B] target mode indices

        Returns:
            total_loss: Combined loss
            metrics: Dict of individual losses
        """
        # Mode classification loss (with label smoothing for +3% accuracy)
        mode_loss = F.cross_entropy(mode_logits, target_mode, label_smoothing=0.1)

        # Goal regularization (encourage unit-norm goals)
        goal_norm = torch.norm(goal, dim=-1)
        goal_loss = F.mse_loss(goal_norm, torch.ones_like(goal_norm))

        # Confidence should match mode prediction correctness
        with torch.no_grad():
            pred_mode = mode_logits.argmax(dim=-1)
            correct = (pred_mode == target_mode).float()

        # Clamp confidence to avoid log(0) = -inf in BCE
        conf_clamped = confidence.squeeze(-1).clamp(1e-7, 1 - 1e-7)
        confidence_loss = F.binary_cross_entropy(conf_clamped, correct)

        # Total
        total_loss = (self.mode_weight * mode_loss +
                     self.goal_weight * goal_loss +
                     self.confidence_weight * confidence_loss)

        # Accuracy
        accuracy = correct.mean().item()

        metrics = {
            'mode_loss': mode_loss.item(),
            'goal_loss': goal_loss.item(),
            'confidence_loss': confidence_loss.item(),
            'total_loss': total_loss.item(),
            'accuracy': accuracy,
            'avg_confidence': confidence.mean().item()
        }

        return total_loss, metrics


def train_strategist(
    data_dir: str,
    output_path: str = "strategist.pt",
    config: Optional[StrategistConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    device: str = "cuda",
    pretrained_path: Optional[str] = None
) -> Strategist:
    """
    Train the Strategist model.

    Args:
        data_dir: Directory with training recordings
        output_path: Where to save the model
        config: Model configuration
        training_config: Training hyperparameters
        device: cuda or cpu
        pretrained_path: Path to pretrained model for fine-tuning (optional)
    """
    if config is None:
        config = StrategistConfig()
    if training_config is None:
        training_config = TrainingConfig()

    print("\n" + "="*60)
    print("  V2 STRATEGIST TRAINING")
    print("="*60)

    # Load data
    print("\n[1/4] Loading training data...")
    dataset = StrategistDataset(
        data_dir,
        history_length=config.history_seconds,  # history_seconds at 1Hz = history_length
        state_dim=config.state_dim
    )

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
    print("\n[2/4] Creating Strategist model...")
    if pretrained_path and Path(pretrained_path).exists():
        print(f"   Loading pretrained weights from: {pretrained_path}")
        from ..models.strategist import load_strategist
        model = load_strategist(pretrained_path, device=device)
        print("   ✅ Fine-tuning from pretrained model")
    else:
        model = create_strategist(config, device=device)
        if pretrained_path:
            print(f"   ⚠️ Pretrained path not found: {pretrained_path}")
            print("   Training from scratch instead")
    model.train()

    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = StrategistLoss(
        mode_weight=1.0,
        goal_weight=0.3,
        confidence_weight=0.1
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.strategist_epochs,
        eta_min=training_config.learning_rate * 0.01
    )

    # TensorBoard logging
    from .training_utils import TrainingLogger
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TrainingLogger(
        log_dir="runs",
        experiment_name=f"strategist_{timestamp}",
        use_tensorboard=True,
        log_to_file=True
    )

    # Training loop
    print("\n[3/4] Training...")
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(training_config.strategist_epochs):
        model.train()
        train_metrics = {'mode_loss': 0, 'accuracy': 0, 'total_loss': 0}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_config.strategist_epochs}")

        for batch_data in pbar:
            history, mask, target_mode, visual_history = batch_data
            history = history.to(device)
            mask = mask.to(device)
            target_mode = target_mode.to(device)
            visual_history = visual_history.to(device)

            optimizer.zero_grad()

            # Forward (use visual features if model supports it)
            if hasattr(model, 'forward_with_visual') and dataset.has_visual_features():
                goal, mode_logits, confidence = model.forward_with_visual(
                    history, visual_history, mask)
            else:
                goal, mode_logits, confidence = model(history, mask)

            # Loss
            loss, metrics = criterion(mode_logits, goal, confidence, target_mode)

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
        val_metrics = {'mode_loss': 0, 'accuracy': 0, 'total_loss': 0}
        num_val_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                history, mask, target_mode, visual_history = batch_data
                history = history.to(device)
                mask = mask.to(device)
                target_mode = target_mode.to(device)
                visual_history = visual_history.to(device)

                if hasattr(model, 'forward_with_visual') and dataset.has_visual_features():
                    goal, mode_logits, confidence = model.forward_with_visual(
                        history, visual_history, mask)
                else:
                    goal, mode_logits, confidence = model(history, mask)
                loss, metrics = criterion(mode_logits, goal, confidence, target_mode)

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
    save_strategist(model, output_path)

    # Close logger
    logger.close()

    print("\n" + "="*60)
    print(f"  Training complete! Best val loss: {best_val_loss:.4f}")
    print("="*60)

    return model


def main():
    parser = argparse.ArgumentParser(description='Train V2 Strategist')
    parser.add_argument('--data', type=str, default='darkorbit_bot/data/recordings_v2',
                       help='Directory containing training recordings (default: darkorbit_bot/data/recordings_v2)')
    parser.add_argument('--output', type=str, default='models/v2/strategist/best_model.pt',
                       help='Output model path (default: models/v2/strategist/best_model.pt)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained model for fine-tuning')

    args = parser.parse_args()

    config = StrategistConfig()
    training_config = TrainingConfig(
        strategist_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    train_strategist(
        data_dir=args.data,
        output_path=args.output,
        config=config,
        training_config=training_config,
        device=args.device,
        pretrained_path=args.pretrained
    )


if __name__ == "__main__":
    main()
