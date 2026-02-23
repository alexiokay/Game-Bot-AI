"""
V2 Unified Hierarchical Policy

Combines Strategist, Tactician, and Executor into a single interface.
Manages the different update rates for each component.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from collections import deque
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .strategist import Strategist, create_strategist
from .tactician import Tactician, create_tactician
from .executor import Executor, ExecutorV2, create_executor


@dataclass
class HierarchicalState:
    """Current state of the hierarchical system."""
    # Strategist outputs (updated at 1Hz)
    goal: np.ndarray = None
    mode: str = "EXPLORE"
    mode_idx: int = 3
    mode_probs: np.ndarray = None
    strategist_confidence: float = 0.5
    last_strategist_update: float = 0.0

    # Tactician outputs (updated at 10Hz)
    target_idx: int = -1
    target_weights: np.ndarray = None
    approach: Dict = None
    target_info: np.ndarray = None  # 32-dim learned embedding from Tactician
    target_x: float = 0.5  # Target x position (normalized 0-1)
    target_y: float = 0.5  # Target y position (normalized 0-1)
    last_tactician_update: float = 0.0

    # Executor outputs (updated at 60Hz)
    action: Dict = None
    last_executor_update: float = 0.0


class HierarchicalPolicy(nn.Module):
    """
    Unified hierarchical policy combining all three layers.

    Manages timing:
    - Strategist: 1Hz (every 1000ms)
    - Tactician: 10Hz (every 100ms)
    - Executor: 60Hz (every frame)
    """

    def __init__(self,
                 strategist: Optional[Strategist] = None,
                 tactician: Optional[Tactician] = None,
                 executor: Optional[Executor] = None,
                 device: str = "cuda"):
        super().__init__()

        self.device = device

        # Create components if not provided
        self.strategist = strategist or create_strategist(device=device)
        self.tactician = tactician or create_tactician(device=device)
        self.executor = executor or create_executor(device=device)

        # Detect ExecutorV2
        self.is_executor_v2 = isinstance(self.executor, ExecutorV2)

        # Timing configuration (in seconds)
        self.strategist_interval = 1.0      # 1Hz
        self.tactician_interval = 0.1       # 10Hz

        # Current state
        self.state = HierarchicalState()

        # Default goal (used before first strategist update)
        self.default_goal = np.zeros(64, dtype=np.float32)
        self.default_target_info = np.zeros(34, dtype=np.float32)  # 34-dim target info for executor

        # Frame stacking ring buffer (for ExecutorV2)
        self._frame_stack_size = getattr(self.executor, 'frame_stack_size', 1) if self.is_executor_v2 else 1
        self._state_stack: deque = deque(maxlen=self._frame_stack_size)

        # Check if models support visual features
        self.strategist_has_visual = hasattr(self.strategist, 'visual_dim')
        self.tactician_has_visual = hasattr(self.tactician, 'visual_dim')
        self.executor_has_visual = hasattr(self.executor, 'visual_dim') and getattr(self.executor, 'visual_dim', 0) > 0

        # Visual feature dimensions (for zero-padding when visual not provided)
        self.strategist_visual_dim = getattr(self.strategist, 'visual_dim', 512)
        self.tactician_visual_dim = getattr(self.tactician, 'visual_dim', 128)
        self.executor_visual_dim = getattr(self.executor, 'visual_dim', 64)

    def step(self,
             state_history: np.ndarray,
             current_state: np.ndarray,
             objects: np.ndarray,
             object_mask: np.ndarray,
             player_x: float = 0.5,
             player_y: float = 0.5,
             force_strategist: bool = False,
             force_tactician: bool = False,
             visual_history: Optional[np.ndarray] = None,
             roi_visual: Optional[np.ndarray] = None,
             local_visual: Optional[np.ndarray] = None) -> Dict:
        """
        Single step through the hierarchy.

        Components are updated based on their intervals:
        - Strategist: if enough time has passed (or forced)
        - Tactician: if enough time has passed (or forced)
        - Executor: always (every frame)

        Args:
            state_history: [T, state_dim] for Strategist
            current_state: [state_dim] for Executor
            objects: [max_objects, object_dim] for Tactician
            object_mask: [max_objects] valid objects
            player_x, player_y: player position
            force_strategist: Force strategist update
            force_tactician: Force tactician update
            visual_history: [T, visual_dim] global visual features for Strategist
            roi_visual: [max_objects, visual_dim] per-object visual features for Tactician
            local_visual: [visual_dim] local visual features for Executor

        Returns:
            Dict with action and debug info
        """
        now = time.time()

        # === STRATEGIST (1Hz) ===
        if force_strategist or (now - self.state.last_strategist_update >= self.strategist_interval):
            self._update_strategist(state_history, visual_history)
            self.state.last_strategist_update = now

        # === TACTICIAN (10Hz) ===
        if force_tactician or (now - self.state.last_tactician_update >= self.tactician_interval):
            self._update_tactician(objects, object_mask, roi_visual)
            self.state.last_tactician_update = now

        # === EXECUTOR (60Hz - always) ===
        self._update_executor(current_state, objects, object_mask, local_visual)
        self.state.last_executor_update = now

        # Return action with debug info
        return {
            'action': self.state.action,
            'mode': self.state.mode,
            'target_idx': self.state.target_idx,
            'goal': self.state.goal,
            'confidence': self.state.strategist_confidence,
            'approach': self.state.approach
        }

    def _update_strategist(self, state_history: np.ndarray, visual_history: Optional[np.ndarray] = None):
        """Update strategist with new history."""
        # If visual model and visual features provided, concatenate
        if self.strategist_has_visual:
            T = state_history.shape[0]
            if visual_history is None:
                # Pad with zeros if no visual features
                visual_history = np.zeros((T, self.strategist_visual_dim), dtype=np.float32)
            # Concatenate state and visual
            combined_history = np.concatenate([state_history, visual_history], axis=-1)
            result = self.strategist.get_goal(combined_history, device=self.device)
        else:
            result = self.strategist.get_goal(state_history, device=self.device)

        self.state.goal = result['goal']
        self.state.mode = result['mode']
        self.state.mode_idx = result['mode_idx']
        self.state.mode_probs = result['mode_probs']
        self.state.strategist_confidence = result['confidence']

    def _update_tactician(self, objects: np.ndarray, object_mask: np.ndarray, roi_visual: Optional[np.ndarray] = None):
        """Update tactician with current objects."""
        # Use goal from strategist (or default)
        goal = self.state.goal if self.state.goal is not None else self.default_goal

        # If visual model, concatenate object and visual features
        if self.tactician_has_visual:
            N = objects.shape[0]
            if roi_visual is None:
                # Pad with zeros if no visual features
                roi_visual = np.zeros((N, self.tactician_visual_dim), dtype=np.float32)
            # Concatenate object and visual features
            combined_objects = np.concatenate([objects, roi_visual], axis=-1)
            result = self.tactician.get_target(combined_objects, object_mask, goal, device=self.device)
        else:
            result = self.tactician.get_target(objects, object_mask, goal, device=self.device)

        self.state.target_idx = result['target_idx']
        self.state.target_weights = result['target_weights']
        self.state.approach = result['approach']
        self.state.target_info = result['target_info']
        # Note: target_x, target_y are now extracted in _update_executor
        # along with velocity/size features for the 34-dim target_info

    def _update_executor(self, current_state: np.ndarray, objects: np.ndarray, object_mask: np.ndarray, local_visual: Optional[np.ndarray] = None):
        """
        Update executor with current state and target object features.

        Builds target_info matching the TRAINING format (34-dim for base):
        - [0-1]: Object x, y position
        - [2-3]: Object velocity x, y
        - [4]: Object speed
        - [5-6]: Object size (width, height)
        - [7]: Object confidence
        - [8]: Is enemy
        - [9]: Is loot
        - [10-11]: Aim offset (will be learned by model)
        - [12-31]: Additional object features
        - [32-33]: Reserved

        For ExecutorV2: visual features passed separately, frame-stacked state.
        For legacy Executor: visual features concatenated to target_info.
        """
        goal = self.state.goal if self.state.goal is not None else self.default_goal
        target_idx = self.state.target_idx

        # Build target_info from raw object features (matching training format)
        target_info = np.zeros(34, dtype=np.float32)

        # DEBUG: Track what objects we have
        valid_objects = int(np.sum(object_mask))

        if target_idx >= 0 and target_idx < len(objects) and object_mask[target_idx] > 0:
            obj = objects[target_idx]
            # Object feature layout from TrackedObject.to_feature_vector (20-dim):
            # [0-3]:   x, y, distance_to_player, angle_to_player (position)
            # [4-7]:   vx, vy, speed, heading (velocity)
            # [8-11]:  width, height, confidence, age_normalized (bbox)
            # [12-15]: hits_norm, time_since_update_norm, is_tracked, is_lost (tracking)
            # [16-19]: is_enemy, is_loot, is_player, is_other (class)

            target_info[0] = obj[0]   # x position
            target_info[1] = obj[1]   # y position
            target_info[2] = obj[4] if len(obj) > 4 else 0.0   # velocity x
            target_info[3] = obj[5] if len(obj) > 5 else 0.0   # velocity y
            target_info[4] = obj[6] if len(obj) > 6 else 0.0   # speed
            target_info[5] = obj[8] if len(obj) > 8 else 0.05  # width
            target_info[6] = obj[9] if len(obj) > 9 else 0.05  # height
            target_info[7] = obj[10] if len(obj) > 10 else 1.0 # confidence
            target_info[8] = obj[16] if len(obj) > 16 else 0.0 # is_enemy
            target_info[9] = obj[17] if len(obj) > 17 else 0.0 # is_loot
            # [10-11] aim offset - model will learn to predict this
            # Copy remaining object features as context
            for j in range(min(20, len(obj))):
                if 12 + j < 32:
                    target_info[12 + j] = obj[j]
        else:
            # No valid target - use center position, zero features
            target_info[0] = 0.5  # x = center
            target_info[1] = 0.5  # y = center

        # Also store target position for diagnostics
        self.state.target_x = target_info[0]
        self.state.target_y = target_info[1]

        # DEBUG: Log target selection every 30 frames
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0

        if self._debug_counter % 30 == 0:
            print(f"   [UNIFIED-DEBUG] Valid objects: {valid_objects} | "
                  f"Target idx: {target_idx} | Target pos: ({target_info[0]:.3f}, {target_info[1]:.3f})")

        # Compact state for executor (64 dims)
        state_compact = current_state[:64] if len(current_state) > 64 else current_state

        if self.is_executor_v2:
            # === ExecutorV2 path: frame stacking + separate visual ===
            # Push current compact state to frame stack
            self._state_stack.append(state_compact.copy())

            # Build frame-stacked state: [state_dim * frame_stack_size]
            if len(self._state_stack) < self._frame_stack_size:
                # Pad with copies of first frame until we have enough
                frames = []
                for _ in range(self._frame_stack_size - len(self._state_stack)):
                    frames.append(self._state_stack[0])
                frames.extend(self._state_stack)
            else:
                frames = list(self._state_stack)
            stacked_state = np.concatenate(frames, axis=0)  # [state_dim * frame_stack_size]

            # Visual features passed separately to ExecutorV2
            vis = local_visual if self.executor_has_visual else None

            action = self.executor.get_action(
                stacked_state, goal, target_info,
                device=self.device, visual_features=vis
            )
        else:
            # === Legacy Executor path: visual concatenated to target_info ===
            if self.executor_has_visual:
                if local_visual is None:
                    local_visual = np.zeros(self.executor_visual_dim, dtype=np.float32)
                target_info = np.concatenate([target_info, local_visual])

            action = self.executor.get_action(state_compact, goal, target_info, device=self.device)

        self.state.action = action

    def reset(self):
        """Reset all components (call at start of episode)."""
        self.executor.reset_hidden()
        self.state = HierarchicalState()
        self._state_stack.clear()

    def set_mode_override(self, mode: str):
        """
        Manually override the current mode.

        Useful for testing or when you want to force specific behavior.
        """
        mode_map = {
            'FIGHT': 0, 'LOOT': 1, 'FLEE': 2, 'EXPLORE': 3, 'CAUTIOUS': 4
        }
        if mode in mode_map:
            self.state.mode = mode
            self.state.mode_idx = mode_map[mode]

    def get_diagnostics(self) -> Dict:
        """Get diagnostic information for debugging."""
        return {
            'mode': self.state.mode,
            'mode_probs': self.state.mode_probs.tolist() if self.state.mode_probs is not None else None,
            'confidence': self.state.strategist_confidence,
            'target_idx': self.state.target_idx,
            'target_x': self.state.target_x,
            'target_y': self.state.target_y,
            'target_weights': self.state.target_weights.tolist() if self.state.target_weights is not None else None,
            'approach': self.state.approach,
            'action': self.state.action,
            'goal_magnitude': float(np.linalg.norm(self.state.goal)) if self.state.goal is not None else 0
        }


def create_hierarchical_policy(config=None, device: str = "cuda") -> HierarchicalPolicy:
    """
    Create a complete hierarchical policy.

    Args:
        config: V2Config (optional)
        device: Target device

    Returns:
        HierarchicalPolicy ready for use
    """
    if config is None:
        from ..config import V2Config
        config = V2Config()

    strategist = create_strategist(config.strategist, device=device)
    tactician = create_tactician(config.tactician, device=device)
    executor = create_executor(config.executor, device=device)

    return HierarchicalPolicy(
        strategist=strategist,
        tactician=tactician,
        executor=executor,
        device=device
    )


def save_hierarchical_policy(policy: HierarchicalPolicy, directory: str):
    """
    Save all components of the hierarchical policy.

    Saves three files:
    - {directory}/strategist.pt
    - {directory}/tactician.pt
    - {directory}/executor.pt
    """
    from pathlib import Path
    import os

    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    from .strategist import save_strategist
    from .tactician import save_tactician
    from .executor import save_executor

    save_strategist(policy.strategist, str(dir_path / "strategist.pt"))
    save_tactician(policy.tactician, str(dir_path / "tactician.pt"))
    save_executor(policy.executor, str(dir_path / "executor.pt"))

    print(f"[V2] Hierarchical policy saved to {directory}")


def load_hierarchical_policy(directory: str, device: str = "cuda") -> HierarchicalPolicy:
    """
    Load all components of the hierarchical policy.

    Args:
        directory: Directory containing the checkpoint files
        device: Target device

    Returns:
        Loaded HierarchicalPolicy
    """
    from pathlib import Path

    from .strategist import load_strategist
    from .tactician import load_tactician
    from .executor import load_executor

    dir_path = Path(directory)

    strategist = load_strategist(str(dir_path / "strategist" / "best_model.pt"), device=device)
    tactician = load_tactician(str(dir_path / "tactician" / "best_model.pt"), device=device)
    executor = load_executor(str(dir_path / "executor" / "best_model.pt"), device=device)

    return HierarchicalPolicy(
        strategist=strategist,
        tactician=tactician,
        executor=executor,
        device=device
    )
