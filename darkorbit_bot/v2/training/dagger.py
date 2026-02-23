"""
DAgger (Dataset Aggregation) - Corrective Imitation Learning

Extends shadow training by detecting when human corrections differ from
bot predictions, recording these corrections with higher priority, and
using MC Dropout uncertainty estimation to identify where the bot is unsure.

Usage:
    dagger = DAggerTrainer(shadow_trainer, executor, device='cuda')

    # In bot play loop:
    if dagger.detect_correction(bot_action, human_action):
        dagger.record_correction(state, goal, target_info, human_action)

    # Periodically check uncertainty:
    uncertainty = dagger.estimate_uncertainty(state, goal, target_info)
"""

import torch
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DAggerTrainer:
    """
    DAgger corrective learning wrapper around ShadowTrainer.

    Detects when human corrections deviate from bot predictions and
    records these corrections with higher sampling weight (added N times
    to the demonstration buffer).

    Also provides MC Dropout uncertainty estimation.
    """

    def __init__(self,
                 shadow_trainer,
                 executor: torch.nn.Module,
                 device: str = "cuda",
                 correction_weight: int = 3,
                 mouse_threshold: float = 0.1,
                 click_mismatch_weight: int = 2,
                 uncertainty_samples: int = 5,
                 uncertainty_threshold: float = 0.05):
        """
        Args:
            shadow_trainer: ShadowTrainer instance (provides buffer + training)
            executor: The executor model (for uncertainty estimation)
            device: Torch device
            correction_weight: How many times to add a correction to the buffer
            mouse_threshold: Mouse distance threshold to detect correction
            click_mismatch_weight: Extra additions for click mismatch corrections
            uncertainty_samples: Number of MC Dropout forward passes
            uncertainty_threshold: Variance above which bot is "uncertain"
        """
        self.shadow_trainer = shadow_trainer
        self.executor = executor
        self.device = device
        self.correction_weight = correction_weight
        self.mouse_threshold = mouse_threshold
        self.click_mismatch_weight = click_mismatch_weight
        self.uncertainty_samples = uncertainty_samples
        self.uncertainty_threshold = uncertainty_threshold

        # Stats
        self.total_corrections = 0
        self.total_frames = 0
        self.correction_rate = 0.0

    def detect_correction(self, bot_action: Optional[Dict], human_action: Dict) -> bool:
        """
        Detect if human is correcting the bot's action.

        A correction is detected when:
        1. Mouse position differs by more than threshold
        2. Click state disagrees (human clicks when bot doesn't or vice versa)
        3. Any keyboard key disagrees

        Args:
            bot_action: Bot's predicted action dict (or None if no prediction)
            human_action: Human's actual action dict

        Returns:
            True if this is a correction
        """
        self.total_frames += 1

        if bot_action is None:
            return False

        # Mouse distance
        bot_x = bot_action.get('mouse_x', 0.5)
        bot_y = bot_action.get('mouse_y', 0.5)
        human_x = human_action.get('mouse_x', 0.5)
        human_y = human_action.get('mouse_y', 0.5)
        mouse_dist = np.sqrt((bot_x - human_x) ** 2 + (bot_y - human_y) ** 2)

        if mouse_dist > self.mouse_threshold:
            return True

        # Click mismatch
        bot_click = bot_action.get('should_click', False)
        human_click = human_action.get('should_click', False)
        if bot_click != human_click:
            return True

        # Keyboard mismatch (any key differs)
        bot_kb = bot_action.get('keyboard', {})
        human_kb = human_action.get('keyboard', {})
        for key in human_kb:
            if human_kb.get(key, False) != bot_kb.get(key, False):
                return True

        return False

    def record_correction(self,
                          state: np.ndarray,
                          goal: np.ndarray,
                          target_info: np.ndarray,
                          human_action: Dict,
                          bot_action: Optional[Dict] = None,
                          local_visual: Optional[np.ndarray] = None):
        """
        Record a human correction with higher priority in the demo buffer.

        The correction is added multiple times (correction_weight) to effectively
        upweight it during random sampling.

        Args:
            state: Current compact state [64] or frame-stacked state
            goal: Goal embedding [64]
            target_info: Target info [34]
            human_action: The human's corrective action
            bot_action: Bot's predicted action (for logging)
            local_visual: Optional visual features
        """
        from ..config import KEYBOARD_KEYS, NUM_KEYBOARD_KEYS

        self.total_corrections += 1
        if self.total_frames > 0:
            self.correction_rate = self.total_corrections / self.total_frames

        # Build action array matching shadow trainer format:
        # [mouse_x, mouse_y, click_logit, ...keyboard keys...]
        action = np.zeros(3 + NUM_KEYBOARD_KEYS, dtype=np.float32)
        action[0] = human_action.get('mouse_x', 0.5)
        action[1] = human_action.get('mouse_y', 0.5)
        action[2] = 3.0 if human_action.get('should_click', False) else -3.0

        keyboard = human_action.get('keyboard', {})
        for i, key_name in enumerate(KEYBOARD_KEYS):
            action[3 + i] = 3.0 if keyboard.get(key_name, False) else -3.0

        # Build demo dict
        demo = {
            'state': state.copy(),
            'goal': goal.copy(),
            'target_info': target_info.copy(),
            'action': action,
            'is_correction': True,
        }
        if local_visual is not None:
            demo['local_visual'] = local_visual.copy()

        # Determine weight: click mismatches get extra weight
        weight = self.correction_weight
        if bot_action is not None:
            bot_click = bot_action.get('should_click', False)
            human_click = human_action.get('should_click', False)
            if bot_click != human_click:
                weight += self.click_mismatch_weight

        # Add to buffer multiple times for higher sampling probability
        for _ in range(weight):
            self.shadow_trainer.buffer.add(demo)

        if self.total_corrections % 50 == 1:
            logger.info(
                f"[DAgger] Correction #{self.total_corrections} "
                f"(rate: {self.correction_rate:.1%}, weight: {weight}x)"
            )

    def estimate_uncertainty(self,
                             state: np.ndarray,
                             goal: np.ndarray,
                             target_info: np.ndarray,
                             visual_features: Optional[np.ndarray] = None) -> float:
        """
        Estimate action uncertainty via MC Dropout.

        Runs multiple forward passes with dropout enabled and measures
        variance of mouse predictions. High variance = uncertain = should
        defer to human.

        Args:
            state: Compact state (or frame-stacked)
            goal: Goal embedding
            target_info: Target info
            visual_features: Optional visual features

        Returns:
            Uncertainty score (average mouse variance across samples).
            Values > uncertainty_threshold suggest the bot is unsure.
        """
        # Enable dropout for MC sampling
        was_training = self.executor.training
        self.executor.train()

        mouse_preds = []

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            goal_t = torch.tensor(goal, dtype=torch.float32).unsqueeze(0).to(self.device)
            target_t = torch.tensor(target_info, dtype=torch.float32).unsqueeze(0).to(self.device)
            vis_t = None
            if visual_features is not None:
                vis_t = torch.tensor(visual_features, dtype=torch.float32).unsqueeze(0).to(self.device)

            for _ in range(self.uncertainty_samples):
                # Forward with dropout active (model is in train mode)
                from ..models.executor import ExecutorV2
                if isinstance(self.executor, ExecutorV2):
                    pred = self.executor.forward(state_t, goal_t, target_t,
                                                 visual_features=vis_t)
                    mouse_raw = pred['mouse'].cpu().numpy()[0]
                    if mouse_raw.ndim > 1:
                        mouse_raw = mouse_raw[0]  # Take first timestep
                    # Decode to [0,1] position
                    mx, my = self.executor._decode_mouse(mouse_raw)
                else:
                    action, _ = self.executor.forward(state_t, goal_t, target_t)
                    action_np = action.cpu().numpy()[0]
                    def sigmoid(x):
                        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
                    mx = float(sigmoid(action_np[0]))
                    my = float(sigmoid(action_np[1]))

                mouse_preds.append([mx, my])

        # Restore training mode
        if not was_training:
            self.executor.eval()

        mouse_preds = np.array(mouse_preds)  # [N, 2]
        variance = np.mean(np.var(mouse_preds, axis=0))

        return float(variance)

    def is_uncertain(self,
                     state: np.ndarray,
                     goal: np.ndarray,
                     target_info: np.ndarray,
                     visual_features: Optional[np.ndarray] = None) -> bool:
        """Convenience: check if bot is uncertain on this input."""
        return self.estimate_uncertainty(
            state, goal, target_info, visual_features
        ) > self.uncertainty_threshold

    def get_stats(self) -> Dict:
        """Get DAgger statistics."""
        return {
            'total_corrections': self.total_corrections,
            'total_frames': self.total_frames,
            'correction_rate': self.correction_rate,
            'uncertainty_threshold': self.uncertainty_threshold,
        }

    def reset_stats(self):
        """Reset correction counters (e.g., at start of new session)."""
        self.total_corrections = 0
        self.total_frames = 0
        self.correction_rate = 0.0
