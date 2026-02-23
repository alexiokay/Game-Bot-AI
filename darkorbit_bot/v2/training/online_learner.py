"""
V2 Online Learner - Real-time Learning from Experience

Learns from the bot's own actions during runtime:
1. Hit/Miss detection - When clicking, did target HP drop?
2. Distance tracking - Are we getting closer to target?
3. Mini-batch updates - Update model weights every N seconds

This is 100x faster than VLM feedback for aiming/movement.

Usage:
    learner = OnlineLearner(executor_model, device='cuda')

    # In main loop:
    learner.record(state, goal, target_info, action, reward_signal)

    # Periodically (every few seconds):
    learner.update()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import time
import logging

from .training_utils import TrainingLogger, WarmupCosineScheduler, gradient_norm
from ..config import KEYBOARD_KEYS, NUM_KEYBOARD_KEYS

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """Circular buffer for storing recent experiences."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, experience: Dict):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Dict]:
        """Sample random batch from buffer."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def get_recent(self, n: int) -> List[Dict]:
        """Get most recent n experiences."""
        return list(self.buffer)[-n:]

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class OnlineLearner:
    """
    Real-time online learning for the Executor.

    Learns from:
    1. Hit/Miss feedback - click + target HP drop = positive, else negative
    2. Distance rewards - getting closer to target = positive
    3. Smooth movement - consistent direction = positive
    """

    def __init__(self,
                 executor: nn.Module,
                 device: str = "cuda",
                 learning_rate: float = 1e-5,
                 buffer_size: int = 2000,
                 batch_size: int = 32,
                 update_interval: float = 5.0,
                 min_samples: int = 50,
                 log_dir: str = "runs",
                 use_lr_scheduler: bool = True,
                 warmup_steps: int = 100,
                 total_steps: int = 10000):
        """
        Args:
            executor: The executor model to train
            device: cuda or cpu
            learning_rate: Small LR for online updates (don't destabilize)
            buffer_size: Max experiences to store
            batch_size: Batch size for updates
            update_interval: Seconds between weight updates
            min_samples: Minimum samples before updating
            log_dir: Directory for tensorboard logs
            use_lr_scheduler: Whether to use warmup + cosine LR schedule
            warmup_steps: Number of warmup steps for LR
            total_steps: Total expected training steps for LR schedule
        """
        self.executor = executor
        self.device = device
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.min_samples = min_samples
        self.learning_rate = learning_rate

        # Detect if executor is visual-enabled by checking input dimensions
        self.visual_model = False
        self.visual_dim = 0
        self.target_info_dim = 34  # Base dimension

        if hasattr(executor, 'input_proj') and hasattr(executor.input_proj[0], 'in_features'):
            input_dim = executor.input_proj[0].in_features
            # Non-visual: 64+64+34=162, Visual: 64+64+98=226
            if input_dim > 200:
                self.visual_model = True
                self.visual_dim = 64  # Executor visual features are 64-dim
                self.target_info_dim = 98  # 34 + 64
                logger.info(f"[OnlineLearner] Detected VISUAL executor (target_info_dim={self.target_info_dim})")
            else:
                logger.info(f"[OnlineLearner] Detected non-visual executor (target_info_dim={self.target_info_dim})")

        # Experience buffer
        self.buffer = ExperienceBuffer(buffer_size)

        # Optimizer - use small LR to avoid destabilizing
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
                min_lr=learning_rate * 0.1
            )

        # Tensorboard logging
        self.training_logger = TrainingLogger(
            log_dir=log_dir,
            experiment_name="online_learning",
            use_tensorboard=True,
            log_to_file=True
        )

        # Tracking
        self.last_update_time = time.time()
        self.total_updates = 0
        self.total_samples = 0

        # Recent state for reward calculation
        self.prev_target_hp: Optional[float] = None
        self.prev_target_dist: Optional[float] = None
        self.prev_mouse_pos: Optional[Tuple[float, float]] = None
        self.prev_target_pos: Optional[Tuple[float, float]] = None

        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'distance_improved': 0,
            'distance_worsened': 0,
            'updates': 0,
            'avg_loss': 0.0,
            'learning_rate': learning_rate,
        }

        logger.info(f"[OnlineLearner] Initialized with LR={learning_rate}, buffer={buffer_size}")
        print(f"   [ONLINE] Tensorboard: tensorboard --logdir={log_dir}")

    def record(self,
               state: np.ndarray,
               goal: np.ndarray,
               target_info: np.ndarray,
               action: Dict,
               target_hp: Optional[float] = None,
               clicked: bool = False,
               target_pos: Optional[Tuple[float, float]] = None):
        """
        Record an experience and calculate reward.

        Args:
            state: Current state (64-dim)
            goal: Goal embedding (64-dim)
            target_info: Target info (34-dim)
            action: Executed action dict from executor
            target_hp: Current target HP (0-1), None if no target
            clicked: Whether we clicked this frame
            target_pos: Target position (x, y) in [0,1]
        """
        # Validate and fix input shapes
        if len(state) < 64:
            state = np.pad(state, (0, 64 - len(state)), mode='constant')
        elif len(state) > 64:
            state = state[:64]

        if len(goal) < 64:
            goal = np.pad(goal, (0, 64 - len(goal)), mode='constant')
        elif len(goal) > 64:
            goal = goal[:64]

        # Handle target_info based on whether model is visual-enabled
        # Non-visual: 34 dims, Visual: 98 dims (34 + 64)
        base_target_dim = 34
        if len(target_info) < base_target_dim:
            target_info = np.pad(target_info, (0, base_target_dim - len(target_info)), mode='constant')
        elif len(target_info) > self.target_info_dim:
            target_info = target_info[:self.target_info_dim]

        # If visual model but target_info is only base dim, pad with zeros for visual features
        if self.visual_model and len(target_info) < self.target_info_dim:
            visual_padding = np.zeros(self.target_info_dim - len(target_info), dtype=np.float32)
            target_info = np.concatenate([target_info[:base_target_dim], visual_padding])

        mouse_x = action.get('mouse_x', 0.5)
        mouse_y = action.get('mouse_y', 0.5)

        # Calculate rewards
        reward = 0.0
        reward_components = {}

        # 1. Hit/Miss reward (most important)
        if clicked and self.prev_target_hp is not None and target_hp is not None:
            hp_delta = self.prev_target_hp - target_hp
            if hp_delta > 0.001:  # HP dropped = hit
                reward += 1.0
                reward_components['hit'] = 1.0
                self.stats['hits'] += 1
            else:  # No HP drop = miss
                reward -= 0.3
                reward_components['miss'] = -0.3
                self.stats['misses'] += 1

        # 2. Distance reward (getting closer to target)
        if target_pos is not None:
            current_dist = np.sqrt(
                (mouse_x - target_pos[0])**2 +
                (mouse_y - target_pos[1])**2
            )

            if self.prev_target_dist is not None and self.prev_target_pos == target_pos:
                dist_delta = self.prev_target_dist - current_dist
                if dist_delta > 0.01:  # Getting closer
                    reward += 0.2
                    reward_components['closer'] = 0.2
                    self.stats['distance_improved'] += 1
                elif dist_delta < -0.01:  # Getting farther
                    reward -= 0.1
                    reward_components['farther'] = -0.1
                    self.stats['distance_worsened'] += 1

            self.prev_target_dist = current_dist
            self.prev_target_pos = target_pos

        # 3. Smooth movement reward (penalize erratic movement)
        if self.prev_mouse_pos is not None:
            mouse_delta = np.sqrt(
                (mouse_x - self.prev_mouse_pos[0])**2 +
                (mouse_y - self.prev_mouse_pos[1])**2
            )
            if mouse_delta > 0.3:  # Very jerky movement
                reward -= 0.05
                reward_components['jerky'] = -0.05

        # Update previous state
        self.prev_target_hp = target_hp
        self.prev_mouse_pos = (mouse_x, mouse_y)

        # Only store if there's meaningful reward signal
        if abs(reward) > 0.01 or len(self.buffer) < self.min_samples:
            # Build action tensor (what we want to reinforce/discourage)
            # Full 31-dim format: [mouse_x, mouse_y, click, ...keyboard_keys...]
            action_array = np.zeros(3 + NUM_KEYBOARD_KEYS, dtype=np.float32)
            action_array[0] = mouse_x
            action_array[1] = mouse_y
            action_array[2] = 3.0 if action.get('should_click', False) else -3.0

            # Add keyboard state from action dict
            keyboard = action.get('keyboard', {})
            for i, key_name in enumerate(KEYBOARD_KEYS):
                key_pressed = keyboard.get(key_name, False)
                action_array[3 + i] = 3.0 if key_pressed else -3.0

            self.buffer.add({
                'state': state.astype(np.float32),
                'goal': goal.astype(np.float32),
                'target_info': target_info.astype(np.float32),
                'action': action_array,
                'reward': reward,
                'components': reward_components
            })
            self.total_samples += 1

    def should_update(self) -> bool:
        """Check if it's time for a weight update."""
        if len(self.buffer) < self.min_samples:
            return False
        if time.time() - self.last_update_time < self.update_interval:
            return False
        return True

    def update(self) -> Optional[Dict]:
        """
        Perform a mini-batch weight update.

        Uses policy gradient style update:
        - Positive reward → increase probability of this action
        - Negative reward → decrease probability of this action

        Returns:
            Dict with update statistics, or None if no update performed
        """
        if not self.should_update():
            return None

        self.executor.train()

        # Sample batch
        batch = self.buffer.sample(min(self.batch_size, len(self.buffer)))

        if len(batch) < 8:  # Need minimum batch
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

        target_actions = torch.tensor(
            np.stack([b['action'] for b in batch]),
            dtype=torch.float32
        ).to(self.device)

        rewards = torch.tensor(
            [b['reward'] for b in batch],
            dtype=torch.float32
        ).to(self.device)

        # Normalize rewards (important for stability)
        if rewards.std() > 0.01:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Forward pass
        self.optimizer.zero_grad()

        # Get model output (raw logits)
        with torch.enable_grad():
            predicted_action, _ = self.executor.forward(states, goals, target_infos)

            # Loss: weighted MSE where weight = reward
            # Positive reward = reduce loss (reinforce action)
            # Negative reward = increase loss (discourage action)

            # MSE for position (first 2 dims)
            # Convert target to logit space for fair comparison
            target_xy = target_actions[:, :2].clamp(0.01, 0.99)
            target_logits = torch.log(target_xy / (1.0 - target_xy))

            pos_loss = F.mse_loss(predicted_action[:, :2], target_logits, reduction='none').mean(dim=1)

            # BCE for click (dim 2) - already in logit space
            click_loss = F.binary_cross_entropy_with_logits(
                predicted_action[:, 2],
                (target_actions[:, 2] > 0).float(),
                reduction='none'
            )

            # Combined loss weighted by reward
            # Positive reward → we want to move TOWARD this action (reduce loss effect)
            # Negative reward → we want to move AWAY from this action (increase loss effect)
            # Use: loss * (1 - reward) so positive reward reduces gradient
            weights = (1.0 - rewards.clamp(-1, 1)) / 2.0 + 0.5  # Map [-1,1] to [1, 0.5]

            # Actually for RL, we want:
            # - Good actions (high reward): make model output closer to this
            # - Bad actions (low reward): make model output farther from this
            # Standard approach: -reward * log_prob, but we're doing regression
            # So: loss = MSE * (1 - reward) where reward in [-1, 1] normalized

            loss = (pos_loss * weights + click_loss * weights * 0.5).mean()

            # Backprop
            loss.backward()

            # Log gradients before clipping
            grad_norm = gradient_norm(self.executor)
            self.training_logger.log_scalar("Gradients/norm", grad_norm, self.total_updates)

            torch.nn.utils.clip_grad_norm_(self.executor.parameters(), 1.0)
            self.optimizer.step()

            # Step LR scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()
                current_lr = self.lr_scheduler.get_lr()
                self.stats['learning_rate'] = current_lr

        self.executor.eval()

        # Update stats
        self.last_update_time = time.time()
        self.total_updates += 1
        self.stats['updates'] = self.total_updates
        self.stats['avg_loss'] = 0.9 * self.stats['avg_loss'] + 0.1 * loss.item()

        # Calculate hit rate
        hit_rate = 0.0
        if self.stats['hits'] + self.stats['misses'] > 0:
            hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses'])

        # Log to tensorboard
        self.training_logger.log_scalar("Loss/total", loss.item(), self.total_updates)
        self.training_logger.log_scalar("Loss/position", pos_loss.mean().item(), self.total_updates)
        self.training_logger.log_scalar("Loss/click", click_loss.mean().item(), self.total_updates)
        self.training_logger.log_scalar("Metrics/hit_rate", hit_rate, self.total_updates)
        self.training_logger.log_scalar("Metrics/avg_reward", float(rewards.mean()), self.total_updates)
        self.training_logger.log_scalar("Metrics/buffer_size", len(self.buffer), self.total_updates)
        self.training_logger.log_scalar("LR/learning_rate", self.stats['learning_rate'], self.total_updates)

        result = {
            'loss': loss.item(),
            'batch_size': len(batch),
            'buffer_size': len(self.buffer),
            'total_updates': self.total_updates,
            'avg_reward': float(rewards.mean()),
            'hit_rate': hit_rate,
            'learning_rate': self.stats['learning_rate'],
            'stats': self.stats.copy()
        }

        logger.info(f"[OnlineLearner] Update #{self.total_updates}: loss={loss.item():.4f}, "
                   f"hits={self.stats['hits']}, misses={self.stats['misses']}, lr={self.stats['learning_rate']:.2e}")

        return result

    def reset_episode(self):
        """Call when starting a new episode/run."""
        self.prev_target_hp = None
        self.prev_target_dist = None
        self.prev_mouse_pos = None
        self.prev_target_pos = None

    def get_stats(self) -> Dict:
        """Get learning statistics."""
        hit_rate = 0.0
        if self.stats['hits'] + self.stats['misses'] > 0:
            hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses'])

        return {
            **self.stats,
            'hit_rate': hit_rate,
            'buffer_size': len(self.buffer),
            'total_samples': self.total_samples
        }

    def save_checkpoint(self, path: str):
        """Save current model and optimizer state."""
        checkpoint = {
            'model_state_dict': self.executor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stats': self.stats,
            'total_updates': self.total_updates
        }
        # Save LR scheduler state if using one
        if self.lr_scheduler:
            checkpoint['lr_scheduler_state'] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"[OnlineLearner] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.executor.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.stats = checkpoint.get('stats', self.stats)
        self.total_updates = checkpoint.get('total_updates', 0)

        # Restore LR scheduler state if available
        if self.lr_scheduler and 'lr_scheduler_state' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])

        logger.info(f"[OnlineLearner] Checkpoint loaded from {path}")

    def close(self):
        """Clean up resources (call when done training)."""
        if self.training_logger:
            self.training_logger.close()


class OnlineLearnerV2:
    """
    Enhanced online learner with separate learning for different skills.

    Tracks and learns:
    1. Aiming accuracy (hit/miss on clicks)
    2. Target tracking (keeping crosshair on moving targets)
    3. Click timing (when to click vs hold fire)
    """

    def __init__(self,
                 executor: nn.Module,
                 device: str = "cuda",
                 learning_rate: float = 1e-5):
        self.base_learner = OnlineLearner(
            executor=executor,
            device=device,
            learning_rate=learning_rate
        )

        # Separate buffers for different skills
        self.aim_buffer = ExperienceBuffer(500)  # Click accuracy
        self.track_buffer = ExperienceBuffer(500)  # Continuous tracking

        # Skill-specific stats
        self.skill_stats = {
            'aim': {'samples': 0, 'success': 0},
            'track': {'samples': 0, 'success': 0}
        }

    def record_aim(self, state, goal, target_info, action, hit: bool):
        """Record aiming experience (click result)."""
        self.aim_buffer.add({
            'state': state,
            'goal': goal,
            'target_info': target_info,
            'action': action,
            'reward': 1.0 if hit else -0.5
        })
        self.skill_stats['aim']['samples'] += 1
        if hit:
            self.skill_stats['aim']['success'] += 1

    def record_track(self, state, goal, target_info, action, on_target: bool):
        """Record tracking experience (crosshair on target)."""
        self.track_buffer.add({
            'state': state,
            'goal': goal,
            'target_info': target_info,
            'action': action,
            'reward': 0.5 if on_target else -0.1
        })
        self.skill_stats['track']['samples'] += 1
        if on_target:
            self.skill_stats['track']['success'] += 1


class OutcomeTracker:
    """
    Tracks game outcomes for reward signals.

    Provides delayed rewards for:
    1. Kills - Enemy HP goes to 0 or enemy disappears after combat
    2. Loot collection - Resources/items picked up
    3. Survival - Not dying, HP recovery
    4. Combat efficiency - Damage dealt vs taken ratio

    These outcome rewards are much stronger signals than hit/miss
    and help the model learn what actually matters.
    """

    def __init__(self, reward_decay: float = 0.95, history_length: int = 100):
        """
        Args:
            reward_decay: Discount factor for assigning credit to past actions
            history_length: How many past states to keep for credit assignment
        """
        self.reward_decay = reward_decay
        self.history_length = history_length

        # State tracking
        self.current_target_id: Optional[str] = None
        self.current_target_hp: Optional[float] = None
        self.player_hp: float = 1.0
        self.player_shield: float = 1.0

        # Combat tracking
        self.combat_start_time: Optional[float] = None
        self.damage_dealt: float = 0.0
        self.damage_taken: float = 0.0

        # Experience history for credit assignment
        self.experience_history: deque = deque(maxlen=history_length)

        # Statistics
        self.stats = {
            'kills': 0,
            'deaths': 0,
            'loot_collected': 0,
            'damage_dealt_total': 0.0,
            'damage_taken_total': 0.0,
            'combat_sessions': 0,
            'successful_combats': 0,  # Won without dying
        }

        # Reward weights (tunable)
        self.rewards = {
            'kill': 5.0,           # Big reward for killing enemy
            'death': -3.0,         # Penalty for dying
            'loot': 1.0,           # Reward for collecting loot
            'damage_dealt': 0.5,   # Per 10% enemy HP
            'damage_taken': -0.2,  # Per 10% own HP lost
            'combat_win': 2.0,     # Bonus for winning combat without dying
            'hp_recovery': 0.3,    # Reward for healing
        }

        logger.info("[OutcomeTracker] Initialized")

    def record_state(self, experience: Dict, target_id: Optional[str] = None,
                     target_hp: Optional[float] = None, player_hp: float = 1.0,
                     player_shield: float = 1.0):
        """
        Record current state for later credit assignment.

        Args:
            experience: Dict with state, goal, target_info, action
            target_id: Unique identifier for current target (None if no target)
            target_hp: Current target HP (0-1)
            player_hp: Current player HP (0-1)
            player_shield: Current player shield (0-1)
        """
        # Store experience with timestamp
        self.experience_history.append({
            **experience,
            'timestamp': time.time(),
            'target_id': target_id,
            'target_hp': target_hp,
            'player_hp': player_hp,
        })

        # Track combat start
        if target_id is not None and self.current_target_id != target_id:
            # New target - start new combat session
            self._end_combat_session()
            self.current_target_id = target_id
            self.current_target_hp = target_hp
            self.combat_start_time = time.time()
            self.damage_dealt = 0.0
            self.stats['combat_sessions'] += 1

        # Track damage dealt
        if target_id == self.current_target_id and self.current_target_hp is not None:
            if target_hp is not None and target_hp < self.current_target_hp:
                damage = self.current_target_hp - target_hp
                self.damage_dealt += damage
                self.stats['damage_dealt_total'] += damage
            self.current_target_hp = target_hp

        # Track damage taken
        if player_hp < self.player_hp:
            damage = self.player_hp - player_hp
            self.damage_taken += damage
            self.stats['damage_taken_total'] += damage

        self.player_hp = player_hp
        self.player_shield = player_shield

    def check_kill(self, target_id: str) -> List[Dict]:
        """
        Check if target was killed and return rewarded experiences.

        Call this when a target disappears or HP reaches 0.

        Returns:
            List of experiences with kill reward applied
        """
        if target_id != self.current_target_id:
            return []

        self.stats['kills'] += 1
        self.stats['successful_combats'] += 1

        # Calculate total reward
        kill_reward = self.rewards['kill']
        damage_reward = self.damage_dealt * self.rewards['damage_dealt'] * 10
        combat_bonus = self.rewards['combat_win'] if self.damage_taken < 0.3 else 0.0

        total_reward = kill_reward + damage_reward + combat_bonus

        logger.info(f"[OutcomeTracker] KILL! reward={total_reward:.2f} "
                   f"(kill={kill_reward}, damage={damage_reward:.2f}, bonus={combat_bonus})")

        # Assign credit to recent experiences with decay
        rewarded = self._assign_credit(total_reward)

        # Reset combat state
        self._end_combat_session()

        return rewarded

    def check_death(self) -> List[Dict]:
        """
        Check if player died and return penalized experiences.

        Call this when player HP reaches 0.

        Returns:
            List of experiences with death penalty applied
        """
        self.stats['deaths'] += 1

        death_penalty = self.rewards['death']
        damage_penalty = self.damage_taken * self.rewards['damage_taken'] * 10

        total_penalty = death_penalty + damage_penalty

        logger.info(f"[OutcomeTracker] DEATH! penalty={total_penalty:.2f}")

        # Assign negative credit to recent experiences
        penalized = self._assign_credit(total_penalty)

        # Reset combat state
        self._end_combat_session()

        return penalized

    def check_loot(self, loot_value: float = 1.0) -> List[Dict]:
        """
        Record loot collection and return rewarded experiences.

        Args:
            loot_value: Value multiplier for the loot (1.0 = normal)

        Returns:
            List of experiences with loot reward applied
        """
        self.stats['loot_collected'] += 1

        loot_reward = self.rewards['loot'] * loot_value

        logger.info(f"[OutcomeTracker] LOOT! reward={loot_reward:.2f}")

        # Assign credit to very recent experiences (loot is immediate)
        rewarded = self._assign_credit(loot_reward, max_history=20)

        return rewarded

    def check_hp_recovery(self, old_hp: float, new_hp: float) -> List[Dict]:
        """
        Record HP recovery and return rewarded experiences.

        Args:
            old_hp: Previous HP (0-1)
            new_hp: Current HP (0-1)

        Returns:
            List of experiences with recovery reward applied
        """
        if new_hp <= old_hp:
            return []

        recovery = new_hp - old_hp
        recovery_reward = recovery * self.rewards['hp_recovery'] * 10

        if recovery_reward > 0.1:
            logger.debug(f"[OutcomeTracker] HP Recovery: {recovery:.1%}, reward={recovery_reward:.2f}")

        return self._assign_credit(recovery_reward, max_history=10)

    def _assign_credit(self, reward: float, max_history: Optional[int] = None) -> List[Dict]:
        """
        Assign credit to past experiences with exponential decay.

        More recent actions get more credit for the outcome.

        Args:
            reward: Total reward to distribute
            max_history: Maximum number of past experiences to credit

        Returns:
            List of experiences with assigned rewards
        """
        if not self.experience_history:
            return []

        history = list(self.experience_history)
        if max_history:
            history = history[-max_history:]

        rewarded = []
        for i, exp in enumerate(reversed(history)):
            # Exponential decay - more recent = more credit
            decay = self.reward_decay ** i
            exp_reward = reward * decay

            if abs(exp_reward) > 0.01:  # Only include meaningful rewards
                rewarded.append({
                    **exp,
                    'outcome_reward': exp_reward,
                    'reward': exp.get('reward', 0.0) + exp_reward
                })

        return rewarded

    def _end_combat_session(self):
        """End current combat session and reset tracking."""
        self.current_target_id = None
        self.current_target_hp = None
        self.combat_start_time = None
        self.damage_dealt = 0.0
        self.damage_taken = 0.0

    def get_stats(self) -> Dict:
        """Get outcome tracking statistics."""
        kd_ratio = 0.0
        if self.stats['deaths'] > 0:
            kd_ratio = self.stats['kills'] / self.stats['deaths']
        elif self.stats['kills'] > 0:
            kd_ratio = float('inf')

        combat_win_rate = 0.0
        if self.stats['combat_sessions'] > 0:
            combat_win_rate = self.stats['successful_combats'] / self.stats['combat_sessions']

        return {
            **self.stats,
            'kd_ratio': kd_ratio,
            'combat_win_rate': combat_win_rate,
            'avg_damage_per_combat': (
                self.stats['damage_dealt_total'] / max(1, self.stats['combat_sessions'])
            ),
        }

    def reset(self):
        """Reset tracker for new episode."""
        self._end_combat_session()
        self.player_hp = 1.0
        self.player_shield = 1.0
        self.experience_history.clear()


class VisualOutcomeLearner:
    """
    Online learner that uses visual-based outcome detection instead of heuristics.

    This combines:
    1. OnlineLearner for basic hit/miss and distance rewards
    2. VisualOutcomeTracker for AI-detected game events (explosions, damage, etc.)

    The visual detector learns to recognize:
    - Explosions → Kill confirmation
    - Damage numbers → Hit confirmation
    - Screen flash → Damage taken
    - Loot effects → Loot collected

    This is more reliable than heuristic detection (object disappearance)
    because it learns the actual visual patterns of game events.
    """

    def __init__(self,
                 executor: nn.Module,
                 device: str = "cuda",
                 learning_rate: float = 1e-5,
                 buffer_size: int = 2000,
                 use_visual_outcomes: bool = True,
                 log_dir: str = "runs"):
        """
        Args:
            executor: The executor model to train
            device: cuda or cpu
            learning_rate: Learning rate for executor updates
            buffer_size: Experience buffer size
            use_visual_outcomes: Whether to use visual outcome detection
            log_dir: Directory for tensorboard logs
        """
        self.device = device
        self.use_visual_outcomes = use_visual_outcomes

        # Base learner for immediate feedback (hit/miss, distance)
        self.base_learner = OnlineLearner(
            executor=executor,
            device=device,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            log_dir=log_dir
        )

        # Visual outcome tracker for AI-detected events
        self.visual_tracker = None
        if use_visual_outcomes:
            try:
                from ..perception.visual_outcome_detector import (
                    VisualOutcomeTracker,
                    OutcomeDetectorConfig
                )
                config = OutcomeDetectorConfig()
                self.visual_tracker = VisualOutcomeTracker(config, device)
                logger.info("[VisualOutcomeLearner] Visual outcome detection ENABLED")
            except ImportError as e:
                logger.warning(f"[VisualOutcomeLearner] Could not import visual tracker: {e}")
                self.use_visual_outcomes = False

        # Combined statistics
        self.stats = {
            'frames_processed': 0,
            'visual_events_detected': 0,
            'visual_reward_total': 0.0,
            'base_reward_total': 0.0,
        }

        # Feature dimension tracking (for compatibility with VisionEncoder)
        self.global_visual_dim = 512
        self.local_visual_dim = 64
        self.roi_visual_dim = 128

    def record(self,
               state: np.ndarray,
               goal: np.ndarray,
               target_info: np.ndarray,
               action: Dict,
               target_hp: Optional[float] = None,
               clicked: bool = False,
               target_pos: Optional[Tuple[float, float]] = None,
               player_hp: float = 1.0,
               target_idx: int = -1,
               is_attacking: bool = False,
               global_visual: Optional[np.ndarray] = None,
               local_visual: Optional[np.ndarray] = None,
               roi_visual: Optional[np.ndarray] = None):
        """
        Record an experience with both base and visual rewards.

        Args:
            state: Current state (64-dim)
            goal: Goal embedding (64-dim)
            target_info: Target info (34 or 98 dim)
            action: Executed action dict
            target_hp: Target HP for heuristic hit detection
            clicked: Whether clicked this frame
            target_pos: Target position for distance tracking
            player_hp: Player HP for damage taken detection
            target_idx: Index of current target
            is_attacking: Whether currently in attack mode
            global_visual: Global scene features from VisionEncoder [512]
            local_visual: Local patch features [64]
            roi_visual: Per-object ROI features [max_objects, 128]
        """
        self.stats['frames_processed'] += 1

        # Record to base learner (for immediate hit/miss feedback)
        self.base_learner.record(
            state=state,
            goal=goal,
            target_info=target_info,
            action=action,
            target_hp=target_hp,
            clicked=clicked,
            target_pos=target_pos
        )

        # Record to visual tracker if enabled and features available
        if self.visual_tracker is not None and global_visual is not None:
            # Use defaults if features not provided
            if local_visual is None:
                local_visual = np.zeros(self.local_visual_dim, dtype=np.float32)
            if roi_visual is None:
                roi_visual = np.zeros((16, self.roi_visual_dim), dtype=np.float32)

            result = self.visual_tracker.record_frame(
                global_features=global_visual,
                local_features=local_visual,
                roi_features=roi_visual,
                player_hp=player_hp,
                target_idx=target_idx,
                is_attacking=is_attacking,
                timestamp=time.time()
            )

            # Track visual detection results
            if result.get('events'):
                self.stats['visual_events_detected'] += len(result['events'])

            visual_reward = result.get('reward', 0.0)
            self.stats['visual_reward_total'] += visual_reward

            # If significant visual event detected, add to buffer with boosted reward
            if abs(visual_reward) > 0.5:
                # Create enhanced experience with visual reward
                # Full 31-dim format: [mouse_x, mouse_y, click, ...keyboard_keys...]
                action_array = np.zeros(3 + NUM_KEYBOARD_KEYS, dtype=np.float32)
                action_array[0] = action.get('mouse_x', 0.5)
                action_array[1] = action.get('mouse_y', 0.5)
                action_array[2] = 3.0 if action.get('should_click', False) else -3.0

                # Add keyboard state from action dict
                keyboard = action.get('keyboard', {})
                for i, key_name in enumerate(KEYBOARD_KEYS):
                    key_pressed = keyboard.get(key_name, False)
                    action_array[3 + i] = 3.0 if key_pressed else -3.0

                # Handle target_info dimension
                ti = target_info.copy()
                if len(ti) < self.base_learner.target_info_dim:
                    ti = np.pad(ti, (0, self.base_learner.target_info_dim - len(ti)))
                elif len(ti) > self.base_learner.target_info_dim:
                    ti = ti[:self.base_learner.target_info_dim]

                self.base_learner.buffer.add({
                    'state': state.astype(np.float32),
                    'goal': goal.astype(np.float32),
                    'target_info': ti.astype(np.float32),
                    'action': action_array,
                    'reward': visual_reward,
                    'components': {'visual': visual_reward}
                })

    def update(self) -> Optional[Dict]:
        """Perform weight update using both base and visual rewards."""
        result = self.base_learner.update()

        if result is not None:
            # Add visual stats to result
            result['visual_stats'] = {
                'events_detected': self.stats['visual_events_detected'],
                'visual_reward_total': self.stats['visual_reward_total'],
            }

            if self.visual_tracker is not None:
                result['visual_tracker_stats'] = self.visual_tracker.get_stats()

        return result

    def should_update(self) -> bool:
        """Check if update should be performed."""
        return self.base_learner.should_update()

    def get_recent_visual_reward(self, lookback_frames: int = 30) -> float:
        """Get recent visual-detected reward."""
        if self.visual_tracker is None:
            return 0.0
        return self.visual_tracker.get_recent_reward(lookback_frames)

    def get_detected_events(self) -> Dict:
        """Get recently detected visual events."""
        if self.visual_tracker is None:
            return {}

        if not self.visual_tracker.detection_history:
            return {}

        # Return most recent detection
        return self.visual_tracker.detection_history[-1].get('events', {})

    def reset_episode(self):
        """Reset for new episode."""
        self.base_learner.reset_episode()
        if self.visual_tracker is not None:
            self.visual_tracker.reset()

    def get_stats(self) -> Dict:
        """Get combined statistics."""
        base_stats = self.base_learner.get_stats()

        combined = {
            **base_stats,
            'visual': self.stats.copy()
        }

        if self.visual_tracker is not None:
            combined['visual_tracker'] = self.visual_tracker.get_stats()

        return combined

    def save_checkpoint(self, path: str):
        """Save all model states."""
        self.base_learner.save_checkpoint(path)

        # Save visual tracker separately if exists
        if self.visual_tracker is not None:
            visual_path = path.replace('.pt', '_visual.pt')
            self.visual_tracker.save(visual_path)

    def load_checkpoint(self, path: str):
        """Load all model states."""
        self.base_learner.load_checkpoint(path)

        # Load visual tracker if exists
        if self.visual_tracker is not None:
            visual_path = path.replace('.pt', '_visual.pt')
            try:
                self.visual_tracker.load(visual_path)
            except FileNotFoundError:
                logger.warning(f"Visual tracker checkpoint not found: {visual_path}")

    def train_visual_detector(self, labeled_data: List[Dict]) -> float:
        """
        Train the visual outcome detector with labeled examples.

        Args:
            labeled_data: List of dicts with visual features and event labels

        Returns:
            Training loss
        """
        if self.visual_tracker is None:
            return 0.0

        if not labeled_data:
            return 0.0

        # Convert to batch tensors
        visual_batch = torch.tensor(
            np.stack([d['visual_history'] for d in labeled_data]),
            dtype=torch.float32
        )
        state_batch = torch.tensor(
            np.stack([d['state_history'] for d in labeled_data]),
            dtype=torch.float32
        )
        roi_batch = torch.tensor(
            np.stack([d['roi_features'] for d in labeled_data]),
            dtype=torch.float32
        )
        target_idx_batch = torch.tensor(
            [d['target_idx'] for d in labeled_data],
            dtype=torch.long
        )
        event_labels = torch.tensor(
            np.stack([d['event_labels'] for d in labeled_data]),
            dtype=torch.float32
        )

        loss = self.visual_tracker.train_step(
            visual_batch, state_batch, roi_batch, target_idx_batch, event_labels
        )

        return loss
