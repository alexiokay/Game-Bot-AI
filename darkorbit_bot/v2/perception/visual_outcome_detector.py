"""
Visual Outcome Detector - Deep Learning-based Game Event Recognition

Instead of unreliable heuristics (object disappearance = kill), this module
learns to recognize visual patterns that indicate game outcomes:
- Explosions -> Kill
- Damage numbers -> Hit
- Loot sparkle -> Loot collected
- Screen flash -> Damage taken
- UI changes -> HP/shield changes

The model learns these patterns through experience with delayed reward signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import time

logger = logging.getLogger(__name__)


@dataclass
class OutcomeDetectorConfig:
    """Configuration for visual outcome detector."""
    # Input dimensions (from VisionEncoder)
    global_visual_dim: int = 512  # Global scene features
    roi_visual_dim: int = 128     # Per-object RoI features
    local_visual_dim: int = 64    # Local patch features

    # Architecture
    hidden_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.1

    # Temporal modeling
    history_length: int = 30       # ~0.5 seconds at 60fps
    temporal_hidden: int = 128

    # Output event types
    num_event_types: int = 8       # explosion, damage, loot, flash, etc.

    # Training
    learning_rate: float = 1e-4
    gradient_clip: float = 1.0

    # Detection thresholds - HIGH by default since untrained model outputs noise
    # These will be lowered once model is trained
    confidence_threshold: float = 0.85  # High threshold to suppress untrained noise

    # Minimum training samples before detection is enabled
    min_training_samples: int = 100


class TemporalConvBlock(nn.Module):
    """1D temporal convolution for detecting events in time series."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2
        )
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)  # Back to [B, T, C]
        x = self.norm(x)
        x = self.activation(x)
        return x


class VisualOutcomeDetector(nn.Module):
    """
    Neural network that detects game outcomes from visual features.

    Event Types:
    0: explosion_small - Small explosion (normal enemy death)
    1: explosion_large - Large explosion (boss/elite death)
    2: damage_dealt - Damage numbers appearing on enemy
    3: damage_taken - Screen flash/shake, red vignette
    4: loot_collected - Loot sparkle/pickup effect
    5: shield_effect - Shield activation/hit
    6: ability_effect - Special ability visual
    7: death_screen - Player death (full screen change)

    The model learns temporal patterns - e.g., enemy targeted + explosion = kill
    """

    def __init__(self, config: Optional[OutcomeDetectorConfig] = None):
        super().__init__()
        self.config = config or OutcomeDetectorConfig()

        # Encoders for different visual inputs
        total_input_dim = (
            self.config.global_visual_dim +
            self.config.local_visual_dim +
            32  # Additional state info (current target, player state, etc.)
        )

        self.input_encoder = nn.Sequential(
            nn.Linear(total_input_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout)
        )

        # Temporal convolutions for detecting events over time
        self.temporal_conv1 = TemporalConvBlock(
            self.config.hidden_dim,
            self.config.temporal_hidden,
            kernel_size=3  # Short-term patterns
        )
        self.temporal_conv2 = TemporalConvBlock(
            self.config.temporal_hidden,
            self.config.temporal_hidden,
            kernel_size=5  # Medium-term patterns
        )
        self.temporal_conv3 = TemporalConvBlock(
            self.config.temporal_hidden,
            self.config.temporal_hidden,
            kernel_size=7  # Longer patterns (explosions take time)
        )

        # Attention over temporal features
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.config.temporal_hidden,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True
        )

        # Event detection heads - one per event type
        self.event_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.config.temporal_hidden, 64),
                nn.GELU(),
                nn.Linear(64, 1)  # Logit for this event
            )
            for _ in range(self.config.num_event_types)
        ])

        # Object-specific event detection (for per-target events like kills)
        # Takes ROI features + temporal context
        self.target_event_encoder = nn.Sequential(
            nn.Linear(self.config.roi_visual_dim + self.config.temporal_hidden, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(self.config.dropout)
        )

        self.target_kill_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)  # Kill probability for this target
        )

        self.target_damage_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)  # Damage dealt probability
        )

        # Frame differencing features (learn what visual changes mean)
        self.diff_encoder = nn.Sequential(
            nn.Linear(self.config.global_visual_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64)
        )

        # Intensity estimation (how strong was the effect?)
        self.intensity_head = nn.Sequential(
            nn.Linear(self.config.temporal_hidden, 32),
            nn.GELU(),
            nn.Linear(32, self.config.num_event_types)  # Intensity per event
        )

    def forward(self,
                visual_history: torch.Tensor,
                state_history: torch.Tensor,
                current_roi_features: Optional[torch.Tensor] = None,
                target_idx: Optional[int] = None
                ) -> Dict[str, torch.Tensor]:
        """
        Detect game events from visual feature history.

        Args:
            visual_history: [B, T, global_dim + local_dim] visual features over time
            state_history: [B, T, 32] game state info (player HP, target, etc.)
            current_roi_features: [B, max_objects, roi_dim] current frame ROI features
            target_idx: Which object is currently targeted (for kill detection)

        Returns:
            Dict containing:
                'event_probs': [B, num_events] probability of each event occurring
                'event_intensities': [B, num_events] estimated intensity
                'target_kill_prob': [B] kill probability for targeted object
                'target_damage_prob': [B] damage probability for targeted object
        """
        B, T, _ = visual_history.shape

        # Combine visual and state features
        combined = torch.cat([visual_history, state_history], dim=-1)

        # Encode per-frame
        encoded = self.input_encoder(combined)  # [B, T, hidden]

        # Temporal convolutions at multiple scales
        t1 = self.temporal_conv1(encoded)
        t2 = self.temporal_conv2(t1)
        t3 = self.temporal_conv3(t2)

        # Combine scales
        temporal_features = t1 + t2 + t3  # [B, T, temporal_hidden]

        # Self-attention over time
        attended, _ = self.temporal_attention(
            temporal_features, temporal_features, temporal_features
        )

        # Take most recent frame's attended features for event detection
        current_features = attended[:, -1, :]  # [B, temporal_hidden]

        # Detect each event type
        event_logits = []
        for head in self.event_heads:
            logit = head(current_features)
            event_logits.append(logit)
        event_logits = torch.cat(event_logits, dim=-1)  # [B, num_events]
        event_probs = torch.sigmoid(event_logits)

        # Estimate intensity
        intensities = torch.sigmoid(self.intensity_head(current_features))

        # Target-specific detection
        target_kill_prob = torch.zeros(B, device=visual_history.device)
        target_damage_prob = torch.zeros(B, device=visual_history.device)

        if current_roi_features is not None and target_idx is not None and target_idx >= 0:
            # Get target's ROI features
            target_roi = current_roi_features[:, target_idx, :]  # [B, roi_dim]

            # Combine with temporal context
            target_combined = torch.cat([target_roi, current_features], dim=-1)
            target_encoded = self.target_event_encoder(target_combined)

            target_kill_prob = torch.sigmoid(self.target_kill_head(target_encoded)).squeeze(-1)
            target_damage_prob = torch.sigmoid(self.target_damage_head(target_encoded)).squeeze(-1)

        return {
            'event_probs': event_probs,
            'event_intensities': intensities,
            'target_kill_prob': target_kill_prob,
            'target_damage_prob': target_damage_prob,
            'temporal_features': current_features  # For debugging/analysis
        }


class VisualOutcomeTracker:
    """
    Tracks game outcomes using visual features and the learned detector.

    This replaces the heuristic-based OutcomeTracker with AI-based detection.
    """

    EVENT_NAMES = [
        'explosion_small', 'explosion_large', 'damage_dealt', 'damage_taken',
        'loot_collected', 'shield_effect', 'ability_effect', 'death_screen'
    ]

    def __init__(self,
                 config: Optional[OutcomeDetectorConfig] = None,
                 device: str = "cuda"):
        self.config = config or OutcomeDetectorConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize detector model
        self.detector = VisualOutcomeDetector(self.config).to(self.device)
        self.detector.eval()  # Start in eval mode

        # IMPORTANT: Detector is NOT trained by default - outputs will be random noise
        # Detection is disabled until model is loaded or trained
        self._is_trained = False
        self._training_samples = 0

        # History buffers
        self.visual_history = deque(maxlen=self.config.history_length)
        self.state_history = deque(maxlen=self.config.history_length)
        self.roi_history = deque(maxlen=self.config.history_length)

        # Detection results history (for credit assignment)
        self.detection_history: List[Dict] = []
        self.max_history = 1000

        # Current target tracking
        self.current_target_idx: int = -1

        # Reward accumulator
        self.reward_config = {
            'explosion_small': 2.0,    # Small kill
            'explosion_large': 5.0,    # Boss kill
            'damage_dealt': 0.5,       # Hit confirm
            'damage_taken': -0.3,      # Taking damage
            'loot_collected': 1.0,     # Got loot
            'shield_effect': 0.1,      # Shield hit (neutral)
            'ability_effect': 0.0,     # Ability used
            'death_screen': -5.0,      # Died
        }

        # Optimizer for online learning
        self.optimizer = torch.optim.AdamW(
            self.detector.parameters(),
            lr=self.config.learning_rate
        )

        # Statistics
        self.stats = {name: 0 for name in self.EVENT_NAMES}
        self.total_reward = 0.0

        # ═══════════════════════════════════════════════════════════
        # WEAK SUPERVISION - Auto-generate labels from game state
        # ═══════════════════════════════════════════════════════════
        self._weak_supervision_enabled = True
        self._prev_player_hp = 1.0
        self._prev_target_idx = -1
        self._target_tracking: Dict[int, Dict] = {}  # track_id -> {first_seen, last_seen, was_attacking}
        self._pending_labels: List[Dict] = []  # Labels waiting to be applied
        self._weak_label_buffer: deque = deque(maxlen=200)  # Buffer of (features, weak_label) pairs
        self._weak_train_interval = 50  # Train every N weak labels
        self._weak_train_batch_size = 16

        logger.info(f"VisualOutcomeTracker initialized on {self.device}")
        logger.info("Weak supervision ENABLED - will self-train from game state signals")

    def record_frame(self,
                     global_features: np.ndarray,
                     local_features: np.ndarray,
                     roi_features: np.ndarray,
                     player_hp: float,
                     target_idx: int,
                     is_attacking: bool,
                     timestamp: float) -> Dict:
        """
        Record a frame's visual features and detect events.

        Args:
            global_features: [512] global scene features
            local_features: [64] local patch features
            roi_features: [max_objects, 128] per-object features
            player_hp: Current player HP (0-1)
            target_idx: Currently targeted object index
            is_attacking: Whether currently attacking
            timestamp: Current time

        Returns:
            Dict with detected events and rewards
        """
        # Build visual feature vector
        visual = np.concatenate([global_features, local_features])

        # Build state vector
        state = np.zeros(32, dtype=np.float32)
        state[0] = player_hp
        state[1] = float(target_idx >= 0)
        state[2] = float(is_attacking)
        state[3] = float(target_idx) / 16.0 if target_idx >= 0 else 0.0
        # Add previous frame HP for delta
        if len(self.state_history) > 0:
            state[4] = self.state_history[-1][0]  # Previous HP
            state[5] = player_hp - state[4]  # HP delta

        # Store in history
        self.visual_history.append(visual)
        self.state_history.append(state)
        self.roi_history.append(roi_features)
        self.current_target_idx = target_idx

        # Need full history buffer before detection
        if len(self.visual_history) < self.config.history_length:
            return {
                'events': {},
                'reward': 0.0,
                'buffer_filling': True
            }

        # Run detection (only if trained)
        result = self._detect_events(roi_features, target_idx)
        result['timestamp'] = timestamp

        # Store for credit assignment
        self.detection_history.append(result)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)

        # ═══════════════════════════════════════════════════════════
        # WEAK SUPERVISION - Auto-train from game state signals
        # ═══════════════════════════════════════════════════════════
        if self._weak_supervision_enabled:
            self._process_weak_supervision(player_hp, target_idx, is_attacking)

        return result

    @torch.no_grad()
    def _detect_events(self,
                       current_roi: np.ndarray,
                       target_idx: int) -> Dict:
        """Run the neural network to detect events."""

        # IMPORTANT: If detector not trained, return empty results to avoid false positives
        if not self._is_trained:
            return {
                'events': {},
                'raw_probs': np.zeros(len(self.EVENT_NAMES), dtype=np.float32),
                'reward': 0.0,
                'target_kill_prob': 0.0,
                'target_damage_prob': 0.0,
                'not_trained': True
            }

        # Convert histories to tensors
        visual_tensor = torch.tensor(
            np.array(list(self.visual_history)),
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)  # [1, T, visual_dim]

        state_tensor = torch.tensor(
            np.array(list(self.state_history)),
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)  # [1, T, 32]

        roi_tensor = torch.tensor(
            current_roi,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)  # [1, max_objects, roi_dim]

        # Forward pass
        output = self.detector(
            visual_tensor,
            state_tensor,
            roi_tensor,
            target_idx
        )

        # Process results
        event_probs = output['event_probs'].cpu().numpy()[0]
        intensities = output['event_intensities'].cpu().numpy()[0]
        target_kill_prob = output['target_kill_prob'].cpu().numpy()[0]
        target_damage_prob = output['target_damage_prob'].cpu().numpy()[0]

        # Threshold events - use high threshold to reduce false positives
        threshold = self.config.confidence_threshold
        detected_events = {}
        reward = 0.0

        for i, (name, prob) in enumerate(zip(self.EVENT_NAMES, event_probs)):
            if prob > threshold:
                detected_events[name] = {
                    'probability': float(prob),
                    'intensity': float(intensities[i])
                }
                # Accumulate reward
                base_reward = self.reward_config.get(name, 0.0)
                reward += base_reward * intensities[i]
                self.stats[name] += 1

        # Target-specific events
        if target_kill_prob > threshold:
            detected_events['target_killed'] = {
                'probability': float(target_kill_prob),
                'target_idx': target_idx
            }
            reward += 3.0 * target_kill_prob  # Bonus for confirmed kill on target

        if target_damage_prob > threshold:
            detected_events['target_damaged'] = {
                'probability': float(target_damage_prob),
                'target_idx': target_idx
            }
            reward += 0.3 * target_damage_prob

        self.total_reward += reward

        return {
            'events': detected_events,
            'raw_probs': event_probs,
            'reward': reward,
            'target_kill_prob': float(target_kill_prob),
            'target_damage_prob': float(target_damage_prob)
        }

    def get_recent_reward(self, lookback_frames: int = 30) -> float:
        """Get total reward from recent frames."""
        if not self.detection_history:
            return 0.0

        recent = self.detection_history[-lookback_frames:]
        return sum(r['reward'] for r in recent)

    def _process_weak_supervision(self, player_hp: float, target_idx: int, is_attacking: bool):
        """
        Generate weak labels from game state signals and auto-train.

        This is the key to self-learning:
        - HP drop → damage_taken label
        - Target disappears while attacking → explosion/kill label
        - HP recovery → (negative damage_taken)
        - Death (HP=0) → death_screen label

        These are "weak" labels because they're noisy (e.g., target might fly off screen
        instead of dying), but with enough data the model learns the visual patterns.
        """
        weak_labels = np.zeros(len(self.EVENT_NAMES), dtype=np.float32)
        has_label = False

        # EVENT INDICES:
        # 0: explosion_small, 1: explosion_large, 2: damage_dealt, 3: damage_taken
        # 4: loot_collected, 5: shield_effect, 6: ability_effect, 7: death_screen

        # ─────────────────────────────────────────────────────────────
        # 1. DAMAGE TAKEN - HP dropped
        # ─────────────────────────────────────────────────────────────
        hp_delta = player_hp - self._prev_player_hp
        if hp_delta < -0.05:  # Took significant damage
            weak_labels[3] = min(1.0, abs(hp_delta) * 5)  # damage_taken
            has_label = True
            logger.debug(f"Weak label: damage_taken (hp_delta={hp_delta:.2f})")

        # ─────────────────────────────────────────────────────────────
        # 2. DEATH SCREEN - HP reached 0
        # ─────────────────────────────────────────────────────────────
        if player_hp < 0.01 and self._prev_player_hp > 0.1:
            weak_labels[7] = 1.0  # death_screen
            has_label = True
            logger.debug("Weak label: death_screen")

        # ─────────────────────────────────────────────────────────────
        # 3. KILL DETECTION - Target disappeared while attacking
        # ─────────────────────────────────────────────────────────────
        if self._prev_target_idx >= 0 and is_attacking:
            # Was attacking a target that's no longer the target
            if target_idx != self._prev_target_idx or target_idx < 0:
                # Target changed or lost - possible kill
                weak_labels[0] = 0.7  # explosion_small (uncertain)
                has_label = True
                logger.debug(f"Weak label: possible kill (target {self._prev_target_idx} -> {target_idx})")

        # ─────────────────────────────────────────────────────────────
        # 4. Store weak label with current visual features
        # ─────────────────────────────────────────────────────────────
        if has_label and len(self.visual_history) >= self.config.history_length:
            # Save visual history snapshot with weak label
            self._weak_label_buffer.append({
                'visual_history': np.array(list(self.visual_history)).copy(),
                'state_history': np.array(list(self.state_history)).copy(),
                'roi_features': np.array(list(self.roi_history)[-1]).copy(),
                'target_idx': target_idx,
                'event_labels': weak_labels.copy(),
                'confidence': 0.5  # Weak labels have lower confidence
            })

            # Periodically train on weak labels
            if len(self._weak_label_buffer) >= self._weak_train_interval:
                self._train_on_weak_labels()

        # ─────────────────────────────────────────────────────────────
        # 5. Also generate NEGATIVE examples (nothing happening)
        # ─────────────────────────────────────────────────────────────
        # Every ~30 frames with no events, add a "nothing" example
        if not has_label and len(self.visual_history) >= self.config.history_length:
            if len(self._weak_label_buffer) % 30 == 0:  # Every 30th frame
                self._weak_label_buffer.append({
                    'visual_history': np.array(list(self.visual_history)).copy(),
                    'state_history': np.array(list(self.state_history)).copy(),
                    'roi_features': np.array(list(self.roi_history)[-1]).copy(),
                    'target_idx': target_idx,
                    'event_labels': np.zeros(len(self.EVENT_NAMES), dtype=np.float32),
                    'confidence': 0.8  # More confident about "nothing happening"
                })

        # Update tracking
        self._prev_player_hp = player_hp
        self._prev_target_idx = target_idx

    def _train_on_weak_labels(self):
        """Train detector on accumulated weak labels."""
        if len(self._weak_label_buffer) < self._weak_train_batch_size:
            return

        # Sample a batch
        indices = np.random.choice(
            len(self._weak_label_buffer),
            min(self._weak_train_batch_size, len(self._weak_label_buffer)),
            replace=False
        )
        batch = [self._weak_label_buffer[i] for i in indices]

        # Convert to tensors
        visual_batch = torch.tensor(
            np.stack([d['visual_history'] for d in batch]),
            dtype=torch.float32
        ).to(self.device)

        state_batch = torch.tensor(
            np.stack([d['state_history'] for d in batch]),
            dtype=torch.float32
        ).to(self.device)

        roi_batch = torch.tensor(
            np.stack([d['roi_features'] for d in batch]),
            dtype=torch.float32
        ).to(self.device)

        labels = torch.tensor(
            np.stack([d['event_labels'] for d in batch]),
            dtype=torch.float32
        ).to(self.device)

        confidences = torch.tensor(
            [d['confidence'] for d in batch],
            dtype=torch.float32
        ).to(self.device)

        # Training step with confidence weighting
        self.detector.train()
        self.optimizer.zero_grad()

        output = self.detector(visual_batch, state_batch, roi_batch, None)
        event_probs = output['event_probs']

        # Weighted BCE loss (lower weight for uncertain labels)
        loss = F.binary_cross_entropy(event_probs, labels, reduction='none')
        loss = (loss * confidences.unsqueeze(1)).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.detector.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        self.detector.eval()

        # Track training
        self._training_samples += len(batch)

        # Enable detection after enough training
        if not self._is_trained and self._training_samples >= self.config.min_training_samples:
            self._is_trained = True
            logger.info(f"Visual detector ENABLED after {self._training_samples} weak supervision samples!")
            print(f"\n   [VISUAL] ✅ Self-training complete! Detection ENABLED after {self._training_samples} samples")

        # (deque auto-trims to maxlen=200)

        # Periodic logging
        if self._training_samples % 50 == 0:
            print(f"   [VISUAL] Self-training: {self._training_samples} samples, loss={loss.item():.4f}")

    def provide_ground_truth(self,
                             event_name: str,
                             occurred: bool,
                             frames_ago: int = 0):
        """
        Provide ground truth feedback for training.

        When we have external confirmation (e.g., from game log/memory),
        use this to train the detector.

        Args:
            event_name: Name of the event
            occurred: Whether it actually occurred
            frames_ago: How many frames ago it happened
        """
        if event_name not in self.EVENT_NAMES:
            logger.warning(f"Unknown event type: {event_name}")
            return

        event_idx = self.EVENT_NAMES.index(event_name)

        # Get the detection from that frame
        if frames_ago >= len(self.detection_history):
            return

        detection = self.detection_history[-(frames_ago + 1)]
        predicted_prob = detection['raw_probs'][event_idx]

        # Create training signal
        target = 1.0 if occurred else 0.0
        loss = F.binary_cross_entropy(
            torch.tensor([predicted_prob]),
            torch.tensor([target])
        )

        # Log for debugging
        logger.debug(f"Ground truth: {event_name}={occurred}, predicted={predicted_prob:.3f}, loss={loss:.4f}")

    def train_step(self,
                   visual_batch: torch.Tensor,
                   state_batch: torch.Tensor,
                   roi_batch: torch.Tensor,
                   target_idx_batch: torch.Tensor,
                   event_labels: torch.Tensor) -> float:
        """
        Training step with labeled data.

        Args:
            visual_batch: [B, T, visual_dim]
            state_batch: [B, T, 32]
            roi_batch: [B, max_objects, roi_dim]
            target_idx_batch: [B]
            event_labels: [B, num_events] ground truth

        Returns:
            Loss value
        """
        self.detector.train()
        self.optimizer.zero_grad()

        # Forward
        output = self.detector(
            visual_batch.to(self.device),
            state_batch.to(self.device),
            roi_batch.to(self.device),
            None  # Will handle per-batch target idx differently
        )

        # Event classification loss
        event_probs = output['event_probs']
        loss = F.binary_cross_entropy(
            event_probs,
            event_labels.to(self.device)
        )

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.detector.parameters(),
            self.config.gradient_clip
        )
        self.optimizer.step()

        self.detector.eval()

        # Track training progress - enable detection after sufficient training
        batch_size = visual_batch.shape[0]
        self._training_samples += batch_size
        if not self._is_trained and self._training_samples >= self.config.min_training_samples:
            self._is_trained = True
            logger.info(f"Visual detector ENABLED after {self._training_samples} training samples")

        return loss.item()

    def enable_detection(self, enable: bool = True):
        """Manually enable/disable detection (e.g., after loading trained weights)."""
        self._is_trained = enable
        if enable:
            logger.info("Visual detector manually ENABLED")
        else:
            logger.info("Visual detector manually DISABLED")

    def is_trained(self) -> bool:
        """Check if detector is trained and detection is enabled."""
        return self._is_trained

    def get_stats(self) -> Dict:
        """Get detection statistics."""
        return {
            'event_counts': dict(self.stats),
            'total_reward': self.total_reward,
            'frames_processed': len(self.detection_history),
            'history_buffer_size': len(self.visual_history),
            'is_trained': self._is_trained,
            'training_samples': self._training_samples,
            'weak_supervision': {
                'enabled': self._weak_supervision_enabled,
                'buffer_size': len(self._weak_label_buffer),
                'samples_collected': self._training_samples
            }
        }

    def reset(self):
        """Reset tracker state (call between episodes)."""
        self.visual_history.clear()
        self.state_history.clear()
        self.roi_history.clear()
        self.detection_history.clear()
        self.current_target_idx = -1
        # Reset weak supervision tracking for new episode
        self._prev_player_hp = 1.0
        self._prev_target_idx = -1
        # Don't reset stats or weak_label_buffer - those accumulate across episodes

    def save(self, path: str):
        """Save detector model."""
        torch.save({
            'model_state_dict': self.detector.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'stats': self.stats,
            'total_reward': self.total_reward,
            'training_samples': self._training_samples,
            'is_trained': self._is_trained
        }, path)
        logger.info(f"VisualOutcomeDetector saved to {path}")

    def load(self, path: str):
        """Load detector model."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.detector.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'stats' in checkpoint:
            self.stats = checkpoint['stats']
        if 'total_reward' in checkpoint:
            self.total_reward = checkpoint['total_reward']
        if 'training_samples' in checkpoint:
            self._training_samples = checkpoint['training_samples']

        # Enable detection since we loaded trained weights
        self._is_trained = checkpoint.get('is_trained', True)
        logger.info(f"VisualOutcomeDetector loaded from {path} (trained={self._is_trained})")


class VisualRewardSignal:
    """
    Generates reward signals from visual outcome detection for RL training.

    This bridges the VisualOutcomeTracker with the OnlineLearner.
    """

    def __init__(self,
                 outcome_tracker: VisualOutcomeTracker,
                 reward_scale: float = 1.0,
                 discount: float = 0.95):
        self.tracker = outcome_tracker
        self.reward_scale = reward_scale
        self.discount = discount

        # Credit assignment buffer
        self.action_buffer: List[Dict] = []
        self.max_buffer = 100

    def record_action(self, action: Dict, state_features: np.ndarray):
        """Record an action for credit assignment."""
        self.action_buffer.append({
            'action': action,
            'state': state_features.copy(),
            'timestamp': time.time(),
            'reward': 0.0  # Will be filled in by outcome detection
        })

        if len(self.action_buffer) > self.max_buffer:
            self.action_buffer.pop(0)

    def get_reward_for_transition(self,
                                  prev_state: np.ndarray,
                                  action: Dict,
                                  next_state: np.ndarray) -> float:
        """
        Get reward for a state-action-state transition.

        Uses visual detection results to compute reward.
        """
        # Get most recent detection result
        if not self.tracker.detection_history:
            return 0.0

        result = self.tracker.detection_history[-1]
        reward = result['reward'] * self.reward_scale

        return reward

    def assign_credit(self, outcome_reward: float, lookback: int = 30):
        """
        Assign credit to recent actions based on an outcome.

        Uses exponential decay - recent actions get more credit.
        """
        if not self.action_buffer:
            return

        recent_actions = self.action_buffer[-lookback:]

        # Exponential decay assignment
        for i, action_record in enumerate(reversed(recent_actions)):
            decay = self.discount ** i
            action_record['reward'] += outcome_reward * decay

    def get_training_batch(self) -> Optional[List[Dict]]:
        """Get actions with assigned rewards for training."""
        if len(self.action_buffer) < 10:
            return None

        # Return actions that have had time to receive rewards
        batch = self.action_buffer[:-10]  # Exclude most recent (waiting for outcomes)
        self.action_buffer = self.action_buffer[-50:]  # Keep recent

        return batch if batch else None


def create_visual_outcome_tracker(
    config: Optional[OutcomeDetectorConfig] = None,
    device: str = "cuda"
) -> VisualOutcomeTracker:
    """Factory function to create VisualOutcomeTracker."""
    return VisualOutcomeTracker(config, device)
