"""
V2 Executor - Fast Motor Control

The Executor runs every frame (60fps) and outputs precise actions.
It doesn't need to understand strategy - it just executes:
"Move toward target X with urgency Y"

Two implementations:
1. Mamba (preferred) - State Space Model, O(1) inference
2. LSTM (fallback) - If Mamba not available

Both are ~3ms on RTX 3080.

Action output format:
- [0-1]: mouse_x, mouse_y (continuous, sigmoid applied)
- [2]: click (binary logit)
- [3:3+NUM_KEYBOARD_KEYS]: keyboard keys (binary logits, one per key in KEYBOARD_KEYS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from collections import deque
from typing import Dict, Tuple, Optional, List

from ..config import KEYBOARD_KEYS, NUM_KEYBOARD_KEYS, KEYBOARD_KEY_TO_IDX

logger = logging.getLogger(__name__)

# Try to import Mamba, fall back to LSTM if not available
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    logger.warning("Mamba not installed, using LSTM fallback")


class Executor(nn.Module):
    """
    Mamba-based Executor for fast motor control.

    Takes current state + goal + target info and outputs precise actions.
    Maintains hidden state across frames for smooth behavior.
    """

    def __init__(self,
                 state_dim: int = 64,
                 goal_dim: int = 64,
                 target_dim: int = 34,
                 hidden_dim: int = 256,
                 d_state: int = 64,
                 d_conv: int = 4,
                 expand: int = 2,
                 action_dim: int = 3 + NUM_KEYBOARD_KEYS):
        """
        Args:
            state_dim: Input state features
            goal_dim: Goal embedding from Strategist
            target_dim: Target info from Tactician
            hidden_dim: Mamba hidden size
            d_state: State space dimension
            d_conv: Convolution width
            expand: Expansion factor
            action_dim: Output actions [mouse_x, mouse_y, click_logit, ...keyboard_keys]
        """
        super().__init__()

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Input projection
        total_input = state_dim + goal_dim + target_dim
        self.input_proj = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Mamba backbone (if available)
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            self.use_mamba = True
        else:
            # Fallback to LSTM
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.1
            )
            self.use_mamba = False

        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # For LSTM: hidden state storage
        self.hidden = None

    def forward(self,
                state: torch.Tensor,
                goal: torch.Tensor,
                target_info: torch.Tensor,
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            state: [B, state_dim] current state
            goal: [B, goal_dim] goal embedding
            target_info: [B, target_dim] target information (first 2 dims are target x,y)
            hidden: Optional hidden state from previous frame

        Returns:
            action: [B, action_dim] raw action outputs (mouse_x, mouse_y are in logit space)
            hidden: Updated hidden state (for LSTM only)
        """
        # Concatenate inputs
        x = torch.cat([state, goal, target_info], dim=-1)
        x = self.input_proj(x)

        # Add sequence dimension
        x = x.unsqueeze(1)  # [B, 1, hidden_dim]

        if self.use_mamba:
            # Mamba forward (stateful)
            x = self.mamba(x)
            new_hidden = None
        else:
            # LSTM forward
            if hidden is None:
                x, (h, c) = self.lstm(x)
                new_hidden = (h, c)
            else:
                x, (h, c) = self.lstm(x, hidden)
                new_hidden = (h, c)

        # Remove sequence dimension
        x = x.squeeze(1)  # [B, hidden_dim]

        # Action output (raw logits for mouse position + click/ability)
        action = self.action_head(x)

        # RESIDUAL CONNECTION: Add target position to mouse output
        # This ensures even untrained model aims at target, training learns corrections
        # target_info[0:2] contains target x,y in [0,1] range
        # Convert to logit space: logit(p) = log(p/(1-p))
        target_xy = target_info[:, :2].clamp(0.01, 0.99)  # Avoid log(0)
        target_logits = torch.log(target_xy / (1.0 - target_xy))  # Inverse sigmoid

        # Add target position as bias to mouse output (first 2 dims)
        action = action.clone()
        action[:, 0] = action[:, 0] + target_logits[:, 0]
        action[:, 1] = action[:, 1] + target_logits[:, 1]

        # NO BIASES - Let the model learn WHEN to click/press keys from training data
        # The model must learn from demonstrations, not hardcoded rules
        #
        # Previously had click_bias +25.0 when has_valid_target
        # Now: Pure model output, no modifications

        # DEBUG: Log executor outputs every 30 frames
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1

        if self._debug_counter % 30 == 0:
            raw_x = torch.sigmoid(action[0, 0]).item()
            raw_y = torch.sigmoid(action[0, 1]).item()
            raw_click = action[0, 2].item()
            target_x = target_xy[0, 0].item()
            target_y = target_xy[0, 1].item()
            has_valid = (target_info[0, 0] != 0.5) | (target_info[0, 1] != 0.5)
            is_enemy_val = target_info[0, 8].item() if target_info.shape[1] > 8 else -1.0

            print(f"   [EXECUTOR-DEBUG] Raw output: pos({raw_x:.2f}, {raw_y:.2f}) click({raw_click:.2f}) | "
                  f"Target: ({target_x:.2f}, {target_y:.2f}) | "
                  f"Valid: {has_valid} | Enemy: {is_enemy_val:.2f}")

        return action, new_hidden if not self.use_mamba else None

    def get_action(self,
                   state: np.ndarray,
                   goal: np.ndarray,
                   target_info: np.ndarray,
                   device: str = "cuda") -> Dict:
        """
        Convenience method to get action from numpy arrays.

        Returns:
            Dict with:
                'mouse_x': float (0-1)
                'mouse_y': float (0-1)
                'should_click': bool
                'raw_click': float (logit)
                'keyboard': Dict[str, bool] - pressed state for each key in KEYBOARD_KEYS
        """
        self.eval()

        with torch.no_grad():
            # Convert to tensors
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            goal_t = torch.tensor(goal, dtype=torch.float32).unsqueeze(0).to(device)
            target_t = torch.tensor(target_info, dtype=torch.float32).unsqueeze(0).to(device)

            # Forward pass
            action, self.hidden = self.forward(state_t, goal_t, target_t, self.hidden)
            action = action.cpu().numpy()[0]

        # Use numpy sigmoid: 1 / (1 + exp(-x))
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

        # Parse action: [mouse_x, mouse_y, click, ...keyboard keys...]
        # NOTE: Using threshold > 1.0 for click too - requires confident output
        CLICK_THRESHOLD = 1.0
        result = {
            'mouse_x': float(np.clip(sigmoid(action[0]), 0, 1)),
            'mouse_y': float(np.clip(sigmoid(action[1]), 0, 1)),
            'should_click': bool(action[2] > CLICK_THRESHOLD),
            'raw_click': float(action[2]),
        }

        # Parse keyboard outputs (generic, not hardcoded)
        # Keys start at index 3, one per KEYBOARD_KEYS entry
        # NOTE: Using threshold > 1.0 instead of > 0.0 to prevent random key presses
        # from untrained/poorly-trained models. Well-trained models output ~+3.0 for
        # pressed keys and ~-3.0 for unpressed. Threshold 1.0 requires confident output.
        keyboard = {}
        keyboard_start_idx = 3
        KEYBOARD_THRESHOLD = 1.0  # Require confident positive output to press key
        for i, key_name in enumerate(KEYBOARD_KEYS):
            idx = keyboard_start_idx + i
            if idx < len(action):
                keyboard[key_name] = bool(action[idx] > KEYBOARD_THRESHOLD)
            else:
                keyboard[key_name] = False

        result['keyboard'] = keyboard

        return result

    def reset_hidden(self):
        """Reset hidden state (call when starting new episode)."""
        self.hidden = None


class ExecutorLSTM(nn.Module):
    """
    LSTM-based Executor (fallback if Mamba not available).

    Simpler architecture, still fast enough at ~3-5ms.
    """

    def __init__(self,
                 state_dim: int = 64,
                 goal_dim: int = 64,
                 target_dim: int = 34,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 action_dim: int = 3 + NUM_KEYBOARD_KEYS):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        total_input = state_dim + goal_dim + target_dim
        self.input_proj = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        # Output
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        self.hidden = None

    def forward(self, state, goal, target_info, hidden=None):
        x = torch.cat([state, goal, target_info], dim=-1)
        x = self.input_proj(x)
        x = x.unsqueeze(1)

        if hidden is None:
            x, (h, c) = self.lstm(x)
        else:
            x, (h, c) = self.lstm(x, hidden)

        x = x.squeeze(1)
        action = self.action_head(x)

        # RESIDUAL CONNECTION: Add target position to mouse output
        target_xy = target_info[:, :2].clamp(0.01, 0.99)
        target_logits = torch.log(target_xy / (1.0 - target_xy))
        action = action.clone()
        action[:, 0] = action[:, 0] + target_logits[:, 0]
        action[:, 1] = action[:, 1] + target_logits[:, 1]

        # DEBUG CHECK (Simple print for LSTM)
        # We can't access self._debug_counter easily here without init changes, 
        # so we'll just check a global probability or simple counter if needed,
        # or simplified: just print if we have a valid target but raw click is low
        has_valid_target = (target_info[:, 0] != 0.5) | (target_info[:, 1] != 0.5)

        # NO BIASES - Let the model learn WHEN to click/press keys from training data
        # The model must learn from demonstrations, not hardcoded rules
        #
        # Previously had:
        # - click_bias +12.0 when has_valid_target
        # - ammo_bias +8.0 when is_enemy (for click, ctrl, space, 1)
        #
        # Now: Pure model output, no modifications
        # The model will only click/press keys if it learned to do so from training data

        if np.random.random() < 0.01:  # Sample 1% of frames for debug
            raw_click = action[0, 2].item()
            raw_space = action[0, 6].item()
            raw_ctrl = action[0, 3].item()
            is_valid = has_valid_target[0].item()
            is_enemy_val = target_info[0, 8].item() if target_info.shape[1] > 8 else -1.0
            print(f"   [LSTM-DEBUG] Click:{raw_click:.1f} Ctrl:{raw_ctrl:.1f} Space:{raw_space:.1f} | Enemy:{is_enemy_val:.0f} Valid:{is_valid}")

        return action, (h, c)

    def get_action(self, state, goal, target_info, device="cuda"):
        """Get action from numpy arrays (same interface as Executor)."""
        self.eval()

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            goal_t = torch.tensor(goal, dtype=torch.float32).unsqueeze(0).to(device)
            target_t = torch.tensor(target_info, dtype=torch.float32).unsqueeze(0).to(device)

            action, self.hidden = self.forward(state_t, goal_t, target_t, self.hidden)
            action = action.cpu().numpy()[0]

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

        # Parse action: [mouse_x, mouse_y, click, ...keyboard keys...]
        # NO THRESHOLDS - Pure model output for natural learning
        # Model must learn from training data when to click/press keys

        # NOTE: Using threshold > 1.0 for click - requires confident output
        CLICK_THRESHOLD = 1.0
        result = {
            'mouse_x': float(np.clip(sigmoid(action[0]), 0, 1)),
            'mouse_y': float(np.clip(sigmoid(action[1]), 0, 1)),
            'should_click': bool(action[2] > CLICK_THRESHOLD),
            'raw_click': float(action[2]),
        }

        # Parse keyboard outputs (generic, not hardcoded)
        # NOTE: Using threshold > 1.0 instead of > 0.0 to prevent random key presses
        # from untrained/poorly-trained models. Well-trained models output ~+3.0 for
        # pressed keys and ~-3.0 for unpressed. Threshold 1.0 requires confident output.
        keyboard = {}
        keyboard_start_idx = 3
        KEYBOARD_THRESHOLD = 1.0  # Require confident positive output to press key
        for i, key_name in enumerate(KEYBOARD_KEYS):
            idx = keyboard_start_idx + i
            if idx < len(action):
                keyboard[key_name] = bool(action[idx] > KEYBOARD_THRESHOLD)
            else:
                keyboard[key_name] = False

        result['keyboard'] = keyboard

        return result

    def reset_hidden(self):
        self.hidden = None


class ExecutorWithVisual(Executor):
    """
    Executor with visual feature support.

    Target info changes from [34] to [98] (34 base + 64 visual).
    Visual features help with precision aiming (find gaps in shields, critical spots).
    """

    def __init__(self,
                 state_dim: int = 64,
                 goal_dim: int = 64,
                 target_dim: int = 34,
                 visual_dim: int = 64,
                 hidden_dim: int = 256,
                 d_state: int = 64,
                 d_conv: int = 4,
                 expand: int = 2,
                 action_dim: int = 3 + NUM_KEYBOARD_KEYS):
        # Initialize parent with combined target dimension
        total_target_dim = target_dim + visual_dim
        super().__init__(
            state_dim=state_dim,
            goal_dim=goal_dim,
            target_dim=total_target_dim,  # Combined dimension
            hidden_dim=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            action_dim=action_dim
        )

        # Store original dimensions
        self.base_target_dim = target_dim
        self.visual_dim = visual_dim

    def forward_with_visual(self,
                            state: torch.Tensor,
                            goal: torch.Tensor,
                            target_info: torch.Tensor,
                            visual_features: torch.Tensor,
                            hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with separate target and visual inputs.

        Args:
            state: [B, state_dim]
            goal: [B, goal_dim]
            target_info: [B, base_target_dim] position + learned features
            visual_features: [B, visual_dim] local precision features
            hidden: Optional hidden state

        Returns:
            Same as forward()
        """
        # Concatenate target info and visual features
        combined_target = torch.cat([target_info, visual_features], dim=-1)
        return self.forward(state, goal, combined_target, hidden)

    def forward_no_visual(self,
                          state: torch.Tensor,
                          goal: torch.Tensor,
                          target_info: torch.Tensor,
                          hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass without visual features (zero-padded).
        """
        B = state.shape[0]
        device = state.device

        visual_zeros = torch.zeros(B, self.visual_dim, device=device, dtype=state.dtype)
        combined_target = torch.cat([target_info, visual_zeros], dim=-1)
        return self.forward(state, goal, combined_target, hidden)


class ExecutorV2(nn.Module):
    """
    V2 Executor with separate action heads, Beta distribution mouse output,
    action chunking, frame stacking, and confidence-gated early exit.

    Improvements over Executor:
    1. Separate heads: mouse (Beta dist), click (BCE), keyboard (Focal Loss)
    2. Action chunking: predict next chunk_size frames, temporal ensembling
    3. Frame stacking: concatenate last N frames for motion context
    4. Confidence gate: skip backbone on easy frames (CTM-inspired)
    """

    def __init__(self,
                 state_dim: int = 64,
                 goal_dim: int = 64,
                 target_dim: int = 34,
                 hidden_dim: int = 256,
                 d_state: int = 64,
                 d_conv: int = 4,
                 expand: int = 2,
                 # V2 params
                 frame_stack_size: int = 3,
                 chunk_size: int = 8,
                 replan_every: int = 4,
                 temporal_ensemble_lambda: float = 0.01,
                 mouse_head_hidden: int = 128,
                 click_head_hidden: int = 64,
                 keyboard_head_hidden: int = 64,
                 num_keyboard_keys: int = NUM_KEYBOARD_KEYS,
                 use_beta_distribution: bool = True,
                 confidence_threshold: float = 0.8,
                 visual_dim: int = 0):
        super().__init__()

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.frame_stack_size = frame_stack_size
        self.chunk_size = chunk_size
        self.replan_every = replan_every
        self.temporal_ensemble_lambda = temporal_ensemble_lambda
        self.num_keyboard_keys = num_keyboard_keys
        self.use_beta_distribution = use_beta_distribution
        self.confidence_threshold = confidence_threshold
        self.visual_dim = visual_dim
        self.action_dim = 3 + num_keyboard_keys  # for legacy compat

        # Input projection (frame-stacked state + goal + target + optional visual)
        effective_state_dim = state_dim * frame_stack_size
        total_input = effective_state_dim + goal_dim + target_dim + visual_dim
        self.input_proj = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Mamba backbone
        self.use_mamba = False
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            self.use_mamba = True
        else:
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.1
            )

        # Confidence gate (CTM-inspired early exit)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # === Separate action heads ===
        # Mouse: Beta distribution params (alpha_x, beta_x, alpha_y, beta_y)
        mouse_out = 4 if use_beta_distribution else 2
        self.mouse_head = nn.Sequential(
            nn.Linear(hidden_dim, mouse_head_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mouse_head_hidden, mouse_out)
        )

        # Click: single logit
        self.click_head = nn.Sequential(
            nn.Linear(hidden_dim, click_head_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(click_head_hidden, 1)
        )

        # Keyboard: one logit per key
        self.keyboard_head = nn.Sequential(
            nn.Linear(hidden_dim, keyboard_head_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(keyboard_head_hidden, num_keyboard_keys)
        )

        # Action chunking decoder (predict multiple future frames)
        if chunk_size > 1:
            self.chunk_pos_embed = nn.Parameter(
                torch.randn(1, chunk_size, hidden_dim) * 0.02
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            )
            self.chunk_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # LSTM hidden state
        self.hidden = None

        # Temporal ensembling buffers (set up in get_action)
        self._overlapping_chunks: deque = deque(maxlen=3)
        self._chunk_step = 0

    def _run_backbone(self, x: torch.Tensor, hidden=None):
        """Run Mamba/LSTM backbone on projected input."""
        x = x.unsqueeze(1)  # [B, 1, hidden_dim]
        if self.use_mamba:
            x = self.mamba(x)
            new_hidden = None
        else:
            if hidden is None:
                x, (h, c) = self.lstm(x)
            else:
                x, (h, c) = self.lstm(x, hidden)
            new_hidden = (h, c)
        return x.squeeze(1), new_hidden  # [B, hidden_dim]

    def _apply_heads(self, features: torch.Tensor, target_info: torch.Tensor):
        """Apply separate action heads to features. Handles both [B, H] and [B, C, H]."""
        mouse_raw = self.mouse_head(features)
        click_logit = self.click_head(features)
        keyboard_logits = self.keyboard_head(features)

        # Residual target bias for mouse
        target_xy = target_info[:, :2].clamp(0.01, 0.99)

        if self.use_beta_distribution:
            # Shift Beta distribution mean toward target by biasing alpha
            # Beta mean = alpha/(alpha+beta). To bias mean toward target_xy,
            # we add a logit-space bias to the alpha parameter
            target_logits = torch.log(target_xy / (1.0 - target_xy))
            if mouse_raw.dim() == 3:
                # Chunked: [B, C, 4] - broadcast target bias
                mouse_raw = mouse_raw.clone()
                mouse_raw[:, :, 0] = mouse_raw[:, :, 0] + target_logits[:, 0:1]  # alpha_x bias
                mouse_raw[:, :, 2] = mouse_raw[:, :, 2] + target_logits[:, 1:2]  # alpha_y bias
            else:
                mouse_raw = mouse_raw.clone()
                mouse_raw[:, 0] = mouse_raw[:, 0] + target_logits[:, 0]
                mouse_raw[:, 2] = mouse_raw[:, 2] + target_logits[:, 1]
        else:
            # Logit-space bias (same as old Executor)
            target_logits = torch.log(target_xy / (1.0 - target_xy))
            if mouse_raw.dim() == 3:
                mouse_raw = mouse_raw.clone()
                mouse_raw[:, :, 0] = mouse_raw[:, :, 0] + target_logits[:, 0:1]
                mouse_raw[:, :, 1] = mouse_raw[:, :, 1] + target_logits[:, 1:2]
            else:
                mouse_raw = mouse_raw.clone()
                mouse_raw[:, 0] = mouse_raw[:, 0] + target_logits[:, 0]
                mouse_raw[:, 1] = mouse_raw[:, 1] + target_logits[:, 1]

        return mouse_raw, click_logit, keyboard_logits

    def forward(self,
                state: torch.Tensor,
                goal: torch.Tensor,
                target_info: torch.Tensor,
                hidden=None,
                visual_features: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass with separate heads and optional chunking.

        Args:
            state: [B, state_dim * frame_stack_size] frame-stacked state
            goal: [B, goal_dim]
            target_info: [B, target_dim]
            hidden: Optional LSTM hidden state
            visual_features: Optional [B, visual_dim] local precision features

        Returns:
            Dict with 'mouse', 'click', 'keyboard', 'confidence', 'hidden'
            Shapes: [B, C, dim] if chunk_size > 1, else [B, dim]
        """
        # Build input
        parts = [state, goal, target_info]
        if visual_features is not None and self.visual_dim > 0:
            parts.append(visual_features)
        elif self.visual_dim > 0:
            parts.append(torch.zeros(state.shape[0], self.visual_dim,
                                     device=state.device, dtype=state.dtype))
        x = torch.cat(parts, dim=-1)
        x = self.input_proj(x)  # [B, hidden_dim]

        # Confidence gate: can we skip the backbone?
        confidence = self.confidence_head(x)  # [B, 1]

        # Run backbone (always during training, gated during eval)
        if self.training:
            backbone_out, new_hidden = self._run_backbone(x, hidden)
        else:
            # Early exit: if confident, use projected features directly
            if confidence.mean().item() > self.confidence_threshold:
                backbone_out = x  # Skip backbone
                new_hidden = hidden
            else:
                backbone_out, new_hidden = self._run_backbone(x, hidden)

        # Action chunking
        if self.chunk_size > 1 and hasattr(self, 'chunk_decoder'):
            B = backbone_out.shape[0]
            memory = backbone_out.unsqueeze(1)  # [B, 1, hidden_dim]
            query = self.chunk_pos_embed.expand(B, -1, -1)  # [B, chunk_size, hidden_dim]
            chunk_features = self.chunk_decoder(query, memory)  # [B, chunk_size, hidden_dim]
            mouse_raw, click_logit, keyboard_logits = self._apply_heads(
                chunk_features, target_info)
        else:
            mouse_raw, click_logit, keyboard_logits = self._apply_heads(
                backbone_out, target_info)

        return {
            'mouse': mouse_raw,
            'click': click_logit,
            'keyboard': keyboard_logits,
            'confidence': confidence,
            'hidden': new_hidden
        }

    def forward_with_visual(self,
                            state: torch.Tensor,
                            goal: torch.Tensor,
                            target_info: torch.Tensor,
                            visual_features: torch.Tensor,
                            hidden=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Legacy-compatible forward with visual features. Returns (action_flat, hidden)."""
        pred = self.forward(state, goal, target_info, hidden, visual_features)
        action_flat = self._dict_to_flat(pred)
        return action_flat, pred['hidden']

    def forward_legacy(self,
                       state: torch.Tensor,
                       goal: torch.Tensor,
                       target_info: torch.Tensor,
                       hidden=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Legacy-compatible forward. Returns (action_flat[B, 31], hidden)."""
        pred = self.forward(state, goal, target_info, hidden)
        action_flat = self._dict_to_flat(pred)
        return action_flat, pred['hidden']

    def _dict_to_flat(self, pred: Dict) -> torch.Tensor:
        """Convert dict output to flat [B, action_dim] tensor for legacy compat."""
        mouse = pred['mouse']
        click = pred['click']
        keyboard = pred['keyboard']

        # If chunked, take first timestep
        if mouse.dim() == 3:
            mouse = mouse[:, 0]
            click = click[:, 0]
            keyboard = keyboard[:, 0]

        # Convert Beta params to logit-space mouse for legacy
        if self.use_beta_distribution:
            alpha_x = F.softplus(mouse[:, 0]) + 1.0
            beta_x = F.softplus(mouse[:, 1]) + 1.0
            alpha_y = F.softplus(mouse[:, 2]) + 1.0
            beta_y = F.softplus(mouse[:, 3]) + 1.0
            mean_x = alpha_x / (alpha_x + beta_x)
            mean_y = alpha_y / (alpha_y + beta_y)
            # Convert to logit space
            mean_x = mean_x.clamp(0.01, 0.99)
            mean_y = mean_y.clamp(0.01, 0.99)
            mouse_logits = torch.stack([
                torch.log(mean_x / (1.0 - mean_x)),
                torch.log(mean_y / (1.0 - mean_y))
            ], dim=-1)
        else:
            mouse_logits = mouse

        return torch.cat([mouse_logits, click, keyboard], dim=-1)

    def _decode_mouse(self, mouse_raw: np.ndarray) -> Tuple[float, float]:
        """Decode mouse head output to (x, y) in [0, 1]."""
        if self.use_beta_distribution:
            def softplus(x):
                return np.log1p(np.exp(np.clip(x, -20, 20)))
            alpha_x = softplus(mouse_raw[0]) + 1.0
            beta_x = softplus(mouse_raw[1]) + 1.0
            alpha_y = softplus(mouse_raw[2]) + 1.0
            beta_y = softplus(mouse_raw[3]) + 1.0
            mx = float(np.clip(alpha_x / (alpha_x + beta_x), 0, 1))
            my = float(np.clip(alpha_y / (alpha_y + beta_y), 0, 1))
        else:
            def sigmoid(x):
                return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
            mx = float(np.clip(sigmoid(mouse_raw[0]), 0, 1))
            my = float(np.clip(sigmoid(mouse_raw[1]), 0, 1))
        return mx, my

    def get_action(self,
                   state: np.ndarray,
                   goal: np.ndarray,
                   target_info: np.ndarray,
                   device: str = "cuda",
                   visual_features: Optional[np.ndarray] = None) -> Dict:
        """
        Get action from numpy arrays with temporal ensembling for chunks.

        Returns same dict format as old Executor:
            {'mouse_x', 'mouse_y', 'should_click', 'raw_click', 'keyboard': {str: bool}}
        """
        self.eval()

        self._chunk_step += 1

        # Re-plan if needed
        needs_replan = (
            len(self._overlapping_chunks) == 0 or
            self._chunk_step >= self.replan_every
        )

        if needs_replan:
            self._chunk_step = 0

            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                goal_t = torch.tensor(goal, dtype=torch.float32).unsqueeze(0).to(device)
                target_t = torch.tensor(target_info, dtype=torch.float32).unsqueeze(0).to(device)
                vis_t = None
                if visual_features is not None and self.visual_dim > 0:
                    vis_t = torch.tensor(visual_features, dtype=torch.float32).unsqueeze(0).to(device)

                pred = self.forward(state_t, goal_t, target_t, self.hidden, vis_t)
                self.hidden = pred['hidden']

                # Extract chunk as numpy: list of (mouse_raw, click, keyboard) per timestep
                mouse_np = pred['mouse'].cpu().numpy()[0]  # [C, 4] or [4]
                click_np = pred['click'].cpu().numpy()[0]   # [C, 1] or [1]
                kb_np = pred['keyboard'].cpu().numpy()[0]   # [C, 28] or [28]

                if mouse_np.ndim == 1:
                    # No chunking, single action
                    chunk = [(mouse_np, click_np, kb_np)]
                else:
                    chunk = [(mouse_np[t], click_np[t], kb_np[t])
                             for t in range(mouse_np.shape[0])]

                self._overlapping_chunks.append(chunk)

        # Temporal ensembling across overlapping chunks
        action = self._temporal_ensemble(self._chunk_step)

        # Parse action
        mouse_raw, click_raw, keyboard_raw = action
        mx, my = self._decode_mouse(mouse_raw)

        CLICK_THRESHOLD = 1.0
        KEYBOARD_THRESHOLD = 1.0

        result = {
            'mouse_x': mx,
            'mouse_y': my,
            'should_click': bool(click_raw[0] > CLICK_THRESHOLD if click_raw.ndim > 0 else click_raw > CLICK_THRESHOLD),
            'raw_click': float(click_raw[0] if click_raw.ndim > 0 else click_raw),
        }

        keyboard = {}
        for i, key_name in enumerate(KEYBOARD_KEYS):
            if i < len(keyboard_raw):
                keyboard[key_name] = bool(keyboard_raw[i] > KEYBOARD_THRESHOLD)
            else:
                keyboard[key_name] = False
        result['keyboard'] = keyboard

        return result

    def _temporal_ensemble(self, current_step: int):
        """Weighted average of overlapping chunk predictions for smoothness."""
        lam = self.temporal_ensemble_lambda
        weights = []
        mouse_vals = []
        click_vals = []
        kb_vals = []

        for i, chunk in enumerate(self._overlapping_chunks):
            # Offset within this chunk for the current frame
            chunk_offset = current_step + (len(self._overlapping_chunks) - 1 - i) * self.replan_every
            if 0 <= chunk_offset < len(chunk):
                w = np.exp(-lam * chunk_offset)
                weights.append(w)
                m, c, k = chunk[chunk_offset]
                mouse_vals.append(m)
                click_vals.append(c)
                kb_vals.append(k)

        if not weights:
            # Fallback: use latest chunk, first timestep
            return self._overlapping_chunks[-1][0]

        weights = np.array(weights)
        weights = weights / weights.sum()

        mouse_avg = sum(w * m for w, m in zip(weights, mouse_vals))
        click_avg = sum(w * c for w, c in zip(weights, click_vals))
        kb_avg = sum(w * k for w, k in zip(weights, kb_vals))

        return mouse_avg, click_avg, kb_avg

    def reset_hidden(self):
        """Reset hidden state and chunk buffers for new episode."""
        self.hidden = None
        self._overlapping_chunks.clear()
        self._chunk_step = 0


class ExecutorV2WithVisual(ExecutorV2):
    """ExecutorV2 with visual feature support."""

    def __init__(self, visual_dim: int = 64, **kwargs):
        super().__init__(visual_dim=visual_dim, **kwargs)


def create_executor(config=None, use_mamba: bool = True, with_visual: bool = False,
                    use_v2: bool = True, device: str = "cuda"):
    """
    Factory function to create the appropriate Executor.

    Args:
        config: ExecutorConfig (optional)
        use_mamba: Whether to try Mamba (falls back to LSTM if unavailable)
        with_visual: Whether to create visual-enabled version
        use_v2: Whether to create ExecutorV2 (separate heads, chunking, etc.)
        device: Target device

    Returns:
        Executor model on device
    """
    if config is None:
        from ..config import ExecutorConfig
        config = ExecutorConfig()

    if use_v2:
        visual_dim = config.visual_dim if (with_visual and config.use_visual_features) else 0
        model = ExecutorV2(
            state_dim=config.state_dim,
            goal_dim=config.goal_dim,
            target_dim=config.target_dim,
            hidden_dim=config.hidden_dim,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            frame_stack_size=config.frame_stack_size,
            chunk_size=config.chunk_size,
            replan_every=config.replan_every,
            temporal_ensemble_lambda=config.temporal_ensemble_lambda,
            mouse_head_hidden=config.mouse_head_hidden,
            click_head_hidden=config.click_head_hidden,
            keyboard_head_hidden=config.keyboard_head_hidden,
            num_keyboard_keys=NUM_KEYBOARD_KEYS,
            use_beta_distribution=config.use_beta_distribution,
            confidence_threshold=config.confidence_threshold,
            visual_dim=visual_dim,
        )
    elif with_visual and hasattr(config, 'use_visual_features') and config.use_visual_features:
        model = ExecutorWithVisual(
            state_dim=config.state_dim,
            goal_dim=config.goal_dim,
            target_dim=config.target_dim,
            visual_dim=config.visual_dim,
            hidden_dim=config.hidden_dim,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            action_dim=config.action_dim
        )
    elif use_mamba and MAMBA_AVAILABLE:
        model = Executor(
            state_dim=config.state_dim,
            goal_dim=config.goal_dim,
            target_dim=config.target_dim,
            hidden_dim=config.hidden_dim,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            action_dim=config.action_dim
        )
    else:
        model = ExecutorLSTM(
            state_dim=config.state_dim,
            goal_dim=config.goal_dim,
            target_dim=config.target_dim,
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim
        )

    return model.to(device)


def save_executor(model: nn.Module, path: str):
    """Save executor checkpoint."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'use_mamba': hasattr(model, 'mamba'),
    }
    # Save V2 config for reliable restoration
    if isinstance(model, ExecutorV2):
        save_dict['v2_config'] = {
            'state_dim': model.state_dim,
            'goal_dim': model.goal_dim,
            'target_dim': model.target_dim,
            'hidden_dim': model.hidden_dim,
            'frame_stack_size': model.frame_stack_size,
            'chunk_size': model.chunk_size,
            'replan_every': model.replan_every,
            'num_keyboard_keys': model.num_keyboard_keys,
            'use_beta_distribution': model.use_beta_distribution,
            'visual_dim': model.visual_dim,
        }
    torch.save(save_dict, path)
    logger.info(f"Executor saved to {path} ({model.__class__.__name__})")


def load_executor(path: str, device: str = "cuda", force_v2: bool = True) -> nn.Module:
    """Load executor from checkpoint, with automatic V1→V2 migration."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Executor checkpoint not found: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load executor checkpoint: {e}")

    model_class = checkpoint.get('model_class', '')
    use_mamba = checkpoint.get('use_mamba', MAMBA_AVAILABLE)
    state_dict = checkpoint['model_state_dict']

    from ..config import ExecutorConfig
    config = ExecutorConfig()

    # Detect if checkpoint is already ExecutorV2
    is_v2_checkpoint = (
        model_class in ('ExecutorV2', 'ExecutorV2WithVisual') or
        'mouse_head.0.weight' in state_dict
    )

    if is_v2_checkpoint:
        # Load V2 directly
        v2_cfg = checkpoint.get('v2_config', {})
        visual_dim = v2_cfg.get('visual_dim', 0)
        model = ExecutorV2(
            state_dim=v2_cfg.get('state_dim', config.state_dim),
            goal_dim=v2_cfg.get('goal_dim', config.goal_dim),
            target_dim=v2_cfg.get('target_dim', config.target_dim),
            hidden_dim=v2_cfg.get('hidden_dim', config.hidden_dim),
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            frame_stack_size=v2_cfg.get('frame_stack_size', config.frame_stack_size),
            chunk_size=v2_cfg.get('chunk_size', config.chunk_size),
            replan_every=v2_cfg.get('replan_every', config.replan_every),
            num_keyboard_keys=v2_cfg.get('num_keyboard_keys', NUM_KEYBOARD_KEYS),
            use_beta_distribution=v2_cfg.get('use_beta_distribution', config.use_beta_distribution),
            visual_dim=visual_dim,
        ).to(device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        logger.info(f"ExecutorV2 loaded from {path}")
        return model

    # Old checkpoint → migrate to V2 if force_v2
    if force_v2:
        logger.info(f"Migrating old {model_class} checkpoint to ExecutorV2...")

        # Detect visual from old checkpoint
        with_visual = model_class == 'ExecutorWithVisual'
        if not with_visual and 'input_proj.0.weight' in state_dict:
            if state_dict['input_proj.0.weight'].shape[1] > 200:
                with_visual = True
        visual_dim = config.visual_dim if with_visual else 0

        model = ExecutorV2(
            state_dim=config.state_dim,
            goal_dim=config.goal_dim,
            target_dim=config.target_dim,
            hidden_dim=config.hidden_dim,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            frame_stack_size=config.frame_stack_size,
            chunk_size=config.chunk_size,
            replan_every=config.replan_every,
            num_keyboard_keys=NUM_KEYBOARD_KEYS,
            use_beta_distribution=config.use_beta_distribution,
            visual_dim=visual_dim,
        ).to(device)

        # Partial weight migration: copy backbone, skip old action_head
        model_dict = model.state_dict()
        loaded = 0
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                model_dict[k] = v
                loaded += 1
            elif 'action_head' in k:
                logger.info(f"  Skipped old {k} (replaced by separate heads)")
            elif k in model_dict:
                logger.warning(f"  Shape mismatch {k}: {v.shape} vs {model_dict[k].shape}")
            else:
                logger.info(f"  Skipped {k}: not in ExecutorV2")

        model.load_state_dict(model_dict, strict=False)
        model.eval()
        logger.info(f"Migration complete: {loaded} layers copied. New heads need training.")
        return model

    # Load as old model type (no migration)
    with_visual = model_class == 'ExecutorWithVisual'
    if not with_visual and 'input_proj.0.weight' in state_dict:
        if state_dict['input_proj.0.weight'].shape[1] > 200:
            with_visual = True

    checkpoint_action_dim = None
    if 'action_head.3.weight' in state_dict:
        checkpoint_action_dim = state_dict['action_head.3.weight'].shape[0]
    elif 'action_head.2.weight' in state_dict:
        checkpoint_action_dim = state_dict['action_head.2.weight'].shape[0]

    target_action_dim = config.action_dim
    needs_migration = (checkpoint_action_dim is not None and
                       checkpoint_action_dim != target_action_dim)

    if with_visual and config.use_visual_features:
        model = ExecutorWithVisual(
            state_dim=config.state_dim, goal_dim=config.goal_dim,
            target_dim=config.target_dim, visual_dim=config.visual_dim,
            hidden_dim=config.hidden_dim, d_state=config.d_state,
            d_conv=config.d_conv, expand=config.expand,
            action_dim=target_action_dim
        )
    elif use_mamba and MAMBA_AVAILABLE:
        model = Executor(
            state_dim=config.state_dim, goal_dim=config.goal_dim,
            target_dim=config.target_dim, hidden_dim=config.hidden_dim,
            d_state=config.d_state, d_conv=config.d_conv, expand=config.expand,
            action_dim=target_action_dim
        )
    else:
        model = ExecutorLSTM(
            state_dim=config.state_dim, goal_dim=config.goal_dim,
            target_dim=config.target_dim, hidden_dim=config.hidden_dim,
            action_dim=target_action_dim
        )

    model = model.to(device)

    if needs_migration:
        model_dict = model.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                filtered[k] = v
            elif k in model_dict:
                if 'action_head' in k:
                    old_dim = v.shape[0]
                    model_dict[k][:min(old_dim, model_dict[k].shape[0])] = v[:min(old_dim, model_dict[k].shape[0])]
                    filtered[k] = model_dict[k]
        model.load_state_dict(filtered, strict=False)
    else:
        model.load_state_dict(state_dict)

    model.eval()
    logger.info(f"Executor loaded from {path} ({model_class})")
    return model
