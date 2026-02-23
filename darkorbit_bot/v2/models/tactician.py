"""
V2 Tactician - Target Selection with Cross-Attention

The Tactician runs at 10Hz and decides:
- Which object to target (explicit attention over objects)
- How to approach (velocity, urgency, aggression)

Uses cross-attention: Goal attends to objects.
This learns "given FIGHT goal, which object matters?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)


class Tactician(nn.Module):
    """
    Target selection using cross-attention.

    The goal embedding (from Strategist) queries the set of detected objects.
    Attention weights indicate which object is most relevant.
    """

    def __init__(self,
                 object_dim: int = 20,
                 goal_dim: int = 64,
                 hidden_dim: int = 128,
                 num_heads: int = 2,
                 num_layers: int = 2,
                 approach_dim: int = 4,
                 max_objects: int = 16,
                 dropout: float = 0.1):
        """
        Args:
            object_dim: Features per tracked object
            goal_dim: Goal embedding dimension
            hidden_dim: Hidden layer size
            num_heads: Attention heads
            num_layers: Number of transformer layers
            approach_dim: Output approach vector [vx, vy, urgency, aggression]
            max_objects: Maximum objects to consider
            dropout: Dropout rate
        """
        super().__init__()

        self.object_dim = object_dim
        self.goal_dim = goal_dim
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        self.approach_dim = approach_dim

        # Object encoder
        self.object_encoder = nn.Sequential(
            nn.Linear(object_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Goal encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Self-attention over objects (to understand relationships)
        self.object_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers,
        )

        # Cross-attention: Goal attends to objects
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output heads
        self.approach_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, approach_dim)
        )

        # Target info head (compressed info about selected target)
        self.target_info_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 32)  # 32-dim target embedding for Executor
        )

    def forward(self,
                objects: torch.Tensor,
                object_mask: torch.Tensor,
                goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            objects: [B, max_objects, object_dim] object features
            object_mask: [B, max_objects] 1 for valid objects, 0 for padding
            goal: [B, goal_dim] goal embedding from Strategist

        Returns:
            target_weights: [B, max_objects] attention weights (which object selected)
            approach: [B, approach_dim] approach vector
            target_info: [B, 32] compressed target info for Executor
        """
        B = objects.shape[0]

        # Encode objects
        obj_embed = self.object_encoder(objects)  # [B, max_objects, hidden]

        # Create attention mask (True = masked/invalid)
        # PyTorch expects True for positions to ignore
        # IMPORTANT: Only use mask if there are actually masked elements AND valid elements
        # to avoid PyTorch nested tensor "to_padded_tensor" error
        attn_mask = None
        has_valid = (object_mask == 1).any()
        has_invalid = (object_mask == 0).any()
        if has_valid and has_invalid:
            attn_mask = (object_mask == 0)  # [B, max_objects]

        # Self-attention over objects (with masking)
        # Need to handle the mask properly for TransformerEncoder
        obj_embed = self.object_self_attn(
            obj_embed,
            src_key_padding_mask=attn_mask
        )

        # Encode goal
        goal_embed = self.goal_encoder(goal)  # [B, hidden]
        goal_query = goal_embed.unsqueeze(1)  # [B, 1, hidden]

        # Cross-attention: goal queries objects
        # key_padding_mask: [B, max_objects], True = ignore
        # Use same attn_mask computed above (already handles edge cases)
        target_embed, attn_weights = self.cross_attention(
            query=goal_query,
            key=obj_embed,
            value=obj_embed,
            key_padding_mask=attn_mask
        )
        # target_embed: [B, 1, hidden]
        # attn_weights: [B, 1, max_objects]

        target_embed = target_embed.squeeze(1)  # [B, hidden]
        target_weights = attn_weights.squeeze(1)  # [B, max_objects]

        # Ensure numerical stability - attention can produce NaN if all keys are masked
        # This can happen when no objects are detected
        if torch.isnan(target_weights).any() or torch.isinf(target_weights).any():
            # Fallback: uniform distribution over valid objects, or uniform over all if none valid
            mask_sum = object_mask.sum(dim=-1, keepdim=True)
            if (mask_sum > 0).all():
                target_weights = object_mask / mask_sum.clamp(min=1.0)
            else:
                # All objects masked - use uniform distribution (first object will be selected)
                target_weights = torch.ones_like(target_weights) / target_weights.shape[-1]

        # Generate outputs
        approach = self.approach_head(target_embed)  # [B, approach_dim]
        target_info = self.target_info_head(target_embed)  # [B, 32]

        return target_weights, approach, target_info

    def get_target(self,
                   objects: np.ndarray,
                   object_mask: np.ndarray,
                   goal: np.ndarray,
                   device: str = "cuda") -> Dict:
        """
        Convenience method for inference.

        Returns:
            Dict with:
                'target_idx': int (which object is targeted)
                'target_weights': np.ndarray (full attention distribution)
                'approach': Dict with vx, vy, urgency, aggression
                'target_info': np.ndarray (32-dim for Executor)
        """
        self.eval()

        with torch.no_grad():
            objects_t = torch.tensor(objects, dtype=torch.float32).unsqueeze(0).to(device)
            mask_t = torch.tensor(object_mask, dtype=torch.float32).unsqueeze(0).to(device)
            goal_t = torch.tensor(goal, dtype=torch.float32).unsqueeze(0).to(device)

            weights, approach, target_info = self.forward(objects_t, mask_t, goal_t)

            weights = weights.cpu().numpy()[0]
            approach = approach.cpu().numpy()[0]
            target_info = target_info.cpu().numpy()[0]

        # Find the target with highest attention (only among valid objects)
        valid_weights = weights * object_mask

        # Handle edge case where all objects are masked (no valid targets)
        if np.sum(object_mask) == 0 or np.all(valid_weights == 0):
            target_idx = -1  # No valid target
        else:
            target_idx = int(np.argmax(valid_weights))

        # Use numpy sigmoid
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

        return {
            'target_idx': target_idx,
            'target_weights': weights,
            'approach': {
                'vx': float(approach[0]),
                'vy': float(approach[1]),
                'urgency': float(sigmoid(approach[2])),
                'aggression': float(sigmoid(approach[3]))
            },
            'target_info': target_info
        }


class TacticianWithHistory(Tactician):
    """
    Extended Tactician that also considers recent state history.

    This helps with predicting object motion and making smoother decisions.
    """

    def __init__(self,
                 object_dim: int = 20,
                 goal_dim: int = 64,
                 hidden_dim: int = 128,
                 history_len: int = 10,
                 **kwargs):
        super().__init__(object_dim, goal_dim, hidden_dim, **kwargs)

        self.history_len = history_len

        # Temporal encoder for recent states
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Fuse temporal info with goal
        self.temporal_fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward_with_history(self,
                            objects: torch.Tensor,
                            object_mask: torch.Tensor,
                            goal: torch.Tensor,
                            history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with state history.

        Args:
            objects: [B, max_objects, object_dim]
            object_mask: [B, max_objects]
            goal: [B, goal_dim]
            history: [B, history_len, state_dim] recent states

        Returns:
            Same as forward()
        """
        B = objects.shape[0]

        # Encode history
        # Project history to hidden_dim first
        history_proj = self.object_encoder[0](history)  # Reuse first linear
        _, (h_n, _) = self.temporal_encoder(history_proj)
        history_embed = h_n[-1]  # [B, hidden]

        # Fuse with goal
        goal_embed = self.goal_encoder(goal)
        fused_goal = self.temporal_fusion(
            torch.cat([goal_embed, history_embed], dim=-1)
        )

        # Now use fused goal for attention
        obj_embed = self.object_encoder(objects)

        # Only use mask if there are both valid and invalid elements
        attn_mask = None
        has_valid = (object_mask == 1).any()
        has_invalid = (object_mask == 0).any()
        if has_valid and has_invalid:
            attn_mask = (object_mask == 0)

        obj_embed = self.object_self_attn(obj_embed, src_key_padding_mask=attn_mask)

        goal_query = fused_goal.unsqueeze(1)
        target_embed, attn_weights = self.cross_attention(
            query=goal_query,
            key=obj_embed,
            value=obj_embed,
            key_padding_mask=attn_mask
        )

        target_embed = target_embed.squeeze(1)
        target_weights = attn_weights.squeeze(1)

        approach = self.approach_head(target_embed)
        target_info = self.target_info_head(target_embed)

        return target_weights, approach, target_info


class TacticianWithVisual(Tactician):
    """
    Tactician with visual feature support.

    Object features change from [N, 20] to [N, 148] (20 YOLO + 128 visual).
    Visual features help distinguish boss vs normal enemy, shielded vs unshielded, etc.
    """

    def __init__(self,
                 object_dim: int = 20,
                 visual_dim: int = 128,
                 goal_dim: int = 64,
                 hidden_dim: int = 128,
                 num_heads: int = 2,
                 num_layers: int = 2,
                 approach_dim: int = 4,
                 max_objects: int = 16,
                 dropout: float = 0.1):
        # Initialize parent with combined object dimension
        total_object_dim = object_dim + visual_dim
        super().__init__(
            object_dim=total_object_dim,  # Combined dimension
            goal_dim=goal_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            approach_dim=approach_dim,
            max_objects=max_objects,
            dropout=dropout
        )

        # Store original dimensions
        self.base_object_dim = object_dim
        self.visual_dim = visual_dim

    def forward_with_visual(self,
                            objects: torch.Tensor,
                            visual_features: torch.Tensor,
                            object_mask: torch.Tensor,
                            goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with separate object and visual inputs.

        Args:
            objects: [B, max_objects, base_object_dim] YOLO-based features
            visual_features: [B, max_objects, visual_dim] RoI visual features
            object_mask: [B, max_objects]
            goal: [B, goal_dim]

        Returns:
            Same as forward()
        """
        # Concatenate YOLO and visual features
        combined = torch.cat([objects, visual_features], dim=-1)
        return self.forward(combined, object_mask, goal)

    def forward_no_visual(self,
                          objects: torch.Tensor,
                          object_mask: torch.Tensor,
                          goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass without visual features (zero-padded).
        """
        B, N, _ = objects.shape
        device = objects.device

        visual_zeros = torch.zeros(B, N, self.visual_dim, device=device, dtype=objects.dtype)
        combined = torch.cat([objects, visual_zeros], dim=-1)
        return self.forward(combined, object_mask, goal)


def create_tactician(config=None, with_history: bool = False, with_visual: bool = False, device: str = "cuda"):
    """Factory function to create Tactician."""
    if config is None:
        from ..config import TacticianConfig
        config = TacticianConfig()

    if with_visual and hasattr(config, 'use_visual_features') and config.use_visual_features:
        model = TacticianWithVisual(
            object_dim=config.object_dim,
            visual_dim=config.visual_dim,
            goal_dim=config.goal_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            approach_dim=config.approach_dim,
            max_objects=config.max_objects,
            dropout=config.dropout
        )
    elif with_history:
        model = TacticianWithHistory(
            object_dim=config.object_dim,
            goal_dim=config.goal_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            approach_dim=config.approach_dim,
            max_objects=config.max_objects,
            dropout=config.dropout
        )
    else:
        model = Tactician(
            object_dim=config.object_dim,
            goal_dim=config.goal_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            approach_dim=config.approach_dim,
            max_objects=config.max_objects,
            dropout=config.dropout
        )

    return model.to(device)


def save_tactician(model: nn.Module, path: str):
    """Save tactician checkpoint."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }, path)
    logger.info(f"Tactician saved to {path}")


def load_tactician(path: str, device: str = "cuda") -> nn.Module:
    """Load tactician from checkpoint."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Tactician checkpoint not found: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load tactician checkpoint: {e}")

    # Determine model type from checkpoint
    model_class = checkpoint.get('model_class', '')
    with_history = model_class == 'TacticianWithHistory'
    with_visual = model_class == 'TacticianWithVisual'

    # Check if this is a migrated checkpoint (has visual layers in state dict)
    state_dict = checkpoint['model_state_dict']
    if not with_visual and 'object_encoder.0.weight' in state_dict:
        # Check input dimension to detect visual model
        input_weight = state_dict['object_encoder.0.weight']
        if input_weight.shape[1] > 50:  # Visual model has 148 input dim (20+128)
            with_visual = True

    model = create_tactician(with_history=with_history, with_visual=with_visual, device=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Tactician loaded from {path} (visual={with_visual})")
    return model
