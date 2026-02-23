"""
V2 Strategist - Goal Selection Transformer

The Strategist runs at 1Hz and decides the high-level goal:
- What mode? (FIGHT, LOOT, FLEE, EXPLORE, CAUTIOUS)
- What goal embedding? (rich 64-dim representation of intent)

Uses a Transformer to process long-term state history (60 seconds).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from ..config import FULL_STATE_DIM
import logging
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 120, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Strategist(nn.Module):
    """
    Transformer-based goal selection.

    Processes long-term state history and outputs:
    - Discrete mode (FIGHT, LOOT, FLEE, EXPLORE, CAUTIOUS)
    - Continuous goal embedding (64-dim)
    """

    def __init__(self,
                 state_dim: int = FULL_STATE_DIM,
                 hidden_dim: int = 256,
                 goal_dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 4,
                 num_modes: int = 5,
                 max_history: int = 120,
                 dropout: float = 0.1):
        """
        Args:
            state_dim: Input state dimension (from StateEncoderV2.full_state)
            hidden_dim: Transformer hidden size
            goal_dim: Output goal embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_modes: Number of discrete modes
            max_history: Maximum history length (in samples)
            dropout: Dropout rate
        """
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.goal_dim = goal_dim
        self.num_modes = num_modes
        self.max_history = max_history

        # Mode names for reference
        self.mode_names = ['FIGHT', 'LOOT', 'FLEE', 'EXPLORE', 'CAUTIOUS']

        # Input projection
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_history, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN: +10% faster training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=True  # Flash Attention: 30% faster (PyTorch 2.0+)
        )

        # Temporal aggregation (attention pooling)
        self.temporal_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output heads
        self.goal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, goal_dim)
        )

        self.mode_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_modes)
        )

        # Additional outputs for explainability
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self,
                state_history: torch.Tensor,
                history_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state_history: [B, T, state_dim] historical states
            history_mask: [B, T] optional mask (1 for valid, 0 for padding)

        Returns:
            goal: [B, goal_dim] goal embedding
            mode_logits: [B, num_modes] mode classification logits
            confidence: [B, 1] confidence score
        """
        B, T, _ = state_history.shape

        # Encode states
        x = self.state_encoder(state_history)  # [B, T, hidden]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask if needed
        # Only use mask if there are BOTH valid (1) AND invalid (0) elements
        # to avoid PyTorch nested tensor "to_padded_tensor" error
        attn_mask = None
        if history_mask is not None:
            has_valid = (history_mask == 1).any()
            has_invalid = (history_mask == 0).any()
            if has_valid and has_invalid:
                attn_mask = (history_mask == 0)  # True = masked

        # Transformer forward
        x = self.transformer(x, src_key_padding_mask=attn_mask)  # [B, T, hidden]

        # Temporal aggregation using learned query
        query = self.temporal_query.expand(B, -1, -1)  # [B, 1, hidden]
        context, _ = self.temporal_attn(query, x, x, key_padding_mask=attn_mask)
        context = context.squeeze(1)  # [B, hidden]

        # Generate outputs
        goal = self.goal_head(context)  # [B, goal_dim]
        mode_logits = self.mode_head(context)  # [B, num_modes]
        confidence = torch.sigmoid(self.confidence_head(context))  # [B, 1]

        return goal, mode_logits, confidence

    def get_goal(self,
                 state_history: np.ndarray,
                 history_mask: Optional[np.ndarray] = None,
                 device: str = "cuda") -> Dict:
        """
        Convenience method for inference.

        Args:
            state_history: [T, state_dim] historical states
            history_mask: [T] optional mask

        Returns:
            Dict with:
                'goal': np.ndarray [goal_dim]
                'mode': str (mode name)
                'mode_idx': int
                'mode_probs': np.ndarray [num_modes]
                'confidence': float
        """
        self.eval()

        with torch.no_grad():
            # Add batch dimension
            history_t = torch.tensor(state_history, dtype=torch.float32).unsqueeze(0).to(device)

            mask_t = None
            if history_mask is not None:
                mask_t = torch.tensor(history_mask, dtype=torch.float32).unsqueeze(0).to(device)

            goal, mode_logits, confidence = self.forward(history_t, mask_t)

            goal = goal.cpu().numpy()[0]
            mode_probs = F.softmax(mode_logits, dim=-1).cpu().numpy()[0]
            confidence = confidence.cpu().numpy()[0, 0]

        mode_idx = int(np.argmax(mode_probs))
        mode_name = self.mode_names[mode_idx]

        return {
            'goal': goal,
            'mode': mode_name,
            'mode_idx': mode_idx,
            'mode_probs': mode_probs,
            'confidence': confidence
        }


class StrategistWithContext(Strategist):
    """
    Extended Strategist that also considers current context explicitly.

    Useful when you want to bias decisions based on immediate threats.
    """

    def __init__(self,
                 state_dim: int = FULL_STATE_DIM,
                 context_dim: int = 32,
                 **kwargs):
        super().__init__(state_dim, **kwargs)

        self.context_dim = context_dim

        # Context encoder (for immediate threats, etc.)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )

        # Fusion layer
        self.context_fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    def forward_with_context(self,
                            state_history: torch.Tensor,
                            current_context: torch.Tensor,
                            history_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with explicit current context.

        Args:
            state_history: [B, T, state_dim]
            current_context: [B, context_dim] immediate context (threats, etc.)
            history_mask: [B, T] optional

        Returns:
            Same as forward()
        """
        B, T, _ = state_history.shape

        # Standard transformer processing
        x = self.state_encoder(state_history)
        x = self.pos_encoder(x)

        # Only use mask if there are BOTH valid (1) AND invalid (0) elements
        attn_mask = None
        if history_mask is not None:
            has_valid = (history_mask == 1).any()
            has_invalid = (history_mask == 0).any()
            if has_valid and has_invalid:
                attn_mask = (history_mask == 0)

        x = self.transformer(x, src_key_padding_mask=attn_mask)

        query = self.temporal_query.expand(B, -1, -1)
        temporal_context, _ = self.temporal_attn(query, x, x, key_padding_mask=attn_mask)
        temporal_context = temporal_context.squeeze(1)

        # Encode and fuse current context
        immediate_context = self.context_encoder(current_context)
        fused = self.context_fusion(
            torch.cat([temporal_context, immediate_context], dim=-1)
        )

        # Generate outputs from fused context
        goal = self.goal_head(fused)
        mode_logits = self.mode_head(fused)
        confidence = torch.sigmoid(self.confidence_head(fused))

        return goal, mode_logits, confidence


class StrategistWithVisual(Strategist):
    """
    Strategist with visual feature support.

    Input changes from [T, state_dim] to [T, state_dim+512] (state + 512 visual).
    Visual features provide environmental awareness (fire, poison, phase changes).
    """

    def __init__(self,
                 state_dim: int = FULL_STATE_DIM,
                 visual_dim: int = 512,
                 hidden_dim: int = 256,
                 goal_dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 4,
                 num_modes: int = 5,
                 max_history: int = 120,
                 dropout: float = 0.1):
        # Initialize parent with combined input dimension
        total_dim = state_dim + visual_dim
        super().__init__(
            state_dim=total_dim,  # Combined dimension
            hidden_dim=hidden_dim,
            goal_dim=goal_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_modes=num_modes,
            max_history=max_history,
            dropout=dropout
        )

        # Store original dimensions for reference
        self.base_state_dim = state_dim
        self.visual_dim = visual_dim

    def forward_with_visual(self,
                            state_history: torch.Tensor,
                            visual_history: torch.Tensor,
                            history_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with separate state and visual inputs.

        Args:
            state_history: [B, T, base_state_dim] YOLO-based state history
            visual_history: [B, T, visual_dim] global visual features
            history_mask: [B, T] optional mask

        Returns:
            Same as forward()
        """
        # Concatenate state and visual features
        combined = torch.cat([state_history, visual_history], dim=-1)
        return self.forward(combined, history_mask)

    def forward_no_visual(self,
                          state_history: torch.Tensor,
                          history_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass without visual features (zero-padded).
        Useful for backwards compatibility or when vision is unavailable.
        """
        B, T, _ = state_history.shape
        device = state_history.device

        # Pad with zeros for visual features
        visual_zeros = torch.zeros(B, T, self.visual_dim, device=device, dtype=state_history.dtype)
        combined = torch.cat([state_history, visual_zeros], dim=-1)
        return self.forward(combined, history_mask)


def create_strategist(config=None, with_context: bool = False, with_visual: bool = False, device: str = "cuda"):
    """Factory function to create Strategist."""
    if config is None:
        from ..config import StrategistConfig
        config = StrategistConfig()

    if with_visual and hasattr(config, 'use_visual_features') and config.use_visual_features:
        model = StrategistWithVisual(
            state_dim=config.state_dim,
            visual_dim=config.visual_dim,
            hidden_dim=config.hidden_dim,
            goal_dim=config.goal_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            num_modes=config.num_modes,
            dropout=config.dropout
        )
    elif with_context:
        model = StrategistWithContext(
            state_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            goal_dim=config.goal_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            num_modes=config.num_modes,
            dropout=config.dropout
        )
    else:
        model = Strategist(
            state_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            goal_dim=config.goal_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            num_modes=config.num_modes,
            dropout=config.dropout
        )

    return model.to(device)


def save_strategist(model: nn.Module, path: str):
    """Save strategist checkpoint."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }, path)
    logger.info(f"Strategist saved to {path}")


def load_strategist(path: str, device: str = "cuda") -> nn.Module:
    """Load strategist from checkpoint."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Strategist checkpoint not found: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load strategist checkpoint: {e}")

    # Determine model type from checkpoint
    model_class = checkpoint.get('model_class', '')
    with_context = model_class == 'StrategistWithContext'
    with_visual = model_class == 'StrategistWithVisual'

    # Check if this is a migrated checkpoint (has visual layers in state dict)
    state_dict = checkpoint['model_state_dict']
    if not with_visual and 'state_encoder.0.weight' in state_dict:
        # Check input dimension to detect visual model
        input_weight = state_dict['state_encoder.0.weight']
        if input_weight.shape[1] > 300:  # Visual model has 704 input dim
            with_visual = True

    model = create_strategist(with_context=with_context, with_visual=with_visual, device=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Strategist loaded from {path} (visual={with_visual})")
    return model
