"""
HP Bar Regression Model

Learns to predict enemy HP percentage from visual RoI features.
Uses the shared VisionBackbone (MobileNetV3) and adds a lightweight regression head.

Architecture:
    Enemy RoI → VisionBackbone (576-dim) → HP Head → HP% (0.0-1.0)

Training:
    - Frozen backbone (pretrained on ImageNet)
    - Train only HP head (~100k params)
    - Loss: Huber or MSE
    - Dataset: 5k-10k labeled enemy RoIs

Inference:
    - ~1-2ms for 10 enemies (runs in parallel with Tactician RoI encoding)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from pathlib import Path


class HPRegressionHead(nn.Module):
    """
    Lightweight regression head for HP prediction.

    Input: 576-dim features from MobileNetV3 backbone
    Output: HP percentage [0.0, 1.0]
    """

    def __init__(self,
                 input_dim: int = 576,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Backbone feature dimension (576 for MobileNetV3)
            hidden_dim: Hidden layer size
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output: [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim] backbone features

        Returns:
            hp_percent: [B, 1] predicted HP percentage
        """
        return self.head(x)


class HPShieldReader(nn.Module):
    """
    Multi-task reader for HP + Shield percentages.

    Input: 576-dim features from MobileNetV3 backbone
    Output: HP% [0.0, 1.0], Shield% [0.0, 1.0]
    """

    def __init__(self,
                 input_dim: int = 576,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Backbone feature dimension
            hidden_dim: Hidden layer size
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Shared encoder
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # HP head
        self.hp_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Shield head
        self.shield_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, input_dim] backbone features

        Returns:
            Dictionary with:
                'hp': [B, 1] HP percentage
                'shield': [B, 1] Shield percentage
        """
        shared_features = self.shared(x)

        return {
            'hp': self.hp_head(shared_features),
            'shield': self.shield_head(shared_features)
        }


class HPReader:
    """
    Wrapper for HP bar reading using VisionBackbone + Regression Head.

    Usage:
        reader = HPReader()
        reader.load('hp_reader.pt')

        # From VisionBackbone features
        hp_percentages = reader.predict_from_features(backbone_features)

        # Or directly from RoI images
        hp_percentages = reader.predict_from_rois(roi_images)
    """

    def __init__(self,
                 backbone=None,
                 device: str = "cuda",
                 use_shield: bool = False):
        """
        Args:
            backbone: VisionBackbone instance (optional, will load if None)
            device: Computation device
            use_shield: Use multi-task HP+Shield reader
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_shield = use_shield

        # Load backbone if not provided
        if backbone is None:
            from ..perception.vision_encoder import VisionBackbone, VisionConfig
            config = VisionConfig()
            self.backbone = VisionBackbone(config).to(self.device)
            self.backbone.eval()
        else:
            self.backbone = backbone

        # Initialize regression head
        if use_shield:
            self.model = HPShieldReader().to(self.device)
        else:
            self.model = HPRegressionHead().to(self.device)

        self.model.eval()

    def load(self, checkpoint_path: str):
        """Load trained HP reader weights."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"HP reader checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"[HP-READER] Loaded from {checkpoint_path}")

    def save(self, checkpoint_path: str):
        """Save HP reader weights."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"[HP-READER] Saved to {checkpoint_path}")

    @torch.no_grad()
    def predict_from_features(self, backbone_features: torch.Tensor) -> np.ndarray:
        """
        Predict HP% from backbone features.

        Args:
            backbone_features: [B, 576] features from VisionBackbone

        Returns:
            hp_percentages: [B] HP values (0.0-1.0)
        """
        self.model.eval()

        if backbone_features.device != self.device:
            backbone_features = backbone_features.to(self.device)

        # Forward pass
        if self.use_shield:
            output = self.model(backbone_features)
            hp = output['hp']
        else:
            hp = self.model(backbone_features)

        return hp.cpu().numpy().flatten()

    @torch.no_grad()
    def predict_from_rois(self, roi_images: list) -> np.ndarray:
        """
        Predict HP% directly from RoI images.

        Args:
            roi_images: List of RoI images [H, W, 3] (BGR/RGB)

        Returns:
            hp_percentages: [N] HP values (0.0-1.0)
        """
        if len(roi_images) == 0:
            return np.array([], dtype=np.float32)

        # Preprocess RoIs → backbone features
        from torchvision import transforms as T

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),  # RoI size
            T.ToTensor()
        ])

        roi_tensors = []
        for roi in roi_images:
            # Convert BGR to RGB if needed
            if roi.shape[2] == 3:
                roi_rgb = roi[:, :, ::-1].copy()
            else:
                roi_rgb = roi

            tensor = transform(roi_rgb)
            roi_tensors.append(tensor)

        # Batch forward
        batch = torch.stack(roi_tensors).to(self.device)

        # Extract backbone features
        backbone_features = self.backbone.forward_backbone(batch)

        # Predict HP
        return self.predict_from_features(backbone_features)

    @torch.no_grad()
    def predict_with_shield(self, backbone_features: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Predict HP% and Shield% from backbone features.

        Args:
            backbone_features: [B, 576] features from VisionBackbone

        Returns:
            Dictionary with:
                'hp': [B] HP percentages
                'shield': [B] Shield percentages
        """
        if not self.use_shield:
            raise ValueError("HPReader was not initialized with use_shield=True")

        self.model.eval()

        if backbone_features.device != self.device:
            backbone_features = backbone_features.to(self.device)

        # Forward pass
        output = self.model(backbone_features)

        return {
            'hp': output['hp'].cpu().numpy().flatten(),
            'shield': output['shield'].cpu().numpy().flatten()
        }


def create_hp_reader(checkpoint_path: Optional[str] = None,
                     backbone=None,
                     device: str = "cuda",
                     use_shield: bool = False) -> HPReader:
    """
    Factory function to create HP reader.

    Args:
        checkpoint_path: Path to trained weights (optional)
        backbone: VisionBackbone instance (optional)
        device: Computation device
        use_shield: Use multi-task HP+Shield reader

    Returns:
        HPReader instance
    """
    reader = HPReader(backbone=backbone, device=device, use_shield=use_shield)

    if checkpoint_path and Path(checkpoint_path).exists():
        reader.load(checkpoint_path)

    return reader


# Example usage
if __name__ == "__main__":
    # Create reader
    reader = create_hp_reader(device="cuda")

    # Example: Predict from backbone features
    dummy_features = torch.randn(10, 576).cuda()  # 10 enemies
    hp_percentages = reader.predict_from_features(dummy_features)

    print(f"Predicted HP%: {hp_percentages}")
    print(f"Shape: {hp_percentages.shape}")  # (10,)
