"""
Vision Encoder for V2 Bot

Extracts visual features (colors, textures) alongside YOLO coordinates.
This gives the bot "Game Sense" - understanding status effects, environmental
hazards, and fine-grained object recognition (e.g., Boss vs Normal enemy).

Architecture:
- Shared Vision Backbone (MobileNetV3-Small or ResNet18, frozen initially)
- Multi-scale outputs for different brains:
  - Strategist: Global context (224x224 -> 512-dim)
  - Tactician: RoI features (cropped objects -> 128-dim per object)
  - Executor: Local precision patches (64x64 around target -> 64-dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

try:
    import torchvision.models as models
    import torchvision.transforms as T
    from torchvision.ops import roi_align
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    models = None
    T = None

logger = logging.getLogger(__name__)


@dataclass
class VisionConfig:
    """Configuration for vision encoder."""
    # Backbone
    backbone: str = "mobilenet_v3_small"  # or "resnet18", "efficientnet_b0"
    pretrained: bool = True
    freeze_backbone: bool = True  # Freeze initially for efficiency

    # Feature dimensions (output)
    global_dim: int = 512      # For Strategist (full screen context)
    roi_dim: int = 128         # Per-object features for Tactician
    local_dim: int = 64        # For Executor precision

    # Input sizes
    global_size: int = 448     # Full screen resize (448 = 2x resolution, ~14x14 final features)
    roi_size: int = 64         # Object crop size
    local_size: int = 64       # Executor patch size

    # Heatmap resolution
    use_intermediate_features: bool = True  # Use 28x28 intermediate layer instead of 14x14 final

    # Processing
    max_rois: int = 16         # Maximum objects to extract features for
    device: str = "cuda"
    half_precision: bool = True  # Use FP16 for speed

    # Caching
    cache_global: bool = True   # Cache global features (only compute 1Hz)
    cache_ttl_ms: int = 100     # Cache time-to-live


class VisionBackbone(nn.Module):
    """
    Shared vision backbone that extracts multi-scale features.
    Uses a lightweight CNN pretrained on ImageNet.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config

        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision required for VisionEncoder")

        # Load pretrained backbone
        if config.backbone == "mobilenet_v3_small":
            self.backbone = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if config.pretrained else None
            )
            # MobileNetV3-Small output: 576 channels
            self.backbone_channels = 576
            # Remove classifier, keep features
            self.backbone.classifier = nn.Identity()

        elif config.backbone == "resnet18":
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if config.pretrained else None
            )
            self.backbone_channels = 512
            # Remove FC layer
            self.backbone.fc = nn.Identity()

        elif config.backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if config.pretrained else None
            )
            self.backbone_channels = 1280
            self.backbone.classifier = nn.Identity()

        else:
            raise ValueError(f"Unknown backbone: {config.backbone}")

        # Freeze backbone if configured
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection heads for different outputs
        self.global_proj = nn.Sequential(
            nn.Linear(self.backbone_channels, config.global_dim),
            nn.ReLU(),
            nn.LayerNorm(config.global_dim)
        )

        self.roi_proj = nn.Sequential(
            nn.Linear(self.backbone_channels, config.roi_dim),
            nn.ReLU(),
            nn.LayerNorm(config.roi_dim)
        )

        self.local_proj = nn.Sequential(
            nn.Linear(self.backbone_channels, config.local_dim),
            nn.ReLU(),
            nn.LayerNorm(config.local_dim)
        )

        # Image normalization (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize image tensor to ImageNet stats."""
        return (x - self.mean) / self.std

    def forward_features(self, x: torch.Tensor, intermediate: bool = False) -> torch.Tensor:
        """Run backbone and return spatial feature map [B, C, H, W].

        Args:
            x: Input tensor [B, 3, H, W]
            intermediate: If True, return intermediate layer (higher resolution, ~14x14)
                         If False, return final layer (lower resolution, ~7x7)
        """
        x = self.normalize(x)

        if self.config.backbone.startswith("mobilenet"):
            # MobileNet has .features which is a Sequential
            # For intermediate features, stop earlier (around layer 8-9 gives ~14x14)
            if intermediate:
                # Run up to inverted residual block 8 (index ~8) for 14x14 features
                for i, layer in enumerate(self.backbone.features):
                    x = layer(x)
                    if i == 8:  # After block 8, we have ~14x14 spatial
                        return x
            return self.backbone.features(x)
        elif self.config.backbone.startswith("efficientnet"):
             # EfficientNet has .features
            return self.backbone.features(x)
        elif self.config.backbone.startswith("resnet"):
            # ResNet needs manual sequential or use the modified backbone if layers removed
            # But here we just removed fc. The avgpool is still there in forward().
            # So we manually run the layers:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            if intermediate:
                return x  # layer2 gives ~28x28
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            return x

        # Fallback
        return self.backbone(x)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone and return flattened features."""
        # Get spatial features
        features = self.forward_features(x)

        # Global Average Pooling
        if features.dim() == 4:  # [B, C, H, W]
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)

        return features

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global context features (for Strategist)."""
        features = self.forward_backbone(x)
        return self.global_proj(features)

    def forward_roi(self, x: torch.Tensor) -> torch.Tensor:
        """Extract RoI features (for Tactician)."""
        features = self.forward_backbone(x)
        return self.roi_proj(features)

    def forward_local(self, x: torch.Tensor) -> torch.Tensor:
        """Extract local precision features (for Executor)."""
        features = self.forward_backbone(x)
        return self.local_proj(features)


class VisionEncoder:
    """
    Main vision encoder that provides visual features for the hierarchical bot.

    Usage:
        encoder = VisionEncoder(config)

        # Every 1 second (Strategist)
        global_features = encoder.encode_global(frame)

        # Every 100ms (Tactician)
        roi_features = encoder.encode_rois(frame, bboxes)

        # Every 16ms (Executor)
        local_features = encoder.encode_local(frame, target_pos)
    """

    def __init__(self, config: Optional[VisionConfig] = None):
        self.config = config or VisionConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        if not TORCHVISION_AVAILABLE:
            logger.warning("torchvision not available - VisionEncoder disabled")
            self.enabled = False
            return

        self.enabled = True

        # Initialize backbone
        self.backbone = VisionBackbone(self.config).to(self.device)
        self.backbone.eval()

        # Use half precision if configured and on CUDA
        if self.config.half_precision and self.device.type == "cuda":
            self.backbone = self.backbone.half()
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        # Transform for preprocessing
        self.global_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.config.global_size, self.config.global_size)),
            T.ToTensor()
        ])

        self.roi_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.config.roi_size, self.config.roi_size)),
            T.ToTensor()
        ])

        self.local_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.config.local_size, self.config.local_size)),
            T.ToTensor()
        ])

        # Cache for global features
        self._global_cache = None
        self._global_cache_time = 0

        logger.info(f"VisionEncoder initialized with {self.config.backbone} on {self.device}")

    @torch.no_grad()
    def encode_global(self, frame: np.ndarray, current_time_ms: int = 0,
                      return_map: bool = False, high_res_map: bool = True) -> Any:
        """
        Encode full screen for Strategist (environmental awareness).

        Args:
            frame: BGR or RGB frame [H, W, 3] (uint8)
            current_time_ms: Current timestamp for caching
            return_map: If True, returns tuple (features, activation_map_uint8)
            high_res_map: If True, use intermediate layer for higher resolution

        Returns:
            Global feature vector [global_dim] OR (vector, map) if return_map=True
        """
        if not self.enabled:
            return np.zeros(self.config.global_dim, dtype=np.float32)

        # Check cache (only if not asking for map, as we don't cache maps to save RAM)
        if self.config.cache_global and not return_map:
            if (self._global_cache is not None and
                current_time_ms - self._global_cache_time < self.config.cache_ttl_ms):
                return self._global_cache

        # Preprocess
        if frame.shape[2] == 4:  # BGRA
            frame = frame[:, :, :3]

        # Convert BGR to RGB if needed (assuming OpenCV format)
        frame_rgb = frame[:, :, ::-1].copy()

        # Transform and move to device
        tensor = self.global_transform(frame_rgb).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=self.dtype)

        # Extract features (final layer for the feature vector)
        spatial_features = self.backbone.forward_features(tensor, intermediate=False)

        # Collapse to vector
        flat_features = F.adaptive_avg_pool2d(spatial_features, 1).flatten(1)
        projected = self.backbone.global_proj(flat_features)

        result_vec = projected.cpu().float().numpy()[0]

        # Update cache
        if self.config.cache_global:
            self._global_cache = result_vec
            self._global_cache_time = current_time_ms

        if return_map:
            # Use intermediate layer for HIGHER RESOLUTION heatmap
            # With 448x448 input + intermediate layer = ~28x28 features
            # Each cell = ~69 pixels (vs 274 pixels with old 224/7x7 setup)
            if self.config.use_intermediate_features:
                heatmap_features = self.backbone.forward_features(tensor, intermediate=True)
            else:
                heatmap_features = spatial_features

            # Get spatial dimensions
            feat_h, feat_w = heatmap_features.shape[2], heatmap_features.shape[3]

            # MAX activation across all channels - shows strongest feature response
            max_activation, _ = torch.max(heatmap_features, dim=1)
            heatmap = max_activation.squeeze(0)  # [H, W]

            # Per-frame normalization to see relative activations clearly
            hmin, hmax = heatmap.min(), heatmap.max()
            if hmax > hmin:
                heatmap = (heatmap - hmin) / (hmax - hmin)
            else:
                heatmap = torch.zeros_like(heatmap)

            # NEAREST neighbor upscale - keeps the pixelated grid visible
            # Output size scales with input resolution
            output_size = (feat_h * 4, feat_w * 4)  # e.g., 28x28 -> 112x112
            heatmap = heatmap.unsqueeze(0).unsqueeze(0)
            heatmap = F.interpolate(heatmap, size=output_size, mode='nearest')
            heatmap = heatmap.squeeze()

            heatmap = (heatmap * 255).byte().cpu().numpy()

            return result_vec, heatmap

        return result_vec

    @torch.no_grad()
    def encode_rois(self, frame: np.ndarray,
                    bboxes: List[Tuple[float, float, float, float]]) -> np.ndarray:
        """
        Encode object regions for Tactician (object state/identity).

        Args:
            frame: BGR or RGB frame [H, W, 3] (uint8)
            bboxes: List of normalized bboxes [(x_center, y_center, width, height), ...]

        Returns:
            RoI features [num_objects, roi_dim] padded to max_rois
        """
        if not self.enabled:
            return np.zeros((self.config.max_rois, self.config.roi_dim), dtype=np.float32)

        num_boxes = min(len(bboxes), self.config.max_rois)
        features = np.zeros((self.config.max_rois, self.config.roi_dim), dtype=np.float32)

        if num_boxes == 0:
            return features

        # Convert frame
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frame_rgb = frame[:, :, ::-1].copy()

        h, w = frame.shape[:2]

        # Extract and process each RoI
        roi_tensors = []
        for i, (cx, cy, bw, bh) in enumerate(bboxes[:num_boxes]):
            # Convert normalized coords to pixel coords
            x1 = int(max(0, (cx - bw/2) * w))
            y1 = int(max(0, (cy - bh/2) * h))
            x2 = int(min(w, (cx + bw/2) * w))
            y2 = int(min(h, (cy + bh/2) * h))

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Crop and transform
            crop = frame_rgb[y1:y2, x1:x2]
            tensor = self.roi_transform(crop)
            roi_tensors.append(tensor)

        if not roi_tensors:
            return features

        # Batch process
        batch = torch.stack(roi_tensors).to(self.device, dtype=self.dtype)
        roi_features = self.backbone.forward_roi(batch)

        # Store results
        roi_np = roi_features.cpu().float().numpy()
        features[:len(roi_np)] = roi_np

        return features

    @torch.no_grad()
    def encode_local(self, frame: np.ndarray,
                     target_x: float, target_y: float,
                     patch_size: float = 0.15) -> np.ndarray:
        """
        Encode local region around target for Executor (precision aiming).

        Args:
            frame: BGR or RGB frame [H, W, 3] (uint8)
            target_x, target_y: Normalized target position
            patch_size: Normalized patch size (fraction of screen)

        Returns:
            Local feature vector [local_dim]
        """
        if not self.enabled:
            return np.zeros(self.config.local_dim, dtype=np.float32)

        # Convert frame
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frame_rgb = frame[:, :, ::-1].copy()

        h, w = frame.shape[:2]

        # Calculate patch bounds
        patch_w = int(patch_size * w)
        patch_h = int(patch_size * h)

        cx = int(target_x * w)
        cy = int(target_y * h)

        x1 = max(0, cx - patch_w // 2)
        y1 = max(0, cy - patch_h // 2)
        x2 = min(w, x1 + patch_w)
        y2 = min(h, y1 + patch_h)

        # Adjust if at edge
        if x2 - x1 < patch_w:
            x1 = max(0, x2 - patch_w)
        if y2 - y1 < patch_h:
            y1 = max(0, y2 - patch_h)

        # Crop and transform
        crop = frame_rgb[y1:y2, x1:x2]

        if crop.size == 0:
            return np.zeros(self.config.local_dim, dtype=np.float32)

        tensor = self.local_transform(crop).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=self.dtype)

        # Extract features
        features = self.backbone.forward_local(tensor)
        return features.cpu().float().numpy()[0]

    @torch.no_grad()
    def encode_colors(self, frame: np.ndarray,
                      bboxes: List[Tuple[float, float, float, float]]) -> np.ndarray:
        """
        Extract simple color statistics from object regions.
        Fast alternative to full CNN features.

        Args:
            frame: BGR frame [H, W, 3] (uint8)
            bboxes: List of normalized bboxes

        Returns:
            Color features [num_objects, 12] (mean RGB, std RGB, HSV mean, HSV std)
        """
        num_boxes = min(len(bboxes), self.config.max_rois)
        features = np.zeros((self.config.max_rois, 12), dtype=np.float32)

        if num_boxes == 0:
            return features

        h, w = frame.shape[:2]

        try:
            import cv2
            use_cv2 = True
        except ImportError:
            use_cv2 = False

        for i, (cx, cy, bw, bh) in enumerate(bboxes[:num_boxes]):
            # Convert to pixel coords
            x1 = int(max(0, (cx - bw/2) * w))
            y1 = int(max(0, (cy - bh/2) * h))
            x2 = int(min(w, (cx + bw/2) * w))
            y2 = int(min(h, (cy + bh/2) * h))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]

            # RGB statistics (normalized 0-1)
            rgb = crop.astype(np.float32) / 255.0
            rgb_mean = np.mean(rgb, axis=(0, 1))  # [3]
            rgb_std = np.std(rgb, axis=(0, 1))    # [3]

            # HSV statistics (if cv2 available)
            if use_cv2:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 0] /= 180.0  # H: 0-180 -> 0-1
                hsv[:, :, 1:] /= 255.0  # S, V: 0-255 -> 0-1
                hsv_mean = np.mean(hsv, axis=(0, 1))
                hsv_std = np.std(hsv, axis=(0, 1))
            else:
                hsv_mean = np.zeros(3, dtype=np.float32)
                hsv_std = np.zeros(3, dtype=np.float32)

            features[i] = np.concatenate([rgb_mean, rgb_std, hsv_mean, hsv_std])

        return features

    def get_feature_dims(self) -> Dict[str, int]:
        """Get output dimensions for each feature type."""
        return {
            'global': self.config.global_dim,
            'roi': self.config.roi_dim,
            'local': self.config.local_dim,
            'color': 12  # RGB mean/std + HSV mean/std
        }


class LightweightColorEncoder:
    """
    Ultra-fast color/texture encoder for real-time use.
    No neural network - just statistical features.
    Use this if VisionEncoder is too slow for 60fps.
    """

    def __init__(self, feature_dim: int = 32):
        self.feature_dim = feature_dim

    def encode_object(self, frame: np.ndarray,
                      cx: float, cy: float,
                      w: float, h: float) -> np.ndarray:
        """
        Extract lightweight visual features from an object region.

        Features (32-dim):
        - Mean RGB (3)
        - Std RGB (3)
        - Color histogram (9) - simplified 3x3x1 bins
        - Edge density (1)
        - Brightness (1)
        - Dominant color hue (1)
        - Saturation mean/std (2)
        - Texture variance (1)
        - Red ratio (for health/damage) (1)
        - Blue ratio (for shields/freeze) (1)
        - Green ratio (for poison/heal) (1)
        - Glow detection (1) - bright pixels ratio
        - Reserved (7)

        Returns:
            Feature vector [32]
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)

        frame_h, frame_w = frame.shape[:2]

        # Convert to pixel coords
        x1 = int(max(0, (cx - w/2) * frame_w))
        y1 = int(max(0, (cy - h/2) * frame_h))
        x2 = int(min(frame_w, (cx + w/2) * frame_w))
        y2 = int(min(frame_h, (cy + h/2) * frame_h))

        if x2 <= x1 or y2 <= y1:
            return features

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return features

        # Normalize to 0-1
        crop_f = crop.astype(np.float32) / 255.0

        # Mean RGB [0-2]
        features[0:3] = np.mean(crop_f, axis=(0, 1))

        # Std RGB [3-5]
        features[3:6] = np.std(crop_f, axis=(0, 1))

        # Simplified color histogram [6-14]
        # Quantize to 3 levels per channel
        quantized = (crop_f * 2.99).astype(np.int32)
        for i in range(3):
            hist, _ = np.histogram(quantized[:, :, i].flatten(), bins=3, range=(0, 3))
            features[6+i*3:9+i*3] = hist / (hist.sum() + 1e-6)

        # Brightness [15]
        features[15] = np.mean(crop_f)

        # Color ratios for status effects [16-19]
        total = features[0] + features[1] + features[2] + 1e-6
        features[16] = features[2] / total  # Red ratio (BGR format)
        features[17] = features[0] / total  # Blue ratio
        features[18] = features[1] / total  # Green ratio

        # Glow detection (bright pixels > 0.8) [19]
        brightness = np.mean(crop_f, axis=2)
        features[19] = np.mean(brightness > 0.8)

        # Texture variance [20]
        features[20] = np.var(crop_f)

        # Edge density (simple gradient) [21]
        if crop.shape[0] > 2 and crop.shape[1] > 2:
            gray = np.mean(crop_f, axis=2)
            dx = np.abs(np.diff(gray, axis=1))
            dy = np.abs(np.diff(gray, axis=0))
            features[21] = (np.mean(dx) + np.mean(dy)) / 2

        # Reserved [22-31] for future use

        return features

    def encode_global(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract global scene features.

        Features (32-dim):
        - Overall brightness (1)
        - Overall color means (3)
        - Color variance (1)
        - Screen region brightness (4) - quadrants
        - Dominant colors (3)
        - Reserved (20)
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)

        crop_f = frame.astype(np.float32) / 255.0

        # Overall brightness [0]
        features[0] = np.mean(crop_f)

        # Color means [1-3]
        features[1:4] = np.mean(crop_f, axis=(0, 1))

        # Color variance [4]
        features[4] = np.var(crop_f)

        # Quadrant brightness [5-8]
        h, w = crop_f.shape[:2]
        h2, w2 = h // 2, w // 2
        features[5] = np.mean(crop_f[:h2, :w2])       # Top-left
        features[6] = np.mean(crop_f[:h2, w2:])       # Top-right
        features[7] = np.mean(crop_f[h2:, :w2])       # Bottom-left
        features[8] = np.mean(crop_f[h2:, w2:])       # Bottom-right

        # Dominant color ratios [9-11]
        total = features[1] + features[2] + features[3] + 1e-6
        features[9] = features[3] / total   # Red dominance
        features[10] = features[1] / total  # Blue dominance
        features[11] = features[2] / total  # Green dominance

        return features


# Factory function
def create_vision_encoder(config: Optional[VisionConfig] = None,
                          lightweight: bool = False) -> object:
    """
    Create appropriate vision encoder based on hardware/config.

    Args:
        config: Vision encoder configuration
        lightweight: If True, use LightweightColorEncoder (no GPU needed)

    Returns:
        VisionEncoder or LightweightColorEncoder
    """
    if lightweight or not TORCHVISION_AVAILABLE:
        logger.info("Using LightweightColorEncoder (no neural network)")
        return LightweightColorEncoder()

    return VisionEncoder(config)
