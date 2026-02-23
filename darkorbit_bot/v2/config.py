"""
V2 Configuration - Hierarchical Temporal Architecture

All configurable parameters for the V2 system.
"""

from dataclasses import dataclass, field
from typing import List, Optional, FrozenSet, Tuple, Dict
from enum import IntEnum
import logging

# Setup module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Mode Enum - Central definition of bot modes
# =============================================================================
class Mode(IntEnum):
    """Bot operational modes."""
    FIGHT = 0
    LOOT = 1
    FLEE = 2
    EXPLORE = 3
    CAUTIOUS = 4

    @classmethod
    def names(cls) -> List[str]:
        """Get list of mode names."""
        return [m.name for m in cls]


# =============================================================================
# Centralized Class Name Constants
# =============================================================================
# These are used across tracker.py, state_encoder.py, and other modules.
# Defined as frozensets for O(1) lookup performance.

ENEMY_CLASSES: FrozenSet[str] = frozenset([
    # Base names
    'Devo', 'Lordakia', 'Mordon', 'Saimon', 'Sibelon', 'Struener', 'Streuner',
    'Kristallon', 'Kristallin', 'Sibelonit', 'Protegit',
    
    # YOLO detected names (observed in logs)
    'Enemy_npc_mordon', 'enemy_npc_mordon', 
    'Enemy_npc_streuner', 'enemy_npc_streuner',
    'Enemy_npc_saimon', 'enemy_npc_saimon',
    'Enemy_npc_lordakia', 'enemy_npc_lordakia',
    'Enemy_npc_devolarium', 'enemy_npc_devolarium',
    'Enemy_npc_sibelon', 'enemy_npc_sibelon',
    'Enemy_npc_kristalin', 'Enemy_npc_kristallin',
    'Enemy_npc_kristalion', 'Enemy_npc_kristallon',
    
    # Boss/Uber/Elite variations
    'Enemy_npc_boss_mordon', 'Enemy_npc_uber_mordon',
    'Enemy_npc_boss_saimon', 'Enemy_npc_uber_saimon',
    'Enemy_npc_boss_streuner', 'enemy_npc_boss_streuner',
    'Enemy_npc_boss_lordakia', 'Enemy_npc_uber_lordakia',
    'Enemy_npc_boss_devolarium', 'enemy_npc_boss_devolarium',
    'Enemy_npc_boss_sibelon', 'Enemy_npc_uber_sibelon',
    'Enemy_npc_boss_kristalin', 'Enemy_npc_boss_kristalion',
    'Enemy_npc_uber_kristallon', 'enemy_npc_uber_kristallon',
    'Enemy_npc_uber_kristallion', 'Enemy_npc_elite_saimon',
    'Enemy_npc_elite_sylox', 'Enemy_npc_sylox',

    'damage_numbers',
    # Generic
    'npc', 'enemy',
    # Add player ships as potential enemies so the model sees them as targets
    'Ship_goliath', 'Ship_vengeance', 'Ship_bigboy', 'Ship_nostromo', 
    'Ship_lenov', 'Ship_leonov', 'Ship_venom', 'Ship_solace', 
    'Ship_diminisher', 'Ship_spectrum', 'Ship_sentinel', 'Ship_cyborg',
    'PlayerShip', 'Player'
])

LOOT_CLASSES: FrozenSet[str] = frozenset([
    'BonusBox', 'box', 'bonus_box'
])

# Player classes are now also in ENEMY_CLASSES, but we keep this for specific identification
PLAYER_CLASSES: FrozenSet[str] = frozenset([
    'Player', 'player_ship',
    'Ship_goliath', 'Ship_vengeance', 'Ship_bigboy', 'Ship_nostromo', 
    'Ship_lenov', 'Ship_leonov', 'Ship_venom', 'Ship_solace', 
    'Ship_diminisher', 'Ship_spectrum', 'Ship_sentinel', 'Ship_cyborg'
])

IGNORE_CLASSES: FrozenSet[str] = frozenset([
    'portal'
])


# =============================================================================
# Keyboard Action Keys (for training and inference)
# =============================================================================
# These are the game keys the model can learn to press.
# This list is ordered and determines the keyboard output dimension.
KEYBOARD_KEYS: Tuple[str, ...] = (
    'ctrl', 'shift', 'alt', 'space',  # Modifiers
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',  # Abilities/ammo
    'q', 'w', 'e', 'r', 't',  # Top row
    'a', 's', 'd', 'f', 'g',  # Middle row
    'z', 'x', 'c', 'v', 'b',  # Bottom row
    'j', 'tab', 'esc',  # Special
)

# Number of keyboard keys the model outputs
NUM_KEYBOARD_KEYS = len(KEYBOARD_KEYS)  # Currently 32 keys

# Mapping from key name to index (for fast lookup)
KEYBOARD_KEY_TO_IDX: Dict[str, int] = {k: i for i, k in enumerate(KEYBOARD_KEYS)}

# =============================================================================
# Hotkeys per Script Type (excluded from keyboard recording)
# =============================================================================
# These keys control the bot itself, not gameplay actions.
# They are excluded when recording human keyboard input.
SCRIPT_HOTKEYS: Dict[str, Tuple[str, ...]] = {
    'bot': ('f1', 'f2', 'f3', 'f4', 'f5', 'f6'),      # Pause, BAD STOP, EMERGENCY, Debug, Reasoning, Mode
    'shadow': ('f1', 'f2', 'f3', 'f4', 'f5', 'f6'),   # Same for shadow training
    'default': ('f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12'),
}


# =============================================================================
# State Encoder Constants
# =============================================================================
NUM_OBJECTS_IN_FLAT_STATE = 16  # Max objects included in flattened state
ASSUMED_FPS = 30                # For velocity estimation
VELOCITY_SCALE = 10             # Velocity normalization factor
POSITION_HISTORY_LEN = 10       # Frames of position history to keep

# Object feature dimension from TrackedObject.to_feature_vector()
# This is computed dynamically but we define the expected breakdown here:
#   Position (4): x, y, distance_to_player, angle_to_player
#   Velocity (4): vx, vy, speed, heading
#   Bbox (4): width, height, confidence, age_normalized
#   Tracking (4): hits, time_since_update, is_tracked, is_lost
#   Class (4): is_enemy, is_loot, is_player, is_other
# Total: 20 features (can be extended by modifying TrackedObject.to_feature_vector)
OBJECT_FEATURE_DIM = 20

# State vector layout constants (derived, do not edit independently)
PLAYER_DIM = 16
CONTEXT_DIM = 16
OBJECTS_START = PLAYER_DIM                                              # 16
CONTEXT_START = PLAYER_DIM + NUM_OBJECTS_IN_FLAT_STATE * OBJECT_FEATURE_DIM  # 336
FULL_STATE_DIM = CONTEXT_START + CONTEXT_DIM                            # 352



@dataclass
class VisionEncoderConfig:
    """Vision encoder configuration for pixel/color features."""
    # Enable/disable
    enabled: bool = True
    use_lightweight: bool = False       # Use LightweightColorEncoder (no GPU)

    # Backbone
    backbone: str = "mobilenet_v3_small"  # "mobilenet_v3_small", "resnet18", "efficientnet_b0"
    pretrained: bool = True
    freeze_backbone: bool = True        # Freeze for efficiency

    # Feature dimensions
    global_dim: int = 512               # Strategist: full screen context
    roi_dim: int = 128                  # Tactician: per-object features
    local_dim: int = 64                 # Executor: precision patch
    color_dim: int = 32                 # Lightweight color stats

    # Input sizes
    global_size: int = 224              # Full screen resize
    roi_size: int = 64                  # Object crop size
    local_size: int = 64                # Executor patch size

    # Processing
    max_rois: int = 16                  # Max objects for feature extraction
    half_precision: bool = True         # FP16 for speed

    # Caching (Strategist runs at 1Hz, can cache)
    cache_global: bool = True
    cache_ttl_ms: int = 100


@dataclass
class PerceptionConfig:
    """Object detection and tracking configuration."""
    # YOLO
    yolo_model_path: str = "F:/dev/bot/best.pt"
    yolo_confidence: float = 0.5
    yolo_device: str = "cuda"

    # ByteTrack
    track_high_thresh: float = 0.6      # High confidence threshold for new tracks
    track_low_thresh: float = 0.1       # Low confidence threshold for matching
    track_buffer: int = 6               # Frames to keep lost tracks (Reduced 30->6 to fix hallucinations)
    match_thresh: float = 0.8           # IoU threshold for matching

    # Vision encoder
    vision: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)

    # Object classes
    enemy_classes: List[str] = field(default_factory=lambda: [
        'Devo', 'Lordakia', 'Mordon', 'Saimon', 'Sibelon', 'Struener',
        'npc', 'enemy'
    ])
    loot_classes: List[str] = field(default_factory=lambda: [
        'BonusBox', 'box', 'bonus_box'
    ])
    player_classes: List[str] = field(default_factory=lambda: [
        'Player', 'player_ship'
    ])
    ignore_classes: List[str] = field(default_factory=lambda: [
        'portal'
    ])


@dataclass
class StrategistConfig:
    """Strategist (goal selection) configuration."""
    # Architecture
    # state_dim = player(16) + flat_objects(NUM_OBJECTS_IN_FLAT_STATE * OBJECT_FEATURE_DIM) + context(16)
    # Default: 16 + (16 * 20) + 16 = 352
    state_dim: int = 16 + (NUM_OBJECTS_IN_FLAT_STATE * OBJECT_FEATURE_DIM) + 16
    visual_dim: int = 512           # Global visual context dimension
    hidden_dim: int = 256           # Transformer hidden size
    goal_dim: int = 64              # Output goal embedding size
    num_heads: int = 4              # Attention heads
    num_layers: int = 4             # Transformer layers
    dropout: float = 0.1

    # Temporal
    history_seconds: int = 60       # How many seconds of history to consider
    history_sample_rate: int = 1    # Sample rate (1 = every second)

    # Modes
    num_modes: int = 5              # FIGHT, LOOT, FLEE, EXPLORE, CAUTIOUS
    mode_names: List[str] = field(default_factory=lambda: [
        'FIGHT', 'LOOT', 'FLEE', 'EXPLORE', 'CAUTIOUS'
    ])

    # Timing
    update_interval_ms: int = 1000  # Update every 1 second

    # Feature fusion mode
    use_visual_features: bool = True

    @property
    def state_total_dim(self) -> int:
        """Total state features = base + visual (computed dynamically)."""
        return self.state_dim + self.visual_dim if self.use_visual_features else self.state_dim


@dataclass
class TacticianConfig:
    """Tactician (target selection) configuration."""
    # Architecture
    # object_dim is set from OBJECT_FEATURE_DIM by default (from TrackedObject.to_feature_vector)
    object_dim: int = OBJECT_FEATURE_DIM  # Base features per tracked object
    visual_dim: int = 128           # Visual features per object (from VisionEncoder)
    goal_dim: int = 64              # Goal embedding from Strategist
    hidden_dim: int = 128           # Hidden layer size
    num_heads: int = 2              # Attention heads
    num_layers: int = 2             # Transformer layers
    dropout: float = 0.1

    # Objects
    max_objects: int = 16           # Maximum objects to consider

    # Output
    approach_dim: int = 4           # [vx, vy, urgency, aggression]

    # Timing
    update_interval_ms: int = 100   # Update every 100ms (10 Hz)

    # Feature fusion mode
    use_visual_features: bool = True

    @property
    def object_total_dim(self) -> int:
        """Total object features = base + visual (computed dynamically)."""
        return self.object_dim + self.visual_dim if self.use_visual_features else self.object_dim


@dataclass
class ExecutorConfig:
    """Executor (motor control) configuration."""
    # Architecture
    state_dim: int = 64             # Compact state features
    goal_dim: int = 64              # Goal from Strategist
    target_dim: int = 34            # Target info: 2 (x,y position) + 32 (learned embedding from Tactician)
    visual_dim: int = 64            # Local visual features (precision patch)
    hidden_dim: int = 256           # Mamba hidden size

    # Mamba specific
    d_state: int = 64               # State space dimension
    d_conv: int = 4                 # Convolution width
    expand: int = 2                 # Expansion factor

    # Output dimensions (computed dynamically from KEYBOARD_KEYS)
    mouse_dim: int = 2              # mouse_x, mouse_y
    click_dim: int = 1              # click

    # Timing
    target_fps: int = 60            # Run at 60 FPS

    # Feature fusion mode
    use_visual_features: bool = True

    # === V2 Executor enhancements ===
    # Separate head architecture
    mouse_head_hidden: int = 128
    click_head_hidden: int = 64
    keyboard_head_hidden: int = 64
    use_beta_distribution: bool = True  # Beta dist for mouse (bounded [0,1])

    # Action chunking
    chunk_size: int = 8             # Predict 8 future frames at once
    replan_every: int = 4           # Re-plan every 4 frames
    temporal_ensemble_lambda: float = 0.01  # Exponential weighting for overlapping chunks

    # Frame stacking
    frame_stack_size: int = 3       # Stack last 3 frames for motion context

    # Focal loss for keyboard
    focal_alpha: float = 0.25       # Focal loss alpha
    focal_gamma: float = 2.0        # Focal loss gamma (higher = more focus on hard examples)

    # Confidence-gated early exit (CTM-inspired)
    confidence_threshold: float = 0.8  # Skip backbone if confidence > this

    @property
    def keyboard_dim(self) -> int:
        """Number of keyboard action outputs (from KEYBOARD_KEYS)."""
        return NUM_KEYBOARD_KEYS

    @property
    def action_dim(self) -> int:
        """Total action dimension = mouse + click + keyboard."""
        return self.mouse_dim + self.click_dim + self.keyboard_dim

    @property
    def target_total_dim(self) -> int:
        """Total target features = base + visual (computed dynamically)."""
        return self.target_dim + self.visual_dim if self.use_visual_features else self.target_dim

    @property
    def effective_state_dim(self) -> int:
        """State dim accounting for frame stacking."""
        return self.state_dim * self.frame_stack_size


@dataclass
class TrainingConfig:
    """Training configuration."""
    # General
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 50

    # Per-component epochs (can override num_epochs)
    executor_epochs: int = 100      # Executor needs most training
    tactician_epochs: int = 50      # Tactician needs medium training
    strategist_epochs: int = 30     # Strategist needs least training

    # Data
    sequence_length: int = 50       # Frames per training sequence
    train_val_split: float = 0.9

    # Optimizer
    use_amp: bool = True            # Automatic Mixed Precision
    gradient_clip: float = 1.0

    # Checkpointing
    save_every_epochs: int = 5
    checkpoint_dir: str = "data/checkpoints/v2"

    # VLM corrections
    correction_weight: float = 2.0  # Higher weight for VLM corrections


@dataclass
class V2Config:
    """Main V2 configuration combining all components."""
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    strategist: StrategistConfig = field(default_factory=StrategistConfig)
    tactician: TacticianConfig = field(default_factory=TacticianConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Runtime
    device: str = "cuda"
    monitor: int = 1
    debug_mode: bool = False

    # Humanization (shared with V1)
    reaction_delay_ms: int = 50
    precision_noise: float = 0.05

    # Safety
    emergency_stop_key: str = "f3"

    # VLM
    vlm_enabled: bool = True
    vlm_url: str = "http://localhost:1234"
    vlm_model: str = "qwen/qwen3-vl-8b"


# Default configuration
DEFAULT_CONFIG = V2Config()


def load_config(path: str = None) -> V2Config:
    """Load configuration from file or return default."""
    if path is None:
        return DEFAULT_CONFIG

    # TODO: Implement JSON/YAML loading
    import json
    from pathlib import Path

    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            data = json.load(f)
        # TODO: Parse into config dataclasses
        pass

    return DEFAULT_CONFIG


def save_config(config: V2Config, path: str):
    """Save configuration to file."""
    import json
    from dataclasses import asdict

    with open(path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
