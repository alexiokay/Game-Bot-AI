"""
DarkOrbit Bot V2 - Hierarchical Temporal Architecture

A modern game bot architecture using:
- ByteTrack for object persistence
- Transformer for strategic decisions (Strategist, 1Hz)
- Cross-Attention for target selection (Tactician, 10Hz)
- Mamba (SSM) for fast execution (Executor, 60Hz)

Coexists with V1 - your old models still work!

Quick Start:
    from darkorbit_bot.v2 import BotControllerV2, BotConfigV2

    config = BotConfigV2(policy_dir="path/to/trained/policy")
    bot = BotControllerV2(config)
    bot.start()

Training:
    python -m darkorbit_bot.v2.training.train_executor --data recordings/
    python -m darkorbit_bot.v2.training.train_tactician --data recordings/
    python -m darkorbit_bot.v2.training.train_strategist --data recordings/
"""

__version__ = "2.0.0"

# Main controller
from .bot_controller_v2 import BotControllerV2, BotConfigV2

# Models
from .models.unified import (
    HierarchicalPolicy,
    HierarchicalState,
    create_hierarchical_policy,
    load_hierarchical_policy,
    save_hierarchical_policy
)
from .models.executor import Executor, ExecutorV2, create_executor, load_executor, save_executor
from .models.tactician import Tactician, create_tactician, load_tactician, save_tactician
from .models.strategist import Strategist, create_strategist, load_strategist, save_strategist

# Perception
from .perception.tracker import ObjectTracker, TrackedObject
from .perception.state_encoder import StateEncoderV2, StateSummarizer, PlayerState

# Config
from .config import (
    V2Config,
    PerceptionConfig,
    StrategistConfig,
    TacticianConfig,
    ExecutorConfig,
    TrainingConfig,
    # Enums and constants
    Mode,
    ENEMY_CLASSES,
    LOOT_CLASSES,
    PLAYER_CLASSES,
    IGNORE_CLASSES,
)

__all__ = [
    # Version
    '__version__',
    # Controller
    'BotControllerV2',
    'BotConfigV2',
    # Models
    'HierarchicalPolicy',
    'HierarchicalState',
    'create_hierarchical_policy',
    'load_hierarchical_policy',
    'save_hierarchical_policy',
    'Executor',
    'ExecutorV2',
    'create_executor',
    'load_executor',
    'save_executor',
    'Tactician',
    'create_tactician',
    'load_tactician',
    'save_tactician',
    'Strategist',
    'create_strategist',
    'load_strategist',
    'save_strategist',
    # Perception
    'ObjectTracker',
    'TrackedObject',
    'StateEncoderV2',
    'StateSummarizer',
    'PlayerState',
    # Config
    'V2Config',
    'PerceptionConfig',
    'StrategistConfig',
    'TacticianConfig',
    'ExecutorConfig',
    'TrainingConfig',
    # Enums and constants
    'Mode',
    'ENEMY_CLASSES',
    'LOOT_CLASSES',
    'PLAYER_CLASSES',
    'IGNORE_CLASSES',
]
