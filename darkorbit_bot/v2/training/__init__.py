"""
V2 Training Module

Training scripts for each component:
- Executor: Behavior cloning from human recordings
- Tactician: Target selection from human target choices
- Strategist: Mode selection from human mode patterns
- Full: End-to-end training

Usage:
    # Train each component separately
    python -m darkorbit_bot.v2.training.train_executor --data recordings/
    python -m darkorbit_bot.v2.training.train_tactician --data recordings/
    python -m darkorbit_bot.v2.training.train_strategist --data recordings/

    # Or train all at once
    python -m darkorbit_bot.v2.training.train_full --data recordings/
"""

from .train_executor import train_executor, ExecutorDataset, ExecutorLoss
from .train_tactician import train_tactician, TacticianDataset, TacticianLoss
from .train_strategist import train_strategist, StrategistDataset, StrategistLoss
from .train_full import train_full
from .training_utils import (
    TrainingLogger,
    CheckpointManager,
    WarmupCosineScheduler,
    StepScheduler,
    PrioritizedReplayBuffer,
    compute_score,
    gradient_norm
)

__all__ = [
    'train_full',
    'train_executor',
    'train_tactician',
    'train_strategist',
    'ExecutorDataset',
    'ExecutorLoss',
    'TacticianDataset',
    'TacticianLoss',
    'StrategistDataset',
    'StrategistLoss',
    # Training utilities
    'TrainingLogger',
    'CheckpointManager',
    'WarmupCosineScheduler',
    'StepScheduler',
    'PrioritizedReplayBuffer',
    'compute_score',
    'gradient_norm'
]
