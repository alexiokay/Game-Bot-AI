"""
V2 Neural Network Models

Hierarchical architecture:
- Strategist: Goal selection (Transformer)
- Tactician: Target selection (Cross-Attention)
- Executor: Motor control (Mamba/LSTM fallback)
"""

from .executor import Executor, ExecutorLSTM
from .tactician import Tactician
from .strategist import Strategist
from .unified import HierarchicalPolicy

__all__ = [
    'Executor', 'ExecutorLSTM',
    'Tactician',
    'Strategist',
    'HierarchicalPolicy'
]
