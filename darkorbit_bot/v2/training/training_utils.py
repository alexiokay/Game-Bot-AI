"""
Training Utilities - Industry Standard ML Infrastructure

- Tensorboard logging for visualization
- Smart checkpoint management with versioning
- Learning rate scheduling with warmup
- Gradient monitoring
"""

import os
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

# Try to import tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("Tensorboard not installed. Run: pip install tensorboard")


# ═══════════════════════════════════════════════════════════════════════════════
# TENSORBOARD LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingLogger:
    """
    Centralized logging for training metrics.

    Supports:
    - Tensorboard visualization
    - Console output
    - JSON file logging for later analysis

    Usage:
        logger = TrainingLogger("runs/experiment_1")
        logger.log_scalar("loss", 0.5, step=100)
        logger.log_scalars("rewards", {"hit": 1.0, "miss": -0.3}, step=100)
    """

    def __init__(self,
                 log_dir: str = "runs",
                 experiment_name: Optional[str] = None,
                 use_tensorboard: bool = True,
                 log_to_file: bool = True):
        """
        Args:
            log_dir: Base directory for logs
            experiment_name: Name for this run (auto-generated if None)
            use_tensorboard: Whether to use tensorboard
            log_to_file: Whether to also log to JSON file
        """
        # Generate experiment name with timestamp
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"train_{timestamp}"

        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Tensorboard writer
        self.writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(str(self.log_dir))
            logger.info(f"Tensorboard logging to: {self.log_dir}")
            print(f"   [LOG] Tensorboard: tensorboard --logdir={log_dir}")

        # JSON file logging
        self.log_to_file = log_to_file
        if log_to_file:
            self.json_log_path = self.log_dir / "metrics.jsonl"
            self._json_file = open(self.json_log_path, "a")

        # Track global step
        self.global_step = 0

        # Gradient statistics
        self._grad_norms: List[float] = []
        self._clip_count = 0
        self._total_updates = 0

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log a single scalar value."""
        step = step if step is not None else self.global_step

        if self.writer:
            self.writer.add_scalar(tag, value, step)

        if self.log_to_file:
            self._write_json({"tag": tag, "value": value, "step": step})

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float],
                    step: Optional[int] = None):
        """Log multiple related scalars."""
        step = step if step is not None else self.global_step

        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

        if self.log_to_file:
            self._write_json({
                "tag": main_tag,
                "values": tag_scalar_dict,
                "step": step
            })

    def log_histogram(self, tag: str, values: torch.Tensor, step: Optional[int] = None):
        """Log histogram of values (e.g., weights, gradients)."""
        step = step if step is not None else self.global_step

        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_gradients(self, model: nn.Module, step: Optional[int] = None):
        """Log gradient statistics for a model."""
        step = step if step is not None else self.global_step

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        self._grad_norms.append(total_norm)
        self._total_updates += 1

        if self.writer:
            self.writer.add_scalar("Gradients/total_norm", total_norm, step)

        # Track if clipping would be triggered (assuming clip_value=1.0)
        if total_norm > 1.0:
            self._clip_count += 1

        # Log clip rate every 100 updates
        if self._total_updates % 100 == 0 and self._total_updates > 0:
            clip_rate = self._clip_count / self._total_updates
            if self.writer:
                self.writer.add_scalar("Gradients/clip_rate", clip_rate, step)
            if clip_rate > 0.5:
                logger.warning(f"High gradient clipping rate: {clip_rate:.1%}")

    def log_training_update(self,
                           loss: float,
                           metrics: Dict[str, float],
                           step: Optional[int] = None):
        """Log a complete training update with all metrics."""
        step = step if step is not None else self.global_step

        # Log loss
        self.log_scalar("Loss/train", loss, step)

        # Log all metrics
        for name, value in metrics.items():
            self.log_scalar(f"Metrics/{name}", value, step)

        self.global_step = step + 1

    def log_episode(self,
                   episode_num: int,
                   episode_return: float,
                   episode_length: int,
                   metrics: Optional[Dict[str, float]] = None):
        """Log episode-level statistics."""
        if self.writer:
            self.writer.add_scalar("Episode/return", episode_return, episode_num)
            self.writer.add_scalar("Episode/length", episode_length, episode_num)

            if metrics:
                for name, value in metrics.items():
                    self.writer.add_scalar(f"Episode/{name}", value, episode_num)

    def _write_json(self, data: Dict):
        """Write a JSON line to the log file."""
        if self._json_file:
            data["timestamp"] = time.time()
            self._json_file.write(json.dumps(data) + "\n")
            self._json_file.flush()

    def step(self):
        """Increment global step."""
        self.global_step += 1

    def close(self):
        """Close all resources."""
        if self.writer:
            self.writer.close()
        if self.log_to_file and hasattr(self, '_json_file'):
            self._json_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SMART CHECKPOINT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CheckpointMetadata:
    """Metadata stored with each checkpoint."""
    version: str
    timestamp: str
    step: int
    loss: float
    score: float  # Combined metric for ranking
    metrics: Dict[str, float]
    config: Dict[str, Any]

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "CheckpointMetadata":
        return cls(**d)


class CheckpointManager:
    """
    Smart checkpoint management with versioning and auto-selection.

    Features:
    - Keeps top N checkpoints by score
    - Automatic best model selection
    - Rollback capability
    - Metadata tracking

    Usage:
        manager = CheckpointManager("checkpoints/executor", keep_top_n=5)

        # Save checkpoint
        manager.save(model, optimizer, score=0.85, metrics={"hit_rate": 0.9})

        # Load best
        manager.load_best(model, optimizer)

        # Rollback to previous
        manager.rollback(model, optimizer)
    """

    def __init__(self,
                 checkpoint_dir: str,
                 model_name: str = "model",
                 keep_top_n: int = 5,
                 keep_latest_n: int = 3):
        """
        Args:
            checkpoint_dir: Directory for checkpoints
            model_name: Base name for checkpoint files
            keep_top_n: Keep top N checkpoints by score
            keep_latest_n: Also keep N most recent (regardless of score)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.keep_top_n = keep_top_n
        self.keep_latest_n = keep_latest_n

        # Track all checkpoints
        self.checkpoints: List[Dict] = []
        self._load_checkpoint_index()

        # Track current version
        self.current_version = len(self.checkpoints)

    def _load_checkpoint_index(self):
        """Load existing checkpoint index."""
        index_path = self.checkpoint_dir / "checkpoint_index.json"
        if index_path.exists():
            with open(index_path) as f:
                self.checkpoints = json.load(f)

    def _save_checkpoint_index(self):
        """Save checkpoint index."""
        index_path = self.checkpoint_dir / "checkpoint_index.json"
        with open(index_path, "w") as f:
            json.dump(self.checkpoints, f, indent=2)

    def save(self,
             model: nn.Module,
             optimizer: torch.optim.Optimizer,
             score: float,
             step: int = 0,
             loss: float = 0.0,
             metrics: Optional[Dict[str, float]] = None,
             config: Optional[Dict] = None) -> str:
        """
        Save a checkpoint with metadata.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            score: Score for ranking (higher = better)
            step: Training step
            loss: Current loss
            metrics: Additional metrics
            config: Training configuration

        Returns:
            Path to saved checkpoint
        """
        self.current_version += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create metadata
        metadata = CheckpointMetadata(
            version=f"v{self.current_version}",
            timestamp=timestamp,
            step=step,
            loss=loss,
            score=score,
            metrics=metrics or {},
            config=config or {}
        )

        # Checkpoint filename
        filename = f"{self.model_name}_v{self.current_version}_{score:.4f}.pt"
        filepath = self.checkpoint_dir / filename

        # Save checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata.to_dict()
        }, filepath)

        # Update index
        self.checkpoints.append({
            "filename": filename,
            "score": score,
            "timestamp": timestamp,
            "version": self.current_version,
            "step": step
        })

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

        # Save index
        self._save_checkpoint_index()

        logger.info(f"Checkpoint saved: {filename} (score={score:.4f})")
        return str(filepath)

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping top N by score and latest N."""
        if len(self.checkpoints) <= self.keep_top_n + self.keep_latest_n:
            return

        # Sort by score (descending)
        by_score = sorted(self.checkpoints, key=lambda x: x["score"], reverse=True)
        top_by_score = set(c["filename"] for c in by_score[:self.keep_top_n])

        # Sort by timestamp (descending)
        by_time = sorted(self.checkpoints, key=lambda x: x["timestamp"], reverse=True)
        latest = set(c["filename"] for c in by_time[:self.keep_latest_n])

        # Keep checkpoints in either set
        keep = top_by_score | latest

        # Remove others
        to_remove = []
        for ckpt in self.checkpoints:
            if ckpt["filename"] not in keep:
                filepath = self.checkpoint_dir / ckpt["filename"]
                if filepath.exists():
                    filepath.unlink()
                    logger.debug(f"Removed old checkpoint: {ckpt['filename']}")
                to_remove.append(ckpt)

        for ckpt in to_remove:
            self.checkpoints.remove(ckpt)

    def load_best(self, model: nn.Module,
                  optimizer: Optional[torch.optim.Optimizer] = None,
                  device: str = "cuda") -> Optional[CheckpointMetadata]:
        """Load the best checkpoint by score."""
        if not self.checkpoints:
            logger.warning("No checkpoints available")
            return None

        # Find best by score
        best = max(self.checkpoints, key=lambda x: x["score"])
        return self._load_checkpoint(best["filename"], model, optimizer, device)

    def load_latest(self, model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    device: str = "cuda") -> Optional[CheckpointMetadata]:
        """Load the most recent checkpoint."""
        if not self.checkpoints:
            logger.warning("No checkpoints available")
            return None

        # Find latest by timestamp
        latest = max(self.checkpoints, key=lambda x: x["timestamp"])
        return self._load_checkpoint(latest["filename"], model, optimizer, device)

    def load_version(self, version: int, model: nn.Module,
                     optimizer: Optional[torch.optim.Optimizer] = None,
                     device: str = "cuda") -> Optional[CheckpointMetadata]:
        """Load a specific version."""
        for ckpt in self.checkpoints:
            if ckpt["version"] == version:
                return self._load_checkpoint(ckpt["filename"], model, optimizer, device)

        logger.warning(f"Version {version} not found")
        return None

    def _load_checkpoint(self, filename: str, model: nn.Module,
                         optimizer: Optional[torch.optim.Optimizer],
                         device: str) -> Optional[CheckpointMetadata]:
        """Load a checkpoint by filename."""
        filepath = self.checkpoint_dir / filename
        if not filepath.exists():
            logger.error(f"Checkpoint not found: {filepath}")
            return None

        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        metadata = CheckpointMetadata.from_dict(checkpoint.get("metadata", {}))
        logger.info(f"Loaded checkpoint: {filename} (score={metadata.score:.4f})")

        return metadata

    def rollback(self, model: nn.Module,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 steps_back: int = 1,
                 device: str = "cuda") -> Optional[CheckpointMetadata]:
        """Rollback to a previous checkpoint."""
        if len(self.checkpoints) <= steps_back:
            logger.warning(f"Not enough checkpoints to rollback {steps_back} steps")
            return None

        # Sort by version descending
        by_version = sorted(self.checkpoints, key=lambda x: x["version"], reverse=True)
        target = by_version[steps_back]

        return self._load_checkpoint(target["filename"], model, optimizer, device)

    def get_best_score(self) -> float:
        """Get the best score among all checkpoints."""
        if not self.checkpoints:
            return 0.0
        return max(c["score"] for c in self.checkpoints)

    def list_checkpoints(self) -> List[Dict]:
        """List all checkpoints with their metadata."""
        return sorted(self.checkpoints, key=lambda x: x["score"], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULING
# ═══════════════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing.

    - Warmup: Linear increase from 0 to base_lr over warmup_steps
    - Cosine: Decay from base_lr to min_lr over remaining steps

    Usage:
        scheduler = WarmupCosineScheduler(optimizer, base_lr=1e-4, warmup_steps=100)

        for step in range(total_steps):
            scheduler.step()
            # training...
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 base_lr: float,
                 warmup_steps: int = 100,
                 total_steps: int = 10000,
                 min_lr: float = 1e-6):
        """
        Args:
            optimizer: The optimizer to schedule
            base_lr: Target learning rate after warmup
            warmup_steps: Number of warmup steps
            total_steps: Total training steps (for cosine decay)
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        self.current_step = 0
        self._last_lr = 0.0

        # Initialize with warmup starting LR
        self._set_lr(0.0)

    def step(self):
        """Update learning rate for current step."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

        self._set_lr(lr)

    def _set_lr(self, lr: float):
        """Set learning rate for all param groups."""
        self._last_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self._last_lr

    def state_dict(self) -> Dict:
        """Get scheduler state for checkpointing."""
        return {
            "current_step": self.current_step,
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr
        }

    def load_state_dict(self, state: Dict):
        """Load scheduler state from checkpoint."""
        self.current_step = state["current_step"]
        self.base_lr = state["base_lr"]
        self.warmup_steps = state["warmup_steps"]
        self.total_steps = state["total_steps"]
        self.min_lr = state["min_lr"]
        # Restore current LR
        self.step()
        self.current_step -= 1  # Undo the step increment


class StepScheduler:
    """
    Simple step-based LR decay.

    Multiplies LR by gamma every step_size steps.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 base_lr: float,
                 step_size: int = 1000,
                 gamma: float = 0.9,
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr

        self.current_step = 0
        self._set_lr(base_lr)

    def step(self):
        """Update learning rate."""
        self.current_step += 1

        if self.current_step % self.step_size == 0:
            current_lr = self.optimizer.param_groups[0]["lr"]
            new_lr = max(self.min_lr, current_lr * self.gamma)
            self._set_lr(new_lr)

    def _set_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ═══════════════════════════════════════════════════════════════════════════════
# PRIORITIZED EXPERIENCE REPLAY (Bonus)
# ═══════════════════════════════════════════════════════════════════════════════

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.

    Samples experiences proportional to their priority (e.g., TD error).
    Uses sum-tree for efficient O(log n) sampling.

    Usage:
        buffer = PrioritizedReplayBuffer(max_size=10000)
        buffer.add(experience, priority=1.0)
        batch, indices, weights = buffer.sample(batch_size=32)
        # After computing TD errors:
        buffer.update_priorities(indices, new_priorities)
    """

    def __init__(self,
                 max_size: int = 10000,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001):
        """
        Args:
            max_size: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full priority)
            beta: Importance sampling exponent (annealed to 1)
            beta_increment: How much to increase beta each sample
        """
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        # Use simple list + priority array (not full sum-tree for simplicity)
        self.buffer: List[Any] = []
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, experience: Any, priority: Optional[float] = None):
        """Add experience with priority."""
        if priority is None:
            # Default to max priority for new experiences
            priority = self.priorities[:self.size].max() if self.size > 0 else 1.0

        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> tuple:
        """
        Sample a batch with priorities.

        Returns:
            (experiences, indices, importance_weights)
        """
        if self.size < batch_size:
            batch_size = self.size

        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # Compute importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize

        experiences = [self.buffer[i] for i in indices]

        return experiences, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-6) ** self.alpha

    def __len__(self):
        return self.size


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_score(metrics: Dict[str, float],
                  weights: Optional[Dict[str, float]] = None) -> float:
    """
    Compute a combined score from metrics for checkpoint ranking.

    Default weights prioritize hit_rate > survival > efficiency.
    """
    if weights is None:
        weights = {
            "hit_rate": 0.4,
            "click_accuracy": 0.2,
            "survival_rate": 0.2,
            "position_error": -0.2,  # Lower is better
        }

    score = 0.0
    for metric, weight in weights.items():
        if metric in metrics:
            score += weight * metrics[metric]

    return score


def gradient_norm(model: nn.Module) -> float:
    """Compute total gradient norm for a model."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5
