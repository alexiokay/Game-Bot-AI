"""
DarkOrbit Bot - Safe Trainer

Training loop with VRAM safety for RTX 5070 Ti.
Features:
- Button-triggered training (not auto-background)
- Small batch sizes to prevent VRAM spikes
- Training progress logging
- Model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import threading

# Local imports
try:
    from .policy_network import DualHeadPolicy, create_policy, save_policy
    from .filters import BufferFrame, GaussianSmoother
except ImportError:
    from policy_network import DualHeadPolicy, create_policy, save_policy
    from filters import BufferFrame, GaussianSmoother


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs_per_session: int = 10
    sequence_length: int = 50
    max_vram_gb: float = 8.0  # Leave headroom for game
    checkpoint_interval: int = 100  # Steps between checkpoints
    device: str = "cuda"


class BehavioralDataset(Dataset):
    """
    Dataset of filtered behavioral sequences.
    
    Each item is a (state_sequence, action, mode) tuple.
    """
    
    def __init__(self, data_dir: str, sequence_length: int = 50):
        self.sequence_length = sequence_length
        self.samples: List[Dict] = []
        
        self._load_data(data_dir)
        
    def _load_data(self, data_dir: str):
        """Load all saved successful sequences"""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            print(f"⚠️ Data directory not found: {data_dir}")
            return
            
        # Look for sequence files
        for seq_file in data_path.glob("**/sequence_*.json"):
            try:
                with open(seq_file, 'r') as f:
                    seq_data = json.load(f)
                    
                # Only load "SUCCESS" labeled sequences
                if seq_data.get('label') == 'SUCCESS':
                    self.samples.append({
                        'states': np.array(seq_data['states'], dtype=np.float32),
                        'actions': np.array(seq_data['actions'], dtype=np.float32),
                        'mode': seq_data.get('mode', 'PASSIVE')
                    })
            except Exception as e:
                print(f"Error loading {seq_file}: {e}")
                
        print(f"📊 Loaded {len(self.samples)} successful sequences")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[idx]
        states = torch.tensor(sample['states'], dtype=torch.float32)
        actions = torch.tensor(sample['actions'], dtype=torch.float32)
        mode = sample['mode']
        return states, actions, mode


class SafeTrainer:
    """
    VRAM-safe trainer for the policy network.
    
    Features:
    - Manual trigger (button press) to start training
    - Small batches to prevent VRAM spikes
    - Background training thread
    - Progress logging and checkpoints
    """
    
    def __init__(self, 
                 policy: DualHeadPolicy,
                 config: TrainingConfig = None):
        self.policy = policy
        self.config = config or TrainingConfig()
        
        # Optimizer
        self.optimizer = optim.Adam(
            policy.parameters(), 
            lr=self.config.learning_rate
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training state
        self.is_training = False
        self.training_thread: Optional[threading.Thread] = None
        self.total_steps = 0
        self.total_loss = 0.0
        
        # Data buffer (in RAM, not GPU)
        self.data_buffer: List[Dict] = []
        
        # Paths
        self.checkpoint_dir = Path("data/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def add_sequence(self, 
                    states: np.ndarray, 
                    actions: np.ndarray, 
                    mode: str,
                    apply_smoothing: bool = True):
        """
        Add a successful sequence to the training buffer (RAM).
        
        Args:
            states: State sequence array
            actions: Action sequence array
            mode: "PASSIVE" or "AGGRESSIVE"
            apply_smoothing: Apply Gaussian smoothing to mouse movements
        """
        if apply_smoothing:
            # Smooth the mouse components of actions
            smoother = GaussianSmoother(sigma=3.0)
            mouse_coords = actions[:, :2]  # First 2 columns are mouse x, y
            smoothed = smoother.smooth_path(mouse_coords)
            actions[:, :2] = smoothed
            
        self.data_buffer.append({
            'states': states,
            'actions': actions,
            'mode': mode
        })
        
        print(f"📝 Added {mode} sequence to buffer. Total: {len(self.data_buffer)}")
        
    def get_buffer_size(self) -> int:
        """Get number of sequences in buffer"""
        return len(self.data_buffer)
    
    def get_vram_usage(self) -> float:
        """Get current VRAM usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    def can_train(self) -> bool:
        """Check if safe to train (VRAM check)"""
        vram = self.get_vram_usage()
        return vram < self.config.max_vram_gb and len(self.data_buffer) > 0
    
    def start_training(self, epochs: int = None):
        """
        Start training in background thread.
        Call this on button press.
        """
        if self.is_training:
            print("⚠️ Training already in progress")
            return False
            
        if not self.can_train():
            print(f"⚠️ Cannot train: VRAM={self.get_vram_usage():.1f}GB, Buffer={len(self.data_buffer)}")
            return False
            
        epochs = epochs or self.config.epochs_per_session
        
        self.is_training = True
        self.training_thread = threading.Thread(
            target=self._training_loop,
            args=(epochs,),
            daemon=True
        )
        self.training_thread.start()
        
        print(f"🚀 Training started ({epochs} epochs)")
        return True
    
    def stop_training(self):
        """Stop training"""
        self.is_training = False
        if self.training_thread:
            self.training_thread.join(timeout=5.0)
        print("⏹️ Training stopped")
        
    def _training_loop(self, epochs: int):
        """Main training loop (runs in background thread)"""
        self.policy.train()
        
        try:
            for epoch in range(epochs):
                if not self.is_training:
                    break
                    
                epoch_loss = 0.0
                epoch_steps = 0
                
                # Shuffle buffer
                np.random.shuffle(self.data_buffer)
                
                # Process in batches
                for i in range(0, len(self.data_buffer), self.config.batch_size):
                    if not self.is_training:
                        break
                        
                    batch = self.data_buffer[i:i + self.config.batch_size]
                    loss = self._train_batch(batch)
                    
                    epoch_loss += loss
                    epoch_steps += 1
                    self.total_steps += 1
                    
                    # Checkpoint
                    if self.total_steps % self.config.checkpoint_interval == 0:
                        self._save_checkpoint()
                        
                    # VRAM safety check
                    if self.get_vram_usage() > self.config.max_vram_gb:
                        print("⚠️ VRAM limit reached, pausing...")
                        torch.cuda.empty_cache()
                        time.sleep(1.0)
                        
                avg_loss = epoch_loss / max(epoch_steps, 1)
                self.total_loss = avg_loss
                print(f"📈 Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
                
        except Exception as e:
            print(f"❌ Training error: {e}")
        finally:
            self.is_training = False
            self._save_checkpoint()
            self.policy.eval()
            print("✅ Training complete")
    
    def _train_batch(self, batch: List[Dict]) -> float:
        """
        Train on a single batch using proper sequence learning.

        Now trains on ALL frames using sliding windows, not just last frame.
        Uses BCE loss for click prediction, MSE for positions.
        """
        self.optimizer.zero_grad()

        # Separate by mode
        passive_states, passive_targets = [], []
        aggressive_states, aggressive_targets = [], []

        sequence_length = 50

        for sample in batch:
            states = sample['states']
            actions = sample['actions']
            mode = sample['mode']

            # Create sliding window samples - train on EVERY frame
            for i in range(sequence_length, len(states)):
                state_seq = torch.tensor(states[i - sequence_length:i], dtype=torch.float32)
                target = torch.tensor(actions[i - 1], dtype=torch.float32)

                if mode == 'AGGRESSIVE':
                    aggressive_states.append(state_seq)
                    aggressive_targets.append(target)
                else:
                    passive_states.append(state_seq)
                    passive_targets.append(target)

        total_loss = 0.0
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()

        # Train passive samples
        if passive_states:
            states_batch = torch.stack(passive_states).to(self.config.device)
            targets_batch = torch.stack(passive_targets).to(self.config.device)

            pred = self.policy(states_batch, mode="PASSIVE")

            # Position loss + click loss + other loss
            pos_loss = mse_loss(pred[:, :2], targets_batch[:, :2])
            click_loss = bce_loss(pred[:, 2], targets_batch[:, 2])
            other_loss = mse_loss(pred[:, 3:], targets_batch[:, 3:])

            loss = pos_loss + click_loss * 2.0 + other_loss
            loss.backward()
            total_loss += loss.item()

        # Train aggressive samples
        if aggressive_states:
            states_batch = torch.stack(aggressive_states).to(self.config.device)
            targets_batch = torch.stack(aggressive_targets).to(self.config.device)

            pred = self.policy(states_batch, mode="AGGRESSIVE")

            pos_loss = mse_loss(pred[:, :2], targets_batch[:, :2])
            fire_loss = bce_loss(pred[:, 2], targets_batch[:, 2])
            other_loss = mse_loss(pred[:, 3:], targets_batch[:, 3:])

            loss = pos_loss + fire_loss * 2.0 + other_loss
            loss.backward()
            total_loss += loss.item()

        self.optimizer.step()

        num_samples = len(passive_states) + len(aggressive_states)
        return total_loss / max(num_samples, 1)
    
    def _save_checkpoint(self):
        """Save model checkpoint"""
        path = self.checkpoint_dir / f"policy_step_{self.total_steps}.pt"
        save_policy(self.policy, str(path))
        
        # Also save as "latest"
        latest_path = self.checkpoint_dir / "policy_latest.pt"
        save_policy(self.policy, str(latest_path))
        
    def get_training_status(self) -> Dict:
        """Get current training status"""
        return {
            'is_training': self.is_training,
            'total_steps': self.total_steps,
            'loss': self.total_loss,
            'buffer_size': len(self.data_buffer),
            'vram_usage': self.get_vram_usage()
        }


class TrainingController:
    """
    High-level controller for training.
    Handles keyboard shortcuts and status display.
    """
    
    def __init__(self, trainer: SafeTrainer):
        self.trainer = trainer
        self.hotkey_registered = False
        
    def register_hotkeys(self):
        """Register training hotkeys"""
        try:
            import keyboard
            
            # F9 = Start training
            keyboard.add_hotkey('f9', self._on_start_training)
            
            # F10 = Stop training
            keyboard.add_hotkey('f10', self._on_stop_training)
            
            # F11 = Show status
            keyboard.add_hotkey('f11', self._on_show_status)
            
            self.hotkey_registered = True
            print("🎮 Training hotkeys registered:")
            print("   F9  = Start training")
            print("   F10 = Stop training")
            print("   F11 = Show status")
            
        except ImportError:
            print("⚠️ 'keyboard' module not installed. Hotkeys disabled.")
            print("   Install with: pip install keyboard")
            
    def _on_start_training(self):
        print("\n🔥 F9 pressed - Starting training...")
        self.trainer.start_training()
        
    def _on_stop_training(self):
        print("\n⏹️ F10 pressed - Stopping training...")
        self.trainer.stop_training()
        
    def _on_show_status(self):
        status = self.trainer.get_training_status()
        print(f"\n📊 Training Status:")
        print(f"   Active: {status['is_training']}")
        print(f"   Steps: {status['total_steps']}")
        print(f"   Loss: {status['loss']:.4f}")
        print(f"   Buffer: {status['buffer_size']} sequences")
        print(f"   VRAM: {status['vram_usage']:.2f} GB")


def create_trainer(state_size: int = 128, device: str = "cuda") -> Tuple[SafeTrainer, TrainingController]:
    """
    Create trainer and controller.
    
    Returns:
        (trainer, controller) tuple
    """
    policy = create_policy(state_size=state_size, device=device)
    trainer = SafeTrainer(policy)
    controller = TrainingController(trainer)
    
    return trainer, controller
