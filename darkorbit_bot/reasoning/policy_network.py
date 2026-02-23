"""
DarkOrbit Bot - Bi-LSTM Policy Network

Dual-head behavioral policy that learns from filtered gameplay:
- PASSIVE HEAD: Looting, navigation, exploration
- AGGRESSIVE HEAD: Combat, targeting, pursuit

The network processes sequences of game states and outputs actions
that mimic your "best" gameplay moments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


class DualHeadPolicy(nn.Module):
    """
    Bi-directional LSTM with separate heads for Passive and Aggressive behavior.
    
    Architecture:
    ┌─────────────────────────────────────────┐
    │         Shared Feature Extractor        │
    │  (Bi-LSTM processes state sequences)    │
    └─────────────────┬───────────────────────┘
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
    ┌───────────────┐   ┌───────────────┐
    │ Passive Head  │   │ Aggressive    │
    │ (5 outputs)   │   │ Head (5 out)  │
    └───────────────┘   └───────────────┘
    """
    
    def __init__(self, 
                 input_size: int = 128,  # State vector size
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            input_size: Size of each state vector
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout rate between layers
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Shared Bi-LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer (focus on important frames)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Passive Head - for looting, navigation, exploration
        # Outputs: [move_x, move_y, should_click, is_enemy, distance, ctrl, space, shift]
        self.passive_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8)  # Extended to 8 outputs including keyboard actions
        )

        # Aggressive Head - for combat, targeting, pursuit
        # Outputs: [aim_x, aim_y, should_fire, is_enemy, distance, ctrl, space, shift]
        self.aggressive_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8)  # Extended to 8 outputs including keyboard actions
        )

        # Mode Selector Head - learns WHEN to be aggressive vs passive from YOUR gameplay
        # Output: single value, >0.5 means AGGRESSIVE, <0.5 means PASSIVE
        self.mode_selector = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, 
               state_sequence: torch.Tensor, 
               mode: str = "PASSIVE") -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state_sequence: Tensor of shape (batch, sequence_length, input_size)
            mode: "PASSIVE" or "AGGRESSIVE"
            
        Returns:
            Action tensor of shape (batch, 5)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(state_sequence)
        # lstm_out shape: (batch, seq_len, hidden_size * 2)
        
        # Apply attention
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden_size * 2)
        
        # Select head based on mode
        if mode == "AGGRESSIVE":
            return self.aggressive_head(context)
        else:
            return self.passive_head(context)
    
    def forward_both(self, state_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get outputs from both heads (for visualization/debugging).

        Returns:
            (passive_output, aggressive_output)
        """
        lstm_out, _ = self.lstm(state_sequence)
        attention_weights = self.attention(lstm_out)
        context = torch.sum(lstm_out * attention_weights, dim=1)

        return self.passive_head(context), self.aggressive_head(context)

    def forward_auto(self, state_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Auto-select mode based on learned behavior and return action.

        The mode selector learns from YOUR gameplay when you were aggressive
        (attacking enemies) vs passive (looting, navigating). This removes
        the need for hardcoded mode switching logic.

        Returns:
            (action_output, mode_confidence, predicted_mode)
        """
        lstm_out, _ = self.lstm(state_sequence)
        attention_weights = self.attention(lstm_out)
        context = torch.sum(lstm_out * attention_weights, dim=1)

        # Get mode prediction from learned selector
        mode_conf = self.mode_selector(context)  # (batch, 1), 0-1 range

        # Select head based on learned mode
        # >0.5 = AGGRESSIVE, <0.5 = PASSIVE
        is_aggressive = mode_conf > 0.5

        # Get outputs from both heads
        passive_out = self.passive_head(context)
        aggressive_out = self.aggressive_head(context)

        # Use mode prediction to select output (differentiable soft selection for training)
        action_out = mode_conf * aggressive_out + (1 - mode_conf) * passive_out

        predicted_mode = "AGGRESSIVE" if is_aggressive.item() else "PASSIVE"

        return action_out, mode_conf, predicted_mode

    def predict_mode(self, state_sequence: torch.Tensor) -> Tuple[float, str]:
        """
        Get just the mode prediction without computing actions.

        Returns:
            (confidence, mode_string) where confidence is 0-1
        """
        with torch.no_grad():
            lstm_out, _ = self.lstm(state_sequence)
            attention_weights = self.attention(lstm_out)
            context = torch.sum(lstm_out * attention_weights, dim=1)
            mode_conf = self.mode_selector(context)

            conf_value = mode_conf.item()
            mode_str = "AGGRESSIVE" if conf_value > 0.5 else "PASSIVE"
            return conf_value, mode_str

    def get_action(self, 
                  state_sequence: np.ndarray, 
                  mode: str = "PASSIVE",
                  device: str = "cuda") -> Dict:
        """
        Convenience method to get action from numpy state.
        
        Args:
            state_sequence: Numpy array of shape (sequence_length, input_size)
            mode: "PASSIVE" or "AGGRESSIVE"
            device: "cuda" or "cpu"
            
        Returns:
            Dict with action components
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension
            x = torch.tensor(state_sequence, dtype=torch.float32).unsqueeze(0)
            x = x.to(device)
            
            output = self.forward(x, mode)
            output = output.cpu().numpy()[0]
            
        if mode == "AGGRESSIVE":
            return {
                'aim_x': float(np.clip(output[0], 0, 1)),
                'aim_y': float(np.clip(output[1], 0, 1)),
                'should_fire': output[2] > 0.0,  # Using sigmoid in training, so threshold at 0
                'raw_fire_value': float(output[2]),  # Raw value for debugging
                'is_enemy': output[3] > 0.5,
                'distance': float(output[4]),
                # Keyboard actions (learned from recordings)
                'ctrl_attack': output[5] > 0.0 if len(output) > 5 else False,
                'space_rocket': output[6] > 0.0 if len(output) > 6 else False,
                'shift_special': output[7] > 0.0 if len(output) > 7 else False,
            }
        else:
            return {
                'move_x': float(np.clip(output[0], 0, 1)),
                'move_y': float(np.clip(output[1], 0, 1)),
                'should_click': output[2] > 0.0,  # Using sigmoid in training, so threshold at 0
                'raw_click_value': float(output[2]),  # Raw value for debugging
                'is_enemy': output[3] > 0.5,
                'distance': float(output[4]),
                # Keyboard actions (learned from recordings)
                'ctrl_attack': output[5] > 0.0 if len(output) > 5 else False,
                'space_rocket': output[6] > 0.0 if len(output) > 6 else False,
                'shift_special': output[7] > 0.0 if len(output) > 7 else False,
            }

    def get_action_auto(self,
                        state_sequence: np.ndarray,
                        device: str = "cuda") -> Dict:
        """
        Get action using learned mode selection - no manual mode needed!

        The network learns from YOUR gameplay when to be aggressive vs passive.
        This replaces hardcoded logic like "if enemy detected: mode = AGGRESSIVE".

        Args:
            state_sequence: Numpy array of shape (sequence_length, input_size)
            device: "cuda" or "cpu"

        Returns:
            Dict with action components + predicted mode info
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(state_sequence, dtype=torch.float32).unsqueeze(0)
            x = x.to(device)

            action_out, mode_conf, predicted_mode = self.forward_auto(x)
            output = action_out.cpu().numpy()[0]
            confidence = mode_conf.cpu().numpy()[0, 0]

        # Build result dict based on predicted mode
        result = {
            'predicted_mode': predicted_mode,
            'mode_confidence': float(confidence),
            'target_x': float(np.clip(output[0], 0, 1)),
            'target_y': float(np.clip(output[1], 0, 1)),
            'should_act': output[2] > 0.0,  # click or fire depending on mode
            'raw_act_value': float(output[2]),
            'is_enemy': output[3] > 0.5,
            'distance': float(output[4]),
            # Keyboard actions
            'ctrl_attack': output[5] > 0.0 if len(output) > 5 else False,
            'space_rocket': output[6] > 0.0 if len(output) > 6 else False,
            'shift_special': output[7] > 0.0 if len(output) > 7 else False,
        }

        # Add mode-specific aliases for clarity
        if predicted_mode == "AGGRESSIVE":
            result['aim_x'] = result['target_x']
            result['aim_y'] = result['target_y']
            result['should_fire'] = result['should_act']
        else:
            result['move_x'] = result['target_x']
            result['move_y'] = result['target_y']
            result['should_click'] = result['should_act']

        return result


class ActionEncoder:
    """
    Encodes/decodes actions for training.
    
    Converts raw mouse movements and key presses to normalized vectors
    that the network can learn.
    """
    
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Key to index mapping
        self.key_map = {
            '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
            '6': 6, '7': 7, '8': 8, '9': 9, '0': 10,
            'q': 11, 'w': 12, 'e': 13, 'r': 14,
            'space': 15, 'ctrl': 16, 'shift': 17
        }
        
    def encode_passive_action(self, 
                             mouse_x: float, 
                             mouse_y: float,
                             clicked: bool,
                             key_pressed: str = None,
                             wait_time: float = 0.0) -> np.ndarray:
        """
        Encode passive mode action to vector.
        
        Returns:
            np.array of shape (5,)
        """
        return np.array([
            mouse_x / self.screen_width,  # Normalize to 0-1
            mouse_y / self.screen_height,
            1.0 if clicked else 0.0,
            self.key_map.get(key_pressed, 0) / len(self.key_map),
            min(wait_time / 1.0, 1.0)  # Normalize wait time
        ], dtype=np.float32)
    
    def encode_aggressive_action(self,
                                aim_x: float,
                                aim_y: float,
                                fired: bool,
                                ability_key: str = None,
                                dodge_dir: float = 0.0) -> np.ndarray:
        """
        Encode aggressive mode action to vector.
        
        Returns:
            np.array of shape (5,)
        """
        return np.array([
            aim_x / self.screen_width,
            aim_y / self.screen_height,
            1.0 if fired else 0.0,
            self.key_map.get(ability_key, 0) / len(self.key_map),
            (dodge_dir + 1.0) / 2.0  # Normalize -1 to 1 -> 0 to 1
        ], dtype=np.float32)
    
    def decode_passive_action(self, action_vector: np.ndarray) -> Dict:
        """Decode passive action vector to usable values"""
        return {
            'mouse_x': int(action_vector[0] * self.screen_width),
            'mouse_y': int(action_vector[1] * self.screen_height),
            'click': action_vector[2] > 0.5,
            'key': self._index_to_key(int(action_vector[3] * len(self.key_map))),
            'wait': action_vector[4] * 1.0  # Max 1 second
        }
    
    def decode_aggressive_action(self, action_vector: np.ndarray) -> Dict:
        """Decode aggressive action vector to usable values"""
        return {
            'aim_x': int(action_vector[0] * self.screen_width),
            'aim_y': int(action_vector[1] * self.screen_height),
            'fire': action_vector[2] > 0.5,
            'ability': self._index_to_key(int(action_vector[3] * len(self.key_map))),
            'dodge': (action_vector[4] * 2.0) - 1.0  # Back to -1 to 1
        }
    
    def _index_to_key(self, index: int) -> Optional[str]:
        """Convert key index back to key string"""
        for key, idx in self.key_map.items():
            if idx == index:
                return key
        return None


def create_policy(state_size: int = 128, 
                 hidden_size: int = 256,
                 device: str = "cuda") -> DualHeadPolicy:
    """
    Create and initialize the policy network.
    
    Args:
        state_size: Size of state vectors (from StateBuilder)
        hidden_size: LSTM hidden size
        device: "cuda" or "cpu"
        
    Returns:
        Initialized DualHeadPolicy on specified device
    """
    policy = DualHeadPolicy(
        input_size=state_size,
        hidden_size=hidden_size
    )
    
    # Initialize weights
    for name, param in policy.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
            
    return policy.to(device)


def save_policy(policy: DualHeadPolicy, path: str):
    """Save policy to file"""
    # Get input_size from LSTM layer
    input_size = policy.lstm.input_size

    torch.save({
        'model_state_dict': policy.state_dict(),
        'hidden_size': policy.hidden_size,
        'num_layers': policy.num_layers,
        'input_size': input_size  # Save for compatibility checking
    }, path)
    print(f"✅ Policy saved to {path} (input_size={input_size})")
    

def load_policy(path: str, device: str = "cuda", expected_input_size: int = None) -> DualHeadPolicy:
    """
    Load policy from file with automatic migration.

    Handles:
    - 5-dim to 8-dim output migration (keyboard actions)
    - Mode selector head addition
    - Input size changes (movement pattern features)

    Args:
        path: Path to saved model
        device: "cuda" or "cpu"
        expected_input_size: If provided, verifies compatibility with current StateBuilder
    """
    checkpoint = torch.load(path, map_location=device)

    # Get input_size from checkpoint or infer from weights
    saved_input_size = checkpoint.get('input_size')
    if saved_input_size is None:
        # Old checkpoint without input_size - infer from LSTM weights
        lstm_weight = checkpoint['model_state_dict'].get('lstm.weight_ih_l0')
        if lstm_weight is not None:
            saved_input_size = lstm_weight.shape[1]
        else:
            saved_input_size = 128  # Default fallback

    # Warn if input size mismatch
    if expected_input_size is not None and saved_input_size != expected_input_size:
        print(f"⚠️ Input size mismatch: model={saved_input_size}, current={expected_input_size}")
        print(f"   Model was trained with different state features.")
        print(f"   Consider retraining or using StateBuilder(include_movement_patterns=False)")

    policy = DualHeadPolicy(
        input_size=saved_input_size,  # Use saved input size
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers']
    )

    # Check if this is an old 5-dim model that needs migration
    old_state = checkpoint['model_state_dict']
    new_state = policy.state_dict()

    # Find the output layer index (could be 4 or 5 depending on model version)
    # The output layer is the last Linear layer in each head
    output_layer_idx = None
    for key in old_state.keys():
        if 'passive_head' in key and 'weight' in key:
            # Extract layer index (e.g., "passive_head.4.weight" -> 4)
            parts = key.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                idx = int(parts[1])
                if output_layer_idx is None or idx > output_layer_idx:
                    output_layer_idx = idx

    # Check if the output layer has 5 dims (old format)
    old_passive_out = old_state.get(f'passive_head.{output_layer_idx}.weight')
    needs_migration = old_passive_out is not None and old_passive_out.shape[0] == 5

    if needs_migration:
        print(f"🔄 Migrating old 5-dim model to new 8-dim format (layer index {output_layer_idx})...")

        # Migrate each head's output layer
        for head in ['passive_head', 'aggressive_head']:
            old_weight = old_state[f'{head}.{output_layer_idx}.weight']  # [5, 64]
            old_bias = old_state[f'{head}.{output_layer_idx}.bias']      # [5]

            # Create new expanded weight/bias
            new_weight = new_state[f'{head}.{output_layer_idx}.weight'].clone()  # [8, 64]
            new_bias = new_state[f'{head}.{output_layer_idx}.bias'].clone()      # [8]

            # Copy old weights to first 5 outputs
            new_weight[:5, :] = old_weight
            new_bias[:5] = old_bias

            # Initialize new keyboard outputs (5,6,7) to zero/small values
            # This means ctrl/space/shift start as "off" until fine-tuned
            nn.init.zeros_(new_weight[5:, :])
            nn.init.constant_(new_bias[5:], -2.0)  # Sigmoid(-2) ≈ 0.12, defaults to "not pressed"

            old_state[f'{head}.{output_layer_idx}.weight'] = new_weight
            old_state[f'{head}.{output_layer_idx}.bias'] = new_bias

        print("   ✅ Migration complete - keyboard outputs initialized to 'off'")
        print("   Fine-tune with new recordings to learn keyboard actions")

    # Check if old model is missing mode_selector (new feature)
    has_mode_selector = any('mode_selector' in key for key in old_state.keys())
    if not has_mode_selector:
        print("🔄 Adding mode_selector head (new feature)...")
        # Initialize mode_selector with new model's random weights
        # It will learn from fine-tuning
        for key in new_state.keys():
            if 'mode_selector' in key:
                old_state[key] = new_state[key]
        print("   ✅ Mode selector initialized - will learn from fine-tuning")

    policy.load_state_dict(old_state)
    policy = policy.to(device)
    policy.eval()

    print(f"✅ Policy loaded from {path}")
    return policy
