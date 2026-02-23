"""
DarkOrbit Bot - Training Script

Trains the Bi-LSTM policy network on your recorded gameplay.
Now also supports learning from VLM corrections (self-improvement).

Usage:
    python train.py                    # Train on all recordings (from scratch)
    python train.py --epochs 20        # Train for 20 epochs
    python train.py --corrections      # Include VLM corrections from bot sessions
    python train.py --finetune         # Continue training existing model (recommended!)
    python train.py --finetune --corrections  # Fine-tune with new corrections only

Training Modes:
    FROM SCRATCH (default): Creates new model, trains on ALL data
    FINE-TUNE (--finetune): Loads existing model, continues training
                            Much faster and better for incremental improvement!
"""

import sys
import json
from pathlib import Path
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim

from reasoning.policy_network import create_policy, save_policy, load_policy


def load_all_sequences(data_dir: str = "data/recordings", sequence_length: int = 50):
    """
    Load all recorded sequences and create training samples.

    Creates sliding window samples so EVERY frame gets trained on,
    not just the last frame of each recording.

    If sequences have VLM annotations, uses quality info for weighting.
    Run `python reasoning/vlm_annotator.py` to add annotations.
    """
    data_path = Path(data_dir)
    samples = []
    annotated_count = 0

    print(f"📂 Loading data from: {data_path}")

    # Find all sequence files
    for seq_file in data_path.glob("**/sequence_*.json"):
        try:
            with open(seq_file, 'r') as f:
                data = json.load(f)

            # Only load SUCCESS labeled sequences
            if data.get('label') != 'SUCCESS':
                continue

            states = np.array(data['states'], dtype=np.float32)
            actions = np.array(data['actions'], dtype=np.float32)
            mode = data.get('mode', 'PASSIVE')

            # Check for VLM annotations
            vlm_context = data.get('vlm_context', {})
            if vlm_context:
                annotated_count += 1
                # Show VLM source type
                if vlm_context.get('source') == 'vlm_image':
                    pass  # Real VLM annotation with screenshots

            # Quality weight from VLM annotation (default 1.0)
            # VLM tells us if player action was good or needs improvement
            quality = vlm_context.get('quality', 'good')
            if quality == 'good':
                weight = 1.0  # Full weight for good actions
            elif quality == 'needs_improvement':
                weight = 0.3  # Lower weight - learn less from bad examples
            else:
                weight = 0.7  # Unknown quality - moderate weight

            # Create sliding window samples - EVERY frame becomes a training target
            for i in range(sequence_length, len(states)):
                samples.append({
                    'states': states[i - sequence_length:i],
                    'action': actions[i - 1],
                    'mode': mode,
                    'weight': weight  # For potential weighted loss
                })

        except Exception as e:
            print(f"   ⚠️ Error loading {seq_file.name}: {e}")

    print(f"   ✅ Created {len(samples)} training samples from recordings")
    if annotated_count > 0:
        print(f"   📝 {annotated_count} sequences have VLM annotations")
    return samples


def load_corrections(corrections_dir: str = "data/vlm_corrections") -> list:
    """
    Load VLM corrections from self-improvement and enhanced VLM sessions.

    Supports three formats:
    1. Self-improve: session_*/corrections.json (basic VLM critique)
    2. Enhanced VLM: enhanced_*.json (multi-level Strategic/Tactical/Execution)
    3. Bad stop: bad_stop_*.json (user pressed F2 to mark as wrong)

    These are corrections generated when VLM watches the bot play
    and tells it what it SHOULD have done, OR when user manually marks
    bot behavior as wrong (bad stop).

    Returns:
        List of correction samples (single-frame, not sequences)
    """
    corrections_path = Path(corrections_dir)
    samples = []
    self_improve_count = 0
    enhanced_count = 0
    bad_stop_count = 0

    if not corrections_path.exists():
        print(f"   ⚠️ Corrections directory not found: {corrections_path}")
        return samples

    # ═══════════════════════════════════════════════════════════════
    # Format 1: Self-improve corrections (session_*/corrections.json)
    # ═══════════════════════════════════════════════════════════════
    for session_dir in corrections_path.glob("session_*"):
        corrections_file = session_dir / "corrections.json"
        if not corrections_file.exists():
            continue

        try:
            with open(corrections_file, 'r') as f:
                data = json.load(f)

            for c in data.get('corrections', []):
                correct_action = c.get('correct_action', {})
                state_vector = c.get('state_vector')

                # Skip if no state vector (can't train without it)
                if not state_vector:
                    continue

                # Weight by quality - bad actions = high weight (need to learn most)
                quality = c.get('quality', 'unknown')
                if quality == 'bad':
                    weight = 2.0  # High weight - definitely wrong
                elif quality == 'needs_improvement':
                    weight = 1.5  # Medium weight
                else:
                    weight = 0.5  # Low weight - already good

                # Build 8-dim action vector (includes keyboard)
                samples.append({
                    'state': np.array(state_vector, dtype=np.float32),
                    'action': np.array([
                        correct_action.get('move_x', 0.5),
                        correct_action.get('move_y', 0.5),
                        1.0 if correct_action.get('should_click', False) else 0.0,
                        0.0,  # is_enemy (placeholder)
                        0.0,  # distance (placeholder)
                        1.0 if correct_action.get('ctrl_attack', False) else 0.0,
                        1.0 if correct_action.get('space_rocket', False) else 0.0,
                        1.0 if correct_action.get('shift_special', False) else 0.0,
                    ], dtype=np.float32),
                    'mode': c.get('bot_action', {}).get('mode', 'PASSIVE'),
                    'weight': weight,
                    'source': 'self_improve'
                })
                self_improve_count += 1

        except Exception as e:
            print(f"   ⚠️ Error loading corrections from {session_dir.name}: {e}")

    # ═══════════════════════════════════════════════════════════════
    # Format 2: Enhanced VLM corrections (enhanced_*.json)
    # ═══════════════════════════════════════════════════════════════
    for enhanced_file in corrections_path.glob("enhanced_*.json"):
        try:
            with open(enhanced_file, 'r') as f:
                data = json.load(f)

            for c in data.get('corrections', []):
                vlm_correction = c.get('vlm_correction', {})
                level = c.get('level', 'tactical')
                context = c.get('context', {})
                state_vector = c.get('state_vector')

                # Skip if no state vector (can't train without it)
                if not state_vector:
                    continue

                # Weight by analysis level and confidence
                # Execution-level corrections are most actionable
                if level == 'execution':
                    weight = 2.5  # Highest - immediate action correction
                elif level == 'tactical':
                    weight = 2.0  # High - target/tactic correction
                else:  # strategic
                    weight = 1.0  # Lower - general guidance

                # Determine mode from context
                mode = context.get('mode', 'PASSIVE')

                # Build action from VLM correction
                # Enhanced VLM provides richer correction data
                move_x = vlm_correction.get('move_x', 0.5)
                move_y = vlm_correction.get('move_y', 0.5)
                should_attack = vlm_correction.get('should_attack', False)

                # For strategic corrections, infer attack from strategy
                if level == 'strategic':
                    strategy = vlm_correction.get('recommended_strategy', 'farm')
                    should_attack = strategy in ['farm', 'hunt']
                    # Strategic doesn't give positions, use center
                    move_x = 0.5
                    move_y = 0.5

                samples.append({
                    'state': np.array(state_vector, dtype=np.float32),
                    'action': np.array([
                        move_x,
                        move_y,
                        1.0 if should_attack else 0.0,
                        1.0 if context.get('num_enemies', 0) > 0 else 0.0,  # is_enemy
                        0.0,  # distance (not available)
                        1.0 if should_attack else 0.0,  # ctrl_attack
                        0.0,  # space_rocket (not in correction)
                        0.0,  # shift_special (not in correction)
                    ], dtype=np.float32),
                    'mode': mode,
                    'weight': weight,
                    'source': f'enhanced_{level}',
                    'level': level,
                    'vlm_full': c.get('vlm_full_result', {}),  # Keep full VLM response
                })
                enhanced_count += 1

        except Exception as e:
            print(f"   ⚠️ Error loading enhanced corrections from {enhanced_file.name}: {e}")

    # ═══════════════════════════════════════════════════════════════
    # Format 3: Bad stop corrections (bad_stop_*.json)
    # User pressed F2 to mark bot behavior as WRONG
    # These are NEGATIVE examples - the action the bot took was bad
    # ═══════════════════════════════════════════════════════════════
    for bad_stop_file in corrections_path.glob("bad_stop_*.json"):
        try:
            with open(bad_stop_file, 'r') as f:
                data = json.load(f)

            for c in data.get('corrections', []):
                state_vector = c.get('state_vector')
                bot_action = c.get('bot_action', {})
                mode = c.get('mode', 'PASSIVE')

                # Skip if no state vector
                if not state_vector:
                    continue

                # Bad stop = highest weight negative example
                # We want the model to AVOID this action
                # Training approach: we don't know the "correct" action,
                # but we know this action was wrong. Use inverse of bot_action
                # or train to output "do nothing" / opposite direction
                weight = 3.0  # Highest weight - user explicitly said this was wrong

                # For negative training, we push AWAY from the bad action
                # Simple approach: train toward neutral (center, no click)
                # The idea is "when in this state, DON'T do what the bot did"
                samples.append({
                    'state': np.array(state_vector, dtype=np.float32),
                    'action': np.array([
                        0.5,  # Neutral position (don't move there)
                        0.5,
                        0.0,  # Don't click
                        0.0,  # is_enemy (placeholder)
                        0.0,  # distance (placeholder)
                        0.0,  # Don't press ctrl
                        0.0,  # Don't press space
                        0.0,  # Don't press shift
                    ], dtype=np.float32),
                    'mode': mode,
                    'weight': weight,
                    'source': 'bad_stop',
                    'bad_action': bot_action,  # Keep record of what was wrong
                })
                bad_stop_count += 1

        except Exception as e:
            print(f"   ⚠️ Error loading bad stop corrections from {bad_stop_file.name}: {e}")

    # Summary
    if samples:
        print(f"   🔄 Loaded {len(samples)} VLM correction samples:")
        if self_improve_count:
            print(f"      - {self_improve_count} from self-improve")
        if enhanced_count:
            print(f"      - {enhanced_count} from enhanced VLM")
        if bad_stop_count:
            print(f"      - {bad_stop_count} from bad stop (F2)")
    return samples


def train(epochs: int = 10, data_dir: str = "data/recordings", batch_size: int = 32,
          use_corrections: bool = False, finetune: bool = False,
          corrections_only: bool = False):
    """
    Train the policy network using proper sequence-to-sequence learning.

    Key improvements:
    1. Uses ALL frames from recordings (sliding window), not just the last
    2. Uses BCE loss for click prediction (binary classification)
    3. Uses MSE loss for mouse position (regression)
    4. Proper batching for efficiency
    5. Can include VLM corrections from self-improvement sessions
    6. FINE-TUNE mode: continue training existing model instead of starting fresh

    Args:
        epochs: Number of training epochs
        data_dir: Directory with recorded sequences
        batch_size: Training batch size
        use_corrections: Include VLM correction samples
        finetune: Load existing model and continue training (recommended!)
        corrections_only: Only train on corrections, not full sequences (faster)
    """

    print("\n" + "="*60)
    if finetune:
        print("  🔧 FINE-TUNING Bi-LSTM POLICY NETWORK")
    else:
        print("  🧠 TRAINING Bi-LSTM POLICY NETWORK (from scratch)")
    print("="*60)

    # Load data with sliding window sampling
    samples = []
    if not corrections_only:
        samples = load_all_sequences(data_dir, sequence_length=50)

    # Optionally load corrections from VLM self-improvement
    correction_samples = []
    if use_corrections or corrections_only:
        print("\n📚 Loading VLM corrections...")
        correction_samples = load_corrections()

    if len(samples) == 0 and len(correction_samples) == 0:
        print("\n❌ No training data found!")
        if corrections_only:
            print("   No corrections found. Run bot with --self-improve first.")
        else:
            print("   Record some gameplay first with filtered_recorder.py")
        return None

    # Determine state size from data
    if samples:
        state_size = samples[0]['states'].shape[1]
    elif correction_samples:
        state_size = len(correction_samples[0]['state'])
    else:
        state_size = 128  # Default

    total_samples = len(samples) + len(correction_samples)
    print(f"\n📊 State vector size: {state_size}")
    if not corrections_only:
        print(f"   Sequence samples: {len(samples)}")
    if correction_samples:
        print(f"   Correction samples: {len(correction_samples)}")
    print(f"   Total training samples: {total_samples}")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {epochs}")

    # Create or load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")

    model_path = Path("data/checkpoints/policy_latest.pt")

    if finetune and model_path.exists():
        print(f"\n🔄 Loading existing model for fine-tuning...")
        policy = load_policy(str(model_path), device=device)
        policy.train()  # Set to training mode
        # Use lower learning rate for fine-tuning (prevents catastrophic forgetting)
        learning_rate = 0.0003
        print(f"   Using lower learning rate for fine-tuning: {learning_rate}")
    else:
        if finetune:
            print(f"\n⚠️ No existing model found at {model_path}")
            print("   Creating new model instead...")
        policy = create_policy(state_size=state_size, device=device)
        learning_rate = 0.001

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Separate losses for different action components
    mse_loss = nn.MSELoss()  # For mouse position (continuous)
    bce_loss = nn.BCEWithLogitsLoss()  # For click prediction (binary)

    # Training loop
    print("\n" + "-"*60)
    print("🚀 Starting training...")
    print("-"*60)

    policy.train()
    best_loss = float('inf')
    interrupted = False

    try:
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_pos_loss = 0
            epoch_click_loss = 0
            num_batches = 0

            np.random.shuffle(samples)

            for i in range(0, len(samples), batch_size):
                batch = samples[i:i + batch_size]
                if len(batch) < 2:
                    continue

                optimizer.zero_grad()

                # Separate by mode for proper head selection
                passive_states, passive_targets = [], []
                aggressive_states, aggressive_targets = [], []

                for sample in batch:
                    states = torch.tensor(sample['states'], dtype=torch.float32)
                    target = torch.tensor(sample['action'], dtype=torch.float32)

                    if sample['mode'] == 'AGGRESSIVE':
                        aggressive_states.append(states)
                        aggressive_targets.append(target)
                    else:
                        passive_states.append(states)
                        passive_targets.append(target)

                total_batch_loss = 0

                # Train passive samples
                if passive_states:
                    states_batch = torch.stack(passive_states).to(device)
                    targets_batch = torch.stack(passive_targets).to(device)

                    # Forward through LSTM to get context
                    lstm_out, _ = policy.lstm(states_batch)
                    attention_weights = policy.attention(lstm_out)
                    context = torch.sum(lstm_out * attention_weights, dim=1)

                    # Get predictions from passive head
                    pred = policy.passive_head(context)

                    # Train mode selector: PASSIVE = 0
                    mode_pred = policy.mode_selector(context)
                    mode_target = torch.zeros_like(mode_pred)  # 0 = PASSIVE
                    mode_loss = bce_loss(mode_pred, mode_target)

                    # Position loss (indices 0,1 = mouse x,y)
                    pos_loss = mse_loss(pred[:, :2], targets_batch[:, :2])

                    # Click loss (index 2 = should_click)
                    click_loss = bce_loss(pred[:, 2], targets_batch[:, 2])

                    # Meta info loss (indices 3,4 = is_enemy, distance)
                    meta_loss = mse_loss(pred[:, 3:5], targets_batch[:, 3:5])

                    # Keyboard actions loss (indices 5,6,7 = ctrl, space, shift)
                    # Only compute if targets have keyboard data (8-dim actions)
                    keyboard_loss = torch.tensor(0.0, device=device)
                    if targets_batch.shape[1] >= 8:
                        ctrl_loss = bce_loss(pred[:, 5], targets_batch[:, 5])
                        space_loss = bce_loss(pred[:, 6], targets_batch[:, 6])
                        shift_loss = bce_loss(pred[:, 7], targets_batch[:, 7])
                        keyboard_loss = (ctrl_loss + space_loss + shift_loss) * 1.5

                    # Include mode selector loss (learns when to be passive)
                    loss = pos_loss + click_loss * 2.0 + meta_loss + keyboard_loss + mode_loss
                    loss.backward()
                    total_batch_loss += loss.item()
                    epoch_pos_loss += pos_loss.item()
                    epoch_click_loss += click_loss.item()

                # Train aggressive samples
                if aggressive_states:
                    states_batch = torch.stack(aggressive_states).to(device)
                    targets_batch = torch.stack(aggressive_targets).to(device)

                    # Forward through LSTM to get context
                    lstm_out, _ = policy.lstm(states_batch)
                    attention_weights = policy.attention(lstm_out)
                    context = torch.sum(lstm_out * attention_weights, dim=1)

                    # Get predictions from aggressive head
                    pred = policy.aggressive_head(context)

                    # Train mode selector: AGGRESSIVE = 1
                    mode_pred = policy.mode_selector(context)
                    mode_target = torch.ones_like(mode_pred)  # 1 = AGGRESSIVE
                    mode_loss = bce_loss(mode_pred, mode_target)

                    # Aim position loss (indices 0,1)
                    pos_loss = mse_loss(pred[:, :2], targets_batch[:, :2])

                    # Fire loss (index 2 = should_fire)
                    fire_loss = bce_loss(pred[:, 2], targets_batch[:, 2])

                    # Meta info loss (indices 3,4)
                    meta_loss = mse_loss(pred[:, 3:5], targets_batch[:, 3:5])

                    # Keyboard actions loss (indices 5,6,7 = ctrl, space, shift)
                    keyboard_loss = torch.tensor(0.0, device=device)
                    if targets_batch.shape[1] >= 8:
                        ctrl_loss = bce_loss(pred[:, 5], targets_batch[:, 5])
                        space_loss = bce_loss(pred[:, 6], targets_batch[:, 6])
                        shift_loss = bce_loss(pred[:, 7], targets_batch[:, 7])
                        keyboard_loss = (ctrl_loss + space_loss + shift_loss) * 1.5

                    # Include mode selector loss (learns when to be aggressive)
                    loss = pos_loss + fire_loss * 2.0 + meta_loss + keyboard_loss + mode_loss
                    loss.backward()
                    total_batch_loss += loss.item()
                    epoch_pos_loss += pos_loss.item()
                    epoch_click_loss += fire_loss.item()

                optimizer.step()
                epoch_loss += total_batch_loss
                num_batches += 1

            # Train on correction samples (single-frame, not sequences)
            # These teach the network what it SHOULD have done
            if correction_samples:
                np.random.shuffle(correction_samples)
                # Use smaller batch size for corrections (often have few samples)
                corr_batch_size = min(batch_size, max(2, len(correction_samples) // 4))
                for i in range(0, len(correction_samples), corr_batch_size):
                    corr_batch = correction_samples[i:i + corr_batch_size]
                    if len(corr_batch) < 1:  # Process even single samples
                        continue

                    optimizer.zero_grad()

                    # Corrections are single frames, expand to match sequence input
                    # by repeating the state (simple approach)
                    passive_states, passive_targets, passive_weights = [], [], []
                    aggressive_states, aggressive_targets, aggressive_weights = [], [], []

                    for sample in corr_batch:
                        # Repeat state to create a "sequence" of 50 identical frames
                        state_seq = torch.tensor(
                            np.tile(sample['state'], (50, 1)),
                            dtype=torch.float32
                        )
                        target = torch.tensor(sample['action'], dtype=torch.float32)
                        weight = sample.get('weight', 1.0)

                        if sample['mode'] == 'AGGRESSIVE':
                            aggressive_states.append(state_seq)
                            aggressive_targets.append(target)
                            aggressive_weights.append(weight)
                        else:
                            passive_states.append(state_seq)
                            passive_targets.append(target)
                            passive_weights.append(weight)

                    # Train passive corrections
                    if passive_states:
                        states_batch = torch.stack(passive_states).to(device)
                        targets_batch = torch.stack(passive_targets).to(device)
                        weights_batch = torch.tensor(passive_weights, dtype=torch.float32).to(device)

                        pred = policy(states_batch, mode="PASSIVE")
                        pos_loss = ((pred[:, :2] - targets_batch[:, :2])**2).mean(dim=1)
                        click_loss = nn.functional.binary_cross_entropy_with_logits(
                            pred[:, 2], targets_batch[:, 2], reduction='none'
                        )
                        # Weight the loss by correction importance
                        weighted_loss = (pos_loss + click_loss * 2.0) * weights_batch
                        weighted_loss.mean().backward()
                        epoch_loss += weighted_loss.mean().item()
                        epoch_pos_loss += pos_loss.mean().item()
                        epoch_click_loss += click_loss.mean().item()

                    # Train aggressive corrections
                    if aggressive_states:
                        states_batch = torch.stack(aggressive_states).to(device)
                        targets_batch = torch.stack(aggressive_targets).to(device)
                        weights_batch = torch.tensor(aggressive_weights, dtype=torch.float32).to(device)

                        pred = policy(states_batch, mode="AGGRESSIVE")
                        pos_loss = ((pred[:, :2] - targets_batch[:, :2])**2).mean(dim=1)
                        fire_loss = nn.functional.binary_cross_entropy_with_logits(
                            pred[:, 2], targets_batch[:, 2], reduction='none'
                        )
                        weighted_loss = (pos_loss + fire_loss * 2.0) * weights_batch
                        weighted_loss.mean().backward()
                        epoch_loss += weighted_loss.mean().item()
                        epoch_pos_loss += pos_loss.mean().item()
                        epoch_click_loss += fire_loss.mean().item()

                    optimizer.step()
                    num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            avg_pos = epoch_pos_loss / max(num_batches, 1)
            avg_click = epoch_click_loss / max(num_batches, 1)

            corr_indicator = " +corr" if correction_samples else ""
            print(f"   Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Pos: {avg_pos:.4f} | Click: {avg_click:.4f}{corr_indicator}")

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss

    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user (Ctrl+C)!")
        print("   Saving current progress...")
        interrupted = True
        
    # Save model
    checkpoint_dir = Path("data/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = checkpoint_dir / "policy_latest.pt"
    save_policy(policy, str(model_path))
    
    print("\n" + "="*60)
    if interrupted:
        print("  ⚠️ TRAINING INTERRUPTED - PROGRESS SAVED!")
    else:
        print("  ✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📁 Model saved to: {model_path}")
    print(f"   Best loss: {best_loss:.6f}")
    print(f"   Total samples trained: {len(samples)}")
    print("\n🤖 To run the bot with this model:")
    print(f"   python reasoning/bot_controller.py --policy {model_path}")

    return policy


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train the policy network')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--data', type=str, default='data/recordings', help='Data directory')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--corrections', action='store_true',
                       help='Include VLM corrections from self-improvement sessions')
    parser.add_argument('--finetune', action='store_true',
                       help='Continue training existing model (recommended for incremental improvement)')
    parser.add_argument('--corrections-only', action='store_true',
                       help='Only train on corrections, skip full sequences (fast incremental update)')

    args = parser.parse_args()

    train(epochs=args.epochs, data_dir=args.data, batch_size=args.batch,
          use_corrections=args.corrections, finetune=args.finetune,
          corrections_only=args.corrections_only)


if __name__ == "__main__":
    main()
