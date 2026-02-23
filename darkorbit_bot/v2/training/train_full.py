"""
V2 Full Training Pipeline - Train All Components

Trains all three hierarchical policy components in order:
1. Executor (60Hz motor control) - Needs most data, trains first
2. Tactician (10Hz target selection) - Uses executor features
3. Strategist (1Hz goal/mode selection) - Uses both

Usage:
    python -m darkorbit_bot.v2.training.train_full --data data/recordings --epochs 20

Options:
    --data          Directory containing recordings (.npz or .json)
    --output-dir    Where to save trained models (default: v2/checkpoints/)
    --epochs        Base epochs (executor gets this, tactician 50%, strategist 30%)
    --batch-size    Batch size
    --device        cuda or cpu
    --skip-executor Skip executor training (use existing)
    --skip-tactician Skip tactician training (use existing)
    --skip-strategist Skip strategist training (use existing)
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .train_executor import train_executor
from .train_tactician import train_tactician
from .train_strategist import train_strategist
from ..config import ExecutorConfig, TacticianConfig, StrategistConfig, TrainingConfig


def train_full(
    data_dir: str,
    output_dir: str = "v2/checkpoints",
    base_epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    skip_executor: bool = False,
    skip_tactician: bool = False,
    skip_strategist: bool = False,
):
    """
    Train the full V2 hierarchical policy.

    Args:
        data_dir: Directory containing training recordings
        output_dir: Where to save models
        base_epochs: Base epoch count (adjusted per component)
        batch_size: Training batch size
        learning_rate: Base learning rate
        device: cuda or cpu
        skip_executor: Skip executor training
        skip_tactician: Skip tactician training
        skip_strategist: Skip strategist training
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("  V2 FULL TRAINING PIPELINE - Hierarchical Policy")
    print("="*70)
    print(f"\n  Data: {data_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {device}")
    print(f"  Base epochs: {base_epochs}")
    print(f"  Batch size: {batch_size}")
    print("-"*70)

    # Check GPU
    if device == "cuda":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\n  ✅ GPU: {gpu_name}")
            torch.backends.cudnn.benchmark = True
        else:
            print("\n  ⚠️ CUDA not available, falling back to CPU")
            device = "cpu"

    start_time = time.time()
    trained_models = {}

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Train Executor (60Hz motor control)
    # Most data-hungry, gets full epochs
    # ═══════════════════════════════════════════════════════════════════

    executor_path = output_path / "executor.pt"

    if skip_executor and executor_path.exists():
        print(f"\n[1/3] EXECUTOR - Skipping (using existing: {executor_path})")
    else:
        print(f"\n[1/3] EXECUTOR - Training 60Hz motor control...")
        print(f"      Epochs: {base_epochs}")

        executor_config = ExecutorConfig()
        training_config = TrainingConfig(
            executor_epochs=base_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        try:
            executor = train_executor(
                data_dir=data_dir,
                output_path=str(executor_path),
                config=executor_config,
                training_config=training_config,
                device=device
            )
            trained_models['executor'] = executor
            print(f"      ✅ Saved to {executor_path}")
        except Exception as e:
            print(f"      ❌ Executor training failed: {e}")
            if not skip_executor:
                raise

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: Train Tactician (10Hz target selection)
    # Needs less data, 50% of base epochs
    # ═══════════════════════════════════════════════════════════════════

    tactician_path = output_path / "tactician.pt"
    tactician_epochs = max(10, base_epochs // 2)

    if skip_tactician and tactician_path.exists():
        print(f"\n[2/3] TACTICIAN - Skipping (using existing: {tactician_path})")
    else:
        print(f"\n[2/3] TACTICIAN - Training 10Hz target selection...")
        print(f"      Epochs: {tactician_epochs}")

        tactician_config = TacticianConfig()
        training_config = TrainingConfig(
            tactician_epochs=tactician_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        try:
            tactician = train_tactician(
                data_dir=data_dir,
                output_path=str(tactician_path),
                config=tactician_config,
                training_config=training_config,
                device=device
            )
            trained_models['tactician'] = tactician
            print(f"      ✅ Saved to {tactician_path}")
        except Exception as e:
            print(f"      ❌ Tactician training failed: {e}")
            if not skip_tactician:
                raise

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: Train Strategist (1Hz goal/mode selection)
    # Long-term patterns, 30% of base epochs
    # ═══════════════════════════════════════════════════════════════════

    strategist_path = output_path / "strategist.pt"
    strategist_epochs = max(6, base_epochs // 3)

    if skip_strategist and strategist_path.exists():
        print(f"\n[3/3] STRATEGIST - Skipping (using existing: {strategist_path})")
    else:
        print(f"\n[3/3] STRATEGIST - Training 1Hz goal/mode selection...")
        print(f"      Epochs: {strategist_epochs}")

        strategist_config = StrategistConfig()
        training_config = TrainingConfig(
            strategist_epochs=strategist_epochs,
            batch_size=max(16, batch_size // 4),  # Smaller batches for longer sequences
            learning_rate=learning_rate * 0.5  # Lower LR for strategist
        )

        try:
            strategist = train_strategist(
                data_dir=data_dir,
                output_path=str(strategist_path),
                config=strategist_config,
                training_config=training_config,
                device=device
            )
            trained_models['strategist'] = strategist
            print(f"      ✅ Saved to {strategist_path}")
        except Exception as e:
            print(f"      ❌ Strategist training failed: {e}")
            if not skip_strategist:
                raise

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════

    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print("="*70)
    print(f"\n  Total time: {total_time/60:.1f} minutes")
    print(f"\n  Models saved to: {output_path}")

    # Check what was trained
    for name, path in [("executor", executor_path),
                       ("tactician", tactician_path),
                       ("strategist", strategist_path)]:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"    ✅ {name}.pt ({size_mb:.1f} MB)")
        else:
            print(f"    ❌ {name}.pt (not found)")

    print(f"\n  To run the bot with trained models:")
    print(f"    python -m darkorbit_bot.v2.bot_controller_v2 --policy-dir {output_path}")
    print("="*70 + "\n")

    return trained_models


def main():
    parser = argparse.ArgumentParser(
        description='Train V2 Hierarchical Policy (all components)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train from recordings:
    python -m darkorbit_bot.v2.training.train_full --data data/recordings --epochs 20

  Train only executor:
    python -m darkorbit_bot.v2.training.train_full --data data/recordings --skip-tactician --skip-strategist

  Resume training (skip already trained):
    python -m darkorbit_bot.v2.training.train_full --data data/recordings --skip-executor
"""
    )

    parser.add_argument('--data', type=str, required=True,
                       help='Directory containing training recordings')
    parser.add_argument('--output-dir', type=str, default='v2/checkpoints',
                       help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Base epochs (executor=100%%, tactician=50%%, strategist=30%%)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--skip-executor', action='store_true',
                       help='Skip executor training (use existing checkpoint)')
    parser.add_argument('--skip-tactician', action='store_true',
                       help='Skip tactician training (use existing checkpoint)')
    parser.add_argument('--skip-strategist', action='store_true',
                       help='Skip strategist training (use existing checkpoint)')

    args = parser.parse_args()

    # Validate data directory
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Data directory not found: {args.data}")
        sys.exit(1)

    # Count recordings (search recursively in subdirectories)
    npz_files = list(data_path.glob("**/*.npz"))
    json_files = list(data_path.glob("**/*.json"))
    pkl_files = list(data_path.glob("**/*.pkl"))

    # Filter out non-recording JSON files (metadata, index, etc.)
    json_files = [f for f in json_files if 'sequence_' in f.name or 'recording' in f.name.lower()]

    # Filter pickle files to only shadow recordings
    pkl_files = [f for f in pkl_files if 'shadow_recording' in f.name.lower() or 'recording' in f.name.lower()]

    total_files = len(npz_files) + len(json_files) + len(pkl_files)

    if total_files == 0:
        print(f"❌ No recording files found in {args.data}")
        print(f"   Looking for: **/*.npz, **/sequence_*.json, or **/shadow_recording_*.pkl")
        sys.exit(1)

    print(f"Found {total_files} recording files ({len(npz_files)} .npz, {len(json_files)} .json, {len(pkl_files)} .pkl)")

    train_full(
        data_dir=args.data,
        output_dir=args.output_dir,
        base_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        skip_executor=args.skip_executor,
        skip_tactician=args.skip_tactician,
        skip_strategist=args.skip_strategist,
    )


if __name__ == "__main__":
    main()
