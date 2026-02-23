"""
Weight Migration Utility for V2 Visual Feature Upgrade

Copies learned weights from pre-visual V2 models to new visual-enabled models.
Only the new visual feature layers need training from scratch.

Strategy:
1. Load old model weights
2. Create new model with visual features enabled
3. Copy matching weights (YOLO feature processors stay the same)
4. Initialize new visual layers with sensible defaults
5. Save migrated model

The old model's learned behaviors are preserved - it just gains new "eyes".
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import json
import shutil

logger = logging.getLogger(__name__)


@dataclass
class MigrationReport:
    """Report of weight migration results."""
    model_name: str
    copied_params: int
    new_params: int
    total_params: int
    copied_layers: List[str]
    new_layers: List[str]
    warnings: List[str]


def get_param_count(state_dict: Dict[str, torch.Tensor]) -> int:
    """Count total parameters in state dict."""
    return sum(p.numel() for p in state_dict.values())


def copy_matching_weights(
    old_state: Dict[str, torch.Tensor],
    new_state: Dict[str, torch.Tensor],
    strict_shape: bool = True
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    """
    Copy weights from old model to new model where keys and shapes match.

    Args:
        old_state: State dict from old model
        new_state: State dict from new model (will be modified)
        strict_shape: If True, shapes must match exactly. If False, copy partial.

    Returns:
        Tuple of (updated_state, copied_keys, new_keys)
    """
    copied_keys = []
    new_keys = []
    updated_state = new_state.copy()

    for key in new_state.keys():
        if key in old_state:
            old_shape = old_state[key].shape
            new_shape = new_state[key].shape

            if old_shape == new_shape:
                # Exact match - copy directly
                updated_state[key] = old_state[key].clone()
                copied_keys.append(key)
            elif not strict_shape and len(old_shape) == len(new_shape):
                # Partial copy - for expanded layers (e.g., input projection)
                # Copy what fits, leave rest as initialized
                slices = tuple(slice(0, min(o, n)) for o, n in zip(old_shape, new_shape))
                updated_state[key][slices] = old_state[key][slices].clone()
                copied_keys.append(f"{key} (partial: {old_shape} -> {new_shape})")
            else:
                new_keys.append(f"{key} (shape mismatch: {old_shape} vs {new_shape})")
        else:
            new_keys.append(key)

    return updated_state, copied_keys, new_keys


def migrate_strategist(
    old_checkpoint_path: str,
    new_model: nn.Module,
    output_path: Optional[str] = None
) -> MigrationReport:
    """
    Migrate Strategist weights to visual-enabled version.

    The Strategist's input changes from [T, 192] to [T, 704] (192 + 512 visual).
    - Temporal attention layers: KEEP (operate on sequence, not features)
    - Input projection: EXPAND (192 -> 704 input dim)
    - Output layers: KEEP (goal_dim stays 64)
    """
    logger.info(f"Migrating Strategist from {old_checkpoint_path}")

    # Load old weights
    old_checkpoint = torch.load(old_checkpoint_path, map_location='cpu')
    if 'model_state_dict' in old_checkpoint:
        old_state = old_checkpoint['model_state_dict']
    else:
        old_state = old_checkpoint

    # Get new model state
    new_state = new_model.state_dict()

    # Copy matching weights (allow partial for input layers)
    updated_state, copied, new_layers = copy_matching_weights(
        old_state, new_state, strict_shape=False
    )

    # Special handling for input projection layer
    # The first linear layer needs to expand from 192 to 704 input
    input_layer_keys = [k for k in new_state.keys() if 'input' in k.lower() or k.endswith('.0.weight')]

    for key in input_layer_keys:
        if key in old_state and key in updated_state:
            old_w = old_state[key]
            new_w = updated_state[key]

            if len(old_w.shape) == 2 and len(new_w.shape) == 2:
                # Linear layer: [out_features, in_features]
                if old_w.shape[0] == new_w.shape[0] and old_w.shape[1] < new_w.shape[1]:
                    # Same output dim, expanded input - copy old weights to first part
                    updated_state[key][:, :old_w.shape[1]] = old_w
                    # Initialize visual part with small random values
                    nn.init.xavier_uniform_(updated_state[key][:, old_w.shape[1]:])
                    logger.info(f"Expanded input layer {key}: {old_w.shape} -> {new_w.shape}")

    # Load migrated weights
    new_model.load_state_dict(updated_state)

    # Save if output path provided
    if output_path:
        save_checkpoint = {
            'model_state_dict': updated_state,
            'migration_info': {
                'source': old_checkpoint_path,
                'copied_layers': copied,
                'new_layers': new_layers
            }
        }
        # Preserve other checkpoint data
        for key in ['optimizer_state_dict', 'epoch', 'config']:
            if key in old_checkpoint:
                save_checkpoint[key] = old_checkpoint[key]

        torch.save(save_checkpoint, output_path)
        logger.info(f"Saved migrated Strategist to {output_path}")

    return MigrationReport(
        model_name="Strategist",
        copied_params=sum(old_state[k].numel() for k in old_state if k in copied),
        new_params=sum(new_state[k].numel() for k in new_layers if k in new_state),
        total_params=get_param_count(new_state),
        copied_layers=copied,
        new_layers=new_layers,
        warnings=[]
    )


def migrate_tactician(
    old_checkpoint_path: str,
    new_model: nn.Module,
    output_path: Optional[str] = None
) -> MigrationReport:
    """
    Migrate Tactician weights to visual-enabled version.

    Object features change from [N, 20] to [N, 148] (20 + 128 visual).
    - Object encoder: EXPAND input (20 -> 148)
    - Self-attention layers: KEEP (operate on hidden dim)
    - Cross-attention with goal: KEEP
    - Output layers: KEEP
    """
    logger.info(f"Migrating Tactician from {old_checkpoint_path}")

    old_checkpoint = torch.load(old_checkpoint_path, map_location='cpu')
    if 'model_state_dict' in old_checkpoint:
        old_state = old_checkpoint['model_state_dict']
    else:
        old_state = old_checkpoint

    new_state = new_model.state_dict()

    # Copy matching weights
    updated_state, copied, new_layers = copy_matching_weights(
        old_state, new_state, strict_shape=False
    )

    # Special handling for object encoder (first layer processing objects)
    for key in new_state.keys():
        if ('object' in key.lower() and 'encoder' in key.lower()) or \
           (key.endswith('.0.weight') and 'object' in key.lower()):
            if key in old_state:
                old_w = old_state[key]
                new_w = updated_state[key]

                if len(old_w.shape) == 2 and old_w.shape[1] < new_w.shape[1]:
                    # Expand object input: copy YOLO features, init visual features
                    updated_state[key][:, :old_w.shape[1]] = old_w
                    nn.init.xavier_uniform_(updated_state[key][:, old_w.shape[1]:])
                    logger.info(f"Expanded object encoder {key}: {old_w.shape} -> {new_w.shape}")

    new_model.load_state_dict(updated_state)

    if output_path:
        save_checkpoint = {
            'model_state_dict': updated_state,
            'migration_info': {
                'source': old_checkpoint_path,
                'copied_layers': copied,
                'new_layers': new_layers
            }
        }
        for key in ['optimizer_state_dict', 'epoch', 'config']:
            if key in old_checkpoint:
                save_checkpoint[key] = old_checkpoint[key]

        torch.save(save_checkpoint, output_path)
        logger.info(f"Saved migrated Tactician to {output_path}")

    return MigrationReport(
        model_name="Tactician",
        copied_params=sum(old_state[k].numel() for k in old_state if k in copied),
        new_params=sum(new_state[k].numel() for k in new_layers if k in new_state),
        total_params=get_param_count(new_state),
        copied_layers=copied,
        new_layers=new_layers,
        warnings=[]
    )


def migrate_executor(
    old_checkpoint_path: str,
    new_model: nn.Module,
    output_path: Optional[str] = None
) -> MigrationReport:
    """
    Migrate Executor weights to visual-enabled version.

    Target info changes from [34] to [98] (34 + 64 visual).
    - State encoder: KEEP (64 dim unchanged)
    - Goal encoder: KEEP (64 dim unchanged)
    - Target encoder: EXPAND (34 -> 98)
    - Mamba/LSTM core: KEEP (hidden dim unchanged)
    - Output heads: KEEP
    """
    logger.info(f"Migrating Executor from {old_checkpoint_path}")

    old_checkpoint = torch.load(old_checkpoint_path, map_location='cpu')
    if 'model_state_dict' in old_checkpoint:
        old_state = old_checkpoint['model_state_dict']
    else:
        old_state = old_checkpoint

    new_state = new_model.state_dict()

    updated_state, copied, new_layers = copy_matching_weights(
        old_state, new_state, strict_shape=False
    )

    # Special handling for target encoder
    for key in new_state.keys():
        if 'target' in key.lower() and ('encoder' in key.lower() or 'embed' in key.lower()):
            if key in old_state:
                old_w = old_state[key]
                new_w = updated_state[key]

                if len(old_w.shape) == 2 and old_w.shape[1] < new_w.shape[1]:
                    updated_state[key][:, :old_w.shape[1]] = old_w
                    nn.init.xavier_uniform_(updated_state[key][:, old_w.shape[1]:])
                    logger.info(f"Expanded target encoder {key}: {old_w.shape} -> {new_w.shape}")

    new_model.load_state_dict(updated_state)

    if output_path:
        save_checkpoint = {
            'model_state_dict': updated_state,
            'migration_info': {
                'source': old_checkpoint_path,
                'copied_layers': copied,
                'new_layers': new_layers
            }
        }
        for key in ['optimizer_state_dict', 'epoch', 'config']:
            if key in old_checkpoint:
                save_checkpoint[key] = old_checkpoint[key]

        torch.save(save_checkpoint, output_path)
        logger.info(f"Saved migrated Executor to {output_path}")

    return MigrationReport(
        model_name="Executor",
        copied_params=sum(old_state[k].numel() for k in old_state if k in copied),
        new_params=sum(new_state[k].numel() for k in new_layers if k in new_state),
        total_params=get_param_count(new_state),
        copied_layers=copied,
        new_layers=new_layers,
        warnings=[]
    )


def migrate_all_models(
    old_checkpoint_dir: str,
    new_checkpoint_dir: str,
    config=None
) -> Dict[str, MigrationReport]:
    """
    Migrate all V2 models to visual-enabled versions.

    Args:
        old_checkpoint_dir: Directory with old checkpoints
        new_checkpoint_dir: Directory for new checkpoints
        config: V2Config with visual settings

    Returns:
        Dict of migration reports
    """
    old_dir = Path(old_checkpoint_dir)
    new_dir = Path(new_checkpoint_dir)
    new_dir.mkdir(parents=True, exist_ok=True)

    reports = {}

    # Import models
    from ..models.strategist import StrategistWithVisual
    from ..models.tactician import TacticianWithVisual
    from ..models.executor import ExecutorWithVisual

    if config is None:
        from ..config import V2Config
        config = V2Config()

    # Migrate Strategist
    old_strategist = old_dir / "strategist.pt"
    if old_strategist.exists():
        cfg = config.strategist
        new_strategist = StrategistWithVisual(
            state_dim=cfg.state_dim,
            visual_dim=cfg.visual_dim,
            hidden_dim=cfg.hidden_dim,
            goal_dim=cfg.goal_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            num_modes=cfg.num_modes,
            dropout=cfg.dropout
        )
        reports['strategist'] = migrate_strategist(
            str(old_strategist),
            new_strategist,
            str(new_dir / "strategist_visual.pt")
        )

    # Migrate Tactician
    old_tactician = old_dir / "tactician.pt"
    if old_tactician.exists():
        cfg = config.tactician
        new_tactician = TacticianWithVisual(
            object_dim=cfg.object_dim,
            visual_dim=cfg.visual_dim,
            goal_dim=cfg.goal_dim,
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            approach_dim=cfg.approach_dim,
            max_objects=cfg.max_objects,
            dropout=cfg.dropout
        )
        reports['tactician'] = migrate_tactician(
            str(old_tactician),
            new_tactician,
            str(new_dir / "tactician_visual.pt")
        )

    # Migrate Executor
    old_executor = old_dir / "executor.pt"
    if old_executor.exists():
        cfg = config.executor
        new_executor = ExecutorWithVisual(
            state_dim=cfg.state_dim,
            goal_dim=cfg.goal_dim,
            target_dim=cfg.target_dim,
            visual_dim=cfg.visual_dim,
            hidden_dim=cfg.hidden_dim,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            action_dim=cfg.action_dim
        )
        reports['executor'] = migrate_executor(
            str(old_executor),
            new_executor,
            str(new_dir / "executor_visual.pt")
        )

    # Save migration report
    report_path = new_dir / "migration_report.json"
    report_data = {
        name: {
            'model_name': r.model_name,
            'copied_params': r.copied_params,
            'new_params': r.new_params,
            'total_params': r.total_params,
            'copied_layers_count': len(r.copied_layers),
            'new_layers_count': len(r.new_layers)
        }
        for name, r in reports.items()
    }
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)

    logger.info(f"Migration complete. Report saved to {report_path}")
    return reports


def print_migration_report(report: MigrationReport):
    """Print a formatted migration report."""
    print(f"\n{'='*60}")
    print(f"Migration Report: {report.model_name}")
    print(f"{'='*60}")
    print(f"Total Parameters: {report.total_params:,}")
    print(f"Copied Parameters: {report.copied_params:,} ({100*report.copied_params/max(1,report.total_params):.1f}%)")
    print(f"New Parameters: {report.new_params:,} ({100*report.new_params/max(1,report.total_params):.1f}%)")
    print(f"\nCopied Layers ({len(report.copied_layers)}):")
    for layer in report.copied_layers[:10]:
        print(f"  [OK] {layer}")
    if len(report.copied_layers) > 10:
        print(f"  ... and {len(report.copied_layers) - 10} more")
    print(f"\nNew Layers ({len(report.new_layers)}):")
    for layer in report.new_layers[:10]:
        print(f"  [+] {layer}")
    if len(report.new_layers) > 10:
        print(f"  ... and {len(report.new_layers) - 10} more")
    if report.warnings:
        print(f"\nWarnings:")
        for w in report.warnings:
            print(f"  [!] {w}")
    print(f"{'='*60}\n")


# CLI interface
if __name__ == "__main__":
    import argparse
    import sys

    # Fix imports when running as standalone script
    script_dir = Path(__file__).parent.parent.parent.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    parser = argparse.ArgumentParser(description="Migrate V2 weights to visual-enabled models")
    parser.add_argument("--old-dir", type=str, default="data/checkpoints/v2",
                        help="Directory with old checkpoints")
    parser.add_argument("--new-dir", type=str, default="data/checkpoints/v2_visual",
                        help="Directory for new checkpoints")
    parser.add_argument("--model", type=str, choices=["all", "strategist", "tactician", "executor"],
                        default="all", help="Which model to migrate")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print(f"Migrating models from {args.old_dir} to {args.new_dir}")
    print("This will preserve your trained weights and only initialize new visual layers.\n")

    if args.model == "all":
        # Import here after path fix
        from darkorbit_bot.v2.models.strategist import StrategistWithVisual
        from darkorbit_bot.v2.models.tactician import TacticianWithVisual
        from darkorbit_bot.v2.models.executor import ExecutorWithVisual
        from darkorbit_bot.v2.config import V2Config

        old_dir = Path(args.old_dir)
        new_dir = Path(args.new_dir)
        new_dir.mkdir(parents=True, exist_ok=True)

        config = V2Config()
        reports = {}

        # Migrate Strategist
        old_strategist = old_dir / "strategist.pt"
        if old_strategist.exists():
            cfg = config.strategist
            new_strategist = StrategistWithVisual(
                state_dim=cfg.state_dim,
                visual_dim=cfg.visual_dim,
                hidden_dim=cfg.hidden_dim,
                goal_dim=cfg.goal_dim,
                num_heads=cfg.num_heads,
                num_layers=cfg.num_layers,
                num_modes=cfg.num_modes,
                dropout=cfg.dropout
            )
            reports['strategist'] = migrate_strategist(
                str(old_strategist),
                new_strategist,
                str(new_dir / "strategist_visual.pt")
            )
        else:
            print(f"Warning: {old_strategist} not found, skipping")

        # Migrate Tactician
        old_tactician = old_dir / "tactician.pt"
        if old_tactician.exists():
            cfg = config.tactician
            new_tactician = TacticianWithVisual(
                object_dim=cfg.object_dim,
                visual_dim=cfg.visual_dim,
                goal_dim=cfg.goal_dim,
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                num_layers=cfg.num_layers,
                approach_dim=cfg.approach_dim,
                max_objects=cfg.max_objects,
                dropout=cfg.dropout
            )
            reports['tactician'] = migrate_tactician(
                str(old_tactician),
                new_tactician,
                str(new_dir / "tactician_visual.pt")
            )
        else:
            print(f"Warning: {old_tactician} not found, skipping")

        # Migrate Executor
        old_executor = old_dir / "executor.pt"
        if old_executor.exists():
            cfg = config.executor
            new_executor = ExecutorWithVisual(
                state_dim=cfg.state_dim,
                goal_dim=cfg.goal_dim,
                target_dim=cfg.target_dim,
                visual_dim=cfg.visual_dim,
                hidden_dim=cfg.hidden_dim,
                d_state=cfg.d_state,
                d_conv=cfg.d_conv,
                expand=cfg.expand,
                action_dim=cfg.action_dim
            )
            reports['executor'] = migrate_executor(
                str(old_executor),
                new_executor,
                str(new_dir / "executor_visual.pt")
            )
        else:
            print(f"Warning: {old_executor} not found, skipping")

        # Save migration report
        report_path = new_dir / "migration_report.json"
        report_data = {
            name: {
                'model_name': r.model_name,
                'copied_params': r.copied_params,
                'new_params': r.new_params,
                'total_params': r.total_params,
                'copied_layers_count': len(r.copied_layers),
                'new_layers_count': len(r.new_layers)
            }
            for name, r in reports.items()
        }
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"Migration report saved to {report_path}")

        for name, report in reports.items():
            print_migration_report(report)
    else:
        print(f"Single model migration not yet implemented. Use --model all")
