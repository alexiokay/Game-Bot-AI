"""Inspect YOLO model details - architecture, parameters, training config."""
from ultralytics import YOLO
import torch

def inspect_model(model_path):
    print("="*60)
    print(f"  INSPECTING MODEL: {model_path}")
    print("="*60)

    # Load model
    model = YOLO(model_path)

    # Basic info
    print(f"\nModel Type: {model.task}")
    print(f"Model Variant: {model.ckpt.get('model', 'unknown')}")

    # Check if it's a detection or segmentation model
    if hasattr(model.model, 'names'):
        print(f"Number of classes: {len(model.names)}")
        print(f"Class names: {model.names}")

    # Training configuration used
    if 'train_args' in model.ckpt:
        print(f"\n{'='*60}")
        print("TRAINING CONFIGURATION USED:")
        print(f"{'='*60}")
        train_args = model.ckpt['train_args']
        for key, value in train_args.items():
            print(f"  {key}: {value}")

    # Model architecture details
    print(f"\n{'='*60}")
    print("MODEL ARCHITECTURE:")
    print(f"{'='*60}")

    # Count parameters
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

    # Training metrics if available
    if 'metrics/mAP50(B)' in model.ckpt:
        print(f"\n{'='*60}")
        print("TRAINING METRICS:")
        print(f"{'='*60}")
        print(f"  mAP50: {model.ckpt.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  mAP50-95: {model.ckpt.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"  Best fitness: {model.ckpt.get('best_fitness', 'N/A')}")
        print(f"  Epoch: {model.ckpt.get('epoch', 'N/A')}")

    # Check input image size
    if 'imgsz' in model.ckpt.get('train_args', {}):
        print(f"\nInput image size: {model.ckpt['train_args']['imgsz']}")

    print(f"\n{'='*60}")

    # Try to get model base architecture
    try:
        model_name = model.ckpt.get('train_args', {}).get('model', 'unknown')
        print(f"Base model: {model_name}")
    except:
        pass

    # Dataset it was trained on
    try:
        data_path = model.ckpt.get('train_args', {}).get('data', 'unknown')
        print(f"Dataset: {data_path}")
    except:
        pass

if __name__ == '__main__':
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'F:/dev/bot/best.pt'
    inspect_model(model_path)
