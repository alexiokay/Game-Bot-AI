"""
DarkOrbit Bot - YOLO Training Script
Trains YOLOv8 on your labeled dataset.

Usage:
    python train_yolo.py

Prerequisites:
    1. Run prepare_dataset.py to create dataset structure
    2. Label your images using Roboflow/CVAT/LabelImg
    3. Place labels in data/yolo_dataset/labels/
"""

import os
import json
from pathlib import Path


def check_labels(dataset_dir: Path) -> dict:
    """Check if labels exist and count them"""
    train_labels = list((dataset_dir / "labels" / "train").glob("*.txt"))
    val_labels = list((dataset_dir / "labels" / "val").glob("*.txt"))
    
    train_images = list((dataset_dir / "images" / "train").glob("*.png"))
    val_images = list((dataset_dir / "images" / "val").glob("*.png"))
    
    return {
        "train_images": len(train_images),
        "train_labels": len(train_labels),
        "val_images": len(val_images),
        "val_labels": len(val_labels),
        "labeled": len(train_labels) > 0 or len(val_labels) > 0
    }


def train(dataset_yaml: str, epochs: int = 100, imgsz: int = 640, 
          model_size: str = "n", resume: bool = False):
    """
    Train YOLOv8 model.
    
    Args:
        dataset_yaml: Path to dataset.yaml
        epochs: Number of training epochs
        imgsz: Image size (640 recommended)
        model_size: Model size - n(ano), s(mall), m(edium), l(arge), x(large)
        resume: Resume from last checkpoint
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed!")
        print("   Run: uv pip install ultralytics --python .venv\\Scripts\\python.exe")
        return None
    
    # Model selection
    model_name = f"yolov8{model_size}.pt"
    print(f"\n🤖 Using model: {model_name}")
    
    # Load model
    if resume:
        # Find last checkpoint
        runs_dir = Path("runs/detect")
        if runs_dir.exists():
            last_run = sorted(runs_dir.glob("train*"))[-1] if list(runs_dir.glob("train*")) else None
            if last_run and (last_run / "weights" / "last.pt").exists():
                model = YOLO(str(last_run / "weights" / "last.pt"))
                print(f"   Resuming from: {last_run}")
            else:
                model = YOLO(model_name)
        else:
            model = YOLO(model_name)
    else:
        model = YOLO(model_name)
    
    # Train
    print(f"\n🚀 Starting training...")
    print(f"   Dataset: {dataset_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print("-" * 60)
    
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=-1,  # Auto batch size
        patience=20,  # Early stopping
        save=True,
        plots=True,
        verbose=True,
        # Augmentation for game screenshots
        augment=True,
        hsv_h=0.015,  # Hue variation
        hsv_s=0.4,    # Saturation
        hsv_v=0.3,    # Value/brightness
        flipud=0.0,   # No vertical flip (game UI is fixed)
        fliplr=0.5,   # Horizontal flip OK
        scale=0.3,    # Scale variation
    )
    
    # Results
    print("\n" + "="*60)
    print("  ✅ TRAINING COMPLETE")
    print("="*60)
    
    # Find best model
    runs_dir = Path("runs/detect")
    latest_run = sorted(runs_dir.glob("train*"))[-1]
    best_model = latest_run / "weights" / "best.pt"
    
    print(f"\n📊 Results saved to: {latest_run}")
    print(f"   Best model: {best_model}")
    
    # Copy best model to models folder
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    import shutil
    dest_model = models_dir / "detector.pt"
    shutil.copy2(best_model, dest_model)
    print(f"   Copied to: {dest_model}")
    
    return str(dest_model)


def main():
    print("\n" + "="*60)
    print("  DARKORBIT BOT - YOLO TRAINING")
    print("="*60)
    
    # Find dataset
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    dataset_dir = data_dir / "yolo_dataset"
    dataset_yaml = dataset_dir / "dataset.yaml"
    
    if not dataset_dir.exists():
        print("\n❌ Dataset not found!")
        print("   Run prepare_dataset.py first.")
        return
    
    # Check labels
    label_info = check_labels(dataset_dir)
    
    print(f"\n📊 Dataset status:")
    print(f"   Training images: {label_info['train_images']}")
    print(f"   Training labels: {label_info['train_labels']}")
    print(f"   Validation images: {label_info['val_images']}")
    print(f"   Validation labels: {label_info['val_labels']}")
    
    if not label_info['labeled']:
        print("\n⚠️  NO LABELS FOUND!")
        print("""
You need to label your images first:

1. Go to https://roboflow.com (free tier)
2. Upload images from: """ + str(dataset_dir / "images" / "train"))
        print("""3. Draw boxes around: boxes, NPCs, portals, etc.
4. Export as "YOLO v8"
5. Extract the downloaded labels to:
   - """ + str(dataset_dir / "labels" / "train"))
        print("   - " + str(dataset_dir / "labels" / "val"))
        print("""
After labeling, run this script again!
""")
        return
    
    # Training options
    print("\n🎯 Training options:")
    print("   1. Quick test (10 epochs)")
    print("   2. Standard (50 epochs)")
    print("   3. Full training (100 epochs)")
    print("   4. Extended (200 epochs)")
    
    choice = input("\nChoice [1-4, default=2]: ").strip()
    
    epochs_map = {"1": 10, "2": 50, "3": 100, "4": 200}
    epochs = epochs_map.get(choice, 50)
    
    # Model size
    print("\n📦 Model size:")
    print("   n - Nano (fastest, least accurate)")
    print("   s - Small (balanced)")
    print("   m - Medium (slower, more accurate)")
    
    size = input("\nSize [n/s/m, default=n]: ").strip().lower()
    if size not in ["n", "s", "m", "l", "x"]:
        size = "n"
    
    # Train
    print("\n" + "-"*60)
    print(f"  Training with {epochs} epochs, model size '{size}'")
    print("-"*60)
    
    model_path = train(
        dataset_yaml=str(dataset_yaml),
        epochs=epochs,
        model_size=size
    )
    
    if model_path:
        print(f"\n🎉 Model saved to: {model_path}")
        print("   Next: Run the detector or integrate into bot!")


if __name__ == "__main__":
    main()
