"""YOLO Fine-tuning: Continue training existing model with new data.

This script fine-tunes an already trained model (best.pt) with additional
training data from darkorbit_v6. Uses lower learning rate to preserve
existing knowledge while adapting to new examples.

Usage:
    python yolo_finetuning.py
"""
from ultralytics import YOLO


def main():
    # Load existing trained model instead of pretrained weights
    existing_model = 'F:/dev/bot/best.pt'
    model = YOLO(existing_model)

    print("="*60)
    print("  YOLO FINE-TUNING")
    print("="*60)
    print(f"Base model: {existing_model}")
    print(f"Dataset: darkorbit_v6")
    print(f"Strategy: Lower LR to preserve existing knowledge")
    print("="*60)

    # Fine-tune with darkorbit_v6 dataset
    results = model.train(
        data='F:/dev/bot/yolo/datasets/darkorbit_v6/data.yaml',
        epochs=50,         # Fewer epochs for fine-tuning
        imgsz=1280,        # Match original training resolution
        project='F:/dev/bot/yolo/runs',
        name='darkorbit_v6_finetuned',

        # Fine-tuning specific settings
        lr0=0.001,         # Lower learning rate to avoid catastrophic forgetting
        lrf=0.0001,        # Final learning rate
        warmup_epochs=3,   # Fewer warmup epochs

        # Hardware settings
        batch=-1,          # Auto-detect optimal batch size for 16GB VRAM
        workers=0,         # Windows fix
        device=0,          # GPU

        # Training settings
        patience=15,       # Early stopping
        plots=True,        # Generate training plots
        verbose=True,
    )

    print(f"\n{'='*60}")
    print("FINE-TUNING COMPLETE!")
    print(f"{'='*60}")
    print(f"Fine-tuned model: F:/dev/bot/yolo/runs/darkorbit_v6_finetuned/weights/best.pt")
    print(f"\nNext steps:")
    print(f"1. Copy best.pt to F:/dev/bot/best.pt to replace old model")
    print(f"2. Test with: detector.py --live --model F:/dev/bot/best.pt --show")
    print(f"3. Compare performance vs original model")


if __name__ == '__main__':
    main()
