"""Improved YOLO26 training with proper hyperparameters and augmentation.

Based on Ultralytics best practices for YOLO26.
Includes proper augmentation, learning rate tuning, and optimizer settings.
"""
from ultralytics import YOLO

def main():
    # Load YOLO26 medium model
    model = YOLO('yolo26m.pt')

    # Train with optimized hyperparameters
    results = model.train(
        data='F:/dev/bot/yolo/datasets/darkorbit_v6/data.yaml',

        # Training duration
        epochs=200,           # More epochs (YOLO26 needs longer to converge)
        patience=50,          # Less aggressive early stopping

        # Image & Batch
        imgsz=1280,           # Good for small objects
        batch=5,              # Reduced for 1280 resolution on 16GB GPU
        # batch=-1,           # Alternative: Auto-adjust to 60% GPU memory

        # Optimizer settings (YOLO26 uses MuSGD by default)
        optimizer='auto',     # Let YOLO26 use its MuSGD optimizer
        lr0=0.01,             # Initial learning rate
        lrf=0.01,             # Final learning rate (lr0 * lrf)
        momentum=0.937,       # SGD momentum
        weight_decay=0.0005,  # Weight decay for regularization
        warmup_epochs=3.0,    # Warmup epochs
        warmup_momentum=0.8,  # Warmup momentum

        # Data Augmentation - HSV
        hsv_h=0.015,          # Hue augmentation (0.0-1.0)
        hsv_s=0.7,            # Saturation augmentation (0.0-1.0)
        hsv_v=0.4,            # Value/brightness augmentation (0.0-1.0)

        # Data Augmentation - Geometric
        degrees=0.0,          # Rotation (+/- deg) - 0 for game UI
        translate=0.1,        # Translation (+/- fraction)
        scale=0.5,            # Scaling (+/- gain)
        shear=0.0,            # Shear (+/- deg) - 0 for game UI
        perspective=0.0,      # Perspective (+/- fraction) - 0 for game UI
        flipud=0.0,           # Vertical flip probability (no flip for game)
        fliplr=0.5,           # Horizontal flip probability (50% chance)

        # Data Augmentation - Advanced
        mosaic=1.0,           # Mosaic augmentation probability
        mixup=0.0,            # Mixup augmentation probability
        copy_paste=0.0,       # Copy-paste augmentation probability

        # Loss weights
        box=7.5,              # Box loss weight
        cls=0.5,              # Class loss weight
        dfl=1.5,              # DFL loss weight

        # System
        project='F:/dev/bot/yolo/runs',
        name='darkorbit_v6_yolo26m_improved',
        workers=0,            # Windows fix
        device=0,             # GPU
        plots=True,           # Generate training plots
        cache=False,          # Don't cache (153 images not worth it)

        # Validation
        val=True,             # Validate during training
        save=True,            # Save checkpoints
        save_period=-1,       # Save checkpoint every epoch (-1 = only last)

        # Advanced YOLO26 features
        close_mosaic=10,      # Disable mosaic in last N epochs for better accuracy
    )

    print(f"\n=== Training Complete ===")
    print(f"Best model: F:/dev/bot/yolo/runs/darkorbit_v6_yolo26m_improved/weights/best.pt")
    print(f"Best mAP50: {results.results_dict['metrics/mAP50(B)']:.3f}")
    print(f"Best mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.3f}")

if __name__ == '__main__':
    main()
