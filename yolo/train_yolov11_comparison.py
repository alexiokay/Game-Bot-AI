"""YOLOv11 training for comparison with YOLO26.

YOLOv11 is more mature and stable, better for small objects.
Use this to compare against YOLO26 results.
"""
from ultralytics import YOLO

def main():
    # Load YOLOv11 medium model
    model = YOLO('yolo11m.pt')

    # Train with same hyperparameters as YOLO26 for fair comparison
    results = model.train(
        data='F:/dev/bot/yolo/datasets/darkorbit_v6/data.yaml',

        # Training duration
        epochs=200,
        patience=50,

        # Image & Batch
        imgsz=1280,
        batch=6,

        # Optimizer settings
        optimizer='auto',     # YOLOv11 uses AdamW by default
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,

        # Data Augmentation - HSV
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # Data Augmentation - Geometric
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,

        # Data Augmentation - Advanced
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,

        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # System
        project='F:/dev/bot/yolo/runs',
        name='darkorbit_v6_yolo11m_comparison',
        workers=0,
        device=0,
        plots=True,
        cache=False,

        # Validation
        val=True,
        save=True,
        save_period=-1,

        # Advanced
        close_mosaic=10,
    )

    print(f"\n=== Training Complete ===")
    print(f"Best model: F:/dev/bot/yolo/runs/darkorbit_v6_yolo11m_comparison/weights/best.pt")
    print(f"Best mAP50: {results.results_dict['metrics/mAP50(B)']:.3f}")
    print(f"Best mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.3f}")

if __name__ == '__main__':
    main()
