"""Fine-tune pretrained model on simplified darkorbit_v7_simple dataset.

The pretrained model already knows:
  - BonusBox, Explosion detection
  - GUI elements
  - Lordakia (alien enemies)
  - Struener enemies
  - player_attack (lasers)
  - selection_ring

Fine-tuning will adapt this knowledge to DarkOrbit game specifically.
"""
from ultralytics import YOLO

def main():
    # Load pretrained model
    print("Loading pretrained model...")
    model = YOLO('C:/Users/alexispace/Downloads/best (3).pt')

    print(f"Pretrained model classes: {model.names}")
    print(f"Number of classes: {len(model.names)}")

    # Fine-tune on simplified dataset
    print("\nStarting fine-tuning...")
    results = model.train(
        data='F:/dev/bot/yolo/datasets/darkorbit_v7_simple/data.yaml',

        # Fine-tuning settings (different from training from scratch)
        epochs=100,           # Less epochs than from scratch
        patience=25,          # Early stopping

        # Lower learning rate for fine-tuning (preserve pretrained knowledge)
        lr0=0.001,            # 10x lower than from-scratch training
        lrf=0.01,
        warmup_epochs=3.0,

        # Image & Batch
        imgsz=1280,
        batch=6,

        # Optimizer
        optimizer='auto',     # Will use AdamW
        momentum=0.937,
        weight_decay=0.0005,

        # Augmentation (lighter since pretrained model saw different data)
        hsv_h=0.01,           # Reduced from 0.015
        hsv_s=0.5,            # Reduced from 0.7
        hsv_v=0.3,            # Reduced from 0.4
        degrees=0.0,
        translate=0.1,
        scale=0.3,            # Reduced from 0.5
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.8,           # Reduced from 1.0
        mixup=0.0,
        copy_paste=0.0,

        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # System
        project='F:/dev/bot/yolo/runs',
        name='darkorbit_v7_finetuned',
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

        # Fine-tuning specific
        freeze=None,          # Don't freeze any layers, allow full fine-tuning
    )

    print(f"\n=== Fine-tuning Complete ===")
    print(f"Best model: F:/dev/bot/yolo/runs/darkorbit_v7_finetuned/weights/best.pt")
    print(f"Best mAP50: {results.results_dict['metrics/mAP50(B)']:.3f}")
    print(f"Best mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.3f}")

    print(f"\n\nYou can now use the fine-tuned model:")
    print(f"  model = YOLO('F:/dev/bot/yolo/runs/darkorbit_v7_finetuned/weights/best.pt')")
    print(f"  results = model.predict('screenshot.png')")

if __name__ == '__main__':
    main()
