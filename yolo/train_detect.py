"""Train YOLO26 detection model on darkorbit_v6 dataset.

Based on official Roboflow YOLO26 training notebook.
Detection-only model (bounding boxes, no segmentation masks).
"""
from ultralytics import YOLO

def main():
    # Load YOLO26 detection model (official from Roboflow notebook)
    # Using medium variant for balance of speed/accuracy
    model = YOLO('yolo26m.pt')

    # Train on darkorbit_v6 dataset
    results = model.train(
        data='F:/dev/bot/yolo/datasets/darkorbit_v6/data.yaml',
        epochs=100,
        imgsz=1280,   # Best for small objects (ships, lasers, UI)
        project='F:/dev/bot/yolo/runs',
        name='darkorbit_v6_yolo26_detect',
        patience=20,  # Early stopping if no improvement for 20 epochs
        batch=5,      # Small batch to fit 1280 resolution in GPU
        workers=0,    # Windows fix: use 0 workers to avoid multiprocessing issues
        device=0,     # GPU
        plots=True,   # Generate training plots
        cache=False,  # Don't cache images in RAM
    )

    print(f"\nTraining complete!")
    print(f"Best model: F:/dev/bot/yolo/runs/darkorbit_v6_yolo26_detect/weights/best.pt")

if __name__ == '__main__':
    main()
