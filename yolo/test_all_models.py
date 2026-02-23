"""Test all trained models on a training image to see which works best."""
from ultralytics import YOLO
import cv2
from pathlib import Path

models_to_test = [
    ('Friend model (current best.pt)', 'F:/dev/bot/best.pt'),
    ('v3 YOLO26 detect', 'F:/dev/bot/yolo/runs/darkorbit_v6_yolo26_detect/weights/best.pt'),
    ('v3 YOLO11n', 'F:/dev/bot/yolo/runs/darkorbit_v6_yolo11n/weights/best.pt'),
    ('v3 Finetuned', 'F:/dev/bot/yolo/runs/darkorbit_v6_finetuned/weights/best.pt'),
    ('v2 YOLO26 detect', 'F:/dev/bot/yolo/runs/darkorbit_v6_yolo26_detect/weights/best.pt'),
]

# Get test image from v3 dataset
train_dir = Path('F:/dev/bot/yolo/datasets/darkorbit_v6/train/images')
images = list(train_dir.glob('*.jpg'))
if not images:
    print("No training images found!")
    exit(1)

test_img = str(images[0])
img = cv2.imread(test_img)

print("="*80)
print(f"Testing on: {images[0].name}")
print(f"Image shape: {img.shape}")
print("="*80)

for name, model_path in models_to_test:
    print(f"\n{name}:")
    print(f"  Path: {model_path}")

    if not Path(model_path).exists():
        print("  ❌ Model not found")
        continue

    try:
        model = YOLO(model_path)
        print(f"  Classes: {len(model.names)}")

        # Test at different confidence levels
        for conf in [0.01, 0.05, 0.1, 0.25]:
            results = model(img, conf=conf, verbose=False)
            detections = len(results[0].boxes)

            if detections > 0:
                max_conf = max([float(box.conf[0]) for box in results[0].boxes])
                print(f"  conf={conf}: {detections} detections (max conf: {max_conf:.3f})")

                # Show class distribution
                if conf == 0.1:
                    class_counts = {}
                    for box in results[0].boxes:
                        cls = int(box.cls[0])
                        class_name = model.names[cls]
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    print(f"    Classes detected: {dict(list(class_counts.items())[:5])}")
            else:
                print(f"  conf={conf}: 0 detections")

    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("  Choose the model with:")
print("  1. Highest detection count at conf=0.25")
print("  2. If all are 0 at 0.25, choose highest at conf=0.1")
print("  3. Check if detected classes match what you need")
print("="*80)
