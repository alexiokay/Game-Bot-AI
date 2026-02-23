from ultralytics import YOLO
import cv2
from pathlib import Path

model = YOLO('F:/dev/bot/best.pt')

# Get first training image from v3 dataset
train_dir = Path('F:/dev/bot/yolo/datasets/darkorbit_v6/train/images')
images = list(train_dir.glob('*.jpg'))

if not images:
    print("No images found!")
    exit(1)

img_path = str(images[0])
print(f"Testing on: {images[0].name}")

img = cv2.imread(img_path)
if img is None:
    print(f"Failed to load image: {img_path}")
    exit(1)

print(f'Image shape: {img.shape}')

# Test with very low confidence
for conf_thresh in [0.01, 0.05, 0.1, 0.25, 0.5]:
    results = model(img, conf=conf_thresh, verbose=False)
    detections = len(results[0].boxes)
    print(f'Confidence {conf_thresh}: {detections} detections')

    if detections > 0 and conf_thresh == 0.01:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f'  Class {cls} ({model.names[cls]}): {conf:.3f}')

        # Save visualization
        result_img = results[0].plot()
        cv2.imwrite('F:/dev/bot/test_result.jpg', result_img)
        print(f'\nSaved visualization to: F:/dev/bot/test_result.jpg')
