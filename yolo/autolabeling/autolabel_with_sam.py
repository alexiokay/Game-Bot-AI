"""Auto-label training screenshots using SAM2 for segmentation + YOLO for classification.

This script:
1. Uses your existing YOLO model to detect objects (bounding boxes + class)
2. Uses SAM2 to refine each detection into a precise segmentation mask
3. Outputs YOLO-format polygon labels for training

Usage:
    python autolabel_with_sam.py --input "F:/dev/bot/yolo/training_screenshots" --output "F:/dev/bot/yolo/autolabeled"
"""
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
from tqdm import tqdm

# Add detection module to path
sys.path.insert(0, str(Path(__file__).parent / "darkorbit_bot"))

from ultralytics import YOLO


def load_sam():
    """Load SAM model for mask refinement (tries SAM3, SAM2, then Ultralytics)."""
    # Check available SAM models
    sam_models = [
        (Path("F:/dev/bot/darkorbit_bot/models/sam3.pt"), "SAM3"),
        (Path("F:/dev/bot/darkorbit_bot/models/sam2_hiera_tiny.pt"), "SAM2-tiny"),
        (Path("F:/dev/bot/darkorbit_bot/sam2.1_s.pt"), "SAM2.1-small"),
        (Path("F:/dev/bot/darkorbit_bot/FastSAM-s.pt"), "FastSAM-s"),
    ]

    # Try Ultralytics SAM first (easiest)
    for model_path, name in sam_models:
        if model_path.exists():
            try:
                from ultralytics import SAM
                model = SAM(str(model_path))
                print(f"Loaded {name} from {model_path}")
                return model, "ultralytics"
            except Exception as e:
                print(f"Failed to load {name} with Ultralytics: {e}")
                continue

    # Try SAM2 API
    for model_path, name in sam_models:
        if model_path.exists() and "sam2" in name.lower():
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                config = "sam2_hiera_t.yaml"  # tiny config
                model = build_sam2(config, str(model_path), device="cuda")
                predictor = SAM2ImagePredictor(model)
                print(f"Loaded {name} with SAM2 API from {model_path}")
                return predictor, "sam2"
            except Exception as e:
                print(f"Failed to load {name} with SAM2 API: {e}")
                continue

    print("No SAM model could be loaded!")
    return None, None


def load_ultralytics_sam():
    """Fallback: Use Ultralytics SAM (FastSAM or SAM)."""
    try:
        # Try FastSAM (much faster)
        from ultralytics import FastSAM
        model = FastSAM("FastSAM-x.pt")
        print("Loaded FastSAM from Ultralytics")
        return model, "fastsam"
    except:
        try:
            # Try regular SAM
            from ultralytics import SAM
            model = SAM("sam_b.pt")
            print("Loaded SAM from Ultralytics")
            return model, "sam"
        except:
            print("No SAM model available")
            return None, None


def mask_to_polygon(mask: np.ndarray, simplify: bool = True) -> List[Tuple[float, float]]:
    """Convert binary mask to polygon points (normalized 0-1)."""
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    # Get largest contour
    contour = max(contours, key=cv2.contourArea)

    # Simplify contour if requested
    if simplify:
        epsilon = 0.005 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)

    # Convert to normalized coordinates
    h, w = mask.shape
    polygon = []
    for point in contour:
        x, y = point[0]
        polygon.append((x / w, y / h))

    return polygon


def autolabel_images(input_dir: str, output_dir: str, yolo_model_path: str, use_sam: bool = True, conf_threshold: float = 0.1):
    """
    Auto-label images using YOLO + SAM.

    Args:
        input_dir: Directory with images to label
        output_dir: Directory to save labeled images + YOLO txt files
        yolo_model_path: Path to YOLO model for detection
        use_sam: Whether to use SAM for mask refinement (if False, just uses YOLO boxes)
        conf_threshold: Confidence threshold for detections
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create labels directory
    labels_dir = output_path / "labels"
    labels_dir.mkdir(exist_ok=True)

    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)

    print("="*60)
    print("  AUTO-LABELING WITH SAM")
    print("="*60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"YOLO model: {yolo_model_path}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Use SAM: {use_sam}")
    print("="*60)

    # Load YOLO model
    print("\nLoading YOLO model...")
    yolo = YOLO(yolo_model_path)
    print(f"Model has {len(yolo.names)} classes")

    # Load SAM if requested
    sam_predictor = None
    sam_type = None
    if use_sam:
        print("\nLoading SAM model...")
        sam_predictor, sam_type = load_sam()

    # Get images
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    print(f"\nFound {len(image_files)} images")

    if not image_files:
        print("No images found!")
        return

    # Process each image
    total_labels = 0
    for img_path in tqdm(image_files, desc="Auto-labeling"):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # Run YOLO detection
        results = yolo(img, conf=conf_threshold, verbose=False)

        if len(results[0].boxes) == 0:
            continue  # No detections

        # Prepare label file
        label_lines = []

        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if use_sam and sam_predictor:
                # Get bbox for SAM
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = [int(x1), int(y1), int(x2), int(y2)]

                # Use SAM to get mask
                try:
                    if sam_type == "ultralytics":
                        # Ultralytics SAM3/SAM - call directly with bboxes parameter
                        sam_results = sam_predictor(img, bboxes=[bbox], verbose=False)
                        if len(sam_results) > 0 and sam_results[0].masks is not None:
                            mask = sam_results[0].masks.data[0].cpu().numpy()
                        else:
                            raise Exception("No mask generated")
                    else:
                        # SAM2 API - use set_image/predict
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        sam_predictor.set_image(img_rgb)
                        masks, scores, _ = sam_predictor.predict(
                            box=np.array(bbox),
                            multimask_output=False
                        )
                        mask = masks[0]

                    # Convert mask to polygon
                    polygon = mask_to_polygon(mask)

                    if len(polygon) >= 3:  # Need at least 3 points
                        # Format: class x1 y1 x2 y2 x3 y3 ...
                        coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in polygon])
                        label_lines.append(f"{cls} {coords}\n")
                        total_labels += 1
                    else:
                        # Polygon too small, use bbox
                        raise Exception("Polygon too small")
                except Exception as e:
                    # Fallback to bbox
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    label_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    total_labels += 1
            else:
                # Just use YOLO bbox (no SAM)
                x_center, y_center, width, height = box.xywhn[0].cpu().numpy()
                label_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                total_labels += 1

        # Save label file
        if label_lines:
            label_file = labels_dir / f"{img_path.stem}.txt"
            label_file.write_text("".join(label_lines))

            # Copy image
            import shutil
            shutil.copy(img_path, images_dir / img_path.name)

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"Labeled images: {len(list(labels_dir.glob('*.txt')))}")
    print(f"Total labels: {total_labels}")
    print(f"\nOutput structure:")
    print(f"  {output_path}/images/  - Images")
    print(f"  {output_path}/labels/  - YOLO format labels")
    print(f"\nNext steps:")
    print(f"1. Review labels in Roboflow or Label Studio")
    print(f"2. Merge with existing dataset")
    print(f"3. Retrain YOLO model")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Auto-label screenshots with SAM')
    parser.add_argument('--input', type=str, default='F:/dev/bot/yolo/training_screenshots',
                       help='Input directory with images')
    parser.add_argument('--output', type=str, default='F:/dev/bot/yolo/autolabeled',
                       help='Output directory for labeled data')
    parser.add_argument('--model', type=str, default='F:/dev/bot/best.pt',
                       help='YOLO model for detection')
    parser.add_argument('--conf', type=float, default=0.1,
                       help='Confidence threshold (default: 0.1 for auto-labeling)')
    parser.add_argument('--no-sam', action='store_true',
                       help='Skip SAM, just use YOLO bounding boxes')

    args = parser.parse_args()

    autolabel_images(
        input_dir=args.input,
        output_dir=args.output,
        yolo_model_path=args.model,
        use_sam=not args.no_sam,
        conf_threshold=args.conf
    )


if __name__ == '__main__':
    main()
