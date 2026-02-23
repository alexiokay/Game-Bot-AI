"""Auto-label using SAM-first approach: Segment everything, then classify.

Strategy:
1. SAM auto-segments ALL objects in the image
2. YOLO classifies each segment (finds best matching class)
3. Outputs YOLO polygon labels

This finds way more objects than YOLO-first approach!

Usage:
    python autolabel_sam_first.py
"""
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO, SAM


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate Intersection over Union of two masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def remove_duplicate_masks(masks: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """Remove duplicate/overlapping masks using NMS."""
    if len(masks) == 0:
        return masks

    # Calculate area of each mask
    areas = [mask.sum() for mask in masks]

    # Sort by area (largest first)
    sorted_indices = sorted(range(len(masks)), key=lambda i: areas[i], reverse=True)

    kept_indices = []

    for i in sorted_indices:
        # Check if this mask overlaps significantly with any kept mask
        keep = True
        for j in kept_indices:
            iou = calculate_iou(masks[i], masks[j])
            if iou > iou_threshold:
                keep = False
                break

        if keep:
            kept_indices.append(i)

    return masks[kept_indices]


def mask_to_polygon(mask: np.ndarray, simplify: bool = True) -> List[Tuple[float, float]]:
    """Convert binary mask to polygon points (normalized 0-1)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    # Get largest contour
    contour = max(contours, key=cv2.contourArea)

    # Simplify
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


def classify_segment(img: np.ndarray, mask: np.ndarray, yolo: YOLO) -> Tuple[int, float]:
    """
    Classify a segmented region using YOLO.

    Returns:
        class_id, confidence
    """
    # Get bounding box of mask
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return -1, 0.0

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop region
    crop = img[y_min:y_max+1, x_min:x_max+1]

    if crop.size == 0:
        return -1, 0.0

    # Run YOLO on crop
    results = yolo(crop, conf=0.01, verbose=False)

    if len(results[0].boxes) == 0:
        return -1, 0.0

    # Get highest confidence detection
    best_box = results[0].boxes[0]  # Already sorted by confidence
    cls = int(best_box.cls[0])
    conf = float(best_box.conf[0])

    return cls, conf


def autolabel_sam_first(input_dir: str, output_dir: str, yolo_model_path: str, sam_model_path: str):
    """
    Auto-label using SAM-first approach.

    Args:
        input_dir: Images to label
        output_dir: Output for labeled data
        yolo_model_path: YOLO model for classification
        sam_model_path: SAM model for segmentation
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    (output_path / 'images').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels').mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("  SAM-FIRST AUTO-LABELING")
    print("="*60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"YOLO (classifier): {yolo_model_path}")
    print(f"SAM (segmenter): {sam_model_path}")
    print("="*60)

    # Load models
    print("\nLoading SAM...")
    sam = SAM(sam_model_path)

    print("Loading YOLO...")
    yolo = YOLO(yolo_model_path)
    print(f"YOLO classes: {len(yolo.names)}")

    # Get images
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    print(f"\nFound {len(image_files)} images\n")

    if not image_files:
        print("No images found!")
        return

    total_segments = 0
    total_classified = 0
    total_duplicates = 0
    total_low_conf = 0
    total_no_polygon = 0
    images_processed = 0

    for img_path in tqdm(image_files, desc="Auto-labeling"):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # SAM auto-segment (finds ALL objects)
        sam_results = sam(img, verbose=False)

        if len(sam_results) == 0 or sam_results[0].masks is None:
            continue

        masks = sam_results[0].masks.data.cpu().numpy()
        num_segments = len(masks)
        total_segments += num_segments

        # Remove duplicate/overlapping segments
        unique_masks = remove_duplicate_masks(masks, iou_threshold=0.5)
        num_duplicates = num_segments - len(unique_masks)
        total_duplicates += num_duplicates

        label_lines = []

        # Classify each segment
        for mask in unique_masks:
            # Get class from YOLO
            cls, conf = classify_segment(img, mask, yolo)

            if cls == -1 or conf < 0.02:  # Skip if unclassified or very low confidence
                total_low_conf += 1
                continue

            # Convert mask to polygon
            polygon = mask_to_polygon(mask)

            if len(polygon) < 3:
                total_no_polygon += 1
                continue

            # YOLO format: class x1 y1 x2 y2 ...
            coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in polygon])
            label_lines.append(f"{cls} {coords}\n")
            total_classified += 1

        # Save if we got any labels
        if label_lines:
            label_file = output_path / 'labels' / f"{img_path.stem}.txt"
            label_file.write_text("".join(label_lines))

            # Copy image
            import shutil
            shutil.copy(img_path, output_path / 'images' / img_path.name)

            images_processed += 1

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"Images processed: {images_processed}")
    print(f"\nSegment Statistics:")
    print(f"  Total segments found by SAM: {total_segments}")
    print(f"  Duplicate segments removed: {total_duplicates}")
    print(f"  Unique segments: {total_segments - total_duplicates}")
    print(f"\nFiltering Statistics:")
    print(f"  Segments with low YOLO confidence: {total_low_conf}")
    print(f"  Segments with invalid polygons: {total_no_polygon}")
    print(f"  Successfully classified: {total_classified}")
    print(f"\nAverages:")
    print(f"  Segments per image: {total_segments/len(image_files):.1f}")
    print(f"  Labels per image: {total_classified/images_processed:.1f}" if images_processed > 0 else "0")
    print(f"\nOutput:")
    print(f"  {output_path}/images/")
    print(f"  {output_path}/labels/")
    print(f"\nNext: Upload to Roboflow, review labels, merge with dataset")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SAM-first auto-labeling')
    parser.add_argument('--input', type=str, default='F:/dev/bot/yolo/training_screenshots',
                       help='Input directory')
    parser.add_argument('--output', type=str, default='F:/dev/bot/yolo/autolabeled_sam_first',
                       help='Output directory')
    parser.add_argument('--yolo', type=str, default='F:/dev/bot/best.pt',
                       help='YOLO model for classification')
    parser.add_argument('--sam', type=str, default='F:/dev/bot/darkorbit_bot/models/sam3.pt',
                       help='SAM model for segmentation')

    args = parser.parse_args()

    autolabel_sam_first(
        input_dir=args.input,
        output_dir=args.output,
        yolo_model_path=args.yolo,
        sam_model_path=args.sam
    )


if __name__ == '__main__':
    main()
