"""SAM + CLIP Auto-labeling: Segment with SAM, classify with CLIP.

How it works:
1. SAM automatically segments ALL objects in the image
2. CLIP classifies each segment using text descriptions
3. You provide class names, CLIP matches segments to them

This is similar to Grounding DINO but using SAM3 + CLIP!

Usage:
    python sam_clip_autolabel.py --classes "enemy ship" "player ship" "explosion" "laser" "UI button"
"""
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import SAM
import torch


def load_clip():
    """Load CLIP for classification."""
    from transformers import CLIPProcessor, CLIPModel

    print("Loading CLIP...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"CLIP loaded on {device}")

    return model, processor, device


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


def mask_to_polygon(mask: np.ndarray) -> List[Tuple[float, float]]:
    """Convert mask to polygon."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)

    h, w = mask.shape
    polygon = [(x / w, y / h) for [[x, y]] in contour]
    return polygon


def classify_segment_with_clip(img: np.ndarray, mask: np.ndarray, class_names: List[str],
                                clip_model, clip_processor, device) -> Tuple[int, float]:
    """
    Classify a segment using CLIP.

    Args:
        img: Full image
        mask: Binary mask of segment
        class_names: List of class names to match against
        clip_model: CLIP model
        clip_processor: CLIP processor
        device: cuda/cpu

    Returns:
        class_id, confidence
    """
    from PIL import Image

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

    # Convert to PIL
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    # Encode image and text
    inputs = clip_processor(
        images=pil_img,
        text=class_names,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # Image-text similarity scores
        probs = logits_per_image.softmax(dim=1)  # Probabilities

    # Get best match
    class_id = probs.argmax().item()
    confidence = probs[0, class_id].item()

    return class_id, confidence


def autolabel_with_sam_clip(input_dir: str, output_dir: str, class_names: List[str],
                            sam_model_path: str, min_confidence: float = 0.3):
    """
    Auto-label using SAM (segmentation) + CLIP (classification).

    Args:
        input_dir: Images to label
        output_dir: Output directory
        class_names: List of class names (e.g., ["enemy ship", "player ship", "explosion"])
        sam_model_path: Path to SAM model
        min_confidence: Minimum CLIP confidence to accept (0-1)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    (output_path / 'images').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels').mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("  SAM + CLIP AUTO-LABELING")
    print("="*60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"SAM model: {sam_model_path}")
    print(f"Classes: {class_names}")
    print(f"Min confidence: {min_confidence}")
    print("="*60)

    # Load models
    print("\nLoading SAM...")
    sam = SAM(sam_model_path)

    clip_model, clip_processor, device = load_clip()

    # Get images
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    print(f"\nFound {len(image_files)} images\n")

    if not image_files:
        print("No images found!")
        return

    total_segments = 0
    total_labeled = 0
    total_duplicates = 0
    total_low_conf = 0
    total_no_polygon = 0
    images_processed = 0

    # Create class name to ID mapping
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    for img_path in tqdm(image_files, desc="Processing"):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # SAM auto-segment
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
        segment_count = 0

        # Classify each segment with CLIP
        for mask in unique_masks:
            # Classify with CLIP
            cls_id, confidence = classify_segment_with_clip(
                img, mask, class_names, clip_model, clip_processor, device
            )

            if cls_id == -1 or confidence < min_confidence:
                total_low_conf += 1
                continue

            # Convert to polygon
            polygon = mask_to_polygon(mask)
            if len(polygon) < 3:
                total_no_polygon += 1
                continue

            # YOLO format
            coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in polygon])
            label_lines.append(f"{cls_id} {coords}\n")
            segment_count += 1
            total_labeled += 1

        # Save labels
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
    print(f"  Segments with low CLIP confidence: {total_low_conf}")
    print(f"  Segments with invalid polygons: {total_no_polygon}")
    print(f"  Successfully labeled: {total_labeled}")
    print(f"\nAverages:")
    print(f"  Segments per image: {total_segments/len(image_files):.1f}")
    print(f"  Labels per image: {total_labeled/images_processed:.1f}" if images_processed > 0 else "0")
    print(f"\nClass distribution:")

    # Show class distribution
    all_labels = []
    for label_file in (output_path / 'labels').glob('*.txt'):
        for line in label_file.read_text().splitlines():
            cls_id = int(line.split()[0])
            all_labels.append(cls_id)

    for idx, name in enumerate(class_names):
        count = all_labels.count(idx)
        print(f"  {name}: {count}")

    print(f"\nOutput:")
    print(f"  {output_path}/images/")
    print(f"  {output_path}/labels/")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SAM + CLIP auto-labeling')
    parser.add_argument('--input', type=str, default='F:/dev/bot/yolo/training_screenshots',
                       help='Input directory')
    parser.add_argument('--output', type=str, default='F:/dev/bot/yolo/autolabeled_sam_clip',
                       help='Output directory')
    parser.add_argument('--sam', type=str, default='F:/dev/bot/darkorbit_bot/models/sam3.pt',
                       help='SAM model path')
    parser.add_argument('--classes', nargs='+', required=True,
                       help='Class names (e.g., --classes "enemy ship" "player ship" "explosion")')
    parser.add_argument('--min-conf', type=float, default=0.3,
                       help='Minimum CLIP confidence (default: 0.3)')

    args = parser.parse_args()

    autolabel_with_sam_clip(
        input_dir=args.input,
        output_dir=args.output,
        class_names=args.classes,
        sam_model_path=args.sam,
        min_confidence=args.min_conf
    )


if __name__ == '__main__':
    main()
