"""
DarkOrbit Bot - AI Auto-Labeler (YOLO-World)
Uses YOLO-World for zero-shot object detection - no training needed!

Simply describe what to detect, and it finds them automatically.

Usage:
    python auto_labeler.py
"""

import os
from pathlib import Path
from typing import List
import shutil


def auto_label_with_yolo_world(images_dir: str, labels_dir: str, 
                                classes: List[str], confidence: float = 0.3):
    """
    Auto-label images using YOLO-World zero-shot detection.
    
    Args:
        images_dir: Directory with images to label
        labels_dir: Directory to save YOLO format labels
        classes: List of class names to detect
        confidence: Minimum confidence threshold
    """
    from ultralytics import YOLO
    
    print("\n" + "="*60)
    print("  YOLO-WORLD AUTO-LABELER")
    print("="*60)
    
    # Load YOLO-World model
    print("\n📦 Loading YOLO-World model...")
    model = YOLO("yolov8s-worldv2.pt")  # Small model, good balance
    
    # Set custom classes to detect
    print(f"🎯 Setting detection classes: {classes}")
    model.set_classes(classes)
    
    # Get images
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    labels_path.mkdir(parents=True, exist_ok=True)
    
    images = list(images_path.glob("*.png")) + list(images_path.glob("*.jpg"))
    print(f"\n📸 Found {len(images)} images to process")
    
    if not images:
        print("❌ No images found!")
        return
    
    # Process each image
    labeled_count = 0
    total_detections = 0
    
    print("\n🔍 Processing images...")
    
    for i, img_path in enumerate(images):
        # Progress
        if (i + 1) % 10 == 0 or i == 0:
            print(f"   {i+1}/{len(images)} images processed...")
        
        # Run detection
        results = model.predict(
            str(img_path), 
            conf=confidence,
            verbose=False
        )
        
        # Get detections
        result = results[0]
        
        if len(result.boxes) == 0:
            continue
        
        # Convert to YOLO format
        label_lines = []
        
        for box in result.boxes:
            class_id = int(box.cls[0])
            
            # Get normalized center coordinates
            xyxyn = box.xyxyn[0].cpu().numpy()
            x_center = (xyxyn[0] + xyxyn[2]) / 2
            y_center = (xyxyn[1] + xyxyn[3]) / 2
            width = xyxyn[2] - xyxyn[0]
            height = xyxyn[3] - xyxyn[1]
            
            label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        if label_lines:
            # Save label file
            label_path = labels_path / f"{img_path.stem}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(label_lines))
            
            labeled_count += 1
            total_detections += len(label_lines)
    
    # Summary
    print("\n" + "="*60)
    print("  ✅ AUTO-LABELING COMPLETE")
    print("="*60)
    print(f"\n📊 Results:")
    print(f"   Images with detections: {labeled_count}/{len(images)}")
    print(f"   Total objects labeled: {total_detections}")
    print(f"   Labels saved to: {labels_path}")
    
    # Show class distribution
    if labeled_count > 0:
        print(f"\n📈 Average objects per image: {total_detections/labeled_count:.1f}")
    
    return labeled_count, total_detections


def preview_detections(image_path: str, classes: List[str]):
    """Preview detections on a single image"""
    from ultralytics import YOLO
    import cv2
    
    print(f"\n🔍 Detecting objects in: {image_path}")
    
    model = YOLO("yolov8s-worldv2.pt")
    model.set_classes(classes)
    
    results = model.predict(image_path, conf=0.25)
    
    # Show results
    result = results[0]
    print(f"\n📦 Found {len(result.boxes)} objects:")
    
    for box in result.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
        print(f"   - {class_name}: {conf:.2f}")
    
    # Save annotated image
    annotated = result.plot()
    output_path = Path(image_path).parent / f"preview_{Path(image_path).name}"
    cv2.imwrite(str(output_path), annotated)
    print(f"\n💾 Saved preview to: {output_path}")
    
    return result


def main():
    print("\n" + "="*60)
    print("  🤖 SPACE ACES AUTO-LABELER")
    print("  Using YOLO-World (Zero-Shot Detection)")
    print("="*60)
    
    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    dataset_dir = data_dir / "yolo_dataset"
    
    train_images = dataset_dir / "images" / "train"
    train_labels = dataset_dir / "labels" / "train"
    val_images = dataset_dir / "images" / "val"
    val_labels = dataset_dir / "labels" / "val"
    
    if not train_images.exists():
        print(f"\n❌ Dataset not found: {train_images}")
        print("   Run prepare_dataset.py first!")
        return
    
    # Classes to detect - using visual descriptions
    classes = [
        "small bright dot",      # Ships, boxes, items
        "glowing ring",          # Portal gate
        "small square icon",     # Collectible boxes
        "blue planet sphere",    # Planets
        "bright white object",   # Ships/NPCs
    ]
    
    print(f"\n📋 Classes to detect:")
    for i, c in enumerate(classes):
        print(f"   {i}: {c}")
    
    print("\n🎮 Options:")
    print("  1. Preview detection on one image first")
    print("  2. Auto-label ALL training images")
    print("  3. Auto-label ALL images (train + val)")
    
    choice = input("\nChoice [1/2/3, default=1]: ").strip() or "1"
    
    if choice == "1":
        # Preview on one image
        sample_images = list(train_images.glob("*.png"))[:1]
        if sample_images:
            preview_detections(str(sample_images[0]), classes)
            print("\n👀 Check the preview image!")
            print("   If detections look good, run again with option 2 or 3.")
        else:
            print("No images found!")
    
    elif choice == "2":
        # Label training images only
        auto_label_with_yolo_world(
            str(train_images),
            str(train_labels),
            classes,
            confidence=0.25
        )
    
    elif choice == "3":
        # Label all images
        print("\n📁 Labeling training images...")
        auto_label_with_yolo_world(
            str(train_images),
            str(train_labels),
            classes,
            confidence=0.25
        )
        
        print("\n📁 Labeling validation images...")
        auto_label_with_yolo_world(
            str(val_images),
            str(val_labels),
            classes,
            confidence=0.25
        )
    
    print("\n" + "="*60)
    print("  🎉 NEXT STEPS")
    print("="*60)
    print("""
1. Review some labels to check quality
2. Optionally fix any mistakes in Roboflow
3. Train your custom YOLO model:
   python detection/train_yolo.py
""")


if __name__ == "__main__":
    main()
