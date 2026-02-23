"""Analyze YOLO dataset statistics - class distribution, label counts, etc."""

import os
from pathlib import Path
from collections import Counter
import yaml

def analyze_dataset(dataset_path):
    dataset_path = Path(dataset_path)

    # Read data.yaml to get class names
    yaml_path = dataset_path / 'data.yaml'
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    class_names = data_config['names']
    print(f"Dataset: {dataset_path}")
    print(f"Number of classes: {len(class_names)}")
    print(f"\nClass names:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")

    # Analyze each split
    splits = ['train', 'valid', 'test']
    all_stats = {}

    for split in splits:
        label_dir = dataset_path / split / 'labels'
        if not label_dir.exists():
            print(f"\n{split.upper()}: Not found")
            continue

        label_files = list(label_dir.glob('*.txt'))
        image_dir = dataset_path / split / 'images'
        image_files = list(image_dir.glob('*.*')) if image_dir.exists() else []

        # Count classes
        class_counter = Counter()
        total_labels = 0
        polygon_labels = 0
        bbox_labels = 0

        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    class_id = int(parts[0])
                    class_counter[class_id] += 1
                    total_labels += 1

                    # Check if polygon (more than 5 values) or bbox (exactly 5 values)
                    if len(parts) > 5:
                        polygon_labels += 1
                    elif len(parts) == 5:
                        bbox_labels += 1

        all_stats[split] = {
            'images': len(image_files),
            'label_files': len(label_files),
            'total_labels': total_labels,
            'polygon_labels': polygon_labels,
            'bbox_labels': bbox_labels,
            'class_counter': class_counter
        }

        print(f"\n{'='*60}")
        print(f"{split.upper()} SET:")
        print(f"{'='*60}")
        print(f"Images: {len(image_files)}")
        print(f"Label files: {len(label_files)}")
        print(f"Total labels: {total_labels}")
        print(f"  Polygon labels: {polygon_labels}")
        print(f"  Bbox labels: {bbox_labels}")
        print(f"\nClass distribution:")
        print(f"{'Class ID':<10} {'Class Name':<30} {'Count':<10}")
        print("-" * 60)

        for class_id in sorted(class_counter.keys()):
            count = class_counter[class_id]
            class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown-{class_id}"
            print(f"{class_id:<10} {class_name:<30} {count:<10}")

        # Classes with very few examples
        rare_classes = [(cid, cnt) for cid, cnt in class_counter.items() if cnt < 5]
        if rare_classes:
            print(f"\nWARNING: {len(rare_classes)} classes with less than 5 examples:")
            for cid, cnt in sorted(rare_classes, key=lambda x: x[1]):
                class_name = class_names[cid] if cid < len(class_names) else f"Unknown-{cid}"
                print(f"  Class {cid} ({class_name}): {cnt} examples")

    # Summary
    print(f"\n{'='*60}")
    print("DATASET SUMMARY:")
    print(f"{'='*60}")
    total_images = sum(s['images'] for s in all_stats.values())
    total_labels = sum(s['total_labels'] for s in all_stats.values())
    print(f"Total images: {total_images}")
    print(f"Total labels: {total_labels}")
    print(f"Average labels per image: {total_labels / total_images if total_images > 0 else 0:.2f}")

    # Check if dataset is too small
    if all_stats.get('train', {}).get('images', 0) < 100:
        print(f"\n⚠️  WARNING: Training set has only {all_stats.get('train', {}).get('images', 0)} images.")
        print(f"   Recommended: At least 100-500 images for training")
        print(f"   For {len(class_names)} classes, you ideally need 1000+ images")

if __name__ == '__main__':
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'F:/dev/bot/yolo/datasets/darkorbit_v6'
    analyze_dataset(dataset_path)
