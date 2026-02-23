#!/usr/bin/env python
"""
View YOLO training data with FiftyOne

Usage:
    python view_dataset.py                    # View data/bootstrap
    python view_dataset.py --data path/to/dir # View custom directory
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='View YOLO dataset with FiftyOne')
    parser.add_argument('--data', type=str, default='data/bootstrap',
                       help='Dataset directory (with images/ and labels/ folders)')
    parser.add_argument('--port', type=int, default=5151,
                       help='FiftyOne app port')
    args = parser.parse_args()

    data_dir = Path(args.data)
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    classes_file = data_dir / "classes.json"

    if not images_dir.exists():
        print(f"Error: {images_dir} not found")
        return

    # Load class names
    if classes_file.exists():
        with open(classes_file) as f:
            class_map = json.load(f)
        # Invert: {name: id} -> {id: name}
        classes = {v: k for k, v in class_map.items()}
        max_id = max(classes.keys()) if classes else -1
        class_names = [classes.get(i, f"class_{i}") for i in range(max_id + 1)]
        print(f"Loaded {len(class_names)} classes: {class_names}")
    else:
        print("Warning: classes.json not found, using generic names")
        class_names = None

    try:
        import fiftyone as fo
    except ImportError:
        print("FiftyOne not installed. Run: pip install fiftyone")
        return

    print(f"\nLoading dataset from: {data_dir}")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")

    # Count files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    label_files = list(labels_dir.glob("*.txt"))
    print(f"  Found {len(image_files)} images, {len(label_files)} label files")

    if not image_files:
        print("No images found!")
        return

    # Create FiftyOne dataset
    dataset_name = f"bootstrap_{data_dir.name}"

    # Delete existing dataset with same name
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    # Check for dataset.yaml
    yaml_path = data_dir / "dataset.yaml"

    if yaml_path.exists():
        # Load using YAML file
        print(f"  Using: {yaml_path}")
        dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.YOLOv5Dataset,
            yaml_path=str(yaml_path),
            name=dataset_name
        )
    else:
        # Manual loading - create samples directly
        print("  No dataset.yaml found, loading manually...")
        dataset = fo.Dataset(name=dataset_name)

        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"

            sample = fo.Sample(filepath=str(img_path))

            if label_path.exists():
                # Parse YOLO format labels
                detections = []
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            # Convert YOLO format (center, normalized) to FiftyOne (top-left, normalized)
                            x = cx - w/2
                            y = cy - h/2
                            label = class_names[class_id] if class_names and class_id < len(class_names) else f"class_{class_id}"
                            detections.append(
                                fo.Detection(label=label, bounding_box=[x, y, w, h])
                            )

                sample["ground_truth"] = fo.Detections(detections=detections)

            dataset.add_sample(sample)

    print(f"\nDataset loaded: {len(dataset)} samples")
    print(f"Classes: {dataset.default_classes}")

    # Print stats
    print("\n--- Dataset Statistics ---")
    print(dataset.stats())

    # Launch app
    print(f"\nLaunching FiftyOne app on port {args.port}...")
    print("Press Ctrl+C to stop\n")

    session = fo.launch_app(dataset, port=args.port)
    session.wait()


if __name__ == "__main__":
    main()
