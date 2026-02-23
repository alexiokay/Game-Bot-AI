"""
Fix mixed label format issue in YOLO dataset for YOLO26 segmentation training.

Problem: Dataset has mixed labels - some with polygons, some with bounding boxes only.
Ultralytics requires ALL labels to be in polygon format for segmentation training.

Solution: Convert bbox-only labels (5 values) to polygon format (4 corner points).

References:
- https://github.com/ultralytics/ultralytics/issues/1008
- https://github.com/ultralytics/ultralytics/issues/3592
- https://blog.roboflow.com/train-yolo26-instance-segmentation-custom-data/
"""

import os
from pathlib import Path


def is_bbox_format(parts):
    """Check if label line is bbox format (exactly 5 values: cls x y w h)."""
    return len(parts) == 5


def is_polygon_format(parts):
    """Check if label line is polygon format (more than 5 values)."""
    return len(parts) > 5


def bbox_to_polygon(x_center, y_center, width, height):
    """
    Convert YOLO bbox (x_center, y_center, w, h) to polygon corner points.
    All values are normalized (0-1).
    Returns 4 corner points: top-left, top-right, bottom-right, bottom-left
    """
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    # Return 4 corner points as flat list
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def convert_label_file(filepath):
    """
    Convert a label file, changing any bbox-only lines to polygon format.
    Returns (converted_count, total_count, was_modified)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    new_lines = []
    converted = 0
    total = len(lines)
    modified = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        if is_bbox_format(parts):
            # Convert bbox to polygon
            cls = parts[0]
            x_center, y_center, w, h = map(float, parts[1:5])
            polygon = bbox_to_polygon(x_center, y_center, w, h)
            polygon_str = ' '.join([f'{coord:.6f}' for coord in polygon])
            new_lines.append(f'{cls} {polygon_str}')
            converted += 1
            modified = True
        elif is_polygon_format(parts):
            # Already polygon format, keep as-is
            new_lines.append(line)
        else:
            # Invalid format, keep as-is
            print(f"  Warning: Invalid line format in {filepath}: {line}")
            new_lines.append(line)

    if modified:
        with open(filepath, 'w') as f:
            f.write('\n'.join(new_lines) + '\n')

    return converted, total, modified


def fix_dataset_labels(dataset_path):
    """Fix all label files in a dataset."""
    dataset_path = Path(dataset_path)

    # Find all label directories
    label_dirs = []
    for split in ['train', 'valid', 'test']:
        label_dir = dataset_path / split / 'labels'
        if label_dir.exists():
            label_dirs.append(label_dir)

    if not label_dirs:
        print(f"No label directories found in {dataset_path}")
        return

    total_files = 0
    total_modified = 0
    total_converted = 0

    for label_dir in label_dirs:
        print(f"\nProcessing: {label_dir}")

        label_files = list(label_dir.glob('*.txt'))
        print(f"  Found {len(label_files)} label files")

        for label_file in label_files:
            converted, total, was_modified = convert_label_file(label_file)
            total_files += 1
            if was_modified:
                total_modified += 1
                total_converted += converted
                print(f"  Converted {converted}/{total} labels in {label_file.name}")

    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Total files processed: {total_files}")
    print(f"  Files modified: {total_modified}")
    print(f"  Labels converted (bbox -> polygon): {total_converted}")
    print(f"{'='*50}")


def verify_dataset(dataset_path):
    """Verify all labels are now in polygon format."""
    dataset_path = Path(dataset_path)

    bbox_count = 0
    polygon_count = 0

    for split in ['train', 'valid', 'test']:
        label_dir = dataset_path / split / 'labels'
        if not label_dir.exists():
            continue

        for label_file in label_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if is_bbox_format(parts):
                        bbox_count += 1
                    elif is_polygon_format(parts):
                        polygon_count += 1

    print(f"\nVerification:")
    print(f"  Bbox-only labels: {bbox_count}")
    print(f"  Polygon labels: {polygon_count}")

    if bbox_count == 0:
        print("  All labels are now in polygon format!")
        return True
    else:
        print("  Some labels still in bbox format")
        return False


if __name__ == '__main__':
    dataset_path = 'F:/dev/bot/yolo/datasets/darkorbit_v6'

    print("YOLO Label Format Fixer")
    print("="*50)
    print(f"Dataset: {dataset_path}")
    print("Converting bbox-only labels to polygon format...")

    fix_dataset_labels(dataset_path)
    verify_dataset(dataset_path)
