"""Rebalance train/val split to 80/20 ratio.

Current: 140 train / 13 val (91%/9%)
Target: 122 train / 31 val (80%/20%)
Action: Move 18 random images from train to validation
"""
import os
import random
import shutil
from pathlib import Path

def main():
    dataset_root = Path(__file__).parent
    train_images = dataset_root / 'train' / 'images'
    train_labels = dataset_root / 'train' / 'labels'
    val_images = dataset_root / 'valid' / 'images'
    val_labels = dataset_root / 'valid' / 'labels'

    # Get all training images
    train_image_files = list(train_images.glob('*.*'))
    print(f"Found {len(train_image_files)} training images")

    # Randomly select 18 images to move
    random.seed(42)  # For reproducibility
    images_to_move = random.sample(train_image_files, 18)

    print(f"\nMoving 18 images from train to validation...")

    moved_count = 0
    for img_path in images_to_move:
        # Get corresponding label file
        label_name = img_path.stem + '.txt'
        label_path = train_labels / label_name

        # Check if label exists
        if not label_path.exists():
            print(f"Warning: Label not found for {img_path.name}, skipping")
            continue

        # Move image
        dest_img = val_images / img_path.name
        shutil.move(str(img_path), str(dest_img))

        # Move label
        dest_label = val_labels / label_name
        shutil.move(str(label_path), str(dest_label))

        moved_count += 1
        print(f"  Moved: {img_path.name}")

    print(f"\n✓ Successfully moved {moved_count} images and labels")

    # Verify new counts
    new_train_count = len(list(train_images.glob('*.*')))
    new_val_count = len(list(val_images.glob('*.*')))

    print(f"\nNew split:")
    print(f"  Train: {new_train_count} images ({new_train_count/(new_train_count+new_val_count)*100:.1f}%)")
    print(f"  Val:   {new_val_count} images ({new_val_count/(new_train_count+new_val_count)*100:.1f}%)")

if __name__ == '__main__':
    main()
