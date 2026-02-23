"""
DarkOrbit Bot - Dataset Preparation Tool
Helps prepare your screenshots for YOLO training.

This script:
1. Finds all your screenshots
2. Creates a dataset structure for YOLO
3. Provides instructions for labeling with Roboflow/CVAT

Usage:
    python prepare_dataset.py
"""

import os
import shutil
import random
from pathlib import Path


def find_screenshots(recordings_dir: str) -> list:
    """Find all screenshots from recording sessions"""
    screenshots = []
    recordings_path = Path(recordings_dir)
    
    if not recordings_path.exists():
        return screenshots
    
    # Find all session folders
    for session_dir in recordings_path.glob("session_*"):
        screenshots_dir = session_dir / "screenshots"
        if screenshots_dir.exists():
            for img in screenshots_dir.glob("*.png"):
                screenshots.append(str(img))
    
    return screenshots


def create_dataset_structure(base_dir: str, screenshots: list, 
                            train_ratio: float = 0.8):
    """Create YOLO dataset structure"""
    dataset_dir = Path(base_dir) / "yolo_dataset"
    
    # Create directories
    (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Shuffle and split
    random.shuffle(screenshots)
    split_idx = int(len(screenshots) * train_ratio)
    train_images = screenshots[:split_idx]
    val_images = screenshots[split_idx:]
    
    # Copy images
    print(f"\n📁 Creating dataset structure...")
    print(f"   Training images: {len(train_images)}")
    print(f"   Validation images: {len(val_images)}")
    
    for img_path in train_images:
        dest = dataset_dir / "images" / "train" / Path(img_path).name
        if not dest.exists():
            shutil.copy2(img_path, dest)
    
    for img_path in val_images:
        dest = dataset_dir / "images" / "val" / Path(img_path).name
        if not dest.exists():
            shutil.copy2(img_path, dest)
    
    # Create dataset.yaml
    yaml_content = f"""# Space Aces / DarkOrbit Dataset
path: {dataset_dir.absolute()}
train: images/train
val: images/val

# Classes - adjust based on what you want to detect
names:
  0: box           # Collectible boxes/cargo
  1: npc           # Enemy NPCs/aliens
  2: player_ship   # Your ship (optional)
  3: bonus_box     # Special bonus boxes
  4: portal        # Jump portals/gates

# You can add more classes as needed:
#  5: asteroid
#  6: station
#  7: other_player
"""
    
    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    print(f"\n✅ Dataset structure created at: {dataset_dir}")
    print(f"   - dataset.yaml (edit classes as needed)")
    print(f"   - images/train/ ({len(train_images)} images)")
    print(f"   - images/val/ ({len(val_images)} images)")
    
    return dataset_dir


def print_labeling_instructions(dataset_dir: Path):
    """Print instructions for labeling"""
    print("\n" + "="*60)
    print("  📝 NEXT STEP: LABEL YOUR IMAGES")
    print("="*60)
    
    print("""
You need to label the objects in your screenshots. 

OPTION 1: Roboflow (Recommended - Free tier available)
--------------------------------------------------------
1. Go to https://roboflow.com and create account
2. Create new project → Object Detection
3. Upload images from: """ + str(dataset_dir / "images" / "train"))
    
    print("""4. Draw boxes around:
   - 🟦 Boxes (cargo/collectibles)
   - 🟥 NPCs (enemies/aliens)
   - 🟨 Bonus boxes (if different from regular)
   - 🟣 Portals (jump gates)

5. Export as "YOLO v8" format
6. Download and extract to: """ + str(dataset_dir))
    
    print("""
OPTION 2: CVAT (Free, self-hosted)
--------------------------------------------------------
1. Go to https://cvat.ai
2. Create task, upload images
3. Label objects
4. Export as "YOLO 1.1"

OPTION 3: LabelImg (Offline tool)
--------------------------------------------------------
1. pip install labelImg
2. labelImg """ + str(dataset_dir / "images" / "train") + """
3. Save in YOLO format

""")
    
    print("="*60)
    print("  After labeling, run: python train_yolo.py")
    print("="*60)


def main():
    print("\n" + "="*60)
    print("  DARKORBIT BOT - DATASET PREPARATION")
    print("="*60)
    
    # Find recordings
    script_dir = Path(__file__).parent
    recordings_dir = script_dir.parent / "data" / "recordings"
    
    print(f"\n🔍 Looking for screenshots in: {recordings_dir}")
    
    screenshots = find_screenshots(str(recordings_dir))
    
    if not screenshots:
        print("\n❌ No screenshots found!")
        print("   Run game_recorder.py first to capture gameplay.")
        return
    
    print(f"   Found {len(screenshots)} screenshots")
    
    # Show sample
    print("\n📸 Sample screenshots:")
    for img in screenshots[:5]:
        print(f"   - {Path(img).name}")
    if len(screenshots) > 5:
        print(f"   ... and {len(screenshots) - 5} more")
    
    # Create dataset
    data_dir = script_dir.parent / "data"
    dataset_dir = create_dataset_structure(str(data_dir), screenshots)
    
    # Print instructions
    print_labeling_instructions(dataset_dir)
    
    # Save dataset path for training script
    config_path = data_dir / "dataset_config.json"
    import json
    with open(config_path, "w") as f:
        json.dump({
            "dataset_path": str(dataset_dir),
            "yaml_path": str(dataset_dir / "dataset.yaml"),
            "num_images": len(screenshots)
        }, f, indent=2)
    
    print(f"\n Config saved to: {config_path}")


if __name__ == "__main__":
    main()
